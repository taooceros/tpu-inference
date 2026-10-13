"""Persistent per-host controller used only by scripts/ici_engine.py."""
from __future__ import annotations
import gc
import os
import time
import traceback


def engine_config(tp, workers=None):
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(model='/models/gemma-4-31b', tensor_parallel_size=tp,
        distributed_executor_backend='ray', dtype='bfloat16', kv_cache_dtype='float32',
        additional_config={'activation_dtype': 'float32', 'matmul_precision': 'highest'},
        max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=1024,
        gpu_memory_utilization=0.9, enable_prefix_caching=False, enforce_eager=False,
        seed=0, trust_remote_code=True, disable_log_stats=True,
        limit_mm_per_prompt={'image': 0, 'audio': 0, 'video': 0})
    config = args.create_engine_config()
    config.scheduler_config.async_scheduling = False
    if workers is not None:
        config.additional_config['ici_workers'] = workers
    return config


class ResidentWorker:
    def __init__(self, rank, environment):
        os.environ.update(environment)
        for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            os.environ.pop(key, None)
        self.rank = rank
        self.phases = []
        self.failure = None
        self.worker = None
        self.resident = {}
        self.layout = None
        self.cache_events = []

    def timed(self, name, call):
        start = time.time()
        row = {'process': self.rank, 'phase': name, 'start': start}
        try:
            return call()
        except Exception:
            self.failure = traceback.format_exc()
            row['error'] = self.failure
            raise
        finally:
            row['end'] = time.time()
            row['seconds'] = row['end'] - start
            self.phases.append(row)

    def start(self):
        tp = int(os.environ.get('ICI_INITIAL_TP', '16'))
        def initialize():
            import jax
            jax.distributed.initialize(coordinator_address=os.environ['ICI_COORDINATOR'],
                num_processes=int(os.environ.get('ICI_PROCESS_COUNT', '4')), process_id=self.rank, initialization_timeout=180)
            if os.environ.get('ICI_SUBMESH_CACHE_WRITER') == '1':
                from hetero.ici_resident import install_submesh_cache_writer
                install_submesh_cache_writer(self.cache_events)
            from tpu_inference.worker.tpu_worker import TPUWorker
            from hetero.native_backend import verify_backend
            verify_backend()
            jax.config.update('jax_default_matmul_precision', 'highest')
            config = engine_config(tp)
            self.worker = TPUWorker(config, 0, self.rank, '',
                is_driver_worker=self.rank == 0, devices=jax.devices())
            self.worker.init_device()
            self.layout = f'TP{tp}'
        self.timed('runtime_and_runner_init', initialize)
        from tpu_inference.models.common import model_loader
        from hetero.ici_resident import merged_layouts
        original_builder = model_loader._get_nnx_model
        import jax
        import numpy as np
        from tpu_inference.layers.common import utils as common_utils
        original_cpu_mesh = common_utils.cpu_mesh
        common_utils.cpu_mesh = lambda: jax.sharding.Mesh(np.array(jax.local_devices(backend='cpu')[:1]), ('cpu',))
        def capture(*args, **kwargs):
            model = original_builder(*args, **kwargs)
            self.packing = merged_layouts(model)
            return model
        try:
            model_loader._get_nnx_model = capture
            self.timed('initial_checkpoint_load', self.worker.load_model)
        finally:
            model_loader._get_nnx_model = original_builder
            common_utils.cpu_mesh = original_cpu_mesh
        self.resident = {f'TP{tp}': self.worker.model_runner.state}
        if self.rank == 0:
            self.catalogue = [{'path': jax.tree_util.keystr(path), 'shape': list(array.shape),
                'dtype': str(array.dtype), 'partition_spec': str(array.sharding.spec)}
                for path, array in jax.tree_util.tree_flatten_with_path(self.worker.model_runner.state)[0]]
        return self.report()

    def rpc(self, method, args, kwargs, index):
        from vllm.config import set_current_vllm_config
        self.worker.is_driver_worker = index == 0
        with set_current_vllm_config(self.worker.vllm_config):
            if method in ('determine_available_memory', 'initialize_from_config', 'compile_or_warm_up_model'):
                result = self.timed(method, lambda: getattr(self.worker, method)(*args, **kwargs))
                phase = self.phases[-1]
                if method == 'compile_or_warm_up_model':
                    self.serving_ready_at = phase['end']
                if method == 'determine_available_memory':
                    phase['available_memory_bytes'] = result
                if method == 'initialize_from_config':
                    import jax
                    phase['kv_num_blocks'] = args[0].num_blocks
                    phase['kv_shapes'] = [list(array.shape) for array in jax.tree_util.tree_leaves(self.worker.model_runner.kv_caches)]
                return result
            if callable(method):
                return method(self.worker, *args, **kwargs)
            if method == 'first_sample_tokens':
                result = self.timed('first_token', lambda: self.worker.sample_tokens(*args, **kwargs))
                self.phases[-1]['ready_to_first_sample_s'] = self.phases[-1]['end'] - self.serving_ready_at
                return result
            if method == 'drain_execute_model':
                return self.timed('drain_finished_batch', lambda: self.worker.execute_model(*args, **kwargs))
            return getattr(self.worker, method)(*args, **kwargs)

    def retire(self):
        def drain():
            import jax
            runner = self.worker.model_runner
            if runner.requests or runner.input_batch.num_reqs:
                raise RuntimeError('runner has live requests at transition')
            pending = getattr(runner, '_pre_async_results', None)
            if pending is not None:
                raise RuntimeError('async sampling results remain at transition')
            runner.compilation_manager._flush_compilations()
            jax.block_until_ready(runner.state_leaves)
        self.timed('drain', drain)
        def free():
            self.worker.model_runner.delete_kv_cache()
            self.worker.model_runner = None
            gc.collect()
            import jax
            jax.clear_caches()
        self.timed('kv_and_runner_free', free)
        return self.report()

    def switch(self, layout):
        import jax
        import numpy as np
        from flax import nnx
        from vllm.config import set_current_vllm_config
        from tpu_inference.models.common import model_loader
        from tpu_inference.runner.tpu_runner import TPUModelRunner
        from hetero.ici_resident import reshard_tree, attach_resident_model, repack_state, resident_views
        devices = jax.devices()
        groups = {'TP16': devices} if layout == 'TP16' else {
            'TP8_A': [d for d in devices if d.process_index in (0, 1)],
            'TP8_B': [d for d in devices if d.process_index in (2, 3)]}
        name = 'TP16' if layout == 'TP16' else ('TP8_A' if self.rank < 2 else 'TP8_B')
        source = self.resident['TP16'] if self.layout == 'TP16' else self.global_resident
        if self.layout == 'TP16':
            source = self.timed('merged_weight_repacking', lambda: repack_state(source, self.packing, 16, 8))
        self.resident = {}
        self.global_resident = None
        gc.collect()
        meshes = {}
        for engine, selected in groups.items():
            config = engine_config(len(selected))
            if engine == name:
                runner = self.timed('runner_rebuild:' + engine, lambda: TPUModelRunner(
                    config, selected, self.rank if layout == 'TP16' else self.rank % 2, True, True))
                own_runner = runner
            else:
                runner = TPUModelRunner.__new__(TPUModelRunner)
                runner.vllm_config, runner.devices = config, selected
                runner._init_mesh()
            meshes[engine] = runner.mesh
        runner = own_runner
        config = runner.vllm_config
        def build():
            original_dtype = config.model_config.dtype
            config.model_config.dtype = jax.numpy.bfloat16
            config.quant_config = model_loader.get_tpu_quantization_config(config)
            cls = model_loader._get_model_architecture(config.model_config.hf_config)
            try:
                with set_current_vllm_config(config), jax.set_mesh(runner.mesh):
                    return nnx.eval_shape(lambda: cls(config, jax.random.key(0), runner.mesh))
            finally:
                config.model_config.dtype = original_dtype
        # Each host needs one graph only, for its own engine. All hosts still
        # construct the SAME global replica sharding and issue the reshard.
        graph = self.timed('abstract_model_rebuild:' + name, build)
        abstract = nnx.state(graph)
        if layout == '2TP8':
            small = meshes['TP8_A']
            replica_mesh = jax.sharding.Mesh(np.stack([small.devices, meshes['TP8_B'].devices]),
                                             ('ici_replica',) + small.axis_names)
            global_specs = nnx.get_named_sharding(abstract, replica_mesh)
            self.global_resident = self.timed('reshard:global_2TP8', lambda: reshard_tree(source, abstract, global_specs))
            specs = nnx.get_named_sharding(abstract, meshes[name])
            state = self.timed('submesh_array_handles:' + name,
                lambda: resident_views([self.global_resident], abstract, specs))
        else:
            specs = nnx.get_named_sharding(abstract, meshes['TP16'])
            state = self.timed('reshard:TP16', lambda: reshard_tree(source, abstract, specs))
            state = self.timed('merged_weight_repacking', lambda: repack_state(state, self.packing, 8, 16))
        nnx.update(graph, state)
        self.resident = {name: state}
        source = None
        for attr in ('vllm_config', 'model_config', 'cache_config', 'parallel_config',
                     'scheduler_config', 'load_config', 'device_config', 'compilation_config',
                     'lora_config', 'speculative_config', 'observability_config', 'kv_transfer_config'):
            setattr(self.worker, attr, config if attr == 'vllm_config' else getattr(config, attr))
        self.worker.devices = list(groups[name])
        self.worker.rank = self.rank if layout == 'TP16' else self.rank % 2
        self.worker.topology_order_id = self.worker.rank
        self.worker.model_runner = runner
        self.timed('model_closure_rebuild', lambda: attach_resident_model(runner, graph))
        self.layout = layout
        gc.collect()
        return self.report()

    def report(self):
        return {'process': self.rank, 'pid': os.getpid(), 'layout': self.layout,
                'phases': list(self.phases), 'failure': self.failure, 'parameter_catalogue': getattr(self, 'catalogue', None),
                'submesh_cache_writer': os.environ.get('ICI_SUBMESH_CACHE_WRITER') == '1',
                'cache_writer_events': list(self.cache_events)}
