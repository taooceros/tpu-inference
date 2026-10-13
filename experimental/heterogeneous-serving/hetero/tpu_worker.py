"""Instance-local TPU worker: rank remapping and effective-environment evidence.

vLLM's TPU backend creates one Ray worker actor per *host* (4 chips), not per
chip, and forcibly sets ``parallel_config.worker_cls`` inside
``TpuPlatform.check_and_update_config``.  The engine actor therefore overrides
``worker_cls`` *after* config creation and ships the resolved layout through
``vllm_config.additional_config['omp_hetero']`` so that each worker can:

* apply this instance's subslice TPU environment before any device init,
* translate the slice-wide physical ``TPU_WORKER_ID`` into an instance-local
  process rank derived from the ordered ``TPU_PROCESS_ADDRESSES``, and
* report the *effective* environment and real device membership as evidence.

Carrying the payload on the config rather than on the process environment keeps
this correct regardless of Ray's environment-variable copy allowlist.
"""

from __future__ import annotations

import os
import time
import socket
from typing import Any, Dict, List, Optional, Sequence

import ray

from tpu_inference.worker.tpu_worker import TPUWorker  # type: ignore

from .native_backend import BACKEND_REVISION, RUNTIME_REVISION, verify_backend
from .compile_cache import configure_compile_cache, verify_compile_cache

EVIDENCE_ENV_KEYS = (
    "TPU_MULTIHOST_BACKEND",
    "TPU_TOPOLOGY",
    "TPU_HOST_BOUNDS",
    "TPU_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_VISIBLE_CHIPS",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_NAME",
    "CLOUD_TPU_TASK_ID",
    "JAX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
    "VLLM_XLA_CACHE_PATH",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_LOG_COMPILES",
    "JAX_EXPLAIN_CACHE_MISSES",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "OMP_HETERO_INSTANCE_ID",
)


class HeteroTPUWorker(TPUWorker):
    """TPU worker bound to one logical serving instance's execution domain."""

    # Engine start-up phases are timed for the reconfiguration-cost breakdown.
    # Explicit overrides, not __getattribute__: worker attributes are on the step path.
    def _timed(self, name, call, *args, **kwargs):
        started = time.time()
        try:
            return call(*args, **kwargs)
        finally:
            # Not "omp_init_phases": an instance attribute of that name would shadow the RPC method.
            self.__dict__.setdefault("_omp_phase_clocks", {})[name] = {
                "start": started, "end": time.time()}

    def determine_available_memory(self, *args, **kwargs):  # type: ignore[override]
        return self._timed("determine_available_memory", super().determine_available_memory,
                           *args, **kwargs)

    def initialize_from_config(self, *args, **kwargs):  # type: ignore[override]
        verify_compile_cache(self._omp_compile_cache_expected)
        return self._startup_profiled("initialize_from_config", super().initialize_from_config,
                                     *args, **kwargs)

    def compile_or_warm_up_model(self, *args, **kwargs):  # type: ignore[override]
        verify_compile_cache(self._omp_compile_cache_expected)
        return self._startup_profiled("compile_or_warm_up_model", super().compile_or_warm_up_model,
                                     *args, **kwargs)

    def _startup_profiled(self, name, call, *args, **kwargs):
        payload = self._hetero_payload()
        bundle = payload.get("startup_compile_bundle")
        trace = payload.get("startup_device_trace", False)
        if not payload.get("startup_profile", False) and not bundle and not trace:
            return self._timed(name, call, *args, **kwargs)
        import jax
        import shutil
        import tempfile
        from contextlib import ExitStack
        from tpu_inference.runner import kv_cache_manager
        from .compile_overlap import StartupProfile, summarize_device_trace
        profile = StartupProfile()
        manager = getattr(self.model_runner, "compilation_manager", None)
        marker = "omp-startup:" + name
        directory = tempfile.mkdtemp(prefix="omp-overlap-trace-") if trace else None
        tracing = False
        setup_start = start = time.time()
        setup_s = 0.0
        try:
            if trace:
                options = jax.profiler.ProfileOptions()
                options.python_tracer_level = 0
                options.host_tracer_level = 2
                jax.profiler.start_trace(directory, profiler_options=options)
                tracing = True
                setup_s = time.time() - setup_start
            start = time.time()
            with ExitStack() as stack:
                stack.enter_context(profile.hooks(manager=manager, allocation_module=kv_cache_manager))
                if bundle:
                    from pathlib import Path
                    from scripts.overlap_replay import capture
                    stack.enter_context(capture(Path(bundle) / payload["instance_id"] / name))
                if trace:
                    stack.enter_context(jax.profiler.TraceAnnotation(marker))
                result = self._timed(name, call, *args, **kwargs)
                if name == "initialize_from_config":
                    jax.block_until_ready(self.model_runner.kv_caches)
                return result
        finally:
            end = time.time()
            report = profile.report(start, end)
            from jax._src import distributed
            report["pjrt_process_index"] = jax.process_index()
            report["cache_writer_process_id"] = distributed.global_state.process_id
            report["compilation_cache_dir"] = jax.config.jax_compilation_cache_dir
            report["persistent_cache_min_compile_time_secs"] = jax.config.jax_persistent_cache_min_compile_time_secs
            report["persistent_cache_min_entry_size_bytes"] = jax.config.jax_persistent_cache_min_entry_size_bytes
            report["jax_version"] = jax.__version__
            report["trace_options"] = ({"python_tracer_level": options.python_tracer_level,
                                         "host_tracer_level": options.host_tracer_level} if trace else None)
            report["host"] = self.omp_hostname
            report["node_ip"] = self.omp_node_ip
            report["pid"] = os.getpid()
            export_start = time.time()
            try:
                if tracing:
                    jax.profiler.stop_trace()
                    report["device_trace"] = summarize_device_trace(directory, marker)
            finally:
                if directory:
                    shutil.rmtree(directory)
                if trace:
                    report["trace_diagnostic_setup_s"] = setup_s
                    report["trace_diagnostic_export_parse_cleanup_s"] = time.time() - export_start
                self.__dict__.setdefault("_omp_startup_profiles", {})[name] = report

    def _hetero_payload(self) -> Dict[str, Any]:
        additional = getattr(self.vllm_config, "additional_config", None) or {}
        payload = additional.get("omp_hetero")
        if not payload:
            raise RuntimeError(
                "HeteroTPUWorker requires vllm_config.additional_config['omp_hetero']; "
                "the engine actor must attach the resolved instance layout"
            )
        return payload

    def init_device(self, *args, **kwargs):  # type: ignore[override]
        payload = self._hetero_payload()
        local_ip = ray.util.get_node_ip_address()

        host_order: List[str] = list(payload["host_ips"])
        if len(set(host_order)) != len(host_order):
            raise RuntimeError(f"instance host order repeats a host: {host_order}")
        if local_ip not in host_order:
            raise RuntimeError(
                f"worker on {local_ip} is not a member of instance "
                f"{payload['instance_id']} hosts {host_order}"
            )
        process_rank = host_order.index(local_ip)

        addresses = payload["runtime_env"]["TPU_PROCESS_ADDRESSES"].split(",")
        if [a.rsplit(":", 1)[0] for a in addresses] != host_order:
            raise RuntimeError(
                "TPU_PROCESS_ADDRESSES order does not match the resolved host order"
            )
        if self.rank == 0 and process_rank != 0:
            raise RuntimeError(
                f"executor rank 0 landed on {local_ip} (process rank {process_rank}); "
                "coordinator and rank zero must agree"
            )

        self.omp_physical_tpu_worker_id = os.environ.get("TPU_WORKER_ID")
        self.omp_process_rank = process_rank
        self.omp_instance_id = payload["instance_id"]
        self.omp_node_ip = local_ip
        self.omp_hostname = socket.gethostname()
        self.omp_intended_env = dict(payload["runtime_env"])
        self.omp_intended_env["CLOUD_TPU_TASK_ID"] = str(process_rank)
        self.omp_intended_env["TPU_WORKER_ID"] = str(process_rank)

        # Applied before super().init_device() because that call performs JAX/PJRT
        # device initialization and the libtpu rendezvous.
        for key, value in self.omp_intended_env.items():
            os.environ[key] = value

        if payload.get("backend_revision") != BACKEND_REVISION:
            raise RuntimeError("worker backend revision does not match the engine configuration")
        if payload.get("runtime_revision") != RUNTIME_REVISION:
            raise RuntimeError("worker runtime revision does not match the engine configuration")
        self.omp_backend_evidence = verify_backend()
        additional = self.vllm_config.additional_config
        for field in ("activation_dtype", "matmul_precision"):
            if additional.get(field) != payload.get(field):
                raise RuntimeError(f"worker {field} does not match the engine configuration")
        if additional.get("activation_dtype") != "float32":
            raise RuntimeError("native Gemma4 FP32-state backend requires activation_dtype=float32")
        if additional.get("matmul_precision") not in ("default", "high", "highest"):
            raise RuntimeError("unsupported worker matmul_precision")
        if self.vllm_config.cache_config.cache_dtype != payload.get("kv_cache_dtype"):
            raise RuntimeError("worker cache dtype does not match the engine configuration")
        import jax

        # Set before native runner construction or any model tracing, not in a
        # thread-local context that expires before the execution thread runs.
        jax.config.update("jax_default_matmul_precision", additional["matmul_precision"])
        # TPUWorker imports JAX before this instance payload is applied. Updating
        # os.environ alone cannot change JAX's already-initialized config states.
        self._omp_compile_cache_expected = configure_compile_cache(self.omp_intended_env)
        result = self._timed("init_device", super().init_device, *args, **kwargs)
        verify_compile_cache(self._omp_compile_cache_expected)

        self.omp_effective_env = {k: os.environ.get(k) for k in EVIDENCE_ENV_KEYS}
        self.omp_device_evidence = self._collect_device_evidence(payload)
        return result

    def load_model(self):
        mode = self._hetero_payload().get("weight_snapshot", "off")
        self.omp_snapshot_status = {
            "mode": mode, "restored": False,
            "checksums_requested": self._hetero_payload().get("weight_snapshot_verify", True),
            "verified_checksums": False, "verified_bytes": 0,
        }
        if mode == "read":
            result = self._timed("load_model", self._load_snapshot_or_normal)
        else:
            result = self._timed("load_model", super().load_model)
        self.omp_precision_evidence = self._verify_model_precision()
        if mode == "write":
            self.omp_snapshot_status["snapshot"] = self.omp_snapshot_weights()
        return result

    def _snapshot_location(self):
        from pathlib import Path
        payload = self._hetero_payload()
        return Path("/dev/shm/omp_weights") / payload["layout_id"] / payload["instance_id"] / f"rank{self.omp_process_rank}"

    def _snapshot_context(self):
        from .weight_snapshot import model_identity
        payload = self._hetero_payload()
        config = self.vllm_config.model_config
        return {
            **model_identity(config.model),
            "tp_size": payload["tp_size"],
            "dtype": str(config.dtype).removeprefix("torch."),
            "quantization": getattr(config, "quantization", None),
            "backend_revision": payload["backend_revision"],
            "runtime_revision": payload["runtime_revision"],
            "host_ips": payload["host_ips"],
            "chip_coords_by_host": payload["chip_coords_by_host"],
        }

    def omp_snapshot_weights(self, generation=None):
        from .weight_snapshot import snapshot
        generation = generation or self._hetero_payload().get("weight_snapshot_generation")
        report = self._timed("snapshot_write", snapshot, self._snapshot_location(),
                             self.model_runner.state, self._snapshot_context(), generation=generation)
        return {"executor_rank": self.rank, "process_rank": self.omp_process_rank,
                "node_ip": self.omp_node_ip, **report}

    def omp_weight_fingerprint(self):
        from .weight_snapshot import fingerprint
        return {"executor_rank": self.rank, "process_rank": self.omp_process_rank,
                "node_ip": self.omp_node_ip, **fingerprint(self.model_runner.state)}

    def _load_snapshot_or_normal(self):
        """Restore before closure capture, without an inference-step hook.

        All processes vote before loader/placement so one invalid host cannot
        enter normal checkpoint collectives while its peers take another path.
        The load-time builder patch is removed unconditionally.
        """
        import logging
        import jax
        import numpy as np
        from flax import nnx
        from jax.experimental import multihost_utils
        from tpu_inference.models.common import model_loader
        from .weight_snapshot import SnapshotRefused, prepare, restore, validate_tree

        clocks = self.__dict__.setdefault("_omp_phase_clocks", {})
        runner = self.model_runner
        configured_dtype = runner.model_config.dtype
        error = None
        prepared = None
        try:
            prepared = prepare(self._snapshot_location(), self._snapshot_context(),
                               runner.mesh, clocks,
                               verify_checksums=self.omp_snapshot_status["checksums_requested"])
            manifest = prepared[0]
            checksum_kind = manifest.get('checksum_kind', 'sha256')
            total_bytes = sum(s['bytes'] for a in manifest['arrays'] for s in a['shards'])
            verified = checksum_kind != 'none' and self.omp_snapshot_status['checksums_requested']
            self.omp_snapshot_status.update(
                manifest_version=manifest['version'], checksum_kind=checksum_kind,
                bytes=total_bytes, verified_checksums=verified,
                verified_bytes=total_bytes if verified else 0)
        except Exception as exc:
            error = str(exc)

        def agree(local_error):
            votes = multihost_utils.process_allgather(np.int32(local_error is None))
            if not np.all(votes):
                raise SnapshotRefused(local_error or "another worker refused its snapshot")

        original_builder = model_loader._get_nnx_model

        def snapshot_builder(model_class, config, rng, mesh, pooler=None, is_draft_model=False):
            problem = None
            abstract = None
            try:
                if is_draft_model or config.model_config.quantization is not None:
                    raise SnapshotRefused("draft/quantized snapshot graph construction is unsupported")
                with jax.set_mesh(mesh):
                    abstract = nnx.eval_shape(lambda: model_class(config, rng, mesh))
                validate_tree(nnx.state(abstract), mesh, prepared)
            except Exception as exc:
                problem = str(exc)
            agree(problem)
            restored = restore(nnx.state(abstract), mesh, prepared, clocks)
            nnx.update(abstract, restored)
            self.omp_snapshot_status["restored"] = True
            return abstract

        try:
            agree(error)
            vote_start = time.time()
            manifest = prepared[0]
            generation = bytes.fromhex(manifest['generation']) if manifest['version'] == 2 else bytes(16)
            token = np.frombuffer(bytes([manifest['version']]) + generation, dtype=np.uint8)
            epochs = np.asarray(multihost_utils.process_allgather(token)).reshape(-1, len(token))
            clocks['snapshot_generation_vote'] = {'start': vote_start, 'end': time.time()}
            if not np.all(epochs == epochs[0]):
                raise SnapshotRefused("mixed snapshot generations across workers")
            self.omp_snapshot_status['generation'] = manifest.get('generation')
            model_loader._get_nnx_model = snapshot_builder
            return super().load_model()
        except SnapshotRefused as exc:
            if prepared is not None:
                prepared.close()
            self.omp_snapshot_status.update(restored=False, fallback_reason=str(exc))
            logging.getLogger(__name__).warning("weight snapshot refused; normal loader: %s", exc)
            runner.model_config.dtype = configured_dtype
            model_loader._get_nnx_model = original_builder
            return super().load_model()
        finally:
            model_loader._get_nnx_model = original_builder
            if prepared is not None:
                prepared.close()

    def _verify_model_precision(self) -> Dict[str, Any]:
        """Inspect once after loading, before cache allocation and captured execution."""
        from collections import Counter

        import jax
        import jax.numpy as jnp
        from flax import nnx
        from tpu_inference.models.jax.gemma4 import Gemma4Model

        runner = self.model_runner
        payload = self._hetero_payload()
        if runner.kv_caches:
            raise RuntimeError("model precision must be verified before cache allocation")
        configured_dtype = str(runner.model_config.dtype).removeprefix("torch.")
        requested_dtype = payload["dtype"]
        requested_dtype = {"half": "float16", "float": "float32"}.get(
            requested_dtype, requested_dtype
        )
        if requested_dtype != "auto" and configured_dtype != requested_dtype:
            raise RuntimeError("native model parameter dtype differs from requested dtype")
        parameter_dtypes = dict(Counter(
            str(array.dtype)
            for array in jax.tree_util.tree_leaves(nnx.state(runner.model, nnx.Param))
        ))
        if not parameter_dtypes or set(parameter_dtypes) != {configured_dtype}:
            raise RuntimeError(
                f"model parameters {parameter_dtypes} do not match {configured_dtype}"
            )

        roots = [module for _, module in nnx.iter_graph(runner.model)
                 if isinstance(module, Gemma4Model)]
        if len(roots) != 1:
            raise RuntimeError(f"expected one native Gemma4 text backbone, found {len(roots)}")
        root = roots[0]
        activation_dtype = str(jnp.dtype(root.activation_dtype))
        if activation_dtype != payload["activation_dtype"]:
            raise RuntimeError("native model activation dtype differs from requested dtype")
        norm_dtypes = Counter()
        attention_dtypes = Counter()
        for layer in root.layers:
            for name in (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                dtype = getattr(layer, name).dtype
                if dtype is None or str(jnp.dtype(dtype)) != activation_dtype:
                    raise RuntimeError(f"decoder {name} is not configured for {activation_dtype}")
                norm_dtypes[str(jnp.dtype(dtype))] += 1
            attention_dtypes[str(jnp.dtype(layer.self_attn.kv_cache_quantized_dtype))] += 1
        if not norm_dtypes:
            raise RuntimeError("native model has no decoder norm precision evidence")
        cache_dtype = runner.cache_config.cache_dtype
        runner_cache_dtype = str(runner.kv_cache_dtype).removeprefix("torch.")
        expected_cache_dtype = configured_dtype if cache_dtype == "auto" else cache_dtype
        if cache_dtype != payload["kv_cache_dtype"] or runner_cache_dtype != expected_cache_dtype:
            raise RuntimeError("native runner KV dtype differs from the configured cache dtype")
        if set(attention_dtypes) != {expected_cache_dtype}:
            raise RuntimeError("native attention KV dtype differs from the runner cache dtype")
        precision = jax.config.jax_default_matmul_precision
        if precision != payload["matmul_precision"]:
            raise RuntimeError("effective JAX matmul precision differs from requested precision")
        return {
            "verified_before_cache_allocation": True,
            "model_dtype": configured_dtype,
            "parameter_dtypes": parameter_dtypes,
            "activation_dtype": activation_dtype,
            "decoder_norm_dtypes": dict(norm_dtypes),
            "attention_cache_dtypes": dict(attention_dtypes),
            "configured_cache_dtype": cache_dtype,
            "runner_cache_dtype": runner_cache_dtype,
            "matmul_precision": precision,
            "backend_revision": BACKEND_REVISION,
            "runtime_revision": RUNTIME_REVISION,
        }

    def _collect_device_evidence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Qualifies the execution domain the runtime actually built.

        A subslice runtime re-bases chip coordinates on its own origin, so the
        parent-slice absolute coordinates are not directly comparable.  What must
        hold is structural: the runtime sees exactly this instance's chip count
        and box shape, this host owns exactly one host-sized sub-box, and that
        sub-box sits at the same *relative* position inside the instance box that
        the resolved layout assigned to this host.
        """
        import jax

        local = jax.local_devices()
        glob = jax.devices()
        selected = list(self.devices or [])
        observed_local = sorted(tuple(int(v) for v in d.coords) for d in local)
        observed_global = sorted(tuple(int(v) for v in d.coords) for d in glob)

        parent_local = [tuple(c) for c in payload["chip_coords_by_host"][self.omp_node_ip]]
        parent_global = [tuple(c) for c in payload["chip_coords"]]
        instance_origin = tuple(min(c[axis] for c in parent_global) for axis in range(3))
        expected_relative = sorted(
            tuple(c[axis] - instance_origin[axis] for axis in range(3)) for c in parent_local
        )
        expected_global_relative = sorted(
            tuple(c[axis] - instance_origin[axis] for axis in range(3)) for c in parent_global
        )

        if len(glob) != int(payload["tp_size"]):
            raise RuntimeError(
                f"{payload['instance_id']} runtime sees {len(glob)} devices, expected "
                f"{payload['tp_size']}; unintended rendezvous outside the execution domain "
                f"(observed coords {observed_global})"
            )
        if len(local) != len(parent_local):
            raise RuntimeError(
                f"{payload['instance_id']} worker on {self.omp_node_ip} owns {len(local)} "
                f"chips, expected {len(parent_local)}"
            )
        if _box_dims(observed_global) != _box_dims(expected_global_relative):
            raise RuntimeError(
                f"{payload['instance_id']} runtime box {_box_dims(observed_global)} does not "
                f"match the resolved subslice box {_box_dims(expected_global_relative)} "
                f"(observed {observed_global})"
            )
        relative_match = observed_local == expected_relative
        if not relative_match:
            raise RuntimeError(
                f"{payload['instance_id']} worker on {self.omp_node_ip} (process rank "
                f"{self.omp_process_rank}) owns runtime chips {observed_local}; the resolved "
                f"layout places it at relative chips {expected_relative} inside instance "
                f"origin {instance_origin}"
            )
        return {
            "jax_process_index": int(jax.process_index()),
            "jax_process_count": int(jax.process_count()),
            "global_device_count": len(glob),
            "local_device_count": len(local),
            "instance_chip_origin_in_parent": list(instance_origin),
            "runtime_global_chip_coords": [list(c) for c in observed_global],
            "runtime_local_chip_coords": [list(c) for c in observed_local],
            "parent_local_chip_coords": [list(c) for c in sorted(parent_local)],
            "expected_relative_local_coords": [list(c) for c in expected_relative],
            "relative_position_match": relative_match,
            "selected_device_ids": [int(d.id) for d in selected],
            "selected_chip_coords": [list(int(v) for v in d.coords) for d in selected],
            "topology_order_id": int(getattr(self, "topology_order_id", -1)),
            "hbm_gb": _hbm(selected or local),
        }

    def omp_worker_evidence(self) -> Dict[str, Any]:
        """Collective-RPC entry point returning this worker's qualification record."""
        return {
            "instance_id": getattr(self, "omp_instance_id", None),
            "node_ip": getattr(self, "omp_node_ip", None),
            "hostname": getattr(self, "omp_hostname", None),
            "physical_tpu_worker_id": getattr(self, "omp_physical_tpu_worker_id", None),
            "instance_process_rank": getattr(self, "omp_process_rank", None),
            "executor_rank": getattr(self, "rank", None),
            "executor_local_rank": getattr(self, "local_rank", None),
            "pid": os.getpid(),
            "intended_env": getattr(self, "omp_intended_env", {}),
            "effective_env": getattr(self, "omp_effective_env", {}),
            "devices": getattr(self, "omp_device_evidence", {}),
            "native_backend": getattr(self, "omp_backend_evidence", {}),
            "precision": getattr(self, "omp_precision_evidence", {}),
            "weight_snapshot": getattr(self, "omp_snapshot_status", {}),
        }

    def omp_init_phases(self) -> Dict[str, Any]:
        """Report startup clocks and the compile-cache configuration actually used."""
        import jax
        from jax._src import distributed

        verify_compile_cache(self._omp_compile_cache_expected)
        cache_dir = jax.config.jax_compilation_cache_dir
        return {"executor_rank": getattr(self, "rank", None),
                "weight_snapshot": getattr(self, "omp_snapshot_status", {}),
                "startup_profiles": self.__dict__.get("_omp_startup_profiles", {}),
                "phases": self.__dict__.get("_omp_phase_clocks", {}),
                "compile_cache": {
                    "dir": cache_dir,
                    "min_compile_time_secs": jax.config.jax_persistent_cache_min_compile_time_secs,
                    "min_entry_size_bytes": jax.config.jax_persistent_cache_min_entry_size_bytes,
                    "log_compiles": jax.config.jax_log_compiles,
                    "explain_cache_misses": jax.config.jax_explain_cache_misses,
                    "writer_process_id": distributed.global_state.process_id,
                    "hostname": getattr(self, "omp_hostname", None),
                    "node_ip": getattr(self, "omp_node_ip", None),
                    "pid": os.getpid(),
                }}

    def omp_kv_cache_evidence(self) -> Dict[str, Any]:
        """Reports the worker-side KV cache shape actually allocated."""
        runner = getattr(self, "model_runner", None)
        caches = getattr(runner, "kv_caches", None)
        rows = []
        if caches:
            for index, cache in enumerate(caches):
                shape = getattr(cache, "shape", None)
                dtype = getattr(cache, "dtype", None)
                rows.append(
                    {
                        "layer_index": index,
                        "shape": list(shape) if shape is not None else None,
                        "dtype": str(dtype) if dtype is not None else None,
                        "nbytes": int(getattr(cache, "nbytes", 0) or 0),
                    }
                )
        return {
            "node_ip": getattr(self, "omp_node_ip", None),
            "backend_revision": BACKEND_REVISION,
            "configured_cache_dtype": getattr(getattr(runner, "cache_config", None), "cache_dtype", None),
            "runner_cache_dtype": str(getattr(runner, "kv_cache_dtype", None)),
            "cache_dtypes": sorted({r["dtype"] for r in rows}),
            "num_cache_arrays": len(rows),
            "caches": rows[:4],
            "total_cache_bytes": sum(r["nbytes"] for r in rows),
        }

    # ------------------------------------------------- host-staged KV transfer

    def omp_kv_layout(self) -> Dict[str, Any]:
        """Collective RPC describing native groups, physical arrays and device slices."""
        from .kv_transport import cache_layout

        return cache_layout(self.model_runner)

    def _omp_kv_staging(self) -> Dict[Any, Any]:
        staging = getattr(self, "_omp_kv_staging_state", None)
        if staging is None:
            staging = self._omp_kv_staging_state = {}
        return staging

    def omp_export_gather(self, transaction_id: str, addressing: Dict[str, Any],
                          start: int, budget_bytes: int, notifier: Any) -> Dict[str, Any]:
        """Device gather of one chunk; caller must drain and exclude execution.

        Host staging continues on a background thread and reports to notifier.
        """
        from .kv_transport import export_gather

        return export_gather(self.model_runner, self._omp_kv_staging(), transaction_id,
                             addressing, start, budget_bytes, notifier)

    def omp_import_advance(self, transaction_id: str, commit: bool, start: int,
                           budget_bytes: int, notifier: Any,
                           addressing: Optional[Dict[str, Any]] = None,
                           payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Scatter a staged chunk and/or queue the next; caller must drain execution."""
        from .kv_transport import import_advance

        return import_advance(self.model_runner, self._omp_kv_staging(), transaction_id,
                              commit=commit, start=start, budget_bytes=budget_bytes,
                              notifier=notifier, addressing=addressing, payload=payload)

    def omp_kv_discard(self, transaction_id: str) -> bool:
        from .kv_transport import discard

        discard(self._omp_kv_staging(), transaction_id)
        return True

    def omp_kv_debug(self) -> Dict[str, Any]:
        """Background transport thread stacks and staging progress, for fail-closed errors."""
        import sys
        import threading
        import traceback

        names = {thread.ident: thread.name for thread in threading.enumerate()}
        stacks = {names[ident]: "".join(traceback.format_stack(frame)[-12:])
                  for ident, frame in sys._current_frames().items()
                  if names.get(ident, "").startswith("omp-kv")}
        staging = {str(key): {k: v for k, v in value.items()
                              if k in ("cursor", "pending", "progress", "error")}
                   for key, value in self._omp_kv_staging().items() if isinstance(value, dict)}
        return {"hostname": socket.gethostname(), "threads": stacks, "staging": staging}

    def omp_resume_fenced_request_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore authoritative token history before native resumed scheduling.

        The actor must drain the execution queue and exclude stepping first.
        Native async output retirement can skip a fenced request after it leaves
        the persistent batch, retaining placeholder IDs in CachedRequestState.
        Finalize pending host bookkeeping exactly as native zero-token execution
        does, then repair only this resident request. The scheduler must emit a
        resumed cached-request row so native update_states rebuilds input_batch.
        This does not touch KV, allocation, sampling/RNG state or output budget.
        """
        runner = self.model_runner
        request_id = payload["request_id"]
        state = runner.requests.get(request_id)
        if state is None:
            raise ValueError("fenced source request is not resident on this worker")
        prompt = list(payload["prompt_token_ids"])
        output = list(payload["generated_token_ids"])
        computed = payload["num_computed_tokens"]
        if not prompt or prompt != state.prompt_token_ids:
            raise ValueError("fenced source prompt identity changed or is empty")
        if any(not isinstance(token, int) or token < 0 for token in output):
            raise ValueError("fenced source output contains placeholder token IDs")
        if not isinstance(computed, int) or not 0 <= computed < len(prompt) + len(output):
            raise ValueError("fenced source needs an uncomputed continuation token")
        if state.mm_features or state.lora_request or getattr(runner, "uses_mrope", False):
            raise ValueError("multimodal, LoRA and M-RoPE source resumption is unsupported")
        blocks = tuple(list(group) for group in payload["block_ids"])
        if len(blocks) != len(runner.kv_cache_config.kv_cache_groups):
            raise ValueError("source resumption requires every native cache-group block table")
        pending = getattr(runner, "_pre_async_results", None) is not None
        if pending:
            runner._modify_prev_results()
            runner._pre_async_results = None
        # Keep the native object: outstanding metadata and continuation ownership
        # may reference it. Native resumed scheduling handles batch reinstallation.
        state.output_token_ids[:] = output
        state.num_computed_tokens = computed
        state.block_ids = blocks
        return {"request_id": request_id, "history_restored": True,
                "num_computed_tokens": computed, "num_generated_tokens": len(output),
                "last_token_id": output[-1] if output else prompt[-1],
                "native_async_results_flushed": pending}


    def omp_prepare_continuation_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Seed a PREEMPTED native request without admitting it to the batch.

        The actor supplies native SamplingParams and full native block tables,
        including sliding-window null entries. It must keep this request fenced
        until all workers acknowledge preparation and any KV import completes.
        """
        from tpu_inference.runner.input_batch import CachedRequestState
        from vllm.sampling_params import SamplingParams

        runner = self.model_runner
        request_id = payload["request_id"]
        identity = (payload["transaction_id"], payload["destination_generation"])
        if not request_id or not all(identity):
            raise ValueError("continuation requires request, transaction and generation identity")
        records = getattr(self, "_omp_pending_continuations", None)
        if records is None:
            records = self._omp_pending_continuations = {}
        # Native completion removes runner.requests; do not retain token histories.
        for key in list(records):
            if key not in runner.requests:
                del records[key]
        prior = records.get(request_id)
        if prior is not None:
            if prior["identity"] != identity:
                raise ValueError("continuation belongs to another transaction/generation")
            if runner.requests.get(request_id) is not prior["state"]:
                raise ValueError("native continuation state was replaced")
            return {"request_id": request_id, "prepared": True, "active": prior["active"]}
        if request_id in runner.requests or request_id in runner.input_batch.req_id_to_index:
            raise ValueError("native request ID is already resident")
        prompt = list(payload["prompt_token_ids"])
        output = list(payload["generated_token_ids"])
        computed = payload["num_computed_tokens"]
        if not prompt or not output or not isinstance(computed, int) or not 0 <= computed < len(prompt) + len(output):
            raise ValueError("continuation needs token history and an uncomputed token")
        params = payload["sampling_params"]
        if not isinstance(params, SamplingParams):
            raise ValueError("continuation requires native SamplingParams")
        if payload.get("mm_features") or payload.get("lora_request") or getattr(runner, "uses_mrope", False):
            raise ValueError("multimodal, LoRA and M-RoPE continuations are unsupported")
        blocks = tuple(list(group) for group in payload["block_ids"])
        if len(blocks) != len(runner.kv_cache_config.kv_cache_groups):
            raise ValueError("continuation block tables must cover every native cache group")
        state = CachedRequestState(
            req_id=request_id, prompt_token_ids=prompt, output_token_ids=output,
            mm_features=[], sampling_params=params, pooling_params=None,
            generator=None, block_ids=blocks, num_computed_tokens=computed,
            lora_request=None,
        )
        runner.requests[request_id] = state
        records[request_id] = {"identity": identity, "state": state, "active": False}
        return {"request_id": request_id, "prepared": True, "active": False}

    def omp_activate_continuation_state(
        self, request_id: str, transaction_id: str, destination_generation: str
    ) -> Dict[str, Any]:
        """Commit preparation ownership before the scheduler resumes execution."""
        record = getattr(self, "_omp_pending_continuations", {}).get(request_id)
        if record is None or record["identity"] != (transaction_id, destination_generation):
            raise ValueError("no continuation prepared under this transaction/generation")
        if self.model_runner.requests.get(request_id) is not record["state"]:
            raise ValueError("native continuation state was replaced")
        record["active"] = True
        return {"request_id": request_id, "active": True}

    def omp_abort_continuation_state(
        self, request_id: str, transaction_id: str, destination_generation: str
    ) -> Dict[str, Any]:
        """Discard unactivated worker state; native scheduler owns block freeing."""
        records = getattr(self, "_omp_pending_continuations", {})
        record = records.get(request_id)
        if record is None:
            return {"request_id": request_id, "aborted": False}
        if record["identity"] != (transaction_id, destination_generation):
            raise ValueError("continuation belongs to another transaction/generation")
        runner = self.model_runner
        if record["active"] or request_id in runner.input_batch.req_id_to_index:
            raise ValueError("cannot abort activated continuation state")
        if runner.requests.get(request_id) is not record["state"]:
            raise ValueError("native continuation state was replaced")
        del runner.requests[request_id]
        del records[request_id]
        return {"request_id": request_id, "aborted": True}


def _box_dims(coords: Sequence[Any]) -> tuple:
    if not coords:
        return (0, 0, 0)
    return tuple(
        max(c[axis] for c in coords) - min(c[axis] for c in coords) + 1 for axis in range(3)
    )

def _hbm(devices: List[Any]) -> Optional[float]:
    try:
        from tpu_inference import utils  # type: ignore

        return float(utils.hbm_usage_gb(devices))
    except Exception:
        return None
