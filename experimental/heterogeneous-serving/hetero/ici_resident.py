"""Load-time resident-state cutover helpers; no serving-path instrumentation."""
from __future__ import annotations


def reshard_tree(source, abstract, shardings):
    """Keep destination metadata and match source arrays by path, not treedef.

    Every source/destination controller must call this in the same order. Never
    convert non-addressable arrays to host buffers. Destination variable metadata
    can differ between TP sizes, so retaining the source treedef is incorrect.
    """
    import jax
    from jax.sharding import NamedSharding
    source_pairs = jax.tree_util.tree_flatten_with_path(source)[0]
    destination_pairs, destination_def = jax.tree_util.tree_flatten_with_path(abstract)
    sharding_pairs = jax.tree_util.tree_flatten_with_path(
        shardings, is_leaf=lambda x: isinstance(x, NamedSharding))[0]
    by_path = {jax.tree_util.keystr(path): array for path, array in source_pairs}
    specs = {jax.tree_util.keystr(path): spec for path, spec in sharding_pairs}
    target_paths = {jax.tree_util.keystr(path) for path, _ in destination_pairs}
    if set(by_path) != target_paths or set(specs) != target_paths:
        raise ValueError('resident parameter paths differ from destination graph')
    arrays = []
    for path, target in destination_pairs:
        key = jax.tree_util.keystr(path)
        value = by_path[key]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f'resident parameter shape/dtype mismatch: {key}')
        arrays.append(jax.device_put(value, specs[key]))
    jax.block_until_ready(arrays)
    return jax.tree_util.tree_unflatten(destination_def, arrays)


def attach_resident_model(runner, model):
    """Rebuild native loader closures around an already populated NNX graph."""
    from tpu_inference.models.common import model_loader
    original = model_loader._get_nnx_model

    def resident_builder(model_class, config, rng, mesh, **kwargs):
        if kwargs.get('is_draft_model') or config.model_config.quantization is not None:
            raise ValueError('prototype supports only the unquantized primary model')
        if mesh != runner.mesh:
            raise ValueError('resident model attached to a different mesh')
        return model

    try:
        model_loader._get_nnx_model = resident_builder
        runner.load_model()
    finally:
        model_loader._get_nnx_model = original


def repack_merged(value, sizes, source_tp, target_tp):
    """Convert the merged projection layout without host staging."""
    import jax.numpy as jnp
    if sum(sizes) != value.shape[-1] or any(n % source_tp or n % target_tp for n in sizes):
        raise ValueError('merged projection sizes do not match TP degrees')
    grouped = value.reshape(value.shape[:-1] + (source_tp, -1))
    parts = []
    offset = 0
    for size in sizes:
        width = size // source_tp
        parts.append(grouped[..., offset:offset + width].reshape(value.shape[:-1] + (size,)))
        offset += width
    return jnp.concatenate([part.reshape(value.shape[:-1] + (target_tp, size // target_tp))
                            for part, size in zip(parts, sizes)], axis=-1).reshape(value.shape)


def merged_layouts(graph):
    from flax import nnx
    from tpu_inference.layers.jax.linear import JaxMergedColumnParallelLinear
    result = {}
    for path, module in nnx.iter_graph(graph):
        if isinstance(module, JaxMergedColumnParallelLinear):
            for attr in ('weight', 'bias'):
                if getattr(module, attr, None) is not None:
                    result[path + (attr,)] = tuple(module.output_sizes)
    return result


def repack_state(state, layouts, source_tp, target_tp):
    import jax
    kernels = {}
    def convert(path, array):
        keys = tuple(getattr(key, 'key', getattr(key, 'idx', getattr(key, 'name', None))) for key in path)
        module_path = keys[:-1] if keys and keys[-1] == 'value' else keys
        if module_path not in layouts:
            return array
        sizes = layouts[module_path]
        key = (sizes, array.sharding)
        if key not in kernels:
            kernels[key] = jax.jit(lambda x: repack_merged(x, sizes, source_tp, target_tp),
                                   out_shardings=array.sharding)
        converted = kernels[key](array)
        converted.block_until_ready()
        return converted
    return jax.tree_util.tree_map_with_path(convert, state)



def resident_views(sources, abstract, shardings):
    """Rebind identical device shards to submesh/global handles without transfer."""
    import jax
    from jax.sharding import NamedSharding
    source_maps = [dict((jax.tree_util.keystr(p), a) for p, a in jax.tree_util.tree_flatten_with_path(s)[0]) for s in sources]
    targets, treedef = jax.tree_util.tree_flatten_with_path(abstract)
    specs = dict((jax.tree_util.keystr(p), a) for p, a in jax.tree_util.tree_flatten_with_path(
        shardings, is_leaf=lambda x: isinstance(x, NamedSharding))[0])
    arrays = []
    def index_key(index, shape):
        return tuple((s.start or 0, s.stop if s.stop is not None else n, s.step or 1)
                     if isinstance(s, slice) else s for s, n in zip(index, shape))
    for path, target in targets:
        key = jax.tree_util.keystr(path)
        by_device = {}
        for source in source_maps:
            array = source[key]
            if array.shape != target.shape or array.dtype != target.dtype:
                raise ValueError('resident view shape/dtype mismatch: ' + key)
            for shard in array.addressable_shards:
                by_device[shard.device] = shard
        sharding = specs[key]
        buffers = []
        for device, index in sharding.addressable_devices_indices_map(target.shape).items():
            shard = by_device[device]
            if index_key(shard.index, target.shape) != index_key(index, target.shape):
                raise ValueError('resident view changes device shard index: ' + key)
            buffers.append(shard.data)
        arrays.append(jax.make_array_from_single_device_arrays(target.shape, sharding, buffers, dtype=target.dtype))
    return jax.tree_util.tree_unflatten(treedef, arrays)


def install_submesh_cache_writer(events):
    """Opt-in JAX 0.10.2 compile-time patch; never mutate distributed runtime state.

    TPU cache keys retain device assignments. One lowest participating process
    writes each key; disjoint TPU meshes have different keys. Preserve the native
    callback, duration, serialization and error policy in a private function copy.
    """
    import contextvars
    import functools
    import types
    import jax
    from jax._src import compiler, distributed
    if jax.__version__ != '0.10.2':
        raise RuntimeError('submesh cache writer is qualified only for JAX 0.10.2')
    participants = contextvars.ContextVar('ici_cache_participants', default=None)
    original_compile = compiler.compile_or_get_cached
    original_write = compiler._cache_write
    globals_copy = dict(original_write.__globals__)
    globals_copy['distributed'] = types.SimpleNamespace(
        global_state=types.SimpleNamespace(process_id=0))
    permitted_write = types.FunctionType(original_write.__code__, globals_copy,
        original_write.__name__, original_write.__defaults__, original_write.__closure__)
    permitted_write.__kwdefaults__ = original_write.__kwdefaults__

    @functools.wraps(original_compile)
    def compile_with_participants(backend, computation, devices, *args, **kwargs):
        token = participants.set((backend.platform, tuple(sorted({device.process_index for device in devices.flat}))))
        try:
            return original_compile(backend, computation, devices, *args, **kwargs)
        finally:
            participants.reset(token)

    @functools.wraps(original_write)
    def write_from_mesh_leader(cache_key, *args, **kwargs):
        context = participants.get()
        if context is None or context[0] == 'gpu':
            return original_write(cache_key, *args, **kwargs)
        platform, group = context
        rank = distributed.global_state.process_id
        eligible = rank == min(group)
        events.append({'cache_key': cache_key, 'backend': platform, 'process': rank,
            'participating_processes': list(group), 'eligible_writer': eligible})
        if eligible:
            return permitted_write(cache_key, *args, **kwargs)

    compiler.compile_or_get_cached = compile_with_participants
    compiler._cache_write = write_from_mesh_leader


