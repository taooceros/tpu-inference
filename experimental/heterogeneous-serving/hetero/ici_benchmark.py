"""Synthetic BF16, resident-weight reshard measurements (no checkpoint I/O)."""
from __future__ import annotations
import dataclasses
import importlib.metadata
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


@dataclasses.dataclass(frozen=True)
class RelativeDevice:
    device: object
    coords: tuple
    device_kind: str
    core_on_chip: int


def native_order(devices):
    origin = tuple(min(d.coords[a] for d in devices) for a in range(3))
    wrappers = [RelativeDevice(d, tuple(d.coords[a] - origin[a] for a in range(3)),
                               d.device_kind, d.core_on_chip) for d in devices]
    # tpu_runner._create_single_slice_mesh: singleton axes + model TP axis.
    ordered = mesh_utils.create_device_mesh((1, 1, 1, 1, len(devices), 1, 1), wrappers,
                                             allow_split_physical_axes=True)
    return [d.device for d in ordered.flat]


def meshes():
    devices = jax.devices()
    if devices[0].platform == 'cpu':
        small_a, small_b = devices[:8], devices[8:16]
        large = [small_a[0], small_a[1], small_b[0], small_b[1], small_b[3], small_b[2], small_a[3], small_a[6], small_b[6], small_b[7], small_b[4], small_b[5], small_a[4], small_a[5], small_a[7], small_a[2]]
    else:
        box = [d for d in devices if d.coords[1] // 4 == jax.local_devices()[0].coords[1] // 4]
        large = native_order(box)
        xmin = min(d.coords[0] for d in box)
        small_a = native_order([d for d in box if d.coords[0] < xmin + 2])
        small_b = native_order([d for d in box if d.coords[0] >= xmin + 2])
    interleaved = [d for pair in zip(small_a, small_b) for d in pair]
    return {'a': Mesh(np.array(small_a), ('model',)),
            'b': Mesh(np.array(small_b), ('model',)),
            'h': Mesh(np.array(large), ('model',)),
            'interleaved': Mesh(np.array(interleaved), ('model',)),
            'replicas': Mesh(np.array([small_a, small_b]), ('replica', 'model'))}

def slice_meshes():
    """Use all 32 devices: libtpu v5e rejects subset reshard assignments.

    The unused box axis represents identical model replicas. Both boxes
    reshard concurrently, but communication groups stay box-local.
    """
    devices = jax.devices()
    if len(devices) == 16:
        return meshes()
    rows = []
    for box_index in (0, 1):
        if devices[0].platform == 'cpu':
            base = meshes()
            order = {k: [devices[d.id + 16 * box_index] for d in base[k].devices.flat]
                     for k in ('a', 'b', 'h', 'interleaved')}
        else:
            box = [d for d in devices if d.coords[1] // 4 == box_index]
            a = native_order([d for d in box if d.coords[0] < 2])
            b = native_order([d for d in box if d.coords[0] >= 2])
            order = {'a': a, 'b': b, 'h': native_order(box),
                     'interleaved': [d for pair in zip(a,b) for d in pair]}
        rows.append(order)
    result = {key: Mesh(np.array([r[key] for r in rows]), ('box','model'))
              for key in ('a','b','h','interleaved')}
    result['replicas'] = Mesh(np.array([[r['a'],r['b']] for r in rows]), ('box','replica','model'))
    return result


def orders():
    ms = meshes()
    def fact(d):
        return {'id': d.id, 'coords': list(getattr(d, 'coords', [])), 'process_index': d.process_index,
                'kind': d.device_kind}
    return {'jax': jax.__version__, 'jaxlib': importlib.metadata.version('jaxlib'),
            'libtpu': importlib.metadata.version('libtpu') if jax.default_backend() == 'tpu' else None,
            'process_index': jax.process_index(), 'devices': [fact(d) for d in jax.devices()],
            'orders': {k: [fact(d) for d in v.devices.flat] for k, v in ms.items()}}


def subgroup():
    rank = jax.process_index()
    which = 'a' if rank % 4 < 2 else 'b'
    group = [d for d in jax.devices() if d.process_index // 2 == rank // 2]
    mesh = Mesh(np.array(native_order(group)), ('model',))
    sharding = NamedSharding(mesh, P('model'))
    count = 2 if which == 'a' else 5
    from jax import shard_map
    def body(x):
        for _ in range(count):
            x = jax.lax.psum(x, 'model') + 1
        return x
    f = jax.jit(shard_map(body, mesh=mesh, in_specs=P('model'), out_specs=P('model'), check_vma=False))
    x = jax.jit(lambda: jnp.ones((8 * 16,), jnp.float32), out_shardings=sharding)()
    t = time.perf_counter()
    y = f(x).block_until_ready()
    expected = 1
    for _ in range(count):
        expected = 8 * expected + 1
    checks = [bool(np.asarray(jax.jit(lambda a: jnp.all(a == expected))(s.data))) for s in y.addressable_shards]
    from hetero.ici_payload import barrier
    barrier('subgroup_complete')
    return {'process_index': rank, 'mesh': which, 'collective_count': count, 'elapsed_s': time.perf_counter()-t,
            'expected': expected, 'correct': all(checks)}


def run(mode, sizes, repetitions, profile):
    if mode == 'subgroup':
        return subgroup()
    if mode == 'mixed':
        from hetero.ici_mixed import run_mixed
        return run_mixed(repetitions)
    if mode == 'reshard':
        from hetero.ici_payload import reshard
        return reshard(sizes, repetitions, profile)
    raise ValueError(f'Unknown mode {mode}')
