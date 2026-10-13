"""Device-only synthetic canonical Gemma4 parameter catalogue and transfers."""
from __future__ import annotations
import ast
import base64
import contextlib
import functools
import hashlib
import json
import os
from pathlib import Path
import time
import zlib
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from hetero.ici_benchmark import slice_meshes
from jax._src import distributed


@functools.cache
def model_catalogue():
    """Measured resident leaf shapes/specs; values remain synthetic BF16."""
    encoded = os.environ.get('ICI_PARAMETER_CATALOGUE')
    raw = zlib.decompress(base64.b64decode(encoded)) if encoded else (
        Path(__file__).resolve().parents[1] / 'results/ici/ICIBench/parameter_catalogue.json').read_bytes()
    value = json.loads(raw)
    value['sha256'] = hashlib.sha256(raw).hexdigest()
    return value


@functools.cache
def schema(size):
    if size != 'model':
        count = {'tiny': 32768, '256MiB': 256 * 2**20, '1GiB': 2**30}[size] // 2
        return {'parameter': ((128, count // 128), P(None, 'model'))}
    # PartitionSpec repr contains only tuple/None/string literals. Never eval.
    return {row['path']: (tuple(row['shape']), P(*ast.literal_eval('(' + row['partition_spec'][2:-1] + ')')))
            for row in model_catalogue()['parameters']}


def normalized(index, shape):
    return tuple((s.start or 0, s.stop if s.stop is not None else n) for s, n in zip(index, shape))


def deterministic(shape, start, seed):
    value = jnp.zeros(shape, jnp.int32)
    for axis, n in enumerate(shape):
        dims = [1] * len(shape)
        dims[axis] = n
        value = value + (jnp.arange(n, dtype=jnp.int32) + start[axis]).reshape(dims) * (axis + 1)
    return ((value + seed) % 127).astype(jnp.bfloat16)


_generate_cache, _check_cache, _slice_cache = {}, {}, {}


def generate(catalogue, mesh):
    tree, compilation_s = {}, 0.
    for seed, (name, (shape, spec)) in enumerate(catalogue.items()):
        sharding = NamedSharding(mesh, spec)
        buffers = []
        for device, index in sharding.addressable_devices_indices_map(shape).items():
            region = normalized(index, shape)
            local_shape = tuple(hi-lo for lo, hi in region)
            start = tuple(lo for lo, _ in region)
            key = (local_shape, device)
            if key not in _generate_cache:
                t = time.perf_counter()
                f = jax.jit(lambda offset, salt, s=local_shape: deterministic(s, offset, salt), device=device)
                _generate_cache[key] = f.lower(jnp.zeros(len(shape), jnp.int32), jnp.int32(0)).compile()
                compilation_s += time.perf_counter()-t
            buffers.append(_generate_cache[key](jnp.asarray(start, jnp.int32), jnp.int32(seed)))
        tree[name] = jax.make_array_from_single_device_arrays(shape, sharding, buffers)
    jax.block_until_ready(tree)
    return tree, compilation_s


def verify(tree, catalogue):
    checks, compilation_s = [], 0.
    for seed, (name, (shape, _)) in enumerate(catalogue.items()):
        for shard in tree[name].addressable_shards:
            start = tuple(lo for lo, _ in normalized(shard.index, shape))
            key = (shard.data.shape, shard.device)
            if key not in _check_cache:
                t = time.perf_counter()
                def check(x, offset, salt):
                    expected = deterministic(x.shape, offset, salt)
                    return jnp.sum(x.astype(jnp.float32)), jnp.sum(x != expected, dtype=jnp.int32)
                _check_cache[key] = jax.jit(check, device=shard.device).lower(shard.data, jnp.asarray(start, jnp.int32), jnp.int32(seed)).compile()
                compilation_s += time.perf_counter()-t
            checksum, errors = _check_cache[key](shard.data, jnp.asarray(start, jnp.int32), jnp.int32(seed))
            count = int(np.asarray(errors))
            if count:
                raise ValueError(f'{name} device {shard.device} has {count} wrong values')
            checks.append({'leaf': name, 'device': shard.device.id, 'checksum': float(np.asarray(checksum)), 'errors': count})
    return checks, compilation_s


def merge_local(source, catalogue, interleaved):
    tree, compilation_s = {}, 0.
    for name, (shape, spec) in catalogue.items():
        target = NamedSharding(interleaved, spec)
        buffers = []
        by_device = {s.device: s for s in source[name].addressable_shards}
        for device, index in target.addressable_devices_indices_map(shape).items():
            shard = by_device[device]
            old, new = normalized(shard.index, shape), normalized(index, shape)
            slices = tuple(slice(lo-old[a][0], hi-old[a][0]) for a, (lo, hi) in enumerate(new))
            if old == new:
                buffers.append(shard.data)
                continue
            key = (shard.data.shape, tuple((s.start, s.stop) for s in slices), device)
            if key not in _slice_cache:
                t = time.perf_counter()
                _slice_cache[key] = jax.jit(lambda x, sl=slices: x[sl], device=device).lower(shard.data).compile()
                compilation_s += time.perf_counter()-t
            buffers.append(_slice_cache[key](shard.data))
        tree[name] = jax.make_array_from_single_device_arrays(shape, target, buffers)
    return tree, compilation_s


def put_tree(tree, catalogue, mesh):
    return {name: jax.device_put(tree[name], NamedSharding(mesh, spec)) for name, (_, spec) in catalogue.items()}


def movement(catalogue, source_mesh, target_mesh):
    received = {}
    for name, (shape, spec) in catalogue.items():
        before = NamedSharding(source_mesh, spec).devices_indices_map(shape)
        after = NamedSharding(target_mesh, spec).devices_indices_map(shape)
        for device, index in after.items():
            target = normalized(index, shape)
            old = normalized(before[device], shape) if device in before else None
            kept = 0 if old is None else int(np.prod([max(0, min(hi, old[a][1])-max(lo, old[a][0])) for a, (lo, hi) in enumerate(target)]))
            total = int(np.prod([hi-lo for lo, hi in target]))
            received[str(device.id)] = received.get(str(device.id), 0) + 2 * (total-kept)
    return {'logical_bytes_moved': sum(received.values()), 'per_device_received_bytes': received,
            'interpretation': 'Exact missing destination bytes; not measured wire bytes or multi-hop traffic'}


def memory():
    return {str(d.id): d.memory_stats() for d in jax.local_devices()}


def barrier(name):
    """Host-only rendezvous: a global TPU allgather can race subset work."""
    if jax.process_count() > 1:
        distributed.global_state.client.wait_at_barrier(name, 300_000)


def host_status(name, message, process_ids=None):
    if jax.process_count() == 1:
        return [message]
    client = distributed.global_state.client
    prefix = 'ici-status/' + name + '/'
    client.key_value_set(prefix + str(jax.process_index()), message)
    ranks = range(jax.process_count()) if process_ids is None else process_ids
    values = [client.blocking_key_value_get(prefix + str(i), 300_000) for i in ranks]
    return [v.decode() if isinstance(v, bytes) else v for v in values]


@contextlib.contextmanager
def implicit_compile_spans(enabled):
    """Cold-only native JAX spans; backend includes compilation or cache restore."""
    spans = []
    def collect(event, start_time, end_time, **metadata):
        if event in {
            '/jax/core/compile/jaxpr_trace_duration',
            '/jax/core/compile/jaxpr_to_mlir_module_duration',
            '/jax/core/compile/backend_compile_duration',
        } and '_device_put_reshard' in metadata.get('fun_name', ''):
            spans.append({'event': event, 'start_s': start_time, 'end_s': end_time,
                          'elapsed_s': end_time - start_time, 'fun_name': metadata['fun_name']})
    if enabled:
        jax.monitoring.register_event_time_span_listener(collect)
    try:
        yield spans
    finally:
        if enabled:
            jax.monitoring.unregister_event_time_span_listener(collect)



def reshard(sizes, repetitions, profile):
    ms, rows, rank = slice_meshes(), [], jax.process_index()
    for size in sizes:
        catalogue = schema(size)
        for route in ('generic', 'replica_aware', 'interleaved'):
            for direction in ('S_to_H', 'H_to_S'):
                source_mesh = ms['replicas'] if direction == 'S_to_H' else ms['interleaved' if route == 'interleaved' else 'h']
                target_mesh = ms['interleaved' if route == 'interleaved' else 'h'] if direction == 'S_to_H' else ms['replicas']
                source, gen_compile = generate(catalogue, source_mesh)
                before = memory()
                _, check_compile = verify(source, catalogue)
                ledger = movement(catalogue, source_mesh, target_mesh)
                for rep in range(repetitions + 1):
                    label = f'{size}-{route}-{direction}-{rep}'
                    barrier(label + '-start')
                    trace = profile and size == 'model' and route == 'generic' and direction == 'S_to_H' and rep == 1
                    trace_dir = f'/tmp/ici-profile-process{rank}'
                    if trace:
                        jax.profiler.start_trace(trace_dir)
                    with implicit_compile_spans(rep == 0) as compilation_events:
                        t = time.perf_counter()
                        route_compile = 0.
                        if direction == 'S_to_H' and route != 'generic':
                            assembled, route_compile = merge_local(source, catalogue, ms['interleaved'])
                            output = assembled if route == 'interleaved' else put_tree(assembled, catalogue, target_mesh)
                        elif direction == 'H_to_S' and route == 'replica_aware':
                            expanded = put_tree(source, catalogue, ms['replicas'])
                            pieces = []
                            for mesh in (ms['a'], ms['b']):
                                piece = {}
                                for name, (shape, spec) in catalogue.items():
                                    sharding = NamedSharding(mesh, spec)
                                    by_device = {s.device: s.data for s in expanded[name].addressable_shards}
                                    piece[name] = jax.make_array_from_single_device_arrays(shape, sharding,
                                        [by_device[d] for d in sharding.addressable_devices_indices_map(shape)], dtype=expanded[name].dtype)
                                pieces.append(piece)
                            output = {}
                            for name, (shape, spec) in catalogue.items():
                                sharding = NamedSharding(ms['replicas'], spec)
                                by_device = {s.device: s.data for tree in pieces for s in tree[name].addressable_shards}
                                output[name] = jax.make_array_from_single_device_arrays(shape, sharding,
                                    [by_device[d] for d in sharding.addressable_devices_indices_map(shape)])
                            assembled = expanded
                            del expanded, pieces, piece, by_device
                        else:
                            assembled = None
                            output = put_tree(source, catalogue, target_mesh)
                        jax.block_until_ready(output)
                        elapsed = time.perf_counter()-t
                    if trace:
                        jax.profiler.stop_trace()
                    after = memory()
                    verified, compilation = verify(output, catalogue)
                    rows.append({'size': size, 'route': route, 'direction': direction, 'repetition': rep,
                                 'cold': rep == 0, 'elapsed_s': elapsed, 'route_compile_s': route_compile,
                                 'implicit_compile_spans': compilation_events if rep == 0 else None,
                                 'generation_compile_s': gen_compile if rep == 0 else 0,
                                 'verification_compile_s': check_compile + compilation if rep == 0 else compilation,
                                 'checksums': verified if rep in (0, repetitions) else None,
                                 'memory_before': before, 'memory_after': after, **ledger,
                                 'effective_GB_s': ledger['logical_bytes_moved'] / elapsed / 1e9,
                                 'trace_dir': trace_dir if trace else None})
                    del output, assembled
                del source
                barrier(f'{size}-{route}-{direction}-done')
    return {'process_index': rank, 'rows': rows,
            'model_schema': {k: {'shape': v[0], 'spec': list(v[1])} for k, v in schema('model').items()},
            'unique_model_bytes': sum(int(np.prod(v[0])) * 2 for v in schema('model').values()),
            'model_source_run': model_catalogue()['run_id'], 'catalogue_sha256': model_catalogue()['sha256'],
            'scope_limit': 'Synthetic values for the complete measured resident leaf catalogue; no TP-specific packed-value conversion, KV or executable buffers'}
