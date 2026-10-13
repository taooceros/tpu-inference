"""Box0 S→H while box1 keeps native TP16 weights and runs a separate program.

v5e reshards need all-device assignment. Source rows are interleaved box0
(local halves of its TP8 replicas) and unchanged native box1. Only box0 has
missing destination bytes. Host-only preflight avoids launching a global TPU
allgather while box1 is still warming its subset program.
"""
import threading
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from hetero.ici_benchmark import meshes, slice_meshes
from hetero.ici_payload import barrier, host_status, generate, merge_local, put_tree, schema, verify, movement, memory


def run_mixed(repetitions):
    if jax.device_count() != 32:
        raise ValueError('mixed experiment requires the full 32-chip runtime')
    local, whole = meshes(), slice_meshes()
    box = jax.local_devices()[0].coords[1] // 4
    catalogue = schema('model')
    if box == 0:
        resident, compile_s = generate(catalogue, local['replicas'])
        prepared, slicing_compile = merge_local(resident, catalogue, local['interleaved'])
        compile_s += slicing_compile
    else:
        resident, compile_s = generate(catalogue, local['h'])
        prepared = resident
    source_mesh = Mesh(np.array([whole['interleaved'].devices[0], whole['h'].devices[1]]), ('box','model'))
    source = {}
    for name, (shape, spec) in catalogue.items():
        sharding = NamedSharding(source_mesh, spec)
        by_device = {s.device: s.data for s in prepared[name].addressable_shards}
        source[name] = jax.make_array_from_single_device_arrays(shape, sharding,
            [by_device[d] for d in sharding.addressable_devices_indices_map(shape)], dtype=prepared[name].dtype)
    verify(source, catalogue)
    barrier('mixed-weights-ready')
    probe, error = {}, ''
    if box == 1:
        try:
            from jax import shard_map
            sharding = NamedSharding(local['h'], P('model'))
            def step(x):
                return jax.lax.psum(x, 'model') / 16 + 1
            f = jax.jit(shard_map(step, mesh=local['h'], in_specs=P('model'), out_specs=P('model'), check_vma=False))
            x = jax.jit(lambda: jnp.zeros((16 * 4096,), jnp.float32), out_shardings=sharding)().block_until_ready()
        except Exception as exc:
            error = str(exc)
    errors = host_status('mixed-subset-initialized', error)
    if any(errors):
        return failure(box, 'subset_compile_failed', errors)
    if box == 1:
        try:
            t = time.perf_counter()
            x = f(x).block_until_ready()
            probe['compute_compile_s'] = time.perf_counter()-t
        except Exception as exc:
            error = str(exc)
    errors = host_status('mixed-subset-compiled', error)
    if any(errors):
        return failure(box, 'subset_compile_failed', errors)
    if box == 1:
        t = time.perf_counter()
        for _ in range(1000):
            x = f(x).block_until_ready()
        probe.update(baseline_steps=1000, baseline_s=time.perf_counter()-t)
    barrier('mixed-baseline-complete')
    start_wall_ns = time.time_ns()
    thread = None
    before = memory()
    if box == 1:
        compute_ranks = sorted({d.process_index for d in local['h'].devices.flat})
        def compute():
            nonlocal x
            events = []
            t = time.perf_counter()
            try:
                for i in range(10000):
                    x = f(x).block_until_ready()
                    if i % 100 == 0:
                        events.append({'step': i+1, 'wall_ns': time.time_ns()})
                elapsed = time.perf_counter()-t
                # No checksum kernel races the last subset collective.
                host_status('mixed-neighbor-steps-done', '', compute_ranks)
                probe.update(steps=10000, elapsed_s=elapsed, expected_value=11001,
                             progress_events=events)
            except Exception as exc:
                probe.update(correct=False, error=str(exc), progress_events=events)
        thread = threading.Thread(target=compute)
        thread.start()
    rows = []
    ledger = movement(catalogue, source_mesh, whole['h'])
    for rep in range(repetitions + 1):
        t, start = time.perf_counter(), time.time_ns()
        output = put_tree(source, catalogue, whole['h'])
        jax.block_until_ready(output)
        elapsed, end = time.perf_counter()-t, time.time_ns()
        rows.append({'repetition': rep, 'elapsed_s': elapsed, 'start_wall_ns': start,
                     'end_wall_ns': end, 'memory_before': before, 'memory': memory(), **ledger})
        if rep != repetitions:
            del output
    if thread:
        thread.join()
    barrier('mixed-programs-finished')
    if box == 1 and 'error' not in probe:
        expected = probe['expected_value']
        probe['correct'] = all(bool(np.asarray(jax.jit(lambda a: jnp.all(a == expected))(s.data))) for s in x.addressable_shards)
    checksum, _ = verify(output, catalogue)
    del output
    end_wall_ns = time.time_ns()
    barrier('mixed-complete')
    return {'process_index': jax.process_index(), 'box': box,
            'role': 'reshard' if box == 0 else 'independent_compute',
            'correct': probe.get('correct', True), 'generation_and_slicing_compile_s': compile_s,
            'start_wall_ns': start_wall_ns, 'end_wall_ns': end_wall_ns,
            'rows': rows, 'output_checksums': checksum, **probe,
            'weight_movement_scope': 'box0 only; box1 source and target have identical native TP16 shard indices'}


def failure(box, role, errors):
    return {'process_index': jax.process_index(), 'box': box, 'role': role,
            'correct': False, 'errors': errors}
