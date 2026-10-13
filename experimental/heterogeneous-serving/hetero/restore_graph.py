"""Restore tail without the serial graph build.

The snapshot restore in ``tpu_worker._load_snapshot_or_normal`` runs four
dependent steps back to back: hash the tmpfs shards, build the abstract nnx
graph (``nnx.eval_shape`` of the model constructor), validate the manifest
against that graph, then ``device_put`` the shards. Only the last step needs
the TPU; the first two are host work on independent inputs. This module lets
the caller:

* start host-to-device placement straight from the manifest, before any
  abstract graph exists (``place_from_manifest``);
* run a GIL-releasing verification (sha256 of memory-mapped tmpfs pages) in a
  worker thread while the calling thread traces the model constructor
  (``build_while_verifying``);
* attach the placed arrays to the abstract graph once it exists, with the
  same manifest validation as ``weight_snapshot.restore`` (``assemble``).

Any validation failure after placement drops the device arrays before the
caller falls back to the normal loader, so a refused snapshot leaves no HBM
behind.
"""
from __future__ import annotations

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .weight_snapshot import SnapshotRefused, validate_tree


@contextlib.contextmanager
def cheap_tracer_sharding_errors():
    """Make ``Tracer.sharding`` fail fast while an nnx graph is traced.

    jax 0.10.x builds the full tracer-origin message (``_origin_msg`` ->
    ``find_progenitors`` over every equation traced so far) before raising the
    ``AttributeError`` that ``getattr(x, 'sharding', None)`` swallows. flax
    0.12's ``Variable.set_value`` does that ``getattr`` twice for every
    ``rngs.params()`` counter update, so ``nnx.eval_shape`` of a model with N
    parameters costs O(N^2) equation walks: ~13 s of a 14 s build for the
    1028-parameter Gemma-4-31B graph on one CPU host. jax 0.11 dropped the
    origin message from this property; this restores that behaviour for the
    duration of the block only.
    """
    from jax._src import core

    original = core.Tracer.__dict__.get('sharding')
    if not isinstance(original, property):
        yield
        return

    def sharding(self):
        raise AttributeError(
            f"The 'sharding' attribute is not available on {self._error_repr()}.")

    core.Tracer.sharding = property(sharding)
    try:
        yield
    finally:
        core.Tracer.sharding = original


def _sharding(mesh, row):
    from jax.sharding import NamedSharding, PartitionSpec
    spec = PartitionSpec(*(tuple(x) if isinstance(x, list) else x for x in row['spec']))
    return NamedSharding(mesh, spec)


def place_from_manifest(manifest, buffers, mesh):
    """Issue ``device_put`` for every local shard using manifest metadata only.

    Returns the global arrays in manifest order without blocking; transfers
    proceed while the caller does other host work. Shard membership and
    indices are checked here, so a wrong manifest fails before any transfer.
    """
    import jax
    import jax.numpy as jnp

    arrays = []
    for row, local in zip(manifest['arrays'], buffers):
        sharding = _sharding(mesh, row)
        shape = tuple(row['shape'])
        expected = sharding.addressable_devices_indices_map(shape)
        by_id = {s['device_id']: (s, raw) for s, raw in zip(row['shards'], local)}
        if set(by_id) != {int(d.id) for d in expected} or len(row['shards']) != len(expected):
            raise SnapshotRefused(f"missing/duplicate shard devices: {row['path']}")
        dtype = np.dtype(jnp.dtype(row['dtype']))
        device_arrays = []
        for device in sharding.addressable_devices:
            shard, raw = by_id[int(device.id)]
            if shard['bytes'] != int(np.prod(shard['shape'])) * dtype.itemsize or raw.nbytes != shard['bytes']:
                raise SnapshotRefused(f"shard byte count mismatch: {row['path']}")
            host = raw.view(dtype).reshape(shard['shape'])
            # Same ownership rule as weight_snapshot.restore: CPU device_put aliases
            # the tmpfs map, which Prepared.close() unmaps after restore/fallback.
            if device.platform == 'cpu':
                host = np.array(host, copy=True, order='C')
            device_arrays.append(jax.device_put(host, device))
        arrays.append(jax.make_array_from_single_device_arrays(shape, sharding, device_arrays))
    return arrays


def build_while_verifying(build, verify):
    """Run ``verify`` on a worker thread while ``build`` runs on this thread.

    ``build`` must run on the calling thread: JAX mesh/config contexts are
    thread-local. ``verify`` should release the GIL (hashlib does for buffers
    larger than 2 KiB), otherwise the two merely interleave. Returns
    ``(build_result, verify_result)``; an exception from either propagates
    after both have finished.
    """
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='snapshot-verify') as pool:
        pending = pool.submit(verify)
        try:
            built = build()
        finally:
            # Always drain the thread so a failed build does not leave the
            # verifier racing the fallback loader for tmpfs pages.
            verified = pending.result()
    return built, verified


def assemble(abstract_tree, mesh, prepared, arrays, clocks):
    """Validate the abstract graph against the manifest and attach placed arrays.

    Blocks on the transfers and records ``snapshot_device_placement`` as the
    wait from here to readiness, so the clock no longer includes host-side
    dispatch that already ran concurrently with the graph build.
    """
    import jax

    manifest, _ = prepared
    try:
        treedef, shardings = validate_tree(abstract_tree, mesh, prepared)
        if len(arrays) != len(shardings):
            raise SnapshotRefused('placed array count does not match manifest')
        for array, sharding, row in zip(arrays, shardings, manifest['arrays']):
            if not array.sharding.is_equivalent_to(sharding, array.ndim):
                raise SnapshotRefused(f"placed sharding differs from graph sharding: {row['path']}")
    except Exception:
        arrays.clear()
        raise
    begin = time.time()
    jax.block_until_ready(arrays)
    clocks['snapshot_device_placement'] = {'start': begin, 'end': time.time()}
    return jax.tree_util.tree_unflatten(treedef, arrays)
