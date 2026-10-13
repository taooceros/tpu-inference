"""Explicit, host-local snapshots of final JAX runner state (no runtime hooks).

Files contain raw shard bytes, including bfloat16's exact bit representation.
The manifest is published last; a missing/partial snapshot is never usable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class SnapshotRefused(ValueError):
    pass


def model_identity(model):
    """Fingerprint configuration contents and checkpoint filesystem identity.

    Avoid rereading the 62 GB checkpoint at restore. Changed checkpoint size or
    mtime refuses restore; immutable checkpoint contents remain an assumption.
    """
    root = Path(model)
    entries = []
    for path in sorted(root.glob('*')):
        if path.is_file() and (path.suffix in ('.json', '.safetensors', '.bin')):
            stat = path.stat()
            row = [path.name, stat.st_size, stat.st_mtime_ns]
            if path.suffix == '.json':
                row.append(hashlib.sha256(path.read_bytes()).hexdigest())
            entries.append(row)
    if not entries:
        raise SnapshotRefused(f'no checkpoint identity files under {model}')
    return {'model_path': str(root.resolve()), 'model_hash': hashlib.sha256(
        json.dumps(entries, separators=(',', ':')).encode()).hexdigest(),
        'hash_kind': 'config-content-and-checkpoint-size-mtime'}


def _devices(mesh):
    return [{'id': int(d.id), 'process_index': int(d.process_index),
             'platform': d.platform} for d in mesh.devices.flat]


def _index(index):
    return [[x.start, x.stop, x.step] if isinstance(x, slice) else int(x) for x in index]


def _spec(spec):
    return [list(x) if isinstance(x, tuple) else x for x in spec]


def _path(path):
    import jax
    return jax.tree_util.keystr(path)


def snapshot(directory, tree, context):
    import jax
    from jax.sharding import NamedSharding

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate any prior publication before replacing individual shards.
    (directory / 'manifest.json').unlink(missing_ok=True)
    started = time.time()
    manifest = {'version': 1, 'context': context, 'arrays': []}
    total = 0
    for number, (path, array) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
        if not isinstance(array, jax.Array) or not isinstance(array.sharding, NamedSharding):
            raise SnapshotRefused(f'{_path(path)} is not a named-sharded JAX array')
        sharding = array.sharding
        row = {'path': _path(path), 'shape': list(array.shape), 'dtype': str(array.dtype),
               'mesh_shape': list(sharding.mesh.devices.shape),
               'mesh_axes': list(sharding.mesh.axis_names),
               'devices': _devices(sharding.mesh), 'spec': _spec(sharding.spec), 'shards': []}
        for shard_number, shard in enumerate(array.addressable_shards):
            host = np.ascontiguousarray(np.asarray(shard.data)).reshape(shard.data.shape)
            filename = f'{number:05d}-{shard_number:02d}.raw'
            raw = memoryview(host.reshape(-1).view(np.uint8)).cast('B')
            checksum = hashlib.sha256(raw).hexdigest()
            with (directory / filename).open('wb') as output:
                output.write(raw)
            total += raw.nbytes
            row['shards'].append({'file': filename, 'device_id': int(shard.device.id),
                                  'index': _index(shard.index), 'shape': list(host.shape),
                                  'bytes': raw.nbytes, 'sha256': checksum})
        manifest['arrays'].append(row)
    temporary = directory / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, directory / 'manifest.json')
    return {'directory': str(directory), 'bytes': total, 'snapshot_s': time.time() - started,
            'arrays': len(manifest['arrays'])}


def prepare(directory, context, mesh, clocks, *, verify_checksums=True):
    """Check identity and local shards; optionally skip byte hashes for an A/B."""
    directory = Path(directory)
    begin = time.time()
    try:
        manifest = json.loads((directory / 'manifest.json').read_text())
        if manifest.get('version') != 1:
            raise SnapshotRefused('unsupported manifest version')
        if manifest['context'] != context:
            fields = sorted(k for k in set(context) | set(manifest['context'])
                            if context.get(k) != manifest['context'].get(k))
            raise SnapshotRefused(f'incompatible context fields: {fields}')
        devices = _devices(mesh)
        for row in manifest['arrays']:
            if row['devices'] != devices or row['mesh_axes'] != list(mesh.axis_names) or row['mesh_shape'] != list(mesh.devices.shape):
                raise SnapshotRefused(f"mesh/device order mismatch: {row['path']}")
            for shard in row['shards']:
                path = directory / shard['file']
                if path.parent != directory or not path.is_file():
                    raise SnapshotRefused(f'missing shard: {path}')
                if path.stat().st_size != shard['bytes']:
                    raise SnapshotRefused(f'shard size mismatch: {path}')
        clocks['snapshot_manifest_verify'] = {'start': begin, 'end': time.time()}
        begin = time.time()
        buffers = []
        for row in manifest['arrays']:
            local = []
            for shard in row['shards']:
                # Map tmpfs directly, avoiding a second full host copy. Hashing
                # faults the pages in; device_put reads the same mapping.
                raw = np.memmap(directory / shard['file'], mode='r', dtype=np.uint8)
                local.append(raw)
            buffers.append(local)
        clocks['snapshot_read_tmpfs'] = {'start': begin, 'end': time.time()}
        begin = time.time()
        try:
            if verify_checksums:
                for row, local in zip(manifest['arrays'], buffers):
                    for shard, raw in zip(row['shards'], local):
                        if hashlib.sha256(raw).hexdigest() != shard['sha256']:
                            raise SnapshotRefused(f"checksum mismatch: {shard['file']}")
        finally:
            # Includes reading/faulting mapped tmpfs bytes; separately visible
            # from mapping setup and host-to-device placement.
            clocks['snapshot_verify'] = {'start': begin, 'end': time.time()}
        return manifest, buffers
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SnapshotRefused(str(exc)) from exc


def validate_tree(abstract_tree, mesh, prepared):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    manifest, _ = prepared
    paths, treedef = jax.tree_util.tree_flatten_with_path(abstract_tree)
    rows = manifest['arrays']
    if len(paths) != len(rows):
        raise SnapshotRefused('parameter count mismatch')
    shardings = []
    for (path, abstract), row in zip(paths, rows):
        if _path(path) != row['path'] or list(abstract.shape) != row['shape'] or str(abstract.dtype) != row['dtype']:
            raise SnapshotRefused(f'parameter path/shape/dtype mismatch: {_path(path)}')
        spec = PartitionSpec(*(tuple(x) if isinstance(x, list) else x for x in row['spec']))
        sharding = NamedSharding(mesh, spec)
        expected = sharding.addressable_devices_indices_map(tuple(row['shape']))
        shards = row['shards']
        if {s['device_id'] for s in shards} != {int(d.id) for d in expected} or len(shards) != len(expected):
            raise SnapshotRefused(f"missing/duplicate shard devices: {row['path']}")
        for shard in shards:
            device = next(d for d in expected if int(d.id) == shard['device_id'])
            if shard['index'] != _index(expected[device]):
                raise SnapshotRefused(f"shard index mismatch: {row['path']}")
            shape = [len(range(*index.indices(dim))) for index, dim in
                     zip(expected[device], row['shape']) if isinstance(index, slice)]
            if shard['shape'] != shape or shard['bytes'] != int(np.prod(shape)) * abstract.dtype.itemsize:
                raise SnapshotRefused(f"shard shape/byte count mismatch: {row['path']}")
        shardings.append(sharding)
    return treedef, shardings


def restore(abstract_tree, mesh, prepared, clocks):
    import jax
    import jax.numpy as jnp

    treedef, shardings = validate_tree(abstract_tree, mesh, prepared)
    manifest, buffers = prepared
    rows = manifest['arrays']
    begin = time.time()
    arrays = []
    for row, local, sharding in zip(rows, buffers, shardings):
        by_id = {s['device_id']: (s, raw) for s, raw in zip(row['shards'], local)}
        device_arrays = []
        for device in sharding.addressable_devices:
            shard, raw = by_id[int(device.id)]
            host = raw.view(np.dtype(jnp.dtype(row['dtype']))).reshape(shard['shape'])
            device_arrays.append(jax.device_put(host, device))
        array = jax.make_array_from_single_device_arrays(tuple(row['shape']), sharding, device_arrays)
        arrays.append(array)
    jax.block_until_ready(arrays)
    clocks['snapshot_device_placement'] = {'start': begin, 'end': time.time()}
    return jax.tree_util.tree_unflatten(treedef, arrays)
