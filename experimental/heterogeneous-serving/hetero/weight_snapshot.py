"""Explicit, host-local snapshots of final JAX runner state (no runtime hooks).

Files contain raw shard bytes, including bfloat16's exact bit representation.
The manifest is published last; a missing/partial snapshot is never usable.
Version 2 defaults to explicit no-byte-checksum operation with identity checks;
SHA-256 remains available, and version 1 SHA-256 manifests remain readable.
"""
from __future__ import annotations

import fcntl
import mmap
import uuid
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


class SnapshotRefused(ValueError):
    pass


def _close_buffers(buffers):
    for local in buffers:
        for raw in local:
            mapped = getattr(raw, '_mmap', None)
            if mapped is not None and not mapped.closed:
                mapped.close()


class Prepared(tuple):
    """Tuple-compatible manifest/maps with a reader lock through H2D readiness."""

    def __new__(cls, manifest, buffers, lock):
        self = super().__new__(cls, (manifest, buffers))
        self.lock = lock
        return self

    def close(self):
        if self.lock is not None:
            try:
                _close_buffers(self[1])
            finally:
                self.lock.close()
                self.lock = None

    def __del__(self):
        if getattr(self, 'lock', None) is not None:
            self.close()


def _lock(directory, *, exclusive):
    lock = (directory / '.snapshot.lock').open('a+b')
    try:
        fcntl.flock(lock, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BaseException as exc:
        lock.close()
        if isinstance(exc, BlockingIOError):
            raise SnapshotRefused('snapshot is busy') from exc
        raise
    return lock


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def open_for_write(directory):
    """Exclusive writer lock on ``directory`` and the old manifest invalidated (unlinked + synced).

    Done before any reused shard byte is touched, including on a later failure. Readers hold the
    same stable lock (shared) through H2D completion. Returns the lock; the caller closes it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = _lock(directory, exclusive=True)
    try:
        (directory / 'manifest.json').unlink(missing_ok=True)
        _sync_directory(directory)
    except BaseException:
        lock.close()
        raise
    return lock


def publish_manifest(directory, manifest):
    """Publish ``manifest`` atomically (temp + fsync + rename + directory sync) under the writer lock."""
    directory = Path(directory)
    temporary = directory / f"manifest.{manifest['generation']}.tmp"
    try:
        with temporary.open('x') as output:
            json.dump(manifest, output, separators=(',', ':'))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / 'manifest.json')
        _sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


class ShardFile:
    """One reusable shard file: allocate, copy, flush/fsync, record its mtime identity.

    ``allocate`` reserves real pages before mmap (fallocate), so ENOSPC is a catchable error
    rather than a SIGBUS half-way through the copy. Use as a context manager: the map and the
    descriptor are closed on every path.
    """

    def __init__(self, path, nbytes):
        self.path, self.nbytes = Path(path), nbytes
        self.fd = self.mapped = None
        self.reused = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.mapped is not None:
            self.mapped.close()
            self.mapped = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def allocate(self):
        self.reused = self.path.is_file() and self.path.stat().st_size == self.nbytes
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(self.fd, self.nbytes)
        if self.nbytes:
            os.posix_fallocate(self.fd, 0, self.nbytes)
            self.mapped = mmap.mmap(self.fd, self.nbytes, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                                    prot=mmap.PROT_READ | mmap.PROT_WRITE)

    def copy(self, shape, host):
        """Typed copy into the file (handles strided sources directly in the destination)."""
        if self.mapped is None:
            target = np.empty(tuple(shape), dtype=host.dtype)
        else:
            target = np.ndarray(tuple(shape), dtype=host.dtype, buffer=self.mapped)
        np.copyto(target, host, casting='no')
        del target

    def finish(self):
        """Flush and fsync; returns the file's mtime_ns (the v2 identity check)."""
        if self.mapped is not None:
            self.mapped.flush()
            self.mapped.close()
            self.mapped = None
        os.fsync(self.fd)
        return os.fstat(self.fd).st_mtime_ns


def write_shard(directory, part, host):
    """Write one shard (``part``: file/shape/bytes) from a host array; sets ``part['mtime_ns']``."""
    if host.nbytes != part['bytes'] or list(host.shape) != list(part['shape']):
        raise SnapshotRefused(f"shard {part['file']}: host array does not match its shape/byte count")
    with ShardFile(Path(directory) / part['file'], part['bytes']) as shard:
        shard.allocate()
        shard.copy(part['shape'], host)
        part['mtime_ns'] = shard.finish()
        return shard.reused


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


def fingerprint(tree):
    """Validation-only SHA-256 of every live device shard, never timed as a snapshot."""
    import jax
    from jax.sharding import NamedSharding

    begin = time.perf_counter()
    groups = {}
    paths = jax.tree_util.tree_flatten_with_path(tree)[0]
    for number, (path, array) in enumerate(paths):
        if not isinstance(array, jax.Array) or not isinstance(array.sharding, NamedSharding):
            raise SnapshotRefused(f'{_path(path)} is not a named-sharded JAX array')
        for shard_number, shard in enumerate(array.addressable_shards):
            row = {'array': number, 'shard': shard_number, 'path': _path(path),
                   'device_id': int(shard.device.id), 'index': _index(shard.index),
                   'shape': list(shard.data.shape), 'dtype': str(shard.data.dtype),
                   'bytes': shard.data.nbytes}
            groups.setdefault(int(shard.device.id), []).append((shard.data, row))
            shard.data.copy_to_host_async()

    def hash_chip(items):
        rows = []
        for data, row in items:
            host, _ = data._single_device_array_to_np_array_did_copy()
            raw = np.ascontiguousarray(host).reshape(-1).view(np.uint8)
            row['sha256'] = hashlib.sha256(raw).hexdigest()
            rows.append(row)
        return rows

    with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
        rows = [row for chip in pool.map(hash_chip, groups.values()) for row in chip]
    rows.sort(key=lambda r: (r['array'], r['shard']))
    return {'bytes': sum(r['bytes'] for r in rows), 'arrays': len(paths), 'shards': rows,
            'sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            'validation_s': time.perf_counter() - begin,
            'outside_snapshot_and_restore_timer': True}


def file_fingerprint(manifest_path, *, threads=4):
    """Validation-only SHA-256 of every shard file of one published snapshot (shared lock held).

    Shard rows are the ``scripts/snapshot_validate.py file-fingerprint`` rows (array, shard, path,
    device_id, index, shape, dtype, bytes, sha256). ``array_meta`` adds the per-array layout
    (path, shape, dtype, spec, mesh shape/axes, devices) for exactness comparisons.
    """
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    with (directory / '.snapshot.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest = json.loads(manifest_path.read_text())
        jobs = [(number, shard_number, array, shard)
                for number, array in enumerate(manifest['arrays'])
                for shard_number, shard in enumerate(array['shards'])]

        def hash_file(job):
            number, shard_number, array, shard = job
            file = directory / shard['file']
            if file.parent != directory or file.stat().st_size != shard['bytes']:
                raise RuntimeError(f'missing/wrong-sized shard: {file}')
            raw = np.memmap(file, mode='r', dtype=np.uint8) if shard['bytes'] else np.empty(0, np.uint8)
            try:
                digest = hashlib.sha256(raw).hexdigest()
            finally:
                if isinstance(raw, np.memmap):
                    raw._mmap.close()
            return {'array': number, 'shard': shard_number, 'path': array['path'],
                    'device_id': shard['device_id'], 'index': shard['index'],
                    'shape': shard['shape'], 'dtype': array['dtype'],
                    'bytes': shard['bytes'], 'sha256': digest}

        with ThreadPoolExecutor(max_workers=threads) as pool:
            shards = list(pool.map(hash_file, jobs))
    shards.sort(key=lambda s: (s['array'], s['shard']))
    return {'layout': directory.parent.parent.name, 'instance': directory.parent.name,
            'process_rank': int(directory.name.removeprefix('rank')),
            'bytes': sum(s['bytes'] for s in shards), 'shards': shards,
            'sha256': hashlib.sha256(json.dumps(shards, sort_keys=True).encode()).hexdigest(),
            'generation': manifest.get('generation'),
            'array_meta': [{k: a[k] for k in ('path', 'shape', 'dtype', 'spec', 'mesh_shape',
                                              'mesh_axes', 'devices')} for a in manifest['arrays']]}


def snapshot(directory, tree, context, *, checksum='none', generation=None):
    """Copy all local shards to reusable, shared files; publish only on completion.

    Transfers are queued together. Each chip then allocates/maps and copies its
    files concurrently. A typed copy handles PJRT's strided host layouts directly
    in the destination, without an intermediate contiguous host allocation.
    """
    import jax
    from jax.sharding import NamedSharding

    if checksum not in ('none', 'sha256'):
        raise SnapshotRefused('unsupported checksum kind')
    generation = uuid.uuid4().hex if generation is None else uuid.UUID(hex=generation).hex
    begin = time.perf_counter()
    directory = Path(directory)
    # Invalidate before touching any reused bytes, including on validation
    # failure. Readers hold this same stable lock through H2D completion.
    lock = open_for_write(directory)
    try:
        manifest = {'version': 2, 'context': context, 'generation': generation,
                    'checksum_kind': checksum, 'arrays': []}
        groups = {}
        cached_arrays = cached_bytes = 0
        for number, (path, array) in enumerate(jax.tree_util.tree_flatten_with_path(tree)[0]):
            if not isinstance(array, jax.Array) or not isinstance(array.sharding, NamedSharding):
                raise SnapshotRefused(f'{_path(path)} is not a named-sharded JAX array')
            sharding = array.sharding
            row = {'path': _path(path), 'shape': list(array.shape), 'dtype': str(array.dtype),
                   'mesh_shape': list(sharding.mesh.devices.shape),
                   'mesh_axes': list(sharding.mesh.axis_names),
                   'devices': _devices(sharding.mesh), 'spec': _spec(sharding.spec), 'shards': []}
            for shard_number, shard in enumerate(array.addressable_shards):
                data = shard.data
                part = {'file': f'{number:05d}-{shard_number:02d}.raw',
                        'device_id': int(shard.device.id), 'index': _index(shard.index),
                        'shape': list(data.shape), 'bytes': int(data.nbytes)}
                row['shards'].append(part)
                groups.setdefault(int(shard.device.id), []).append((data, part))
                if data._npy_value is not None:
                    cached_arrays += 1
                    cached_bytes += part['bytes']
            manifest['arrays'].append(row)
        metadata_end = time.perf_counter()
        submit_start = time.perf_counter()
        for items in groups.values():
            for data, _ in items:
                data.copy_to_host_async()
        submit_end = time.perf_counter()

        def copy_chip(items):
            stats = {'device_id': items[0][1]['device_id'], 'bytes': 0, 'files': 0,
                     'reused_files': 0, 'new_files': 0, 'did_copy_arrays': 0,
                     'strided_arrays': 0, 'allocation_s': 0., 'transfer_wait_s': 0.,
                     'copy_s': 0., 'checksum_s': 0., 'flush_s': 0.,
                     'transfer_end_s': submit_end - submit_start}
            for data, part in items:
                tick = time.perf_counter()
                with ShardFile(directory / part['file'], part['bytes']) as shard:
                    shard.allocate()
                    stats['allocation_s'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    # The exposed native collector avoids filling _npy_value
                    # with a second 31 GB Python cache. JAX 0.10.2/0.11.2 expose
                    # it; unlike a caller-owned D2H buffer it still returns a
                    # runtime-owned ndarray (see the bandwidth API evidence).
                    host, did_copy = data._single_device_array_to_np_array_did_copy()
                    acquired = time.perf_counter()
                    stats['transfer_wait_s'] += acquired - tick
                    stats['transfer_end_s'] = acquired - submit_start
                    stats['did_copy_arrays'] += int(did_copy)
                    stats['strided_arrays'] += int(not host.flags.c_contiguous)
                    tick = time.perf_counter()
                    shard.copy(part['shape'], host)
                    stats['copy_s'] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    if checksum == 'sha256':
                        # Source-derived: also detects an incorrect destination
                        # copy, not just corruption after manifest publication.
                        logical = np.ascontiguousarray(host).reshape(-1).view(np.uint8)
                        part['sha256'] = hashlib.sha256(logical).hexdigest()
                        del logical
                    del host
                    stats['checksum_s'] += time.perf_counter() - tick if checksum != 'none' else 0.
                    tick = time.perf_counter()
                    part['mtime_ns'] = shard.finish()
                    stats['flush_s'] += time.perf_counter() - tick
                    stats['bytes'] += part['bytes']
                    stats['files'] += 1
                    stats['reused_files' if shard.reused else 'new_files'] += 1
            return stats

        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
            chip_times = list(pool.map(copy_chip, groups.values()))
        publish_start = time.perf_counter()
        publish_manifest(directory, manifest)
        published = time.perf_counter()
        active = lambda name: max((c[name] for c in chip_times), default=0.)
        return {'directory': str(directory), 'bytes': sum(c['bytes'] for c in chip_times),
                'snapshot_s': published - begin, 'arrays': len(manifest['arrays']),
                'generation': generation, 'manifest_version': 2, 'checksum_kind': checksum,
                'host_cache_arrays_before': cached_arrays, 'host_cache_bytes_before': cached_bytes,
                'source_freshness': 'live immutable weights; Python cache counted, PJRT runtime caching on repeats is possible',
                'reused_files': sum(c['reused_files'] for c in chip_times),
                'new_files': sum(c['new_files'] for c in chip_times), 'chip_timings': chip_times,
                'timings': {'metadata_s': metadata_end - begin, 'enqueue_s': submit_end - submit_start,
                            'transfer_s': active('transfer_end_s'),
                            'transfer_wait_s': active('transfer_wait_s'),
                            'allocation_s': active('allocation_s'), 'copy_s': active('copy_s'),
                            'checksum_s': active('checksum_s'), 'flush_s': active('flush_s'),
                            'publish_s': published - publish_start},
                'phase_semantics': 'snapshot_s includes publication; transfer_s is submit to last host acquisition (overlaps and includes preceding allocation/copies); wait/allocation/copy/checksum/flush are max per-chip active sums, not additive wall phases'}
    finally:
        lock.close()


def prepare(directory, context, mesh, clocks, *, verify_checksums=True):
    """Validate identity and map a complete snapshot, holding a shared lock.

    Version 1 SHA-256 snapshots remain readable. Version 2 states its checksum
    policy explicitly and uses file identity checks even when byte hashes are
    disabled. The returned Prepared owns the maps until restore/close.
    """
    directory = Path(directory)
    begin = time.time()
    lock = None
    buffers = []
    try:
        lock = _lock(directory, exclusive=False)
        manifest = json.loads((directory / 'manifest.json').read_text())
        version = manifest.get('version')
        if type(version) is not int or version not in (1, 2):
            raise SnapshotRefused('unsupported manifest version')
        kind = 'sha256' if version == 1 else manifest['checksum_kind']
        if kind not in ('none', 'sha256'):
            raise SnapshotRefused('unsupported checksum kind')
        if version == 2:
            generation = manifest['generation']
            if not isinstance(generation, str) or len(generation) != 32 or len(bytes.fromhex(generation)) != 16:
                raise SnapshotRefused('invalid snapshot generation')
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
                if path.parent != directory or not path.is_file() or path.is_symlink():
                    raise SnapshotRefused(f'missing shard: {path}')
                stat = path.stat()
                if type(shard['bytes']) is not int or stat.st_size != shard['bytes']:
                    raise SnapshotRefused(f'shard size mismatch: {path}')
                if version == 2 and kind == 'none' and stat.st_mtime_ns != shard['mtime_ns']:
                    raise SnapshotRefused(f'shard identity mismatch: {path}')
        clocks['snapshot_manifest_verify'] = {'start': begin, 'end': time.time()}
        begin = time.time()
        for row in manifest['arrays']:
            local = []
            buffers.append(local)
            for shard in row['shards']:
                raw = (np.memmap(directory / shard['file'], mode='r', dtype=np.uint8)
                       if shard['bytes'] else np.empty(0, dtype=np.uint8))
                local.append(raw)
        clocks['snapshot_read_tmpfs'] = {'start': begin, 'end': time.time()}
        begin = time.time()
        try:
            if verify_checksums and kind == 'sha256':
                for row, local in zip(manifest['arrays'], buffers):
                    for shard, raw in zip(row['shards'], local):
                        if hashlib.sha256(raw).hexdigest() != shard['sha256']:
                            raise SnapshotRefused(f"checksum mismatch: {shard['file']}")
        finally:
            clocks['snapshot_verify'] = {'start': begin, 'end': time.time()}
        return Prepared(manifest, buffers, lock)
    except Exception as exc:
        _close_buffers(buffers)
        if lock is not None:
            lock.close()
        if isinstance(exc, (OSError, KeyError, TypeError, ValueError)) and not isinstance(exc, SnapshotRefused):
            raise SnapshotRefused(str(exc)) from exc
        raise


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

    try:
        treedef, shardings = validate_tree(abstract_tree, mesh, prepared)
        if not isinstance(prepared, Prepared) or prepared.lock is None:
            raise SnapshotRefused('prepared snapshot is closed or unprotected')
        manifest, buffers = prepared
        rows = manifest['arrays']
        begin = time.time()
        groups = {}
        for number, (row, local, sharding) in enumerate(zip(rows, buffers, shardings)):
            by_id = {s['device_id']: (s, raw) for s, raw in zip(row['shards'], local)}
            for device in sharding.addressable_devices:
                shard, raw = by_id[int(device.id)]
                host = raw.view(np.dtype(jnp.dtype(row['dtype']))).reshape(shard['shape'])
                groups.setdefault(int(device.id), []).append((number, device, host))

        def place_chip(items):
            start = time.time()
            placed = {}
            try:
                for number, device, host in items:
                    # CPU device_put aliases host memory even with may_alias=False
                    # (verified on local JAX). Give it independent heap ownership;
                    # TPU transfers were separately poison-checked by the bench.
                    if device.platform == 'cpu':
                        host = np.array(host, copy=True, order='C')
                    placed[number] = jax.device_put(host, device, may_alias=False)
            finally:
                jax.block_until_ready(list(placed.values()))
                clocks[f'snapshot_device_placement_chip_{items[0][1].id}'] = {
                    'start': start, 'end': time.time()}
            return items[0][1].id, placed

        with ThreadPoolExecutor(max_workers=max(1, len(groups))) as pool:
            by_device = dict(pool.map(place_chip, groups.values()))
        arrays = []
        for number, (row, sharding) in enumerate(zip(rows, shardings)):
            parts = [by_device[d.id][number] for d in sharding.addressable_devices]
            arrays.append(jax.make_array_from_single_device_arrays(tuple(row['shape']), sharding, parts))
        jax.block_until_ready(arrays)
        clocks['snapshot_device_placement'] = {'start': begin, 'end': time.time()}
        return jax.tree_util.tree_unflatten(treedef, arrays)
    finally:
        if isinstance(prepared, Prepared):
            prepared.close()
