"""Resident weight store and reshard (plan C4).

One ``ResidentStore`` lives in each host kernel. ``put`` keeps a stopped engine's
parameter leaves in device memory (this host's addressable shards only).
``reshard`` turns the leaves of replica engines into the leaves of target
engines, in the physical view, with every host of the union calling it
concurrently:

1. *exchange*: hosts swap the source engines' specs, mesh orders and leaf specs
   (a tiny fixed-shape gather over the participating hosts), then verify that
   the sources are replicas (identical ``LeafSpec`` tuples).
2. *local* path (S-block -> H-block with target order ``halves_interleaved``):
   when every target shard lies inside a source shard held by the same chip, each
   target buffer is a slice/repack of a buffer already on that chip
   (``_piece_fn``), or an on-chip copy of it, so sources stay deletable. No
   device-to-device copy (the path runs under
   ``jax.transfer_guard_device_to_device("disallow")``).
3. *general* path (everything else), as in U2 (``unified_runtime.py:741``): stitch
   the sources' buffers into one stacked "pair view" array
   (``make_array_from_single_device_arrays``, replica x model mesh), repack fused
   leaves (``ici_resident.repack_merged``), ``device_put`` to the stacked target
   sharding, then rebind each target's buffers (no copy).

Cross-host ``device_put`` needs identical device sets (jax on CPU and TPU), so the
general path requires the source hosts to be inside the targets' hosts and the
target chips to be inside the source chips; unused chips receive filler
replicas that are dropped. Hosts only touch addressable shards; a non-addressable
array is never converted to numpy.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import pickle
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import catalog
from .cpu_block import resolve
from .types import ENGINE_KINDS, EngineSpec, LeafSpec, ReshardPlan, ResidentManifest

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"  # as unified_runtime.py:37
EXCHANGE_BYTES = 1 << 17  # fixed row size of the metadata gather (one compile for any content)

_COMPILES: List[float] = []
_LISTENING = False


def _install_compile_listener() -> None:
    global _LISTENING
    if _LISTENING:
        return
    import jax

    def on_duration(event, duration, **kwargs):
        if event == BACKEND_COMPILE_EVENT:
            _COMPILES.append(float(duration))

    jax.monitoring.register_event_duration_secs_listener(on_duration)
    _LISTENING = True


# --- device/mesh helpers ---------------------------------------------------------------------

def _physical_devices() -> list:
    import jax
    return sorted(jax.devices(), key=lambda d: (d.process_index, d.id))


def _mesh(rows: Sequence[Sequence[Any]], axes: Tuple[str, ...]):
    from jax.sharding import Mesh
    grid = np.empty((len(rows), len(rows[0])) if len(axes) == 2 else (len(rows[0]),), dtype=object)
    for i, row in enumerate(rows):
        for j, device in enumerate(row):
            grid[i, j] = device
    return Mesh(grid.reshape(len(rows), -1) if len(axes) == 2 else grid, axes)


def _pspec(leaf: LeafSpec):
    from jax.sharding import PartitionSpec as P
    if leaf.tp_axis is None:
        return P()
    return P(*([None] * (leaf.tp_axis % len(leaf.shape)) + ["model"]))


def _engine_sharding(devices: Sequence[Any], leaf: LeafSpec):
    """The sharding an engine builds for a leaf: Mesh(devices, ("model",)), "model" at tp_axis."""
    from jax.sharding import Mesh, NamedSharding
    grid = np.empty((len(devices),), dtype=object)
    for i, device in enumerate(devices):
        grid[i] = device
    return NamedSharding(Mesh(grid, ("model",)), _pspec(leaf))


def _stack_sharding(rows: Sequence[Sequence[Any]], leaf: LeafSpec):
    """Replica x model view over several engines' chips; replicated over "replica"."""
    from jax.sharding import NamedSharding
    return NamedSharding(_mesh(rows, ("replica", "model")), _pspec(leaf))


@functools.lru_cache(maxsize=None)
def _identity_to(sharding):
    import jax
    return jax.jit(lambda x: x, out_shardings=sharding)


@functools.lru_cache(maxsize=None)
def _repack_to(sizes: Tuple[int, ...], tp_s: int, tp_t: int, sharding):
    import jax
    from hetero.ici_resident import repack_merged  # hetero/ici_resident.py:53
    return jax.jit(lambda x: repack_merged(x, sizes, tp_s, tp_t), out_shardings=sharding)


@functools.lru_cache(maxsize=None)
def _copy_fn():
    import jax
    import jax.numpy as jnp
    return jax.jit(lambda x: jnp.copy(x))


@functools.lru_cache(maxsize=None)
def _piece_fn(axis: int, r: int, tp_s: int, sizes: Tuple[int, ...]):
    """Single-chip program: sub-slice ``sub`` of 1/r of a source shard (fused parts repacked)."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    def piece(x, sub):
        if not sizes:
            width = x.shape[axis] // r
            return lax.dynamic_slice_in_dim(x, sub * width, width, axis)
        out, offset = [], 0
        for size in sizes:
            part = size // tp_s
            width = part // r
            out.append(lax.dynamic_slice_in_dim(x, offset + sub * width, width, axis))
            offset += part
        return jnp.concatenate(out, axis=axis)

    return jax.jit(piece)


def _norm(index, shape) -> tuple:
    return tuple((s.start or 0, n if s.stop is None else s.stop) for s, n in zip(index, shape))


def local_bytes(array) -> int:
    """Bytes of this host's addressable shards of one array."""
    return sum(int(s.data.nbytes) for s in array.addressable_shards)


def _pointers(arrays) -> set:
    return {s.data.unsafe_buffer_pointer() for a in arrays for s in a.addressable_shards}


def reshard_leaves(leaves: Dict[str, Any], specs: Sequence[LeafSpec], targets: Dict[str, Any],
                   tp_source: int, tp_target: int, *, chunk_bytes: int = 1 << 30,
                   delete_sources: bool = True, discard: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Move one engine's parameter leaves onto other shardings of the same chips (one SPMD call).

    The general path of ``ResidentStore.reshard`` for a single live engine: per leaf, repack a fused
    projection from ``tp_source`` to ``tp_target`` shards (jit on the source sharding, so no mesh
    change inside the program), then ``jax.device_put`` to ``targets[path]``. ``targets`` may be any
    shardings over the same device set as the sources (e.g. a ``(data=2, model=8)`` mesh, replicated
    over ``data``). A leaf whose placement already is the target's is returned as is.

    Leaves go in chunks of ``chunk_bytes`` (global bytes): after each chunk completes, its source
    arrays (and repack intermediates) are deleted, so peak HBM is the weights plus one chunk, not
    twice the weights. A deleted source stays in ``leaves`` (metadata only). Every process of the
    sources' hosts must call this with the same arguments. Returns ``(new_leaves, stats)``.

    ``discard`` (priming): run the same programs but keep the sources and delete every output chunk by
    chunk; used to compile the reverse transition ahead of its first use (no persistent extra HBM).
    """
    import jax
    by_path = {l.path: l for l in specs}
    if set(leaves) != set(by_path) or set(targets) != set(by_path):
        raise ValueError("reshard_leaves: leaves, specs and targets must name the same paths")
    out: Dict[str, Any] = {}
    stats = {"leaves": len(leaves), "repacked": 0, "moved": 0, "kept": 0, "chunks": 0, "bytes_moved": 0}
    pending_out: List[Any] = []
    pending_drop: List[Any] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending_bytes
        if not pending_out:
            return
        jax.block_until_ready(pending_out)
        stats["chunks"] += 1
        if delete_sources or discard:
            for array in pending_drop:
                if not array.is_deleted():
                    array.delete()
        pending_out.clear()
        pending_drop.clear()
        pending_bytes = 0

    for path, source in leaves.items():
        leaf, target = by_path[path], targets[path]
        if tuple(source.shape) != tuple(leaf.shape) or str(source.dtype) != leaf.dtype:
            raise ValueError(f"{path}: {source.shape} {source.dtype} differs from its LeafSpec")
        x = source
        if leaf.fused and tp_source != tp_target:
            x = _repack_to(tuple(leaf.fused_sizes), tp_source, tp_target, source.sharding)(source)
            stats["repacked"] += 1
        if x.sharding.is_equivalent_to(target, len(leaf.shape)):
            # same placement: an own copy (so the old array can be deleted like every other), carrying
            # the exact target sharding object the layout's executables were compiled with
            copy = _copy_fn()(x)
            moved = copy if copy.sharding == target else jax.make_array_from_single_device_arrays(
                x.shape, target, [s.data for s in copy.addressable_shards])
            stats["kept"] += 1
        else:
            moved = jax.device_put(x, target)
            stats["moved"] += 1
            stats["bytes_moved"] += int(np.prod(leaf.shape)) * x.dtype.itemsize
        out[path] = moved
        pending_out.append(moved)
        if discard:
            pending_drop.append(moved)
            if x is not source:
                pending_drop.append(x)
        else:
            pending_drop.append(source)
            if x is not source:
                pending_drop.append(x)
        pending_bytes += int(np.prod(leaf.shape)) * source.dtype.itemsize
        if pending_bytes >= chunk_bytes:
            flush()
    flush()
    return out, stats


def local_assignment(source_meshes: Sequence[Sequence[int]], target_meshes: Sequence[Sequence[int]],
                     sharded: bool) -> Optional[Dict[Tuple[int, int], Tuple[int, int]]]:
    """Chip-local construction map or None. Pure function of device-id meshes.

    ``(target index, mesh position j) -> (source index k, source shard m)`` such that the
    chip ``target_meshes[e][j]`` already holds source shard ``m = j // r`` of replica ``k``
    (``r = tp_target / tp_source``; sharded leaves) or any copy (replicated leaves).
    """
    tp_s = len(source_meshes[0])
    tp_t = len(target_meshes[0])
    if sharded and tp_t % tp_s:
        return None
    r = tp_t // tp_s if sharded else 1
    assignment = {}
    for e, mesh in enumerate(target_meshes):
        for j, device in enumerate(mesh):
            found = None
            for k, source in enumerate(source_meshes):
                if sharded:
                    if source[j // r] == device:
                        found = (k, j // r)
                        break
                elif device in source:
                    found = (k, list(source).index(device))
                    break
            if found is None:
                return None
            assignment[(e, j)] = found
    return assignment


def local_share(source_meshes: Sequence[Sequence[int]], target_meshes: Sequence[Sequence[int]]) -> int:
    """Number of target mesh positions (over all targets) whose chip already holds the source
    shard ``j // r`` of some replica, i.e. positions a sharded leaf can build without a transfer."""
    tp_s, tp_t = len(source_meshes[0]), len(target_meshes[0])
    if tp_t % tp_s:
        return 0
    r = tp_t // tp_s
    return sum(1 for mesh in target_meshes for j, device in enumerate(mesh)
               if any(source[j // r] == device for source in source_meshes))


def tp_device_ids(spec: EngineSpec, specs: Sequence[LeafSpec], leaves: Dict[str, Any]) -> Tuple[int, ...]:
    """Device ids in TP order of an engine's leaves; every leaf must share one mesh.

    Accepts any ``NamedSharding`` whose mesh has a ``model`` axis of size ``spec.tp_size`` and only
    size-1 other axes (tpu-inference: ``("data", "model")`` of shape ``(1, tp)``, or the NEW_MODEL_DESIGN
    mesh whose axes other than ``model`` are all 1), so the flattened device order is the TP order. The
    ``PartitionSpec`` must put ``model`` on ``LeafSpec.tp_axis`` (nowhere for replicated leaves) and name
    no other axis but size-1 ones.
    """
    by_path = {l.path: l for l in specs}
    orders = set()
    for path, array in leaves.items():
        sharding, leaf = array.sharding, by_path[path]
        mesh, pspec = getattr(sharding, "mesh", None), getattr(sharding, "spec", None)
        if mesh is None or pspec is None:
            raise ValueError(f"{spec.engine_id}: {path} is not on a NamedSharding ({type(sharding).__name__})")
        sizes = dict(zip(mesh.axis_names, mesh.devices.shape))
        if sizes.get("model") != spec.tp_size or any(n != 1 for a, n in sizes.items() if a != "model"):
            raise ValueError(f"{spec.engine_id}: {path} mesh axes {sizes}; need model={spec.tp_size} "
                             "and every other axis of size 1")
        on_model = []
        for dim, entry in enumerate(pspec):
            names = entry if isinstance(entry, tuple) else (entry,)
            for name in names:
                if isinstance(name, str) and name not in sizes:
                    raise ValueError(f"{spec.engine_id}: {path} partition names unknown axis {name!r}")
            if "model" in names:
                on_model.append(dim)
        expect = [] if leaf.tp_axis is None else [leaf.tp_axis % len(leaf.shape)]
        if on_model != expect:
            raise ValueError(f"{spec.engine_id}: {path} shards 'model' on dims {on_model}, LeafSpec.tp_axis "
                             f"says {expect}")
        orders.add(tuple(int(d.id) for d in mesh.devices.flat))
    if len(orders) != 1:
        raise ValueError(f"{spec.engine_id}: leaves use {len(orders)} different TP device orders")
    return orders.pop()


@dataclasses.dataclass
class _Entry:
    spec: EngineSpec
    leaves: Dict[str, Any]
    specs: Tuple[LeafSpec, ...]
    checksums: Dict[str, str]


class ResidentStore:
    """Parameter leaves kept in device memory between engines, one per host kernel."""

    def __init__(self, rank: int):
        _install_compile_listener()
        self.rank = rank
        self._entries: Dict[str, _Entry] = {}

    # --- store ---------------------------------------------------------------------------

    def put(self, spec: EngineSpec, leaves: Dict[str, Any], specs: Tuple[LeafSpec, ...],
            checksums: bool = False) -> ResidentManifest:
        """Keep an exported engine's leaves under ``spec.engine_id``. ``checksums`` gathers every
        leaf over the engine's own hosts (SPMD over ``spec.hosts``)."""
        if spec.engine_id in self._entries:
            raise ValueError(f"resident key {spec.engine_id!r} already held")
        self._check_leaves(spec, leaves, specs)
        tp_device_ids(spec, specs, leaves)  # reject unsupported meshes here, before any collective
        sums = self._gather_checksums(leaves, specs, spec.tp_size) if checksums else {}
        self._entries[spec.engine_id] = _Entry(spec, dict(leaves), tuple(specs), sums)
        return self.manifest(spec.engine_id)

    def manifest(self, key: str) -> ResidentManifest:
        e = self._entries[key]
        return ResidentManifest(e.spec.engine_id, e.spec.generation, e.spec.hosts, e.spec.tp_size,
                                e.specs, sum(local_bytes(a) for a in e.leaves.values()), dict(e.checksums))

    def peek(self, key: str) -> Dict[str, Any]:
        """The leaves without removing them (tests, inspection)."""
        return dict(self._entries[key].leaves)

    def take(self, key: str) -> Dict[str, Any]:
        """Hand the leaves to an engine and forget them (the engine owns the buffers now)."""
        return self._entries.pop(key).leaves

    def drop(self, key: str) -> None:
        self._release(self._entries.pop(key), keep=self._entries.values())

    def keys(self) -> Tuple[str, ...]:
        return tuple(self._entries)

    def bytes_on_host(self) -> int:
        return sum(local_bytes(a) for e in self._entries.values() for a in e.leaves.values())

    @staticmethod
    def _check_leaves(spec, leaves, specs):
        if set(leaves) != {l.path for l in specs}:
            raise ValueError(f"{spec.engine_id}: leaves differ from specs")
        for leaf in specs:
            array = leaves[leaf.path]
            if tuple(array.shape) != tuple(leaf.shape) or str(array.dtype) != leaf.dtype:
                raise ValueError(f"{spec.engine_id}: {leaf.path} is {array.shape} {array.dtype}, "
                                 f"spec says {leaf.shape} {leaf.dtype}")
            if leaf.fused and leaf.tp_axis not in (len(leaf.shape) - 1, -1):
                raise ValueError(f"{leaf.path}: fused leaves must be sharded on the last axis")

    @staticmethod
    def _release(entry: _Entry, keep) -> None:
        """Delete an entry's buffers (a buffer a kept entry still shares is never deleted)."""
        shared = set()
        for other in keep:
            shared |= _pointers(other.leaves.values())
        for array in entry.leaves.values():
            if array.is_deleted():
                continue
            if shared & _pointers([array]):
                continue
            array.delete()

    @staticmethod
    def _gather_checksums(leaves, specs, tp) -> Dict[str, str]:
        """Logical checksum per leaf; SPMD over the hosts of the leaves' mesh."""
        from jax.sharding import NamedSharding, PartitionSpec as P
        out = {}
        for leaf in specs:
            array = leaves[leaf.path]
            full = _identity_to(NamedSharding(array.sharding.mesh, P()))(array)
            logical = catalog.to_logical(np.asarray(full.addressable_data(0)), leaf, tp)
            out[leaf.path] = catalog.logical_checksum(logical)
        return out

    # --- reshard -------------------------------------------------------------------------

    def reshard(self, plan: ReshardPlan) -> Dict[str, Any]:
        """Build ``f"{plan_id}/{engine_id}"`` entries for ``plan.targets`` from ``plan.sources``.

        Contract: called concurrently by every host of ``union(source hosts, target hosts)``;
        source hosts must lie inside the target hosts (cross-host ``device_put`` needs one device
        set) and the target chips inside the source chips. Returns ``segments`` (seconds; dispatch
        and ``block_until_ready`` separately), ``compiles`` (backend compiles during the call),
        ``targets`` (manifests of the entries this host holds), ``paths`` and ``provenance``.
        """
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P

        started = time.perf_counter()
        compiles0 = len(_COMPILES)
        seg: Dict[str, float] = collections.defaultdict(float)

        @contextlib.contextmanager
        def timed(name):
            t = time.perf_counter()
            try:
                yield
            finally:
                seg[name] += time.perf_counter() - t

        def block(arrays):
            with timed("block_until_ready"):
                jax.block_until_ready(arrays)

        held = {s: self._entries[s] for s in plan.sources if s in self._entries}
        target_hosts = {h for t in plan.targets for h in t.hosts}
        mine = {h for e in held.values() for h in e.spec.hosts}
        if not mine <= target_hosts:
            raise ValueError(f"host {self.rank}: source hosts {sorted(mine - target_hosts)} are outside "
                             f"the targets' hosts {sorted(target_hosts)}")
        participants = tuple(sorted(mine | target_hosts))
        report: Dict[str, Any] = {"rank": self.rank, "targets": {}, "paths": {}, "provenance": {}}
        if self.rank not in participants:
            report.update(segments={"total": 0.0}, compiles=0, compile_s=0.0)
            return report

        # 1. exchange metadata, verify replicas ------------------------------------------------
        with timed("exchange"):
            meta = self._exchange(participants, held)
        missing = [s for s in plan.sources if s not in meta]
        if missing:
            raise ValueError(f"sources {missing} are held by no participating host {participants}")
        specs = meta[plan.sources[0]][2]
        for s in plan.sources:
            if meta[s][2] != specs:
                raise ValueError(f"source {s!r} has different LeafSpecs from {plan.sources[0]!r}: "
                                 "not a replica")
        tps = {meta[s][0].tp_size for s in plan.sources}
        if len(tps) != 1:
            raise ValueError(f"sources have different TP degrees {sorted(tps)}")
        tp_s = tps.pop()
        tp_ts = {t.tp_size for t in plan.targets}
        if len(tp_ts) != 1:
            raise ValueError(f"targets have different TP degrees {sorted(tp_ts)}")
        tp_t = tp_ts.pop()
        physical = _physical_devices()
        by_id = {d.id: d for d in physical}
        source_ids = [list(meta[s][1]) for s in plan.sources]
        target_devs = [self._target_mesh(t, physical) for t in plan.targets]
        target_ids = [[d.id for d in m] for m in target_devs]
        source_hosts = {h for s in plan.sources for h in meta[s][0].hosts}
        if not source_hosts <= target_hosts or source_hosts | target_hosts != set(participants):
            raise ValueError(f"source hosts {sorted(source_hosts)} must lie inside target hosts "
                             f"{sorted(target_hosts)}")
        source_set = {i for ids in source_ids for i in ids}
        used = [i for ids in target_ids for i in ids]
        if len(set(used)) != len(used) or not set(used) <= source_set:
            raise ValueError("target chips must be distinct and inside the source chips "
                             "(cross-host reshard needs one device set)")
        filler = sorted(source_set - set(used))
        if len(filler) % tp_t:
            raise ValueError(f"{len(filler)} unused chips do not form TP{tp_t} filler replicas")

        sources_sums = {}
        if plan.checksums:
            with timed("checksums"):
                for s, entry in held.items():
                    sources_sums[s] = entry.checksums or self._gather_checksums(
                        entry.leaves, entry.specs, entry.spec.tp_size)

        # 2. classify leaves ---------------------------------------------------------------------
        sharded_map = local_assignment(source_ids, target_ids, True)
        replicated_map = local_assignment(source_ids, target_ids, False)
        local_leaves, general_leaves = [], []
        for leaf in specs:
            (local_leaves if (sharded_map if leaf.tp_axis is not None else replicated_map) is not None
             else general_leaves).append(leaf)
        outputs: Dict[str, Dict[str, Any]] = {t.engine_id: {} for t in plan.targets}
        mine_targets = [(e, t) for e, t in enumerate(plan.targets) if self.rank in t.hosts]
        provenance = report["provenance"]
        copied = sliced = 0

        # 3a. chip-local construction ------------------------------------------------------------
        if local_leaves:
            pieces = []
            with timed("local_slice"), jax.transfer_guard_device_to_device("disallow"):
                for leaf in local_leaves:
                    assignment = sharded_map if leaf.tp_axis is not None else replicated_map
                    axis = None if leaf.tp_axis is None else leaf.tp_axis % len(leaf.shape)
                    r = tp_t // tp_s if leaf.tp_axis is not None else 1
                    sizes = tuple(leaf.fused_sizes) if leaf.fused and r > 1 else ()
                    for e, target in mine_targets:
                        sharding = _engine_sharding(target_devs[e], leaf)
                        position = {d: j for j, d in enumerate(target_devs[e])}
                        buffers = []
                        for device in sharding.addressable_devices_indices_map(leaf.shape):
                            j = position[device]
                            k, m = assignment[(e, j)]
                            source = held[plan.sources[k]]
                            shard = {s.device: s for s in source.leaves[leaf.path].addressable_shards}[device]
                            if axis is not None:
                                width = leaf.shape[axis] // tp_s
                                expect = tuple((0, n) if a != axis else (m * width, (m + 1) * width)
                                               for a, n in enumerate(leaf.shape))
                                if _norm(shard.index, leaf.shape) != expect:
                                    raise ValueError(f"{leaf.path}: chip {device.id} holds shard "
                                                     f"{shard.index}, expected {expect}")
                            if r == 1:
                                buffers.append(_copy_fn()(shard.data))  # own buffer: sources stay deletable
                                copied += 1
                            else:
                                sub = jax.device_put(np.int32(j % r), device)
                                buffers.append(_piece_fn(axis, r, tp_s, sizes)(shard.data, sub))
                                sliced += 1
                            provenance.setdefault(leaf.path, []).append(
                                (device.id, plan.sources[k], m, shard.device.id, j % r))
                        outputs[target.engine_id][leaf.path] = jax.make_array_from_single_device_arrays(
                            leaf.shape, sharding, buffers, dtype=np.dtype(catalog._np_dtype(leaf.dtype)))
                        pieces.extend(buffers)
            block(pieces)

        # 3b. general path: stitch -> repack -> device_put -> rebind -------------------------------
        if general_leaves:
            src_rows = [[by_id[i] for i in ids] for ids in source_ids]
            rows = target_devs + [[by_id[i] for i in filler[n:n + tp_t]] for n in range(0, len(filler), tp_t)]
            flat = [d for row in src_rows for d in row]
            held_shards = {}
            for entry in held.values():
                for leaf in entry.specs:
                    held_shards.setdefault(leaf.path, {}).update(
                        {s.device: s for s in entry.leaves[leaf.path].addressable_shards})
            stacked = {}
            with timed("stitch"):
                for leaf in general_leaves:
                    sharding = _stack_sharding(src_rows, leaf)
                    buffers = []
                    for device, index in sharding.addressable_devices_indices_map(leaf.shape).items():
                        shard = held_shards[leaf.path][device]
                        if _norm(shard.index, leaf.shape) != _norm(index, leaf.shape):
                            raise ValueError(f"{leaf.path}: chip {device.id} shard index differs from the "
                                             "stacked view")
                        buffers.append(shard.data)
                    stacked[leaf.path] = jax.make_array_from_single_device_arrays(
                        leaf.shape, sharding, buffers, dtype=np.dtype(catalog._np_dtype(leaf.dtype)))
            repacked = {}
            with timed("repack"):
                for leaf in general_leaves:
                    if leaf.fused and tp_s != tp_t:
                        mid = _mesh([flat[n:n + tp_t] for n in range(0, len(flat), tp_t)],
                                    ("replica", "model"))
                        repacked[leaf.path] = _repack_to(tuple(leaf.fused_sizes), tp_s, tp_t,
                                                         NamedSharding(mid, _pspec(leaf)))(stacked[leaf.path])
            block(list(repacked.values()))
            moved = {}
            with timed("transfer"):
                for leaf in general_leaves:
                    moved[leaf.path] = jax.device_put(repacked.get(leaf.path, stacked[leaf.path]),
                                                      _stack_sharding(rows, leaf))
            block(list(moved.values()))
            with timed("rebind"):
                for leaf in general_leaves:
                    out_shards = {s.device: s for s in moved[leaf.path].addressable_shards}
                    for e, target in mine_targets:
                        sharding = _engine_sharding(target_devs[e], leaf)
                        buffers = []
                        for device, index in sharding.addressable_devices_indices_map(leaf.shape).items():
                            shard = out_shards[device]
                            if _norm(shard.index, leaf.shape) != _norm(index, leaf.shape):
                                raise ValueError(f"{leaf.path}: stacked shard index differs on chip {device.id}")
                            buffers.append(shard.data)
                        outputs[target.engine_id][leaf.path] = jax.make_array_from_single_device_arrays(
                            leaf.shape, sharding, buffers, dtype=np.dtype(catalog._np_dtype(leaf.dtype)))
            del stacked, repacked, moved
        block([a for leaves in outputs.values() for a in leaves.values()])

        # 4. checksums of the targets, cross-host agreement --------------------------------------
        target_sums = {}
        if plan.checksums:
            with timed("checksums"):
                for e, target in mine_targets:
                    leaves = {l.path: outputs[target.engine_id][l.path] for l in specs}
                    target_sums[target.engine_id] = self._gather_checksums(leaves, specs, target.tp_size)
                self._agree(participants, plan, specs, {**sources_sums, **target_sums})

        # 5. install targets, release sources -----------------------------------------------------
        for e, target in mine_targets:
            key = f"{plan.plan_id}/{target.engine_id}"
            if key in self._entries:
                raise ValueError(f"resident key {key!r} already held")
            self._entries[key] = _Entry(target, outputs[target.engine_id], tuple(specs),
                                        target_sums.get(target.engine_id, {}))
            report["targets"][target.engine_id] = self.manifest(key)
        dropped = []
        if plan.drop_sources:
            with timed("drop"):
                for s in list(held):
                    entry = self._entries.pop(s)
                    self._release(entry, keep=self._entries.values())
                    dropped.append(s)
        report["dropped"] = dropped
        report["paths"] = {"local_leaves": len(local_leaves), "general_leaves": len(general_leaves),
                           "copied_buffers": copied, "sliced_buffers": sliced,
                           "chip_local_positions": local_share(source_ids, target_ids),
                           "target_positions": sum(len(m) for m in target_ids),
                           "tp_source": tp_s, "tp_target": tp_t, "filler_chips": len(filler)}
        report["checksums_verified"] = bool(plan.checksums)
        report["compiles"] = len(_COMPILES) - compiles0
        report["compile_s"] = float(sum(_COMPILES[compiles0:]))
        report["segments"] = dict(seg, total=time.perf_counter() - started)
        return report

    # --- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _target_mesh(spec: EngineSpec, physical: Sequence[Any]) -> list:
        """Chips of the target in the TP mesh order of its engine class."""
        if spec.dp_size > 1:
            raise ValueError(f"{spec.engine_id}: dp_size {spec.dp_size} engines are not reshard targets "
                             "(mesh_device_ids is (data, model)); they are entered in place with "
                             "Engine.switch_layout, which uses reshard_leaves")
        devices = [d for d in physical if d.process_index in spec.hosts]
        cls = resolve(ENGINE_KINDS.get(spec.kind, spec.kind))
        ids = tuple(cls.mesh_device_ids(spec, devices))
        by_id = {d.id: d for d in devices}
        if len(ids) != spec.tp_size or len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
            raise ValueError(f"{spec.kind}.mesh_device_ids returned {ids} for TP{spec.tp_size} "
                             f"on hosts {spec.hosts}")
        return [by_id[i] for i in ids]

    def _exchange(self, participants: Tuple[int, ...], held: Dict[str, _Entry]) -> Dict[str, tuple]:
        """key -> (EngineSpec, mesh device ids, LeafSpecs) of every source, on every participant."""
        payload = {}
        for key, entry in held.items():
            payload[key] = (entry.spec, tp_device_ids(entry.spec, entry.specs, entry.leaves), entry.specs)
        blob = zlib.compress(pickle.dumps(payload))
        if len(blob) + 4 > EXCHANGE_BYTES:
            raise ValueError(f"metadata {len(blob)} bytes exceeds EXCHANGE_BYTES={EXCHANGE_BYTES}")
        row = np.zeros(EXCHANGE_BYTES, np.uint8)
        row[:4] = np.frombuffer(np.uint32(len(blob)).tobytes(), np.uint8)
        row[4:4 + len(blob)] = np.frombuffer(blob, np.uint8)
        merged: Dict[str, tuple] = {}
        for received in _union_gather(participants, row):
            n = int(np.frombuffer(received[:4].tobytes(), np.uint32)[0])
            for key, value in pickle.loads(zlib.decompress(received[4:4 + n].tobytes())).items():
                if merged.setdefault(key, value) != value:
                    raise ValueError(f"hosts disagree about source {key!r}")
        return merged

    def _agree(self, participants, plan, specs, sums: Dict[str, Dict[str, str]]) -> None:
        """Every engine's logical checksum of every leaf, as known to any host, must be equal."""
        engines = list(plan.sources) + [t.engine_id for t in plan.targets]
        row = np.zeros(len(engines) * len(specs) * 9, np.uint8)
        for a, engine in enumerate(engines):
            for b, leaf in enumerate(specs):
                digest = sums.get(engine, {}).get(leaf.path)
                if digest:
                    at = (a * len(specs) + b) * 9
                    row[at] = 1
                    row[at + 1:at + 9] = np.frombuffer(bytes.fromhex(digest), np.uint8)
        table = _union_gather(participants, row).reshape(len(participants), len(engines), len(specs), 9)
        for b, leaf in enumerate(specs):
            seen = {bytes(table[h, a, b, 1:]) for h in range(len(participants))
                    for a in range(len(engines)) if table[h, a, b, 0]}
            known = {a for a in range(len(engines)) if table[:, a, b, 0].any()}
            if len(seen) != 1 or len(known) != len(engines):
                raise ValueError(f"logical checksum mismatch for {leaf.path}: {len(seen)} distinct "
                                 f"digests, known for engines {sorted(known)} of {len(engines)}")


def _union_gather(hosts: Sequence[int], row: np.ndarray) -> np.ndarray:
    """All-gather one uint8 row per participating host; returns the rows in ``hosts`` order.

    The mesh is every chip of the participating hosts, each host's row on each of its chips. A
    whole-host set is what the block's compile matrix allows. One chip per host (chips 0, 2, 8, 10
    on a v5e 4x4 block) is a set that holds chip 0 without being a dense prefix: libtpu fails it
    at compile (net_util.cc:2304) and aborts the process (G2 run vproc-G2-23efaf8, 2026-10-22).
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    devices = [d for d in _physical_devices() if d.process_index in hosts]
    mesh = _mesh([devices], ("replica", "hosts"))
    sharding = NamedSharding(mesh, P("hosts"))
    shards = [jax.device_put(row.reshape(1, -1), d) for d in devices if d.process_index == jax.process_index()]
    array = jax.make_array_from_single_device_arrays((len(devices), row.size), sharding, shards)
    out = np.asarray(_identity_to(NamedSharding(mesh, P()))(array).addressable_data(0))
    owners = [d.process_index for d in devices]
    return np.stack([out[owners.index(h)] for h in hosts])
