"""C4 resident store: S-block <-> H-block reshard on the 4x4 CPU block, exact leaf values.

Every leaf is built from ``catalog.synthetic_value`` + ``to_stored`` in each process, resharded
(TP8 x2 -> TP16 -> TP8 x2) and compared leaf by leaf, per shard and after a gather (gather in the
test only). The test engine class defines the two mesh orders the plan pins.
"""
import json
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from hetero.vproc import catalog
from hetero.vproc.cpu_block import CPU_ID_STRIDE, CpuBlock
from hetero.vproc.resident import local_assignment
from hetero.vproc.types import ILLEGAL_HOST_SETS, EngineSpec, ReshardPlan

HERE = __name__
KIND = f"{HERE}:TestEngine"
SEED = 7
_STORE = None
_KEPT = {}


# EngineAdapter's real v5e ring orders by physical chip index k (host k // 4, chip k % 4).
V5E_RINGS = {(0, 1): (0, 1, 2, 3, 7, 6, 5, 4),
             (2, 3): (8, 9, 10, 11, 15, 14, 13, 12),
             (0, 1, 2, 3): (0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4)}


class TestEngine:
    """Only the mesh-order contract of ``types.Engine``."""
    __test__ = False

    @staticmethod
    def mesh_device_ids(spec, devices):
        ids = [d.id for d in devices]
        order = spec.model.get("device_order", "natural")
        if order == "natural":
            return tuple(ids)
        if order == "v5e_ring":
            return tuple((k // 4) * CPU_ID_STRIDE + k % 4 for k in V5E_RINGS[tuple(spec.hosts)])
        hosts = sorted({d.process_index for d in devices})
        first = set(hosts[:len(hosts) // 2])
        a = [d.id for d in devices if d.process_index in first]
        b = [d.id for d in devices if d.process_index not in first]
        return tuple(i for pair in zip(a, b) for i in pair)


def spec_of(engine_id, hosts, tp, order="natural", mesh="1d"):
    """``mesh`` is the mesh kind the engine's leaves live on: "1d" (model,), "2d" (data=1, model),
    "7d" (six size-1 axes + model, the NEW_MODEL_DESIGN shape), "bad" (data=2, model=tp/2)."""
    return EngineSpec(engine_id, KIND, tuple(hosts), tp, {"device_order": order, "mesh": mesh})


# --- per-process programs (run inside the CpuBlock processes) -----------------------------------

def _store():
    global _STORE
    import jax
    from hetero.vproc.resident import ResidentStore
    if _STORE is None:
        _STORE = ResidentStore(jax.process_index())
    return _STORE


def _mesh_devices(spec):
    import jax
    devices = sorted((d for d in jax.devices() if d.process_index in spec.hosts),
                     key=lambda d: (d.process_index, d.id))
    by_id = {d.id: d for d in devices}
    return [by_id[i] for i in TestEngine.mesh_device_ids(spec, devices)]


def _sharding(spec, leaf):
    from hetero.vproc.resident import _engine_sharding
    return _engine_sharding(_mesh_devices(spec), leaf)


def _source_sharding(spec, leaf):
    """The sharding an exporting engine's leaves carry (tpu-inference meshes, see ``spec_of``)."""
    from jax.sharding import Mesh, NamedSharding
    kind, tp = spec.model.get("mesh", "1d"), spec.tp_size
    if kind == "1d":
        return _sharding(spec, leaf)
    shape, names = {
        "2d": ((1, tp), ("data", "model")),
        "7d": ((1,) * 6 + (tp,), ("data", "attn_dp", "attn_dp_expert", "expert", "pipeline", "seq", "model")),
        "bad": ((2, tp // 2), ("data", "model")),
    }[kind]
    from hetero.vproc.resident import _pspec
    grid = np.empty(len(_mesh_devices(spec)), dtype=object)
    for i, device in enumerate(_mesh_devices(spec)):
        grid[i] = device
    return NamedSharding(Mesh(grid.reshape(shape), names), _pspec(leaf))


def reset():
    store = _store()
    for key in store.keys():
        store.drop(key)
    _KEPT.clear()


def build(items, checksums):
    """items: [(spec, seed, divisor)]; builds the engines this process is a host of."""
    import jax
    out = {}
    for spec, seed, divisor in items:
        if jax.process_index() not in spec.hosts:
            continue
        leaves, specs = {}, catalog.scaled(divisor=divisor)
        for leaf in specs:
            stored = catalog.to_stored(catalog.synthetic_value(leaf, seed), leaf, spec.tp_size)
            leaves[leaf.path] = jax.make_array_from_callback(
                leaf.shape, _source_sharding(spec, leaf), lambda index, s=stored: s[index],
                dtype=stored.dtype)
        _KEPT[spec.engine_id] = leaves
        out[spec.engine_id] = _store().put(spec, leaves, specs, checksums=checksums)
    return out


def do_reshard(plan):
    return _store().reshard(plan)


_MESHES = []


def record_meshes():
    """Wrap the reshard's mesh constructors so every device set it builds a program over is kept."""
    from hetero.vproc import resident
    if getattr(resident, "_recording", False):
        return
    mesh, engine_sharding = resident._mesh, resident._engine_sharding

    def recording_mesh(rows, axes):
        out = mesh(rows, axes)
        _MESHES.append(("mesh", sorted(d.id for d in out.devices.flat)))
        return out

    def recording_engine_sharding(devices, leaf):
        _MESHES.append(("engine", sorted(d.id for d in devices)))
        return engine_sharding(devices, leaf)

    resident._mesh, resident._engine_sharding, resident._recording = recording_mesh, recording_engine_sharding, True
    _MESHES.clear()


def recorded_meshes():
    return sorted({(kind, tuple(ids)) for kind, ids in _MESHES})


def state():
    import jax
    store = _store()
    alive = {k: [p for p, a in v.items() if not a.is_deleted()] for k, v in _KEPT.items()}
    return {"keys": store.keys(), "bytes": store.bytes_on_host(), "kept_alive": alive,
            "rank": jax.process_index()}


def verify(key, spec, seed):
    """Every leaf of ``key`` equals the synthetic value: per local shard (stored layout) and
    logically after a gather over the engine's own hosts."""
    from hetero.vproc.resident import _identity_to
    from jax.sharding import NamedSharding, PartitionSpec as P
    leaves = _store().peek(key)
    count = 0
    for leaf in catalog.scaled():
        array = leaves[leaf.path]
        expect = _sharding(spec, leaf)
        assert array.sharding.is_equivalent_to(expect, len(leaf.shape)), leaf.path
        assert list(array.sharding.mesh.devices.flat) == list(expect.mesh.devices.flat), leaf.path
        logical = catalog.synthetic_value(leaf, SEED if seed is None else seed)
        stored = catalog.to_stored(logical, leaf, spec.tp_size)
        for shard in array.addressable_shards:
            assert np.array_equal(np.asarray(shard.data), stored[shard.index]), (leaf.path, shard.index)
        full = _identity_to(NamedSharding(expect.mesh, P()))(array)
        gathered = catalog.to_logical(np.asarray(full.addressable_data(0)), leaf, spec.tp_size)
        assert gathered.dtype == logical.dtype and np.array_equal(gathered, logical), leaf.path
        count += 1
    return count


def rewrap(key, spec, kind):
    """What ``VllmEngine.build`` does: ``device_put`` each leaf onto the runner's (data=1, model=tp)
    mesh with the same flattened device order. Must move no data."""
    import jax
    from jax.sharding import Mesh, NamedSharding
    from hetero.vproc.resident import _pspec
    tp = spec.tp_size
    shape, names = {"2d": ((1, tp), ("data", "model")),
                    "7d": ((1,) * 6 + (tp,), ("data", "attn_dp", "attn_dp_expert", "expert", "pipeline",
                                              "seq", "model"))}[kind]
    grid = np.empty(tp, dtype=object)
    for i, device in enumerate(_mesh_devices(spec)):
        grid[i] = device
    mesh = Mesh(grid.reshape(shape), names)
    leaves, same, zero_copy = _store().peek(key), 0, 0
    for leaf in catalog.scaled():
        array = leaves[leaf.path]
        before = {s.device: s.data.unsafe_buffer_pointer() for s in array.addressable_shards}
        with jax.transfer_guard("disallow"):
            out = jax.device_put(array, NamedSharding(mesh, _pspec(leaf)))
        out.block_until_ready()
        assert out.sharding.mesh.devices.shape == shape
        assert list(out.sharding.mesh.devices.flat) == list(grid)
        after = {s.device: s.data.unsafe_buffer_pointer() for s in out.addressable_shards}
        same += before == after
        for a, b in zip(sorted(array.addressable_shards, key=lambda s: s.device.id),
                        sorted(out.addressable_shards, key=lambda s: s.device.id)):
            assert a.index == b.index and np.array_equal(np.asarray(a.data), np.asarray(b.data)), leaf.path
        # Zero-copy variant (ici_resident.resident_views): rebind the same buffers to the new sharding.
        sharding = NamedSharding(mesh, _pspec(leaf))
        by_device = {s.device: s.data for s in array.addressable_shards}
        with jax.transfer_guard("disallow"):
            view = jax.make_array_from_single_device_arrays(
                leaf.shape, sharding, [by_device[d] for d in sharding.addressable_devices_indices_map(leaf.shape)],
                dtype=array.dtype)
        assert {s.device: s.data.unsafe_buffer_pointer() for s in view.addressable_shards} == before, leaf.path
        zero_copy += 1
    return {"leaves": len(leaves), "same_buffers": same, "zero_copy_rebinds": zero_copy}


# --- driver side ------------------------------------------------------------------------------

def expected_bytes_per_host(tp, chips=4):
    total = 0
    for leaf in catalog.scaled():
        n = int(np.prod(leaf.shape)) * np.dtype(catalog._np_dtype(leaf.dtype)).itemsize
        total += (n // tp if leaf.tp_axis is not None else n) * chips
    return total


def spmd(block, name, args_by_rank):
    return block.call([(r, f"{HERE}:{name}", a) for r, a in args_by_rank])


def merge_segments(reports):
    keys = {k for r in reports for k in r["segments"]}
    return {k: round(max(r["segments"].get(k, 0.0) for r in reports), 4) for k in sorted(keys)}


class LocalAssignmentTests(unittest.TestCase):
    def meshes(self, order):
        s = [list(range(8)), list(range(8, 16))]
        spec = spec_of("H", (0, 1, 2, 3), 16, order)

        class D:
            def __init__(self, i):
                self.id, self.process_index = i, i // 4
        h = list(TestEngine.mesh_device_ids(spec, [D(i) for i in range(16)]))
        return s, [h]

    def test_interleaved_is_chip_local(self):
        s, t = self.meshes("halves_interleaved")
        plan = local_assignment(s, t, True)
        self.assertIsNotNone(plan)
        for (e, j), (k, m) in plan.items():
            self.assertEqual(s[k][m], t[e][j])
            self.assertEqual(m, j // 2)

    def test_natural_is_not_chip_local(self):
        s, t = self.meshes("natural")
        self.assertIsNone(local_assignment(s, t, True))
        self.assertIsNotNone(local_assignment(s, t, False))  # replicated leaves are everywhere

    def test_reverse_is_not_chip_local(self):
        s, t = self.meshes("halves_interleaved")
        self.assertIsNone(local_assignment(t, s, True))


class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock()
        cls.leaves = catalog.scaled()
        cls.timings = []

    @classmethod
    def tearDownClass(cls):
        cls.block.close()
        for row in cls.timings:
            print("RESHARD " + json.dumps(row, sort_keys=True))

    def setUp(self):
        spmd(self.block, "reset", [(r, ()) for r in range(4)])

    def build(self, specs, checksums=False, divisors=None):
        items = [(s, SEED, (divisors or {}).get(s.engine_id, 64)) for s in specs]
        return spmd(self.block, "build", [(r, (items, checksums)) for r in range(4)])

    def verify(self, key, spec):
        calls = [(r, (key, spec, SEED)) for r in spec.hosts]
        self.assertEqual(spmd(self.block, "verify", calls), [len(self.leaves)] * len(spec.hosts))

    def reshard(self, plan, label):
        reports = spmd(self.block, "do_reshard", [(r, (plan,)) for r in range(4)])
        row = {"label": label, "segments_max_over_hosts": merge_segments(reports),
               "compiles_per_host": [r["compiles"] for r in reports],
               "compile_s_max": round(max(r["compile_s"] for r in reports), 4),
               "paths": reports[0]["paths"]}
        self.timings.append(row)
        return reports

    def state(self):
        return {s["rank"]: s for s in spmd(self.block, "state", [(r, ()) for r in range(4)])}

    def check_manifests(self, reports, spec, key):
        for r, report in enumerate(reports):
            m = report["targets"][spec.engine_id]
            self.assertEqual((m.engine_id, m.hosts, m.tp_size), (spec.engine_id, spec.hosts, spec.tp_size))
            self.assertEqual(m.leaves, self.leaves)
            self.assertEqual(m.bytes_per_host, expected_bytes_per_host(spec.tp_size))
        for rank, state in self.state().items():
            self.assertEqual(key in state["keys"], rank in spec.hosts)

    def cycle(self, order, tag, checksums, src_order="natural", mesh="1d"):
        a, b = spec_of(f"A{tag}", (0, 1), 8, src_order, mesh), spec_of(f"B{tag}", (2, 3), 8, src_order, mesh)
        h = spec_of(f"H{tag}", (0, 1, 2, 3), 16, order)
        a2, b2 = spec_of(f"A2{tag}", (0, 1), 8, src_order), spec_of(f"B2{tag}", (2, 3), 8, src_order)
        spmd(self.block, "reset", [(r, ()) for r in range(4)])
        manifests = self.build([a, b], checksums)
        self.assertEqual(manifests[0][a.engine_id].bytes_per_host, expected_bytes_per_host(8))
        self.verify(a.engine_id, a)
        self.verify(b.engine_id, b)

        forward = ReshardPlan(f"fwd{tag}", (a.engine_id, b.engine_id), (h,), drop_sources=True,
                              checksums=checksums)
        reports = self.reshard(forward, f"S->H {order} {tag}")
        key = f"fwd{tag}/{h.engine_id}"
        self.check_manifests(reports, h, key)
        self.verify(key, h)
        state = self.state()
        for rank, s in state.items():
            self.assertEqual(s["keys"], (key,))
            self.assertEqual(s["bytes"], expected_bytes_per_host(16))
            for kept in (a.engine_id, b.engine_id):  # every source buffer is freed
                self.assertEqual(s["kept_alive"].get(kept, []), [], kept)
        if checksums:
            for report in reports:
                self.assertTrue(report["checksums_verified"])
                self.assertEqual(len(report["targets"][h.engine_id].checksums), len(self.leaves))

        backward = ReshardPlan(f"bwd{tag}", (key,), (a2, b2), drop_sources=True, checksums=checksums)
        reports_back = self.reshard(backward, f"H->S {order} {tag}")
        for rank, report in enumerate(reports_back):
            spec = a2 if rank in a2.hosts else b2
            manifest = report["targets"][spec.engine_id]
            self.assertEqual((manifest.hosts, manifest.tp_size, manifest.leaves),
                             (spec.hosts, 8, self.leaves))
            self.assertEqual(manifest.bytes_per_host, expected_bytes_per_host(8))
        for spec in (a2, b2):
            self.verify(f"bwd{tag}/{spec.engine_id}", spec)
        for rank, s in self.state().items():
            host_spec = a2 if rank in a2.hosts else b2
            self.assertEqual(s["keys"], (f"bwd{tag}/{host_spec.engine_id}",))
            self.assertEqual(s["bytes"], expected_bytes_per_host(8))
        return reports, reports_back

    def test_round_trip_halves_interleaved(self):
        reports, back = self.cycle("halves_interleaved", "cold", checksums=True)
        n_replicated = sum(1 for l in self.leaves if l.tp_axis is None)
        for report in reports:
            paths = report["paths"]
            self.assertEqual((paths["local_leaves"], paths["general_leaves"]), (len(self.leaves), 0))
            self.assertGreater(paths["sliced_buffers"], 0)
            self.assertGreaterEqual(paths["copied_buffers"], n_replicated)
            # No bytes crossed chips: every target buffer was built on the chip of its source shard.
            self.assertEqual(set(report["provenance"]), {l.path for l in self.leaves})
            for path, entries in report["provenance"].items():
                self.assertEqual(len(entries), 4, path)
                for target_chip, source, shard, source_chip, sub in entries:
                    self.assertEqual(target_chip, source_chip, path)
        for report in back:
            self.assertEqual(report["paths"]["local_leaves"], n_replicated)
        warm, _ = self.cycle("halves_interleaved", "warm", checksums=False)
        self.assertEqual([r["compiles"] for r in warm], [0] * 4, "warm local S->H must not compile")

    def test_round_trip_natural(self):
        reports, back = self.cycle("natural", "cold", checksums=True)
        n_sharded = sum(1 for l in self.leaves if l.tp_axis is not None)
        for report in reports:
            self.assertEqual(report["paths"]["general_leaves"], n_sharded)
        self.cycle("natural", "warm", checksums=False)

    def test_s_block_to_s_diag(self):
        a, b = spec_of("A", (0, 1), 8), spec_of("B", (2, 3), 8)
        c, d = spec_of("C", (0, 3), 8), spec_of("D", (1, 2), 8)
        self.build([a, b], checksums=True)
        reports = self.reshard(ReshardPlan("diag", ("A", "B"), (c, d), checksums=True), "S->Sdiag")
        self.verify("diag/C", c)
        self.verify("diag/D", d)
        for report in reports:
            self.assertEqual(report["paths"]["general_leaves"],
                             sum(1 for l in self.leaves if l.tp_axis is not None))
            self.assertTrue(report["checksums_verified"])
        for rank, s in self.state().items():
            self.assertEqual(s["keys"], ("diag/C",) if rank in c.hosts else ("diag/D",))
            self.assertEqual(s["bytes"], expected_bytes_per_host(8))

    def test_every_program_spans_whole_hosts(self):
        """libtpu on a v5e 4x4 block compiles only programs over unions of whole hosts outside
        ``ILLEGAL_HOST_SETS``; any other chip set holding chip 0 aborts the process
        (net_util.cc:2304). The metadata gather once used one chip per host and killed host
        kernel 2 in G2 run vproc-G2-23efaf8. CPU compiles anything, so check the device sets."""
        spmd(self.block, "record_meshes", [(r, ()) for r in range(4)])
        self.cycle("natural", "hosts", checksums=True)  # general path: gather, stitch, repack, transfer
        self.cycle("halves_interleaved", "hostsl", checksums=True)  # chip-local path
        sets = {m for per_host in spmd(self.block, "recorded_meshes", [(r, ()) for r in range(4)])
                for m in per_host}
        self.assertTrue(any(kind == "mesh" for kind, _ in sets))
        for kind, ids in sets:
            hosts = {i // CPU_ID_STRIDE for i in ids}
            whole = {h * CPU_ID_STRIDE + c for h in hosts for c in range(self.block.devices)}
            self.assertEqual(set(ids), whole, f"{kind} program over part of hosts {sorted(hosts)}: {ids}")
            self.assertNotIn(frozenset(hosts), ILLEGAL_HOST_SETS, f"{kind} program over {sorted(hosts)}")

    def test_non_replica_sources_rejected(self):
        a, b = spec_of("A", (0, 1), 8), spec_of("B", (2, 3), 8)
        h = spec_of("H", (0, 1, 2, 3), 16)
        self.build([a, b], divisors={"B": 32})
        with self.assertRaises(Exception) as ctx:
            self.reshard(ReshardPlan("bad", ("A", "B"), (h,)), "bad")
        self.assertIn("not a replica", str(ctx.exception))


    def test_v5e_ring_orders_2d_sources(self):
        """EngineAdapter's real v5e ring orders, sources on (data=1, model=8) meshes."""
        reports, back = self.cycle("v5e_ring", "cold2d", checksums=True, src_order="v5e_ring", mesh="2d")
        n_sharded = sum(1 for l in self.leaves if l.tp_axis is not None)
        for report in reports:
            paths = report["paths"]
            self.assertEqual(paths["chip_local_positions"], 4)  # target positions 0, 11, 12, 15
            self.assertEqual(paths["target_positions"], 16)
            self.assertEqual(paths["general_leaves"], n_sharded)
        warm, warm_back = self.cycle("v5e_ring", "warm2d", checksums=False, src_order="v5e_ring", mesh="2d")
        self.assertEqual([r["compiles"] for r in warm + warm_back], [0] * 8)

    def test_seven_axis_sources(self):
        reports, _ = self.cycle("halves_interleaved", "cold7d", checksums=False, mesh="7d")
        for report in reports:
            self.assertEqual(report["paths"]["general_leaves"], 0)
            self.assertEqual(report["paths"]["chip_local_positions"], 16)

    def test_rewrap_onto_runner_mesh_is_transfer_free(self):
        for order, kind in (("halves_interleaved", "2d"), ("v5e_ring", "2d"), ("natural", "7d")):
            tag = f"{order}{kind}"
            a, b = spec_of(f"A{tag}", (0, 1), 8, "natural"), spec_of(f"B{tag}", (2, 3), 8, "natural")
            h = spec_of(f"H{tag}", (0, 1, 2, 3), 16, order)
            self.build([a, b])
            self.reshard(ReshardPlan(f"p{tag}", (a.engine_id, b.engine_id), (h,)), f"rewrap {tag}")
            out = spmd(self.block, "rewrap", [(r, (f"p{tag}/{h.engine_id}", h, kind)) for r in range(4)])
            self.assertEqual({o["leaves"] for o in out}, {len(self.leaves)})
            self.timings.append({"label": f"rewrap {tag}", "same_buffers": [o["same_buffers"] for o in out],
                                "zero_copy_rebinds": [o["zero_copy_rebinds"] for o in out]})
            self.assertEqual({o["zero_copy_rebinds"] for o in out}, {len(self.leaves)})

    def test_unsupported_source_mesh_rejected(self):
        a = spec_of("A", (0, 1), 8, "natural", "bad")
        with self.assertRaises(Exception) as ctx:
            self.build([a])
        self.assertIn("need model=8", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
