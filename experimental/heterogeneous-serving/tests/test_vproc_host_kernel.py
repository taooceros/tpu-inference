"""C1: host kernel + synthetic engine on the CPU block and on local Ray."""
import os
import time
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc import catalog
from hetero.vproc.cpu_block import CPU_ID_STRIDE, BlockCallError, CpuBlock
from hetero.vproc.synthetic_engine import SyntheticEngine, expected_tokens
from hetero.vproc.types import EngineSpec, ReshardPlan, ResidentManifest

HERE = __name__
FACTORY = "hetero.vproc.host_kernel:HostKernel"
REQUEST = {"prompt": [[1, 2, 3], [4, 5], [9]], "steps": 6}
N_LEAVES = len(catalog.scaled())


def spec_for(engine_id, hosts, order="natural", seed=7, generation=0, **model):
    return EngineSpec(engine_id, "synthetic", tuple(hosts), 4 * len(hosts),
                      {"seed": seed, "device_order": order, **model}, generation)


def probe_adoption(hosts, seed):
    """Per-rank program (all of ``hosts`` run it): adoption validation and repack sensitivity."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding
    from hetero.vproc.synthetic_engine import _partition_spec
    hosts = tuple(hosts)
    spec = spec_for("probe", hosts, seed=seed)
    first = SyntheticEngine(spec)
    first.build(None)
    leaves = first.export()
    out = {}
    wrong_order = SyntheticEngine(spec_for("probe2", hosts, "halves_interleaved", seed))
    try:
        wrong_order.build(leaves)
    except ValueError as error:
        out["order_error"] = str(error)
    good = SyntheticEngine(spec)
    good.build(leaves)
    out["good"] = good.step(REQUEST)["tokens"]
    leaves = good.export()
    fused = next(l for l in good.leaf_specs() if l.fused)
    wrong_tp = spec.tp_size * 2 if spec.tp_size == 8 else 8
    logical = catalog.synthetic_value(fused, seed)
    stored = catalog.to_stored(logical, fused, wrong_tp)
    sharding = NamedSharding(leaves[fused.path].sharding.mesh, _partition_spec(fused))
    leaves[fused.path] = jax.make_array_from_callback(fused.shape, sharding, lambda i: stored[i])
    bad = SyntheticEngine(spec)
    bad.build(leaves)
    out["bad"] = bad.step(REQUEST)["tokens"]
    leaves[fused.path] = leaves[fused.path].astype(np.float16)
    dtype = SyntheticEngine(spec)
    try:
        dtype.build(leaves)
    except ValueError as error:
        out["dtype_error"] = str(error)
    return out


class OracleTests(unittest.TestCase):
    def test_mesh_orders(self):
        class D:
            def __init__(self, id):
                self.id = id
        devices = [D(i) for i in range(8)]
        natural = spec_for("a", (0, 1))
        self.assertEqual(SyntheticEngine.mesh_device_ids(natural, devices), tuple(range(8)))
        halves = spec_for("a", (0, 1), "halves_interleaved")
        self.assertEqual(SyntheticEngine.mesh_device_ids(halves, devices), (0, 4, 1, 5, 2, 6, 3, 7))
        with self.assertRaises(ValueError):
            SyntheticEngine(spec_for("a", (0, 1), "diagonal"))

    def test_oracle_depends_on_seed(self):
        self.assertNotEqual(expected_tokens(spec_for("a", (0, 1), seed=1), REQUEST),
                            expected_tokens(spec_for("a", (0, 1), seed=2), REQUEST))


class CpuBlockKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock(factory=FACTORY)
        cls.runtime_ids = {d.runtime_id for d in cls.block.call_all("describe")}

    @classmethod
    def tearDownClass(cls):
        cls.block.close()

    def tearDown(self):
        for rank, health in enumerate(self.block.call_all("health")):
            if health["engine"]:
                self.block.call([(rank, "stop_engine", (False,))])

    def start_all(self, specs, resident=None):
        """specs: one EngineSpec per engine; each host runs the spec containing it."""
        calls = []
        for spec in specs:
            for rank in spec.hosts:
                calls.append((rank, "start_engine", (spec, resident(spec) if resident else None)))
        return self.block.call(sorted(calls, key=lambda c: c[0]))

    def step_all(self, specs):
        calls = sorted(((r, "engine_call", ("step", REQUEST)) for s in specs for r in s.hosts))
        results = self.block.call(calls)
        by_spec = {}
        for (rank, _, _), result in zip(calls, results):
            spec = next(s for s in specs if rank in s.hosts)
            by_spec.setdefault(spec.engine_id, []).append(result["tokens"])
        for engine_id, rows in by_spec.items():
            self.assertTrue(all(r == rows[0] for r in rows), f"{engine_id}: hosts disagree")
        return {k: v[0] for k, v in by_spec.items()}

    def stop_all(self, specs, export=True):
        calls = sorted((r, "stop_engine", (export,)) for s in specs for r in s.hosts)
        return dict(zip([(c[0]) for c in calls], self.block.call(calls)))

    def test_describe(self):
        self.assertEqual(len(self.runtime_ids), 1)
        for rank, description in enumerate(self.block.call_all("describe")):
            self.assertEqual((description.rank, description.processes, description.platform),
                             (rank, 4, "cpu"))
            self.assertEqual(description.device_ids, tuple(rank * CPU_ID_STRIDE + i for i in range(4)))
            self.assertIsNone(description.engine)
            self.assertEqual(description.runtime_id, next(iter(self.runtime_ids)))
        for health in self.block.call_all("health"):
            self.assertTrue(health["alive"] and health["ok"])
            self.assertIsNone(health["hbm"])  # CPU devices have no memory_stats

    def test_h_block_export_restart(self):
        h = spec_for("h-restart", range(4))
        segments = self.start_all([h])
        self.assertTrue(all({"load", "graph", "warmup"} <= set(s["segments"]) for s in segments))
        expected = expected_tokens(h, REQUEST)
        self.assertEqual(self.step_all([h])["h-restart"], expected)
        for description in self.block.call_all("describe"):
            self.assertEqual(description.engine, h)
        with self.assertRaises(BlockCallError) as raised:
            self.block.call_all("start_engine", h)
        self.assertIn("stop_engine first", str(raised.exception))
        manifests = self.stop_all([h])
        for rank, manifest in manifests.items():
            self.assertIsInstance(manifest, ResidentManifest)
            self.assertEqual((manifest.engine_id, manifest.tp_size, manifest.hosts),
                             ("h-restart", 16, (0, 1, 2, 3)))
            self.assertEqual(len(manifest.leaves), N_LEAVES)
            self.assertGreater(manifest.bytes_per_host, 0)
        for description in self.block.call_all("describe"):
            self.assertIn("h-restart", description.resident)
            self.assertIsNone(description.engine)
        again = spec_for("h-restart", range(4), generation=1)
        segments = self.start_all([again], resident=lambda s: s.engine_id)
        self.assertEqual(self.step_all([again])["h-restart"], expected)
        for description in self.block.call_all("describe"):
            self.assertNotIn("h-restart", description.resident)
        self.assertEqual(self.stop_all([again], export=False), {r: None for r in range(4)})
        self.assertEqual({d.runtime_id for d in self.block.call_all("describe")}, self.runtime_ids)

    def test_s_block_equals_h_block(self):
        seed = 11
        s0, s1 = spec_for("s0", (0, 1), seed=seed), spec_for("s1", (2, 3), seed=seed)
        h = spec_for("h", range(4), seed=seed)
        self.start_all([s0, s1])
        tokens = self.step_all([s0, s1])
        self.stop_all([s0, s1], export=False)
        self.start_all([h])
        h_tokens = self.step_all([h])["h"]
        self.assertEqual(tokens["s0"], h_tokens)
        self.assertEqual(tokens["s1"], h_tokens)
        self.assertEqual(h_tokens, expected_tokens(h, REQUEST))
        self.stop_all([h], export=False)
        self.assertEqual({d.runtime_id for d in self.block.call_all("describe")}, self.runtime_ids)

    def test_build_compile_counts(self):
        """Measured: SyntheticEngine.build lowers+compiles one fresh executable per build, so a
        second build of the same shape in the same process compiles again (no in-process reuse);
        stop/export compiles nothing."""
        spec = spec_for("compiles", (0, 1), catalog={"language_layers": 2, "divisor": 48})
        counts = []
        for _ in range(2):
            results = self.block.call([(r, "start_engine", (spec,)) for r in spec.hosts])
            for result in results:
                self.assertGreater(result["compile_s"], 0.0)
            counts.append([r["compiles"] for r in results])
            self.block.call([(r, "stop_engine", (True,)) for r in spec.hosts])
            for health in self.block.call_all("health", ranks=spec.hosts):
                self.assertEqual(health["last_stop"]["compiles"], 0)
            self.block.call([(r, "start_engine", (spec, spec.engine_id)) for r in spec.hosts])
            self.block.call([(r, "stop_engine", (False,)) for r in spec.hosts])
        self.assertTrue(all(c > 0 for c in counts[0]), counts)
        self.assertEqual(counts[1], counts[0], counts)

    def test_failed_build_leaves_kernel_alive(self):
        bad = EngineSpec("bad", "synthetic", (0, 1), 12, {"seed": 1})
        with self.assertRaises(BlockCallError) as raised:
            self.block.call([(0, "start_engine", (bad,)), (1, "start_engine", (bad,))])
        self.assertIn("tp_size 12", str(raised.exception))
        with self.assertRaises(BlockCallError) as raised:
            self.block.call([(2, "start_engine", (spec_for("x", (0, 1)),))])
        self.assertIn("not in engine hosts", str(raised.exception))
        with self.assertRaises(BlockCallError) as raised:
            self.block.call([(0, "start_engine", (spec_for("r", (0, 1)), "no-such-key"))])
        self.assertIn("no-such-key", str(raised.exception))
        with self.assertRaises(BlockCallError):
            self.block.call([(0, "engine_call", ("step", REQUEST))])
        for health in self.block.call_all("health"):
            self.assertIsNone(health["engine"])
        good = spec_for("after-fail", (0, 1))
        self.start_all([good])
        self.assertEqual(self.step_all([good])["after-fail"], expected_tokens(good, REQUEST))
        self.stop_all([good], export=False)

    def test_adoption_checks(self):
        for hosts in ((0, 1), (0, 1, 2, 3)):
            results = self.block.call([(r, f"{HERE}:probe_adoption", (hosts, 5)) for r in hosts])
            expected = expected_tokens(spec_for("p", hosts, seed=5), REQUEST)
            for out in results:
                self.assertIn("mesh device order", out["order_error"])
                self.assertEqual(out["good"], expected)
                self.assertNotEqual(out["bad"], expected, "a wrong fused layout must change tokens")
                self.assertIn("dtype", out["dtype_error"])

    def test_reshard_s_h_s(self):
        for order in ("halves_interleaved", "natural"):
            with self.subTest(order=order):
                self.s_h_s(order)

    def s_h_s(self, order):
        tag, seed = order[:4], 3
        s = [spec_for(f"{tag}-s0", (0, 1), order, seed), spec_for(f"{tag}-s1", (2, 3), order, seed)]
        h = spec_for(f"{tag}-h", range(4), order, seed, generation=1)
        s2 = [spec_for(f"{tag}-s0", (0, 1), order, seed, 2), spec_for(f"{tag}-s1", (2, 3), order, seed, 2)]
        expected = expected_tokens(h, REQUEST)
        self.start_all(s)
        for tokens in self.step_all(s).values():
            self.assertEqual(tokens, expected)
        self.stop_all(s)
        forward = ReshardPlan(f"{tag}-fwd", tuple(x.engine_id for x in s), (h,), checksums=True)
        results = self.block.call_all("reshard", forward)
        for result in results:
            self.assertIn("segments", result)
            self.assertIn(f"{tag}-h", result["targets"])
        self.start_all([h], resident=lambda x: f"{forward.plan_id}/{x.engine_id}")
        self.assertEqual(self.step_all([h])[h.engine_id], expected)
        self.stop_all([h])
        reverse = ReshardPlan(f"{tag}-rev", (h.engine_id,), tuple(s2))
        self.block.call_all("reshard", reverse)
        self.start_all(s2, resident=lambda x: f"{reverse.plan_id}/{x.engine_id}")
        for tokens in self.step_all(s2).values():
            self.assertEqual(tokens, expected)
        self.stop_all(s2, export=False)
        self.assertEqual({d.runtime_id for d in self.block.call_all("describe")}, self.runtime_ids)


class RayKernelTests(unittest.TestCase):
    def test_spawn_block_health_during_long_call(self):
        import ray
        from hetero.vproc.host_kernel import kill_block, spawn_block
        if ray.is_initialized():  # an earlier test module (test_gateway_lifecycle) leaves its session open
            ray.shutdown()
        ray.init(address="local", num_cpus=8, include_dashboard=False, logging_level="ERROR")
        actors = []
        try:
            actors = spawn_block(4, 4)
            descriptions = ray.get([a.describe.remote() for a in actors])
            self.assertEqual([d.rank for d in descriptions], [0, 1, 2, 3])
            self.assertEqual(len({d.runtime_id for d in descriptions}), 1)
            self.assertEqual(len({d.pid for d in descriptions}), 4)
            h = spec_for("ray-h", range(4))
            ray.get([a.start_engine.remote(h) for a in actors], timeout=120)
            tokens = ray.get([a.engine_call.remote("step", REQUEST) for a in actors], timeout=120)
            for result in tokens:
                self.assertEqual(result["tokens"], expected_tokens(h, REQUEST))
            long_call = actors[0].engine_call.remote("idle", 6.0)
            time.sleep(1.0)
            started = time.time()
            health = ray.get(actors[0].health.remote(), timeout=4)
            description = ray.get(actors[0].describe.remote(), timeout=4)
            self.assertLess(time.time() - started, 3.0)
            self.assertEqual(health["busy"], "engine_call(idle)")
            self.assertEqual(description.engine, h)
            done, _ = ray.wait([long_call], timeout=0)
            self.assertEqual(done, [], "health must answer while the long call is still running")
            self.assertEqual(ray.get(long_call, timeout=30), 6.0)
            self.assertIsNone(ray.get(actors[0].health.remote())["busy"])
            ray.get([a.stop_engine.remote(False) for a in actors], timeout=60)
            kill_block(actors)
            deadline, dead = time.time() + 30, False
            while not dead and time.time() < deadline:  # ray.kill is asynchronous
                try:
                    ray.get(actors[0].health.remote(), timeout=5)
                    time.sleep(0.2)
                except Exception:
                    dead = True
            self.assertTrue(dead, "actor still answering 30 s after kill_block")
            actors = []
        finally:
            if actors:
                kill_block(actors)
            ray.shutdown()


if __name__ == "__main__":
    unittest.main()
