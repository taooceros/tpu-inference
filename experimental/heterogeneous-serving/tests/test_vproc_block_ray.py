"""C5-C7 integration: BlockController over a 4 x 4 CPU block of Ray host kernels, synthetic engines
served through SyntheticFrontend actors (namespace omp_hetero), InMemoryEndpoints.

S-block -> H-block -> S-block with resident weights, token equality against the oracle in every
phase, one runtime for the whole cycle, a report JSON, then a killed host kernel and the rebuild,
then a killed frontend and the rebuild.
"""
import json
import os
import tempfile
import time
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc.controller import BlockController, Timeouts
from hetero.vproc.endpoints import InMemoryEndpoints
from hetero.vproc.synthetic_engine import expected_tokens

REQUEST = {"prompt": [[1, 2, 3], [4, 5], [9]], "steps": 6}
SEED = 7
LAYOUT_MODELS = {"H-block": {"device_order": "halves_interleaved"}}


class BlockRayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ray
        if ray.is_initialized():  # an earlier test module (test_gateway_lifecycle) leaves its session open
            ray.shutdown()
        ray.init(address="local", num_cpus=8, include_dashboard=False, logging_level="ERROR")
        from hetero.vproc.host_kernel import spawn_block
        # names/namespace are what the frontends use to find the kernels (controller default refs)
        cls.actors = spawn_block(4, 4, name_prefix="vproc-kernel-e0", actor_options={"namespace": "omp_hetero"})
        cls.endpoints = InMemoryEndpoints()
        cls.ctl = BlockController(
            cls.actors, cls.endpoints, engine_model={"seed": SEED}, layout_models=LAYOUT_MODELS,
            timeouts=Timeouts(health_s=10, start_s=300, reshard_s=300, stop_s=120, step_s=120),
            checksums=True)

    @classmethod
    def tearDownClass(cls):
        import ray
        try:
            cls.ctl.close()
        finally:
            ray.shutdown()

    def serve_and_check(self, label, engine_ids):
        """Serve every engine and compare its tokens to the oracle for that engine's spec."""
        for engine_id in engine_ids:
            spec = self.ctl.engines[engine_id]
            out = self.ctl.serve(engine_id, REQUEST)
            self.assertTrue(out["serve"]["ranks_agree"], label)
            check = self.ctl.measure.check_tokens(f"{label}/{engine_id}", out["tokens"],
                                                  expected_tokens(spec, REQUEST))
            self.assertTrue(check["equal"], check)

    def test_switch_cycle_then_kernel_failure(self):
        ctl = self.ctl
        deploy = ctl.deploy("S-block")
        self.assertEqual([r["name"] for r in deploy["phases"]], ["build", "publish"])
        runtime_ids = ctl.runtime_ids()
        self.assertEqual(len(set(runtime_ids)), 1)
        pids = [d.pid for d in ctl.describe()]
        self.serve_and_check("S", ["S-block-0", "S-block-1"])
        import ray
        for engine_id in ("S-block-0", "S-block-1"):  # frontends are named actors in the gateway's namespace
            ray.get_actor(f"vproc-{engine_id}-g0", namespace="omp_hetero")

        to_h = ctl.switch("H-block")
        self.serve_and_check("H", ["H-block-0"])
        to_s = ctl.switch("S-block")
        self.serve_and_check("S-again", ["S-block-0", "S-block-1"])

        # one runtime, same processes, across the whole cycle
        self.assertEqual(ctl.runtime_ids(), runtime_ids)
        self.assertEqual([d.pid for d in ctl.describe()], pids)
        self.assertEqual(to_h["generations"], {"H-block-0": 0})
        self.assertEqual(to_s["generations"], {"S-block-0": 1, "S-block-1": 1})
        for report in (to_h, to_s):
            self.assertEqual(report["mode"], "resident")
            self.assertEqual(list(report["segments"]),
                             ["drain", "withdraw", "stop_export", "reshard", "build", "publish"])
            self.assertTrue(all(v >= 0 for v in report["segments"].values()))
            reshard = next(r for r in report["phases"] if r["name"] == "reshard")
            self.assertTrue(reshard["parts"], "kernels report reshard segments")
            self.assertLessEqual(report["max_clock_uncertainty_s"], 1.0)
            build = next(r for r in report["phases"] if r["name"] == "build")
            self.assertGreater(build["kernel_span_s"], 0)
            self.assertLess(build["kernel_outside_bracket_s"], 0.5)  # kernel clocks corrected onto the driver's

        out = os.environ.get("VPROC_REPORT") or os.path.join(tempfile.mkdtemp(), "vproc_block_ray.json")
        ctl.write_report(out)
        data = json.load(open(out))
        self.assertEqual(data["verdict"]["status"], "passed", data["verdict"])
        self.assertEqual([s["target"] for s in data["switches"]], ["H-block", "S-block"])
        self.assertEqual(data["block"]["hosts"], 4)
        self.assertTrue(all("total_s" in s and "compiles" in s for s in data["switches"]))

        # kill one host kernel: the next request rebuilds the block and is served token-equal
        ray.kill(self.actors[2], no_restart=True)
        started = time.time()
        out = ctl.serve("S-block-1", REQUEST)
        self.assertTrue(out["serve"]["retried"])
        self.assertEqual(out["serve"]["generation"], 2)
        self.assertEqual(out["tokens"], expected_tokens(ctl.engines["S-block-1"], REQUEST))
        (rebuild,) = ctl.measure.rebuilds
        self.assertTrue(rebuild["ok"], rebuild)
        self.assertEqual(rebuild["layout"], "S-block")
        self.assertIn(2, rebuild["failed_hosts"])
        self.assertEqual(list(rebuild["segments"]), ["teardown", "respawn", "describe", "build", "publish"])
        new_ids = ctl.runtime_ids()
        self.assertEqual(len(set(new_ids)), 1)
        self.assertNotEqual(new_ids, runtime_ids)  # a new block runtime
        self.assertEqual(sorted(ctl.engines), ["S-block-0", "S-block-1"])
        self.serve_and_check("S-rebuilt", ["S-block-0", "S-block-1"])
        # the rebuilt block switches again with resident weights
        ctl.switch("H-block")
        self.serve_and_check("H-rebuilt", ["H-block-0"])
        self.assertEqual(ctl.report()["verdict"]["status"], "passed")

        # a killed frontend is the same failure domain: the block is rebuilt, the request retried
        ray.kill(ctl.frontends["H-block-0"], no_restart=True)
        out = ctl.serve("H-block-0", REQUEST)
        self.assertTrue(out["serve"]["retried"])
        self.assertEqual(out["tokens"], expected_tokens(ctl.engines["H-block-0"], REQUEST))
        self.assertEqual(len(ctl.measure.rebuilds), 2)
        self.assertEqual(ctl.measure.rebuilds[1]["failed_frontends"], ["H-block-0"])
        self.assertEqual(ctl.report()["verdict"]["status"], "passed")
        self.assertLess(time.time() - started, 300)


if __name__ == "__main__":
    unittest.main()
