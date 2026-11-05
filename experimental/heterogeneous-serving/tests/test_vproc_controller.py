"""C5-C7 unit tests with in-process fake host kernels and frontends (no jax, no Ray, < 10 s)."""
import json
import os
import tempfile
import threading
import time
import unittest
import uuid

from hetero.vproc import measure as M
from hetero.vproc.controller import (BlockController, BlockDown, EngineNotFound, FrontendCallError,
                                     FrontendFailure, KernelCallError, KernelDead, KernelFailure,
                                     SwitchFailed, ThreadCaller, Timeouts, frontend_name)
from hetero.vproc.endpoints import (EndpointError, GatewayEndpoints, InMemoryEndpoints,
                                    engine_descriptor, layout_digest, layout_info)
from hetero.vproc.types import (BLOCK_LAYOUTS, EngineSpec, HostDescription, IllegalHostSet,
                                ResidentManifest)

FAST = Timeouts(describe_s=2, health_s=0.5, stop_s=10, reshard_s=10, start_s=10, step_s=10,
                drain_wait_s=5, serve_wait_s=10)
FRONTEND_COMPILES = 3


class Cluster:
    """Shared state of the fakes: ordered event log, SPMD barriers, exports, every object ever made."""

    def __init__(self, hosts=4, offsets=None):
        self.hosts, self.offsets = hosts, offsets or {}
        self.log, self.lock = [], threading.Lock()
        self.barriers, self.exported = {}, {}
        self.fail = {}  # (method, rank) -> exception to raise
        self.runtimes = 0
        self.kernels = []
        self.all_kernels, self.frontends, self.by_name = [], [], {}
        self.spawn()

    def spawn(self):
        self.runtimes += 1
        self.runtime_id = f"runtime-{self.runtimes}"
        self.kernels = [FakeKernel(r, self) for r in range(self.hosts)]
        self.all_kernels.append(self.kernels)
        return self.kernels

    def make_frontend(self, spec, refs, settings, name):
        self.record("fe.create", spec.engine_id)
        front = FakeFrontend(spec, refs, settings, name, self, list(self.kernels))
        self.frontends.append(front)
        self.by_name[name] = front
        return front

    def record(self, *event):
        with self.lock:
            self.log.append(event)

    def barrier(self, key, parties):
        with self.lock:
            return self.barriers.setdefault(key, threading.Barrier(parties))

    def events(self, *kinds):
        return [e for e in self.log if e[0] in kinds]


class FakeKernel:
    def __init__(self, rank, cluster):
        self.rank, self.cluster = rank, cluster
        self.runtime_id = cluster.runtime_id
        self.spec, self.store = None, {}
        self.dead = self.sick = False
        self.hang_health = threading.Event()
        self.offset = cluster.offsets.get(rank, 0.0)

    def _enter(self, method):
        if self.dead:
            raise KernelDead(f"host {self.rank} is gone")
        failure = self.cluster.fail.pop((method, self.rank), None)
        if failure is not None:
            raise failure

    def now(self):
        return time.time() + self.offset

    def describe(self):
        self._enter("describe")
        return HostDescription(self.rank, self.cluster.hosts, "cpu",
                               tuple(self.rank * 2048 + i for i in range(4)), f"h{self.rank}", 1,
                               self.runtime_id, self.spec, tuple(self.store), self.now())

    def start_engine(self, spec, resident=None):
        self._enter("start_engine")
        self.cluster.record("start", self.rank, spec.engine_id, resident, spec.generation)
        self.cluster.barrier(("start", self.runtime_id, spec.engine_id, spec.generation), len(spec.hosts)).wait(5)
        t0 = self.now()
        if resident is not None and resident not in self.store:
            raise KeyError(f"no resident state {resident}")
        if self.spec is not None:
            raise RuntimeError("engine already running")
        time.sleep(0.005)
        self.spec = spec
        self.store.pop(resident, None)
        return {"t0": t0, "t1": self.now(), "segments": {"load": 0.002, "graph": 0.003},
                "compiles": 0 if resident else 2}

    def engine_call(self, method, *args):
        self._enter("engine_call")
        assert method == "step" and self.spec is not None
        request = args[0]
        self.cluster.record("engine_call", self.rank, request.get("id"))
        time.sleep(request.get("sleep", 0))
        return {"tokens": [[(sum(p) + i) % 50 for i in range(4)] for p in request["prompts"]]}

    def stop_engine(self, export=True):
        self._enter("stop_engine")
        spec, self.spec = self.spec, None
        self.cluster.record("stop", self.rank, spec.engine_id, export)
        if not export:
            return None
        self.store[spec.engine_id] = object()
        self.cluster.exported[spec.engine_id] = spec.hosts
        return ResidentManifest(spec.engine_id, spec.generation, spec.hosts, spec.tp_size, (), 1)

    def reshard(self, plan):
        self._enter("reshard")
        self.cluster.record("reshard", self.rank, plan.plan_id, plan.sources)
        involved = {h for s in plan.sources for h in self.cluster.exported[s]}
        involved |= {h for t in plan.targets for h in t.hosts}
        assert self.rank in involved
        self.cluster.barrier(("reshard", plan.plan_id), len(involved)).wait(5)
        t0 = self.now()
        for target in plan.targets:
            if self.rank in target.hosts:
                self.store[f"{plan.plan_id}/{target.engine_id}"] = object()
        for source in plan.sources:
            self.store.pop(source, None)
        return {"t0": t0, "t1": self.now(), "segments": {"transfer": 0.001}, "compiles": 0, "targets": {}}

    def health(self):
        self._enter("health")
        if self.hang_health.is_set():
            time.sleep(2.0)
        return {"ok": not self.sick, "hbm": None}

    def shutdown(self):
        self.dead = True


class FakeFrontend:
    """Frontend actor stand-in: serialises the fanout to its view's kernels, speaks the HeteroEngine surface."""

    def __init__(self, spec, refs, settings, name, cluster, kernels):
        self.spec, self.refs, self.settings, self.name = spec, refs, settings, name
        self.cluster, self.kernels = cluster, kernels
        self.layout_id, self.layout_digest = settings["layout_id"], settings["layout_digest"]
        self.dead = self.poisoned = False
        self.inflight, self.max_concurrent_fanout = 0, 0
        self._fanout = threading.Lock()
        self.fanout_orders = []

    def _enter(self):
        if self.dead:
            raise KernelDead(f"frontend {self.name} is gone")

    def initialize(self):
        self._enter()
        self.cluster.record("fe.init", self.spec.engine_id)
        return {"segments": {"kv_alloc": 0.001, "warmup": 0.002}, "compiles": FRONTEND_COMPILES}

    def identity(self):
        self._enter()
        return {"instance_id": self.spec.engine_id, "layout_id": self.layout_id,
                "layout_digest": self.layout_digest, "generation": self.spec.generation,
                "node_ip": f"10.0.0.{self.spec.hosts[0]}", "endpoint_port": 8000 + self.spec.hosts[0],
                "state": "ready", "tp_size": self.spec.tp_size}

    def status(self):
        self._enter()
        return {"generation": self.spec.generation, "state": "ready", "total_blocks": 10, "free_blocks": 10,
                "block_size": 16}

    def adopt_identity(self, layout_id, layout_digest_):
        self._enter()
        self.cluster.record("fe.adopt", self.spec.engine_id, layout_id)
        self.layout_id, self.layout_digest = layout_id, layout_digest_
        return self.identity()

    def drain(self, timeout_s=0):
        self._enter()
        self.cluster.record("fe.drain", self.spec.engine_id)
        return {"drained": True, "inflight": self.inflight}

    def generate(self, request):
        self._enter()
        if self.poisoned:
            raise RuntimeError("engine poisoned by an earlier failed step")
        self.inflight += 1
        try:
            with self._fanout:  # one fanout at a time: every host sees the requests in the same order
                self.fanout_orders.append(request.get("id"))
                try:
                    results = ThreadCaller(self.kernels).call(
                        [(h, "engine_call", ("step", request)) for h in self.spec.hosts], 10)
                except (KernelFailure, KernelCallError) as error:
                    raise RuntimeError(f"a host kernel failed under the frontend: {error}") from error
            tokens = [r["tokens"] for r in results]
            return {"tokens": tokens[0], "ranks_agree": all(t == tokens[0] for t in tokens)}
        finally:
            self.inflight -= 1

    def health(self):
        self._enter()
        return {"ok": True, "poisoned": self.poisoned, "inflight": self.inflight}

    def shutdown(self):
        self.cluster.record("fe.shutdown", self.spec.engine_id)
        self.dead = True


class LoggedEndpoints(InMemoryEndpoints):
    def __init__(self, cluster, **kw):
        super().__init__(**kw)
        self.cluster = cluster

    def _enter(self, op, ids):
        self.cluster.record("ep." + op, tuple(ids))
        super()._enter(op, ids)


def make(hosts=4, offsets=None, endpoints=None, **kw):
    cluster = Cluster(hosts, offsets)
    endpoints = endpoints(cluster) if endpoints else LoggedEndpoints(cluster)
    controller = BlockController(cluster.kernels, endpoints, respawn=cluster.spawn, timeouts=FAST,
                                 engine_model={"seed": 1}, frontend_factory=cluster.make_frontend, **kw)
    return cluster, endpoints, controller


def phases(events):
    """Collapse the event log into the sequence of protocol phases (frontend events are not phases)."""
    names = {"ep.drain": "drain", "ep.withdraw": "withdraw", "stop": "stop_export",
             "reshard": "reshard", "start": "build", "ep.publish": "publish"}
    out = []
    for event in events:
        name = names.get(event[0])
        if name is None:
            continue
        if not out or out[-1] != name:
            out.append(name)
        elif name in ("drain", "withdraw", "publish"):
            out.append(name)
    return out


def kinds_collapsed(events):
    out = []
    for event in events:
        if not out or out[-1] != event[0]:
            out.append(event[0])
    return out


REQUEST = {"prompts": [[1, 2, 3], [4, 5]]}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cluster, self.endpoints, self.ctl = make()
        self.addCleanup(self.ctl.close)

    def test_deploy_then_resident_switch_order(self):
        report = self.ctl.deploy("S-block")
        self.assertEqual(sorted(self.ctl.engines), ["S-block-0", "S-block-1"])
        self.assertEqual(self.ctl.placement, {0: "S-block-0", 1: "S-block-0", 2: "S-block-1", 3: "S-block-1"})
        self.assertEqual([r["name"] for r in report["phases"]], ["build", "publish"])
        self.assertEqual(sorted(self.ctl.frontends), ["S-block-0", "S-block-1"])
        mark = len(self.cluster.log)
        switch = self.ctl.switch("H-block")
        events = self.cluster.log[mark:]
        self.assertEqual(phases(events), ["drain", "withdraw", "stop_export", "reshard", "build", "publish"])
        # endpoint drain, then the frontends drain, then withdraw; frontends are shut down before any
        # kernel stops its engine; the targets' frontends are created and initialised after every start
        self.assertEqual(kinds_collapsed(events),
                         ["ep.drain", "fe.drain", "ep.withdraw", "fe.shutdown", "stop", "reshard", "start",
                          "fe.create", "fe.init", "ep.publish"])
        kinds = [e[0] for e in events]
        self.assertEqual(kinds.count("stop"), 4)
        self.assertEqual(kinds.count("reshard"), 4)
        self.assertEqual(kinds.count("start"), 4)
        self.assertEqual(events[0][1], ("S-block-0", "S-block-1"))
        self.assertEqual({e[1] for e in events if e[0] == "stop"}, {0, 1, 2, 3})
        self.assertTrue(all(e[3] is True for e in events if e[0] == "stop"))  # export
        plan = switch["plan_id"]
        self.assertEqual({e[3] for e in events if e[0] == "start"}, {f"{plan}/H-block-0"})
        self.assertEqual(set(self.endpoints.published), {"H-block-0"})
        self.assertEqual(sorted(self.ctl.frontends), ["H-block-0"])
        self.assertEqual(self.ctl.placement, {h: "H-block-0" for h in range(4)})
        self.assertEqual(switch["mode"], "resident")
        self.assertEqual(list(switch["segments"]), ["drain", "withdraw", "stop_export", "reshard", "build", "publish"])
        build = next(r for r in switch["phases"] if r["name"] == "build")
        self.assertEqual(build["parts"]["kv_alloc"], 0.001)  # frontend initialize time is in the build segment
        self.assertEqual(build["parts"]["warmup"], 0.002)
        self.assertEqual(build["parts"]["load"], 0.002)

    def test_compiles_sum_kernel_reshard_and_frontend(self):
        self.ctl.deploy("S-block")
        resident = self.ctl.switch("H-block")
        # reshard 0 + start_engine 0 (resident) + one frontend initialize of 3
        self.assertEqual(resident["compiles"], FRONTEND_COMPILES)
        build = next(r for r in resident["phases"] if r["name"] == "build")
        self.assertEqual((build["compiles"], build["compiles_by_frontend"]), (3, {"H-block-0": 3}))
        cold = self.ctl.switch("S-block", resident=False)
        # cold start_engine compiles 2 on every host (max over hosts) + two frontends of 3 each
        self.assertEqual(cold["compiles"], 2 + 2 * FRONTEND_COMPILES)
        self.assertEqual(self.ctl.measure.deploys[0]["compiles"], 2 + 2 * FRONTEND_COMPILES)

    def test_cold_switch_skips_reshard_and_export(self):
        self.ctl.deploy("S-block")
        mark = len(self.cluster.log)
        switch = self.ctl.switch("H-block", resident=False)
        events = self.cluster.log[mark:]
        self.assertEqual(phases(events), ["drain", "withdraw", "stop_export", "build", "publish"])
        self.assertTrue(all(e[3] is False for e in events if e[0] == "stop"))
        self.assertTrue(all(e[3] is None for e in events if e[0] == "start"))
        self.assertNotIn("reshard", switch["segments"])
        self.assertEqual(switch["mode"], "cold")
        self.assertEqual(switch["compiles"], 2 + FRONTEND_COMPILES)

    def test_generation_increments_per_rebuilt_id(self):
        gens = [self.ctl.deploy("S-block")["generations"]]
        gens.append(self.ctl.switch("H-block")["generations"])
        gens.append(self.ctl.switch("S-block")["generations"])
        gens.append(self.ctl.switch("H-block")["generations"])
        self.assertEqual(gens, [{"S-block-0": 0, "S-block-1": 0}, {"H-block-0": 0},
                                {"S-block-0": 1, "S-block-1": 1}, {"H-block-0": 1}])
        self.assertEqual(self.ctl.engines["H-block-0"].generation, 1)
        self.assertEqual(self.cluster.frontends[-1].name, frontend_name(self.ctl.engines["H-block-0"]))
        self.assertEqual(self.cluster.frontends[-1].name, "vproc-H-block-0-g1")

    def test_serve_goes_through_the_frontend_once_per_request(self):
        self.ctl.deploy("S-block")
        out = self.ctl.serve("S-block-1", REQUEST)
        self.assertEqual(out["tokens"], [[6, 7, 8, 9], [9, 10, 11, 12]])
        self.assertTrue(out["serve"]["ranks_agree"])
        self.assertEqual(out["serve"]["hosts"], [2, 3])
        with self.assertRaises(EngineNotFound):
            self.ctl.serve("H-block-0", REQUEST)

    def test_concurrent_serves_reach_the_hosts_in_one_order(self):
        self.ctl.deploy("S-block")
        threads = [threading.Thread(target=lambda i=i: self.ctl.serve("S-block-0", {**REQUEST, "id": i, "sleep": 0.01}))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        front = self.ctl.frontends["S-block-0"]
        self.assertEqual(sorted(front.fanout_orders), list(range(8)))
        # both hosts of the engine saw the requests in the same order (the frontend serialises the fanout)
        orders = {rank: [e[2] for e in self.cluster.events("engine_call") if e[1] == rank] for rank in (0, 1)}
        self.assertEqual(sorted(orders[0]), list(range(8)))
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[0], front.fanout_orders)

    def test_same_layout_is_noop_and_kept_engines_stay(self):
        self.ctl.deploy("S-block")
        mark = len(self.cluster.log)
        self.assertTrue(self.ctl.switch("S-block")["noop"])
        self.assertEqual(self.cluster.log[mark:], [])
        # S-block-0 unchanged, second engine replaced: only the changed engine is touched
        keep = self.ctl.engines["S-block-0"]
        kept_frontend = self.ctl.frontends["S-block-0"]
        new = EngineSpec("other", "synthetic", (2, 3), 8, {"seed": 1})
        report = self.ctl.switch([keep, new])
        self.assertEqual((report["kept"], report["sources"], report["targets"]), (["S-block-0"], ["S-block-1"], ["other"]))
        self.assertEqual({e[1] for e in self.cluster.log[mark:] if e[0] == "stop"}, {2, 3})
        self.assertEqual(set(self.endpoints.published), {"S-block-0", "other"})
        self.assertIs(self.ctl.frontends["S-block-0"], kept_frontend)  # the kept engine keeps its frontend
        self.assertFalse(kept_frontend.dead)
        # ... which adopts the new layout identity, so every frontend names the published layout
        layout = self.endpoints.layouts[-1]
        self.assertEqual(layout["engines"], ["S-block-0", "other"])
        self.assertIn(("fe.adopt", "S-block-0", layout["layout_id"]), self.cluster.log)
        for front in self.ctl.frontends.values():
            self.assertEqual((front.layout_id, front.layout_digest), (layout["layout_id"], layout["layout_digest"]))

    def test_failed_switch_in_unrelated_endpoint_step_leaves_block_intact(self):
        self.ctl.deploy("S-block")
        self.endpoints.fail_on["drain"] = EndpointError("gateway busy")
        with self.assertRaises(EndpointError):
            self.ctl.switch("H-block")
        self.assertEqual(sorted(self.ctl.engines), ["S-block-0", "S-block-1"])
        self.assertEqual(self.cluster.events("stop"), [])
        self.assertEqual(self.ctl.measure.rebuilds, [])
        self.assertEqual(self.ctl.serve("S-block-0", REQUEST)["serve"]["generation"], 0)


class LayoutIdentityTests(unittest.TestCase):
    def test_one_digest_per_layout_in_settings_publish_and_identity(self):
        cluster, endpoints, ctl = make()
        self.addCleanup(ctl.close)
        deploy = ctl.deploy("S-block")
        layout = endpoints.layouts[-1]
        self.assertEqual((layout["layout_id"], layout["layout_digest"]), (deploy["layout_id"], deploy["layout_digest"]))
        self.assertEqual(layout["engines"], ["S-block-0", "S-block-1"])
        for front in cluster.frontends:  # frontends were built knowing the layout they will be published in
            self.assertEqual(front.settings["layout_id"], layout["layout_id"])
            self.assertEqual(front.settings["layout_digest"], layout["layout_digest"])
        specs = [ctl.engines[i] for i in sorted(ctl.engines)]
        self.assertEqual(layout["layout_digest"], layout_digest(layout["layout_id"], specs))
        h = ctl.switch("H-block")
        self.assertNotEqual(h["layout_digest"], layout["layout_digest"])
        self.assertEqual(endpoints.layouts[-1]["layout_digest"], h["layout_digest"])
        s = ctl.switch("S-block")  # same engine ids, new generations: a new digest again
        self.assertEqual(s["layout_id"], layout["layout_id"])
        self.assertNotEqual(s["layout_digest"], layout["layout_digest"])

    def test_frontend_identity_disagreeing_with_the_layout_blocks_publish(self):
        cluster, endpoints, ctl = make()
        self.addCleanup(ctl.close)
        original = cluster.make_frontend

        def wrong(spec, refs, settings, name):
            return original(spec, refs, {**settings, "layout_digest": "stale"}, name)

        ctl.frontend_factory = wrong
        with self.assertRaises(RuntimeError) as caught:
            ctl.deploy("S-block")
        self.assertIn("identity disagrees", str(caught.exception))
        self.assertEqual(endpoints.published, {})


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.cluster, self.endpoints, self.ctl = make()
        self.addCleanup(self.ctl.close)

    def test_illegal_host_sets_refused_before_anything_runs(self):
        for bad in ([0], [0, 2], [1], [0, 1, 1], [0, 4]):
            spec = EngineSpec("bad", "synthetic", tuple(bad), 4 * len(bad))
            with self.assertRaises(IllegalHostSet, msg=str(bad)):
                self.ctl.deploy([spec])
        overlap = [EngineSpec("a", "synthetic", (0, 1), 8), EngineSpec("b", "synthetic", (1, 2), 8)]
        with self.assertRaises(IllegalHostSet):
            self.ctl.deploy(overlap)
        with self.assertRaises(ValueError):
            self.ctl.deploy("no-such-layout")
        with self.assertRaises(ValueError):  # TP degree does not match 2 hosts x 4 chips
            self.ctl.deploy([EngineSpec("a", "synthetic", (0, 1), 16)])
        self.assertEqual(self.cluster.events("start", "stop", "reshard", "ep.publish", "fe.create"), [])
        self.assertEqual(self.ctl.engines, {})

    def test_illegal_switch_target_leaves_running_layout_untouched(self):
        self.ctl.deploy("S-block")
        mark = len(self.cluster.log)
        with self.assertRaises(IllegalHostSet):
            self.ctl.switch([EngineSpec("c0", "synthetic", (0, 2), 8), EngineSpec("c1", "synthetic", (1, 3), 8)])
        self.assertEqual(self.cluster.log[mark:], [])
        self.assertEqual(sorted(self.ctl.engines), ["S-block-0", "S-block-1"])

    def test_every_named_layout_is_legal(self):
        for name in BLOCK_LAYOUTS:
            self.ctl.switch(name) if self.ctl.engines else self.ctl.deploy(name)
        self.assertEqual(sorted(self.ctl.engines), ["H-block-0"])

    def test_resident_switch_refused_when_sources_leave_the_target_hosts(self):
        self.ctl.deploy("S-block")
        mark = len(self.cluster.log)
        shrink = [EngineSpec("small", "synthetic", (0, 1), 8, {"seed": 1})]
        with self.assertRaises(ValueError):
            self.ctl.switch(shrink)
        self.assertEqual(self.cluster.log[mark:], [])  # nothing ran, nothing stopped
        self.assertEqual(sorted(self.ctl.engines), ["S-block-0", "S-block-1"])
        self.ctl.switch(shrink, resident=False)
        self.assertEqual(sorted(self.ctl.engines), ["small"])
        self.assertEqual(self.ctl.engines["small"].generation, 0)  # the refused attempt did not burn a generation

    def test_layout_models_apply_per_layout(self):
        cluster, _, ctl = make(layout_models={"H-block": {"device_order": "halves_interleaved"}})
        self.addCleanup(ctl.close)
        ctl.deploy("S-block")
        ctl.switch("H-block")
        self.assertEqual(ctl.engines["H-block-0"].model, {"seed": 1, "device_order": "halves_interleaved"})
        self.assertEqual(ctl.engines["H-block-0"].tp_size, 16)


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.cluster, self.endpoints, self.ctl = make()
        self.addCleanup(self.ctl.close)
        self.ctl.deploy("S-block")
        self.old_kernels = list(self.cluster.kernels)
        self.old_frontends = list(self.cluster.frontends)

    def assert_rebuilt_on_s(self, generation):
        self.assertEqual(sorted(self.ctl.engines), ["S-block-0", "S-block-1"])
        self.assertEqual({e.generation for e in self.ctl.engines.values()}, {generation})
        self.assertEqual(set(self.endpoints.published), {"S-block-0", "S-block-1"})
        self.assertEqual({e.generation for e in self.endpoints.published.values()}, {generation})
        self.assertEqual(self.ctl.runtime_ids(), ["runtime-2"] * 4)
        self.assertTrue(all(k.dead for k in self.old_kernels))  # the old block is torn down whole
        self.assertTrue(all(f.dead for f in self.old_frontends))  # and its frontends with it
        self.assertEqual(sorted(f.spec.engine_id for f in self.ctl.frontends.values()), ["S-block-0", "S-block-1"])

    def test_dead_host_during_serve_rebuilds_block_and_retries(self):
        self.cluster.kernels[2].dead = True
        out = self.ctl.serve("S-block-1", REQUEST)
        self.assertEqual(out["tokens"], [[6, 7, 8, 9], [9, 10, 11, 12]])
        self.assertTrue(out["serve"]["retried"])
        self.assertEqual(out["serve"]["generation"], 1)
        self.assert_rebuilt_on_s(1)
        (rebuild,) = self.ctl.measure.rebuilds
        self.assertTrue(rebuild["ok"])
        self.assertEqual(rebuild["failed_hosts"], [2])
        self.assertEqual(list(rebuild["segments"]), ["teardown", "respawn", "describe", "build", "publish"])
        self.assertEqual(rebuild["layout"], "S-block")
        # the healthy engine on the other pair was rebuilt too: one failure domain
        self.assertEqual(self.ctl.serve("S-block-0", REQUEST)["serve"]["generation"], 1)

    def test_dead_host_without_retry_raises_after_rebuilding(self):
        self.cluster.kernels[0].dead = True
        with self.assertRaises(KernelCallError):
            self.ctl.serve("S-block-0", REQUEST, retry=False)
        self.assert_rebuilt_on_s(1)

    def test_ordinary_request_error_does_not_rebuild(self):
        self.cluster.fail[("engine_call", 0)] = ValueError("bad prompt")
        with self.assertRaises(FrontendCallError):
            self.ctl.serve("S-block-0", REQUEST)
        self.assertEqual(self.ctl.measure.rebuilds, [])
        self.assertEqual(self.ctl.serve("S-block-0", REQUEST)["serve"]["generation"], 0)

    def test_dead_frontend_rebuilds_the_whole_block(self):
        self.ctl.frontends["S-block-1"].dead = True
        out = self.ctl.serve("S-block-1", REQUEST)
        self.assertTrue(out["serve"]["retried"])
        self.assert_rebuilt_on_s(1)
        (rebuild,) = self.ctl.measure.rebuilds
        self.assertEqual((rebuild["failed_frontends"], rebuild["failed_hosts"]), (["S-block-1"], [2, 3]))

    def test_poisoned_frontend_rebuilds_on_serve(self):
        self.ctl.frontends["S-block-0"].poisoned = True
        out = self.ctl.serve("S-block-0", REQUEST)  # generate raises, health says poisoned: rebuild, retry
        self.assertTrue(out["serve"]["retried"])
        self.assert_rebuilt_on_s(1)
        self.assertEqual(self.ctl.measure.rebuilds[0]["failed_frontends"], ["S-block-0"])

    def test_poisoned_frontend_found_by_switch_preflight(self):
        self.ctl.frontends["S-block-1"].poisoned = True
        probe = self.ctl.health_check()
        self.assertEqual((probe["healthy"], probe["failed"], probe["failed_frontends"]), (False, [], ["S-block-1"]))
        report = self.ctl.switch("H-block")
        self.assertEqual(len(self.ctl.measure.rebuilds), 1)
        self.assertEqual(set(self.ctl.engines), {"H-block-0"})
        self.assertEqual(report["epoch"], 1)
        self.assertTrue(all(f.dead for f in self.old_frontends))

    def test_dead_host_found_by_switch_preflight_rebuilds_first(self):
        self.cluster.kernels[3].dead = True
        report = self.ctl.switch("H-block")
        self.assertEqual(len(self.ctl.measure.rebuilds), 1)
        self.assertEqual(set(self.ctl.engines), {"H-block-0"})
        self.assertEqual(report["generations"], {"H-block-0": 0})
        self.assertEqual(report["epoch"], 1)

    def test_kernel_dies_mid_switch_rebuilds_source_layout(self):
        self.cluster.fail[("reshard", 1)] = KernelDead("host 1 vanished")
        with self.assertRaises(SwitchFailed) as caught:
            self.ctl.switch("H-block")
        self.assertFalse(caught.exception.report["ok"])
        self.assertEqual([r["name"] for r in caught.exception.report["phases"]],
                         ["drain", "withdraw", "stop_export", "reshard"])
        self.assertTrue(caught.exception.rebuild["ok"])
        self.assert_rebuilt_on_s(1)
        self.assertEqual(self.ctl.layout_name, "S-block")
        self.assertEqual(self.ctl.measure.report()["verdict"]["status"], "failed")

    def test_remote_error_in_target_build_rebuilds_source_layout(self):
        self.cluster.fail[("start_engine", 0)] = RuntimeError("engine build blew up")
        with self.assertRaises(SwitchFailed):
            self.ctl.switch("H-block")
        self.assert_rebuilt_on_s(1)
        # and the next switch works on the new block
        self.assertEqual(self.ctl.switch("H-block")["generations"], {"H-block-0": 1})

    def test_unhealthy_and_hung_kernels_trigger_rebuild(self):
        self.assertIsNone(self.ctl.ensure_healthy())
        self.assertEqual(self.ctl.measure.rebuilds, [])
        self.cluster.kernels[1].sick = True
        probe = self.ctl.health_check()
        self.assertEqual((probe["healthy"], probe["failed"]), (False, [1]))
        self.assertTrue(self.ctl.ensure_healthy()["ok"])
        self.assert_rebuilt_on_s(1)
        self.old_kernels = list(self.cluster.kernels)
        self.cluster.kernels[2].hang_health.set()
        probe = self.ctl.health_check()
        self.assertEqual((probe["failed"], probe["kernels"][2]["status"]), ([2], "timeout"))
        self.assertTrue(self.ctl.ensure_healthy()["ok"])
        self.assertEqual({e.generation for e in self.ctl.engines.values()}, {2})

    def test_failed_rebuild_leaves_block_down_until_rebuild_succeeds(self):
        calls = []

        def respawn():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("no capacity")
            return self.cluster.spawn()

        self.ctl.respawn = respawn
        self.cluster.kernels[0].dead = True
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-1", REQUEST)
        self.assertFalse(self.ctl.measure.rebuilds[0]["ok"])
        self.assertTrue(self.ctl.rebuild("operator")["ok"])
        self.assertEqual(self.ctl.serve("S-block-0", REQUEST)["serve"]["generation"], 1)


class RecoveryLeakTests(unittest.TestCase):
    """A replacement block that fails to come up must not outlive the failed rebuild."""

    def setUp(self):
        self.cluster, self.endpoints, self.ctl = make()
        self.addCleanup(self.ctl.close)
        self.ctl.deploy("S-block")
        self.cluster.kernels[1].dead = True  # the failure that starts the rebuild
        self.old = len(self.cluster.all_kernels)

    def assert_replacement_cleaned_up(self, publish_attempted=False):
        replacements = self.cluster.all_kernels[self.old:]
        self.assertEqual(len(replacements), 1)
        self.assertTrue(all(k.dead for k in replacements[0]), "replacement kernels were not shut down")
        self.assertTrue(all(f.dead for f in self.cluster.frontends), "a frontend survived")
        self.assertEqual(self.endpoints.published, {})
        self.assertEqual((self.ctl.engines, self.ctl.frontends, self.ctl.down), ({}, {}, True))
        self.assertFalse(self.ctl.measure.rebuilds[-1]["ok"])

    def test_describe_failure(self):
        self.cluster.fail[("describe", 3)] = RuntimeError("runtime does not match")
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assert_replacement_cleaned_up()

    def test_start_failure(self):
        self.cluster.fail[("start_engine", 2)] = RuntimeError("engine build blew up")
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assert_replacement_cleaned_up()
        self.assertEqual(len(self.cluster.events("fe.create")), 2)  # none made for the failed build

    def test_frontend_initialize_failure_kills_the_created_frontends(self):
        made = []
        real = self.cluster.make_frontend

        def factory(spec, refs, settings, name):
            front = real(spec, refs, settings, name)
            made.append(front)
            if len(made) == 4:  # the second frontend of the rebuilt S-block (two at deploy)
                front.initialize = lambda: (_ for _ in ()).throw(RuntimeError("warm-up failed"))
            return front

        made.extend([None, None])  # the two deploy-time frontends were made before this wrapper
        self.ctl.frontend_factory = factory
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assertTrue(all(f.dead for f in made if f is not None))
        self.assert_replacement_cleaned_up()

    def test_publish_failure_withdraws_and_kills(self):
        self.endpoints.fail_on["publish"] = EndpointError("gateway rejected the layout")
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assert_replacement_cleaned_up()

    def test_partially_published_replacement_is_withdrawn(self):
        real = self.endpoints.publish

        def publish_then_fail(specs, frontends=None, layout=None):
            real(specs, frontends=frontends, layout=layout)
            raise EndpointError("gateway reconcile failed after registering")

        self.endpoints.publish = publish_then_fail
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assertIn(("ep.withdraw", ("S-block-0", "S-block-1")), self.cluster.log[-6:])
        self.assert_replacement_cleaned_up()

    def test_operator_rebuild_recovers_afterwards(self):
        self.cluster.fail[("describe", 0)] = RuntimeError("transient")
        with self.assertRaises(BlockDown):
            self.ctl.serve("S-block-0", REQUEST)
        self.assertTrue(self.ctl.rebuild("operator")["ok"])
        self.assertEqual(self.ctl.serve("S-block-1", REQUEST)["serve"]["generation"], 1)


class GateTests(unittest.TestCase):
    def test_switch_waits_for_inflight_requests_and_blocks_new_ones(self):
        cluster, endpoints, ctl = make()
        self.addCleanup(ctl.close)
        ctl.deploy("S-block")
        results = {}

        def slow():
            results["slow"] = ctl.serve("S-block-0", {**REQUEST, "sleep": 0.4})

        thread = threading.Thread(target=slow)
        thread.start()
        time.sleep(0.1)
        started = time.time()
        report = ctl.switch("H-block")
        thread.join()
        self.assertGreaterEqual(report["segments"]["drain"], 0.25)  # waited for the request
        self.assertIn("tokens", results["slow"])
        with self.assertRaises(EngineNotFound):
            ctl.serve("S-block-0", REQUEST)
        self.assertEqual(ctl.serve("H-block-0", REQUEST)["serve"]["hosts"], [0, 1, 2, 3])
        self.assertLess(time.time() - started, 5)


class ClockAndReportTests(unittest.TestCase):
    def test_offsets_from_rpc_midpoints(self):
        cluster, _, ctl = make(offsets={0: 0.0, 1: 100.0, 2: -3.5, 3: 0.25})
        self.addCleanup(ctl.close)
        ctl.deploy("S-block")
        for rank, offset in {0: 0.0, 1: 100.0, 2: -3.5, 3: 0.25}.items():
            sample = ctl.measure.clocks.offsets[rank]
            self.assertAlmostEqual(sample.offset_s, offset, delta=max(0.01, sample.rtt_s))

    def test_skewed_kernel_clocks_are_corrected_in_switch_rows(self):
        cluster, _, ctl = make(offsets={1: 100.0, 2: -50.0})
        self.addCleanup(ctl.close)
        ctl.deploy("S-block")
        report = ctl.switch("H-block")
        rows = {r["name"]: r for r in report["phases"]}
        for name in ("reshard", "build"):
            row = rows[name]
            self.assertLessEqual(row["kernel_span_s"], row["duration_s"] + 0.01)
            self.assertLess(row["kernel_outside_bracket_s"], 0.05, name)  # raw skew would be 50-100 s
        self.assertEqual(rows["build"]["parts"], {"load": 0.002, "graph": 0.003, "kv_alloc": 0.001, "warmup": 0.002})
        self.assertEqual(rows["reshard"]["parts"], {"transfer": 0.001})

    def test_recorder_math_on_a_fake_clock(self):
        now = [1000.0]
        clocks = M.ClockMap({0: M.ClockOffset(0, 10.0, 0.02, 3), 1: M.ClockOffset(1, -5.0, 0.04, 3)})
        rec = M.SwitchRecorder(clocks, clock=lambda: now[0])
        rec.begin()
        now[0] += 0.5  # outside any phase
        with rec.phase("drain"):
            now[0] += 2.0
        with rec.phase("reshard", [0, 1]) as ph:
            start = now[0]
            now[0] += 4.0
            ph.results = {0: {"t0": start + 10.0 + 0.1, "t1": start + 10.0 + 3.6, "segments": {"a": 1.0, "b": 2.0},
                              "compiles": 2},
                          1: {"t0": start - 5.0 + 0.2, "t1": start - 5.0 + 3.9, "segments": {"a": 1.5},
                              "compiles": {"x": 1, "y": 3}}}
        with rec.phase("build") as ph:
            now[0] += 1.0
            ph.results = {0: {"compiles": 1}, 1: {"compiles": 2}}
            ph.frontends = {"e0": {"segments": {"warmup": 0.7}, "compiles": 5}, "e1": {"segments": {"warmup": 0.9}, "compiles": 4}}
        out = rec.finish()
        self.assertAlmostEqual(out["total_s"], 7.5)
        self.assertEqual(out["segments"], {"drain": 2.0, "reshard": 4.0, "build": 1.0})
        self.assertAlmostEqual(out["unaccounted_s"], 0.5)
        self.assertAlmostEqual(sum(out["segments"].values()) + out["unaccounted_s"], out["total_s"])
        row = out["phases"][1]
        self.assertAlmostEqual(row["kernel_span_s"], 3.8)  # 0.1 .. 3.9 after correction
        self.assertAlmostEqual(row["rpc_overhead_s"], 0.2)
        self.assertAlmostEqual(row["start_s"], 2.5)
        self.assertEqual(row["parts"], {"a": 1.5, "b": 2.0})
        self.assertEqual((row["compiles"], row["compiles_by_host"]), (4, {"0": 2, "1": 4}))
        build = out["phases"][2]  # kernels: max(1, 2); frontends: 5 + 4
        self.assertEqual((build["compiles"], build["compiles_by_frontend"]), (2 + 9, {"e0": 5, "e1": 4}))
        self.assertEqual(build["parts"], {"warmup": 0.9})
        self.assertEqual(out["compiles"], 4 + 11)
        self.assertAlmostEqual(out["max_clock_uncertainty_s"], 0.02)

    def test_phase_row_records_failure(self):
        rec = M.SwitchRecorder(M.ClockMap())
        with self.assertRaises(ValueError):
            with rec.phase("stop_export"):
                raise ValueError("boom")
        self.assertFalse(rec.rows[0]["ok"])
        self.assertIn("boom", rec.rows[0]["error"])

    def test_estimate_offsets_picks_smallest_rtt_round(self):
        ticks = iter([0.0, 1.0, 10.0, 10.1, 20.0, 20.5])  # rtts 1.0, 0.1, 0.5
        remote = iter([500.5, 510.05, 520.9])
        clocks = M.estimate_offsets(lambda rank: {"clock_s": next(remote)}, [7], rounds=3, clock=lambda: next(ticks))
        sample = clocks.offsets[7]
        self.assertAlmostEqual(sample.rtt_s, 0.1)
        self.assertAlmostEqual(sample.offset_s, 510.05 - 10.05)
        self.assertAlmostEqual(clocks.to_driver(7, 600.0), 100.0)

    def test_token_equality(self):
        self.assertTrue(M.token_check("S", [[1, 2], [3]], [[1, 2], [3]])["equal"])
        bad = M.token_check("H", [[1, 2], [3, 9]], [[1, 2], [3, 4]])
        self.assertFalse(bad["equal"])
        self.assertEqual(bad["mismatches"], [{"key": 1, "got_len": 2, "ref_len": 2, "first_diff": 1}])
        short = M.token_check("H", [[1]], {0: [1], 1: [2]})
        self.assertEqual(short["mismatches"][0]["got_len"], None)

    def test_hbm_summary(self):
        self.assertIsNone(M.hbm_summary({0: None, 1: None}))

        class Cpu:
            id = 0

            def memory_stats(self):
                return None

        class Tpu:
            id = 3

            def memory_stats(self):
                return {"bytes_in_use": 7, "peak_bytes_in_use": 9, "bytes_limit": 16, "other": 1}

        self.assertIsNone(M.device_memory_stats([Cpu()]))
        stats = M.device_memory_stats([Cpu(), Tpu()])
        self.assertEqual(stats, {3: {"bytes_in_use": 7, "peak_bytes_in_use": 9, "bytes_limit": 16}})
        summary = M.hbm_summary({0: stats, 1: {4: {"bytes_in_use": 11, "peak_bytes_in_use": 12, "bytes_limit": 16}}})
        self.assertEqual((summary["chips"], summary["max_bytes_in_use_per_chip"],
                          summary["max_peak_bytes_in_use_per_chip"]), (2, 11, 12))

    def test_compare_constants(self):
        self.assertEqual((M.INPLACE_ENGINE_WARM_S, M.O3_S), (100.6, 27.5))
        c = M.compare(55.0)
        self.assertAlmostEqual(c["vs_inplace_engine"], 55.0 / 100.6)
        self.assertEqual((c["faster_than_inplace_engine"], c["faster_than_o3"]), (True, False))

    def test_report_json_roundtrip_and_verdict(self):
        cluster, _, ctl = make()
        self.addCleanup(ctl.close)
        ctl.deploy("S-block")
        ctl.switch("H-block")
        ctl.switch("S-block", resident=False)
        out = ctl.serve("S-block-0", REQUEST)
        ctl.measure.check_tokens("S", out["tokens"], out["tokens"])
        with tempfile.TemporaryDirectory() as tmp:
            path = ctl.write_report(os.path.join(tmp, "sub", "report.json"))
            data = json.loads(path.read_text())
        self.assertEqual(data["schema"], 1)
        self.assertEqual([s["target"] for s in data["switches"]], ["H-block", "S-block"])
        self.assertEqual([s["mode"] for s in data["switches"]], ["resident", "cold"])
        self.assertEqual(data["verdict"]["status"], "passed", data["verdict"])
        self.assertEqual(data["switches"][0]["comparison"]["inplace_engine_warm_s"], 100.6)
        self.assertEqual(data["block"]["hosts"], 4)
        self.assertIsNone(data["switches"][0]["hbm"])  # kernels report no HBM (CPU)
        data["token_checks"].append(M.token_check("bad", [[1]], [[2]]))
        self.assertEqual(M.verdict(data)["status"], "failed")
        self.assertIn("tokens bad", M.verdict(data)["failures"][0])
        self.assertTrue(M.verdict(data, expect_no_compile=True)["failures"])  # cold switch compiled


# --------------------------------------------------------------------------- gateway

class FakeRegistry:
    """ServingInstanceRegistry surface (registry.py:100-218), same stale-generation rule."""

    def __init__(self):
        self.records, self.layouts, self.log = {}, {}, []

    def register(self, d):
        existing = self.records.get(d["instance_id"])
        if existing and d["generation"] < existing["generation"]:
            return {"accepted": False, "reason": "stale_generation"}
        self.records[d["instance_id"]] = dict(d)
        self.log.append(("register", d["instance_id"], d["layout_id"], d["generation"]))
        return {"accepted": True, "generation": d["generation"]}

    def withdraw(self, instance_id, generation=None):
        row = self.records.get(instance_id)
        self.log.append(("withdraw", instance_id, generation))
        if row is None or (generation is not None and generation != row["generation"]):
            return False
        del self.records[instance_id]
        return True

    def get(self, instance_id):
        return self.records.get(instance_id)

    def get_layout(self, layout_id):
        return self.layouts.get(layout_id)

    def snapshot(self, layout_id=None):
        rows = []
        for record in self.records.values():
            if layout_id is not None and record["layout_id"] != layout_id:
                continue
            body = dict(record)
            body["routable"] = body.get("state") == "ready" and bool(body.get("endpoint_host") and body.get("endpoint_port"))
            body["health_age_s"] = 0.0
            rows.append(body)
        return sorted(rows, key=lambda r: r["instance_id"])

    def put_layout(self, layout):
        self.layouts[layout["layout_id"]] = layout
        self.log.append(("put_layout", layout["layout_id"], layout.get("layout_digest"),
                         [i["instance_id"] for i in layout["instances"]]))


class FakeGatewayCore:
    """OmpGateway as GatewayEndpoints uses it: maintenance, retarget, join, finish and the reconcile
    checks of ``hetero/gateway/service.py:243-310`` (digest equality, namespace, actor identity)."""

    NAMESPACE = "omp_hetero"

    def __init__(self, registry, actors, layout_id):
        self.registry, self.actors, self.layout_id = registry, actors, layout_id
        self.maintenance, self.joining, self.log = {}, set(), []
        self.clients = {}  # instance_id -> (row, actor)

    @staticmethod
    def _binding(row):
        return tuple(row.get(k) for k in ("instance_id", "layout_id", "generation", "engine_actor_name",
                                          "config_digest"))

    def _reconcile(self, joined=frozenset(), restore=False):
        rows = self.registry.snapshot(self.layout_id)
        layout = self.registry.get_layout(self.layout_id)
        if not layout:
            raise RuntimeError("registered static layout is missing")
        expected = {row["instance_id"] for row in layout["instances"]}
        joining = set(self.joining) - set(joined)
        excluded = (set(self.maintenance.get("instances", {})) if not restore else set()) | joining
        active = [row for row in rows if row["instance_id"] not in excluded]
        if ({row["instance_id"] for row in active} != expected - excluded or not excluded <= expected
                or any(row.get("config_digest") != layout["layout_digest"]
                       or row.get("layout_id") != self.layout_id for row in rows)):
            raise RuntimeError("registry differs from the declared static layout outside maintenance")
        before = {name: row for name, (row, _) in self.clients.items()}
        if joined:
            live = {row["instance_id"] for row in active}
            if live - set(before) != set(joined) or set(before) - live or any(
                    self._binding(row) != self._binding(before[row["instance_id"]])
                    for row in active if row["instance_id"] in before):
                raise RuntimeError("joining must be purely additive to the live bindings")
        if self.maintenance:
            for row in active:
                original = self.maintenance["instances"].get(row["instance_id"])
                if original is not None and (not restore or row["instance_id"] not in self.maintenance["drained"]
                                             or row["generation"] <= original["generation"]):
                    raise RuntimeError("maintenance requires an acknowledged drain and a fresh generation")
        fresh = {}
        for row in active:
            name = row["instance_id"]
            if not row["routable"] or row["state"] != "ready" or row.get("ray_namespace") != self.NAMESPACE:
                raise RuntimeError(f"instance is not ready in the owned namespace: {name}")
            actor = self.actors.get(row["engine_actor_name"])
            if actor is None or actor.dead:
                raise RuntimeError(f"no live actor {row['engine_actor_name']} in {self.NAMESPACE}")
            previous = self.clients.get(name)
            if previous is not None and actor is not previous[1] and row["generation"] <= previous[0]["generation"]:
                raise RuntimeError(f"replacement actor reused a retired generation: {name}")
            identity = actor.identity()
            wanted = {"instance_id": name, "layout_id": self.layout_id, "generation": row["generation"],
                      "node_ip": row["endpoint_host"], "layout_digest": row["config_digest"], "state": "ready"}
            if any(identity.get(key) != value for key, value in wanted.items()):
                raise RuntimeError(f"actor and registry identity disagree: {name}: {identity} vs {wanted}")
            status = actor.status()
            if status.get("generation") != row["generation"] or status.get("state") != "ready":
                raise RuntimeError(f"actor generation is not admitting: {name}")
            fresh[name] = (row, actor)
        if joined:
            self.clients.update({n: v for n, v in fresh.items() if n in joined})
        else:
            self.clients = fresh
        return {"instances": sorted(self.clients)}

    def reconcile(self, bind_labels=True):
        self.log.append(("reconcile", self.layout_id))
        return self._reconcile()

    def begin_maintenance(self, instance_ids, *, reason):
        assert reason.strip() and set(instance_ids) <= set(self.clients)
        self.maintenance = {"id": uuid.uuid4().hex, "reason": reason, "drained": {},
                            "instances": {i: dict(self.clients[i][0]) for i in sorted(instance_ids)}}
        self.clients = {k: v for k, v in self.clients.items() if k not in instance_ids}
        self.log.append(("begin_maintenance", sorted(instance_ids)))
        return dict(self.maintenance)

    def drain_maintenance_instance(self, instance_id, *, maintenance_id, timeout_s=600.0):
        assert self.maintenance["id"] == maintenance_id and instance_id in self.maintenance["instances"]
        receipt = {"generation": self.maintenance["instances"][instance_id]["generation"], "drain": {"drained": True}}
        self.maintenance["drained"][instance_id] = receipt
        self.log.append(("drain", instance_id))
        return receipt

    def finish_maintenance(self, *, maintenance_id):
        if self.maintenance.get("id") != maintenance_id:
            raise RuntimeError("maintenance scope changed")
        if set(self.maintenance["drained"]) != set(self.maintenance["instances"]):
            raise RuntimeError("finish requires every maintained instance to be drained")
        result = self._reconcile(restore=True)
        self.maintenance = {}
        self.log.append(("finish_maintenance",))
        return result

    def retarget_layout(self, new_layout_id, joining_ids):
        joining = set(joining_ids)
        if self.joining or not joining or not self.maintenance or new_layout_id == self.layout_id:
            raise RuntimeError("retarget requires declared maintenance, a new layout and a joining set")
        if set(self.maintenance["drained"]) != set(self.maintenance["instances"]):
            raise RuntimeError("retarget requires every maintained instance to be drained")
        live = {(i, r["generation"]) for i, r in self.registry.records.items()}
        if any((n, o["generation"]) in live for n, o in self.maintenance["instances"].items()):
            raise RuntimeError("retarget requires the released instances withdrawn from the registry")
        layout = self.registry.get_layout(new_layout_id)
        if not layout:
            raise RuntimeError("target layout is not registered")
        expected = {row["instance_id"] for row in layout["instances"]}
        if not joining <= expected or set(self.clients) != expected - joining:
            raise RuntimeError("kept engines and the joining set must exactly partition the target layout")
        previous = (self.layout_id, self.maintenance, self.clients)
        self.layout_id, self.maintenance, self.joining = new_layout_id, {}, joining
        try:
            result = self._reconcile()
        except Exception:
            self.layout_id, self.maintenance, self.clients = previous
            self.joining = set()
            raise
        self.log.append(("retarget", new_layout_id, sorted(joining)))
        return {**result, "layout_id": new_layout_id, "joining": sorted(joining)}

    def join_instances(self, instance_ids, policy_after=None):
        ids = frozenset(instance_ids)
        if not ids or not ids <= self.joining or self.maintenance:
            raise RuntimeError("join names instances that are not joining this layout")
        result = self._reconcile(joined=ids)
        self.joining -= ids
        self.log.append(("join", sorted(ids), policy_after))
        return {**result, "joined": sorted(ids)}


def direct(target, method, *args, **kwargs):
    return getattr(target, method)(*args, **kwargs)


class EndpointTests(unittest.TestCase):
    def spec(self, engine_id, hosts, generation=0):
        return EngineSpec(engine_id, "synthetic", hosts, 4 * len(hosts), {"seed": 1}, generation)

    def test_in_memory_endpoints_rules(self):
        ep = InMemoryEndpoints()
        s, h = self.spec("S0", (0, 1)), self.spec("H0", (0, 1, 2, 3))
        with self.assertRaises(EndpointError):
            ep.drain(["S0"])
        ep.publish([s])
        with self.assertRaises(EndpointError):
            ep.publish([s])
        ep.drain(["S0"])
        self.assertEqual(ep.draining, {"S0"})
        self.assertEqual(ep.withdraw(["S0", "ghost"]), {"withdrawn": ["S0"], "missing": ["ghost"]})
        ep.publish([h], frontends={"H0": {"actor_name": "a"}}, layout=layout_info("L", [h]))
        self.assertEqual(ep.ops(), ["drain", "publish", "publish", "drain", "withdraw", "publish"])
        self.assertEqual(list(ep.published), ["H0"])
        self.assertEqual(ep.layouts[-1]["engines"], ["H0"])

    def test_descriptor_addresses_the_frontend_actor_not_a_host_kernel(self):
        from hetero.registry import InstanceDescriptor
        spec = self.spec("e", (0, 1), 3)
        frontend = {"actor_name": "vproc-e-g3", "namespace": "omp_hetero",
                    "identity": {"node_ip": "10.1.1.1", "endpoint_port": 9000, "max_model_len": 4096}}
        body = engine_descriptor(spec, "L", "digest-1", frontend)
        InstanceDescriptor(**body)  # every key is a descriptor field, every required field is present
        self.assertEqual((body["engine_actor_name"], body["ray_namespace"]), ("vproc-e-g3", "omp_hetero"))
        self.assertEqual((body["generation"], body["tp_size"], body["endpoint_host"], body["endpoint_port"]),
                         (3, 8, "10.1.1.1", 9000))
        self.assertEqual((body["config_digest"], body["max_model_len"], body["state"]), ("digest-1", 4096, "ready"))
        with self.assertRaises(EndpointError):  # no node_ip: nothing to bind discovery to
            engine_descriptor(spec, "L", "d", {"actor_name": "x", "namespace": "omp_hetero", "identity": {}})

    def gateway_controller(self, gateway_layout="vproc-S-block"):
        """Controller + fake frontends + GatewayEndpoints over a faithful fake gateway/registry."""
        registry = FakeRegistry()
        holder = {}

        def endpoints(cluster):
            gateway = FakeGatewayCore(registry, cluster.by_name, gateway_layout)
            holder["gateway"] = gateway
            return GatewayEndpoints(gateway, registry, rpc=direct, policy_after={"migration_enabled": False})

        cluster, gw_endpoints, ctl = make(endpoints=endpoints)
        self.addCleanup(ctl.close)
        return cluster, registry, holder["gateway"], gw_endpoints, ctl

    def test_gateway_accepts_publish_through_a_full_switch_cycle(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        ctl.deploy("S-block")
        self.assertEqual(sorted(gateway.clients), ["S-block-0", "S-block-1"])
        digest = registry.layouts["vproc-S-block"]["layout_digest"]
        for row in registry.records.values():  # descriptors address frontends, one digest for the layout
            self.assertEqual((row["config_digest"], row["ray_namespace"]), (digest, "omp_hetero"))
            self.assertTrue(row["engine_actor_name"].startswith("vproc-S-block-"))
        self.assertEqual(registry.records["S-block-1"]["endpoint_host"], "10.0.0.2")

        ctl.switch("H-block")
        self.assertEqual((gateway.layout_id, sorted(gateway.clients)), ("vproc-H-block", ["H-block-0"]))
        ops = [e[0] for e in gateway.log]
        self.assertEqual(ops, ["reconcile", "begin_maintenance", "drain", "drain", "retarget", "join"])
        self.assertEqual(gateway.log[-1][2], {"migration_enabled": False})
        self.assertEqual(registry.records["H-block-0"]["config_digest"],
                         registry.layouts["vproc-H-block"]["layout_digest"])
        self.assertEqual(sorted(registry.records), ["H-block-0"])  # sources were withdrawn

        ctl.switch("S-block")
        self.assertEqual((gateway.layout_id, sorted(gateway.clients)), ("vproc-S-block", ["S-block-0", "S-block-1"]))
        self.assertEqual({r["generation"] for r in registry.records.values()}, {1})
        self.assertEqual(ctl.serve("S-block-0", REQUEST)["serve"]["generation"], 1)

    def test_gateway_adopts_kept_engines_with_the_new_layout_digest(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        ctl.deploy("S-block")
        keep = ctl.engines["S-block-0"]
        ctl.switch([keep, EngineSpec("other", "synthetic", (2, 3), 8, {"seed": 1})])
        layout = registry.layouts[gateway.layout_id]
        self.assertEqual(sorted(gateway.clients), ["S-block-0", "other"])
        # the retained descriptor and the new one both carry the layout's digest, and the kept
        # frontend identity was adopted (the gateway compares them)
        self.assertEqual({r["config_digest"] for r in registry.records.values()}, {layout["layout_digest"]})
        self.assertEqual(registry.records["S-block-0"]["generation"], 0)
        self.assertEqual(ctl.frontends["S-block-0"].identity()["layout_digest"], layout["layout_digest"])

    def test_gateway_rejects_a_descriptor_with_another_digest(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        endpoints.descriptor_for = lambda spec, layout_id, digest, frontend: {
            **engine_descriptor(spec, layout_id, digest, frontend, endpoint_port=1), "config_digest": "other"}
        with self.assertRaises(RuntimeError) as caught:
            ctl.deploy("S-block")
        self.assertIn("registry differs", str(caught.exception))

    def test_gateway_rejects_a_host_kernel_descriptor(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        endpoints.descriptor_for = lambda spec, layout_id, digest, frontend: {
            **engine_descriptor(spec, layout_id, digest, frontend), "engine_actor_name": f"vproc-kernel-e0-{spec.hosts[0]}"}
        with self.assertRaises(RuntimeError) as caught:
            ctl.deploy("S-block")
        self.assertIn("no live actor", str(caught.exception))

    def test_gateway_after_block_rebuild_reregisters_new_generations(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        ctl.deploy("S-block")
        cluster.kernels[1].dead = True
        out = ctl.serve("S-block-0", REQUEST)
        self.assertEqual(out["serve"]["generation"], 1)
        self.assertEqual(sorted(gateway.clients), ["S-block-0", "S-block-1"])
        self.assertEqual({r["generation"] for r in registry.records.values()}, {1})
        self.assertEqual(ctl.measure.rebuilds[0]["ok"], True)

    def test_gateway_failed_switch_then_rebuild_finishes_the_maintenance(self):
        cluster, registry, gateway, endpoints, ctl = self.gateway_controller()
        ctl.deploy("S-block")
        cluster.fail[("reshard", 1)] = KernelDead("host 1 vanished")
        with self.assertRaises(SwitchFailed):
            ctl.switch("H-block")
        self.assertEqual(gateway.maintenance, {})
        self.assertEqual((gateway.layout_id, sorted(gateway.clients)), ("vproc-S-block", ["S-block-0", "S-block-1"]))
        self.assertIn(("finish_maintenance",), gateway.log)
        self.assertEqual({r["generation"] for r in registry.records.values()}, {1})

    def test_publish_without_layout_or_frontends_is_refused(self):
        endpoints = GatewayEndpoints(object(), FakeRegistry(), rpc=direct)
        with self.assertRaises(EndpointError):
            endpoints.publish([self.spec("e", (0, 1))])


class CallerTests(unittest.TestCase):
    def test_thread_caller_is_concurrent_and_fails_fast(self):
        barrier = threading.Barrier(3)

        class K:
            def meet(self):
                barrier.wait(2)
                return "ok"

            def die(self):
                raise KernelDead("gone")

            def hang(self):
                time.sleep(1.5)

        caller = ThreadCaller([K(), K(), K()])
        self.assertEqual(caller.call([(r, "meet", ()) for r in range(3)], 5), ["ok"] * 3)
        started = time.time()
        with self.assertRaises(Exception) as caught:
            caller.call([(0, "hang", ()), (1, "die", ())], 5)
        self.assertEqual((caught.exception.rank, caught.exception.kind), (1, "dead"))
        self.assertLess(time.time() - started, 1.0)  # did not wait for the hung host
        with self.assertRaises(Exception) as caught:
            caller.call([(0, "hang", ())], 0.2)
        self.assertEqual(caught.exception.kind, "timeout")


if __name__ == "__main__":
    unittest.main()
