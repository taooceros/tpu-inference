"""Pre-spawned target engines (hetero/prespawn.py, deployment pre-spawn/commit, switch wiring).

CPU only: no Ray cluster, TPU or gateway.  The libtpu lockfile / TPU device node are emulated by a
POSIX lock on a temp file and an open temp "device" file held by a child process.
"""

import asyncio
import fcntl
import multiprocessing as mp
import os
import pathlib
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from hetero import deployment, placement, prespawn, switch
from hetero.engine_actor import EngineSettings
from scripts import switch_dry_run as dry

SCALE = 0.05
# A line read from /proc/locks on a serving TPU host: the worker (pid 256341) holds a POSIX write
# lock on /tmp/libtpu_lockfile (inode 1330595).
REAL_LOCKS_LINE = "87: POSIX  ADVISORY  WRITE 256341 00:19c:1330595 0 EOF\n"


def _hold(lock, dev, ready, stop):
    fd = os.open(lock, os.O_RDWR)
    fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dfd = os.open(dev, os.O_RDWR)
    ready.set()
    stop.wait(30)
    os.close(dfd)
    os.close(fd)


class DevicePredicateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)
        self.lock, self.dev = str(root / "libtpu_lockfile"), str(root / "vfio0")
        for path in (self.lock, self.dev):
            pathlib.Path(path).touch()

    def _holder(self):
        ready, stop = mp.Event(), mp.Event()
        child = mp.Process(target=_hold, args=(self.lock, self.dev, ready, stop))
        child.start()
        self.assertTrue(ready.wait(10))
        self.addCleanup(lambda: (stop.set(), child.join(5)))
        return child, stop

    def test_parses_real_proc_locks_line_and_skips_waiters(self):
        locks = pathlib.Path(self.tmp.name) / "locks"
        locks.write_text(REAL_LOCKS_LINE + "88: POSIX  ADVISORY  WRITE 999 00:19c:7 0 EOF\n"
                         "88: -> POSIX  ADVISORY  WRITE 1234 00:19c:1330595 0 EOF\n")
        with mock.patch.object(prespawn.os, "stat", return_value=SimpleNamespace(st_ino=1330595)):
            self.assertEqual(prespawn.lock_holders("/tmp/libtpu_lockfile", str(locks)), [256341])

    def test_foreign_holder_detected_then_released(self):
        child, stop = self._holder()
        self.assertEqual(prespawn.lock_holders(self.lock), [child.pid])
        self.assertEqual(prespawn.device_holders([self.dev]), [child.pid])
        stop_at = time.time() + 0.4

        def sleep(s):
            if time.time() >= stop_at:
                stop.set()
            time.sleep(s)
        got = prespawn.wait_device_free(lockfile=self.lock, device_globs=[self.dev], timeout_s=10,
                                        poll_s=0.02, sleep=sleep)
        self.assertEqual(got["holders_seen"], {"lock": [child.pid], "vfio": [child.pid]})
        self.assertGreaterEqual(got["lock_free_at"], stop_at - 0.05)
        self.assertGreaterEqual(got["vfio_free_at"], stop_at - 0.05)
        self.assertGreater(got["polls"], 1)

    def test_bounded_wait_raises_instead_of_proceeding(self):
        self._holder()
        with self.assertRaisesRegex(TimeoutError, "TPU still held"):
            prespawn.wait_device_free(lockfile=self.lock, device_globs=[self.dev], timeout_s=0.2, poll_s=0.05)

    def test_free_immediately_and_absent_files(self):
        got = prespawn.wait_device_free(lockfile=self.lock, device_globs=[self.dev], timeout_s=1)
        self.assertEqual((got["polls"], got["holders_seen"]), (1, {}))
        self.assertEqual(prespawn.lock_holders(self.lock + ".absent"), [])
        self.assertEqual(prespawn.device_holders([self.dev + "-none*"]), [])

    def test_own_lock_is_not_foreign(self):
        fd = os.open(self.lock, os.O_RDWR)
        self.addCleanup(os.close, fd)
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual(prespawn.lock_holders(self.lock), [])

    def test_group_busy_after_holder_exit_keeps_waiting(self):
        # O2 attempt 2: /proc shows no holder, but the group is still EBUSY during the kernel release.
        answers = [["/dev/vfio/0"], ["/dev/vfio/0"], []]
        got = prespawn.wait_device_free(lockfile=self.lock, device_globs=[self.dev], timeout_s=5,
                                        poll_s=0.01, busy_groups=lambda: answers.pop(0))
        self.assertEqual((got["busy_polls"], got["polls"]), (2, 3))
        self.assertEqual(got["holders_seen"], {"busy_groups": ["/dev/vfio/0"]})
        self.assertGreaterEqual(got["groups_openable_at"], got["vfio_free_at"])
        with self.assertRaisesRegex(TimeoutError, "busy groups"):
            prespawn.wait_device_free(lockfile=self.lock, device_globs=[self.dev], timeout_s=0.1,
                                      poll_s=0.02, busy_groups=lambda: ["/dev/vfio/0"])

    def test_busy_probe_reports_only_ebusy_and_closes(self):
        import errno

        free, busy = str(pathlib.Path(self.tmp.name) / "7"), str(pathlib.Path(self.tmp.name) / "8")
        pathlib.Path(free).touch()
        pathlib.Path(busy).touch()
        real_open, opened = os.open, []

        def fake_open(path, flags, *a):
            if path == busy:
                raise OSError(errno.EBUSY, "Device or resource busy")
            fd = real_open(path, flags, *a)
            opened.append(fd)
            return fd
        with mock.patch.object(prespawn.os, "open", fake_open):
            got = prespawn.busy_vfio_groups(str(pathlib.Path(self.tmp.name) / "[0-9]*"))
        self.assertEqual(got, [busy])
        for fd in opened:
            with self.assertRaises(OSError):
                os.fstat(fd)  # closed again


class SelfCheckTests(unittest.TestCase):
    def _proc(self, maps="", fds=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        (root / "maps").write_text(maps)
        (root / "fd").mkdir()
        for i, target in enumerate(fds):
            os.symlink(target, root / "fd" / str(i))
        return str(root)

    def test_clean_process(self):
        got = prespawn.self_check(proc_self=self._proc("7f 00 r-xp /usr/lib/libc.so.6\n", ["/dev/null"]))
        self.assertTrue(got["clean"])

    def test_libtpu_mapped_vfio_fd_or_lockfile_fd_is_a_hard_failure(self):
        lib = "7f00-7f01 r-xp 0 00:00 1 /site-packages/libtpu/libtpu.so\n"
        sdk = "7f00-7f01 r-xp 0 00:00 1 /site-packages/libtpu/sdk.so\n"
        self.assertTrue(prespawn.self_check(proc_self=self._proc(sdk))["clean"])  # tpu_info's sdk.so is not libtpu
        for proc in (self._proc(lib), self._proc(fds=["/dev/vfio/0"]),
                     self._proc(fds=[os.path.realpath(prespawn.LIBTPU_LOCKFILE)])):
            with self.assertRaisesRegex(RuntimeError, "touched the TPU runtime before the release gate"):
                prespawn.assert_untouched("parked", proc_self=proc)


class GateTests(unittest.TestCase):
    def test_wait_release_status(self):
        async def scenario():
            gate = prespawn.ReleaseGate(["10.0.0.2", "10.0.0.1"])
            waiter = asyncio.create_task(gate.wait("10.0.0.1", 5))
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            status = await gate.status()
            self.assertEqual(sorted(status["parked"]), ["10.0.0.1"])
            released = await gate.release(["10.0.0.1"])
            receipt = await waiter
            self.assertEqual(receipt["released_at"], released["10.0.0.1"])
            self.assertLessEqual(receipt["parked_at"], receipt["released_at"])
            # A release before the wait opens it immediately; foreign keys are refused both ways.
            await gate.release(["10.0.0.2"])
            self.assertEqual((await gate.wait("10.0.0.2", 0.1))["key"], "10.0.0.2")
            with self.assertRaises(ValueError):
                await gate.wait("10.9.9.9", 0.1)
            with self.assertRaises(ValueError):
                await gate.release(["10.9.9.9"])
            with self.assertRaises(asyncio.TimeoutError):
                await prespawn.ReleaseGate(["a"]).wait("a", 0.05)
        asyncio.run(scenario())

    def test_park_and_wait_timeline(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        lock = os.path.join(tmp.name, "lock")
        pathlib.Path(lock).touch()
        handle = SimpleNamespace(wait=SimpleNamespace(remote=lambda key, t: {"parked_at": 1.0, "released_at": 2.0}))
        with mock.patch.object(prespawn, "self_check", return_value={"clean": True, "at": 0.5}) as check:
            got = prespawn.park_and_wait({"name": "g", "namespace": "ns", "timeout_s": 5, "device_timeout_s": 1},
                                         "10.0.0.1", get=lambda ref, timeout: ref, find=lambda: handle,
                                         lockfile=lock, device_globs=[os.path.join(tmp.name, "none*")])
        self.assertEqual(check.call_count, 2)  # before parking and right after the gate returned
        for key in ("parked_at", "released_at", "gate_returned_at", "lock_free_at", "vfio_free_at",
                    "self_check_parked", "self_check_released"):
            self.assertIn(key, got)
        self.assertEqual((got["key"], got["released_at"]), ("10.0.0.1", 2.0))

    def test_dirty_worker_never_reaches_the_gate(self):
        find = mock.Mock()
        with mock.patch.object(prespawn, "self_check", return_value={"clean": False, "libtpu_mapped": ["x"]}):
            with self.assertRaises(RuntimeError):
                prespawn.park_and_wait({"name": "g", "namespace": "ns", "timeout_s": 5}, "k", find=find)
        find.assert_not_called()


class PlacementAndExecutorTests(unittest.TestCase):
    INSTANCE = SimpleNamespace(hosts=[SimpleNamespace(node_ip="10.0.0.1"), SimpleNamespace(node_ip="10.0.0.2")])

    def test_cpu_only_bundles_omit_the_tpu_key(self):
        self.assertEqual(placement.intended_bundles(self.INSTANCE, 8.0, 0.0),
                         [{"CPU": 8.0, "node:10.0.0.1": 0.001}, {"CPU": 8.0, "node:10.0.0.2": 0.001}])
        tpu = placement.intended_bundles(self.INSTANCE, 8.0, 4.0)
        self.assertEqual([list(b) for b in tpu], [["TPU", "CPU", "node:10.0.0.1"], ["TPU", "CPU", "node:10.0.0.2"]])
        self.assertEqual(placement.tpu_hold_bundles(self.INSTANCE, 4.0)[1], {"TPU": 4.0, "node:10.0.0.2": 0.001})
        name = placement.tpu_hold_pg_name("H", "large_a")
        self.assertEqual(placement.parse_pg_name(name), ("H", "large_a-tpuhold"))

    def test_executor_worker_kwargs_carry_no_tpu(self):
        import ray

        class TpuExecutor:  # tpu_inference ray_distributed_executor_v2.py:135-145 shape
            placement_group = SimpleNamespace(bundle_specs=[{"CPU": 8.0}])

            def _get_actor_resource_kwargs(self):
                return {"num_gpus": 0, "resources": {"TPU": self.placement_group.bundle_specs[0].get("TPU", 0)}}

        cls = prespawn.prespawn_executor_class(TpuExecutor)
        self.assertTrue(issubclass(cls, TpuExecutor))
        self.assertEqual(cls()._get_actor_resource_kwargs(), {"num_gpus": 0})
        actor = ray.remote(type("A", (), {}))
        with self.assertRaisesRegex(ValueError, "not a supported chip configuration"):
            actor.options(num_cpus=0, **TpuExecutor()._get_actor_resource_kwargs())
        actor.options(num_cpus=0, **cls()._get_actor_resource_kwargs())

    def test_cpu_bundles_get_ranks_in_the_normal_driver_first_order(self):
        pg = SimpleNamespace(bundle_specs=[{"CPU": 8.0}] * 4)
        nodes = [{"NodeID": n, "NodeManagerAddress": f"10.0.0.{i}", "Alive": True}
                 for i, n in enumerate(("node-c", "node-a", "node-b", "node-d"))]
        context = SimpleNamespace(get_node_id=lambda: "node-b")
        with mock.patch("ray.util.placement_group_table",
                        return_value={"bundles_to_node_id": {0: "node-c", 1: "node-a", 2: "node-b", 3: "node-d"}}), \
                mock.patch("ray.nodes", return_value=nodes), \
                mock.patch("ray.get_runtime_context", return_value=context):
            order = prespawn.bundles_sorted_by_node_any_device(pg)
        self.assertEqual(order, [(2, "node-b", "10.0.0.2"), (1, "node-a", "10.0.0.1"),
                                 (0, "node-c", "10.0.0.0"), (3, "node-d", "10.0.0.3")])

    def test_rank_discovery_is_swapped_only_inside_init_executor(self):
        import sys
        import types

        module = types.ModuleType("vllm.v1.executor.ray_executor_v2")
        original = object()
        module.get_bundles_sorted_by_node = original
        seen = []

        class Base:
            def _init_executor(self):
                seen.append(module.get_bundles_sorted_by_node)

        packages = {name: sys.modules.get(name) or types.ModuleType(name)
                    for name in ("vllm", "vllm.v1", "vllm.v1.executor")}
        packages["vllm.v1.executor"].ray_executor_v2 = module
        with mock.patch.dict(sys.modules, {**packages, "vllm.v1.executor.ray_executor_v2": module}):
            prespawn.prespawn_executor_class(Base)()._init_executor()
        self.assertEqual(seen, [prespawn.bundles_sorted_by_node_any_device])
        self.assertIs(module.get_bundles_sorted_by_node, original)

    def test_engine_setting(self):
        self.assertFalse(EngineSettings().prespawn)
        self.assertTrue(EngineSettings.from_dict({"prespawn": True}).prespawn)
        with self.assertRaises(ValueError):
            EngineSettings.from_dict({"prespawn": "yes"})


# --------------------------------------------------------------------------- deployment split

class Fut:
    def __init__(self, value=None, error=None, done=False):
        self.value, self.error, self.done = value, error, done


class FakeRay:
    killed = []

    @staticmethod
    def get(value, timeout=None):
        if isinstance(value, list):
            return [FakeRay.get(v) for v in value]
        if isinstance(value, Fut):
            if value.error:
                raise value.error
            return value.value
        return value

    @staticmethod
    def wait(refs, num_returns=1, timeout=None):
        done = [r for r in refs if r.done]
        return done, [r for r in refs if not r.done]

    @staticmethod
    def kill(actor, no_restart=True):
        FakeRay.killed.append(actor)


def _instance(iid, ips):
    return SimpleNamespace(instance_id=iid, host_ips=ips, hosts=[SimpleNamespace(node_ip=ip) for ip in ips],
                           tp_size=4 * len(ips), coordinator_ip=ips[0], to_dict=lambda: {"instance_id": iid})


class FakeGate:
    def __init__(self, parked=()):
        self.parked = {k: 1.0 for k in parked}
        self.released = []
        self.status = SimpleNamespace(remote=lambda: {"parked": dict(self.parked)})
        self.release = SimpleNamespace(remote=lambda keys: (self.released.extend(keys), {k: 5.0 for k in keys})[1])


INVENTORY = SimpleNamespace(to_dict=lambda: {"inventory": 1})


class DeploymentSplitTests(unittest.TestCase):
    def setUp(self):
        FakeRay.killed = []
        self.inst = _instance("large_b", ["10.0.0.3", "10.0.0.4"])
        self.config = deployment.DeploymentConfig(
            layout=SimpleNamespace(layout_id="Mb", to_dict=lambda: {}),
            engine_settings={"prespawn": True}, pg_ready_timeout_s=5.0, engine_init_timeout_s=5.0)
        self.registry = mock.Mock()
        self.gate = FakeGate()
        self.init_calls = []
        actor = mock.Mock()
        actor.initialize.remote = lambda **kw: (self.init_calls.append(kw), Fut(value={"init": True}))[1]
        self.actor = actor
        self.holds = []
        patches = [
            mock.patch.object(deployment, "ray", FakeRay),
            mock.patch.object(deployment, "_select_instances",
                              return_value=(None, [self.inst], {"layout_digest": "d"}, "Mb", "d")),
            mock.patch.object(deployment, "_find_engine_actor", return_value=None),
            mock.patch.object(deployment, "_prepare_backend", return_value=({"ready": True}, 0.1)),
            mock.patch.object(deployment, "_place", side_effect=self._place),
            mock.patch.object(deployment, "_spawn_engine_actors", side_effect=self._spawn),
            mock.patch.object(deployment, "get_or_create_registry", return_value=self.registry),
            mock.patch.object(deployment, "_rollback"),
            mock.patch.object(deployment, "_serve_and_register", return_value=[{"descriptor": {}}]),
            mock.patch("hetero.prespawn.create_gate", return_value=self.gate),
            mock.patch("ray.util.placement_group.placement_group", side_effect=self._hold_pg),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _place(self, instances, layout_id, config, tpu_per_host, created_pgs):
        self.assertEqual(tpu_per_host, 0.0)  # pre-spawn groups hold no chips
        created_pgs.append(deployment.pg_name(layout_id, "large_b"))
        return {"large_b": "pg"}, {"large_b": {"record": {"pg_id": "x"}}}, 0.01

    def _spawn(self, instances, layout_id, digest, config, handles, existing, created_actors):
        created_actors.append(deployment.engine_actor_name(layout_id, "large_b"))
        return {"large_b": self.actor}, {"large_b": False}, [{"env": 1}], 0.01

    def _hold_pg(self, bundles, strategy, name, lifetime):
        self.holds.append((bundles, strategy, name))
        return SimpleNamespace(ready=lambda: "ready")

    def test_prespawn_submits_gated_initialize_and_registers_nothing(self):
        prepared = deployment.prespawn_deployment(None, self.config, instance_ids=["large_b"], gate_timeout_s=99)
        self.assertEqual(self.init_calls[0]["prespawn_gate"]["name"], prepared.gate_name)
        self.assertEqual(self.init_calls[0]["prespawn_gate"]["timeout_s"], 99.0)
        self.assertEqual(self.registry.mock_calls, [])  # no registry (hence no gateway) before commit
        self.assertEqual(prepared.host_keys(), ["10.0.0.3", "10.0.0.4"])

    def test_prespawn_refuses_without_the_setting_or_with_live_actors(self):
        config = deployment.DeploymentConfig(layout=self.config.layout, engine_settings={})
        with self.assertRaisesRegex(ValueError, "prespawn=true"):
            deployment.prespawn_deployment(None, config)
        with mock.patch.object(deployment, "_find_engine_actor", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "fresh engine actors"):
                deployment.prespawn_deployment(None, self.config)

    def test_deploy_refuses_prespawn_settings(self):
        with self.assertRaisesRegex(ValueError, "prespawn_deployment"):
            deployment.deploy(None, self.config)

    def test_park_release_commit(self):
        prepared = deployment.prespawn_deployment(None, self.config, instance_ids=["large_b"])
        self.gate.parked = {"10.0.0.3": 1.0}
        with self.assertRaisesRegex(TimeoutError, "not parked"):
            deployment.await_parked(prepared, timeout_s=0.05, poll_s=0.01)
        self.gate.parked = {"10.0.0.3": 1.0, "10.0.0.4": 1.2}
        self.assertIn("all_parked_at", deployment.await_parked(prepared, timeout_s=1, poll_s=0.01))
        with mock.patch.object(deployment, "_check_partial_gateway") as gateway_check:
            with self.assertRaisesRegex(RuntimeError, "commit before release"):
                deployment.commit_prespawned(prepared)
            gateway_check.assert_not_called()
        self.assertTrue(prepared.aborted)

    def test_release_then_commit_happy_path(self):
        prepared = deployment.prespawn_deployment(INVENTORY, self.config, instance_ids=["large_b"])
        receipt = deployment.release_prepared(prepared)
        self.assertEqual(self.gate.released, ["10.0.0.3", "10.0.0.4"])
        self.assertEqual(self.holds, [([{"TPU": 4.0, "node:10.0.0.3": 0.001}, {"TPU": 4.0, "node:10.0.0.4": 0.001}],
                                       "STRICT_SPREAD", "omp-hetero--Mb--large_b-tpuhold")])
        self.assertEqual(receipt["tpu_hold_pgs"], ["omp-hetero--Mb--large_b-tpuhold"])
        with mock.patch.object(deployment, "_check_partial_gateway") as gateway_check:
            report = deployment.commit_prespawned(prepared)
        gateway_check.assert_called_once()  # the partial gateway check happens at commit, after the retarget
        self.assertTrue(prepared.committed)
        self.assertIn(self.gate, FakeRay.killed)
        self.assertIn("ready_at", report["prespawn"]["tpu_hold"]["large_b"])
        self.assertIn("tpu_hold_wait_s", report["stage_clocks"])
        self.assertEqual(deployment.abort_prespawned(prepared)["aborted"], False)  # committed: untouched

    def test_early_finish_or_init_error_fails_the_park_wait(self):
        prepared = deployment.prespawn_deployment(None, self.config, instance_ids=["large_b"])
        prepared.futures = {"large_b": Fut(error=RuntimeError("worker import failed"), done=True)}
        with self.assertRaisesRegex(RuntimeError, "worker import failed"):
            deployment.await_parked(prepared, timeout_s=1, poll_s=0.01)
        prepared.futures = {"large_b": Fut(value={}, done=True)}
        with self.assertRaisesRegex(RuntimeError, "before its hosts were released"):
            deployment.await_parked(prepared, timeout_s=1, poll_s=0.01)

    def test_failures_remove_only_own_resources(self):
        prepared = deployment.prespawn_deployment(INVENTORY, self.config, instance_ids=["large_b"])
        deployment.release_prepared(prepared)
        prepared.futures = {"large_b": Fut(error=RuntimeError("device init failed"))}
        with mock.patch.object(deployment, "_check_partial_gateway"):
            with self.assertRaisesRegex(RuntimeError, "device init failed"):
                deployment.commit_prespawned(prepared)
        rollback = deployment._rollback
        rollback.assert_called_once_with(["omp-hetero-engine--Mb--large_b"],
                                         ["omp-hetero--Mb--large_b", "omp-hetero--Mb--large_b-tpuhold"])
        self.assertIn(self.gate, FakeRay.killed)
        self.assertEqual(deployment.abort_prespawned(prepared)["reason"], "already aborted")
        rollback.assert_called_once()
        # A failure inside prespawn_deployment rolls back what it created so far.
        with mock.patch.object(deployment, "_spawn_engine_actors", side_effect=RuntimeError("actor")):
            with self.assertRaisesRegex(RuntimeError, "actor"):
                deployment.prespawn_deployment(None, self.config, instance_ids=["large_b"])
        self.assertEqual(rollback.call_args.args, ([], ["omp-hetero--Mb--large_b"]))


# --------------------------------------------------------------------------- switch ordering

def run(mode, **kwargs):
    tmp = tempfile.TemporaryDirectory()
    fabric, client, kw = dry.run_dry(mode, pathlib.Path(tmp.name), scale=SCALE, overlap=True, **kwargs)
    try:
        report = switch.run_switch(fabric, client, **kw)
        error = None
    except Exception as exc:  # noqa: BLE001
        report = __import__("json").loads((pathlib.Path(tmp.name) / kw["run_id"] / "switch_report.json").read_text())
        error = exc
    return tmp, fabric, client, report, error


class SwitchPrespawnTests(unittest.TestCase):
    def test_full_order_gate_and_segments(self):
        tmp, fabric, client, report, error = run("full")
        self.addCleanup(tmp.cleanup)
        self.assertIsNone(error)
        self.assertEqual(fabric.order(), ["prespawn", "await_parked", "drain", "gateway_down", "release",
                                          "commit", "gateway_up", "start_capture"])
        dry.check_order(report, fabric, client)  # incl. pre-spawn before T0 and no dispatch before the gate
        self.assertEqual([s["segment"] for s in report["segments"]],
                         ["controller_wake", "prespawn_wait", "drain_release", "gateway_down", "prespawn_release",
                          "device_handoff", "engine_init", "endpoint_register", "gateway_up", "gate_observe",
                          "capture_arm", "dispatch"])
        self.assertLess(report["segments"][1]["s"], 0.05)  # parked before T0: nothing left to wait for
        pre = report["prespawn"]
        self.assertTrue(pre["parked_before_T0"])
        self.assertEqual(set(pre["window"]) >= {"before", "during", "after"}, True)
        anatomy = report["timeline"]["prespawn"]
        self.assertTrue(anatomy["all_parked_before_release"] and anatomy["all_self_checks_clean"])
        self.assertLess(anatomy["driver_engine_init_started_s"]["large_a"], 0)  # engine start before T0

    def test_partial_order_and_kept_segments(self):
        tmp, fabric, client, report, error = run("partial")
        self.addCleanup(tmp.cleanup)
        self.assertIsNone(error)
        self.assertEqual(fabric.order(), ["prespawn", "await_parked", "drain", "release", "adopt", "retarget",
                                          "start_capture", "commit", "join"])
        dry.check_order(report, fabric, client)
        self.assertEqual([s["segment"] for s in report["segments"]][:4],
                         ["controller_wake", "prespawn_wait", "drain_release", "prespawn_release"])
        self.assertEqual(report["prespawn"]["instances"], ["large_b"])
        self.assertIn("release_s", report["endpoints"]["joiner_timeline"])

    def test_late_parking_is_waited_for_and_recorded(self):
        tmp, fabric, client, report, error = run("full", service_by_label={"phase1": 0.01})
        self.addCleanup(tmp.cleanup)
        self.assertIsNone(error)
        self.assertFalse(report["prespawn"]["parked_before_T0"])
        wait = next(s for s in report["segments"] if s["segment"] == "prespawn_wait")
        self.assertGreater(wait["s"], 0.05)
        self.assertTrue(report["timeline"]["segments_reconcile"])
        self.assertTrue(report["completed"])

    def test_prespawn_at_s_delays_the_spawn(self):
        tmp, fabric, client, report, error = run("full", prespawn_at_s=0.3)
        self.addCleanup(tmp.cleanup)
        self.assertIsNone(error)
        st = report["stamps"]
        self.assertGreaterEqual(st["prespawn_begin"] - st["phase1_begin"], 0.3 - 1e-3)

    def test_prespawn_failure_before_T0_cancels_phase1_and_cleans_up(self):
        tmp, fabric, client, report, error = run("full", fail_at="prespawn")
        self.addCleanup(tmp.cleanup)
        self.assertIsNotNone(error)
        self.assertEqual(report["failed_step"], "prespawn")
        self.assertIn("injected failure in prespawn", report["error"])
        self.assertIn("phase1", client.cancelled_labels)
        self.assertNotIn("controller_wake", report["stamps"])
        self.assertNotIn("drain", fabric.order())  # the source keeps serving; nothing is torn down
        self.assertTrue(fabric.aborted and report["prespawn_abort"]["aborted"])

    def test_park_failure_before_T0(self):
        tmp, fabric, client, report, error = run("partial", fail_at="park")
        self.addCleanup(tmp.cleanup)
        self.assertEqual(report["failed_step"], "prespawn")
        self.assertTrue(fabric.aborted)
        self.assertNotIn("drain", fabric.order())

    def test_commit_failure_aborts_the_uncommitted_prespawn(self):
        tmp, fabric, client, report, error = run("full", fail_at="commit")
        self.addCleanup(tmp.cleanup)
        self.assertIsNotNone(error)
        self.assertEqual(report["failed_step"], "deploy")
        self.assertIn("abort_prespawn", fabric.order())
        self.assertFalse(any(d["label"] == "target" for d in client.dispatches))

    def test_phase1_error_is_not_blamed_on_prespawn(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fabric, client, kw = dry.run_dry("full", pathlib.Path(tmp.name), scale=SCALE, overlap=True)
        original = client.dispatch

        async def failing(items, **kwargs):
            if kwargs.get("label") == "phase1":
                return [{"request_id": i["request_id"], "error": "boom", "client_finished_at_s": time.time()}
                        for i in items]
            return await original(items, **kwargs)
        client.dispatch = failing
        with self.assertRaisesRegex(RuntimeError, "phase 1 delivered"):
            switch.run_switch(fabric, client, **kw)
        self.assertTrue(fabric.aborted)
        self.assertNotIn("drain", fabric.order())

    def test_offline_recompute_matches_report(self):
        import csv
        import gzip
        import json

        tmp, fabric, client, report, error = run("full")
        self.addCleanup(tmp.cleanup)
        out = pathlib.Path(report["artifact_dir"])
        read = lambda name: json.load(gzip.open(out / name, "rt"))  # noqa: E731
        with gzip.open(out / "phase2_r1_iterations.csv.gz", "rt") as handle:
            rows = [(r["instance_id"], float(r["completed_at_s"]), int(float(r["emitted_output_tokens"] or 0)))
                    for r in csv.DictReader(handle)]
        again = switch.build_timeline(
            mode="full", T0=report["T0"], stamps=report["stamps"], records2=read("phase2_records.json.gz"),
            deploy_report=read("deploy_report.json.gz"), drain_report=read("drain_report.json.gz"),
            iter_rows=rows, plan=None, static_target_s=report["static_target_s"])
        self.assertEqual([s["s"] for s in again["segments"]], [s["s"] for s in report["segments"]])
        self.assertEqual(again["prespawn"], report["timeline"]["prespawn"])

    def test_mixed_prespawn_settings_are_refused(self):
        fabric = switch.RayFabric.__new__(switch.RayFabric)
        fabric.config = deployment.DeploymentConfig(
            layout=SimpleNamespace(layout_id="Mb"), engine_settings={},
            instance_engine_settings={"large_b": {"prespawn": True}})
        self.assertTrue(fabric.prespawn_requested(["large_b"]))
        self.assertFalse(fabric.prespawn_requested(["small_a"]))
        with self.assertRaisesRegex(ValueError, "must agree"):
            fabric.prespawn_requested(["large_b", "small_a"])


if __name__ == "__main__":
    unittest.main()
