"""Parked-target pre-read (hetero/preread.py): device table, exactness, failure and isolation.

Run with the CPU venv built per results/preread/venv-build.md:

    . results/preread/env.sh && /dev/shm/hetero-scratch/preread/venv/bin/python -m pytest tests/test_preread.py

The device-table test needs only jax. The model tests need flax + tpu_inference + vllm + torch;
without them they are skipped, so the Bf16Kv ``.venv`` suite still runs.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "results" / "switch" / "inventory-2026-10-12.json"
O3_IDENTITY = ROOT / "results" / "preread" / "o3-H-identity.json"
O3_MESH_IDS = [0, 1, 2, 3, 7, 6, 5, 9, 10, 11, 15, 14, 13, 12, 8, 4]
HAVE_JAX = importlib.util.find_spec("jax") is not None
HAVE_STACK = HAVE_JAX and all(importlib.util.find_spec(m) is not None
                              for m in ("flax", "torch", "vllm", "tpu_inference", "cloudpickle"))
CONTEXT = {"model_hash": "tiny", "tp_size": 16, "dtype": "bfloat16"}


def _h_payload(instance_id: str = "large_a"):
    """The instance payload hetero/engine_actor.py builds, for H resolved on the inventory."""
    from hetero.layout import builtin_layout, resolve_layout
    from hetero.slice_inventory import SliceInventory

    inventory = SliceInventory.from_dict(json.loads(INVENTORY.read_text()))
    instance = next(i for i in resolve_layout(inventory, builtin_layout("H")).to_dict()["instances"]
                    if i["instance_id"] == instance_id)
    chips = {h["node_ip"]: [list(c) for c in h["chip_coords"]] for h in instance["hosts"]}
    return {"instance_id": instance_id, "layout_id": "H", "tp_size": instance["tp_size"],
            "host_ips": list(instance["host_ips"]), "runtime_env": dict(instance["intended_runtime_env"]),
            "chip_coords_by_host": chips}


@unittest.skipUnless(HAVE_JAX, "jax is not installed")
class DeviceTableTests(unittest.TestCase):
    def test_h_table_matches_o3_mesh_and_rank_identities(self):
        from hetero.preread import device_table

        o3 = json.loads(O3_IDENTITY.read_text())
        for instance in ("large_a", "large_b"):
            table = device_table(_h_payload(instance))
            self.assertEqual([d["id"] for d in table["devices"]], O3_MESH_IDS)
            self.assertEqual(table["devices"][0]["platform"], "tpu")
            self.assertEqual(table["mesh_shape"], o3["parallel"]["mesh_shape"])
            self.assertEqual(table["mesh_axes"], o3["parallel"]["mesh_axis_names"])
            for worker in o3["workers"]:
                rank = worker["rank"]
                self.assertEqual(table["local_device_ids"][str(rank["process_index"])],
                                 sorted(rank["local_device_ids"]))
                positions = table["local_positions"][str(rank["process_index"])]
                self.assertTrue(all(table["devices"][k]["process_index"] == rank["process_index"]
                                    for k in positions))
        # jax.devices() order (process, id) equals the O3 selected_device_ids.
        table = device_table(_h_payload())
        by_process = sorted(table["devices"], key=lambda d: (d["process_index"], d["id"]))
        self.assertEqual([d["id"] for d in by_process], o3["worker_evidence"][0]["selected_device_ids"])

    def test_inconsistent_payload_is_refused(self):
        from hetero.preread import device_table

        payload = _h_payload()
        payload["runtime_env"]["TPU_TOPOLOGY"] = "2x8"
        with self.assertRaises(ValueError):
            device_table(payload)
        payload = _h_payload()
        payload["host_ips"] = list(reversed(payload["host_ips"]))
        with self.assertRaises(ValueError):
            device_table(payload)


class SettingsAndAnatomyTests(unittest.TestCase):
    def test_preread_settings_validation_and_payload(self):
        from hetero.engine_actor import EngineSettings

        ok = EngineSettings(weight_snapshot="read", prespawn=True, preread=True, preread_timeout_s=120)
        self.assertTrue(ok.preread)
        self.assertFalse(EngineSettings().preread)
        self.assertEqual(EngineSettings().preread_timeout_s, 900.0)
        for bad in ({"weight_snapshot": "read", "preread": True},
                    {"prespawn": True, "preread": True},
                    {"prespawn": True, "weight_snapshot": "write", "preread": True},
                    {"prespawn": True, "weight_snapshot": "read", "preread": 1},
                    {"prespawn": True, "weight_snapshot": "read", "preread": True, "preread_timeout_s": 0}):
            with self.assertRaises(ValueError, msg=bad):
                EngineSettings(**bad)
        self.assertEqual(EngineSettings.from_dict(ok.to_dict()), ok)

    def test_payload_carries_preread(self):
        from hetero.engine_actor import EngineSettings, HeteroEngine

        cls = HeteroEngine.__ray_metadata__.modified_class if hasattr(HeteroEngine, "__ray_metadata__") else HeteroEngine
        actor = object.__new__(cls)
        actor.settings = EngineSettings(weight_snapshot="read", prespawn=True, preread=True, preread_timeout_s=60)
        actor.instance_id, actor.layout_id, actor.layout_digest = "large_a", "H", "d"
        actor.instance = {"tp_size": 16, "host_ips": ["a"], "intended_runtime_env": {},
                          "hosts": [{"node_ip": "a", "chip_coords": [[0, 0, 0]]}]}
        actor._prespawn_gate = None
        payload = actor._hetero_payload()
        self.assertIs(payload["preread"], True)
        self.assertEqual(payload["preread_timeout_s"], 60.0)

    def test_switch_extra_and_prespawn_anatomy_preread_timeline(self):
        from hetero import switch

        self.assertIn("preread_wait", switch.EXTRA)
        t0 = 1000.0
        worker = {"executor_rank": 0, "prespawn_gate": {"key": "a", "parked_at": 990.0, "released_at": 1001.0,
                                                        "self_check_parked": {"clean": True},
                                                        "self_check_released": {"clean": True}},
                  "preread": {"started_at": 989.0, "ended_at": 999.5, "wait_started_at": 1003.0,
                              "wait_ended_at": 1003.1, "used": True, "error": None, "exit_code": 0,
                              "killed": False,
                              "report": {"wall_s": 10.5, "vm_hwm_kb": 1, "bytes": 2, "phases_s": {"load": 3},
                                         "isolation": {"platforms_cpu_only": True, "self_checks_clean": True,
                                                       "audit_events": []}}}}
        report = {"instances": [{"initialization": {"instance_id": "large_a", "engine_init_started_at": 980.0,
                                                    "worker_init_phases": [worker]}}]}
        anatomy = switch.prespawn_anatomy(report, t0)
        row = anatomy["workers"][0]["preread"]
        self.assertEqual(row["started_at_s"], -11.0)
        self.assertEqual(row["ended_at_s"], -0.5)
        self.assertTrue(row["done_before_T0"] and row["isolation_clean"] and row["used"])
        self.assertTrue(anatomy["preread_all_used"] and anatomy["preread_all_done_before_T0"])
        worker.pop("preread")
        self.assertNotIn("preread_all_used", switch.prespawn_anatomy(report, t0))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_group(phase: str, spec_path: Path, log_dir: Path, timeout: float = 600) -> None:
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    procs = []
    for rank in range(4):
        log = open(log_dir / f"{phase}-rank{rank}.log", "wb")
        procs.append((subprocess.Popen([sys.executable, str(ROOT / "tests" / "preread_ref_worker.py"), phase,
                                        str(rank), str(spec_path)], env=env, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT), log))
    deadline = time.time() + timeout
    try:
        for proc, log in procs:
            proc.wait(timeout=max(1.0, deadline - time.time()))
            log.close()
    finally:
        for proc, _ in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    for rank, (proc, _) in enumerate(procs):
        if proc.returncode != 0:
            tail = (log_dir / f"{phase}-rank{rank}.log").read_text()[-3000:]
            raise AssertionError(f"{phase} rank {rank} exited {proc.returncode}:\n{tail}")


@unittest.skipUnless(HAVE_STACK, "flax/torch/vllm/tpu_inference CPU stack is not installed")
class PrereadModelTests(unittest.TestCase):
    """Tiny real Gemma4ForConditionalGeneration checkpoint; 4 simulated ranks of a TP16 mesh."""

    @classmethod
    def setUpClass(cls):
        from hetero.preread import device_table, write_spec
        from tests.preread_support import build_tiny_checkpoint, build_vllm_config, tiny_payload

        cls.tmp = Path(tempfile.mkdtemp(prefix="preread-test-", dir=os.environ.get("TMPDIR")))
        cls.model = str(build_tiny_checkpoint(cls.tmp / "model"))
        cls.h_table = device_table(tiny_payload())
        # Built in this (parent-like) process: no JAX backend may be initialized by it.
        cls.vllm_config = build_vllm_config(cls.model, tp=16)
        cls.generation = uuid.uuid4().hex
        cls.write_spec = staticmethod(write_spec)

        # Phase 1: the normal loader on a 4-process x 4-CPU-device mesh with the H position layout.
        cls.spec = {"port": _free_port(), "model": cls.model, "root": str(cls.tmp / "groups"),
                    "positions_of": [d["process_index"] for d in cls.h_table["devices"]],
                    "context": CONTEXT, "generation": uuid.uuid4().hex,
                    "reference_root": str(cls.tmp / "snap" / "reference" / "H" / "large_a"),
                    "preread_root": str(cls.tmp / "snap" / "preread" / "H" / "large_a")}
        cls.spec_path = cls.tmp / "group-spec.json"
        cls.spec_path.write_text(json.dumps(cls.spec))
        (cls.tmp / "logs").mkdir()
        _run_group("normal", cls.spec_path, cls.tmp / "logs")
        cls.normal = [json.loads((cls.tmp / "groups" / "normal" / f"rank{r}" / "result.json").read_text())
                      for r in range(4)]
        devices = cls.normal[0]["devices"]
        cls.cpu_table = {"devices": devices, "mesh_shape": [1, 16], "mesh_axes": ["data", "model"],
                         "local_positions": {str(p): [k for k, d in enumerate(devices) if d["process_index"] == p]
                                             for p in range(4)}}

        # Phase 2: four pre-read children (one per rank), started as a parked worker starts them.
        cls.children = {}
        from hetero.preread import start_child
        for rank in range(4):
            spec = write_spec(cls.tmp / "spec" / f"rank{rank}", vllm_config=cls.vllm_config, rank=rank,
                              table=cls.cpu_table, out_dir=str(Path(cls.spec["preread_root"]) / f"rank{rank}"),
                              generation=cls.generation, context=CONTEXT)
            cls.children[rank] = start_child(spec, tp=16)
        cls.outcomes = {rank: child.wait(600) for rank, child in cls.children.items()}
        from jax._src import xla_bridge
        cls.parent_backends_after_children = sorted(xla_bridge._backends)

    @classmethod
    def tearDownClass(cls):
        for child in getattr(cls, "children", {}).values():
            child.kill()
        if os.environ.get("PREREAD_KEEP_TMP") != "1":
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_children_succeed_with_isolation_evidence(self):
        for rank, outcome in self.outcomes.items():
            self.assertTrue(outcome["ok"], (rank, outcome["error"], (outcome["report"] or {}).get("traceback")))
            report = outcome["report"]
            iso = report["isolation"]
            self.assertEqual(iso["jax_platforms"], ["cpu"])
            self.assertTrue(iso["platforms_cpu_only"])
            self.assertTrue(iso["self_checks_clean"])
            self.assertGreaterEqual(len(iso["self_checks"]), 2)  # start and exit (+ every 10 s)
            self.assertEqual(iso["audit_events"], [])
            for field in ("vm_hwm_kb", "cpu_user_s", "cpu_sys_s", "ru_majflt", "meminfo", "phases_s", "wall_s"):
                self.assertIn(field, report)
            for key in ("graph", "load", "convert", "write", "publish"):
                self.assertIsNotNone(report["phases_s"][key])
            for key in ("MemAvailable_kb", "Cached_kb", "Shmem_kb", "Dirty_kb", "Writeback_kb"):
                self.assertIn(key, report["meminfo"])
            self.assertIsNotNone(report["meminfo"]["shm_free_min_bytes"])
            self.assertEqual(report["env"]["JAX_PLATFORMS"], "cpu")

    def test_parent_has_no_jax_backend(self):
        self.assertEqual(self.parent_backends_after_children, [])

    def test_union_of_rank_snapshots_equals_normal_loader_snapshot(self):
        from hetero.weight_snapshot import file_fingerprint
        from scripts.preread_compare import compare

        root = self.tmp / "snap"
        normal = [file_fingerprint(p) for p in sorted((root / "reference").glob("*/*/rank*/manifest.json"))]
        preread = [file_fingerprint(p) for p in sorted((root / "preread").glob("*/*/rank*/manifest.json"))]
        self.assertEqual(len(normal), 4)
        self.assertEqual(len(preread), 4)
        result = compare(normal, preread)
        self.assertTrue(result["equal"], result["problems"][:20])
        self.assertGreater(result["shards_compared"], 0)
        # The CLI over saved files agrees.
        a, b = self.tmp / "a.json", self.tmp / "b.json"
        a.write_text(json.dumps(normal))
        b.write_text(json.dumps(preread))
        cli = subprocess.run([sys.executable, str(ROOT / "scripts" / "preread_compare.py"), str(a), str(b)],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(cli.returncode, 0, cli.stdout[-2000:] + cli.stderr[-2000:])
        # Every rank wrote only its own mesh positions.
        for record in preread:
            ids = {d["id"] for k, d in enumerate(self.cpu_table["devices"])
                   if k in self.cpu_table["local_positions"][str(record["process_rank"])]}
            self.assertEqual({s["device_id"] for s in record["shards"]}, ids)

    def test_restore_of_preread_equals_normal_state(self):
        _run_group("restore", self.spec_path, self.tmp / "logs")
        for rank in range(4):
            restored = json.loads((self.tmp / "groups" / "restore" / f"rank{rank}" / "result.json").read_text())
            self.assertEqual(restored["fingerprint"]["sha256"], self.normal[rank]["fingerprint"]["sha256"])
            self.assertEqual(restored["fingerprint"]["shards"], self.normal[rank]["fingerprint"]["shards"])
            self.assertIn("snapshot_device_placement", restored["restore_clocks"])

    def _child(self, name, *, seed=False, **extra):
        from hetero.preread import start_child
        out = self.tmp / name / "out"
        if seed:
            # A previously published, valid snapshot in the child's directory: must be withdrawn.
            shutil.copytree(Path(self.spec["preread_root"]) / "rank0", out)
            self.assertTrue((out / "manifest.json").is_file())
        spec = self.write_spec(self.tmp / name / "spec", vllm_config=self.vllm_config, rank=0,
                               table=self.cpu_table, out_dir=str(out), generation=uuid.uuid4().hex,
                               context=CONTEXT, extra=extra)
        return out, start_child(spec, tp=16)

    def test_failed_child_publishes_nothing_and_read_path_refuses(self):
        from hetero.preread import gate_outcome
        from hetero.weight_snapshot import SnapshotRefused, prepare

        out, child = self._child("failed", seed=True, fault_after_calls=5)
        outcome = child.wait(300)
        self.assertFalse(outcome["ok"])
        self.assertIn("injected pre-read fault", outcome["report"]["error"])
        self.assertNotEqual(outcome["exit_code"], 0)
        self.assertFalse((out / "manifest.json").exists())
        error = gate_outcome(outcome)
        self.assertIsNotNone(error)
        with self.assertRaises(SnapshotRefused):
            prepare(out, CONTEXT, None, {})

    def test_killed_child_publishes_nothing_and_read_path_refuses(self):
        from hetero.preread import gate_outcome

        started = time.time()
        out, child = self._child("killed", seed=True, pause_after_calls=[3, 300])

        def rewritten():
            return any(p.stat().st_mtime > started for p in out.glob("*.raw"))
        deadline = time.time() + 300
        while time.time() < deadline and not rewritten():
            time.sleep(0.02)
        self.assertTrue(rewritten(), "child never started writing")
        os.kill(child.process.pid, signal.SIGKILL)
        outcome = child.wait(60)
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["exit_code"], -signal.SIGKILL)
        self.assertFalse((out / "manifest.json").exists())
        self.assertIsNotNone(gate_outcome(outcome))

    def test_timeout_kills_and_reaps_child(self):
        out, child = self._child("timeout")
        outcome = child.wait(0.5)
        self.assertFalse(outcome["ok"])
        self.assertIn("timed out", outcome["error"])
        self.assertTrue(outcome["killed"])
        self.assertIsNotNone(child.process.returncode)
        self.assertFalse((out / "manifest.json").exists())

    # ------------------------------------------------------------- worker integration (tpu_worker)

    def _fake_worker(self, root: Path):
        """A HeteroTPUWorker shell with just what the pre-read start/wait and read path use."""
        from types import SimpleNamespace

        from hetero.native_backend import BACKEND_REVISION, RUNTIME_REVISION
        from hetero.tpu_worker import HeteroTPUWorker
        from tests.preread_support import tiny_payload

        payload = {**tiny_payload(), "weight_snapshot": "read", "weight_snapshot_root": str(root),
                   "weight_snapshot_generation": uuid.uuid4().hex, "preread": True, "preread_timeout_s": 300.0,
                   "backend_revision": BACKEND_REVISION, "runtime_revision": RUNTIME_REVISION}
        worker = object.__new__(HeteroTPUWorker)
        worker.vllm_config = self.vllm_config
        worker._hetero_payload = lambda: payload
        worker.omp_process_rank = 2
        worker.omp_snapshot_status = {"mode": "read", "restored": False, "checksums_requested": False}
        worker.model_runner = SimpleNamespace(model_config=SimpleNamespace(dtype="bfloat16"), mesh=None)
        return worker, payload

    def test_worker_starts_child_before_park_and_reads_its_snapshot(self):
        from hetero.weight_snapshot import file_fingerprint

        worker, payload = self._fake_worker(self.tmp / "worker-ok")
        worker._omp_start_preread(payload)
        record = worker._omp_preread
        self.assertNotIn("error", record)
        spec_dir = self.tmp / "worker-ok" / ".preread" / "H" / "large_a" / "rank2"
        self.assertTrue((spec_dir / "spec.json").is_file() and (spec_dir / "preread.log").is_file())
        self.assertIsNone(worker._omp_wait_preread())
        self.assertTrue(worker.omp_snapshot_status["preread"]["used"])
        manifest = worker._snapshot_location() / "manifest.json"
        fp = file_fingerprint(manifest)
        self.assertEqual(fp["array_meta"][0]["devices"][0]["platform"], "tpu")
        self.assertEqual({s["device_id"] for s in fp["shards"]}, {8, 9, 12, 13})  # rank 2 of the H table
        context = json.loads(manifest.read_text())["context"]
        self.assertEqual(context, worker._snapshot_context())

    def test_worker_read_path_refuses_and_falls_back_on_failed_child(self):
        from unittest import mock

        from tpu_inference.worker.tpu_worker import TPUWorker

        worker, payload = self._fake_worker(self.tmp / "worker-fail")
        payload["preread_timeout_s"] = 0.2  # the child cannot finish: timeout -> kill -> refuse
        worker._omp_start_preread(payload)
        with mock.patch.object(TPUWorker, "load_model", return_value="normal-loader") as normal:
            self.assertEqual(worker._load_snapshot_or_normal(), "normal-loader")
        normal.assert_called_once()
        status = worker.omp_snapshot_status
        self.assertFalse(status["restored"])
        self.assertIn("pre-read", status["fallback_reason"])
        self.assertFalse(status["preread"]["used"])
        self.assertTrue(status["preread"]["killed"])
        self.assertIn("preread_wait", worker._omp_phase_clocks)
        self.assertIsNotNone(worker._omp_preread_child.process.returncode)

    def test_worker_kill_on_abort_reaps_child(self):
        worker, payload = self._fake_worker(self.tmp / "worker-abort")
        worker._omp_start_preread(payload)
        worker._omp_kill_preread("release gate aborted")
        self.assertIsNotNone(worker._omp_preread_child.process.returncode)
        self.assertEqual(worker._omp_preread["killed_reason"], "release gate aborted")


if __name__ == "__main__":
    unittest.main()
