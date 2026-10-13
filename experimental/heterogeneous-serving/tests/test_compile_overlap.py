"""Real CPU AOT/cache-hit smoke and exclusive interval accounting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hetero.compile_overlap import partition_intervals
from hetero.engine_actor import EngineSettings
from scripts.analyze_overlap import cost_ledger, routing_signature


class CompileOverlapTests(unittest.TestCase):
    def test_layout_bindings_change_without_hiding_routing_knob_changes(self):
        small = {"small_instances": ["small_a"], "large_instances": [], "decode_length_blind": True}
        large = {"small_instances": [], "large_instances": ["large_a"], "decode_length_blind": True}
        self.assertEqual(routing_signature(small), routing_signature(large))
        large["decode_length_blind"] = False
        self.assertNotEqual(routing_signature(small), routing_signature(large))

    def test_parallel_nested_wall_accounting(self):
        rows = [{'kind': 'aot_compile', 'start': 1, 'end': 8},
                {'kind': 'backend_compile_and_load', 'start': 2, 'end': 5},
                {'kind': 'backend_compile_and_load', 'start': 3, 'end': 6},
                {'kind': 'cache_load', 'start': 4, 'end': 5},
                {'kind': 'warmup_execute_and_sync', 'start': 7, 'end': 9}]
        result = partition_intervals(0, 10, rows)
        self.assertEqual(sum(result.values()), 10)
        self.assertEqual(result['backend_compile_and_load'], 3)
        self.assertEqual(result['cache_load'], 1)
        self.assertEqual(result['other'], 2)

    def test_charge_slowdown_and_provisioning_without_double_counting_late_join(self):
        source = {"makespan_s": 120, "last_finished_request": 120,
                  "during_run": {"result": {"start": 10, "end": 130, "process_cpu_s": 80}}}
        control = {"makespan_s": 100, "run_id": "undisturbed"}
        result = cost_ledger(50, source, control, {"charged_s": 200})
        self.assertEqual(result["economic_cost_s"], {"reused_provisioning": 70, "first_use": 270})
        self.assertEqual(result["late_compile_after_source_s"], 10)
        self.assertEqual(result["precompile_task_wall_s"], 120)
        self.assertEqual(result["precompile_process_cpu_s"], 80)

    def test_faster_noisy_source_pair_is_not_negative_cost_credit(self):
        source = {"makespan_s": 90, "last_finished_request": 90, "during_run": None}
        control = {"makespan_s": 100, "run_id": "undisturbed"}
        result = cost_ledger(50, source, control, {"charged_s": 200})
        self.assertEqual(result["source_serving_delta_s"], -10)
        self.assertEqual(result["economic_cost_s"], {"reused_provisioning": 50, "first_use": 250})
        self.assertIsNone(result["precompile_task_wall_s"])

    def test_disabled_settings(self):
        self.assertFalse(EngineSettings.from_dict({}).startup_profile)
        self.assertIsNone(EngineSettings.from_dict({}).compilation_cache_dir)
        self.assertIsNone(EngineSettings.from_dict({}).startup_compile_bundle)
        self.assertFalse(EngineSettings.from_dict({}).startup_device_trace)
        with self.assertRaisesRegex(ValueError, 'startup_profile'):
            EngineSettings.from_dict({'startup_profile': 'true'})
        from scripts.overlap_replay import replay
        self.assertEqual(replay("/nonexistent", "/nonexistent", "v5e:4x4"),
                         {"enabled": False, "executables": [], "device_launches": 0})

    def test_abstract_aot_cache_matches_real_call(self):
        script = '''import json, sys, time
import jax
import jax.numpy as jnp
from hetero.compile_overlap import StartupProfile
from pathlib import Path
from scripts.overlap_replay import capture, replay
root = Path(sys.argv[1])
if sys.argv[2] == "replay":
    print(json.dumps(replay(root/"bundle", str(root/"background"), "attached:cpu", enabled=True)))
    raise SystemExit(0)
jax.config.update("jax_compilation_cache_dir", str(root/("normal" if sys.argv[2] == "aot" else "background")))
jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
@jax.jit
def next_phase(x):
    return jnp.sin(x) + 2
p=StartupProfile(); start=time.time()
with p.hooks():
    if sys.argv[2]=='aot':
        with capture(root/"bundle"):
            next_phase.lower(jax.ShapeDtypeStruct((1024,), jnp.float32)).compile()
    else:
        result=next_phase(jnp.ones((1024,), jnp.float32))
        result.block_until_ready()
        assert abs(float(result[0]) - 2.8414709848) < 1e-5
print(json.dumps(p.report(start,time.time())))
'''
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}
            first = json.loads(subprocess.check_output(
                [sys.executable, '-c', script, directory, 'aot'], env=env, text=True))
            replayed = json.loads(subprocess.check_output(
                [sys.executable, "-c", script, directory, "replay"],
                env={**env, "XLA_FLAGS": "", "JAX_LOG_COMPILES": "1"}, text=True))
            self.assertTrue(replayed["all_keys_match"])
            self.assertTrue(replayed["all_compiled"])
            effective = replayed["effective_cache_config"]
            self.assertEqual(effective["jax_persistent_cache_min_compile_time_secs"], 0)
            self.assertEqual(effective["jax_persistent_cache_min_entry_size_bytes"], 0)
            self.assertFalse(effective["jax_log_compiles"])
            self.assertFalse(effective["jax_explain_cache_misses"])
            second = json.loads(subprocess.check_output(
                [sys.executable, '-c', script, directory, 'execute'], env=env, text=True))
            cold = [r for r in first['intervals'] if r.get('module') == 'jit_next_phase']
            warm = [r for r in second['intervals'] if r.get('module') == 'jit_next_phase']
            self.assertEqual([r['hit'] for r in cold], [False])
            self.assertEqual([r['hit'] for r in warm], [True])
            self.assertEqual(cold[0]['cache_key'], warm[0]['cache_key'])
            self.assertEqual(replayed["executables"][0]["actual_key"], warm[0]["cache_key"])
            self.assertGreater(first["exclusive_wall_s"]["trace"], 0)
            self.assertGreater(first["exclusive_wall_s"]["lower"], 0)
            for event in first["monitoring_events"]:
                if "duration_s" in event:
                    self.assertGreaterEqual(event["end"] - event["duration_s"], first["start"] - .001)
                    self.assertLessEqual(event["end"], first["end"])
            self.assertTrue(any(e['name'].endswith('backend_compile_duration')
                                for e in second['monitoring_events']))

    def test_device_trace_excludes_host_and_nested_op_lanes(self):
        import gzip
        from hetero.compile_overlap import summarize_device_trace
        events = [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:TPU:0"}},
            {"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "/device:TPU:1"}},
            {"ph": "M", "pid": 1, "tid": 3, "name": "thread_name", "args": {"name": "XLA Modules"}},
            {"ph": "M", "pid": 2, "tid": 3, "name": "thread_name", "args": {"name": "XLA Modules"}},
            {"ph": "M", "pid": 1, "tid": 4, "name": "thread_name", "args": {"name": "XLA Ops"}},
            {"ph": "X", "pid": 0, "tid": 1, "name": "phase", "ts": 0, "dur": 12},
            {"ph": "X", "pid": 0, "tid": 1, "name": "host_wait", "ts": 0, "dur": 1000},
            {"ph": "X", "pid": 1, "tid": 3, "name": "module_a", "ts": 0, "dur": 8},
            {"ph": "X", "pid": 1, "tid": 3, "name": "module_b", "ts": 3, "dur": 7},
            {"ph": "X", "pid": 1, "tid": 3, "name": "module_c", "ts": 11, "dur": 7},
            {"ph": "X", "pid": 2, "tid": 3, "name": "module_d", "ts": 1, "dur": 10},
            {"ph": "X", "pid": 1, "tid": 4, "name": "nested_op", "ts": 0, "dur": 1000},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.trace.json.gz"
            path.write_bytes(gzip.compress(json.dumps({"traceEvents": events}).encode()))
            row = summarize_device_trace(directory, "phase")["files"][0]
            self.assertAlmostEqual(row["device_module_execution_wall_s"], 12e-6)
            self.assertAlmostEqual(row["per_device_module_execution_s"]["/device:TPU:0"], 11e-6)
            self.assertAlmostEqual(row["per_device_module_execution_s"]["/device:TPU:1"], 10e-6)
            self.assertIsNone(summarize_device_trace(directory, "missing")["files"][0]["device_module_execution_wall_s"])


if __name__ == '__main__':
    unittest.main()
