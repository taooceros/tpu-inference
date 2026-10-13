"""Published config-only regressions from WarmupSetFable commit 2a3dd37."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import jax

from hetero.compile_cache import configure_compile_cache, verify_compile_cache
from scripts import analyze_overlap as analysis


class CompileCacheConfig(unittest.TestCase):
    def setUp(self):
        options = ("jax_persistent_cache_min_compile_time_secs",
                   "jax_persistent_cache_min_entry_size_bytes", "jax_log_compiles",
                   "jax_explain_cache_misses")
        old = {key: getattr(jax.config, key) for key in options}
        self.addCleanup(lambda: [jax.config.update(key, value) for key, value in old.items()])

    def test_late_worker_env_changes_effective_config_across_threads(self):
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        jax.config.update("jax_log_compiles", False)
        env = {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
               "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
               "JAX_LOG_COMPILES": "1", "JAX_EXPLAIN_CACHE_MISSES": "true"}
        with mock.patch.dict(os.environ, env):
            configure_compile_cache(env)
        with ThreadPoolExecutor(max_workers=1) as pool:
            effective = pool.submit(lambda: (
                jax.config.jax_persistent_cache_min_compile_time_secs,
                jax.config.jax_persistent_cache_min_entry_size_bytes,
                jax.config.jax_log_compiles,
                jax.config.jax_explain_cache_misses)).result()
        self.assertEqual(effective, (0.0, 0, True, True))

    def test_backend_override_is_rejected_instead_of_silently_reapplied(self):
        expected = configure_compile_cache({"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"})
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        with self.assertRaisesRegex(RuntimeError, "differs from requested"):
            verify_compile_cache(expected)

    def test_unspecified_options_are_preserved(self):
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 2.5)
        configure_compile_cache({"JAX_LOG_COMPILES": "false"})
        self.assertEqual(jax.config.jax_persistent_cache_min_compile_time_secs, 2.5)
        self.assertFalse(jax.config.jax_log_compiles)


class CoverageMatchedAcceptance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patcher = mock.patch.object(analysis, "ROOT", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.path = root / "deploy.json"
        self.instances = []
        for name in ("small_a", "small_b", "small_c", "small_d"):
            workers = []
            for rank in (0, 1):
                workers.append({"executor_rank": rank, "compile_cache": {
                    "dir": "/matched-cache", "min_compile_time_secs": 0,
                    "min_entry_size_bytes": 0, "log_compiles": False, "explain_cache_misses": False},
                    "startup_profiles": {"compile_or_warm_up_model": {
                        "host": f"{name}-{rank}", "compilation_cache_dir": "/matched-cache",
                        "persistent_cache_min_compile_time_secs": 0,
                        "persistent_cache_min_entry_size_bytes": 0, "intervals": [], "executables": []}}})
            self.instances.append({"descriptor": {"instance_id": name}, "initialization": {
                "engine_settings": {"compilation_cache_dir": "/matched-cache", "worker_env": {
                    key: "0" for key in ("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                                         "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                                         "JAX_LOG_COMPILES", "JAX_EXPLAIN_CACHE_MISSES")}},
                "resolved_engine_config": {"max_model_len": 16384, "max_num_seqs": 512,
                                           "max_num_batched_tokens": 2048, "enable_prefix_caching": False},
                "worker_init_phases": workers}})

    def evaluate(self):
        self.path.write_text(json.dumps({"measurements": {"ok": True,
                                "deployment": {"instances": self.instances}}}))
        return analysis.worker_rows(self.path, "compile_or_warm_up_model")

    def test_every_worker_must_have_coverage_matched_threshold(self):
        self.assertEqual(self.evaluate()["qualification"], "controlled")
        worker = self.instances[-1]["initialization"]["worker_init_phases"][-1]
        worker["compile_cache"]["min_compile_time_secs"] = 1
        result = self.evaluate()
        self.assertEqual(result["qualification"], "diagnostic_only")
        self.assertEqual(result["workers"][-1]["compile_cache"]["min_compile_time_secs"], 1)

    def test_duplicate_rank_is_not_complete_evidence(self):
        self.instances[0]["initialization"]["worker_init_phases"][1]["executor_rank"] = 0
        self.assertEqual(self.evaluate()["qualification"], "diagnostic_only")

    def test_absent_effective_option_cannot_be_assumed_disabled(self):
        del self.instances[0]["initialization"]["worker_init_phases"][0]["compile_cache"]["log_compiles"]
        self.assertEqual(self.evaluate()["qualification"], "diagnostic_only")


if __name__ == '__main__':
    unittest.main()
