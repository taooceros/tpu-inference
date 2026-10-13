"""HeteroTPUWorker restore tail on CPU: sub-clocks, fast-graph switch, KV-window split.

tpu_inference and ray are stubbed at module level; the restore path itself
(prepare -> vote -> nnx.eval_shape -> validate -> device_put -> nnx.update)
runs for real on eight host devices.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def _install_stubs():
    if "tpu_inference.worker.tpu_worker" in sys.modules:
        return
    try:
        import ray  # noqa: F401
    except ImportError:
        ray = types.ModuleType("ray")
        ray.util = SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        sys.modules["ray"] = ray
    pkg = types.ModuleType("tpu_inference")
    worker_pkg = types.ModuleType("tpu_inference.worker")
    worker_mod = types.ModuleType("tpu_inference.worker.tpu_worker")
    models_pkg = types.ModuleType("tpu_inference.models")
    common_pkg = types.ModuleType("tpu_inference.models.common")
    loader_mod = types.ModuleType("tpu_inference.models.common.model_loader")

    class TPUWorker:
        def load_model(self):
            runner = self.model_runner
            model = loader_mod._get_nnx_model(runner.model_class, runner.vllm_config,
                                              runner.rng_key, runner.mesh)
            runner.state = nnx.state(model)

        def initialize_from_config(self, kv_cache_config):
            manager = self.model_runner.compilation_manager
            manager._precompile_sampling()
            manager._precompile_gather_logprobs()
            manager._flush_compilations()
            self.model_runner.initialize_kv_cache(kv_cache_config, 0)

    worker_mod.TPUWorker = TPUWorker

    def normal_builder(model_class, vllm_config, rng, mesh, **_):
        # Mirrors model_loader._get_nnx_model: eval_shape of the constructor, and
        # records whether Tracer.sharding was the patched property while tracing.
        from jax._src import core
        loader_mod.saw_patched_property = core.Tracer.__dict__["sharding"] is not loader_mod.original_property
        with jax.set_mesh(mesh):
            return nnx.eval_shape(lambda: model_class(vllm_config, rng, mesh))

    from jax._src import core as _core
    loader_mod.original_property = _core.Tracer.__dict__["sharding"]
    loader_mod.saw_patched_property = None
    loader_mod._get_nnx_model = normal_builder
    for name, module in (("tpu_inference", pkg), ("tpu_inference.worker", worker_pkg),
                         ("tpu_inference.worker.tpu_worker", worker_mod),
                         ("tpu_inference.models", models_pkg),
                         ("tpu_inference.models.common", common_pkg),
                         ("tpu_inference.models.common.model_loader", loader_mod)):
        sys.modules[name] = module


def core_property():
    from jax._src import core
    return core.Tracer.__dict__["sharding"]


_install_stubs()

from hetero.tpu_worker import HeteroTPUWorker  # noqa: E402
from hetero.weight_snapshot import snapshot  # noqa: E402

uniform = nnx.initializers.uniform()


class Tiny(nnx.Module):
    """Two sharded kernels and a replicated scale, built like tpu-inference layers."""

    def __init__(self, vllm_config, rng_key, mesh):
        rngs = nnx.Rngs(rng_key)
        hidden = vllm_config.model_config.hidden
        self.up = nnx.Param(nnx.with_partitioning(uniform, (None, "model"))(
            rngs.params(), (hidden, 4 * hidden), jnp.bfloat16))
        self.down = nnx.Param(nnx.with_partitioning(uniform, ("model", None))(
            rngs.params(), (4 * hidden, hidden), jnp.bfloat16))
        self.scale = nnx.Param(nnx.with_partitioning(uniform, (None,))(
            rngs.params(), (hidden,), jnp.float32))


class RestoreTailWorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text(json.dumps({"hidden": 16}))
        self.snapshot_dir = root / "snap"
        devices = jax.devices("cpu")[:8]
        self.mesh = Mesh(np.array(devices).reshape(1, 8), ("data", "model"))
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hidden=16, model=str(self.model_dir), dtype="bfloat16",
                                         quantization=None),
            additional_config={})
        with jax.set_mesh(self.mesh):
            concrete = Tiny(self.vllm_config, jax.random.key(0), self.mesh)
        state = nnx.state(concrete)
        for path, variable in state.flat_state():
            sharding = variable.sharding
            if not isinstance(sharding, NamedSharding):
                sharding = NamedSharding(self.mesh, PartitionSpec(*sharding))
            variable.set_value(jax.device_put(variable.get_value(), sharding))
        nnx.update(concrete, state)
        self.reference = nnx.state(concrete)

    def tearDown(self):
        self.tmp.cleanup()

    def _worker(self, **settings):
        payload = {"instance_id": "i0", "layout_id": "S", "tp_size": 8, "host_ips": ["127.0.0.1"],
                   "chip_coords_by_host": {"127.0.0.1": [[0, 0, 0, 0]]}, "backend_revision": "b",
                   "runtime_revision": "r", "weight_snapshot": "read", "weight_snapshot_verify": True,
                   **settings}
        worker = object.__new__(HeteroTPUWorker)
        worker.rank = 0
        worker.omp_process_rank = 0
        worker.omp_node_ip = "127.0.0.1"
        worker.vllm_config = SimpleNamespace(additional_config={"omp_hetero": payload},
                                             model_config=self.vllm_config.model_config)
        worker.model_runner = SimpleNamespace(mesh=self.mesh, model_config=self.vllm_config.model_config,
                                              vllm_config=self.vllm_config, model_class=Tiny,
                                              rng_key=jax.random.key(0))
        worker._snapshot_location = lambda: self.snapshot_dir
        worker._verify_model_precision = lambda: {}  # Gemma-specific inspection, out of scope here
        snapshot(self.snapshot_dir, self.reference, worker._snapshot_context())
        return worker

    def _assert_restored(self, worker):
        restored = worker.model_runner.state
        for (path, before), (_, after) in zip(self.reference.flat_state(), restored.flat_state()):
            np.testing.assert_array_equal(np.asarray(before.get_value()), np.asarray(after.get_value()))
            self.assertEqual(before.get_value().sharding, after.get_value().sharding)

    def test_restore_records_graph_subclocks_in_order(self):
        worker = self._worker()
        worker.load_model()
        self.assertTrue(worker.omp_snapshot_status["restored"])
        self.assertFalse(worker.omp_snapshot_status["fast_graph"])
        self._assert_restored(worker)
        clocks = worker._omp_phase_clocks
        order = ["snapshot_manifest_verify", "snapshot_read_tmpfs", "snapshot_verify", "snapshot_agree",
                 "loader_entry", "graph_build", "graph_validate", "graph_agree",
                 "snapshot_device_placement", "graph_update"]
        for key in order:
            self.assertIn(key, clocks)
        starts = [clocks[key]["start"] for key in order]
        self.assertEqual(starts, sorted(starts))
        self.assertLessEqual(clocks["graph_update"]["end"], clocks["load_model"]["end"])
        self.assertGreaterEqual(clocks["graph_build"]["start"], clocks["loader_entry"]["end"])

    def test_fast_graph_restores_identically_and_restores_tracer_property(self):
        from jax._src import core
        before = core.Tracer.__dict__["sharding"]
        worker = self._worker(fast_graph=True)
        worker.load_model()
        self.assertTrue(worker.omp_snapshot_status["fast_graph"])
        self.assertTrue(worker.omp_snapshot_status["restored"])
        self._assert_restored(worker)
        self.assertIs(core.Tracer.__dict__["sharding"], before)

    def test_normal_loader_traces_under_patch_only_when_enabled(self):
        loader = sys.modules["tpu_inference.models.common.model_loader"]
        for enabled in (False, True):
            with self.subTest(fast_graph=enabled):
                loader.saw_patched_property = None
                original_eval_shape = nnx.eval_shape
                worker = self._worker(weight_snapshot="off", fast_graph=enabled)
                worker.load_model()
                self.assertEqual(loader.saw_patched_property, enabled)
                self.assertFalse(worker.omp_snapshot_status["restored"])
                self.assertIs(core_property(), loader.original_property)
                self.assertIs(nnx.eval_shape, original_eval_shape)
                clocks = worker._omp_phase_clocks
                self.assertLessEqual(clocks["load_model"]["start"], clocks["graph_build"]["start"])
                self.assertLessEqual(clocks["graph_build"]["end"], clocks["load_model"]["end"])
                self.assertEqual([p for p, _ in worker.model_runner.state.flat_state()],
                                 [p for p, _ in self.reference.flat_state()])

    def test_initialize_from_config_splits_compile_from_allocation(self):
        worker = self._worker()
        calls = []
        manager = SimpleNamespace(_precompile_sampling=lambda: calls.append("sampling"),
                                  _precompile_gather_logprobs=lambda: calls.append("logprobs"),
                                  _flush_compilations=lambda: calls.append("flush"))
        worker.model_runner.compilation_manager = manager
        worker.model_runner.initialize_kv_cache = lambda cfg, order: calls.append(("kv", cfg, order))
        worker.initialize_from_config("cfg")
        self.assertEqual(calls, ["sampling", "logprobs", "flush", ("kv", "cfg", 0)])
        clocks = worker._omp_phase_clocks
        for key in ("precompile_sampling", "precompile_gather_logprobs", "precompile_flush",
                    "kv_cache_allocate", "initialize_from_config"):
            self.assertIn(key, clocks)
        self.assertLessEqual(clocks["initialize_from_config"]["start"], clocks["precompile_sampling"]["start"])
        self.assertLessEqual(clocks["kv_cache_allocate"]["end"], clocks["initialize_from_config"]["end"])
        # Wrappers are removed again: later calls are not timed twice.
        self.assertIs(worker.model_runner.compilation_manager._flush_compilations, manager._flush_compilations)



class RestoreTailSettingsTests(unittest.TestCase):
    def test_defaults_leave_baseline_unchanged(self):
        from hetero.engine_actor import EngineSettings
        settings = EngineSettings.from_dict({})
        self.assertFalse(settings.fast_graph)

    def test_fast_graph_round_trips_and_rejects_non_bool(self):
        from hetero.engine_actor import EngineSettings
        self.assertTrue(EngineSettings.from_dict({"fast_graph": True}).fast_graph)
        with self.assertRaisesRegex(ValueError, "fast_graph"):
            EngineSettings.from_dict({"fast_graph": "yes"})


if __name__ == "__main__":
    unittest.main()
