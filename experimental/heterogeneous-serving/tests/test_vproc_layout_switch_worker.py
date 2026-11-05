"""In-engine layout switch (H = TP16 <-> S-dp = DP2 x TP8) at the worker level.

* CPU block (4 processes x 4 CPU devices): the synthetic engine through ``HostKernel.switch_engine_layout``,
  H -> S-dp -> H -> S-dp: tokens and logical leaf checksums equal in every layout, ``built`` only on first
  visits, zero compiles on revisits, old arrays deleted, every compiled program over whole hosts.
* ``VllmEngine`` pool bookkeeping with stub runners (vllm, flax and tpu_inference are not installed in the
  test interpreters; what only the cluster can check is listed in the final report of the task).
* The DP mesh hook and ``mesh_device_ids`` on fake v5e 4x4 devices, config guards, env.
"""
import dataclasses
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from hetero.vproc import catalog, engine_vllm
from hetero.vproc.cpu_block import CPU_ID_STRIDE, BlockCallError, CpuBlock
from hetero.vproc.synthetic_engine import SyntheticEngine, expected_tokens
from hetero.vproc.types import ILLEGAL_HOST_SETS, EngineSpec

HERE = __name__
FACTORY = "hetero.vproc.host_kernel:HostKernel"
REQUEST = {"prompt": [[1, 2, 3], [4, 5], [9]], "steps": 6}
SEED = 7


# --------------------------------------------------------------------------------------------------
# per-process helpers (run inside the CpuBlock processes)

class CheckedEngine(SyntheticEngine):
    """Synthetic engine plus inspection methods for the tests (called through ``engine_call``)."""

    def checksums(self):
        from jax.sharding import NamedSharding, PartitionSpec as P
        from hetero.vproc.resident import _identity_to
        out = {}
        for leaf in self.leaf_specs():
            array = self._params[leaf.path]
            full = _identity_to(NamedSharding(array.sharding.mesh, P()))(array)
            logical = catalog.to_logical(np.asarray(full.addressable_data(0)), leaf, self.spec.tp_size)
            out[leaf.path] = catalog.logical_checksum(logical)
        return out

    def remember(self):
        self._old = list(self._params.values()) + list(self._coefs.values())
        return len(self._old)

    def old_deleted(self):
        return [a.is_deleted() for a in self._old]

    def layout_info(self):
        array = next(iter(self._params.values()))
        mesh = array.sharding.mesh
        names = lambda e: e if isinstance(e, tuple) else (e,)
        return {"axes": tuple(mesh.axis_names), "shape": tuple(mesh.devices.shape),
                "ids": tuple(int(d.id) for d in mesh.devices.flat),
                "data_in_pspec": any("data" in names(e) for a in self._params.values() for e in a.sharding.spec)}


_PROGRAMS = []


def record_programs():
    """Keep the device set of every program this process compiles."""
    from jax._src import compiler
    if getattr(compiler, "_vproc_recording", False):
        _PROGRAMS.clear()
        return
    original = compiler.compile_or_get_cached

    def recording(backend, computation, devices, *args, **kwargs):
        _PROGRAMS.append(tuple(sorted(int(d.id) for d in getattr(devices, "flat", devices))))
        return original(backend, computation, devices, *args, **kwargs)

    compiler.compile_or_get_cached, compiler._vproc_recording = recording, True
    _PROGRAMS.clear()


def recorded_programs():
    return sorted(set(_PROGRAMS))


# --- VllmEngine pool bookkeeping with stub runners ---------------------------------------------

def _config_for(spec):
    import types
    cfg = types.SimpleNamespace(
        name=f"cfg{spec.layout}",
        sharding_config=types.SimpleNamespace(model_dp_size=spec.dp_size, tp_size=spec.tp_size,
                                              total_devices=spec.chips, device_indexes=None))
    for alias in engine_vllm._CONFIG_ALIASES:
        setattr(cfg, alias, f"{cfg.name}.{alias}")
    return cfg


def pool_scenario():
    """What ``VllmEngine`` does around a layout switch, with stub runner/config and real arrays.
    Runs on all four ranks (SPMD)."""
    import contextlib
    import types
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from hetero.vproc import resident
    from hetero.vproc.resident import _COMPILES, _identity_to, _install_compile_listener, _pspec

    _install_compile_listener()
    devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
    leaves, key_of = [], {}
    for l in catalog.scaled():  # state is a dict keyed by the catalog path; LeafSpec.path is its keystr
        path = jax.tree_util.keystr((jax.tree_util.DictKey(l.path),))
        leaves.append(dataclasses.replace(l, path=path))
        key_of[path] = l.path
    config_stack = []

    @contextlib.contextmanager
    def set_current_vllm_config(config):
        config_stack.append(config)
        try:
            yield
        finally:
            config_stack.pop()

    class StubRunner:
        def __init__(self, config, devs, rank, first, last):
            self.vllm_config, self.devices, self.rank = config, devs, rank
            self.mesh = self._create_2d_mesh()
            self.requests, self.input_batch = {}, types.SimpleNamespace(num_reqs=0)
            self._pre_async_results = self.execute_model_state = None
            self.kv_caches, self.kv_cache_config = [], None
            self.state, self.state_leaves, self.model = None, (), object()
            self.model_fn = jax.jit(lambda ls: sum(jnp.sum(l.astype(jnp.float32)) for l in ls))

        def _create_2d_mesh(self):  # stands in for tpu-inference's: natural (data, model) order
            sc = self.vllm_config.sharding_config
            grid = np.array(self.devices[:sc.total_devices], dtype=object).reshape(sc.model_dp_size, sc.tp_size)
            return Mesh(grid, ("data", "model"))

        def _alloc(self):
            self.kv_caches = [jax.device_put(np.zeros((self.mesh.devices.shape[0] * 2, 4), np.float32),
                                             NamedSharding(self.mesh, P("data")))]

        def initialize_kv_cache(self, kv_cache_config):
            self.kv_cache_config = kv_cache_config
            self._alloc()

        def delete_kv_cache(self):
            for a in self.kv_caches:
                a.delete()
            self.kv_caches = []

        def reinitialize_kv_cache(self):
            if self.kv_cache_config is None:
                raise RuntimeError("no kv_cache_config")
            self._alloc()

    updates = []

    class StubEngine(engine_vllm.VllmEngine):
        @staticmethod
        def _derive_leaf_specs(runner, spec):
            return tuple(leaves)

        def _update_model(self, runner, state):
            updates.append(runner)

        def _attach_leaves(self, runner, leaf_arrays, rewrap):
            abstract = {key_of[l.path]: jax.ShapeDtypeStruct(l.shape, np.dtype(catalog._np_dtype(l.dtype)))
                        for l in leaves}
            wanted = {key_of[l.path]: NamedSharding(runner.mesh, _pspec(l)) for l in leaves}
            state, conformed = engine_vllm.conform_resident(abstract, wanted, leaf_arrays)
            runner.state = state
            runner.state_leaves = tuple(jax.tree_util.tree_leaves(state))
            return conformed

    def fake_config(spec, executor_class=None):
        return _config_for(spec)

    checked = []
    modules = {"vllm": types.ModuleType("vllm"), "vllm.config": types.ModuleType("vllm.config"),
               "tpu_inference": types.ModuleType("tpu_inference"),
               "tpu_inference.runner": types.ModuleType("tpu_inference.runner"),
               "tpu_inference.runner.tpu_runner": types.ModuleType("tpu_inference.runner.tpu_runner")}
    for name in ("tpu_inference.layers", "tpu_inference.layers.common", "tpu_inference.layers.common.sharding"):
        modules[name] = types.ModuleType(name)
    modules["tpu_inference.layers.common.sharding"].MESH_AXIS_NAMES_2D = ("data", "model")
    modules["vllm.config"].set_current_vllm_config = set_current_vllm_config
    modules["tpu_inference.runner.tpu_runner"].TPUModelRunner = StubRunner

    def spec_of(tp, dp):
        return EngineSpec("e", "vllm", (0, 1, 2, 3), tp, {}, dp_size=dp)

    def stored_sums(spec):
        out = {}
        for leaf in leaves:
            array = engine.worker.model_runner.state[key_of[leaf.path]]
            full = _identity_to(NamedSharding(array.sharding.mesh, P()))(array)
            out[leaf.path] = catalog.logical_checksum(
                catalog.to_logical(np.asarray(full.addressable_data(0)), leaf, spec.tp_size))
        return out

    want = {l.path: catalog.logical_checksum(catalog.synthetic_value(
        dataclasses.replace(l, path=key_of[l.path]), SEED)) for l in leaves}
    h, s = spec_of(16, 1), spec_of(8, 2)
    obs = {"reports": []}

    def model_value(runner):
        before = len(_COMPILES)
        value = float(np.asarray(runner.model_fn(runner.state_leaves).addressable_data(0)))
        return value, len(_COMPILES) - before

    patches = [mock.patch.dict(sys.modules, modules),
               mock.patch.object(engine_vllm, "_tp_ring_ids",
                                 lambda tp, devs, env=None: tuple(int(d.id) for d in list(devs)[:tp])),
               mock.patch.object(engine_vllm, "engine_config", fake_config),
               mock.patch.object(engine_vllm, "check_view_config", lambda c, sp, n: checked.append(sp.layout))]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        cfg_h = _config_for(h)
        runner = StubRunner(cfg_h, devices, jax.process_index(), True, True)
        runner.state = {}
        for leaf in leaves:
            logical = catalog.synthetic_value(dataclasses.replace(leaf, path=key_of[leaf.path]), SEED)
            stored = catalog.to_stored(logical, leaf, 16)
            runner.state[key_of[leaf.path]] = jax.make_array_from_callback(
                leaf.shape, NamedSharding(runner.mesh, _pspec(leaf)), lambda i, a=stored: a[i])
        runner.state_leaves = tuple(jax.tree_util.tree_leaves(runner.state))
        runner.initialize_kv_cache("kv-H")
        engine = StubEngine(h)
        engine.worker = types.SimpleNamespace(devices=devices, rank=jax.process_index(), model_runner=runner,
                                              vllm_config=cfg_h)
        engine._specs = tuple(leaves)
        engine._register(h, cfg_h, runner, tuple(leaves))
        runner_h = runner
        obs["value"], _ = model_value(runner)
        first_arrays = {k: v for k, v in runner.state.items()}

        def switch(target, kv_name=None):
            r = engine.switch_layout(target)
            active = engine.worker.model_runner
            if r["built"]:  # what the frontend does through the executor after a first visit
                active.initialize_kv_cache(kv_name)
            value, compiles = model_value(active)
            obs["reports"].append({"layout": target.layout, "built": r["built"], "compiles": r["compiles"],
                                   "segments": r["segments"], "reshard": r["reshard"], "hbm": r["hbm"],
                                   "conformed": r["conformed"], "model_fn_compiles": compiles,
                                   "value": value, "kv": active.kv_cache_config,
                                   "kv_alive": [not a.is_deleted() for a in active.kv_caches],
                                   "mesh_ids": tuple(int(d.id) for d in active.mesh.devices.flat),
                                   "mesh_shape": tuple(active.mesh.devices.shape),
                                   "sums_equal": stored_sums(target) == want,
                                   "worker_config": engine.worker.vllm_config.name,
                                   "alias": engine.worker.model_config, "config_stack": list(config_stack),
                                   "spec": engine.spec.layout})
            return r

        slot_h = engine._slots[(16, 1)]
        switch(s, "kv-S")
        runner_s = engine.worker.model_runner
        slot_s = engine._slots[(8, 2)]
        obs["h_parked"] = {"parked": slot_h.parked, "state_leaves": len(runner_h.state_leaves),
                           "kv": len(runner_h.kv_caches), "kv_saved": slot_h.kv_cache_config,
                           "state_abstract": all(isinstance(v, jax.ShapeDtypeStruct)
                                                 for v in runner_h.state.values()),
                           "old_deleted": all(a.is_deleted() for a in first_arrays.values())}
        obs["s_shardings"] = {
            "data_in_pspec": any("data" in (e if isinstance(e, tuple) else (e,))
                                 for a in runner_s.state.values() for e in a.sharding.spec),
            "meshes_equal": all(a.sharding.mesh == runner_s.mesh for a in runner_s.state.values())}
        s_arrays = dict(runner_s.state)
        switch(h)
        obs["s_parked"] = {"parked": slot_s.parked, "state_leaves": len(runner_s.state_leaves),
                           "kv": len(runner_s.kv_caches), "kv_saved": slot_s.kv_cache_config,
                           "old_deleted": all(a.is_deleted() for a in s_arrays.values())}
        obs["same_runner_h"] = engine.worker.model_runner is runner_h
        h_arrays = dict(runner_h.state)
        switch(s)
        obs["same_runner_s"] = engine.worker.model_runner is runner_s
        obs["h2_old_deleted"] = all(a.is_deleted() for a in h_arrays.values())
        obs["layouts"] = {k: (v["active"], v["visits"], v["kv_cache_config"]) for k, v in engine.layouts().items()}
        obs["updates"] = len(updates)
        obs["checked"] = list(checked)

        # a no-op switch, then refusals that must free nothing
        noop = engine.switch_layout(s)
        obs["noop"] = (noop["noop"], noop["built"], noop["compiles"])
        errors = {}
        runner_s.requests["r"] = object()
        try:
            engine.switch_layout(h)
        except RuntimeError as error:
            errors["busy"] = str(error)
        runner_s.requests.clear()
        errors["busy_state_intact"] = (not any(a.is_deleted() for a in runner_s.state.values())
                                       and all(not a.is_deleted() for a in runner_s.kv_caches))
        try:
            engine.switch_layout(dataclasses.replace(h, hosts=(0, 1, 2, 3), model={"x": 1}))
        except ValueError as error:
            errors["differs"] = str(error)
        slot_h.runner.kv_cache_config = None  # a layout entered but never given a KV config
        try:
            engine.switch_layout(h)
        except ValueError as error:
            errors["no_kv"] = str(error)
        errors["no_kv_state_intact"] = not any(a.is_deleted() for a in runner_s.state.values())
        slot_h.runner.kv_cache_config = "kv-H"
        with mock.patch.object(resident, "reshard_leaves", side_effect=RuntimeError("boom")):
            try:
                engine.switch_layout(h)
            except RuntimeError as error:
                errors["boom"] = str(error)
        try:
            engine._check_usable()
        except RuntimeError as error:
            errors["unusable"] = str(error)
        obs["errors"] = errors
    return obs


# --------------------------------------------------------------------------------------------------
# driver side

def synthetic_spec(layout, order):
    tp, dp = layout
    return EngineSpec("e", f"{HERE}:CheckedEngine", (0, 1, 2, 3), tp, {"seed": SEED, "device_order": order},
                      dp_size=dp)


class SyntheticLayoutSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock(factory=FACTORY)

    @classmethod
    def tearDownClass(cls):
        cls.block.close()

    def all(self, method, *args):
        return self.block.call([(r, method, args) for r in range(4)])

    def engine(self, method, *args):
        return self.all("engine_call", method, *args)

    def tearDown(self):
        try:
            self.all("stop_engine", False)
        except BlockCallError:
            pass

    def cycle(self, order):
        h, s = synthetic_spec((16, 1), order), synthetic_spec((8, 2), order)
        want_tokens = expected_tokens(h, REQUEST)
        want_sums = {l.path: catalog.logical_checksum(catalog.synthetic_value(l, SEED)) for l in catalog.scaled()}
        self.block.call([(r, f"{HERE}:record_programs", ()) for r in range(4)])
        self.all("start_engine", h)

        def check(spec):
            for tokens in self.engine("step", REQUEST):
                self.assertEqual(tokens["tokens"], want_tokens, spec.layout)
            for sums in self.engine("checksums"):
                self.assertEqual(sums, want_sums, spec.layout)
            for description in self.all("describe"):
                self.assertEqual(description.engine, spec)

        check(h)
        visits = []
        for target in (s, h, s, h):  # H -> S-dp -> H -> S-dp (-> H)
            self.engine("remember")
            reports = self.all("switch_engine_layout", target)
            visits.append(reports)
            check(target)
            deleted = self.engine("old_deleted")
            for flags in deleted:
                self.assertTrue(flags and all(flags), "every array of the previous layout is deleted")
            info = self.engine("layout_info")
            devices = [type("D", (), {"id": p * CPU_ID_STRIDE + i, "process_index": p})()
                       for p in range(4) for i in range(4)]
            expect_ids = SyntheticEngine.mesh_device_ids(target, devices)
            for i in info:
                self.assertEqual(i["ids"], expect_ids)
                if target.dp_size > 1:
                    self.assertEqual((i["axes"], i["shape"], i["data_in_pspec"]), (("data", "model"), (2, 8), False))
            for report in reports:
                self.assertEqual((report["rank"], report["engine_id"]), (reports.index(report), "e"))
        self.assertEqual([[r["built"] for r in reports] for reports in visits],
                         [[True] * 4, [False] * 4, [False] * 4, [False] * 4])
        self.assertTrue(all(r["compiles"] > 0 for r in visits[0]))
        # The first visit primes the way back (``prime`` segment: its compiles are charged to that visit),
        # so every revisit, S-dp -> H included, reports zero compiles.
        self.assertTrue(all(r["segments"]["prime"] > 0 for r in visits[0]))
        for reports in visits[1:]:
            self.assertEqual([r["compiles"] for r in reports], [0] * 4, "revisits must not compile")
        for reports in visits:
            for report in reports:
                self.assertEqual(set(report["segments"]), set(engine_vllm.SWITCH_SEGMENTS) | {"total"})
                self.assertGreater(report["reshard"]["moved"] + report["reshard"]["kept"], 0)
        programs = {p for per_rank in self.block.call([(r, f"{HERE}:recorded_programs", ()) for r in range(4)])
                    for p in per_rank}
        self.assertTrue(programs)
        for ids in programs:
            hosts = {i // CPU_ID_STRIDE for i in ids}
            whole = {p * CPU_ID_STRIDE + c for p in hosts for c in range(4)}
            self.assertEqual(set(ids), whole, f"program over part of hosts {sorted(hosts)}: {ids}")
            self.assertNotIn(frozenset(hosts), ILLEGAL_HOST_SETS)

    def test_natural(self):
        self.cycle("natural")

    def test_halves_interleaved(self):
        self.cycle("halves_interleaved")

    def test_errors_and_guards(self):
        h, s = synthetic_spec((16, 1), "natural"), synthetic_spec((8, 2), "natural")
        self.all("start_engine", h)
        with self.assertRaisesRegex(BlockCallError, "not callable through engine_call"):
            self.engine("switch_layout", s)
        other = dataclasses.replace(s, model={"seed": 1})
        with self.assertRaisesRegex(BlockCallError, "differs"):
            self.all("switch_engine_layout", other)
        reports = self.all("switch_engine_layout", h)  # same layout: nothing happens
        self.assertTrue(all(r["noop"] and not r["built"] and r["compiles"] == 0 for r in reports))
        self.all("switch_engine_layout", s)
        with self.assertRaisesRegex(BlockCallError, "dp_size"):  # S-dp leaves are not exportable
            self.all("stop_engine", True)


class PoolBookkeeping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock()
        try:
            cls.results = cls.block.call([(r, f"{HERE}:pool_scenario", ()) for r in range(4)], timeout_s=300)
        except BaseException:
            cls.block.close()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.block.close()

    def test_visits(self):
        for obs in self.results:
            first, back, again = obs["reports"]
            self.assertEqual([r["built"] for r in obs["reports"]], [True, False, False])
            self.assertEqual([r["layout"] for r in obs["reports"]], [(8, 2), (16, 1), (8, 2)])
            self.assertEqual([r["spec"] for r in obs["reports"]], [(8, 2), (16, 1), (8, 2)])
            self.assertTrue(all(r["sums_equal"] for r in obs["reports"]))
            self.assertTrue(all(abs(r["value"] - obs["value"]) < 1e-3 for r in obs["reports"]))
            self.assertEqual([back["compiles"], again["compiles"]], [0, 0])
            self.assertEqual([back["model_fn_compiles"], again["model_fn_compiles"]], [0, 0],
                             "rebound leaves must hit the executables of the layout")
            self.assertGreater(first["model_fn_compiles"], 0)
            self.assertEqual(obs["noop"], (True, False, 0))

    def test_runner_pool(self):
        for obs in self.results:
            first, back, again = obs["reports"]
            self.assertEqual(first["mesh_shape"], (2, 8))
            self.assertEqual(first["mesh_ids"], tuple(p * CPU_ID_STRIDE + i for p in range(4) for i in range(4)))
            self.assertEqual(first["worker_config"], "cfg(8, 2)")
            self.assertEqual(first["alias"], "cfg(8, 2).model_config")
            self.assertEqual(back["worker_config"], "cfg(16, 1)")
            self.assertEqual(back["alias"], "cfg(16, 1).model_config")
            self.assertTrue(obs["same_runner_h"] and obs["same_runner_s"])
            self.assertEqual(obs["checked"], [(8, 2)])  # the config of a layout is built and checked once
            self.assertEqual(obs["updates"], 2)  # two rebinds into parked runners
            self.assertEqual(set(obs["layouts"]), {(16, 1), (8, 2)})
            self.assertEqual(obs["layouts"][(8, 2)], (True, 2, "kv-S"))
            self.assertEqual(obs["layouts"][(16, 1)], (False, 2, "kv-H"))

    def test_parked_runner_holds_nothing(self):
        for obs in self.results:
            self.assertEqual(obs["h_parked"], {"parked": True, "state_leaves": 0, "kv": 0, "kv_saved": "kv-H",
                                               "state_abstract": True, "old_deleted": True})
            self.assertEqual(obs["s_parked"], {"parked": True, "state_leaves": 0, "kv": 0, "kv_saved": "kv-S",
                                               "old_deleted": True})
            self.assertTrue(obs["h2_old_deleted"])
            self.assertEqual(obs["s_shardings"], {"data_in_pspec": False, "meshes_equal": True})

    def test_kv_follows_the_active_layout(self):
        for obs in self.results:
            first, back, again = obs["reports"]
            self.assertEqual((first["kv"], back["kv"], again["kv"]), ("kv-S", "kv-H", "kv-S"))
            for report in obs["reports"]:
                self.assertTrue(report["kv_alive"] and all(report["kv_alive"]))

    def test_refusals_free_nothing_and_failure_poisons(self):
        for obs in self.results:
            errors = obs["errors"]
            self.assertIn("idle", errors["busy"])
            self.assertTrue(errors["busy_state_intact"])
            self.assertIn("differs", errors["differs"])
            self.assertIn("never got a KV cache config", errors["no_kv"])
            self.assertTrue(errors["no_kv_state_intact"])
            self.assertEqual(errors["boom"], "boom")
            self.assertIn("unusable", errors["unusable"])


# --------------------------------------------------------------------------------------------------
# mesh, hook, config on fake v5e 4x4 devices

class MeshHook(unittest.TestCase):
    def setUp(self):
        from tests.test_vproc_surface import EXPECTED, v5e_4x4
        self.EXPECTED, self.devices = EXPECTED, v5e_4x4()

    def runner_cls(self, spec):
        import types
        calls = []

        class Runner:
            def __init__(self, devices):
                self.devices = devices
                self.vllm_config = types.SimpleNamespace(sharding_config=types.SimpleNamespace(
                    model_dp_size=spec.dp_size, tp_size=spec.tp_size, device_indexes=None))
                self.mesh = self._create_2d_mesh()

            def _create_2d_mesh(self):  # make_optimized_mesh for a v5e TP-only engine (utils.py:244-324)
                import jax
                ordered = sorted(self.devices[:spec.chips], key=lambda d: d.coords)
                calls.append("original")
                return jax.make_mesh((spec.dp_size, spec.tp_size), ("data", "model"),
                                     (jax.sharding.AxisType.Auto,) * 2, devices=ordered)
        return Runner, calls

    def test_dp2_tp8_mesh_is_one_s_ring_per_data_group(self):
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        ring = self.EXPECTED
        want = ring[(0, 1)] + ring[(2, 3)]
        self.assertEqual(engine_vllm.VllmEngine.mesh_device_ids(spec, self.devices), want)
        self.assertEqual(engine_vllm.VllmEngine.mesh_device_ids(spec, list(reversed(self.devices))), want)
        Runner, calls = self.runner_cls(spec)
        original = Runner.__dict__["_create_2d_mesh"]
        with engine_vllm.layout_mesh_hook(spec, Runner, ("data", "model")):
            runner = Runner(self.devices)
        self.assertEqual(calls, [], "the hooked constructor must not call tpu-inference's mesh builder")
        self.assertEqual(tuple(int(d.id) for d in runner.mesh.devices.flat), want)
        self.assertEqual((runner.mesh.axis_names, runner.mesh.devices.shape), (("data", "model"), (2, 8)))
        self.assertIs(Runner.__dict__["_create_2d_mesh"], original)
        for g, hosts in enumerate(spec.data_groups()):  # row g only holds chips of its hosts
            self.assertEqual({d.process_index for d in runner.mesh.devices[g]}, set(hosts))
        # tpu-inference's own order differs (deviceless-dp-switch-libtpu0.0.43.json), hence the hook
        Runner(self.devices)
        self.assertEqual(calls, ["original"])

    def test_tp16_is_not_hooked(self):
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 16)
        Runner, calls = self.runner_cls(spec)
        original = Runner.__dict__["_create_2d_mesh"]
        with engine_vllm.layout_mesh_hook(spec, Runner, ("data", "model")) as hook:
            self.assertIsNone(hook)
            self.assertIs(Runner.__dict__["_create_2d_mesh"], original)
            runner = Runner(self.devices)
        self.assertEqual(calls, ["original"])
        ids = tuple(int(d.id) for d in runner.mesh.devices.flat)
        self.assertEqual(ids, engine_vllm.VllmEngine.mesh_device_ids(spec, self.devices))
        self.assertEqual(ids, self.EXPECTED[(0, 1, 2, 3)])

    def test_hook_restores_on_error_and_checks_the_runner_config(self):
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        Runner, _ = self.runner_cls(spec)
        original = Runner.__dict__["_create_2d_mesh"]
        wrong = dataclasses.replace(spec, tp_size=4)
        with self.assertRaises(engine_vllm.EngineGuardError):
            with engine_vllm.layout_mesh_hook(wrong, Runner, ("data", "model")):
                Runner(self.devices)
        self.assertIs(Runner.__dict__["_create_2d_mesh"], original)

    def test_dp_ids_errors(self):
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        with self.assertRaises(ValueError):
            engine_vllm.VllmEngine.mesh_device_ids(spec, self.devices[:12])
        with self.assertRaises(NotImplementedError):
            engine_vllm.tp_mesh_device_ids(8, self.devices, {"NEW_MODEL_DESIGN": "1"}, spec.data_groups())

    def test_engine_mesh(self):
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        mesh = engine_vllm.engine_mesh(spec, self.devices)
        self.assertEqual(tuple(int(d.id) for d in mesh.devices.flat),
                         engine_vllm.VllmEngine.mesh_device_ids(spec, self.devices))


class ConfigGuards(unittest.TestCase):
    def config(self, spec, **over):
        import types
        sharding = types.SimpleNamespace(total_devices=spec.chips, device_indexes=None, model_dp_size=spec.dp_size,
                                         total_dp_size=spec.dp_size, tp_size=spec.tp_size)
        parallel = types.SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1)
        cfg = types.SimpleNamespace(parallel_config=parallel, sharding_config=sharding, kv_transfer_config=None)
        for key, value in over.items():
            owner, _, name = key.partition("__")
            setattr(getattr(cfg, owner) if name else cfg, name or owner, value)
        return cfg

    def check(self, spec, multiprocess_dp=False, count=16, **over):
        import types
        envs = types.ModuleType("tpu_inference.envs")
        envs.TPU_MULTIPROCESS_DP, envs.TPU_MULTIHOST_BACKEND = multiprocess_dp, "ray"
        package = types.ModuleType("tpu_inference")
        package.envs = envs
        with mock.patch.dict(sys.modules, {"tpu_inference": package, "tpu_inference.envs": envs}):
            engine_vllm.check_view_config(self.config(spec, **over), spec, count)

    def test_accepts_tp_and_single_process_dp(self):
        self.check(EngineSpec("e", "vllm", (0, 1, 2, 3), 16))
        self.check(EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2))
        self.check(EngineSpec("e", "vllm", (0, 1), 8), count=8)

    def test_rejects(self):
        s = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        cases = [dict(multiprocess_dp=True), dict(parallel_config__data_parallel_size=2),
                 dict(parallel_config__pipeline_parallel_size=2), dict(sharding_config__device_indexes=[0, 1]),
                 dict(sharding_config__total_devices=8), dict(sharding_config__total_dp_size=4),
                 dict(kv_transfer_config=object()), dict(count=8)]
        for case in cases:
            with self.assertRaises(engine_vllm.EngineGuardError, msg=str(case)):
                self.check(s, **case)

    def test_preinit_sets_single_process_dp(self):
        env = {}
        engine_vllm.preinit_environment(env)
        self.assertEqual((env["TPU_MULTIPROCESS_DP"], env["TPU_MULTIHOST_BACKEND"]), ("0", "ray"))
        env = {"TPU_MULTIPROCESS_DP": "1"}  # an explicit value stays; check_view_config then refuses it
        engine_vllm.preinit_environment(env)
        self.assertEqual(env["TPU_MULTIPROCESS_DP"], "1")

    def test_engine_config_passes_dp(self):
        import types
        seen = {}

        class EngineArgs:
            def __init__(self, **kwargs):
                seen.update(kwargs)

            def create_engine_config(self):
                return types.SimpleNamespace(scheduler_config=types.SimpleNamespace(async_scheduling=True))

        arg_utils = types.ModuleType("vllm.engine.arg_utils")
        arg_utils.EngineArgs = EngineArgs
        modules = {"vllm": types.ModuleType("vllm"), "vllm.engine": types.ModuleType("vllm.engine"),
                   "vllm.engine.arg_utils": arg_utils}
        spec = EngineSpec("e", "vllm", (0, 1, 2, 3), 8, dp_size=2)
        with mock.patch.dict(sys.modules, modules):
            config = engine_vllm.engine_config(spec, executor_class="X")
        self.assertEqual((seen["tensor_parallel_size"], seen["data_parallel_size"],
                          seen["distributed_executor_backend"]), (8, 2, "X"))
        self.assertFalse(config.scheduler_config.async_scheduling)


class SyntheticMeshOrder(unittest.TestCase):
    def devices(self):
        return [type("D", (), {"id": p * 10 + i, "process_index": p})() for p in range(4) for i in range(4)]

    def test_dp_rows_follow_data_groups(self):
        spec = EngineSpec("e", "synthetic", (0, 1, 2, 3), 8, {"device_order": "natural"}, dp_size=2)
        ids = SyntheticEngine.mesh_device_ids(spec, self.devices())
        self.assertEqual(ids, (0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33))
        spec = dataclasses.replace(spec, model={"device_order": "halves_interleaved"})
        ids = SyntheticEngine.mesh_device_ids(spec, self.devices())
        self.assertEqual(ids[:8], (0, 10, 1, 11, 2, 12, 3, 13))

    def test_dp_must_split_hosts(self):
        with self.assertRaises(ValueError):
            EngineSpec("e", "synthetic", (0, 1, 2), 8, dp_size=2)


if __name__ == "__main__":
    unittest.main()
