"""C2 conformance: the virtual JAX view on the CPU block (hetero/vproc/view.py).

Run: ``python -m unittest tests.test_vproc_view`` (jax 0.10.2 and 0.11.2).
"""
import importlib
import os
import sys
import tempfile
import textwrap
import time
import types
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc.cpu_block import CPU_ID_STRIDE, CpuBlock
from hetero.vproc.types import ViewSpec

HERE = __name__
PUBLIC = ("devices", "device_count", "local_devices", "process_index", "process_count",
          "process_indices")


def _view(hosts):
    return ViewSpec("e" + "".join(map(str, hosts)), tuple(hosts))


def _ids(host, n=4):
    return [host * CPU_ID_STRIDE + i for i in range(n)]


# ---------------------------------------------------------------- per-process programs
# (module-level so spawned host processes can resolve them by "module:function")

def _engine_body(cache_dir):
    """Engine-shaped: jax.devices -> Mesh("model") -> TP matmul + psum, process_allgather vote,
    broadcast_one_to_all, sync_global_devices, a jax.devices("cpu")[0] use, persistent cache write."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    writes = []
    if cache_dir:
        from jax._src import compilation_cache as cc
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        cc.reset_cache()
        real_put = cc.put_executable_and_time

        def spy(key, *args, **kwargs):
            writes.append(key)
            return real_put(key, *args, **kwargs)

        cc.put_executable_and_time = spy
    try:
        rank, count = jax.process_index(), jax.process_count()
        devs = jax.devices()
        procs = sorted({d.process_index for d in devs})

        def norm(d):
            return (procs.index(d.process_index), d.id % CPU_ID_STRIDE)

        mesh = Mesh(np.array(devs), ("model",))
        n = len(devs)
        k = 4 * n
        x_host = np.arange(2 * k, dtype=np.float32).reshape(2, k) % 5
        w_host = np.arange(k * 8, dtype=np.float32).reshape(k, 8) % 7
        x = jax.make_array_from_callback(x_host.shape, NamedSharding(mesh, P(None, "model")),
                                         lambda i: x_host[i])
        w = jax.make_array_from_callback(w_host.shape, NamedSharding(mesh, P("model", None)),
                                         lambda i: w_host[i])
        step = jax.jit(jax.shard_map(lambda a, b: jax.lax.psum(a @ b, "model"), mesh=mesh,
                                     in_specs=(P(None, "model"), P("model", None)), out_specs=P()))
        y = np.asarray(step(x, w).addressable_data(0))
        np.testing.assert_array_equal(y, x_host @ w_host)

        vote = mh.process_allgather(np.array([rank + 1, rank * 2], dtype=np.int32))
        scalar_vote = mh.process_allgather(np.int32(rank))
        bcast = mh.broadcast_one_to_all(np.array([rank + 100, 7], dtype=np.int32))
        mh.sync_global_devices("conformance")
        mh.assert_equal(np.int32(5))

        cpu0 = jax.devices("cpu")[0]
        cpu_local = cpu0.id in {d.id for d in jax.local_devices()}
        cpu_val = None
        if cpu_local:
            cpu_val = float(np.asarray(jax.device_put(np.arange(4, dtype=np.float32), cpu0).sum()))
        return {
            "rank": rank, "count": count, "indices": list(jax.process_indices()),
            "device_count": jax.device_count(), "local_device_count": jax.local_device_count(),
            "devices": [norm(d) for d in devs],
            "local": [norm(d) for d in jax.local_devices()],
            "local_of": [[norm(d) for d in jax.local_devices(process_index=r)] for r in range(count)],
            "mesh": (dict(mesh.shape), tuple(mesh.axis_names), [norm(d) for d in mesh.devices.flat]),
            "y": y.tolist(), "vote": vote.tolist(), "scalar_vote": scalar_vote.tolist(),
            "bcast": bcast.tolist(), "cpu_count": len(jax.devices("cpu")),
            "writes": len(writes), "write_keys": sorted(writes),
            "raw_devices": [d.id for d in devs],
            "cpu_local": cpu_local, "cpu_val": cpu_val,
        }
    finally:
        if cache_dir:
            cc.put_executable_and_time = real_put
            jax.config.update("jax_compilation_cache_dir", None)
            cc.reset_cache()


def engine_program(cache_dir, hosts):
    """The engine-shaped program, standalone (hosts=None) or under a view on ``hosts``."""
    from hetero.vproc import view as vw
    if hosts is None:
        return _engine_body(cache_dir)
    with vw.active(_view(hosts)):
        return _engine_body(cache_dir)


def view_semantics(hosts):
    import jax
    import numpy as np
    from jax._src import distributed, xla_bridge as xb
    from jax.sharding import Mesh
    from hetero.vproc import view as vw

    def snapshot():
        current = vw.current()
        return {"index": jax.process_index(), "count": jax.process_count(),
                "indices": list(jax.process_indices()),
                "devices": [d.id for d in jax.devices()], "device_count": jax.device_count(),
                "cpu": [d.id for d in jax.devices("cpu")],
                "local": [d.id for d in jax.local_devices()],
                "local_device_count": jax.local_device_count(),
                "current": None if current is None else current.engine_id}

    out = {"before": snapshot()}
    with vw.active(_view(hosts)):
        out["active"] = snapshot()
        out["local_of"] = {r: [d.id for d in jax.local_devices(process_index=r)]
                           for r in range(len(hosts))}
        try:
            jax.local_devices(process_index=len(hosts))
            out["bad_rank"] = "no error"
        except ValueError:
            out["bad_rank"] = "ValueError"
        mesh = Mesh(np.array(jax.devices()), ("model",))
        out["internals"] = {
            "xb_index": xb.process_index(), "xb_count": xb.process_count(),
            "xb_devices": len(xb.devices()),
            "process_id": distributed.global_state.process_id,
            "num_processes": distributed.global_state.num_processes,
            "local_mesh": sorted(d.id for d in mesh.local_mesh.devices.flat),
            "xb_is_patched": xb.process_index is jax.process_index}
        with vw.physical():
            out["suspended"] = snapshot()
            with vw.physical():
                out["nested"] = snapshot()
            out["after_nested"] = snapshot()
        out["resumed"] = snapshot()
    out["after"] = snapshot()
    return out


def _owned_originals():
    """(module name, attr) -> object for every jax.* attribute that is one of the patched originals."""
    import jax
    from jax._src import compiler
    targets = {id(getattr(jax, n)): n for n in PUBLIC}
    targets[id(compiler._cache_write)] = "_cache_write"
    found = {}
    for name, module in list(sys.modules.items()):
        if isinstance(module, types.ModuleType) and (name == "jax" or name.startswith("jax.")):
            for attr, value in list(vars(module).items()):
                if id(value) in targets:
                    found[(name, attr)] = value
    return found


def restore_identity(hosts):
    import jax
    from jax._src import xla_bridge as xb
    from hetero.vproc import view as vw
    fake = types.ModuleType("jax.fake_reexport")
    fake.devices = jax.devices
    fake.process_index = jax.process_index
    sys.modules["jax.fake_reexport"] = fake
    try:
        before = _owned_originals()
        meta_before = list(sys.meta_path)
        physical_fns = {n: getattr(xb, n) for n in PUBLIC}
        with vw.active(_view(hosts)):
            during = {key: getattr(sys.modules[key[0]], key[1]) for key in before}
            changed = sorted(key for key in before if during[key] is before[key])
            reexport = (fake.devices is jax.devices, fake.process_index is jax.process_index)
            internals_physical = all(getattr(xb, n) is f for n, f in physical_fns.items())
        after = {key: getattr(sys.modules[key[0]], key[1]) for key in before}
        return {"unchanged_while_active": changed, "reexport_rebound": reexport,
                "internals_physical": internals_physical,
                "restored": all(after[key] is before[key] for key in before),
                "meta_path_restored": list(sys.meta_path) == meta_before,
                "n_patched_names": len(before), "current_after": vw.current()}
    finally:
        del sys.modules["jax.fake_reexport"]


def error_cases(rank):
    import jax
    from hetero.vproc import view as vw
    from hetero.vproc.types import ViewSpec
    out = {}
    originals = {n: getattr(jax, n) for n in PUBLIC}

    def untouched():
        return all(getattr(jax, n) is f for n, f in originals.items()) and vw.current() is None

    for label, view in (("foreign", ViewSpec("x", tuple(h for h in (0, 1, 2, 3) if h != rank)[:2])),
                        ("out_of_runtime", ViewSpec("y", (rank, 9)))):
        try:
            vw.activate(view)
            out[label] = "no error"
            vw.deactivate()
        except ValueError:
            out[label] = "ValueError"
        out[label + "_untouched"] = untouched()
    first = ViewSpec("a", (0, 1, 2, 3))
    vw.activate(first)
    try:
        try:
            vw.activate(ViewSpec("b", (0, 1, 2, 3)))
            out["double"] = "no error"
        except RuntimeError:
            out["double"] = "RuntimeError"
        out["still_first"] = vw.current().engine_id == "a" and jax.process_index() == rank
    finally:
        vw.deactivate()
    out["restored"] = untouched()
    try:
        vw.deactivate()
        out["double_deactivate"] = "no error"
    except RuntimeError:
        out["double_deactivate"] = "RuntimeError"
    return out


UTILS_SRC = '''
from jax._src import xla_bridge as xb

def make_optimized_mesh(devices=None):
    # mirrors tpu_inference/utils.py:249
    if devices is None:
        devices = xb.devices()
    return list(devices)
'''

# Verbatim from tpu_inference/distributed/utils.py:100-150 (without the docstring).
DIST_SRC = '''
import logging
logger = logging.getLogger("fake")

def get_device_topology_order_id(local_devices, global_devices) -> int:
    if not local_devices:
        raise ValueError("local_devices cannot be empty")
    if not global_devices:
        raise ValueError("global_devices cannot be empty")

    if not all(hasattr(d, "coords") for d in local_devices):
        logger.error(
            f"Expect TPU device but got {[type(d) for d in local_devices]}")

    local_anchor = min(d.coords for d in local_devices)

    process_anchors = {}
    for d in global_devices:
        pid = d.process_index
        if pid not in process_anchors or d.coords < process_anchors[pid]:
            process_anchors[pid] = d.coords

    sorted_anchors = sorted(process_anchors.values())

    try:
        return sorted_anchors.index(local_anchor)
    except ValueError:
        raise ValueError(
            f"Local devices: {local_devices} do not exist in the global device: {global_devices} list."
        )
'''

WORKER_SRC = '''
from tpu_inference.distributed.utils import get_device_topology_order_id
from tpu_inference import utils
'''


def _fake_topology_devices():
    """Physical devices dressed with host-grid coords (host k at (k % 2, k // 2))."""
    from jax._src import xla_bridge as xb
    return [types.SimpleNamespace(process_index=d.process_index, id=d.id,
                                  coords=(d.process_index % 2, d.process_index // 2, d.id % CPU_ID_STRIDE))
            for d in xb.devices()]


def _tpu_checks(hosts, utils, dist, worker_binding):
    import jax
    from jax._src import xla_bridge as xb
    from hetero.vproc import view as vw
    allfake = _fake_topology_devices()
    me = xb.process_index()
    local = [d for d in allfake if d.process_index == me]
    out = {"physical_rank": dist.get_device_topology_order_id(local, allfake)}
    with vw.active(_view(hosts)):
        out["xb_devices"] = [d.id for d in utils.make_optimized_mesh()]
        out["xb_forwarded"] = (utils.xb.get_backend is xb.get_backend
                               and utils.xb.process_index is xb.process_index)
        out["virtual_rank"] = dist.get_device_topology_order_id(local, allfake)
        out["worker_binding_rank"] = worker_binding(local, allfake)
        out["virtual_rank_virtual_devs"] = dist.get_device_topology_order_id(
            local, [d for d in allfake if d.process_index in hosts])
        with vw.physical():
            out["physical_xb_devices"] = len(utils.make_optimized_mesh())
            out["physical_rank_inside"] = dist.get_device_topology_order_id(local, allfake)
    out["after_xb_devices"] = len(utils.make_optimized_mesh())
    return out


def fake_tpu_preloaded(hosts):
    """Fake tpu_inference modules already in sys.modules when the view activates."""
    import jax
    from jax._src import xla_bridge as xb
    from hetero.vproc import view as vw
    names = ["tpu_inference", "tpu_inference.utils", "tpu_inference.distributed",
             "tpu_inference.distributed.utils", "tpu_inference.worker"]
    assert not any(n in sys.modules for n in names)
    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        exec(UTILS_SRC, sys.modules["tpu_inference.utils"].__dict__)
        exec(DIST_SRC, sys.modules["tpu_inference.distributed.utils"].__dict__)
        utils, dist = sys.modules["tpu_inference.utils"], sys.modules["tpu_inference.distributed.utils"]
        worker = sys.modules["tpu_inference.worker"]
        worker.get_device_topology_order_id = dist.get_device_topology_order_id  # import-time binding
        original_fn = dist.get_device_topology_order_id
        out = _tpu_checks(hosts, utils, dist, lambda *a: worker.get_device_topology_order_id(*a))
        out["restored"] = (utils.xb is xb and dist.get_device_topology_order_id is original_fn
                           and worker.get_device_topology_order_id is original_fn)
        return out
    finally:
        for name in names:
            sys.modules.pop(name, None)


def fake_tpu_late(hosts, root):
    """Fake tpu_inference first imported while the view is active (import hook)."""
    from jax._src import xla_bridge as xb
    from hetero.vproc import view as vw
    names = [n for n in sys.modules if n == "tpu_inference" or n.startswith("tpu_inference.")]
    assert not names
    sys.path.insert(0, root)
    importlib.invalidate_caches()
    try:
        vw.activate(_view(hosts))
        try:
            utils = importlib.import_module("tpu_inference.utils")
            dist = importlib.import_module("tpu_inference.distributed.utils")
            worker = importlib.import_module("tpu_inference.worker")
            patched = (hasattr(dist.get_device_topology_order_id, "__vproc_original__"),
                       utils.xb is not xb)
            late_devices = len(utils.make_optimized_mesh())
        finally:
            vw.deactivate()
        local = [d for d in _fake_topology_devices() if d.process_index == xb.process_index()]
        return {"patched": patched, "late_devices": late_devices,
                "xb_restored": utils.xb is xb,
                "fn_restored": not hasattr(dist.get_device_topology_order_id, "__vproc_original__"),
                "worker_wrapper_passthrough": worker.get_device_topology_order_id(
                    local, _fake_topology_devices()) == dist.get_device_topology_order_id(
                    local, _fake_topology_devices())}
    finally:
        sys.path.remove(root)
        for name in [n for n in sys.modules if n == "tpu_inference" or n.startswith("tpu_inference.")]:
            del sys.modules[name]


def view_collectives_quick(hosts, barrier_name):
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from hetero.vproc import view as vw
    with vw.active(_view(hosts)):
        import jax
        vote = mh.process_allgather(np.int32(jax.process_index())).tolist()
        mh.sync_global_devices("quick")
        vw.barrier(barrier_name)
        return time.time(), vote


def view_collectives_late(hosts, sleep_s):
    import numpy as np
    from jax.experimental import multihost_utils as mh
    from hetero.vproc import view as vw
    with vw.active(_view(hosts)):
        import jax
        time.sleep(sleep_s)
        woke = time.time()
        vote = mh.process_allgather(np.int32(jax.process_index())).tolist()
        mh.sync_global_devices("late")
        return woke, time.time(), vote


def coordination(hosts, token):
    import jax
    from jax._src import distributed
    from hetero.vproc import view as vw
    view = _view(hosts)
    client = distributed.global_state.client
    with vw.active(view):
        rank = jax.process_index()
        vw.kv_set(f"{token}/from{rank}", f"{view.engine_id}:{rank}")
        vw.barrier(f"{token}-b")
        peers = [vw.kv_get(f"{token}/from{r}", 20_000) for r in range(jax.process_count())]
    physical = client.blocking_key_value_get(f"{view.namespace}/{token}/from{rank}", 20_000)
    return {"peers": peers, "physical_key": physical}


class ViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock(processes=4, devices=4)

    @classmethod
    def tearDownClass(cls):
        cls.block.close()

    def call(self, fn, per_rank, ranks=range(4)):
        return self.block.call([(r, f"{HERE}:{fn}", per_rank(r)) for r in ranks])

    def test_virtualized_table(self):
        from hetero.vproc.view import VIRTUALIZED
        self.assertEqual(set(VIRTUALIZED), {
            "jax.devices", "jax.device_count", "jax.local_devices", "jax.process_index",
            "jax.process_count", "jax.process_indices", "jax.experimental.multihost_utils",
            "tpu_inference.utils.xb", "tpu_inference.distributed.utils.get_device_topology_order_id",
            "jax._src.compiler._cache_write", "jax.distributed.coordination"})

    def test_api_surface_under_views(self):
        for layout in (((0, 1), (2, 3)), ((0, 3), (1, 2)), ((0, 1, 2, 3),)):
            where = {h: hosts for hosts in layout for h in hosts}
            out = self.call("view_semantics", lambda r: (where[r],))
            for r, res in enumerate(out):
                hosts = where[r]
                rank, mine = hosts.index(r), _ids(r)
                everyone = [i for h in hosts for i in _ids(h)]
                physical = [i for h in range(4) for i in _ids(h)]
                with self.subTest(layout=layout, process=r):
                    self.assertEqual(res["before"]["devices"], physical)
                    self.assertEqual((res["before"]["index"], res["before"]["count"]), (r, 4))
                    a = res["active"]
                    self.assertEqual((a["index"], a["count"], a["indices"]),
                                     (rank, len(hosts), list(range(len(hosts)))))
                    self.assertEqual((a["devices"], a["device_count"]), (everyone, 4 * len(hosts)))
                    self.assertEqual(a["local"], mine)
                    self.assertEqual(a["local_device_count"], 4)
                    self.assertEqual(a["cpu"][:4], mine)
                    self.assertEqual(sorted(a["cpu"]), everyone)
                    self.assertEqual(a["current"], _view(hosts).engine_id)
                    self.assertEqual(res["local_of"], {k: _ids(h) for k, h in enumerate(hosts)})
                    self.assertEqual(res["bad_rank"], "ValueError")
                    i = res["internals"]  # jax._src stays physical
                    self.assertEqual((i["xb_index"], i["xb_count"], i["xb_devices"]), (r, 4, 16))
                    self.assertEqual((i["process_id"], i["num_processes"]), (r, 4))
                    self.assertEqual(i["local_mesh"], mine)
                    self.assertFalse(i["xb_is_patched"])
                    for key in ("suspended", "nested", "after_nested", "before", "after"):
                        s = res[key]
                        self.assertEqual((s["index"], s["count"], s["devices"], s["current"]),
                                         (r, 4, physical, None), key)
                        self.assertEqual(s["indices"], [0, 1, 2, 3])
                        self.assertEqual(s["device_count"], 16)
                    self.assertEqual(res["resumed"], res["active"])

    def test_deactivate_restores_identity(self):
        out = self.call("restore_identity", lambda r: (((0, 1), (2, 3))[r // 2],))
        for r, res in enumerate(out):
            with self.subTest(process=r):
                self.assertGreaterEqual(res["n_patched_names"], 7)  # 6 jax names + _cache_write
                # Only jax._src.xla_bridge keeps the originals (internals stay physical).
                self.assertEqual(res["unchanged_while_active"],
                                 [("jax._src.xla_bridge", n) for n in sorted(PUBLIC)])
                self.assertEqual(res["reexport_rebound"], (True, True))
                self.assertTrue(res["internals_physical"])
                self.assertTrue(res["restored"])
                self.assertTrue(res["meta_path_restored"])
                self.assertIsNone(res["current_after"])

    def test_activation_errors(self):
        for r, res in enumerate(self.call("error_cases", lambda r: (r,))):
            with self.subTest(process=r):
                self.assertEqual((res["out_of_runtime"], res["double"], res["double_deactivate"]),
                                 ("ValueError", "RuntimeError", "RuntimeError"))
                self.assertTrue(res["out_of_runtime_untouched"] and res["still_first"] and res["restored"])
                self.assertEqual(res["foreign"], "ValueError")
                self.assertTrue(res["foreign_untouched"])

    def test_fake_tpu_inference_preloaded(self):
        hosts_of = lambda r: ((0, 1), (2, 3))[r // 2]
        for r, res in enumerate(self.call("fake_tpu_preloaded", lambda r: (hosts_of(r),))):
            hosts = hosts_of(r)
            everyone = [i for h in hosts for i in _ids(h)]
            with self.subTest(process=r):
                self.assertEqual(res["xb_devices"], everyone)
                self.assertTrue(res["xb_forwarded"])
                self.assertEqual(res["physical_rank"], {0: 0, 1: 2, 2: 1, 3: 3}[r])  # (x, y) sort
                self.assertEqual(res["virtual_rank"], hosts.index(r))
                self.assertEqual(res["worker_binding_rank"], hosts.index(r))
                self.assertEqual(res["virtual_rank_virtual_devs"], hosts.index(r))
                self.assertEqual(res["physical_xb_devices"], 16)
                self.assertEqual(res["physical_rank_inside"], res["physical_rank"])
                self.assertEqual(res["after_xb_devices"], 16)
                self.assertTrue(res["restored"])

    def test_fake_tpu_inference_late_import(self):
        with tempfile.TemporaryDirectory() as root:
            files = {"tpu_inference/__init__.py": "", "tpu_inference/distributed/__init__.py": "",
                     "tpu_inference/utils.py": UTILS_SRC,
                     "tpu_inference/distributed/utils.py": DIST_SRC,
                     "tpu_inference/worker.py": WORKER_SRC.replace("from tpu_inference import utils\n", "")}
            for name, text in files.items():
                path = os.path.join(root, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(textwrap.dedent(text))
            hosts_of = lambda r: ((0, 1), (2, 3))[r // 2]
            for r, res in enumerate(self.call("fake_tpu_late", lambda r: (hosts_of(r), root))):
                with self.subTest(process=r):
                    self.assertEqual(res["patched"], (True, True))
                    self.assertEqual(res["late_devices"], 8)
                    self.assertTrue(res["xb_restored"] and res["fn_restored"])
                    self.assertTrue(res["worker_wrapper_passthrough"])

    def test_view_collective_never_waits_on_other_view(self):
        sleep_s = 12
        calls = [(r, f"{HERE}:view_collectives_quick", ((0, 1), "neg-a")) for r in (0, 1)]
        calls += [(r, f"{HERE}:view_collectives_late", ((2, 3), sleep_s)) for r in (2, 3)]
        out = self.block.call(calls)
        done_a = [o[0] for o in out[:2]]
        woke_b = [o[0] for o in out[2:]]
        self.assertLess(max(done_a), min(woke_b), "view {0,1} finished only after view {2,3} woke up")
        self.assertEqual([o[1] for o in out[:2]], [[0, 1]] * 2)
        self.assertEqual([o[2] for o in out[2:]], [[0, 1]] * 2)  # view {2,3} completes afterwards

    def test_coordination_keys_and_barriers_are_scoped(self):
        hosts_of = lambda r: ((0, 1), (2, 3))[r // 2]
        out = self.call("coordination", lambda r: (hosts_of(r), "tok"))
        for r, res in enumerate(out):
            engine = _view(hosts_of(r)).engine_id
            with self.subTest(process=r):
                self.assertEqual(res["peers"], [f"{engine}:0", f"{engine}:1"])
                self.assertEqual(res["physical_key"], f"{engine}:{hosts_of(r).index(r)}")

    def test_noncontiguous_views_match_standalone(self):
        with CpuBlock(processes=2, devices=4) as alone:
            ref = alone.call_all(f"{HERE}:engine_program", None, None)
        for layout in (((0, 3), (1, 2)),):
            where = {h: hosts for hosts in layout for h in hosts}
            out = self.call("engine_program", lambda r: (None, where[r]))
            for r, res in enumerate(out):
                rank = where[r].index(r)
                self.assertEqual(_comparable(res, rank), _comparable(ref[rank], rank))


def _comparable(res, rank):
    drop = {"raw_devices", "write_keys", "cpu_local", "cpu_val", "writes"}
    return {k: v for k, v in res.items() if k not in drop}


class ConformanceTests(unittest.TestCase):
    def test_standalone_equals_virtualized_on_two_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = {name: os.path.join(tmp, name) for name in ("alone", "A", "B")}
            with CpuBlock(processes=2, devices=4) as alone:
                ref = alone.call_all(f"{HERE}:engine_program", dirs["alone"], None)
            with CpuBlock(processes=4, devices=4) as block:
                calls = [(r, f"{HERE}:engine_program", (dirs["A"], (0, 1))) for r in (0, 1)]
                calls += [(r, f"{HERE}:engine_program", (dirs["B"], (2, 3))) for r in (2, 3)]
                out = block.call(calls)
            views = {"A": (out[0:2], (0, 1)), "B": (out[2:4], (2, 3))}

            # Observable results identical after normalizing device ids.
            self.assertEqual([(r["rank"], r["count"]) for r in ref], [(0, 2), (1, 2)])
            for name, (results, hosts) in views.items():
                for rank, res in enumerate(results):
                    with self.subTest(view=name, rank=rank):
                        self.assertEqual(_comparable(res, rank), _comparable(ref[rank], rank))
                        self.assertEqual(res["raw_devices"], [i for h in hosts for i in _ids(h)])
                        # jax.devices("cpu")[0] is a usable local device under a view.
                        self.assertTrue(res["cpu_local"])
                        self.assertEqual(res["cpu_val"], 6.0)
            # On the CPU stand-in the CPU client spans processes, so standalone rank 1 sees
            # a remote device first; only rank 0 is comparable (a TPU host's CPU client is local).
            self.assertTrue(ref[0]["cpu_local"])
            self.assertEqual(ref[0]["cpu_val"], 6.0)

            # Persistent compile cache: exactly one writer per view (view rank 0), same count.
            self.assertGreater(ref[0]["writes"], 0)
            self.assertEqual(ref[1]["writes"], 0)
            for name, (results, hosts) in views.items():
                self.assertEqual([r["writes"] for r in results], [ref[0]["writes"], 0], name)
                self.assertEqual(results[0]["write_keys"], sorted(set(results[0]["write_keys"])))
            files = {name: sorted(os.listdir(path)) for name, path in dirs.items()}
            self.assertEqual(len(files["alone"]), len(files["A"]))
            self.assertEqual(len(files["alone"]), len(files["B"]))
            for name, (results, _) in views.items():
                entries = [f for f in files[name] if f.endswith("-cache")]
                self.assertEqual(sorted(entries), sorted(k + "-cache" for k in results[0]["write_keys"]))
            self.assertEqual(set(files["A"]) & set(files["B"]), set())  # different device assignments


if __name__ == "__main__":
    unittest.main()
