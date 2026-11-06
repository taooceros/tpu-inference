"""CPU block harness: one 4-process x 4-device runtime, submesh programs per host pair."""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.vproc.cpu_block import CPU_ID_STRIDE, BlockCallError, CpuBlock
from hetero.vproc.types import (BLOCK_LAYOUTS, EngineSpec, IllegalHostSet, ViewSpec,
                                check_host_set)

HERE = __name__


def identity():
    import jax
    return {"process_index": jax.process_index(), "process_count": jax.process_count(),
            "devices": [d.id for d in jax.devices()], "local": [d.id for d in jax.local_devices()]}


def tp_sum(hosts, scale):
    """psum over the chips of ``hosts`` only; launched by those hosts alone."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    devices = [d for d in jax.devices() if d.process_index in hosts]
    mesh = Mesh(np.array(devices), ("model",))
    n = len(devices)
    host = np.arange(n * 4, dtype=np.float32).reshape(n, 4) * scale
    sharding = NamedSharding(mesh, P("model", None))
    x = jax.make_array_from_callback(host.shape, sharding, lambda index: host[index])
    total = jax.jit(lambda a: jnp.sum(a, axis=0), out_shardings=NamedSharding(mesh, P()))(x)
    return np.asarray(total.addressable_data(0)).tolist()


def boom():
    raise ValueError("expected failure")


class TypesTests(unittest.TestCase):
    def test_host_sets(self):
        self.assertEqual(check_host_set([3, 2]), (2, 3))
        for bad in ([0], [0, 2], [1], [1, 1], [0, 4]):
            with self.assertRaises(IllegalHostSet):
                check_host_set(bad)
        for layout in BLOCK_LAYOUTS.values():
            for hosts in layout:
                check_host_set(hosts)

    def test_view(self):
        view = EngineSpec("tp8b", "synthetic", (3, 2), 8).view()
        self.assertEqual((view.hosts, view.namespace, view.rank_of(3)), ((2, 3), "vproc/tp8b", 1))
        with self.assertRaises(ValueError):
            ViewSpec("x", (2, 2))


class CpuBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.block = CpuBlock()

    @classmethod
    def tearDownClass(cls):
        cls.block.close()

    def test_identity(self):
        out = self.block.call_all(f"{HERE}:identity")
        everyone = [p * CPU_ID_STRIDE + i for p in range(4) for i in range(4)]
        for rank, ident in enumerate(out):
            self.assertEqual(ident["process_index"], rank)
            self.assertEqual(ident["process_count"], 4)
            self.assertEqual(ident["devices"], everyone)
            self.assertEqual(ident["local"], everyone[rank * 4:rank * 4 + 4])

    def test_whole_block_collective(self):
        out = self.block.call_all(f"{HERE}:tp_sum", (0, 1, 2, 3), 1.0)
        expect = [sum(range(c, 64, 4)) for c in range(4)]
        self.assertEqual(out, [expect] * 4)

    def test_disjoint_pairs_run_concurrently(self):
        calls = [(r, f"{HERE}:tp_sum", ((0, 1), 1.0)) for r in (0, 1)]
        calls += [(r, f"{HERE}:tp_sum", ((2, 3), 2.0)) for r in (2, 3)]
        out = self.block.call(calls)
        a = [float(sum(range(c, 32, 4))) for c in range(4)]
        self.assertEqual(out, [a, a, [2 * v for v in a], [2 * v for v in a]])

    def test_remote_error(self):
        with self.assertRaises(BlockCallError) as ctx:
            self.block.call([(1, f"{HERE}:boom", ())])
        self.assertIn("expected failure", ctx.exception.remote_traceback)
        self.assertEqual(self.block.call([(1, f"{HERE}:identity", ())])[0]["process_index"], 1)


if __name__ == "__main__":
    unittest.main()
