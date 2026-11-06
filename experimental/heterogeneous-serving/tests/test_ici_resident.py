"""CPU invariant tests; these do not qualify TPU multi-controller behavior."""
import unittest
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from hetero.ici_resident import reshard_tree, repack_merged, resident_views


@partial(jax.tree_util.register_dataclass, data_fields=['value'], meta_fields=['tp'])
@dataclass
class MeshBoundVariable:
    value: object
    tp: int

class ResidentTests(unittest.TestCase):

    def test_nnx_rebuilt_graph_uses_new_tp_and_resident_values(self):
        try:
            from flax import nnx
        except ImportError:
            self.skipTest('Flax absent from control venv; run pinned local command')
        from hetero.ici_resident import repack_state
        devices = jax.devices('cpu')[:16]
        if len(devices) < 16:
            self.skipTest('requires sixteen forced CPU devices')
        host = jnp.arange(3 * 128, dtype=jnp.float32).reshape(3, 128)
        class Merged(nnx.Module):
            def __init__(self, tp):
                self.tp = tp
                self.weight = nnx.Param(jnp.zeros((3, 128), jnp.float32), sharding=(None, 'model'), eager_sharding=False)
            def __call__(self, x):
                logical = repack_merged(self.weight.get_value(), (64, 64), self.tp, 1)
                return x @ logical
        large = Mesh(np.array(devices), ('model',))
        small = Mesh(np.array(devices[:8]), ('model',))
        graph16 = Merged(16)
        graph16.weight.set_value(jax.device_put(repack_merged(host, (64, 64), 1, 16), NamedSharding(large, P(None, 'model'))))
        source = nnx.state(graph16)
        transferred = repack_state(source, {('weight',): (64, 64)}, 16, 8)
        graph8 = nnx.eval_shape(lambda: Merged(8))
        abstract = nnx.state(graph8)
        specs = nnx.get_named_sharding(abstract, small)
        resident = reshard_tree(transferred, abstract, specs)
        nnx.update(graph8, resident)
        x = jnp.array([[1., 2., 3.]])
        np.testing.assert_array_equal(np.asarray(graph8(x)), np.asarray(x @ host))
        self.assertEqual(graph8.tp, 8)
        naive_graph8 = nnx.eval_shape(lambda: Merged(8))
        nnx.update(naive_graph8, reshard_tree(source, nnx.state(naive_graph8), specs))
        self.assertFalse(np.array_equal(np.asarray(naive_graph8(x)), np.asarray(x @ host)))
    def test_replica_global_array_to_submesh_handles_and_back(self):
        devices = jax.devices('cpu')[:16]
        if len(devices) < 16:
            self.skipTest('requires sixteen forced CPU devices')
        replica_mesh = Mesh(np.array(devices).reshape(2, 8), ('replica', 'model'))
        meshes = [Mesh(np.array(devices[:8]), ('model',)), Mesh(np.array(devices[8:]), ('model',))]
        host = np.arange(1024, dtype=np.float32).reshape(64, 16)
        source = {'weight': jax.device_put(host, NamedSharding(replica_mesh, P(None, 'model')))}
        abstract = {'weight': jax.ShapeDtypeStruct(host.shape, host.dtype)}
        views = [resident_views([source], abstract, {'weight': NamedSharding(mesh, P(None, 'model'))}) for mesh in meshes]
        for view, mesh in zip(views, meshes):
            np.testing.assert_array_equal(np.asarray(view['weight']), host)
            self.assertEqual(view['weight'].sharding.mesh, mesh)
        back = resident_views(views, abstract, {'weight': NamedSharding(replica_mesh, P(None, 'model'))})
        np.testing.assert_array_equal(np.asarray(back['weight']), host)
        with self.assertRaisesRegex(ValueError, 'changes device shard index'):
            resident_views([source], abstract, {'weight': NamedSharding(meshes[0], P('model', None))})

    def test_roundtrip_disjoint_mesh_values_and_shardings(self):
        devices = jax.devices('cpu')
        if len(devices) < 16:
            self.skipTest('requires sixteen forced CPU devices')
        large = Mesh(np.array(devices[:16]), ('model',))
        small_a = Mesh(np.array(devices[:8]), ('model',))
        small_b = Mesh(np.array(devices[8:16]), ('model',))
        host = np.arange(1024, dtype=np.float32).reshape(64, 16)
        source = {'weight': jax.device_put(host, NamedSharding(large, P('model', None)))}
        abstract = {'weight': jax.ShapeDtypeStruct(host.shape, host.dtype)}
        destinations = []
        for mesh in (small_a, small_b):
            shardings = {'weight': NamedSharding(mesh, P(None, 'model'))}
            result = reshard_tree(source, abstract, shardings)
            np.testing.assert_array_equal(np.asarray(result['weight']), host)
            self.assertEqual(result['weight'].sharding, shardings['weight'])
            destinations.append(result)
        back = reshard_tree(destinations[0], abstract, {'weight': NamedSharding(large, P('model', None))})
        np.testing.assert_array_equal(np.asarray(back['weight']), host)
    def test_destination_static_metadata_replaces_source(self):
        mesh = Mesh(np.array(jax.devices('cpu')[:1]), ('model',))
        sharding = NamedSharding(mesh, P())
        source = {'w': MeshBoundVariable(jax.device_put(jnp.arange(32), sharding), 16)}
        abstract = {'w': MeshBoundVariable(jax.ShapeDtypeStruct((32,), jnp.int32), 8)}
        specs = {'w': MeshBoundVariable(sharding, 8)}
        result = reshard_tree(source, abstract, specs)
        self.assertEqual(result['w'].tp, 8)
        np.testing.assert_array_equal(np.asarray(result['w'].value), np.arange(32))


    def test_merged_equal_and_unequal_projection_repacking(self):
        for sizes in ((64, 64), (128, 32, 32)):
            for shape in ((sum(sizes),), (3, sum(sizes))):
                with self.subTest(sizes=sizes, shape=shape):
                    canonical = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
                    packed16 = repack_merged(canonical, sizes, 1, 16)
                    packed8 = jax.jit(lambda x: repack_merged(x, sizes, 16, 8))(packed16)
                    logical = repack_merged(packed8, sizes, 8, 1)
                    np.testing.assert_array_equal(np.asarray(logical), np.asarray(canonical))
                    self.assertFalse(np.array_equal(np.asarray(packed16), np.asarray(packed8)))
                    back = repack_merged(packed8, sizes, 8, 16)
                    np.testing.assert_array_equal(np.asarray(back), np.asarray(packed16))
                    if len(shape) == 2:
                        x = jnp.array([[1., 2., 3.]])
                        np.testing.assert_array_equal(np.asarray(x @ logical), np.asarray(x @ canonical))

    def test_refuse_incompatible_destination(self):
        mesh = Mesh(np.array(jax.devices('cpu')[:1]), ('model',))
        source = {'w': jax.device_put(jnp.ones((16, 8)), NamedSharding(mesh, P()))}
        for abstract, shardings in (
            ({'different': jax.ShapeDtypeStruct((16, 8), jnp.float32)}, {'different': NamedSharding(mesh, P())}),
            ({'w': jax.ShapeDtypeStruct((8, 16), jnp.float32)}, {'w': NamedSharding(mesh, P())}),
            ({'w': jax.ShapeDtypeStruct((16, 8), jnp.bfloat16)}, {'w': NamedSharding(mesh, P())}),
        ):
            with self.assertRaises(ValueError):
                reshard_tree(source, abstract, shardings)
        with self.assertRaises(ValueError):
            repack_merged(jnp.ones((3, 64)), (31, 33), 16, 8)


if __name__ == '__main__':
    unittest.main()
