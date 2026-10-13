"""Placement from the manifest before the abstract graph exists; verify/build overlap."""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from hetero.restore_graph import (assemble, build_while_verifying, cheap_tracer_sharding_errors,
                                  place_from_manifest)
from hetero.weight_snapshot import SnapshotRefused, prepare, restore, snapshot


class RestoreGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        devices = jax.devices('cpu')[:8]
        self.mesh = Mesh(np.array(devices), ('tp',))
        n = len(devices)
        self.tree = {'layer': {'weight': jax.device_put(
            jnp.arange(n * 16, dtype=jnp.bfloat16).reshape(n * 4, 4),
            NamedSharding(self.mesh, PartitionSpec('tp', None)))},
            'scale': jax.device_put(jnp.array([1., -0., 3.], dtype=jnp.float32),
                                   NamedSharding(self.mesh, PartitionSpec())),
            'scalar': jax.device_put(jnp.array(-0., dtype=jnp.bfloat16),
                                    NamedSharding(self.mesh, PartitionSpec()))}
        self.context = {'model_path': '/models/test', 'model_hash': 'abc',
                        'tp_size': n, 'dtype': 'bfloat16', 'quantization': None}
        snapshot(self.directory, self.tree, self.context)
        self.abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), self.tree)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_placement_matches_serial_restore(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        manifest, buffers = prepared
        arrays = place_from_manifest(manifest, buffers, self.mesh)
        clocks = {}
        assembled = assemble(self.abstract, self.mesh, prepared, arrays, clocks)
        serial = restore(self.abstract, self.mesh, prepared, {})
        self.assertIn('snapshot_device_placement', clocks)
        for a, b in zip(jax.tree_util.tree_leaves(assembled), jax.tree_util.tree_leaves(serial)):
            np.testing.assert_array_equal(np.asarray(a).reshape(-1).view(np.uint8),
                                          np.asarray(b).reshape(-1).view(np.uint8))
            self.assertEqual(a.sharding, b.sharding)
            self.assertEqual([s.index for s in a.addressable_shards], [s.index for s in b.addressable_shards])
        self.assertEqual(jax.tree_util.tree_structure(assembled), jax.tree_util.tree_structure(self.abstract))

    def test_graph_mismatch_after_placement_drops_arrays(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        arrays = place_from_manifest(prepared[0], prepared[1], self.mesh)
        abstract = copy.deepcopy(self.abstract)
        abstract['layer']['weight'] = jax.ShapeDtypeStruct(self.tree['layer']['weight'].shape, jnp.float32)
        with self.assertRaisesRegex(SnapshotRefused, 'dtype mismatch'):
            assemble(abstract, self.mesh, prepared, arrays, {})
        self.assertEqual(arrays, [])

    def test_placement_refuses_wrong_shard_devices_before_transfer(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        manifest = copy.deepcopy(prepared[0])
        manifest['arrays'][0]['shards'][0]['device_id'] = manifest['arrays'][0]['shards'][-1]['device_id']
        with self.assertRaisesRegex(SnapshotRefused, 'shard devices'):
            place_from_manifest(manifest, prepared[1], self.mesh)

    def test_build_runs_concurrently_with_verify(self):
        order = []
        release = threading.Event()

        def verify():
            order.append('verify-start')
            release.wait(5)
            order.append('verify-end')
            return 'ok'

        def build():
            order.append('build-start')
            release.set()
            return 'graph'

        built, verified = build_while_verifying(build, verify)
        self.assertEqual((built, verified), ('graph', 'ok'))
        self.assertEqual(order[:2], ['verify-start', 'build-start'])
        self.assertEqual(order[-1], 'verify-end')

    def test_build_failure_still_drains_verifier(self):
        finished = threading.Event()

        def verify():
            time.sleep(0.05)
            finished.set()

        with self.assertRaisesRegex(RuntimeError, 'boom'):
            build_while_verifying(lambda: (_ for _ in ()).throw(RuntimeError('boom')), verify)
        self.assertTrue(finished.is_set())

    def test_verify_failure_propagates_after_build(self):
        def verify():
            raise SnapshotRefused('checksum mismatch: x')

        with self.assertRaisesRegex(SnapshotRefused, 'checksum mismatch'):
            build_while_verifying(lambda: 'graph', verify)

    def test_tracer_sharding_probe_skips_origin_walk_inside_block_only(self):
        from jax._src import core
        from jax._src.interpreters import partial_eval

        walks = []
        original_origin = partial_eval.DynamicJaxprTracer._origin_msg
        partial_eval.DynamicJaxprTracer._origin_msg = lambda self: walks.append(1) or ''
        try:
            def probe():
                # Fresh closure each call: jax caches traces per function object.
                return lambda x: jnp.zeros(()) if getattr(x + 1, 'sharding', None) is None else x

            before = core.Tracer.__dict__['sharding']
            with cheap_tracer_sharding_errors():
                jax.eval_shape(probe(), jnp.zeros((2,)))
                self.assertEqual(walks, [])
                self.assertIsNot(core.Tracer.__dict__['sharding'], before)
            self.assertIs(core.Tracer.__dict__['sharding'], before)
            jax.eval_shape(probe(), jnp.zeros((2,)))
            self.assertEqual(walks, [1])
        finally:
            partial_eval.DynamicJaxprTracer._origin_msg = original_origin

    def test_tracer_sharding_patch_restored_after_error(self):
        from jax._src import core
        before = core.Tracer.__dict__['sharding']
        with self.assertRaises(KeyError):
            with cheap_tracer_sharding_errors():
                raise KeyError('x')
        self.assertIs(core.Tracer.__dict__['sharding'], before)


if __name__ == '__main__':
    unittest.main()
