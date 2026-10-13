"""Host-only exact shard restore, compatibility, and explicit opt-in contract."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from hetero.engine_actor import EngineSettings
from hetero.weight_snapshot import SnapshotRefused, prepare, restore, snapshot


class WeightSnapshotTests(unittest.TestCase):
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
        self.report = snapshot(self.directory, self.tree, self.context)
        self.abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), self.tree)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_roundtrip_final_shards(self):
        clocks = {}
        restored = restore(self.abstract, self.mesh, prepare(self.directory, self.context, self.mesh, clocks), clocks)
        for before, after in zip(jax.tree_util.tree_leaves(self.tree), jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(np.asarray(before).reshape(-1).view(np.uint8), np.asarray(after).reshape(-1).view(np.uint8))
            self.assertEqual(before.sharding, after.sharding)
            self.assertEqual([s.index for s in before.addressable_shards], [s.index for s in after.addressable_shards])
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.assertEqual(manifest['context'], self.context)
        self.assertEqual(self.report['bytes'], sum(s['bytes'] for a in manifest['arrays'] for s in a['shards']))

    def test_composite_partition_axis_roundtrip(self):
        if self.mesh.size != 8:
            self.skipTest('run with eight forced CPU devices')
        mesh = Mesh(self.mesh.devices.reshape(2, 4), ('x', 'y'))
        sharding = NamedSharding(mesh, PartitionSpec(('x', 'y'), None))
        original = {'weight': jax.device_put(
            jnp.arange(256, dtype=jnp.bfloat16).reshape(32, 8), sharding)}
        directory = self.directory / 'composite'
        snapshot(directory, original, self.context)
        abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), original)
        restored = restore(abstract, mesh, prepare(directory, self.context, mesh, {}), {})
        np.testing.assert_array_equal(np.asarray(original['weight']), np.asarray(restored['weight']))
        self.assertEqual(restored['weight'].sharding.spec, PartitionSpec(('x', 'y'), None))

    def test_refuse_tp_model_hash_dtype_quantization_changes(self):
        for key, value in [('tp_size', 16), ('model_path', '/other'), ('model_hash', 'changed'),
                           ('dtype', 'float32'), ('quantization', 'int8')]:
            with self.subTest(key=key):
                context = {**self.context, key: value}
                with self.assertRaisesRegex(SnapshotRefused, key):
                    prepare(self.directory, context, self.mesh, {})

    def test_refuse_device_order(self):
        if self.mesh.size < 2:
            self.skipTest('run with XLA_FLAGS=--xla_force_host_platform_device_count=8')
        reversed_mesh = Mesh(self.mesh.devices[::-1], ('tp',))
        with self.assertRaisesRegex(SnapshotRefused, 'device order'):
            prepare(self.directory, self.context, reversed_mesh, {})

    def test_refuse_missing_shard(self):
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        (self.directory / manifest['arrays'][0]['shards'][0]['file']).unlink()
        with self.assertRaisesRegex(SnapshotRefused, 'missing shard'):
            prepare(self.directory, self.context, self.mesh, {})

    def test_refuse_checksum_mismatch(self):
        manifest = json.loads((self.directory / 'manifest.json').read_text())
        shard = self.directory / manifest['arrays'][0]['shards'][0]['file']
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(SnapshotRefused, 'checksum mismatch'):
            prepare(self.directory, self.context, self.mesh, {})

    def test_refuse_abstract_shape_and_dtype(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        for dtype in [jnp.float16, jnp.float32]:
            abstract = copy.deepcopy(self.abstract)
            abstract['layer']['weight'] = jax.ShapeDtypeStruct(self.tree['layer']['weight'].shape, dtype)
            with self.assertRaisesRegex(SnapshotRefused, 'dtype mismatch'):
                restore(abstract, self.mesh, prepared, {})

    def test_setting_off_by_default_and_reject_invalid(self):
        self.assertEqual(EngineSettings.from_dict({}).weight_snapshot, 'off')
        for mode in ['read', 'write']:
            self.assertEqual(EngineSettings.from_dict({'weight_snapshot': mode}).weight_snapshot, mode)
        with self.assertRaisesRegex(ValueError, 'weight_snapshot'):
            EngineSettings.from_dict({'weight_snapshot': 'typo'})
