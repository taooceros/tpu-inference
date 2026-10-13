"""Host-only exact shard restore, compatibility, and explicit opt-in contract."""
import copy
import json
from pathlib import Path
import tempfile
from unittest import mock
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from hetero.engine_actor import EngineSettings
from hetero.weight_snapshot import SnapshotRefused, fingerprint, prepare, restore, snapshot


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

    def test_restored_arrays_survive_snapshot_overwrite(self):
        expected = [np.asarray(a).copy() for a in jax.tree_util.tree_leaves(self.tree)]
        before = self._inodes(self.directory)
        restored = self._roundtrip(self.directory, self.tree)
        second = self._make_tree(offset=50.0)
        snapshot(self.directory, second, self.context)
        self.assertEqual(self._inodes(self.directory), before)
        for want, got in zip(expected, jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(want.reshape(-1).view(np.uint8),
                                          np.asarray(got).reshape(-1).view(np.uint8))
        self._roundtrip(self.directory, second)

    def test_snapshot_refused_while_prepared_is_held(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        with self.assertRaisesRegex(SnapshotRefused, 'busy'):
            snapshot(self.directory, self._make_tree(3.0), self.context)
        prepared.close()
        prepared.close()  # idempotent
        second = self._make_tree(3.0)
        snapshot(self.directory, second, self.context)
        self._roundtrip(self.directory, second)

    def test_restore_releases_lock_after_readiness(self):
        self._roundtrip(self.directory, self.tree)
        snapshot(self.directory, self._make_tree(4.0), self.context)

    def test_prepare_refused_while_snapshot_is_writing(self):
        real = np.copyto
        seen = []

        def probing(*args, **kwargs):
            try:
                prepare(self.directory, self.context, self.mesh, {})
            except SnapshotRefused as exc:
                seen.append(str(exc))
            else:
                seen.append(None)
            return real(*args, **kwargs)

        with mock.patch.object(np, 'copyto', probing):
            snapshot(self.directory, self._make_tree(6.0), self.context)
        self.assertTrue(seen)
        for message in seen:
            self.assertIn('busy', message or '')

    def test_fingerprint_stable_sensitive_and_matches_restore(self):
        first = fingerprint(self.tree)
        self.assertEqual(first['sha256'], fingerprint(self.tree)['sha256'])
        self.assertEqual(first['bytes'], self.report['bytes'])
        changed = fingerprint(self._make_tree(offset=1.0))
        self.assertNotEqual(first['sha256'], changed['sha256'])
        restored = self._roundtrip(self.directory, self.tree)
        self.assertEqual(fingerprint(restored)['sha256'], first['sha256'])
        self.assertEqual(fingerprint(restored)['bytes'], first['bytes'])

    def test_fingerprint_refuses_non_sharded_leaf(self):
        with self.assertRaises(SnapshotRefused):
            fingerprint({'host': np.zeros(3, np.float32)})

    def test_composite_partition_axis_roundtrip(self):
        if self.mesh.size not in (4, 8):
            self.skipTest('run with four or eight forced CPU devices')
        mesh = Mesh(self.mesh.devices.reshape(2, self.mesh.size // 2), ('x', 'y'))
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

    def _make_tree(self, offset=0.0, rows=None):
        n = self.mesh.size
        rows = rows or n * 4
        return {'layer': {'weight': jax.device_put(
            (jnp.arange(rows * 4, dtype=jnp.float32) + offset).astype(jnp.bfloat16).reshape(rows, 4),
            NamedSharding(self.mesh, PartitionSpec('tp', None)))},
            'scale': jax.device_put(jnp.array([1., -0., 3.], dtype=jnp.float32) + offset,
                                   NamedSharding(self.mesh, PartitionSpec())),
            'scalar': jax.device_put(jnp.array(-0., dtype=jnp.bfloat16) + offset,
                                    NamedSharding(self.mesh, PartitionSpec()))}

    def _roundtrip(self, directory, tree, context=None):
        abstract = jax.tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), tree)
        restored = restore(abstract, self.mesh, prepare(directory, context or self.context, self.mesh, {}), {})
        for before, after in zip(jax.tree_util.tree_leaves(tree), jax.tree_util.tree_leaves(restored)):
            np.testing.assert_array_equal(np.asarray(before).reshape(-1).view(np.uint8),
                                          np.asarray(after).reshape(-1).view(np.uint8))
            self.assertEqual(before.sharding, after.sharding)
        return restored

    def _manifest(self, directory=None):
        return json.loads(((directory or self.directory) / 'manifest.json').read_text())

    def _inodes(self, directory):
        return {p.name: p.stat().st_ino for p in directory.glob('*.raw')}

    def test_default_is_version2_without_checksums(self):
        manifest = self._manifest()
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['checksum_kind'], 'none')
        self._roundtrip(self.directory, self.tree)

    def test_sha256_roundtrip_and_corruption_refused(self):
        directory = self.directory / 'sha'
        snapshot(directory, self.tree, self.context, checksum='sha256')
        manifest = self._manifest(directory)
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['checksum_kind'], 'sha256')
        self._roundtrip(directory, self.tree)
        shard = directory / manifest['arrays'][0]['shards'][0]['file']
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(SnapshotRefused, 'checksum mismatch'):
            prepare(directory, self.context, self.mesh, {})

    def test_version1_manifest_still_readable(self):
        directory = self.directory / 'v1'
        snapshot(directory, self.tree, self.context, checksum='sha256')
        manifest = self._manifest(directory)
        manifest['version'] = 1
        manifest.pop('checksum_kind')
        (directory / 'manifest.json').write_text(json.dumps(manifest))
        self._roundtrip(directory, self.tree)
        shard = directory / manifest['arrays'][0]['shards'][0]['file']
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(SnapshotRefused, 'checksum mismatch'):
            prepare(directory, self.context, self.mesh, {})

    def test_refuse_unknown_manifest_version(self):
        for version in [0, 3, None]:
            with self.subTest(version=version):
                manifest = self._manifest()
                manifest['version'] = version
                (self.directory / 'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(SnapshotRefused, 'version'):
                    prepare(self.directory, self.context, self.mesh, {})

    def test_reuse_preserves_inodes_and_restores_new_payload(self):
        before = self._inodes(self.directory)
        self.assertTrue(before)
        second = self._make_tree(offset=100.0)
        snapshot(self.directory, second, self.context)
        self.assertEqual(self._inodes(self.directory), before)
        restored = self._roundtrip(self.directory, second)
        self.assertNotEqual(float(np.asarray(restored['scale'])[0]), 1.0)

    def test_reuse_with_different_shape_restores_exactly(self):
        third = self._make_tree(offset=7.0, rows=self.mesh.size * 2)
        snapshot(self.directory, third, self.context, checksum='sha256')
        self._roundtrip(self.directory, third)
        for array in self._manifest()['arrays']:
            for shard in array['shards']:
                self.assertEqual((self.directory / shard['file']).stat().st_size, shard['bytes'])

    def test_reuse_switching_checksum_kind(self):
        snapshot(self.directory, self._make_tree(1.0), self.context, checksum='sha256')
        self.assertEqual(self._manifest()['checksum_kind'], 'sha256')
        second = self._make_tree(2.0)
        snapshot(self.directory, second, self.context, checksum='none')
        self.assertEqual(self._manifest()['checksum_kind'], 'none')
        self._roundtrip(self.directory, second)

    def test_failed_write_publishes_no_manifest(self):
        # A later leaf is invalid, so earlier shards are already written when it fails.
        bad = dict(self._make_tree(5.0))
        bad['zz_host'] = np.zeros(3, np.float32)
        directory = self.directory / 'failed'
        with self.assertRaises(SnapshotRefused):
            snapshot(directory, bad, self.context)
        self.assertFalse((directory / 'manifest.json').exists())
        with self.assertRaises(SnapshotRefused):
            prepare(directory, self.context, self.mesh, {})

    def test_failed_rewrite_removes_previous_manifest(self):
        bad = dict(self._make_tree(5.0))
        bad['zz_host'] = np.zeros(3, np.float32)
        with self.assertRaises(SnapshotRefused):
            snapshot(self.directory, bad, self.context)
        self.assertFalse((self.directory / 'manifest.json').exists())
        with self.assertRaises(SnapshotRefused):
            prepare(self.directory, self.context, self.mesh, {})

    def test_copy_failure_midway_publishes_no_manifest(self):
        # Assumes the writer copies shard bytes with numpy.copyto.
        directory = self.directory / 'copyfail'
        real = np.copyto
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise OSError('injected copy failure')
            return real(*args, **kwargs)

        with mock.patch.object(np, 'copyto', flaky):
            with self.assertRaises(OSError):
                snapshot(directory, self.tree, self.context)
        self.assertGreaterEqual(len(calls), 3)
        self.assertFalse((directory / 'manifest.json').exists())

    def test_failed_copy_on_reuse_removes_old_manifest(self):
        real = np.copyto
        calls = []

        def flaky(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise OSError('injected copy failure')
            return real(*args, **kwargs)

        with mock.patch.object(np, 'copyto', flaky):
            with self.assertRaises(OSError):
                snapshot(self.directory, self._make_tree(9.0), self.context)
        self.assertFalse((self.directory / 'manifest.json').exists())

    def test_refuse_truncated_or_extended_shard(self):
        manifest = self._manifest()
        shard = self.directory / manifest['arrays'][0]['shards'][0]['file']
        original = shard.read_bytes()
        for data in [original[:-1], b'', original + b'\0']:
            with self.subTest(size=len(data)):
                shard.write_bytes(data)
                with self.assertRaisesRegex(SnapshotRefused, 'size mismatch'):
                    prepare(self.directory, self.context, self.mesh, {})

    def test_refuse_changed_tree(self):
        prepared = prepare(self.directory, self.context, self.mesh, {})
        missing = copy.deepcopy(self.abstract)
        del missing['scalar']
        with self.assertRaisesRegex(SnapshotRefused, 'parameter count mismatch'):
            restore(missing, self.mesh, prepared, {})
        renamed = copy.deepcopy(self.abstract)
        renamed['renamed'] = renamed.pop('scalar')
        with self.assertRaisesRegex(SnapshotRefused, 'mismatch'):
            restore(renamed, self.mesh, prepared, {})
        reshaped = copy.deepcopy(self.abstract)
        reshaped['layer']['weight'] = jax.ShapeDtypeStruct((self.mesh.size * 8, 4), jnp.bfloat16)
        with self.assertRaisesRegex(SnapshotRefused, 'mismatch'):
            restore(reshaped, self.mesh, prepared, {})

    def test_refuse_checksum_mismatch(self):
        directory = self.directory / 'sha-legacy'
        snapshot(directory, self.tree, self.context, checksum='sha256')
        manifest = self._manifest(directory)
        shard = directory / manifest['arrays'][0]['shards'][0]['file']
        data = bytearray(shard.read_bytes())
        data[0] ^= 1
        shard.write_bytes(data)
        with self.assertRaisesRegex(SnapshotRefused, 'checksum mismatch'):
            prepare(directory, self.context, self.mesh, {})

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
