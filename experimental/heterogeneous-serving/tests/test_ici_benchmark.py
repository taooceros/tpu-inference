"""Value preservation and movement in isolated 16/32-device CPU meshes."""
import os
import pathlib
import subprocess
import sys
import unittest


class ReshardLogic(unittest.TestCase):
    def test_roundtrip_and_interleaved_local_retention(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = r'''
import jax
import numpy as np
from hetero.ici_benchmark import meshes
from hetero.ici_payload import generate, merge_local, movement, put_tree, schema, verify
from jax.sharding import PartitionSpec as P
m = meshes()
# Cover different partition axes and replicated leaves, including global-K
# head_dim fallback rather than only a flattened one-dimensional weight.
c = {'column': ((32, 64), P(None, 'model')),
     'row': ((64, 32), P('model', None)),
     'global_k': ((32, 4, 64), P(None, None, 'model')),
     'norm': ((32,), P(None))}
s, _ = generate(c, m['replicas'])
for destination in ('h', 'interleaved'):
    local, _ = merge_local(s, c, m['interleaved'])
    merged = put_tree(local, c, m[destination])
    jax.block_until_ready(merged)
    verify(merged, c)
    restored = put_tree(merged, c, m['replicas'])
    jax.block_until_ready(restored)
    verify(restored, c)
    for name in c:
        np.testing.assert_array_equal(np.asarray(restored[name]), np.asarray(s[name]))
assert movement(c, m['replicas'], m['interleaved'])['logical_bytes_moved'] == 0
w = schema('tiny')
assert movement(w, m['replicas'], m['h'])['logical_bytes_moved'] == 32768 * 5 // 8
assert movement(w, m['h'], m['replicas'])['logical_bytes_moved'] == 32768 * 13 // 8
# Corrupted resident content must be caught by device-side verification.
bad = {name: x for name, x in merged.items()}
leaf = 'column'
buffers = [sh.data for sh in merged[leaf].addressable_shards]
buffers[0] = jax.jit(lambda x: x + 1)(buffers[0])
bad[leaf] = jax.make_array_from_single_device_arrays(merged[leaf].shape, merged[leaf].sharding, buffers)
try:
    verify(bad, c)
except ValueError:
    pass
else:
    raise AssertionError('verification missed corrupted shard placement')
# Full runtime uses an unused replica-box axis, not a TP32 model.
from hetero.ici_benchmark import slice_meshes
from jax.sharding import Mesh
full = slice_meshes()
gs, _ = generate(c, full['replicas'])
gh = put_tree(gs, c, full['h'])
jax.block_until_ready(gh)
verify(gh, c)
back = put_tree(gh, c, full['replicas'])
jax.block_until_ready(back)
verify(back, c)
assert movement(w, full['replicas'], full['h'])['logical_bytes_moved'] == 32768 * 2 * 5 // 8
mixed = Mesh(np.array([full['interleaved'].devices[0], full['h'].devices[1]]), ('box','model'))
ledger = movement(w, mixed, full['h'])
assert ledger['logical_bytes_moved'] == 32768 * 5 // 8
assert all(ledger['per_device_received_bytes'][str(i)] == 0 for i in range(16,32))
print('roundtrip, byte ledger, corruption detection, and neighbor retention passed')
'''
        env = dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32')
        result = subprocess.run([sys.executable, '-c', script], cwd=root, env=env,
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
