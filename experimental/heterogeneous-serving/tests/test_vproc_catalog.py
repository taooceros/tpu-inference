"""Gemma-4 leaf catalog: measured shapes, scaled copies, fused stored layouts."""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from hetero.ici_resident import repack_merged
from hetero.vproc import catalog


class CatalogTests(unittest.TestCase):
    def test_measured_catalogue(self):
        leaves = catalog.gemma4_31b()
        self.assertEqual(len(leaves), 1028)
        by_kind = {l.fused for l in leaves}
        self.assertEqual(by_kind, {None, "qkv", "gate_up"})
        for leaf in leaves:
            if leaf.fused:
                self.assertEqual(leaf.tp_axis, len(leaf.shape) - 1, leaf.path)
        bytes_gb = catalog.nbytes(leaves) / 1e9
        self.assertGreater(bytes_gb, 55)
        self.assertLess(bytes_gb, 70)

    def test_scaled_divides_every_tp_axis(self):
        leaves = catalog.scaled()
        self.assertTrue(any("['layers'][5]" in l.path and "q_proj" in l.path for l in leaves))
        self.assertFalse(any("['layers'][6]" in l.path for l in leaves))
        for leaf in leaves:
            if leaf.tp_axis is not None:
                self.assertEqual(leaf.shape[leaf.tp_axis] % 16, 0, leaf.path)
            for size in leaf.fused_sizes:
                self.assertEqual(size % 16, 0, leaf.path)
            if leaf.fused:
                self.assertEqual(sum(leaf.fused_sizes), leaf.shape[-1])
        self.assertLess(catalog.nbytes(leaves), 64 << 20)

    def test_fused_layout_matches_ici_resident_repack(self):
        leaf = next(l for l in catalog.scaled() if l.fused == "qkv")
        logical = catalog.synthetic_value(leaf, seed=7)
        tp8 = catalog.to_stored(logical, leaf, 8)
        tp16 = catalog.to_stored(logical, leaf, 16)
        self.assertFalse(np.array_equal(tp8, tp16))
        np.testing.assert_array_equal(catalog.to_logical(tp8, leaf, 8), logical)
        np.testing.assert_array_equal(
            np.asarray(repack_merged(tp8, leaf.fused_sizes, 8, 16)), tp16)
        self.assertEqual(catalog.logical_checksum(catalog.to_logical(tp16, leaf, 16)),
                         catalog.logical_checksum(logical))

    def test_synthetic_values_deterministic(self):
        leaf = catalog.scaled()[0]
        a, b = catalog.synthetic_value(leaf, 1), catalog.synthetic_value(leaf, 1)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, catalog.synthetic_value(leaf, 2)))
        self.assertLessEqual(np.abs(a).max(), 2)


if __name__ == "__main__":
    unittest.main()
