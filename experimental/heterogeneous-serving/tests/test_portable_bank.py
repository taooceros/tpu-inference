"""Portable executable bank tooling: placement congruence, relocation bytes, cross-host comparison.

No Ray, TPU or Kubernetes: the synthetic inventory is the recorded v5e slice geometry (as in
test_staged_transition.py) and banks are hand-made manifests on temporary directories.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bank_compare  # noqa: E402
import bank_relocate  # noqa: E402
from hetero import exec_bank as eb  # noqa: E402
from tests.test_staged_transition import inventory  # noqa: E402


def fake_bank(root, digest, rank, payloads, host_order):
    path = Path(root) / digest / f"rank{rank}"
    path.mkdir(parents=True)
    entries = {}
    for i, data in enumerate(payloads):
        name = f"fn-{i}.bin"
        (path / name).write_bytes(data)
        entries[name] = {"fn": "fn", "key": f"k{i}", "sha256": hashlib.sha256(data).hexdigest(),
                         "bytes": len(data)}
    manifest = {"format": eb.FORMAT, "identity": {"instance": {"x": 1}, "rank": {"process_index": rank}},
                "identity_digest": digest, "placement": {"host_order": host_order}, "entries": entries,
                "entry_set_digest": eb.entry_set_digest(e["key"] for e in entries.values())}
    (path / eb.MANIFEST).write_text(json.dumps(manifest))


class CongruenceTests(unittest.TestCase):
    def test_tp16_blocks_pair_rank_for_rank(self):
        pairs = bank_relocate.congruent_pairs(inventory(), "H:large_a", "H:large_b")
        self.assertEqual([p["rank"] for p in pairs], [0, 1, 2, 3])
        self.assertEqual([p["relative_host_position"] for p in pairs],
                         [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.assertTrue(all(p["source_host"] != p["target_host"] for p in pairs))
        # Mb's joiner sits exactly on H's large_b hosts.
        self.assertEqual(bank_relocate.congruent_pairs(inventory(), "H:large_a", "Mb:large_b"), pairs)

    def test_tp8_pairs_and_shape_mismatch(self):
        self.assertEqual(len(bank_relocate.congruent_pairs(inventory(), "S:small_a", "S:small_d")), 2)
        with self.assertRaises(SystemExit):
            bank_relocate.congruent_pairs(inventory(), "H:large_a", "S:small_a")
        with self.assertRaises(SystemExit):  # same placement: nothing to relocate
            bank_relocate.congruent_pairs(inventory(), "H:large_b", "Mb:large_b")


class RelocationBytesTests(unittest.TestCase):
    def test_relocated_copy_is_byte_identical_and_recorded(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            fake_bank(a, "d" * 64, 1, [b"one", b"two"], ["10.0.0.1", "10.0.0.6"])
            bundle = bank_relocate.read_bundle(a, 1, None)
            report = bank_relocate.install_bundle(b, bundle, {"source_host": "10.0.0.6"})
            self.assertTrue(report["verified_after_install"])
            src, dst = eb.bank_inventory(a), eb.bank_inventory(b)
            self.assertEqual(src[0]["manifest_sha256"], dst[0]["manifest_sha256"])
            self.assertEqual(dst[0]["relocation"]["source_host"], "10.0.0.6")
            self.assertEqual(dst[0]["placement"], {"host_order": ["10.0.0.1", "10.0.0.6"]})
            hosts = [{"node_ip": "10.0.0.6", "banks": src}, {"node_ip": "10.0.0.7", "banks": dst}]
            (group,) = bank_compare.compare(hosts)
            self.assertTrue(group["identical"])
            self.assertEqual(group["banks"][1]["relocated_from"], "10.0.0.6")

    def test_read_bundle_needs_exactly_one_bank_per_rank(self):
        with tempfile.TemporaryDirectory() as a:
            fake_bank(a, "d" * 64, 0, [b"x"], ["h"])
            fake_bank(a, "e" * 64, 0, [b"y"], ["h"])
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                bank_relocate.read_bundle(a, 0, None)
            self.assertEqual(bank_relocate.read_bundle(a, 0, "e" * 64)["files"], {"fn-0.bin": b"y"})

    def test_compare_reports_differing_bytes(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            fake_bank(a, "d" * 64, 0, [b"same", b"left"], ["10.0.0.1"])
            fake_bank(b, "d" * 64, 0, [b"same", b"right"], ["10.0.0.7"])
            (group,) = bank_compare.compare([{"node_ip": "1", "banks": eb.bank_inventory(a)},
                                             {"node_ip": "7", "banks": eb.bank_inventory(b)}])
            self.assertEqual((group["entries_identical_across_hosts"], group["entries"]), (1, 2))
            self.assertFalse(group["identical"])
            self.assertEqual(group["placements"], 2)


if __name__ == "__main__":
    unittest.main()
