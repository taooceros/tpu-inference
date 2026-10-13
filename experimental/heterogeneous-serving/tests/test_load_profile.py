"""Accounting boundaries for the opt-in load profiler, without TPU dependencies."""
import unittest
from types import SimpleNamespace
from unittest import mock

from hetero.load_profile import LoadProfile


class LoadProfileAccounting(unittest.TestCase):
    def test_nested_categories_partition_wall_time(self):
        profile = LoadProfile()
        with mock.patch("hetero.load_profile.time.perf_counter", side_effect=[0, 2, 5, 7]):
            with profile.span("host_tensor_ops", "outer"):
                with profile.span("host_to_device", "transfer", byte_count=120):
                    pass
        self.assertEqual(profile.totals["host_tensor_ops"], 4)
        self.assertEqual(profile.totals["host_to_device"], 3)
        self.assertEqual(sum(profile.totals.values()), 7)
        self.assertEqual(profile.calls["transfer"], [3, 1, 120])

    def test_iterator_excludes_consumer_processing_and_clears_stack_on_exhaustion(self):
        profile = LoadProfile()
        def source():
            yield "weights.a", 1
            yield "weights.b", 2
        iterator = profile.iterator(source)()
        with mock.patch("hetero.load_profile.time.perf_counter", side_effect=[0, 1, 100, 102, 200, 201]):
            self.assertEqual(next(iterator), ("weights.a", 1))
            self.assertEqual(profile.tensor, "weights.a")
            self.assertEqual(next(iterator), ("weights.b", 2))
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assertEqual(profile.totals["file_deserialize"], 4)
        self.assertEqual(profile.stack, [])

    def test_restore_unwinds_multiple_patches_even_after_error(self):
        profile = LoadProfile()
        def fail():
            raise ValueError("failed load")
        owner = SimpleNamespace(call=fail)
        profile.patch(owner, "call", profile.wrap("post_load", "call"))
        profile.patch(owner, "call", profile.wrap("host_tensor_ops", "wrapper"))
        try:
            with self.assertRaisesRegex(ValueError, "failed load"):
                owner.call()
        finally:
            profile._restore()
        self.assertIs(owner.call, fail)
        self.assertEqual(profile.stack, [])
        self.assertEqual(profile.counts["post_load"], 1)

    def test_top_calls_retain_slowest_boundaries(self):
        profile = LoadProfile()
        times = [value for i in range(25) for value in (100 * i, 100 * i + i + 1)]
        with mock.patch("hetero.load_profile.time.perf_counter", side_effect=times):
            for i in range(25):
                with profile.span("host_tensor_ops", f"tensor_{i}"):
                    pass
        evidence = profile.evidence()
        self.assertEqual([row["wall_s"] for row in evidence["top20"]], list(range(25, 5, -1)))
        self.assertEqual(evidence["category_calls"]["host_tensor_ops"], 25)
