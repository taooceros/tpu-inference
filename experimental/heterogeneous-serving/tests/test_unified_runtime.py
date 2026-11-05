"""Unified-runtime harness (plan U1/U2): topology, exact values, verdicts, CPU fleet.

The fleet tests run the real protocols in one jax.distributed runtime of four
local processes x four CPU devices (gloo collectives): each half's programs are
launched only by its own two processes, and the switch reshards across all four.
"""
import copy
import dataclasses
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from hetero.unified_runtime import (
    PROTOCOLS, SALT_UP, CpuFleet, Shape, UnifiedConfig, checksum_jnp, checksum_np, forward_fn,
    inputs_np, reference_forward, split_halves, switch_segments, target_devices, verdict_u1,
    verdict_u2, weights_jnp, weights_np)

TINY = dict(batch=8, small_layers=2, small_d_model=64, small_d_ff=128, compile_layers=16,
            compile_d_model=32, compile_d_ff=64, model_layers=4, model_d_model=64, model_d_ff=256,
            control_s=0.4, stall_s=1.5, batch_target_s=0.02, min_weight_bytes_per_chip=1,
            min_native_compile_s=0.0, min_contained_steps=1,
            # Four CPU processes share one host (and a blocked gloo partner may spin), so
            # CPU rates are noisy; the TPU floor stays at the default 0.8.
            stall_rate_floor=0.3)


def tpu_block():
    """The 4x4 v5e block as JAX reports it (ids and coords from the CompileExecutionSmoke runs)."""
    devices = []
    for y in range(4):
        for x in range(4):
            host = (y // 2) * 2 + (x // 2)
            devices.append(SimpleNamespace(id=y * 4 + x, coords=(x, y, 0), process_index=host,
                                           platform="tpu"))
    return devices


class TopologyTest(unittest.TestCase):
    def test_x_halves_are_the_s_layout_tp8_boxes_and_y_halves_are_host_rows(self):
        lower, upper = split_halves(tpu_block(), "x")
        self.assertEqual([d.id for d in lower], [0, 1, 4, 5, 8, 9, 12, 13])
        self.assertEqual([d.id for d in upper], [2, 3, 6, 7, 10, 11, 14, 15])
        self.assertEqual({d.process_index for d in lower}, {0, 2})
        self.assertEqual({d.process_index for d in upper}, {1, 3})
        lower, upper = split_halves(tpu_block())
        self.assertEqual([d.id for d in lower], list(range(8)))
        self.assertEqual([d.id for d in upper], list(range(8, 16)))
        self.assertEqual({d.process_index for d in lower}, {0, 1})
        self.assertEqual({d.process_index for d in upper}, {2, 3})

    def test_local_target_order_keeps_every_tp16_shard_on_its_tp8_chip(self):
        lower, upper = split_halves(tpu_block())
        order = target_devices(lower, upper, "local")
        self.assertEqual([d.id for d in order][:4], [0, 8, 1, 9])
        cols = 12288
        for j, device in enumerate(order):
            half = lower if device in lower else upper
            k = half.index(device)
            tp16 = range(j * cols // 16, (j + 1) * cols // 16)
            tp8 = range(k * cols // 8, (k + 1) * cols // 8)
            self.assertTrue(tp16.start >= tp8.start and tp16.stop <= tp8.stop)
        natural = target_devices(lower, upper, "natural")
        self.assertEqual([d.id for d in natural], list(range(16)))

    def test_process_spanning_halves_is_refused(self):
        devices = tpu_block()
        devices[2].process_index = 0  # chip x=2 on rank 0's host
        with self.assertRaises(ValueError):
            split_halves(devices, "x")
        devices = tpu_block()
        devices[8].process_index = 0  # chip y=2 on rank 0's host
        with self.assertRaises(ValueError):
            split_halves(devices, "y")

    def test_config_refuses_natural_order_with_aot(self):
        with self.assertRaises(ValueError):
            UnifiedConfig(target_order="natural", reshard="aot").validate()
        UnifiedConfig().validate()
        self.assertGreaterEqual(UnifiedConfig().model().weight_bytes_per_chip(8), 1_000_000_000)


class ValuesTest(unittest.TestCase):
    def test_device_weights_equal_host_and_checksum_is_position_sensitive(self):
        import jax
        import numpy as np
        shape = Shape(3, 32, 64, 4)
        up_shape, _ = shape.weight_shapes()
        device = np.asarray(jax.jit(lambda: weights_jnp(up_shape, SALT_UP, shape.d_model))())
        host = np.stack([weights_np(up_shape, SALT_UP, shape.d_model, l) for l in range(3)])
        self.assertTrue(np.array_equal(device.view(np.uint16), host.view(np.uint16)))
        self.assertEqual(int(jax.jit(checksum_jnp)(device)), checksum_np(host))
        swapped = host.copy()
        swapped[:, :, :8], swapped[:, :, 8:16] = host[:, :, 8:16], host[:, :, :8]
        self.assertNotEqual(checksum_np(swapped), checksum_np(host))

    def test_device_forward_matches_host_reference(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        shape = Shape(8, 256, 512, 8)
        up_shape, down_shape = shape.weight_shapes()
        w_up = jax.jit(lambda: weights_jnp(up_shape, SALT_UP, shape.d_model))()
        w_down = jax.jit(lambda: weights_jnp(down_shape, 23, shape.d_ff))()
        y = jax.jit(forward_fn(shape.layers))(jnp.asarray(inputs_np(shape)), w_up, w_down)
        self.assertLess(float(np.max(np.abs(np.asarray(y) - reference_forward(shape)))), 0.01)


def switch_record(rank, direction, last, enter, ready, view, first, compiles=0):
    from hetero.unified_runtime import UnifiedWorker

    def at(ns):
        return {"epoch_ns": ns, "mono_ns": ns}
    times = {"last_source_step_end": at(last), "enter": at(enter), "reshard_dispatched": at(enter),
             "reshard_ready": at(ready), "view": at(view), "first_dispatched": at(first),
             "first_ready": at(first)}
    return {"rank": rank, "direction": direction, "times": times,
            "local_s": UnifiedWorker._local_segments(times),
            "backend_compiles_during_switch": compiles, "backend_compile_s_during_switch": 0.0}


class AnalysisTest(unittest.TestCase):
    def test_switch_segments_sum_and_use_the_slowest_rank(self):
        records = {0: switch_record(0, "forward", 100, 150, 300, 120, 700),
                   1: switch_record(1, "forward", 1_000, 1_100, 1_500, 1_050, 2_000)}
        seg = switch_segments(records)
        self.assertEqual(seg["switch_s"], 1e-6)
        self.assertEqual(seg["stop_s"], 1e-7)
        self.assertEqual(seg["reshard_s"], 4e-7)
        self.assertEqual(seg["first_exec_s"], 5e-7)
        self.assertAlmostEqual(seg["source_last_step_skew_s"], 9e-7)
        self.assertEqual(seg["local_max"]["switch"], 1e-6)
        self.assertEqual(seg["local_min"]["switch"], 6e-7)
        # Driver-clock correction: rank 1's clock runs 400 ns ahead.
        shifted = switch_segments(records, {0: {"offset_ns": 0}, 1: {"offset_ns": 400}})
        self.assertAlmostEqual(shifted["switch_s"], 1e-6)
        self.assertAlmostEqual(shifted["source_last_step_skew_s"], 5e-7)
        # reverse: the split into half views ends the reshard segment
        reverse = switch_segments({0: switch_record(0, "reverse", 0, 10, 20, 40, 50)})
        self.assertEqual(reverse["reshard_s"], 3e-8)
        self.assertEqual(reverse["local_max"]["reshard"], 3e-8)

    def test_verdict_u2_rejects_a_moved_element_and_a_changed_reverse_output(self):
        config = UnifiedConfig(**TINY, experiment="u2")
        progress = run_fleet("u2", {})
        self.assertEqual(verdict_u2(config, progress)["status"], "passed")
        broken = copy.deepcopy(progress)
        broken["check_rounds"]["target_checksums"][0]["w_up"] ^= 1
        self.assertIn("weight checksums differ", " ".join(verdict_u2(config, broken)["failures"]))
        broken = copy.deepcopy(progress)
        broken["switch_reverse"][3]["equals_pre_switch_half_output"] = False
        self.assertIn("reverse switch", " ".join(verdict_u2(config, broken)["failures"]))
        broken = copy.deepcopy(progress)
        broken["check_rounds"]["setup"]["cross_half_max_abs_diff"] = 1.0
        self.assertIn("halves differ", " ".join(verdict_u2(config, broken)["failures"]))
        short = dataclasses.replace(config, min_native_compile_s=1e6)
        self.assertEqual(verdict_u2(short, progress)["status"], "inconclusive_overlap_too_short")

    def test_verdict_u2_fails_a_compile_on_the_switch_when_it_was_prepared(self):
        cold = run_fleet("u2", {"warm_device_put": False})
        warm_config = UnifiedConfig(**TINY, experiment="u2")
        failures = " ".join(verdict_u2(warm_config, cold)["failures"])
        self.assertIn("backend compiles on the critical path", failures)

    def test_verdict_u1_rejects_an_unblocked_stall_and_a_slow_active_half(self):
        config = UnifiedConfig(**TINY, experiment="u1")
        progress = run_fleet("u1", {})
        self.assertEqual(verdict_u1(config, progress)["status"], "passed", verdict_u1(config, progress))
        broken = copy.deepcopy(progress)
        for report in broken["workers"]:
            phase = report["phases"].get("small/b2_upper_stalled")
            if phase:
                phase["max_step_s"] = 0.01
            phase = report["phases"].get("small/b2_lower_active_upper_stalled")
            if phase:
                phase["steps_per_wall_s"] /= 10
        failures = " ".join(verdict_u1(config, broken)["failures"])
        self.assertIn("partner blocked only", failures)
        self.assertIn("lower rate ratio", failures)


_FLEET_CACHE = {}


def run_fleet(experiment, overrides):
    key = (experiment, tuple(sorted(overrides.items())))
    if key not in _FLEET_CACHE:
        config = UnifiedConfig(**{**TINY, **overrides, "experiment": experiment})
        fleet = CpuFleet(config)
        try:
            progress = {}
            result = PROTOCOLS[experiment](config, fleet, lambda *_, **__: None, progress)
            progress["result"] = result
        finally:
            fleet.close()
        _FLEET_CACHE[key] = progress
    return _FLEET_CACHE[key]


class CpuFleetTest(unittest.TestCase):
    def test_u1_halves_run_independently_and_stall_blocks_only_its_half(self):
        result = run_fleet("u1", {})["result"]
        self.assertEqual(result["status"], "passed", result["failures"])
        self.assertIn("e_target_compile_impact", result)
        self.assertTrue(all(c["ok"] for v in result["compiles"].values() for c in v.values()))
        for stalled in ("upper", "lower"):
            b2 = result["b2"][stalled]
            self.assertGreaterEqual(b2["stalled_partner_max_step_s"], 1.35)
            self.assertLess(b2["active_max_step_s"], 1.0)
            self.assertEqual(b2["stalled_steps"], [5, 5])

    def test_u2_device_put_switch_both_directions_with_no_compile_on_the_switch(self):
        progress = run_fleet("u2", {})
        result = progress["result"]
        self.assertEqual(result["status"], "passed", result["failures"])
        for direction in ("forward", "reverse"):
            self.assertEqual(result[direction]["backend_compiles_during_switch"], 0)
            seg = result[direction]
            self.assertAlmostEqual(seg["switch_s"], seg["stop_s"] + seg["reshard_s"] + seg["first_exec_s"])
        for impact in result["serving_impact"].values():
            self.assertTrue(all(n > 0 for n in impact["native_contained_steps"]))
            # 6 target buckets + 4 warm device_put programs, one native interval each
            self.assertTrue(all(i == 10 for i in impact["native_intervals"]), impact["native_intervals"])
        setup = progress["setup"]
        self.assertEqual(len({v["output_sha256"] for v in setup.values()}), 1)

    def test_u2_without_warm_compiles_on_the_switch_and_aot_does_not(self):
        cold = run_fleet("u2", {"warm_device_put": False})["result"]
        self.assertEqual(cold["status"], "passed", cold["failures"])
        self.assertGreater(cold["forward"]["backend_compiles_during_switch"], 0)
        aot = run_fleet("u2", {"reshard": "aot"})["result"]
        self.assertEqual(aot["status"], "passed", aot["failures"])
        self.assertEqual(aot["forward"]["backend_compiles_during_switch"], 0)
        natural = run_fleet("u2", {"target_order": "natural"})["result"]
        self.assertEqual(natural["status"], "passed", natural["failures"])

    def test_u1d_round_trips_each_half_executable(self):
        result = run_fleet("u1d", {})["result"]
        self.assertEqual(result["status"], "passed")
        self.assertEqual(sorted(result["probes"]), ["lower", "upper"])


if __name__ == "__main__":
    unittest.main()
