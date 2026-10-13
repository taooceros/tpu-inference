"""Standalone JAX compile-only/execution overlap; no vLLM or global cache hooks.

The target uses a subset of the *attached* topology. This does not qualify
portability to a fresh, smaller runtime. Only rank zero's local monotonic clock
is used to establish fully contained execution during its native compile call.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import os
import socket
import statistics
import threading
import time
import traceback


@dataclasses.dataclass(frozen=True)
class Config:
    depth: int = 4096
    batch: int = 128
    width: int = 256
    source_depth: int = 4
    step_batch: int = 16
    control_s: float = 3.0
    compile_timeout_s: float = 300.0
    min_compile_s: float = 5.0
    min_contained_steps: int = 20
    target_half: str = "lower"

    def validate(self):
        if self.target_half not in ("lower", "upper"):
            raise ValueError("target_half must be lower or upper")
        if not (1 <= self.depth <= 8192 and 1 <= self.source_depth <= 16):
            raise ValueError("depth must be 1..8192; source_depth must be 1..16")
        if not (1 <= self.batch <= 1024 and 1 <= self.width <= 2048):
            raise ValueError("batch must be 1..1024; width must be 1..2048")
        if not (1 <= self.step_batch <= 256 and self.control_s > 0
                and self.compile_timeout_s > 0 and self.min_compile_s >= 0
                and self.min_contained_steps >= 1):
            raise ValueError("invalid measurement bounds")


def program(depth):
    import jax
    import jax.numpy as jnp
    import numpy as np

    def evaluate(x, weight):
        anchor = x
        # Intentionally unrolled. Runtime weight input prevents constant folding.
        for i in range(depth):
            x = jnp.tanh(np.float32(.5) * jnp.matmul(
                x, weight, precision=jax.lax.Precision.HIGHEST)
                + np.float32(.25) * anchor + np.float32((i % 7 - 3) * .01))
        return x
    return evaluate


def inputs_and_reference(config, depth):
    import numpy as np
    x = np.linspace(-.2, .2, config.batch * config.width, dtype=np.float32).reshape(
        config.batch, config.width)
    weight = np.roll(np.eye(config.width, dtype=np.float32), 1, axis=1)
    result = x.copy()
    # Independent reference exploits this runtime input, not a compiler shortcut.
    for i in range(depth):
        result = np.tanh(np.float32(.5) * np.roll(result, 1, axis=-1)
                         + np.float32(.25) * x + np.float32((i % 7 - 3) * .01))
    return x, weight, result


def device_record(device):
    return {"id": device.id, "process_index": device.process_index,
            "platform": device.platform, "device_kind": device.device_kind,
            "coords": list(getattr(device, "coords", ()) or ()),
            "core_on_chip": getattr(device, "core_on_chip", None)}


def select_target(devices, half):
    if len(devices) < 2 or len(devices) % 2:
        raise ValueError("need an even source device count >= 2")
    if devices[0].platform == "tpu":
        xs = sorted({d.coords[0] for d in devices})
        if len(xs) % 2:
            raise ValueError("cannot halve physical x dimension")
        selected_xs = xs[:len(xs) // 2] if half == "lower" else xs[len(xs) // 2:]
        target = [d for d in devices if d.coords[0] in selected_xs]
    else:
        target = devices[:len(devices) // 2] if half == "lower" else devices[len(devices) // 2:]
    if len(target) != len(devices) // 2:
        raise ValueError("target is not an exact half-submesh")
    return target


def sample_summary(samples):
    if not samples:
        return {"steps": 0}
    durations = sorted((end - start) / 1e9 for start, end in samples)
    wall = (samples[-1][1] - samples[0][0]) / 1e9
    active = sum(durations)
    return {"steps": len(samples), "first_start_ns": samples[0][0],
            "last_end_ns": samples[-1][1], "wall_s": wall,
            "active_step_s": active, "between_step_s": max(0., wall - active),
            "steps_per_wall_s": len(samples) / wall if wall else None,
            "p50_step_s": statistics.median(durations),
            "p95_step_s": durations[min(len(durations) - 1, int(.95 * len(durations)))]}


class Session:
    """Lives in one initialized JAX process; RPCs keep dispatching during compile."""

    def __init__(self, config: Config, platform: str):
        import jax
        import numpy as np
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        config.validate()
        self.config = config
        self.rank = jax.process_index()
        self.devices = list(jax.devices())
        if any(d.platform != platform for d in self.devices):
            raise ValueError(f"expected only {platform} devices")
        if config.batch % len(self.devices):
            raise ValueError("batch must divide evenly across source devices")
        self.target = select_target(self.devices, config.target_half)
        # These workers are new; never change another serving process's cache.
        jax.config.update("jax_enable_compilation_cache", False)
        self.samples = {p: [] for p in ("baseline", "overlap", "postcontrol")}
        self.compile_done = threading.Event()
        self.compile_result = None
        self.target_plan = None
        self.source_checks = []
        self.target_check = None
        self.target_info = None
        mesh = Mesh(np.array(self.devices), ("data",))
        self.source_shardings = (NamedSharding(mesh, P("data", None)),
                                 NamedSharding(mesh, P()))
        x, weight, self.source_reference = inputs_and_reference(config, config.source_depth)
        self.source_inputs = tuple(jax.make_array_from_callback(
            value.shape, sharding, lambda index, value=value: value[index])
            for value, sharding in zip((x, weight), self.source_shardings))
        started = time.monotonic()
        self.source = jax.jit(program(config.source_depth),
            in_shardings=self.source_shardings,
            out_shardings=self.source_shardings[0]).lower(*self.source_inputs).compile()
        for _ in range(3):
            self.last_source = self.source(*self.source_inputs)
            self.last_source.block_until_ready()
        self.source_prepare_s = time.monotonic() - started
        self.validate_source()
        versions = {"jax": jax.__version__}
        for package in ("jaxlib", "libtpu"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.identity = {"rank": self.rank, "pid": os.getpid(), "hostname": socket.gethostname(),
                         "versions": versions, "devices": [device_record(d) for d in self.devices],
                         "target_devices": [device_record(d) for d in self.target],
                         "cpu_count": os.cpu_count(),
                         "cpu_affinity": sorted(os.sched_getaffinity(0)),
                         "cache_enabled": bool(jax.config.jax_enable_compilation_cache)}

    def prepare_target(self):
        import jax
        import numpy as np
        from jax._src import compiler
        from jax._src.lib import xla_client as xc
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        if self.rank != 0:
            raise ValueError("only rank zero prepares/compiles the target")
        topology = xc.get_topology_for_devices(self.devices)
        by_id = {d.id: d for d in topology._make_compile_only_devices()}
        virtual = [by_id[d.id] for d in self.target]
        client = virtual[0].client
        if type(client).__name__ != "CompileOnlyPyClient":
            raise TypeError(f"not a compile-only client: {type(client)}")
        mesh = Mesh(np.array(virtual), ("data",))
        sx, sw = NamedSharding(mesh, P("data", None)), NamedSharding(mesh, P())
        c = self.config
        function = jax.jit(program(c.depth), in_shardings=(sx, sw), out_shardings=sx)
        start = time.monotonic()
        traced = function.trace(jax.ShapeDtypeStruct((c.batch, c.width), np.float32),
                                jax.ShapeDtypeStruct((c.width, c.width), np.float32))
        traced_at = time.monotonic()
        lowered = traced.lower()
        lowered_at = time.monotonic()
        module = lowered.compiler_ir()
        options = compiler.get_compile_options(num_replicas=1, num_partitions=len(virtual),
            device_assignment=np.array([[d.id for d in virtual]]), backend=client)
        options.executable_build_options.use_spmd_partitioning = True
        options.executable_build_options.use_shardy_partitioner = True
        mlir = str(module).encode()
        dot_count = mlir.count(b"stablehlo.dot_general")
        if dot_count != c.depth:
            raise ValueError(f"expected {c.depth} dependent dots, found {dot_count}")
        self.target_plan = (client, module, xc.DeviceList(tuple(virtual)), options)
        self.target_info = {"client_type": type(client).__name__,
            "topology": str(topology), "trace_s": traced_at - start,
            "lower_s": lowered_at - traced_at, "mlir_bytes": len(mlir),
            "mlir_sha256": hashlib.sha256(mlir).hexdigest(), "dot_general_count": dot_count,
            "compile_options_sha256": hashlib.sha256(options.SerializeAsString()).hexdigest(),
            "target_ids": [d.id for d in virtual],
            "preparation_overlapped": False,
            "call": "CompileOnlyPyClient.compile(module, executable_devices, compile_options)"}
        return self.target_info

    def start_compile(self):
        if self.target_plan is None or self.compile_result is not None or hasattr(self, "thread"):
            raise RuntimeError("target missing or compilation already started")

        def work():
            client, module, devices, options = self.target_plan
            record = {"start_ns": time.monotonic_ns(), "start_epoch_ns": time.time_ns()}
            process_start, thread_start = time.process_time(), time.thread_time()
            try:
                # Deliberately NOT lowered.compile(): bypass live-client rerouting.
                self.unloaded = client.compile(module, executable_devices=devices,
                                               compile_options=options)
                record["end_ns"] = time.monotonic_ns()
                record["executable_type"] = type(self.unloaded).__name__
                if record["executable_type"] != "Executable":
                    raise TypeError("expected an unloaded executable")
                record["ok"] = True
            except Exception:
                record.setdefault("end_ns", time.monotonic_ns())
                record.update(ok=False, traceback=traceback.format_exc())
            finally:
                record["call_s"] = (record["end_ns"] - record["start_ns"]) / 1e9
                record["whole_process_cpu_s_including_serving"] = time.process_time() - process_start
                record["calling_thread_cpu_s_excluding_native_pool"] = time.thread_time() - thread_start
                self.compile_result = record
                self.compile_done.set()

        self.thread = threading.Thread(target=work, name="target-compile-only", daemon=True)
        self.thread.start()
        return {"started": True}

    def compile_status(self):
        return {"done": self.compile_done.is_set(), "result": self.compile_result}

    def steps(self, phase, count):
        if phase not in self.samples:
            raise ValueError(phase)
        for _ in range(count):
            start = time.monotonic_ns()
            self.last_source = self.source(*self.source_inputs)
            self.last_source.block_until_ready()
            self.samples[phase].append([start, time.monotonic_ns()])
        return {"rank": self.rank, "phase": phase, "completed": count}

    def validate_source(self):
        import numpy as np
        errors = []
        for shard in self.last_source.addressable_shards:
            actual, expected = np.asarray(shard.data), self.source_reference[shard.index]
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            errors.append(float(np.max(np.abs(actual - expected))))
        check = {"local_shards": len(errors), "max_abs_error": max(errors), "ok": True}
        self.source_checks.append(check)
        return check

    def export_target(self):
        if not self.compile_done.is_set() or not self.compile_result["ok"]:
            raise RuntimeError("no successful completed native compilation")
        start = time.monotonic()
        blob = self.unloaded.serialize()
        options = self.target_plan[3].SerializeAsString()
        return {"blob": blob, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                "serialization_s": time.monotonic() - start, "compile_options": options,
                "compile_options_sha256": hashlib.sha256(options).hexdigest()}

    def load_and_check_target(self, blob, serialized_options):
        import jax
        import numpy as np
        from jax._src.lib import xla_client as xc
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        if self.rank not in {d.process_index for d in self.target}:
            self.target_check = {"rank": self.rank, "participates": False}
            return self.target_check
        start = time.monotonic()
        # executable_devices alone only configures the IFRT wrapper. Supplying
        # options lets the real client translate IFRT global IDs to native PJRT IDs.
        options = xc.CompileOptions.ParseFromString(serialized_options)
        loaded = self.devices[0].client.deserialize_executable(
            blob, executable_devices=xc.DeviceList(tuple(self.target)), compile_options=options)
        loaded_at = time.monotonic()
        x, weight, expected = inputs_and_reference(self.config, self.config.depth)
        mesh = Mesh(np.array(self.target), ("data",))
        sx, sw = NamedSharding(mesh, P("data", None)), NamedSharding(mesh, P())
        inputs = [jax.make_array_from_callback(value.shape, sharding,
                    lambda index, value=value: value[index])
                  for value, sharding in zip((x, weight), (sx, sw))]
        outputs = loaded.execute_sharded(inputs).disassemble_into_single_device_arrays()
        local_devices = list(loaded.local_devices())
        if len(outputs) != 1 or len(outputs[0]) != len(local_devices) or not local_devices:
            raise ValueError("unexpected target result shard count")
        indexes = sx.devices_indices_map(expected.shape)
        errors = []
        for device, array in zip(local_devices, outputs[0]):
            actual, reference = np.asarray(array), expected[indexes[device]]
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
            errors.append(float(np.max(np.abs(actual - reference))))
        self.target_check = {"rank": self.rank, "participates": True, "ok": True,
            "load_s": loaded_at - start, "reference_input_execute_check_s": time.monotonic() - loaded_at,
            "local_device_ids": [d.id for d in local_devices], "max_abs_error": max(errors),
            "loaded_type": type(loaded).__name__}
        return self.target_check

    def report(self):
        result = {"identity": self.identity, "source_prepare_s": self.source_prepare_s,
            "source_checks": self.source_checks, "target_check": self.target_check,
            "phases": {p: sample_summary(s) for p, s in self.samples.items()}}
        if self.rank == 0:
            result.update(target_plan=self.target_info, native_compile=self.compile_result,
                          rank0_samples_ns=self.samples)
            if self.compile_result:
                start, end = self.compile_result["start_ns"], self.compile_result["end_ns"]
                contained = [s for s in self.samples["overlap"] if start <= s[0] and s[1] <= end]
                result["contained_source_steps"] = sample_summary(contained)
        return result


def verdict(config, reports):
    rank0 = next(r for r in reports if r["identity"]["rank"] == 0)
    native = rank0.get("native_compile") or {}
    participants = [r["target_check"] for r in reports
                    if (r.get("target_check") or {}).get("participates")]
    expected_target_ids = sorted(d["id"] for d in rank0["identity"]["target_devices"])
    executed_target_ids = sorted(d for p in participants for d in p.get("local_device_ids", []))
    expected_ranks = {d["process_index"] for d in rank0["identity"]["devices"]}
    complete_coverage = (executed_target_ids == expected_target_ids
                         and {r["identity"]["rank"] for r in reports} == expected_ranks
                         and len(reports) == len(expected_ranks))
    correct = complete_coverage and bool(participants) and all(p.get("ok") for p in participants)
    correct = correct and all(len(r["source_checks"]) == 2
                              and all(c["ok"] for c in r["source_checks"]) for r in reports)
    steps = rank0.get("contained_source_steps", {}).get("steps", 0)
    enough_work = native.get("call_s", 0) >= config.min_compile_s
    enough_overlap = steps >= config.min_contained_steps
    if not native.get("ok") or not correct:
        status = "failed"
    elif not enough_work:
        status = "inconclusive_compile_too_short"
    elif not enough_overlap:
        status = "failed_no_sufficient_completed_overlap"
    else:
        status = "passed"
    return {"status": status, "correct": correct, "native_compile_s": native.get("call_s"),
            "fully_contained_rank0_steps": steps, "minimum_compile_s_met": enough_work,
            "minimum_contained_steps_met": enough_overlap}
