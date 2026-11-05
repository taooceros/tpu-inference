"""Unified 16-chip JAX runtime with two in-process 8-chip submeshes (plan U1/U2).

One runtime spans a 4-host block (4x4 v5e chips). Its two physical x-halves are
exactly the S-layout TP8 instances (2 hosts each), so in a 4-process runtime the
lower half is driven by processes {0, 2} and the upper half by {1, 3}.

Ordering contract. Every device execution in a process happens on that
process's single command thread, in the order the driver submitted commands.
The driver submits identical command sequences to the two processes of a half
(and to all four processes for 16-chip work), so every multi-process program is
launched in the same order on every participant, and the halves never share a
barrier. Compilation runs on a separate thread and never executes on devices.

The model program is TP-sharded like a transformer MLP: per layer, a
column-sharded up projection, then a row-sharded down projection whose partial
sums are all-reduced. Weights come from an integer hash of the global index, so
they are identical in both halves and on the host, and an exact
position-weighted checksum (wrapping uint32 sum of bf16 bit patterns) is
independent of the sharding. A reshard that moves or loses any element changes
the checksum.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import importlib.metadata
import math
import os
import queue
import socket
import statistics
import threading
import time
import traceback

BACKEND_COMPILE_EVENT = "/jax/core/compile/backend_compile_duration"
HALVES = ("lower", "upper")
EXPERIMENTS = ("u1", "u2", "u1d")
SALT_UP, SALT_DOWN, SALT_X, SALT_CHECK = 11, 23, 37, 53
# Residual scale 0.25 keeps the unrolled tanh stack contractive, so f32
# summation-order differences (TP8 vs TP16, device vs host) stay ~1e-2 or less
# instead of growing chaotically with depth (at 1.0: 0.28 max error at 64 layers).
RESIDUAL_SCALE = 0.25
_M1, _M2, _M3, _M4 = 0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F


@dataclasses.dataclass(frozen=True)
class Shape:
    layers: int
    d_model: int
    d_ff: int
    batch: int

    def weight_shapes(self):
        return ((self.layers, self.d_model, self.d_ff), (self.layers, self.d_ff, self.d_model))

    def weight_bytes_per_chip(self, tp):
        return 2 * 2 * self.layers * self.d_model * self.d_ff // tp


@dataclasses.dataclass(frozen=True)
class UnifiedConfig:
    experiment: str = "u1"
    batch: int = 32
    # U1 stepping program (host reference is computed for it).
    small_layers: int = 8
    small_d_model: int = 1024
    small_d_ff: int = 4096
    # U1 (c): program compiled by one half while the other executes.
    compile_layers: int = 384
    compile_d_model: int = 256
    compile_d_ff: int = 512
    # U2 model-shaped program: 64 x 4096 x 12288 bf16 = 1.61 GB per chip at TP8.
    model_layers: int = 64
    model_d_model: int = 4096
    model_d_ff: int = 12288
    # local: TP16 shard 2k/2k+1 on lower[k]/upper[k], so every TP16 shard is a
    # slice of the TP8 shard already on that chip. natural: device-id order.
    target_order: str = "local"
    # Physical axis the block is halved along; see split_halves.
    half_split: str = "y"
    # aot: reshard with a background-compiled identity program (what
    # jax.device_put runs for a same-device-set reshard); device_put: call
    # jax.device_put at switch time.
    reshard: str = "device_put"
    warm_device_put: bool = True
    # Batch buckets of the TP16 target compiled in the background (the engines
    # precompile a set of shapes). The switch executes the `batch` bucket.
    target_buckets: tuple = (8, 16, 32, 64, 128, 256)
    # U1 (e): compile the U2 target set in the background while both halves step.
    u1_calibrate_target: bool = True
    control_s: float = 10.0
    stall_s: float = 30.0
    batch_target_s: float = 0.1
    queue_depth: int = 3
    poll_s: float = 0.01
    compile_timeout_s: float = 900.0
    command_timeout_s: float = 900.0
    output_atol: float = 0.05
    deep_output_atol: float = 0.1  # the (c) program: 384 layers, CPU error 0.028
    min_weight_bytes_per_chip: int = 1_000_000_000
    stall_rate_floor: float = 0.8
    stall_max_step_s: float = 1.0
    stall_blocked_fraction: float = 0.9
    # Overlap evidence (as CompileExecutionSmoke): native compile time and fully
    # contained serving steps per rank inside the native-compile intervals.
    min_native_compile_s: float = 5.0
    min_contained_steps: int = 20

    def validate(self):
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}")
        if self.half_split not in ("x", "y"):
            raise ValueError("half_split must be x or y")
        if self.target_order not in ("local", "natural"):
            raise ValueError("target_order must be local or natural")
        if self.reshard not in ("aot", "device_put"):
            raise ValueError("reshard must be aot or device_put")
        if self.target_order == "natural" and self.reshard == "aot":
            raise ValueError("natural order needs device_put: jit refuses mixed device orders")
        for shape in (self.small(), self.compile_shape(), self.model()):
            if min(dataclasses.astuple(shape)) < 1 or shape.d_ff % 16 or shape.d_model % 16:
                raise ValueError(f"invalid shape {shape}")
        if not (self.control_s > 0 and self.stall_s > 0 and self.batch_target_s > 0
                and self.queue_depth >= 1 and self.poll_s > 0):
            raise ValueError("invalid timing bounds")
        if self.batch not in self.target_buckets or any(b < 1 for b in self.target_buckets):
            raise ValueError("target_buckets must be positive and contain batch")

    def small(self):
        return Shape(self.small_layers, self.small_d_model, self.small_d_ff, self.batch)

    def compile_shape(self):
        return Shape(self.compile_layers, self.compile_d_model, self.compile_d_ff, self.batch)

    def model(self):
        return Shape(self.model_layers, self.model_d_model, self.model_d_ff, self.batch)


def clock():
    return {"mono_ns": time.monotonic_ns(), "epoch_ns": time.time_ns()}


# ----------------------------------------------------------------------------
# Deterministic values, identical on host (numpy) and device (jax.numpy).


def _hash_u32(xp, l, i, j, salt):
    u32 = xp.uint32
    v = l * u32(_M1) + i * u32(_M2) + j * u32(_M3) + u32(salt)
    v = v ^ (v >> u32(15))
    v = v * u32(_M4)
    v = v ^ (v >> u32(13))
    v = v * u32(_M2)
    return v ^ (v >> u32(16))


def _unit(xp, v):
    # Uniform in [-0.5, 0.5); exact in float32 (24-bit mantissa).
    return (v >> xp.uint32(8)).astype(xp.float32) * xp.float32(2.0 ** -24) - xp.float32(0.5)


def _scale(fan_in):
    import numpy as np
    return np.float32(math.sqrt(12.0 / fan_in))


def weights_np(full_shape, salt, fan_in, layer):
    """One layer of a weight array, computed on the host."""
    import ml_dtypes
    import numpy as np
    _, rows, cols = full_shape
    l = np.full((1, 1, 1), layer, np.uint32)
    i = np.arange(rows, dtype=np.uint32)[None, :, None]
    j = np.arange(cols, dtype=np.uint32)[None, None, :]
    values = _unit(np, _hash_u32(np, l, i, j, salt)) * _scale(fan_in)
    return values.astype(ml_dtypes.bfloat16)[0]


def weights_jnp(full_shape, salt, fan_in):
    import jax.numpy as jnp
    from jax import lax
    l, i, j = (lax.broadcasted_iota(jnp.uint32, full_shape, axis) for axis in range(3))
    return (_unit(jnp, _hash_u32(jnp, l, i, j, salt)) * _scale(fan_in)).astype(jnp.bfloat16)


def inputs_np(shape):
    import numpy as np
    i = np.arange(shape.batch, dtype=np.uint32)[:, None]
    j = np.arange(shape.d_model, dtype=np.uint32)[None, :]
    return _unit(np, _hash_u32(np, np.uint32(0), i, j, SALT_X))


def checksum_np(array):
    """Position-weighted wrapping uint32 checksum of a bf16 array (host)."""
    import numpy as np
    bits = np.ascontiguousarray(array).view(np.uint16).astype(np.uint32)
    l, i, j = np.indices(array.shape, dtype=np.uint32)
    return int(np.sum(bits * (_hash_u32(np, l, i, j, SALT_CHECK) | np.uint32(1)), dtype=np.uint32))


def checksum_jnp(array):
    import jax.numpy as jnp
    from jax import lax
    bits = lax.bitcast_convert_type(array, jnp.uint16).astype(jnp.uint32)
    l, i, j = (lax.broadcasted_iota(jnp.uint32, array.shape, axis) for axis in range(3))
    return jnp.sum(bits * (_hash_u32(jnp, l, i, j, SALT_CHECK) | jnp.uint32(1)), dtype=jnp.uint32)


def forward_fn(layers):
    import jax.numpy as jnp
    import numpy as np

    def forward(x, w_up, w_down):
        h = x
        for layer in range(layers):  # unrolled, like the engines' layer loops
            a = jnp.tanh(jnp.dot(h.astype(jnp.bfloat16), w_up[layer],
                                 preferred_element_type=jnp.float32))
            o = jnp.dot(a.astype(jnp.bfloat16), w_down[layer], preferred_element_type=jnp.float32)
            h = jnp.tanh(h + np.float32(RESIDUAL_SCALE) * o)
        return h
    return forward


def reference_forward(shape):
    """Host float32 reference with the device's bf16 rounding points."""
    import ml_dtypes
    import numpy as np
    up_shape, down_shape = shape.weight_shapes()
    h = inputs_np(shape)
    for layer in range(shape.layers):
        w_up = weights_np(up_shape, SALT_UP, shape.d_model, layer).astype(np.float32)
        w_down = weights_np(down_shape, SALT_DOWN, shape.d_ff, layer).astype(np.float32)
        a = np.tanh(h.astype(ml_dtypes.bfloat16).astype(np.float32) @ w_up)
        h = np.tanh(h + np.float32(RESIDUAL_SCALE)
                    * (a.astype(ml_dtypes.bfloat16).astype(np.float32) @ w_down))
    return h


# ----------------------------------------------------------------------------
# Topology.


def split_halves(devices, split="y"):
    """(lower, upper) device lists, each sorted by id.

    TPU: halves along physical ``split`` (x or y). x-halves are the S-layout TP8
    boxes (host columns, ranks {0, 2} / {1, 3}); y-halves are host rows (ranks
    {0, 1} / {2, 3}). libtpu 0.0.43 cannot compile ICI collectives on the
    origin-anchored x-half (U1 attempt 1), so y is the default. CPU emulation
    mirrors the TPU rank layout. Every process must belong to exactly one half.
    """
    devices = sorted(devices, key=lambda d: d.id)
    if len(devices) < 4 or len(devices) % 2:
        raise ValueError("need an even device count >= 4")
    if split not in ("x", "y"):
        raise ValueError("split must be x or y")
    if devices[0].platform == "tpu":
        axis = 0 if split == "x" else 1
        values = sorted({d.coords[axis] for d in devices})
        if len(values) % 2:
            raise ValueError(f"cannot halve the physical {split} dimension")
        cut = set(values[:len(values) // 2])
        lower = [d for d in devices if d.coords[axis] in cut]
    else:
        processes = sorted({d.process_index for d in devices})
        lower_processes = set(processes[0::2] if split == "x" else processes[:len(processes) // 2])
        lower = [d for d in devices if d.process_index in lower_processes]
    upper = [d for d in devices if d not in lower]
    if len(lower) != len(upper):
        raise ValueError("halves differ in size")
    if {d.process_index for d in lower} & {d.process_index for d in upper}:
        raise ValueError("a process spans both halves")
    return lower, upper


def target_devices(lower, upper, order):
    if order == "local":
        return [d for pair in zip(lower, upper) for d in pair]
    return sorted(lower + upper, key=lambda d: d.id)


def device_record(device):
    return {"id": device.id, "process_index": device.process_index, "platform": device.platform,
            "coords": list(getattr(device, "coords", ()) or ())}


def sample_summary(samples):
    if not samples:
        return {"steps": 0}
    durations = sorted((end - start) / 1e9 for start, end in samples)
    wall = (samples[-1][1] - samples[0][0]) / 1e9
    return {"steps": len(samples), "first_start_ns": samples[0][0], "last_end_ns": samples[-1][1],
            "wall_s": wall, "active_step_s": sum(durations),
            "steps_per_wall_s": len(samples) / wall if wall > 0 else None,
            "p50_step_s": statistics.median(durations),
            "p95_step_s": durations[min(len(durations) - 1, int(.95 * len(durations)))],
            "max_step_s": durations[-1]}


@dataclasses.dataclass
class Program:
    name: str
    shape: Shape
    scope: str  # half | target
    compiled: object
    x: object
    w_up: object
    w_down: object
    last: object = None
    last_end_ns: int = 0
    reference_hash: str = None
    reference_output: object = None


# ----------------------------------------------------------------------------
# Per-process worker.


class UnifiedWorker:
    """Lives in one initialized JAX process of the 16-chip runtime."""

    def __init__(self, config: UnifiedConfig, platform: str):
        import jax
        import numpy as np
        from jax.sharding import Mesh
        config.validate()
        self.config = config
        # New workers only; never touch a serving process's cache.
        jax.config.update("jax_enable_compilation_cache", False)
        self.rank = jax.process_index()
        self.devices = sorted(jax.devices(), key=lambda d: d.id)
        if any(d.platform != platform for d in self.devices):
            raise ValueError(f"expected only {platform} devices")
        self.lower, self.upper = split_halves(self.devices, config.half_split)
        self.half = "lower" if any(d.process_index == self.rank for d in self.lower) else "upper"
        self.half_devices = self.lower if self.half == "lower" else self.upper
        self.half_ranks = sorted({d.process_index for d in self.half_devices})
        self.local_devices = [d for d in self.devices if d.process_index == self.rank]
        self.half_mesh = Mesh(np.array(self.half_devices), ("model",))
        self.target_list = target_devices(self.lower, self.upper, config.target_order)
        self.target_mesh = Mesh(np.array(self.target_list), ("model",))
        # Two TP8 replicas as one 16-device array: row k holds shard k on
        # (lower[k], upper[k]). Its flattened order equals the local target order.
        self.pair_mesh = Mesh(np.array(list(zip(self.lower, self.upper))), ("model", "replica"))
        self.programs = {}
        self.precompiled = {}
        self.samples = collections.defaultdict(list)
        self.batches = []
        self.checks = []
        self.switches = []
        self.memory = []
        self.events = []
        self.compiles = {}
        self.compile_events = []
        self._references = {}
        self._lock = threading.Lock()
        self._commands = queue.Queue()
        self._results = {}
        self._submitted = 0
        self._completed = 0
        self._running = None
        self._error = None
        jax.monitoring.register_event_duration_secs_listener(self._on_duration)
        versions = {"jax": jax.__version__}
        for package in ("jaxlib", "libtpu"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        self.identity = {
            "rank": self.rank, "pid": os.getpid(), "hostname": socket.gethostname(),
            "versions": versions, "half": self.half, "half_ranks": self.half_ranks,
            "local_devices": [device_record(d) for d in self.local_devices],
            "lower_ids": [d.id for d in self.lower], "upper_ids": [d.id for d in self.upper],
            "target_ids": [d.id for d in self.target_list], "target_order": config.target_order,
            "half_split": config.half_split,
            "process_count": jax.process_count(), "device_count": jax.device_count(),
            "cpu_count": os.cpu_count(), "cache_enabled": bool(jax.config.jax_enable_compilation_cache)}
        self._thread = threading.Thread(target=self._loop, name="unified-commands", daemon=True)
        self._thread.start()

    # -- RPC surface (fast; never touches devices) ---------------------------

    def describe(self):
        return self.identity

    def submit(self, command):
        index = self._submitted
        self._submitted += 1
        self._commands.put((index, tuple(command)))
        return index

    def status(self):
        return {"rank": self.rank, "submitted": self._submitted, "completed": self._completed,
                "running": self._running, "error": self._error}

    def result(self, index):
        return self._results.get(index)

    def clock(self):
        return time.time_ns()

    def start_compile(self, job, spec):
        if job in self.compiles:
            raise RuntimeError(f"compile job {job} already started")
        record = {"job": job, "spec": spec, "thread": "unified-compile-" + job, "start": clock(),
                  "done": False}
        with self._lock:
            self.compiles[job] = record

        def work():
            process_cpu = time.process_time()
            try:
                record["parts"] = self._compile(job, spec)
                record["ok"] = True
            except Exception:
                record.update(ok=False, traceback=traceback.format_exc())
            finally:
                record["end"] = clock()
                record["call_s"] = (record["end"]["mono_ns"] - record["start"]["mono_ns"]) / 1e9
                record["whole_process_cpu_s"] = time.process_time() - process_cpu
                record["done"] = True

        threading.Thread(target=work, name=record["thread"], daemon=True).start()
        return {"started": True, "rank": self.rank}

    def compile_status(self, job):
        record = self.compiles.get(job)
        if record is None:
            return {"done": False, "started": False}
        return {"done": record["done"], "ok": record.get("ok"), "call_s": record.get("call_s"),
                "traceback": record.get("traceback")}

    def _native_intervals(self, thread):
        # The backend-compile event fires at the end of each native compile.
        return sorted((at - int(duration * 1e9), at) for name, at, duration in list(self.compile_events)
                      if name == thread)

    def report(self):
        with self._lock:
            samples = {key: list(value) for key, value in self.samples.items()}
            compiles = {job: dict(record) for job, record in self.compiles.items()}
        phases = {f"{program}/{phase}": sample_summary(s) for (program, phase), s in samples.items()}
        windows = {}
        for job, record in compiles.items():
            if not record.get("done"):
                continue
            start, end = record["start"]["mono_ns"], record["end"]["mono_ns"]
            intervals = self._native_intervals(record["thread"])
            native_s = sum(b - a for a, b in intervals) / 1e9
            call, native = {}, {}
            for (program, phase), s in samples.items():
                key = f"{program}/{phase}"
                inside = [x for x in s if start <= x[0] and x[1] <= end]
                if inside:
                    call[key] = sample_summary(inside)
                contained = [x for x in inside if any(a <= x[0] and x[1] <= b for a, b in intervals)]
                if contained:
                    native[key] = {**sample_summary(contained),
                                   "steps_per_native_s": len(contained) / native_s if native_s else None}
            windows[job] = {"call": call, "native": native, "native_total_s": native_s,
                            "native_intervals": len(intervals),
                            "native_max_s": max((b - a for a, b in intervals), default=0) / 1e9}
        compile_events = collections.Counter(thread for thread, _, _ in list(self.compile_events))
        return {"identity": self.identity, "status": self.status(), "phases": phases,
                "compile_windows": windows, "compiles": compiles, "checks": self.checks,
                "switches": self.switches, "memory": self.memory, "events": self.events,
                "batches": list(self.batches), "backend_compiles_by_thread": dict(compile_events),
                "timelines": self._timelines(samples)}

    def shutdown(self):
        self._commands.put((None, None))
        self._thread.join(timeout=10)

    # -- internals -------------------------------------------------------------

    def _on_duration(self, event, duration, **_):
        if event == BACKEND_COMPILE_EVENT:
            self.compile_events.append((threading.current_thread().name, time.monotonic_ns(), duration))

    def _loop(self):
        while True:
            index, command = self._commands.get()
            if index is None:
                return
            if self._error is not None:
                # Multi-process programs after a failure would hang peers.
                self._results[index] = {"skipped_after_error": True}
                self._completed += 1
                continue
            self._running = command[0]
            try:
                self._results[index] = getattr(self, "_cmd_" + command[0])(*command[1:])
            except Exception:
                self._error = traceback.format_exc()
                self._results[index] = {"error": self._error}
            finally:
                self._running = None
                self._completed += 1

    def _event(self, __event, /, **fields):
        self.events.append({"event": __event, **clock(), **fields})

    def _timelines(self, samples_by_key, bucket_s=1.0):
        out = {}
        for (program, phase), samples in samples_by_key.items():
            if not samples:
                continue
            origin = samples[0][0]
            counts = collections.Counter(int((end - origin) / 1e9 / bucket_s) for _, end in samples)
            out[f"{program}/{phase}"] = {"origin_mono_ns": origin, "bucket_s": bucket_s,
                                         "steps": [counts.get(i, 0) for i in range(max(counts) + 1)]}
        return out

    def _shardings(self, mesh):
        from jax.sharding import NamedSharding, PartitionSpec as P
        return {"x": NamedSharding(mesh, P()),
                "w_up": NamedSharding(mesh, P(None, None, "model")),
                "w_down": NamedSharding(mesh, P(None, "model", None)),
                "out": NamedSharding(mesh, P())}

    def _specs(self, shape, shardings):
        import jax
        import jax.numpy as jnp
        up_shape, down_shape = shape.weight_shapes()
        return (jax.ShapeDtypeStruct((shape.batch, shape.d_model), jnp.float32, sharding=shardings["x"]),
                jax.ShapeDtypeStruct(up_shape, jnp.bfloat16, sharding=shardings["w_up"]),
                jax.ShapeDtypeStruct(down_shape, jnp.bfloat16, sharding=shardings["w_down"]))

    def _compile_forward(self, shape, mesh):
        import jax
        s = self._shardings(mesh)
        function = jax.jit(forward_fn(shape.layers), in_shardings=(s["x"], s["w_up"], s["w_down"]),
                           out_shardings=s["out"])
        return function.lower(*self._specs(shape, s)).compile()

    def _compile(self, job, spec):
        import jax
        from jax._src import dispatch
        parts = {}

        def timed(name, thunk):
            start = time.monotonic()
            value = thunk()
            parts[name] = time.monotonic() - start
            return value

        shape = Shape(**spec["shape"])
        if spec["scope"] == "half":
            self.precompiled[spec.get("as", job)] = timed(
                "forward", lambda: self._compile_forward(shape, self.half_mesh))
            return parts
        target = self._shardings(self.target_mesh)
        pair = self._shardings(self.pair_mesh)
        entry = {"buckets": {}}
        for bucket in spec.get("buckets", [shape.batch]):
            variant = dataclasses.replace(shape, batch=bucket)
            entry["buckets"][bucket] = timed(f"forward_b{bucket}", lambda variant=variant: (
                self._compile_forward(variant, self.target_mesh)))
        entry["forward"] = entry["buckets"][shape.batch]
        _, up_spec, down_spec = self._specs(shape, pair)
        _, up_target, down_target = self._specs(shape, target)
        if self.config.reshard == "aot":
            identity = lambda a, b: (a, b)  # noqa: E731
            entry["to_target"] = timed("reshard_to_target", lambda: jax.jit(
                identity, out_shardings=(target["w_up"], target["w_down"])).lower(
                    up_spec, down_spec).compile())
            entry["to_pair"] = timed("reshard_to_pair", lambda: jax.jit(
                identity, out_shardings=(pair["w_up"], pair["w_down"])).lower(
                    up_target, down_target).compile())
        elif self.config.warm_device_put and self.config.target_order == "local":
            # Same function and out_shardings device_put uses for a same-device-list
            # reshard (jax/_src/dispatch.py _different_device_order_reshard).
            for name, (source, sharding) in {
                    "warm_device_put_to_target_up": (up_spec, target["w_up"]),
                    "warm_device_put_to_target_down": (down_spec, target["w_down"]),
                    "warm_device_put_to_pair_up": (up_target, pair["w_up"]),
                    "warm_device_put_to_pair_down": (down_target, pair["w_down"])}.items():
                timed(name, lambda source=source, sharding=sharding: jax.jit(
                    dispatch._device_put_reshard, out_shardings=sharding).lower(source).compile())
        self.precompiled[job] = entry
        return parts

    def _memory(self, label):
        stats = []
        for device in self.local_devices:
            raw = device.memory_stats() or {}
            stats.append({"id": device.id, **{k: raw.get(k) for k in (
                "bytes_in_use", "peak_bytes_in_use", "bytes_limit")}})
        record = {"label": label, **clock(), "devices": stats}
        self.memory.append(record)
        return record

    def _local_output(self, array):
        import numpy as np
        shards = [np.asarray(s.data) for s in array.addressable_shards]
        if any(not np.array_equal(shards[0], s) for s in shards[1:]):
            raise ValueError("replicated output differs between local shards")
        return shards[0]

    def _x(self, shape, sharding):
        import jax
        x = inputs_np(shape)
        return jax.make_array_from_callback(x.shape, sharding, lambda index: x[index])

    # -- commands (command thread only) -----------------------------------

    def _cmd_setup(self, name, shape_spec, use_precompiled=False):
        """Create weights on this process's half, compile (or adopt), warm up."""
        import jax
        shape = Shape(**shape_spec)
        s = self._shardings(self.half_mesh)
        up_shape, down_shape = shape.weight_shapes()
        start = time.monotonic()
        init = jax.jit(lambda: (weights_jnp(up_shape, SALT_UP, shape.d_model),
                                weights_jnp(down_shape, SALT_DOWN, shape.d_ff)),
                       out_shardings=(s["w_up"], s["w_down"]))
        w_up, w_down = init()
        jax.block_until_ready((w_up, w_down))
        init_s = time.monotonic() - start
        before = len([e for e in self.compile_events if e[0] == "unified-commands"])
        start = time.monotonic()
        if use_precompiled:
            compiled = self.precompiled.pop(name)
        else:
            compiled = self._compile_forward(shape, self.half_mesh)
        compile_s = time.monotonic() - start
        program = Program(name, shape, "half", compiled, self._x(shape, s["x"]), w_up, w_down)
        self.programs[name] = program
        durations = []
        for _ in range(5):
            t0 = time.monotonic_ns()
            program.last = compiled(program.x, w_up, w_down)
            program.last.block_until_ready()
            durations.append((time.monotonic_ns() - t0) / 1e9)
        program.last_end_ns = time.monotonic_ns()
        output = self._local_output(program.last)
        program.reference_hash = hashlib.sha256(output.tobytes()).hexdigest()
        program.reference_output = output
        result = {"rank": self.rank, "half": self.half, "name": name, "shape": shape_spec,
                  "init_s": init_s, "compile_s": compile_s, "precompiled": use_precompiled,
                  "warmup_step_s": durations, "step_s": statistics.median(durations[1:]),
                  "weight_bytes_per_chip": shape.weight_bytes_per_chip(len(self.half_devices)),
                  "output_sha256": program.reference_hash,
                  "command_thread_backend_compiles": len([e for e in self.compile_events
                                                          if e[0] == "unified-commands"]) - before}
        self._event("SETUP", name=name)
        return result

    def _cmd_steps(self, name, count, phase):
        program = self.programs[name]
        with self._lock:
            samples = self.samples[(name, phase)]
        first = len(samples)
        for _ in range(count):
            start = time.monotonic_ns()
            program.last = program.compiled(program.x, program.w_up, program.w_down)
            program.last.block_until_ready()
            end = time.monotonic_ns()
            samples.append((start, end))
        program.last_end_ns = samples[-1][1]
        batch = samples[first:]
        record = {"program": name, "phase": phase, "steps": count, "start_ns": batch[0][0],
                  "end_ns": batch[-1][1], "max_step_s": max(e - s for s, e in batch) / 1e9,
                  "epoch_offset_ns": time.time_ns() - time.monotonic_ns()}
        self.batches.append(record)
        return {"steps": count, "max_step_s": record["max_step_s"],
                "mean_step_s": (batch[-1][1] - batch[0][0]) / 1e9 / count}

    def _cmd_stall(self, seconds, phase):
        self._event("STALL_BEGIN", phase=phase, seconds=seconds)
        time.sleep(seconds)
        self._event("STALL_END", phase=phase)
        return {"stalled_s": seconds}

    def _cmd_check(self, name, label, with_reference):
        import numpy as np
        program = self.programs[name]
        output = self._local_output(program.last)
        record = {"rank": self.rank, "half": self.half, "name": name, "label": label,
                  "scope": program.scope, "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                  "finite": bool(np.isfinite(output).all()),
                  "equals_setup_output": None, "max_abs_diff_vs_reference_output": None}
        if program.reference_output is not None:
            record["equals_setup_output"] = record["sha256"] == program.reference_hash
            record["max_abs_diff_vs_reference_output"] = float(
                np.max(np.abs(output - program.reference_output)))
        if with_reference:
            key = dataclasses.astuple(program.shape)
            if key not in self._references:
                self._references[key] = reference_forward(program.shape)
            record["max_abs_error_vs_host"] = float(np.max(np.abs(output - self._references[key])))
        self.checks.append(record)
        # The driver compares outputs across ranks with a tolerance, then drops them.
        return {**record, "_output": output}

    def _cmd_checksum(self, name, label):
        import jax
        program = self.programs[name]
        up = jax.jit(checksum_jnp)(program.w_up)
        down = jax.jit(checksum_jnp)(program.w_down)
        record = {"rank": self.rank, "half": self.half, "name": name, "label": label,
                  "scope": program.scope, "w_up": int(self._local_output(up)),
                  "w_down": int(self._local_output(down))}
        self.checks.append(record)
        return record

    def _cmd_memory(self, label):
        return self._memory(label)

    def _cmd_drop(self, name):
        program = self.programs.pop(name)
        for array in (program.w_up, program.w_down):
            array.delete()
        return {"dropped": name}

    def _cmd_drop_half_weights(self, name):
        """Free the TP8 weights after the switch; keep the loaded executable."""
        program = self.programs[name]
        for array in (program.w_up, program.w_down):
            array.delete()
        program.w_up = program.w_down = None
        return {"dropped_weights": name}

    def _cmd_prepare_target(self, source_name):
        record = self.compiles.get("target")
        if not record or not record.get("done") or not record.get("ok"):
            raise RuntimeError("target compile missing, running or failed")
        s = self._shardings(self.target_mesh)
        source = self.programs[source_name]
        self.target_x = self._x(source.shape, s["x"])
        self.target_x.block_until_ready()
        return {"prepared": True}

    def _compiles_in_thread(self, start_ns, end_ns):
        return [d for thread, at, d in self.compile_events
                if thread == "unified-commands" and start_ns <= at <= end_ns]

    def _cmd_switch_forward(self, source_name, target_name):
        """Stop (queue drained) -> reshard 2x TP8 -> TP16 -> first target step."""
        import jax
        source = self.programs[source_name]
        entry = self.precompiled["target"]
        target = self._shardings(self.target_mesh)
        pair = self._shardings(self.pair_mesh)
        t = {"last_source_step_end": {"mono_ns": source.last_end_ns,
                                      "epoch_ns": source.last_end_ns + time.time_ns() - time.monotonic_ns()},
             "enter": clock()}
        up_view = jax.make_array_from_single_device_arrays(
            source.w_up.shape, pair["w_up"], [s.data for s in source.w_up.addressable_shards])
        down_view = jax.make_array_from_single_device_arrays(
            source.w_down.shape, pair["w_down"], [s.data for s in source.w_down.addressable_shards])
        t["view"] = clock()
        if self.config.reshard == "aot":
            w_up, w_down = entry["to_target"](up_view, down_view)
        else:
            w_up, w_down = jax.device_put((up_view, down_view), (target["w_up"], target["w_down"]))
        t["reshard_dispatched"] = clock()
        jax.block_until_ready((w_up, w_down))
        t["reshard_ready"] = clock()
        y = entry["forward"](self.target_x, w_up, w_down)
        t["first_dispatched"] = clock()
        y.block_until_ready()
        t["first_ready"] = clock()
        program = Program(target_name, source.shape, "target", entry["forward"], self.target_x,
                          w_up, w_down, last=y, last_end_ns=t["first_ready"]["mono_ns"])
        output = self._local_output(y)
        program.reference_hash = hashlib.sha256(output.tobytes()).hexdigest()
        program.reference_output = output
        self.programs[target_name] = program
        compiles = self._compiles_in_thread(t["enter"]["mono_ns"], t["first_ready"]["mono_ns"])
        record = {"direction": "forward", "rank": self.rank, "half": self.half,
                  "reshard": self.config.reshard, "target_order": self.config.target_order,
                  "times": t, "local_s": self._local_segments(t),
                  "backend_compiles_during_switch": len(compiles),
                  "backend_compile_s_during_switch": sum(compiles),
                  "target_output_sha256": program.reference_hash,
                  "max_abs_diff_vs_source_output": float(abs(output - source.reference_output).max())}
        self.switches.append(record)
        return record

    def _cmd_switch_reverse(self, target_name, half_name):
        """Stop TP16 -> reshard TP16 -> 2x TP8 -> first step on each half."""
        import jax
        source = self.programs[target_name]
        half = self.programs[half_name]
        entry = self.precompiled["target"]
        pair = self._shardings(self.pair_mesh)
        s = self._shardings(self.half_mesh)
        t = {"last_source_step_end": {"mono_ns": source.last_end_ns,
                                      "epoch_ns": source.last_end_ns + time.time_ns() - time.monotonic_ns()},
             "enter": clock()}
        if self.config.reshard == "aot":
            up_pair, down_pair = entry["to_pair"](source.w_up, source.w_down)
        else:
            up_pair, down_pair = jax.device_put((source.w_up, source.w_down), (pair["w_up"], pair["w_down"]))
        t["reshard_dispatched"] = clock()
        jax.block_until_ready((up_pair, down_pair))
        t["reshard_ready"] = clock()
        w_up = jax.make_array_from_single_device_arrays(
            up_pair.shape, s["w_up"], [x.data for x in up_pair.addressable_shards])
        w_down = jax.make_array_from_single_device_arrays(
            down_pair.shape, s["w_down"], [x.data for x in down_pair.addressable_shards])
        t["view"] = clock()
        y = half.compiled(half.x, w_up, w_down)
        t["first_dispatched"] = clock()
        y.block_until_ready()
        t["first_ready"] = clock()
        half.w_up, half.w_down, half.last, half.last_end_ns = w_up, w_down, y, t["first_ready"]["mono_ns"]
        output = self._local_output(y)
        compiles = self._compiles_in_thread(t["enter"]["mono_ns"], t["first_ready"]["mono_ns"])
        digest = hashlib.sha256(output.tobytes()).hexdigest()
        record = {"direction": "reverse", "rank": self.rank, "half": self.half,
                  "reshard": self.config.reshard, "target_order": self.config.target_order,
                  "times": t, "local_s": self._local_segments(t),
                  "backend_compiles_during_switch": len(compiles),
                  "backend_compile_s_during_switch": sum(compiles),
                  "half_output_sha256": digest, "equals_pre_switch_half_output": digest == half.reference_hash}
        self.switches.append(record)
        return record

    @staticmethod
    def _local_segments(t):
        """Monotonic (single-host) segments, the same split as switch_segments."""
        def span(a, b):
            return (t[b]["mono_ns"] - t[a]["mono_ns"]) / 1e9
        # Forward wraps views before resharding; reverse splits views after it.
        ready = "reshard_ready" if t["view"]["mono_ns"] <= t["reshard_ready"]["mono_ns"] else "view"
        return {"stop": span("last_source_step_end", "enter"), "reshard": span("enter", ready),
                "first_exec": span(ready, "first_ready"),
                "switch": span("last_source_step_end", "first_ready"),
                "reshard_dispatched_to_ready": span("reshard_dispatched", "reshard_ready"),
                "first_dispatched_to_ready": span("first_dispatched", "first_ready")}

    def _cmd_drop_target_inputs(self, target_name):
        program = self.programs.pop(target_name)
        for array in (program.w_up, program.w_down):
            array.delete()
        return {"dropped": target_name}

    def _cmd_probe_deserialize(self, name):
        """Serialize this half's live-compiled executable and load it back.

        This is the libtpu LoadSerializedExecutable path that both the JAX
        persistent cache and the executable bank use; it aborted for the
        upper-half target in CompileExecutionSmoke.
        """
        from jax.experimental import serialize_executable
        program = self.programs[name]
        self._event("PROBE_DESERIALIZE_BEGIN", name=name)
        start = time.monotonic()
        payload, in_tree, out_tree = serialize_executable.serialize(program.compiled)
        serialized_at = time.monotonic()
        loaded = serialize_executable.deserialize_and_load(
            payload, in_tree, out_tree, execution_devices=self.half_devices)
        loaded_at = time.monotonic()
        y = loaded(program.x, program.w_up, program.w_down)
        y.block_until_ready()
        output = self._local_output(y)
        digest = hashlib.sha256(output.tobytes()).hexdigest()
        record = {"rank": self.rank, "half": self.half, "name": name, "label": "probe_deserialize",
                  "payload_bytes": len(payload), "serialize_s": serialized_at - start,
                  "load_s": loaded_at - serialized_at, "sha256": digest,
                  "equals_setup_output": digest == program.reference_hash}
        self.checks.append(record)
        self._event("PROBE_DESERIALIZE_END", name=name)
        return record


# ----------------------------------------------------------------------------
# Driver side (backend-agnostic: Ray actors on TPU, subprocesses on CPU).


class Driver:
    """Submits ordered command sequences and keeps per-half queues full."""

    def __init__(self, config, fleet, emit):
        self.config, self.fleet, self.emit = config, fleet, emit
        identities = fleet.call([(r, "describe", ()) for r in fleet.ranks], 120)
        self.identities = {i["rank"]: i for i in identities}
        self.halves = {h: sorted(r for r, i in self.identities.items() if i["half"] == h)
                       for h in HALVES}
        if any(len(v) != 2 for v in self.halves.values()):
            raise RuntimeError(f"expected two processes per half, got {self.halves}")
        self.all = sorted(fleet.ranks)
        self.log = []

    def topology(self):
        first = self.identities[self.all[0]]
        return {"halves": self.halves, "lower_ids": first["lower_ids"], "upper_ids": first["upper_ids"],
                "target_ids": first["target_ids"], "target_order": first["target_order"],
                "identities": self.identities}

    def submit(self, ranks, command, timeout_s=60):
        indexes = self.fleet.call([(r, "submit", (command,)) for r in ranks], timeout_s)
        return dict(zip(ranks, indexes))

    def statuses(self, ranks):
        statuses = {s["rank"]: s for s in self.fleet.call([(r, "status", ()) for r in ranks], 60)}
        for status in statuses.values():
            if status["error"]:
                raise RuntimeError(f"rank {status['rank']} command failed:\n{status['error']}")
        return statuses

    def wait(self, pending, timeout_s):
        deadline = time.monotonic() + timeout_s
        while True:
            statuses = self.statuses(list(pending))
            if all(statuses[r]["completed"] > index for r, index in pending.items()):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"commands {pending} not complete: {statuses}")
            time.sleep(self.config.poll_s)
        results = self.fleet.call([(r, "result", (index,)) for r, index in pending.items()], 60)
        return dict(zip(pending, results))

    def run(self, ranks, command, timeout_s=None):
        return self.wait(self.submit(ranks, command), timeout_s or self.config.command_timeout_s)

    def check(self, name, label, reference, ranks=None):
        """Output check on ``ranks``; outputs are compared here, then dropped.

        Ranks of one half must agree bitwise (one replicated output). The two
        halves are separately compiled programs, so they are compared with a
        tolerance; their hash equality is recorded as data only.
        """
        import numpy as np
        results = self.run(ranks or self.all, ("check", name, label, reference))
        outputs = {r: v.pop("_output") for r, v in results.items()}
        by_half = {h: [r for r in self.halves[h] if r in results] for h in HALVES}
        within = all(len({results[r]["sha256"] for r in rs}) == 1 for rs in by_half.values() if rs)
        cross = None
        if all(by_half.values()):
            cross = float(np.max(np.abs(outputs[by_half["lower"][0]] - outputs[by_half["upper"][0]])))
        return {"ranks": results, "within_half_equal": within, "cross_half_max_abs_diff": cross,
                "all_hashes_equal": len({v["sha256"] for v in results.values()}) == 1}

    def clock_offsets(self, ranks, rounds=8):
        """Remote epoch minus driver epoch, from the minimum-RTT round trip."""
        offsets = {}
        for rank in ranks:
            best = None
            for _ in range(rounds):
                start = time.time_ns()
                remote = self.fleet.call([(rank, "clock", ())], 30)[0]
                end = time.time_ns()
                if best is None or end - start < best["rtt_ns"]:
                    best = {"rtt_ns": end - start, "offset_ns": remote - (start + end) // 2}
            offsets[rank] = best
        return offsets

    def compile_done(self, ranks, job):
        statuses = self.fleet.call([(r, "compile_status", (job,)) for r in ranks], 60)
        for status in statuses:
            if status["done"] and not status["ok"]:
                raise RuntimeError(f"compile {job} failed:\n{status['traceback']}")
        return all(s["done"] for s in statuses)

    def start_compile(self, ranks, job, spec):
        self.log.append({"event": "start_compile", "job": job, "ranks": ranks, **clock()})
        return self.fleet.call([(r, "start_compile", (job, spec)) for r in ranks], 60)

    def serve(self, groups, phase, min_s, until=None, max_s=None, final=None, until_poll_s=0.25):
        """Keep each group's queue ``queue_depth`` batches deep.

        groups: {group: (program, ranks, steps_per_batch)}. Runs at least
        ``min_s`` and until ``until()`` is true, then stops issuing. ``final``
        = (ranks, command) is enqueued right behind the last batches, so a
        switch starts with no driver round trip. Returns the final results.
        """
        outstanding = {g: collections.deque() for g in groups}
        ranks = sorted({r for _, rs, _ in groups.values() for r in rs})
        start = time.monotonic()
        last_until, until_ok = 0.0, until is None
        self.log.append({"event": "serve_begin", "phase": phase, "groups": {
            g: {"program": p, "ranks": rs, "steps_per_batch": k} for g, (p, rs, k) in groups.items()},
            **clock()})
        issued = collections.Counter()
        while True:
            statuses = self.statuses(ranks)
            for g, (_, group_ranks, _) in groups.items():
                while outstanding[g] and all(statuses[r]["completed"] > outstanding[g][0][r]
                                             for r in group_ranks):
                    outstanding[g].popleft()
            elapsed = time.monotonic() - start
            if until is not None and not until_ok and time.monotonic() - last_until >= until_poll_s:
                until_ok, last_until = bool(until()), time.monotonic()
            if elapsed >= min_s and until_ok:
                break
            if max_s is not None and elapsed > max_s:
                raise TimeoutError(f"phase {phase} exceeded {max_s} s")
            for g, (program, group_ranks, steps) in groups.items():
                while len(outstanding[g]) < self.config.queue_depth:
                    outstanding[g].append(self.submit(group_ranks, ("steps", program, steps, phase)))
                    issued[g] += 1
            time.sleep(self.config.poll_s)
        stop = clock()
        final_result = None
        pending = {}
        if final is not None:
            pending = self.submit(*final)
        for g in groups:
            if outstanding[g]:
                self.wait(outstanding[g][-1], self.config.command_timeout_s)
        if pending:
            final_result = self.wait(pending, self.config.command_timeout_s)
        self.log.append({"event": "serve_end", "phase": phase, "stop_issuing": stop,
                         "batches_issued": dict(issued), **clock()})
        self.emit("UNIFIED_PHASE_DONE", phase=phase, batches=dict(issued),
                  elapsed_s=time.monotonic() - start)
        return final_result

    def steps_per_batch(self, setup_results, ranks):
        step_s = max(setup_results[r]["step_s"] for r in ranks)
        return max(1, min(100_000, int(round(self.config.batch_target_s / max(step_s, 1e-6)))))

    def reports(self):
        return self.fleet.call([(r, "report", ()) for r in self.all], 300)


def _half_groups(driver, program, steps):
    return {h: (program, driver.halves[h], steps[h]) for h in HALVES}


def _target_spec(config):
    return {"scope": "target", "shape": dataclasses.asdict(config.model()),
            "buckets": sorted(set(int(b) for b in config.target_buckets))}


def protocol_u1(config, fleet, emit, progress):
    """Submesh independence: (a) both halves, (b) idle and stalled half, (c) compile vs
    execute, (e) background compile of the U2 target set while both halves step."""
    d = Driver(config, fleet, emit)
    progress["topology"] = d.topology()
    progress["driver_log"] = d.log
    small, comp = config.small(), config.compile_shape()
    setup = d.run(d.all, ("setup", "small", dataclasses.asdict(small)))
    progress["setup"] = setup
    emit("UNIFIED_SETUP", step_s={r: v["step_s"] for r, v in setup.items()},
         compile_s={r: v["compile_s"] for r, v in setup.items()})
    steps = {h: d.steps_per_batch(setup, d.halves[h]) for h in HALVES}
    progress["steps_per_batch"] = steps
    groups = _half_groups(d, "small", steps)
    checks = progress.setdefault("check_rounds", {})

    def check(label, name="small"):
        checks[label] = d.check(name, label, True)
        emit("UNIFIED_CHECK", label=label, within_half_equal=checks[label]["within_half_equal"],
             cross_half_max_abs_diff=checks[label]["cross_half_max_abs_diff"])

    # (a0) each half alone, (a) both concurrently.
    d.serve({"lower": groups["lower"]}, "a0_lower_alone", config.control_s)
    d.serve({"upper": groups["upper"]}, "a0_upper_alone", config.control_s)
    d.serve(groups, "a_both", 2 * config.control_s)
    check("a")
    # (b1) one half idle for stall_s while the other steps.
    d.serve({"lower": groups["lower"]}, "b1_lower_active_upper_idle", config.stall_s)
    d.serve({"upper": groups["upper"]}, "b1_upper_active_lower_idle", config.stall_s)
    d.serve(groups, "b1_both_after", config.control_s)
    check("b1")
    # (c) each half compiles a new program while the other executes.
    for compiling, active in (("lower", "upper"), ("upper", "lower")):
        job = f"c_{compiling}"
        d.start_compile(d.halves[compiling], job,
                        {"scope": "half", "shape": dataclasses.asdict(comp), "as": "c"})
        d.serve({active: groups[active]}, f"c_{active}_active_{compiling}_compiles", 0.0,
                until=lambda job=job, compiling=compiling: d.compile_done(d.halves[compiling], job),
                max_s=config.compile_timeout_s)
    # Each rank adopts its own half's background-compiled executable as "c".
    d.run(d.all, ("setup", "c", dataclasses.asdict(comp), True))
    d.run(d.all, ("steps", "c", 3, "c_program"))
    check("c", name="c")
    d.run(d.all, ("drop", "c"))
    d.serve(groups, "c_both_after", config.control_s)
    check("c_after")
    # (e) all four processes compile the U2 target set (TP16, model shape, all
    # buckets) in the background while both halves keep stepping (calibration
    # for U2, and the dispatch-bound counterpart of CompileExecutionSmoke).
    if config.u1_calibrate_target:
        d.serve(groups, "e_baseline", config.control_s)
        d.start_compile(d.all, "target", _target_spec(config))
        d.serve(groups, "e_target_compiles", 0.0, until=lambda: d.compile_done(d.all, "target"),
                max_s=config.compile_timeout_s)
        d.serve(groups, "e_post", config.control_s)
        check("e")
    # (b2) one half stalled inside a step: its leader sleeps stall_s while its
    # partner has already launched the step and waits in the collective.
    for stalled, active in (("upper", "lower"), ("lower", "upper")):
        leader = d.halves[stalled][0]
        phase = f"b2_{stalled}_stalled"
        d.submit([leader], ("stall", config.stall_s, phase))
        pending = d.submit(d.halves[stalled], ("steps", "small", 5, phase))

        def resumed(pending=pending):
            statuses = d.statuses(list(pending))
            return all(statuses[r]["completed"] > i for r, i in pending.items())

        d.serve({active: groups[active]}, f"b2_{active}_active_{stalled}_stalled", config.stall_s,
                until=resumed, max_s=config.stall_s + config.command_timeout_s)
        d.wait(pending, config.command_timeout_s)
        check(f"b2_{stalled}")
    d.serve(groups, "b2_both_after", config.control_s)
    check("final")
    reports = d.reports()
    progress["workers"] = reports
    return {"workers": reports, **verdict_u1(config, progress)}


def protocol_u2(config, fleet, emit, progress):
    """In-runtime switch 2x TP8 -> TP16 -> 2x TP8 with background target compile."""
    d = Driver(config, fleet, emit)
    progress["topology"] = d.topology()
    progress["driver_log"] = d.log
    model = config.model()
    setup = d.run(d.all, ("setup", "model", dataclasses.asdict(model)))
    progress["setup"] = setup
    emit("UNIFIED_SETUP", step_s={r: v["step_s"] for r, v in setup.items()},
         weight_bytes_per_chip=model.weight_bytes_per_chip(8))
    checks = progress.setdefault("check_rounds", {})
    checks["setup"] = d.check("model", "setup", False)
    checks["setup_checksums"] = d.run(d.all, ("checksum", "model", "setup"))
    progress["memory"] = {"setup": d.run(d.all, ("memory", "after_setup"))}
    steps = {h: d.steps_per_batch(setup, d.halves[h]) for h in HALVES}
    progress["steps_per_batch"] = steps
    groups = _half_groups(d, "model", steps)
    d.serve(groups, "baseline", config.control_s)
    d.start_compile(d.all, "target", _target_spec(config))
    d.serve(groups, "overlap", 0.0, until=lambda: d.compile_done(d.all, "target"),
            max_s=config.compile_timeout_s)
    emit("UNIFIED_TARGET_COMPILED")
    d.serve(groups, "postcontrol", config.control_s)
    d.run(d.all, ("prepare_target", "model"))
    progress["clock_offsets_forward"] = offsets = d.clock_offsets(d.all)
    # Short pre-switch phase so the switch command sits right behind live batches.
    forward = d.serve(groups, "preswitch", min(2.0, config.control_s),
                      final=(d.all, ("switch_forward", "model", "target")))
    progress["switch_forward"] = forward
    progress["switch_forward_segments"] = switch_segments(forward, offsets)
    emit("UNIFIED_SWITCH_FORWARD", **{k: v for k, v in progress["switch_forward_segments"].items()
                                      if not isinstance(v, dict)})
    checks["target_first"] = d.check("target", "target_first", False)
    checks["target_checksums"] = d.run(d.all, ("checksum", "target", "target"))
    progress["memory"]["forward"] = d.run(d.all, ("memory", "after_forward"))
    d.run(d.all, ("drop_half_weights", "model"))
    warm = d.run(d.all, ("steps", "target", 3, "target_warm"))
    target_step = max(v["mean_step_s"] for v in warm.values())
    target_steps = max(1, int(round(config.batch_target_s / max(target_step, 1e-6))))
    progress["target_steps_per_batch"] = target_steps
    progress["clock_offsets_reverse"] = offsets = d.clock_offsets(d.all)
    reverse = d.serve({"block": ("target", d.all, target_steps)}, "target", config.control_s,
                      final=(d.all, ("switch_reverse", "target", "model")))
    progress["switch_reverse"] = reverse
    progress["switch_reverse_segments"] = switch_segments(reverse, offsets)
    emit("UNIFIED_SWITCH_REVERSE", **{k: v for k, v in progress["switch_reverse_segments"].items()
                                      if not isinstance(v, dict)})
    checks["reverse_first"] = d.check("model", "reverse_first", False)
    checks["reverse_checksums"] = d.run(d.all, ("checksum", "model", "reverse"))
    d.run(d.all, ("drop_target_inputs", "target"))
    progress["memory"]["reverse"] = d.run(d.all, ("memory", "after_reverse"))
    d.serve(groups, "post_reverse", config.control_s)
    checks["final"] = d.check("model", "final", False)
    reports = d.reports()
    progress["workers"] = reports
    return {"workers": reports, **verdict_u2(config, progress)}


def protocol_u1d(config, fleet, emit, progress):
    """Serialize/deserialize each half's live executable: lower first (control), then upper."""
    d = Driver(config, fleet, emit)
    progress["topology"] = d.topology()
    progress["driver_log"] = d.log
    setup = d.run(d.all, ("setup", "small", dataclasses.asdict(config.small())))
    progress["setup"] = setup
    probes = progress.setdefault("probes", {})
    for half in HALVES:
        # A native abort kills the half's processes; keep what completed.
        progress["workers_before_" + half] = d.reports()
        emit("UNIFIED_PROBE_BEGIN", half=half)
        probes[half] = d.run(d.halves[half], ("probe_deserialize", "small"))
        emit("UNIFIED_PROBE_DONE", half=half, **{str(r): v["equals_setup_output"]
                                                 for r, v in probes[half].items()})
    reports = d.reports()
    progress["workers"] = reports
    ok = all(v["equals_setup_output"] for p in probes.values() for v in p.values())
    return {"workers": reports, "status": "passed" if ok else "failed", "probes": probes}


PROTOCOLS = {"u1": protocol_u1, "u2": protocol_u2, "u1d": protocol_u1d}


# ----------------------------------------------------------------------------
# Analysis.


def switch_segments(records, offsets=None):
    """Switch segments, last source step -> first target step ready.

    switch = stop + reshard + first_exec, where
      stop       = last source step (latest rank) -> every rank inside the switch command
                   (steps are synchronous, so this is command-loop latency),
      reshard    = -> every rank's reshard and replica views complete,
      first_exec = -> every rank's first target step complete.
    Cross-host values use per-rank epoch stamps shifted onto the driver clock by
    ``offsets`` (min-RTT RPC estimate; uncertainty about RTT/2). ``local`` gives
    the same split per rank from its monotonic clock (no cross-host error).
    """
    if not isinstance(records, dict):
        records = {r["rank"]: r for r in records}
    direction = next(iter(records.values()))["direction"]
    ready_key = "reshard_ready" if direction == "forward" else "view"

    def split(corrected):
        def at(rank, key):
            ns = records[rank]["times"][key]["epoch_ns"]
            return ns - offsets[rank]["offset_ns"] if corrected and offsets else ns
        last = max(at(r, "last_source_step_end") for r in records)
        first_last = min(at(r, "last_source_step_end") for r in records)
        enter = max(at(r, "enter") for r in records)
        ready = max(at(r, ready_key) for r in records)
        first = max(at(r, "first_ready") for r in records)
        return {"switch_s": (first - last) / 1e9, "stop_s": (enter - last) / 1e9,
                "reshard_s": (ready - enter) / 1e9, "first_exec_s": (first - ready) / 1e9,
                "source_last_step_skew_s": (last - first_last) / 1e9,
                "earliest_source_downtime_s": (first - first_last) / 1e9}

    local = {r: v["local_s"] for r, v in records.items()}
    keys = ("stop", "reshard", "first_exec", "switch")
    return {"direction": direction, "clock": "driver-corrected" if offsets else "raw-epoch",
            **split(True), "raw_epoch": split(False),
            "clock_offsets": offsets, "local_per_rank": local,
            "local_max": {k: max(v[k] for v in local.values()) for k in keys},
            "local_min": {k: min(v[k] for v in local.values()) for k in keys},
            "backend_compiles_during_switch": sum(r["backend_compiles_during_switch"]
                                                  for r in records.values()),
            "backend_compile_s_during_switch": max(r["backend_compile_s_during_switch"]
                                                   for r in records.values())}


def _by_half(reports):
    return {h: [r for r in reports if r["identity"]["half"] == h] for h in HALVES}


def _mean(values):
    values = [v for v in values]
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def _rate(reports, key):
    return _mean([r["phases"].get(key, {}).get("steps_per_wall_s") for r in reports])


def _ratio(a, b):
    return a / b if a and b else None


def _check_round(check, atol, need_reference, require_all_equal=False):
    values = list(check["ranks"].values())
    problems = []
    if not check["within_half_equal"]:
        problems.append("ranks of one half disagree")
    if require_all_equal and not check["all_hashes_equal"]:
        problems.append("ranks of one 16-chip program disagree")
    if not all(v["finite"] for v in values):
        problems.append("non-finite output")
    if check["cross_half_max_abs_diff"] is not None and not check["cross_half_max_abs_diff"] <= atol:
        problems.append(f"halves differ by {check['cross_half_max_abs_diff']} > {atol}")
    if need_reference:
        worst = max(v.get("max_abs_error_vs_host", math.inf) for v in values)
        if not worst <= atol:
            problems.append(f"host reference error {worst} > {atol}")
    return problems


def overlap_impact(halves, program, baseline_phase, overlap_phase, post_phase, job):
    """Per half: baseline vs steps fully inside each rank's own compile windows.

    ``call`` = the whole compile-thread call (trace, lower, every native
    compile); ``native`` = only the backend-compile intervals (the window
    CompileExecutionSmoke's -19% used, which excluded trace/lower).
    """
    out = {}
    key = f"{program}/{overlap_phase}"
    for h, reports in halves.items():
        base = [r["phases"].get(f"{program}/{baseline_phase}", {}) for r in reports]
        windows = [r["compile_windows"].get(job, {}) for r in reports]
        call = [w.get("call", {}).get(key, {}) for w in windows]
        native = [w.get("native", {}).get(key, {}) for w in windows]
        base_rate = _mean([b.get("steps_per_wall_s") for b in base])
        call_rate = _mean([c.get("steps_per_wall_s") for c in call])
        native_rate = _mean([n.get("steps_per_native_s") for n in native])
        base_p50 = _mean([b.get("p50_step_s") for b in base])
        native_p50 = _mean([n.get("p50_step_s") for n in native])
        out[h] = {
            "baseline_steps_per_s": base_rate, "call_window_steps_per_s": call_rate,
            "native_window_steps_per_s": native_rate,
            "post_steps_per_s": _rate(reports, f"{program}/{post_phase}"),
            "call_change": _ratio(call_rate, base_rate) - 1 if _ratio(call_rate, base_rate) else None,
            "native_change": (_ratio(native_rate, base_rate) - 1
                              if _ratio(native_rate, base_rate) else None),
            "baseline_mean_step_ms": 1e3 / base_rate if base_rate else None,
            "native_mean_step_ms": 1e3 / native_rate if native_rate else None,
            "baseline_p50_step_ms": base_p50 * 1e3 if base_p50 else None,
            "native_p50_step_ms": native_p50 * 1e3 if native_p50 else None,
            "native_contained_steps": [n.get("steps", 0) for n in native],
            "native_total_s": [w.get("native_total_s") for w in windows],
            "native_intervals": [w.get("native_intervals") for w in windows],
            "call_s": [r["compiles"].get(job, {}).get("call_s") for r in reports]}
    return out


def _overlap_problems(config, impact):
    problems = []
    for h, v in impact.items():
        if any((s or 0) < config.min_native_compile_s for s in v["native_total_s"]):
            problems.append(f"{h}: native compile {v['native_total_s']} s < {config.min_native_compile_s}")
        if any(n < config.min_contained_steps for n in v["native_contained_steps"]):
            problems.append(f"{h}: contained steps {v['native_contained_steps']} "
                            f"< {config.min_contained_steps}")
    return problems


def verdict_u1(config, progress):
    reports = progress["workers"]
    halves = _by_half(reports)
    failures, rates = [], {}
    for label, check in progress["check_rounds"].items():
        atol = config.deep_output_atol if label == "c" else config.output_atol
        failures += [f"check {label}: {p}" for p in _check_round(check, atol, True)]
    keys = set().union(*(r["phases"] for r in reports))
    for h in HALVES:
        rates[h] = {k.split("/", 1)[1]: _rate(halves[h], k) for k in sorted(keys)
                    if k.startswith("small/") and all(k in r["phases"] for r in halves[h])}
    b2 = {}
    for stalled, active in (("upper", "lower"), ("lower", "upper")):
        key = f"small/b2_{active}_active_{stalled}_stalled"
        active_rate = _rate(halves[active], key)
        baseline = rates[active].get("a_both")
        max_step = max(r["phases"].get(key, {}).get("max_step_s", math.inf) for r in halves[active])
        stalled_steps = [r["phases"].get(f"small/b2_{stalled}_stalled", {}) for r in halves[stalled]]
        blocked = max(s.get("max_step_s", 0) for s in stalled_steps)
        ratio = _ratio(active_rate, baseline)
        b2[stalled] = {"active_half": active, "active_rate": active_rate, "active_a_both_rate": baseline,
                       "active_rate_ratio": ratio, "active_max_step_s": max_step,
                       "stalled_partner_max_step_s": blocked,
                       "stalled_steps": [s.get("steps", 0) for s in stalled_steps]}
        if ratio is None or ratio < config.stall_rate_floor:
            failures.append(f"b2 {stalled} stalled: {active} rate ratio {ratio}")
        if not max_step <= config.stall_max_step_s:
            failures.append(f"b2 {stalled} stalled: {active} max step {max_step} s")
        if blocked < config.stall_blocked_fraction * config.stall_s:
            failures.append(f"b2 {stalled} stalled: partner blocked only {blocked} s")

    def other(h):
        return "upper" if h == "lower" else "lower"

    independence = {h: {"a_both_over_a0": _ratio(rates[h].get("a_both"), rates[h].get(f"a0_{h}_alone")),
                        "b1_active_over_a_both": _ratio(
                            rates[h].get(f"b1_{h}_active_{other(h)}_idle"), rates[h].get("a_both")),
                        "c_active_over_a_both": _ratio(
                            rates[h].get(f"c_{h}_active_{other(h)}_compiles"), rates[h].get("a_both"))}
                    for h in HALVES}
    compiles = {r["identity"]["rank"]: {j: {"ok": c.get("ok"), "call_s": c.get("call_s"),
                                            "parts": c.get("parts")}
                                        for j, c in r["compiles"].items()} for r in reports}
    if not all(c["ok"] for v in compiles.values() for c in v.values()):
        failures.append("a background compile failed")
    result = {"status": "passed" if not failures else "failed", "failures": failures,
              "rates_steps_per_s": rates, "independence": independence, "b2": b2,
              "compiles": compiles,
              "cross_half_max_abs_diff": {k: v["cross_half_max_abs_diff"]
                                          for k, v in progress["check_rounds"].items()},
              "cross_half_bitwise_equal": {k: v["all_hashes_equal"]
                                           for k, v in progress["check_rounds"].items()}}
    if config.u1_calibrate_target:
        impact = overlap_impact(halves, "small", "e_baseline", "e_target_compiles", "e_post", "target")
        result["e_target_compile_impact"] = impact
        result["e_overlap_evidence_problems"] = _overlap_problems(config, impact)  # data only in U1
    return result


def verdict_u2(config, progress):
    reports = progress["workers"]
    halves = _by_half(reports)
    checks = progress["check_rounds"]
    failures = []
    setup = progress["setup"]
    bytes_per_chip = min(v["weight_bytes_per_chip"] for v in setup.values())
    if bytes_per_chip < config.min_weight_bytes_per_chip:
        failures.append(f"weights {bytes_per_chip} B per chip < {config.min_weight_bytes_per_chip}")
    for label, require_all in (("setup", False), ("target_first", True), ("reverse_first", False),
                               ("final", False)):
        failures += [f"check {label}: {p}" for p in _check_round(
            checks[label], config.output_atol, False, require_all)]
    sums = {label: {(v["w_up"], v["w_down"]) for v in checks[label].values()}
            for label in ("setup_checksums", "target_checksums", "reverse_checksums")}
    if any(len(s) != 1 for s in sums.values()) or len(set().union(*sums.values())) != 1:
        failures.append(f"weight checksums differ: {sums}")
    forward = list(progress["switch_forward"].values())
    worst = max(r["max_abs_diff_vs_source_output"] for r in forward)
    if not worst <= config.output_atol:
        failures.append(f"TP16 output differs from TP8 by {worst} > {config.output_atol}")
    reverse = list(progress["switch_reverse"].values())
    if not all(r["equals_pre_switch_half_output"] for r in reverse):
        failures.append("2x TP8 outputs after the reverse switch differ from before")
    if not all(v["equals_setup_output"] for v in checks["final"]["ranks"].values()):
        failures.append("final half outputs differ from setup outputs")
    expect_no_compile = config.reshard == "aot" or (config.target_order == "local"
                                                    and config.warm_device_put)
    for name in ("switch_forward_segments", "switch_reverse_segments"):
        compiles = progress[name]["backend_compiles_during_switch"]
        if expect_no_compile and compiles:
            failures.append(f"{name}: {compiles} backend compiles on the critical path")
    compiles = {}
    for r in reports:
        c = r["compiles"].get("target", {})
        compiles[r["identity"]["rank"]] = {"ok": c.get("ok"), "call_s": c.get("call_s"),
                                           "parts": c.get("parts"),
                                           "whole_process_cpu_s": c.get("whole_process_cpu_s")}
    if not all(v["ok"] for v in compiles.values()):
        failures.append("target compile failed")
    impact = overlap_impact(halves, "model", "baseline", "overlap", "postcontrol", "target")
    for h in HALVES:
        impact[h]["post_reverse_steps_per_s"] = _rate(halves[h], "model/post_reverse")
    overlap_problems = _overlap_problems(config, impact)
    status = "failed" if failures else ("inconclusive_overlap_too_short" if overlap_problems
                                        else "passed")
    return {"status": status, "failures": failures, "overlap_evidence_problems": overlap_problems,
            "weight_bytes_per_chip_tp8": bytes_per_chip, "target_compile": compiles,
            "serving_impact": impact, "target_steps_per_s": _rate(reports, "target/target"),
            "forward": progress["switch_forward_segments"],
            "reverse": progress["switch_reverse_segments"],
            "max_abs_diff_tp16_vs_tp8": worst,
            "cross_half_bitwise_equal_setup": checks["setup"]["all_hashes_equal"],
            "cross_half_max_abs_diff_setup": checks["setup"]["cross_half_max_abs_diff"]}


# ----------------------------------------------------------------------------
# CPU emulation fleet: 4 processes x 4 CPU devices, one jax.distributed runtime.


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cpu_worker_main(rank, processes, devices, port, config, conn):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    import jax
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(f"127.0.0.1:{port}", num_processes=processes, process_id=rank,
                               initialization_timeout=120)
    worker = UnifiedWorker(UnifiedConfig(**config), "cpu")
    conn.send((True, "ready"))
    while True:
        message = conn.recv()
        if message[0] == "__shutdown__":
            break
        method, args = message
        try:
            conn.send((True, getattr(worker, method)(*args)))
        except Exception:
            conn.send((False, traceback.format_exc()))
    worker.shutdown()
    conn.send((True, "bye"))
    jax.distributed.shutdown()


class CpuFleet:
    def __init__(self, config, processes=4, devices=4, timeout_s=180):
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        port = _free_port()
        self.ranks = list(range(processes))
        self.conns, self.procs = [], []
        for rank in self.ranks:
            parent, child = context.Pipe()
            process = context.Process(target=_cpu_worker_main, daemon=True, args=(
                rank, processes, devices, port, dataclasses.asdict(config), child))
            process.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(process)
        for conn in self.conns:
            if not conn.poll(timeout_s):
                self.close()
                raise TimeoutError("CPU worker did not initialize")
            ok, value = conn.recv()
            if not ok:
                raise RuntimeError(value)

    def call(self, calls, timeout_s):
        for rank, method, args in calls:
            self.conns[rank].send((method, args))
        results = []
        for rank, method, _ in calls:
            if not self.conns[rank].poll(timeout_s):
                raise TimeoutError(f"rank {rank} {method} timed out")
            ok, value = self.conns[rank].recv()
            if not ok:
                raise RuntimeError(f"rank {rank} {method}:\n{value}")
            results.append(value)
        return results

    def close(self):
        for conn in self.conns:
            try:
                conn.send(("__shutdown__",))
            except (BrokenPipeError, OSError):
                pass
        for process in self.procs:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
