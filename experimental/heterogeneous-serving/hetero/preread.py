"""Parked-target pre-read (design A): a CPU-only child process writes this rank's TP snapshot.

A pre-spawned target worker (hetero/prespawn.py) parks before its first device touch. With engine
setting ``preread`` it first starts ``python -m hetero.preread --spec SPEC.json``. The child is
forced onto JAX's CPU backend. It runs tpu_inference's own abstract-model path
(``nnx.eval_shape`` of the model class under a mesh, as ``model_loader._get_nnx_model``) and the
real ``loader.load_weights(model, model_config)``. ``assign_and_shard_param`` is intercepted in
``tpu_inference.models.jax.utils.weight_utils`` and in
``tpu_inference.layers.jax.quantization.unquantized``, which imports it by name. Each call:
computes the param's sharding exactly as the original (``shard_put``), maps it on a CPU mesh
whose position k stands for TPU device ``table[k]``, slices only this rank's positions from the
converted host array, writes those shards immediately (``weight_snapshot.write_shard``), and
drops the converted buffer. The param stays abstract and is marked ``_is_loaded``.

The child then publishes a ``weight_snapshot`` v2 manifest, so the released worker takes the
existing restore path (``weight_snapshot='read'``). Rows are in
``tree_flatten_with_path(nnx.state(abstract_model))`` order, which is how both ``validate_tree``
and ``snapshot(runner.state)`` order them. The published specs are those the normal loader's
``create_jit_model`` (a donated ``nnx.jit``) returns: one abstract identity jit over all leaves,
compiled on the CPU mesh (nothing is allocated), gives the canonical output shardings.

The parent never imports a JAX backend for this. It derives the TPU device table from the
instance payload (``device_table``), pickles its own ``vllm_config`` (the engine's, as Ray
delivered it to the worker), and supplies the snapshot generation and context.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

MESH_AXES_2D = ("data", "model")  # tpu_inference.layers.common.sharding.MESH_AXIS_NAMES_2D
V5_LITE_KINDS = ("TPU v5 lite", "TPU v5e")  # jax mesh_utils._TPU_V5_LITE / _TPU_V5E
WATCHED_PATHS = ("/dev/vfio", "/dev/accel", "/tmp/libtpu_lockfile")
WATCHED_LIBS = ("libtpu.so",)
SELF_CHECK_EVERY_S = 10.0
SAMPLE_EVERY_S = 1.0
PR_SET_PDEATHSIG = 1


# --------------------------------------------------------------------------- device table (deviceless)

@dataclasses.dataclass(frozen=True)
class _StubDevice:
    """What jax.experimental.mesh_utils reads of a TPU device; never a real device."""
    id: int
    coords: tuple
    process_index: int
    core_on_chip: int = 0
    device_kind: str = "TPU v5 lite"
    platform: str = "tpu"
    slice_index: int = 0


def _bounds(value: str) -> tuple:
    parts = tuple(int(v) for v in value.replace("x", ",").split(","))
    return parts + (1,) * (3 - len(parts))


def device_table(payload: Dict[str, Any], *, device_kind: str = "TPU v5 lite") -> Dict[str, Any]:
    """The TPU mesh the runner will build, derived from the instance payload alone (no JAX backend).

    Device identity:
    - The runtime rebases chip coordinates on the instance's own origin
      (``tpu_worker._collect_device_evidence``).
    - A v5e subslice numbers devices ``x + X*(y + Y*z)`` within ``TPU_TOPOLOGY`` X x Y (x Z).
    - ``process_index`` is the host's rank in ``host_ips``, which equals the
      ``TPU_PROCESS_ADDRESSES`` order (checked by the worker).
    - ``jax.devices()`` is ordered by ``(process_index, id)``.

    Mesh, as tpu_inference @ 8d31f0866 builds it:
    - ``runner/tpu_runner.py`` ``_create_2d_mesh``: shape ``(model_dp_size, tp_size)`` = ``(1, tp)``,
      axes ``('data', 'model')``.
    - ``utils.make_optimized_mesh``: devices sorted by coords, then ``jax.make_mesh``.
    - ``jax.make_mesh`` is ``mesh_utils.create_device_mesh`` with ``allow_split_physical_axes`` set
      for v5 lite.

    Refused (never guessed): ``NEW_MODEL_DESIGN``, data parallelism, or a payload whose topology
    disagrees with its chip coordinates.
    """
    from jax._src import mesh_utils

    if os.environ.get("NEW_MODEL_DESIGN", "").lower() in ("1", "true"):
        raise ValueError("NEW_MODEL_DESIGN meshes are not derived here")
    env = payload["runtime_env"]
    hosts: List[str] = list(payload["host_ips"])
    tp = int(payload["tp_size"])
    coords = {h: [tuple(int(v) for v in c) for c in payload["chip_coords_by_host"][h]] for h in hosts}
    every = [c for h in hosts for c in coords[h]]
    origin = tuple(min(c[a] for c in every) for a in range(3))
    dims = _bounds(env["TPU_TOPOLOGY"])
    box = tuple(max(c[a] for c in every) - origin[a] + 1 for a in range(3))
    if box != dims or len(every) != tp or len(set(every)) != tp:
        raise ValueError(f"payload chips {sorted(every)} do not form the {env['TPU_TOPOLOGY']} topology "
                         f"of {tp} chips")
    per_host = _bounds(env["TPU_CHIPS_PER_HOST_BOUNDS"])
    if any(len(coords[h]) != per_host[0] * per_host[1] * per_host[2] for h in hosts):
        raise ValueError("chips per host disagree with TPU_CHIPS_PER_HOST_BOUNDS")
    addresses = [a.rsplit(":", 1)[0] for a in env["TPU_PROCESS_ADDRESSES"].split(",")]
    if addresses != hosts:
        raise ValueError("TPU_PROCESS_ADDRESSES order does not match host_ips")
    devices = []
    for rank, host in enumerate(hosts):
        for c in coords[host]:
            rel = tuple(c[a] - origin[a] for a in range(3))
            devices.append(_StubDevice(id=rel[0] + dims[0] * (rel[1] + dims[1] * rel[2]), coords=rel,
                                       process_index=rank, device_kind=device_kind))
    devices.sort(key=lambda d: (d.process_index, d.id))  # jax.devices() order
    mesh_shape = (1, tp)
    ordered = sorted(devices, key=lambda d: d.coords)   # make_optimized_mesh
    mesh = mesh_utils.create_device_mesh(mesh_shape, ordered,
                                         allow_split_physical_axes=device_kind in V5_LITE_KINDS)
    flat = list(mesh.flat)
    table = [{"id": int(d.id), "process_index": int(d.process_index), "platform": "tpu"} for d in flat]
    local_positions = {str(r): [k for k, d in enumerate(flat) if d.process_index == r] for r in range(len(hosts))}
    return {"devices": table, "mesh_shape": list(mesh_shape), "mesh_axes": list(MESH_AXES_2D),
            "coords": [list(d.coords) for d in flat], "device_kind": device_kind,
            "local_positions": local_positions,
            "local_device_ids": {r: sorted(table[k]["id"] for k in ks) for r, ks in local_positions.items()}}


# --------------------------------------------------------------------------- parent side (deviceless)

def force_tpu_platform():
    """Make vllm's ``current_platform`` the TPU platform without probing for libtpu.

    vllm resolves its platform lazily by importing ``libtpu`` (vllm/platforms/__init__.py,
    tpu_platform_plugin); the pre-read child must not import it. Call before any vllm module
    binds ``current_platform`` (``from vllm.platforms import current_platform``).
    """
    import vllm.platforms as platforms
    from vllm.utils.import_utils import resolve_obj_by_qualname

    cls = resolve_obj_by_qualname("tpu_inference.platforms.tpu_platform.TpuPlatform")
    if not isinstance(platforms._current_platform, cls):
        platforms._current_platform = cls()
    return platforms._current_platform


def child_env(base: Optional[Dict[str, str]] = None, *, tp: int, package_root: str) -> Dict[str, str]:
    """Sanitized child environment: no TPU runtime, Ray, XLA or JAX settings of the parent; CPU only."""
    base = dict(os.environ if base is None else base)
    dropped = ("TPU_", "CLOUD_TPU", "MEGASCALE", "LIBTPU", "JAX_", "XLA_", "RAY_", "TPU_INFERENCE_")
    env = {k: v for k, v in base.items() if not k.startswith(dropped)}
    env.update({
        "JAX_PLATFORMS": "cpu",
        "TPU_SKIP_MDS_QUERY": "1",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={tp}",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "JAX_COMPILATION_CACHE_DIR": "",
        "PYTHONPATH": os.pathsep.join([package_root] + [p for p in base.get("PYTHONPATH", "").split(os.pathsep) if p]),
        "PYTHONUNBUFFERED": "1",
        "OMP_HETERO_PREREAD": "1",
    })
    return env


def write_spec(directory: Path, *, vllm_config: Any, rank: int, table: Dict[str, Any], out_dir: str,
               generation: str, context: Dict[str, Any], parent_pid: Optional[int] = None,
               nice: int = 10, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Spec + pickled vllm_config for the child; returns the spec path.

    The worker's own ``vllm_config`` is pickled rather than rebuilt in the child. Rebuilding would
    rerun ``EngineArgs.create_engine_config`` (HF config download/parse and platform checks) and
    could drift from the engine. Pickling needs no JAX backend: the config holds no device data at
    ``init_device`` time. The Ray placement group is dropped from the copy, so the child never
    needs Ray.
    """
    import copy

    import cloudpickle

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.copy(vllm_config)
    config.parallel_config = copy.copy(vllm_config.parallel_config)
    config.parallel_config.placement_group = None
    (directory / "vllm_config.pkl").write_bytes(cloudpickle.dumps(config))
    spec = {"vllm_config": str(directory / "vllm_config.pkl"), "rank": int(rank), "table": table,
            "out_dir": str(out_dir), "generation": generation, "context": context,
            "report": str(directory / "report.json"), "parent_pid": parent_pid or os.getpid(),
            "nice": nice, **(extra or {})}
    path = directory / "spec.json"
    path.write_text(json.dumps(spec, sort_keys=True))
    (directory / "report.json").unlink(missing_ok=True)
    return path


class PrereadChild:
    """Handle of a started child: wait (bounded), kill, reap, report."""

    def __init__(self, process, spec_path: Path, log_path: Path, started_at: float):
        self.process, self.spec_path, self.log_path, self.started_at = process, spec_path, log_path, started_at
        self.ended_at: Optional[float] = None
        self.killed = False

    @property
    def report_path(self) -> Path:
        return self.spec_path.parent / "report.json"

    def wait(self, timeout_s: float) -> Dict[str, Any]:
        """Wait at most ``timeout_s``; kill + reap on timeout. Returns the outcome record."""
        import subprocess
        error = None
        try:
            self.process.wait(timeout=max(0.0, timeout_s))
        except subprocess.TimeoutExpired:
            error = f"pre-read child timed out after {timeout_s:.1f} s"
            self.kill()
        self.ended_at = self.ended_at or time.time()
        report = None
        try:
            report = json.loads(self.report_path.read_text())
        except (OSError, ValueError) as exc:
            error = error or f"pre-read child left no report: {exc}"
        code = self.process.returncode
        if error is None and code != 0:
            error = f"pre-read child exited with {code}"
        if error is None and not (report or {}).get("ok"):
            error = f"pre-read child reported failure: {(report or {}).get('error')}"
        return {"ok": error is None, "error": error, "exit_code": code, "pid": self.process.pid,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "killed": self.killed, "log": str(self.log_path), "report": report}

    def kill(self) -> None:
        if self.process.poll() is None:
            self.killed = True
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
        with contextlib.suppress(Exception):
            self.process.wait(timeout=30)
        self.ended_at = self.ended_at or time.time()


def gate_outcome(outcome: Dict[str, Any]) -> Optional[str]:
    """None if the worker may read the pre-read snapshot, else the reason it refuses (and falls back).

    Requires a zero exit, an ``ok`` report and clean isolation evidence (CPU-only platforms, clean
    self-checks, no audited TPU opens). A refusal makes ``prepare`` refuse locally, so the
    all-rank vote falls back to the normal loader.
    """
    if not outcome.get("ok"):
        return outcome.get("error") or "pre-read failed"
    isolation = (outcome.get("report") or {}).get("isolation") or {}
    if not (isolation.get("platforms_cpu_only") and isolation.get("self_checks_clean")
            and not isolation.get("audit_events")):
        return "pre-read isolation evidence not clean"
    return None


def start_child(spec_path: Path, *, tp: int, python: Optional[str] = None,
                package_root: Optional[str] = None) -> PrereadChild:
    """``python -m hetero.preread --spec SPEC``.

    The child sets its own ``PR_SET_PDEATHSIG`` (SIGKILL) and verifies its parent pid, and lowers
    its own priority, so no ``preexec_fn`` runs inside the multithreaded Ray worker. stdout and
    stderr go to ``preread.log`` next to the spec.
    """
    import subprocess

    spec_path = Path(spec_path)
    package_root = package_root or str(Path(__file__).resolve().parents[1])
    log_path = spec_path.parent / "preread.log"
    started = time.time()
    with open(log_path, "wb") as log:
        process = subprocess.Popen([python or sys.executable, "-m", "hetero.preread", "--spec", str(spec_path)],
                                   env=child_env(tp=tp, package_root=package_root), cwd=package_root,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   close_fds=True)
    return PrereadChild(process, spec_path, log_path, started)


# --------------------------------------------------------------------------- child: evidence

class _Audit:
    """``sys.addaudithook`` record of TPU-relevant opens and dlopens (``events``: any is a failure)
    and of attempted ``libtpu`` / ``tpu_raiden`` imports (``imports``: evidence only; tpu_inference
    tries ``tpu_raiden`` at import, absent from the worker image). An audit hook cannot be removed.
    Native dlopens by C extensions bypass audit hooks; ``prespawn.self_check`` (mapped libtpu.so)
    covers those."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []

    def __call__(self, event: str, args: tuple) -> None:
        try:
            if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                path = os.fsdecode(args[0])
                if path.startswith(WATCHED_PATHS) or any(lib in path for lib in WATCHED_LIBS):
                    self.events.append({"event": event, "path": path, "at": time.time()})
            elif event == "ctypes.dlopen" and args and args[0]:
                path = os.fsdecode(args[0]) if isinstance(args[0], (str, bytes)) else str(args[0])
                if any(lib in path for lib in WATCHED_LIBS):
                    self.events.append({"event": event, "path": path, "at": time.time()})
            elif event == "import" and args and isinstance(args[0], str) and (
                    args[0] == "libtpu" or args[0].startswith(("libtpu.", "tpu_raiden"))):
                self.imports.append({"event": event, "module": args[0], "at": time.time()})
        except Exception:  # an audit hook must never raise into the audited call
            pass


def _meminfo() -> Dict[str, int]:
    rows = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        rows[key] = int(value.split()[0])
    return rows


def _status(field: str) -> Optional[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    return None


class _Sampler(threading.Thread):
    """1 s meminfo + /dev/shm samples; ``prespawn.self_check`` every 10 s; parent-death watchdog."""

    KEYS = ("MemAvailable", "Cached", "Shmem", "Dirty", "Writeback")

    def __init__(self, parent_pid: Optional[int]):
        super().__init__(daemon=True, name="omp-preread-sampler")
        self.parent_pid = parent_pid
        self.samples: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.stop = threading.Event()

    def sample(self) -> None:
        import shutil
        info = _meminfo()
        row = {k: info.get(k) for k in self.KEYS}
        row["shm_free_bytes"] = shutil.disk_usage("/dev/shm").free
        row["at"] = time.time()
        self.samples.append(row)

    def check(self) -> None:
        from .prespawn import self_check
        self.checks.append(self_check())

    def run(self) -> None:
        last_check = time.monotonic()
        while not self.stop.wait(SAMPLE_EVERY_S):
            if self.parent_pid and os.getppid() != self.parent_pid:
                os._exit(97)  # orphaned: the parent worker is gone (PDEATHSIG backstop)
            self.sample()
            if time.monotonic() - last_check >= SELF_CHECK_EVERY_S:
                self.check()
                last_check = time.monotonic()

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"samples": len(self.samples), "interval_s": SAMPLE_EVERY_S}
        for key in self.KEYS:
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if values:
                out[f"{key}_kb"] = {"min": min(values), "max": max(values)}
        free = [s["shm_free_bytes"] for s in self.samples]
        out["shm_free_min_bytes"] = min(free) if free else None
        return out


# --------------------------------------------------------------------------- child: writer

class Writer:
    """``assign_and_shard_param`` replacement: shard this rank's positions to files immediately."""

    def __init__(self, mesh, table: Dict[str, Any], rank: int, out_dir: Path, var_numbers: Dict[int, int],
                 *, fault_after_calls: Optional[int] = None, pause_after_calls: Optional[list] = None):
        self.mesh, self.table, self.rank, self.out_dir = mesh, table, rank, out_dir
        self.positions = list(table["local_positions"][str(rank)])
        self.var_numbers = var_numbers
        self.fault_after_calls = fault_after_calls  # tests only: fail mid-load
        self.pause_after_calls = pause_after_calls  # tests only: [calls, seconds] stall mid-load
        self.rows: Dict[int, Dict[str, Any]] = {}
        self.stats = {"calls": 0, "reassigned": 0, "bytes": 0, "files": 0, "reused_files": 0,
                      "hook_s": 0.0, "slice_s": 0.0, "write_s": 0.0}

    def target_sharding(self, jax_param, jax_weight, mesh):
        """``assign_and_shard_param`` + ``shard_put`` sharding resolution, without placement."""
        import math

        from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding, get_mesh

        spec = jax_param.get_metadata().get("out_sharding", ())
        if isinstance(spec, NamedSharding):
            spec = spec.spec
        elif isinstance(spec, SingleDeviceSharding):
            spec = ()
        param_mesh = jax_param.get_metadata().get("mesh") or mesh
        if param_mesh is None:
            param_mesh = get_mesh()
        if math.prod(param_mesh.axis_sizes) == 1:
            raise RuntimeError("single-device meshes are not pre-read")
        if param_mesh is not self.mesh and (list(param_mesh.devices.flat) != list(self.mesh.devices.flat)
                                            or param_mesh.axis_names != self.mesh.axis_names):
            raise RuntimeError("parameter mesh differs from the pre-read mesh")
        if spec is None:
            spec = ()
        if isinstance(spec, tuple):
            return NamedSharding(param_mesh, P(*spec))
        if isinstance(spec, P):
            return NamedSharding(param_mesh, spec)
        if isinstance(spec, NamedSharding):
            return spec
        raise RuntimeError(f"unsupported parameter sharding {spec!r}")

    def __call__(self, jax_param, jax_weight, param_name: str = "Unknown", mesh=None) -> None:
        import numpy as np

        from .weight_snapshot import _index, write_shard

        begin = time.perf_counter()
        try:
            if self.fault_after_calls is not None and self.stats["calls"] >= self.fault_after_calls:
                raise RuntimeError("injected pre-read fault")
            if self.pause_after_calls and self.stats["calls"] == self.pause_after_calls[0]:
                time.sleep(float(self.pause_after_calls[1]))
            number = self.var_numbers.get(id(jax_param))
            if number is None:
                raise RuntimeError("parameter is not a leaf of the abstract model state")
            sharding = self.target_sharding(jax_param, jax_weight, mesh)
            shape = tuple(jax_weight.shape)
            abstract = jax_param.get_value()
            if tuple(abstract.shape) != shape or str(abstract.dtype) != str(jax_weight.dtype):
                raise RuntimeError(f"converted {shape}/{jax_weight.dtype} does not match the parameter "
                                   f"{tuple(abstract.shape)}/{abstract.dtype}")
            indices = sharding.devices_indices_map(shape)
            flat = list(self.mesh.devices.flat)
            tick = time.perf_counter()
            host = np.asarray(jax_weight)  # CPU backend: a view of the converted buffer
            self.stats["slice_s"] += time.perf_counter() - tick
            row = {"spec": sharding.spec, "shape": list(shape), "dtype": str(jax_weight.dtype), "shards": []}
            if number in self.rows:
                self.stats["reassigned"] += 1
            for shard_number, position in enumerate(self.positions):
                index = indices[flat[position]]
                tick = time.perf_counter()
                local = host[index]
                self.stats["slice_s"] += time.perf_counter() - tick
                part = {"file": f"{number:05d}-{shard_number:02d}.raw",
                        "device_id": int(self.table["devices"][position]["id"]), "index": _index(index),
                        "shape": list(local.shape), "bytes": int(local.nbytes)}
                tick = time.perf_counter()
                reused = write_shard(self.out_dir, part, local)
                self.stats["write_s"] += time.perf_counter() - tick
                self.stats["bytes"] += part["bytes"]
                self.stats["files"] += 1
                self.stats["reused_files"] += int(reused)
                row["shards"].append(part)
                del local
            del host
            self.rows[number] = row
            # Keep the param abstract; the original assigns the device array here.
            jax_param.set_metadata("_is_loaded", True)
            self.stats["calls"] += 1
        except Exception as exc:
            raise RuntimeError(f"pre-read of weight '{param_name}' failed: {exc}") from exc
        finally:
            del jax_weight
            self.stats["hook_s"] += time.perf_counter() - begin


def _leaf_graph_path(keypath) -> tuple:
    """Graph path of a ``tree_flatten_with_path(nnx.state(...))`` leaf: the keys before ``.value``."""
    keys = []
    for key in keypath:
        for attr in ("key", "idx", "name"):
            if hasattr(key, attr):
                keys.append(getattr(key, attr))
                break
    if keys and keys[-1] == "value":
        keys.pop()
    return tuple(keys)


def state_leaves(model):
    """(keystr paths, abstract leaves, id(Variable) -> leaf number) of ``nnx.state(model)``.

    Leaf order is ``tree_flatten_with_path(nnx.state(model))``: the order ``validate_tree`` reads and
    ``snapshot(runner.state)`` writes. Variables are matched to leaves by graph path.
    """
    import jax
    from flax import nnx

    flat = jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]
    by_graph = {_leaf_graph_path(keypath): number for number, (keypath, _) in enumerate(flat)}
    if len(by_graph) != len(flat):
        raise RuntimeError("ambiguous state leaf graph paths")
    numbers = {}
    for path, node in nnx.iter_graph(model):
        if isinstance(node, nnx.Variable):
            number = by_graph.get(tuple(path))
            if number is not None:
                numbers[id(node)] = number
    return [jax.tree_util.keystr(k) for k, _ in flat], [leaf for _, leaf in flat], numbers


def canonical_specs(mesh, leaves: Sequence[Any], specs: Sequence[Any]) -> List[Any]:
    """Output specs of the normal loader's ``create_jit_model`` (donated jit of the state).

    One identity jit over all leaves, lowered and compiled abstractly on the CPU mesh (nothing is
    allocated). JAX canonicalizes output NamedShardings (e.g. trailing ``None`` axes dropped), and
    ``snapshot(runner.state)`` records those.
    """
    import jax
    from jax.sharding import NamedSharding

    args = [jax.ShapeDtypeStruct(tuple(a.shape), a.dtype, sharding=NamedSharding(mesh, s))
            for a, s in zip(leaves, specs)]
    compiled = jax.jit(lambda xs: xs, donate_argnums=0).lower(args).compile()
    result = [s.spec for s in jax.tree_util.tree_leaves(compiled.output_shardings,
                                                        is_leaf=lambda x: isinstance(x, NamedSharding))]
    if len(result) != len(args):
        raise RuntimeError("canonical spec count mismatch")
    return result


def run(spec: Dict[str, Any], audit: _Audit, sampler: _Sampler) -> Dict[str, Any]:
    """The pre-read itself. Publishes nothing unless every state leaf was written."""
    clocks: Dict[str, Dict[str, float]] = {}

    def clock(name):
        @contextlib.contextmanager
        def timed():
            start = time.time()
            try:
                yield
            finally:
                clocks[name] = {"start": start, "end": time.time()}
        return timed()

    with clock("imports"):
        force_tpu_platform()
        import cloudpickle
        import jax
        import numpy as np
        from flax import nnx
        from jax.sharding import AxisType, Mesh
        from vllm.config import set_current_vllm_config

        from tpu_inference.distributed import jax_parallel_state
        from tpu_inference.models.common import model_loader
        from tpu_inference.models.jax.utils import weight_utils
        import tpu_inference.layers.jax.quantization.unquantized as unquantized

        from .weight_snapshot import _devices, _spec, open_for_write, publish_manifest

    table = spec["table"]
    rank = int(spec["rank"])
    out_dir = Path(spec["out_dir"])
    tp = int(np.prod(table["mesh_shape"]))
    with clock("config"):
        vllm_config = cloudpickle.loads(Path(spec["vllm_config"]).read_bytes())
    cpu = jax.devices("cpu")
    if len(cpu) < tp:
        raise RuntimeError(f"{len(cpu)} CPU devices for a {tp}-device mesh (XLA_FLAGS device count)")
    # Position k of this mesh stands for TPU device table[k]: specs and shard indices depend on
    # mesh position only.
    mesh = Mesh(np.array(cpu[:tp]).reshape(table["mesh_shape"]), tuple(table["mesh_axes"]),
                axis_types=(AxisType.Auto,) * len(table["mesh_shape"]))
    # tpu_worker.init_device (pipeline_parallel_size 1): the PP group the model constructor reads.
    jax_parallel_state.init_pp_distributed_environment("", 0, 1, cpu[0], need_pp=False)

    lock = open_for_write(out_dir)  # invalidates any manifest before a byte is written
    try:
        state: Dict[str, Any] = {}
        originals = [(weight_utils, "assign_and_shard_param", weight_utils.assign_and_shard_param),
                     (unquantized, "assign_and_shard_param", unquantized.assign_and_shard_param)]
        original_builder = model_loader._get_nnx_model

        class _Loaded(Exception):
            """Raised by the builder after load_weights: get_flax_model stops before any jit."""

        def builder(model_class, config, rng, mesh_, pooler=None, is_draft_model=False):
            # tpu_inference _get_nnx_model, load_format 'auto', LoadableWithIterator branch
            # (model_loader.py:247-285 @ 8d31f0866) up to create_jit_model, which needs devices.
            if (is_draft_model or pooler is not None or config.model_config.quantization is not None
                    or config.load_config.load_format in ("dummy", "jax_dummy", "pathways_dummy")
                    or model_loader.apply_qwix_on_abstract_model(config)
                    or getattr(model_class, "_self_manages_sharding", False)):
                raise RuntimeError("only the plain checkpoint loader path is pre-read")
            with clock("graph_build"), jax.set_mesh(mesh_):
                model = nnx.eval_shape(lambda: model_class(config, rng, mesh_))
            if not isinstance(model, model_loader.LoadableWithIterator):
                raise RuntimeError(f"{model_class.__name__} does not load with an iterator")
            paths, values, numbers = state_leaves(model)
            writer = Writer(mesh_, table, rank, out_dir, numbers,
                            fault_after_calls=spec.get("fault_after_calls"),
                            pause_after_calls=spec.get("pause_after_calls"))
            state.update(model=model, writer=writer, paths=paths, values=values)
            for module, attr, _ in originals:
                setattr(module, attr, writer)
            with clock("weight_load"), jax.set_mesh(mesh_):
                loader = model_loader.get_model_loader(config.load_config)
                loader.load_weights(model, config.model_config)
            raise _Loaded()

        try:
            model_loader._get_nnx_model = builder
            with clock("load_model"), set_current_vllm_config(vllm_config):
                try:
                    # get_model -> get_flax_model: model dtype / quant config exactly as the runner.
                    model_loader.get_model(vllm_config, jax.random.key(vllm_config.model_config.seed), mesh)
                except _Loaded:
                    pass
                else:
                    raise RuntimeError("the pre-read builder was not used")
        finally:
            model_loader._get_nnx_model = original_builder
            for module, attr, original in originals:
                setattr(module, attr, original)
        writer: Writer = state["writer"]
        paths, values = state["paths"], state["values"]

        with clock("assemble"):
            missing = [paths[n] for n in range(len(paths)) if n not in writer.rows]
            if missing:
                raise RuntimeError(f"{len(missing)} state leaves were never loaded: {missing[:5]}")
            rows = [writer.rows[n] for n in range(len(paths))]
        with clock("canonical_specs"):
            specs = canonical_specs(mesh, values, [r["spec"] for r in rows])
        with clock("publish"):
            manifest = {"version": 2, "context": spec["context"], "generation": spec["generation"],
                        "checksum_kind": "none", "arrays": []}
            for path, row, canon in zip(paths, rows, specs):
                manifest["arrays"].append({
                    "path": path, "shape": row["shape"], "dtype": row["dtype"],
                    "mesh_shape": list(table["mesh_shape"]), "mesh_axes": list(table["mesh_axes"]),
                    "devices": list(table["devices"]), "spec": _spec(canon), "shards": row["shards"]})
            publish_manifest(out_dir, manifest)
    finally:
        lock.close()
    duration = lambda name: (clocks[name]["end"] - clocks[name]["start"]) if name in clocks else None
    load = duration("weight_load")
    return {"arrays": len(manifest["arrays"]), "bytes": writer.stats["bytes"], "directory": str(out_dir),
            "generation": spec["generation"], "writer": writer.stats, "clocks": clocks,
            "phases_s": {"imports": duration("imports"), "config": duration("config"),
                         "graph": duration("graph_build"), "load": load, "write": writer.stats["write_s"],
                         "convert": (load - writer.stats["hook_s"]) if load is not None else None,
                         "canonical_specs": duration("canonical_specs"), "publish": duration("publish")},
            "phase_semantics": "graph = nnx.eval_shape of the model; load = loader.load_weights (checkpoint "
                               "reads + conversion + shard writes); write = shard file allocate/copy/fsync "
                               "inside the hook; convert = load minus all hook time (checkpoint page faults, "
                               "torch reshape/permute, t2j, merges)"}


def _platforms() -> List[str]:
    from jax._src import xla_bridge
    return sorted(xla_bridge._backends)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    audit = _Audit()
    sys.addaudithook(audit)
    begin = time.time()
    spec = json.loads(Path(args.spec).read_text())
    parent = spec.get("parent_pid")
    if sys.platform.startswith("linux"):
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if parent and os.getppid() != parent:
        return 97  # the parent died before PDEATHSIG was armed
    with contextlib.suppress(OSError):
        os.nice(int(spec.get("nice", 10)))
    with contextlib.suppress(OSError):
        # Under memory pressure the kernel must pick this child, never a serving S engine.
        Path("/proc/self/oom_score_adj").write_text("1000")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("pre-read child requires JAX_PLATFORMS=cpu")

    sampler = _Sampler(parent)
    sampler.check()
    sampler.sample()
    sampler.start()
    report: Dict[str, Any] = {"ok": False, "pid": os.getpid(), "rank": spec["rank"], "started_at": begin,
                              "env": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "TPU_SKIP_MDS_QUERY",
                                                                     "XLA_FLAGS", "JAX_ENABLE_COMPILATION_CACHE")}}
    code = 1
    try:
        report.update(run(spec, audit, sampler))
        report["ok"] = True
        code = 0
    except BaseException as exc:  # report every failure, then exit non-zero
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-4000:]
    finally:
        sampler.stop.set()
        sampler.sample()
        sampler.check()
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        platforms = _platforms() if "jax" in sys.modules else []
        report.update({
            "ended_at": time.time(), "wall_s": time.time() - begin,
            "vm_hwm_kb": _status("VmHWM"), "cpu_user_s": usage.ru_utime, "cpu_sys_s": usage.ru_stime,
            "ru_majflt": usage.ru_majflt, "ru_minflt": usage.ru_minflt,
            "meminfo": sampler.summary(),
            "isolation": {"self_checks": sampler.checks,
                          "self_checks_clean": all(c["clean"] for c in sampler.checks),
                          "audit_events": audit.events, "audit_imports": audit.imports,
                          "jax_platforms": platforms, "platforms_cpu_only": platforms == ["cpu"]},
        })
        if report["ok"] and not (report["isolation"]["self_checks_clean"] and not audit.events
                                 and platforms == ["cpu"]):
            # Isolation failure fails the pre-read: withdraw the published manifest.
            report["ok"] = False
            report["error"] = "isolation evidence not clean"
            code = 1
            with contextlib.suppress(OSError, ValueError):
                from .weight_snapshot import open_for_write
                open_for_write(Path(spec["out_dir"])).close()
        path = Path(spec["report"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, sort_keys=True, default=str))
        os.replace(temporary, path)
    return code


if __name__ == "__main__":
    sys.exit(main())
