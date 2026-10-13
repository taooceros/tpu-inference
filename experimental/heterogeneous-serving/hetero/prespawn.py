"""Pre-spawned target engines: CPU-only bring-up up to the first device touch, then a release gate.

A pre-spawned target engine (driver actor + one RayWorkerProc per host) is started while the
source engines still own the TPUs on the same hosts.  Its placement group carries no TPU
resource and its worker actors request none (``prespawn_executor_class``), so Ray schedules it
next to the source.  Everything up to ``HeteroTPUWorker.init_device``'s ``super().init_device()``
-- imports, vLLM config, executor and worker construction -- runs during the source phase; the
first device touch is ``jax.devices()`` in tpu_inference/worker/tpu_worker.py:410 (installed
v0.26.0).  Each worker parks right before it (``park_and_wait``) on a named ``ReleaseGate`` actor
keyed by its node IP.  The controller releases a host after the source engine on it has been
drained and its placement group removed; the worker then waits, bounded, until no other process
holds libtpu's lockfile (``/tmp/libtpu_lockfile``; libtpu takes a POSIX write lock on it at
backend init) or a TPU device node (``/dev/vfio/*``), and only then initialises the device.

Fail closed everywhere: a parked worker that already mapped libtpu.so or holds a TPU fd raises
(it must never have touched the device), and every wait has a deadline that raises instead of
proceeding.
"""

from __future__ import annotations

import asyncio
import glob
import os
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

LIBTPU_LOCKFILE = "/tmp/libtpu_lockfile"
TPU_DEVICE_GLOBS = ("/dev/vfio/[0-9]*", "/dev/accel*")
GATE_PREFIX = "omp-hetero-release"
DEFAULT_DEVICE_FREE_TIMEOUT_S = 120.0


def gate_name(layout_id: str) -> str:
    """A fresh gate name per pre-spawn: a stale gate of a failed run is never reused."""
    return f"{GATE_PREFIX}--{layout_id}--{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- process introspection

def _realpaths(patterns: Iterable[str]) -> set:
    return {os.path.realpath(p) for pattern in patterns for p in glob.glob(pattern)}


def self_check(*, lockfile: str = LIBTPU_LOCKFILE, device_globs: Iterable[str] = TPU_DEVICE_GLOBS,
               proc_self: str = "/proc/self") -> Dict[str, Any]:
    """What this process holds of the TPU runtime: mapped libtpu.so, TPU device fds, lockfile fds."""
    with open(os.path.join(proc_self, "maps")) as handle:
        mapped = sorted({line.split()[-1] for line in handle
                         if line.split() and os.path.basename(line.split()[-1]) == "libtpu.so"})
    devices, locks = [], []
    lock_path = os.path.realpath(lockfile)
    fd_dir = os.path.join(proc_self, "fd")
    for fd in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.startswith(("/dev/vfio/", "/dev/accel")):
            devices.append(target)
        elif target == lock_path:
            locks.append(target)
    return {"libtpu_mapped": mapped, "tpu_device_fds": sorted(devices), "lockfile_fds": sorted(locks),
            "clean": not (mapped or devices or locks), "at": time.time()}


def assert_untouched(stage: str, **kwargs: Any) -> Dict[str, Any]:
    """Hard failure if this process touched the TPU runtime before the gate opened."""
    check = self_check(**kwargs)
    if not check["clean"]:
        raise RuntimeError(f"pre-spawned worker touched the TPU runtime before the release gate "
                           f"({stage}): {check}")
    return check


def lock_holders(lockfile: str = LIBTPU_LOCKFILE, proc_locks: str = "/proc/locks") -> List[int]:
    """PIDs other than ours holding a lock on ``lockfile`` (by inode, from /proc/locks).

    A holder outside this pid namespace is listed as pid 0: still foreign, still held.
    """
    try:
        inode = os.stat(lockfile).st_ino
    except FileNotFoundError:
        return []
    me = os.getpid()
    holders = set()
    with open(proc_locks) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 6 or "->" in fields[:2]:  # blocked waiters are not holders
                continue
            try:
                pid = int(fields[4])
                lock_inode = int(fields[5].rsplit(":", 1)[1])
            except (ValueError, IndexError):
                continue
            if lock_inode == inode and pid != me:
                holders.add(pid)
    return sorted(holders)


def device_holders(device_globs: Iterable[str] = TPU_DEVICE_GLOBS, proc: str = "/proc") -> List[int]:
    """PIDs other than ours with a TPU device node open."""
    targets = _realpaths(device_globs)
    if not targets:
        return []
    me = str(os.getpid())
    holders = []
    for entry in os.listdir(proc):
        if not entry.isdigit() or entry == me:
            continue
        fd_dir = os.path.join(proc, entry, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:  # exited, or not readable
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) in targets:
                    holders.append(int(entry))
                    break
            except OSError:
                continue
    return sorted(holders)


VFIO_GROUP_GLOB = "/dev/vfio/[0-9]*"


def busy_vfio_groups(group_glob: str = VFIO_GROUP_GLOB) -> List[str]:
    """VFIO groups that cannot be opened yet (EBUSY), probed by an open + immediate close.

    A killed source closes its fds early in exit, but the group stays busy until the kernel's
    release path (device reset) finishes, so /proc shows no holder while libtpu's open still gets
    EBUSY (O2 attempt 2: "open(/dev/vfio/0): 16: Device or resource busy" ~1 s after release).
    A VFIO group admits one opener, so the probe cannot take a group from a live owner; it is only
    run after the lockfile and /proc checks are clear, and the fd is closed before returning.
    """
    import errno

    busy = []
    for path in sorted(glob.glob(group_glob)):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except OSError as exc:
            if exc.errno == errno.EBUSY:
                busy.append(path)
            continue  # absent/unreadable groups are libtpu's error to report, not a wait
        os.close(fd)
    return busy


def wait_device_free(*, timeout_s: float = DEFAULT_DEVICE_FREE_TIMEOUT_S, poll_s: float = 0.02,
                     lockfile: str = LIBTPU_LOCKFILE, device_globs: Iterable[str] = TPU_DEVICE_GLOBS,
                     clock: Callable[[], float] = time.time,
                     sleep: Callable[[float], None] = time.sleep,
                     busy_groups: Callable[[], List[str]] = busy_vfio_groups) -> Dict[str, Any]:
    """Blocks until the TPU is free for this process; raises on timeout.

    Free = no foreign lock on the libtpu lockfile, no foreign TPU fd in /proc, and every VFIO
    group opens (``busy_groups`` empty).  ``lock_free_at`` / ``vfio_free_at`` /
    ``groups_openable_at`` are the instants each condition was last observed to become true (a
    re-acquired lock or a busy group resets its clock).
    """
    globs = tuple(device_globs)
    started = clock()
    seen: Dict[str, List[Any]] = {}
    lock_free_at: Optional[float] = None
    vfio_free_at: Optional[float] = None
    groups_openable_at: Optional[float] = None
    polls = busy_polls = 0
    busy: List[str] = []
    while True:
        polls += 1
        now = clock()
        locks, devices = lock_holders(lockfile), device_holders(globs)
        if locks:
            seen["lock"] = sorted(set(seen.get("lock", [])) | set(locks))
            lock_free_at = None
        elif lock_free_at is None:
            lock_free_at = now
        if devices:
            seen["vfio"] = sorted(set(seen.get("vfio", [])) | set(devices))
            vfio_free_at = None
        elif vfio_free_at is None:
            vfio_free_at = now
        if lock_free_at is not None and vfio_free_at is not None:
            busy = busy_groups()
            if busy:
                busy_polls += 1
                seen["busy_groups"] = sorted(set(seen.get("busy_groups", [])) | set(busy))
                groups_openable_at = None
            else:
                groups_openable_at = clock()
                return {"lock_free_at": lock_free_at, "vfio_free_at": vfio_free_at,
                        "groups_openable_at": groups_openable_at, "polls": polls,
                        "busy_polls": busy_polls, "holders_seen": seen,
                        "device_free_wait_s": clock() - started}
        if now - started > timeout_s:
            raise TimeoutError(f"TPU still held {timeout_s:.0f} s after release: "
                               f"lockfile holders {locks}, device holders {devices}, busy groups {busy}")
        sleep(poll_s)


# --------------------------------------------------------------------------- gate

class ReleaseGate:
    """Controller-side gate (a named, non-detached Ray actor owned by the switch job).

    Workers ``wait(key)`` with their node IP; the controller ``release(keys)`` per host.
    The gate dies with the job, so a parked worker of a dead controller fails instead of hanging.
    """

    def __init__(self, expected_keys: List[str]) -> None:
        self.expected = sorted(set(expected_keys))
        self._events: Dict[str, asyncio.Event] = {}
        self.parked_at: Dict[str, float] = {}
        self.released_at: Dict[str, float] = {}
        self.returned_at: Dict[str, float] = {}

    def _event(self, key: str) -> asyncio.Event:
        if key not in self._events:
            self._events[key] = asyncio.Event()
        return self._events[key]

    async def wait(self, key: str, timeout_s: float) -> Dict[str, Any]:
        if key not in self.expected:
            raise ValueError(f"gate key {key} is not one of the pre-spawned hosts {self.expected}")
        self.parked_at.setdefault(key, time.time())
        await asyncio.wait_for(self._event(key).wait(), timeout_s)
        self.returned_at.setdefault(key, time.time())
        return {"key": key, "parked_at": self.parked_at[key], "released_at": self.released_at[key]}

    async def release(self, keys: List[str]) -> Dict[str, float]:
        unknown = set(keys) - set(self.expected)
        if unknown:
            raise ValueError(f"cannot release hosts outside the pre-spawn: {sorted(unknown)}")
        now = time.time()
        for key in keys:
            self.released_at.setdefault(key, now)
            self._event(key).set()
        return {key: self.released_at[key] for key in keys}

    async def status(self) -> Dict[str, Any]:
        return {"expected": list(self.expected), "parked": dict(self.parked_at),
                "released": dict(self.released_at), "returned": dict(self.returned_at)}


def create_gate(name: str, namespace: str, expected_keys: List[str]) -> Any:
    import ray

    return ray.remote(num_cpus=0, max_concurrency=1024)(ReleaseGate).options(
        name=name, namespace=namespace).remote(sorted(set(expected_keys)))


def park_and_wait(gate: Dict[str, Any], key: str, *, get: Optional[Callable] = None,
                  find: Optional[Callable] = None, **device_kwargs: Any) -> Dict[str, Any]:
    """Worker side, immediately before the first device touch.

    ``gate``: the ``prespawn_gate`` payload ({"name", "namespace", "timeout_s",
    optional "device_timeout_s"}).  Returns the worker's gate timeline.
    """
    import ray

    get = get or ray.get
    find = find or (lambda: ray.get_actor(gate["name"], namespace=gate["namespace"]))
    timeout_s = float(gate["timeout_s"])
    probe = {k: v for k, v in device_kwargs.items() if k in ("lockfile", "device_globs")}
    parked = assert_untouched("parked", **probe)
    handle = find()
    receipt = get(handle.wait.remote(key, timeout_s), timeout=timeout_s + 60.0)
    gate_returned_at = time.time()
    released = assert_untouched("released", **probe)
    free = wait_device_free(timeout_s=float(gate.get("device_timeout_s", DEFAULT_DEVICE_FREE_TIMEOUT_S)),
                            **device_kwargs)
    return {"key": key, "pid": os.getpid(), "parked_at": parked["at"], "gate_parked_at": receipt["parked_at"],
            "released_at": receipt["released_at"], "gate_returned_at": gate_returned_at,
            "lock_free_at": free["lock_free_at"], "vfio_free_at": free["vfio_free_at"],
            "groups_openable_at": free.get("groups_openable_at"), "busy_polls": free.get("busy_polls"),
            "device_free_wait_s": free["device_free_wait_s"], "holders_seen": free["holders_seen"],
            "self_check_parked": parked, "self_check_released": released}


# --------------------------------------------------------------------------- executor

def bundles_sorted_by_node_any_device(placement_group) -> List[Tuple[int, str, str]]:
    """vllm ``get_bundles_sorted_by_node`` without its device-key filter.

    The installed version keeps only bundles that carry ``ray_device_key`` (vllm
    v1/executor/ray_utils.py:429-432), so a CPU-only group yields no worker ranks and
    RayExecutorV2 fails at ``bundle_assignments[0]`` (ray_executor_v2.py:330).  Same order as the
    original: driver node first, then by node id, stable in bundle index (ray_utils.py:434-440),
    so pre-spawned ranks land exactly where a normal deploy puts them.
    """
    import ray
    from ray.util import placement_group_table

    bundle_to_node = placement_group_table(placement_group)["bundles_to_node_id"]
    node_id_to_ip = {n["NodeID"]: n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]}
    entries = [(index, bundle_to_node[index], node_id_to_ip[bundle_to_node[index]])
               for index in range(len(placement_group.bundle_specs))]
    driver_node = ray.get_runtime_context().get_node_id()
    entries.sort(key=lambda item: (0 if item[1] == driver_node else 1, item[1]))
    return entries


def prespawn_executor_class(base: type) -> type:
    """``base`` (tpu_inference RayDistributedExecutorV2) whose worker actors request no TPU.

    The stock override returns ``{"num_gpus": 0, "resources": {"TPU": bundle TPU}}``
    (tpu_inference/executors/ray_distributed_executor_v2.py:135-145); with a CPU-only bundle
    that is ``TPU: 0``, which Ray 2.56 rejects ("not a supported chip configuration").  A TPU-free
    actor also gets ``get_accelerator_ids()["TPU"] == []`` and Ray leaves the TPU_* env alone;
    HeteroTPUWorker.init_device writes the instance's TPU env before the device init anyway.
    Rank discovery uses :func:`bundles_sorted_by_node_any_device` for the duration of
    ``_init_executor`` (a pre-spawned engine owns its process: no other executor is affected).
    """

    def _get_actor_resource_kwargs(self=None) -> Dict[str, Any]:
        return {"num_gpus": 0}

    def _init_executor(self) -> None:
        from vllm.v1.executor import ray_executor_v2

        original = ray_executor_v2.get_bundles_sorted_by_node
        ray_executor_v2.get_bundles_sorted_by_node = bundles_sorted_by_node_any_device
        try:
            base._init_executor(self)
        finally:
            ray_executor_v2.get_bundles_sorted_by_node = original

    return type(f"Prespawn{base.__name__}", (base,), {
        "_get_actor_resource_kwargs": _get_actor_resource_kwargs,
        "_init_executor": _init_executor,
        "__doc__": prespawn_executor_class.__doc__,
    })
