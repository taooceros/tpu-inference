"""CPU stand-in for a block: N spawned processes x D CPU devices, one jax.distributed runtime.

Each process plays one host kernel. It initializes the runtime once, builds an
optional hosted object from a ``"module:attr"`` factory, then serves calls over
a pipe until closed. A call names either a method of the hosted object or a
``"module:function"`` to run with the given args, so tests can ship per-rank
programs without a hosted object.

``init_cpu_runtime`` is the bootstrap the Ray host kernel uses too.
"""

from __future__ import annotations

import importlib
import os
import signal
import socket
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# CPU device ids of process p are p * CPU_ID_STRIDE + i (jax 0.10.2 and 0.11.2).
CPU_ID_STRIDE = 2048


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def init_cpu_runtime(rank: int, processes: int, devices: int, coordinator: str,
                     timeout_s: int = 120) -> None:
    """Join a CPU block runtime. Must run before anything else imports jax in this process."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = [f for f in os.environ.get("XLA_FLAGS", "").split()
             if not f.startswith("--xla_force_host_platform_device_count")]
    os.environ["XLA_FLAGS"] = " ".join(flags + [f"--xla_force_host_platform_device_count={devices}"])
    import jax
    if jax._src.xla_bridge.backends_are_initialized():
        raise RuntimeError("jax backends already initialized; init_cpu_runtime must run first")
    jax.config.update("jax_cpu_collectives_implementation", "gloo")
    jax.distributed.initialize(coordinator, num_processes=processes, process_id=rank,
                               initialization_timeout=timeout_s)
    local = [d.id for d in jax.local_devices()]
    expected = [rank * CPU_ID_STRIDE + i for i in range(devices)]
    if local != expected:
        raise RuntimeError(f"rank {rank}: local CPU device ids {local}, expected {expected}")


def resolve(path: str) -> Any:
    module, _, attr = path.partition(":")
    value = importlib.import_module(module)
    for part in attr.split("."):
        value = getattr(value, part)
    return value


def _block_main(rank, processes, devices, coordinator, factory, kwargs, env, conn):
    os.environ.update(env or {})
    try:
        init_cpu_runtime(rank, processes, devices, coordinator)
        hosted = resolve(factory)(rank=rank, processes=processes, devices=devices,
                                  **(kwargs or {})) if factory else None
    except BaseException:
        conn.send((False, traceback.format_exc()))
        return
    conn.send((True, "ready"))
    while True:
        message = conn.recv()
        if message[0] == "__shutdown__":
            break
        method, args = message
        try:
            target = resolve(method) if ":" in method else getattr(hosted, method)
            conn.send((True, target(*args)))
        except BaseException:
            conn.send((False, traceback.format_exc()))
    try:
        if hosted is not None and hasattr(hosted, "shutdown"):
            hosted.shutdown()
        import jax
        jax.distributed.shutdown()
    finally:
        conn.send((True, "bye"))


class BlockCallError(RuntimeError):
    def __init__(self, rank: int, method: str, remote_traceback: str):
        super().__init__(f"rank {rank} {method}:\n{remote_traceback}")
        self.rank, self.method, self.remote_traceback = rank, method, remote_traceback


class CpuBlock:
    """``processes`` spawned host processes sharing one CPU jax.distributed runtime.

    ``factory`` ("module:attr", optional) is called in every process as
    ``factory(rank=, processes=, devices=, **kwargs)``; ``env`` is applied
    before jax is imported there.
    """

    def __init__(self, factory: Optional[str] = None, kwargs: Optional[Dict[str, Any]] = None,
                 processes: int = 4, devices: int = 4, timeout_s: float = 180,
                 env: Optional[Dict[str, str]] = None):
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        self.processes, self.devices = processes, devices
        self.coordinator = f"127.0.0.1:{free_port()}"
        self.conns, self.procs = [], []
        self.dead: set = set()
        for rank in range(processes):
            parent, child = context.Pipe()
            process = context.Process(target=_block_main, daemon=True, args=(
                rank, processes, devices, self.coordinator, factory, kwargs, env, child))
            process.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(process)
        try:
            for rank, conn in enumerate(self.conns):
                if not conn.poll(timeout_s):
                    raise TimeoutError(f"rank {rank} did not initialize in {timeout_s}s")
                ok, value = conn.recv()
                if not ok:
                    raise BlockCallError(rank, "__init__", value)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "CpuBlock":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def call(self, calls: Sequence[Tuple[int, str, tuple]], timeout_s: float = 120) -> List[Any]:
        """Send every call first (so SPMD programs run concurrently), then collect in order."""
        for rank, method, args in calls:
            self.conns[rank].send((method, tuple(args)))
        results, errors = [], []
        for rank, method, _ in calls:
            if not self.conns[rank].poll(timeout_s):
                raise TimeoutError(f"rank {rank} {method} timed out after {timeout_s}s")
            ok, value = self.conns[rank].recv()
            if not ok:
                errors.append(BlockCallError(rank, method, value))
            results.append(value)
        if errors:
            raise errors[0]
        return results

    def call_all(self, method: str, *args: Any, ranks: Optional[Sequence[int]] = None,
                 timeout_s: float = 120) -> List[Any]:
        ranks = range(self.processes) if ranks is None else ranks
        return self.call([(r, method, args) for r in ranks], timeout_s=timeout_s)

    def kill(self, rank: int) -> None:
        """SIGKILL one host process (failure-domain tests)."""
        os.kill(self.procs[rank].pid, signal.SIGKILL)
        self.procs[rank].join(timeout=30)
        self.dead.add(rank)

    def close(self) -> None:
        for rank, conn in enumerate(self.conns):
            if rank in self.dead:
                continue
            try:
                conn.send(("__shutdown__",))
            except (BrokenPipeError, OSError):
                pass
        for process in self.procs:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        self.conns, self.procs = [], []
