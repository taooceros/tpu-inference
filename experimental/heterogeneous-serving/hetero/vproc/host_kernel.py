"""C1: the host kernel. One long-lived process per host that owns the block runtime.

``HostKernel`` bootstraps the block's ``jax.distributed`` runtime once, keeps it
alive across layout switches, owns the resident weight store (C4) and runs at
most one engine (C3) under that engine's view (C2). It implements
``types.HostKernelAPI``; the controller calls it as a Ray actor
(``spawn_block``) and the CPU test harness calls it through
``cpu_block.CpuBlock(factory="hetero.vproc.host_kernel:HostKernel")``.

Bootstrap, one source per platform:
- cpu: ``cpu_block.init_cpu_runtime`` (gloo collectives), skipped when the
  process already joined a runtime (the CpuBlock case).
- tpu: InPlaceEngine's ``hetero/ici_worker.py:47-57``
  (``jax.distributed.initialize(coordinator, num_processes, process_id=rank)``)
  under the per-host environment of ``scripts/ici_engine.py:82-86`` (libtpu
  ``TPU_*`` variables from ``ResolvedInstance.process_env`` plus
  ``TPU_VISIBLE_CHIPS=0,1,2,3``), which the caller passes as ``env`` /
  ``rank_env`` and ``spawn_block`` installs through ``runtime_env`` so it is set
  before jax is imported. Not exercised locally.

Every host of a runtime must construct/bootstrap concurrently: rank 0 mints the
``runtime_id`` in the coordination service and the others read it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from .cpu_block import free_port, init_cpu_runtime, resolve
from .types import ENGINE_KINDS, EngineSpec, HostDescription, ReshardPlan, ResidentManifest

RUNTIME_ID_KEY = "vproc/runtime_id"
LIFECYCLE_METHODS = frozenset({"build", "export", "shutdown", "switch_layout"})


class HostKernelError(RuntimeError):
    """A host kernel call failed; the message names the rank, call and cause."""


class HostKernel:
    """``types.HostKernelAPI`` for one host. See the module docstring.

    ``coordinator`` set -> bootstrap in the constructor. Unset: the runtime must
    already be initialized in this process, or the caller runs ``bootstrap``
    (Ray: the constructor must return before the other ranks can join).
    """

    def __init__(self, rank: int, processes: int, devices: int = 4, platform: str = "cpu",
                 coordinator: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                 init_timeout_s: int = 180,
                 preinit: Optional[str] = "hetero.vproc.engine_vllm:preinit_environment"):
        if platform not in ("cpu", "tpu"):
            raise ValueError(f"platform must be 'cpu' or 'tpu', got {platform!r}")
        self.rank, self.processes, self.devices, self.platform = rank, processes, devices, platform
        self.env, self.init_timeout_s = dict(env or {}), init_timeout_s
        self.preinit = preinit  # tpu only: "module:function" run before the backend starts
        self.runtime_id: Optional[str] = None
        self.device_ids: tuple = ()
        self.spec: Optional[EngineSpec] = None
        self._engine = None
        self._store = None
        self._owns_runtime = False
        self._op = threading.Lock()
        self._busy: Optional[tuple] = None
        self._last_stop: Optional[Dict[str, Any]] = None
        self._hostname, self._pid = socket.gethostname(), os.getpid()
        self._started = time.time()
        if coordinator is not None or self._runtime_up():
            self.bootstrap(coordinator)

    # -- bootstrap --------------------------------------------------------

    @staticmethod
    def _runtime_up() -> bool:
        try:
            import jax
            return jax.distributed.is_initialized()
        except Exception:
            return False

    def address(self) -> str:
        """``ip:port`` this host offers as the coordinator (rank 0's answer is used)."""
        import ray
        return f"{ray.util.get_node_ip_address()}:{free_port()}"

    def bootstrap(self, coordinator: Optional[str] = None) -> str:
        """Join the block runtime once and return the shared ``runtime_id``."""
        if self.runtime_id is not None:
            return self.runtime_id
        try:
            self._bootstrap(coordinator)
        except Exception as error:
            raise HostKernelError(f"rank {self.rank} bootstrap({coordinator}): "
                                  f"{type(error).__name__}: {error}") from error
        return self.runtime_id

    def _bootstrap(self, coordinator: Optional[str]) -> None:
        if not self._runtime_up():
            if coordinator is None:
                raise ValueError("runtime not initialized and no coordinator given")
            if self.platform == "cpu":
                init_cpu_runtime(self.rank, self.processes, self.devices, coordinator,
                                 self.init_timeout_s)
            else:
                os.environ.update(self.env)
                if self.preinit:
                    resolve(self.preinit)()  # libtpu reads XLA_FLAGS/LIBTPU_INIT_ARGS once, at backend start
                import jax
                jax.distributed.initialize(coordinator, num_processes=self.processes,
                                           process_id=self.rank,
                                           initialization_timeout=self.init_timeout_s)
            self._owns_runtime = True
        import jax
        local = jax.local_devices()
        problems = []
        if jax.process_index() != self.rank:
            problems.append(f"jax.process_index() {jax.process_index()}")
        if jax.process_count() != self.processes:
            problems.append(f"jax.process_count() {jax.process_count()}")
        if len(local) != self.devices:
            problems.append(f"{len(local)} local devices")
        if any(d.platform != self.platform for d in local):
            problems.append(f"platform {sorted({d.platform for d in local})}")
        if problems:
            raise RuntimeError(f"runtime does not match rank {self.rank}/{self.processes} x "
                               f"{self.devices} {self.platform}: " + ", ".join(problems))
        self.device_ids = tuple(int(d.id) for d in local)
        self.runtime_id = self._mint_runtime_id()

    def _mint_runtime_id(self) -> str:
        from jax._src import distributed
        client = distributed.global_state.client
        timeout_ms = int(self.init_timeout_s * 1000)
        if self.rank == 0:
            value = f"{self.platform}-{uuid.uuid4().hex[:12]}"
            try:
                client.key_value_set(RUNTIME_ID_KEY, value)
                return value
            except Exception:
                pass  # key already set by an earlier kernel object of this runtime
        return client.blocking_key_value_get(RUNTIME_ID_KEY, timeout_ms)

    # -- observation (lock-free, never touches devices) -------------------

    def _resident_keys(self) -> tuple:
        return tuple(self._store.keys()) if self._store is not None else ()

    def describe(self) -> HostDescription:
        return HostDescription(
            rank=self.rank, processes=self.processes, platform=self.platform,
            device_ids=self.device_ids, hostname=self._hostname, pid=self._pid,
            runtime_id=self.runtime_id, engine=self.spec,
            resident=self._resident_keys(), clock_s=time.time())

    def health(self) -> Dict[str, Any]:
        busy = self._busy
        hbm = None
        if self.runtime_id is not None:  # backend is up; never initialize it from health
            from .measure import device_memory_stats
            hbm = device_memory_stats()
        return {"rank": self.rank, "pid": self._pid, "runtime_id": self.runtime_id,
                "alive": True, "ok": True, "bootstrapped": self.runtime_id is not None,
                "engine": self.spec.engine_id if self.spec else None,
                "busy": busy[0] if busy else None,
                "busy_s": time.time() - busy[1] if busy else 0.0,
                "hbm": hbm, "last_stop": self._last_stop,
                "uptime_s": time.time() - self._started, "clock_s": time.time()}

    # -- calls ------------------------------------------------------------

    @contextlib.contextmanager
    def _call(self, name: str):
        """Serialize state-changing calls, publish ``busy`` for ``health``, add context to errors."""
        if self.runtime_id is None:
            raise HostKernelError(f"rank {self.rank} {name}: runtime not bootstrapped")
        with self._op:
            self._busy = (name, time.time())
            try:
                yield
            except HostKernelError:
                raise
            except Exception as error:
                raise HostKernelError(f"rank {self.rank} {name}: "
                                      f"{type(error).__name__}: {error}") from error
            finally:
                self._busy = None

    def _store_for_use(self):
        if self._store is None:
            from .resident import ResidentStore
            self._store = ResidentStore(self.rank)
        return self._store

    def start_engine(self, spec: EngineSpec, resident: Optional[str] = None) -> Dict[str, Any]:
        with self._call(f"start_engine({spec.engine_id}, resident={resident})"):
            from . import view
            if self._engine is not None:
                raise RuntimeError(f"engine {self.spec.engine_id} is still running; stop_engine first")
            if self.rank not in spec.hosts:
                raise ValueError(f"host {self.rank} is not in engine hosts {spec.hosts}")
            t0 = time.time()
            leaves = None
            if resident is not None:
                store = self._store_for_use()
                manifest = store.manifest(resident)
                if manifest.tp_size != spec.tp_size or tuple(manifest.hosts) != spec.hosts:
                    raise ValueError(
                        f"resident {resident!r} is tp{manifest.tp_size} on hosts {tuple(manifest.hosts)}, "
                        f"engine wants tp{spec.tp_size} on hosts {spec.hosts}")
                leaves = store.take(resident)  # consumed even if the build then fails
            from .resident import _COMPILES, _install_compile_listener
            _install_compile_listener()  # same backend-compile counter as ResidentStore.reshard
            compiles0 = len(_COMPILES)
            engine = resolve(ENGINE_KINDS.get(spec.kind, spec.kind))(spec)
            try:
                with view.active(spec.view()):
                    segments = engine.build(leaves)
            except BaseException:
                self._discard(engine)
                raise
            self._engine, self.spec = engine, spec
            t1 = time.time()
            return {"rank": self.rank, "engine_id": spec.engine_id, "generation": spec.generation,
                    "hosts": spec.hosts, "resident": resident, "segments": dict(segments),
                    "compiles": len(_COMPILES) - compiles0,
                    "compile_s": float(sum(_COMPILES[compiles0:])),
                    "t0": t0, "t1": t1, "seconds": t1 - t0}

    @staticmethod
    def _discard(engine) -> None:
        try:
            engine.shutdown()
        except Exception:
            pass

    def engine_call(self, method: str, *args: Any) -> Any:
        with self._call(f"engine_call({method})"):
            from . import view
            if self._engine is None:
                raise RuntimeError("no engine is running")
            if method.startswith("_") or method in LIFECYCLE_METHODS:
                raise ValueError(f"{method!r} is not callable through engine_call")
            with view.active(self.spec.view()):
                return getattr(self._engine, method)(*args)

    def stop_engine(self, export: bool = True) -> Optional[ResidentManifest]:
        with self._call(f"stop_engine(export={export})"):
            from . import view
            if self._engine is None:
                raise RuntimeError("no engine is running")
            from .resident import _COMPILES, _install_compile_listener
            _install_compile_listener()
            compiles0 = len(_COMPILES)
            engine, spec = self._engine, self.spec
            self._engine, self.spec = None, None  # an engine that fails to stop is gone either way
            try:
                with view.active(spec.view()):
                    manifest = None
                    if export:
                        specs = engine.leaf_specs()
                        manifest = self._store_for_use().put(spec, engine.export(), specs)
            finally:
                self._discard(engine)
                # ResidentManifest has no room for it: health()["last_stop"] carries the count
                self._last_stop = {"engine_id": spec.engine_id, "export": export,
                                   "compiles": len(_COMPILES) - compiles0}
            return manifest

    def switch_engine_layout(self, target: EngineSpec) -> Dict[str, Any]:
        """SPMD on all hosts of the running engine: ``Engine.switch_layout`` under the engine's view.

        ``target`` is the running spec with another ``(tp_size, dp_size)``. The engine's answer
        (``built``, ``segments``, ``compiles``, ``hbm``, ...) is returned with this host's ``rank``,
        ``engine_id``, ``t0``/``t1``/``seconds``. The kernel's spec follows the engine's, so a failed
        switch leaves ``describe().engine`` showing the layout the engine reports, and the error
        propagates like any other engine call."""
        with self._call(f"switch_engine_layout({target.engine_id}, tp{target.tp_size} x dp{target.dp_size})"):
            from . import view
            if self._engine is None:
                raise RuntimeError("no engine is running")
            if not hasattr(self._engine, "switch_layout"):
                raise TypeError(f"engine kind {self.spec.kind!r} has no switch_layout")
            if target != dataclasses.replace(self.spec, tp_size=target.tp_size, dp_size=target.dp_size):
                raise ValueError(f"switch target {target} differs from the running {self.spec} by more "
                                 "than tp_size/dp_size")
            t0 = time.time()
            try:
                with view.active(self.spec.view()):
                    result = dict(self._engine.switch_layout(target))
            finally:
                self.spec = getattr(self._engine, "spec", self.spec)
            t1 = time.time()
            result.update({"rank": self.rank, "engine_id": target.engine_id, "t0": t0, "t1": t1,
                           "seconds": t1 - t0})
            return result

    def reshard(self, plan: ReshardPlan) -> Dict[str, Any]:
        with self._call(f"reshard({plan.plan_id})"):
            from . import view
            if self._engine is not None:
                raise RuntimeError(f"engine {self.spec.engine_id} is running; "
                                   "stop every engine of the plan before reshard")
            t0 = time.time()
            with view.physical():
                result = dict(self._store_for_use().reshard(plan))
            t1 = time.time()
            result.update({"rank": self.rank, "plan_id": plan.plan_id, "t0": t0, "t1": t1,
                           "seconds": t1 - t0})
            return result

    def shutdown(self) -> None:
        with self._call("shutdown"):
            if self._engine is not None:
                engine, self._engine, self.spec = self._engine, None, None
                self._discard(engine)
            if self._store is not None:
                for key in list(self._store.keys()):
                    self._store.drop(key)
            if self._owns_runtime:
                import jax
                jax.distributed.shutdown()
                self._owns_runtime = False


# -- Ray ---------------------------------------------------------------------

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _actor_class():
    """HostKernel as a Ray actor: ``describe``/``health``/``address`` run in their
    own concurrency group, so they answer while a SPMD call occupies the default
    group."""
    import ray

    class HostKernelActor(HostKernel):
        @ray.method(concurrency_group="control")
        def describe(self):
            return super().describe()

        @ray.method(concurrency_group="control")
        def health(self):
            return super().health()

        @ray.method(concurrency_group="control")
        def address(self):
            return super().address()

    return ray.remote(num_cpus=0, max_restarts=0, concurrency_groups={"control": 2})(HostKernelActor)


def spawn_block(processes: int = 4, devices: int = 4, platform: str = "cpu",
                env: Optional[Dict[str, str]] = None,
                rank_env: Optional[Sequence[Dict[str, str]]] = None,
                actor_options: Any = None, init_timeout_s: int = 180,
                name_prefix: Optional[str] = None,
                preinit: Optional[str] = "hetero.vproc.engine_vllm:preinit_environment") -> List[Any]:
    """Create one host-kernel Ray actor per rank and bootstrap the runtime on all.

    ``env`` / ``rank_env[rank]`` are process environment installed through
    ``runtime_env`` before jax is imported (TPU: ``ResolvedInstance.process_env``
    and chip visibility). ``actor_options`` (dict, or one per rank) is passed to
    ``.options`` (placement group, resources). The coordinator is rank 0's
    address. ``preinit`` (TPU only, "module:function") runs before the backend starts; the
    default is the vLLM engine's env preparation, pass None for synthetic-only blocks.
    Returns the actors in rank order; ray must be initialized.
    """
    import ray
    if not ray.is_initialized():
        raise RuntimeError("spawn_block needs an initialized ray (ray.init first)")
    base = {"PYTHONPATH": os.pathsep.join(filter(None, [_repo_root(), os.environ.get("PYTHONPATH")])),
            "PYTHONUNBUFFERED": "1"}
    if platform == "cpu":
        base.update({"JAX_PLATFORMS": "cpu",
                     "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"})
    else:
        base.update({"TPU_VISIBLE_CHIPS": ",".join(str(i) for i in range(devices)),
                     "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1"})
    cls = _actor_class()
    actors = []
    try:
        for rank in range(processes):
            environment = {**base, **(env or {}), **(rank_env[rank] if rank_env else {})}
            options = dict(actor_options[rank] if isinstance(actor_options, (list, tuple))
                           else actor_options or {})
            if "runtime_env" in options:
                raise ValueError("pass process environment as env/rank_env, not runtime_env")
            if name_prefix:
                options["name"] = f"{name_prefix}-{rank}"
            actors.append(cls.options(runtime_env={"env_vars": environment}, **options).remote(
                rank, processes, devices, platform, None, environment if platform == "tpu" else None,
                init_timeout_s, preinit))
        coordinator = ray.get(actors[0].address.remote(), timeout=init_timeout_s)
        ids = ray.get([a.bootstrap.remote(coordinator) for a in actors], timeout=init_timeout_s + 60)
        if len(set(ids)) != 1:
            raise RuntimeError(f"hosts disagree on runtime_id: {ids}")
    except BaseException:
        kill_block(actors)
        raise
    return actors


def kill_block(actors: Sequence[Any], timeout_s: float = 60) -> None:
    """Shut every kernel down (concurrently, so the runtime leaves together), then kill the actors."""
    import ray
    try:
        ray.get([a.shutdown.remote() for a in actors], timeout=timeout_s)
    except Exception:
        pass
    for actor in actors:
        ray.kill(actor, no_restart=True)
