"""C5: the block controller (Ray driver side of the virtual-process runtime).

The controller owns the placement table (host -> engine) of one block and drives
layout switches over the host kernels (C1). Switch protocol, resident weights:

    drain -> withdraw -> stop_engine(export) on source hosts
          -> one ReshardPlan over every involved host
          -> start_engine(resident=f"{plan_id}/{engine_id}") on target hosts -> publish

Cold (``resident=False``): stop without export, no reshard, start from the model source.
Engine ids are per layout (``"S-block-0"``); the generation of an id increments each
time the controller builds it again.

The block is one failure domain. A dead or unresponsive host kernel (Ray actor death,
call or health timeout) tears the whole block down; the controller respawns it with
``respawn`` (default ``host_kernel.spawn_block``), redeploys the last layout cold and
records a rebuild report. Remote application errors during deploy or switch also
leave the SPMD state undefined, so they rebuild too, except when nothing was touched
yet.

The call layer is injectable: ``Caller`` sends calls to all hosts of an SPMD step
concurrently. ``RayCaller`` drives Ray actor handles, ``ThreadCaller`` in-process objects.

In-engine layout switch (``switch_layout``, non-transparent, one engine over whole hosts): the engine id,
its host kernels, JAX runtime, frontend actor and gateway endpoint stay; only the engine's (tp_size,
dp_size) changes:

    block admission -> drain frontend + await idle -> ``switch_engine_layout`` on all its hosts (weights
    resharded in HBM, parked layouts keep their executables) -> frontend ``switch_layout(target, built)``
    (KV init + warm-up + scheduler on a layout's first visit, else swap of the parked scheduler)
    -> identity check, ``resume_admission`` -> admission resumes

Gateway/registry: the published layout id, layout digest, generation, actor name and endpoint are NOT
changed (the gateway binds on them and ``reconcile`` demands an idle gateway for any change), and there
is no withdraw/publish cycle: ``Endpoints`` has no in-place update, so ``endpoints.update_engine(spec,
frontends=..., layout=...)`` is called only when an implementation offers it. The registry row's
``tp_size`` therefore keeps the layout it was published with until the engine is next published;
frontend ``identity()``/``status()`` report the live ``tp_size``/``dp_size``. On a failure after the hosts
were touched the engine is left stopped: frontend killed, engine dropped from the controller and the
endpoints, hosts untouched, NO rebuild. ``rebuild()`` brings the block back on ``last_layout`` (which
records the engine's current in-engine layout).
"""

from __future__ import annotations

import dataclasses
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import measure as measure_mod
from .endpoints import FRONTEND_NAMESPACE, Endpoints, InMemoryEndpoints, layout_info
from .measure import Measure
from .types import (BLOCK_LAYOUTS, IN_ENGINE_LAYOUTS, EngineSpec, HostDescription, IllegalHostSet, ReshardPlan,
                    check_host_set)


# --------------------------------------------------------------------------- errors

class KernelFailure(RuntimeError):
    """A host kernel is dead or did not answer: the block must be rebuilt."""

    def __init__(self, rank: int, method: str, kind: str, cause: Any = None,
                 ranks: Sequence[int] = ()):
        super().__init__(f"host kernel {rank} {method}: {kind}" + (f" ({cause!r})" if cause else ""))
        self.rank, self.method, self.kind, self.cause = rank, method, kind, cause
        self.ranks = tuple(ranks) or (rank,)


class KernelCallError(RuntimeError):
    """A host kernel raised while handling a call (it is alive)."""

    def __init__(self, rank: int, method: str, cause: Any):
        super().__init__(f"host kernel {rank} {method} raised: {cause!r}")
        self.rank, self.method, self.cause = rank, method, cause


class KernelDead(ConnectionError):
    """Raised by in-process kernels to say 'this host process is gone'."""


class BlockDown(RuntimeError):
    """The block is down and could not be rebuilt (or is being rebuilt)."""


class EngineNotFound(KeyError):
    pass


class LayoutSwitchFailed(RuntimeError):
    """An in-engine layout switch failed after the hosts were touched. The engine was left stopped (frontend
    killed, engine dropped, endpoint withdrawn); the block was NOT rebuilt (call ``rebuild()``).
    ``report`` has the partial phases."""

    def __init__(self, message: str, report: Dict[str, Any]):
        super().__init__(message)
        self.report = report


class SwitchFailed(RuntimeError):
    """A switch failed after it started changing the block; the block was rebuilt
    on the source layout. ``report`` has the partial segments, ``rebuild`` the rebuild."""

    def __init__(self, message: str, report: Dict[str, Any], rebuild: Optional[Dict[str, Any]]):
        super().__init__(message)
        self.report, self.rebuild = report, rebuild


class WeightMismatch(RuntimeError):
    pass


# --------------------------------------------------------------------------- call layer

_DEAD_ERRORS = (KernelDead, ConnectionError, EOFError)
Outcome = Tuple[str, Any]  # ("ok" | "dead" | "timeout" | "error" | "aborted", value or exception)
Call = Tuple[int, str, tuple]


class Caller:
    """Concurrent RPC to the host kernels of one block, addressed by host rank."""

    size: int

    def _run(self, calls: Sequence[Call], timeout_s: float, fail_fast: bool) -> List[Outcome]:
        raise NotImplementedError

    def kill_all(self) -> None:
        raise NotImplementedError

    def call_each(self, calls: Sequence[Call], timeout_s: float) -> List[Outcome]:
        """Every outcome, waiting out ``timeout_s`` for slow kernels (health probes)."""
        return self._run(calls, timeout_s, False)

    def call(self, calls: Sequence[Call], timeout_s: float) -> List[Any]:
        """Results in call order. Fails fast when a kernel is dead; raises
        ``KernelFailure`` (dead/timeout) before ``KernelCallError`` (remote exception)."""
        outcomes = self._run(calls, timeout_s, True)
        failed = [(c, o) for c, o in zip(calls, outcomes) if o[0] in ("dead", "timeout")]
        if failed:
            (rank, method, _), (kind, cause) = failed[0]
            raise KernelFailure(rank, method, kind, cause, ranks=[c[0] for c, _ in failed])
        for (rank, method, _), (kind, value) in zip(calls, outcomes):
            if kind == "error":
                raise KernelCallError(rank, method, value)
        return [value for _, value in outcomes]


class ThreadCaller(Caller):
    """In-process kernels (objects with the ``HostKernelAPI`` methods), one thread per call.
    ``KernelDead``/``ConnectionError`` mean death, ``TimeoutError`` a kernel-side timeout."""

    def __init__(self, kernels: Sequence[Any]):
        self.kernels = list(kernels)
        self.size = len(self.kernels)

    def _run(self, calls, timeout_s, fail_fast):
        outcomes: List[Optional[Outcome]] = [None] * len(calls)
        cond = threading.Condition()

        def work(i: int, rank: int, method: str, args: tuple) -> None:
            try:
                out: Outcome = ("ok", getattr(self.kernels[rank], method)(*args))
            except _DEAD_ERRORS as exc:
                out = ("dead", exc)
            except TimeoutError as exc:
                out = ("timeout", exc)
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                out = ("error", exc)
            with cond:
                outcomes[i] = out
                cond.notify_all()

        for i, (rank, method, args) in enumerate(calls):
            threading.Thread(target=work, args=(i, rank, method, tuple(args)), daemon=True).start()
        deadline = time.monotonic() + timeout_s
        aborted = False
        with cond:
            while any(o is None for o in outcomes):
                if fail_fast and any(o is not None and o[0] != "ok" for o in outcomes):
                    aborted = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                cond.wait(remaining)
            return [o if o is not None else ("aborted" if aborted else "timeout", None)
                    for o in outcomes]

    def kill_all(self) -> None:
        for kernel in self.kernels:
            shutdown = getattr(kernel, "shutdown", None)
            if shutdown is None:
                continue
            thread = threading.Thread(target=lambda s=shutdown: _quietly(s), daemon=True)
            thread.start()
            thread.join(timeout=5.0)


class RayCaller(Caller):
    """Ray actor handles. ``RayActorError`` is death, ``GetTimeoutError`` a timeout,
    ``RayTaskError`` an exception raised inside a live kernel."""

    def __init__(self, handles: Sequence[Any]):
        self.handles = list(handles)
        self.size = len(self.handles)

    def _run(self, calls, timeout_s, fail_fast):
        import ray
        from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError

        refs = [getattr(self.handles[rank], method).remote(*args) for rank, method, args in calls]
        index = {ref: i for i, ref in enumerate(refs)}
        outcomes: List[Optional[Outcome]] = [None] * len(calls)
        pending = list(refs)
        deadline = time.monotonic() + timeout_s
        aborted = False
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=max(0.0, deadline - time.monotonic()))
            if not ready:
                break
            i = index[ready[0]]
            try:
                outcomes[i] = ("ok", ray.get(ready[0]))
            except RayActorError as exc:
                outcomes[i] = ("dead", exc)
            except GetTimeoutError as exc:
                outcomes[i] = ("timeout", exc)
            except RayTaskError as exc:
                outcomes[i] = ("error", exc)
            if fail_fast and outcomes[i][0] != "ok":
                aborted = True
                break
        return [o if o is not None else ("aborted" if aborted else "timeout", None) for o in outcomes]

    def kill_all(self) -> None:
        import ray

        for handle in self.handles:
            try:
                ray.kill(handle, no_restart=True)
            except Exception:  # noqa: BLE001 - already dead
                pass


def make_caller(kernels: Sequence[Any]) -> Caller:
    ray = sys.modules.get("ray")
    if ray is not None and kernels and all(isinstance(k, ray.actor.ActorHandle) for k in kernels):
        return RayCaller(kernels)
    return ThreadCaller(kernels)


def _quietly(fn: Callable[[], Any]) -> None:
    try:
        fn()
    except BaseException:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- controller

@dataclasses.dataclass
class Timeouts:
    describe_s: float = 60.0
    health_s: float = 15.0
    stop_s: float = 300.0
    reshard_s: float = 600.0
    start_s: float = 1800.0
    step_s: float = 600.0
    drain_wait_s: float = 600.0
    serve_wait_s: float = 3600.0
    layout_s: float = 3600.0  # one in-engine layout switch: worker reshard / frontend KV init + warm-up


def default_respawn(processes: int, devices: int, **options: Any) -> List[Any]:
    from . import host_kernel

    return list(host_kernel.spawn_block(processes=processes, devices=devices, **options))


class FrontendFailure(KernelFailure):
    """A frontend actor is dead, unresponsive or poisoned. Same consequence as a dead kernel."""

    def __init__(self, engine_id: str, method: str, kind: str, cause: Any = None,
                 ranks: Sequence[int] = (), engine_ids: Sequence[str] = ()):
        super().__init__(ranks[0] if ranks else -1, method, kind, cause, ranks=ranks)
        self.args = (f"frontend {engine_id} {method}: {kind}" + (f" ({cause!r})" if cause else ""),)
        self.engine_id, self.engine_ids = engine_id, tuple(engine_ids) or (engine_id,)


class FrontendCallError(KernelCallError):
    """A frontend raised while handling a call (it is alive)."""

    def __init__(self, engine_id: str, method: str, cause: Any):
        super().__init__(-1, method, cause)
        self.args = (f"frontend {engine_id} {method} raised: {cause!r}",)
        self.engine_id = engine_id


@dataclasses.dataclass
class Block:
    """What a ``respawn`` returns: host kernels in rank order and, optionally, the
    ``{"name", "namespace"}`` refs the frontends use to find them (kept when None)."""

    kernels: Sequence[Any]
    refs: Optional[Sequence[Dict[str, str]]] = None


def frontend_name(spec: EngineSpec) -> str:
    return f"vproc-{spec.engine_id}-g{spec.generation}"


def default_frontend_factory(spec: EngineSpec, kernel_refs: Sequence[Dict[str, str]],
                             settings: Dict[str, Any], name: str) -> Any:
    """Create the engine's frontend actor (``frontend.SyntheticFrontend`` / ``VllmFrontend``, Ray actor
    classes built as ``(spec, kernels, settings)``) named ``name`` in namespace ``omp_hetero``.

    Pinned to the controller's node: Ray places zero-resource actors on a random node, and a vLLM
    EngineCore on a TPU host would start the TPU platform next to that host's kernel. On the cluster
    the controller is the Ray job driver on the head node, where InPlaceEngine's EngineCore ran."""
    import os

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from . import frontend, host_kernel

    classes = {"synthetic": frontend.SyntheticFrontend, "vllm": frontend.VllmFrontend}
    if spec.kind not in classes:
        raise ValueError(f"no frontend for engine kind {spec.kind!r}")
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [host_kernel._repo_root(), os.environ.get("PYTHONPATH")])),
           "PYTHONUNBUFFERED": "1", "TPU_MULTIPROCESS_DP": "0"}  # in-engine DP is single-process SPMD
    here = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
    return classes[spec.kind].options(name=name, namespace=FRONTEND_NAMESPACE, scheduling_strategy=here,
                                      runtime_env={"env_vars": env}).remote(spec, list(kernel_refs), dict(settings))


class BlockController:
    """One block of whole hosts. ``kernels[i]`` is the host kernel of host rank ``i``.

    ``engine_model`` is the ``EngineSpec.model`` of named layouts, ``layout_models[name]``
    overrides it per layout (e.g. ``{"H-block": {"device_order": "halves_interleaved"}}``).
    ``respawn()`` returns new host kernels (a list, or a ``Block`` with their refs) after a
    failure; the default calls ``host_kernel.spawn_block`` with per-epoch actor names.

    Every engine has a frontend actor, created by ``frontend_factory(spec, kernel_refs,
    settings, name)`` (default ``default_frontend_factory``) once its ``start_engine`` has
    succeeded on all view hosts. ``kernel_refs`` = ``{"name", "namespace"}`` of the host kernels
    in rank order (default ``f"{kernel_prefix}-e0-{rank}"`` in ``kernel_namespace``: spawn the
    block with those names). ``settings`` = ``frontend_settings`` + ``layout_id`` and
    ``layout_digest`` (``layout_id_for(layout_name)``, default ``"vproc-<name>"``).
    """

    def __init__(self, kernels: Sequence[Any], endpoints: Optional[Endpoints] = None, *,
                 caller: Optional[Caller] = None,
                 respawn: Optional[Callable[[], Sequence[Any]]] = None,
                 spawn_options: Optional[Dict[str, Any]] = None,
                 engine_kind: str = "synthetic", engine_model: Optional[Dict[str, Any]] = None,
                 layout_models: Optional[Dict[str, Dict[str, Any]]] = None,
                 frontend_factory: Optional[Callable[..., Any]] = None,
                 frontend_settings: Optional[Dict[str, Any]] = None,
                 kernel_refs: Optional[Sequence[Dict[str, str]]] = None,
                 kernel_prefix: str = "vproc-kernel", kernel_namespace: str = FRONTEND_NAMESPACE,
                 layout_id_for: Optional[Callable[[str], str]] = None,
                 timeouts: Optional[Timeouts] = None, measure: Optional[Measure] = None,
                 checksums: bool = False, clock_rounds: int = 3,
                 clock: Callable[[], float] = time.time):
        self.kernels = list(kernels)
        self.caller = caller or make_caller(self.kernels)
        self.endpoints: Endpoints = endpoints if endpoints is not None else InMemoryEndpoints()
        self.respawn = respawn
        self.spawn_options = {"platform": "cpu", **(spawn_options or {})}
        self.engine_kind = engine_kind
        self.engine_model = dict(engine_model or {})
        self.layout_models = {k: dict(v) for k, v in (layout_models or {}).items()}
        self.frontend_factory = frontend_factory or default_frontend_factory
        self.frontend_settings = dict(frontend_settings or {})
        self.kernel_prefix, self.kernel_namespace = kernel_prefix, kernel_namespace
        self.kernel_refs = list(kernel_refs) if kernel_refs is not None else [
            {"name": f"{kernel_prefix}-e0-{h}", "namespace": kernel_namespace} for h in range(len(self.kernels))]
        self.layout_id_for = layout_id_for or (lambda name: f"vproc-{name}")
        self.timeouts = timeouts or Timeouts()
        self.checksums, self.clock_rounds, self.clock = checksums, clock_rounds, clock
        self.measure = measure or Measure(clock)

        self.engines: Dict[str, EngineSpec] = {}
        self.placement: Dict[int, str] = {}
        self.generations: Dict[str, int] = {}
        self.frontends: Dict[str, Any] = {}
        self.frontend_specs: Dict[str, EngineSpec] = {}
        self.frontend_names: Dict[str, str] = {}
        self.frontend_layout: Dict[str, Tuple[str, str]] = {}
        self.layout_name: Optional[str] = None
        self.last_layout: Optional[Tuple[str, List[EngineSpec]]] = None
        self.descriptions: List[HostDescription] = []
        self.epoch = 0
        self.switch_count = 0
        self.down = False

        self._admin = threading.RLock()
        self._cv = threading.Condition()
        self._blocked: Set[str] = set()
        self._block_all = False
        self._inflight: Dict[str, int] = {}

    # ----- block facts

    @property
    def hosts(self) -> int:
        return len(self.kernels)

    @property
    def chips_per_host(self) -> Optional[int]:
        return len(self.descriptions[0].device_ids) if self.descriptions else None

    def describe(self) -> List[HostDescription]:
        """``describe()`` on every kernel; checks one runtime and rank == position."""
        found = self.caller.call([(h, "describe", ()) for h in range(self.hosts)],
                                 self.timeouts.describe_s)
        for h, d in enumerate(found):
            if d.rank != h or d.processes != self.hosts:
                raise BlockDown(f"kernel {h} reports rank {d.rank} of {d.processes}, block has {self.hosts}")
        if len({d.runtime_id for d in found}) != 1:
            raise BlockDown(f"kernels run different runtimes: {sorted({d.runtime_id for d in found})}")
        self.descriptions = list(found)
        d0 = found[0]
        self.measure.block = {"hosts": self.hosts, "chips_per_host": len(d0.device_ids),
                              "platform": d0.platform, "runtime_id": d0.runtime_id,
                              "hostnames": [d.hostname for d in found]}
        return self.descriptions

    def runtime_ids(self) -> List[str]:
        return [d.runtime_id for d in self.describe()]

    def _sync_clocks(self) -> None:
        self.measure.clocks = measure_mod.estimate_offsets(
            lambda rank: self.caller.call([(rank, "describe", ())], self.timeouts.describe_s)[0],
            range(self.hosts), rounds=self.clock_rounds, clock=self.clock)

    # ----- layouts

    def _resolve(self, layout: Any) -> Tuple[str, List[EngineSpec]]:
        """Validated target specs (canonical hosts, no generation). Raises
        ``IllegalHostSet``/``ValueError`` before anything runs."""
        chips = self.chips_per_host
        if isinstance(layout, str):
            if layout not in BLOCK_LAYOUTS:
                raise ValueError(f"unknown layout {layout!r}; known: {sorted(BLOCK_LAYOUTS)}")
            name = layout
            model = {**self.engine_model, **self.layout_models.get(layout, {})}
            specs = [EngineSpec(f"{layout}-{i}", self.engine_kind, hosts, len(hosts) * (chips or 4), dict(model))
                     for i, hosts in enumerate(BLOCK_LAYOUTS[layout])]
        else:
            specs = [dataclasses.replace(s, generation=0) for s in layout]
            name = "+".join(s.engine_id for s in specs)
        if not specs:
            raise ValueError("a layout needs at least one engine")
        ids = [s.engine_id for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate engine ids in {ids}")
        used: Dict[int, str] = {}
        canonical = []
        for spec in specs:
            hosts = check_host_set(spec.hosts, block_hosts=self.hosts)
            for h in hosts:
                if h in used:
                    raise IllegalHostSet(f"host {h} is in both {used[h]} and {spec.engine_id}")
                used[h] = spec.engine_id
            if chips is not None and spec.chips != len(hosts) * chips:
                raise ValueError(f"{spec.engine_id}: tp_size {spec.tp_size} x dp_size {spec.dp_size} != "
                                 f"{len(hosts)} hosts x {chips} chips")
            canonical.append(dataclasses.replace(spec, hosts=hosts))
        return name, canonical

    def _next_generation(self, specs: Sequence[EngineSpec]) -> List[EngineSpec]:
        out = []
        for spec in specs:
            gen = self.generations.get(spec.engine_id, -1) + 1
            self.generations[spec.engine_id] = gen
            out.append(dataclasses.replace(spec, generation=gen))
        return out

    @staticmethod
    def _same_engine(a: EngineSpec, b: EngineSpec) -> bool:
        return ((a.engine_id, a.kind, a.hosts, a.tp_size, a.dp_size, a.model)
                == (b.engine_id, b.kind, b.hosts, b.tp_size, b.dp_size, b.model))

    def _set_engine(self, spec: EngineSpec) -> None:
        with self._cv:
            self.engines[spec.engine_id] = spec
            for h in spec.hosts:
                self.placement[h] = spec.engine_id

    def _drop_engine(self, engine_id: str) -> None:
        with self._cv:
            spec = self.engines.pop(engine_id, None)
            for h in (spec.hosts if spec else ()):
                if self.placement.get(h) == engine_id:
                    del self.placement[h]

    # ----- admission gate for serve()

    def _block(self, ids: Sequence[str], everything: bool = False) -> None:
        with self._cv:
            self._blocked.update(ids)
            self._block_all = self._block_all or everything

    def _unblock(self, ids: Sequence[str] = (), everything: bool = False) -> None:
        with self._cv:
            self._blocked.difference_update(ids)
            if everything:
                self._blocked.clear()
                self._block_all = False
            self._cv.notify_all()

    def _wait_idle(self, ids: Sequence[str]) -> None:
        deadline = time.monotonic() + self.timeouts.drain_wait_s
        with self._cv:
            while any(self._inflight.get(i, 0) for i in ids):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"in-flight requests on {list(ids)} did not finish")
                self._cv.wait(remaining)

    def _admit(self, engine_id: str) -> Tuple[int, EngineSpec]:
        deadline = time.monotonic() + self.timeouts.serve_wait_s
        with self._cv:
            while self._block_all or engine_id in self._blocked:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BlockDown(f"engine {engine_id} stayed blocked for {self.timeouts.serve_wait_s}s")
                self._cv.wait(remaining)
            if self.down:
                raise BlockDown("the block is down; call rebuild()")
            spec = self.engines.get(engine_id)
            if spec is None:
                raise EngineNotFound(engine_id)
            self._inflight[engine_id] = self._inflight.get(engine_id, 0) + 1
            return self.epoch, spec

    def _release(self, engine_id: str) -> None:
        with self._cv:
            self._inflight[engine_id] -= 1
            self._cv.notify_all()

    # ----- serve

    def serve(self, engine_id: str, request: Dict[str, Any], retry: bool = True) -> Dict[str, Any]:
        """One ``generate(request)`` on the engine's frontend, which orders the fanout to the view
        hosts. A dead kernel or frontend, or a poisoned frontend, rebuilds the block; with ``retry``
        the request is then served once more on the rebuilt engine. An ordinary request error
        (healthy block) is raised as it is."""
        attempts = 2 if retry else 1
        for attempt in range(attempts):
            epoch, spec = self._admit(engine_id)
            try:
                start = self.clock()
                result = self._fcall_many([engine_id], "generate", (request,), self.timeouts.step_s)[engine_id]
                end = self.clock()
            except (KernelFailure, KernelCallError) as exc:
                self._release(engine_id)
                failure = exc if isinstance(exc, KernelFailure) else self._diagnose(exc)
                if failure is None:
                    raise
                self._recover(failure, epoch=epoch)
                if attempt + 1 == attempts:
                    raise
                continue
            except BaseException:
                self._release(engine_id)
                raise
            self._release(engine_id)
            first = dict(result)
            first["serve"] = {"engine_id": engine_id, "generation": spec.generation, "hosts": list(spec.hosts),
                              "elapsed_s": end - start, "ranks_agree": first.get("ranks_agree"),
                              "retried": attempt > 0}
            return first
        raise AssertionError("unreachable")

    # ----- deploy

    def deploy(self, layout: Any) -> Dict[str, Any]:
        """Cold start of ``layout`` (a ``BLOCK_LAYOUTS`` name or EngineSpecs) on an empty block."""
        with self._admin:
            if self.engines:
                raise RuntimeError(f"block already runs {sorted(self.engines)}; use switch()")
            if self.down:
                raise BlockDown("the block is down; call rebuild()")
            self.describe()
            name, specs = self._resolve(layout)
            self._sync_clocks()
            targets = self._next_generation(specs)
            info = self._layout_info(name, targets)
            rec = self.measure.recorder()
            rec.begin()
            self._block([s.engine_id for s in targets])
            try:
                self._start_engines(rec, targets, None, info)
                self._publish(rec, targets, info, [])
            except BaseException as exc:
                self._abort(exc, rec, kind="deploy", target=name)
                self._recover(exc, epoch=None, redeploy=False)
                raise
            finally:
                self._unblock([s.engine_id for s in targets])
            self.layout_name, self.last_layout = name, (name, specs)
            report = {"kind": "deploy", "ok": True, "layout": name, "epoch": self.epoch,
                      "layout_id": info["layout_id"], "layout_digest": info["layout_digest"],
                      "generations": {s.engine_id: s.generation for s in targets},
                      "runtime_ids": [d.runtime_id for d in self.descriptions], **rec.finish()}
            self.measure.deploys.append(report)
            return report

    def _start_engines(self, rec, targets: Sequence[EngineSpec], plan_id: Optional[str],
                       layout: Dict[str, Any]) -> None:
        """Build phase: ``start_engine`` on every view host, then each engine's frontend is created
        and ``initialize()``d (EngineCore, KV allocation, warm-up). All of it is the build segment."""
        calls: List[Call] = []
        for spec in targets:
            key = f"{plan_id}/{spec.engine_id}" if plan_id else None
            calls += [(h, "start_engine", (spec, key)) for h in spec.hosts]
        with rec.phase("build", sorted({c[0] for c in calls})) as ph:
            ph.note["engines"] = [s.engine_id for s in targets]
            results = self.caller.call(calls, self.timeouts.start_s)
            ph.results = {c[0]: r for c, r in zip(calls, results)}
            for spec in targets:
                self._set_engine(spec)
            settings = {**self.frontend_settings, "layout_id": layout["layout_id"],
                        "layout_digest": layout["layout_digest"]}
            ids = [s.engine_id for s in targets]
            for spec in targets:
                name = frontend_name(spec)
                self.frontends[spec.engine_id] = self.frontend_factory(
                    spec, list(self.kernel_refs), dict(settings), name)
                self.frontend_specs[spec.engine_id] = spec
                self.frontend_names[spec.engine_id] = name
                self.frontend_layout[spec.engine_id] = (layout["layout_id"], layout["layout_digest"])
            ph.frontends = self._fcall_many(ids, "initialize", (), self.timeouts.start_s)

    def _layout_info(self, name: str, specs: Sequence[EngineSpec]) -> Dict[str, Any]:
        return layout_info(self.layout_id_for(name), specs)

    def _publish(self, rec, targets: Sequence[EngineSpec], layout: Dict[str, Any],
                 kept: Sequence[EngineSpec]) -> None:
        """Publish phase: kept frontends adopt the new layout identity, the targets' frontend
        identities are checked against it, then the endpoints publish."""
        with rec.phase("publish") as ph:
            ph.note["engines"] = [s.engine_id for s in targets]
            current = (layout["layout_id"], layout["layout_digest"])
            stale = [s.engine_id for s in kept if self.frontend_layout.get(s.engine_id) != current]
            if stale:
                self._fcall_many(stale, "adopt_identity", current, self.timeouts.describe_s)
                for engine_id in stale:
                    self.frontend_layout[engine_id] = current
            ids = [s.engine_id for s in targets]
            identities = self._fcall_many(ids, "identity", (), self.timeouts.describe_s)
            for engine_id, identity in identities.items():
                if not isinstance(identity, dict):
                    continue
                wrong = {k: identity.get(k) for k, v in (("layout_id", current[0]), ("layout_digest", current[1]),
                                                         ("generation", self.engines[engine_id].generation))
                         if identity.get(k, v) != v}
                if wrong:
                    raise RuntimeError(f"frontend {engine_id} identity disagrees with the published layout: {wrong}")
            info = {i: {"actor_name": self.frontend_names[i], "namespace": FRONTEND_NAMESPACE,
                        "identity": identities[i]} for i in ids}
            self.endpoints.publish(list(targets), frontends=info, layout=layout)

    # ----- frontends

    def _fcall_many(self, ids: Sequence[str], method: str, args: tuple, timeout_s: float) -> Dict[str, Any]:
        """``method(*args)`` on the frontends of ``ids`` concurrently. Dead/unresponsive ->
        ``FrontendFailure``, raised inside -> ``FrontendCallError``."""
        ids = list(ids)
        handles = []
        for engine_id in ids:
            if engine_id not in self.frontends:
                raise EngineNotFound(engine_id)
            handles.append(self.frontends[engine_id])
        try:
            results = make_caller(handles).call([(k, method, args) for k in range(len(ids))], timeout_s)
        except FrontendFailure:
            raise
        except KernelFailure as failure:
            bad = [ids[k] for k in failure.ranks if 0 <= k < len(ids)] or ids[:1]
            hosts = sorted({h for i in bad for h in self._frontend_hosts(i)})
            raise FrontendFailure(bad[0], method, failure.kind, failure.cause, ranks=hosts,
                                  engine_ids=bad) from failure
        except KernelCallError as error:
            raise FrontendCallError(ids[error.rank], method, error.cause) from error
        return dict(zip(ids, results))

    def _frontend_hosts(self, engine_id: str) -> Tuple[int, ...]:
        spec = self.frontend_specs.get(engine_id) or self.engines.get(engine_id)
        return tuple(spec.hosts) if spec else ()

    def _drain_frontends(self, ids: Sequence[str]) -> None:
        drained = self._fcall_many(ids, "drain", (self.timeouts.drain_wait_s,), self.timeouts.drain_wait_s + 60.0)
        refused = [i for i, r in drained.items() if isinstance(r, dict) and r.get("drained") is False]
        if refused:
            raise RuntimeError(f"frontends {refused} did not drain: {[drained[i] for i in refused]}")

    def _retire_frontend(self, engine_id: str, graceful: bool = True) -> None:
        """Shut the frontend down (graceful: ``shutdown()`` first) and kill its actor."""
        handle = self.frontends.pop(engine_id, None)
        self.frontend_specs.pop(engine_id, None)
        self.frontend_names.pop(engine_id, None)
        self.frontend_layout.pop(engine_id, None)
        if handle is None:
            return
        caller = make_caller([handle])
        if graceful:
            _quietly(lambda: caller.call_each([(0, "shutdown", ())], self.timeouts.stop_s))
        caller.kill_all()

    # ----- switch

    def switch(self, target: Any, resident: bool = True) -> Dict[str, Any]:
        """Replace the running layout by ``target``. Engines identical in id, hosts and
        options are kept. See the module docstring for the protocol."""
        with self._admin:
            if self.down:
                raise BlockDown("the block is down; call rebuild()")
            self.ensure_healthy()  # a dead kernel or poisoned frontend is rebuilt before the switch
            name, specs = self._resolve(target)  # illegal sets are refused before anything runs
            keep = {s.engine_id for s in specs
                    if s.engine_id in self.engines and self._same_engine(self.engines[s.engine_id], s)}
            sources = [e for i, e in sorted(self.engines.items()) if i not in keep]
            fresh = [s for s in specs if s.engine_id not in keep]
            if not sources and not fresh:
                return {"kind": "switch", "ok": True, "noop": True, "target": name}
            use_resident = bool(resident and sources and fresh)
            if use_resident:  # ResidentStore.reshard contract; refused here rather than after the sources stopped
                outside = sorted({h for s in sources for h in s.hosts} - {h for s in fresh for h in s.hosts})
                if outside:
                    raise ValueError(f"resident switch needs the source hosts inside the target hosts, "
                                     f"{outside} are not; use resident=False")
                if len({s.tp_size for s in fresh}) > 1:
                    raise ValueError("resident switch needs targets of one TP degree; use resident=False")
            targets = self._next_generation(fresh)
            kept_specs = [self.engines[i] for i in sorted(keep)]
            layout = self._layout_info(name, kept_specs + targets)
            self.switch_count += 1
            plan_id = f"e{self.epoch}s{self.switch_count}"
            src_ids = [s.engine_id for s in sources]
            tgt_ids = [s.engine_id for s in targets]
            src_hosts = sorted({h for s in sources for h in s.hosts})
            tgt_hosts = sorted({h for s in targets for h in s.hosts})
            self._sync_clocks()
            rec = self.measure.recorder()
            rec.begin()
            source_name, source_specs = self.layout_name, self.last_layout
            touched = False
            self._block(src_ids + tgt_ids)
            try:
                if sources:
                    with rec.phase("drain") as ph:
                        ph.note["engines"] = src_ids
                        self.endpoints.drain(src_ids)
                        self._wait_idle(src_ids)
                        self._drain_frontends(src_ids)
                    with rec.phase("withdraw") as ph:
                        ph.note["engines"] = src_ids
                        self.endpoints.withdraw(src_ids)
                    touched = True
                    manifests = self._stop(rec, sources, src_hosts, export=use_resident)
                if use_resident:
                    self._reshard(rec, plan_id, sources, targets, src_hosts, tgt_hosts, manifests)
                self._start_engines(rec, targets, plan_id if use_resident else None, layout)
                self._publish(rec, targets, layout, kept_specs)
            except BaseException as exc:
                partial = self._abort(exc, rec, kind="switch", target=name, mode=self._mode(use_resident),
                                      plan_id=plan_id, sources=src_ids, targets=tgt_ids)
                if touched or isinstance(exc, KernelFailure):
                    rebuild = self._recover(exc, epoch=None, layout=source_specs)
                    raise SwitchFailed(f"switch to {name} failed ({type(exc).__name__}: {exc}); "
                                       f"block rebuilt on {source_name}", partial, rebuild) from exc
                raise
            finally:
                self._unblock(src_ids + tgt_ids)
            previous = self.layout_name
            self.layout_name, self.last_layout = name, (name, specs)
            report = {"kind": "switch", "ok": True, "index": self.switch_count, "source": previous,
                      "target": name, "mode": self._mode(use_resident), "plan_id": plan_id,
                      "epoch": self.epoch, "sources": src_ids, "targets": tgt_ids, "kept": sorted(keep),
                      "layout_id": layout["layout_id"], "layout_digest": layout["layout_digest"],
                      "generations": {s.engine_id: s.generation for s in targets},
                      "runtime_ids": [d.runtime_id for d in self.descriptions], **rec.finish()}
            report["hbm"] = self._hbm()
            return self.measure.add_switch(report)

    @staticmethod
    def _mode(resident: bool) -> str:
        return "resident" if resident else "cold"

    # ----- in-engine layout switch

    def _layout_target(self, current: EngineSpec, target: Any) -> EngineSpec:
        """The validated target spec of ``switch_layout``: ``current`` with another (tp_size, dp_size). ``target``
        is a ``types.IN_ENGINE_LAYOUTS`` name, a ``(tp_size, dp_size)`` pair or an ``EngineSpec`` that equals
        ``current`` in everything else (same id, kind, hosts, model)."""
        if isinstance(target, str):
            if target not in IN_ENGINE_LAYOUTS:
                raise ValueError(f"unknown in-engine layout {target!r}; known: {sorted(IN_ENGINE_LAYOUTS)}")
            tp, dp = IN_ENGINE_LAYOUTS[target]
            spec = dataclasses.replace(current, tp_size=tp, dp_size=dp)
        elif isinstance(target, EngineSpec):
            spec = dataclasses.replace(target, generation=current.generation)
        else:
            tp, dp = target
            spec = dataclasses.replace(current, tp_size=int(tp), dp_size=int(dp))
        if (spec.engine_id, spec.kind, spec.hosts, spec.model) != (current.engine_id, current.kind, current.hosts,
                                                                   current.model):
            raise ValueError(f"layout target for {current.engine_id} must keep its id, kind, hosts and model")
        chips = self.chips_per_host
        wanted = len(spec.hosts) * chips if chips is not None else current.chips
        if spec.tp_size < 1 or spec.chips != wanted or spec.chips != current.chips:
            raise ValueError(f"{current.engine_id}: layout tp {spec.tp_size} x dp {spec.dp_size} must span "
                             f"{wanted} chips ({len(spec.hosts)} hosts)")
        return spec

    def switch_layout(self, engine_id: str, target: Any) -> Dict[str, Any]:
        """Switch one live engine to another (tp_size, dp_size) in place (module docstring). Returns the report
        (``kind: "layout_switch"``; phases drain, worker_switch, frontend_switch, publish; ``built``; per-host
        ``worker_segments``; ``frontend`` segments; runtime ids and host pids), also kept in ``measure.switches``.
        A target equal to the current layout is a no-op. Refused before anything changed (unknown engine,
        bad target, drain failure): raised as it is, the engine keeps serving. Failed after the hosts were
        touched: the engine is left stopped and ``LayoutSwitchFailed`` carries the report; no rebuild."""
        with self._admin:
            if self.down:
                raise BlockDown("the block is down; call rebuild()")
            self.ensure_healthy()
            current = self.engines.get(engine_id)
            if current is None:
                raise EngineNotFound(engine_id)
            spec = self._layout_target(current, target)
            if spec.layout == current.layout:
                return {"kind": "layout_switch", "ok": True, "noop": True, "engine_id": engine_id,
                        "target_layout": list(spec.layout)}
            self.switch_count += 1
            self._sync_clocks()
            identity = self.frontend_layout[engine_id]
            rec = self.measure.recorder()
            rec.begin()
            touched, built = False, None
            self._block([engine_id])
            try:
                with rec.phase("drain") as ph:
                    ph.note["engines"] = [engine_id]
                    self._drain_frontends([engine_id])  # latch the frontend's refusal first (gateway/direct callers)
                    self._wait_idle([engine_id])
                    # the scheduler tail (finished-request cleanup, batch queue) is not part of ``drain``
                    idle = self._fcall_many([engine_id], "await_idle", (self.timeouts.drain_wait_s,),
                                            self.timeouts.drain_wait_s + 60.0)[engine_id]
                    if not (isinstance(idle, dict) and idle.get("idle")):
                        raise RuntimeError(f"frontend {engine_id} is not idle after its drain: {idle}")
                touched = True
                with rec.phase("worker_switch", spec.hosts) as ph:
                    ph.note["engines"] = [engine_id]
                    calls = [(h, "switch_engine_layout", (spec,)) for h in spec.hosts]
                    results = self.caller.call(calls, self.timeouts.layout_s)
                    ph.results = dict(zip(spec.hosts, results))
                    ph.note["host_segments"] = {str(h): (r.get("segments") if isinstance(r, dict) else None)
                                                for h, r in ph.results.items()}
                    flags = {h: (r.get("built") if isinstance(r, dict) else None) for h, r in ph.results.items()}
                    ph.note["built_by_host"] = {str(h): v for h, v in flags.items()}
                    if any(not isinstance(v, bool) for v in flags.values()) or len(set(flags.values())) != 1:
                        raise RuntimeError(f"hosts disagree on whether layout {spec.layout} was built: {flags}")
                    built = next(iter(flags.values()))
                with rec.phase("frontend_switch") as ph:
                    ph.note["engines"] = [engine_id]
                    ph.frontends = self._fcall_many([engine_id], "switch_layout", (spec, built),
                                                    self.timeouts.layout_s)
                    ph.note["built"] = built
                self._set_engine(spec)
                self.frontend_specs[engine_id] = spec
                with rec.phase("publish") as ph:
                    ph.note["engines"] = [engine_id]
                    self._publish_layout(ph, spec, identity)
            except BaseException as exc:
                partial = self._abort(exc, rec, kind="layout_switch", engine_id=engine_id,
                                      source_layout=list(current.layout), target_layout=list(spec.layout),
                                      built=built, stopped=touched)
                if not touched:
                    _quietly(lambda: self._fcall_many([engine_id], "resume_admission", (), self.timeouts.describe_s))
                    raise
                self._stop_engine_after_failed_layout_switch(engine_id)
                raise LayoutSwitchFailed(f"layout switch of {engine_id} to {spec.layout} failed "
                                         f"({type(exc).__name__}: {exc}); engine left stopped, no rebuild",
                                         partial) from exc
            finally:
                self._unblock([engine_id])
            specs_by_id = {s.engine_id: s for s in (self.last_layout[1] if self.last_layout else ())}
            if engine_id in specs_by_id and self.last_layout is not None:
                specs_by_id[engine_id] = dataclasses.replace(specs_by_id[engine_id], tp_size=spec.tp_size,
                                                             dp_size=spec.dp_size)
                self.last_layout = (self.last_layout[0], [specs_by_id[s.engine_id] for s in self.last_layout[1]])
            front = rec.rows[-2]["parts"] if len(rec.rows) >= 2 else {}
            report = {"kind": "layout_switch", "ok": True, "index": self.switch_count, "engine_id": engine_id,
                      "source_layout": list(current.layout), "target_layout": list(spec.layout), "built": built,
                      "epoch": self.epoch, "generation": spec.generation, "layout_id": identity[0],
                      "layout_digest": identity[1], "frontend_segments": front,
                      "worker_segments": rec.rows[1].get("host_segments"),
                      "runtime_ids": [d.runtime_id for d in self.descriptions], **rec.finish()}
            report["host_pids"] = [d.pid for d in self.describe()]
            report["hbm"] = self._hbm()
            return self.measure.add_switch(report)

    def _publish_layout(self, ph, spec: EngineSpec, identity: Tuple[str, str]) -> None:
        """Publish phase of an in-engine switch: the frontend still reports the published layout identity
        and generation, the endpoint (if it can) learns the new layout, admission resumes."""
        engine_id = spec.engine_id
        reported = self._fcall_many([engine_id], "identity", (), self.timeouts.describe_s)[engine_id]
        if isinstance(reported, dict):
            wrong = {k: reported.get(k) for k, v in (("layout_id", identity[0]), ("layout_digest", identity[1]),
                                                     ("generation", spec.generation))
                     if reported.get(k, v) != v}
            wrong.update({k: reported.get(k) for k, v in (("tp_size", spec.tp_size), ("dp_size", spec.dp_size))
                          if k in reported and reported.get(k) != v})
            if wrong:
                raise RuntimeError(f"frontend {engine_id} identity disagrees after the layout switch: {wrong}")
        update = getattr(self.endpoints, "update_engine", None)
        if callable(update):
            info = {engine_id: {"actor_name": self.frontend_names[engine_id], "namespace": FRONTEND_NAMESPACE,
                                "identity": reported}}
            ph.note["endpoint_update"] = update(spec, frontends=info,
                                                layout={"layout_id": identity[0], "layout_digest": identity[1]})
        else:
            ph.note["endpoint_update"] = "none: layout id, digest, generation and endpoint unchanged"
        self._fcall_many([engine_id], "resume_admission", (), self.timeouts.describe_s)

    def _stop_engine_after_failed_layout_switch(self, engine_id: str) -> None:
        """Leave the engine stopped: frontend retired and killed, engine and placement dropped, endpoint
        withdrawn. The host kernels are not called (an SPMD step may be wedged); ``rebuild()`` is the way back."""
        self._retire_frontend(engine_id, graceful=False)
        self._drop_engine(engine_id)
        _quietly(lambda: self.endpoints.withdraw([engine_id]))

    def _stop(self, rec, sources: Sequence[EngineSpec], hosts: Sequence[int], export: bool) -> Dict[int, Any]:
        with rec.phase("stop_export", hosts) as ph:
            ph.note["engines"] = [s.engine_id for s in sources]
            for spec in sources:  # frontend gone (shutdown, then killed) before any kernel stops its engine
                self._retire_frontend(spec.engine_id)
            results = self.caller.call([(h, "stop_engine", (export,)) for h in hosts], self.timeouts.stop_s)
            ph.results = dict(zip(hosts, results))
        for spec in sources:
            self._drop_engine(spec.engine_id)
        manifests = dict(zip(hosts, results))
        if export and any(m is None for m in manifests.values()):
            raise RuntimeError(f"hosts {[h for h, m in manifests.items() if m is None]} exported nothing")
        return manifests

    def _reshard(self, rec, plan_id: str, sources, targets, src_hosts, tgt_hosts, manifests) -> None:
        hosts = sorted(set(src_hosts) | set(tgt_hosts))
        plan = ReshardPlan(plan_id, tuple(s.engine_id for s in sources), tuple(targets),
                           drop_sources=True, checksums=self.checksums)
        with rec.phase("reshard", hosts) as ph:
            ph.note["engines"] = [s.engine_id for s in targets]
            results = self.caller.call([(h, "reshard", (plan,)) for h in hosts], self.timeouts.reshard_s)
            ph.results = dict(zip(hosts, results))
        if self.checksums:
            unverified = [h for h, r in ph.results.items() if not (isinstance(r, dict) and r.get("checksums_verified"))]
            if unverified:
                raise WeightMismatch(f"hosts {unverified} did not verify logical weight checksums")

    # ----- failure domain

    def health_check(self) -> Dict[str, Any]:
        """Concurrent ``health()`` on every kernel and every frontend. A kernel is failed when
        it is dead, does not answer within ``timeouts.health_s``, or reports ``ok: False``; a
        frontend likewise, or when it reports ``poisoned``."""
        calls = [(h, "health", ()) for h in range(self.hosts)]
        outcomes = self.caller.call_each(calls, self.timeouts.health_s)
        kernels, failed = {}, []
        for h, (kind, value) in enumerate(outcomes):
            ok = kind == "ok" and not (isinstance(value, dict)
                                       and (value.get("ok") is False or value.get("alive") is False))
            kernels[h] = {"status": kind if kind != "ok" or ok else "unhealthy",
                          "detail": value if kind == "ok" else repr(value)}
            if not ok:
                failed.append(h)
        ids = list(self.frontends)
        frontends, failed_frontends = {}, []
        if ids:
            handles = [self.frontends[i] for i in ids]
            outs = make_caller(handles).call_each([(k, "health", ()) for k in range(len(ids))],
                                                  self.timeouts.health_s)
            for engine_id, (kind, value) in zip(ids, outs):
                ok = kind == "ok" and not (isinstance(value, dict) and (
                    value.get("ok") is False or value.get("poisoned") or value.get("alive") is False))
                frontends[engine_id] = {"status": kind if kind != "ok" or ok else "unhealthy",
                                        "detail": value if kind == "ok" else repr(value)}
                if not ok:
                    failed_frontends.append(engine_id)
        return {"healthy": not failed and not failed_frontends, "failed": failed,
                "failed_frontends": failed_frontends, "kernels": kernels, "frontends": frontends}

    def _failure_from_probe(self, probe: Dict[str, Any]) -> KernelFailure:
        if probe["failed"]:
            first = probe["failed"][0]
            return KernelFailure(first, "health", probe["kernels"][first]["status"], ranks=probe["failed"])
        ids = probe["failed_frontends"]
        hosts = sorted({h for i in ids for h in self.frontend_specs[i].hosts})
        return FrontendFailure(ids[0], "health", probe["frontends"][ids[0]]["status"], ranks=hosts,
                               engine_ids=ids)

    def _diagnose(self, error: BaseException) -> Optional[KernelFailure]:
        """A call raised inside a live frontend. Poisoned frontend or dead kernel behind it is
        a block failure; an ordinary request error is not."""
        probe = self.health_check()
        return None if probe["healthy"] else self._failure_from_probe(probe)

    def ensure_healthy(self) -> Optional[Dict[str, Any]]:
        """Probe; rebuild the block when any kernel or frontend is failed. Returns the rebuild report."""
        with self._admin:
            probe = self.health_check()
            if probe["healthy"]:
                return None
            return self._recover(self._failure_from_probe(probe), epoch=None)

    def rebuild(self, cause: str = "manual") -> Optional[Dict[str, Any]]:
        """Tear the block down and bring it back (also the way out of ``BlockDown``)."""
        with self._admin:
            return self._recover(RuntimeError(cause), epoch=None)

    def _abort(self, exc: BaseException, rec, **fields: Any) -> Dict[str, Any]:
        partial = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}", **fields,
                   "epoch": self.epoch, **rec.finish()}
        (self.measure.switches if fields.get("kind") in ("switch", "layout_switch")
         else self.measure.deploys).append(partial)
        return partial

    def _recover(self, cause: BaseException, epoch: Optional[int],
                 layout: Optional[Tuple[str, List[EngineSpec]]] = None,
                 redeploy: bool = True) -> Optional[Dict[str, Any]]:
        """Tear down, respawn, redeploy ``layout`` (default: the last layout) cold, publish.
        ``epoch`` set: skip when the block was already rebuilt since the caller started.
        When the replacement block fails to come up, it is torn down again (published engines
        withdrawn, frontends and kernels killed) before ``BlockDown`` is raised."""
        with self._admin:
            if epoch is not None and epoch != self.epoch:
                return None
            self._block([], everything=True)
            rec = self.measure.recorder()
            rec.begin()
            layout = (layout or self.last_layout) if redeploy else None
            report: Dict[str, Any] = {
                "kind": "rebuild", "cause": f"{type(cause).__name__}: {str(cause)[:300]}",
                "failed_hosts": list(getattr(cause, "ranks", ())),
                "failed_frontends": list(getattr(cause, "engine_ids", ())), "from_epoch": self.epoch,
                "layout": layout[0] if layout else None, "ok": False}
            try:
                self.down = True
                self._wait_quiet()
                with rec.phase("teardown"):
                    self._teardown_block()
                with rec.phase("respawn"):
                    block = self._respawn()
                    self.kernels = list(block.kernels)
                    if block.refs is not None:
                        self.kernel_refs = list(block.refs)
                    self.caller = make_caller(self.kernels)
                    self.epoch += 1
                with rec.phase("describe"):
                    self.describe()
                    self._sync_clocks()
                if layout is not None:
                    targets = self._next_generation(layout[1])
                    info = self._layout_info(layout[0], targets)
                    self._start_engines(rec, targets, None, info)
                    self._publish(rec, targets, info, [])
                    self.layout_name, self.last_layout = layout[0], layout
                    report["generations"] = {s.engine_id: s.generation for s in targets}
                self.down = False
                report.update(ok=True, epoch=self.epoch,
                              runtime_ids=[d.runtime_id for d in self.descriptions])
            except BaseException as exc:
                _quietly(self._teardown_block)  # the replacement block must not outlive a failed rebuild
                report["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                report.update(rec.finish())
                self.measure.rebuilds.append(report)
                raise BlockDown(f"block rebuild failed: {report['error']}") from exc
            finally:
                self._unblock(everything=True)
            report.update(rec.finish())
            self.measure.rebuilds.append(report)
            return report

    def _teardown_block(self) -> None:
        """Withdraw everything published, kill every frontend (no graceful shutdown: their kernels
        may be gone) and the kernels. Never raises."""
        ids = sorted(set(self.engines) | set(self.endpoints_live()))
        if ids:
            _quietly(lambda: self.endpoints.withdraw(ids))
        for engine_id in list(self.frontends):
            _quietly(lambda e=engine_id: self._retire_frontend(e, graceful=False))
        _quietly(self.caller.kill_all)
        with self._cv:
            self.engines.clear()
            self.placement.clear()

    def endpoints_live(self) -> List[str]:
        for attr in ("published", "live"):
            live = getattr(self.endpoints, attr, None)
            if isinstance(live, (dict, set)):
                return list(live)
        return []

    def _wait_quiet(self) -> None:
        """Requests in flight on dead kernels fail by themselves; give them a moment to leave."""
        deadline = time.monotonic() + 5.0
        with self._cv:
            while any(self._inflight.values()) and time.monotonic() < deadline:
                self._cv.wait(0.1)

    def _respawn(self) -> Block:
        if self.respawn is not None:
            out = self.respawn()
            return out if isinstance(out, Block) else Block(list(out))
        prefix = f"{self.kernel_prefix}-e{self.epoch + 1}"
        options = {"name_prefix": prefix, "actor_options": {"namespace": self.kernel_namespace},
                   **self.spawn_options}
        namespace = (options["actor_options"] or {}).get("namespace", self.kernel_namespace) \
            if isinstance(options["actor_options"], dict) else self.kernel_namespace
        refs = [{"name": f"{options['name_prefix']}-{h}", "namespace": namespace} for h in range(self.hosts)]
        return Block(default_respawn(self.hosts, self.chips_per_host or 4, **options), refs)

    # ----- reporting

    def _hbm(self) -> Optional[Dict[str, Any]]:
        """HBM per chip from ``health()["hbm"]`` (None on CPU or when kernels do not report it)."""
        try:
            probes = self.caller.call([(h, "health", ()) for h in range(self.hosts)], self.timeouts.health_s)
        except (KernelFailure, KernelCallError):
            return None
        return measure_mod.hbm_summary({h: p.get("hbm") if isinstance(p, dict) else None
                                        for h, p in enumerate(probes)})

    def report(self) -> Dict[str, Any]:
        return self.measure.report()

    def write_report(self, path: Any):
        return self.measure.write(path)

    def close(self) -> None:
        """Retire the frontends, withdraw the engines, ask the kernels to shut down, kill what is left."""
        with self._admin:
            for engine_id in list(self.frontends):
                self._retire_frontend(engine_id)
            ids = sorted(self.engines)
            if ids:
                _quietly(lambda: self.endpoints.withdraw(ids))
            _quietly(lambda: self.caller.call_each([(h, "shutdown", ()) for h in range(self.hosts)], 30.0))
            self.caller.kill_all()
            with self._cv:
                self.engines.clear()
                self.placement.clear()
