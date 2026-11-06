"""Shared contracts for the virtual-process block runtime (plan: results/vproc/plan.md).

One JAX runtime spans a block of whole hosts. Each host runs one long-lived OS
process (the *host kernel*, one Ray actor) whose ``jax.process_index()`` is its
host rank. A host kernel runs at most one engine at a time. An engine is a
*virtual process*: under a view (C2) it sees a JAX world made of its hosts only.

Physical facts these contracts rely on:
- Block A on v5e: 4 hosts x 4 chips, libtpu topology 4x4. Host k sits at
  host-grid position (k % 2, k // 2).
- CPU stand-in (``hetero.vproc.cpu_block``): N processes x D CPU devices, gloo
  collectives, one ``jax.distributed`` runtime. Process p owns device ids
  ``p * 2048 + i`` (jax 0.10.2 and 0.11.2).
- Device ``.id/.process_index/.coords`` are read-only C++ attributes; only the
  engine-facing API is virtualized, ``jax._src`` stays physical.

Everything here is plain data (picklable for Ray and multiprocessing pipes),
except the two Protocols that pin the method names peers call.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence, Tuple

BLOCK_HOSTS = 4
CHIPS_PER_HOST = 4

# Deviceless compile of a TP program over the hosts' chips, libtpu 0.0.43,
# v5e 4x4 and v6e 4x4 (results/vproc/deviceless-host-sets-libtpu0.0.43.json):
# every host set compiles except these two (net_util.cc:2304).
ILLEGAL_HOST_SETS = frozenset({frozenset({0}), frozenset({0, 2})})

# Gemma-4 31B bf16 needs >= 2 hosts, so engines are TP8 (2 hosts) or TP16 (4).
MIN_ENGINE_HOSTS = 2

# Layouts inside one block: name -> host sets, one per engine.
# Columns {0,2}+{1,3} are not a layout because {0,2} does not compile.
BLOCK_LAYOUTS: Dict[str, Tuple[Tuple[int, ...], ...]] = {
    "S-block": ((0, 1), (2, 3)),
    "S-diag": ((0, 3), (1, 2)),
    "H-block": ((0, 1, 2, 3),),
}

# In-engine layouts of one engine over all four hosts of a block (non-transparent
# switch, results/in_engine_dp/): name -> (tp_size, dp_size). "S-dp" is one
# engine with single-process SPMD data parallelism (tpu-inference ``data`` mesh
# axis + DPScheduler); data group g runs on hosts (2g, 2g+1) like S-block's TP8s.
IN_ENGINE_LAYOUTS: Dict[str, Tuple[int, int]] = {
    "H": (16, 1),
    "S-dp": (8, 2),
}


def host_grid(host: int) -> Tuple[int, int]:
    return host % 2, host // 2


class IllegalHostSet(ValueError):
    """The host set cannot carry an engine (compile matrix or model size)."""


def check_host_set(hosts: Iterable[int], block_hosts: int = BLOCK_HOSTS,
                   min_hosts: int = MIN_ENGINE_HOSTS) -> Tuple[int, ...]:
    """Return the canonical (sorted) host tuple or raise ``IllegalHostSet``."""
    canonical = tuple(sorted(hosts))
    if len(set(canonical)) != len(canonical):
        raise IllegalHostSet(f"duplicate hosts in {canonical}")
    if not canonical or canonical[0] < 0 or canonical[-1] >= block_hosts:
        raise IllegalHostSet(f"hosts {canonical} outside block of {block_hosts}")
    if len(canonical) < min_hosts:
        raise IllegalHostSet(f"hosts {canonical}: an engine needs >= {min_hosts} hosts")
    if frozenset(canonical) in ILLEGAL_HOST_SETS:
        raise IllegalHostSet(f"hosts {canonical} do not compile on a 4x4 block (net_util.cc:2304)")
    return canonical


@dataclasses.dataclass(frozen=True)
class ViewSpec:
    """The JAX world one engine sees. ``hosts`` are physical process indices.

    Virtual rank r is ``hosts[r]``; ``namespace`` prefixes every coordination
    service key and barrier id the engine creates.
    """

    engine_id: str
    hosts: Tuple[int, ...]
    namespace: str = ""

    def __post_init__(self):
        hosts = tuple(int(h) for h in self.hosts)
        if list(hosts) != sorted(set(hosts)) or not hosts:
            raise ValueError(f"view hosts must be unique and ascending, got {self.hosts}")
        object.__setattr__(self, "hosts", hosts)
        if not self.namespace:
            object.__setattr__(self, "namespace", f"vproc/{self.engine_id}")

    @property
    def size(self) -> int:
        return len(self.hosts)

    def rank_of(self, physical_process: int) -> int:
        return self.hosts.index(physical_process)

    def physical_of(self, rank: int) -> int:
        return self.hosts[rank]


@dataclasses.dataclass(frozen=True)
class EngineSpec:
    """One engine instance. ``kind`` selects the class in ``ENGINE_KINDS``.

    ``model`` is kind-specific config: synthetic engines take a catalog name and
    seed (``hetero.vproc.catalog``); vLLM engines take the vLLM engine args.
    ``generation`` increments every time the controller rebuilds this id.
    ``dp_size`` > 1 is single-process SPMD data parallelism inside one engine:
    the mesh is (data=dp_size, model=tp_size) over ``hosts``, and data group g
    runs on the g-th consecutive ``len(hosts) // dp_size`` hosts.
    """

    engine_id: str
    kind: str
    hosts: Tuple[int, ...]
    tp_size: int
    model: Dict[str, Any] = dataclasses.field(default_factory=dict)
    generation: int = 0
    dp_size: int = 1

    def __post_init__(self):
        object.__setattr__(self, "hosts", tuple(sorted(int(h) for h in self.hosts)))
        if self.dp_size < 1 or len(self.hosts) % self.dp_size:
            raise ValueError(f"{self.engine_id}: dp_size {self.dp_size} does not split hosts {self.hosts}")

    @property
    def chips(self) -> int:
        """Chips the engine's mesh spans: tp_size x dp_size."""
        return self.tp_size * self.dp_size

    @property
    def layout(self) -> Tuple[int, int]:
        """In-engine layout key (tp_size, dp_size)."""
        return self.tp_size, self.dp_size

    def data_groups(self) -> Tuple[Tuple[int, ...], ...]:
        """Hosts of each data group, in data-axis order."""
        per = len(self.hosts) // self.dp_size
        return tuple(self.hosts[g * per:(g + 1) * per] for g in range(self.dp_size))

    def view(self) -> ViewSpec:
        return ViewSpec(self.engine_id, self.hosts)


# Engine kind -> "module:Class". The class implements ``Engine``.
ENGINE_KINDS: Dict[str, str] = {
    "synthetic": "hetero.vproc.synthetic_engine:SyntheticEngine",
    "vllm": "hetero.vproc.engine_vllm:VllmEngine",
}


@dataclasses.dataclass(frozen=True)
class LeafSpec:
    """One parameter leaf of the logical (unsharded) model.

    ``tp_axis`` is the dimension sharded over the engine's TP mesh axis (None =
    replicated). ``fused`` names a per-shard interleaved layout whose bytes must
    be repacked when the TP degree changes: ``"qkv"`` or ``"gate_up"``, with
    ``fused_sizes`` the logical widths of the fused parts along ``tp_axis``.
    """

    path: str
    shape: Tuple[int, ...]
    dtype: str
    tp_axis: Optional[int]
    fused: Optional[str] = None
    fused_sizes: Tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class ResidentManifest:
    """What a stopped engine left in device memory, as seen from one host.

    ``checksums`` maps leaf path -> hex digest of the *logical* array, so a
    TP8 replica and a TP16 copy of the same weights compare equal. Empty when
    the caller did not ask for checksums (they cost a gather).
    """

    engine_id: str
    generation: int
    hosts: Tuple[int, ...]
    tp_size: int
    leaves: Tuple[LeafSpec, ...]
    bytes_per_host: int
    checksums: Dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ReshardPlan:
    """Turn resident states of stopped engines into resident states for targets.

    Every host in the union of source and target hosts calls
    ``HostKernel.reshard(plan)`` concurrently (one SPMD program in the physical
    view). Afterwards each target's state is held under key
    ``f"{plan.plan_id}/{target.engine_id}"`` until ``start_engine`` consumes it.

    The target sharding is the one the target engine will build: its mesh
    device order comes from ``Engine.mesh_device_ids(spec, devices)`` of the
    target's class (``ENGINE_KINDS``). The reshard never picks an order itself;
    the controller picks engine options (e.g. a synthetic engine's
    ``model["device_order"]``) and the reshard follows.
    """

    plan_id: str
    sources: Tuple[str, ...]
    targets: Tuple[EngineSpec, ...]
    drop_sources: bool = True
    checksums: bool = False


@dataclasses.dataclass(frozen=True)
class HostDescription:
    rank: int
    processes: int
    platform: str
    device_ids: Tuple[int, ...]
    hostname: str
    pid: int
    runtime_id: str
    engine: Optional[EngineSpec]
    resident: Tuple[str, ...]
    clock_s: float


class Engine(Protocol):
    """An engine as a virtual process. Every method runs under its view and is
    SPMD: the host kernels of all of the view's hosts call it concurrently.
    Constructed as ``cls(spec)``; nothing touches devices before ``build``.
    """

    spec: EngineSpec

    @staticmethod
    def mesh_device_ids(spec: EngineSpec, devices: Sequence[Any]) -> Tuple[int, ...]:
        """Device ids, in mesh order, of the mesh this engine builds when its
        view shows ``devices`` (the view's devices in physical order). For
        ``dp_size`` > 1 the ids are (data, model) row-major: row g is the TP
        ring the engine would use for a TP``tp_size`` engine on
        ``spec.data_groups()[g]``. Pure: no global state, callable in the
        physical view by the reshard (C4)."""

    def build(self, resident: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Build from resident leaves (path -> jax.Array on the view's devices,
        stored layout for ``spec.tp_size``) or, when None, from the model
        source. Returns build segments in seconds (e.g. ``load``, ``graph``,
        ``kv_alloc``, ``warmup``)."""

    def step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Serve one request batch; returns at least ``{"tokens": [[int]]}``."""

    def export(self) -> Dict[str, Any]:
        """Hand the parameter leaves over (path -> jax.Array in the engine's
        stored layout, i.e. fused leaves interleaved for ``spec.tp_size``) and
        drop everything else (KV cache, executables). Unusable afterwards."""

    def leaf_specs(self) -> Tuple[LeafSpec, ...]:
        """Logical specs of the leaves ``export`` returns (``hetero.vproc.catalog``)."""

    def switch_layout(self, target: EngineSpec) -> Dict[str, Any]:
        """Optional, non-transparent: switch this live engine to ``target``'s
        (tp_size, dp_size) in place. ``target`` equals ``self.spec`` except for
        ``tp_size``/``dp_size`` (same id, kind, hosts, model). Requires no
        request in flight. Reshards the parameter leaves in device memory,
        keeps the device runtime, the worker and the executables of every
        layout visited before; afterwards ``self.spec`` is ``target``.
        Returns ``{"built": bool, "segments": {name: s}, "compiles": int,
        "hbm": {...}}``. ``built`` is True on the first visit of a layout; a
        vLLM engine then still needs KV init and warm-up from its EngineCore.
        False: the engine serves in ``target`` as soon as this returns."""

    def shutdown(self) -> None:
        ...


class HostKernelAPI(Protocol):
    """RPC surface of one host kernel (C1). The controller (C5) calls these on
    Ray actors; SPMD calls must be issued to all involved hosts concurrently.
    ``describe`` and ``health`` must answer while another call is running.
    """

    def describe(self) -> HostDescription: ...

    def start_engine(self, spec: EngineSpec, resident: Optional[str] = None) -> Dict[str, Any]:
        """Build the engine under ``spec.view()``. ``resident`` is a resident
        store key to consume: ``f"{plan_id}/{engine_id}"`` after a reshard, or
        the ``engine_id`` of a state exported on the same hosts (restart without
        reshard). None: load from the model source."""

    def engine_call(self, method: str, *args: Any) -> Any:
        """Run an engine method under the view (e.g. ``"step", request``)."""

    def stop_engine(self, export: bool = True) -> Optional[ResidentManifest]: ...

    def switch_engine_layout(self, target: EngineSpec) -> Dict[str, Any]:
        """SPMD on all hosts of the running engine: ``Engine.switch_layout``
        under the engine's view; updates the kernel's engine spec."""

    def reshard(self, plan: ReshardPlan) -> Dict[str, Any]: ...

    def health(self) -> Dict[str, Any]: ...

    def shutdown(self) -> None: ...
