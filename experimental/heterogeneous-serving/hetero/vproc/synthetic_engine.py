"""Synthetic engine: Gemma-4-shaped sharded weights and an exact integer forward.

Stands in for a vLLM engine in the host-kernel and controller tests. It holds
the leaves of ``catalog.scaled(**model["catalog"])`` sharded over a ``"model"``
mesh axis (fused leaves in the stored layout for ``spec.tp_size``) and serves
tokens that depend on every leaf's contents.

Forward: for each leaf, ``s = sum(w * c)`` in wrapping uint32, where ``c`` is a
fixed per-leaf coefficient array that is position-dependent in the *logical*
layout and held in the same stored layout as ``w``. Wrapping integer addition
is associative, so the reduction is exact for any TP degree or device order,
and ``s`` equals the logical dot product only if the stored bytes are the
correct layout for ``spec.tp_size``. A wrong reshard or repack changes ``s``
and so the tokens. Tokens are a deterministic hash chain over the prompt and
all ``s``.

``spec.model`` keys (all optional): ``catalog`` (kwargs of ``catalog.scaled``),
``seed`` (int, 0), ``device_order`` ("natural" | "halves_interleaved"),
``vocab`` (int, 32000).
"""

from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from . import catalog
from .types import EngineSpec, LeafSpec

DEVICE_ORDERS = ("natural", "halves_interleaved")
MASK = 0xFFFFFFFF


def _order(spec: EngineSpec) -> str:
    order = spec.model.get("device_order", "natural")
    if order not in DEVICE_ORDERS:
        raise ValueError(f"engine {spec.engine_id}: device_order {order!r} not in {DEVICE_ORDERS}")
    return order


def _partition_spec(leaf: LeafSpec):
    from jax.sharding import PartitionSpec as P
    if leaf.tp_axis is None:
        return P()
    return P(*([None] * leaf.tp_axis + ["model"]))


def _coefficients(leaf: LeafSpec, seed: int) -> np.ndarray:
    """Logical coefficient array: small positive ints, deterministic per leaf."""
    digest = hashlib.sha256(f"coef:{seed}:{leaf.path}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.integers(1, 256, size=leaf.shape).astype(np.int32)


def _ring(ids: Sequence[int], order: str) -> Tuple[int, ...]:
    if order == "natural":
        return tuple(ids)
    if len(ids) % 2:
        raise ValueError(f"halves_interleaved needs an even device count, got {len(ids)}")
    half = len(ids) // 2
    out = []
    for a, b in zip(ids[:half], ids[half:]):
        out += [a, b]
    return tuple(out)


def chain(sums: Sequence[int], prompts: Sequence[Sequence[int]], steps: int, vocab: int):
    """Token hash chain over the prompt and every leaf sum (pure Python, exact)."""
    tokens = []
    for row in prompts:
        state = 2166136261
        for token in row:
            state = ((state ^ int(token)) * 16777619) & MASK
        generated = []
        for _ in range(steps):
            mix = 0
            for index, s in enumerate(sums):
                mix = (mix + s * (((state >> (index % 24)) & 0xFF) | 1)) & MASK
            state = (state * 1664525 + 1013904223 + mix) & MASK
            generated.append((state >> 8) % vocab)
        tokens.append(generated)
    return tokens


def expected_tokens(spec: EngineSpec, request: Dict[str, Any]) -> list:
    """Tokens the engine must produce, from the logical weights alone (numpy; no
    sharding, no stored layout): the oracle for reshard/repack tests."""
    seed = int(spec.model.get("seed", 0))
    sums = []
    for leaf in catalog.scaled(**spec.model.get("catalog", {})):
        w = catalog.synthetic_value(leaf, seed).astype(np.int32).astype(np.uint32)
        c = _coefficients(leaf, seed).astype(np.uint32)
        sums.append(int(np.sum(w * c, dtype=np.uint32)))
    return chain(sums, request["prompt"], int(request["steps"]), int(spec.model.get("vocab", 32000)))


class SyntheticEngine:
    """``types.Engine`` over synthetic weights. See the module docstring."""

    def __init__(self, spec: EngineSpec):
        _order(spec)
        self.spec = spec
        self._specs: Optional[Tuple[LeafSpec, ...]] = None
        self._mesh = None
        self._params: Optional[Dict[str, Any]] = None
        self._coefs: Optional[Dict[str, Any]] = None
        self._program = None
        self._layouts: Dict[Tuple[int, int], Tuple[Any, Any]] = {}  # (tp, dp) -> (mesh, compiled step)
        self._meshes: Dict[Tuple[int, int], Any] = {}
        self._closed = False

    # -- topology ---------------------------------------------------------

    @staticmethod
    def mesh_device_ids(spec: EngineSpec, devices: Sequence[Any]) -> Tuple[int, ...]:
        """natural: the view's devices in physical order. halves_interleaved:
        A = devices of the first half of the view's hosts, B = the second half
        (each in physical order); mesh order [A0, B0, A1, B1, ...], so an
        S-block -> H-block switch is chip-local. ``dp_size`` > 1: (data, model) row-major, row g
        is that order over the devices of ``spec.data_groups()[g]`` (the TP ring of an S-block engine)."""
        if spec.dp_size == 1:
            return _ring([int(d.id) for d in devices], _order(spec))
        rows: Tuple[int, ...] = ()
        for hosts in spec.data_groups():
            group = [int(d.id) for d in devices if d.process_index in hosts]
            if len(group) != spec.tp_size:
                raise ValueError(f"data group on hosts {hosts} has {len(group)} devices, "
                                 f"TP{spec.tp_size} needs {spec.tp_size}")
            rows += _ring(group, _order(spec))
        return rows

    def leaf_specs(self) -> Tuple[LeafSpec, ...]:
        if self._specs is None:
            self._specs = catalog.scaled(**self.spec.model.get("catalog", {}))
        return self._specs

    def _mesh_for(self, spec: EngineSpec):
        """The mesh of ``spec``'s layout (cached per ``(tp, dp)``, so shardings compare equal across
        visits): ``("model",)`` for dp 1, ``(data, model)`` row-major in ``mesh_device_ids`` order else."""
        import jax
        from jax.sharding import Mesh
        cached = self._meshes.get(spec.layout)
        if cached is not None:
            return cached
        devices = [d for d in jax.devices() if d.process_index in spec.hosts]
        if len(devices) != spec.chips:
            raise ValueError(f"engine {spec.engine_id}: tp_size {spec.tp_size} x dp_size {spec.dp_size} but "
                             f"the view shows {len(devices)} devices of hosts {spec.hosts}")
        by_id = {int(d.id): d for d in devices}
        grid = np.array([by_id[i] for i in self.mesh_device_ids(spec, devices)])
        mesh = (Mesh(grid, ("model",)) if spec.dp_size == 1
                else Mesh(grid.reshape(spec.dp_size, spec.tp_size), ("data", "model")))
        self._meshes[spec.layout] = mesh
        return mesh

    # -- build ------------------------------------------------------------

    def build(self, resident: Optional[Dict[str, Any]]) -> Dict[str, float]:
        import jax
        if self._closed or self._params is not None:
            raise RuntimeError(f"engine {self.spec.engine_id}: build on a used engine")
        spec, seed = self.spec, int(self.spec.model.get("seed", 0))
        leaves = self.leaf_specs()
        mesh = self._mesh_for(spec)
        t0 = time.perf_counter()
        if resident is None:
            params = {l.path: self._put(catalog.synthetic_value(l, seed), l, spec.tp_size, mesh) for l in leaves}
        else:
            if spec.dp_size > 1:
                raise ValueError("resident leaves are TP-only; a dp_size > 1 engine is built from the model "
                                 "source or entered with switch_layout")
            params = self._adopt(resident, leaves, mesh)
        coefs = self._make_coefs(leaves, seed, spec.tp_size, mesh)
        jax.block_until_ready((params, coefs))
        t1 = time.perf_counter()
        compiled = self._compile(leaves, mesh, params, coefs)
        t2 = time.perf_counter()
        self._params, self._coefs, self._program, self._mesh = params, coefs, compiled, mesh
        self._layouts[spec.layout] = (mesh, compiled)
        self._read()
        t3 = time.perf_counter()
        return {"load": t1 - t0, "graph": t2 - t1, "warmup": t3 - t2}

    @staticmethod
    def _put(logical: np.ndarray, leaf: LeafSpec, tp: int, mesh):
        import jax
        from jax.sharding import NamedSharding
        stored = catalog.to_stored(logical, leaf, tp)
        sharding = NamedSharding(mesh, _partition_spec(leaf))
        return jax.make_array_from_callback(leaf.shape, sharding, lambda index: stored[index])

    def _make_coefs(self, leaves, seed: int, tp: int, mesh) -> Dict[str, Any]:
        return {l.path: self._put(_coefficients(l, seed), l, tp, mesh) for l in leaves}

    @staticmethod
    def _compile(leaves, mesh, params, coefs):
        import jax
        from jax.sharding import NamedSharding, PartitionSpec
        paths = [l.path for l in leaves]
        replicated = NamedSharding(mesh, PartitionSpec())

        def read_all(params, coefs):
            import jax.numpy as jnp
            out = []
            for path in paths:
                w = params[path].astype(jnp.int32).astype(jnp.uint32)
                c = coefs[path].astype(jnp.int32).astype(jnp.uint32)
                out.append(jnp.sum(w * c, dtype=jnp.uint32))
            return jnp.stack(out)

        return jax.jit(read_all, out_shardings=replicated).lower(params, coefs).compile()

    # -- in-place layout switch ---------------------------------------------

    def switch_layout(self, target: EngineSpec) -> Dict[str, Any]:
        """Same semantics as ``VllmEngine.switch_layout``. The step program of every layout visited is
        kept (compiled for that layout's mesh, one per ``(tp, dp)``); the parameter leaves are
        resharded with ``resident.reshard_leaves`` onto the target mesh (a ``(data, model)`` mesh for
        ``dp_size`` > 1, replicated over ``data``) and the sources are deleted. First visit: compile
        the program, ``built=True``. Revisit: reuse it (zero compiles), ``built=False``."""
        from jax.sharding import NamedSharding
        from .engine_vllm import SWITCH_SEGMENTS, hbm_snapshot
        from .resident import _COMPILES, _install_compile_listener, reshard_leaves
        if self._params is None:
            raise RuntimeError(f"engine {self.spec.engine_id} is not built or was exported")
        source = self.spec
        if target != dataclasses.replace(source, tp_size=target.tp_size, dp_size=target.dp_size):
            raise ValueError(f"switch_layout target {target} differs from {source} by more than tp_size/dp_size")
        _install_compile_listener()
        started, compiles0 = time.perf_counter(), len(_COMPILES)
        seg: Dict[str, float] = dict.fromkeys(SWITCH_SEGMENTS, 0.0)
        hbm = {"before": hbm_snapshot()}
        report: Dict[str, Any] = {"layout": tuple(target.layout), "noop": target.layout == source.layout,
                                  "reshard": {}}
        built = False
        if not report["noop"]:
            leaves, seed = self.leaf_specs(), int(source.model.get("seed", 0))
            known = self._layouts.get(target.layout)
            t = time.perf_counter()
            mesh = known[0] if known else self._mesh_for(target)
            targets = {l.path: NamedSharding(mesh, _partition_spec(l)) for l in leaves}
            seg["config"] = time.perf_counter() - t
            t = time.perf_counter()
            moved, report["reshard"] = reshard_leaves(self._params, leaves, targets,
                                                      source.tp_size, target.tp_size)
            seg["reshard"] = time.perf_counter() - t
            t = time.perf_counter()
            self._adopt(moved, leaves, mesh)  # layout, order and sharding checks
            old_coefs = self._coefs
            coefs = self._make_coefs(leaves, seed, target.tp_size, mesh)
            import jax
            jax.block_until_ready(coefs)
            for array in old_coefs.values():
                array.delete()
            seg["rebind"] = time.perf_counter() - t
            if known is None:
                t = time.perf_counter()
                program = self._compile(leaves, mesh, moved, coefs)
                seg["runner_build"] = time.perf_counter() - t
                self._layouts[target.layout] = (mesh, program)
                built = True
                t = time.perf_counter()  # prime the way back (see VllmEngine._prime_reverse)
                back = {l.path: NamedSharding(self._meshes[source.layout], _partition_spec(l)) for l in leaves}
                reshard_leaves(moved, leaves, back, target.tp_size, source.tp_size, discard=True)
                seg["prime"] = time.perf_counter() - t
            else:
                program = known[1]
            self._params, self._coefs, self._program, self._mesh = moved, coefs, program, mesh
            self.spec = target
            t = time.perf_counter()
            self._read()
            seg["attach"] = time.perf_counter() - t
        hbm["after"] = hbm_snapshot()
        seg["total"] = time.perf_counter() - started
        report.update(built=built, segments=seg, hbm=hbm, compiles=len(_COMPILES) - compiles0,
                      compile_s=float(sum(_COMPILES[compiles0:])))
        return report

    def _adopt(self, resident: Dict[str, Any], leaves: Tuple[LeafSpec, ...], mesh) -> Dict[str, Any]:
        """Validate given arrays against this engine's mesh/layout and take them."""
        from jax.sharding import NamedSharding
        expected_ids = [int(d.id) for d in mesh.devices.flat]
        missing = [l.path for l in leaves if l.path not in resident]
        extra = sorted(set(resident) - {l.path for l in leaves})
        if missing or extra:
            raise ValueError(f"engine {self.spec.engine_id}: resident leaves mismatch, "
                             f"missing {missing[:3]} ({len(missing)}), extra {extra[:3]} ({len(extra)})")
        for leaf in leaves:
            array = resident[leaf.path]
            problems = []
            if tuple(array.shape) != leaf.shape:
                problems.append(f"shape {tuple(array.shape)} != {leaf.shape}")
            if str(array.dtype) != leaf.dtype:
                problems.append(f"dtype {array.dtype} != {leaf.dtype}")
            sharding = array.sharding
            if not isinstance(sharding, NamedSharding):
                problems.append(f"sharding {type(sharding).__name__} is not NamedSharding")
            else:
                ids = [int(d.id) for d in sharding.mesh.devices.flat]
                if ids != expected_ids:
                    problems.append(f"mesh device order {ids} != {expected_ids}")
                if sharding.spec != _partition_spec(leaf) and not sharding.is_equivalent_to(
                        NamedSharding(mesh, _partition_spec(leaf)), len(leaf.shape)):
                    problems.append(f"partition spec {sharding.spec} != {_partition_spec(leaf)}")
            if problems:
                raise ValueError(f"engine {self.spec.engine_id}: resident leaf {leaf.path}: "
                                 + "; ".join(problems))
        return {l.path: resident[l.path] for l in leaves}

    # -- serving ----------------------------------------------------------

    def _read(self) -> np.ndarray:
        if self._program is None:
            raise RuntimeError(f"engine {self.spec.engine_id} is not built or was exported")
        out = self._program(self._params, self._coefs)
        return np.asarray(out.addressable_data(0))

    def step(self, request: Dict[str, Any]) -> Dict[str, Any]:
        prompts, steps = request["prompt"], int(request["steps"])
        vocab = int(self.spec.model.get("vocab", 32000))
        t0 = time.perf_counter()
        sums = [int(s) for s in self._read()]
        t1 = time.perf_counter()
        tokens = chain(sums, prompts, steps, vocab)
        return {"tokens": tokens, "forward_s": t1 - t0, "decode_s": time.perf_counter() - t1}

    def idle(self, seconds: float) -> float:
        """Occupy the engine for ``seconds`` (long-call tests)."""
        time.sleep(seconds)
        return seconds

    # -- teardown ---------------------------------------------------------

    def export(self) -> Dict[str, Any]:
        if self._params is None:
            raise RuntimeError(f"engine {self.spec.engine_id}: nothing to export")
        if self.spec.dp_size > 1:
            raise ValueError(f"engine {self.spec.engine_id} runs data-parallel {self.spec.layout}: "
                             "ResidentStore holds TP-only leaves; switch_layout to dp_size 1 first")
        params = self._params
        self._params = self._coefs = self._program = self._mesh = None
        self._layouts.clear()
        self._meshes.clear()
        self._closed = True
        return params

    def shutdown(self) -> None:
        self._params = self._coefs = self._program = self._mesh = None
        self._layouts.clear()
        self._meshes.clear()
        self._closed = True
