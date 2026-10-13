"""Executable bank: serialized per-rank XLA executables that let a target engine
skip trace, lower and backend compile for every precompiled bucket.

Write mode (provisioning deploy): runner jit callables are replaced by
``BankedFunction`` objects that capture an AOT ``Compiled`` per bucket (warm-up
and ``CompilationManager._run_compilation`` both go through them); after
warm-up each rank serializes its own executables into a pod-local directory
``<exec_bank_dir>/<identity digest>/rank<process index>/`` (atomic: temporary
directory + rename; manifest with identity, entries and per-payload sha256).

Read mode: after ``initialize_from_config`` each rank loads its bank, all ranks
of the instance vote through Ray internal_kv, and only a unanimous bank is
installed. A bucket miss or a ``Compiled`` signature rejection (raised before
execution) takes the original jit path and is counted; never a partial mix of
banked and unbanked ranks.

Installed tpu_inference call sites (md5-identical to the fork at 8d31f0866):
  run_model      models/common/model_loader.py:375-391 (closure cell
                 ``jitted_model_fn`` of ``wrapped_model_fn`` :467-471)
  compute_logits model_loader.py:412, runner attribute tpu_runner.py:1203
  sample, compute_and_gather_logprobs  layers/jax/sample/sampling.py:105,154;
                 module globals of tpu_runner.py:62-65, compilation_manager.py:33-34
  compute_and_gather_prompt_logprobs   sampling.py:165 (global used by :197)
  select_from_array    tpu_runner.py:2262-2264 (staticmethod -> instance attr)
  substitute_placeholder_token  tpu_runner.py:535, attribute :844
  structured_decode    structured_decoding_manager.py:33 (bound, static self)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

MANIFEST = "manifest.json"
FORMAT = 1
DEFAULT_DIR = "/root/.cache/omp_exec_bank"
VOTE_NAMESPACE = "omp_exec_bank"
VOTE_TIMEOUT_S = 120.0
# Static arguments that are per-process singletons: canonicalized by type.
SINGLETON_STATIC_TYPES = frozenset({
    "tpu_inference.runner.structured_decoding_manager.StructuredDecodingManager",
})
RUNNER_MODULES = (
    "tpu_inference.runner.tpu_runner",
    "tpu_inference.runner.compilation_manager",
    "tpu_inference.runner.structured_decoding_manager",
    "tpu_inference.models.common.model_loader",
    "tpu_inference.layers.jax.sample.sampling",
    "tpu_inference.models.jax.gemma4",
)
JAX_CONFIG_KEYS = (
    "jax_default_matmul_precision",
    "jax_enable_x64",
    "jax_use_shardy_partitioner",
    "jax_threefry_partitionable",
    "jax_default_prng_impl",
    "jax_optimization_level",
    "jax_memory_fitting_level",
    "jax_exec_time_optimization_effort",
    "jax_memory_fitting_effort",
)


class ExecBankError(RuntimeError):
    """The bank cannot be used; the engine takes the normal compile path."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


# ----------------------------------------------------------------- identity

def _version(dist: str) -> Optional[str]:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def runner_file_digests(modules: Iterable[str] = RUNNER_MODULES) -> Dict[str, Optional[str]]:
    """sha256 of the installed source of every module whose callables the bank replaces."""
    out = {}
    for name in modules:
        module = sys.modules.get(name)
        path = getattr(module, "__file__", None)
        out[name] = _sha256(Path(path).read_bytes()) if path else None
    return out


def runtime_identity(env=os.environ) -> Dict[str, Any]:
    import jax
    import jaxlib

    device = jax.devices()[0]
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "libtpu": _version("libtpu"),
        "vllm": _version("vllm"),
        "tpu_inference": _version("tpu_inference") or _version("tpu-inference"),
        "platform": device.client.platform,
        "platform_version": device.client.platform_version,
        "device_kind": device.device_kind,
        "XLA_FLAGS": env.get("XLA_FLAGS", ""),
        "LIBTPU_INIT_ARGS": env.get("LIBTPU_INIT_ARGS", ""),
        # Absent options (renamed across jax versions) record None, deterministically.
        "jax_config": {key: str(jax.config.values.get(key)) for key in JAX_CONFIG_KEYS},
    }


def engine_identity(runner, payload: Dict[str, Any], *, backend_revision: str,
                    runtime_revision: str, runtime: Dict[str, Any],
                    files: Dict[str, Optional[str]], num_blocks: int) -> Dict[str, Dict[str, Any]]:
    """{"instance": shared by every rank (digest names the bank), "rank": this process}."""
    import jax

    config = runner.vllm_config
    model, cache, scheduler = config.model_config, config.cache_config, config.scheduler_config
    try:
        hf_config = model.hf_config.to_dict()
    except AttributeError:
        hf_config = repr(model.hf_config)
    env = payload["runtime_env"]
    mesh = runner.mesh
    kv = [[list(a.shape), str(a.dtype), str(getattr(a.sharding, "spec", a.sharding))]
          for a in runner.kv_caches]
    instance = {
        "format": FORMAT,
        "model": {
            "model": model.model,
            "revision": getattr(model, "revision", None),
            "dtype": str(model.dtype).removeprefix("torch."),
            "quantization": model.quantization,
            "max_model_len": model.max_model_len,
            "max_logprobs": model.max_logprobs,
            "enforce_eager": model.enforce_eager,
            "hf_config_sha256": digest(hf_config),
        },
        "cache": {
            "kv_cache_dtype": cache.cache_dtype,
            "block_size": cache.block_size,
            "num_blocks": int(num_blocks),
            "enable_prefix_caching": cache.enable_prefix_caching,
            "kv_caches_sha256": digest(kv),
            "kv_cache_arrays": len(kv),
        },
        "scheduler": {
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "async_scheduling": bool(getattr(scheduler, "async_scheduling", False)),
            "speculative": config.speculative_config is not None,
        },
        "additional_config": {k: config.additional_config.get(k)
                              for k in ("activation_dtype", "matmul_precision")},
        "parallel": {
            "tp_size": payload["tp_size"],
            "mesh_axis_names": list(mesh.axis_names),
            "mesh_shape": [int(v) for v in mesh.devices.shape],
            "mesh_device_ids": [int(d.id) for d in mesh.devices.flat],
            "process_count": int(jax.process_count()),
            "host_order": [a.rsplit(":", 1)[0] for a in env["TPU_PROCESS_ADDRESSES"].split(",")],
            "TPU_TOPOLOGY": env.get("TPU_TOPOLOGY"),
            "TPU_HOST_BOUNDS": env.get("TPU_HOST_BOUNDS"),
            "TPU_CHIPS_PER_HOST_BOUNDS": env.get("TPU_CHIPS_PER_HOST_BOUNDS"),
        },
        "runtime": runtime,
        "code": {"backend_revision": backend_revision, "runtime_revision": runtime_revision,
                 "runner_files": files},
    }
    rank = {
        "process_index": int(jax.process_index()),
        "local_device_ids": sorted(int(d.id) for d in jax.local_devices()),
    }
    return {"instance": instance, "rank": rank}


# ----------------------------------------------------------- bucket keys

_UNSEEN = object()


def canonical_static(value) -> Optional[str]:
    """Process-independent text for a static argument; None if not bankable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, (tuple, list)):
        parts = [canonical_static(v) for v in value]
        if any(p is None for p in parts):
            return None
        return ("(" if isinstance(value, tuple) else "[") + ",".join(parts) + ")"
    from jax.sharding import Mesh
    if isinstance(value, Mesh):
        return (f"Mesh({list(value.axis_names)},{list(value.devices.shape)},"
                f"{[int(d.id) for d in value.devices.flat]})")
    name = f"{type(value).__module__}.{type(value).__qualname__}"
    if name in SINGLETON_STATIC_TYPES:
        return f"<{name}>"
    return None


def _leaf_key(leaf):
    shape = getattr(leaf, "shape", None)
    if shape is None:
        return type(leaf)
    return (tuple(shape), leaf.dtype)


def _leaf_text(key) -> str:
    if isinstance(key, type):
        return key.__name__
    return f"{list(key[0])}:{key[1]}"


class BankedFunction:
    """Call-site replacement for one jitted function.

    ``original`` is the exact callable previously at the call site (fallback);
    ``jitted`` the underlying ``jax.jit`` object (static argnums, AOT lower);
    ``bound`` an object prepended as first argument (jitted methods).
    Arguments at ``invariant_argnums`` (weights, KV caches) are left out of the
    bucket key; ``Compiled`` still validates them on every call.

    Deliberately has no ``lower`` attribute: compile_overlap's profiler wraps
    anything with ``lower`` and must hand this object through unchanged.
    """

    def __init__(self, fn_id: str, original: Callable, jitted, *, bound=None,
                 invariant_argnums: Iterable[int] = ()):
        info = jitted._jit_info
        self.fn_id = fn_id
        self.original = original
        self.jitted = jitted
        self.bound = bound
        self.static_argnums = frozenset(info.static_argnums)
        self.static_argnames = frozenset(info.static_argnames)
        self.invariant_argnums = frozenset(invariant_argnums)
        self.capture = False
        self._raw: Dict[Any, tuple] = {}
        self._table: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.captured = 0
        self.misses: Counter = Counter()
        self.rejections: Counter = Counter()
        self.rejection_samples: Dict[str, str] = {}

    # -- keys
    def _split(self, full, kwargs):
        statics, dynamic = [], []
        for index, arg in enumerate(full):
            if index in self.static_argnums:
                statics.append(arg)
            elif index not in self.invariant_argnums:
                dynamic.append(arg)
        static_kw, dynamic_kw = [], {}
        for name, arg in kwargs.items():
            if name in self.static_argnames:
                static_kw.append((name, arg))
            else:
                dynamic_kw[name] = arg
        return statics, sorted(static_kw), dynamic, dynamic_kw

    def _keys(self, full, kwargs):
        import jax

        statics, static_kw, dynamic, dynamic_kw = self._split(full, kwargs)
        leaves, treedef = jax.tree_util.tree_flatten((dynamic, dynamic_kw))
        leaf_keys = tuple(_leaf_key(leaf) for leaf in leaves)
        raw = (tuple(statics), tuple(static_kw), treedef, leaf_keys)
        return raw, statics, static_kw, treedef, leaf_keys

    def canonical(self, full, kwargs) -> Optional[str]:
        _, statics, static_kw, treedef, leaf_keys = self._keys(full, kwargs)
        return self._canonical(statics, static_kw, treedef, leaf_keys)

    def _canonical(self, statics, static_kw, treedef, leaf_keys) -> Optional[str]:
        parts = [canonical_static(v) for v in statics]
        named = [(k, canonical_static(v)) for k, v in static_kw]
        if any(p is None for p in parts) or any(p is None for _, p in named):
            return None
        parts += [f"{k}={p}" for k, p in named]
        return (f"{self.fn_id}|static[{';'.join(parts)}]|{treedef}|"
                f"[{','.join(_leaf_text(k) for k in leaf_keys)}]")

    def _entry(self, full, kwargs):
        """(canonical key or None, Compiled or None) for this call, memoized per raw key."""
        raw, statics, static_kw, treedef, leaf_keys = self._keys(full, kwargs)
        try:
            found = self._raw.get(raw, _UNSEEN)
        except TypeError:  # unhashable static argument
            return None, None
        if found is _UNSEEN or (found[1] is None and self.capture):
            key = self._canonical(statics, static_kw, treedef, leaf_keys)
            compiled = self._table.get(key) if key is not None else None
            found = (key, compiled)
            if compiled is not None or not self.capture:
                self._raw[raw] = found
        return found

    def _dynamic_args(self, full, kwargs):
        args = tuple(a for i, a in enumerate(full) if i not in self.static_argnums)
        kw = {k: v for k, v in kwargs.items() if k not in self.static_argnames}
        return args, kw

    # -- table
    def lookup(self, full, kwargs):
        return self._entry(full, kwargs)[1]

    def store(self, full, kwargs, compiled) -> Optional[str]:
        key = self.canonical(full, kwargs)
        if key is not None:
            with self._lock:
                self._table[key] = compiled
                self._raw.clear()
                self.captured += 1
        return key

    def load(self, entries: Dict[str, Any]) -> None:
        with self._lock:
            self._table.update(entries)
            self._raw.clear()

    def entries(self) -> Dict[str, Any]:
        return dict(self._table)

    def compile_for(self, full, kwargs):
        compiled = self.jitted.lower(*full, **kwargs).compile()
        self.store(full, kwargs, compiled)
        return compiled

    # -- dispatch
    def __call__(self, *args, **kwargs):
        full = args if self.bound is None else (self.bound,) + args
        key, compiled = self._entry(full, kwargs)
        if compiled is None:
            if self.capture and key is not None:
                compiled = self.compile_for(full, kwargs)
            else:
                self.misses[key or "<unbankable>"] += 1
                return self.original(*args, **kwargs)
        dyn_args, dyn_kwargs = self._dynamic_args(full, kwargs)
        try:
            out = compiled(*dyn_args, **dyn_kwargs)
        except (TypeError, ValueError) as exc:
            # Compiled validates avals/shardings/pytree before dispatching.
            self.rejections[key] += 1
            self.rejection_samples.setdefault(key, f"{type(exc).__name__}: {str(exc)[:300]}")
            return self.original(*args, **kwargs)
        self.hits += 1
        return out

    def report(self) -> Dict[str, Any]:
        return {"entries": len(self._table), "hits": self.hits, "captured": self.captured,
                "misses": sum(self.misses.values()), "miss_buckets": len(self.misses),
                "miss_samples": sorted(self.misses)[:5],
                "rejections": sum(self.rejections.values()),
                "rejection_samples": dict(list(self.rejection_samples.items())[:3])}


# ------------------------------------------------------- serialization

def serialize_entry(compiled) -> bytes:
    """Pickled (payload, in_tree, out_tree, device ids); ExecBankError when not serializable.

    The device ids are the executable's device assignment in order (the runner mesh order,
    e.g. [0, 1, 4, 5, ...] on a TP16 sub-slice).  ``deserialize_and_load`` otherwise loads the
    XLA executable onto ``backend.devices()`` in id order (jax serialize_executable.py:63-64,
    116-119) while the unpickled shardings keep the mesh order, and the first call aborts in
    PjRt ("PjRtBuffer's memory space is addressed by device TPU_5 vs sharding is on device
    TPU_4"; O2 attempt 1).
    """
    from jax.experimental import serialize_executable

    executable = compiled._executable
    if compiled._params.const_args:
        raise ExecBankError("closed-over constants (const_args)")
    unsafe_call = getattr(executable, "unsafe_call", None)
    if getattr(unsafe_call, "has_host_callbacks", False):
        raise ExecBankError("host callbacks")
    try:
        device_ids = [int(d.id) for d in executable._unloaded_executable.device_list]
        payload, in_tree, out_tree = serialize_executable.serialize(compiled)
        return pickle.dumps((payload, in_tree, out_tree, device_ids), protocol=pickle.HIGHEST_PROTOCOL)
    except (ValueError, NotImplementedError, TypeError, AttributeError,
            pickle.PicklingError) as exc:
        raise ExecBankError(f"not serializable: {type(exc).__name__}: {exc}") from exc


def deserialize_entry(data: bytes):
    import jax
    from jax.experimental import serialize_executable

    payload, in_tree, out_tree, device_ids = pickle.loads(data)
    by_id = {d.id: d for d in jax.devices()}
    missing = [i for i in device_ids if i not in by_id]
    if missing:
        raise ExecBankError(f"device ids {missing} of the banked assignment are not in this client")
    compiled = serialize_executable.deserialize_and_load(
        payload, in_tree, out_tree, execution_devices=[by_id[i] for i in device_ids])
    # The XLA executable's own addressable devices must follow the banked assignment.
    local = jax.process_index()
    expected = [i for i in device_ids if by_id[i].process_index == local]
    loaded = [int(d.id) for d in compiled._executable.xla_executable.local_devices()]
    if loaded != expected:
        raise ExecBankError(f"loaded local devices {loaded} != banked assignment {expected}")
    return compiled


def entry_set_digest(keys: Iterable[str]) -> str:
    return digest(sorted(keys))


def bank_path(root: str, identity: Dict[str, Any]) -> Path:
    return Path(root) / digest(identity["instance"]) / f"rank{identity['rank']['process_index']}"


def write_bank(root: str, identity: Dict[str, Any], functions: Iterable[BankedFunction],
               *, nonce: Optional[str] = None) -> Dict[str, Any]:
    """Serialize every captured entry; atomic replace of this rank's directory."""
    started = time.time()
    target = bank_path(root, identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".tmp-{target.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    entries, skipped, total = {}, [], 0
    try:
        for banked in functions:
            for key, compiled in sorted(banked.entries().items()):
                try:
                    data = serialize_entry(compiled)
                except ExecBankError as exc:
                    skipped.append({"key": key, "reason": str(exc)})
                    continue
                name = f"{banked.fn_id}-{_sha256(key.encode())[:20]}.bin"
                (staging / name).write_bytes(data)
                entries[name] = {"fn": banked.fn_id, "key": key, "sha256": _sha256(data),
                                 "bytes": len(data)}
                total += len(data)
        manifest = {"format": FORMAT, "identity": identity,
                    "identity_digest": digest(identity["instance"]), "nonce": nonce,
                    "entries": entries, "skipped": skipped,
                    "entry_set_digest": entry_set_digest(e["key"] for e in entries.values()),
                    "written_at": time.time()}
        (staging / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=1))
        old = None
        if target.exists():
            old = target.parent / f".old-{target.name}-{uuid.uuid4().hex}"
            os.rename(target, old)
        os.rename(staging, target)
        if old is not None:
            shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"path": str(target), "entries": len(entries), "bytes": total,
            "skipped": skipped, "entry_set_digest": manifest["entry_set_digest"],
            "write_s": time.time() - started}


def _identity_difference(stored, expected, prefix="") -> list:
    if isinstance(stored, dict) and isinstance(expected, dict):
        out = []
        for key in sorted(set(stored) | set(expected)):
            out += _identity_difference(stored.get(key), expected.get(key), f"{prefix}{key}.")
        return out
    return [] if stored == expected else [prefix.rstrip(".")]


def _nearest_bank(root: str, identity: Dict[str, Any]) -> str:
    """Why no bank matched: identity fields that differ from this rank's newest bank."""
    rank = f"rank{identity['rank']['process_index']}"
    newest = None
    for manifest_file in Path(root).glob(f"*/{rank}/{MANIFEST}"):
        try:
            manifest = json.loads(manifest_file.read_text())
        except (OSError, ValueError):
            continue
        if newest is None or manifest.get("written_at", 0) > newest.get("written_at", 0):
            newest = manifest
    if newest is None:
        return " (no bank for this rank under the root)"
    differ = _identity_difference(newest["identity"], identity)
    return f" (newest bank {newest['identity_digest'][:12]}: identity mismatch: {differ[:12]})"


def load_bank(root: str, identity: Dict[str, Any]) -> Dict[str, Any]:
    """Verify identity and checksums, deserialize every entry. Raises ExecBankError."""
    started = time.time()
    path = bank_path(root, identity)
    manifest_file = path / MANIFEST
    if not manifest_file.exists():
        raise ExecBankError(f"no bank at {path}{_nearest_bank(root, identity)}")
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("format") != FORMAT:
        raise ExecBankError(f"bank format {manifest.get('format')} != {FORMAT}")
    differ = _identity_difference(manifest["identity"], identity)
    if differ:
        raise ExecBankError(f"identity mismatch: {differ[:12]}")
    tables: Dict[str, Dict[str, Any]] = {}
    for name, entry in sorted(manifest["entries"].items()):
        data = (path / name).read_bytes()
        if _sha256(data) != entry["sha256"]:
            raise ExecBankError(f"checksum mismatch for {name}")
        try:
            compiled = deserialize_entry(data)
        except Exception as exc:  # noqa: BLE001 - any load failure refuses the bank
            raise ExecBankError(f"deserialize failed for {name}: {type(exc).__name__}: {exc}") from exc
        tables.setdefault(entry["fn"], {})[entry["key"]] = compiled
    keys = [e["key"] for e in manifest["entries"].values()]
    return {"path": str(path), "tables": tables, "entries": len(keys),
            "entry_set_digest": entry_set_digest(keys), "load_s": time.time() - started,
            "written_nonce": manifest.get("nonce")}


# ------------------------------------------------------------- unanimity

class RayKV:
    """Ray GCS internal_kv (reachable from every worker of the cluster)."""

    def put(self, key: str, value: bytes) -> None:
        from ray.experimental import internal_kv
        internal_kv._internal_kv_put(key, value, overwrite=True, namespace=VOTE_NAMESPACE)

    def get(self, key: str) -> Optional[bytes]:
        from ray.experimental import internal_kv
        return internal_kv._internal_kv_get(key, namespace=VOTE_NAMESPACE)


def unanimity(kv, *, instance_id: str, nonce: str, process_rank: int, process_count: int,
              vote: Dict[str, Any], timeout_s: float = VOTE_TIMEOUT_S,
              poll_s: float = 0.05, clock=time.monotonic, sleep=time.sleep) -> Dict[str, Any]:
    """Rank 0 collects every rank's vote (bounded) and publishes one verdict.

    Every rank applies rank 0's verdict, so all ranks take the same path. A rank
    that never sees the verdict raises: proceeding alone could mix paths.
    """
    prefix = f"{instance_id}/{nonce}"
    started = clock()
    kv.put(f"{prefix}/vote/{process_rank}", json.dumps(vote, sort_keys=True).encode())
    if process_rank == 0:
        votes: Dict[int, Dict[str, Any]] = {}
        deadline = started + timeout_s
        while True:
            for rank in range(process_count):
                if rank not in votes:
                    raw = kv.get(f"{prefix}/vote/{rank}")
                    if raw is not None:
                        votes[rank] = json.loads(raw)
            if len(votes) == process_count or clock() >= deadline:
                break
            sleep(poll_s)
        missing = sorted(set(range(process_count)) - set(votes))
        problems = {r: v["error"] for r, v in votes.items() if v.get("error")}
        signatures = {(v.get("identity_digest"), v.get("entry_set_digest"))
                      for v in votes.values() if not v.get("error")}
        if missing:
            verdict = {"use": False, "reason": f"no vote from ranks {missing} within {timeout_s}s"}
        elif problems:
            verdict = {"use": False, "reason": f"ranks refused: {problems}"}
        elif len(signatures) != 1:
            verdict = {"use": False, "reason": f"ranks disagree on identity/entry set: {sorted(signatures)}"}
        else:
            verdict = {"use": True, "reason": "unanimous"}
        verdict["votes"] = {str(r): v for r, v in sorted(votes.items())}
        kv.put(f"{prefix}/verdict", json.dumps(verdict, sort_keys=True).encode())
    else:
        deadline = started + 2 * timeout_s + 30.0
        while True:
            raw = kv.get(f"{prefix}/verdict")
            if raw is not None:
                verdict = json.loads(raw)
                break
            if clock() >= deadline:
                raise ExecBankError(
                    f"no exec-bank verdict from rank 0 for {prefix}; refusing to proceed alone")
            sleep(poll_s)
    verdict["wait_s"] = clock() - started
    return verdict


# ----------------------------------------------------------- installation

def _closure_cell(fn, name):
    code = getattr(fn, "__code__", None)
    if code is None or name not in code.co_freevars:
        raise ExecBankError(f"{getattr(fn, '__qualname__', fn)!r} has no closure variable {name!r}")
    return fn.__closure__[code.co_freevars.index(name)]


class Installation:
    """Replaces runner jit callables with BankedFunctions; ``restore`` undoes it."""

    def __init__(self):
        self.functions: Dict[str, BankedFunction] = {}
        self._undo: list = []

    def add(self, banked: BankedFunction, sites: Iterable[tuple]) -> None:
        """sites: (kind, owner, name) with kind in attr|global|cell."""
        for kind, owner, name in sites:
            if kind == "cell":
                cell = owner
                previous = cell.cell_contents
                self._undo.append(lambda c=cell, p=previous: setattr(c, "cell_contents", p))
                cell.cell_contents = banked
            elif kind == "attr":
                had = name in vars(owner)
                previous = vars(owner).get(name)
                self._undo.append(lambda o=owner, n=name, h=had, p=previous:
                                  setattr(o, n, p) if h else delattr(o, n))
                setattr(owner, name, banked)
            elif kind == "global":
                previous = getattr(owner, name)
                self._undo.append(lambda o=owner, n=name, p=previous: setattr(o, n, p))
                setattr(owner, name, banked)
            else:
                raise ValueError(kind)
        self.functions[banked.fn_id] = banked

    def restore(self) -> None:
        while self._undo:
            self._undo.pop()()
        self.functions.clear()


def install_runner(runner, modules: Dict[str, Any]) -> Installation:
    """Bank every precompiled tpu_inference jit at its serving and precompile call sites.

    ``modules``: {"tpu_runner", "compilation_manager", "sampling"} module objects.
    """
    tpu_runner, manager_module, sampling = (modules["tpu_runner"], modules["compilation_manager"],
                                            modules["sampling"])
    inst = Installation()
    try:
        cell = _closure_cell(runner.model_fn, "jitted_model_fn")
        run_model = cell.cell_contents
        inst.add(BankedFunction("run_model", run_model, run_model, invariant_argnums=(0, 1)),
                 [("cell", cell, None)])
        logits = runner.compute_logits_fn
        inst.add(BankedFunction("compute_logits", logits, logits, invariant_argnums=(0,)),
                 [("attr", runner, "compute_logits_fn")])
        for fn_id, owners in (
            ("sample", (tpu_runner, manager_module)),
            ("compute_and_gather_logprobs", (tpu_runner, manager_module)),
            ("compute_and_gather_prompt_logprobs", (manager_module, sampling)),
        ):
            fn = getattr(owners[0], fn_id)
            if any(getattr(o, fn_id) is not fn for o in owners):
                raise ExecBankError(f"{fn_id} differs between modules")
            inst.add(BankedFunction(fn_id, fn, fn), [("global", o, fn_id) for o in owners])
        select = runner._select_from_array_fn
        inst.add(BankedFunction("select_from_array", select, select),
                 [("attr", runner, "_select_from_array_fn")])
        substitute = runner._substitute_placeholder_token_fn
        inst.add(BankedFunction("substitute_placeholder_token", substitute, substitute),
                 [("attr", runner, "_substitute_placeholder_token_fn")])
        structured = getattr(runner, "structured_decoding_manager", None)
        if structured is not None:
            bound_method = structured.structured_decode_fn
            jitted = type(structured).__dict__["structured_decode_fn"]
            inst.add(BankedFunction("structured_decode", bound_method, jitted, bound=structured),
                     [("attr", structured, "structured_decode_fn")])
        for banked in inst.functions.values():
            if not hasattr(banked.jitted, "lower"):
                raise ExecBankError(f"{banked.fn_id} is not a jax.jit function")
    except BaseException:
        inst.restore()
        raise
    return inst


class _CapturingLowered:
    def __init__(self, lowered, banked: BankedFunction, args, kwargs):
        self._lowered, self._banked, self._args, self._kwargs = lowered, banked, args, kwargs

    def compile(self, *a, **k):
        compiled = self._lowered.compile(*a, **k)
        self._banked.store(self._args, self._kwargs, compiled)
        return compiled


class _CaptureProxy:
    """Looks like a jit to CompilationManager._run_compilation; stores each AOT Compiled."""

    def __init__(self, banked: BankedFunction):
        self._banked = banked

    def lower(self, *args, **kwargs):
        lowered = self._banked.jitted.lower(*args, **kwargs)
        return _CapturingLowered(lowered, self._banked, args, kwargs)

    def __call__(self, *args, **kwargs):
        return self._banked(*args, **kwargs)


def patch_run_compilation(manager, counters: Counter):
    """Wrap ``CompilationManager._run_compilation`` (compilation_manager.py:126-196).

    Banked hit: skip lower and compile; the warm-up task (unless compile_only)
    still executes through the BankedFunction. Capture (write mode): the
    original runs on a proxy whose AOT ``Compiled`` is stored. Otherwise the
    original runs on the original callable. Returns an undo callable.
    """
    import functools

    original = manager._run_compilation
    had = "_run_compilation" in vars(manager)

    def run(name, fn, *args, call_kwargs=None, warmup_handler=None, aot=True,
            compile_only=False, **kwargs):
        call_kwargs = dict(call_kwargs or {})
        while isinstance(fn, functools.partial):  # as compilation_manager.py:139-142
            args = fn.args + args
            call_kwargs = {**fn.keywords, **call_kwargs}
            fn = fn.func
        if not isinstance(fn, BankedFunction):
            return original(name, fn, *args, call_kwargs=call_kwargs,
                            warmup_handler=warmup_handler, aot=aot,
                            compile_only=compile_only, **kwargs)
        if fn.lookup(args, call_kwargs) is not None:
            counters["aot_skipped"] += 1
            if not compile_only:
                manager._warmup_tasks.append((name, fn, args, call_kwargs, warmup_handler))
            return None
        if fn.capture:
            counters["aot_captured"] += 1
            target = _CaptureProxy(fn)
        else:
            counters["aot_unbanked"] += 1
            target = fn.original
        return original(name, target, *args, call_kwargs=call_kwargs,
                        warmup_handler=warmup_handler, aot=aot, compile_only=compile_only,
                        **kwargs)

    manager._run_compilation = run

    def undo():
        if had:
            manager._run_compilation = original
        else:
            del manager._run_compilation

    return undo


# ---------------------------------------------------------------- worker

class WorkerBank:
    """Per-worker bank lifecycle and evidence (hetero/tpu_worker.py drives it)."""

    def __init__(self, mode: str, root: str, *, instance_id: str, nonce: str,
                 process_rank: int, process_count: int, sabotage_rank: Optional[int] = None,
                 kv=None, vote_timeout_s: float = VOTE_TIMEOUT_S):
        if mode not in ("write", "read"):
            raise ValueError(f"unsupported exec_bank mode {mode!r}")
        self.mode, self.root = mode, root
        self.instance_id, self.nonce = instance_id, nonce
        self.process_rank, self.process_count = process_rank, process_count
        self.sabotage_rank = sabotage_rank
        self.kv = kv if kv is not None else RayKV()
        self.vote_timeout_s = vote_timeout_s
        self.installation: Optional[Installation] = None
        self.counters: Counter = Counter()
        self._undo_run: Optional[Callable] = None
        self.status: Dict[str, Any] = {"mode": mode, "dir": root, "installed": False,
                                       "process_rank": process_rank}

    def _activate(self, runner, modules) -> Installation:
        installation = install_runner(runner, modules)
        try:
            self._undo_run = patch_run_compilation(runner.compilation_manager, self.counters)
        except BaseException:
            installation.restore()
            raise
        self.installation = installation
        self.status["installed"] = True
        self.status["functions_installed"] = sorted(installation.functions)
        return installation

    def deactivate(self) -> None:
        if self._undo_run is not None:
            self._undo_run()
            self._undo_run = None
        if self.installation is not None:
            self.installation.restore()
            self.installation = None
        self.status["installed"] = False

    # -- write mode
    def start_capture(self, runner, modules) -> None:
        """Before the first precompile: every banked bucket compiles AOT and is kept.

        An uninstallable runner is recorded and the deploy continues unbanked."""
        try:
            installation = self._activate(runner, modules)
        except ExecBankError as exc:
            self.status["write_error"] = f"install: {exc}"
            return
        for banked in installation.functions.values():
            banked.capture = True

    def finish_capture(self, identity_fn: Callable[[], Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self.installation is None:
            return None
        functions = list(self.installation.functions.values())
        for banked in functions:
            banked.capture = False
        identity = identity_fn()
        self.status["identity_digest"] = digest(identity["instance"])
        self.status["identity"] = identity
        report = write_bank(self.root, identity, functions, nonce=self.nonce)
        self.status["written"] = report
        return report

    # -- read mode
    def load(self, runner, modules, identity_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Load, vote, and install only on a unanimous verdict (else restore everything).

        Every local failure becomes a refusing vote, so all ranks take one path."""
        vote: Dict[str, Any] = {}
        try:
            identity = identity_fn()
            self.status["identity_digest"] = vote["identity_digest"] = digest(identity["instance"])
            self.status["identity"] = identity
            if self.sabotage_rank is not None and self.sabotage_rank == self.process_rank:
                raise ExecBankError(f"sabotaged (exec_bank_sabotage_rank={self.sabotage_rank})")
            loaded = load_bank(self.root, identity)
            installation = self._activate(runner, modules)
            unknown = set(loaded["tables"]) - set(installation.functions)
            if unknown:
                raise ExecBankError(f"bank has functions this runner cannot install: {sorted(unknown)}")
            for fn_id, table in loaded["tables"].items():
                installation.functions[fn_id].load(table)
            vote["entry_set_digest"] = loaded["entry_set_digest"]
            self.status["loaded"] = {k: loaded[k] for k in
                                     ("path", "entries", "entry_set_digest", "load_s", "written_nonce")}
        except Exception as exc:  # noqa: BLE001 - refuse in the vote rather than diverge
            self.deactivate()
            message = str(exc) if isinstance(exc, ExecBankError) else f"{type(exc).__name__}: {exc}"
            vote["error"] = message
            self.status["local_error"] = message
        verdict = unanimity(self.kv, instance_id=self.instance_id, nonce=self.nonce,
                            process_rank=self.process_rank, process_count=self.process_count,
                            vote=vote, timeout_s=self.vote_timeout_s)
        self.status["unanimity"] = verdict
        if not verdict["use"]:
            self.deactivate()
            self.status["fallback_reason"] = verdict["reason"]
        return self.status

    def report(self) -> Dict[str, Any]:
        functions = ({fn_id: b.report() for fn_id, b in self.installation.functions.items()}
                     if self.installation is not None else {})
        return {**self.status, "run_compilation": dict(self.counters), "functions": functions}
