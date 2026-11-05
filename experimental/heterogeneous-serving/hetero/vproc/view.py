"""C2: the virtual JAX view (results/vproc/plan.md, C2).

One host kernel = one OS process in one physical JAX runtime. While a view is
active, the engine-facing JAX API shows only the view's hosts; ``jax._src``
stays physical (read-only C++ ``Device.process_index`` is compared with
``xla_bridge.process_index()`` inside JAX, so those two must keep agreeing).

Verified against jax 0.10.2 and 0.11.2 (identical in both):
- ``jax.experimental.multihost_utils`` (``process_allgather``,
  ``broadcast_one_to_all``, ``sync_global_devices``/``assert_equal``) reads
  only the public names ``jax.devices/process_count/process_index/
  local_device_count/local_devices`` (multihost_utils.py:62-130, 177). Patching
  those names scopes the collectives to the view; the devices handed to the
  jitted collective are the view's, so only the view's processes launch it.
- Internal comparisons use ``xla_bridge`` directly and stay physical:
  mesh.py:357 (``local_mesh``), dispatch.py:475/508, array.py:146/948,
  pjit.py:1049, sharding.py:169, sharding_impls.py:182.
- ``jax.devices`` & co. are bound by ``from jax._src.xla_bridge import ...`` in
  ``jax/__init__.py`` only; no public ``jax.*`` submodule re-exports them, but
  ``activate`` still rebinds any that does (``_PUBLIC_NAMES`` scan).
- The persistent-cache write guard is ``compiler._cache_write``
  (``distributed.global_state.process_id != 0``). Other ``process_id`` readers
  in ``compiler`` (share-binary path, GPU autotune) must stay physical, so only
  ``_cache_write`` is replaced, by a scoped wrapper (same idea as
  ``hetero/ici_resident.py:133-179``): view rank 0 runs a private copy of the
  original function that sees ``process_id == 0``; other ranks skip.
- ``wait_at_barrier(barrier_id, timeout_in_ms, process_ids=None)`` exists in
  jaxlib 0.10.2 and 0.11.2. ``kv_set``/``kv_get``/``barrier`` below prefix keys
  with ``view.namespace`` and scope barriers to ``view.hosts``.

Not covered (see ``VIRTUALIZED``): callers that pass ``devices=None`` to
``jax.make_mesh`` / ``mesh_utils.create_device_mesh`` get ``xb.devices()``
(physical); ``jax.pmap``; ``jax.experimental.array_serialization`` barriers use
``global_state.client`` unscoped; ``from jax import devices`` bound before
activation.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.abc
import logging
import sys
import threading
import types
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hetero.vproc.types import ViewSpec

logger = logging.getLogger(__name__)

# What is virtualized while a view is active (value = how). Peers rely on the keys.
VIRTUALIZED: Dict[str, str] = {
    "jax.devices": "view's devices in physical order; backend 'cpu' on a multi-process CPU client lists this host's devices first",
    "jax.device_count": "len(view devices)",
    "jax.local_devices": "process_index= argument mapped virtual rank -> physical host; no argument = unchanged",
    "jax.process_index": "rank within the view",
    "jax.process_count": "number of hosts in the view",
    "jax.process_indices": "range(view size)",
    "jax.experimental.multihost_utils": "not patched: reads the public jax.* names above, so its collectives span the view only",
    "tpu_inference.utils.xb": "proxy module: devices() virtual, everything else forwarded to jax._src.xla_bridge",
    "tpu_inference.distributed.utils.get_device_topology_order_id": "global_devices filtered to the view's hosts before ranking",
    "jax._src.compiler._cache_write": "scoped wrapper: view rank 0 writes, other ranks skip; global_state is not touched",
    "jax.distributed.coordination": "view.kv_set/kv_get/barrier: keys prefixed with view.namespace, barriers use process_ids=view.hosts",
}

_PUBLIC_NAMES = ("devices", "device_count", "local_devices", "process_index",
                 "process_count", "process_indices")


class _Activation:
    def __init__(self, view: ViewSpec, physical_index: int):
        self.view = view
        self.physical_index = physical_index
        self.rank = view.rank_of(physical_index)
        self.patches: List[Tuple[Any, str, Any, Any]] = []  # (owner, name, original, new)
        self.finder: Optional["_LateImportFinder"] = None


_LOCK = threading.RLock()
_ACTIVE: Optional[_Activation] = None
_SUSPENDED = 0  # depth of physical() contexts; process-global like the view


def current() -> Optional[ViewSpec]:
    """The view the engine API currently shows (None when inactive or inside ``physical()``)."""
    act = _ACTIVE
    return act.view if act is not None and _SUSPENDED == 0 else None


def _current_activation() -> Optional[_Activation]:
    act = _ACTIVE
    return act if act is not None and _SUSPENDED == 0 else None


@contextlib.contextmanager
def physical() -> Iterator[None]:
    """Suspend the active view (no-op when none): every patched name shows physical values. Reentrant."""
    global _SUSPENDED
    with _LOCK:
        _SUSPENDED += 1
    try:
        yield
    finally:
        with _LOCK:
            _SUSPENDED -= 1


@contextlib.contextmanager
def active(view: ViewSpec) -> Iterator[ViewSpec]:
    activate(view)
    try:
        yield view
    finally:
        deactivate()


def _set(act: _Activation, owner: Any, name: str, new: Any) -> None:
    original = getattr(owner, name)
    setattr(owner, name, new)
    act.patches.append((owner, name, original, new))


def _mark(wrapper, original):
    wrapper.__vproc_original__ = original
    return wrapper


# ---------------------------------------------------------------- jax.* names

def _build_jax_wrappers(orig: Dict[str, Any]) -> Dict[str, Any]:
    o_devices, o_count = orig["devices"], orig["device_count"]
    o_local, o_index = orig["local_devices"], orig["process_index"]
    o_pcount, o_indices = orig["process_count"], orig["process_indices"]

    def is_block(backend) -> bool:
        return o_pcount(backend) > 1

    @functools.wraps(o_devices)
    def devices(backend=None):
        devs = o_devices(backend)
        act = _current_activation()
        if act is None or len({d.process_index for d in devs}) <= 1:
            return devs
        hosts = act.view.hosts
        keep = [d for d in devs if d.process_index in hosts]
        if backend is not None and keep and keep[0].platform == "cpu":
            mine = [d for d in keep if d.process_index == act.physical_index]
            keep = mine + [d for d in keep if d.process_index != act.physical_index]
        return keep

    @functools.wraps(o_count)
    def device_count(backend=None):
        act = _current_activation()
        if act is None or not is_block(backend):
            return o_count(backend)
        return len(devices(backend))

    @functools.wraps(o_local)
    def local_devices(process_index=None, backend=None, host_id=None):
        act = _current_activation()
        virtual = process_index if process_index is not None else host_id
        if act is None or virtual is None or not is_block(backend):
            return o_local(process_index, backend, host_id)
        if not (0 <= virtual < act.view.size):
            raise ValueError(f"Unknown process_index {virtual}")
        return o_local(act.view.physical_of(virtual), backend)

    @functools.wraps(o_index)
    def process_index(backend=None):
        physical_index = o_index(backend)
        act = _current_activation()
        if act is None or not is_block(backend):
            return physical_index
        return act.view.rank_of(physical_index)

    @functools.wraps(o_pcount)
    def process_count(backend=None):
        act = _current_activation()
        count = o_pcount(backend)
        return act.view.size if act is not None and count > 1 else count

    @functools.wraps(o_indices)
    def process_indices(backend=None):
        act = _current_activation()
        if act is None or not is_block(backend):
            return o_indices(backend)
        return list(range(act.view.size))

    return {"devices": devices, "device_count": device_count,
            "local_devices": local_devices, "process_index": process_index,
            "process_count": process_count, "process_indices": process_indices}


def _patch_jax_names(act: _Activation) -> None:
    import jax
    orig = {name: getattr(jax, name) for name in _PUBLIC_NAMES}
    wrappers = _build_jax_wrappers(orig)
    for name, wrapper in wrappers.items():
        _mark(wrapper, orig[name])
    by_id = {id(orig[name]): wrappers[name] for name in _PUBLIC_NAMES}
    # The jax module itself plus any other public jax.* module that re-exports them.
    for modname, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType):
            continue
        if not (modname == "jax" or (modname.startswith("jax.") and not modname.startswith("jax._src"))):
            continue
        for attr, value in list(vars(module).items()):
            replacement = by_id.get(id(value))
            if replacement is not None and value is not replacement:
                _set(act, module, attr, replacement)


# ------------------------------------------------------- persistent-cache write

def _patch_cache_write(act: _Activation) -> None:
    from jax._src import compiler
    original = compiler._cache_write
    if "distributed" not in original.__code__.co_names:
        raise RuntimeError("jax._src.compiler._cache_write does not read distributed.global_state: "
                           "persistent-cache guard not qualified for this jax")
    globals_copy = dict(original.__globals__)
    globals_copy["distributed"] = types.SimpleNamespace(
        global_state=types.SimpleNamespace(process_id=0))
    permitted = types.FunctionType(original.__code__, globals_copy, original.__name__,
                                   original.__defaults__, original.__closure__)
    permitted.__kwdefaults__ = original.__kwdefaults__

    @functools.wraps(original)
    def cache_write(*args, **kwargs):
        current_act = _current_activation()
        if current_act is None:
            return original(*args, **kwargs)
        if current_act.rank == 0:
            return permitted(*args, **kwargs)
        logger.debug("view %s rank %d: not writing persistent cache entry",
                     current_act.view.engine_id, current_act.rank)
        return None

    _set(act, compiler, "_cache_write", _mark(cache_write, original))


# ------------------------------------------------------------ tpu_inference sites

class _XbProxy(types.ModuleType):
    """Stand-in for ``jax._src.xla_bridge`` inside tpu_inference.utils: ``devices()`` is virtual."""

    def __init__(self, real):
        super().__init__(real.__name__)
        self.__dict__["_vproc_real"] = real
        import jax

        def devices(backend=None):
            return jax.devices(backend)

        self.__dict__["devices"] = devices

    def __getattr__(self, name):
        return getattr(self.__dict__["_vproc_real"], name)

    def __dir__(self):
        return sorted(set(dir(self.__dict__["_vproc_real"])) | set(self.__dict__))


def _patch_tpu_utils(act: _Activation, module) -> None:
    xb = getattr(module, "xb", None)
    if xb is not None and not isinstance(xb, _XbProxy):
        _set(act, module, "xb", _XbProxy(xb))


def _patch_tpu_distributed_utils(act: _Activation, module) -> None:
    fn = getattr(module, "get_device_topology_order_id", None)
    if fn is None or hasattr(fn, "__vproc_original__"):
        return

    @functools.wraps(fn)
    def get_device_topology_order_id(local_devices, global_devices):
        current_act = _current_activation()
        if current_act is not None:
            hosts = current_act.view.hosts
            global_devices = [d for d in global_devices if d.process_index in hosts]
        return fn(local_devices, global_devices)

    wrapper = _mark(get_device_topology_order_id, fn)
    _set(act, module, "get_device_topology_order_id", wrapper)
    # Importers that already did ``from ...distributed.utils import get_device_topology_order_id``.
    for modname, other in list(sys.modules.items()):
        if other is module or not modname.startswith("tpu_inference") or not isinstance(other, types.ModuleType):
            continue
        for attr, value in list(vars(other).items()):
            if value is fn:
                _set(act, other, attr, wrapper)


_LATE_PATCHES = {
    "tpu_inference.utils": _patch_tpu_utils,
    "tpu_inference.distributed.utils": _patch_tpu_distributed_utils,
}


class _LateImportFinder(importlib.abc.MetaPathFinder):
    """Applies the tpu_inference patches to modules first imported while a view is active."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _LATE_PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None:
                continue
            loader = spec.loader
            if loader is None or not hasattr(loader, "exec_module"):
                return spec
            exec_module = loader.exec_module

            def patched_exec_module(module, _exec=exec_module, _name=fullname):
                _exec(module)
                with _LOCK:
                    if _ACTIVE is not None:
                        _LATE_PATCHES[_name](_ACTIVE, module)

            loader.exec_module = patched_exec_module
            return spec
        return None


def _patch_loaded_tpu_modules(act: _Activation) -> None:
    for name, patch in _LATE_PATCHES.items():
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType):
            patch(act, module)


# ------------------------------------------------------------------- lifecycle

def activate(view: ViewSpec) -> None:
    """Show ``view`` through the engine-facing JAX API of this process (process-global)."""
    global _ACTIVE
    import jax
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError(f"view {_ACTIVE.view.engine_id!r} is already active; deactivate() first")
        physical_index, physical_count = jax.process_index(), jax.process_count()
        if physical_index not in view.hosts:
            raise ValueError(f"process {physical_index} is not in view {view.engine_id!r} hosts {view.hosts}")
        if view.hosts[-1] >= physical_count:
            raise ValueError(f"view {view.engine_id!r} hosts {view.hosts} exceed the runtime's {physical_count} processes")
        act = _Activation(view, physical_index)
        try:
            _patch_jax_names(act)
            _patch_cache_write(act)
            _patch_loaded_tpu_modules(act)
            act.finder = _LateImportFinder()
            sys.meta_path.insert(0, act.finder)
        except BaseException:
            _restore(act)
            raise
        _ACTIVE = act


def _restore(act: _Activation) -> None:
    if act.finder is not None and act.finder in sys.meta_path:
        sys.meta_path.remove(act.finder)
    for owner, name, original, _ in reversed(act.patches):
        setattr(owner, name, original)
    act.patches.clear()


def deactivate() -> None:
    """Restore the exact original objects. Raises if someone replaced a patched name meanwhile."""
    global _ACTIVE
    with _LOCK:
        act = _ACTIVE
        if act is None:
            raise RuntimeError("no active view")
        if _SUSPENDED:
            raise RuntimeError("deactivate() inside physical()")
        stale = [f"{getattr(owner, '__name__', owner)}.{name}" for owner, name, _, new in act.patches
                 if getattr(owner, name) is not new]
        if stale:
            raise RuntimeError(f"patched names were replaced by someone else, refusing to restore: {stale}")
        _restore(act)
        _ACTIVE = None


# ----------------------------------------------------------------- coordination

def _view_or_current(view: Optional[ViewSpec]) -> ViewSpec:
    if view is not None:
        return view
    act = _ACTIVE
    if act is None:
        raise RuntimeError("no active view")
    return act.view


def _client():
    from jax._src import distributed
    client = distributed.global_state.client
    if client is None:
        raise RuntimeError("jax.distributed is not initialized")
    return client


def namespaced(key: str, view: Optional[ViewSpec] = None) -> str:
    return f"{_view_or_current(view).namespace}/{key}"


def kv_set(key: str, value: str, allow_overwrite: bool = False, view: Optional[ViewSpec] = None) -> None:
    _client().key_value_set(namespaced(key, view), value, allow_overwrite)


def kv_get(key: str, timeout_ms: int = 60_000, view: Optional[ViewSpec] = None) -> str:
    return _client().blocking_key_value_get(namespaced(key, view), timeout_ms)


def barrier(name: str, timeout_ms: int = 60_000, view: Optional[ViewSpec] = None) -> None:
    """Barrier over the view's hosts only; ``name`` must be unused in this namespace."""
    view = _view_or_current(view)
    _client().wait_at_barrier(namespaced(name, view), timeout_ms, list(view.hosts))
