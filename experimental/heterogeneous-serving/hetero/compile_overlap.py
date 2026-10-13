"""Opt-in startup attribution; all hooks are removed before serving begins.

Intervals are raw wall-clock intervals, not additive CPU time. Parallel compile
spans may overlap; use partition_intervals to account for each second once.
JAX's compile-and-load API is indivisible on a cache miss: this is explicitly
reported as such, never relabelled as pure CPU compile.
"""
from __future__ import annotations

import functools
import threading
import time
from contextlib import ExitStack
from unittest.mock import patch


def partition_intervals(start, end, rows):
    """Exclusive wall-clock attribution using most-specific interval priority."""
    priorities = ("executable_load", "cache_lookup_and_decompress", "cache_load",
                  "backend_compile_and_load", "trace", "lower",
                  "warmup_execute_and_sync", "allocation_and_sync", "aot_compile")
    points = sorted({start, end} | {max(start, min(end, r[k]))
                                    for r in rows for k in ('start', 'end')})
    totals = {k: 0.0 for k in (*priorities, 'other')}
    for a, b in zip(points, points[1:]):
        active = {r['kind'] for r in rows if r['start'] <= a and r['end'] >= b}
        key = next((k for k in priorities if k in active), 'other')
        totals[key] += b - a
    return totals

def summarize_device_trace(directory, marker):
    """Retain every TPU module interval used in timing; no event-name absence proof."""
    import collections
    import gzip
    import hashlib
    import json
    from pathlib import Path
    def union(intervals):
        end = None
        total = 0.0
        for a, b in sorted(intervals):
            total += max(0.0, b - max(a, end if end is not None else a))
            end = max(b, end if end is not None else b)
        return total / 1e6
    files = []
    for path in sorted(Path(directory).rglob("*.trace.json.gz")):
        raw = path.read_bytes()
        events = json.loads(gzip.decompress(raw))["traceEvents"]
        devices = {e["pid"]: e["args"]["name"] for e in events
                   if e.get("ph") == "M" and e.get("name") == "process_name"
                   and e.get("args", {}).get("name", "").startswith("/device:TPU:")}
        lanes = {(e["pid"], e["tid"]) for e in events
                 if e.get("ph") == "M" and e.get("name") == "thread_name"
                 and e.get("args", {}).get("name") == "XLA Modules" and e["pid"] in devices}
        markers = [e for e in events if e.get("ph") == "X" and e.get("name") == marker]
        window = (min(e["ts"] for e in markers), max(e["ts"] + e["dur"] for e in markers)) if markers else None
        rows = []
        for e in events:
            if e.get("ph") != "X" or (e.get("pid"), e.get("tid")) not in lanes:
                continue
            start, end = e["ts"], e["ts"] + e["dur"]
            clipped = (max(start, window[0]), min(end, window[1])) if window else None
            rows.append({"device": devices[e["pid"]], "name": e["name"],
                         "start_us": start, "end_us": end, "args": e.get("args", {}),
                         "phase_interval_us": list(clipped) if clipped and clipped[1] > clipped[0] else None})
        inside = [r for r in rows if r["phase_interval_us"]]
        grouped = collections.defaultdict(list)
        for row in inside:
            grouped[row["device"]].append(row["phase_interval_us"])
        files.append({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "events": len(events), "device_lanes": list(devices.values()), "marker_window_us": window,
                      "device_module_execution_wall_s": union([r["phase_interval_us"] for r in inside]) if devices and window else None,
                      "per_device_module_execution_s": {k: union(v) for k, v in grouped.items()},
                      "module_events": rows})
    return {"files": files, "interpretation": "Union of observed TPU XLA Modules on device lanes within the explicit phase annotation. Not allocator API time, memory residency, or proof that unobserved device work is zero."}


class StartupProfile:
    def __init__(self):
        self.rows = []
        self.events = []
        self.executables = []
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._lowered = {}

    def timed(self, kind, call, *args, metadata=None, **kwargs):
        start, cpu = time.time(), time.thread_time()
        try:
            return call(*args, **kwargs)
        finally:
            row = {'kind': kind, 'start': start, 'end': time.time(),
                   'thread_cpu_s': time.thread_time() - cpu,
                   'thread_id': threading.get_ident(),
                   'executable': metadata or getattr(self._tls, 'executable', None)}
            with self._lock:
                self.rows.append(row)

    def event(self, name, start_time, end_time, **kwargs):
        duration = end_time - start_time
        end = end_time  # dispatch.log_elapsed_time records time.time epochs.
        with self._lock:
            self.events.append({'name': name, 'duration_s': duration, 'end': end,
                                'executable': getattr(self._tls, 'executable', None),
                                'metadata': kwargs})
            kind = {'/jax/core/compile/jaxpr_trace_duration': 'trace',
                    '/jax/core/compile/jaxpr_to_mlir_module_duration': 'lower'}.get(name)
            if kind:
                self.rows.append({'kind': kind, 'start': end - duration, 'end': end})

    def cache_read(self, original, *args, **kwargs):
        start = time.time()
        result = original(*args, **kwargs)
        hit = result[0] is not None
        row = {'kind': 'cache_load' if hit else 'cache_miss_lookup',
               'start': start, 'end': time.time(), 'hit': hit,
               'module': str(args[0]), 'cache_key': str(args[1]),
               'executable': getattr(self._tls, 'executable', None)}
        with self._lock:
            self.rows.append(row)
        return result

    def report(self, start, end):
        return {'start': start, 'end': end, 'wall_s': end - start,
                'exclusive_wall_s': partition_intervals(start, end, self.rows),
                'intervals': self.rows, 'monitoring_events': self.events,
                'executables': self.executables,
                "limitations": ["executable_load is backend deserialize/load wall time on a cache hit",
                                'backend_compile_and_load cannot separate backend compilation from loading',
                                'warmup_execute_and_sync includes host dispatch and synchronization; nested compile intervals are subtracted']}

    def hooks(self, manager=None, allocation_module=None):
        import jax
        from jax import stages
        from jax._src import compiler
        stack = ExitStack()
        jax.monitoring.register_event_time_span_listener(self.event)
        stack.callback(jax.monitoring.unregister_event_time_span_listener, self.event)
        jax.monitoring.register_event_listener(self.count_event)
        stack.callback(jax.monitoring.unregister_event_listener, self.count_event)
        from jax._src import compilation_cache
        def load_cached(key, options, backend, devices, host_callbacks=()):
            def lookup():
                cache = compilation_cache._get_cache(backend)
                value = cache.get(key) if cache is not None else None
                if value is None:
                    return None
                return compilation_cache.extract_executable_and_time(
                    compilation_cache.decompress_executable(value))
            value = self.timed("cache_lookup_and_decompress", lookup, metadata={"cache_key": key})
            if value is None:
                return None, None
            binary, compile_time = value
            args = (binary, devices, options)
            if host_callbacks:
                args += (host_callbacks,)
            executable = self.timed("executable_load", backend.deserialize_executable,
                                    *args, metadata={"cache_key": key})
            return executable, compile_time
        stack.enter_context(patch.object(compilation_cache, "get_executable_and_time", load_cached))
        original_read = compiler._cache_read
        stack.enter_context(patch.object(compiler, '_cache_read',
                            lambda *a, **k: self.cache_read(original_read, *a, **k)))
        original_backend = compiler.backend_compile_and_load
        stack.enter_context(patch.object(compiler, 'backend_compile_and_load',
                            lambda *a, **k: self.timed('backend_compile_and_load', original_backend, *a, **k)))
        original_compile = stages.Lowered.compile
        def compile_lowered(lowered, *args, **kwargs):
            prior = getattr(self._tls, 'executable', None)
            self._tls.executable = self._lowered.get(id(lowered))
            try:
                return self.timed('aot_compile', original_compile, lowered, *args, **kwargs)
            finally:
                self._tls.executable = prior
        stack.enter_context(patch.object(stages.Lowered, 'compile', compile_lowered))
        if manager is not None:
            original_run = manager._run_compilation
            profile = self
            warm_descriptors = {}
            def run(name, fn, *args, **kwargs):
                call_kwargs = kwargs.get('call_kwargs', {})
                while isinstance(fn, functools.partial):
                    args = fn.args + args
                    call_kwargs = {**fn.keywords, **call_kwargs}
                    fn = fn.func
                kwargs['call_kwargs'] = call_kwargs
                shapes = [{'shape': list(a.shape), 'dtype': str(a.dtype)}
                          for a in jax.tree.leaves((args, call_kwargs))
                          if hasattr(a, "shape") and hasattr(a, "dtype")]
                descriptor = {'name': name, 'buckets': {k: str(v) for k, v in kwargs.items()
                                                       if k not in ('call_kwargs', 'warmup_handler')},
                              'arguments': shapes, 'index': len(profile.executables)}
                profile.executables.append(descriptor)
                if hasattr(fn, 'lower'):
                    class Proxy:
                        def lower(self, *a, **k):
                            value = profile.timed('lower', fn.lower, *a, metadata=descriptor, **k)
                            profile._lowered[id(value)] = descriptor
                            return value
                        def __call__(self, *a, **k):
                            return fn(*a, **k)
                    wrapped = Proxy()
                else:
                    wrapped = fn
                before = len(manager._warmup_tasks)
                result = original_run(name, wrapped, *args, **kwargs)
                for task in manager._warmup_tasks[before:]:
                    warm_descriptors[id(task)] = descriptor
                return result
            stack.enter_context(patch.object(manager, '_run_compilation', run))
            original_flush = manager._flush_compilations
            def flush():
                first_dispatch = None
                wrapped = []
                for task in manager._warmup_tasks:
                    name, fn, args, kwargs, handler = task
                    descriptor = warm_descriptors.get(id(task), {"name": name})
                    def execute(f, a, k, h=handler, d=descriptor):
                        nonlocal first_dispatch
                        if first_dispatch is None:
                            first_dispatch = time.time()
                        call = lambda: h(f, a, k) if h else f(*a, **k)
                        return profile.timed("warmup_dispatch", call, metadata=d)
                    wrapped.append((name, fn, args, kwargs, execute))
                manager._warmup_tasks = wrapped
                try:
                    return original_flush()
                finally:
                    # Preserve original pipelining and final output barriers.
                    if first_dispatch is not None:
                        profile.rows.append({"kind": "warmup_execute_and_sync",
                                             "start": first_dispatch, "end": time.time()})
            stack.enter_context(patch.object(manager, '_flush_compilations', flush))
        if allocation_module is not None:
            original = allocation_module.create_kv_caches
            def allocate(*args, **kwargs):
                def call():
                    out = original(*args, **kwargs)
                    jax.block_until_ready(out)
                    return out
                return self.timed('allocation_and_sync', call,
                                  metadata={k: str(v) for k, v in kwargs.items() if k != 'mesh'})
            stack.enter_context(patch.object(allocation_module, 'create_kv_caches', allocate))
        return stack

    def count_event(self, name, **kwargs):
        with self._lock:
            self.events.append({'name': name, 'end': time.time(), 'metadata': kwargs,
                                'executable': getattr(self._tls, 'executable', None)})
