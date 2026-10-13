"""Opt-in, reversible load profiling; never installed on the serving step path.

Exclusive categories partition elapsed time. Synchronizing transfer outputs prevents
asynchronous TPU work from being charged to the next tensor. This deliberately
serializes transfers in profiling mode; reported bandwidth is observed, not peak.
"""
from __future__ import annotations

import functools
import heapq
import importlib
import os
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager


class LoadProfile:
    def __init__(self, *, sample_stacks=False):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.calls = defaultdict(lambda: [0.0, 0, 0])
        self.top = []
        self.stack = []
        self.patches = []
        self.tensor = None
        self.serial = 0
        self.started = None
        self.elapsed = None
        self.implementation = None
        self.sample_stacks = sample_stacks
        self.stack_samples = []
        self.samplers = []

    @contextmanager
    def span(self, category, name, tensor=None, byte_count=0):
        start = time.perf_counter()
        entry = [category, 0.0]
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += elapsed
            exclusive = max(0.0, elapsed - entry[1])
            self.totals[category] += exclusive
            self.counts[category] += 1
            row = self.calls[name]
            row[0] += exclusive
            row[1] += 1
            row[2] += byte_count
            self.serial += 1
            item = (elapsed, self.serial, {
                "call": name, "tensor": tensor or self.tensor,
                "wall_s": elapsed, "exclusive_s": exclusive,
                "category": category, "bytes": byte_count})
            if len(self.top) < 20:
                heapq.heappush(self.top, item)
            elif elapsed > self.top[0][0]:
                heapq.heapreplace(self.top, item)

    def patch(self, owner, name, replacement):
        original = getattr(owner, name)
        self.patches.append((owner, name, original))
        setattr(owner, name, replacement(original))

    def wrap(self, category, label, *, sync=False, tensor_arg=None):
        def decorate(original):
            @functools.wraps(original)
            def call(*args, **kwargs):
                tensor = kwargs.get("param_name")
                if tensor_arg is not None and len(args) > tensor_arg:
                    tensor = str(args[tensor_arg])
                with self.span(category, label, tensor):
                    result = original(*args, **kwargs)
                    if sync and hasattr(result, "block_until_ready"):
                        result.block_until_ready()
                    return result
            return call
        return decorate

    def iterator(self, original):
        @functools.wraps(original)
        def iterate(*args, **kwargs):
            source = iter(original(*args, **kwargs))
            while True:
                with self.span("file_deserialize", original.__name__):
                    try:
                        item = next(source)
                    except StopIteration:
                        return
                self.tensor = str(item[0])
                yield item
        return iterate

    def __enter__(self):
        try:
            import jax
            from tpu_inference.models.jax.utils import weight_utils as weights
            from vllm.model_executor.model_loader import default_loader
            from vllm.model_executor.model_loader import weight_utils as vweights
            from tpu_inference.models.common import model_loader
            # Patch imported aliases, not just the original defining module.
            for module in (vweights, default_loader, weights):
                for name in ("safetensors_weights_iterator", "model_weights_single_file_generator"):
                    if hasattr(module, name):
                        self.patch(module, name, self.iterator)
            for name in ("jax_array_from_reshaped_torch", "load_nnx_param_from_reshaped_torch",
                         "assign_and_shard_param", "_load_and_shard_weight"):
                self.patch(weights, name, self.wrap("host_tensor_ops", name))
            self.patch(weights, "shard_put", self.wrap("host_to_device", "shard_put", sync=True))
            pxla = importlib.import_module("jax._src.interpreters.pxla")
            def raw_transfer_wrapper(original):
                @functools.wraps(original)
                def call(aval, sharding, xs, devices, *args, **kwargs):
                    if not any(device.platform == "tpu" for device in devices):
                        return original(aval, sharding, xs, devices, *args, **kwargs)
                    local_bytes = sum(x.nbytes for x in xs)
                    with self.span("host_to_device", "raw_batched_device_put", byte_count=local_bytes):
                        result = original(aval, sharding, xs, devices, *args, **kwargs)
                        result.block_until_ready()
                        return result
                return call
            self.patch(pxla, "batched_device_put", raw_transfer_wrapper)
            # CPU indexing in the multihost callback may itself compile kernels.
            def callback_wrapper(original):
                @functools.wraps(original)
                def call(shape, sharding, data_callback, *args, **kwargs):
                    wrapped = self.wrap("host_tensor_ops", "local_shard_selection", sync=True)(data_callback)
                    return original(shape, sharding, wrapped, *args, **kwargs)
                return call
            self.patch(jax, "make_array_from_callback", callback_wrapper)
            compiler = importlib.import_module("jax._src.compiler")
            self.patch(compiler, "compile_or_get_cached", self.wrap("jit_compile", "compile_or_get_cached"))
            self.patch(jax, "clear_caches", self.wrap("post_load", "jax.clear_caches"))
            for impl in ("get_flax_model", "get_vllm_model"):
                def implementation_wrapper(original, impl=impl):
                    @functools.wraps(original)
                    def call(*args, **kwargs):
                        self.implementation = impl
                        return original(*args, **kwargs)
                    return call
                self.patch(model_loader, impl, implementation_wrapper)
            for modname in ("unquantized", "fp8", "mxfp4"):
                module = importlib.import_module("tpu_inference.layers.jax.quantization." + modname)
                for name, cls in vars(module).items():
                    if isinstance(cls, type) and cls.__module__ == module.__name__:
                        for method in ("process_weights_after_loading", "_load_merged_tensor"):
                            if method in cls.__dict__:
                                descriptor = cls.__dict__[method]
                                decorator = self.wrap("post_load" if method.startswith("process") else "host_tensor_ops", name + "." + method)
                                if isinstance(descriptor, staticmethod):
                                    self.patches.append((cls, method, descriptor))
                                    setattr(cls, method, staticmethod(decorator(descriptor.__func__)))
                                else:
                                    self.patch(cls, method, decorator)
            # Measure final addressable bytes at the synchronized transfer boundary.
            original_shard_put = weights.shard_put
            def counted(*args, **kwargs):
                result = original_shard_put(*args, **kwargs)
                local_bytes = sum(s.data.nbytes for s in result.addressable_shards)
                self.calls["shard_put"][2] += local_bytes
                return result
            self.patches.append((weights, "shard_put", original_shard_put))
            weights.shard_put = counted
        except BaseException:
            self._restore()
            raise
        self.started = time.perf_counter()
        self.root = self.span("host_tensor_ops", "load_model_unwrapped")
        self.root.__enter__()
        if self.sample_stacks and shutil.which("py-spy"):
            for delay in (30, 90):
                timer = threading.Timer(delay, self._sample_stack)
                timer.daemon = True
                self.samplers.append(timer)
                timer.start()
        return self

    def _sample_stack(self):
        try:
            result = subprocess.run(
                ["py-spy", "dump", "--pid", str(os.getpid()), "--nonblocking"],
                capture_output=True, text=True, timeout=10)
            self.stack_samples.append({"load_elapsed_s": time.perf_counter() - self.started,
                                       "returncode": result.returncode,
                                       "stdout": result.stdout, "stderr": result.stderr})
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.stack_samples.append({"error": str(exc)})

    def _restore(self):
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()

    def __exit__(self, *exc):
        try:
            self.root.__exit__(*exc)
            self.elapsed = time.perf_counter() - self.started
        finally:
            self._restore()
            for timer in self.samplers:
                timer.cancel()
                timer.join(timeout=11)

    def evidence(self):
        categories = ("file_deserialize", "host_tensor_ops", "host_to_device", "jit_compile", "post_load")
        transfer = self.calls.get("raw_batched_device_put", [0, 0, 0])
        return {
            "schema_version": 1,
            "implementation": self.implementation,
            "elapsed_s": self.elapsed,
            "categories_s": {key: self.totals[key] for key in categories},
            "category_calls": {key: self.counts[key] for key in categories},
            "calls": {name: {"exclusive_s": row[0], "count": row[1], "local_bytes": row[2]}
                      for name, row in sorted(self.calls.items())},
            "top20": [item[2] for item in sorted(self.top, reverse=True)],
            "transfer_local_bytes": transfer[2],
            "transfer_observed_GBps": transfer[2] / 1e9 / transfer[0] if transfer[0] else None,
            "py_spy_dumps": self.stack_samples,
            "notes": ["Categories are nested exclusive wall times, not CPU time.",
                      "Host residual includes construction, bookkeeping and unwrapped operations.",
                      "File category is iterator advancement; mmap page faults during tensor conversion remain in host ops.",
                      "Transfers are synchronized; bandwidth includes dispatch overhead, excludes measured callback slicing and compile time.",
                      "compile_or_get_cached includes cache lookup/loading; not only fresh compilation."]}
