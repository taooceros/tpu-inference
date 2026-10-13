"""Executable bank (hetero/exec_bank.py) on 8 fake CPU devices.

The jax scenarios run this file as a script in fresh interpreters (write, then
read), so a bank is always produced and consumed by different processes. The
fake runner reproduces the installed tpu_inference call shapes (md5-identical
to the fork at 8d31f0866):

* ``run_model``: jit with donate_argnums=1 and static (4, 5, 6), reached via a
  non-jit ``wrapped_model_fn`` closure (models/common/model_loader.py:375-391, 467-471);
* ``compute_logits_fn`` jit on (state_leaves, hidden) (model_loader.py:412);
* ``sample`` static_argnames=["mesh"] and ``compute_and_gather_*logprobs``
  static max_logprobs as module globals (layers/jax/sample/sampling.py:105,154,165);
* ``_select_from_array_fn`` staticmethod jit static (2, 3) (runner/tpu_runner.py:2262-2264);
* ``_substitute_placeholder_token_fn`` donate (0, 1, 2) (tpu_runner.py:535);
* ``structured_decode_fn`` jitted method, static self (runner/structured_decoding_manager.py:33);
* ``FakeCompilationManager._run_compilation`` / ``_flush_compilations`` copy
  runner/compilation_manager.py:126-229 (partial unwrap, compile_only, AOT
  lower on the caller, compile in a thread pool, warm-up after the flush).
"""
import functools
import json
import os
import subprocess
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hetero import exec_bank as eb  # noqa: E402

LAYERS, HIDDEN, VOCAB, MAX_LOGPROBS = 2, 64, 128, 4
TOKENS = (16, 32)
REQS = (8,)
LAYER_MAP = tuple((f"layer.{i}", i) for i in range(LAYERS))
BANKED = ("run_model", "run_compute_logits", "sample", "compute_and_gather_logprobs",
          "compute_and_gather_prompt_logprobs", "select_from_array", "substitute",
          "structured_decode_fn")
SNAPSHOT_CONTEXT = {"model_path": "/models/fake", "tp_size": 8, "dtype": "float32", "quantization": None}


# --------------------------------------------------------------- fake runtime

# Module level: serialized in_tree references them by import path (as tpu_inference's
# AttentionMetadata / TPUSupportedSamplingMetadata, registered in _jax_setup).
@dataclass
class AttnMD:
    positions: object
    seq_lens: object
    padded_num_reqs: int


@dataclass
class SamplingMD:
    temperature: object
    do_sampling: bool


def _jax_setup():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    jax.tree_util.register_dataclass(AttnMD, data_fields=["positions", "seq_lens"],
                                     meta_fields=["padded_num_reqs"])
    jax.tree_util.register_dataclass(SamplingMD, data_fields=["temperature"],
                                     meta_fields=["do_sampling"])
    # Device order not in id order, as on a TPU sub-slice (H mesh ids [0, 1, 4, 5, 2, 3, ...]):
    # a bank that loads executables in backend.devices() order aborts here (O2 attempt 1).
    import numpy as np
    from jax.sharding import Mesh
    by_id = {d.id: d for d in jax.devices()}
    mesh = Mesh(np.array([[by_id[i] for i in (0, 1, 4, 5, 2, 3, 6, 7)]]), ("data", "model"))
    col, rep = NamedSharding(mesh, P(None, "model")), NamedSharding(mesh, P())

    @functools.partial(jax.jit, out_shardings=([col] * LAYERS, rep), donate_argnums=1,
                       static_argnums=(4, 5, 6))
    def run_model(state_leaves, kv_caches, input_ids, md, layer_map, is_first, is_last):
        h = state_leaves[0][input_ids % VOCAB]
        new_kv = []
        for i in range(LAYERS):
            h = jnp.tanh(h @ state_leaves[1 + i]) + md.seq_lens[:1].sum() * 0
            new_kv.append(kv_caches[i].at[: h.shape[0]].add(h))
        if is_last:
            h = h * 2.0
        return new_kv, h + len(layer_map)

    def make_model_fn(jitted_model_fn):
        def wrapped_model_fn(*args, **kwargs):  # model_loader.py:467-471
            kwargs.pop("shared_attention_metadata", None)
            return jitted_model_fn(*args, **kwargs)
        return wrapped_model_fn

    @functools.partial(jax.jit, out_shardings=rep)
    def run_compute_logits(state_leaves, hidden):
        return hidden @ state_leaves[0].T

    @functools.partial(jax.jit, static_argnames=["mesh"])
    def sample(rng, mesh, logits, md):
        scaled = logits / md.temperature if md.do_sampling else logits
        return jnp.argmax(scaled + jax.random.uniform(rng, logits.shape) * 0, axis=-1)

    @functools.partial(jax.jit, static_argnames=("max_logprobs",))
    def compute_and_gather_logprobs(logits, token_ids, max_logprobs):
        lp = jax.nn.log_softmax(logits, axis=-1)
        return jax.lax.top_k(lp, max_logprobs)[0], jnp.take_along_axis(lp, token_ids[:, None], 1)

    @functools.partial(jax.jit, static_argnames=("max_logprobs",))
    def compute_and_gather_prompt_logprobs(logits, token_ids, max_logprobs):
        return jax.lax.top_k(jax.nn.log_softmax(logits, axis=-1), max_logprobs)[0]

    @functools.partial(jax.jit, static_argnums=(2, 3))
    def select_from_array(array, indices, mesh, pcp_size=1):
        return array[indices] * pcp_size

    @functools.partial(jax.jit, donate_argnums=(0, 1, 2))
    def substitute(input_ids, cur_indices, pre_indices, next_tokens, placeholder_num):
        return input_ids.at[cur_indices].set(next_tokens[pre_indices] + placeholder_num * 0)

    class StructuredDecodingManager:
        @functools.partial(jax.jit, static_argnums=(0,))
        def structured_decode_fn(self, require, logits):
            return jnp.where(require[:, None], logits - 1.0, logits)

    modules = {name: types.ModuleType(f"fake_{name}")
               for name in ("tpu_runner", "compilation_manager", "sampling")}
    for name in ("sample", "compute_and_gather_logprobs"):
        setattr(modules["tpu_runner"], name, locals()[name])
        setattr(modules["compilation_manager"], name, locals()[name])
    for name in ("compilation_manager", "sampling"):
        setattr(modules[name], "compute_and_gather_prompt_logprobs", compute_and_gather_prompt_logprobs)
    eb.SINGLETON_STATIC_TYPES = eb.SINGLETON_STATIC_TYPES | {
        f"{StructuredDecodingManager.__module__}.{StructuredDecodingManager.__qualname__}"}

    class FakeCompilationManager:
        def __init__(self, runner):
            self.runner = runner
            self._compile_executor = ThreadPoolExecutor(max_workers=2)
            self._compile_futures = []
            self._warmup_tasks = []

        def _run_compilation(self, name, fn, *args, call_kwargs=dict(), warmup_handler=None,
                             aot=True, compile_only=False, **kwargs):
            while isinstance(fn, functools.partial):
                args = fn.args + args
                call_kwargs = {**fn.keywords, **call_kwargs}
                fn = fn.func
            is_jit = hasattr(fn, 'lower')
            if compile_only:
                if not is_jit:
                    raise ValueError(f"compile_only=True requires a JITted function: {name}")
            else:
                self._warmup_tasks.append((name, fn, args, call_kwargs, warmup_handler))
            if not compile_only and (not aot or not is_jit):
                return
            with jax.set_mesh(self.runner.mesh):
                lowered = fn.lower(*args, **call_kwargs)

            def _compile(lowered, mesh):
                with jax.set_mesh(mesh):
                    return lowered.compile()
            self._compile_futures.append(
                self._compile_executor.submit(_compile, lowered, self.runner.mesh))

        def _flush_compilations(self):
            futures, self._compile_futures = self._compile_futures, []
            tasks, self._warmup_tasks = self._warmup_tasks, []
            for fut in futures:
                fut.result()
            with jax.set_mesh(self.runner.mesh):
                for name, fn, args, call_kwargs, warmup_handler in tasks:
                    out = (warmup_handler(fn, args, call_kwargs) if warmup_handler is not None
                           else fn(*args, **call_kwargs))
                    jax.tree.map(lambda r: r.block_until_ready(), out)

        def _skip_self_arg_warmup_handler(self, fn, args, call_kwargs):
            return fn(*args[1:], **call_kwargs)

        def capture_model(self):
            r = self.runner
            with jax.set_mesh(r.mesh):
                for t in TOKENS:
                    for n in REQS:
                        ids, md = r.inputs(t, n)

                        def model_fn_warmup(_fn, _args, _kw, ids=ids, md=md):
                            out = r.model_fn(r.state_leaves, r.kv_caches, ids, md, LAYER_MAP,
                                             True, True, shared_attention_metadata=None)
                            r.kv_caches = out[0]
                            return out
                        self._run_compilation("backbone", r.model_fn, r.state_leaves, r.kv_caches,
                                              ids, md, LAYER_MAP, True, True,
                                              warmup_handler=model_fn_warmup, num_tokens=t)
                self._flush_compilations()
                for t in TOKENS:
                    ids, cur, pre, nxt = r.substitute_inputs(t)
                    self._run_compilation("_substitute_placeholder_token_fn",
                                          r._substitute_placeholder_token_fn, ids, cur, pre, nxt, 3,
                                          compile_only=False, num_tokens=t)
                self._flush_compilations()
                for n in REQS:
                    for t in TOKENS:  # array_size x index_size, as compilation_manager.py:779-834
                        hidden, idx = r.select_inputs(t, n)
                        self._run_compilation("select_from_array [all]", r._select_from_array_fn,
                                              hidden, idx, r.mesh, 1, compile_only=True)
                    self._run_compilation("worker0 compute_logits", r.compute_logits_fn,
                                          r.state_leaves, r.logit_hidden(n), compile_only=True)
                    for do_sampling in (True, False):
                        self._run_compilation("worker0 sample", self_module("sample"),
                                              r.rng, r.mesh, r.logits(n), r.sampling_md(n, do_sampling))
                    self._run_compilation("worker0 gather_logprobs",
                                          self_module("compute_and_gather_logprobs"), r.logits(n),
                                          r.tokens(n), MAX_LOGPROBS, compile_only=True)
                    self._run_compilation("worker0 prompt_logprobs",
                                          self_module("compute_and_gather_prompt_logprobs"),
                                          r.logits(n), r.tokens(n), MAX_LOGPROBS, compile_only=True)
                    sdm = r.structured_decoding_manager
                    self._run_compilation("structured_decode", sdm.structured_decode_fn, sdm,
                                          r.require(n), r.logits(n),
                                          warmup_handler=self._skip_self_arg_warmup_handler)
                self._flush_compilations()

    def self_module(name):
        return getattr(modules["compilation_manager"], name)

    class FakeRunner:
        _select_from_array_fn = staticmethod(select_from_array)

        def __init__(self):
            self.mesh = mesh
            k = jax.random.key(0)
            self.state_leaves = [jax.device_put(jax.random.normal(k, (VOCAB, HIDDEN)) * 0.1, rep)] + [
                jax.device_put(jax.random.normal(jax.random.fold_in(k, i), (HIDDEN, HIDDEN)) * 0.1, col)
                for i in range(LAYERS)]
            self.kv_caches = self.fresh_kv()
            self.model_fn = make_model_fn(run_model)
            self.compute_logits_fn = run_compute_logits
            self._substitute_placeholder_token_fn = substitute
            self.structured_decoding_manager = StructuredDecodingManager()
            self.compilation_manager = FakeCompilationManager(self)
            self.rng = jax.random.key(1)

        def fresh_kv(self):
            return [jax.device_put(jnp.zeros((64, HIDDEN)), col) for _ in range(LAYERS)]

        def inputs(self, t, n):
            ids = jax.device_put(jnp.arange(t, dtype=jnp.int32), rep)
            md = AttnMD(jax.device_put(jnp.arange(t, dtype=jnp.int32), rep),
                        jax.device_put(jnp.ones((n,), jnp.int32), rep), n)
            return ids, md

        def substitute_inputs(self, t):
            return (jax.device_put(jnp.arange(t, dtype=jnp.int32), rep),
                    jax.device_put(jnp.arange(4, dtype=jnp.int32), rep),
                    jax.device_put(jnp.arange(4, dtype=jnp.int32), rep),
                    jax.device_put(jnp.arange(8, dtype=jnp.int32) + 100, rep))

        def select_inputs(self, t, n):
            return (jax.device_put(jnp.ones((t, HIDDEN)), rep),
                    jax.device_put(jnp.arange(n, dtype=jnp.int32), rep))

        def logit_hidden(self, n):
            return jax.device_put(jnp.ones((n, HIDDEN)), rep)

        def logits(self, n):
            return jax.device_put(jnp.ones((n, VOCAB), jnp.float32), rep)

        def tokens(self, n):
            return jax.device_put(jnp.zeros((n,), jnp.int32), rep)

        def require(self, n):
            return jax.device_put(jnp.arange(n) % 2 == 0, rep)

        def sampling_md(self, n, do_sampling):
            return SamplingMD(jax.device_put(jnp.full((n, 1), 0.5, jnp.float32), rep), do_sampling)

        def step(self, t, n):
            """Serving path order of tpu_runner.py:1630-1701, 1376, 1994, 2055."""
            tr = modules["tpu_runner"]
            ids, md = self.inputs(t, n)
            with jax.set_mesh(self.mesh):
                self.kv_caches, hidden = self.model_fn(
                    self.state_leaves, self.kv_caches, ids, md, LAYER_MAP, True, True,
                    shared_attention_metadata=None)
                hidden = self._select_from_array_fn(hidden, jnp.arange(n, dtype=jnp.int32), self.mesh, 1)
                logits = self.compute_logits_fn(self.state_leaves, hidden)
                logits = self.structured_decoding_manager.structured_decode_fn(self.require(n), logits)
                tokens = tr.sample(self.rng, self.mesh, logits, self.sampling_md(n, True))
                top, picked = tr.compute_and_gather_logprobs(logits, tokens, MAX_LOGPROBS)
                prompt = modules["sampling"].compute_and_gather_prompt_logprobs(
                    logits, tokens, MAX_LOGPROBS)
                ids2, cur, pre, nxt = self.substitute_inputs(t)
                substituted = self._substitute_placeholder_token_fn(ids2, cur, pre, nxt, 3)
            # Host copies now: the next step donates this step's KV arrays.
            import numpy as np
            values = {"tokens": tokens, "top": top, "picked": picked, "prompt": prompt,
                      "kv0": self.kv_caches[0], "substituted": substituted}
            return {**{k: np.asarray(v) for k, v in values.items()},
                    "donated": bool(ids2.is_deleted())}

    return jax, jnp, modules, FakeRunner


class FileKV:
    """Cross-process stand-in for Ray internal_kv."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.root / key.replace("/", "__")

    def put(self, key, value):
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_bytes(value)
        os.replace(tmp, self._path(key))

    def get(self, key):
        path = self._path(key)
        return path.read_bytes() if path.exists() else None


def _identity(jax, **override):
    instance = {"model": "fake", "layers": LAYERS, "jax": jax.__version__,
                "platform_version": jax.devices()[0].client.platform_version, **override}
    return {"instance": instance, "rank": {"process_index": 0}}


def _compile_events(jax):
    events = []

    def listener(name, duration, **kwargs):
        if name in ("/jax/core/compile/jaxpr_trace_duration",
                    "/jax/core/compile/jaxpr_to_mlir_module_duration",
                    "/jax/core/compile/backend_compile_duration"):
            events.append((name, str(kwargs.get("fun_name", ""))))
    jax.monitoring.register_event_duration_secs_listener(listener)
    return events


def _banked_events(events, start):
    return [e for e in events[start:] if any(n in e[1] for n in BANKED)]


def _outputs(jax, results):
    import numpy as np
    return {f"{i}_{k}": np.asarray(v) for i, r in enumerate(results)
            for k, v in r.items() if k != "donated"}


def scenario(mode, root):
    jax, jnp, modules, FakeRunner = _jax_setup()
    import numpy as np
    events = _compile_events(jax)
    runner = FakeRunner()
    kv = FileKV(Path(root) / "kv")
    out = {"mode": mode}
    if mode == "write":
        # As a first (cold) deploy: the checkpoint-loaded weights are snapshotted right after the load
        # (weight_snapshot=write), then the bank is captured during precompilation (exec_bank=write).
        from hetero.weight_snapshot import snapshot
        snapshot(Path(root) / "snap", runner.state_leaves, SNAPSHOT_CONTEXT, generation="a" * 32)
        bank = eb.WorkerBank("write", str(Path(root) / "bank"), instance_id="large_a", nonce="w",
                             process_rank=0, process_count=1, kv=kv)
        bank.start_capture(runner, modules)
        runner.compilation_manager.capture_model()
        report = bank.finish_capture(lambda: _identity(jax))
        runner.kv_caches = runner.fresh_kv()  # same KV start for every compared run
        results = [runner.step(t, REQS[0]) for t in TOKENS]
        np.savez(Path(root) / "refs.npz", **_outputs(jax, results))
        out.update(written=report["entries"], skipped=report["skipped"],
                   functions=bank.report()["functions"], counters=dict(bank.counters))
    elif mode in ("read", "mismatch", "sabotage", "read_restored"):
        if mode == "read_restored":
            # As the later restore deploy: weights come from the snapshot, not the loader that ran
            # when the bank was written; the bank must still be used with no rejection.
            from hetero.weight_snapshot import prepare, restore
            abstract = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in runner.state_leaves]
            runner.state_leaves = None
            clocks = {}
            runner.state_leaves = restore(abstract, runner.mesh, prepare(
                Path(root) / "snap", SNAPSHOT_CONTEXT, runner.mesh, clocks, verify_checksums=False), clocks)
            out["restored_from_snapshot"] = "snapshot_device_placement" in clocks
        identity = _identity(jax, model="other") if mode == "mismatch" else _identity(jax)
        bank = eb.WorkerBank("read", str(Path(root) / "bank"), instance_id="large_a", nonce=mode,
                             process_rank=0, process_count=1, kv=kv,
                             sabotage_rank=0 if mode == "sabotage" else None)
        bank.load(runner, modules, lambda: identity)
        start = len(events)
        runner.compilation_manager.capture_model()
        runner.kv_caches = runner.fresh_kv()
        results = [runner.step(t, REQS[0]) for t in TOKENS]
        out.update(status={k: bank.status.get(k) for k in ("installed", "fallback_reason")},
                   verdict=bank.status["unanimity"]["reason"],
                   banked_compile_events=_banked_events(events, start),
                   counters=dict(bank.counters), donated=[r["donated"] for r in results])
        refs = np.load(Path(root) / "refs.npz")
        got = _outputs(jax, results)
        out["matches_write_process"] = all(np.array_equal(got[k], refs[k]) for k in refs.files)
        if mode == "read_restored":
            functions = bank.installation.functions
            out["hits"] = {k: f.hits for k, f in functions.items()}
            out["reports"] = {k: {"misses": r["misses"], "rejections": r["rejections"]}
                              for k, r in bank.report()["functions"].items()}
        if mode == "read":
            functions = bank.installation.functions
            out["hits"] = {k: f.hits for k, f in functions.items()}
            out["loaded"] = bank.status["loaded"]["entries"]
            # Miss: a bucket that was never banked runs through the original jit.
            runner.step(64, REQS[0])
            # Rejection: an invariant argument (weights, outside the key) changes dtype;
            # Compiled refuses before executing and the jit path serves the call.
            with jax.set_mesh(runner.mesh):
                fallback = runner.compute_logits_fn([a.astype(jnp.bfloat16) for a in runner.state_leaves],
                                                    runner.logit_hidden(REQS[0]))
            out["rejection_output_dtype"] = str(fallback.dtype)
            report = bank.report()["functions"]
            out["run_model_report"] = report["run_model"]
            out["compute_logits_report"] = report["compute_logits"]
            # Same steps on the native jit path agree with the banked executables.
            bank.deactivate()
            out["restored"] = modules["tpu_runner"].sample.__class__.__name__ != "BankedFunction"
            runner.kv_caches = runner.fresh_kv()
            native = [runner.step(t, REQS[0]) for t in TOKENS]
            out["matches_native_jit"] = all(
                np.array_equal(a, b) for a, b in zip(_outputs(jax, native).values(), got.values()))
    print("RESULT " + json.dumps(out, default=str))


def _run(mode, root):
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, __file__, mode, str(root)], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(proc.stderr[-4000:])
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


# ----------------------------------------------------------------- tests

class FreshProcessBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.TemporaryDirectory(prefix="exec-bank-test-")
        cls.root = Path(cls._tmp.name)
        cls.write = _run("write", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_write_captures_every_precompiled_bucket(self):
        functions = self.write["functions"]
        self.assertEqual(self.write["skipped"], [])
        # run_model x2 tokens, substitute x2, select x2, logits, sample x2, logprobs, prompt, structured.
        self.assertEqual(self.write["written"], 12)
        self.assertEqual(functions["run_model"]["captured"], 2)
        self.assertEqual(functions["sample"]["captured"], 2)
        self.assertEqual(functions["structured_decode"]["captured"], 1)
        self.assertEqual(self.write["counters"]["aot_captured"], 10)  # all but run_model (not a jit)

    def test_read_round_trip_skips_trace_lower_compile(self):
        out = _run("read", self.root)
        self.assertTrue(out["status"]["installed"], out)
        self.assertEqual(out["verdict"], "unanimous")
        self.assertEqual(out["loaded"], 12)
        self.assertEqual(out["banked_compile_events"], [])  # no trace/lower/compile for banked fns
        self.assertEqual(out["counters"], {"aot_skipped": 10})  # every AOT bucket skipped
        self.assertTrue(out["matches_write_process"])
        self.assertTrue(out["matches_native_jit"])
        self.assertTrue(all(out["donated"]))  # donate_argnums preserved through Compiled
        self.assertTrue(all(v > 0 for v in out["hits"].values()), out["hits"])
        # static args (layer map, flags, mesh, self) were stripped: no rejection on banked calls
        self.assertEqual(out["run_model_report"]["rejections"], 0)
        self.assertEqual(out["run_model_report"]["misses"], 1)
        self.assertIn("[64]:int32", out["run_model_report"]["miss_samples"][0])
        self.assertEqual(out["compute_logits_report"]["rejections"], 1)
        self.assertIn("TypeError", next(iter(out["compute_logits_report"]["rejection_samples"].values())))
        self.assertEqual(out["rejection_output_dtype"], "float32")  # served by the jit (bf16 @ f32)
        self.assertTrue(out["restored"])

    def test_identity_mismatch_refuses_bank_and_compiles_normally(self):
        out = _run("mismatch", self.root)
        self.assertFalse(out["status"]["installed"])
        self.assertIn("identity mismatch: ['instance.model']", out["status"]["fallback_reason"])
        self.assertTrue(out["banked_compile_events"])  # the normal path traced and compiled
        self.assertEqual(out["counters"], {})
        self.assertTrue(out["matches_write_process"])

    def test_sabotaged_rank_falls_back(self):
        out = _run("sabotage", self.root)
        self.assertFalse(out["status"]["installed"])
        self.assertIn("sabotaged", out["status"]["fallback_reason"])
        self.assertTrue(out["matches_write_process"])

    def test_bank_from_a_checkpoint_load_serves_snapshot_restored_weights(self):
        """F0: the bank written by the first (normal-loader) deploy is used, with no rejection, by a
        later deploy whose weights come from the inline snapshot of that same first deploy."""
        out = _run("read_restored", self.root)
        self.assertTrue(out["restored_from_snapshot"])
        self.assertTrue(out["status"]["installed"], out)
        self.assertEqual(out["verdict"], "unanimous")
        self.assertEqual(out["banked_compile_events"], [])
        self.assertEqual(out["counters"], {"aot_skipped": 10})
        self.assertTrue(out["matches_write_process"])
        self.assertTrue(all(v > 0 for v in out["hits"].values()), out["hits"])
        self.assertEqual({k: r["rejections"] for k, r in out["reports"].items() if r["rejections"]}, {})
        self.assertEqual({k: r["misses"] for k, r in out["reports"].items() if r["misses"]}, {})


class DictKV:
    def __init__(self):
        self.data, self.lock = {}, threading.Lock()

    def put(self, key, value):
        with self.lock:
            self.data[key] = value

    def get(self, key):
        with self.lock:
            return self.data.get(key)


class UnanimityTests(unittest.TestCase):
    def _vote(self, kv, votes, timeout_s=5.0):
        results = {}

        def run(rank, vote):
            try:
                results[rank] = eb.unanimity(kv, instance_id="large_a", nonce="n", process_rank=rank,
                                             process_count=len(votes), vote=vote,
                                             timeout_s=timeout_s, poll_s=0.001)
            except eb.ExecBankError as exc:
                results[rank] = exc
        threads = [threading.Thread(target=run, args=(r, v)) for r, v in enumerate(votes) if v]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def test_all_agree(self):
        vote = {"identity_digest": "a", "entry_set_digest": "e"}
        results = self._vote(DictKV(), [vote] * 4)
        self.assertTrue(all(r["use"] for r in results.values()))

    def test_one_refusal_makes_every_rank_fall_back(self):
        vote = {"identity_digest": "a", "entry_set_digest": "e"}
        results = self._vote(DictKV(), [vote, vote, {"identity_digest": "a", "error": "no bank"}, vote])
        self.assertEqual({r["use"] for r in results.values()}, {False})
        self.assertIn("no bank", results[1]["reason"])

    def test_entry_set_disagreement_falls_back(self):
        results = self._vote(DictKV(), [{"identity_digest": "a", "entry_set_digest": "e"},
                                         {"identity_digest": "a", "entry_set_digest": "f"}])
        self.assertEqual({r["use"] for r in results.values()}, {False})
        self.assertIn("disagree", results[0]["reason"])

    def test_missing_vote_times_out_to_fallback(self):
        vote = {"identity_digest": "a", "entry_set_digest": "e"}
        results = self._vote(DictKV(), [vote, vote, None], timeout_s=0.05)
        self.assertFalse(results[0]["use"])
        self.assertIn("no vote from ranks [2]", results[0]["reason"])
        self.assertFalse(results[1]["use"])

    def test_rank_without_verdict_refuses_to_proceed(self):
        ticks = iter(range(1000))
        with self.assertRaises(eb.ExecBankError):
            eb.unanimity(DictKV(), instance_id="i", nonce="n", process_rank=1, process_count=2,
                         vote={}, timeout_s=1.0, clock=lambda: next(ticks), sleep=lambda s: None)


class IdentityTests(unittest.TestCase):
    def test_instance_digest_covers_engine_shape_and_runtime(self):
        import jax
        import jax.numpy as jnp
        from types import SimpleNamespace as NS

        def runner(num_seqs=512):
            hf = NS(to_dict=lambda: {"hidden_size": 5376})
            config = NS(model_config=NS(model="/models/gemma-4-31b", revision=None, dtype="torch.bfloat16",
                                        quantization=None, max_model_len=8192, max_logprobs=32,
                                        enforce_eager=False, hf_config=hf),
                        cache_config=NS(cache_dtype="bfloat16", block_size=512, enable_prefix_caching=False),
                        scheduler_config=NS(max_num_seqs=num_seqs, max_num_batched_tokens=2048,
                                            async_scheduling=True),
                        speculative_config=None,
                        additional_config={"activation_dtype": None, "matmul_precision": "default"})
            return NS(vllm_config=config, mesh=jax.make_mesh((1, 1), ("data", "model")),
                      kv_caches=[jnp.zeros((4, 8))])

        payload = {"tp_size": 16, "runtime_env": {
            "TPU_PROCESS_ADDRESSES": "10.0.0.1:8471,10.0.0.2:8471", "TPU_TOPOLOGY": "4x4"}}

        def identity(r=None, p=payload, env=None, blocks=283):
            return eb.engine_identity(r or runner(), p, backend_revision="b", runtime_revision="r",
                                      runtime=eb.runtime_identity(env or {}), files={"m": "x"},
                                      num_blocks=blocks)

        base = eb.digest(identity()["instance"])
        self.assertEqual(base, eb.digest(identity()["instance"]))
        swapped = {**payload, "runtime_env": {**payload["runtime_env"],
                                              "TPU_PROCESS_ADDRESSES": "10.0.0.2:8471,10.0.0.1:8471"}}
        for variant in (identity(blocks=284), identity(runner(num_seqs=256)), identity(p=swapped),
                        identity(env={"XLA_FLAGS": "--xla_foo=1"})):
            self.assertNotEqual(base, eb.digest(variant["instance"]))
        rank = identity()["rank"]
        self.assertEqual(rank["process_index"], 0)
        self.assertIn("process_count", identity()["instance"]["parallel"])
        self.assertIn("platform_version", identity()["instance"]["runtime"])

    def test_engine_settings_validate_exec_bank(self):
        from hetero.engine_actor import EngineSettings
        self.assertEqual(EngineSettings(exec_bank="read", exec_bank_sabotage_rank=1).exec_bank, "read")
        for bad in ({"exec_bank": "on"}, {"exec_bank": "write", "exec_bank_sabotage_rank": 0},
                    {"exec_bank": "read", "exec_bank_dir": "relative"},
                    {"exec_bank": "read", "enforce_eager": True},
                    {"exec_bank": "read", "exec_bank_sabotage_rank": -1}):
            with self.assertRaises(ValueError, msg=bad):
                EngineSettings(**bad)


if __name__ == "__main__":
    scenario(sys.argv[1], sys.argv[2])
