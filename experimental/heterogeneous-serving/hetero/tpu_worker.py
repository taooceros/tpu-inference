"""Instance-local TPU worker: rank remapping and effective-environment evidence.

vLLM's TPU backend creates one Ray worker actor per *host* (4 chips), not per
chip, and forcibly sets ``parallel_config.worker_cls`` inside
``TpuPlatform.check_and_update_config``.  The engine actor therefore overrides
``worker_cls`` *after* config creation and ships the resolved layout through
``vllm_config.additional_config['omp_hetero']`` so that each worker can:

* apply this instance's subslice TPU environment before any device init,
* translate the slice-wide physical ``TPU_WORKER_ID`` into an instance-local
  process rank derived from the ordered ``TPU_PROCESS_ADDRESSES``, and
* report the *effective* environment and real device membership as evidence.

Carrying the payload on the config rather than on the process environment keeps
this correct regardless of Ray's environment-variable copy allowlist.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Dict, List, Optional, Sequence

import ray

from tpu_inference.worker.tpu_worker import TPUWorker  # type: ignore

from .native_backend import BACKEND_REVISION, RUNTIME_REVISION, verify_backend

EVIDENCE_ENV_KEYS = (
    "TPU_MULTIHOST_BACKEND",
    "TPU_TOPOLOGY",
    "TPU_HOST_BOUNDS",
    "TPU_PROCESS_BOUNDS",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_CHIPS_PER_PROCESS_BOUNDS",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TPU_VISIBLE_CHIPS",
    "TPU_WORKER_ID",
    "TPU_WORKER_HOSTNAMES",
    "TPU_NAME",
    "CLOUD_TPU_TASK_ID",
    "JAX_PLATFORMS",
    "JAX_COMPILATION_CACHE_DIR",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "OMP_HETERO_INSTANCE_ID",
)


class HeteroTPUWorker(TPUWorker):
    """TPU worker bound to one logical serving instance's execution domain."""

    def _hetero_payload(self) -> Dict[str, Any]:
        additional = getattr(self.vllm_config, "additional_config", None) or {}
        payload = additional.get("omp_hetero")
        if not payload:
            raise RuntimeError(
                "HeteroTPUWorker requires vllm_config.additional_config['omp_hetero']; "
                "the engine actor must attach the resolved instance layout"
            )
        return payload

    def init_device(self, *args, **kwargs):  # type: ignore[override]
        payload = self._hetero_payload()
        local_ip = ray.util.get_node_ip_address()

        host_order: List[str] = list(payload["host_ips"])
        if len(set(host_order)) != len(host_order):
            raise RuntimeError(f"instance host order repeats a host: {host_order}")
        if local_ip not in host_order:
            raise RuntimeError(
                f"worker on {local_ip} is not a member of instance "
                f"{payload['instance_id']} hosts {host_order}"
            )
        process_rank = host_order.index(local_ip)

        addresses = payload["runtime_env"]["TPU_PROCESS_ADDRESSES"].split(",")
        if [a.rsplit(":", 1)[0] for a in addresses] != host_order:
            raise RuntimeError(
                "TPU_PROCESS_ADDRESSES order does not match the resolved host order"
            )
        if self.rank == 0 and process_rank != 0:
            raise RuntimeError(
                f"executor rank 0 landed on {local_ip} (process rank {process_rank}); "
                "coordinator and rank zero must agree"
            )

        self.omp_physical_tpu_worker_id = os.environ.get("TPU_WORKER_ID")
        self.omp_process_rank = process_rank
        self.omp_instance_id = payload["instance_id"]
        self.omp_node_ip = local_ip
        self.omp_hostname = socket.gethostname()
        self.omp_intended_env = dict(payload["runtime_env"])
        self.omp_intended_env["CLOUD_TPU_TASK_ID"] = str(process_rank)
        self.omp_intended_env["TPU_WORKER_ID"] = str(process_rank)

        # Applied before super().init_device() because that call performs JAX/PJRT
        # device initialization and the libtpu rendezvous.
        for key, value in self.omp_intended_env.items():
            os.environ[key] = value

        if payload.get("backend_revision") != BACKEND_REVISION:
            raise RuntimeError("worker backend revision does not match the engine configuration")
        if payload.get("runtime_revision") != RUNTIME_REVISION:
            raise RuntimeError("worker runtime revision does not match the engine configuration")
        self.omp_backend_evidence = verify_backend()
        additional = self.vllm_config.additional_config
        for field in ("activation_dtype", "matmul_precision"):
            if additional.get(field) != payload.get(field):
                raise RuntimeError(f"worker {field} does not match the engine configuration")
        if additional.get("activation_dtype") != "float32":
            raise RuntimeError("native Gemma4 FP32-state backend requires activation_dtype=float32")
        if additional.get("matmul_precision") not in ("default", "high", "highest"):
            raise RuntimeError("unsupported worker matmul_precision")
        if self.vllm_config.cache_config.cache_dtype != payload.get("kv_cache_dtype"):
            raise RuntimeError("worker cache dtype does not match the engine configuration")
        import jax

        # Set before native runner construction or any model tracing, not in a
        # thread-local context that expires before the execution thread runs.
        jax.config.update("jax_default_matmul_precision", additional["matmul_precision"])
        result = super().init_device(*args, **kwargs)

        self.omp_effective_env = {k: os.environ.get(k) for k in EVIDENCE_ENV_KEYS}
        self.omp_device_evidence = self._collect_device_evidence(payload)
        return result

    def load_model(self):
        result = super().load_model()
        self.omp_precision_evidence = self._verify_model_precision()
        return result

    def _verify_model_precision(self) -> Dict[str, Any]:
        """Inspect once after loading, before cache allocation and captured execution."""
        from collections import Counter

        import jax
        import jax.numpy as jnp
        from flax import nnx
        from tpu_inference.models.jax.gemma4 import Gemma4Model

        runner = self.model_runner
        payload = self._hetero_payload()
        if runner.kv_caches:
            raise RuntimeError("model precision must be verified before cache allocation")
        configured_dtype = str(runner.model_config.dtype).removeprefix("torch.")
        requested_dtype = payload["dtype"]
        requested_dtype = {"half": "float16", "float": "float32"}.get(
            requested_dtype, requested_dtype
        )
        if requested_dtype != "auto" and configured_dtype != requested_dtype:
            raise RuntimeError("native model parameter dtype differs from requested dtype")
        parameter_dtypes = dict(Counter(
            str(array.dtype)
            for array in jax.tree_util.tree_leaves(nnx.state(runner.model, nnx.Param))
        ))
        if not parameter_dtypes or set(parameter_dtypes) != {configured_dtype}:
            raise RuntimeError(
                f"model parameters {parameter_dtypes} do not match {configured_dtype}"
            )

        roots = [module for _, module in nnx.iter_graph(runner.model)
                 if isinstance(module, Gemma4Model)]
        if len(roots) != 1:
            raise RuntimeError(f"expected one native Gemma4 text backbone, found {len(roots)}")
        root = roots[0]
        activation_dtype = str(jnp.dtype(root.activation_dtype))
        if activation_dtype != payload["activation_dtype"]:
            raise RuntimeError("native model activation dtype differs from requested dtype")
        norm_dtypes = Counter()
        attention_dtypes = Counter()
        for layer in root.layers:
            for name in (
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
            ):
                dtype = getattr(layer, name).dtype
                if dtype is None or str(jnp.dtype(dtype)) != activation_dtype:
                    raise RuntimeError(f"decoder {name} is not configured for {activation_dtype}")
                norm_dtypes[str(jnp.dtype(dtype))] += 1
            attention_dtypes[str(jnp.dtype(layer.self_attn.kv_cache_quantized_dtype))] += 1
        if not norm_dtypes:
            raise RuntimeError("native model has no decoder norm precision evidence")
        cache_dtype = runner.cache_config.cache_dtype
        runner_cache_dtype = str(runner.kv_cache_dtype).removeprefix("torch.")
        expected_cache_dtype = configured_dtype if cache_dtype == "auto" else cache_dtype
        if cache_dtype != payload["kv_cache_dtype"] or runner_cache_dtype != expected_cache_dtype:
            raise RuntimeError("native runner KV dtype differs from the configured cache dtype")
        if set(attention_dtypes) != {expected_cache_dtype}:
            raise RuntimeError("native attention KV dtype differs from the runner cache dtype")
        precision = jax.config.jax_default_matmul_precision
        if precision != payload["matmul_precision"]:
            raise RuntimeError("effective JAX matmul precision differs from requested precision")
        return {
            "verified_before_cache_allocation": True,
            "model_dtype": configured_dtype,
            "parameter_dtypes": parameter_dtypes,
            "activation_dtype": activation_dtype,
            "decoder_norm_dtypes": dict(norm_dtypes),
            "attention_cache_dtypes": dict(attention_dtypes),
            "configured_cache_dtype": cache_dtype,
            "runner_cache_dtype": runner_cache_dtype,
            "matmul_precision": precision,
            "backend_revision": BACKEND_REVISION,
            "runtime_revision": RUNTIME_REVISION,
        }

    def _collect_device_evidence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Qualifies the execution domain the runtime actually built.

        A subslice runtime re-bases chip coordinates on its own origin, so the
        parent-slice absolute coordinates are not directly comparable.  What must
        hold is structural: the runtime sees exactly this instance's chip count
        and box shape, this host owns exactly one host-sized sub-box, and that
        sub-box sits at the same *relative* position inside the instance box that
        the resolved layout assigned to this host.
        """
        import jax

        local = jax.local_devices()
        glob = jax.devices()
        selected = list(self.devices or [])
        observed_local = sorted(tuple(int(v) for v in d.coords) for d in local)
        observed_global = sorted(tuple(int(v) for v in d.coords) for d in glob)

        parent_local = [tuple(c) for c in payload["chip_coords_by_host"][self.omp_node_ip]]
        parent_global = [tuple(c) for c in payload["chip_coords"]]
        instance_origin = tuple(min(c[axis] for c in parent_global) for axis in range(3))
        expected_relative = sorted(
            tuple(c[axis] - instance_origin[axis] for axis in range(3)) for c in parent_local
        )
        expected_global_relative = sorted(
            tuple(c[axis] - instance_origin[axis] for axis in range(3)) for c in parent_global
        )

        if len(glob) != int(payload["tp_size"]):
            raise RuntimeError(
                f"{payload['instance_id']} runtime sees {len(glob)} devices, expected "
                f"{payload['tp_size']}; unintended rendezvous outside the execution domain "
                f"(observed coords {observed_global})"
            )
        if len(local) != len(parent_local):
            raise RuntimeError(
                f"{payload['instance_id']} worker on {self.omp_node_ip} owns {len(local)} "
                f"chips, expected {len(parent_local)}"
            )
        if _box_dims(observed_global) != _box_dims(expected_global_relative):
            raise RuntimeError(
                f"{payload['instance_id']} runtime box {_box_dims(observed_global)} does not "
                f"match the resolved subslice box {_box_dims(expected_global_relative)} "
                f"(observed {observed_global})"
            )
        relative_match = observed_local == expected_relative
        if not relative_match:
            raise RuntimeError(
                f"{payload['instance_id']} worker on {self.omp_node_ip} (process rank "
                f"{self.omp_process_rank}) owns runtime chips {observed_local}; the resolved "
                f"layout places it at relative chips {expected_relative} inside instance "
                f"origin {instance_origin}"
            )
        return {
            "jax_process_index": int(jax.process_index()),
            "jax_process_count": int(jax.process_count()),
            "global_device_count": len(glob),
            "local_device_count": len(local),
            "instance_chip_origin_in_parent": list(instance_origin),
            "runtime_global_chip_coords": [list(c) for c in observed_global],
            "runtime_local_chip_coords": [list(c) for c in observed_local],
            "parent_local_chip_coords": [list(c) for c in sorted(parent_local)],
            "expected_relative_local_coords": [list(c) for c in expected_relative],
            "relative_position_match": relative_match,
            "selected_device_ids": [int(d.id) for d in selected],
            "selected_chip_coords": [list(int(v) for v in d.coords) for d in selected],
            "topology_order_id": int(getattr(self, "topology_order_id", -1)),
            "hbm_gb": _hbm(selected or local),
        }

    def omp_worker_evidence(self) -> Dict[str, Any]:
        """Collective-RPC entry point returning this worker's qualification record."""
        return {
            "instance_id": getattr(self, "omp_instance_id", None),
            "node_ip": getattr(self, "omp_node_ip", None),
            "hostname": getattr(self, "omp_hostname", None),
            "physical_tpu_worker_id": getattr(self, "omp_physical_tpu_worker_id", None),
            "instance_process_rank": getattr(self, "omp_process_rank", None),
            "executor_rank": getattr(self, "rank", None),
            "executor_local_rank": getattr(self, "local_rank", None),
            "pid": os.getpid(),
            "intended_env": getattr(self, "omp_intended_env", {}),
            "effective_env": getattr(self, "omp_effective_env", {}),
            "devices": getattr(self, "omp_device_evidence", {}),
            "native_backend": getattr(self, "omp_backend_evidence", {}),
            "precision": getattr(self, "omp_precision_evidence", {}),
        }

    def omp_kv_cache_evidence(self) -> Dict[str, Any]:
        """Reports the worker-side KV cache shape actually allocated."""
        runner = getattr(self, "model_runner", None)
        caches = getattr(runner, "kv_caches", None)
        rows = []
        if caches:
            for index, cache in enumerate(caches):
                shape = getattr(cache, "shape", None)
                dtype = getattr(cache, "dtype", None)
                rows.append(
                    {
                        "layer_index": index,
                        "shape": list(shape) if shape is not None else None,
                        "dtype": str(dtype) if dtype is not None else None,
                        "nbytes": int(getattr(cache, "nbytes", 0) or 0),
                    }
                )
        return {
            "node_ip": getattr(self, "omp_node_ip", None),
            "backend_revision": BACKEND_REVISION,
            "configured_cache_dtype": getattr(getattr(runner, "cache_config", None), "cache_dtype", None),
            "runner_cache_dtype": str(getattr(runner, "kv_cache_dtype", None)),
            "cache_dtypes": sorted({r["dtype"] for r in rows}),
            "num_cache_arrays": len(rows),
            "caches": rows[:4],
            "total_cache_bytes": sum(r["nbytes"] for r in rows),
        }

    # ------------------------------------------------- host-staged KV transfer

    def omp_kv_layout(self) -> Dict[str, Any]:
        """Collective RPC describing native groups, physical arrays and device slices."""
        from .kv_transport import cache_layout

        return cache_layout(self.model_runner)

    def omp_export_blocks(self, addressing: Dict[str, Any]) -> Dict[str, Any]:
        """Export native pages; caller must drain and exclude concurrent execution."""
        from .kv_transport import export_blocks

        return export_blocks(self.model_runner, addressing)

    def omp_import_blocks(
        self, addressing: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Populate native-reserved pages, never allocate request blocks here."""
        from .kv_transport import import_blocks

        return import_blocks(self.model_runner, addressing, payload)

    def omp_resume_fenced_request_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Restore authoritative token history before native resumed scheduling.

        The actor must drain the execution queue and exclude stepping first.
        Native async output retirement can skip a fenced request after it leaves
        the persistent batch, retaining placeholder IDs in CachedRequestState.
        Finalize pending host bookkeeping exactly as native zero-token execution
        does, then repair only this resident request. The scheduler must emit a
        resumed cached-request row so native update_states rebuilds input_batch.
        This does not touch KV, allocation, sampling/RNG state or output budget.
        """
        runner = self.model_runner
        request_id = payload["request_id"]
        state = runner.requests.get(request_id)
        if state is None:
            raise ValueError("fenced source request is not resident on this worker")
        prompt = list(payload["prompt_token_ids"])
        output = list(payload["generated_token_ids"])
        computed = payload["num_computed_tokens"]
        if not prompt or prompt != state.prompt_token_ids:
            raise ValueError("fenced source prompt identity changed or is empty")
        if any(not isinstance(token, int) or token < 0 for token in output):
            raise ValueError("fenced source output contains placeholder token IDs")
        if not isinstance(computed, int) or not 0 <= computed < len(prompt) + len(output):
            raise ValueError("fenced source needs an uncomputed continuation token")
        if state.mm_features or state.lora_request or getattr(runner, "uses_mrope", False):
            raise ValueError("multimodal, LoRA and M-RoPE source resumption is unsupported")
        blocks = tuple(list(group) for group in payload["block_ids"])
        if len(blocks) != len(runner.kv_cache_config.kv_cache_groups):
            raise ValueError("source resumption requires every native cache-group block table")
        pending = getattr(runner, "_pre_async_results", None) is not None
        if pending:
            runner._modify_prev_results()
            runner._pre_async_results = None
        # Keep the native object: outstanding metadata and continuation ownership
        # may reference it. Native resumed scheduling handles batch reinstallation.
        state.output_token_ids[:] = output
        state.num_computed_tokens = computed
        state.block_ids = blocks
        return {"request_id": request_id, "history_restored": True,
                "num_computed_tokens": computed, "num_generated_tokens": len(output),
                "last_token_id": output[-1] if output else prompt[-1],
                "native_async_results_flushed": pending}


    def omp_prepare_continuation_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Seed a PREEMPTED native request without admitting it to the batch.

        The actor supplies native SamplingParams and full native block tables,
        including sliding-window null entries. It must keep this request fenced
        until all workers acknowledge preparation and any KV import completes.
        """
        from tpu_inference.runner.input_batch import CachedRequestState
        from vllm.sampling_params import SamplingParams

        runner = self.model_runner
        request_id = payload["request_id"]
        identity = (payload["transaction_id"], payload["destination_generation"])
        if not request_id or not all(identity):
            raise ValueError("continuation requires request, transaction and generation identity")
        records = getattr(self, "_omp_pending_continuations", None)
        if records is None:
            records = self._omp_pending_continuations = {}
        # Native completion removes runner.requests; do not retain token histories.
        for key in list(records):
            if key not in runner.requests:
                del records[key]
        prior = records.get(request_id)
        if prior is not None:
            if prior["identity"] != identity:
                raise ValueError("continuation belongs to another transaction/generation")
            if runner.requests.get(request_id) is not prior["state"]:
                raise ValueError("native continuation state was replaced")
            return {"request_id": request_id, "prepared": True, "active": prior["active"]}
        if request_id in runner.requests or request_id in runner.input_batch.req_id_to_index:
            raise ValueError("native request ID is already resident")
        prompt = list(payload["prompt_token_ids"])
        output = list(payload["generated_token_ids"])
        computed = payload["num_computed_tokens"]
        if not prompt or not output or not isinstance(computed, int) or not 0 <= computed < len(prompt) + len(output):
            raise ValueError("continuation needs token history and an uncomputed token")
        params = payload["sampling_params"]
        if not isinstance(params, SamplingParams):
            raise ValueError("continuation requires native SamplingParams")
        if payload.get("mm_features") or payload.get("lora_request") or getattr(runner, "uses_mrope", False):
            raise ValueError("multimodal, LoRA and M-RoPE continuations are unsupported")
        blocks = tuple(list(group) for group in payload["block_ids"])
        if len(blocks) != len(runner.kv_cache_config.kv_cache_groups):
            raise ValueError("continuation block tables must cover every native cache group")
        state = CachedRequestState(
            req_id=request_id, prompt_token_ids=prompt, output_token_ids=output,
            mm_features=[], sampling_params=params, pooling_params=None,
            generator=None, block_ids=blocks, num_computed_tokens=computed,
            lora_request=None,
        )
        runner.requests[request_id] = state
        records[request_id] = {"identity": identity, "state": state, "active": False}
        return {"request_id": request_id, "prepared": True, "active": False}

    def omp_activate_continuation_state(
        self, request_id: str, transaction_id: str, destination_generation: str
    ) -> Dict[str, Any]:
        """Commit preparation ownership before the scheduler resumes execution."""
        record = getattr(self, "_omp_pending_continuations", {}).get(request_id)
        if record is None or record["identity"] != (transaction_id, destination_generation):
            raise ValueError("no continuation prepared under this transaction/generation")
        if self.model_runner.requests.get(request_id) is not record["state"]:
            raise ValueError("native continuation state was replaced")
        record["active"] = True
        return {"request_id": request_id, "active": True}

    def omp_abort_continuation_state(
        self, request_id: str, transaction_id: str, destination_generation: str
    ) -> Dict[str, Any]:
        """Discard unactivated worker state; native scheduler owns block freeing."""
        records = getattr(self, "_omp_pending_continuations", {})
        record = records.get(request_id)
        if record is None:
            return {"request_id": request_id, "aborted": False}
        if record["identity"] != (transaction_id, destination_generation):
            raise ValueError("continuation belongs to another transaction/generation")
        runner = self.model_runner
        if record["active"] or request_id in runner.input_batch.req_id_to_index:
            raise ValueError("cannot abort activated continuation state")
        if runner.requests.get(request_id) is not record["state"]:
            raise ValueError("native continuation state was replaced")
        del runner.requests[request_id]
        del records[request_id]
        return {"request_id": request_id, "aborted": True}


def _box_dims(coords: Sequence[Any]) -> tuple:
    if not coords:
        return (0, 0, 0)
    return tuple(
        max(c[axis] for c in coords) - min(c[axis] for c in coords) + 1 for axis in range(3)
    )

def _hbm(devices: List[Any]) -> Optional[float]:
    try:
        from tpu_inference import utils  # type: ignore

        return float(utils.hbm_usage_gb(devices))
    except Exception:
        return None
