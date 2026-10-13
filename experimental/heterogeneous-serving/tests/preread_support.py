"""CPU-test helpers for the parked-target pre-read (tests/test_preread.py).

A TINY checkpoint of the production architecture (Gemma4ForConditionalGeneration, the class
/models/gemma-4-31b resolves to) and a vllm_config built like ``EngineActor.initialize`` builds
it (EngineArgs.create_engine_config under the TPU platform), on JAX's CPU backend only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

# Production text features kept (k_eq_v, global head dim and kv heads, proportional rope, both
# attention types, tied embeddings, soft-capping), dimensions shrunk; head dims stay multiples of
# 128 (TI pads head_dim to 128, utils.get_padded_head_dim) as in production (256 / 512). TP16 divisibility as in
# production: 32 q heads, 16 kv heads, 4 global kv heads.
TINY_TEXT = {
    "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": True,
    "enable_moe_block": False, "final_logit_softcapping": 30.0, "global_head_dim": 256,
    "head_dim": 128, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64,
    "hidden_size_per_layer_input": 0, "intermediate_size": 128,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
    "max_position_embeddings": 4096, "model_type": "gemma4_text", "num_attention_heads": 32,
    "num_global_key_value_heads": 4, "num_hidden_layers": 3, "num_key_value_heads": 16,
    "num_kv_shared_layers": 0, "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                           "rope_type": "proportional"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 16, "tie_word_embeddings": True, "use_bidirectional_attention": "vision",
    "use_cache": True, "use_double_wide_mlp": False, "vocab_size": 512,
    "vocab_size_per_layer_input": 512, "dtype": "bfloat16",
}
TINY_VISION = {
    "attention_bias": False, "attention_dropout": 0.0, "default_output_length": 4, "dtype": "bfloat16",
    "global_head_dim": 16, "head_dim": 16, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 256,
    "intermediate_size": 256, "max_position_embeddings": 256, "model_type": "gemma4_vision",
    "num_attention_heads": 16, "num_hidden_layers": 1, "num_key_value_heads": 16, "patch_size": 16,
    "pooling_kernel_size": 3, "position_embedding_size": 10240, "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}, "standardize": True,
    "use_clipped_linears": False,
}


def build_tiny_checkpoint(directory: str | os.PathLike, *, seed: int = 0) -> Path:
    """config.json + model.safetensors (random bf16) of a tiny Gemma4ForConditionalGeneration."""
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    directory = Path(directory)
    if (directory / "model.safetensors").is_file():
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    config = Gemma4Config(text_config=dict(TINY_TEXT), vision_config=dict(TINY_VISION),
                          audio_config=None, tie_word_embeddings=True,
                          architectures=["Gemma4ForConditionalGeneration"], vision_soft_tokens_per_image=4,
                          image_token_id=500, boi_token_id=501, eoi_token_id=502, audio_token_id=503,
                          boa_token_id=504, eoa_token_id=505, video_token_id=506, eos_token_id=[1, 106])
    torch.manual_seed(seed)
    model = Gemma4ForConditionalGeneration(config).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():  # non-trivial values everywhere (norm scales too)
            parameter.copy_(torch.randn_like(parameter, dtype=torch.float32).to(torch.bfloat16))
    model.save_pretrained(directory, safe_serialization=True)
    return directory


def tiny_payload(tp: int = 16) -> Dict[str, Any]:
    """An H-like instance payload: 4 hosts x 2x2 chips of a 4x4 box (process ranks 0-3)."""
    hosts = ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]
    coords = {hosts[0]: [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
              hosts[1]: [[2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0]],
              hosts[2]: [[0, 2, 0], [1, 2, 0], [0, 3, 0], [1, 3, 0]],
              hosts[3]: [[2, 2, 0], [3, 2, 0], [2, 3, 0], [3, 3, 0]]}
    return {"instance_id": "large_a", "layout_id": "H", "tp_size": tp, "host_ips": hosts,
            "chip_coords_by_host": coords,
            "chip_coords": [c for h in hosts for c in coords[h]],
            "runtime_env": {"TPU_MULTIHOST_BACKEND": "ray", "TPU_TOPOLOGY": "4x4",
                            "TPU_HOST_BOUNDS": "2,2,1", "TPU_PROCESS_BOUNDS": "2,2,1",
                            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1", "TPU_CHIPS_PER_PROCESS_BOUNDS": "2,2,1",
                            "TPU_PROCESS_ADDRESSES": ",".join(f"{h}:8471" for h in hosts),
                            "TPU_PROCESS_PORT": "8471", "JAX_PLATFORMS": "tpu,cpu"}}


def build_vllm_config(model: str, *, tp: int = 16, payload: Dict[str, Any] | None = None):
    """vllm_config as hetero/engine_actor.py EngineActor.initialize creates it (CPU host, TPU platform)."""
    from hetero.preread import force_tpu_platform
    force_tpu_platform()  # before any vllm module binds `current_platform`
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=str(model), tensor_parallel_size=tp, distributed_executor_backend="ray",
        dtype="bfloat16", kv_cache_dtype="bfloat16",
        additional_config={"activation_dtype": None, "matmul_precision": "default",
                           "omp_hetero": payload or tiny_payload(tp)},
        max_model_len=256, max_num_seqs=8, max_num_batched_tokens=256, gpu_memory_utilization=0.85,
        enable_prefix_caching=False, enforce_eager=True, seed=0, max_logprobs=32,
        trust_remote_code=True, disable_log_stats=False, skip_tokenizer_init=True,
        limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})
    config = engine_args.create_engine_config()
    config.parallel_config.worker_cls = "hetero.tpu_worker.HeteroTPUWorker"
    return config


def write_json(path: str | os.PathLike, data: Any) -> None:
    Path(path).write_text(json.dumps(data, sort_keys=True))
