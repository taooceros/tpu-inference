"""Gemma-4 31B parameter leaves (measured on the cluster) and CPU-sized scaled copies.

Source: ``results/ici/InPlaceEngine/parameter_catalogue.json``, the 1028 leaves
of the real tpu-inference NNX graph (raysubmit_6Kx3zaVPJJwS7UvW). Leaf paths are
kept verbatim so a vLLM engine and a synthetic engine name leaves the same way.

Fused leaves (``qkv_proj``, ``gate_up_proj``) are stored *per TP shard
interleaved* along their last (TP) axis: for each of the engine's ``tp`` shards,
that shard's slice of every part in turn (tpu-inference
``JaxMergedColumnParallelLinear``; repack: ``hetero/ici_resident.py:53``). The
logical layout concatenates whole parts. ``to_stored``/``to_logical`` convert.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .types import LeafSpec

CATALOGUE = Path(__file__).resolve().parents[2] / "results/ici/InPlaceEngine/parameter_catalogue.json"

# Gemma-4 31B fused widths (logical): sliding-layer qkv = 32 q heads + 16 k + 16 v heads x 256.
FUSED_SIZES = {"qkv_proj": ("qkv", (8192, 4096, 4096)), "gate_up_proj": ("gate_up", (21504, 21504))}

_LAYER = re.compile(r"\['(language_model|vision_tower)'\]\['layers'\]\[(\d+)\]")


def _tp_axis(partition_spec: str) -> Optional[int]:
    entries = [e.strip() for e in partition_spec[2:-1].split(",") if e.strip()]
    for axis, entry in enumerate(entries):
        if entry.strip("'\"") == "model":
            return axis
    return None


def gemma4_31b() -> Tuple[LeafSpec, ...]:
    """All 1028 leaves of the served model, logical shapes, catalogue order."""
    rows = json.loads(CATALOGUE.read_text())["parameters"]
    leaves = []
    for row in rows:
        fused, sizes = None, ()
        for name, (kind, parts) in FUSED_SIZES.items():
            if f"['{name}']" in row["path"]:
                fused, sizes = kind, parts
        if fused and sum(sizes) != row["shape"][-1]:
            raise ValueError(f"{row['path']}: fused sizes {sizes} != width {row['shape'][-1]}")
        leaves.append(LeafSpec(row["path"], tuple(row["shape"]), row["dtype"],
                               _tp_axis(row["partition_spec"]), fused, sizes))
    return tuple(leaves)


def _shrink(n: int, divisor: int, multiple: int) -> int:
    if n < 1024:
        return n
    return max(multiple, math.ceil(n / divisor / multiple) * multiple)


def scaled(divisor: int = 64, language_layers: Optional[int] = 6, vision: bool = False,
           dtype: Optional[str] = "float32", multiple: int = 16) -> Tuple[LeafSpec, ...]:
    """A CPU-sized copy with the same leaf structure.

    Every dim >= 1024 is divided by ``divisor`` and rounded up to ``multiple``
    (so TP8 and TP16 both divide every TP axis and every fused part); smaller
    dims (heads, head_dim) are kept. ``language_layers`` keeps the first N
    decoder layers (the default 6 includes the first global layer, index 5).
    """
    out = []
    for leaf in gemma4_31b():
        match = _LAYER.search(leaf.path)
        if match and match.group(1) == "language_model" and language_layers is not None \
                and int(match.group(2)) >= language_layers:
            continue
        if not vision and ("['vision_tower']" in leaf.path or "['embed_vision']" in leaf.path):
            continue
        shape = tuple(_shrink(n, divisor, multiple) for n in leaf.shape)
        sizes = tuple(_shrink(n, divisor, multiple) for n in leaf.fused_sizes)
        if sizes:
            shape = shape[:-1] + (sum(sizes),)
        out.append(LeafSpec(leaf.path, shape, dtype or leaf.dtype, leaf.tp_axis, leaf.fused, sizes))
    return tuple(out)


def nbytes(leaves: Tuple[LeafSpec, ...]) -> int:
    return sum(int(np.prod(l.shape)) * np.dtype(_np_dtype(l.dtype)).itemsize for l in leaves)


def _np_dtype(name: str):
    if name == "bfloat16":
        import ml_dtypes
        return ml_dtypes.bfloat16
    return np.dtype(name)


def synthetic_value(leaf: LeafSpec, seed: int, low: int = -2, high: int = 2) -> np.ndarray:
    """Deterministic logical value: small integers, exact in float32 and bfloat16."""
    digest = hashlib.sha256(f"{seed}:{leaf.path}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.integers(low, high + 1, size=leaf.shape).astype(_np_dtype(leaf.dtype))


def to_stored(logical: np.ndarray, leaf: LeafSpec, tp: int) -> np.ndarray:
    """Logical -> the layout an engine with TP degree ``tp`` holds (identity unless fused)."""
    if not leaf.fused:
        return logical
    lead = logical.shape[:-1]
    parts, offset = [], 0
    for size in leaf.fused_sizes:
        parts.append(logical[..., offset:offset + size].reshape(lead + (tp, size // tp)))
        offset += size
    return np.concatenate(parts, axis=-1).reshape(logical.shape)


def to_logical(stored: np.ndarray, leaf: LeafSpec, tp: int) -> np.ndarray:
    if not leaf.fused:
        return stored
    lead = stored.shape[:-1]
    grouped = stored.reshape(lead + (tp, -1))
    parts, offset = [], 0
    for size in leaf.fused_sizes:
        width = size // tp
        parts.append(grouped[..., offset:offset + width].reshape(lead + (size,)))
        offset += width
    return np.concatenate(parts, axis=-1)


def logical_checksum(logical: np.ndarray) -> str:
    array = np.ascontiguousarray(logical)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode()
                          + array.tobytes()).hexdigest()[:16]


def summary(leaves: Tuple[LeafSpec, ...]) -> Dict[str, int]:
    return {"leaves": len(leaves), "fused": sum(1 for l in leaves if l.fused),
            "replicated": sum(1 for l in leaves if l.tp_axis is None), "bytes": nbytes(leaves)}
