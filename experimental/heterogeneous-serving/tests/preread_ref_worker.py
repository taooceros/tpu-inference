"""One process of a 4-process CPU "TPU instance" for tests/test_preread.py (not a test module).

    python tests/preread_ref_worker.py PHASE RANK SPEC.json

Four processes (``jax.distributed`` with gloo CPU collectives) of 4 CPU devices each form a
16-device mesh. Position k of the mesh belongs to process ``positions_of[k]``, using the H
table's non-contiguous local positions, so each rank owns the same mesh positions as on TPU.

PHASE ``normal``: the real normal loader (``model_loader.get_model``, production
``TPU_MULTIHOST_BACKEND=ray`` placement via ``make_array_from_callback``), then
``weight_snapshot.snapshot`` of ``model.state`` and a device ``fingerprint``.

PHASE ``restore``: the production restore builder (as ``tpu_worker._load_snapshot_or_normal``):
``prepare`` this rank's pre-read snapshot, ``nnx.eval_shape``, ``validate_tree``, ``restore``,
``nnx.update``, then the rest of ``get_flax_model``; fingerprint of the restored state.

Test-harness adaptation: tpu_inference converts checkpoint tensors on
``Mesh(jax.devices('cpu')[:1])``. On TPU hosts that is a process-local CPU device. Under
``jax.distributed`` CPU collectives it would be process 0's device, so it is pinned to this
process's first local device.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PHASE, RANK, SPEC = sys.argv[1], int(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
os.environ.update({"JAX_PLATFORMS": "cpu", "TPU_SKIP_MDS_QUERY": "1", "TPU_MULTIHOST_BACKEND": "ray",
                   "XLA_FLAGS": "--xla_force_host_platform_device_count=4"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hetero.preread import force_tpu_platform  # noqa: E402

force_tpu_platform()
import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(f"127.0.0.1:{SPEC['port']}", num_processes=4, process_id=RANK)

from flax import nnx  # noqa: E402
from jax.sharding import AxisType, Mesh  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402

from hetero.weight_snapshot import (_devices, fingerprint, prepare, restore, snapshot,  # noqa: E402
                                    validate_tree)
from tests.preread_support import build_vllm_config  # noqa: E402
from tpu_inference.distributed import jax_parallel_state  # noqa: E402
from tpu_inference.layers.common import utils as common_utils  # noqa: E402
from tpu_inference.models.common import model_loader  # noqa: E402

common_utils._cpu_mesh = Mesh(jax.local_devices(backend="cpu")[:1], ("cpu",))
by_process = {}
for device in jax.devices():
    by_process.setdefault(device.process_index, []).append(device)
for devices in by_process.values():
    devices.sort(key=lambda d: d.id)
used = {p: 0 for p in by_process}
flat = []
for process in SPEC["positions_of"]:
    flat.append(by_process[process][used[process]])
    used[process] += 1
mesh = Mesh(np.array(flat).reshape(1, 16), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
config = build_vllm_config(SPEC["model"], tp=16)
jax_parallel_state.init_pp_distributed_environment("", 0, 1, jax.local_devices()[0], need_pp=False)
rank_dir = Path(SPEC["root"]) / PHASE / f"rank{RANK}"
rank_dir.mkdir(parents=True, exist_ok=True)
result = {"rank": RANK, "devices": _devices(mesh)}

if PHASE == "normal":
    with set_current_vllm_config(config):
        model = model_loader.get_model(config, jax.random.key(config.model_config.seed), mesh)
    result["snapshot"] = snapshot(Path(SPEC["reference_root"]) / f"rank{RANK}", model.state, SPEC["context"],
                                  generation=SPEC["generation"])
    result["fingerprint"] = fingerprint(model.state)
else:
    clocks = {}
    prepared = prepare(Path(SPEC["preread_root"]) / f"rank{RANK}", SPEC["context"], mesh, clocks,
                       verify_checksums=False)
    original = model_loader._get_nnx_model

    def snapshot_builder(model_class, cfg, rng, mesh_, pooler=None, is_draft_model=False):
        with jax.set_mesh(mesh_):
            abstract = nnx.eval_shape(lambda: model_class(cfg, rng, mesh_))
        validate_tree(nnx.state(abstract), mesh_, prepared)
        restored = restore(nnx.state(abstract), mesh_, prepared, clocks)
        nnx.update(abstract, restored)
        return abstract

    model_loader._get_nnx_model = snapshot_builder
    try:
        with set_current_vllm_config(config):
            model = model_loader.get_model(config, jax.random.key(config.model_config.seed), mesh)
    finally:
        model_loader._get_nnx_model = original
    result["fingerprint"] = fingerprint(model.state)
    result["restore_clocks"] = sorted(clocks)
for key in ("fingerprint",):
    result[key].pop("validation_s", None)
(rank_dir / "result.json").write_text(json.dumps(result, sort_keys=True, default=str))
jax.distributed.shutdown()
