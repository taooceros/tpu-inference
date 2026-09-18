"""
TPU Placement Group & Node-Pinned Resource Management.
======================================================
Implements mandatory explicit node-pinned bundle specifications for multihost TPU meshes
(STRICT_SPREAD, deterministic rank ordering, stale placement group cleanup).
"""

from typing import List, Dict, Any, Optional, Sequence, Union
import ray
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
    PlacementGroup,
)


def cleanup_stale_placement_groups(
    states: Optional[Sequence[str]] = None,
) -> int:
    """
    Cleans up all stale or pending placement groups from earlier runs.
    Strictly targets CREATED / PENDING groups to prevent autoscaler deadlocks.
    Returns the number of placement groups removed.
    """
    if not ray.is_initialized():
        return 0

    target_states = set(states or ("CREATED", "PENDING"))
    removed_count = 0

    try:
        table = placement_group_table()
        for pg_id, pg_info in table.items():
            state = pg_info.get("state", "")
            if state in target_states:
                try:
                    from ray._raylet import PlacementGroupID
                    remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(pg_id)))
                    removed_count += 1
                    print(
                        f"[Placement] Cleared stale placement group {pg_id} (state: {state})",
                        flush=True,
                    )
                except Exception as ex:
                    # Best-effort removal of individual stale group
                    print(
                        f"[Placement] Warning: could not remove placement group {pg_id}: {ex}",
                        flush=True,
                    )
    except Exception as e:
        print(
            f"[Placement] Notice: Could not inspect placement group table: {e}",
            flush=True,
        )

    if removed_count > 0:
        print(
            f"[OMP_EVENT: PLACEMENT_GROUPS_CLEANED removed_count={removed_count}]",
            flush=True,
        )
    return removed_count


def validate_tpu_v5e_topology(
    target_node_ips: Sequence[str],
    tpu_per_node: float = 4.0,
    expected_chips: int = 8,
) -> bool:
    """
    Enforces Cloud TPU v5e (2x4 or 4x4 ICI mesh, 8 or 16 chips) topology constraints.
    Validates node count, IP uniqueness, chip capacity, and active Ray node state.
    """
    if not target_node_ips:
        raise ValueError(
            "Target node IPs cannot be empty for TPU v5e TP=8 placement group."
        )

    # Check uniqueness (each bundle must pin to a distinct physical host)
    unique_ips = list(dict.fromkeys(target_node_ips))
    if len(unique_ips) != len(target_node_ips):
        raise ValueError(
            f"Duplicate host IPs detected in target nodes {target_node_ips}. "
            "STRICT_SPREAD requires strictly unique host nodes."
        )

    total_chips = len(target_node_ips) * tpu_per_node
    if total_chips != float(expected_chips):
        raise ValueError(
            f"Invalid TPU v5e topology: expected {expected_chips} total chips, "
            f"but got {len(target_node_ips)} hosts with {tpu_per_node} TPU/node = {total_chips} chips."
        )

    # If Ray is active, verify nodes exist and are alive
    if ray.is_initialized():
        try:
            alive_nodes = {
                n.get("NodeManagerAddress"): n
                for n in ray.nodes()
                if n.get("Alive", False)
            }
            for ip in target_node_ips:
                if ip not in alive_nodes:
                    print(
                        f"[Placement] Warning: Node IP {ip} not found in active Ray cluster nodes.",
                        flush=True,
                    )
                else:
                    node_res = alive_nodes[ip].get("Resources", {})
                    node_tpu = float(node_res.get("TPU", 0.0))
                    if node_tpu < tpu_per_node:
                        raise ValueError(
                            f"Node {ip} advertises {node_tpu} TPUs, but {tpu_per_node} "
                            "are required for TPU v5e TP=8 worker bundle."
                        )
        except ValueError:
            raise
        except Exception as ex:
            print(f"[Placement] Topology node inspection notice: {ex}", flush=True)

    return True

def validate_tp8_v5e_topology(
    target_node_ips: Sequence[str],
    tpu_per_node: float = 4.0,
    expected_chips: int = 8,
) -> bool:
    """Backwards-compatible wrapper for validate_tpu_v5e_topology."""
    return validate_tpu_v5e_topology(target_node_ips, tpu_per_node, expected_chips)


def remove_placement_group_safe(
    pg_or_id: Union[PlacementGroup, str],
) -> bool:
    """Safely removes a placement group by handle or ID."""
    if not ray.is_initialized():
        return False
    try:
        if isinstance(pg_or_id, str):
            pg = ray.util.placement_group(pg_or_id)
        else:
            pg = pg_or_id
        remove_placement_group(pg)
        return True
    except Exception as e:
        print(f"[Placement] Notice: Failed to remove placement group: {e}", flush=True)
        return False


def create_node_pinned_placement_group(
    target_node_ips: Sequence[str],
    tpu_per_node: float = 4.0,
    strategy: str = "STRICT_SPREAD",
    validate_topology: bool = True,
    timeout_s: float = 60.0,
    name: Optional[str] = None,
    lifetime: Optional[str] = None,
    expected_chips: Optional[int] = None,
) -> PlacementGroup:
    """
    Creates an explicitly node-pinned placement group across target contiguous TPU worker hosts.
    Strictly implements:
      - Stale placement group cleanup (CREATED / PENDING).
      - Node-pinned bundles: [{"TPU": 4.0, f"node:{host_ip}": 0.001} for host_ip in target_node_ips].
      - strategy="STRICT_SPREAD" to guarantee rank ordering and prevent autoscaler deadlocks.
      - Cloud TPU v5e TP=8 (2x4 ICI mesh, 8 chips) topology validation.
    """
    cleanup_stale_placement_groups()

    target_ips_list = list(target_node_ips)
    chips = expected_chips if expected_chips is not None else int(len(target_ips_list) * tpu_per_node)
    if validate_topology:
        validate_tpu_v5e_topology(
            target_node_ips=target_ips_list,
            tpu_per_node=tpu_per_node,
            expected_chips=chips,
        )

    bundles = [
        {"TPU": float(tpu_per_node), f"node:{host_ip}": 0.001}
        for host_ip in target_ips_list
    ]

    print(
        f"[Placement] Creating node-pinned placement group across {target_ips_list} "
        f"(strategy={strategy}, bundles={len(bundles)})...",
        flush=True,
    )
    pg_kwargs: Dict[str, Any] = {"strategy": strategy}
    if name is not None:
        pg_kwargs["name"] = name
    if lifetime is not None:
        pg_kwargs["lifetime"] = lifetime
    pg = placement_group(bundles, **pg_kwargs)
    ray.get(pg.ready(), timeout=timeout_s)

    print(
        f"[OMP_EVENT: PLACEMENT_GROUP_READY hosts={target_ips_list} "
        f"strategy={strategy} bundles={len(bundles)} chips={chips}]",
        flush=True,
    )
    return pg
