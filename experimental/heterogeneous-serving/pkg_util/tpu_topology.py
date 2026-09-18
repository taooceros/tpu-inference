"""
TPU Mesh Topology Discovery Shim.
=================================
Maintains backward compatibility with legacy scripts by re-exporting from
the first-class `admission_control.tpu.topology` module.
"""

from admission_control.tpu.topology import (
    TPUHostInfo,
    ConnectedPair,
    TPUTopologyDiscovery,
)
from admission_control.tpu.placement import (
    cleanup_stale_placement_groups,
    create_node_pinned_placement_group,
    validate_tp8_v5e_topology,
    remove_placement_group_safe,
)

__all__ = [
    "TPUHostInfo",
    "ConnectedPair",
    "TPUTopologyDiscovery",
    "cleanup_stale_placement_groups",
    "create_node_pinned_placement_group",
    "validate_tp8_v5e_topology",
    "remove_placement_group_safe",
]

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Discover connected TPU worker hosts on Ray")
    parser.add_argument("--json", action="store_true", help="Output JSON structure")
    args = parser.parse_args()

    discovery = TPUTopologyDiscovery()
    pair = discovery.select_connected_2host_submesh()

    if args.json:
        print(json.dumps(pair.to_dict(), indent=2))
    else:
        print(f"Selected Connected Host Pair (TP=8, Mesh: {pair.mesh_dimension}):")
        print(f"  Host A: Worker ID {pair.host_a.worker_id}, IP {pair.host_a.node_ip}, Coords {pair.host_a.host_coords}")
        print(f"  Host B: Worker ID {pair.host_b.worker_id}, IP {pair.host_b.node_ip}, Coords {pair.host_b.host_coords}")
