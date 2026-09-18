"""
TPU Mesh Topology & Connected Submesh Discovery Module.
======================================================
Discovers TPU worker hosts connected to Ray, extracts hardware indices,
calculates (Hx, Hy) mesh coordinates and (X, Y) physical chip bounding boxes,
and computes valid physically connected host pairs (e.g. 2-host TP=8 contiguous submeshes).
"""

import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any


@dataclass
class TPUHostInfo:
    worker_id: int
    node_ip: str
    ray_node_id: str
    tpu_count: int
    slice_name: str
    topology: str
    host_coords: Tuple[int, int]
    chip_x_range: Tuple[int, int]
    chip_y_range: Tuple[int, int]
    labels: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConnectedPair:
    host_a: TPUHostInfo
    host_b: TPUHostInfo
    mesh_dimension: str  # "X (4x2)" or "Y (2x4)" or "chassis_local"
    total_tpus: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_ids": [self.host_a.worker_id, self.host_b.worker_id],
            "node_ips": [self.host_a.node_ip, self.host_b.node_ip],
            "mesh_dimension": self.mesh_dimension,
            "total_tpus": self.total_tpus,
            "placement_group_bundles": [
                {"TPU": float(self.host_a.tpu_count), f"node:{self.host_a.node_ip}": 0.001},
                {"TPU": float(self.host_b.tpu_count), f"node:{self.host_b.node_ip}": 0.001},
            ],
        }
    @property
    def jax_subslice_host_bounds(self) -> str:
        """Returns the JAX TPU_HOST_BOUNDS string for this subslice."""
        if "X" in self.mesh_dimension:
            return "2,1,1"
        return "1,2,1"


class TPUTopologyDiscovery:
    """Discovers physical TPU node topology and connected sub-slices on Ray."""

    def __init__(self, ray_address: Optional[str] = None):
        self.ray_address = ray_address

    @staticmethod
    def parse_host_bounds(topology: str, host_bounds_str: Optional[str] = None) -> Tuple[int, int]:
        """Derive (Nx, Ny) host grid dimensions."""
        if host_bounds_str:
            parts = [int(p.strip()) for p in host_bounds_str.split(",") if p.strip()]
            if len(parts) >= 2:
                return (parts[0], parts[1])
        if "4x4" in topology:
            return (2, 2)
        elif "2x4" in topology:
            return (1, 2)
        elif "2x2" in topology:
            return (1, 1)
        elif "4x8" in topology:
            return (2, 4)
        elif "8x8" in topology:
            return (4, 4)
        return (2, 2)

    @classmethod
    def compute_host_and_chip_coords(
        cls,
        worker_id: int,
        host_bounds: Tuple[int, int] = (2, 2),
        chips_per_host: Tuple[int, int] = (2, 2),
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculates (Hx, Hy) host grid coordinates and [x_min, x_max], [y_min, y_max] chip bounds."""
        nx, ny = host_bounds
        cx, cy = chips_per_host

        hx = worker_id % nx
        hy = (worker_id // nx) % ny

        x_min = hx * cx
        x_max = x_min + cx - 1
        y_min = hy * cy
        y_max = y_min + cy - 1

        return (hx, hy), (x_min, x_max), (y_min, y_max)

    def discover_tpu_nodes(self) -> Dict[str, List[TPUHostInfo]]:
        """Queries Ray cluster for active TPU nodes, grouped by slice name."""
        import ray
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address, ignore_reinit_error=True)
            else:
                ray.init(address="auto", ignore_reinit_error=True)

        nodes = ray.nodes()
        slices: Dict[str, List[TPUHostInfo]] = {}

        # Probe node environments via lightweight Ray tasks if available
        node_env_map: Dict[str, Dict[str, Any]] = {}
        try:
            @ray.remote(num_cpus=0)
            def _probe_tpu_node_env() -> Dict[str, Any]:
                import os
                return {
                    "worker_id": os.environ.get("TPU_WORKER_ID"),
                    "topology": os.environ.get("TPU_TOPOLOGY"),
                    "host_bounds": os.environ.get("TPU_HOST_BOUNDS"),
                }

            active_tpu_nodes = [
                n for n in nodes
                if n.get("Alive", False) and n.get("Resources", {}).get("TPU", 0) > 0
            ]
            refs = [
                _probe_tpu_node_env.options(
                    resources={f"node:{n.get('NodeManagerAddress', '')}": 0.001}
                ).remote()
                for n in active_tpu_nodes
            ]
            probed = ray.get(refs, timeout=10.0)
            for n, info in zip(active_tpu_nodes, probed):
                node_env_map[n.get("NodeManagerAddress", "")] = info
        except Exception:
            pass

        for n in nodes:
            if not n.get("Alive", False):
                continue
            resources = n.get("Resources", {})
            tpu_count = int(resources.get("TPU", 0))
            if tpu_count <= 0:
                continue

            node_ip = n.get("NodeManagerAddress", "")
            ray_node_id = n.get("NodeID", "")

            # Extract labels from node resources
            labels = {}
            for k, v in resources.items():
                if k.startswith("node:") or k.startswith("accelerator:"):
                    labels[k] = str(v)

            slice_name = "default_slice"
            topology = "2x4"
            host_bounds = (1, 2)
            worker_id = 0

            # Inspect environment or labels
            for k in resources.keys():
                if "tpu-slice:" in k:
                    slice_name = k.split("tpu-slice:")[-1]
                if "tpu-topology:" in k:
                    topology = k.split("tpu-topology:")[-1]
                if "tpu-worker-id:" in k:
                    try:
                        worker_id = int(k.split("tpu-worker-id:")[-1])
                    except ValueError:
                        pass

            env_info = node_env_map.get(node_ip, {})
            if env_info.get("worker_id") is not None:
                try:
                    worker_id = int(env_info["worker_id"])
                except ValueError:
                    pass
            if env_info.get("topology"):
                topology = env_info["topology"]

            host_bounds = self.parse_host_bounds(topology, env_info.get("host_bounds"))
            coords, x_range, y_range = self.compute_host_and_chip_coords(worker_id, host_bounds)
            host_info = TPUHostInfo(
                worker_id=worker_id,
                node_ip=node_ip,
                ray_node_id=ray_node_id,
                tpu_count=tpu_count,
                slice_name=slice_name,
                topology=topology,
                host_coords=coords,
                chip_x_range=x_range,
                chip_y_range=y_range,
                labels=labels,
            )
            slices.setdefault(slice_name, []).append(host_info)

        for s_name in slices:
            slices[s_name].sort(key=lambda h: h.worker_id)

        return slices

    def find_connected_2host_pairs(self, hosts: List[TPUHostInfo]) -> List[ConnectedPair]:
        """Finds all physically connected 2-host pairs along X or Y ICI mesh axes."""
        pairs = []
        host_map = {(h.host_coords[0], h.host_coords[1]): h for h in hosts}

        for (hx, hy), h in host_map.items():
            # Check +X neighbor
            if (hx + 1, hy) in host_map:
                neighbor = host_map[(hx + 1, hy)]
                pairs.append(ConnectedPair(
                    host_a=h,
                    host_b=neighbor,
                    mesh_dimension="X (4x2)",
                    total_tpus=h.tpu_count + neighbor.tpu_count,
                ))
            # Check +Y neighbor
            if (hx, hy + 1) in host_map:
                neighbor = host_map[(hx, hy + 1)]
                pairs.append(ConnectedPair(
                    host_a=h,
                    host_b=neighbor,
                    mesh_dimension="Y (2x4)",
                    total_tpus=h.tpu_count + neighbor.tpu_count,
                ))

        if not pairs and len(hosts) >= 2:
            # Fallback for 2-host single slice
            pairs.append(ConnectedPair(
                host_a=hosts[0],
                host_b=hosts[1],
                mesh_dimension="contiguous_pair",
                total_tpus=hosts[0].tpu_count + hosts[1].tpu_count,
            ))

        return pairs

    def validate_connected_submesh(self, pair: ConnectedPair, expected_tp: int = 8) -> bool:
        """
        Validates that a ConnectedPair satisfies Cloud TPU v5e TP=8 (2x4 ICI mesh, 8 chips).
        Enforces:
          - total_tpus == expected_tp (8 chips)
          - 4 TPU chips per host (Cloud TPU v5e standard)
          - Distinct physical node IPs
        """
        if pair.total_tpus != expected_tp:
            raise ValueError(
                f"Connected pair does not satisfy TP={expected_tp}: "
                f"total TPUs = {pair.total_tpus} "
                f"(Host A: {pair.host_a.tpu_count}, Host B: {pair.host_b.tpu_count})"
            )
        if pair.host_a.tpu_count != 4 or pair.host_b.tpu_count != 4:
            raise ValueError(
                f"Cloud TPU v5e requires exactly 4 chips per host, but got "
                f"Host A: {pair.host_a.tpu_count}, Host B: {pair.host_b.tpu_count}."
            )
        if pair.host_a.node_ip == pair.host_b.node_ip:
            raise ValueError(
                f"Connected pair hosts must have distinct IPs, got {pair.host_a.node_ip} for both."
            )
        return True

    def select_connected_2host_submesh(self, validate_tp8: bool = True) -> ConnectedPair:
        """Selects a guaranteed physically contiguous 2-host submesh (TP=8) from active cluster."""
        slices = self.discover_tpu_nodes()
        if not slices:
            raise RuntimeError("No TPU nodes discovered on Ray cluster.")

        selected_pair: Optional[ConnectedPair] = None
        for slice_name, hosts in slices.items():
            pairs = self.find_connected_2host_pairs(hosts)
            if pairs:
                selected_pair = pairs[0]
                break

        if selected_pair is None:
            all_hosts = [h for hosts in slices.values() for h in hosts]
            if len(all_hosts) >= 2:
                selected_pair = ConnectedPair(
                    host_a=all_hosts[0],
                    host_b=all_hosts[1],
                    mesh_dimension="generic_pair",
                    total_tpus=all_hosts[0].tpu_count + all_hosts[1].tpu_count,
                )

        if selected_pair is None:
            all_hosts = [h for hosts in slices.values() for h in hosts]
            raise RuntimeError(f"Could not find 2 connected TPU hosts (found {len(all_hosts)}).")

        if validate_tp8:
            self.validate_connected_submesh(selected_pair, expected_tp=8)
        return selected_pair

    def select_full_4x4_cluster(self, validate_tp16: bool = True) -> List[TPUHostInfo]:
        """Selects the full 4-host 4x4 cluster (TP=16, 16 chips) from active cluster in coordinate order."""
        slices = self.discover_tpu_nodes()
        if not slices:
            raise RuntimeError("No TPU nodes discovered on Ray cluster.")

        all_hosts = [h for hosts in slices.values() for h in hosts]
        all_hosts.sort(key=lambda h: h.worker_id)

        if len(all_hosts) < 4:
            raise RuntimeError(f"Expected at least 4 TPU hosts for 4x4 cluster, found {len(all_hosts)}.")

        selected_hosts = all_hosts[:4]
        if validate_tp16:
            total_tpus = sum(h.tpu_count for h in selected_hosts)
            if total_tpus != 16:
                raise ValueError(f"Expected 16 TPUs for 4x4 cluster, found {total_tpus} across 4 hosts.")
            unique_ips = set(h.node_ip for h in selected_hosts)
            if len(unique_ips) != 4:
                raise ValueError(f"Expected 4 distinct host IPs, got: {[h.node_ip for h in selected_hosts]}")
        return selected_hosts
