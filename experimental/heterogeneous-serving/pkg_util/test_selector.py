import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pkg_util.tpu_topology import TPUTopologyDiscovery, TPUHostInfo, ConnectedPair

class TPUSubsliceSelector:
    @staticmethod
    def get_slice_inventory():
        discovery = TPUTopologyDiscovery()
        return discovery.discover_tpu_nodes()

    @classmethod
    def select_connected_pair(cls, slice_name=None) -> ConnectedPair:
        discovery = TPUTopologyDiscovery()
        return discovery.select_connected_2host_submesh(slice_name=slice_name)

if __name__ == '__main__':
    pair = TPUSubsliceSelector.select_connected_pair()
    print(f'Selected Connected Pair: Worker {pair.host_a.worker_id} ({pair.host_a.node_ip}) + Worker {pair.host_b.worker_id} ({pair.host_b.node_ip})')
    print(f'Orientation: {pair.mesh_dimension}, Total TPUs: {pair.total_tpus}')
