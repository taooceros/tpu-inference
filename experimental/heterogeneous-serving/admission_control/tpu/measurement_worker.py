"""TPU worker with process ranks local to the explicitly selected audit mesh."""
import os

import ray
from tpu_inference.worker.tpu_worker import TPUWorker


class MeasurementTPUWorker(TPUWorker):
    def init_device(self, *args, **kwargs):
        hosts = [address.rsplit(':', 1)[0] for address in os.environ['TPU_PROCESS_ADDRESSES'].split(',')]
        local_ip = ray.util.get_node_ip_address()
        if local_ip not in hosts or len(set(hosts)) != len(hosts):
            raise RuntimeError(f'Worker host {local_ip} is not uniquely present in pinned host order {hosts}')
        process_rank = hosts.index(local_ip)
        if self.rank == 0 and process_rank != 0:
            raise RuntimeError('Executor rank zero does not match the mesh coordinator')
        self.physical_tpu_worker_id = os.environ.get('TPU_WORKER_ID')
        os.environ['CLOUD_TPU_TASK_ID'] = str(process_rank)
        os.environ['TPU_WORKER_ID'] = str(process_rank)
        return super().init_device(*args, **kwargs)
