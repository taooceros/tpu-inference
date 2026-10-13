"""Standalone-prototype executor attaching to persistent worker actors."""
from concurrent.futures import Future
from vllm.v1.executor.abstract import Executor


class ResidentExecutor(Executor):
    def _init_executor(self):
        import ray
        self.workers = [ray.get_actor(name, namespace='ici-resident-prototype')
                        for name in self.vllm_config.additional_config['ici_workers']]

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
        import ray
        result = ray.get([actor.rpc.remote(method, args, kwargs or {}, index)
                         for index, actor in enumerate(self.workers)], timeout=timeout or 900)
        if non_block:
            future = Future()
            future.set_result(result)
            return future
        return result

    def initialize_from_config(self, kv_cache_configs):
        import ray
        ray.get([actor.rpc.remote('initialize_from_config', (config,), {}, index)
                 for index, (actor, config) in enumerate(zip(self.workers, kv_cache_configs))], timeout=900)
        self.warmup = []
        if not self.vllm_config.additional_config.get('ici_defer_warmup', False):
            self.start_warmup()
            self.wait_for_warmup()

    def start_warmup(self):
        self.warmup = [actor.rpc.remote('compile_or_warm_up_model', (), {}, index)
                       for index, actor in enumerate(self.workers)]

    def wait_for_warmup(self):
        import ray
        ray.get(self.warmup, timeout=900)
        self.warmup = []

    def _model_result(self, method, args, non_block):
        output = self.collective_rpc(method, args=args)[0]
        if non_block:
            future = Future()
            future.set_result(output)
            return future
        return output

    def execute_model(self, scheduler_output, non_block=False):
        return self._model_result('execute_model', (scheduler_output,), non_block)

    def sample_tokens(self, grammar_output, non_block=False):
        self.sample_tokens = self._sample_tokens
        return self._model_result('first_sample_tokens', (grammar_output,), non_block)

    def _sample_tokens(self, grammar_output, non_block=False):
        return self._model_result('sample_tokens', (grammar_output,), non_block)
    def check_health(self):
        import ray
        reports = ray.get([actor.report.remote() for actor in self.workers], timeout=30)
        failed = [row for row in reports if row['failure'] is not None or row['layout'] is None]
        if failed:
            raise RuntimeError(f'resident controller unhealthy: {failed}')


    def shutdown(self):
        # Actor ownership belongs to the transition driver, not the engine core.
        self.workers = []
