#!/usr/bin/env python3
"""Native CPU continuations for the fixed two-host TP8 Ray service.

Capture real CPU imports/registrations and, for the API, the parsed CLI,
VllmConfig, renderer/tokenizer copies, InputProcessor and OutputProcessor.
No Ray connection, TPU client, listener, engine process or request loop exists.
The API resumes the same prepared AsyncLLM, not a second constructor.

Host geometry, CPU/model inputs and import environment are immutable. All
worker/job identities and generation sockets remain late-bound. Native
real-PID mode, real resource checks and authenticated proxy ownership apply.
"""
from contextlib import contextmanager
from multiprocessing.context import ForkServerProcess
import json
import os
import sys
from pathlib import Path

import pm_ray_zygote as workers

WORKER_LATE_ENV = ('PM_RAY_ZYGOTE_SOCK',)
_DRIVER_IDENTITY = ('TPU_WORKER_ID', 'VLLM_HOST_IP')
API_LATE_ENV = ('RAY_ADDRESS', 'TTL_RAY_RUNTIME_ENV', 'TTL_RAY_JOBID_FILE', 'PM_RAY_ZYGOTE_SOCK',
                *_DRIVER_IDENTITY)
OPTIONAL_LATE_ENV = ('TTL_RUN', 'TTL_RAY_WORKER_ENV')
OPTIONAL_API_LATE_ENV = ('PM_ZYGOTE_SOCK',)
_GENERATION_KEYS = (*API_LATE_ENV, *OPTIONAL_LATE_ENV, *OPTIONAL_API_LATE_ENV)
_FROZEN_KEYS = (*workers.CPU_ENV_KEYS, 'PATH', 'HOME', 'LANG', 'HF_HOME', 'VLLM_CACHE_ROOT',
                'XDG_CACHE_HOME', 'VLLM_LOGGING_CONFIG_PATH', 'VLLM_WORKER_MULTIPROC_METHOD',
                'VLLM_USE_RAY_V2_EXECUTOR_BACKEND', 'TTL_RAY_WORKER_UNSET',
                'PM_RAY_ZYGOTE_STRICT', 'PM_RAY_ZYGOTE_PRELOAD',
                'GLIBC_TUNABLES',
                'PM_REWIND_SERVE_ARGV', 'PM_WARM', 'PM_SB_PRESEED')
_GEOMETRY = {
    'TPU_MULTIHOST_BACKEND': 'ray',
    'TPU_ACCELERATOR_TYPE': 'v5litepod-8',
    'TPU_TOPOLOGY': '4x2',
    'TPU_HOST_BOUNDS': '2,1,1',
    'TPU_CHIPS_PER_HOST_BOUNDS': '2,2,1',
    'TORCH_TPU_TOPOLOGY': '4,2,1',
}
_PREPARED = None
_PREPARING_CPU = False
_GENERATION_ACTIVE = False


def preparing_cpu():
    return _PREPARING_CPU


def _require_glibc_rseq_disabled():
    import ctypes
    if os.getenv('GLIBC_TUNABLES') != 'glibc.pthread.rseq=0':
        raise RuntimeError('native CPU prefix requires startup GLIBC_TUNABLES=glibc.pthread.rseq=0')
    try:
        size = ctypes.c_uint.in_dll(ctypes.CDLL(None), '__rseq_size').value
    except ValueError as exc:
        raise RuntimeError('native CPU prefix requires the public glibc rseq ABI') from exc
    if size:
        raise RuntimeError('glibc rseq is enabled; setting GLIBC_TUNABLES after startup is insufficient')


@contextmanager
def _cpu_preparation():
    global _PREPARING_CPU
    if _PREPARING_CPU or _PREPARED is not None:
        raise RuntimeError('CPU preparation may run only once in a fresh interpreter')
    _require_glibc_rseq_disabled()
    _PREPARING_CPU = True
    try:
        yield
    finally:
        _PREPARING_CPU = False


def activate_generation():
    """Activate deferred real plugins only after this generation's environment exists."""
    global _GENERATION_ACTIVE
    if _PREPARING_CPU or _PREPARED is None or _GENERATION_ACTIVE:
        raise RuntimeError('generation activation requires one unused saved CPU prefix')
    if _PREPARED['role'] == 'workers':
        # Unlike the API driver, Ray workers cannot retain import-time JOB_ID.
        # The actual raylet environment is now installed, so real plugins/Ray
        # imports are safe; no worker identity or client was snapshotted.
        from vllm.plugins import load_general_plugins
        load_general_plugins()
    import ttl_ray_pin
    import pm_torchtpu
    ttl_ray_pin.register()
    workers.apply_prepared_environment(_PREPARED['state'])
    pm_torchtpu.register()
    _require_pinned_executor()
    workers.assert_cpu_boundary(identity=False, ray_free=False, loader_unloaded=True)
    _GENERATION_ACTIVE = True


def _remember(state, role, cpu_objects=None):
    import ctypes
    global _PREPARED
    if _PREPARED is not None:
        raise RuntimeError('a Ray factory was already prepared in this interpreter')
    native = ctypes.create_string_buffer(bytes.fromhex(state['nonce']))
    _PREPARED = {
        'nonce': state['nonce'], 'role': role, 'buffer': native,
        'capture_pid': workers.assert_real_pid(),
        'state': state, 'cpu_objects': cpu_objects or {},
        'saved_cpu_ids': {name: id(value) for name, value in (cpu_objects or {}).items()},
        'saved_module_ids': {name: address for name, _, address in state['modules']},
        'saved_native_address': ctypes.addressof(native), 'saved_native_bytes': native.raw.hex(),
    }
    proof = dict(get_prepared_state(), tpu_initialized=False,
                 ray_imported='ray' in sys.modules,
                 ray_initialized=bool(sys.modules.get('ray') and sys.modules['ray'].is_initialized()),
                 modules={name: hex(address) for name, _, address in state['modules']})
    print('PM_REWIND_RAY_CAPTURE ' + json.dumps(proof, separators=(',', ':')), flush=True)


def get_prepared_state():
    """Saved-state evidence, callable even after Ray/device bootstrap in an actor.

    A fresh import/exec has no witness and fails instead of manufacturing one.
    This does not assert post-bootstrap client state or inference correctness.
    """
    import ctypes
    if _PREPARED is None:
        raise RuntimeError('no saved Ray factory state in this interpreter')
    saved = _PREPARED
    return {
        'nonce': saved['nonce'], 'role': saved['role'],
        'capture_pid': saved['capture_pid'],
        'pid': os.getpid(), 'proc_pid': workers.assert_real_pid(),
        'proxy': workers.get_proxy_identity(),
        'saved_module_ids': dict(saved['saved_module_ids']),
        'current_module_ids': {name: id(sys.modules[name]) if name in sys.modules else None
                               for name in saved['saved_module_ids']},
        'saved_cpu_ids': dict(saved['saved_cpu_ids']),
        'current_cpu_ids': {name: id(value) for name, value in saved['cpu_objects'].items()},
        'cpu_stage': getattr(saved['cpu_objects'].get('engine'), '_pm_cpu_state', None),
        'generated_env': {key: value[1] for key, value in saved['state']['generated_env'].items()},
        'saved_native_address': saved['saved_native_address'],
        'native_address': ctypes.addressof(saved['buffer']),
        'saved_native_bytes': saved['saved_native_bytes'], 'native_bytes': saved['buffer'].raw.hex(),
    }


def _hosts():
    hosts = os.environ.get('TTL_RAY_HOSTS', '').split(',')
    if (len(hosts) != 2 or len(set(hosts)) != 2
            or any(not h or h != h.strip() for h in hosts)):
        raise ValueError('TTL_RAY_HOSTS must contain exactly two distinct hosts in rank order')
    if os.environ.get('TPU_WORKER_HOSTNAMES') != ','.join(hosts):
        raise ValueError('TPU_WORKER_HOSTNAMES must equal frozen TTL_RAY_HOSTS')
    for key, value in _GEOMETRY.items():
        if os.environ.get(key) != value:
            raise ValueError(f'TP8/two-host geometry requires {key}={value}')
    if os.environ.get('VLLM_USE_RAY_V2_EXECUTOR_BACKEND', '1') != '1':
        raise ValueError('TP8 rewind requires the pinned V2 Ray executor')
    return hosts


def _pin_environment():
    overrides = json.loads(os.environ.get('TTL_RAY_WORKER_ENV') or '{}')
    if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                for k, v in overrides.items()):
        raise ValueError('TTL_RAY_WORKER_ENV must be a string environment dictionary')
    for key, value in overrides.items():
        if key in workers.WORKER_IDENTITY_KEYS or key in (
                'RAY_ADDRESS', 'TTL_RAY_RUNTIME_ENV', 'TTL_RAY_JOBID_FILE',
                'TTL_RAY_WORKER_ENV', 'PM_ZYGOTE_SOCK'):
            raise ValueError(f'pin plugin cannot override driver/worker identity: {key}')
        if key in ('TTL_RUN', 'PM_RAY_ZYGOTE_SOCK') and value != os.environ.get(key):
            raise ValueError(f'pin plugin generation mismatch: {key}')
        if key in _FROZEN_KEYS and value != os.environ.get(key):
            raise ValueError(f'pin plugin would change preloaded environment: {key}')
    for key in filter(None, os.environ.get('TTL_RAY_WORKER_UNSET', '').split(',')):
        if key in _FROZEN_KEYS and key in os.environ and overrides.get(key) != os.environ[key]:
            raise ValueError(f'pin plugin would unset preloaded environment without restoring it: {key}')


def _prepare(role):
    workers.assert_cpu_boundary(loader_unloaded=True)
    if os.environ.get('PM_RAY_ZYGOTE_STRICT') != '1':
        raise ValueError('Ray factories require frozen PM_RAY_ZYGOTE_STRICT=1')
    present = [key for key in _GENERATION_KEYS if key in os.environ]
    if present:
        raise ValueError(f'generation keys must be absent until controller restore: {present}')
    _hosts()
    _pin_environment()
    plugins = sys.modules.get('vllm.plugins')
    if plugins is not None and plugins.__dict__.get('plugins_loaded'):
        raise RuntimeError('general plugins ran before controlled CPU preparation')
    state = workers.preload_cpu(role)
    # Name metadata only: both worker and API snapshots must decode torch.device('tpu').
    # SDK options/registration and real device/client initialization remain post-restore.
    workers.bind_tpu_device_name()
    if role == 'api':
        from vllm.plugins import load_general_plugins
        load_general_plugins()  # CPU registries run; job/socket hooks explicitly defer.
    workers.seal_preloaded(state)
    workers.check_preloaded(state)
    state['launcher_sha256'] = _launcher_digest(workers.__file__)
    frozen = {key: os.environ.get(key) for key in _FROZEN_KEYS}
    return state, frozen


def _resume_boundary(state, frozen, required, *, driver=False):
    # These checks precede any socket/fork/CLI import; the generic controller is
    # responsible for admitting only declared late keys into this environment.
    workers.apply_prepared_environment(state)
    workers.check_preloaded(state, allow_identity=_DRIVER_IDENTITY if driver else ())
    for key, value in frozen.items():
        if os.environ.get(key) != value:
            raise ValueError(f'frozen factory environment changed: {key}')
    for key in required:
        if not os.environ.get(key):
            raise ValueError(f'missing controller late binding: {key}')
    hosts = _hosts()
    if driver and (os.environ['TPU_WORKER_ID'] != '0' or os.environ['VLLM_HOST_IP'] != hosts[0]):
        raise ValueError('API identity must be freshly bound to host rank 0 / TTL_RAY_HOSTS[0]')
    _pin_environment()


def prepare_workers():
    """Save a childless CPU worker template; bind a new listener only on restore."""
    with _cpu_preparation():
        state, frozen = _prepare('workers')
        _remember(state, 'workers')

    def resume(argv):
        if list(argv) not in ([], ['serve']):
            raise ValueError('worker continuation accepts no arguments, or serve')
        _resume_boundary(state, frozen, WORKER_LATE_ENV)
        return workers.serve_main(preloaded=state)  # explicitly bypass all preload imports

    return resume


def _launcher_digest(path):
    """Admit the host script's copied launcher, but not arbitrary replacement code."""
    import hashlib
    import stat
    parent = os.path.dirname(path)
    directory = os.stat(parent)
    if (not os.path.isabs(path) or os.path.realpath(parent) != parent
            or directory.st_uid not in (0, os.getuid()) or directory.st_mode & 0o022):
        raise ValueError('Ray launcher must live in a trusted, non-writable directory')
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid())
                or info.st_mode & 0o022):
            raise ValueError('Ray launcher must be a trusted, non-writable regular file')
        return hashlib.file_digest(source, 'sha256').hexdigest()


def _runtime_env(launcher_sha256):
    import shlex
    runtime = json.loads(os.environ['TTL_RAY_RUNTIME_ENV'])
    if not isinstance(runtime, dict) or not isinstance(runtime.get('env_vars'), dict):
        raise ValueError('TTL_RAY_RUNTIME_ENV must contain env_vars and strict py_executable')
    env = runtime['env_vars']
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError('Ray runtime env_vars must be strings')
    if (env.get('PM_RAY_ZYGOTE_STRICT') != '1'
            or env.get('PM_RAY_ZYGOTE_SOCK') != os.environ['PM_RAY_ZYGOTE_SOCK']):
        raise ValueError('Ray workers must target this generation with strict proxy mode')
    if env.get('PM_ZYGOTE_SOCK'):
        raise ValueError('the local API forkserver socket must not propagate to Ray workers')
    if any(key in env for key in _DRIVER_IDENTITY):
        raise ValueError('Ray workers must derive host identity, not inherit the API host')
    if env.get('TTL_RUN') != os.environ.get('TTL_RUN'):
        raise ValueError('Ray workers must carry this generation TTL_RUN')
    if json.loads(env.get('TTL_RAY_WORKER_ENV') or '{}') != json.loads(
            os.environ.get('TTL_RAY_WORKER_ENV') or '{}'):
        raise ValueError('Ray workers must carry the same late pin-plugin environment')
    command = shlex.split(runtime.get('py_executable', ''))
    if (len(command) != 4 or command[1] != '-S' or command[3] != 'exec'
            or not os.path.isabs(command[0]) or not os.path.isabs(command[2])
            or os.path.realpath(command[0]) != os.path.realpath(sys.executable)):
        raise ValueError('py_executable must be "<same python> -S <pm_ray_zygote.py copy> exec"')
    # torchtpu_load_host.sh copies this file to $BASE; path identity is not the contract.
    # Compare with capture-time bytes, not with a potentially changed source file after restore.
    if _launcher_digest(command[2]) != launcher_sha256:
        raise ValueError('Ray launcher copy differs from the saved pm_ray_zygote.py')
    # Worker images have a different safe prefix. Carry the original frozen
    # inputs, not the API image's generated SDK defaults, across this boundary.
    original_env = _PREPARED['state']['original_env']
    for key in _FROZEN_KEYS:
        if key in env and env[key] != original_env.get(key):
            raise ValueError(f'Ray runtime overrides frozen environment: {key}')
    for key in ('TTL_RAY_HOSTS', 'TPU_WORKER_HOSTNAMES', *_GEOMETRY):
        if env.get(key) != os.environ[key]:
            raise ValueError(f'Ray runtime must explicitly carry frozen geometry: {key}')


def _require_pinned_executor():
    from vllm_torchtpu.executors.ray_distributed_executor_v2 import RayDistributedExecutorV2
    if not getattr(RayDistributedExecutorV2._initialize_ray_cluster, '_ttl', False):
        raise RuntimeError('ttl_ray_pin did not install the pinned TP8 Ray executor')


class _RayEngineProcess(ForkServerProcess):
    """Apply fresh API environment and register plugins in the engine child, not its CPU parent."""

    def start(self):
        self._pm_env = dict(os.environ)
        return super().start()

    def run(self):
        workers.install_child_environment(self._pm_env)
        workers.apply_prepared_environment(_PREPARED['state'])
        workers.assert_cpu_boundary(identity=False, ray_free=False)
        activate_generation()
        _install_api_child_bootstrap()
        workers.assert_cpu_boundary(identity=False, ray_free=False)
        return super().run()


def _install_api_child_bootstrap():
    if not os.environ.get('PM_ZYGOTE_SOCK'):
        return
    from multiprocessing import forkserver
    from pm_torchtpu import ZygoteContext
    if forkserver._forkserver._forkserver_address != os.environ['PM_ZYGOTE_SOCK']:
        raise RuntimeError('pm_torchtpu did not bind the fresh API forkserver client')
    # Existing plugin contexts look up Process on this class. Keep its protocol
    # and env propagation, but register child plugins only AFTER the helper forks.
    ZygoteContext.Process = _RayEngineProcess


@contextmanager
def _api_forkserver():
    """Optional local CPU helper, created AFTER restore and BEFORE CLI/plugins.

    The generic reaper owns leftover descendants. The API owns and reaps this
    immediate helper child; no PID saved in a checkpoint is used for shutdown.
    """
    if not os.environ.get('PM_ZYGOTE_SOCK'):
        yield
        return
    import select
    import signal
    import socket
    path = workers.private_socket_path('PM_ZYGOTE_SOCK')
    ready_r, ready_w = os.pipe()
    try:
        child = os.fork()
    except BaseException:
        os.close(ready_r)
        os.close(ready_w)
        raise
    if child == 0:
        os.close(ready_r)
        try:
            workers.assert_cpu_boundary(ray_free=False, allow_identity=_DRIVER_IDENTITY)
            from multiprocessing import forkserver
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(64)
            alive_r, alive_w = os.pipe()  # kept in this fresh helper, never checkpointed
            workers.assert_cpu_boundary(ray_free=False, allow_identity=_DRIVER_IDENTITY)
            os.write(ready_w, b'1')
            os.close(ready_w)
            forkserver.main(listener.detach(), alive_r, [])
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    os.close(ready_w)
    child_fd = None
    try:
        child_fd = workers.pidfd_open(child)  # own an unreaped child before any parent CLI/plugin work
        if not select.select([ready_r], [], [], 120)[0] or os.read(ready_r, 1) != b'1':
            raise RuntimeError('fresh API CPU forkserver failed before ready')
        yield
    finally:
        os.close(ready_r)
        try:
            if child_fd is None:
                os.kill(child, signal.SIGKILL)  # pidfd_open failed before yielding to any other reaper
            else:
                workers.pidfd_send_signal(child_fd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass
        finally:
            if child_fd is not None:
                os.close(child_fd)
        if os.path.lexists(path):
            os.unlink(path)


def _model_inputs(args):
    root = Path(args.model).absolute()
    if root != root.resolve() or not root.is_dir() or args.trust_remote_code:
        raise ValueError('CPU preparation requires the immutable local model, without remote code')
    if args.tokenizer not in (None, args.model):
        raise ValueError('CPU preparation requires the model-local tokenizer')
    paths = sorted(p for p in root.iterdir() if p.suffix in ('.json', '.model', '.txt', '.jinja'))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'non-regular CPU model input: {path}')
    if not {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
            'processor_config.json', 'chat_template.jinja'}.issubset(p.name for p in paths):
        raise ValueError('incomplete fixed Gemma4 CPU metadata')
    if sum(p.stat().st_size for p in paths) > 64 * 1024 * 1024:
        raise ValueError('CPU metadata exceeds the fixed input bound')
    return tuple(str(p) for p in paths)


def prepare_api():
    """Prepare the real API's CPU state; resume only its external-resource suffix."""
    with _cpu_preparation():
        state, _ = _prepare('api')
        argv = json.loads(os.environ['PM_REWIND_SERVE_ARGV'])
        if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv) or argv[:1] != ['serve']:
            raise ValueError('PM_REWIND_SERVE_ARGV must be the immutable real serve argv')
        import vllm.envs as envs
        from vllm.entrypoints.cli.serve import ServeSubcommand
        from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.usage.usage_lib import UsageContext
        from vllm.v1.engine.async_llm import AsyncLLM
        cli_env_setup()
        parser = FlexibleArgumentParser(prog='vllm')
        serve = ServeSubcommand()
        serve.subparser_init(parser.add_subparsers(dest='subparser', required=True))
        args = parser.parse_args(argv)
        serve.validate(args)
        if (args.distributed_executor_backend != 'ray' or args.tensor_parallel_size != 8
                or args.pipeline_parallel_size != 1 or args.data_parallel_size != 1
                or args.prefill_context_parallel_size != 1 or args.nnodes != 1 or args.node_rank != 0
                or args.data_parallel_backend != 'mp' or args.data_parallel_size_local not in (None, 1)
                or args.data_parallel_rank is not None or args.data_parallel_start_rank is not None
                or args.data_parallel_external_lb or args.data_parallel_hybrid_lb
                or args.data_parallel_multi_port_external_lb or args.enable_elastic_ep
                or args.headless or getattr(args, 'grpc', False) or args.api_server_count not in (None, 1)
                or getattr(args, 'decode_context_parallel_size', 1) != 1 or envs.VLLM_USE_RUST_FRONTEND):
            raise ValueError('rewind requires one HTTP API, TP8, Ray V2, PP1/DP1/PCP1 on two hosts')
        if args.model_tag is not None:
            args.model = args.model_tag
        model_inputs = _model_inputs(args)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        config = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
        workers.assert_cpu_boundary(ray_free=False, loader_unloaded=True)
        engine = AsyncLLM.from_vllm_config(
            config, usage_context=UsageContext.OPENAI_API_SERVER,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats, prepare_cpu_only=True)
        from vllm.multimodal import MULTIMODAL_REGISTRY
        from vllm.renderers.hf import resolve_chat_template_content_format
        resolve_chat_template_content_format(
            None, None, 'auto', engine.renderer.tokenizer, model_config=config.model_config)
        MULTIMODAL_REGISTRY.supports_multimodal_inputs(config.model_config)
        for pool in (engine.renderer._executor, engine.renderer._mm_executor):
            if pool._threads:
                raise RuntimeError('renderer work was submitted before the CPU checkpoint')
        workers.seal_preloaded(state)
        workers.check_preloaded(state)
        frozen = {key: os.environ.get(key) for key in _FROZEN_KEYS}
        args._pm_rewind_cpu = (engine_args, engine)
        objects = {'args': args, 'engine_args': engine_args, 'config': config,
                   'hf_config': config.model_config.hf_config, 'engine': engine,
                   'renderer': engine.renderer, 'tokenizer': engine.renderer.tokenizer,
                   'input_processor': engine.input_processor, 'output_processor': engine.output_processor}
        _remember(state, 'api', objects)

    def resume(actual_argv):
        _resume_boundary(state, frozen, API_LATE_ENV, driver=True)
        help_only = list(actual_argv) == ['serve', '--help']
        if not help_only and list(actual_argv) != argv:
            raise ValueError('CPU-prepared API arguments changed; prepare a new image')
        if _model_inputs(args) != model_inputs:
            raise ValueError('CPU model input file set changed after preparation')
        proof = get_prepared_state()
        if proof['saved_cpu_ids'] != proof['current_cpu_ids'] or proof['cpu_stage'] != 'prepared':
            raise RuntimeError('native API CPU objects were replaced or already activated')
        print('PM_REWIND_CPU_REUSE ' + json.dumps(proof, separators=(',', ':')), flush=True)
        _runtime_env(state['launcher_sha256'])
        sys.argv = ['vllm', *argv]
        with _api_forkserver():
            if help_only:
                parser.parse_args(actual_argv)  # actual help, without rebuilding CPU objects
                return 0
            activate_generation()
            _install_api_child_bootstrap()
            return serve.cmd(args)  # SAME namespace/config/AsyncLLM; no second parse or constructor

    resume.rewind_input_files = model_inputs
    return resume
