#!/usr/bin/env python3
"""Qualify a native rewind with a private, real local Ray cluster (CPU only).

Example (the output directory must not exist):
  python scripts/pm_rewind_ray_probe.py --python /path/to/runtime/python \
    --dmtcp-bin /path/to/real-pid-dmtcp/bin --allocator /path/to/libtcmalloc.so \
    --out-dir /dev/shm/pm-ray-proof

--timeout is the per-phase startup/RPC/capture deadline, not an inference timeout.
This never uses a shared Ray address, ray stop, pkill, or TPU/device operations.
Images and runtime artifacts are private and retained, including on failure.
"""
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import select
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback


# Written only at probe execution, inside the newly created private output dir.
# This is a real worker entrypoint through the real proxy, not a protocol server.
HELPER = r'''import json, os, signal, subprocess, sys, time
from pathlib import Path

def identity():
    raw = Path('/proc/self/stat').read_text()
    fields = raw.rpartition(')')[2].split()
    return {'pid': int(raw.split(' ', 1)[0]), 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'raw_stat': raw.strip()}

def dump(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value))
    pending.replace(path)

mode, base = sys.argv[1], Path(sys.argv[2])
if mode == 'leaf':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    dump(str(base) + '.ready.json', {'identity': identity()})
    os.kill(os.getpid(), signal.SIGSTOP)
    while True:
        time.sleep(1)
else:
    Path(str(base) + '.entered').touch()
    def terminated(signum, frame):
        dump(str(base) + '.signal.json', {'signal': signum, 'identity': identity()})
        raise SystemExit(73)
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    signal.signal(signal.SIGHUP, terminated)
    result = {'identity': identity(), 'run_id': os.environ.get('TTL_RUN'),
              'socket': os.environ.get('PM_RAY_ZYGOTE_SOCK')}
    if mode != 'guard':
        from pm_rewind_ray import get_prepared_state
        from pm_ray_zygote import assert_cpu_boundary
        # This helper runs after fresh-env plugin activation, not at capture.
        assert_cpu_boundary(identity=False, ray_free=False, loader_unloaded=True)
        result['cpu_boundary_checked'] = True
        result['prepared'] = get_prepared_state()
    if mode == 'tree':
        leafbase = str(base) + '.leaf'
        child = subprocess.Popen([sys.executable, __file__, 'leaf', leafbase],
                                 start_new_session=True)
        deadline = time.monotonic() + 15
        ready = Path(leafbase + '.ready.json')
        while not ready.exists():
            if child.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(
                    f'owned leaf did not start: pid={child.pid} poll={child.poll()} '
                    f'returncode={child.returncode} wait_elapsed_s={time.monotonic() - (deadline - 15):.3f} '
                    f'ready={ready}')
            time.sleep(.02)
        result['leaf'] = json.loads(ready.read_text())['identity']
    dump(str(base) + '.ready.json', result)
    while not Path(str(base) + '.release').exists():
        time.sleep(.02)
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    pending.chmod(0o600)
    pending.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def identity(pid='self'):
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw.rpartition(')')[2].split()
    return {'pid': int(raw.split(' ', 1)[0]), 'ppid': int(fields[1]),
            'start_ticks': int(fields[19]), 'state': fields[0], 'raw_stat': raw.strip()}


def same_process(who):
    try:
        return identity(who['pid'])['start_ticks'] == int(who['start_ticks'])
    except (FileNotFoundError, ProcessLookupError):
        return False


def until(predicate, timeout, label, process=None):
    deadline = time.monotonic() + timeout
    while True:
        result = predicate()
        if result:
            return result
        if process is not None and process.poll() is not None:
            # A last atomic receipt may have been written immediately before exit.
            result = predicate()
            if result:
                return result
            raise RuntimeError(f'{label}: owned process exited rc={process.returncode}')
        if time.monotonic() >= deadline:
            raise TimeoutError(label)
        time.sleep(.03)


def gone(who, timeout=15, reap_adopted=False):
    def absent():
        try:
            current = identity(who['pid'])
        except (FileNotFoundError, ProcessLookupError):
            return True
        if current['start_ticks'] != int(who['start_ticks']):
            return True
        if reap_adopted and current['ppid'] == os.getpid() and current['state'] == 'Z':
            # A killed controller's completed reaper is adopted here. Its PID
            # remains pinned as our direct zombie until this single waiter reaps.
            os.waitpid(who['pid'], os.WNOHANG)
            return not same_process(who)
        return False
    until(absent, timeout, f'owned process leaked: {who}')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_sources(root):
    source_dir = Path(__file__).resolve().parent
    staged = root / 'src'
    staged.mkdir(mode=0o700)
    receipts = []
    sources = {name: name for name in
               ('pm_rewind.py', 'pm_rewind_ray.py', 'pm_ray_zygote.py', 'pm_rewind_ray_probe.py', 'pm_zygote.py')}
    sources.update({'pm_torchtpu.py': 'pm_torchtpu_plugin.py', 'ttl_probe.py': 'torchtpu_load_plugin.py',
                    'ttl_ray_pin.py': 'torchtpu_ray_pin_plugin.py'})
    for name, original in sources.items():
        source = source_dir / original
        require(source.is_file(), f'unresolved interface: missing first-party source {source}')
        data = source.read_bytes()
        source_sha = hashlib.sha256(data).hexdigest()
        target = staged / name
        with target.open('xb') as stream:
            stream.write(data)
        target.chmod(0o600)
        require(target.read_bytes() == data and digest(source) == source_sha,
                f'first-party source changed during private staging: {source}')
        receipts.append({'source': str(source), 'copy': str(target),
                         'source_sha256': source_sha, 'copy_sha256': digest(target), 'mode': '0600'})
    # Local qualification uses the same real plugin callbacks as the target
    # private venv, without installing anything into a shared Python runtime.
    from importlib.metadata import entry_points
    installed = {entry.name for entry in entry_points(group='vllm.general_plugins')}
    missing = sorted({'pm_torchtpu', 'ttl_probe', 'ttl_ray_pin'} - installed)
    if missing:
        metadata = staged / 'pm_native_probe-0.0.0.dist-info'
        metadata.mkdir(mode=0o700)
        (metadata / 'METADATA').write_text('Metadata-Version: 2.1\nName: pm_native_probe\nVersion: 0.0.0\n')
        (metadata / 'entry_points.txt').write_text(
            '[vllm.general_plugins]\n' + ''.join(f'{name} = {name}:register\n' for name in missing))
    require(stat.S_IMODE(staged.stat().st_mode) == 0o700, 'private source directory is not mode 0700')
    return staged, receipts


def pidfd_preflight():
    from pm_ray_zygote import pidfd_open, pidfd_send_signal
    who = identity()
    require(who['pid'] == os.getpid(), 'probe owner is not using actual kernel PIDs')
    fd = pidfd_open(who['pid'])
    try:
        require(os.readlink(f'/proc/self/fd/{fd}') == 'anon_inode:[pidfd]',
                'pidfd compatibility API returned a non-pidfd')
        info = Path(f'/proc/self/fdinfo/{fd}').read_text()
        fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
        require(int(fields['Pid']) == who['pid'], 'self pidfd identity mismatch')
        pidfd_send_signal(fd, 0)  # Existence/permission check; delivers no signal.
        return {'identity': who, 'fdinfo': info.strip(), 'signal_zero_succeeded': True,
                'api': 'pm_ray_zygote.pidfd_open/pidfd_send_signal',
                'stdlib_open_available': hasattr(os, 'pidfd_open'),
                'stdlib_send_available': hasattr(signal, 'pidfd_send_signal')}
    finally:
        os.close(fd)


def versions():
    packages = {}
    for name in ('ray', 'torch', 'torch-tpu', 'jax', 'jaxlib', 'libtpu', 'vllm', 'vllm-tpu'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'packages': packages}


def check_native(proof, baseline):
    require(proof['role'] == 'workers', 'not a worker factory witness')
    for key in ('nonce', 'role', 'capture_pid', 'saved_module_ids',
                'saved_native_address', 'saved_native_bytes'):
        require(proof[key] == baseline[key], f'native snapshot changed: {key}')
    require(proof['current_module_ids'] == baseline['saved_module_ids'], 'module identity lost')
    require(proof['native_address'] == baseline['saved_native_address'], 'native address lost')
    require(proof['native_bytes'] == baseline['saved_native_bytes'], 'native contents lost')
    require(proof['pid'] == proof['proc_pid'], 'virtual PID reached a real worker')
    require(proof['proc_pid'] != baseline['capture_pid'], 'original capture reused as worker')
    proxy = proof.get('proxy')
    require(isinstance(proxy, dict) and type(proxy.get('pid')) is int and proxy['pid'] > 0
            and type(proxy.get('start_ticks')) is int and proxy['start_ticks'] > 0,
            'native worker lacks an authenticated proxy identity')
    require(proxy['pid'] not in (proof['proc_pid'], baseline['capture_pid']),
            'proxy identity aliases the actual worker or original capture')


def generation_env(cfg, socket, run_id):
    pin = {key: value for key, value in cfg['frozen'].items()
           if key.startswith(('TPU_', 'TORCH_TPU_'))}
    return {'PM_RAY_ZYGOTE_SOCK': str(socket), 'TTL_RUN': run_id,
            'TTL_RAY_WORKER_ENV': json.dumps(pin, sort_keys=True, separators=(',', ':'))}


def _cluster(config_path):
    """Exec-created driver. Its only Ray node is created by address='local'."""
    cfg = load(config_path)
    root = Path(cfg['root'])
    state = {'success': False, 'versions': versions(), 'phase': 'import'}
    dump(root / 'ray-driver.json', state)
    ray = None
    actors = {}
    try:
        import ray
        state['versions']['ray_loaded'] = ray.__version__
        state['versions']['ray_commit'] = getattr(ray, '__commit__', None)
        require(not ray.is_initialized(), 'driver unexpectedly already connected to Ray')
        state['phase'] = 'ray.init'
        dump(root / 'ray-driver.json', state)
        from ray._private.accelerators import get_all_accelerator_resource_names
        # num_gpus=0 alone does not suppress Ray's TPU autodetection. Explicit
        # zero resources bypass accelerator-count/client discovery in Ray 2.56.
        accelerator_names = get_all_accelerator_resource_names()
        disabled = {name: 0 for name in accelerator_names if name != 'GPU'}
        init_args = dict(address='local', num_cpus=2, num_gpus=0, resources=disabled,
                         include_dashboard=False, dashboard_host='127.0.0.1', dashboard_port=0,
                         _node_ip_address='127.0.0.1', _temp_dir=cfg['ray_temp'],
                         object_store_memory=80 * 1024 * 1024,
                         _plasma_directory=cfg['temp'],
                         namespace=cfg['namespace'], log_to_driver=False)
        context = ray.init(**init_args)
        node = ray._private.worker._global_node
        require(node is not None and node.head, 'Ray did not create an owned head node')
        session = Path(node.get_session_dir_path()).resolve()
        require(session.is_relative_to(Path(cfg['ray_temp']).resolve()), 'Ray session escaped private root')
        dashboard = str(context.address_info.get('webui_url') or '')
        require(':8265' not in dashboard, 'unexpected dashboard port 8265')
        cluster_resources = ray.cluster_resources()
        require(all(cluster_resources.get(name, 0) == 0 for name in accelerator_names),
                'local Ray cluster exposed accelerator resources')
        state.update(phase='ready', init_args=init_args, session=str(session),
                     address=context.address_info.get('gcs_address'), dashboard_url=dashboard,
                     driver_identity=identity(), resources=cluster_resources)
        dump(root / 'ray-driver.json', state)
        dump(root / 'ray-ready.json', state)

        # Defined here so Ray cloudpickles the actual actor by value. No factory
        # import in the driver, and no mock actor or fallback interpreter path.
        @ray.remote(max_restarts=0, max_task_retries=0, num_cpus=1)
        class NativeActor:
            def __init__(self):
                self.calls = 0

            def proof(self, challenge):
                import ctypes
                import os
                from pathlib import Path
                import sys
                import torch
                import ray
                from pm_rewind_ray import get_prepared_state
                self.calls += 1
                prepared = get_prepared_state()  # Plain exec MUST fail here.
                raw = Path('/proc/self/stat').read_text()
                fields = raw.rpartition(')')[2].split()
                proc_pid = int(raw.split(' ', 1)[0])
                if proc_pid != os.getpid() or proc_pid != prepared['proc_pid']:
                    raise RuntimeError('kernel/getpid/prepared PID mismatch')
                tensor = torch.arange(16, dtype=torch.float64, device='cpu').reshape(4, 4)
                product = tensor @ torch.eye(4, dtype=torch.float64, device='cpu')
                total = product.sum().item()
                squared = (tensor * tensor).sum().item()
                device_state = {}
                for name in ('torch.tpu', 'torch.cuda', 'torch.distributed'):
                    module = sys.modules.get(name)
                    initialized_predicate = getattr(module, 'is_initialized', None)
                    if not callable(initialized_predicate):
                        raise RuntimeError(f'missing real initialization predicate: {name}')
                    initialized = bool(initialized_predicate())
                    status = {'initialized': initialized}
                    if name != 'torch.distributed':
                        bad_fork_predicate = getattr(module, '_is_in_bad_fork', None)
                        if not callable(bad_fork_predicate):
                            raise RuntimeError(f'missing real fork-state predicate: {name}')
                        status['bad_fork'] = bool(bad_fork_predicate())
                    device_state[name] = status
                    if any(status.values()):
                        raise RuntimeError(f'not a CPU-only actor: {name}')
                for name in ('jax._src.xla_bridge', 'jax.lib.xla_bridge'):
                    bridge = sys.modules.get(name)
                    if bridge and (bridge.__dict__.get('_backends') or
                                   bridge.__dict__.get('_default_backend') is not None):
                        raise RuntimeError('unexpected JAX/PJRT client')
                runtime = ray.get_runtime_context()
                return {'prepared': prepared, 'identity': {'pid': proc_pid, 'ppid': int(fields[1]),
                        'start_ticks': int(fields[19]), 'raw_stat': raw.strip()},
                        'native_readback': ctypes.string_at(prepared['native_address'],
                            len(bytes.fromhex(prepared['saved_native_bytes']))).hex(),
                        'sum': total, 'squared_sum': squared,
                        'device': str(product.device), 'device_state': device_state,
                        'calls': self.calls, 'challenge': challenge,
                        'run_id': os.environ.get('TTL_RUN'),
                        'socket': os.environ.get('PM_RAY_ZYGOTE_SOCK'),
                        'ray_version': ray.__version__, 'torch_version': torch.__version__,
                        'actor_id': runtime.get_actor_id(), 'node_id': runtime.get_node_id(),
                        'job_id': runtime.get_job_id(),
                        'worker_id': ray._private.worker.global_worker.worker_id.hex()}

            def mutate_native(self):
                import ctypes
                from pm_rewind_ray import get_prepared_state
                before = get_prepared_state()
                byte = ctypes.c_ubyte.from_address(before['native_address'])
                byte.value ^= 255
                return {'before': before, 'after': get_prepared_state()}

        from ray._private import state as ray_state
        registration = getattr(ray_state, 'actors', None)
        require(callable(registration), 'unresolved interface: ray._private.state.actors unavailable')
        for line in sys.stdin:
            request = json.loads(line)
            op = request['op']
            if op == 'stop':
                break
            generation = request['generation']
            destination = root / f'{op}-{generation}.json'
            t0 = time.monotonic()
            result = {'success': False, 'generation': generation, 'op': op}
            try:
                if op == 'actors':
                    env = dict(cfg['frozen'], **generation_env(cfg, request['socket'], request['run_id']),
                               TTL_RAY_WORKER='1')
                    runtime_env = {'py_executable': cfg['py_executable'], 'env_vars': env}
                    owned = [NativeActor.options(runtime_env=runtime_env,
                             name=f'{cfg["namespace"]}-{generation}-{n}').remote() for n in range(2)]
                    actors[generation] = owned
                    proofs = []
                    for index, actor in enumerate(owned):
                        pair = []
                        for call in range(2):
                            challenge = f'{cfg["namespace"]}:{generation}:{index}:{call}'
                            proof = ray.get(actor.proof.remote(challenge), timeout=cfg['timeout'])
                            require(proof['challenge'] == challenge and proof['calls'] == call + 1,
                                    'actual actor RPC continuity failed')
                            pair.append(proof)
                        actor_id = actor._actor_id.hex()
                        metadata = until(lambda: registration(actor_id), cfg['timeout'], 'GCS actor registration')
                        require(metadata['ActorID'] == actor_id and metadata['State'] == 'ALIVE',
                                f'actor not registered ALIVE: {metadata}')
                        require(metadata['Pid'] == pair[0]['identity']['pid'],
                                f'GCS registered proxy/virtual PID, not actual actor: {metadata}')
                        require(metadata['Address']['NodeID'] == pair[0]['node_id'], 'GCS node mismatch')
                        require(pair[0]['actor_id'] == actor_id, 'actor context ID mismatch')
                        proofs.append({'rpc': pair, 'registration': {k: metadata[k] for k in
                            ('ActorID', 'ActorClassName', 'Name', 'JobID', 'Address', 'OwnerAddress',
                             'State', 'NumRestarts', 'Pid')}})
                    mutation = ray.get(owned[0].mutate_native.remote(), timeout=cfg['timeout'])
                    require(mutation['after']['native_bytes'] != mutation['before']['native_bytes'],
                            'native COW mutation did not happen')
                    peer = ray.get(owned[1].proof.remote('unmutated-peer'), timeout=cfg['timeout'])
                    result.update(actors=proofs, mutation=mutation, unmutated_peer=peer,
                                  runtime_env=runtime_env)
                elif op == 'dead':
                    dead = []
                    for actor in actors.pop(generation):
                        actor_id = actor._actor_id.hex()
                        def dead_metadata():
                            entry = registration(actor_id)
                            return entry if entry and entry.get('State') == 'DEAD' else None
                        metadata = until(dead_metadata, cfg['timeout'], 'owned actor death registration')
                        try:
                            ray.get(actor.proof.remote('must-not-run'), timeout=cfg['timeout'])
                        except ray.exceptions.RayActorError as exc:
                            error = type(exc).__name__
                        else:
                            raise RuntimeError('actor RPC succeeded after controller cleanup')
                        dead.append({'actor_id': actor_id, 'state': metadata['State'], 'rpc_error': error})
                    result['actors'] = dead
                else:
                    raise ValueError(f'unknown private driver operation: {op}')
                result['success'] = True
            except BaseException as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
                traceback.print_exc()
            finally:
                result['elapsed_s'] = round(time.monotonic() - t0, 3)
                dump(destination, result)
            if not result['success']:
                raise RuntimeError(result['error'])
        state.update(success=True, phase='shutdown')
    except BaseException as exc:
        state['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if ray is not None and ray.is_initialized():
            # This driver can only have created the private head above.
            ray.shutdown()
        dump(root / 'ray-driver.json', state)
    return 0


class Probe:
    def __init__(self, root, env, report):
        self.root, self.env, self.report = root, env, report
        self.processes = []

    def start(self, name, argv, env=None, pipe=False):
        argv = list(map(str, argv))
        record = {'name': name, 'argv': argv, 'invocation': shlex.join(argv),
                  'cwd': str(self.root), 'log': f'{name}.log', 'started_monotonic': time.monotonic()}
        with (self.root / record['log']).open('wb') as log:
            proc = subprocess.Popen(argv, cwd=self.root, env=self.env if env is None else env,
                                    stdin=subprocess.PIPE if pipe else subprocess.DEVNULL,
                                    stdout=log, stderr=log, start_new_session=True)
        record['identity'] = identity(proc.pid)
        self.processes.append((proc, record))
        self.report['commands'].append(record)
        return proc

    def wait(self, proc, timeout, expected=None):
        rc = proc.wait(timeout=timeout)
        for child, record in self.processes:
            if child is proc:
                record.update(rc=rc, elapsed_s=round(time.monotonic() - record['started_monotonic'], 3))
                break
        if expected is not None:
            require(rc == expected, f'owned command exited {rc}, expected {expected}: {proc.args}')
        return rc

    def command(self, name, argv, timeout, expected=0, env=None):
        proc = self.start(name, argv, env=env)
        self.wait(proc, timeout, expected)
        return proc

    def stop(self, proc, sig=signal.SIGTERM):
        # Popen owns an unreaped direct child. Never signal a PID from a receipt.
        if proc.poll() is None:
            proc.send_signal(sig)
        return self.wait(proc, 20)

    def cleanup(self):
        errors = []
        for proc, record in reversed(self.processes):
            try:
                if proc.stdin is not None and not proc.stdin.closed:
                    proc.stdin.close()
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        self.wait(proc, 12)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        self.wait(proc, 12)
            except BaseException as exc:
                errors.append(f'{record["name"]}: {exc}')
        # This exec-created, single-threaded subreaper started childless. All
        # remaining direct children are its own adopted descendants. No other
        # waiter or SIGCHLD handler can reap them between inspection and kill.
        receipt = {'owned': {}, 'cleanup': 'pending', 'errors': errors}
        children = Path(f'/proc/self/task/{os.getpid()}/children')
        overdue = time.monotonic() + 20
        while True:
            pids = list(map(int, children.read_text().split()))
            for pid in pids:
                try:
                    receipt['owned'].setdefault(str(pid), identity(pid))
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    receipt['cleanup'] = 'complete'
                    return receipt
                if not pid:
                    break
                receipt['owned'].setdefault(str(pid), {'pid': pid}).update(
                    reaped=True, rc=os.waitstatus_to_exitcode(status))
            if time.monotonic() >= overdue:
                # Do not orphan a D-state task by discarding the sole owner.
                print('cleanup exceeded 20s; retaining owned subreaper', file=sys.stderr, flush=True)
                overdue = time.monotonic() + 20
            time.sleep(.03)


def events(path, prefix):
    if not Path(path).exists():
        return []
    result = []
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if line.endswith('\n') and line.startswith(prefix if prefix else '{'):
                result.append(json.loads(line[len(prefix):]))
    return result


def proxy_audit(run, prepared, baseline, server, socket, require_exit=False, worker_rc=None):
    proxy = prepared['proxy']
    log = run / 'output.log'

    def receipts():
        identities = [e for e in events(log, 'PM_RAY_ZYGOTE_PROXY_IDENTITY ')
                      if e.get('proxy') == proxy]
        exits = [e for e in events(log, 'PM_RAY_ZYGOTE_PROXY_EXIT ')
                 if e.get('proxy') == proxy]
        if len(identities) < 2 or (require_exit and not exits):
            return None
        return identities, exits

    identities, exits = until(receipts, 15, f'authenticated proxy receipts: {proxy}')
    require(len(identities) == 2 and identities[0]['worker_pid'] is None
            and identities[1]['worker_pid'] == prepared['proc_pid'],
            'proxy handshake lacks ordered pre-fork and worker identity receipts')
    for entry in [*identities, *exits]:
        require(entry['run_id'] == run.name and entry['nonce'] == baseline['nonce']
                and entry['server_pid'] == server['pid'] and entry['socket'] == str(socket),
                'proxy receipt belongs to a different native generation')
    for entry in exits:
        require(entry['exited'] is True and entry['worker_pid'] == prepared['proc_pid'],
                'proxy exit receipt does not identify this actual worker')
        if worker_rc is not None and 'worker_returncode' in entry:
            require(entry['worker_returncode'] == worker_rc, 'proxy receipt worker exit status mismatch')
    if require_exit:
        gone(proxy)  # pidfd readiness alone does not prove Raylet/Popen reaping.
    return {'identity_receipts': identities, 'exit_receipts': exits,
            'exit_receipt_required': require_exit, 'exact_process_gone': not same_process(proxy)}


def supervisor(run, timeout=20):
    path = run / 'supervisor.json'
    def complete():
        try:
            receipt = load(path)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return receipt if receipt.get('cleanup') == 'complete' else None
    receipt = until(complete, timeout, f'generic controller cleanup receipt: {run}')
    require(receipt['bind'][0] == '127.0.0.1', 'coordinator was not loopback-only')
    require(receipt['owned'] and all(w.get('reaped') for w in receipt['owned'].values()),
            'generic controller did not reap its children')
    for who in (receipt['reaper'], *receipt['owned'].values()):
        require(who['start_ticks'] is not None, 'missing supervisor process identity')
        gone(who, reap_adopted=True)
    return receipt


def owned_pidfd(who, owner):
    """Pin and verify a descendant, never turn an untrusted PID into authority."""
    from pm_ray_zygote import pidfd_open
    require(owner.poll() is None, 'descendant ownership anchor already exited')
    fd = pidfd_open(who['pid'])
    try:
        require(same_process(who), 'descendant PID identity changed before pidfd acquisition')
        pid = who['pid']
        seen = set()
        while pid != owner.pid:
            require(pid > 1 and pid not in seen, 'process is not our controller descendant')
            seen.add(pid)
            pid = identity(pid)['ppid']
        info = Path(f'/proc/self/fdinfo/{fd}').read_text()
        fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
        require(int(fields['Pid']) == who['pid'] and same_process(who), 'pidfd target changed')
        return fd
    except BaseException:
        os.close(fd)
        raise


def proxy_pidfds(proxy, worker):
    require(proxy.poll() is None, 'proxy exited before pidfd evidence')
    result = []
    for entry in Path(f'/proc/{proxy.pid}/fd').iterdir():
        try:
            if os.readlink(entry) != 'anon_inode:[pidfd]':
                continue
            info = Path(f'/proc/{proxy.pid}/fdinfo/{entry.name}').read_text()
        except FileNotFoundError:
            continue
        fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
        if int(fields.get('Pid', '-1')) in (worker['pid'], -1):
            result.append({'fd': int(entry.name), 'fdinfo': info.strip()})
    require(result, 'strict proxy has no actual worker pidfd')
    return result


def proxy_ready(proxy, worker, timeout=10):
    def installed():
        fields = dict(line.split(':', 1) for line in Path(f'/proc/{proxy.pid}/status').read_text().splitlines()
                      if ':' in line)
        return bool(int(fields['SigCgt'].strip(), 16) & (1 << (signal.SIGTERM - 1)))
    until(installed, timeout, 'proxy TERM handler installation', proxy)
    return proxy_pidfds(proxy, worker)


def helper(probe, cfg, name, mode, socket, baseline, run_id):
    base = probe.root / name
    env = dict(probe.env, **cfg['frozen'])
    env.update(generation_env(cfg, socket, run_id), TTL_RAY_WORKER='1')
    proc = probe.start(name, [cfg['python'], '-S', cfg['proxy'], 'exec', cfg['helper'], mode, base], env=env)
    path = Path(str(base) + '.ready.json')
    ready = until(lambda: load(path) if path.exists() else None, cfg['timeout'], name, proc)
    check_native(ready['prepared'], baseline)
    require(ready['identity']['pid'] == ready['prepared']['proc_pid'], 'helper identity mismatch')
    require(ready['run_id'] == run_id and ready['socket'] == str(socket)
            and ready['cpu_boundary_checked'], 'helper did not inherit this checked generation')
    actual_proxy = identity(proc.pid)
    require(ready['prepared']['proxy'] == {key: actual_proxy[key] for key in ('pid', 'start_ticks')},
            'helper proxy receipt differs from the owned Popen process')
    ready['proxy_pidfds'] = proxy_ready(proc, ready['identity'])
    return proc, base, ready


def safety_cases(probe, cfg, controller, socket, baseline, server, run):
    from pm_ray_zygote import pidfd_send_signal
    results = probe.report['checks'].setdefault('ownership', {})
    guardbase = probe.root / 'guard'
    guard = probe.start('guard', [cfg['python'], cfg['helper'], 'guard', guardbase])
    guardpath = Path(str(guardbase) + '.ready.json')
    guardproof = until(lambda: load(guardpath) if guardpath.exists() else None, 15, 'owned guard', guard)

    proxy, base, ready = helper(probe, cfg, 'forward-term', 'wait', socket, baseline, run.name)
    require(probe.stop(proxy) == 73, 'proxy did not propagate helper TERM exit status')
    observed = load(str(base) + '.signal.json')
    require(observed['signal'] == signal.SIGTERM and
            observed['identity']['pid'] == ready['identity']['pid'], 'termination was not forwarded')
    gone(ready['identity'])
    ready['server_proxy_receipts'] = proxy_audit(
        run, ready['prepared'], baseline, server, socket, require_exit=True, worker_rc=73)
    results['forwarded_termination'] = dict(ready, observed=observed, rc=73)

    proxy, base, ready = helper(probe, cfg, 'proxy-disappears', 'wait', socket, baseline, run.name)
    require(probe.stop(proxy, signal.SIGKILL) == -signal.SIGKILL, 'proxy did not disappear')
    gone(ready['identity'])
    require(not Path(str(base) + '.signal.json').exists(), 'proxy death was not EOF/KILL cleanup')
    ready['server_proxy_receipts'] = proxy_audit(
        run, ready['prepared'], baseline, server, socket, require_exit=True)
    results['proxy_disappearance'] = ready

    proxy, base, ready = helper(probe, cfg, 'dead-pidfd', 'wait', socket, baseline, run.name)
    serverfd = owned_pidfd(server, controller)
    workerfd = None
    try:
        workerfd = owned_pidfd(ready['identity'], controller)
        pidfd_send_signal(serverfd, signal.SIGSTOP)
        until(lambda: identity(server['pid'])['state'] in ('T', 't'), 10, 'owned server stopped')
        # The real child exits, while the stopped real parent cannot yet send
        # exit status. The real proxy retains its now-exited worker pidfd.
        Path(str(base) + '.release').touch()
        until(lambda: bool(select.select([workerfd], [], [], 0)[0]), 10, 'worker pidfd exit readiness')
        before = proxy_pidfds(proxy, ready['identity'])
        require(proxy.poll() is None, 'proxy exited before dead-pidfd forwarding race')
        proxy.send_signal(signal.SIGTERM)  # Only our unreaped direct proxy.
        time.sleep(.15)
        require(proxy.poll() is None, 'late TERM killed the proxy instead of using its dead pidfd')
        require(guard.poll() is None and not Path(str(guardbase) + '.signal.json').exists(),
                'unrelated owned guard was signaled')
        results['dead_pidfd_forwarding'] = dict(ready, exited_pidfd_readable=True,
            retained_pidfds=before, guard=guardproof, guard_untouched=True,
            scope='real exited-worker race; kernel PID reuse is not forced')
    finally:
        # This fd pins only the verified owned server, even if it has died.
        try:
            pidfd_send_signal(serverfd, signal.SIGCONT)
        except ProcessLookupError:
            pass
        os.close(serverfd)
        if workerfd is not None:
            os.close(workerfd)
    probe.wait(proxy, 15, 0)
    gone(ready['identity'])
    results['dead_pidfd_forwarding']['server_proxy_receipts'] = proxy_audit(
        run, ready['prepared'], baseline, server, socket, require_exit=True, worker_rc=0)
    require(guard.poll() is None, 'guard disappeared during ownership probes')
    Path(str(guardbase) + '.release').touch()
    probe.wait(guard, 15, 0)
    gone(guardproof['identity'])
    return results


def compact_logs(root, limit=48 * 1024):
    compacted = []
    for path in root.rglob('*'):
        if (path.is_symlink() or not path.is_file() or
                path.suffix not in ('.log', '.out', '.err') or not path.resolve().is_relative_to(root)):
            continue
        size = path.stat().st_size
        if size > limit:
            with path.open('rb') as stream:
                first = stream.read(limit // 4)
                stream.seek(-(limit * 3 // 4), os.SEEK_END)
                last = stream.read()
            path.write_bytes(first + b'\n[probe: middle of log omitted]\n' + last)
            compacted.append({'path': str(path), 'original_bytes': size})
    return compacted


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python', required=True, help='runtime interpreter with real Ray/Torch/TorchTPU/JAX')
    parser.add_argument('--dmtcp-bin', required=True, type=Path)
    parser.add_argument('--allocator', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path, help='new private directory, never an existing cache')
    parser.add_argument('--timeout', type=float, default=180, help='per-phase deadline in seconds (default: 180)')
    args = parser.parse_args()
    require(args.timeout >= 40 and args.timeout < 3600, '--timeout must be >=40 and <3600 seconds')
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    root = args.out_dir.absolute()
    require(not os.path.lexists(root), '--out-dir must be new (no reuse or symlinks)')
    require(root.parent.resolve() == root.parent, '--out-dir parent must not contain symlinks')
    require(not root.is_relative_to(Path('/tmp/ray')), 'refusing the shared /tmp/ray tree')
    root.mkdir(mode=0o700)
    start = time.monotonic()
    report = {'success': False, 'invocation': shlex.join([sys.executable, *sys.argv]),
              'argv': [sys.executable, *sys.argv], 'cwd': os.getcwd(), 'commands': [], 'generations': [],
              'checks': {}, 'receipts': {}, 'interface_issues': [], 'cpu_only': True}
    probe = None
    cfg = None
    manifest = None
    paths = []
    work = root / 't'
    old_handlers = {}
    try:
        require(sys.platform == 'linux', 'Linux procfs, subreapers and pidfds are required')
        require(not Path(f'/proc/self/task/{os.getpid()}/children').read_text().strip(),
                'probe must begin childless; will not adopt unrelated children')
        require(len(list(Path('/proc/self/task').iterdir())) == 1, 'probe must begin single-threaded')
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        libc = ctypes.CDLL(None, use_errno=True)
        require(libc.prctl(36, 1, 0, 0, 0) == 0, f'PR_SET_CHILD_SUBREAPER failed: {ctypes.get_errno()}')
        for sig in (signal.SIGINT, signal.SIGTERM):
            def interrupt(signum, frame):
                raise SystemExit(128 + signum)
            old_handlers[sig] = signal.signal(sig, interrupt)
        python = Path(args.python).absolute()  # Do NOT resolve a venv interpreter symlink.
        allocator = args.allocator.absolute()
        dmtcp = args.dmtcp_bin.absolute()
        require(python.is_file() and os.access(python, os.X_OK), '--python is not executable')
        require(allocator.is_file(), '--allocator library is missing')
        for binary in ('dmtcp_launch', 'dmtcp_restart', 'dmtcp_coordinator', 'mtcp_restart'):
            require(os.access(dmtcp / binary, os.X_OK), f'missing executable: {dmtcp / binary}')
        scripts, report['source_staging'] = stage_sources(root)
        sys.path.insert(0, str(scripts))
        sys.dont_write_bytecode = True
        require('pm_ray_zygote' not in sys.modules,
                'probe must import the freshly staged pidfd implementation, not a preloaded source module')
        report['checks']['pidfd_preflight'] = pidfd_preflight()
        controller = scripts / 'pm_rewind.py'
        proxy = scripts / 'pm_ray_zygote.py'
        staged_probe = scripts / 'pm_rewind_ray_probe.py'
        # Ray's AF_UNIX session paths are longer than the generation socket.
        # Keep an explicitly recorded private short temp path when needed; never
        # use Ray's shared /tmp/ray default, nor delete the retained artifacts.
        if len(os.fsencode(work)) > 25:
            work = Path(tempfile.mkdtemp(prefix='pmrr-', dir='/tmp'))
            (root / 'private-temp').symlink_to(work, target_is_directory=True)
        else:
            work.mkdir(mode=0o700)
        report['private_temp'] = str(work)
        ray_temp = work / 'r'
        ray_temp.mkdir(mode=0o700)
        temp = work / 'tmp'
        temp.mkdir(mode=0o700)
        private_env = {'HOME': str(root / 'home'), 'XDG_CACHE_HOME': str(root / 'xdg-cache'),
                       'HF_HOME': str(root / 'hf-cache'), 'TPU_LOG_DIR': str(root / 'tpu-logs'),
                       'VLLM_CACHE_ROOT': str(root / 'vllm-cache')}
        for directory in private_env.values():
            Path(directory).mkdir(mode=0o700)
        tier3 = root / 'tier3-cache'
        tier3.mkdir(mode=0o700)
        frozen = {'PM_RAY_ZYGOTE_STRICT': '1', 'TTL_RAY_HOSTS': '127.0.0.1,127.0.0.2',
                  'TPU_WORKER_HOSTNAMES': '127.0.0.1,127.0.0.2', 'TPU_MULTIHOST_BACKEND': 'ray',
                  'TPU_ACCELERATOR_TYPE': 'v5litepod-8', 'TPU_TOPOLOGY': '4x2',
                  'TPU_TOPOLOGY_WRAP': 'false,false,false', 'TPU_HOST_BOUNDS': '2,1,1',
                  'TPU_CHIPS_PER_HOST_BOUNDS': '2,2,1', 'TORCH_TPU_TOPOLOGY': '4,2,1',
                  'TPU_SKIP_MDS_QUERY': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                  'OPENBLAS_NUM_THREADS': '1', 'TORCH_DEVICE_BACKEND_AUTOLOAD': '0',
                  'TTL_RAY_WORKER_UNSET': 'ENABLE_PJRT_COMPATIBILITY,TPU_NAME,JAX_PLATFORMS',
                  'PYTHONPATH': str(scripts), 'LD_PRELOAD': str(allocator), 'LD_LIBRARY_PATH': '',
                  'GCONV_PATH': '/usr/lib/x86_64-linux-gnu/gconv',
                  'GLIBC_TUNABLES': 'glibc.pthread.rseq=0',
                  'PATH': f'{python.parent}:/usr/local/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8',
                  'VLLM_PLUGINS': 'torchtpu,torchtpu_layers,ttl_probe,ttl_ray_pin,pm_torchtpu',
                  'VLLM_USE_RAY_V2_EXECUTOR_BACKEND': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
                  'TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT': str(tier3),
                  'TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT': str(tier3),
                  'TMPDIR': str(temp), 'RAY_TMPDIR': str(work), 'HF_HUB_OFFLINE': '1',
                  'RAY_USAGE_STATS_ENABLED': '0', 'RAY_ROTATION_MAX_BYTES': '65536',
                  'RAY_ROTATION_BACKUP_COUNT': '1', 'RAY_LD_PRELOAD_ON_WORKERS': '0',
                  'RAY_CLIENT_MODE': '0',
                  'TOKENIZERS_PARALLELISM': 'false'}
        frozen.update(private_env)
        # Whitelist instead of inheriting shared Ray addresses, runtime_env,
        # topology/worker ranks, JAX backends, tracing hooks, or preload settings.
        env = {key: frozen[key] for key in ('PATH', 'LANG', 'LD_LIBRARY_PATH', 'VLLM_PLUGINS',
               'VLLM_USE_RAY_V2_EXECUTOR_BACKEND', 'VLLM_WORKER_MULTIPROC_METHOD',
               'GCONV_PATH', 'GLIBC_TUNABLES')}
        env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp), RAY_TMPDIR=str(work),
                   RAY_USAGE_STATS_ENABLED='0', RAY_ROTATION_MAX_BYTES='65536',
                   RAY_ROTATION_BACKUP_COUNT='1', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0',
                   TTL_RAY_WORKER_UNSET=frozen['TTL_RAY_WORKER_UNSET'],
                   PYTHONPATH=str(scripts), CUDA_VISIBLE_DEVICES='')
        env.update(private_env)
        probe = Probe(root, env, report)
        helperpath = root / 'helper.py'
        helperpath.write_text(HELPER)
        cfg = {'root': str(root), 'python': str(python), 'proxy': str(proxy), 'helper': str(helperpath),
               'ray_temp': str(ray_temp), 'temp': str(temp), 'frozen': frozen,
               'namespace': 'pm-native-' + os.urandom(8).hex(), 'timeout': args.timeout,
               'py_executable': shlex.join([str(python), '-S', str(proxy), 'exec'])}
        dump(root / 'config.json', cfg)
        probe.command('versions', [python, staged_probe, '_versions', root / 'versions.json'],
                      args.timeout)
        report['versions'] = load(root / 'versions.json')
        probe.command('dmtcp-version', [dmtcp / 'dmtcp_launch', '--version'], 15)
        report['dmtcp_version'] = (root / 'dmtcp-version.log').read_text().strip()
        base = [sys.executable, str(controller)]
        cache = root / 'cache'
        create = [*base, 'prepare', cache, '--factory', 'pm_rewind_ray:prepare_workers',
                  '--python', python, '--dmtcp-bin', dmtcp, '--real-pids',
                  '--late-env', 'PM_RAY_ZYGOTE_SOCK', '--late-env', 'TTL_RUN',
                  '--late-env', 'TTL_RAY_WORKER_ENV', '--timeout', str(args.timeout)]
        for key, value in frozen.items():
            create += ['--env', f'{key}={value}']
        create += ['--env', 'PYTHONDONTWRITEBYTECODE=1']
        paths.append(cache / 'build')
        probe.command('prepare', create, args.timeout + 20)
        report['receipts']['capture'] = supervisor(cache / 'build')
        manifest = load(cache / 'manifest.json')
        require(manifest['config']['real_pids'] and len(manifest['images']) == 1,
                'capture is not one real-PID native image')
        captured = events(cache / 'build/backend.log', 'PM_REWIND_RAY_CAPTURE ')
        require(len(captured) == 1, 'expected exactly one factory execution/capture witness')
        baseline = captured[0]
        require(baseline['pid'] == baseline['proc_pid'] == baseline['capture_pid'], 'capture PID mismatch')
        require(not baseline['tpu_initialized'] and not baseline['ray_imported']
                and not baseline['ray_initialized'], 'capture crossed CPU client/device boundary')
        require(baseline['proxy'] is None, 'capture incorrectly retained a generation proxy identity')
        import pm_ray_zygote
        require(set(baseline['saved_module_ids']) == set(pm_ray_zygote.WORKER_CPU_PRELOAD), 'incorrect CPU prefix')
        require(baseline['saved_native_bytes'] == baseline['nonce'] + '00', 'invalid native capture buffer')
        capture_owner = report['receipts']['capture']['owned'].get(str(baseline['capture_pid']))
        require(capture_owner is not None and capture_owner['reaped'],
                'original capture is not accounted for by the generic supervisor')
        gone(capture_owner)
        report['capture'] = {'proof': baseline, 'capture_s': manifest['capture_s'],
                             'prepare_s': manifest['prepared']['prepare_s'], 'images': manifest['images']}
        image = cache / manifest['images'][0]['name']
        initial_hash = digest(image)
        require(initial_hash == manifest['images'][0]['sha256'], 'initial image digest mismatch')
        report['checks']['capture_exited_before_ray_start'] = True

        # Refuse a missing generation socket, bounded independently of Ray's
        # worker startup retries. The helper must not run, even via plain exec.
        missingbase = root / 'missing-socket'
        missing_env = dict(env, **frozen, PM_RAY_ZYGOTE_SOCK=str(work / 'absent.sock'))
        t0 = time.monotonic()
        probe.command('missing-socket', [python, '-S', proxy, 'exec', helperpath, 'wait', missingbase],
                      40, expected=1, env=missing_env)
        refusal = (root / 'missing-socket.log').read_text()
        require('strict exec failed' in refusal and 'within 30s' in refusal,
                'missing socket did not fail through the bounded strict refusal path')
        require(not Path(str(missingbase) + '.entered').exists(), 'strict proxy executed a fallback')
        report['checks']['missing_socket'] = {'elapsed_s': round(time.monotonic() - t0, 3),
                                             'bounded_s': 40, 'fallback_executed': False}

        driver = probe.start('ray-driver', [python, staged_probe, '_cluster', root / 'config.json'], pipe=True)
        ready = until(lambda: load(root / 'ray-ready.json') if (root / 'ray-ready.json').exists() else None,
                      args.timeout, 'real local Ray startup', driver)
        report['ray'] = ready
        seen_pids = {baseline['capture_pid']}
        for generation in range(2):
            run = root / f'run{generation}'
            paths.append(run)
            socket = work / f'g{generation}.sock'
            run_command = [*base, 'run', cache, '--run-dir', run, '--timeout', str(args.timeout * 6)]
            for key, value in generation_env(cfg, socket, run.name).items():
                run_command += ['--env', f'{key}={value}']
            controller_proc = probe.start(f'controller-{generation}', [*run_command, '--', 'serve'])
            t0 = time.monotonic()
            def listening():
                output = run / 'output.log'
                return socket.exists() and output.exists() and 'PM_RAY_ZYGOTE_READY' in output.read_text()
            until(listening, args.timeout, 'native restored worker server', controller_proc)
            restored = [e for e in events(run / 'output.log', '') if e.get('event') == 'PM_REWIND_RESTORED']
            require(len(restored) == 1 and restored[0]['real_pids'], 'missing native controller restore event')
            server = identity(restored[0]['pid'])
            require(server['pid'] not in seen_pids, 'restored server reused an actual PID')
            seen_pids.add(server['pid'])
            driver.stdin.write((json.dumps({'op': 'actors', 'generation': generation,
                               'socket': str(socket), 'run_id': run.name}) + '\n').encode())
            driver.stdin.flush()
            resultpath = root / f'actors-{generation}.json'
            result = until(lambda: load(resultpath) if resultpath.exists() else None,
                           args.timeout * 6, 'real actor registration and CPU RPCs', driver)
            require(result['success'], f'real Ray actor qualification failed: {result}')
            actor_identities = []
            for actor in result['actors']:
                who = actor['rpc'][0]['identity']
                require(who['pid'] not in seen_pids, 'actual worker PID was reused across actors/generations')
                seen_pids.add(who['pid'])
                actor_identities.append(who)
                actor_proxy = actor['rpc'][0]['prepared']['proxy']
                require(actor_proxy['pid'] not in seen_pids and same_process(actor_proxy),
                        'actor proxy is not a fresh live actual process')
                seen_pids.add(actor_proxy['pid'])
                actor['proxy_receipts_before_shutdown'] = proxy_audit(
                    run, actor['rpc'][0]['prepared'], baseline, server, socket)
                for proof in actor['rpc']:
                    check_native(proof['prepared'], baseline)
                    require(proof['prepared']['proxy'] == actor_proxy, 'actor changed its authenticated proxy')
                    require(proof['run_id'] == run.name and proof['socket'] == str(socket),
                            'actor did not inherit the explicitly bound generation')
                    require(all(proof['identity'][key] == who[key] for key in ('pid', 'ppid', 'start_ticks')),
                            'actor process changed between successful RPCs')
                    require(who['ppid'] == server['pid'], 'actor did not fork from the native restored server')
                    require(proof['native_readback'] == baseline['saved_native_bytes'], 'raw native readback mismatch')
                    require(proof['device'] == 'cpu' and proof['sum'] == 120 and proof['squared_sum'] == 1240,
                            'real post-fork CPU torch computation failed')
            check_native(result['unmutated_peer']['prepared'], baseline)
            check_native(result['mutation']['before'], baseline)
            require(result['mutation']['after']['saved_native_bytes'] == baseline['saved_native_bytes'],
                    'mutation replaced saved native witness')
            generation_result = {'generation': generation, 'run_id': run.name,
                                 'restore_s': restored[0]['restore_s'],
                                 'server': server, 'actors': result, 'socket': str(socket)}
            report['generations'].append(generation_result)
            if generation == 0:
                report['checks']['ownership'] = safety_cases(
                    probe, cfg, controller_proc, socket, baseline, server, run)
            tree, treebase, treeproof = helper(
                probe, cfg, f'owned-tree-{generation}', 'tree', socket, baseline, run.name)
            treeproof['proxy_receipts_before_shutdown'] = proxy_audit(
                run, treeproof['prepared'], baseline, server, socket)
            until(lambda: identity(treeproof['leaf']['pid'])['state'] in ('T', 't'), 10, 'owned stopped leaf')
            sig = signal.SIGTERM if generation == 0 else signal.SIGKILL
            expected = 143 if generation == 0 else -signal.SIGKILL
            require(probe.stop(controller_proc, sig) == expected, 'generic controller termination status mismatch')
            receipt = supervisor(run)
            report['receipts'][f'run{generation}'] = receipt
            for who in [server, treeproof['identity'], treeproof['leaf'], *actor_identities]:
                gone(who)
            require(probe.wait(tree, 20) != 0, 'owner disappearance left a successful live helper')
            # Include every authenticated proxy, not just actors that registered
            # successfully: failed Ray worker attempts must not escape this audit.
            registered_proxies = {}
            for entry in events(run / 'output.log', 'PM_RAY_ZYGOTE_PROXY_IDENTITY '):
                who = entry['proxy']
                require(entry['run_id'] == run.name and entry['nonce'] == baseline['nonce']
                        and entry['server_pid'] == server['pid'] and entry['socket'] == str(socket),
                        'generation contains a foreign proxy receipt')
                registered_proxies[(who['pid'], who['start_ticks'])] = who
            require(registered_proxies, 'no authenticated proxies registered')
            for who in registered_proxies.values():
                gone(who, reap_adopted=True)
            generation_result['proxy_shutdown'] = {
                'identities': list(registered_proxies.values()), 'all_exact_processes_gone': True,
                'exit_receipts': events(run / 'output.log', 'PM_RAY_ZYGOTE_PROXY_EXIT '),
                'basis': 'exact pid/start_ticks disappearance; forced controller cleanup need not emit server EXIT'}
            for actor in result['actors']:
                actor['proxy_receipts_after_shutdown'] = proxy_audit(
                    run, actor['rpc'][0]['prepared'], baseline, server, socket)
            treeproof['proxy_receipts_after_shutdown'] = proxy_audit(
                run, treeproof['prepared'], baseline, server, socket)
            generation_result.update(shutdown_signal=sig.name, tree=treeproof,
                                     elapsed_s=round(time.monotonic() - t0, 3))
            driver.stdin.write((json.dumps({'op': 'dead', 'generation': generation}) + '\n').encode())
            driver.stdin.flush()
            deadpath = root / f'dead-{generation}.json'
            dead = until(lambda: load(deadpath) if deadpath.exists() else None,
                         args.timeout * 3, 'actual Ray actor death registration', driver)
            require(dead['success'], f'Ray actor death qualification failed: {dead}')
            generation_result['death_registration'] = dead
        driver.stdin.write(b'{"op":"stop"}\n')
        driver.stdin.flush()
        driver.stdin.close()
        probe.wait(driver, 30, 0)
        probe.command('verify', [*base, 'verify', cache], args.timeout)
        require(len(events(cache / 'build/backend.log', 'PM_REWIND_RAY_CAPTURE ')) == 1 and
                all(not events(root / f'run{n}/output.log', 'PM_REWIND_RAY_CAPTURE ') for n in range(2)),
                'factory unexpectedly executed again after capture')
        report['checks']['factory_executions'] = 1
        report['checks']['two_restores_real_actor_rpcs'] = True
        report['checks']['native_cow_isolated_between_actors_and_generations'] = True
        report['success'] = True
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if (any(name in str(exc) for name in ('interface', 'AttributeError', 'ImportError', 'KeyError'))
                or isinstance(exc, (KeyError, AttributeError, ImportError))):
            report['interface_issues'].append(report['error'])
        traceback.print_exc()
    finally:
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if probe is not None:
                report['probe_cleanup'] = probe.cleanup()
                if report['probe_cleanup']['errors']:
                    report['success'] = False
                for run in paths:
                    if (run / 'supervisor.json').exists():
                        try:
                            report['receipts'][str(run.relative_to(root))] = supervisor(run)
                        except BaseException as exc:
                            report['success'] = False
                            report.setdefault('cleanup_errors', []).append(f'{run}: {exc}')
            if manifest is not None:
                hashes = []
                for entry in manifest['images']:
                    image = root / 'cache' / entry['name']
                    actual = digest(image)
                    unchanged = (actual == entry['sha256'] and image.stat().st_mtime_ns == entry['mtime_ns']
                                 and stat.S_IMODE(image.stat().st_mode) == 0o400)
                    hashes.append({'name': entry['name'], 'before': entry['sha256'],
                                   'after': actual, 'unchanged': unchanged})
                    if not unchanged:
                        report['success'] = False
                report['checks']['immutable_images_after_all_restores'] = hashes
            source_hashes = []
            for entry in report.get('source_staging', []):
                path = Path(entry['copy'])
                actual = digest(path)
                unchanged = (actual == entry['source_sha256'] == entry['copy_sha256']
                             and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600
                             and stat.S_IMODE(path.parent.stat().st_mode) == 0o700)
                source_hashes.append({'copy': str(path), 'sha256': actual, 'unchanged': unchanged})
                if not unchanged:
                    report['success'] = False
            report['checks']['staged_sources_unchanged'] = source_hashes
            if (root / 'ray-driver.json').exists():
                report['ray_driver_final'] = load(root / 'ray-driver.json')
                driver_error = report['ray_driver_final'].get('error', '')
                if any(name in driver_error for name in ('interface', 'AttributeError', 'ImportError', 'TypeError')):
                    if driver_error not in report['interface_issues']:
                        report['interface_issues'].append(driver_error)
            report['compacted_logs'] = compact_logs(root)
            if not work.is_relative_to(root) and work.exists():
                report['compacted_logs'] += compact_logs(work)
        except BaseException as exc:
            report['success'] = False
            report.setdefault('cleanup_errors', []).append(f'{type(exc).__name__}: {exc}')
        report['elapsed_s'] = round(time.monotonic() - start, 3)
        dump(root / 'report.json', report)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        print(json.dumps({'report': str(root / 'report.json'), 'success': report['success']}), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    if sys.argv[1:2] == ['_cluster']:
        sys.exit(_cluster(sys.argv[2]))
    if sys.argv[1:2] == ['_versions']:
        dump(sys.argv[2], dict(versions(), pidfd_preflight=pidfd_preflight()))
        sys.exit(0)
    sys.exit(main())
