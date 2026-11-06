#!/usr/bin/env python3
"""Local, real-DMTCP acceptance probe; never initializes a TPU or starts an engine.

  python scripts/pm_rewind_probe.py --root /dev/shm/rewind-proof \
    --python /path/to/runtime/python --dmtcp-bin /path/to/dmtcp/bin [--full-runtime]

The full-runtime arm imports the real TorchTPU/vLLM stack and uses TPUFlex's
actual forkserver client. A pass is CPU/native-heap evidence, NOT a TPU proof.
Artifacts stay under --root; native images must not be committed or made public.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import time

_STATE = None


def identity():
    pid = next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines()
               if s.startswith('Pid:'))
    return {'real_pid': int(pid),
            'start_ticks': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]}


def gone(who):
    try:
        current = Path(f'/proc/{who["real_pid"]}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return
    assert current != who['start_ticks'], who


def reject_factory():
    if os.getenv('PM_PROBE_FIFO'):
        os.mkfifo(os.environ['PM_PROBE_FIFO'], 0o600)
        fd = os.open(os.environ['PM_PROBE_FIFO'], os.O_RDONLY | os.O_NONBLOCK)
        return lambda argv: os.read(fd, 1)
    if not os.getenv('PM_PROBE_SHARED'):
        raise RuntimeError('intentional capture failure')
    fd = os.open(os.environ['PM_PROBE_SHARED'], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.ftruncate(fd, 4096)
    mmap = ctypes.CDLL(None).mmap
    mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.c_long]
    mmap.restype = ctypes.c_void_p
    address = mmap(None, 4096, int(os.getenv('PM_PROBE_SHARED_PROT', '3')), 1, fd, 0)
    os.close(fd)
    assert address != ctypes.c_void_p(-1).value
    return lambda argv: ctypes.string_at(address, 1)


def worker(output):
    import torch
    state = _STATE
    result = {'sum': state['tensor'].sum().item(), 'nonce': state['nonce'],
              'tensor_ptr': state['tensor'].data_ptr(), 'token': os.getenv('PM_PROBE_TOKEN'),
              'torch_id': id(torch), 'pid': os.getpid(),
              'large_sum': state['bulk'].double().sum().item(), 'identity': identity()}
    if hasattr(torch, 'tpu'):
        result['tpu_initialized'] = torch.tpu.is_initialized()
        result['bad_fork'] = torch.tpu._is_in_bad_fork()
        assert not result['tpu_initialized'] and not result['bad_fork']
        result['private_backend'] = torch._C._get_privateuse1_backend_name()
        if os.getenv('PM_PROBE_FULL') == '1':
            assert result['private_backend'] == 'tpu'
    Path(output).write_text(json.dumps(result))


def prepare():
    global _STATE
    full = os.getenv('PM_PROBE_FULL') == '1'
    serve_resume = None
    if full:
        import pm_zygote
        serve_resume = pm_zygote.prepare_rewind()
    import torch
    tensor = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    bulk = torch.arange(1 << 20, dtype=torch.float32)
    nonce = secrets.token_hex(16)
    _STATE = {'tensor': tensor, 'nonce': nonce, 'bulk': bulk}
    Path(os.environ['PM_PROBE_MARKER']).write_text(nonce)
    calls = [0]
    torch_id = id(torch)

    def resume(argv):
        run = Path(os.environ['_PM_REWIND_RUN_DIR'])
        (run / 'resumed.json').write_text(json.dumps(identity()))
        if argv == ['stopped-child']:
            child = os.fork()
            if child == 0:
                (run / 'stopped.json').write_text(json.dumps(identity()))
                os.kill(os.getpid(), signal.SIGSTOP)
                os._exit(0)
            _, status = os.waitpid(child, os.WUNTRACED)
            assert os.WIFSTOPPED(status)
            return 0
        if argv == ['sleep']:
            print('PROBE_SLEEP_READY', flush=True)
            time.sleep(120)
            return 0
        if argv == ['exit7']:
            return 7
        assert bulk.double().sum().item() == 549755289600
        if argv and argv[0] == 'serve':
            assert serve_resume is not None
            return serve_resume(argv)
        assert calls == [0] and tensor.sum().item() == 120
        assert id(torch) == torch_id
        # Exercise new ctypes types after restart, not only captured API bindings.
        assert len(ctypes.create_string_buffer(12345)) == 12345
        output = run / 'worker.json'
        if full:
            from pm_zygote import rewind_zygote
            with rewind_zygote():
                from vllm.v1.engine.utils import get_mp_context
                proc = get_mp_context().Process(target=worker, args=(str(output),))
                assert proc.__class__.__name__ == 'ZygoteProcess'
                proc.start()
                proc.join(30)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                    raise RuntimeError('restored forkserver worker hung')
                assert proc.exitcode == 0, proc.exitcode
        else:
            child = os.fork()
            if child == 0:
                try:
                    worker(output)
                except BaseException:
                    import traceback
                    traceback.print_exc()
                    os._exit(1)
                os._exit(0)
            _, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0, status
        actual = json.loads(output.read_text())
        assert actual['sum'] == 120 and actual['nonce'] == nonce
        assert actual['tensor_ptr'] == tensor.data_ptr() and actual['torch_id'] == torch_id
        assert actual['token'] == os.environ['PM_PROBE_TOKEN']
        calls[0] += 1
        tensor.add_(1000)  # Next rewind MUST see the pristine native storage again.
        result = {'event': 'PM_REWIND_PROBE', 'argv': argv, 'nonce': nonce, 'calls': calls[0],
                  'mutated_sum': tensor.sum().item(), 'worker': actual,
                  'worker_kind': 'ZygoteProcess' if full else 'fork', 'identity': identity()}
        print(json.dumps(result), flush=True)
        return 0
    return resume


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def no_coordinator(run):
    receipt = json.loads((run / 'supervisor.json').read_text())
    assert receipt['cleanup'] == 'complete', receipt
    assert receipt['bind'][0] == '127.0.0.1', receipt
    for who in (receipt['reaper'], *receipt['owned'].values()):
        gone({'real_pid': who['pid'], 'start_ticks': str(who['start_ticks'])})
    assert all(who['reaped'] for who in receipt['owned'].values()), receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--dmtcp-bin', required=True)
    p.add_argument('--real-pids', action='store_true')
    p.add_argument('--full-runtime', action='store_true')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--allocator-preload', type=Path)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # Native-failure receipts, not multi-GB core files.
    root = a.root.absolute()
    root.mkdir(mode=0o700)
    cache = root / 'cache'
    controller = Path(__file__).with_name('pm_rewind.py')
    base = [sys.executable, str(controller)]
    report = {'full_runtime': a.full_runtime, 'real_pids': a.real_pids,
              'commands': [], 'runs': [], 'success': False}

    def command(args, expected=0):
        t0 = time.monotonic()
        proc = subprocess.run([*base, *map(str, args)], text=True, capture_output=True, timeout=180)
        record = {'args': list(map(str, args)), 'rc': proc.returncode,
                  'elapsed_s': time.monotonic() - t0, 'stdout': proc.stdout, 'stderr': proc.stderr}
        report['commands'].append(record)
        assert proc.returncode == expected, record
        return record

    try:
        create = ['prepare', cache, '--factory', 'pm_rewind_probe:prepare', '--python', a.python,
                  '--dmtcp-bin', a.dmtcp_bin, '--late-env', 'PM_PROBE_TOKEN',
                  '--env', f'PM_PROBE_MARKER={root / "prepared.txt"}',
                  '--env', f'PM_PROBE_FULL={int(a.full_runtime)}',
                  '--env', 'OMP_NUM_THREADS=1', '--env', 'OPENBLAS_NUM_THREADS=1',
                  '--env', 'HF_HUB_OFFLINE=1', '--env', 'TPU_MULTIHOST_BACKEND=mp']
        if a.real_pids:
            create.append('--real-pids')
        if not a.full_runtime:
            create.extend(['--env', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0'])
        else:
            create.extend(['--late-env', 'TORCH_TPU_DP_SIZE', '--late-env', 'MASTER_ADDR'])
        if a.allocator_preload:
            create.extend(['--env', f'LD_PRELOAD={a.allocator_preload.absolute()}'])
        command(create)
        assert len(list((cache / 'build').glob('dmtcp_restart_script_*.sh'))) == 1
        man = json.loads((cache / 'manifest.json').read_text())
        report['capture_s'] = man['capture_s']
        report['prepare_s'] = man['prepared']['prepare_s']
        report['images'] = man['images']
        marker = (root / 'prepared.txt').stat().st_mtime_ns
        no_coordinator(cache / 'build')
        for generation in range(a.repeats):
            run = root / f'run{generation}'
            record = command(['run', cache, '--run-dir', run, '--timeout', '60',
                              '--env', f'PM_PROBE_TOKEN=fresh-{generation}', '--', str(generation)])
            events = [json.loads(s) for s in (run / 'output.log').read_text().splitlines()
                      if s.startswith('{')]
            restored = next(e for e in events if e.get('event') == 'PM_REWIND_RESTORED')
            proof = next(e for e in events if e.get('event') == 'PM_REWIND_PROBE')
            assert proof['argv'] == [str(generation)]
            assert proof['worker']['token'] == f'fresh-{generation}'
            assert proof['nonce'] == (root / 'prepared.txt').read_text()
            assert proof['calls'] == 1 and proof['mutated_sum'] == 16120
            assert proof['worker']['large_sum'] == 549755289600
            if a.real_pids:
                assert restored['real_pids'] is True
                assert proof['worker']['pid'] == proof['worker']['identity']['real_pid']
            report['runs'].append({'restore_s': restored['restore_s'],
                                   'elapsed_s': record['elapsed_s'], 'proof': proof})
            no_coordinator(run)
            gone(proof['identity'])
            gone(proof['worker']['identity'])
        if a.full_runtime:
            command(['run', cache, '--run-dir', root / 'cli', '--', 'serve', '--help'])
            assert 'usage: vllm serve' in (root / 'cli/output.log').read_text()
            no_coordinator(root / 'cli')
            report['real_cli_help'] = True
            config = root / 'unsupported.yaml'
            config.write_text(json.dumps({'model': 'probe-not-loaded', 'nnodes': 2, 'tensor-parallel-size': 2}))
            unsupported = [['--data_parallel_size', '2'], ['-tp', '2', '-n', '2'],
                           ['--data-parallel-backend', 'ray'], ['--config', str(config)]]
            for index, flags in enumerate(unsupported):
                rejected = root / f'rejected-cli{index}'
                command(['run', cache, '--run-dir', rejected, '--', 'serve', *flags], 1)
                text = (rejected / 'output.log').read_text()
                assert 'rewind requires one local HTTP server' in text, text
                assert 'PM_ZYGOTE_READY' in text  # parser needs post-fork plugins; dispatch is still blocked
                no_coordinator(rejected)
            for index, assignment in enumerate(('TORCH_TPU_DP_SIZE=2', 'MASTER_ADDR=192.0.2.1')):
                rejected = root / f'rejected-identity{index}'
                command(['run', cache, '--run-dir', rejected, '--env', assignment, '--', 'serve'], 1)
                text = (rejected / 'output.log').read_text()
                assert 'rewind cannot inherit external distributed identity' in text, text
                assert 'PM_ZYGOTE_READY' not in text
                no_coordinator(rejected)
        hit = command(create)
        assert 'PM_REWIND_CACHE_HIT' in hit['stdout']
        assert (root / 'prepared.txt').stat().st_mtime_ns == marker
        literal = root / 'literal'
        command(['run', cache, '--run-dir', literal, '--env', 'PM_PROBE_TOKEN=literal',
                 '--', '--', '-filename'])
        events = [json.loads(s) for s in (literal / 'output.log').read_text().splitlines()
                  if s.startswith('{')]
        assert next(e for e in events if e.get('event') == 'PM_REWIND_PROBE')['argv'] == ['--', '-filename']
        no_coordinator(literal)
        command(['run', cache, '--run-dir', root / 'forbidden-env', '--env', 'OMP_NUM_THREADS=2'], 1)
        assert not (root / 'forbidden-env').exists()
        command(['run', cache, '--run-dir', root / 'exit7', '--', 'exit7'], 7)
        command(['run', cache, '--run-dir', root / 'stopped', '--', 'stopped-child'])
        gone(json.loads((root / 'stopped/stopped.json').read_text()))
        no_coordinator(root / 'stopped')
        command(['run', cache, '--run-dir', root / 'timeout', '--timeout', '5', '--', 'sleep'], 124)
        no_coordinator(root / 'timeout')
        gone(json.loads((root / 'timeout/resumed.json').read_text()))
        # Controller SIGTERM must clean its coordinator and restored peers too.
        terminate = root / 'terminate'
        with (root / 'terminate-controller.log').open('w') as log:
            proc = subprocess.Popen([*base, 'run', str(cache), '--run-dir', str(terminate),
                                     '--', 'sleep'], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    output = terminate / 'output.log'
                    if output.exists() and 'PROBE_SLEEP_READY' in output.read_text():
                        break
                    if proc.poll() is not None:
                        raise RuntimeError('restore exited before SIGTERM probe')
                    time.sleep(0.05)
                else:
                    raise TimeoutError('restore did not reach SIGTERM probe')
                proc.send_signal(signal.SIGTERM)
                assert proc.wait(timeout=20) == 143
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        no_coordinator(terminate)
        gone(json.loads((terminate / 'resumed.json').read_text()))
        if not a.full_runtime:
            for kind in ('exception', 'shared', 'shared-readonly', 'fifo'):
                failed = root / f'capture-{kind}'
                args = ['prepare', failed, '--factory', 'pm_rewind_probe:reject_factory',
                        '--python', a.python, '--dmtcp-bin', a.dmtcp_bin]
                if a.real_pids:
                    args.append('--real-pids')
                if kind == 'fifo':
                    args += ['--env', f'PM_PROBE_FIFO={root / "external-fifo"}']
                elif kind != 'exception':
                    args += ['--env', f'PM_PROBE_SHARED={root / ("external-" + kind)}',
                             '--env', f'PM_PROBE_SHARED_PROT={1 if kind == "shared-readonly" else 3}']
                command(args, 1)
                assert not (failed / 'manifest.json').exists()
                text = (failed / 'build/backend.log').read_text()
                expected = ('intentional capture failure' if kind == 'exception'
                            else 'unsupported external descriptor' if kind == 'fifo'
                            else 'external shared mapping')
                assert expected in text, text
                no_coordinator(failed / 'build')
        for kind in ('host', 'runtime', 'checksum'):
            invalid = root / f'invalid-{kind}'
            invalid.mkdir(mode=0o700)
            bad = json.loads(json.dumps(man))
            if kind == 'host':
                bad['host'] = {}
            elif kind == 'runtime':
                next(iter(bad['prepared']['files'].values()))['stat'] = []
            else:
                for ent in bad['images']:
                    with (invalid / ent['name']).open('wb') as f:
                        f.truncate(ent['bytes'])  # separate sparse image; never corrupt the template
                    (invalid / ent['name']).chmod(0o400)
                    os.utime(invalid / ent['name'], ns=(time.time_ns(), ent['mtime_ns']))
            (invalid / 'manifest.json').write_text(json.dumps(bad))
            rejected = command(['verify', invalid], 1)
            assert {'host': 'host ABI mismatch', 'runtime': 'installed runtime changed',
                    'checksum': 'checkpoint checksum mismatch'}[kind] in rejected['stderr']
            if kind == 'checksum':
                for ent in bad['images']:
                    (invalid / ent['name']).unlink()
        command(['verify', cache])
        for ent in man['images']:
            assert sha256(cache / ent['name']) == ent['sha256']
            assert (cache / ent['name']).stat().st_mtime_ns == ent['mtime_ns']
            assert (cache / ent['name']).stat().st_mode & 0o777 == 0o400
        report['success'] = True
    except BaseException as e:
        report['error'] = repr(e)
        raise
    finally:
        (root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'report': str(root / 'report.json'), 'success': report['success']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
