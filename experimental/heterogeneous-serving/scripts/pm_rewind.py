#!/usr/bin/env python3
"""Durable CPU-process templates, before device/IPC initialization (DMTCP 4.2).

A factory ``module:function`` runs once and returns a callable ``resume(argv)``.
Its Python closure AND native heap are checkpointed, not pickled. Each rewind
starts from that immutable image, applies explicitly late-bound environment,
and invokes the saved callable. External effects are NOT rolled back.

  python pm_rewind.py prepare CACHE --factory pm_zygote:prepare_rewind \
      --dmtcp-bin /path/to/dmtcp/bin --env TPU_MULTIHOST_BACKEND=mp \
      --env LD_PRELOAD=/path/to/libtcmalloc_minimal.so.4 \
      --late-env PM_GATE_FILE
  python pm_rewind.py run CACHE --run-dir RUN --env PM_GATE_FILE=/tmp/newgate \
      -- serve MODEL --tensor-parallel-size 4 ...

Use the same host and installed runtime/files/CPU ABI for prepare and run. Cache
and run folders are trusted, private inputs (executable memory, possibly secrets).
Only explicitly supplied prepare-time env is captured, plus PATH/HOME/LANG.
Transfer the complete cache directory back to that host; verify hashes before use.
Hot run checks metadata, not full-image hashes. No cross-host/GCS path is qualified.
No power-loss durability guarantee. The manifest is published only after capture.
Lifecycle/acceptance reference: https://github.com/vllm-project/vllm/pull/51360
(its initialized CUDA-engine boundary is deliberately not used here).
"""
import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time

FORMAT = 1
CONTEXT_ENV = '_PM_REWIND_CONTEXT'


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, sort_keys=True), flush=True)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    path.chmod(0o600)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def file_id(path):
    s = Path(path).stat()
    return [s.st_size, s.st_mtime_ns]


def env_pairs(items):
    out = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or not key or key.startswith(('DMTCP_', '_PM_REWIND_')):
            raise ValueError(f'invalid or reserved environment assignment: {item!r}')
        out[key] = value
    return out


def cpu_flags():
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith(('flags\t', 'Features\t')):
            return sorted(line.partition(':')[2].split())
    raise RuntimeError('cannot determine CPU features')


def host_id():
    # Images are boot-local; minimal containers need not provide /etc/machine-id.
    return {'machine': platform.machine(), 'kernel': platform.release(),
            'libc': platform.libc_ver(), 'uid': os.getuid(), 'cpu_flags': cpu_flags(),
            'boot_id': digest('/proc/sys/kernel/random/boot_id')}


def trusted_dir(path, creating=False):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if creating and component == path and not component.exists() and not component.is_symlink():
            continue
        s = component.lstat()
        if component.is_symlink() or not component.is_dir() or s.st_uid not in (0, os.getuid()):
            raise ValueError(f'untrusted directory component: {component}')
        sticky_root = s.st_uid == 0 and s.st_mode & 0o1000
        if s.st_mode & 0o022 and not sticky_root:
            raise ValueError(f'writable directory component: {component}')
        if component == path and (s.st_uid != os.getuid() or s.st_mode & 0o077):
            raise ValueError(f'expected an owner-only directory: {path}')
    return path


def backend(bin_dir, name):
    path = Path(bin_dir).resolve() / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'missing executable: {path}')
    return str(path)


def _owned_identity(pid):
    # Diagnostic only; never use this receipt as authority to signal a PID.
    try:
        ticks = int(Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19])
    except OSError:
        ticks = None
    return {'pid': pid, 'start_ticks': ticks}


def _reap_owned_children(children, processes, receipt):
    """Kill/reap only direct children of this otherwise childless subreaper."""
    deadline = time.monotonic() + 5
    overdue = False
    owned = receipt['owned']
    while True:
        try:
            os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return  # Also handles setup failure before procfs/spawning worked.
        # No concurrent waiters: a direct child pins its PID until we reap it.
        # Killing parents adopts their children here, even across setsid/SIGSTOP.
        for pid in map(int, children.read_text().split()):
            owned.setdefault(str(pid), _owned_identity(pid))
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                # Do not let one refused kill prevent cleanup of the others.
                # Retain ownership and retry until the kernel lets them exit.
                pass
        while True:
            try:
                info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                return
            if info is None:
                break
            pid = info.si_pid
            entry = owned.setdefault(str(pid), _owned_identity(pid))
            _, status = os.waitpid(pid, 0)  # WNOWAIT above pins this exited child.
            entry.update(returncode=os.waitstatus_to_exitcode(status), reaped=True)
            for proc in processes:
                if proc.pid == pid:
                    proc.returncode = os.waitstatus_to_exitcode(status)
        if not overdue and time.monotonic() >= deadline:
            overdue = True
            # The caller waits boundedly; never kill the sole ownership anchor.
            print('owned children still pending after SIGKILL; retaining subreaper',
                  file=sys.stderr, flush=True)
        time.sleep(0.01)


def _supervise_reaper(command, run_dir, coordinator):
    """Private, exec-created Linux subreaper; stdin EOF means owner went away."""
    cancelled = False

    def cancel(*_):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)  # Never auto-reap owned PIDs.
    processes = []
    listener = None
    children = Path(f'/proc/self/task/{os.getpid()}/children')
    result = {}
    run_dir = Path(run_dir)
    receipt = {'reaper': _owned_identity(os.getpid()), 'owned': {},
               'bind': None, 'cleanup': 'pending', 'coordinator_fd_abi': 'DMTCP-4.2.0:821'}
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        children.read_text()  # Fail before spawning if procfs cannot support cleanup.
        dump(run_dir / 'supervisor.json', receipt)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(('127.0.0.1', 0))
        listener.listen(128)
        port = str(listener.getsockname()[1])
        receipt['bind'] = ['127.0.0.1', int(port)]
        # Pinned DMTCP v4.2.0: include/protectedfds.h defines 820 + 1;
        # src/dmtcp_coordinator.cpp:1738 adopts this already-listening fd.
        # There is no bind-host flag. Without this fd it binds INADDR_ANY.
        # --coord-port below is already held by our non-reusable socket, so an
        # incompatible backend that ignores this ABI fails binding, not open-wide.
        coord_fd = 821
        if listener.fileno() != coord_fd:
            try:
                os.fstat(coord_fd)
            except OSError as exc:
                if exc.errno != errno.EBADF:  # Never overwrite an occupied fd.
                    raise
            else:
                raise RuntimeError('coordinator inherited fd is already occupied')
            os.dup2(listener.fileno(), coord_fd, inheritable=True)
        try:
            coord = subprocess.Popen(
                [coordinator, '--coord-port', port, '--port-file', str(run_dir / 'port'),
                 '--tmpdir', str(run_dir / 'tmp'), '--interval', '0',
                 '--coord-logfile', str(run_dir / 'coordinator.log')],
                pass_fds=(coord_fd,), stdin=subprocess.DEVNULL, cwd=run_dir,
                stdout=sys.stderr, stderr=sys.stderr)
            processes.append(coord)
            receipt['coordinator_pid'] = coord.pid
            receipt['owned'][str(coord.pid)] = _owned_identity(coord.pid)
        finally:
            if listener.fileno() != coord_fd:
                os.close(coord_fd)
        # Retain our listener until every child is gone: even if the coordinator
        # dies, this port cannot be reused by an unrelated computation.
        opts = ['--join-coordinator', '--coord-host', '127.0.0.1', '--coord-port', port,
                '--tmpdir', str(run_dir / 'tmp')]
        app = subprocess.Popen([command[0], *opts, *command[1:]],
                               stdin=subprocess.DEVNULL, stdout=sys.stderr,
                               stderr=sys.stderr, start_new_session=True)
        processes.append(app)
        receipt['application_pid'] = app.pid
        receipt['owned'][str(app.pid)] = _owned_identity(app.pid)
        dump(run_dir / 'supervisor.json', receipt)
        while True:
            rc = app.poll()
            if rc is not None:
                result['returncode'] = rc
                receipt['owned'][str(app.pid)].update(returncode=rc, reaped=True)
                break
            if coord.poll() is not None:
                receipt['owned'][str(coord.pid)].update(returncode=coord.returncode, reaped=True)
                raise RuntimeError(f'owned coordinator exited rc={coord.returncode}')
            if cancelled or select.select([sys.stdin], [], [], 0.02)[0]:
                raise RuntimeError('supervision cancelled by owner')
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, OSError):
            result['oserror'] = [exc.errno, exc.strerror, exc.filename]
    finally:
        # Signal the cleanup phase without a pipe-buffer limit on the final
        # result. The owner then applies its cleanup deadline even with timeout=None.
        try:
            print('', flush=True)
        except OSError as exc:
            result['cleanup_error'] = str(exc)
        while True:
            try:
                _reap_owned_children(children, processes, receipt)
                receipt['cleanup'] = 'complete'
                break
            except BaseException as exc:
                # Even a cleanup error must not orphan the computation. The
                # caller times out, while this ownership anchor retries.
                result['cleanup_error'] = str(exc)
                time.sleep(0.02)
        if listener is not None:
            listener.close()
        receipt['result'] = result
        try:
            dump(run_dir / 'supervisor.json', receipt)
        except OSError as exc:
            result['cleanup_error'] = str(exc)
    print(json.dumps(result), flush=True)


def supervise(command, env, run_dir, bin_dir, timeout):
    """Own a loopback coordinator and all locally launched/restored descendants.

    Requires Linux procfs/subreapers and a main-thread caller. DMTCP may restore
    virtual session/process-group IDs, so neither is used as ownership evidence.
    Restore must fork locally below dmtcp_restart (including double-forked roots),
    not delegate creation to an external service. As with any bounded SIGKILL
    cleanup, the kernel must allow tasks to exit (no indefinitely stuck D state).
    On a cleanup overrun the caller fails, but the reaper retains ownership and
    keeps trying rather than orphaning children by killing their sole reaper.
    """
    tmp = run_dir / 'tmp'
    tmp.mkdir(mode=0o700)
    coordinator = backend(bin_dir, 'dmtcp_coordinator')
    bootstrap = ('import json, runpy, sys; '
                 'runpy.run_path(sys.argv[1])["_supervise_reaper"]'
                 '(*json.loads(sys.argv[2]))')
    args = [sys.executable, '-I', '-S', '-c', bootstrap, str(Path(__file__).resolve()),
            json.dumps([command, str(run_dir), coordinator])]
    proc = None
    cleanup_error = None
    previous = {}

    def interrupt(signum, _):
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    with (run_dir / 'backend.log').open('ab') as log, tempfile.TemporaryFile(dir=run_dir) as status:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, interrupt)
            # Do not lose the Popen ownership handle to an asynchronous exception
            # between fork and assignment. The reaper unblocks these on startup.
            mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
            try:
                proc = subprocess.Popen(args, env=env, stdin=subprocess.PIPE,
                                        stdout=status, stderr=log,
                                        start_new_session=True)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, mask)
            deadline = None if timeout is None else time.monotonic() + timeout
            while proc.poll() is None and os.fstat(status.fileno()).st_size == 0:
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(command, timeout)
                time.sleep(0.02)
        finally:
            # Repeated signals must not interrupt cleanup or mask the first error.
            failure = sys.exc_info()[1]
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            try:
                if proc is not None:
                    try:
                        proc.stdin.close()  # Owner death also closes this pipe.
                    except OSError as exc:
                        cleanup_error = str(exc)
                    try:
                        proc.wait(timeout=7)
                    except subprocess.TimeoutExpired:
                        cleanup_error = 'cleanup deadline exceeded; subreaper retains ownership and is still reaping'
            except BaseException as exc:
                cleanup_error = str(exc)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
            if cleanup_error:
                if failure is not None:
                    failure.add_note(cleanup_error)
                else:
                    raise RuntimeError(cleanup_error)
        status.seek(0)
        data = status.read().strip()
        if proc.returncode or not data:
            raise RuntimeError(f'owned subreaper failed rc={proc.returncode}; see {run_dir / "backend.log"}')
        result = json.loads(data)
        if 'error' in result:
            if 'oserror' in result:
                error = OSError(*result['oserror'])
            else:
                error = RuntimeError(result['error'])
            if 'cleanup_error' in result:
                error.add_note(result['cleanup_error'])
            raise error
        if 'cleanup_error' in result:
            if result['returncode']:
                try:
                    emit('PM_REWIND_CLEANUP_ERROR', error=result['cleanup_error'], run_dir=str(run_dir))
                except OSError:
                    pass  # Reporting must not replace the application's failure.
            else:
                raise RuntimeError(result['cleanup_error'])
        return result['returncode']


def check_boundary():
    """Reject observable device/IPC state; factories must also avoid hidden clients."""
    protected = ctypes.CDLL(None).dmtcp_is_protected_fd
    protected.argtypes = [ctypes.c_int]
    protected.restype = ctypes.c_int
    files = set()
    protected_maps = set()
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        if target.startswith(('/dev/vfio/', '/dev/accel', '/dev/nvidia', '/dev/dri/', '/dev/kfd')):
            raise RuntimeError(f'device already opened before checkpoint: {target}')
        if protected(int(fd.name)):
            s = fd.stat()
            protected_maps.add((os.major(s.st_dev), os.minor(s.st_dev), s.st_ino))
            continue
        if int(fd.name) > 2:
            if target.startswith(('socket:', 'pipe:', 'anon_inode:')):
                raise RuntimeError(f'external IPC before checkpoint: {fd}: {target}')
            flags = next(s for s in Path(f'/proc/self/fdinfo/{fd.name}').read_text().splitlines()
                         if s.startswith('flags:'))
            if int(flags.split()[1], 8) & os.O_ACCMODE:
                raise RuntimeError(f'writable file before checkpoint: {target}')
            if target.endswith(' (deleted)'):
                raise RuntimeError(f'deleted open dependency: {target}')
            if target == '/dev/urandom':
                s = fd.stat()
                if stat.S_ISCHR(s.st_mode) and s.st_rdev == os.makedev(1, 9):
                    continue  # Read-only Linux entropy input is fresh, never rewound.
            if target.startswith('/') and Path(target).is_file():
                files.add(target)
            else:
                raise RuntimeError(f'unsupported external descriptor before checkpoint: {target}')
    for path in Path('/proc/self/task').glob('*/children'):
        children = path.read_text().strip()
        if children:
            raise RuntimeError(f'factory spawned children before checkpoint: {children}')
    return files, protected_maps


def check_real_pid():
    kernel_pid = int(Path('/proc/self/stat').read_text().split()[0])
    if os.getpid() != kernel_pid:
        raise RuntimeError(f'external-process mode requires kernel PIDs: {os.getpid()} != {kernel_pid}')
    kernel_tid = int(Path('/proc/thread-self/stat').read_text().split()[0])
    gettid = ctypes.CDLL(None).gettid
    gettid.argtypes = []
    gettid.restype = ctypes.c_int
    if gettid() != kernel_tid:
        raise RuntimeError(f'external-process mode requires kernel TIDs: {gettid()} != {kernel_tid}')


_MM_STAT_FIELDS = {'start_code': 26, 'end_code': 27, 'start_data': 45, 'end_data': 46,
                   'start_brk': 47, 'start_stack': 28, 'arg_start': 48, 'arg_end': 49,
                   'env_start': 50, 'env_end': 51}
_ARG_BOUNDS = ('arg_start', 'arg_end', 'env_start', 'env_end')


class _KernelMemoryMap(ctypes.Structure):
    # Linux <linux/prctl.h>, PR_SET_MM_MAP. The bundled backend is x86_64.
    _fields_ = [(name, ctypes.c_uint64) for name in (
        'start_code', 'end_code', 'start_data', 'end_data', 'start_brk', 'brk',
        'start_stack', 'arg_start', 'arg_end', 'env_start', 'env_end')]
    _fields_ += [('auxv', ctypes.c_void_p), ('auxv_size', ctypes.c_uint32),
                 ('exe_fd', ctypes.c_uint32)]


def _kernel_memory_map():
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        raise RuntimeError('kernel argument restoration requires Linux x86_64')
    # Resolve libc itself, not DMTCP's syscall interposer/virtual program break.
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.syscall.argtypes = [ctypes.c_long, ctypes.c_ulong]
    libc.syscall.restype = ctypes.c_long
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p,
                          ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    fields = Path('/proc/self/stat').read_text().rpartition(')')[2].split()
    mm = _KernelMemoryMap(**{name: int(fields[number - 3])
                            for name, number in _MM_STAT_FIELDS.items()},
                          auxv=None, auxv_size=0, exe_fd=0xffffffff)
    return libc, mm


def capture_argument_bounds():
    libc, mm = _kernel_memory_map()
    size = ctypes.c_uint()
    if libc.prctl(35, 15, ctypes.byref(size), 0, 0) != 0:  # PR_SET_MM_MAP_SIZE
        raise OSError(ctypes.get_errno(), 'PR_SET_MM_MAP_SIZE unavailable')
    if size.value != ctypes.sizeof(mm):
        raise RuntimeError('kernel prctl_mm_map ABI differs from the pinned ABI')
    command = Path('/proc/self/cmdline').read_bytes()
    if not command or len(command) != mm.arg_end - mm.arg_start:
        raise RuntimeError('capture requires intact native command-line bounds')
    return {'bounds': {name: getattr(mm, name) for name in _ARG_BOUNDS}, 'cmdline': command}


def restore_argument_bounds(saved):
    # MTCP restores the original stack but leaves kernel arg/env bounds pointing
    # at its unmapped restart stack. Ray's real spt_setup dereferences arg_start.
    # Rebind only those four bounds, before any application continuation/threads.
    # Keep current heap/code/stack metadata; never rewind brk, auxv or exe_file.
    libc, mm = _kernel_memory_map()
    ranges = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        area, flags, *_ = line.split()
        if flags.startswith('r'):
            ranges.append(tuple(int(value, 16) for value in area.split('-')))
    for first, last in (('arg_start', 'arg_end'), ('env_start', 'env_end')):
        start, end = saved['bounds'][first], saved['bounds'][last]
        if not (0 < start <= end and any(lo <= start <= end <= hi for lo, hi in ranges)):
            raise RuntimeError(f'saved {first}/{last} are not in restored readable memory')
        setattr(mm, first, start)
        setattr(mm, last, end)
    if ctypes.string_at(mm.arg_start, mm.arg_end - mm.arg_start) != saved['cmdline']:
        raise RuntimeError('restored native argv bytes differ from capture')
    pointer, size = ctypes.byref(mm), ctypes.sizeof(mm)
    # SYS_brk(0), from the real libc, at this quiescent pre-client boundary.
    mm.brk = libc.syscall(12, 0)
    if libc.prctl(35, 14, pointer, size, 0) != 0:  # PR_SET_MM_MAP
        raise OSError(ctypes.get_errno(), 'cannot restore kernel argument bounds')
    _, actual = _kernel_memory_map()
    if (any(getattr(actual, name) != value for name, value in saved['bounds'].items())
            or Path('/proc/self/cmdline').read_bytes() != saved['cmdline']):
        raise RuntimeError('kernel did not publish restored native argv bounds')
    emit('PM_REWIND_KERNEL_ARGS', bounds=saved['bounds'],
         cmdline_bytes=len(saved['cmdline']), cmdline_sha256=hashlib.sha256(saved['cmdline']).hexdigest())


def capture(config_path):
    """Runs inside DMTCP. Original execution ends here; only restored copies bind."""
    cfg = json.loads(Path(config_path).read_text())
    cache = Path(cfg['cache'])
    if cfg.get('real_pids'):
        check_real_pid()
    t0 = time.monotonic()
    inputs = {}
    for path in cfg['input_files']:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ValueError(f'CPU preparation input must be a regular non-symlink file: {path}')
        inputs[path] = {'stat': file_id(path), 'sha256': digest(path)}
    module, name = cfg['factory'].split(':')
    resume = getattr(importlib.import_module(module), name)()
    required_inputs = getattr(resume, 'rewind_input_files', ())
    if not set(required_inputs).issubset(inputs):
        raise ValueError(f'factory CPU inputs were not declared with --input: {sorted(set(required_inputs) - inputs.keys())}')
    # Bind the native checkpoint interface and its output storage before sealing.
    lib = ctypes.CDLL(None)
    ckpt = lib.dmtcp_checkpoint
    ckpt.argtypes = []
    ckpt.restype = ctypes.c_int
    get_env = lib.dmtcp_get_restart_env
    get_env.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t]
    get_env.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(65536)
    if not callable(resume):
        raise TypeError('factory must return a callable resume(argv)')
    files, protected_maps = check_boundary()
    files.update(inputs)
    files.update((str(Path(__file__).resolve()), sys.executable, str(Path(sys.executable).resolve())))
    files.update(backend(cfg['bin_dir'], name)
                 for name in ('dmtcp_launch', 'dmtcp_restart', 'dmtcp_coordinator', 'mtcp_restart'))
    files.update(p for p in cfg['frozen_env'].get('LD_PRELOAD', '').replace(':', ' ').split()
                 if Path(p).is_file())
    for mod in tuple(sys.modules.values()):
        path = getattr(mod, '__file__', None)
        if path and Path(path).is_file():
            files.add(str(Path(path).absolute()))
    readonly_shared = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith('/'):
            path = columns[5]
            device = tuple(int(n, 16) for n in columns[3].split(':'))
            if (*device, int(columns[4])) in protected_maps:
                continue  # Exact retained DMTCP FD identity, never a filename-prefix exemption.
            if path == '/dev/zero (deleted)':  # Linux MAP_SHARED|MAP_ANONYMOUS, not an external file
                continue
            if columns[1][-1] == 's':
                # v4.2.0 fileconnlist.cpp:423 creates sidecars iff access(W_OK).
                # Immutable system caches (e.g. glibc gconv) need no sidecar.
                if columns[1][1] == 'w' or os.access(path, os.W_OK):
                    raise RuntimeError(f'external shared mapping before checkpoint: {path}')
                readonly_shared.add(path)
            if path.endswith(' (deleted)') or not Path(path).is_file():
                raise RuntimeError(f'unsupported mapped dependency: {path}')
            files.add(path)
    for directory in sys.path:
        for pattern in ('*.dist-info/METADATA', '*.dist-info/entry_points.txt'):
            files.update(str(p.absolute()) for p in Path(directory).glob(pattern))
    kernel_args = capture_argument_bounds() if cfg.get('real_pids') else None
    info = {'prepare_s': time.monotonic() - t0, 'python': sys.version,
            'native_threads': [p.read_text().strip() for p in Path('/proc/self/task').glob('*/comm')],
            'readonly_shared': sorted(readonly_shared),
            'files': {p: {'stat': file_id(p), 'sha256': digest(p)} for p in sorted(files)}}
    for path, expected in inputs.items():
        if Path(path).is_symlink() or info['files'][path] != expected:
            raise RuntimeError(f'CPU preparation input changed while constructing saved state: {path}')
    info['input_files'] = sorted(inputs)
    dump(cache / 'prepared.json', info)
    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, 'r+b', buffering=0) as null:
        for fd in (0, 1, 2):
            os.dup2(null.fileno(), fd)
    result = ckpt()
    if result == 1:
        os._exit(0)
    # Pinned 4.2.0 include/protectedfds.h: restart keeps its fresh stderr at 828.
    # Restore diagnostics BEFORE parsing context; the snapshot's stdio is /dev/null.
    os.dup2(828, 1)
    os.dup2(828, 2)
    if result != 2:
        raise RuntimeError(f'dmtcp_checkpoint returned {result}, expected restart=2')
    if cfg.get('real_pids'):
        check_real_pid()
    if get_env(CONTEXT_ENV.encode(), buf, len(buf)) != 0:
        raise RuntimeError('missing fresh rewind context')
    context = Path(os.fsdecode(buf.value))
    with open(context.with_name('output.log'), 'ab', buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    ctx = json.loads(context.read_text())
    if kernel_args is not None:
        restore_argument_bounds(kernel_args)
    for key in cfg['late_env']:
        os.environ.pop(key, None)
    if set(ctx['env']) - set(cfg['late_env']):
        raise ValueError('context attempts to modify frozen import-time environment')
    os.environ.update(ctx['env'])
    os.environ['_PM_REWIND_RUN_DIR'] = str(Path(ctx['log']).parent)
    os.chdir(ctx['cwd'])
    emit('PM_REWIND_RESTORED', run_id=ctx['run_id'],
         restore_s=time.monotonic() - ctx['t0'], pid=os.getpid(),
         real_pids=cfg.get('real_pids', False))
    rc = resume(ctx['argv'])
    sys.stdout.flush()
    sys.stderr.flush()
    return 0 if rc is None else rc


def load_cache(cache, verify=False):
    cache = trusted_dir(cache)
    if (cache / 'manifest.json').is_symlink():
        raise ValueError('manifest must not be a symlink')
    man = json.loads((cache / 'manifest.json').read_text())
    if man['format'] != FORMAT or man['host'] != json.loads(json.dumps(host_id())):
        raise ValueError('checkpoint format/host ABI mismatch; prepare a new template')
    if len(man['images']) != 1:
        raise ValueError('expected exactly one process image')
    if man['controller_sha256'] != digest(__file__):
        raise ValueError('checkpoint controller changed; prepare a new template')
    for path in man['config']['input_files']:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ValueError(f'CPU preparation input changed type: {path}')
        if digest(path) != man['prepared']['files'][path]['sha256']:
            raise ValueError(f'CPU preparation input content changed: {path}')
    for path in man['prepared']['readonly_shared']:
        if os.access(path, os.W_OK):
            raise ValueError(f'shared dependency became writable: {path}')
    for path, expected in man['prepared']['files'].items():
        if file_id(path) != expected['stat'] or (verify and digest(path) != expected['sha256']):
            raise ValueError(f'installed runtime changed: {path}; prepare a new template')
    for ent in man['images']:
        path = cache / ent['name']
        s = path.stat()
        if (path.parent != cache or path.is_symlink() or not path.is_file()
                or s.st_size != ent['bytes'] or s.st_mtime_ns != ent['mtime_ns']
                or s.st_uid != os.getuid() or s.st_mode & 0o777 != 0o400):
            raise ValueError(f'invalid checkpoint image: {path}')
        if verify and digest(path) != ent['sha256']:
            raise ValueError(f'checkpoint checksum mismatch: {path}')
    if file_id(backend(man['bin_dir'], 'dmtcp_restart')) != man['backend_id']:
        raise ValueError('DMTCP backend changed; prepare a new template')
    return cache, man


def prepare(a):
    version = subprocess.check_output([backend(a.dmtcp_bin, 'dmtcp_launch'), '--version'], text=True)
    if version.splitlines()[0] != 'dmtcp_launch (DMTCP) 4.2.0':
        raise ValueError('this controller requires the DMTCP 4.2.0 protected-fd ABI')
    cache = trusted_dir(a.cache, creating=True)
    frozen = {k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ}
    frozen.update(env_pairs(a.env))
    frozen['PYTHONUNBUFFERED'] = '1'
    late = sorted(set(a.late_env))
    if any(k.startswith(('DMTCP_', '_PM_REWIND_')) or not k for k in late):
        raise ValueError('reserved late-bound environment key')
    if set(late) & frozen.keys():
        raise ValueError('late-bound keys cannot also be frozen')
    inputs = sorted({str(Path(path).absolute()) for path in a.input})
    for path in inputs:
        if Path(path).is_symlink() or not Path(path).is_file():
            raise ValueError(f'CPU preparation input must be a regular non-symlink file: {path}')
    config = {'cache': str(cache), 'factory': a.factory, 'frozen_env': frozen,
              'late_env': late, 'python': str(Path(a.python).absolute()),
              'bin_dir': str(Path(a.dmtcp_bin).resolve()), 'real_pids': a.real_pids,
              'input_files': inputs}
    if (cache / 'manifest.json').exists():
        _, man = load_cache(cache)
        if config != man['config']:
            raise ValueError('existing template has a different prepare contract')
        emit('PM_REWIND_CACHE_HIT', cache=str(cache))
        return 0
    cache.mkdir(mode=0o700)  # never overwrite an incomplete/unknown template
    run_dir = cache / 'build'
    run_dir.mkdir(mode=0o700)
    dump(cache / 'config.json', config)
    env = dict(frozen)
    env['DMTCP_GZIP'] = '0'
    env['DMTCP_QUIET'] = '2'
    t0 = time.monotonic()
    launch = [backend(a.dmtcp_bin, 'dmtcp_launch')]
    if a.real_pids:
        launch.append('--disable-pid-plugin')
    rc = supervise([*launch, '--no-gzip', '--ckptdir', str(cache), config['python'],
                    str(Path(__file__).resolve()), '_capture', str(cache / 'config.json')],
                   env, run_dir, a.dmtcp_bin, a.timeout)
    if rc:
        raise RuntimeError(f'capture failed rc={rc}; see {run_dir / "backend.log"}')
    images = sorted(cache.glob('*.dmtcp'))
    if len(images) != 1:
        raise RuntimeError(f'expected one CPU template, found {len(images)} images')
    for p in images:
        p.chmod(0o400)
    man = {'format': FORMAT, 'host': host_id(), 'config': config,
           'controller_sha256': digest(__file__), 'bin_dir': str(Path(a.dmtcp_bin).resolve()),
           'backend_id': file_id(backend(a.dmtcp_bin, 'dmtcp_restart')),
           'prepared': json.loads((cache / 'prepared.json').read_text()),
           'images': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p),
                       'mtime_ns': p.stat().st_mtime_ns} for p in images],
           'capture_s': time.monotonic() - t0}
    dump(cache / 'manifest.pending', man)
    (cache / 'manifest.pending').replace(cache / 'manifest.json')
    emit('PM_REWIND_SAVED', cache=str(cache), capture_s=man['capture_s'],
         prepare_s=man['prepared']['prepare_s'], bytes=sum(p.stat().st_size for p in images))
    return 0


def run(a):
    t0 = time.monotonic()
    cache, man = load_cache(a.cache)
    changes = env_pairs(a.env)
    if set(changes) - set(man['config']['late_env']):
        raise ValueError('only --late-env keys declared at prepare may change on rewind')
    run_dir = trusted_dir(a.run_dir, creating=True)
    run_dir.mkdir(mode=0o700)
    ctx = {'run_id': run_dir.name, 'argv': a.argv, 'env': changes, 'cwd': os.getcwd(),
           'log': str(run_dir / 'output.log'), 't0': t0}
    context = run_dir / 'context.json'
    dump(context, ctx)
    # v4.2.0's restart-env scanner needs this entry first (not Python's saved environ).
    env = {CONTEXT_ENV: str(context), 'DMTCP_QUIET': '2'}
    env.update({k: os.environ[k] for k in ('PATH', 'HOME', 'LANG') if k in os.environ})
    rc = 1
    try:
        rc = supervise([backend(man['bin_dir'], 'dmtcp_restart'), '--ckptdir', str(run_dir),
                        *[str(cache / ent['name']) for ent in man['images']]],
                       env, run_dir, man['bin_dir'], a.timeout)
    except subprocess.TimeoutExpired:
        rc = 124
    except KeyboardInterrupt:
        rc = 130
    except SystemExit as e:
        rc = e.code
    finally:
        dump(run_dir / 'result.json', {'rc': rc, 'elapsed_s': time.monotonic() - t0})
    emit('PM_REWIND_EXIT', run_dir=str(run_dir), rc=rc, elapsed_s=time.monotonic() - t0)
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('prepare')
    c.add_argument('cache')
    c.add_argument('--factory', required=True)
    c.add_argument('--python', default=sys.executable)
    c.add_argument('--dmtcp-bin', required=True)
    c.add_argument('--real-pids', action='store_true',
                   help='require the patched kernel-PID backend for external process managers')
    c.add_argument('--env', action='append', default=[])
    c.add_argument('--late-env', action='append', default=[])
    c.add_argument('--input', action='append', default=[],
                   help='immutable CPU preparation input, including files closed before checkpoint')
    c.add_argument('--timeout', type=float, default=180)
    r = sub.add_parser('run')
    r.add_argument('cache')
    r.add_argument('--run-dir', required=True)
    r.add_argument('--env', action='append', default=[])
    r.add_argument('--timeout', type=float, default=None)
    v = sub.add_parser('verify')
    v.add_argument('cache')
    internal = sub.add_parser('_capture', help=argparse.SUPPRESS)
    internal.add_argument('config')
    # Separate resume argv from controller options explicitly; no argv re-parsing.
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        pos = argv.index('--')
        argv, extra = argv[:pos], argv[pos + 1:]
    a = p.parse_args(argv)
    if extra and a.cmd != 'run':
        p.error('trailing arguments are only accepted by run')
    a.argv = extra
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.cmd == '_capture':
        return capture(a.config)
    if a.cmd == 'prepare':
        return prepare(a)
    if a.cmd == 'run':
        return run(a)
    cache, man = load_cache(a.cache, verify=True)
    emit('PM_REWIND_VERIFIED', cache=str(cache), images=len(man['images']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
