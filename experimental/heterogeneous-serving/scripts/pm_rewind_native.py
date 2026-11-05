#!/usr/bin/env python3
"""Transport the pinned native backend, not checkpoints or system libraries.

Only the inspected build is accepted; rebuilding requires requalification and new
pins. Install before preparing snapshots, then keep the destination fixed. This
is artifact verification, not worker ABI/runtime qualification.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import sys
import tarfile


ALLOCATOR = 'allocator/libtcmalloc_minimal.so.4'
# Isolated real-pid-exec/install build: guard fix, real PID/TID restoration, and
# explicit core SharedData mapping after exec. Pins are not runtime qualification.
PINS = dict(line.split() for line in '''
bin/dmtcp_command af94fd08bdd5e1416aaf29b42aa28ba420f26d8688227c4dae0e5e6351cd410d
bin/dmtcp_coordinator 9e3791269090894aebe214cfe9ac3c6b1443cb22b40393924c00cd282d23abe7
bin/dmtcp_get_libc_offset 4fc7ed481417943aa5f513851ff1ffc29e269f49ded770dcae91899750eaab85
bin/dmtcp_launch 24913c65f4d909edc0e6664fb25a5f1192ddccdae907e4ef2005b7f56a53f27a
bin/dmtcp_nocheckpoint f4587f863e89d331cc1621230313e009aa2467363b617c320e884ac7fea9d820
bin/dmtcp_restart fbbedd363be27bc74d7a710d234055a7c57a8be337bd077a043c946ade9ca885
bin/dmtcp_ssh 9aa3b1a4463ca86eedd3f6f1b94f71075f672f8aa669c4213de67092c6f381cf
bin/dmtcp_sshd 247cef9b8e5d1ea6553a783e76c800ba890edf7d6b4649bec792d888265cb5ee
bin/mtcp_restart a5194b2bf10f3a26855af4701e019439fc5d9d4b95a0bde2692b73f3bc0d6936
lib/dmtcp/libdmtcp.so 6f2151cb841b58160751ccaaedd14e5f71bb0eebefc776dc8fcdc25645d792da
lib/dmtcp/libdmtcp_alloc.so 278ec6b531bbdb3a9aeb2b33ff6826fd0739b90133c6a21fe3b01dcb5b7445c9
lib/dmtcp/libdmtcp_dl.so 75108a820ddd0ec0e54d2c23499be4695e6a9c9665884e6b41f8e94d381307eb
lib/dmtcp/libdmtcp_ipc.so 15a145e2f35a8e7b97e2eb4ff47f4cd899abcc1958343163008ebf550ace53c5
lib/dmtcp/libdmtcp_modify-env.so 3350afff4dbedf37ed4609b0d75982908f723d30bcf4ab01d8103500f51a77e0
lib/dmtcp/libdmtcp_pid.so 7d83dd596c46021e92fefb31d15e740268ef2f75ed7d62d8556e923d5445ce1b
lib/dmtcp/libdmtcp_svipc.so b6c0625978a3ee53b2ae2e9be10e50fc840a8133b02ac76be581783334e0d5e3
lib/dmtcp/libdmtcp_timer.so 7dd0cbce4b103429604d1773c07b42171722aa11ee4ab79a59ee3999bd79eac7
lib/dmtcp/libdmtcp_unique-ckpt.so 41d76e42d0df66ff294147b4172f9d4dc581b3d5a6ddd6243dab5b1a7fdb8950
allocator/libtcmalloc_minimal.so.4 572af05b75e2366a3e8c06c29d3a04c0a538c9635c9957f2808846730ab81da5
'''.splitlines() if line)
SOURCE = {
    'backend_version': 'DMTCP 4.2.0',
    'base_revision': 'f8009ce7b4ad211311ca2f72a929b975e4aa1155',
    'patch': 'scripts/pm_rewind_dmtcp.patch',
    'patch_sha256': 'f7ec636cf6929f270be292e241f824f48336575830a0c6e36da13c3078bb1f21',
    'pid_patch': 'scripts/pm_rewind_dmtcp_pid.patch',
    'pid_patch_sha256': 'ee2fc93ec28d6d056e4df6ca00a92c06e9d24dfe155d8331c62e3b52d271bd6a',
    'build_install': '/tmp/pm-rewind-dmtcp/real-pid-exec/install',
    'allocator_package': 'libtcmalloc-minimal4 2.9.1-0ubuntu3 amd64',
    'allocator_resolved_name': 'libtcmalloc_minimal.so.4.5.9',
    'qualification_required': 'CPU/native, external Ray, and target TPU runtime; byte pins are not qualification',
}
# Inspected source: src/util_exec.cpp:563-592 searches prefix/bin and
# prefix/lib{64,}/dmtcp; shareddata.cpp derives prefix from /proc/self/exe.
# dmtcp_launch.cpp:795-836 builds absolute preloads from those relative paths.
# readelf -d on every pinned asset: no RPATH/RUNPATH, only system DT_NEEDED;
# mtcp_restart is static. Pinning ALL bytes prevents an uninspected relocation.
RELOCATION = {
    'layout': 'bin + lib/dmtcp, prefix derived from /proc/self/exe',
    'platform': 'Linux x86_64',
    'system_dependencies': ['ld-linux-x86-64.so.2', 'libc.so.6', 'libm.so.6',
                            'libstdc++.so.6', 'libgcc_s.so.1', 'libatomic.so.1'],
    'scope': 'Native files only; worker ABI must be qualified separately. '
             'Prepare new snapshots after install; do not relocate snapshots.',
}
DIRECTORIES = ('bin', 'lib', 'lib/dmtcp', 'allocator')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def mode(name):
    return 0o644 if name == ALLOCATOR else 0o755


def private_path(path, creating=False):
    """Same owner-only directory policy as pm_rewind.trusted_dir."""
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if creating and component == path and not os.path.lexists(component):
            continue
        s = component.lstat()
        if not stat.S_ISDIR(s.st_mode) or s.st_uid not in (0, os.getuid()):
            raise ValueError(f'untrusted directory: {component}')
        sticky_root = s.st_uid == 0 and s.st_mode & 0o1000
        if s.st_mode & 0o022 and not sticky_root:
            raise ValueError(f'writable directory: {component}')
        if component == path and (s.st_uid != os.getuid() or s.st_mode & 0o077):
            raise ValueError(f'expected owner-only directory: {path}')
    return path


def pack(a):
    root = Path(a.dmtcp_root).resolve(strict=True)
    payload, files = {}, {}
    for name, expected in PINS.items():
        source = (Path(a.allocator) if name == ALLOCATOR else root / name).resolve(strict=True)
        s = source.stat()
        if not stat.S_ISREG(s.st_mode) or stat.S_IMODE(s.st_mode) != mode(name):
            raise ValueError(f'unqualified file type/mode: {source}')
        data = source.read_bytes()
        if sha(data) != expected:
            raise ValueError(f'unqualified source bytes (relocation not established): {source}')
        payload[name] = data
        files[name] = {'sha256': expected, 'mode': mode(name), 'bytes': len(data),
                       'mtime_ns': s.st_mtime_ns, 'source': str(source)}
    manifest = {'format': 1, 'source': SOURCE, 'relocation': RELOCATION, 'files': files}
    payload['manifest.json'] = encoded(manifest)
    out = Path(os.path.abspath(a.output))
    # Exclusive creation: never overwrite any artifact, including a source file.
    with out.open('xb') as raw:
        os.fchmod(raw.fileno(), 0o600)
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for name, data in sorted(payload.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o600 if name == 'manifest.json' else mode(name)
                    tar.addfile(info, io.BytesIO(data))
    return {'bundle': str(out), 'bundle_sha256': sha(out.read_bytes()),
            'manifest_sha256': sha(payload['manifest.json']), 'source': SOURCE}


def load_bundle(path):
    raw = Path(path).read_bytes()
    # Fully decompress to check the gzip trailer/CRC, not just tar's end marker.
    data = gzip.decompress(raw)
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tar:
        for member in tar:
            name = member.name
            # Exact allowlist rejects traversal, absolute names, links, devices,
            # duplicate entries and unneeded files; never use extract/extractall.
            if name not in {*PINS, 'manifest.json'} or name in payload or not member.isfile():
                raise ValueError(f'unsafe or unexpected archive member: {name}')
            expected_mode = 0o600 if name == 'manifest.json' else mode(name)
            if member.mode != expected_mode:
                raise ValueError(f'unexpected archive mode: {name}')
            payload[name] = tar.extractfile(member).read()
    if set(payload) != {*PINS, 'manifest.json'}:
        raise ValueError('incomplete bundle')
    manifest = json.loads(payload['manifest.json'])
    if (manifest.get('format') != 1 or manifest.get('source') != SOURCE
            or manifest.get('relocation') != RELOCATION or set(manifest.get('files', {})) != set(PINS)):
        raise ValueError('unqualified bundle manifest')
    for name, expected in PINS.items():
        entry = manifest['files'][name]
        if (entry.get('sha256') != expected or sha(payload[name]) != expected
                or entry.get('bytes') != len(payload[name]) or entry.get('mode') != mode(name)
                or type(entry.get('mtime_ns')) is not int or entry['mtime_ns'] < 0
                or not isinstance(entry.get('source'), str) or not entry['source'].startswith('/')):
            raise ValueError(f'invalid member identity: {name}')
    return payload, manifest, sha(raw)


def install(a):
    payload, manifest, bundle_id = load_bundle(a.bundle)
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('qualified native artifacts require Linux x86_64')
    dest = private_path(a.dest, creating=True)
    if dest.exists():
        # A receipt alone is not sufficient. Verify every file and directory,
        # but perform no writes/chmod/utime on an identical installation.
        expected = {*payload, *DIRECTORIES}
        if {str(p.relative_to(dest)) for p in dest.rglob('*')} != expected:
            raise ValueError('existing destination is incomplete or has different content')
        for name in DIRECTORIES:
            private_path(dest / name)
        for name, data in payload.items():
            p = dest / name
            s = p.lstat()
            wanted_mode = 0o600 if name == 'manifest.json' else mode(name)
            if (not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid()
                    or stat.S_IMODE(s.st_mode) != wanted_mode or p.read_bytes() != data):
                raise ValueError(f'existing destination differs: {p}')
    else:
        dest.mkdir(mode=0o700)
        for name in DIRECTORIES:
            (dest / name).mkdir(mode=0o700)
        # Receipt last. Interrupted installs remain incomplete and are refused.
        for name in [*PINS, 'manifest.json']:
            p = dest / name
            with p.open('xb') as f:
                f.write(payload[name])
                os.fchmod(f.fileno(), 0o600 if name == 'manifest.json' else mode(name))
            if name != 'manifest.json':
                ns = manifest['files'][name]['mtime_ns']
                os.utime(p, ns=(ns, ns))
    return {'dmtcp_bin': str(dest / 'bin'), 'allocator': str(dest / ALLOCATOR),
            'bundle_sha256': bundle_id, 'manifest_sha256': sha(payload['manifest.json']),
            'source': manifest['source'], 'relocation': manifest['relocation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack', help='package only the pinned native build')
    p.add_argument('--dmtcp-root', required=True)
    p.add_argument('--allocator', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('install', help='verify and install into a new private directory')
    p.add_argument('--bundle', required=True)
    p.add_argument('--dest', required=True)
    a = parser.parse_args()
    try:
        result = pack(a) if a.command == 'pack' else install(a)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, tarfile.TarError) as exc:
        print(f'pm_rewind_native: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
