"""Measure every install profile from its lock: package count, download size, installed size and licences.

For each profile (core and every pyproject extra), each device profile it needs (device-bound profiles: cpu, cuda126,
cuda132, mps where the platform has them; other profiles: one device-independent lock) and each target platform, this
script compiles a PEP 751 lock with ``uv pip compile`` for CPython 3.12 from exactly the requirements ``glossapi install``
would request (``glossapi.package_catalog.lock_requirements``), then measures it:

- package count and download bytes: the lock's own per-wheel ``size`` fields (one wheel per package for the target);
- installed bytes: the sum of the uncompressed file sizes in each wheel's ZIP central directory, read with HTTP range
  requests (no wheel is downloaded); packages without a wheel (sdist or VCS) are listed as unmeasured;
- licences: each release's PyPI metadata (``license_expression``, else a short ``license``, else the classifiers).

Output: the locks ``pylock.<profile>-<device>-<target>.toml`` in ``--locks-dir`` (evidence, not shipped; their SHA-256
is recorded) and the packaged summary ``src/glossapi/data/package_catalog/measured.json`` (read by
``glossapi capabilities --catalog`` and the LLM guide).
Each lock records ``inputs_sha256``; the catalog marks a lock stale when the requirements change, and the tests fail.

Usage: PYTHONPATH=src python tools/catalog/measure_profiles.py --locks-dir DIR [--uv PATH]   (network: PyPI and the PyTorch index)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import tomllib
from typing import Any
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from glossapi import package_catalog as catalog  # noqa: E402

PYTHON = '3.12'
TARGETS = {
    'linux-x86_64': dict(system='Linux', machine='x86_64', uv_platform='x86_64-manylinux_2_28'),
    'macos-arm64': dict(system='Darwin', machine='arm64', uv_platform='aarch64-apple-darwin', macos='14.0'),
}
PYPI = 'https://pypi.org/simple'
OUTPUT = REPO / 'src' / 'glossapi' / 'data' / 'package_catalog' / 'measured.json'
AGENT = {'User-Agent': 'glossapi-catalog-measure/1'}


def fetch(url, headers=None, timeout=60):
    request = urllib.request.Request(url, headers={**AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.headers


def tail(url, size, length):
    start = max(0, size - length)
    data, _ = fetch(url, {'Range': f'bytes={start}-{size - 1}'})
    if len(data) != size - start:
        raise RuntimeError(f'range request not honoured for {url}')
    return start, data


def content_length(url):
    """A wheel's size when its lock entry has none (the PyTorch index publishes no sizes): Content-Range of a 1-byte range."""
    _, headers = fetch(url, {'Range': 'bytes=0-0'})
    found = re.search(r'/(\d+)$', headers.get('Content-Range') or '')
    if not found:
        raise RuntimeError('no Content-Range for ' + url)
    return int(found.group(1))


def installed_bytes(url, size):
    """Sum of uncompressed sizes in the wheel's central directory (ZIP and ZIP64), via range requests only."""
    start, data = tail(url, size, 65536 + 22)
    end = data.rfind(b'PK\x05\x06')
    if end < 0:
        raise RuntimeError('no end of central directory record')
    entries, cd_size, cd_offset = struct.unpack('<HII', data[end + 10:end + 20])
    if cd_offset == 0xFFFFFFFF or entries == 0xFFFF:
        locator = data.rfind(b'PK\x06\x07', 0, end)
        zip64_offset = struct.unpack('<Q', data[locator + 8:locator + 16])[0]
        record_start, record = (start, data) if zip64_offset >= start else (zip64_offset, fetch(url, {'Range': f'bytes={zip64_offset}-{zip64_offset + 55}'})[0])
        offset = zip64_offset - record_start
        entries, cd_size, cd_offset = struct.unpack('<QQQ', record[offset + 32:offset + 56])
    if cd_offset >= start:
        directory = data[cd_offset - start:cd_offset - start + cd_size]
    else:
        directory, _ = fetch(url, {'Range': f'bytes={cd_offset}-{cd_offset + cd_size - 1}'}, timeout=120)
    total, position, count = 0, 0, 0
    while position + 46 <= len(directory) and directory[position:position + 4] == b'PK\x01\x02':
        uncompressed, = struct.unpack('<I', directory[position + 24:position + 28])
        name_length, extra_length, comment_length = struct.unpack('<HHH', directory[position + 28:position + 34])
        if uncompressed == 0xFFFFFFFF:
            extra = directory[position + 46 + name_length:position + 46 + name_length + extra_length]
            cursor = 0
            while cursor + 4 <= len(extra):
                tag, length = struct.unpack('<HH', extra[cursor:cursor + 4])
                if tag == 1:
                    uncompressed, = struct.unpack('<Q', extra[cursor + 4:cursor + 12])
                    break
                cursor += 4 + length
        total += uncompressed
        count += 1
        position += 46 + name_length + extra_length + comment_length
    if count != entries:
        raise RuntimeError(f'central directory has {count} entries, expected {entries}')
    return total


def licence(name, version):
    public = version.split('+', 1)[0]
    try:
        data, _ = fetch(f'https://pypi.org/pypi/{name}/{public}/json')
    except Exception as error:  # a release missing from PyPI (an index-only build)
        return None, repr(error)
    info = json.loads(data)['info']
    if info.get('license_expression'):
        return info['license_expression'], None
    text = (info.get('license') or '').strip()
    if text and len(text) <= 60 and '\n' not in text:
        return text, None
    classifiers = [item.split(' :: ')[-1] for item in info.get('classifiers') or [] if item.startswith('License ::')]
    classifiers = [item for item in classifiers if item != 'OSI Approved']
    return ('; '.join(classifiers) if classifiers else None), None


def vcs_licence(item):
    """A GitHub VCS source: the licence its pyproject declares at the pinned commit, else 'not declared'."""
    found = re.match(r'(?:git\+)?https://github.com/([^/]+/[^/@.]+)', item.get('url') or '')
    if not found or not item.get('commit'):
        return 'unknown (VCS source)'
    try:
        data, _ = fetch(f'https://raw.githubusercontent.com/{found.group(1)}/{item["commit"]}/pyproject.toml')
    except Exception:
        return 'unknown (VCS source)'
    value = tomllib.loads(data.decode()).get('project', {}).get('license')
    if isinstance(value, dict):
        value = value.get('text') or value.get('file')
    return value or 'not declared (no licence in the pyproject at the pinned commit)'


def compile_lock(uv, requirements, torch, target, path):
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / 'requirements.in'
        source.write_text('\n'.join(requirements) + '\n')
        output = Path(temporary) / path.name
        command = [uv, 'pip', 'compile', '--quiet', '--no-header', '--python-version', PYTHON,
                   '--python-platform', target['uv_platform'], '--index-url', PYPI]
        if torch and torch['index_url'] != PYPI:
            command += ['--extra-index-url', torch['index_url'], '--index-strategy', 'unsafe-best-match']
        command += [source.name, '-o', output.name]
        environment = {**os.environ, **({'MACOSX_DEPLOYMENT_TARGET': target['macos']} if target.get('macos') else {})}
        done = subprocess.run(command, cwd=temporary, capture_output=True, text=True, timeout=900, env=environment)
        if done.returncode:
            return None, ' '.join(command[1:]), done.stderr.strip()[-2000:]
        shutil.copyfile(output, path)
        return tomllib.loads(output.read_text()), ' '.join(command[1:]), None


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0])
    parser.add_argument('--uv', default=shutil.which('uv'))
    parser.add_argument('--locks-dir', required=True, type=Path)
    args = parser.parse_args()
    if not args.uv:
        raise SystemExit('uv is required')
    resolver = subprocess.run([args.uv, '--version'], capture_output=True, text=True, check=True).stdout.strip()
    source = catalog.requirement_source()
    assert source.kind == 'source_tree', 'run from the source checkout'
    LOCKS = args.locks_dir.resolve()
    LOCKS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for profile in catalog.profile_names(source):
        if catalog.KINDS.get(profile) == 'alias':
            continue  # an alias is measured by the locks of the profile it names
        bound = catalog.device_bound(source, profile)
        for target_id, target in TARGETS.items():
            for device in (catalog.DEVICES if bound else (None,)):
                if device is not None:
                    try:
                        torch = catalog.torch_build(device, target['system'], target['machine'])
                    except catalog.CatalogError:
                        continue
                else:
                    torch = None
                jobs.append((profile, device, target_id, torch))
    locks = {}
    for profile, device, target_id, torch in jobs:
        target = TARGETS[target_id]
        identity = dict(system=target['system'], machine=target['machine'])
        requirements = catalog.lock_requirements(source, profile, device, identity)
        lock_id = f'{profile}@{device or "any"}@{target_id}'
        path = LOCKS / f'pylock.{profile}-{device or "any"}-{target_id}.toml'
        parsed, command, error = compile_lock(args.uv, requirements, torch, target, path)
        row: dict[str, Any] = dict(id=lock_id, profile=profile, device=device, target=target_id, python=PYTHON, status='resolved' if parsed else 'failed',
                   error=error, command='uv ' + command,
                   inputs_sha256=catalog.inputs_sha256(source, profile, device, identity),
                   lock_file=path.name if parsed else None,
                   lock_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if parsed else None, packages=[])
        if parsed:
            for item in parsed.get('packages', []):
                wheels = item.get('wheels') or []
                entry: dict[str, Any] = dict(name=re.sub(r'[-_.]+', '-', item['name']).lower(), version=item.get('version'))
                if wheels:
                    wheel = wheels[0]
                    entry.update(source='wheel', url=wheel['url'], download_bytes=wheel.get('size'))
                elif item.get('vcs'):
                    entry.update(source='vcs', url=item['vcs'].get('url'), commit=item['vcs'].get('commit-id'), download_bytes=None)
                elif item.get('sdist'):
                    entry.update(source='sdist', url=item['sdist'].get('url'), download_bytes=item['sdist'].get('size'))
                else:
                    entry.update(source='unknown', download_bytes=None)
                row['packages'].append(entry)
        locks[lock_id] = row
        print(lock_id, row['status'], len(row['packages']), error or '', flush=True)
    sizes, licences, problems = {}, {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        unsized = sorted({item['url'] for row in locks.values() for item in row['packages']
                          if item.get('source') == 'wheel' and item.get('download_bytes') is None})
        lengths = {}
        for url, future in [(url, pool.submit(content_length, url)) for url in unsized]:
            try:
                lengths[url] = future.result()
            except Exception as error:
                problems['size:' + url] = repr(error)
        for row in locks.values():
            for item in row['packages']:
                if item.get('source') == 'wheel' and item.get('download_bytes') is None and item['url'] in lengths:
                    item['download_bytes'] = lengths[item['url']]
                    item['size_basis'] = 'content_length'
        wheels = {(item['url'], item['download_bytes']) for row in locks.values() for item in row['packages']
                  if item.get('source') == 'wheel' and item.get('download_bytes')}
        releases = {(item['name'], item['version']) for row in locks.values() for item in row['packages'] if item.get('version') and item.get('source') != 'vcs'}
        futures = {pool.submit(installed_bytes, url, size): url for url, size in wheels}
        for future in concurrent.futures.as_completed(futures):
            try:
                sizes[futures[future]] = future.result()
            except Exception as error:
                problems[futures[future]] = repr(error)
        lic = {pool.submit(licence, name, version): (name, version) for name, version in releases}
        for future in concurrent.futures.as_completed(lic):
            value, error = future.result()
            licences[lic[future]] = value
            if error:
                problems['licence:' + '=='.join(lic[future])] = error
    for row in locks.values():
        for item in row['packages']:
            item['installed_bytes'] = sizes.get(item.get('url')) if item.get('source') == 'wheel' else None
            item['licence'] = licences.get((item['name'], item['version'])) or (vcs_licence(item) if item.get('source') == 'vcs' else None)
            if item.get('source') != 'vcs':
                item.pop('url', None)
    releases = {}
    for row in locks.values():
        table = releases.setdefault(row['target'], {})
        names = []
        for item in row.pop('packages'):
            key = f"{item['name']}=={item['version']}"
            names.append(key)
            table[key] = [item['source'], item.get('download_bytes'), item.get('installed_bytes'), item.get('licence')]
        row['packages'] = names
    result = dict(format=2, measured_at=datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
                  tool='tools/catalog/measure_profiles.py', resolver=resolver, python=PYTHON,
                  targets={key: dict(value) for key, value in TARGETS.items()},
                  size_basis=dict(download_bytes='sum of the lock\'s per-wheel size fields (one wheel per package for the target)',
                                  installed_bytes='sum of uncompressed file sizes in each wheel\'s ZIP central directory (range requests); '
                                                  'sdist/VCS packages are unmeasured: totals cover the measured packages and list the rest',
                                  added='packages of the lock that are not in the core lock of the same target'),
                  release_fields=['source', 'download_bytes', 'installed_bytes', 'licence'],
                  problems=problems, releases={target: dict(sorted(table.items())) for target, table in sorted(releases.items())}, locks=locks)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, separators=(',', ':')).replace('},"', '},\n"') + '\n')
    print(json.dumps(dict(locks=len(locks), failed=[key for key, row in locks.items() if row['status'] != 'resolved'],
                          wheels=len(wheels), measured=len(sizes), problems=len(problems)), indent=1))


if __name__ == '__main__':
    main()
