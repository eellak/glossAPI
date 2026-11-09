"""``glossapi install --device DEV [--package NAME ...] [--yes]``: add optional packages to the running environment.

The default is a dry run: it prints the exact commands (installer, index URLs, requirement lines) and what they would
change, and executes nothing. ``--yes`` runs them, then re-runs the capability check in a fresh process and reports
what changed.

Rules (PLAN rev 5 section 2.4; K3b brief):

- **Exactly what was requested.** The commands carry the requested profiles' own requirement lines
  (``glossapi.package_catalog.install_requirements``), self references such as ``glossapi[download]`` expanded, and
  nothing else: no glossapi reinstall, no other extra, no ``playwright install``, no "helpful" additions. A
  device-bound profile (layout, ocr) first installs the device's torch/torchvision build from its index (the
  installer's own ``torch_build``), and the second command pins the same build so the resolve cannot replace it.
- **Conflicts fail before anything runs**: two device profiles at once, an unknown device or package, a device profile
  this platform does not have, a package that does not support the device, an installed torch build of another device.
- glossAPI itself is never reinstalled; ``install_glossapi.py`` stays the source-checkout installer that builds it.
"""
from __future__ import annotations

from collections import deque
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Final, cast

from .capabilities import canonical_name, capabilities_view, installed_version, module_present
from .package_catalog import (CatalogError, DEVICES, current_target, device_bound, device_support, includes, install_requirements,
                              lock_for, lock_packages, measured, measured_profile, profile_names, requirement_source,
                              requirement_status, torch_build)

PYPI_INDEX: Final[str] = 'https://pypi.org/simple'
INSTALLERS: Final[tuple[str, ...]] = ('auto', 'pip', 'uv')
OUTPUT_TAIL_LINES: Final[int] = 40
# Typed errors of `glossapi install` (code -> exit code, meaning). docs/cli_json_schema.md lists the same codes (tests).
ERRORS: Final[dict[str, tuple[int, str]]] = {
    'device_required': (2, 'No --device was given; choose one device profile: ' + ', '.join(DEVICES) + '.'),
    'conflicting_device_profiles': (2, 'More than one device profile was requested; an environment holds one torch build.'),
    'unknown_device': (2, 'The device profile is not one of ' + ', '.join(DEVICES) + '.'),
    'device_unsupported_on_platform': (2, 'This platform has no such device profile (mps needs Apple Silicon macOS; '
                                          'cuda126/cuda132 need Linux).'),
    'unknown_package': (2, 'A requested package is not in the catalog (`glossapi capabilities --catalog`).'),
    'package_device_unsupported': (2, 'A requested package does not support the device profile (its lock did not resolve '
                                      'or no measured platform has it).'),
    'installed_device_profile_conflict': (2, 'The environment already holds a torch build of another device profile; '
                                             'use a separate environment for this device.'),
    'requirements_unavailable': (2, 'glossapi is neither a source checkout nor an installed distribution, so its '
                                    'profiles cannot be read.'),
    'installer_unavailable': (2, 'Neither pip (in this interpreter) nor uv (on PATH) is available to run --yes.'),
    'install_command_failed': (2, 'An install command exited non-zero; later commands did not run. `data.executed` has '
                                  'the output tail and `data.changes` what changed before the failure.'),
}


def _error(code: str, message: str, **details: Any) -> CatalogError:
    return CatalogError(code, message, details)


def _installer(choice: str, python: str) -> dict[str, Any]:
    pip = module_present('pip')
    uv = shutil.which('uv')
    kind = choice if choice != 'auto' else 'pip' if pip else 'uv' if uv else 'pip'
    available = pip if kind == 'pip' else uv is not None
    base = [python, '-m', 'pip', 'install'] if kind == 'pip' else [uv or 'uv', 'pip', 'install', '--python', python]
    check = [python, '-m', 'pip', 'check'] if kind == 'pip' else [uv or 'uv', 'pip', 'check', '--python', python]
    return dict(kind=kind, requested=choice, available=available, python=python, executable=uv if kind == 'uv' else python,
                install_prefix=base, check=check)


def _distributions() -> dict[str, str]:
    found: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata['Name']
        if name:
            found.setdefault(canonical_name(name), distribution.version)
    return found


def _devices(values: list[str]) -> str:
    requested = list(dict.fromkeys(item.strip() for value in values for item in value.split(',') if item.strip()))
    if not requested:
        raise _error('device_required', ERRORS['device_required'][1])
    if len(requested) > 1:
        raise _error('conflicting_device_profiles', 'Choose one device profile, not ' + ' and '.join(requested),
                     devices=requested)
    device = requested[0]
    if device not in DEVICES:
        raise _error('unknown_device', 'Unknown device profile: ' + device, device=device, known=list(DEVICES))
    return device


def _would_change(rows: list[dict[str, Any]], selected: list[str], bound: dict[str, bool], device: str,
                  system: str, machine: str, source: Any) -> dict[str, Any]:
    data = measured() or {}
    target = current_target(data, system, machine)
    installed = _distributions()
    expected: dict[str, dict[str, Any]] = {}
    locks: list[str] = []
    for name in selected:
        lock = lock_for(data, measured_profile(source, name), device if bound[name] else None, target) if target else None
        if lock is None or lock['status'] != 'resolved':
            continue
        locks.append(cast(str, lock['id']))
        core = lock_for(data, 'core', None, cast(str, target))
        core_names = {item['name'] for item in lock_packages(data, core)} if core is not None else set[str]()
        for item in lock_packages(data, lock):
            if item['name'] not in core_names:  # core packages are installed already; pip keeps satisfied versions
                expected.setdefault(cast(str, item['name']), item)
    new = sorted(name for name in expected if name not in installed)
    changed = sorted(name for name, item in expected.items()
                     if name in installed and installed[name].split('+')[0] != str(item['version']).split('+')[0])
    affected = [expected[name] for name in [*new, *changed]]
    return dict(
        requirements=dict(missing=[row['requirement'] for row in rows if row['status'] == 'missing'],
                          version_mismatch=[row['requirement'] for row in rows if row['status'] == 'version_mismatch'],
                          satisfied=[row['requirement'] for row in rows if row['status'] == 'satisfied']),
        from_lock=dict(basis='the packages the measured locks of the selected profiles add over core, for this platform, '
                             'against the installed distributions; an estimate: the installer resolves for real and may differ',
                       target=target, locks=locks, new=[f"{name}=={expected[name]['version']}" for name in new],
                       changed=[f"{name}: {installed[name]} -> {expected[name]['version']}" for name in changed],
                       download_bytes=sum(cast(int, item.get('download_bytes') or 0) for item in affected),
                       installed_bytes=sum(cast(int, item.get('installed_bytes') or 0) for item in affected),
                       unmeasured=sorted(cast(str, item['name']) for item in affected
                                         if item.get('download_bytes') is None or item.get('installed_bytes') is None)))


def plan_install(devices: list[str], packages: list[str], *, installer: str = 'auto', system: str | None = None,
                 machine: str | None = None, python: str | None = None) -> dict[str, Any]:
    """The install plan (``install --json`` data in dry-run mode). Raises CatalogError with a code from ERRORS."""
    system = system or platform.system()
    machine = machine or platform.machine()
    python = python or sys.executable
    if installer not in INSTALLERS:
        raise _error('invalid_request', 'Installer must be one of ' + ', '.join(INSTALLERS), installer=installer)
    device = _devices(devices)
    build = torch_build(device, system, machine)  # device_unsupported_on_platform, whatever the packages
    source = requirement_source()
    selected = list(dict.fromkeys(item.strip() for value in packages for item in value.split(',') if item.strip()))
    known = profile_names(source)
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise _error('unknown_package', 'Unknown package(s): ' + ', '.join(unknown), packages=unknown, known=known)
    data = measured() or {}
    bound = {name: device_bound(source, name) for name in selected}
    for name in selected:
        support = device_support(source, name, device, data)
        if not support['supported']:
            raise _error('package_device_unsupported', f'{name} does not support the {device} device profile: {support["reason"]}',
                         package=name, device=device, support=support)
    torch = build if any(bound.values()) else None
    if torch is not None:
        have = installed_version('torch')
        suffix = cast(str, torch['suffix'])
        if have is not None and ('+' + have.split('+', 1)[1] if '+' in have else '') != suffix:
            raise _error('installed_device_profile_conflict',
                         f'torch {have} is installed; the {device} profile needs the {suffix or "PyPI"} build. Use a separate '
                         'environment for this device profile.', installed=have, device=device, requested_suffix=suffix)
    requirements = install_requirements(source, selected, torch)
    tool = _installer(installer, python)
    prefix = cast(list[str], tool['install_prefix'])
    commands: list[dict[str, Any]] = []
    if torch is not None:
        argv = [*prefix, *cast(list[str], torch['requirements']), '--index-url', cast(str, torch['index_url'])]
        commands.append(dict(purpose=f'torch/torchvision build for the {device} device profile', argv=argv, shell=shlex.join(argv),
                             index_urls=[torch['index_url']], requirements=list(cast(list[str], torch['requirements']))))
    if requirements:
        argv = [*prefix, '--index-url', PYPI_INDEX, *requirements]
        commands.append(dict(purpose='requirements of ' + ', '.join(selected) + (' (torch build pinned)' if torch is not None else ''),
                             argv=argv, shell=shlex.join(argv), index_urls=[PYPI_INDEX], requirements=requirements))
    rows = [dict(requirement_status(line), package=None) for line in requirements]
    for row in rows:
        row['package'] = next((name for name in selected if row['requirement'] in install_requirements(source, [name], torch)), None)
    warnings: list[dict[str, Any]] = []
    if not selected:
        warnings.append(dict(code='nothing_selected', message='No --package given: nothing to install (core is installed '
                             'with glossapi). Device-bound packages: ' + ', '.join(name for name in known if device_bound(source, name)),
                             stage=None))
    if not tool['available']:
        warnings.append(dict(code='installer_unavailable', message=f"{tool['kind']} is not available here; --yes would fail",
                             stage=None))
    return dict(mode='dry_run', device=device, platform=dict(system=system, machine=machine),
                device_profile=dict(build, applies=torch is not None), packages=selected,
                expanded=[dict(package=name, includes=includes(source, name), device_bound=bound[name]) for name in selected],
                requirements_source=dict(kind=source.kind, location=source.location),
                installer={key: value for key, value in tool.items() if key not in ('install_prefix', 'check')},
                commands=commands, requirements=rows,
                would_change=_would_change(rows, selected, bound, device, system, machine, source),
                executed=None, changes=None, warnings=warnings,
                capabilities_fingerprint=capabilities_view()['fingerprint'])


_SNAPSHOT: Final[str] = r'''
import importlib.metadata, json, re
from glossapi.capabilities import capabilities_view
names = {}
for d in importlib.metadata.distributions():
    n = d.metadata['Name']
    if n:
        names.setdefault(re.sub(r'[-_.]+', '-', n).lower(), d.version)
view = capabilities_view()
print(json.dumps(dict(distributions=names, fingerprint=view['fingerprint'],
                      stages={row['stage']: [row['available'], row['reason_code']] for row in view['stages']})))
'''


def _snapshot(python: str) -> dict[str, Any]:
    """Distributions and stage availability in a fresh interpreter (nothing cached from before the install)."""
    done = subprocess.run([python, '-c', _SNAPSHOT], capture_output=True, text=True, timeout=300)
    if done.returncode:
        raise _error('read_error', 'The capability re-check failed: ' + done.stderr.strip()[-500:])
    return cast(dict[str, Any], json.loads(done.stdout.strip().splitlines()[-1]))


def _changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = cast(dict[str, str], before['distributions'])
    new = cast(dict[str, str], after['distributions'])
    stages = [dict(stage=stage, available_before=value[0], available_after=after['stages'][stage][0],
                   reason_before=value[1], reason_after=after['stages'][stage][1])
              for stage, value in cast(dict[str, list[Any]], before['stages']).items() if value != after['stages'].get(stage)]
    return dict(added=sorted(f'{name}=={new[name]}' for name in new.keys() - old.keys()),
                removed=sorted(f'{name}=={old[name]}' for name in old.keys() - new.keys()),
                changed=sorted(f'{name}: {old[name]} -> {new[name]}' for name in old.keys() & new.keys() if old[name] != new[name]),
                stages=stages, fingerprint_before=before['fingerprint'], fingerprint_after=after['fingerprint'],
                recheck='capabilities_view in a fresh interpreter')


def _tail(text: str) -> list[str]:
    return text.splitlines()[-OUTPUT_TAIL_LINES:]


class _LiveOutput:
    """Combined output of one install command, copied to stderr line by line while it runs (K3B-L5: a long torch install
    shows live progress); the last ``OUTPUT_TAIL_LINES`` lines are kept for ``output_tail``. Pass ``fd`` as the
    command's stdout (stderr joined to it) and call ``close`` once the command has exited."""

    def __init__(self) -> None:
        self.lines: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)
        read_fd, self.fd = os.pipe()
        self.reader = threading.Thread(target=self._pump, args=(read_fd,), name='install-output', daemon=True)
        self.reader.start()

    def _pump(self, read_fd: int) -> None:
        with os.fdopen(read_fd, 'r', encoding='utf-8', errors='replace') as stream:
            for line in stream:
                sys.stderr.write(line)
                sys.stderr.flush()
                self.lines.append(line.rstrip('\n'))

    def close(self) -> None:
        os.close(self.fd)
        self.reader.join(timeout=30)  # EOF arrives once the command (and any child holding the pipe) has exited

    def tail(self, done: Any) -> list[str]:
        parts: list[Any] = [done.stdout, done.stderr]  # None when written to the pipe; a test runner may return text
        captured = ''.join(part.decode('utf-8', 'replace') if isinstance(part, bytes) else str(part) for part in parts if part)
        if captured:
            sys.stderr.write(captured)
            self.lines.extend(captured.splitlines())
        return list(self.lines)


def execute_install(plan: dict[str, Any], *, installer: str = 'auto') -> tuple[dict[str, Any], CatalogError | None]:
    """Run the plan's commands in order, stop at the first failure, then re-check capabilities. Returns the data and the
    typed error of a failed command (or None). The commands are exactly ``plan['commands']``."""
    python = cast(str, plan['installer']['python'])
    tool = _installer(installer, python)
    if not tool['available']:
        raise _error('installer_unavailable', ERRORS['installer_unavailable'][1], installer=tool['kind'])
    before = _snapshot(python)
    steps: list[dict[str, Any]] = []
    failure: CatalogError | None = None
    for index, command in enumerate(cast(list[dict[str, Any]], plan['commands'])):
        started = time.monotonic()
        live = _LiveOutput()
        try:
            done = subprocess.run(cast(list[str], command['argv']), stdout=live.fd, stderr=subprocess.STDOUT)
        finally:
            live.close()
        returncode, output_tail = done.returncode, live.tail(done)
        steps.append(dict(index=index, argv=command['argv'], returncode=returncode,
                          seconds=round(time.monotonic() - started, 3), output_tail=output_tail))
        if returncode:
            failure = _error('install_command_failed', f'Install command {index + 1} exited {returncode}; later commands did not run',
                             index=index, returncode=returncode, argv=command['argv'])
            break
    check: dict[str, Any] | None = None
    if failure is None and steps:
        done = subprocess.run(cast(list[str], tool['check']), capture_output=True, text=True)
        check = dict(argv=tool['check'], returncode=done.returncode, output_tail=_tail(done.stdout + done.stderr))
    after = _snapshot(python)
    data = dict(plan, mode='executed', executed=dict(steps=steps, dependency_check=check),
                changes=_changes(before, after), warnings=list(cast(list[dict[str, Any]], plan['warnings'])))
    if check is not None and check['returncode']:
        data['warnings'].append(dict(code='dependency_check_failed', message='The installer\'s dependency check reported '
                                     'conflicts (they may predate this install); see executed.dependency_check', stage=None))
    return data, failure


__all__ = ['ERRORS', 'INSTALLERS', 'plan_install', 'execute_install']
