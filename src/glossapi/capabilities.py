"""Installed capabilities and visible devices, detected without importing any backend.

Detection reads distribution metadata (``importlib.metadata``) and module locations (``importlib.util.find_spec``)
only. It never imports torch, docling, pandas, pyarrow, numpy or an OCR backend, so ``glossapi capabilities``,
``doctor`` and ``plan`` stay light on a base install. A device that can only be confirmed by its framework (CUDA or
MPS through torch) is reported as ``unknown_until_probe``; ``glossapi doctor --probe-devices`` runs that probe in a
child process and says so.

A stage is *available* under exactly the rule the streaming scheduler applies before dispatch
(``StreamingScheduler._check_package``): every declared distribution is installed; for the OCR stage each pin's
marker is evaluated, its version must satisfy the pin, the Greek OCR toolkit source must be admitted, and its
entry point must be registered. The required native modules of the stage must be locatable as well, and built for the
native API level this package needs (read from the binary's embedded marker, not by importing it).
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Final, Literal, cast

STAGE_ORDER: Final[tuple[str, ...]] = ('download', 'extract', 'clean', 'ocr', 'math', 'sections', 'annotate', 'export')
# Device kinds each stage can use (ocr/defaults.validate_device): only the Greek OCR plugin runs off the CPU.
STAGE_DEVICE_KINDS: Final[dict[str, tuple[str, ...]]] = {stage: ('cpu',) for stage in STAGE_ORDER} | {'ocr': ('cpu', 'mps', 'cuda')}
# Native modules each stage loads; required ones make the stage unavailable when absent. The PDF repair extension
# has a declared Python fallback (the font-repair extra), so it is reported but not required.
NATIVE_MODULES: Final[dict[str, tuple[tuple[str, bool], ...]]] = {
    'extract': (('glossapi_rs_pdfrepair', False),),
    'clean': (('glossapi_rs_cleaner', True), ('glossapi_rs_noise', True)),
}
# INTEG-G1: the native API level this package needs from each required extension. Each build embeds the marker
# ``GLOSSAPI_NATIVE_API:<module>:<level>;`` (rust/<module>/src/lib.rs NATIVE_API_MARKER), read here from the binary
# without importing it. A build without the marker or with another level is stale (built from other source): its
# module is reported missing, with the found and needed levels, so plan/doctor say "rebuild core" instead of the run
# failing per document. Raise a level here and in the crate together.
# The optional PDF repair extension is listed too (K4g fix 3): a stale build is reported, and the repair selector
# (pdf_repair/selector.py) uses the Python reference instead of it.
NATIVE_API_LEVELS: Final[dict[str, int]] = {'glossapi_rs_cleaner': 2, 'glossapi_rs_noise': 1, 'glossapi_rs_pdfrepair': 1}
_NATIVE_API_PREFIX: Final[bytes] = b'GLOSSAPI_NATIVE_API:'
# Stages the streaming run cannot execute, with the reason (ledger/reader.RUN_UNAVAILABLE).
RUN_UNAVAILABLE: Final[dict[str, str]] = {'math': 'math_backend_unavailable'}
INSTALLER_PACKAGES: Final[tuple[str, ...]] = ('core', 'download', 'browser', 'layout', 'ocr', 'annotate', 'annotate-legacy')
SUPPORTED_PYTHON: Final[tuple[tuple[int, int], tuple[int, int]]] = ((3, 11), (3, 15))  # pyproject requires-python
CAPABILITY_FINGERPRINT_VERSION: Final[int] = 1

DeviceState = Literal['available', 'unknown_until_probe', 'unavailable']


@dataclass(frozen=True)
class Requirement:
    text: str
    name: str
    specifiers: tuple[tuple[str, str], ...]
    url: str | None
    marker: str | None


_NAME = re.compile(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*')
_SPECIFIER = re.compile(r'\s*(===|==|!=|~=|<=|>=|<|>)\s*([^\s,;]+)\s*')
_MARKER = re.compile(r'\s*([a-z_]+)\s*(==|!=|<=|>=|<|>)\s*["\']([^"\']*)["\']\s*')


def canonical_name(name: str) -> str:
    return re.sub(r'[-_.]+', '-', name).lower()


def parse_requirement(text: str) -> Requirement:
    """PEP 508 subset used by glossAPI's pins: name, extras, specifiers, a direct URL and a simple marker."""
    body, _, marker = text.partition(';')
    match = _NAME.match(body)
    if match is None:
        raise ValueError('Unparseable requirement: ' + text)
    rest = body[match.end():].strip()
    url = None
    specifiers: list[tuple[str, str]] = []
    if rest.startswith('@'):
        url = rest[1:].strip()
    elif rest:
        for part in rest.split(','):
            found = _SPECIFIER.fullmatch(part)
            if found is None:
                raise ValueError('Unparseable version specifier: ' + text)
            specifiers.append((found.group(1), found.group(2)))
    return Requirement(text, match.group(1), tuple(specifiers), url, marker.strip() or None)


def _environment() -> dict[str, str]:
    return {'python_version': '%d.%d' % sys.version_info[:2],
            'python_full_version': '%d.%d.%d' % sys.version_info[:3],
            'sys_platform': sys.platform, 'platform_system': platform.system(), 'platform_machine': platform.machine(),
            'os_name': os.name, 'implementation_name': sys.implementation.name}


def _version_key(version: str) -> tuple[tuple[int, ...], int, str]:
    """Release tuple, a pre-release rank (pre < final < post) and the local label."""
    public, _, local = version.partition('+')
    match = re.match(r'v?(\d+(?:\.\d+)*)(.*)', public.strip().lower())
    if match is None:
        return ((), 0, local)
    release = tuple(int(part) for part in match.group(1).split('.'))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    suffix = match.group(2)
    rank = 0 if not suffix else 1 if re.match(r'[._-]?(post|rev|r)\d*', suffix) else -1
    return (release, rank, local)


def _compare(installed: str, operator: str, wanted: str) -> bool:
    if operator == '===':
        return installed == wanted
    if operator in ('==', '!=') and wanted.endswith('.*'):
        prefix = _version_key(wanted[:-2])[0]
        equal = _version_key(installed)[0][:len(prefix)] == prefix
        return equal if operator == '==' else not equal
    have, want = _version_key(installed), _version_key(wanted)
    if '+' not in wanted:
        have = (have[0], have[1], '')  # a local label is ignored unless the pin names one
    order = (have > want) - (have < want)
    if operator == '~=':
        prefix = want[0][:-1] if len(want[0]) > 1 else want[0]
        return order >= 0 and have[0][:len(prefix)] == prefix
    return {'==': order == 0, '!=': order != 0, '<': order < 0, '<=': order <= 0, '>': order > 0, '>=': order >= 0}[operator]


def marker_applies(marker: str | None) -> bool:
    """Evaluate ``name OP "value"`` clauses joined by ``and``/``or`` (no parentheses), as pip would."""
    if not marker:
        return True
    environment = _environment()

    def clause(text: str) -> bool:
        found = _MARKER.fullmatch(text)
        if found is None:
            raise ValueError('Unsupported marker: ' + marker)
        name, operator, value = found.groups()
        if name not in environment:
            raise ValueError('Unsupported marker variable: ' + name)
        if name.endswith('version'):
            return _compare(environment[name], operator, value)
        return {'==': environment[name] == value, '!=': environment[name] != value}.get(operator, False)

    return any(all(clause(part) for part in re.split(r'\s+and\s+', alternative)) for alternative in re.split(r'\s+or\s+', marker))


def version_satisfies(installed: str, specifiers: tuple[tuple[str, str], ...]) -> bool:
    return all(_compare(installed, operator, wanted) for operator, wanted in specifiers)


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_present(name: str) -> bool:
    """Locate a top-level module without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def native_binary(name: str) -> Path | None:
    """The extension file a top-level native module resolves to, located without importing it: the module itself
    (``name.abi3.so`` in the root wheel) or the extension inside its package (maturin's ``name/name.abi3.so``)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if spec.origin and spec.origin.endswith(suffixes):
        return Path(spec.origin)
    for location in spec.submodule_search_locations or ():
        found = sorted(path for path in Path(location).glob(name + '.*') if path.name.endswith(suffixes))
        if found:
            return found[0]
    return None


def _marker_level(data: bytes, name: str) -> int | None:
    """The level of ``name``'s marker in a binary's bytes; None without exactly one well-formed marker for it."""
    pattern = re.escape(_NATIVE_API_PREFIX + name.encode() + b':') + rb'(\d{1,6});'
    levels = {int(found.group(1)) for found in re.finditer(pattern, data)}
    return levels.pop() if len(levels) == 1 else None


def native_api_level(name: str) -> int | None:
    """The native API level embedded in the module's binary (None: no binary, or a build without the marker)."""
    binary = native_binary(name)
    if binary is None:
        return None
    try:
        stat = binary.stat()
    except OSError:
        return None
    return _binary_level(str(binary), stat.st_size, stat.st_mtime_ns, name)


@functools.lru_cache(maxsize=16)
def _binary_level(path: str, size: int, mtime_ns: int, name: str) -> int | None:
    try:
        return _marker_level(Path(path).read_bytes(), name)
    except OSError:
        return None


def native_module_status(name: str, required: bool) -> dict[str, Any]:
    """One ``native_modules`` row. A located module whose API level differs from ``NATIVE_API_LEVELS`` is reported as
    missing (the module this package needs is not installed) with the found and required levels."""
    row: dict[str, Any] = dict(module=name, required=required, status='present' if module_present(name) else 'missing')
    wanted = NATIVE_API_LEVELS.get(name)
    if wanted is None:
        return row
    found = native_api_level(name) if row['status'] == 'present' else None
    row.update(api_level=found, required_api_level=wanted, detail=None)
    if row['status'] == 'present' and found != wanted:
        row['status'] = 'missing'
        row['detail'] = (f'stale native module: built for native API level {found} (this glossapi needs {wanted}); '
                         'rebuild core' if found is not None else
                         f'stale native module: built without a native API marker (this glossapi needs level {wanted}); '
                         'rebuild core')
    return row


def install_command(group: str, device: str | None = None) -> str:
    """The dry-run command that adds a missing package group (``glossapi install``; add ``--yes`` to execute). Core (the
    Rust extensions) is rebuilt by the source installer, which reinstalls glossapi itself."""
    if group == 'core':
        return 'python install_glossapi.py core'
    target = 'mps' if device == 'mps' else 'cuda126' if (device or '').startswith('cuda') else 'cpu'
    return f'glossapi install --device {target if group in ("layout", "ocr") else "cpu"} --package {group}'


def _toolkit_status() -> dict[str, Any]:
    from .ocr import defaults  # untyped module: read through Any
    status = cast(dict[str, Any], cast(Any, defaults).toolkit_source_status())
    return {key: status.get(key) for key in ('distribution', 'admitted', 'status', 'reason', 'pinned_commit', 'installed_commit', 'version')}


def _adapter(stage: str) -> Any:
    """The stage adapter class, imported lazily (stage modules import only the standard library at import time)."""
    if stage == 'download':
        from .stages.download import DownloadStage as adapter
    elif stage == 'extract':
        from .stages.extract import ExtractStage as adapter
    elif stage == 'clean':
        from .stages.clean import CleanStage as adapter
    elif stage == 'ocr':
        from .stages.ocr import OcrStage as adapter
    elif stage == 'math':
        from .stages.math import MathStage as adapter
    elif stage == 'sections':
        from .stages.sections import SectionsStage as adapter
    elif stage == 'annotate':
        from .stages.annotate import AnnotateStage as adapter
    elif stage == 'export':
        from .stages.export import ExportStage as adapter
    else:
        raise ValueError('Unknown stage: ' + stage)
    return cast(Any, adapter)()


def declared_requirements(stage: str, settings: dict[str, Any] | None = None, device: str = 'cpu') -> list[Any]:
    """The stage adapter's own package declaration for these settings."""
    from .stages.contracts import StageSpecification
    specification = StageSpecification(stage + '.planned', device_id=device, settings=dict(settings or {}))
    return list(_adapter(stage).packages(specification))


def stage_capability(stage: str, settings: dict[str, Any] | None = None, device: str = 'cpu',
                     *, toolkit: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capability object (schema common.json#/$defs/capability) for one stage under the given settings."""
    check_versions = stage == 'ocr'
    packages: list[dict[str, Any]] = []
    entry_points: list[dict[str, Any]] = []
    toolkit_view: dict[str, Any] | None = None
    reasons: list[str] = []
    missing_groups: list[str] = []
    for requirement in declared_requirements(stage, settings, device):
        group = cast(str, requirement.package_group)
        group_missing = False
        for text in cast(tuple[str, ...], requirement.distributions):
            parsed = parse_requirement(text)
            version = installed_version(parsed.name)
            row: dict[str, Any] = dict(package_group=group, requirement=text, name=canonical_name(parsed.name),
                                       installed_version=version, check='version' if check_versions else 'presence', status='present', detail=None)
            if check_versions and not marker_applies(parsed.marker):
                row['status'] = 'not_applicable'
                row['detail'] = 'marker does not apply here: ' + str(parsed.marker)
            elif version is None:
                row['status'] = 'missing'
            elif check_versions and parsed.specifiers and not version_satisfies(version, parsed.specifiers):
                row['status'] = 'version_mismatch'
                row['detail'] = f'installed {version} does not satisfy {text}'
            elif check_versions and canonical_name(parsed.name) == 'greek-ocr-toolkit':
                toolkit_view = toolkit if toolkit is not None else _toolkit_status()
                if not toolkit_view['admitted']:
                    row['status'] = 'not_admitted'
                    row['detail'] = toolkit_view.get('reason')
            row['blocking'] = row['status'] in ('missing', 'version_mismatch', 'not_admitted')
            if row['blocking']:
                group_missing = True
                reasons.append({'missing': 'missing_package', 'version_mismatch': 'version_mismatch',
                                'not_admitted': 'toolkit_not_admitted'}[row['status']])
            packages.append(row)
        if requirement.entry_point_group:
            registered = any(point.name == requirement.entry_point_name
                             for point in importlib.metadata.entry_points(group=requirement.entry_point_group))
            entry_points.append(dict(group=requirement.entry_point_group, name=requirement.entry_point_name,
                                     status='registered' if registered else 'missing'))
            if not registered:
                group_missing = True
                reasons.append('missing_entry_point')
        if group_missing and group not in missing_groups:
            missing_groups.append(group)
    natives: list[dict[str, Any]] = []
    for module, required in NATIVE_MODULES.get(stage, ()):
        native = native_module_status(module, required)
        natives.append(native)
        if required and native['status'] != 'present':
            reasons.append('missing_native_module')
            if 'core' not in missing_groups:
                missing_groups.append('core')
    available = not reasons
    return dict(stage=stage, available=available, reason_code=reasons[0] if reasons else None,
                packages=packages, entry_points=entry_points, native_modules=natives, toolkit=toolkit_view,
                missing_install_profiles=missing_groups,
                install_commands=[install_command(group, device) for group in missing_groups],
                supported_device_kinds=list(STAGE_DEVICE_KINDS[stage]),
                executable_by_run=stage not in RUN_UNAVAILABLE, unavailable_reason=RUN_UNAVAILABLE.get(stage))


def _nvidia_devices() -> list[int]:
    found = sorted(int(path.name[6:]) for path in Path('/dev').glob('nvidia[0-9]*') if path.name[6:].isdigit())
    if not found and Path('/proc/driver/nvidia/version').exists():
        found = [0]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        if visible.strip().lower() in ('', '-1', 'none', 'nodevfiles'):
            return []
        count = len([item for item in visible.split(',') if item.strip()])
        found = list(range(min(count, len(found)))) if found else []
    return found


def devices(*, probe: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Device objects (schema common.json#/$defs/device). ``probe`` is a framework probe result, if one ran."""
    torch_version = installed_version('torch')
    rows: list[dict[str, Any]] = [dict(device_id='cpu', kind='cpu', hardware='present', state='available',
                                       basis='always present', reason=None)]
    darwin_arm = sys.platform == 'darwin' and platform.machine() == 'arm64'
    framework_reason = None if torch_version else 'torch is not installed (the ocr install profile adds it)'
    mps: dict[str, Any] = dict(device_id='mps', kind='mps', hardware='present' if darwin_arm else 'absent',
                               basis='platform: Apple Silicon macOS' if darwin_arm else 'platform is not Apple Silicon macOS')
    mps.update(state='unknown_until_probe' if darwin_arm and torch_version else 'unavailable',
               reason=None if darwin_arm and torch_version else framework_reason if darwin_arm else 'no Apple GPU on this platform')
    rows.append(mps)
    cuda_ids = _nvidia_devices() if sys.platform.startswith('linux') else []
    cpu_only_torch = bool(torch_version and '+cpu' in torch_version)
    for index in cuda_ids or [0]:
        present = bool(cuda_ids)
        usable = present and torch_version is not None and not cpu_only_torch
        rows.append(dict(device_id=f'cuda:{index}', kind='cuda', hardware='present' if present else 'absent',
                         basis='NVIDIA device node or driver visible' if present else 'no NVIDIA device node or driver visible',
                         state='unknown_until_probe' if usable else 'unavailable',
                         reason=None if usable else 'no NVIDIA GPU visible' if not present else
                         'installed torch is a CPU-only build (' + str(torch_version) + ')' if cpu_only_torch else framework_reason))
    if probe is not None:
        for row in rows:
            if row['kind'] == 'cpu':
                continue
            result = cast(dict[str, Any], probe.get(row['device_id']) or {})
            if 'available' in result:
                row['state'] = 'available' if result['available'] else 'unavailable'
                row['basis'] = 'framework probe: ' + str(probe.get('framework'))
                row['reason'] = None if result['available'] else result.get('reason') or 'framework reports the device unavailable'
    return rows


def device_kind(device_id: str) -> str:
    return 'cuda' if device_id == 'cuda' or device_id.startswith('cuda:') else device_id


def device_state(device_id: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    wanted = 'cuda:0' if device_id == 'cuda' else device_id
    return next((row for row in rows if row['device_id'] == wanted), None)


def capability_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Identity of the installed capabilities a plan depends on (versions, admission, entry points, native modules)."""
    identity = [dict(stage=row['stage'], available=row['available'],
                     packages=[(item['name'], item['status'], item['installed_version']) for item in row['packages']],
                     entry_points=[(item['group'], item['name'], item['status']) for item in row['entry_points']],
                     native_modules=[(item['module'], item['status']) for item in row['native_modules']],
                     toolkit=None if row['toolkit'] is None else (row['toolkit'].get('status'), row['toolkit'].get('installed_commit')))
                for row in rows]
    payload = json.dumps(dict(version=CAPABILITY_FINGERPRINT_VERSION, stages=identity), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def capabilities_view(*, probe: dict[str, Any] | None = None) -> dict[str, Any]:
    """``glossapi capabilities`` data: every stage with its default settings, and the visible devices."""
    toolkit: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        row = stage_capability(stage, toolkit=toolkit)
        toolkit = row['toolkit'] if row['toolkit'] is not None else toolkit
        stages.append(row)
    return dict(stages=stages, devices=devices(probe=probe),
                python=dict(version='%d.%d.%d' % sys.version_info[:3], implementation=sys.implementation.name,
                            supported='>=%d.%d,<%d.%d' % (*SUPPORTED_PYTHON[0], *SUPPORTED_PYTHON[1])),
                platform=dict(system=platform.system(), machine=platform.machine()),
                glossapi_version=installed_version('glossapi'),
                detection=dict(method='importlib.metadata and importlib.util.find_spec only; no backend imported',
                               device_probe='framework probe in a child process' if probe is not None else 'not run'),
                fingerprint=capability_fingerprint(stages))
