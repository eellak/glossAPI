"""The optional-package catalog (``glossapi capabilities --catalog``): every install profile, what it adds, its measured
install size and package count, its device needs, its licences and whether it is installed.

Single sources, never hand-copied:

- **Requirements** come from the files ``pyproject.toml`` names for each extra (``[project.dependencies]`` for core and
  ``[tool.setuptools.dynamic.optional-dependencies]``) when glossAPI runs from a source checkout, and otherwise from the
  installed distribution's own metadata (``Requires-Dist``), which the wheel build generated from the same files.
- **What a profile adds** is derived from each stage adapter's ``packages()`` declaration under the listed settings
  variants, so a stage's package groups are never typed twice.
- **Sizes, package counts and licences** come from the ``uv pip compile`` locks (PEP 751 ``pylock.toml``) written by
  ``tools/catalog/measure_profiles.py`` into ``data/package_catalog/measured.json``. Each lock records the hash of the
  requirement inputs it was compiled from; ``inputs_sha256`` disagreeing with the current requirements marks the
  measurement stale (the tests fail on it).
- **Device profiles** (torch index and build suffix) are the installer's own (``glossapi.scripts.install_glossapi``).

Detection imports no backend: ``importlib.metadata`` only, as in ``glossapi.capabilities``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
from importlib import resources
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Final, cast

from .capabilities import STAGE_ORDER, canonical_name, declared_requirements, installed_version, marker_applies, parse_requirement, version_satisfies

CATALOG_VERSION: Final[int] = 1
DEVICES: Final[tuple[str, ...]] = ('cpu', 'cuda126', 'cuda132', 'mps')
# Purpose of each install profile, in one sentence. Every pyproject extra (plus core) must have one (tests).
PURPOSES: Final[dict[str, str]] = {
    'core': 'The base install: text-layer extraction (PDFium with the bundled Rust font repair), cleaning, sections, '
            'annotation with the packaged neutral classifier, and export. Always installed with glossapi.',
    'download': 'HTTP downloads of URL sources (aiohttp, YAML policies).',
    'browser': 'Browser-assisted downloads (Playwright, Pillow); includes download. Browser engines are a separate '
               'explicit `python -m playwright install chromium` step that glossapi never runs.',
    'layout': 'Docling 2.81 layout extraction and math/code enrichment; needs a device-specific torch build.',
    'docling': 'Alias of layout (kept so existing `glossapi[docling]` installs resolve).',
    'ocr': 'The Greek OCR toolkit (the default OCR plugin, pinned to one commit) with its Docling, torch and ONNX pins.',
    'annotate': 'No extra packages: the packaged neutral section classifier runs on core numpy. Kept so existing '
                '`glossapi[annotate]` installs resolve.',
    'annotate-legacy': 'scikit-learn 1.6.1 and joblib, only to load an explicitly supplied *.joblib classifier or to '
                       'train one.',
    'font-repair': 'The Python reference font repair (pypdf, fontTools); used only when the Rust repair is missing or '
                   'requested explicitly.',
    'docs': 'Documentation tooling (mkdocs); no pipeline capability.',
}
KINDS: Final[dict[str, str]] = {'core': 'base', 'docling': 'alias', 'docs': 'tooling'}
# Proprietary licences (K3B-M2). A package is flagged when its measured licence says proprietary (e.g. 'NVIDIA
# Proprietary Software', 'LicenseRef-NVIDIA-Proprietary'), or when it is an NVIDIA CUDA runtime distribution that
# declares no licence at all (the CUDA 13 wheels): those ship under the NVIDIA software licence agreement.
NVIDIA_RUNTIME_PREFIXES: Final[tuple[str, ...]] = ('nvidia-', 'cuda-toolkit')
PROPRIETARY_BASIS: Final[str] = ('a measured licence naming a proprietary licence, or an NVIDIA CUDA runtime distribution (nvidia-*, '
                                 'cuda-toolkit) that declares no licence (shipped under the NVIDIA software licence agreement)')
# Stage settings variants whose package declarations differ from the defaults (stage adapters' packages()).
STAGE_VARIANTS: Final[tuple[tuple[str, tuple[tuple[str, str], ...]], ...]] = (
    ('download', ()), ('download', (('download_mode', 'browser'),)),
    ('extract', ()), ('extract', (('phase1_backend', 'docling'),)), ('extract', (('repair_backend', 'python'),)),
    ('clean', ()), ('ocr', ()), ('math', ()), ('sections', ()),
    ('annotate', ()), ('annotate', (('classifier_model_format', 'joblib'),)), ('export', ()),
)
MEASURED_RESOURCE: Final[tuple[str, ...]] = ('data', 'package_catalog', 'measured.json')


class CatalogError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class RequirementSource:
    """The requirement lines of core and of every extra, as declared (self references such as glossapi[download] kept)."""
    kind: str  # 'source_tree' | 'installed_metadata'
    location: str
    core: tuple[str, ...]
    extras: dict[str, tuple[str, ...]]


def source_tree() -> Path | None:
    """The checkout root when this package is imported from ``<root>/src/glossapi`` of a glossapi source tree."""
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    project = root / 'pyproject.toml'
    if package.parent.name != 'src' or not project.is_file():
        return None
    try:
        name = cast(dict[str, Any], tomllib.loads(project.read_text(encoding='utf-8')).get('project', {})).get('name')
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return root if name == 'glossapi' else None


def _file_lines(path: Path) -> tuple[str, ...]:
    lines: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        text = line.split(' #', 1)[0].strip()
        if text and not text.startswith('#'):
            lines.append(text)
    return tuple(lines)


def _from_source_tree(root: Path) -> RequirementSource:
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    core = tuple(canonical_requirement(line) for line in cast(list[str], project['project']['dependencies']))
    table = cast(dict[str, dict[str, list[str]]],
                 project['tool']['setuptools']['dynamic']['optional-dependencies'])
    extras = {name: tuple(canonical_requirement(line) for file in entry['file'] for line in _file_lines(root / file))
              for name, entry in table.items()}
    return RequirementSource('source_tree', str(root / 'pyproject.toml'), core, extras)


# K3B-L2: installed metadata lists specifiers in setuptools' sorted order (``aiohttp<4,>=3.14.5``) while the source tree
# writes ``aiohttp>=3.14.5,<4``. Both sources go through one canonical order (exact and lower bounds, then upper bounds,
# then exclusions; stable otherwise), so the requirement text, the install commands and the lock-input hash are the same
# for a wheel install as for a checkout. The order is the source tree's own, so its lines pass through unchanged.
_SPECIFIER_RANK: Final[dict[str, int]] = {'===': 0, '==': 0, '~=': 1, '>=': 1, '>': 1, '<=': 2, '<': 2, '!=': 3}
_REQUIREMENT_HEAD = re.compile(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*\s*(?:\[[^\]]*\])?)\s*')


def canonical_requirement(text: str) -> str:
    """``text`` with its version specifiers in the canonical order; everything else (name, extras, marker) as written."""
    parsed = parse_requirement(text)
    if parsed.url is not None or len(parsed.specifiers) < 2:
        return text
    body, separator, marker = text.partition(';')
    head = _REQUIREMENT_HEAD.match(body)
    if head is None:  # unreachable: parse_requirement accepted the same text
        return text
    ordered = sorted(parsed.specifiers, key=lambda item: _SPECIFIER_RANK[item[0]])
    return head.group(1) + ','.join(operator + version for operator, version in ordered) + (separator + marker if separator else '')


_EXTRA_CLAUSE = re.compile(r'''\s*(?:\band\s+)?extra\s*==\s*["']([^"']+)["']\s*(?:\band\b\s*)?''')


def _from_metadata() -> RequirementSource:
    try:
        distribution = importlib.metadata.distribution('glossapi')
    except importlib.metadata.PackageNotFoundError:
        raise CatalogError('requirements_unavailable', 'glossapi is neither a source checkout nor an installed '
                           'distribution; its install profiles cannot be read') from None
    core: list[str] = []
    extras: dict[str, list[str]] = {name: [] for name in distribution.metadata.get_all('Provides-Extra') or []}
    for line in distribution.requires or []:
        requirement, _, marker = line.partition(';')
        found = _EXTRA_CLAUSE.search(marker)
        if found is None:
            core.append(canonical_requirement(line.strip()))
            continue
        rest = (marker[:found.start()] + ' ' + marker[found.end():]).strip()
        rest = re.sub(r'^(and|or)\s+|\s+(and|or)$', '', rest).strip()
        extras.setdefault(found.group(1), []).append(canonical_requirement(requirement.strip() + ('; ' + rest if rest else '')))
    return RequirementSource('installed_metadata', str(distribution.locate_file('')), tuple(core),
                             {name: tuple(lines) for name, lines in extras.items()})


def requirement_source() -> RequirementSource:
    root = source_tree()
    return _from_source_tree(root) if root is not None else _from_metadata()


def profile_names(source: RequirementSource) -> list[str]:
    return ['core', *source.extras]


def _self_reference(line: str) -> list[str] | None:
    found = re.fullmatch(r'\s*glossapi\s*\[([^\]]+)\]\s*', line.split(';', 1)[0])
    return [item.strip() for item in found.group(1).split(',')] if found else None


def includes(source: RequirementSource, profile: str) -> list[str]:
    """Profiles this one pulls in through ``glossapi[...]`` self references, recursively, in order."""
    found: list[str] = []

    def visit(name: str) -> None:
        for line in source.extras.get(name, ()):
            for item in _self_reference(line) or []:
                if item not in found and item != profile:
                    found.append(item)
                    visit(item)

    visit(profile)
    return found


def expanded_requirements(source: RequirementSource, profile: str) -> list[str]:
    """The profile's own requirement lines with self references expanded (core: the core dependencies)."""
    if profile == 'core':
        return list(source.core)
    if profile not in source.extras:
        raise CatalogError('unknown_package', 'Unknown package: ' + profile,
                           {'package': profile, 'known': profile_names(source)})
    lines: list[str] = []
    for name in [*reversed(includes(source, profile)), profile]:
        for line in source.extras.get(name, ()):
            if _self_reference(line) is None and line not in lines:
                lines.append(line)
    return lines


def torch_profiles() -> tuple[str, ...]:
    from .scripts import install_glossapi as installer  # untyped module: read through Any
    return tuple(cast(tuple[str, ...], cast(Any, installer).TORCH_PACKAGES))


def device_bound(source: RequirementSource, profile: str) -> bool:
    """A profile needs a device-specific torch build when it is, or includes, one of the installer's torch packages."""
    return bool({profile, *includes(source, profile)} & set(torch_profiles()))


def torch_build(device: str, system: str, machine: str) -> dict[str, Any]:
    """The installer's torch index, build suffix and pins for a device profile on a platform (raises CatalogError)."""
    from .scripts import install_glossapi as installer
    module = cast(Any, installer)
    if device not in DEVICES:
        raise CatalogError('unknown_device', 'Unknown device profile: ' + device, {'device': device, 'known': list(DEVICES)})
    try:
        index, suffix = cast(tuple[str, str], module.torch_build(device, system, machine))
    except ValueError as error:
        raise CatalogError('device_unsupported_on_platform', str(error),
                           {'device': device, 'system': system, 'machine': machine}) from None
    return dict(device=device, index_url=index, suffix=suffix,
                requirements=[f'torch=={module.TORCH_VERSION}{suffix}', f'torchvision=={module.TORCHVISION_VERSION}{suffix}'])


TORCH_NAMES: Final[frozenset[str]] = frozenset({'torch', 'torchvision'})


def install_requirements(source: RequirementSource, profiles: list[str], torch: dict[str, Any] | None) -> list[str]:
    """The exact requirement list for these profiles: their expanded lines, in order, without duplicates. With a torch
    build, any torch/torchvision line is replaced by the device pins, which come first (the installer's constraint)."""
    lines: list[str] = list(cast(list[str], torch['requirements'])) if torch is not None else []
    for profile in profiles:
        for line in expanded_requirements(source, profile):
            if torch is not None and canonical_name(parse_requirement(line).name) in TORCH_NAMES:
                continue
            if line not in lines:
                lines.append(line)
    return lines


def lock_requirements(source: RequirementSource, profile: str, device: str | None, target: dict[str, str]) -> list[str]:
    """The input of one lock: the core dependencies plus the profile's exact install requirements for the device."""
    torch = torch_build(device, target['system'], target['machine']) if device is not None else None
    extra = install_requirements(source, [profile], torch) if profile != 'core' else []
    return [*source.core, *(line for line in extra if line not in source.core)]


def inputs_sha256(source: RequirementSource, profile: str, device: str | None, target: dict[str, str]) -> str:
    """Identity of one lock's inputs: the expanded requirements, the core dependencies, the torch build and the target."""
    platform = dict(system=target['system'], machine=target['machine'])
    payload = dict(profile=profile, requirements=lock_requirements(source, profile, device, platform), target=platform)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def measured() -> dict[str, Any] | None:
    """The packaged lock measurements, or None when the file is absent."""
    folder = resources.files(__package__ or 'glossapi')
    item = folder
    for part in MEASURED_RESOURCE:
        item = item.joinpath(part)
    try:
        return cast(dict[str, Any], json.loads(item.read_text(encoding='utf-8')))
    except (FileNotFoundError, OSError):
        return None


def stage_enablement() -> dict[str, list[dict[str, Any]]]:
    """Per package group: the stage variants that need it, with every group each variant needs (from the adapters)."""
    enabled: dict[str, list[dict[str, Any]]] = {}
    for stage, pairs in STAGE_VARIANTS:
        settings: dict[str, Any] = dict(pairs)
        groups: list[str] = []
        for requirement in declared_requirements(stage, settings):
            group = cast(str, requirement.package_group)
            if group not in groups:
                groups.append(group)
        for group in groups or ['core']:
            enabled.setdefault(group, []).append(dict(stage=stage, settings=settings, requires_profiles=groups or ['core']))
    return enabled


def requirement_status(text: str) -> dict[str, Any]:
    """Installed state of one requirement line (presence and version, marker evaluated; direct URLs: presence)."""
    parsed = parse_requirement(text)
    version = installed_version(parsed.name)
    row: dict[str, Any] = dict(requirement=text, name=canonical_name(parsed.name), installed_version=version, status='satisfied')
    if not marker_applies(parsed.marker):
        row['status'] = 'not_applicable'
    elif version is None:
        row['status'] = 'missing'
    elif parsed.specifiers and not version_satisfies(version, parsed.specifiers):
        row['status'] = 'version_mismatch'
    return row


def installed_state(rows: list[dict[str, Any]]) -> str:
    applicable = [row for row in rows if row['status'] != 'not_applicable']
    if not applicable:
        return 'installed'
    satisfied = sum(row['status'] == 'satisfied' for row in applicable)
    return 'installed' if satisfied == len(applicable) else 'missing' if satisfied == 0 else 'partial'


def lock_packages(data: dict[str, Any], lock: dict[str, Any]) -> list[dict[str, Any]]:
    """The packages of one measured lock, each with its release fields for the lock's target."""
    table = cast(dict[str, list[Any]], cast(dict[str, Any], data.get('releases') or {}).get(lock['target']) or {})
    fields = cast(list[str], data.get('release_fields') or ['source', 'download_bytes', 'installed_bytes', 'licence'])
    rows: list[dict[str, Any]] = []
    for key in cast(list[str], lock.get('packages') or []):
        name, _, version = key.partition('==')
        rows.append(dict(name=name, version=version, **dict(zip(fields, table.get(key) or [None] * len(fields)))))
    return rows


def measured_profile(source: RequirementSource, profile: str) -> str:
    """The profile whose locks measure this one (an alias is measured by the profile it names)."""
    return includes(source, profile)[0] if KINDS.get(profile) == 'alias' else profile


def current_target(data: dict[str, Any], system: str, machine: str) -> str | None:
    """The measured target matching this platform, if any."""
    for target_id, target in cast(dict[str, dict[str, str]], data.get('targets') or {}).items():
        if target['system'] == system and target['machine'] in (machine, {'aarch64': 'arm64'}.get(machine, machine)):
            return target_id
    return None


def proprietary_licence(item: dict[str, Any]) -> bool:
    """Whether one measured lock package is under a proprietary licence (``PROPRIETARY_BASIS``)."""
    licence = cast(str | None, item.get('licence'))
    if licence and 'proprietary' in licence.casefold():
        return True
    undeclared = not licence or licence == 'unknown' or licence.startswith('not declared')
    return undeclared and cast(str, item['name']).startswith(NVIDIA_RUNTIME_PREFIXES)


def lock_for(data: dict[str, Any], profile: str, device: str | None, target: str) -> dict[str, Any] | None:
    return cast(dict[str, dict[str, Any]], data.get('locks') or {}).get(f'{profile}@{device or "any"}@{target}')


def _total(items: list[dict[str, Any]], key: str) -> int:
    """Sum over the measured packages; unmeasured ones (sdist/VCS) are listed separately in ``unmeasured``."""
    return sum(cast(int, item[key]) for item in items if item.get(key) is not None)


def _lock_summary(data: dict[str, Any], lock: dict[str, Any], core_lock: dict[str, Any] | None) -> dict[str, Any]:
    packages = lock_packages(data, lock)
    core_names = {item['name'] for item in lock_packages(data, core_lock)} if core_lock is not None and core_lock is not lock else set[str]()
    added = [item for item in packages if item['name'] not in core_names]
    return dict(lock=lock['id'], lock_file=lock.get('lock_file'), lock_sha256=lock.get('lock_sha256'), status=lock['status'],
                error=lock.get('error'), device=lock.get('device'), target=lock['target'], python=lock.get('python'),
                inputs_sha256=lock.get('inputs_sha256'), packages=len(packages), download_bytes=_total(packages, 'download_bytes'),
                installed_bytes=_total(packages, 'installed_bytes'), added_packages=len(added),
                added_download_bytes=_total(added, 'download_bytes'), added_installed_bytes=_total(added, 'installed_bytes'),
                added_names=sorted(cast(str, item['name']) for item in added),
                unmeasured=sorted(cast(str, item['name']) for item in packages
                                  if item.get('download_bytes') is None or item.get('installed_bytes') is None),
                licences=sorted({cast(str, item.get('licence') or 'unknown') for item in added}),
                proprietary_packages=sorted(cast(str, item['name']) for item in packages if proprietary_licence(item)))


def _device_rows(profile: str, bound: bool, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for device in DEVICES:
        platforms: list[dict[str, Any]] = []
        for target_id, target in cast(dict[str, dict[str, str]], data.get('targets') or {}).items():
            try:
                torch_build(device, target['system'], target['machine'])
            except CatalogError:
                continue
            lock = lock_for(data, profile, device if bound else None, target_id)
            platforms.append(dict(target=target_id, lock=lock['id'] if lock else None,
                                  resolved=None if lock is None else lock['status'] == 'resolved'))
        supported = bool(platforms) and all(item['resolved'] is not False for item in platforms)
        rows.append(dict(device=device, supported=supported, platforms=platforms,
                         reason=None if supported else 'no measured platform has this device profile' if not platforms
                         else 'the lock for this device profile did not resolve'))
    return rows


def device_support(source: RequirementSource, profile: str, device: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """The catalog's device row for one profile (supported, platforms, reason)."""
    data = measured() if data is None else data
    rows = _device_rows(measured_profile(source, profile), device_bound(source, profile), data or {})
    return next(row for row in rows if row['device'] == device)


def _proprietary(sizes: list[dict[str, Any]]) -> dict[str, Any]:
    """The profile's explicit proprietary-licence flag (K3B-M2): whether any measured lock of it installs a proprietary
    package, which ones (over every target and device profile), and the rule."""
    names = sorted({name for size in sizes for name in cast(list[str], size['proprietary_packages'])})
    return dict(proprietary=bool(names), proprietary_packages=names, proprietary_basis=PROPRIETARY_BASIS)


def catalog_view(source: RequirementSource | None = None, *, installed: bool = True) -> dict[str, Any]:
    """``capabilities --catalog`` data (schema common.json#/$defs/catalog). ``installed=False`` leaves the
    environment-dependent install state out (the LLM guide uses that form)."""
    source = source or requirement_source()
    data = measured() or {}
    locks = cast(dict[str, dict[str, Any]], data.get('locks') or {})
    targets = cast(dict[str, dict[str, str]], data.get('targets') or {})
    enablement = stage_enablement()
    profiles: list[dict[str, Any]] = []
    for name in profile_names(source):
        bound = device_bound(source, name)
        basis = measured_profile(source, name)
        requirements = expanded_requirements(source, name)
        roots = {canonical_name(parse_requirement(line).name) for line in requirements}
        sizes: list[dict[str, Any]] = []
        stale: list[str] = []
        root_licences: set[str] = set()
        for lock in locks.values():
            if lock['profile'] != basis:
                continue
            core_lock = locks.get(f'core@any@{lock["target"]}')
            summary = _lock_summary(data, lock, core_lock)
            current = inputs_sha256(source, basis, cast(str | None, lock.get('device')) if bound else None, targets[lock['target']])
            summary['current'] = current == lock.get('inputs_sha256')
            if not summary['current']:
                stale.append(cast(str, lock['id']))
            sizes.append(summary)
            root_licences.update(cast(str, item.get('licence') or 'unknown') for item in lock_packages(data, lock) if item['name'] in roots)
        row: dict[str, Any] = dict(
            name=name, kind=KINDS.get(name, 'feature'), purpose=PURPOSES.get(name), alias_of=basis if basis != name else None,
            includes=includes(source, name), requirements=requirements, enables=enablement.get(basis, []),
            device_bound=bound, devices=_device_rows(basis, bound, data),
            licence=dict(roots=sorted(root_licences), all_added=sorted({item for size in sizes for item in size['licences']}),
                         **_proprietary(sizes)),
            install_size=sizes, measurement=dict(state='missing' if not sizes else 'stale' if stale else 'current', stale_locks=stale),
            install_command=None if name == 'core' else f'glossapi install --device cpu --package {name}' if not bound
            else f'glossapi install --device <{"|".join(DEVICES)}> --package {name}')
        if installed:
            statuses = [requirement_status(line) for line in requirements]
            row['installed'] = dict(state=installed_state(statuses), requirements=statuses)
        profiles.append(row)
    return dict(catalog_version=CATALOG_VERSION, requirements_source=dict(kind=source.kind, location=source.location),
                devices=list(DEVICES), torch_profiles=list(torch_profiles()),
                measurement=dict(file='glossapi/' + '/'.join(MEASURED_RESOURCE), present=bool(data),
                                 measured_at=data.get('measured_at'), tool=data.get('tool'), resolver=data.get('resolver'),
                                 python=data.get('python'), targets=targets, size_basis=data.get('size_basis')),
                stages=list(STAGE_ORDER), profiles=profiles)


__all__ = ['CatalogError', 'RequirementSource', 'catalog_view', 'requirement_source', 'expanded_requirements', 'includes',
           'device_bound', 'device_support', 'torch_build', 'install_requirements', 'lock_requirements', 'inputs_sha256',
           'measured', 'lock_packages', 'lock_for', 'current_target', 'measured_profile', 'requirement_status', 'installed_state',
           'profile_names', 'source_tree', 'stage_enablement', 'DEVICES', 'PURPOSES', 'KINDS']
