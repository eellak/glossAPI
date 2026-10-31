"""Input manifest v1: what a caller (Python, the CLI, or an LLM that scraped elsewhere) asks glossAPI to process.

Schema: ``glossapi/schemas/v1/manifest.json``. A v1 manifest is a JSON object::

    {"manifest_version": 1,
     "documents": [{"path": "/abs/a.pdf", "selection_id": "a", "metadata": {...anything...}}, ...],
     "selection": ["a"],                       # optional: only these selection IDs are processed
     "stages": ["extract", "clean", "export"], # optional
     "products": ["document_export"],          # optional: document_export -> export, annotated_sections -> annotate
     "devices": {"ocr": "mps"},                # optional, STAGE -> cpu | mps | cuda:N
     "workers": {"clean": 2},                  # optional, STAGE -> positive integer
     "settings": {"debug": "off", "memory_saving": "normal"}}   # optional, as --settings

Each document entry names exactly one source: ``path`` (an absolute local path), ``url`` (unchanged URL behaviour),
or an existing ledger reference (``document_id``, ``source_key``, ``source_uri``). ``metadata`` and any other entry
fields are the caller's original metadata and come back verbatim in ``status``/``report``. Entries that are not
selected are listed as exclusions and never opened, hashed or processed.

A manifest without ``manifest_version`` (a bare list or ``{"documents": [...]}``) keeps the pre-v1 behaviour. Any other
version, and every malformed entry, fails with a typed :class:`ManifestError` (``unsupported_manifest_version`` or
``invalid_manifest``) naming the entry and field. This module imports only the standard library.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final, Mapping, Sequence, cast

MANIFEST_VERSION: Final[int] = 1
SUPPORTED_VERSIONS: Final[tuple[int, ...]] = (1,)
STAGES: Final[tuple[str, ...]] = ('download', 'extract', 'clean', 'ocr', 'math', 'sections', 'annotate', 'export')
PRODUCTS: Final[dict[str, str]] = {'document_export': 'export', 'annotated_sections': 'annotate'}
SELECTOR_KEYS: Final[tuple[str, ...]] = ('document_id', 'source_key', 'source_uri', 'path', 'url')
TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(('manifest_version', 'documents', 'selection', 'stages', 'products', 'devices', 'workers', 'settings', 'producer', 'note'))
_DEVICE = re.compile(r'cpu|mps|cuda:\d+')


class ManifestError(ValueError):
    """Typed manifest failure; ``code`` is ``invalid_manifest`` or ``unsupported_manifest_version``."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details: dict[str, Any] = details


@dataclass(frozen=True)
class ParsedManifest:
    version: int | None
    selectors: list[str]
    rows: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    stages: list[str] | None
    devices: dict[str, str] = field(default_factory=lambda: {})
    workers: dict[str, int] = field(default_factory=lambda: {})
    settings: dict[str, Any] = field(default_factory=lambda: {})
    sha256: str = ''
    raw: Any = None

    @property
    def selected(self) -> Any:
        """The manifest the plan and run receive: the selected entries only, in their original form (a pre-v1
        manifest is passed on exactly as given)."""
        if self.version is None:
            return self.raw
        return {'documents': self.rows}

    def describe(self) -> dict[str, Any]:
        return dict(manifest_version=self.version, sha256=self.sha256, entries=len(self.rows) + len(self.exclusions),
                    selected=len(self.rows), excluded=len(self.exclusions))


def _invalid(message: str, **details: Any) -> ManifestError:
    return ManifestError('invalid_manifest', message, **details)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _check_json(value: Any, where: str) -> None:
    """Only JSON values (no NaN/Infinity, string keys) can be metadata."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise _invalid('Metadata must be plain JSON: ' + str(error), field=where) from None


def _assignments(value: Any, name: str, *, integer: bool) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _invalid(name + ' must be an object mapping a stage to a value', field=name)
    output: dict[str, Any] = {}
    for stage, item in cast(dict[Any, Any], value).items():
        if stage not in STAGES:
            raise _invalid('Unknown stage in ' + name + ': ' + str(stage), field=name, stage=stage)
        if integer:
            if type(item) is not int or item <= 0:
                raise _invalid('Worker counts must be positive integers', field=name, stage=stage)
        elif not isinstance(item, str) or not _DEVICE.fullmatch(item):
            raise _invalid('Device must be cpu, mps or cuda:N', field=name, stage=stage, value=item)
        output[cast(str, stage)] = item
    return output


def _legacy(records: list[Any], raw: Any) -> ParsedManifest:
    selectors: list[str] = []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        item = cast(dict[str, Any], row) if isinstance(row, dict) else None
        selector = row if isinstance(row, str) else next((item[key] for key in SELECTOR_KEYS if item.get(key)), None) if item is not None else None
        if not isinstance(selector, str):
            raise _invalid('Every row requires an explicit source selector', index=index)
        selectors.append(selector)
        rows.append(item if item is not None else {})
    return ParsedManifest(None, selectors, rows, [], None, sha256=canonical_sha256(raw), raw=raw)


def parse_manifest(value: Any) -> ParsedManifest:
    """Validate a manifest (decoded JSON) and return the selected sources; never touches the files it names."""
    if isinstance(value, list):
        return _legacy(cast(list[Any], value), value)
    if not isinstance(value, dict):
        raise _invalid('Manifest must be a JSON object or a list of documents')
    manifest = cast(dict[str, Any], value)
    if 'manifest_version' not in manifest:
        records = manifest.get('documents', [])
        if not isinstance(records, list):
            raise _invalid('Manifest must contain a documents array', field='documents')
        return _legacy(cast(list[Any], records), manifest)
    version = manifest['manifest_version']
    if type(version) is not int or version not in SUPPORTED_VERSIONS:
        raise ManifestError('unsupported_manifest_version', 'Unsupported manifest_version: ' + json.dumps(version) +
                            '; this glossAPI reads ' + ', '.join(map(str, SUPPORTED_VERSIONS)),
                            manifest_version=version, supported=list(SUPPORTED_VERSIONS))
    unknown = sorted(set(manifest) - TOP_LEVEL_KEYS)
    if unknown:
        raise _invalid('Unknown manifest field(s): ' + ', '.join(unknown), fields=unknown)
    documents = manifest.get('documents')
    if not isinstance(documents, list) or not documents:
        raise _invalid('documents must be a non-empty array', field='documents')
    entries = cast(list[Any], documents)
    seen_ids: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise _invalid('Each document entry must be an object', index=index)
        item = cast(dict[str, Any], entry)
        named = [key for key in SELECTOR_KEYS if key in item]
        if len(named) != 1:
            raise _invalid('Each document entry names exactly one source (' + ', '.join(SELECTOR_KEYS) + ')', index=index, fields=named)
        selector = item[named[0]]
        if not isinstance(selector, str) or not selector:
            raise _invalid(named[0] + ' must be a non-empty string', index=index, field=named[0])
        if named[0] == 'path' and not os.path.isabs(selector):
            raise _invalid('path must be an absolute local path', index=index, field='path', value=selector)
        for key in ('selection_id', 'source_id', 'dataset_id'):
            if key in item and (not isinstance(item[key], str) or not item[key]):
                raise _invalid(key + ' must be a non-empty string', index=index, field=key)
        if 'metadata' in item and not isinstance(item['metadata'], dict):
            raise _invalid('metadata must be a JSON object', index=index, field='metadata')
        _check_json(item, f'documents[{index}]')
        if 'selection_id' in item:
            if item['selection_id'] in seen_ids:
                raise _invalid('Duplicate selection_id: ' + item['selection_id'], index=index, field='selection_id',
                               first_index=seen_ids[item['selection_id']])
            seen_ids[item['selection_id']] = index
    selection = manifest.get('selection')
    if selection is not None:
        if not isinstance(selection, list) or not all(isinstance(item, str) for item in cast(list[Any], selection)):
            raise _invalid('selection must be an array of selection_id strings', field='selection')
        missing = [item for item in cast(list[str], selection) if item not in seen_ids]
        if missing:
            raise _invalid('selection names unknown selection_id(s): ' + ', '.join(missing), field='selection', unknown=missing)
    chosen = set(cast(list[str], selection)) if selection is not None else None
    selectors: list[str] = []
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        item = cast(dict[str, Any], entry)
        selector = next(cast(str, item[key]) for key in SELECTOR_KEYS if key in item)
        if chosen is not None and item.get('selection_id') not in chosen:
            exclusions.append(dict(manifest_index=index, selection_id=item.get('selection_id'), selector=selector,
                                   reason_code='not_selected', basis='manifest selection'))
            continue
        selectors.append(selector)
        rows.append(item)
    stages = manifest.get('stages')
    if stages is not None and (not isinstance(stages, list) or not stages or any(stage not in STAGES for stage in cast(list[Any], stages))):
        raise _invalid('stages must be a non-empty array of: ' + ', '.join(STAGES), field='stages')
    products = manifest.get('products')
    if products is not None and (not isinstance(products, list) or any(product not in PRODUCTS for product in cast(list[Any], products))):
        raise _invalid('products must be an array of: ' + ', '.join(PRODUCTS), field='products')
    requested = list(dict.fromkeys([*cast(list[str], stages or []), *(PRODUCTS[p] for p in cast(list[str], products or []))])) or None
    settings = manifest.get('settings', {})
    if not isinstance(settings, dict):
        raise _invalid('settings must be a JSON object', field='settings')
    return ParsedManifest(MANIFEST_VERSION, selectors, rows, exclusions, requested,
                          _assignments(manifest.get('devices'), 'devices', integer=False),
                          _assignments(manifest.get('workers'), 'workers', integer=True),
                          dict(cast(dict[str, Any], settings)), canonical_sha256(manifest), manifest)


def load_manifest(source: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any]) -> ParsedManifest:
    """Parse a manifest from a decoded JSON value or a JSON file path."""
    if isinstance(source, (str, os.PathLike)):
        try:
            value = json.loads(Path(source).read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise _invalid('Manifest file is not readable JSON: ' + str(error), path=str(source)) from None
        return parse_manifest(value)
    return parse_manifest(source)


def build_manifest(documents: Sequence[Mapping[str, Any]], *, selection: Sequence[str] | None = None,
                   stages: Sequence[str] | None = None, products: Sequence[str] | None = None,
                   devices: Mapping[str, str] | None = None, workers: Mapping[str, int] | None = None,
                   settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Python API: a validated v1 manifest (the same object the CLI reads from ``--manifest``)."""
    manifest: dict[str, Any] = {'manifest_version': MANIFEST_VERSION, 'documents': [dict(entry) for entry in documents]}
    for name, item in (('selection', selection), ('stages', stages), ('products', products)):
        if item is not None:
            manifest[name] = list(item)
    for name, mapping in (('devices', devices), ('workers', workers), ('settings', settings)):
        if mapping is not None:
            manifest[name] = dict(mapping)
    parse_manifest(manifest)
    return manifest


def write_manifest(manifest: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Write a manifest as UTF-8 JSON (Greek text stays readable); validates first."""
    parse_manifest(dict(manifest))
    target = Path(path)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return target
