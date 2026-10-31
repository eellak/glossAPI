"""Versioned JSON Schemas (draft 2020-12) of the glossapi CLI envelope, each command's data and the input manifest.

Package data only: the runtime reads these files but never validates against them (``jsonschema`` is a test
dependency). Compatibility rule (docs/cli_json_schema.md): within v1 only additive optional fields are allowed; any
other change is v2. ``v1/SCHEMA_LOCK.json`` records each file's hash per changelog entry.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, Final, cast

SCHEMA_VERSION: Final[int] = 1
NAMES: Final[tuple[str, ...]] = ('common', 'error', 'envelope', 'plan', 'run', 'status', 'report', 'report_why',
                                 'report_export', 'capabilities', 'doctor', 'manifest')


def schema_files(version: int = SCHEMA_VERSION) -> dict[str, bytes]:
    folder = resources.files(__name__).joinpath('v%d' % version)
    return {name: folder.joinpath(name + '.json').read_bytes() for name in NAMES}


def load_schema(name: str, version: int = SCHEMA_VERSION) -> dict[str, Any]:
    if name not in NAMES:
        raise KeyError('Unknown schema: ' + name)
    folder = resources.files(__name__).joinpath('v%d' % version)
    return cast(dict[str, Any], json.loads(folder.joinpath(name + '.json').read_text(encoding='utf-8')))


def schema_lock(version: int = SCHEMA_VERSION) -> dict[str, Any]:
    folder = resources.files(__name__).joinpath('v%d' % version)
    return cast(dict[str, Any], json.loads(folder.joinpath('SCHEMA_LOCK.json').read_text(encoding='utf-8')))
