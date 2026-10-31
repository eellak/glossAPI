"""Validate glossapi CLI envelopes against the packaged JSON Schemas v1 (test helper; needs the test-group jsonschema).

Also a pytest plugin: ``pytest -p cli_schema_validation tests/test_ledger_cli.py ...`` validates every envelope the
in-process CLI prints during those tests (the existing CLI test corpora) and fails the test that produced an invalid
one. ``GLOSSAPI_SCHEMA_TALLY=<path>`` writes the per-command count of validated envelopes at the end of the session.
"""
from __future__ import annotations

import json
import os
from collections import Counter


def validator():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from glossapi.schemas import NAMES, load_schema
    schemas = {name: load_schema(name) for name in NAMES}
    for value in schemas.values():
        Draft202012Validator.check_schema(value)
    registry = Registry().with_resources([(value['$id'], Resource.from_contents(value)) for value in schemas.values()])
    return {name: Draft202012Validator(value, registry=registry) for name, value in schemas.items()}


_VALIDATORS = None


def validate(envelope, schema='envelope'):
    """Raise AssertionError listing every violation (path and message)."""
    global _VALIDATORS
    if _VALIDATORS is None:
        _VALIDATORS = validator()
    errors = sorted(_VALIDATORS[schema].iter_errors(envelope), key=lambda error: list(error.absolute_path))
    assert not errors, [(list(error.absolute_path), error.message[:300]) for error in errors[:10]]
    return envelope


TALLY = Counter()


def pytest_configure(config):
    import jsonschema  # noqa: F401  (the plugin is requested explicitly; a missing validator must fail, not skip)
    import glossapi.cli as cli
    original = cli._emit

    def checked(envelope, argv, exit_code):
        validate(json.loads(json.dumps(envelope, ensure_ascii=False)))
        TALLY[str(envelope.get('command'))] += 1
        return original(envelope, argv, exit_code)

    cli._emit = checked


def pytest_unconfigure(config):
    target = os.environ.get('GLOSSAPI_SCHEMA_TALLY')
    if target:
        with open(target, 'w') as stream:
            json.dump(dict(TALLY), stream, indent=2, sort_keys=True)
