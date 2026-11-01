"""W18 M14: docs/ledger_field_event_map.json maps every ledger field, event kind, reason code, status and snapshot kind
to its writer, readers and tests, and stays equal to a fresh scan of this tree (K4(b)).

Regenerate after a change: ``python tools/ledger_field_event_map.py``.
"""
import ast
import json
from pathlib import Path
import sqlite3
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
MAP = REPO/'docs/ledger_field_event_map.json'


@pytest.fixture(scope='module')
def generator():
    sys.path.insert(0, str(REPO/'tools'))
    try:
        import ledger_field_event_map as module
    finally:
        sys.path.remove(str(REPO/'tools'))
    return module


@pytest.fixture(scope='module')
def stored():
    return json.loads(MAP.read_text())


def test_checked_in_map_equals_a_fresh_scan_of_this_tree(generator, stored):
    fresh = json.loads(generator.serialize(generator.repository_map(generator.build(REPO))))
    assert fresh['counts'] == stored['counts'], 'regenerate: python tools/ledger_field_event_map.py'
    assert fresh == stored, 'regenerate: python tools/ledger_field_event_map.py'


def test_every_remaining_code_and_field_has_a_writer(stored):
    assert not [entry['name'] for entry in stored['entries'] if entry['status'] == 'no_writer']
    assert {entry['status'] for entry in stored['entries']} <= {'active', 'legacy_import_only', 'computed'}


def test_map_objects_equal_the_catalogue_and_a_fresh_ledger(tmp_path, stored):
    from glossapi.ledger import catalogue
    from glossapi.ledger.writer import LedgerWriter
    by_kind = {}
    for entry in stored['entries']:
        by_kind.setdefault(entry['object'], set()).add(entry['name'])
    assert by_kind['event_kind'] == set(catalogue.EVENT_KINDS)
    assert by_kind['reason_code'] == set(catalogue.REASON_CODES)
    assert by_kind['status'] == set(catalogue.STATUSES)
    with LedgerWriter(tmp_path/'corpus.sqlite3'):
        pass
    connection = sqlite3.connect(tmp_path/'corpus.sqlite3')
    names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'")]
    fields = {f'{name}.{column[1]}' for name in names for column in connection.execute(f'PRAGMA table_info("{name}")')}
    connection.close()
    assert by_kind['field'] == fields, (sorted(fields - by_kind['field'])[:10], sorted(by_kind['field'] - fields)[:10])
    literal = set()
    for path in (REPO/'src').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'snapshot' and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                literal.add(node.args[0].value)
    assert by_kind['snapshot_kind'] == literal


def test_every_writer_symbol_and_cited_test_resolves(stored):
    defined = {}
    def names(path):
        if path not in defined:
            tree = ast.parse((REPO/path).read_text())
            found = {'<module>', 'trigger'}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    found.add(node.name)
            defined[path] = found
        return defined[path]
    for entry in stored['entries']:
        for reference in entry['writers'] + entry['readers'] + entry.get('key_readers', []) + entry['report_cli']:
            path, _, function = reference.partition(':')
            assert (REPO/path).is_file(), reference
            assert all(part in names(path) for part in function.split('.') if part), reference
        for test in entry['tests']:
            path, _, function = test.partition('::')
            assert (REPO/path).is_file(), test
            assert not function or function in names(path), test
