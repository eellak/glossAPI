"""Fix round 18 (ASTRA-FIXES2c13 M1): exact generation and fingerprint types before any certification comparison.

A verdict's ``generation`` must be a plain ``int`` (never ``bool``/``float``/string) or an explicit ``None`` (a first run
on a corpus without a run); the latest ``validation_started`` record's generation must be a plain ``int``; the
``authority_fingerprint`` must be a 64-hex string. Anything else is never current (``malformed_generation``). These are
reader-shape probes: verdicts appended through ``LedgerWriter.event`` after a real run, no physical gate.
"""
from __future__ import annotations

import json

import pytest

from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter
from test_fix17_astra import _first_run, _surfaces, _states


def _append_pass(writer, run_id, details):
    with writer.transaction():
        writer.event(run_id, event_kind='run', reason_code='completion_gate', message='pass', details=details)


def _latest_details(writer):
    return json.loads(writer.connection.execute("SELECT details_json FROM events WHERE reason_code='completion_gate' "
                                                "ORDER BY event_id DESC LIMIT 1").fetchone()[0])


@pytest.mark.parametrize('shape,field', [
    ('bool', 'verdict.generation'), ('float', 'verdict.generation'), ('string', 'verdict.generation'),
    ('missing_fingerprint', 'verdict.authority_fingerprint'), ('malformed_fingerprint', 'verdict.authority_fingerprint'),
    ('uppercase_fingerprint', 'verdict.authority_fingerprint')])
def test_a_malformed_verdict_is_never_current(tmp_path, capfd, shape, field):
    output, run = _first_run(tmp_path)
    run_id = run['run_id']
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        assert writer.begin_validation(run_id) == 1
        details = _latest_details(writer)
        details['generation'] = {'bool': True, 'float': 1.0, 'string': '1'}.get(shape, 1)
        if shape == 'missing_fingerprint':
            details.pop('authority_fingerprint')
        elif shape == 'malformed_fingerprint':
            details['authority_fingerprint'] = details['authority_fingerprint'][:63]
        elif shape == 'uppercase_fingerprint':
            details['authority_fingerprint'] = details['authority_fingerprint'].upper()
        _append_pass(writer, run_id, details)
    seen = _surfaces(output, run_id, capfd)
    assert 'completed' not in _states(seen) and not seen['certification']['certified'], seen
    assert seen['certification']['stale_reason'] == 'malformed_generation' and seen['certification']['malformed_field'] == field


@pytest.mark.parametrize('start', [True, 1.0, '1'])
def test_a_malformed_start_record_is_never_trusted(tmp_path, capfd, start):
    """A ``validation_started`` record whose generation is not a plain int: no pass is current against it, even one
    whose generation compares equal under Python's loose equality."""
    output, run = _first_run(tmp_path)
    run_id = run['run_id']
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        details = _latest_details(writer)
        with writer.transaction():
            writer.event(run_id, event_kind='run', reason_code='validation_started', message='validation started', details={'generation': start})
        details['generation'] = 1
        _append_pass(writer, run_id, details)
        assert W.latest_validation_generation(writer.connection) is None  # the writer never builds on a malformed record
    seen = _surfaces(output, run_id, capfd)
    assert 'completed' not in _states(seen), seen
    assert seen['certification']['stale_reason'] == 'malformed_generation'
    assert seen['certification']['malformed_field'] == 'validation_started.generation'


def test_well_typed_controls_stay_current(tmp_path, capfd):
    """Controls: a plain int generation equal to the latest start, with the real 64-hex fingerprint, is current; the
    first run's explicit None (no start record) is current."""
    output, run = _first_run(tmp_path)
    run_id = run['run_id']
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}  # explicit None, no start record
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        details = _latest_details(writer)
        details['generation'] = writer.begin_validation(run_id)
        _append_pass(writer, run_id, details)
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}
