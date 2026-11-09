"""Fix round 17 (ASTRA-FIXES2c12 on eabd363): a pass without a ``generation`` key is never current; the failed-start
note claims a generation only when its commit happened.

A fix-15 (and earlier) pass verdict has no ``generation`` key at all; a fix-16+ first run on a corpus without a run
records an explicit ``generation: null``. ``reader.certification`` tells them apart: the legacy pass is never current
(``legacy_pass_without_generation``) until a fix-16+ run physically revalidates and commits a new pass.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from glossapi import Corpus, cli
from glossapi.ledger import writer as W
from glossapi.ledger.compact import LedgerConnection
from glossapi.ledger.reader import certification, report_view, select_run, status_view
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES


def _caught(function):
    try:
        return function(), None
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return None, error


def _surfaces(output, run_id, capfd):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        status, report = status_view(connection, run=run), report_view(connection, run=run)
        cert = certification(connection)
    seen = {'status': status['state'], 'report': report['state'], 'certification': cert}
    for command in ('status', 'report'):
        capfd.readouterr()
        code = cli.main([command, '--corpus', str(output), '--run', run_id, '--json'])
        seen['cli_' + command] = {'exit': code, 'state': json.loads(capfd.readouterr().out)['data']['state']}
    return seen


def _states(seen):
    return {seen['status'], seen['report'], seen['cli_status']['state'], seen['cli_report']['state']}


def _first_run(tmp_path):
    output = tmp_path / 'out'
    run = Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    assert run['state'] == 'completed'
    return output, run


def test_explicit_null_generation_of_a_first_run_is_current(tmp_path, capfd):
    output, run = _first_run(tmp_path)
    with open_readonly(output / 'corpus.sqlite3') as connection:
        details = json.loads(connection.execute("SELECT details_json FROM events WHERE reason_code='completion_gate' "
                                                "ORDER BY event_id DESC LIMIT 1").fetchone()[0])
    assert 'generation' in details and details['generation'] is None  # explicit null: no run row existed at the start
    assert _states(_surfaces(output, run['run_id'], capfd)) == {'completed'}


@pytest.mark.parametrize('export', ['intact', 'missing'])
def test_a_pass_without_a_generation_key_is_never_current(tmp_path, capfd, export):
    """The fix-15 ledger shape read by the new reader: the latest verdict is a pass whose details have no ``generation``
    key (as fix 15's gate wrote it, with the fingerprint of the current authority) and no ``validation_started`` record
    exists. Never current, whether or not the products are intact; a fix-16+ run then revalidates and completes. The
    real old-code -> new-reader transition is Astra's c12 probe (``make_fix15_history.py``, run in the receipts)."""
    output, run = _first_run(tmp_path)
    run_id = run['run_id']
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        fingerprint = W.authority_fingerprint(W.expected_products(writer.connection))
        legacy = {'verdict': 'pass', 'boundary': 'completion', 'products': run['completion_gate']['products'], 'bytes_read': 1,
                  'passes': 1, 'elapsed_ns': 1, 'failures': [], 'failure_count': 0, 'authority_fingerprint': fingerprint}
        with writer.transaction():
            writer.event(run_id, event_kind='run', reason_code='completion_gate', message='pass', details=legacy)
        assert writer.connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='validation_started'").fetchone()[0] == 0
    if export == 'missing':
        (output / 'export.jsonl').unlink()
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'unverified'}, seen
    assert seen['certification']['stale_reason'] == 'legacy_pass_without_generation'
    assert not (output / '.glossapi' / 'validation.json').exists()  # nothing to trust or replay
    result, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    assert error is None and result['state'] == 'completed' and result['completion_gate']['generation'] == 1
    assert (output / 'export.jsonl').exists() and _states(_surfaces(output, run_id, capfd)) == {'completed'}


@pytest.mark.parametrize('commit', ['fails', 'succeeds'])
def test_the_failed_start_note_claims_a_generation_only_when_it_was_committed(tmp_path, monkeypatch, capfd, commit):
    output, run = _first_run(tmp_path)
    real_event, real_commit = LedgerWriter.event, LedgerConnection.commit

    def event(self, *args, **kwargs):  # the refusal's own append fails, as on a full ledger
        if kwargs.get('reason_code') == 'completion_gate' and kwargs.get('message') == 'refused':
            raise sqlite3.OperationalError('database or disk is full')
        return real_event(self, *args, **kwargs)

    def failing_commit(connection):
        row = connection.execute(f'SELECT reason_code FROM {connection.event_table} ORDER BY event_id DESC LIMIT 1').fetchone()
        if row and row[0] == 'validation_started':
            raise sqlite3.OperationalError('disk I/O error')
        return real_commit(connection)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    if commit == 'fails':
        monkeypatch.setattr(LedgerConnection, 'commit', failing_commit)
    else:  # the generation commits; the start is then refused (a validation failure)
        monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies',
                            lambda self, *a, **k: (_ for _ in ()).throw(W.UnresolvedRecoveryError([{'path': 'x', 'reason': 'hard_linked'}])))
    _, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run['run_id'], retry_failed=True))
    monkeypatch.undo()
    notes = ' '.join(getattr(error, '__notes__', []))
    assert error.details['refusal_recorded'] is False
    if commit == 'fails':
        assert 'already committed' not in notes and 'validation never started and no new freshness record was committed' in notes
    else:
        assert 'this start already committed validation generation 1' in notes
        assert 'completed' not in _states(_surfaces(output, run['run_id'], capfd))
