"""Fix round 16 (ASTRA-FIXES2c11 on 6d84ddb): validation freshness in the ledger, no side file.

Every ``Corpus.run`` start first commits (synchronous=FULL) a ``validation_started`` run event with a new generation,
before start-up validation or any repair; if that commit fails the start stops, typed, and nothing ran. A pass verdict
records the generation it validated; ``reader.certification`` accepts a pass only for the latest started generation
(with its fingerprint and no later recorded change). A run is completed only if its pass verdict was committed; a failed
verdict append fails the run, typed. The fix-15 side file (``.glossapi/validation.json``) no longer exists, so the c11
side-file histories (deleted, unreadable, replayed, foreign, failed clear) have nothing to act on.
"""
from __future__ import annotations

import errno
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from glossapi import Corpus, cli
from glossapi.ledger import writer as W
from glossapi.ledger.reader import certification, report_view, select_run, status_view
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES

CONTENT = b'{"filename":"uncommitted.pdf","text":"bytes nobody committed"}\n'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _caught(function):
    try:
        return function(), None
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return None, error


def _public(tmp_path):
    output = tmp_path / 'out'
    run = Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    assert run['state'] == 'completed' and run['completion_gate']['verdict'] == 'pass'
    return output, run['run_id'], output / 'export.jsonl'


def _resume(tmp_path, output, run_id):
    return _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))


def _surfaces(output, run_id, capfd):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        status, report = status_view(connection, run=run), report_view(connection, run=run)
        cert = certification(connection)
    seen = {'status': status['state'], 'report': report['state'], 'gate': status['completion_gate'], 'certification': cert}
    for command in ('status', 'report'):
        capfd.readouterr()
        code = cli.main([command, '--corpus', str(output), '--run', run_id, '--json'])
        seen['cli_' + command] = {'exit': code, 'state': json.loads(capfd.readouterr().out)['data']['state']}
    return seen


def _states(seen):
    return {seen['status'], seen['report'], seen['cli_status']['state'], seen['cli_report']['state']}


def _generations(output):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        return [json.loads(row[0])['generation'] for row in
                connection.execute("SELECT details_json FROM events WHERE reason_code='validation_started' ORDER BY event_id")]


def test_every_start_commits_a_new_generation_first_and_the_pass_records_it(tmp_path, capfd):
    output, run_id, path = _public(tmp_path)
    assert not (output / '.glossapi' / 'validation.json').exists()  # no side file
    first = _generations(output)
    run, error = _resume(tmp_path, output, run_id)
    assert error is None and run['state'] == 'completed'
    assert _generations(output) == first + [(first[-1] if first else 0) + 1]
    assert run['completion_gate']['generation'] == _generations(output)[-1]
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'completed'} and seen['certification']['certified']


def test_a_pass_from_an_earlier_generation_is_never_current(tmp_path, monkeypatch, capfd):
    """A start commits its generation and then dies before any verdict (here: validation raises an unexpected error):
    the old pass is no longer current on any surface, though nothing else was recorded."""
    output, run_id, path = _public(tmp_path)
    monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies', lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError('died')))
    _, error = _resume(tmp_path, output, run_id)
    monkeypatch.undo()
    assert isinstance(error, RuntimeError)
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'unverified'} and seen['gate']['stale_reason'] == 'validation_pending', seen
    cert = seen['certification']  # the first public run started on a corpus without a run: its pass has generation None
    assert cert['started_generation'] == _generations(output)[-1] and cert['pass_generation'] != cert['started_generation']
    assert _sha(path) and path.exists()  # the products were never touched; only freshness was revoked


@pytest.mark.parametrize('fault', ['eio', 'missing'])
def test_generation_advanced_then_refusal_unrecordable_is_not_completed_anywhere(tmp_path, monkeypatch, capfd, fault):
    """Astra's c10/c11 history with the side file gone: a real pass; the start's export flush fails (or the export
    vanishes); the refusal's own ledger append fails as on a full ledger. Every surface: not completed."""
    output, run_id, path = _public(tmp_path)
    real_flush, real_event = W._fsync_path, LedgerWriter.event

    def flush(target):
        if Path(target) == path:
            if fault == 'missing':
                path.unlink(missing_ok=True)
                return real_flush(target)
            raise OSError(errno.EIO, 'start flush failed')
        return real_flush(target)

    def event(self, *args, **kwargs):
        if kwargs.get('reason_code') == 'completion_gate' and kwargs.get('message') == 'refused':
            raise sqlite3.OperationalError('database or disk is full')
        return real_event(self, *args, **kwargs)
    monkeypatch.setattr(W, '_fsync_path', flush)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    _, error = _resume(tmp_path, output, run_id)
    monkeypatch.undo()
    assert getattr(error, 'code', None) == 'unresolved_recovery' and error.details['refusal_recorded'] is False
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'unverified'} and seen['gate']['stale_reason'] == 'validation_pending', seen
    assert seen['cli_status']['exit'] == 0 and seen['cli_report']['exit'] == 0


@pytest.mark.parametrize('failure', ['disk_full', 'eio'])
def test_a_generation_that_cannot_be_committed_stops_the_start_before_anything_ran(tmp_path, monkeypatch, capfd, failure):
    output, run_id, path = _public(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.record_uncommitted_write(run_id, path, hashlib.sha256(CONTENT).hexdigest(), len(CONTENT), wrapper_stage='export', existed=True)
    path.write_bytes(CONTENT)
    before = _generations(output)
    repairs = []
    real_verify, real_event = LedgerWriter.verify_excluded_copies, LedgerWriter.event
    monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies', lambda self, *a, **k: repairs.append(1) or real_verify(self, *a, **k))

    def event(self, *args, **kwargs):
        if kwargs.get('reason_code') == 'validation_started':
            raise sqlite3.OperationalError('database or disk is full' if failure == 'disk_full' else 'disk I/O error')
        return real_event(self, *args, **kwargs)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    _, error = _resume(tmp_path, output, run_id)
    monkeypatch.undo()
    assert 'validation_start_unrecorded' in str(error) and 'nothing ran' in str(error), error
    assert getattr(error, 'code', None) == ('disk_full' if failure == 'disk_full' else 'unresolved_recovery')
    assert repairs == [] and path.read_bytes() == CONTENT and _generations(output) == before


@pytest.mark.parametrize('failure', ['eio', 'disk_full'])
def test_a_verdict_that_cannot_be_committed_never_completes_the_run(tmp_path, monkeypatch, capfd, failure):
    """The pass verdict's own append fails: Corpus.run fails typed, CLI run exits non-zero without a completed state,
    and no surface says completed (the generation this start committed has no pass)."""
    output, run_id, path = _public(tmp_path)
    real_event = LedgerWriter.event

    def event(self, *args, **kwargs):
        if kwargs.get('reason_code') == 'completion_gate' and kwargs.get('message') == 'pass':
            raise sqlite3.OperationalError('database or disk is full' if failure == 'disk_full' else 'disk I/O error')
        return real_event(self, *args, **kwargs)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    _, error = _resume(tmp_path, output, run_id)
    assert getattr(error, 'code', None) == ('disk_full' if failure == 'disk_full' else 'unresolved_recovery'), error
    if failure == 'eio':
        assert [entry['reason'] for entry in error.conditions] == ['completion_verdict_unrecorded'] and error.boundary == 'completion'
    capfd.readouterr()
    code = cli.main(['run', '--corpus', str(output), '--resume', run_id, '--retry-failed', '--json'])
    envelope = json.loads(capfd.readouterr().out)
    monkeypatch.undo()
    assert code != 0 and (envelope['data'] or {}).get('state') != 'completed', envelope
    seen = _surfaces(output, run_id, capfd)
    assert 'completed' not in _states(seen) and seen['gate']['stale_reason'] == 'validation_pending', seen
    run, error = _resume(tmp_path, output, run_id)
    assert error is None and run['state'] == 'completed' and _states(_surfaces(output, run_id, capfd)) == {'completed'}
