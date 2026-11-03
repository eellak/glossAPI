"""Fix round 15 (ASTRA-FIXES2c10 on 4f1c98b): path binding, authority fingerprint, durable pending validation.

H1 a registration is committed only by destination-specific membership: the registration itself in its attempt's
output manifest, or a publication record of exactly (path, bytes, attempt). H2 every verdict carries the fingerprint of
the full ``expected_products`` map it verified; every read surface recomputes it, so a change to any input of the
authority after a pass makes the pass stale. H3 a run start durably marks the corpus validation pending (a fixed-size
record overwritten in place) before validation or repair; only a passing gate clears it; if the mark cannot be written
the start stops before any repair.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from glossapi import Corpus, cli
from glossapi.ledger import writer as W
from glossapi.ledger.reader import certification, report_view, select_run, status_view
from glossapi.ledger.writer import LedgerWriter, open_readonly
from glossapi.stages.compatibility import register_publication
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES
from test_fix14_astra import _wrapper_registration

CONTENT = b'{"filename":"uncommitted.pdf","text":"bytes nobody committed"}\n'


def _sha(data_or_path):
    data = data_or_path if isinstance(data_or_path, bytes) else Path(data_or_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


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


def _surfaces(output, run_id, capfd):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        status, report = status_view(connection, run=run), report_view(connection, run=run)
    seen = {'status': status['state'], 'report': report['state'], 'gate': status['completion_gate']}
    for command in ('status', 'report'):
        capfd.readouterr()
        code = cli.main([command, '--corpus', str(output), '--run', run_id, '--json'])
        envelope = json.loads(capfd.readouterr().out)
        seen['cli_' + command] = {'exit': code, 'state': envelope['data']['state']}
    return seen


def _states(seen):
    return {seen['status'], seen['report'], seen['cli_status']['state'], seen['cli_report']['state']}


# ============================================================= H1: destination-specific binding ==============

def test_bytes_produced_at_one_path_never_bind_another(tmp_path, capfd):
    """Astra's H1 shape: the successful producer of a markdown product registers those markdown bytes at export.jsonl
    (no binding of that path, absent from its manifest). Unbound: the export obligation stands; the resume restores
    the export. The same registration with an explicit publication binding of (path, bytes, attempt) is committed."""
    output, run_id, path = _public(tmp_path)
    original = _sha(path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        expected = W.expected_products(writer.connection)
        donor_uri, donor = next((uri, value) for uri, value in expected.items()
                                if isinstance(value, W.ExpectedSha) and uri.endswith('.md') and value.sha256 != original)
        row = writer.file_catalog()[donor.registration]
        data = Path(unquote(urlsplit(donor_uri).path)).read_bytes()
        producer_run = writer.connection.execute('SELECT t.run_id FROM tasks t JOIN attempts a ON a.task_id=t.task_id WHERE a.attempt_id=?',
                                                 (row['producer_attempt_id'],)).fetchone()[0]
        writer.record_uncommitted_write(run_id, path, row['sha256'], len(data), wrapper_stage='export', existed=True)
        path.write_bytes(data)
        with writer.transaction():
            late = writer.artifact(producer_run, row['stage'], 'document_jsonl', path.as_uri(), sha256=row['sha256'], bytes=len(data),
                                   producer_attempt_id=row['producer_attempt_id'], version_id=row['version_id'], retention_class='final')
        assert W._registration_classes(writer.connection, [(late, path.as_uri(), row['sha256'], row['producer_attempt_id'], 0)])[late] == 'unbound'
        assert W.expected_products(writer.connection)[path.as_uri()].sha256 == original
        assert writer.completion_gate(run_id)['verdict'] == 'refused'
    run, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    assert error is None and run['state'] == 'completed' and _sha(path) == original
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}
    with LedgerWriter(output / 'corpus.sqlite3') as writer:  # an explicit binding of exactly (path, bytes, attempt) commits it
        with writer.transaction():
            writer.snapshot('publication_binding', {'kind': 'publication_binding', 'product_uri': path.as_uri(), 'sha256': row['sha256'],
                                                    'producer_attempt_id': row['producer_attempt_id'], 'file_id': late})
        assert W.expected_products(writer.connection)[path.as_uri()].registration == late


# ================================================= H2: authority fingerprint, every input of expected_products =

def _after_pass(writer, run_id, change, path):
    """Change one input of ``expected_products`` after a pass, through the writer's API."""
    expected = W.expected_products(writer.connection)[path.as_uri()]
    committed = writer.file_catalog()[expected.registration]
    if change == 'registration':  # a newer committed generation (with its publication record)
        with writer.transaction():
            manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(b'fix15-newer')]})
            task = writer.task(run_id, 'export', 'export.jsonl', {}, scope_key='corpus:' + manifest)
            attempt = writer.attempt(task, worker_id='fix15', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            writer.artifact(run_id, 'export', 'document_jsonl', path.as_uri(), producer_attempt_id=attempt, sha256=_sha(CONTENT),
                            bytes=len(CONTENT), retention_class='final')
            writer.snapshot('product_projection', {'kind': 'product_projection', 'product_uri': path.as_uri(), 'immutable_uri': None,
                                                   'sha256': _sha(CONTENT), 'producer_attempt_id': attempt})
            writer.finish(run_id, task, attempt, status='ok', reason_code='export_completed')
    elif change == 'binding':  # a snapshot-only publication binding of an observation the pass already covered
        row = writer.file_catalog()[writer.binding_case_observation]
        observed = row['file_id']
        before = writer.connection.execute('SELECT MAX(event_id) FROM events').fetchone()[0]
        with writer.transaction():
            reused = register_publication(writer, run_id, row['stage'], row['kind'], row['uri'], version_id=row['version_id'],
                                          producer_attempt_id=row['producer_attempt_id'], sha256=row['sha256'], bytes=row['bytes'],
                                          retention_class=row['retention_class'])
        assert reused == observed
        traced = writer.connection.execute('SELECT reason_code FROM events WHERE event_id>?', (before,)).fetchall()
        assert [code for code, in traced] == ['publication_bound']  # traced atomically with the binding
    elif change == 'invalidation':
        with writer.transaction():
            writer.event(run_id, event_kind='invalidation', reason_code='registration_invalidated', file_id=committed['file_id'],
                         version_id=committed['version_id'], details={})
    elif change == 'deletion':
        with writer.transaction():
            writer.event(run_id, event_kind='retention', reason_code='delete_completed', file_id=committed['file_id'],
                         version_id=committed['version_id'], details={})
    elif change == 'attempt_status':
        # A bound registration at a new product path whose producer is still running, then that attempt's terminal
        # status: each changes the authority (Refuse(producer_unfinished) -> Sha). A finished attempt's status cannot
        # change (the writer refuses a second terminal event), so this is the status change a pass can meet.
        extra = path.parent / 'fix15-extra.jsonl'
        extra.write_bytes(CONTENT)
        with writer.transaction():
            manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(b'fix15-status')]})
            task = writer.task(run_id, 'export', 'extra.jsonl', {}, scope_key='corpus:' + manifest)
            attempt = writer.attempt(task, worker_id='fix15', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            writer.artifact(run_id, 'export', 'document_jsonl', extra.as_uri(), producer_attempt_id=attempt, sha256=_sha(CONTENT),
                            bytes=len(CONTENT), retention_class='final')
            writer.snapshot('product_projection', {'kind': 'product_projection', 'product_uri': extra.as_uri(), 'immutable_uri': None,
                                                   'sha256': _sha(CONTENT), 'producer_attempt_id': attempt})
        running = W.authority_fingerprint(W.expected_products(writer.connection))
        assert repr(W.expected_products(writer.connection)[extra.as_uri()]) == 'Refuse(producer_unfinished)'
        with writer.transaction():
            writer.finish(run_id, task, attempt, status='ok', reason_code='export_completed')
        assert W.authority_fingerprint(W.expected_products(writer.connection)) != running


@pytest.mark.parametrize('change', ['registration', 'binding', 'invalidation', 'deletion', 'attempt_status'])
def test_any_change_to_the_authority_after_a_pass_makes_it_stale_on_every_surface(tmp_path, capfd, change):
    output, run_id, path = _public(tmp_path)
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        if change == 'binding':  # the observation exists before a fresh passing gate; the binding comes after it
            writer.binding_case_observation = _wrapper_registration(writer, run_id, path, CONTENT, 'document_jsonl')
            assert writer.completion_gate(run_id)['verdict'] == 'pass'
        before = W.authority_fingerprint(W.expected_products(writer.connection))
        _after_pass(writer, run_id, change, path)
        after = W.authority_fingerprint(W.expected_products(writer.connection))
    assert before != after
    seen = _surfaces(output, run_id, capfd)
    assert 'completed' not in _states(seen), seen
    if change in ('binding', 'deletion'):  # no product artifact event follows the pass: only the fingerprint can see it
        assert seen['gate']['stale_reason'] == 'authority_changed', seen
    with open_readonly(output / 'corpus.sqlite3') as connection:
        assert certification(connection)['certified'] is False


def test_the_fingerprint_is_stable_for_an_unchanged_ledger_and_cheap_to_recompute(tmp_path, capfd):
    output, run_id, path = _public(tmp_path)
    with open_readonly(output / 'corpus.sqlite3') as connection:
        first, second = certification(connection), certification(connection)
        recorded = json.loads(connection.execute("SELECT details_json FROM events WHERE reason_code='completion_gate' "
                                                 "ORDER BY event_id DESC LIMIT 1").fetchone()[0])['authority_fingerprint']
        assert recorded == W.authority_fingerprint(W.expected_products(connection))
    assert first['certified'] and second['certified'] and first['authority_check_ns'] > 0


# ============================================================ H3: durable pending validation ================

@pytest.mark.parametrize('recording', ['full', 'writable'])
@pytest.mark.parametrize('fault', ['eio', 'missing'])
def test_a_refusal_the_ledger_cannot_record_still_revokes_the_pass(tmp_path, monkeypatch, capfd, fault, recording):
    """A real pass; at the start's export flush, EIO (or the export vanishes); the refusal's ledger event fails as on a
    full ledger (``recording='full'``) or is written (control). Full: the error says the refusal was not recorded and
    every surface reads ``unverified`` (validation pending). Writable: ``unresolved_recovery``, CLI status exit 3."""
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
        if kwargs.get('reason_code') == 'completion_gate' and kwargs.get('message') == 'refused' and recording == 'full':
            raise sqlite3.OperationalError('database or disk is full')
        return real_event(self, *args, **kwargs)
    monkeypatch.setattr(W, '_fsync_path', flush)
    monkeypatch.setattr(LedgerWriter, 'event', event)
    _, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    monkeypatch.undo()
    assert getattr(error, 'code', None) == 'unresolved_recovery', error
    seen = _surfaces(output, run_id, capfd)
    # Fix 16: the side-file mark is replaced by the ledger's validation generation, committed first at every start.
    with open_readonly(output / 'corpus.sqlite3') as connection:
        assert connection.execute("SELECT COUNT(*) FROM events WHERE reason_code='validation_started'").fetchone()[0] >= 1
    if recording == 'full':
        assert error.details['refusal_recorded'] is False and 'could not be recorded' in ' '.join(getattr(error, '__notes__', []))
        assert _states(seen) == {'unverified'} and seen['gate']['stale_reason'] == 'validation_pending', seen
    else:
        assert error.details['refusal_recorded'] is True
        assert _states(seen) == {'unresolved_recovery'} and seen['cli_status']['exit'] == 3, seen
    run, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    assert error is None and run['state'] == 'completed' and run['completion_gate']['generation'] is not None
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}


@pytest.mark.parametrize('code', [errno.EIO, errno.ENOSPC])
def test_a_start_that_cannot_write_the_pending_mark_stops_before_any_repair(tmp_path, monkeypatch, capfd, code):
    """Fix 16: the start's first commit is the ledger's validation generation (the fix-15 side-file mark is gone); if it
    cannot be committed, the start stops before any validation or repair."""
    output, run_id, path = _public(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.record_uncommitted_write(run_id, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
    path.write_bytes(CONTENT)
    repairs = []
    real_verify = LedgerWriter.verify_excluded_copies
    monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies', lambda self, *a, **k: repairs.append(1) or real_verify(self, *a, **k))

    def unwritable(self, run_id):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(LedgerWriter, 'begin_validation', unwritable)
    _, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    monkeypatch.undo()
    assert 'validation_start_unrecorded' in str(error) and 'nothing ran' in str(error), error
    if code == errno.ENOSPC:  # an exhausted disk stays typed disk_full (resume after freeing space)
        assert getattr(error, 'code', None) == 'disk_full'
    else:
        assert getattr(error, 'code', None) == 'unresolved_recovery' and error.boundary == 'start'
        assert [entry['reason'] for entry in error.conditions] == ['validation_start_unrecorded']
    assert repairs == [] and _sha(path) == _sha(CONTENT)  # nothing validated, nothing repaired
