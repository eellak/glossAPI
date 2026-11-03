"""Fix round 14 (ASTRA-FIXES2c9 on 566f0c8): the completion gate's own predicates and surfaces.

R1 a deletion discharges a path only when it names the path's committed obligation (the newest committed
generation's own registration); R5 a registration is committed only with a positive binding (a publication record, or
the producer attempt's own record of producing exactly these bytes); R2/R3 one completion predicate
(``reader.completion_state``) runs last on every read surface, and a pass is bound to its validation epoch (its verdict
event id): a later recorded mutation, rollback, invalidation or refusal makes it stale; R4 nothing writes a product
after the verdict, and the gate's cost is in the verdict event; FIX13-G1 wrapper-only paths are left out only for an
allowlisted diagnostic role; L1 the refusal text names its boundary.
"""
from __future__ import annotations

import builtins
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi import Corpus, cli
from glossapi.ledger import writer as W
from glossapi.ledger.reader import report_view, select_run, status_view
from glossapi.ledger.writer import LedgerWriter, UnresolvedRecoveryError, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES
from test_fix12_astra import _register_generation
from test_fix13_astra import _break_recovery
from test_k4c2_fixes import published_copy

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


def _resume(tmp_path, output, run_id):
    return _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))


def _surfaces(output, run_id, capfd):
    """Every read surface of the run, each through a fresh read-only connection or CLI process call."""
    with open_readonly(output / 'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        status, report = status_view(connection, run=run), report_view(connection, run=run)
        plan_mode = json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?', (run['plan_sha256'],)).fetchone()[0]).get('mode')
    seen = {'status': status['state'], 'report': report['state'], 'gate': status['completion_gate'], 'plan_mode': plan_mode}
    for command in ('status', 'report'):
        capfd.readouterr()
        code = cli.main([command, '--corpus', str(output), '--run', run_id, '--json'])
        envelope = json.loads(capfd.readouterr().out)
        seen['cli_' + command] = {'exit': code, 'state': envelope['data']['state'], 'errors': [e['code'] for e in envelope['errors']]}
    return seen


def _states(seen):
    return {seen['status'], seen['report'], seen['cli_status']['state'], seen['cli_report']['state']}


def _wrapper_registration(writer, run_id, path, data, kind):
    """A legacy compatibility wrapper (``orchestrator:<pid>``) registers bytes at ``path`` as its observation."""
    with writer.transaction():
        manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(kind.encode())]})
        task = writer.task(run_id, 'clean', 'clean.clean', {}, scope_key='corpus:' + manifest)
        attempt = writer.attempt(task, worker_id='orchestrator:4242', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        file_id = writer.artifact(run_id, 'clean', kind, path.as_uri(), producer_attempt_id=attempt, sha256=_sha(data), bytes=len(data))
        writer.finish(run_id, task, attempt, status='ok', reason_code='clean_completed')
    return file_id


# ===================================================================== R1: deletion bound to the obligation ====

def test_deletion_of_an_unrelated_registration_never_discharges_the_committed_product(tmp_path):
    """Astra's R1 shape, and fix 13's node case: a committed final export; a later wrapper temporary registered at the
    same path and deleted by real retention. The export obligation stands (Sha), and the gate refuses its absence.
    Positive control: a committed deletion of the committed registration itself -> Absent(policy_deletion)."""
    from glossapi.stages.retention import RetentionController
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        uri = path.as_uri()
        data = b'{"temporary":"report"}\n'
        path.write_bytes(data)
        observed = _wrapper_registration(writer, run, path, data, 'temporary_cleaning_report')
        controller = RetentionController(writer, tmp_path, run, {'debug': 'off'})
        consumer = controller.seal([writer.file_catalog()[observed]])
        controller.declare(observed, [consumer])
        controller.delete(observed)
        assert not path.exists()
        expected = W.expected_products(writer.connection)[uri]
        assert repr(expected) == f'Sha({_sha(source)[:12]})' and expected.registration == copy_id
        verdict = writer.completion_gate(run)
        assert verdict['verdict'] == 'refused' and [f['reason'] for f in verdict['all_failures']] == ['completion_gate_missing']
        with writer.transaction():
            writer.event(run, event_kind='retention', reason_code='delete_completed', file_id=copy_id, details={})
        expected = W.expected_products(writer.connection)[uri]
        assert repr(expected) == 'Absent(policy_deletion)' and expected.registration == copy_id
        assert writer.completion_gate(run)['verdict'] == 'pass'


def test_a_report_declared_temporary_is_never_published_as_a_metadata_projection(tmp_path):
    """The node OCR case re-solved (fix 13's cleaning_report.parquet): the OCR stage's nested clean defers its report
    (a ``temporary_cleaning_report`` registration of these bytes) on its own journal; the enclosing journal's metadata
    sealing must not publish those bytes as a committed projection. No obligation is created, so retention's deletion
    of the temporary needs no authority over a product."""
    from glossapi.stages import metadata
    report = tmp_path / 'cleaning_report.parquet'
    report.write_bytes(b'PAR1 temporary report bytes')
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='fix14')
        journal = SimpleNamespace(writer=writer, corpus=SimpleNamespace(output_dir=tmp_path), adapter_stages={'clean'},
                                  cleanup_paths=set(), stage='ocr')
        assert not metadata._deferred_temporary_report(journal, report)
        _wrapper_registration(writer, run, report, report.read_bytes(), 'temporary_cleaning_report')
        assert metadata._deferred_temporary_report(journal, report)
        metadata.seal_metadata_projections(journal)  # nothing to publish: the only candidate is declared temporary
        projections = [json.loads(row[0])['product_uri'] for row in
                       writer.connection.execute("SELECT json FROM snapshots WHERE kind='product_projection'")]
        assert report.as_uri() not in projections
        assert report.as_uri() not in W.expected_products(writer.connection)  # wrapper-only, allowlisted role
        report.write_bytes(b'PAR1 other bytes')  # bytes not declared temporary would still be sealed
        assert not metadata._deferred_temporary_report(journal, report)


# ============================================================= R5: commitment needs a positive binding ========

def test_a_successful_reader_with_no_output_binding_is_never_authoritative(tmp_path):
    """Astra's R5 shape: a successful attempt with an empty output manifest and no publication record registers bytes at
    a product path. Unbound: passed over (the committed generation stands); alone, the path is refused. A second named
    copy of bytes the attempt did record producing is bound."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        uri = path.as_uri()
        with writer.transaction():
            manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(b'sealer')]})
            task = writer.task(run, 'export', 'evidence.seal', {}, scope_key='corpus:' + manifest)
            sealer = writer.attempt(task, worker_id='evidence-sealer', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            late = writer.artifact(run, 'export', 'document_jsonl', uri, producer_attempt_id=sealer, sha256=_sha(CONTENT),
                                   bytes=len(CONTENT), retention_class='final')
            lonely = writer.artifact(run, 'export', 'document_jsonl', (tmp_path / 'lonely.jsonl').as_uri(), producer_attempt_id=sealer,
                                     sha256=_sha(CONTENT), bytes=len(CONTENT), retention_class='final')
            writer.finish(run, task, sealer, status='ok', reason_code='export_completed')
        classes = W._registration_classes(writer.connection, [(late, uri, _sha(CONTENT), sealer, 0), (copy_id, uri, _sha(source), None, 0)])
        assert classes[late] == 'unbound'
        expected = W.expected_products(writer.connection)
        assert expected[uri].registration == copy_id
        assert repr(expected[(tmp_path / 'lonely.jsonl').as_uri()]) == 'Refuse(no_committed_registration)'
        second = tmp_path / 'second.jsonl'
        second.write_bytes(source.read_bytes())
        producer = writer.file_catalog()[copy_id]['producer_attempt_id']
        with writer.transaction():
            named = writer.artifact(run, 'export', 'document_jsonl', second.as_uri(), producer_attempt_id=producer,
                                    sha256=_sha(source), bytes=len(source.read_bytes()), retention_class='final')
        assert W.expected_products(writer.connection)[second.as_uri()].registration == named


# ============================================== R2/R3: one completion predicate, bound to its epoch ===========

def test_no_surface_says_completed_when_the_gate_never_ran(tmp_path, monkeypatch, capfd):
    """R2: EIO at the run's final flush, before the gate; every surface (status, report, both CLI commands), whatever
    the latest task or streaming run event says, reports ``unverified``."""
    output = tmp_path / 'out'
    real = LedgerWriter.durable_barrier
    hits = []

    def fail_final_flush(self, *args, **kwargs):
        if self.completion_gate_run:
            hits.append(self.completion_gate_run)
            raise OSError(errno.EIO, 'final flush failed before the gate')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(LedgerWriter, 'durable_barrier', fail_final_flush)
    _, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6))
    monkeypatch.undo()
    assert hits and isinstance(error, OSError)
    with open_readonly(output / 'corpus.sqlite3') as connection:
        latest = connection.execute("SELECT details_json FROM events WHERE run_id=? AND event_kind='run' AND reason_code<>'completion_gate' "
                                    "ORDER BY event_id DESC LIMIT 1", (hits[0],)).fetchone()
    seen = _surfaces(output, hits[0], capfd)
    assert seen['gate'] is None and _states(seen) == {'unverified'}, (seen, latest)
    assert seen['cli_status']['exit'] == 0 and seen['cli_report']['exit'] == 0


def test_a_refused_start_makes_the_old_pass_stale_on_every_surface(tmp_path, capfd):
    """R3, Astra's case: after a pass, a recorded uncommitted write is installed and hard-linked; two resumes refuse at
    their start; after each, every surface says ``unresolved_recovery`` (CLI status exits 3), never the old pass."""
    output, run_id, path = _public(tmp_path)
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.record_uncommitted_write(run_id, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
    path.write_bytes(CONTENT)
    os.link(path, tmp_path / 'hardlink')
    for _ in range(2):
        _, error = _resume(tmp_path, output, run_id)
        assert getattr(error, 'code', None) == 'unresolved_recovery' and error.boundary == 'start'
        seen = _surfaces(output, run_id, capfd)
        assert _states(seen) == {'unresolved_recovery'}, seen
        assert seen['cli_status']['exit'] == 3 and seen['gate']['verdict'] == 'refused'
        assert _sha(path) == _sha(CONTENT)
    (tmp_path / 'hardlink').unlink()
    run, error = _resume(tmp_path, output, run_id)
    assert error is None and run['state'] == 'completed'
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}


def test_a_later_recorded_write_or_single_phase_call_makes_the_pass_stale_until_verified_again(tmp_path, capfd):
    """R3: a pass certifies the corpus only up to its epoch. A later recorded uncommitted write (no refusal) -> stale
    (``unverified``); a later passing gate re-certifies. A later single-phase call that registers products -> stale."""
    output, run_id, path = _public(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.record_uncommitted_write(run_id, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'unverified'} and seen['gate']['stale_reason'] == 'later_uncommitted_projection_write', seen
    run, error = _resume(tmp_path, output, run_id)
    assert error is None and run['state'] == 'completed'
    assert _states(_surfaces(output, run_id, capfd)) == {'completed'}
    Corpus(tmp_path / 'empty', output).jsonl(output / 'single-phase.jsonl')
    seen = _surfaces(output, run_id, capfd)
    assert _states(seen) == {'unverified'} and seen['gate']['current'] is False, seen
    assert seen['gate']['stale_reason'] in ('later_uncommitted_projection_write', 'later_product_record'), seen


@pytest.mark.parametrize('mode', ['wrong', 'noop'])
def test_falsification_every_surface_including_cli_run(tmp_path, monkeypatch, capfd, mode):
    """Fix 13's falsification, extended to every read surface: recovery broken on purpose; two resumes through the CLI
    ``run``; each exits 3 typed, and status, report, CLI status and CLI report all say ``unresolved_recovery``."""
    output, run_id, path = _public(tmp_path)
    original = _sha(path)
    if mode == 'noop':
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            writer.record_uncommitted_write(run_id, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
    _break_recovery(monkeypatch, mode, output, run_id)
    for _ in range(2):
        capfd.readouterr()
        code = cli.main(['run', '--corpus', str(output), '--resume', run_id, '--retry-failed', '--json'])
        envelope = json.loads(capfd.readouterr().out)
        assert code == 3 and envelope['data'] is None and envelope['errors'][0]['code'] == 'unresolved_recovery'
        seen = _surfaces(output, run_id, capfd)
        assert _states(seen) == {'unresolved_recovery'} and seen['cli_status']['exit'] == 3, seen
        assert _sha(path) != original


# ========================================================= R4: nothing writes a product after the verdict =====

def test_nothing_writes_a_product_after_the_verdict_and_the_cost_is_in_the_verdict(tmp_path, monkeypatch):
    """Every write-capable call on a corpus path after the gate returned (inside the run close) is recorded; none may
    touch a managed product (or any public path). The removed cost log: the environment variable no longer writes
    anything. The verdict event carries the cost (products, bytes read, passes, elapsed)."""
    output = tmp_path / 'out'
    monkeypatch.setenv('GLOSSAPI_COMPLETION_GATE_LOG', str(output / 'export.jsonl'))
    after: list[tuple[str, str]] = []
    state = {'verdict': False}
    real_gate = LedgerWriter.completion_gate

    def gate(self, run_id):
        record = real_gate(self, run_id)
        state['verdict'] = True
        return record
    monkeypatch.setattr(LedgerWriter, 'completion_gate', gate)

    def watch(name, function, write_mode=None):
        def wrapped(*args, **kwargs):
            if state['verdict'] and args:
                target = str(args[1] if name in ('os.replace', 'os.rename', 'os.link', 'shutil.copyfile') and len(args) > 1 else args[0])
                writing = write_mode(*args, **kwargs) if write_mode else True
                if writing and target.startswith(str(output)):
                    after.append((name, target))
            return function(*args, **kwargs)
        return wrapped
    monkeypatch.setattr(builtins, 'open', watch('open', builtins.open,
                        lambda *a, **k: any(flag in str(a[1] if len(a) > 1 else k.get('mode', 'r')) for flag in 'wax+')))
    monkeypatch.setattr(os, 'open', watch('os.open', os.open,
                        lambda *a, **k: bool((a[1] if len(a) > 1 else k.get('flags', 0)) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))))
    for name in ('replace', 'rename', 'unlink', 'link', 'remove'):
        monkeypatch.setattr(os, name, watch('os.' + name, getattr(os, name)))
    monkeypatch.setattr(shutil, 'copyfile', watch('shutil.copyfile', shutil.copyfile))
    run = Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    assert state['verdict'] and run['state'] == 'completed'
    with open_readonly(output / 'corpus.sqlite3') as connection:
        expected = W.expected_products(connection)
        details = json.loads(connection.execute("SELECT details_json FROM events WHERE reason_code='completion_gate' ORDER BY event_id DESC LIMIT 1").fetchone()[0])
    products = {Path(uri[len('file://'):]) for uri in expected}
    ledger = {str(output / name) for name in ('corpus.sqlite3', 'corpus.sqlite3-wal', 'corpus.sqlite3-shm', 'corpus.sqlite3-journal')}
    touched = [entry for entry in after if Path(entry[1]) in products or ('/.glossapi/' not in entry[1] and entry[1] not in ledger)]
    assert touched == [], touched
    assert _sha(output / 'export.jsonl') == expected[(output / 'export.jsonl').as_uri()].sha256
    assert details['verdict'] == 'pass' and details['passes'] == 1 and details['products'] == len(expected)
    assert details['bytes_read'] > 0 and details['elapsed_ns'] > 0 and details['boundary'] == 'completion'


# ============================================================ FIX13-G1 allowlist, L1 wording =================

def test_wrapper_only_paths_are_left_out_only_for_an_allowlisted_diagnostic_role(tmp_path):
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='fix14')
        log, other = tmp_path / 'logs' / 'download_request.logs', tmp_path / 'markdown' / 'only-observed.md'
        log.parent.mkdir()
        other.parent.mkdir()
        log.write_bytes(b'log line\n')
        other.write_bytes(b'# observed\n')
        _wrapper_registration(writer, run, log, log.read_bytes(), 'logs/download_request.logs')
        observed = _wrapper_registration(writer, run, other, other.read_bytes(), 'raw_markdown')
        expected = W.expected_products(writer.connection)
        assert log.as_uri() not in expected
        assert repr(expected[other.as_uri()]) == 'Refuse(unlisted_observation_role)' and expected[other.as_uri()].registrations == [observed]


def test_refusal_text_names_its_boundary():
    gate = UnresolvedRecoveryError([{'path': '/c/export.jsonl', 'reason': 'completion_gate_mismatch'}])
    start = UnresolvedRecoveryError([{'path': '/c/export.jsonl', 'reason': 'hard_linked'}])
    publication = UnresolvedRecoveryError([{'path': '/c/export.jsonl', 'reason': 'hard_linked'}], boundary='publication')
    assert gate.boundary == 'completion' and 'nothing ran' not in str(gate) and "work was done" in str(gate)
    assert start.boundary == 'start' and str(start).endswith('nothing ran')
    assert 'nothing ran' not in str(publication) and gate.details['boundary'] == 'completion'
