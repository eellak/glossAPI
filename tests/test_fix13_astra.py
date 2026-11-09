"""Fix round 13 (ASTRA-FIXES2c8 on 7517a5b; decision B): one authority, an independent completion gate, H1-H4, M1.

``writer.expected_products(connection)`` is THE authority on what every managed public product must hold: pure over the
ledger (no file read, no hash, no presence observation), from committed registrations, each producer attempt's terminal
``ok``, persistent invalidation rows a rollback writes, and committed policy deletions. ``authoritative_target`` takes
its eligibility from it. ``LedgerWriter.completion_gate`` runs once per run completion, after the final flush and under
the workspace lock, hashes every managed public product against it, persists its verdict, and refuses
(``UnresolvedRecoveryError``) on any difference; status and report never say ``completed`` without a passing verdict.

The falsification tests break recovery on purpose (wrong bytes with the registrations marked missing; a no-op) and
require the gate to refuse every time, on every completion surface, across close/reopen and two resumes.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi import Corpus, cli
from glossapi.ledger import writer as W
from glossapi.ledger.reader import report_view, select_run, status_view
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES
from test_fix12_astra import _register_generation, _two_named_copies, _next_close_excluded, _FakeProc
from test_k4c2_fixes import published_copy
from test_k4g_fix4 import products

CONTENT = b'{"filename":"uncommitted.pdf","text":"bytes a failed attempt put in place"}\n'
WRONG = b'{"filename":"wrong.pdf","text":"bytes a broken recovery left"}\n'


def _sha(data_or_path):
    data = data_or_path if isinstance(data_or_path, bytes) else Path(data_or_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _refusal(function):
    try:
        return function(), None
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return None, error


def _expected(path):
    with open_readonly(path / 'corpus.sqlite3') as connection:
        return W.expected_products(connection)


def _public(tmp_path):
    output = tmp_path / 'out'
    run = Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    assert run['state'] == 'completed' and run['completion_gate']['verdict'] == 'pass'
    return output, run['run_id'], output / 'export.jsonl'


def _surfaces(output, run_id):
    """status and report of the run, each read through a fresh read-only connection (close/reopen)."""
    with open_readonly(output / 'corpus.sqlite3') as connection:
        run = select_run(connection, run_id)
        status, report = status_view(connection, run=run), report_view(connection, run=run)
    return status['state'], report['state'], (status['completion_gate'] or {}).get('verdict')


# ================================================================ expected_products: every return =============

def test_expected_products_returns(tmp_path):
    """Sha of the newest ok registration; a failed producer passed over; an unfinished newest producer refuses; a
    persistent invalidation passes a registration over; every registration invalidated -> Absent; a committed policy
    deletion -> Absent; a pending republication newer than every eligible registration -> Refuse."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        uri = path.as_uri()
        assert repr(W.expected_products(writer.connection)[uri]) == f'Sha({_sha(source)[:12]})'
        _register_generation(writer, run, path, b'{"failed":1}\n', tmp_path / '.glossapi/artifacts/failed/x.jsonl', finish='failed')
        assert W.expected_products(writer.connection)[uri].registration == copy_id  # failed producer passed over
        _, unfinished = _register_generation(writer, run, path, b'{"open":1}\n', tmp_path / '.glossapi/artifacts/open/x.jsonl', finish=None)
        assert repr(W.expected_products(writer.connection)[uri]) == 'Refuse(producer_unfinished)'
        with writer.transaction():
            writer.event(run, event_kind='invalidation', reason_code='registration_invalidated', file_id=unfinished, details={})
        assert W.expected_products(writer.connection)[uri].registration == copy_id  # invalidated: passed over for good
        with writer.transaction():
            writer.event(run, event_kind='invalidation', reason_code='pending_republication', file_id=copy_id, details={})
        assert repr(W.expected_products(writer.connection)[uri]) == 'Refuse(pending_republication)'
        _, newer = _register_generation(writer, run, path, b'{"n":1}\n', tmp_path / '.glossapi/artifacts/newer/x.jsonl')
        assert W.expected_products(writer.connection)[uri].registration == newer  # the republication discharges it
        with writer.transaction():
            writer.event(run, event_kind='retention', reason_code='delete_completed', file_id=newer, details={})
        assert repr(W.expected_products(writer.connection)[uri]) == 'Absent(policy_deletion)'
        lonely = tmp_path / 'lonely.jsonl'
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix13')
        _, only = _register_generation(writer, failed, lonely, CONTENT, None, finish='failed')
        assert repr(W.expected_products(writer.connection)[lonely.as_uri()]) == 'Refuse(no_committed_registration)'
        with writer.transaction():
            writer.event(failed, event_kind='invalidation', reason_code='registration_invalidated', file_id=only, details={})
        assert repr(W.expected_products(writer.connection)[lonely.as_uri()]) == 'Absent(every_registration_invalidated)'


def test_expected_products_reads_no_file_and_no_presence(tmp_path, monkeypatch):
    """Purity: with every filesystem read failing, the map is the same; presence observations (missing, corrupt,
    present) of the newest registration change nothing."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        before = repr(W.expected_products(writer.connection))
        for presence in ('missing', 'corrupt', 'present'):
            with writer.transaction():
                writer.event(run, event_kind='artifact', reason_code=presence, file_id=copy_id, details={'observation': 'fix13'})
        import builtins
        reads = []

        def no_files(*args, **kwargs):
            reads.append(args[:1])
            raise OSError(errno.EIO, 'expected_products must not read a file')
        with monkeypatch.context() as patch:
            patch.setattr(builtins, 'open', no_files)
            patch.setattr(os, 'stat', no_files)
            patch.setattr(os, 'lstat', no_files)
            after = repr(W.expected_products(writer.connection))
    assert after == before and reads == []


# ======================================================================= the gate: verdicts and surfaces ======

def test_gate_passes_a_normal_run_and_persists_its_verdict(tmp_path):
    output, run_id, path = _public(tmp_path)
    assert _surfaces(output, run_id) == ('completed', 'completed', 'pass')


@pytest.mark.parametrize('fault', ['mismatch', 'missing', 'extra'])
def test_gate_refuses_every_difference(tmp_path, fault):
    """Direct gate calls: a product with other bytes, a missing product, a product present where its newest
    registration was deleted by policy. Each is refused, typed, the verdict persisted."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        if fault == 'mismatch':
            path.write_bytes(WRONG)
        elif fault == 'missing':
            path.unlink()
        else:
            with writer.transaction():
                writer.event(run, event_kind='retention', reason_code='delete_completed', file_id=copy_id, details={})
        verdict = writer.completion_gate(run)  # fix 14: a direct call returns the persisted verdict; the run close raises
        recorded = writer.completion_gate_record
    assert [entry['reason'] for entry in verdict['all_failures']] == [f'completion_gate_{fault}']
    assert recorded['verdict'] == 'refused' and recorded['failure_count'] == 1


def test_deleted_newer_wrapper_registration_makes_a_published_path_absent(tmp_path):
    """Fix 13 added a universal newer-deletion rule for the node's OCR ``cleaning_report.parquet``; ASTRA-FIXES2c9 R1
    rejected it, and fix 14 binds a deletion to the committed obligation. Kept under its name, with the fix-14
    expectation: a later wrapper registration at a published path, deleted by retention, does not discharge the
    committed generation; the gate refuses its absence. (The OCR case is re-solved at its source:
    ``test_fix14_astra.test_a_report_declared_temporary_is_never_published_as_a_metadata_projection``.)"""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        uri = path.as_uri()
        data = b'{"temporary":"report"}\n'
        with writer.transaction():
            manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(b'fix13-wrapper')]})
            task = writer.task(run, 'clean', 'clean.clean', {}, scope_key='corpus:' + manifest)
            attempt = writer.attempt(task, worker_id='orchestrator:4242', device_id='cpu', fingerprint={}, inputs={},
                                     settings={}, tools={})
            observed = writer.artifact(run, 'clean', 'temporary_cleaning_report', uri, producer_attempt_id=attempt,
                                       sha256=_sha(data), bytes=len(data))
            writer.finish(run, task, attempt, status='ok', reason_code='clean_completed')
        assert W.expected_products(writer.connection)[uri].registration == copy_id  # observation: not authority
        path.write_bytes(data)
        with writer.transaction():
            writer.event(run, event_kind='retention', reason_code='delete_completed', file_id=observed, details={})
        path.unlink()
        expected = W.expected_products(writer.connection)[uri]
        assert repr(expected) == f'Sha({_sha(source)[:12]})' and expected.registration == copy_id
        verdict = writer.completion_gate(run)
        assert verdict['verdict'] == 'refused' and [f['reason'] for f in verdict['all_failures']] == ['completion_gate_missing']


# ===================================================================================== falsification =========

def _break_recovery(monkeypatch, mode, output, run_id):
    """Recovery made deliberately wrong. ``wrong``: every named copy gets other bytes and its registrations are marked
    missing (hiding the evidence); ``noop``: start-up recovery and rollback do nothing at all."""
    real_verify = LedgerWriter.verify_excluded_copies

    def wrong(self):
        self.unresolved_recoveries, self.ambiguous_copies = [], []
        export = output / 'export.jsonl'
        export.write_bytes(WRONG)
        with self.transaction():
            for row in [r for r in self.file_catalog().values() if r['uri'] == export.as_uri() and r['presence'] == 'present']:
                self.event(run_id, event_kind='artifact', reason_code='missing', file_id=row['file_id'],
                           details={'observation': 'falsification'})
        self._file_catalog = None
        return []

    def noop(self):
        self.unresolved_recoveries, self.ambiguous_copies = [], []
        return []
    monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies', wrong if mode == 'wrong' else noop)
    monkeypatch.setattr(LedgerWriter, 'roll_back_uncommitted_writes', lambda self, *a, **k: setattr(self, 'rollback_unresolved', []) or [])
    return real_verify


@pytest.mark.parametrize('mode', ['wrong', 'noop'])
def test_falsification_gate_refuses_a_deliberately_wrong_recovery(tmp_path, monkeypatch, capfd, mode):
    """A gate that passes here is a failure of the round. The inventory is frozen from the completed run before any
    fault. With recovery broken, two resumes (each after close/reopen) never report ``completed``: the run raises
    ``unresolved_recovery``, status and report say ``unresolved_recovery`` (verdict ``refused``), the CLI status exits
    3. With recovery restored, the next resume repairs and passes."""
    output, run_id, path = _public(tmp_path)
    inventory = products(output)
    if mode == 'noop':  # bytes a failed attempt recorded and put in place: a working start-up recovery restores them
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            writer.record_uncommitted_write(run_id, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
    _break_recovery(monkeypatch, mode, output, run_id)
    for _ in range(2):
        result, error = _refusal(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
        assert result is None and getattr(error, 'code', None) == 'unresolved_recovery', (result, error)
        assert products(output) != inventory
        assert _surfaces(output, run_id) == ('unresolved_recovery', 'unresolved_recovery', 'refused')
        capfd.readouterr()
        code = cli.main(['status', '--corpus', str(output), '--run', run_id, '--json'])
        envelope = json.loads(capfd.readouterr().out)
        assert code == 3 and envelope['data']['state'] == 'unresolved_recovery' and envelope['errors'][0]['code'] == 'unresolved_recovery'
    monkeypatch.undo()
    result, error = _refusal(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    if error is None:
        assert result['state'] == 'completed' and products(output) == inventory
        assert _surfaces(output, run_id)[0] == 'completed'
    else:  # bytes nobody recorded are a user's edit: kept and refused, never completed
        assert getattr(error, 'code', None) in ('post_close_edit', 'unresolved_recovery'), repr(error)


# ============================================================ H1: the source-release exception has an obligation ===

def test_released_source_leaves_a_pending_republication_the_gate_refuses(tmp_path):
    """Rollback cannot restore a target whose only source was released by policy: it leaves the failed bytes AND a
    persistent ``pending_republication`` row. Presence changes never discharge it; the gate refuses until a newer
    committed generation of the path exists. 7517a5b returned success and later runs completed with the failed bytes."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            writer.event(run, event_kind='artifact', reason_code='deleted', file_id=source_id, details={'policy_deletion': True})
            failed = writer.run({}, {}, {}, code_revision='fix13-failed')
        source.unlink()
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        assert writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run) == []
        uri = path.as_uri()
        assert repr(W.expected_products(writer.connection)[uri]) == 'Refuse(pending_republication)'
        with writer.transaction():  # an observation cannot discharge the obligation
            writer.event(run, event_kind='artifact', reason_code='present', file_id=copy_id, details={'observation': 'fix13'})
        assert repr(W.expected_products(writer.connection)[uri]) == 'Refuse(pending_republication)'
        verdict = writer.completion_gate(run)  # fix 14: a direct call returns the persisted verdict; the run close raises
        assert verdict['verdict'] == 'refused' and [entry['reason'] for entry in verdict['all_failures']] == ['completion_gate_refused']
        _register_generation(writer, run, path, b'{"republished":1}\n', tmp_path / '.glossapi/artifacts/re/x.jsonl')
        assert writer.completion_gate(run)['verdict'] == 'pass'


# ======================================================== H2: failed producers and invalidation are persistent ====

def test_failed_producer_is_never_a_start_up_target(tmp_path):
    """A newer registration whose producer attempt ended ``failed`` is passed over at start-up too: the copy is
    restored to the committed generation. 7517a5b restored the failed bytes and completed."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix13-failed')
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        _, failed_id = _register_generation(writer, failed, path, CONTENT, tmp_path / '.glossapi/artifacts/f/x.jsonl', finish='failed')
        target = writer.authoritative_target(path)
        restored = writer.verify_excluded_copies()
    assert target.registration['file_id'] == copy_id and restored == [copy_id] and path.read_bytes() == source.read_bytes()


def test_rollback_invalidation_survives_later_observations(tmp_path):
    """The rollback writes ``registration_invalidated`` for the failed attempts' registrations; a later ``missing`` (or
    any) observation does not bring them back as targets."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix13-failed')
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        _, failed_id = _register_generation(writer, failed, path, CONTENT, tmp_path / '.glossapi/artifacts/f/x.jsonl', finish='ok')
        path.unlink()
        writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run)
        for presence in ('missing', 'present', 'missing'):
            with writer.transaction():
                writer.event(run, event_kind='artifact', reason_code=presence, file_id=failed_id, details={'observation': 'fix13'})
        target = writer.authoritative_target(path)
        invalidations = [tuple(row) for row in writer.connection.execute(f"SELECT reason_code FROM {writer.connection.event_table} WHERE event_kind='invalidation' "
                                                  'AND file_id=?', (failed_id,)).fetchall()]
    assert target.registration['file_id'] == copy_id and invalidations == [('registration_invalidated',)]


# ===================================================== H3: rollback uses the same durability phase ==============

def test_rollback_after_start_up_durability_flushes_the_directory_and_voids_durability(tmp_path, monkeypatch):
    """Start-up made the copy durable; a failed attempt then put its bytes in place; the rollback restores. The restore
    is followed by a directory flush, the earlier durable flag is void, and the next close names the copy again.
    7517a5b kept the copy durable with no directory flush after the replacement."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        writer.verify_excluded_copies()
        assert copy_id in writer._durable_files
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix13-failed')
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        trace, real_restore, real_fsync = [], W._restore_copy, os.fsync
        identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)

        def restore(source_path, destination, expected):
            real_restore(source_path, destination, expected)
            trace.append('replace')

        def fsync(descriptor):
            info = os.fstat(descriptor)
            if (info.st_dev, info.st_ino) == identity:
                trace.append('fsync_directory')
            return real_fsync(descriptor)
        monkeypatch.setattr(W, '_restore_copy', restore)
        monkeypatch.setattr(W, 'os', SimpleNamespace(**{key: (fsync if key == 'fsync' else getattr(os, key)) for key in dir(os) if not key.startswith('__')}))
        changed = writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run)
        durable = copy_id in writer._durable_files
        monkeypatch.undo()
        excluded = _next_close_excluded(writer, run)
    assert changed and trace[0] == 'replace' and 'fsync_directory' in trace[1:], trace
    assert not durable and copy_id in excluded and path.read_bytes() == source.read_bytes()


def test_rollback_flush_mismatch_is_listed_not_clean(tmp_path, monkeypatch):
    """The bytes change during the rollback's flush: the post-flush hash catches it (listed ``changed_after_flush``),
    the path's durability stays void. 7517a5b certified the restore with its pre-flush hash."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix13-failed')
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        real = W._fsync_path

        def flush(target):
            if Path(target) == path:
                path.write_bytes(WRONG)
            return real(target)
        monkeypatch.setattr(W, '_fsync_path', flush)
        writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run)
        listed = [entry['reason'] for entry in writer.rollback_unresolved]
    assert listed == ['changed_after_flush'] and copy_id not in getattr(writer, '_durable_files', set())


# ========================================================== H4: incomplete discovery is never fenced ============

def test_detached_descendant_with_incomplete_enumeration_is_untrusted(monkeypatch):
    """A worker (100) with a detached child in its own process group (101); every ``/proc`` enumeration is cut short
    before the child is listed. The worker is signalled and joined, then the workspace is reported untrusted.
    7517a5b returned success with the child still running."""
    from glossapi.stages import execution as E
    table = {1: (0, 1, 'S'), 100: (1, 100, 'S'), 101: (100, 101, 'S')}
    proc, sent = _FakeProc(table, fail=99), []
    monkeypatch.setattr(E, 'Path', proc.path)
    monkeypatch.setattr(E, 'os', SimpleNamespace(kill=lambda *a: sent.append(('kill',) + a), killpg=lambda *a: sent.append(('killpg',) + a)))
    process = SimpleNamespace(pid=100, join=lambda *a: (sent.append(('join',) + a), table.update({100: (1, 100, 'Z')})))
    with pytest.raises(RuntimeError, match='untrusted'):
        E._stop_process(process)
    assert ('kill', 100, signal.SIGTERM) in sent and ('kill', 100, signal.SIGKILL) in sent and ('join',) in sent
    assert not any(len(action) > 1 and action[1] == 101 for action in sent), 'the undiscovered child cannot have been signalled'


def test_pool_close_discards_every_worker_even_if_one_cannot_be_fenced(monkeypatch):
    from glossapi.stages import workers
    stopped = []

    def stop(process):
        stopped.append(process.name)
        if process.name == 'first':
            raise RuntimeError('Worker descendants could not be fenced; workspace remains untrusted')
    monkeypatch.setattr(workers, '_stop_process', stop)
    slots = [{'request': object(), 'process': SimpleNamespace(name=name, exitcode=0, is_alive=lambda: False, close=lambda: None),
              'pipe': SimpleNamespace(close=lambda: None)} for name in ('first', 'second')]
    pool = SimpleNamespace(slots=list(slots), closed=False, _discard=lambda slot: workers.WarmWorkerPool._discard(pool, slot))
    with pytest.raises(RuntimeError, match='untrusted'):
        workers.WarmWorkerPool.close(pool)
    assert stopped == ['first', 'second'] and pool.slots == []
