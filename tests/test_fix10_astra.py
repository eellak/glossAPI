"""Fix round 10 (ASTRA-FIXES2c5 on 32e3183): recovery is fail-closed.

N1 [HIGH] The run-start recovery caught the hard-link refusal (``MetadataAliasError``) and went on, so a public resume
   returned ``completed`` with an export that no longer matched its committed hash. Fix: every recovery target that is
   recognised but not resolved (hard-linked, contradictory or stale lock records, a replaced path, a managed source that
   changed or is not intact, a copy changed right after its restore) is kept, listed and refused before anything runs,
   typed ``UnresolvedRecoveryError`` (``unresolved_recovery``, exit 3). Removing the cause lets the next resume repair
   the file and complete.
N3 [LOW] The run-start recovery took the lock of the newest matching record and ignored conflicting ones. Fix: the
   rollback's lock-agreement rule (``writer._agreed_lock``) over every record after the copy's registration that binds
   the bytes now at the path; a conflict is the same unresolved condition.

Each test fails on 32e3183 on its stated assertion (raw log: fix10/mac/newtest-on-32e3183.log). The table in
fix10/reply.md maps every skip of the recovery code to fail-closed or safe-to-skip.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES, ledger_matches_files
from test_k4c2_fixes import published_copy

RECORDED = b'{"filename": "uncommitted.pdf", "text": "bytes a failed attempt put in place"}\n'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_lock(path):
    resolved = ps.resolve_metadata_target(path)
    return resolved.with_suffix(resolved.suffix + '.lock')


def _refused(writer):
    """The run-start refusal (``refuse_ambiguous_copies``): the error it raises, or None."""
    try:
        writer.refuse_ambiguous_copies()
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return error
    return None


def _conditions(error):
    return [(Path(entry['path']).name, entry['reason']) for entry in getattr(error, 'conditions', [])]


def _own_bytes_copy(writer, root, *, locks=(None,)):
    """A published copy (named by the run close) now holding bytes recorded as the run's own uncommitted write, once
    per lock in ``locks``."""
    run, source, path, source_id, copy_id = published_copy(writer, root)
    path.write_bytes(RECORDED)
    for lock in locks:
        writer.record_uncommitted_write(run, path, _sha(path), len(RECORDED), wrapper_stage='export', existed=True, lock=lock)
    return run, source, path, copy_id


# ======================================================== N1: public resume, hard-linked export ==============

def test_public_resume_refuses_a_hard_linked_export_typed_and_retries(tmp_path):
    """Step: run-start restore refused (hard link) -> run. The resume is refused, typed, naming the path and reason; the
    export and its other name keep their bytes. After the extra link is removed, the next resume restores the committed
    export and completes. 32e3183 returned ``completed`` with the wrong export."""
    output = tmp_path / 'out'
    run = Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)['run_id']
    path = output / 'export.jsonl'
    committed = _sha(path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.record_uncommitted_write(run, path, hashlib.sha256(RECORDED).hexdigest(), len(RECORDED),
                                        wrapper_stage='export', existed=True)
    path.write_bytes(RECORDED)
    other = tmp_path / 'second-name'
    os.link(path, other)
    outcome, error = None, None
    try:
        outcome = Corpus(tmp_path / 'empty', output).run(resume=run, retry_failed=True)
    except Exception as caught:  # noqa: BLE001
        error = caught
    assert outcome is None, f'completed with a hard-linked unresolved export: {outcome and outcome["state"]}'
    assert getattr(error, 'code', None) == 'unresolved_recovery', repr(error)
    assert _conditions(error) == [('export.jsonl', 'hard_linked')]
    assert error.details['paths'] == [str(path)]
    assert path.read_bytes() == other.read_bytes() == RECORDED
    other.unlink()
    retried = Corpus(tmp_path / 'empty', output).run(resume=run, retry_failed=True)
    assert retried['state'] == 'completed' and _sha(path) == committed
    assert not ledger_matches_files(output)


def test_unresolved_recovery_is_a_typed_cli_error():
    """The refusal is typed for the CLI and the LLM guide: code ``unresolved_recovery``, exit 3."""
    from glossapi import llm_guide
    error = W.UnresolvedRecoveryError([{'path': '/c/export.jsonl', 'reason': 'hard_linked'}])
    assert (error.code, error.EXIT_CODE) == ('unresolved_recovery', 3)
    assert error.details == {'boundary': 'start', 'paths': ['/c/export.jsonl'],  # boundary: fix 14 (L1)
                             'conditions': [{'path': '/c/export.jsonl', 'reason': 'hard_linked'}]}
    assert llm_guide._cli_errors()['unresolved_recovery']['exit_code'] == 3


# ========================================================= N1: every run-start skip is fail-closed ===========

def test_startup_hard_linked_copy_is_an_unresolved_condition(tmp_path):
    """Step: restore refused by the metadata lock (hard link). The copy is kept, not made durable, and the start is
    refused. 32e3183 kept the copy but the start went on."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, copy_id = _own_bytes_copy(writer, tmp_path)
        os.link(path, tmp_path / 'second-name')
        assert writer.verify_excluded_copies() == []
        error = _refused(writer)
        assert copy_id not in writer._durable_files
    assert _conditions(error) == [('export.jsonl', 'hard_linked')], repr(error)
    assert path.read_bytes() == RECORDED


@pytest.mark.parametrize('order', ['canonical_last', 'stale_last'])
def test_startup_conflicting_lock_records_are_an_unresolved_condition(tmp_path, order):
    """Step (N3): lock agreement -> restore. Two records of the copy's current bytes bind different locks: the copy is
    kept and the start refused, whatever the record order. 32e3183 restored it when the canonical record was newest."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        locks = [tmp_path / 'stale.lock', _canonical_lock(tmp_path / 'export.jsonl')]
        if order == 'stale_last':
            locks.reverse()
        run, source, path, copy_id = _own_bytes_copy(writer, tmp_path, locks=locks)
        restored = writer.verify_excluded_copies()
        error = _refused(writer)
    assert restored == [] and path.read_bytes() == RECORDED
    assert _conditions(error) == [('export.jsonl', 'conflicting_locks')], repr(error)


def test_startup_conflicting_records_cleared_by_moving_the_copy(tmp_path):
    """The remedy for contradictory records: moving the copy away. The records no longer apply to the bytes at the
    path, so the copy is republished from its managed source and the start goes on."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, copy_id = _own_bytes_copy(writer, tmp_path,
                                                     locks=[tmp_path / 'stale.lock', _canonical_lock(tmp_path / 'export.jsonl')])
        path.rename(tmp_path / 'kept-by-the-user.jsonl')
        assert writer.verify_excluded_copies() == [copy_id]
        assert _refused(writer) is None
    assert path.read_bytes() == source.read_bytes()


def test_startup_stale_recorded_lock_is_an_unresolved_condition(tmp_path):
    """Step: recorded lock is not the lock the path's writers take now -> refused by ``_recover_locked``. 32e3183
    returned "nothing to do" and the start went on."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, copy_id = _own_bytes_copy(writer, tmp_path, locks=[tmp_path / 'elsewhere' / 'export.jsonl.lock'])
        assert writer.verify_excluded_copies() == []
        error = _refused(writer)
    assert _conditions(error) == [('export.jsonl', 'lock_mismatch')], repr(error)
    assert path.read_bytes() == RECORDED


def test_startup_source_changed_during_restore_is_an_unresolved_condition(tmp_path, monkeypatch):
    """Step: verified source -> installed bytes. The source changes after its check (the copied bytes are not the
    registered ones): nothing is replaced and the start is refused. 32e3183 left it "for verification" and went on."""
    import shutil
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, copy_id = _own_bytes_copy(writer, tmp_path)
        real = shutil.copyfile

        def copy(src, dst, *args, **kwargs):
            result = real(src, dst, *args, **kwargs)
            with open(dst, 'ab') as handle:
                handle.write(b'X')
            return result
        monkeypatch.setattr(shutil, 'copyfile', copy)
        assert writer.verify_excluded_copies() == []
        monkeypatch.undo()
        error = _refused(writer)
    assert _conditions(error) == [('export.jsonl', 'source_changed')], repr(error)
    assert path.read_bytes() == RECORDED


def test_startup_copy_without_an_intact_source_is_an_unresolved_condition(tmp_path):
    """Step: torn copy, managed source no longer intact -> nothing to republish. The start is refused. 32e3183 left it
    "for verification" and went on."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path, torn=True)
    source.write_bytes(b'{"a": 2}\n' * 1000)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == []
        error = _refused(writer)
    assert _conditions(error) == [('export.jsonl', 'no_intact_source')], repr(error)


# ============================================================ N1: rollback skips are listed, then refused =====

def _rollback_ledger(tmp_path):
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    with writer.transaction():
        run = writer.run(SETTINGS6, {}, {}, code_revision='fix10')
    return writer, run


def test_rollback_lists_a_path_replaced_by_a_symlink(tmp_path):
    """Step: the recorded physical file was replaced by a symlink. Not followed, and listed as unresolved (the repeated
    barrier refuses on it). 32e3183 skipped it silently."""
    path, elsewhere = tmp_path / 'product.jsonl', tmp_path / 'elsewhere.jsonl'
    path.write_bytes(RECORDED)
    writer, run = _rollback_ledger(tmp_path)
    try:
        writer.record_uncommitted_write(run, path, _sha(path), len(RECORDED), wrapper_stage='export', existed=False)
        path.rename(elsewhere)
        path.symlink_to(elsewhere)
        assert writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run) == []
        listed = [(Path(entry['path']).name, entry['reason']) for entry in getattr(writer, 'rollback_unresolved', [])]
    finally:
        writer.close()
    assert listed == [('product.jsonl', 'path_replaced')]
    assert elsewhere.read_bytes() == RECORDED and path.is_symlink()


@pytest.mark.parametrize('case', ['conflicting', 'stale'])
def test_rollback_lists_contradictory_or_stale_lock_records(tmp_path, case):
    """Steps: lock agreement / recorded lock -> act. The file and its temporary are kept and the path is listed as
    unresolved. 32e3183 kept them but listed nothing, so the repeated publication went on."""
    path, temporary = tmp_path / 'physical', tmp_path / '.physical.tmp'
    path.write_bytes(RECORDED)
    temporary.write_bytes(RECORDED)
    locks = [tmp_path / 'stale.lock'] + ([_canonical_lock(path)] if case == 'conflicting' else [])
    writer, run = _rollback_ledger(tmp_path)
    try:
        for lock in locks:
            writer.record_uncommitted_write(run, path, _sha(path), len(RECORDED), wrapper_stage='clean', existed=False,
                                            temporary=temporary, lock=lock)
        assert writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run) == []
        listed = [(Path(entry['path']).name, entry['reason']) for entry in getattr(writer, 'rollback_unresolved', [])]
    finally:
        writer.close()
    assert listed == [('physical', 'conflicting_locks' if case == 'conflicting' else 'lock_mismatch')]
    assert path.read_bytes() == temporary.read_bytes() == RECORDED


def test_repeated_barrier_refuses_an_unresolved_rollback(tmp_path, monkeypatch):
    """Step: rollback listed an unresolved target -> the repeated publication. ``publish_barrier`` raises the typed
    refusal before the publication runs. 32e3183 went on (the rollback listed nothing and nothing checked)."""
    from glossapi.stages import run as R
    calls = []
    real = LedgerWriter.roll_back_uncommitted_writes

    def rolled_back(self, run_ids, output_dir, *, observation_run):
        changed = real(self, run_ids, output_dir, observation_run=observation_run)
        if run_ids:
            calls.append(run_ids)
            self.rollback_unresolved = [{'path': str(Path(output_dir) / 'export.jsonl'), 'reason': 'conflicting_locks'}]
        return changed
    output = tmp_path / 'out'
    from test_fix8_astra import _failed_clean
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    # Start-up must not decide this test: the copies the failed run's close named are taken as intact.
    monkeypatch.setattr(LedgerWriter, 'verify_excluded_copies', lambda self: setattr(self, 'unresolved_recoveries', []) or [])
    monkeypatch.setattr(LedgerWriter, 'roll_back_uncommitted_writes', rolled_back)
    error = None
    try:
        Corpus(tmp_path / 'empty', output).run(resume=run, retry_failed=True)
    except Exception as caught:  # noqa: BLE001
        error = caught
    monkeypatch.undo()
    assert calls, 'the repeated barrier must have rolled back'
    assert getattr(error, 'code', None) == 'unresolved_recovery', repr(error)
