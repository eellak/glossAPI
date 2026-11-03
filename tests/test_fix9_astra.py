"""Fix round 9 (ASTRA-FIXES2c4 on e8fcb72): recovery locks and temporary cleanup.

F1 [HIGH] Recovery chose the metadata lock from the physical file's suffix (only a lowercase ``.parquet`` path was
   locked), so a metadata file behind a symlink to ``physical.bin`` or ``physical.PARQUET`` (or an extensionless one)
   was rolled back unlocked and a concurrent locked edit was lost. Fix: the writer that makes an uncommitted in-place
   write records the canonical metadata lock it holds (``parquet_schema.held_metadata_lock``) in the intent record;
   recovery, rollback and temporary cleanup take exactly that lock (``writer._recover_locked(lock=...)``), and the
   canonical lock of the path for a record without one. Nothing is inferred from a name.
F2 [MEDIUM] Temporary cleanup hashed the source, then reopened it and compared a prefix: a source changed in between
   was compared as another generation, and a modified temporary was deleted. Fix: ``_verified_temporary`` reads the
   source once, through one descriptor, and both compares and hashes those bytes; the temporary's name is removed
   only if it still names the verified file.

Each test fails on e8fcb72 on its stated assertion (raw log: fix9/mac/newtest-on-e8fcb72.log), except the controls
named as such. The table in fix9/reply.md maps step -> protection -> test.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import encoding
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES, METADATA, _sha

SUFFIXES = ['.bin', '.PARQUET', '']  # a symlinked physical file of any name, and an extensionless metadata file


def _canonical_lock(path):
    """The lock file ``parquet_schema._parquet_lock`` takes for ``path`` (unchanged since e8fcb72): beside the resolved
    physical file, its full name plus ``.lock``. Spelled out here so the tests run unchanged on e8fcb72."""
    resolved = ps.resolve_metadata_target(path)
    return resolved.with_suffix(resolved.suffix + '.lock')


def _ledger(tmp_path):
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    with writer.transaction():
        run = writer.run(SETTINGS6, {}, {}, code_revision='fix9')
    return writer, run


def _aliased_metadata(tmp_path, suffix):
    """``download_results.parquet`` as a symlink to a physical metadata file named ``physical<suffix>``."""
    physical = tmp_path / ('physical' + suffix)
    alias = tmp_path / 'download_results.parquet'
    ps.replace_metadata_parquet(physical, pd.DataFrame({'filename': ['a.pdf']}))
    alias.symlink_to(physical)
    return physical, alias


def _edit_during_second_hash(monkeypatch, physical, alias, state):
    """Right after the rollback's second hash of ``physical`` (the one inside its critical section), a real locked
    ``update_metadata_parquet`` through the alias starts on another thread; it can only finish once that section ends."""
    real, done = encoding.file_hash, threading.Event()
    state['hashes'] = 0

    def user():
        try:
            def edit(frame):
                frame['user_note'] = 'locked user commit'
                return frame
            ps.update_metadata_parquet(alias, edit)
        except BaseException as error:  # noqa: BLE001 - reported in the assertion
            state['writer_error'] = repr(error)
        finally:
            done.set()

    def hashed(target, *args, **kwargs):
        digest = real(target, *args, **kwargs)
        if Path(target) == physical:
            state['hashes'] += 1
            if state['hashes'] == 2:
                state['thread'] = threading.Thread(target=user)
                state['thread'].start()
                state['edit_finished_before_act'] = done.wait(0.5)
        return digest
    monkeypatch.setattr(encoding, 'file_hash', hashed)
    return done


# ====================================================== F1: the writers record the lock they hold ==============

def test_held_metadata_lock_names_the_lock_this_thread_holds(tmp_path):
    """``held_metadata_lock`` is the canonical lock (``metadata_lock_path``) only inside ``metadata_target`` of the same
    file and on the holding thread; nothing is derived from the suffix (``physical.bin`` behind a symlink)."""
    physical, alias = _aliased_metadata(tmp_path, '.bin')
    canonical = ps.metadata_lock_path(alias)
    assert canonical == ps.metadata_lock_path(physical) == ps.resolve_metadata_target(physical).with_name('physical.bin.lock')
    assert ps.held_metadata_lock(alias) is None
    seen = {}
    with ps.metadata_target(alias) as target:
        seen['inside'] = ps.held_metadata_lock(physical)
        thread = threading.Thread(target=lambda: seen.setdefault('other_thread', ps.held_metadata_lock(alias)))
        thread.start()
        thread.join(timeout=10)
        assert target == ps.resolve_metadata_target(physical)
    assert seen == {'inside': canonical, 'other_thread': None}
    assert ps.held_metadata_lock(alias) is None


def test_intent_records_bind_the_lock_their_writer_held(tmp_path):
    """A real run: every intent record of a write made through the metadata choke point (download_results, and the
    cleaning report the clean writes through it) binds the canonical lock of its path; the export publication, made
    without a metadata lock (FIX8-G2), binds none."""
    output = tmp_path / 'out'
    Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    with open_readonly(output / 'corpus.sqlite3') as connection:
        records = [json.loads(row[0]) for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE reason_code='uncommitted_projection_write'")]
    metadata = str(ps.resolve_metadata_target(output / METADATA))
    locked = [r for r in records if 'lock' in r]
    assert metadata in {r['path'] for r in locked}, records
    assert all(r['lock'] == str(_canonical_lock(r['path'])) for r in locked), locked
    assert {Path(r['path']).name for r in records if 'lock' not in r} == {'export.jsonl'}, records


# ========================================== F1: rollback takes the recorded lock, whatever the physical name ===

@pytest.mark.parametrize('suffix', SUFFIXES)
def test_rollback_removal_takes_the_recorded_lock(tmp_path, monkeypatch, suffix):
    """Step: lock selection -> rollback REMOVE. The record binds the canonical lock (as the choke point records it). A
    locked edit through the alias starting inside the critical section must wait, then commit on top of the removal.
    e8fcb72 took no lock for these physical names: the edit committed first and was then removed."""
    physical, alias = _aliased_metadata(tmp_path, suffix)
    state = {}
    writer, run = _ledger(tmp_path)
    try:
        writer.record_uncommitted_writes(run, [(alias, _sha(physical), physical.stat().st_size, False,
                                                {'lock': _canonical_lock(alias)})], wrapper_stage='clean')
        done = _edit_during_second_hash(monkeypatch, physical, alias, state)
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        if 'thread' in state:
            state['thread'].join(timeout=10)
        monkeypatch.undo()
    finally:
        writer.close()
    assert done.is_set() and not state.get('writer_error'), state
    assert state['edit_finished_before_act'] is False, 'the edit must wait for the recovery lock'
    assert physical.is_file() and 'user_note' in pd.read_parquet(physical), 'the committed edit was removed'
    assert alias.is_symlink()


def test_rollback_refuses_a_path_whose_lock_is_not_the_recorded_one(tmp_path):
    """Step: lock selection. A record whose lock is not the lock the path's writers take now (the path no longer
    resolves to the file the writer locked) is not acted on. e8fcb72 ignored the lock and removed the file."""
    physical, alias = _aliased_metadata(tmp_path, '.bin')
    writer, run = _ledger(tmp_path)
    try:
        writer.record_uncommitted_writes(run, [(alias, _sha(physical), physical.stat().st_size, False,
                                                {'lock': tmp_path / 'elsewhere' / 'physical.bin.lock'})], wrapper_stage='clean')
        changed = writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
    finally:
        writer.close()
    assert changed == [] and physical.is_file()


def test_rollback_refuses_a_path_whose_records_bind_two_locks(tmp_path):
    """Step: lock selection. Two records of one physical file binding different locks cannot both be right: the path
    is not acted on. e8fcb72 removed it."""
    physical, alias = _aliased_metadata(tmp_path, '')
    writer, run = _ledger(tmp_path)
    try:
        digest, size = _sha(physical), physical.stat().st_size
        writer.record_uncommitted_writes(run, [(alias, digest, size, False, {'lock': _canonical_lock(alias)}),
                                               (alias, digest, size, False, {'lock': tmp_path / 'other.lock'})],
                                         wrapper_stage='clean')
        changed = writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
    finally:
        writer.close()
    assert changed == [] and physical.is_file()


@pytest.mark.parametrize('suffix', SUFFIXES)
def test_recovery_without_a_recorded_lock_takes_the_canonical_lock(tmp_path, suffix):
    """Step: lock selection for a record or a copy check without a lock identity (older records, products published
    without a metadata lock, the run-start copy check): ``_recover_locked`` takes the canonical lock of the path all
    the same; ``decide`` runs only after a writer holding ``metadata_target`` of the alias releases it.
    e8fcb72 decided at once, while the writer still held the lock."""
    physical, alias = _aliased_metadata(tmp_path, suffix)
    holding, release, order = threading.Event(), threading.Event(), []

    def live_writer():
        with ps.metadata_target(alias):
            holding.set()
            release.wait(timeout=1.0)
            order.append('writer released')
    thread = threading.Thread(target=live_writer)
    thread.start()
    try:
        assert holding.wait(timeout=10)
        W._recover_locked(physical, lambda target: order.append('decided') or None)
    finally:
        release.set()
        thread.join(timeout=10)
    assert order == ['writer released', 'decided'], order


# ============================== F1: temporary cleanup under the recorded lock (target torn, missing, any name) ==

@pytest.mark.parametrize('target_state', ['missing', 'torn'])
@pytest.mark.parametrize('suffix', ['.bin', ''])
def test_temporary_cleanup_waits_for_the_live_writer_of_any_metadata_name(tmp_path, suffix, target_state):
    """Step: temporary verification -> removal, for a metadata file whose physical name has another suffix or none, and
    which is missing (the writer is creating it) or torn (not a readable parquet) at recovery. A writer holding the
    metadata lock owns the recorded temporary and replaces with it before it releases the lock. With the recorded lock
    the cleanup waits and the writer's replace succeeds. e8fcb72 took no lock: it removed the live temporary under the
    writer, whose replace then failed."""
    physical, alias = _aliased_metadata(tmp_path, suffix)
    if target_state == 'missing':
        physical.unlink()
    else:
        physical.write_bytes(physical.read_bytes()[:20])
    content = b'complete new metadata bytes of the live writer'
    temporary = tmp_path / ('.physical' + suffix + '.live.tmp')
    temporary.write_bytes(content)
    writer, run = _ledger(tmp_path)
    holding, finished, outcome = threading.Event(), threading.Event(), {}

    def live_writer():
        with ps.metadata_target(alias) as target:
            holding.set()
            finished.wait(timeout=1.5)  # the cleanup finishes meanwhile only if it does not take the lock
            try:
                os.replace(temporary, target)
                outcome['replace'] = 'ok'
            except FileNotFoundError as error:
                outcome['replace'] = repr(error)
    try:
        writer.record_uncommitted_writes(run, [(alias, hashlib.sha256(content).hexdigest(), len(content), target_state == 'torn',
                                                {'temporary': temporary, 'lock': _canonical_lock(alias)})],
                                         wrapper_stage='clean')
        thread = threading.Thread(target=live_writer)
        thread.start()
        assert holding.wait(timeout=10)
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        finished.set()
        thread.join(timeout=20)
    finally:
        finished.set()
        writer.close()
    assert outcome == {'replace': 'ok'}, outcome
    assert physical.read_bytes() == content


# ======================================================= F2: one pass over the source bytes; identity at unlink ==

RECORDED = b'AAAA' + b'recorded generation tail, the bytes the record binds'
OTHER = b'AAAA' + b'BBBB a different generation of the same source file'


def _temporary_record(tmp_path, temporary_bytes):
    path = tmp_path / 'download_results.parquet'
    ps.replace_metadata_parquet(path, pd.DataFrame({'filename': ['a.pdf']}))
    source = tmp_path / 'immutable'
    source.write_bytes(RECORDED)
    temporary = tmp_path / '.download_results.parquet.recorded.tmp'
    temporary.write_bytes(temporary_bytes)
    writer, run = _ledger(tmp_path)
    writer.record_uncommitted_writes(run, [(path, hashlib.sha256(RECORDED).hexdigest(), len(RECORDED), True,
                                            {'temporary': temporary, 'source': source})], wrapper_stage='clean')
    return writer, run, source, temporary


def _change_source_once_read(monkeypatch, source, change):
    """Change ``source`` right after the predicate first reads it: on this tree after the first block of its one pass
    (blocks of 4 bytes), on e8fcb72 after its separate hash (the reviewer's hook point). Returns the event state."""
    state = {}
    if hasattr(W, '_read_block'):
        monkeypatch.setattr(W, '_TEMPORARY_BLOCK', 4)
        real_read, inode = W._read_block, os.stat(source).st_ino

        def read(descriptor, size):
            block = real_read(descriptor, size)
            if block and not state and os.fstat(descriptor).st_ino == inode:
                change(); state['changed_after'] = 'first block of the single pass'
            return block
        monkeypatch.setattr(W, '_read_block', read)
    else:
        real_hash = W.file_hash

        def hashed(target, *args, **kwargs):
            digest = real_hash(target, *args, **kwargs)
            if Path(target) == source and not state:
                change(); state['changed_after'] = 'the separate source hash'
            return digest
        monkeypatch.setattr(W, 'file_hash', hashed)
    return state


def _rewrite_in_place(source, data):
    def change():
        with open(source, 'r+b') as handle:
            handle.seek(0)
            handle.write(data)
            handle.truncate()
    return change


def test_temporary_kept_when_the_source_changes_in_place_during_verification(tmp_path, monkeypatch):
    """Step: source bytes -> prefix proof -> removal. The temporary (``AAAABBBB``) is a prefix of another generation
    of the source, not of the recorded one. The source is rewritten in place to that generation right after the
    predicate first reads it. Here the bytes compared are the bytes hashed: they are not the recorded generation, so
    the temporary is kept. e8fcb72 hashed the recorded generation, then compared against the new one and deleted it."""
    writer, run, source, temporary = _temporary_record(tmp_path, OTHER[:8])
    try:
        state = _change_source_once_read(monkeypatch, source, _rewrite_in_place(source, OTHER))
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        removed = list(writer.removed_temporaries)
        monkeypatch.undo()
    finally:
        writer.close()
    assert state, 'the source must have been changed during the verification'
    assert removed == [] and temporary.read_bytes() == OTHER[:8], 'a modified temporary was deleted'


def test_temporary_removed_when_a_true_prefix_and_the_source_is_replaced_during_verification(tmp_path, monkeypatch):
    """Control: the temporary is a true prefix of the recorded generation and the source path is replaced by another
    file (rename) during the verification. The descriptor still reads the recorded bytes, so the proof holds and the
    temporary is removed. (e8fcb72 compares against the replacing file and keeps it: safe, but not the same proof.)"""
    writer, run, source, temporary = _temporary_record(tmp_path, RECORDED[:10])
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'ZZZZ unrelated replacement')
    try:
        state = _change_source_once_read(monkeypatch, source, lambda: os.replace(replacement, source))
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        removed = list(writer.removed_temporaries)
        monkeypatch.undo()
    finally:
        writer.close()
    assert state
    assert removed == [str(temporary)] and not temporary.exists()


def test_temporary_name_swapped_after_verification_is_not_removed(tmp_path, monkeypatch):
    """Step: verified temporary -> unlink. Right after the verification, the name is replaced by another file with the
    same bytes (a new inode). Only the verified file may be removed, so the name is kept. e8fcb72 unlinked the name."""
    writer, run, source, temporary = _temporary_record(tmp_path, RECORDED[:10])
    swapped = tmp_path / 'swap'
    name = '_verified_temporary' if hasattr(W, '_verified_temporary') else '_abandoned_temporary'
    real, state = getattr(W, name), {}

    def verified(path, details):
        result = real(path, details)
        if Path(path) == temporary and not state:
            swapped.write_bytes(RECORDED[:10])
            os.replace(swapped, temporary)
            state['swapped'] = os.stat(temporary).st_ino
        return result
    monkeypatch.setattr(W, name, verified)
    try:
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        removed = list(writer.removed_temporaries)
        monkeypatch.undo()
    finally:
        writer.close()
    assert state
    assert removed == [] and temporary.exists() and os.stat(temporary).st_ino == state['swapped']
