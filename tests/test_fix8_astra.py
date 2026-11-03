"""Fix round 8 (ASTRA-FIXES2c3 on 6f24169): every check-then-act step of the recovery and rollback code.

N1 [HIGH] Recovery checked a file's bytes without the metadata lock and then restored or removed it: a locked writer
   (``update_metadata_parquet``) could commit in between, and recovery overwrote that commit. Fix: check and act run in
   one critical section of the canonical metadata lock (``writer._recover_locked`` -> ``metadata_target``); the bytes,
   the symlink and the committed generation are verified again under it. The installed bytes are verified before the
   replace (``_copy_file_locked(expected_sha256=...)``, ``_restore_copy``).
N3 [LOW] A process death after the intent record and before the replace left a temporary in the public directory.
   Fix: the record binds the temporary's path (and a publication's source); the repeated barrier removes it under the
   same lock, only if its bytes are the recorded ones (or a prefix of the recorded source) and it has one name.

Each test injects a deterministic event between a check and its act (a hook on the check's own call, as the
reviewer's probes do) and fails without the protection: on 6f24169, or, for a step that did not exist there, with the
protection disabled (the ``protection='off'`` controls, which assert the hazard). The table in fix8/reply.md maps
step -> protection -> test.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import encoding
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, open_readonly
from glossapi.stages.run import DiskFullError
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES, METADATA, _sha, _fail_noise_and_rollback
from test_k4g_fix4 import products


def _failed_clean(tmp_path, monkeypatch):
    """The reviewer's setup: the clean noise merge meets ENOSPC and the rollback at the failure cannot write, so the
    metadata holds the failed wrapper's recorded, uncommitted bytes over a committed projection."""
    output = tmp_path / 'out'
    _fail_noise_and_rollback(monkeypatch, errno.ENOSPC)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    return output, output / METADATA, stopped.value.details['run_id']


def _clean_wrapper_runs(writer, run):
    return [json.loads(row[0])['wrapper_run_id'] for row in writer.connection.execute(
        f"SELECT details_json FROM {writer.connection.event_table} WHERE reason_code='legacy_observed' "
        "AND message='wrapper:clean' AND run_id=?", (run,))]


def _locked_user_edit(path, note):
    """A cooperating writer on another thread: the normal locked read-modify-write, run to completion."""
    def edit(frame):
        frame['user_note'] = note
        return frame
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(ps.update_metadata_parquet, path, edit).result(timeout=10)


def _hook_first_hash(monkeypatch, path, event):
    """Run ``event`` right after the first hash of ``path`` (the recovery's unlocked check) returns."""
    real, state = encoding.file_hash, {}

    def hashed(target, *args, **kwargs):
        digest = real(target, *args, **kwargs)
        if Path(target) == path and not state:
            state['checked'] = digest
            event()
        return digest
    monkeypatch.setattr(encoding, 'file_hash', hashed)
    return state


def _corrupting_copy(monkeypatch, marker):
    """The source changes after its own verification and before the copy: every copy into a recovery temporary of
    ``marker``'s name gets one extra byte."""
    real = shutil.copyfile
    hits = []

    def copy(source, destination, *args, **kwargs):
        result = real(source, destination, *args, **kwargs)
        if marker in Path(destination).name:
            with open(destination, 'ab') as handle:
                handle.write(b'X')
            hits.append(str(destination))
        return result
    monkeypatch.setattr(shutil, 'copyfile', copy)
    return hits


def _seed_record(tmp_path, *, existed, temporary=None, extra=None):
    """Unit ledger: one run with one recorded uncommitted write of the metadata file's current bytes."""
    path = tmp_path / 'download_results' / 'download_results.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'filename': ['a.pdf', 'b.pdf']}).to_parquet(path, index=False)
    writer = LedgerWriter(tmp_path / 'corpus.sqlite3')
    with writer.transaction():
        run = writer.run(SETTINGS6, {}, {}, code_revision='fix8')
    entry = (ps.resolve_metadata_target(path), _sha(path), path.stat().st_size, existed, dict(extra or {}))
    if temporary is not None:
        entry[4]['temporary'] = temporary
    writer.record_uncommitted_writes(run, [entry], wrapper_stage='clean')
    return writer, run, path


# ================================================ N1: repeated-barrier rollback (roll_back_uncommitted_writes) ===

def test_rollback_removal_keeps_a_locked_edit_committed_after_the_check(tmp_path, monkeypatch):
    """Step: hash check -> REMOVE (the file the failed attempt created). A locked writer commits between the unlocked
    hash and the removal. 6f24169 removes the committed edit; here it is re-verified under the lock and kept."""
    writer, run, path = _seed_record(tmp_path, existed=False)
    try:
        state = _hook_first_hash(monkeypatch, path, lambda: _locked_user_edit(path, 'committed after the check'))
        changed = writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        monkeypatch.undo()
    finally:
        writer.close()
    assert state, 'the hook must have fired at the rollback check'
    assert path.is_file(), 'the committed edit was removed'
    assert 'user_note' in pd.read_parquet(path) and changed == []


def test_rollback_removal_still_removes_the_failed_attempts_own_file(tmp_path):
    """Control for the test above: with no concurrent edit the failed attempt's own file is removed, as on 6f24169."""
    writer, run, path = _seed_record(tmp_path, existed=False)
    expected = str(ps.resolve_metadata_target(path))
    try:
        changed = writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
    finally:
        writer.close()
    assert not path.exists() and changed == [expected]


def test_rollback_restore_keeps_a_locked_edit_committed_after_the_check(tmp_path, monkeypatch):
    """Step: hash check -> RESTORE (the committed projection over the failed attempt's bytes). Same race as the
    reviewer's direct probe, written here with the shared hook: the edit survives, nothing is restored over it."""
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        runs = _clean_wrapper_runs(writer, run)
        state = _hook_first_hash(monkeypatch, path, lambda: _locked_user_edit(path, 'committed after the check'))
        changed = writer.roll_back_uncommitted_writes(runs, output, observation_run=run)
        monkeypatch.undo()
    assert state and changed == []
    assert 'user_note' in pd.read_parquet(path)


def test_rollback_does_not_restore_through_a_symlink_swapped_in_after_the_check(tmp_path, monkeypatch):
    """Step: symlink check -> RESTORE. Between the unlocked look and the restore the path becomes a symlink to another
    file holding the same bytes. 6f24169 resolved the symlink at the restore and overwrote the other file; here the
    path no longer resolves to the recorded physical file under the lock, so nothing is written."""
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    elsewhere = tmp_path / 'elsewhere.parquet'

    def swap():
        shutil.copyfile(path, elsewhere)
        link = path.with_name('.swap-link')
        os.symlink(elsewhere, link)
        os.replace(link, path)
    before = None
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        runs = _clean_wrapper_runs(writer, run)
        state = _hook_first_hash(monkeypatch, path, swap)
        writer.roll_back_uncommitted_writes(runs, output, observation_run=run)
        monkeypatch.undo()
        before = state['checked']
    assert path.is_symlink()
    assert _sha(elsewhere) == before, 'the file behind the swapped-in symlink was overwritten'


def test_rollback_verifies_the_bytes_it_installs(tmp_path, monkeypatch):
    """Step: source verification -> copy. The managed source changes after its check and before the copy. 6f24169
    replaced the metadata with the changed bytes and raised afterwards; here the temporary's bytes are checked
    before the replace, so the file keeps its bytes (a later retry can still roll them back)."""
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    before = _sha(path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        runs = _clean_wrapper_runs(writer, run)
        hits = _corrupting_copy(monkeypatch, METADATA.name)
        with pytest.raises(ValueError):
            writer.roll_back_uncommitted_writes(runs, output, observation_run=run)
        monkeypatch.undo()
    assert hits, 'the rollback must have tried the restore'
    assert _sha(path) == before, 'changed source bytes were installed'
    assert not [p.name for p in path.parent.iterdir() if p.name.endswith('.tmp')]


def test_ledger_cannot_change_during_a_locked_recovery(tmp_path, monkeypatch):
    """Steps: committed-generation choice and the ``existed`` record -> act. Both are read from the ledger, which only
    the owner thread of the process holding the corpus flock can write. Shown inside the critical section itself:
    another thread's ledger write and a second writer on the corpus are both refused there."""
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    seen = {}
    real = W._recover_locked

    def probe(target_path, decide, **kwargs):  # fix 9: _recover_locked also takes the recorded lock
        def inside(target):
            with ThreadPoolExecutor(max_workers=1) as pool:
                seen['thread'] = repr(pool.submit(lambda: writer.record_uncommitted_write(
                    run, path, '0' * 64, 1, wrapper_stage='clean')).exception(timeout=10))
            try:
                LedgerWriter(output / 'corpus.sqlite3')
                seen['second'] = 'opened'
            except Exception as error:  # noqa: BLE001 - the refusal itself is the observation
                seen['second'] = repr(error)
            return decide(target)
        return real(target_path, inside, **kwargs)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        runs = _clean_wrapper_runs(writer, run)
        monkeypatch.setattr(W, '_recover_locked', probe)
        writer.roll_back_uncommitted_writes(runs, output, observation_run=run)
        monkeypatch.undo()
    assert 'Only the owning orchestrator thread may write' in seen['thread'], seen
    assert seen['second'] != 'opened', seen


# ============================================================ N1: run-start recovery (verify_excluded_copies) ===

def _torn_named_metadata(tmp_path, monkeypatch):
    """The reviewer's failed clean, after which the run close named the metadata copy as unflushed; the copy is then
    torn the way a power loss leaves it (a prefix of its registered bytes), with the close's recorded timestamps
    presented (ctime cannot be set from user space). Returns (output, path, torn sha256, registered sha256)."""
    output, path, run = _failed_clean(tmp_path, monkeypatch)
    with open_readonly(output / 'corpus.sqlite3') as db:
        details = json.loads(db.execute(
            f"SELECT details_json FROM {db.event_table} WHERE reason_code='durability_barrier' AND event_kind='retention' "
            "AND (message='barrier_summary' OR message='barrier') ORDER BY event_id DESC LIMIT 1").fetchone()[0])
        named = dict(zip(details['run_close_excluded_copy_file_ids'], details['run_close_excluded_copy_identities']))
        rows = {file_id: db.execute('SELECT uri,sha256 FROM current_file_state WHERE file_id=?', (file_id,)).fetchone()
                for file_id in named}
        file_id = next(key for key, row in rows.items() if row[0].endswith('/' + str(METADATA)))
        registered = rows[file_id][1]
        source = next(row[0] for row in db.execute(
            "SELECT uri FROM current_file_state WHERE sha256=? AND presence='present' AND uri LIKE '%/.glossapi/artifacts/%'",
            (registered,)))
    from urllib.parse import unquote, urlsplit
    data = Path(unquote(urlsplit(source).path)).read_bytes()
    path.write_bytes(data[:len(data) // 2])
    mtime, ctime = named[file_id]
    real = Path.lstat

    class Presented:
        def __init__(self, stat):
            self.stat = stat

        def __getattr__(self, key):
            return {'st_mtime_ns': mtime, 'st_ctime_ns': ctime}[key] if key in ('st_mtime_ns', 'st_ctime_ns') \
                else getattr(self.stat, key)
    monkeypatch.setattr(Path, 'lstat', lambda self: Presented(real(self)) if self == path else real(self))
    return output, path, _sha(path), registered


def test_startup_torn_copy_restore_keeps_a_locked_edit_committed_after_the_check(tmp_path, monkeypatch):
    """Step: torn-image classification (``_left_by_power_loss``) -> RESTORE. A locked writer replaces the torn copy
    with complete new content right after the classification. 6f24169 restored the managed bytes over it; here the copy
    is classified again under the lock: complete different content, kept and refused."""
    output, path, torn, registered = _torn_named_metadata(tmp_path, monkeypatch)
    real, state = LedgerWriter._left_by_power_loss, {}

    def classified(self, copy, source, row):
        verdict = real(self, copy, source, row)
        if Path(copy) == path and not state:
            state['torn'] = verdict
            frame = pd.DataFrame({'filename': ['a.pdf'], 'user_note': ['written after the classification']})
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(ps.replace_metadata_parquet, path, frame).result(timeout=10)
        return verdict
    monkeypatch.setattr(LedgerWriter, '_left_by_power_loss', classified)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        writer.verify_excluded_copies()
        ambiguous = list(getattr(writer, 'ambiguous_copies', []))
    monkeypatch.undo()
    assert state.get('torn') is True, state
    assert pd.read_parquet(path).user_note.tolist() == ['written after the classification']
    assert [entry['reason'] for entry in ambiguous] == ['complete_different_content'], ambiguous


def test_startup_torn_copy_restore_still_republishes_without_an_edit(tmp_path, monkeypatch):
    """Control: without the concurrent edit the torn copy is republished from its managed source (K4C-M2)."""
    output, path, torn, registered = _torn_named_metadata(tmp_path, monkeypatch)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        restored = writer.verify_excluded_copies()
    monkeypatch.undo()
    assert restored and _sha(path) == registered


def test_startup_restore_verifies_the_bytes_it_installs(tmp_path, monkeypatch):
    """Step: source verification -> copy, at the run start. The source changes after its check. 6f24169 installed
    the changed bytes and went on silently; here nothing is replaced and the copy is left for verification."""
    output, path, torn, registered = _torn_named_metadata(tmp_path, monkeypatch)
    hits = _corrupting_copy(monkeypatch, METADATA.name)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        restored = writer.verify_excluded_copies()
    monkeypatch.undo()
    assert hits and restored == []
    assert _sha(path) == torn, 'changed source bytes were installed at the run start'


def test_projection_restore_verifies_the_bytes_it_installs(tmp_path, monkeypatch):
    """Step: ``restore_metadata_projections`` (wrapper start, fix 6): immutable verification -> copy. Its hash check
    and copy were already under the metadata lock; the copy itself is now verified before the replace."""
    from glossapi.stages.metadata import restore_metadata_projections
    output = tmp_path / 'out'
    Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    path = output / METADATA
    ps.replace_metadata_parquet(path, pd.DataFrame({'filename': ['a.pdf'], 'uncommitted': [1]}))
    before = _sha(path)
    hits = _corrupting_copy(monkeypatch, METADATA.name)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        with pytest.raises(ValueError):
            restore_metadata_projections(SimpleNamespace(writer=writer, corpus=SimpleNamespace(output_dir=output)))
    monkeypatch.undo()
    assert hits and _sha(path) == before


# ======================================================== N3: the temporary bound in the intent record ========

def test_intent_record_binds_the_temporary(tmp_path):
    """The choke point's record names its temporary file; the publication's names its temporary and its source."""
    output = tmp_path / 'out'
    Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    with open_readonly(output / 'corpus.sqlite3') as connection:
        records = [json.loads(row[0]) for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE reason_code='uncommitted_projection_write'")]
    real = os.path.realpath
    assert records and all(real(Path(r['temporary']).parent) == real(Path(r['path']).parent) for r in records)
    assert all(Path(r['temporary']).name.startswith((Path(r['path']).name, '.' + Path(r['path']).name)) for r in records)
    assert any('source' in r and '/.glossapi/' in r['source'] for r in records)
    assert not [p for p in output.rglob('*.tmp') if '.glossapi' not in p.parts]


CRASH_CHILD = r'''
import os, sys, json
from pathlib import Path
from types import SimpleNamespace
from glossapi import Corpus
from glossapi.ledger.writer import open_readonly
import glossapi.stages.products as P
from test_fix6_astra import SETTINGS6
root, step = Path(sys.argv[1]), sys.argv[2]
out = root / 'out'
if __name__ == '__main__':
    if step in ('publication_full', 'publication_partial'):
        real = os.replace
        def replace(temporary, written):
            if Path(written).name == 'export.jsonl':
                if step == 'publication_partial':
                    size = os.path.getsize(temporary)
                    os.truncate(temporary, size // 2)
                os._exit(92)
            return real(temporary, written)
        P.os = SimpleNamespace(**{name: getattr(os, name) for name in dir(os) if not name.startswith('__')})
        P.os.replace = replace
    if step == 'resume':
        with open_readonly(out / 'corpus.sqlite3') as db:
            run = db.execute("SELECT run_id FROM tasks WHERE operation='export.compatibility_publish' ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        result = Corpus(root / 'empty', out).run(resume=run, retry_failed=True)
    else:
        result = Corpus(root / 'empty', out).run([Path(p) for p in sys.argv[3:]], stages=('extract', 'clean', 'export'), settings=SETTINGS6)
    print(json.dumps({'state': result['state']}))
'''


@pytest.mark.parametrize('cut', ['publication_full', 'publication_partial'])
def test_process_exit_at_a_publication_replace_leaves_no_temporary(tmp_path, cut):
    """A process death after the publication's intent record, at the replace of ``export.jsonl`` (temporary complete,
    or its copy cut short at half). The resume repeats the export barrier, removes exactly that verified temporary and
    ends with the uninterrupted run's products (no hidden temporary left)."""
    sources = _sources(tmp_path)
    script = tmp_path / 'crash_child.py'
    script.write_text(CRASH_CHILD)

    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(tmp_path / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([*filter(None, [os.environ.get('PYTHONPATH')]), str(repo / 'src'), str(repo / 'tests')])}

    def child(root, step):
        root.mkdir(parents=True, exist_ok=True)
        return subprocess.run([sys.executable, '-B', str(script), str(root), step, *map(str, sources)], capture_output=True,
                              text=True, timeout=300, env=env, cwd=root)
    reference = child(tmp_path / 'ref', 'reference')
    assert reference.returncode == 0, reference.stderr[-2000:]
    killed = child(tmp_path / 'faulted', cut)
    assert killed.returncode == 92, killed.stderr[-2000:]
    left = [p for p in (tmp_path / 'faulted/out').iterdir() if p.name.startswith('.export.jsonl.')]
    assert len(left) == 1, 'the crash must leave the publication temporary'
    resumed = child(tmp_path / 'faulted', 'resume')
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    assert not left[0].exists()
    assert products(tmp_path / 'ref/out') == products(tmp_path / 'faulted/out')


def test_temporary_is_removed_only_when_verified(tmp_path):
    """Recorded path AND recorded bytes AND one name: a recorded temporary holding other bytes, a hard-linked one and an
    unrecorded ``*.tmp`` holding the recorded bytes are all kept; only the verified one is removed."""
    folder = tmp_path / 'download_results'
    folder.mkdir()
    exact, other, linked = (folder / f'download_results.parquet.{name}.tmp' for name in ('exact', 'other', 'linked'))
    unrecorded = folder / 'download_results.parquet.unrecorded.tmp'
    writer, run, path = _seed_record(tmp_path, existed=True, temporary=exact)
    try:
        digest, size = _sha(path), path.stat().st_size
        for extra in (other, linked):
            writer.record_uncommitted_writes(run, [(path, digest, size, True, {'temporary': extra})], wrapper_stage='clean')
        for target in (exact, linked, unrecorded):
            shutil.copyfile(path, target)
        other.write_bytes(path.read_bytes()[:-1] + b'Z')
        os.link(linked, tmp_path / 'second-name')
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        removed = list(writer.removed_temporaries)
    finally:
        writer.close()
    assert removed == [str(exact)]
    assert not exact.exists() and other.exists() and linked.exists() and unrecorded.exists() and path.exists()


@pytest.mark.parametrize('protection', ['on', 'off'])
def test_temporary_of_a_live_locked_writer_is_not_removed(tmp_path, monkeypatch, protection):
    """Step: temporary verification -> removal. A writer still holding the metadata lock owns the recorded temporary
    (verified bytes) and replaces with it before it releases the lock. With the lock (``on``) the removal waits for the
    writer, whose replace succeeds; ``off`` (the lock bypassed) is the control: the removal deletes the live writer's
    temporary under it and the writer's replace fails, which is the hazard the lock prevents."""
    temporary = tmp_path / 'download_results' / 'download_results.parquet.live.tmp'
    writer, run, path = _seed_record(tmp_path, existed=True, temporary=temporary)
    shutil.copyfile(path, temporary)
    if protection == 'off':
        def unlocked(target, decide, **kwargs):  # fix 9: _recover_locked also takes the recorded lock
            action = decide(Path(target))
            if action is not None and action[0] == 'remove':
                os.unlink(action[1])
            return action
        monkeypatch.setattr(W, '_recover_locked', unlocked)
    holding, finished, outcome = threading.Event(), threading.Event(), {}

    def live_writer():
        with ps.metadata_target(path) as physical:
            holding.set()
            finished.wait(timeout=1.5)  # the rollback finishes meanwhile only if it does not take the lock
            try:
                os.replace(temporary, physical)
                outcome['replace'] = 'ok'
            except FileNotFoundError as error:
                outcome['replace'] = repr(error)
    thread = threading.Thread(target=live_writer)
    thread.start()
    try:
        assert holding.wait(timeout=10)
        writer.roll_back_uncommitted_writes([run], tmp_path, observation_run=run)
        finished.set()
        thread.join(timeout=20)
    finally:
        finished.set()
        thread.join(timeout=20)
        writer.close()
    if protection == 'on':
        assert outcome == {'replace': 'ok'}, outcome
    else:
        assert outcome['replace'].startswith('FileNotFoundError'), outcome
