"""Fix round 12 (ASTRA-FIXES2c7 on ed7044a): reconcile every path to its authoritative target; the shutdown race.

Recovery and rollback no longer reason over event history ("a later registration supersedes", "a rollback observation
restored it", "the directory was flushed earlier"). ``LedgerWriter.authoritative_target(path)`` computes, from the ledger
alone and by registration order, what the path must hold (``TargetBytes`` of the newest committed registration,
``TargetAbsent`` after a policy deletion of the path, ``NoTarget`` for a caller-seeded file); recovery makes the bytes
match it from that registration's own source, verifies them, or refuses. Durability is a phase: every restore first,
then each affected directory is flushed once, then IDs become durable. A present registration that contradicts the
installed bytes is refused (``contradictory_registrations``).

R1 [HIGH] durable after a stale directory flush; R2 [HIGH] successor / rollback observation taken as proof; R3 [HIGH]
an obsolete generation restored when the newest one's source was unavailable; R4 [MEDIUM] the sweep's checker read the
ledger after recovery; R5 [HIGH] ``_stop_process`` aborted when advancing the ``/proc`` glob raised ESRCH.

The sweep (``test_fault_sweep``, ``test_rollback_fault_matrix``) checks every completed resume against a product
inventory frozen BEFORE the fault (from the original run's committed outputs), plus the ledger's present registrations
for contradictions. Matrix rows go to ``FIX12_FAULT_MATRIX`` when that variable names a file (fix12/reply.md).
"""
from __future__ import annotations

import builtins
import errno
import hashlib
import json
import os
import pathlib
import shutil
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import encoding
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix6_astra import _sources, SETTINGS6
from test_fix7_astra import STAGES
from test_k4c2_fixes import published_copy
from test_k4g_fix4 import products

CONTENT = b'{"filename":"uncommitted.pdf","text":"bytes a failed attempt put in place"}\n'
NEWER = b'{"filename":"newer.pdf","text":"a newer committed generation"}\n'


def _sha(data_or_path):
    data = data_or_path if isinstance(data_or_path, bytes) else Path(data_or_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _refusal(function):
    try:
        return function(), None
    except Exception as error:  # noqa: BLE001 - the typed refusal is the observation
        return None, error


def _conditions(error_or_list):
    entries = error_or_list if isinstance(error_or_list, list) else getattr(error_or_list, 'conditions', [])
    return [(Path(entry['path']).name, entry['reason']) for entry in entries]


def _register_generation(writer, run, path, data, source, finish='ok'):
    """A newer committed generation of ``path`` (its own task/attempt, its immutable source, the copy). ``finish``: the
    producer attempt's terminal status (fix 13: only ``ok`` makes the registration committed), ``None`` = unfinished."""
    path.write_bytes(data)
    with writer.transaction():
        manifest = writer.snapshot('inputs', {'kind': 'inputs', 'files': [_sha(data)]})
        task = writer.task(run, 'export', 'export.jsonl', {}, scope_key='corpus:' + manifest)
        attempt = writer.attempt(task, worker_id='fix12', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        source_id = None
        if source is not None:
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(data)
            source_id = writer.artifact(run, 'export', 'document_jsonl', source.as_uri(), producer_attempt_id=attempt,
                                        sha256=_sha(data), bytes=len(data), retention_class='final')
        copy_id = writer.artifact(run, 'export', 'document_jsonl', path.as_uri(), producer_attempt_id=attempt,
                                  sha256=_sha(data), bytes=len(data), retention_class='final')
        writer.snapshot('product_projection', {'kind': 'product_projection', 'product_uri': path.as_uri(),
                                               'immutable_uri': source.as_uri() if source is not None else None,
                                               'sha256': _sha(data), 'producer_attempt_id': attempt})
        if finish:
            writer.finish(run, task, attempt, status=finish, reason_code='export_completed' if finish == 'ok' else 'export_exception')
    return source_id, copy_id


# ================================================================ authoritative_target: the per-path state ====

def test_authoritative_target_is_the_newest_registration_by_event_order(tmp_path):
    """Bytes of the newest registration (with ITS source), Absent after a policy deletion of it, NoTarget without any
    managed registration. Presence observations do not reorder registrations."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        newer_source = tmp_path / '.glossapi/artifacts/newer/export.jsonl'
        newer_source_id, newer_id = _register_generation(writer, run, path, NEWER, newer_source)
        with writer.transaction():  # an observation of the OLD registration after the newer one: not a registration
            writer.event(run, event_kind='artifact', reason_code='present', file_id=copy_id, details={'observation': 'fix12'})
        target = writer.authoritative_target(path)
        assert isinstance(target, W.TargetBytes) and target.registration['file_id'] == newer_id
        assert target.sha256 == _sha(NEWER) and target.source['file_id'] == newer_source_id
        with writer.transaction():  # fix 13: a committed policy deletion (retention), never a presence observation
            writer.event(run, event_kind='retention', reason_code='delete_completed', file_id=newer_id, details={'policy_deletion': True})
        assert isinstance(writer.authoritative_target(path), W.TargetAbsent)
        assert isinstance(writer.authoritative_target(tmp_path / 'never-registered.jsonl'), W.NoTarget)


# ================================================== R1: durability is a phase after every replacement ===========

def _two_named_copies(tmp_path):
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, first, source_id, copy_id = published_copy(writer, tmp_path)
        second = tmp_path / 'second.jsonl'
        second.write_bytes(source.read_bytes())
        row = writer.file_catalog()[copy_id]
        with writer.transaction():
            second_id = writer.artifact(run, 'export', 'document_jsonl', second.as_uri(), producer_attempt_id=row['producer_attempt_id'],
                                        sha256=row['sha256'], bytes=row['bytes'], retention_class='final')
            # Fix 15 (ASTRA-FIXES2c10 H1): a second named copy is committed only by its own publication record.
            writer.snapshot('publication_binding', {'kind': 'publication_binding', 'product_uri': second.as_uri(), 'sha256': row['sha256'],
                                                    'producer_attempt_id': row['producer_attempt_id'], 'file_id': second_id})
        writer.durable_barrier()
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        for path in (first, second):
            writer.record_uncommitted_write(run, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
            path.write_bytes(CONTENT)
    return run, source, (first, second), (copy_id, second_id)


def _trace_restores_and_directory_flushes(monkeypatch, directory, fail_directory=False):
    trace, real_restore, real_fsync = [], W._restore_copy, os.fsync
    identity = (directory.stat().st_dev, directory.stat().st_ino)

    def restore(source, destination, expected):
        real_restore(source, destination, expected)
        trace.append(('replace', Path(destination).name))

    def fsync(descriptor):
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == identity:
            trace.append(('fsync_directory', directory.name))
            if fail_directory:
                raise OSError(errno.EIO, 'injected directory flush failure')
        return real_fsync(descriptor)
    monkeypatch.setattr(W, '_restore_copy', restore)
    monkeypatch.setattr(W, 'os', SimpleNamespace(**{key: (fsync if key == 'fsync' else getattr(os, key)) for key in dir(os) if not key.startswith('__')}))
    return trace


def _next_close_excluded(writer, run):
    writer.barrier_context = {'run_id': run, 'debug': 'off'}
    writer.durable_barrier()
    record = json.loads(writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} "
                                                  "WHERE message='barrier_summary' ORDER BY event_id DESC LIMIT 1").fetchone()[0])
    return set(record.get('run_close_excluded_copy_file_ids') or [])


def test_two_copies_are_durable_only_after_the_directory_flush_that_follows_both_restores(tmp_path, monkeypatch):
    """Step: restore -> directory flush -> durable. Both copies in one directory are restored, THEN the directory is
    flushed once, THEN both IDs are durable (so the next close need not name them). ed7044a flushed the directory after
    the first restore and certified the second replacement with that earlier flush."""
    run, source, paths, ids = _two_named_copies(tmp_path)
    trace = _trace_restores_and_directory_flushes(monkeypatch, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        restored = writer.verify_excluded_copies()
        assert _refusal(writer.refuse_ambiguous_copies)[1] is None
        durable = [file_id in writer._durable_files for file_id in ids]
        excluded = _next_close_excluded(writer, run)
    replaces = [index for index, event in enumerate(trace) if event[0] == 'replace']
    flushes = [index for index, event in enumerate(trace) if event[0] == 'fsync_directory']
    assert sorted(restored) == sorted(ids) and durable == [True, True]
    assert len(replaces) == 2 and flushes and min(flushes) > max(replaces), trace
    assert not (excluded & set(ids)), 'durable copies need not be named by the next close'
    assert all(path.read_bytes() == source.read_bytes() for path in paths)


def test_failed_directory_flush_leaves_both_copies_named_for_the_next_start(tmp_path, monkeypatch):
    """The directory flush after the restores fails: neither copy is durable, the start is refused, and the next close
    still names both (the obligation is kept)."""
    run, source, paths, ids = _two_named_copies(tmp_path)
    _trace_restores_and_directory_flushes(monkeypatch, tmp_path, fail_directory=True)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
        durable = [file_id in writer._durable_files for file_id in ids]
        monkeypatch.undo()
        excluded = _next_close_excluded(writer, run)
    assert sorted(_conditions(error)) == [('export.jsonl', 'flush_failed'), ('second.jsonl', 'flush_failed')], repr(error)
    assert durable == [False, False] and set(ids) <= excluded


# ======================================== R2: a successor or an observation is not proof: reconcile the path =====

@pytest.mark.parametrize('state', ['missing', 'corrupt', 'intact'])
def test_successor_outside_the_barrier_is_reconciled(tmp_path, state):
    """A newer committed registration of the named path (not in the old barrier's list): the path is reconciled to it
    (restored from ITS source when missing or holding uncommitted bytes). ed7044a took the successor's existence as
    proof and completed with the export missing or wrong."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:  # no run context: its close records no new barrier
        newer_source_id, newer_id = _register_generation(writer, run, path, NEWER, tmp_path / '.glossapi/artifacts/newer/export.jsonl')
        if state == 'corrupt':
            writer.record_uncommitted_write(run, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
    if state == 'missing':
        path.unlink()
    elif state == 'corrupt':
        path.write_bytes(CONTENT)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        restored = writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
        durable = newer_id in writer._durable_files
    assert error is None, repr(error)
    assert path.read_bytes() == NEWER and durable
    assert restored == ([] if state == 'intact' else [newer_id])


def test_rollback_observation_does_not_discharge_the_named_copy(tmp_path):
    """The rollback found no recorded bytes at a path (the file is gone) and took the failed attempt's registration
    off it. That proves only that the failed bytes are gone: the run start reconciles the path to its committed target
    and restores it. ed7044a took the observation as proof of a restored registration and left no export."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix12-failed')
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        failed_source_id, failed_id = _register_generation(writer, failed, path, CONTENT, tmp_path / '.glossapi/artifacts/failed/export.jsonl',
                                                           finish=None)
        writer.barrier_context = {'run_id': failed, 'debug': 'off'}
        writer.durable_barrier()
    path.unlink()
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        assert writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run) == []
        assert writer.file_catalog()[failed_id]['presence'] == 'missing'
        target = writer.authoritative_target(path)
        restored = writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
    assert isinstance(target, W.TargetBytes) and target.registration['file_id'] == copy_id
    assert error is None and restored == [copy_id] and path.read_bytes() == source.read_bytes()


# ========================== R3: the newest registration decides; an older generation is never substituted =======

@pytest.mark.parametrize('source_state', ['missing', 'released'])
def test_rollback_never_restores_an_obsolete_generation(tmp_path, source_state):
    """The newest committed registration's source is gone (lost, or released by policy). ed7044a restored the OLDER
    generation because its source was intact. Here: lost -> ``no_intact_source`` (listed, file kept); released -> the
    failed bytes stay for the repeating publication and the newest registration no longer claims the path."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, old_source, path, old_source_id, old_id = published_copy(writer, tmp_path)
        newer_source = tmp_path / '.glossapi/artifacts/newer/export.jsonl'
        newer_source_id, newer_id = _register_generation(writer, run, path, NEWER, newer_source)
        with writer.transaction():
            failed = writer.run({}, {}, {}, code_revision='fix12-failed')
            if source_state == 'released':
                writer.event(run, event_kind='artifact', reason_code='deleted', file_id=newer_source_id, details={'policy_deletion': True})
        newer_source.unlink()
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        changed = writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=run)
        listed = _conditions(writer.rollback_unresolved)
        presence = writer.file_catalog()[newer_id]['presence']
    assert changed == [] and path.read_bytes() == CONTENT, 'an obsolete generation was installed'
    if source_state == 'missing':
        assert listed == [('export.jsonl', 'no_intact_source')]
    else:
        assert listed == [] and presence == 'missing'


def test_contradictory_present_registrations_are_refused(tmp_path):
    """Rule 4: after reconciliation every registration still present for the path agrees with the installed bytes. An
    older registration forced back to ``present`` beside the newest one contradicts it: refused."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        run, source, path, source_id, copy_id = published_copy(writer, tmp_path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        _register_generation(writer, run, path, NEWER, tmp_path / '.glossapi/artifacts/newer/export.jsonl')
        with writer.transaction():
            writer.event(run, event_kind='artifact', reason_code='present', file_id=copy_id, details={'observation': 'fix12'})
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        writer.verify_excluded_copies()
        _, error = _refusal(writer.refuse_ambiguous_copies)
    assert _conditions(error) == [('export.jsonl', 'contradictory_registrations')], repr(error)


# ================================================================ R5: the /proc enumeration race ==============

class _FakeProc:
    """A ``/proc`` whose enumeration raises ESRCH while advancing (``fail`` times), then lists ``table``."""
    def __init__(self, table, fail):
        self.table, self.fail, self.attempts = table, fail, 0

    def path(self, value):
        proc = self

        class P:
            def __init__(self, text):
                self.text = str(text)
                self.name = self.text.rsplit('/', 1)[-1]

            @property
            def parent(self):
                return P(self.text.rsplit('/', 1)[0])

            def exists(self):
                return self.text == '/proc/self/stat'

            def read_text(self):
                identifier = int(self.text.split('/')[2])
                if identifier not in proc.table:
                    raise ProcessLookupError(errno.ESRCH, 'No such process', self.text)
                parent, group, state = proc.table[identifier]
                return f'{identifier} (python) {state} {parent} {group} 0'

            def glob(self, pattern):
                proc.attempts += 1
                if proc.attempts <= proc.fail:
                    yield P('/proc/1/stat')
                    raise ProcessLookupError(errno.ESRCH, 'injected exiting process', '/proc/424242/stat')
                for identifier in sorted(proc.table):
                    yield P(f'/proc/{identifier}/stat')
        return P(value)


def _stop_with(monkeypatch, table, fail, expect_untrusted=False):
    from glossapi.stages import execution as E
    proc, sent = _FakeProc(table, fail), []
    monkeypatch.setattr(E, 'Path', proc.path)
    monkeypatch.setattr(E, 'os', SimpleNamespace(kill=lambda *a: sent.append(('kill',) + a),
                                                 killpg=lambda *a: sent.append(('killpg',) + a)))
    process = SimpleNamespace(pid=100, join=lambda *a: (sent.append(('join',) + a), table.update({100: (1, 100, 'Z'), 101: (100, 101, 'Z')})))
    if expect_untrusted:  # fix 13 (H4): an incomplete enumeration is never certified as fenced
        with pytest.raises(RuntimeError, match='untrusted'):
            E._stop_process(process)
    else:
        E._stop_process(process)
    return proc, sent


def test_stop_process_survives_esrch_while_the_proc_glob_advances(monkeypatch):
    """The ``/proc`` glob raises ESRCH once while advancing: the enumeration is retried, the worker and its descendant
    are signalled (TERM, then KILL) and joined. ed7044a raised out of ``_stop_process`` with no signal sent."""
    proc, sent = _stop_with(monkeypatch, {1: (0, 1, 'S'), 100: (1, 100, 'S'), 101: (100, 101, 'S')}, fail=1)
    assert proc.attempts == 2
    assert ('kill', 101, signal.SIGTERM) in sent and ('kill', 100, signal.SIGKILL) in sent and ('join',) in sent


def test_stop_process_still_stops_the_worker_when_enumeration_keeps_failing(monkeypatch):
    """Every enumeration attempt fails: the known family (the worker itself) is still signalled and joined; then, since
    the enumeration never completed, the workspace is reported untrusted (fix 13, H4) instead of certified fenced."""
    proc, sent = _stop_with(monkeypatch, {1: (0, 1, 'S'), 100: (1, 100, 'S')}, fail=99, expect_untrusted=True)
    assert proc.attempts == 3
    assert ('kill', 100, signal.SIGTERM) in sent and ('kill', 100, signal.SIGKILL) in sent and ('join',) in sent


def test_worker_slot_is_cleaned_up_even_if_stopping_fails(monkeypatch):
    """Per-slot cleanup: the pipe is closed even when ``_stop_process`` raises."""
    from glossapi.stages import workers
    closed = []

    def fail(process):
        raise RuntimeError('Worker descendants could not be fenced; workspace remains untrusted')
    monkeypatch.setattr(workers, '_stop_process', fail)
    pool = SimpleNamespace(slots=[])
    slot = {'process': SimpleNamespace(exitcode=None, close=lambda: closed.append('process')),
            'pipe': SimpleNamespace(close=lambda: closed.append('pipe'))}
    pool.slots.append(slot)
    with pytest.raises(RuntimeError):
        workers.WarmWorkerPool._discard(pool, slot)
    assert closed == ['pipe'] and slot not in pool.slots


# ===================================================================================== the fault sweep ===

# One line per operation recovery performs. ``os.*`` hooks the writer's, the metadata choke point's and the hashing
# module's ``os`` (proxies); the others are patched in place. Hooks fire only inside ``verify_excluded_copies`` /
# ``roll_back_uncommitted_writes``, on the Nth call that touches the corpus (path operations) or the Nth call
# (descriptor operations).
OPERATIONS = {
    'os.open': ('os', 'open'), 'os.read': ('os', 'read'), 'os.stat': ('os', 'stat'), 'os.lstat': ('os', 'lstat'),
    'os.fstat': ('os', 'fstat'), 'os.lseek': ('os', 'lseek'), 'os.close': ('os', 'close'), 'os.fsync': ('os', 'fsync'),
    'os.replace': ('os', 'replace'), 'os.unlink': ('os', 'unlink'),
    'Path.stat': ('pathlib', 'stat'), 'Path.lstat': ('pathlib', 'lstat'), 'Path.open': ('pathlib', 'open'),
    'builtins.open': ('builtins', 'open'), 'shutil.copyfile': ('shutil', 'copyfile'), 'file_hash': ('encoding', 'file_hash'),
}
DESCRIPTOR_OPERATIONS = {'os.read', 'os.fstat', 'os.lseek', 'os.close', 'os.fsync'}
FAULTS = ('missing', 'mismatch', 'eio', 'enospc', 'edquot', 'partial')
SCENARIOS = ('startup_intact', 'startup_restore', 'two_copies', 'successor_missing', 'successor_corrupt', 'failed_clean',
             'publication_crash')
# Call-index targeting: the 2nd and 3rd call of the operations whose later calls have their own role (the second copy's
# replace, the directory flush after both restores, the open of that flush).
CALL_INDEXES = [(operation, fault, scenario, 1) for operation in OPERATIONS for fault in FAULTS for scenario in SCENARIOS] + \
    [(operation, fault, 'two_copies', index) for operation in ('os.fsync', 'os.replace', 'os.open') for fault in FAULTS for index in (2, 3)]
_TEMPLATES: dict[str, tuple[Path, Path, str, dict[str, str]]] = {}


def _visible(output):
    """The product map without residue: hidden temporaries are never products (F4, FIX8-G1)."""
    return {relative: digest for relative, digest in products(output).items()
            if not (Path(relative).name.startswith('.') and Path(relative).name.endswith(('.tmp', '.restore')))}


def _contradictions(output):
    """Paths whose present managed registrations disagree with each other or with the file."""
    from urllib.parse import unquote, urlsplit
    with open_readonly(output / 'corpus.sqlite3') as connection:
        rows = connection.execute("SELECT uri, sha256 FROM current_file_state WHERE presence='present' AND ownership='managed'").fetchall()
    by_path: dict[Path, set[str]] = {}
    for uri, digest in rows:
        path = Path(unquote(urlsplit(uri).path))
        if '.glossapi' not in path.parts:
            by_path.setdefault(path, set()).add(digest)
    return sorted(str(path.relative_to(output)) for path, digests in by_path.items()
                  if len(digests) > 1 or not path.is_file() or _sha(path) not in digests)


def _reference(root, scenario):
    """The original run's committed products, computed before any fault: an uninterrupted run of the same inputs."""
    reference = root / 'reference'
    if scenario == 'publication_crash':
        return None  # computed from the crash child's reference run, see _build
    Corpus(reference / 'empty', reference / 'out').run(_sources(reference), stages=STAGES, settings=SETTINGS6)
    return _visible(reference / 'out')


def _build(scenario, root):
    """The recovery state of ``scenario`` at ``root``: (output, resume run id, inventory frozen before any fault)."""
    output = root / 'out'
    if scenario == 'publication_crash':
        import subprocess
        import sys
        from test_fix8_astra import CRASH_CHILD
        root.mkdir(parents=True, exist_ok=True)
        script = root / 'crash_child.py'
        script.write_text(CRASH_CHILD)
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
               'PYTHONPATH': os.pathsep.join([*filter(None, [os.environ.get('PYTHONPATH')]), str(repo / 'src'), str(repo / 'tests')])}

        def child(at, step):
            at.mkdir(parents=True, exist_ok=True)
            return subprocess.run([sys.executable, '-B', str(script), str(at), step, *map(str, _sources(at))],
                                  capture_output=True, text=True, timeout=300, env=env, cwd=at)
        reference = child(root / 'reference', 'reference')
        assert reference.returncode == 0, reference.stderr[-2000:]
        inventory = _visible(root / 'reference' / 'out')
        done = child(root, 'publication_partial')
        assert done.returncode == 92, done.stderr[-2000:]
        with open_readonly(output / 'corpus.sqlite3') as db:
            run = db.execute("SELECT run_id FROM tasks WHERE operation='export.compatibility_publish' ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        return output, run, inventory
    if scenario == 'failed_clean':
        from test_fix8_astra import _failed_clean
        inventory = _reference(root, scenario)
        with pytest.MonkeyPatch.context() as patch:
            output, path, run = _failed_clean(root, patch)
        return output, run, inventory
    run = Corpus(root / 'empty', output).run(_sources(root), stages=STAGES, settings=SETTINGS6)['run_id']
    export = output / 'export.jsonl'
    if scenario in ('successor_missing', 'successor_corrupt'):
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            _register_generation(writer, run, export, NEWER, output / '.glossapi/artifacts/fix12-newer/export.jsonl')
            if scenario == 'successor_corrupt':
                writer.record_uncommitted_write(run, export, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        inventory = _visible(output)  # the successor is the committed export
        if scenario == 'successor_missing':
            export.unlink()
        else:
            export.write_bytes(CONTENT)
        return output, run, inventory
    if scenario == 'two_copies':
        second = output / 'export-copy.jsonl'
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            row = next(r for r in writer.file_catalog().values() if r['uri'] == export.as_uri() and r['presence'] == 'present')
            second.write_bytes(export.read_bytes())
            producer_run = writer.connection.execute('SELECT t.run_id FROM attempts a JOIN tasks t USING(task_id) WHERE a.attempt_id=?',
                                                     (row['producer_attempt_id'],)).fetchone()[0]
            with writer.transaction():
                second_id = writer.artifact(producer_run, row['stage'], row['kind'], second.as_uri(), version_id=row['version_id'],
                                            producer_attempt_id=row['producer_attempt_id'], sha256=row['sha256'], bytes=row['bytes'],
                                            retention_class=row['retention_class'])
                # Fix 15 (ASTRA-FIXES2c10 H1): the second named copy is committed only by its own publication record.
                writer.snapshot('publication_binding', {'kind': 'publication_binding', 'product_uri': second.as_uri(), 'sha256': row['sha256'],
                                                        'producer_attempt_id': row['producer_attempt_id'], 'file_id': second_id})
            writer.barrier_context = {'run_id': run, 'debug': 'off'}
            writer.durable_barrier()
    inventory = _visible(output)
    if scenario in ('startup_restore', 'two_copies'):
        with LedgerWriter(output / 'corpus.sqlite3') as writer:
            for path in [export] + ([output / 'export-copy.jsonl'] if scenario == 'two_copies' else []):
                writer.record_uncommitted_write(run, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
                path.write_bytes(CONTENT)
    return output, run, inventory


def _fresh(scenario, tmp_path_factory):
    if scenario not in _TEMPLATES:
        base = tmp_path_factory.mktemp('fix12-' + scenario)
        work, snapshot = base / 'work', base / 'snapshot'
        output, run, inventory = _build(scenario, work)
        shutil.copytree(work, snapshot, symlinks=True)
        _TEMPLATES[scenario] = (work, snapshot, run, inventory)
    work, snapshot, run, inventory = _TEMPLATES[scenario]
    shutil.rmtree(work)
    shutil.copytree(snapshot, work, symlinks=True)
    return work, work / 'out', run, inventory


def _inject(monkeypatch, operation, fault, index, output, state):
    owner, name = OPERATIONS[operation]
    root = os.path.realpath(output)

    def touches(args):
        if operation in DESCRIPTOR_OPERATIONS:
            return True
        target = args[0] if args else None
        if isinstance(target, pathlib.PurePath) or isinstance(target, (str, bytes, os.PathLike)):
            try:
                return os.path.realpath(os.fspath(target)).startswith(root)
            except (TypeError, ValueError):
                return False
        return False

    def corrupt(path=None):
        # The bytes changed by a "mismatch" fault: the file the call names, else the scenario's recovery targets (never
        # an unrelated product: an arbitrary edit of a file recovery does not handle is FIX8-G2's disclosed boundary).
        candidates = [Path(path)] if path else [output / relative for relative in state.get('targets', ('export.jsonl',))]
        candidates += sorted(output.glob('.export.jsonl.*.tmp')) if not path else []
        for target in candidates:
            if target.is_file():
                data = target.read_bytes()
                target.write_bytes(bytes([data[0] ^ 1]) + data[1:] if data else b'x')
                return

    def named_file(args):
        first = args[0] if args else None
        if isinstance(first, (str, os.PathLike)) and not isinstance(first, bytes):
            candidate = Path(os.fspath(first))
            if candidate.is_file() and os.path.realpath(candidate).startswith(root) and '.glossapi' not in candidate.parts:
                return candidate  # a public product or temporary; private immutable sources are not this fault's subject
        return None

    def wrap(real):
        def hooked(*args, **kwargs):
            if not state['window'] or state.get('fired') or not touches(args):
                return real(*args, **kwargs)
            state['calls'] = state.get('calls', 0) + 1
            if state['calls'] != index:
                return real(*args, **kwargs)
            first = args[0] if args else None
            state['fired'] = f'{operation}#{index}({os.fspath(first) if isinstance(first, (str, os.PathLike)) else first!r})'[-160:]
            if fault == 'missing':
                raise FileNotFoundError(errno.ENOENT, 'injected ' + operation)
            if fault in ('eio', 'enospc', 'edquot'):
                raise OSError({'eio': errno.EIO, 'enospc': errno.ENOSPC, 'edquot': errno.EDQUOT}[fault], 'injected ' + operation)
            if fault == 'partial':
                real(*args, **kwargs)  # the effect happened, then the call reports failure
                raise OSError(errno.EIO, 'injected after the effect of ' + operation)
            if operation == 'file_hash':
                real(*args, **kwargs)
                return '0' * 64
            if operation == 'os.read':
                data = real(*args, **kwargs)
                return bytes([data[0] ^ 1]) + data[1:] if data else data
            if operation in ('os.replace', 'shutil.copyfile'):
                result = real(*args, **kwargs)
                corrupt(args[1])
                return result
            corrupt(named_file(args))
            return real(*args, **kwargs)
        return hooked

    if owner == 'os':
        hooked = wrap(getattr(os, name))
        for module in (W, ps, encoding):
            proxy = SimpleNamespace(**{key: getattr(os, key) for key in dir(os) if not key.startswith('__')})
            setattr(proxy, name, hooked)
            monkeypatch.setattr(module, 'os', proxy)
    elif owner == 'pathlib':
        monkeypatch.setattr(pathlib.Path, name, wrap(getattr(pathlib.Path, name)))
    elif owner == 'builtins':
        monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    elif owner == 'shutil':
        monkeypatch.setattr(shutil, name, wrap(getattr(shutil, name)))
    else:
        hooked = wrap(encoding.file_hash)
        monkeypatch.setattr(encoding, 'file_hash', hooked)
        monkeypatch.setattr(W, 'file_hash', hooked)
    for method in ('verify_excluded_copies', 'roll_back_uncommitted_writes'):
        real = getattr(LedgerWriter, method)

        def windowed(self, *args, real=real, **kwargs):
            state['window'] = True
            try:
                return real(self, *args, **kwargs)
            finally:
                state['window'] = False
        monkeypatch.setattr(LedgerWriter, method, windowed)


def _outcome(function, output, inventory):
    from glossapi.stages.execution import disk_full
    result, error = _refusal(function)
    if error is not None:
        code = 'disk_full' if disk_full(error) else getattr(error, 'code', None) or type(error).__name__
        return {'state': 'refused', 'code': code, 'reasons': sorted({entry['reason'] for entry in getattr(error, 'conditions', [])})}
    visible = _visible(output)
    differs = sorted(set(visible) ^ set(inventory) | {key for key in visible if key in inventory and visible[key] != inventory[key]})
    return {'state': result['state'], 'differs': differs, 'contradictions': _contradictions(output),
            'gate': (result.get('completion_gate') or {}).get('verdict')}


def _record(row):
    if os.environ.get('FIX12_FAULT_MATRIX'):
        with open(os.environ['FIX12_FAULT_MATRIX'], 'a') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


@pytest.mark.parametrize('operation,fault,scenario,index', CALL_INDEXES)
def test_fault_sweep(tmp_path_factory, monkeypatch, operation, fault, scenario, index):
    """Operation x fault x scenario x call index. The hooked public resume never completes with products that differ
    from the inventory frozen before the fault, or with contradictory present registrations; the unhooked retry
    completes matching it, or refuses typed (``unresolved_recovery``; ``post_close_edit`` when the fault left bytes
    nobody recorded)."""
    work, output, run, inventory = _fresh(scenario, tmp_path_factory)
    state = {'window': False, 'targets': {'failed_clean': ('download_results/download_results.parquet',),
                                          'two_copies': ('export.jsonl', 'export-copy.jsonl')}.get(scenario, ('export.jsonl',))}
    _inject(monkeypatch, operation, fault, index, output, state)
    first = _outcome(lambda: Corpus(work / 'empty', output).run(resume=run, retry_failed=True), output, inventory)
    monkeypatch.undo()
    retry = _outcome(lambda: Corpus(work / 'empty', output).run(resume=run, retry_failed=True), output, inventory)
    row = {'kind': 'public', 'operation': operation, 'fault': fault, 'scenario': scenario, 'index': index,
           'fired': state.get('fired'), 'first': first, 'retry': retry}
    _record(row)
    if first['state'] == 'completed':
        assert not first['differs'] and not first['contradictions'], row
        assert first['gate'] == 'pass', row  # fix 13: no completion without a passing completion gate
    assert first['state'] != 'refused' or first['code'] in ('unresolved_recovery', 'post_close_edit', 'disk_full'), row
    assert (retry['state'] == 'completed' and not retry['differs'] and not retry['contradictions']) or \
        (retry['state'] == 'refused' and retry['code'] in ('unresolved_recovery', 'post_close_edit')), row


ROLLBACK_SCENARIOS = ('restore', 'remove', 'temporary')


def _rollback_state(root, scenario):
    """A direct rollback state: (writer, failed run, observation run, product path, target sha or None, temporary)."""
    writer = LedgerWriter(root / 'corpus.sqlite3')
    with writer.transaction():
        failed = writer.run({}, {}, {}, code_revision='fix12-failed')
    if scenario == 'restore':
        run, source, path, source_id, copy_id = published_copy(writer, root)
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=True)
        path.write_bytes(CONTENT)
        return writer, failed, run, path, _sha(source), None
    path = root / 'product.jsonl'
    if scenario == 'remove':
        path.write_bytes(CONTENT)
        writer.record_uncommitted_write(failed, path, _sha(CONTENT), len(CONTENT), wrapper_stage='export', existed=False)
        return writer, failed, failed, path, None, None
    source, temporary = root / 'source.jsonl', root / '.product.jsonl.partial.tmp'
    source.write_bytes(CONTENT)
    temporary.write_bytes(CONTENT[:10])
    writer.record_uncommitted_writes(failed, [(path, _sha(CONTENT), len(CONTENT), False, {'temporary': temporary, 'source': source})],
                                     wrapper_stage='export')
    return writer, failed, failed, path, None, temporary


@pytest.mark.parametrize('scenario', ROLLBACK_SCENARIOS)
@pytest.mark.parametrize('fault', FAULTS)
@pytest.mark.parametrize('operation', list(OPERATIONS))
def test_rollback_fault_matrix(tmp_path, monkeypatch, operation, fault, scenario):
    """The rollback's own operations x faults, called directly. A clean return never leaves the failed attempt's
    recorded bytes at the path and never installs anything but the target; refusals are typed. The unhooked retry
    reaches the target (or lists/raises typed)."""
    from glossapi.stages.execution import disk_full
    writer, failed, observation, path, target, temporary = _rollback_state(tmp_path, scenario)
    state = {'window': False}
    try:
        _inject(monkeypatch, operation, fault, 1, tmp_path, state)

        def attempt():
            changed, error = _refusal(lambda: writer.roll_back_uncommitted_writes([failed], tmp_path, observation_run=observation))
            current = _sha(path) if path.is_file() else None
            if error is not None:
                code = 'disk_full' if disk_full(error) else getattr(error, 'code', None) or type(error).__name__
                return {'state': 'raised', 'code': code, 'current': current}
            listed = sorted(entry['reason'] for entry in writer.rollback_unresolved)
            acted = bool(changed) and str(ps.resolve_metadata_target(path)) in changed
            return {'state': 'listed' if listed else 'clean', 'reasons': listed, 'current': current, 'acted': acted}
        first = attempt()
        monkeypatch.undo()
        retry = attempt()
    finally:
        monkeypatch.undo()
        writer.close()
    row = {'kind': 'rollback', 'operation': operation, 'fault': fault, 'scenario': scenario, 'index': 1,
           'fired': state.get('fired'), 'first': first, 'retry': retry}
    _record(row)
    recorded = _sha(CONTENT)
    for outcome in (first, retry):
        if outcome['state'] == 'clean':
            assert outcome['current'] != recorded or recorded == target, row  # the failed bytes never stay after a clean return
            if outcome['acted']:
                # Fix 13 (M1): a clean reconciliation that acted leaves exactly the target (or the intended absence).
                assert outcome['current'] == target, row
            else:
                # Foreign bytes are tolerated only for a verified no-action outcome (bytes nobody recorded, left alone).
                assert outcome['current'] in (target, None) or outcome['current'] != recorded, row
        if outcome['state'] == 'raised':
            assert outcome['code'] in ('unresolved_recovery', 'disk_full'), row
    assert retry['state'] in ('clean', 'listed') or retry['code'] in ('unresolved_recovery',), row
    if retry['state'] == 'clean' and scenario != 'temporary':
        assert retry['current'] == target or (not retry['acted'] and retry['current'] != recorded and fault == 'mismatch'), row
