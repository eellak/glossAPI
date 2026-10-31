"""Grouped durability barriers: targeted per-path flushes, never syncfs (K4(c) decision 1)."""
import hashlib
import os

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.ledger import writer as writer_module
from glossapi.ledger.writer import SyncBatch, integrity
from glossapi.stages.execution import sha256_file
from glossapi.stages.retention import RetentionController, recover_private_releases, release_private_scratch


def private_producers(writer, root, layout):
    """layout: {stage: {directory: file_count}} of finished private outputs."""
    rows = []
    with writer.transaction():
        run = writer.run({}, {}, {}, code_revision='batched-durability')
    for stage, directories in layout.items():
        for directory, count in directories.items():
            paths = []
            for index in range(count):
                path = root/'.glossapi/artifacts'/stage/directory/f'diagnostic-{index}.txt'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f'private {stage} {directory} {index}')
                paths.append(path)
            with writer.transaction():
                scope = writer.snapshot('inputs', {'files':[], 'directory':directory, 'stage':stage})
                task = writer.task(run, stage, stage+'.private', {}, scope_key='corpus:'+scope)
                attempt = writer.attempt(task, worker_id='test', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
                writer.finish(run, task, attempt, status='ok', reason_code=stage+'_completed', outputs=[{
                    'kind':'attempt_diagnostic', 'role':path.name, 'uri':path.as_uri(), 'sha256':sha256_file(path), 'bytes':path.stat().st_size}
                    for path in paths])
            rows.extend(writer.connection.execute('SELECT * FROM current_file_state WHERE uri=?', (path.as_uri(),)).fetchone() for path in paths)
    return run, rows


def count_fsyncs(monkeypatch):
    calls = []
    original = os.fsync
    def recorded(descriptor):
        calls.append(os.fstat(descriptor).st_ino)
        return original(descriptor)
    monkeypatch.setattr(os, 'fsync', recorded)
    return calls


def test_sync_batch_fallback_flushes_each_path_once(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_module, '_SYNCFS', None)
    calls = count_fsyncs(monkeypatch)
    directory = tmp_path/'d'; directory.mkdir()
    file = directory/'f'; file.write_text('x')
    with SyncBatch() as batch:
        for _ in range(3):
            batch.add_path(directory); batch.add_path(file)
        batch.flush()
    # The file precedes its directory; each is flushed exactly once.
    assert calls == [file.stat().st_ino, directory.stat().st_ino]


def test_sync_batch_never_uses_syncfs_and_reports_errors(tmp_path, monkeypatch):
    devices = []
    monkeypatch.setattr(writer_module, '_SYNCFS', lambda descriptor: devices.append(os.fstat(descriptor).st_dev) or 0)
    calls = count_fsyncs(monkeypatch)
    for index in range(5):
        (tmp_path/str(index)).mkdir()
    with SyncBatch() as batch:
        for index in range(5):
            batch.add_path(tmp_path/str(index))
        batch.flush()
    # Exactly the registered directories, each once; the filesystem is never flushed as a whole.
    assert devices == [] and sorted(calls) == sorted((tmp_path/str(index)).stat().st_ino for index in range(5))
    def failing(descriptor):
        raise OSError(5, 'Input/output error')
    monkeypatch.setattr(os, 'fsync', failing)
    with SyncBatch() as batch:
        batch.add_path(tmp_path)
        with pytest.raises(OSError):
            batch.flush()


@pytest.mark.parametrize('syncfs', [False, True])
def test_durable_barrier_flushes_new_products_once(tmp_path, monkeypatch, syncfs):
    flushed = []
    monkeypatch.setattr(writer_module, '_SYNCFS', (lambda descriptor: flushed.append(descriptor) or 0) if syncfs else None)
    calls = count_fsyncs(monkeypatch)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, rows = private_producers(writer, tmp_path, {'extract': {'a': 3, 'b': 2}})
        writer.durable_barrier()
        # Never syncfs: five files plus their six distinct parents up to the corpus root, each once.
        assert flushed == []
        assert len(calls) == len(set(calls)) == 5 + 6
        flushed.clear(); calls.clear()
        writer.durable_barrier()
        assert flushed == [] and calls == []
        integrity(writer.connection)


@pytest.mark.parametrize('syncfs', [False, True])
def test_private_release_makes_no_private_scratch_durable(tmp_path, monkeypatch, syncfs):
    """Decision 2: released private scratch is never flushed, neither its bytes nor its unlink; the
    release intents still commit (FULL) in one barrier before any unlink."""
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, rows = private_producers(writer, tmp_path, {'extract': {'a': 40, 'b': 30}, 'clean': {'c': 5}})
        flushed = []
        monkeypatch.setattr(writer_module, '_SYNCFS', (lambda descriptor: flushed.append(descriptor) or 0) if syncfs else None)
        calls = count_fsyncs(monkeypatch)
        barriers = []
        original_barrier = writer.durable_barrier
        def barrier(unlinking=(), **options):
            barriers.append((writer.connection.execute(
                "SELECT COUNT(*) FROM events WHERE reason_code='delete_planned'").fetchone()[0], len(set(unlinking)), options.get('private')))
            original_barrier(unlinking, **options)
        monkeypatch.setattr(writer, 'durable_barrier', barrier)
        unlink = os.unlink
        def checked_unlink(*args, **kwargs):
            assert barriers, 'unlink before the release barrier'
            return unlink(*args, **kwargs)
        monkeypatch.setattr(os, 'unlink', checked_unlink)
        release_private_scratch(RetentionController(writer, tmp_path, run), rows)
        # One barrier covers both stage summaries and all 75 files before any unlink.
        assert barriers == [(2, 75, True)]
        # Nothing private was flushed: no file, no directory, no filesystem.
        assert calls == [] and flushed == []
        assert not any(path.exists() for path in (tmp_path/'.glossapi/artifacts').rglob('*.txt'))
        assert {row[0] for row in writer.connection.execute('SELECT presence FROM current_file_state')} == {'deleted'}
        barriers.clear()
        recover_private_releases(RetentionController(writer, tmp_path, run))
        assert barriers == []
        integrity(writer.connection)


def test_delete_many_uses_one_barrier_per_frontier(tmp_path, monkeypatch):
    from test_ledger_retention import consumer
    monkeypatch.setattr(writer_module, '_SYNCFS', None)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='delete-many-barrier')
        files = []
        for index in range(130):
            path = tmp_path/'products'/str(index % 2)/f'{index}.md'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(f'exact {index}')
            with writer.transaction():
                document = writer.document('file:///original-'+str(index)); version = writer.version(document)
                file = writer.artifact(run, 'extract', 'raw_markdown', path.as_uri(), version_id=version,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            task, _ = consumer(writer, run, version, file)
            files.append((file, task))
        controller = RetentionController(writer, tmp_path, run)
        with writer.batch():
            for file, task in files:
                controller.declare(file, [task])
        writer.durable_barrier()
        barriers = []
        original_barrier = writer.durable_barrier
        monkeypatch.setattr(writer, 'durable_barrier', lambda *args, **options: (barriers.append(1), original_barrier(*args, **options)))
        calls = count_fsyncs(monkeypatch)
        controller.delete_many([file for file, _ in files])
        assert barriers == [1]
        # Two 128-file batches each flush their two changed parents once.
        assert len(calls) == 4
        assert not any((tmp_path/'products').rglob('*.md'))
        integrity(writer.connection)
