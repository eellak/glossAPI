"""K4(c) durability decisions 1-3: targeted barriers, no durable private scratch, no duplicate managed copies.

A barrier makes durable exactly what the pending unlink (or the run close) depends on, by per-file and
per-directory fsync; it never flushes the whole filesystem and never flushes released private scratch.
"""
import json
import os
from pathlib import Path

import pytest

from glossapi.ledger import LedgerWriter
from glossapi.ledger import writer as writer_module
from glossapi.ledger.writer import integrity
from glossapi.stages.execution import link_published, links_released_copies, sha256_file
from glossapi.stages.retention import RetentionController, release_private_scratch


def flushed_inodes(monkeypatch):
    calls = []
    original = os.fsync
    def recorded(descriptor):
        calls.append(os.fstat(descriptor).st_ino)
        return original(descriptor)
    monkeypatch.setattr(os, 'fsync', recorded)
    monkeypatch.setattr(writer_module, '_SYNCFS', lambda descriptor: pytest.fail('syncfs must never be called'))
    return calls


def producer(writer, run, stage, files, *, version=None, inputs=()):
    """One finished attempt producing ``files``: [(path, kind, retention_class)]; returns (attempt, {path: file_id})."""
    with writer.transaction():
        scope = writer.snapshot('inputs', {'files': [], 'stage': stage, 'n': len(files), 'first': str(files[0][0])})
        task = writer.task(run, stage, stage+'.fixture:'+scope[:8], {}, version_id=version, scope_key=None if version else 'corpus:'+scope)
        attempt = writer.attempt(task, worker_id='test', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        for file_id, digest in inputs:
            writer.input(attempt, file_id, 'input:'+file_id, digest)
        writer.finish(run, task, attempt, status='ok', reason_code=stage+'_completed', outputs=[
            {'kind': kind, 'role': str(path), 'uri': path.as_uri(), 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
             'retention_class': retention} for path, kind, retention in files])
    ids = {path: writer.connection.execute('SELECT file_id FROM files WHERE uri=?', (path.as_uri(),)).fetchone()[0] for path, _, _ in files}
    return attempt, ids


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text); return path


def test_policy_delete_barrier_flushes_only_copies_and_reader_outputs(tmp_path, monkeypatch):
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='targeted')
            document = writer.document('file:///source'); version = writer.version(document)
        backing = write(root/'.glossapi/artifacts/v/extract/a/doc.md', 'raw text')
        attempt, ids = producer(writer, run, 'extract', [(backing, 'raw_markdown', 'intermediate')], version=version)
        # A user-visible copy of the same producer bytes, a reader's output, and an unrelated dirty product.
        copy = write(root/'markdown/doc.md', 'raw text')
        with writer.transaction():
            writer.artifact(run, 'extract', 'raw_markdown', copy.as_uri(), version_id=version, producer_attempt_id=attempt,
                            sha256=sha256_file(copy), bytes=copy.stat().st_size)
        output = write(root/'.glossapi/artifacts/v/clean/b/doc.md', 'clean text')
        reader, _ = producer(writer, run, 'clean', [(output, 'clean_markdown', 'intermediate')], version=version,
                             inputs=[(ids[backing], sha256_file(backing))])
        unrelated = write(root/'.glossapi/artifacts/v/sections/c/doc.parquet', 'unrelated')
        producer(writer, run, 'sections', [(unrelated, 'sections_parquet', 'final')], version=version)
        calls = flushed_inodes(monkeypatch)
        writer.durable_barrier([ids[backing]])
        files = {copy.stat().st_ino, output.stat().st_ino}
        assert files <= set(calls)
        assert backing.stat().st_ino not in calls and unrelated.stat().st_ino not in calls
        # Every flushed directory is a parent of a covered file; each path once.
        parents = {p.stat().st_ino for f in (copy, output) for p in f.parents if p.is_relative_to(root)}
        assert set(calls) == files | parents and len(calls) == len(set(calls))
        calls.clear()
        # A private release flushes the user-visible copy but no private reader output.
        writer.durable_barrier([ids[backing]], private=True)
        assert calls == []  # the copy is already durable; the private output is never flushed
        # The run close flushes what remains, exactly once.
        writer.durable_barrier()
        assert unrelated.stat().st_ino in calls and output.stat().st_ino not in calls
        integrity(writer.connection)


def test_run_close_skips_a_copy_whose_retained_source_is_durable(tmp_path, monkeypatch):
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='copies')
        source = write(root/'.glossapi/artifacts/corpus/export/a/products/export.jsonl', '{"x":1}\n')
        attempt, ids = producer(writer, run, 'export', [(source, 'document_jsonl', 'final')])
        copy = write(root/'export.jsonl', '{"x":1}\n')
        with writer.transaction():
            writer.artifact(run, 'export', 'document_jsonl', copy.as_uri(), producer_attempt_id=attempt,
                            sha256=sha256_file(copy), bytes=copy.stat().st_size, retention_class='final')
        calls = flushed_inodes(monkeypatch)
        writer.durable_barrier()
        assert source.stat().st_ino in calls and copy.stat().st_ino not in calls


@pytest.mark.parametrize('debug', ['off', 'on'])
def test_barriers_are_recorded_per_barrier_with_debug_on_and_as_one_summary_with_debug_off(tmp_path, debug):
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='recorded')
        writer.barrier_context = {'run_id': run, 'debug': debug}
        product = write(root/'.glossapi/artifacts/corpus/sections/a/s.parquet', 'p')
        producer(writer, run, 'sections', [(product, 'sections_parquet', 'final')])
        scratch = write(root/'.glossapi/artifacts/corpus/extract/b/diag.txt', 'd')
        _, ids = producer(writer, run, 'extract', [(scratch, 'attempt_diagnostic', 'intermediate')])
        writer.durable_barrier([ids[scratch]], private=True)
    with LedgerWriter(root/'corpus.sqlite3') as reader:
        rows = [(message, json.loads(details)) for message, details in reader.connection.execute(
            "SELECT message,details_json FROM events WHERE reason_code='durability_barrier' ORDER BY event_id")]
    if debug == 'on':
        assert [message for message, _ in rows] == ['barrier', 'barrier']
        private, close = rows[0][1], rows[1][1]
        assert private['reason'] == 'private_release' and private['file_ids'] == [] and private['unlink_count'] == 1
        assert close['reason'] == 'run_close' and close['file_count'] == 2 and len(close['file_ids']) == 2
        assert close['byte_count'] == product.stat().st_size + scratch.stat().st_size
    else:
        assert [message for message, _ in rows] == ['barrier_summary']
        summary = rows[0][1]
        assert summary['barrier_count'] == 2 and summary['file_count'] == 2
        assert summary['by_reason']['private_release'] == {'barrier_count': 1, 'byte_count': 0, 'file_count': 0}


def test_released_private_source_may_share_its_inode_only_with_its_registered_copy(tmp_path):
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='links')
            document = writer.document('file:///source'); version = writer.version(document)
        backing = write(root/'.glossapi/artifacts/v/extract/a/doc.md', 'linked text')
        attempt, ids = producer(writer, run, 'extract', [(backing, 'raw_markdown', 'intermediate')], version=version)
        copy = root/'markdown/doc.md'; copy.parent.mkdir()
        assert link_published(backing, copy, sha256_file(backing), backing.stat().st_size)
        assert copy.stat().st_ino == backing.stat().st_ino and copy.read_text() == 'linked text'
        with writer.transaction():
            writer.artifact(run, 'extract', 'raw_markdown', copy.as_uri(), version_id=version, producer_attempt_id=attempt,
                            sha256=sha256_file(copy), bytes=copy.stat().st_size)
        rows = [writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?', (ids[backing],)).fetchone()]
        release_private_scratch(RetentionController(writer, root, run), rows)
        assert not backing.exists() and copy.read_text() == 'linked text' and copy.stat().st_nlink == 1
        # An unregistered extra link stays ambiguous: the private file is not released.
        other = write(root/'.glossapi/artifacts/v/extract/c/doc2.md', 'other text')
        _, more = producer(writer, run, 'extract', [(other, 'raw_markdown', 'intermediate')], version=version)
        os.link(other, root/'stray.md')
        rows = [writer.connection.execute('SELECT * FROM current_file_state WHERE file_id=?', (more[other],)).fetchone()]
        with pytest.raises(ValueError, match='ambiguous'):
            release_private_scratch(RetentionController(writer, root, run), rows)
        assert other.exists()


def test_links_only_for_debug_off_private_intermediates():
    class Journal:
        def __init__(self, debug): self.retention_settings = {'debug': debug}
    assert links_released_copies(Journal('off'), 'intermediate', 'raw_markdown')
    assert not links_released_copies(Journal('on'), 'intermediate', 'raw_markdown')
    assert not links_released_copies(Journal('off'), 'final', 'document_jsonl')
    assert not links_released_copies(Journal('off'), 'intermediate', 'worker_receipt')
    assert not links_released_copies(None, 'intermediate', 'raw_markdown')


def test_republishing_an_existing_link_leaves_no_extra_name(tmp_path):
    """A kill after a link publication and a resumed publication of the same bytes must not leave a temporary link
    (rename() between two links of one inode is a no-op), which would make the private source ambiguous."""
    source = write(tmp_path/'.glossapi/artifacts/v/extract/a/doc.md', 'same bytes')
    destination = tmp_path/'markdown/doc.md'; destination.parent.mkdir()
    digest, size = sha256_file(source), source.stat().st_size
    assert link_published(source, destination, digest, size)
    assert link_published(source, destination, digest, size)
    assert sorted(path.name for path in destination.parent.iterdir()) == ['doc.md']
    assert source.stat().st_nlink == 2


def test_worker_hashes_the_frozen_source_snapshot_once(tmp_path, monkeypatch):
    """Hash once per byte stream: freeze_inputs hashes the snapshot while writing it; run_prepared reuses that hash
    inside the same producer scope, and seal_output_hashes reuses it too. Only a changed snapshot is re-read."""
    from test_ledger_stages import request, simple_pdf
    from glossapi.ledger import encoding
    from glossapi.stages.execution import ReceiptCollector, producer_hash_scope, seal_output_hashes
    from glossapi.stages.extract import ExtractStage
    source = simple_pdf(tmp_path/'source.pdf')
    task = request(tmp_path, source, repair_fonts=False)
    reads = []
    original = encoding.file_hash
    monkeypatch.setattr(encoding, 'file_hash', lambda path, *a, **k: (reads.append(Path(path).name), original(path, *a, **k))[1])
    with producer_hash_scope():
        result = seal_output_hashes(ExtractStage().run_document(task, reporter=ReceiptCollector()), task.work_directory)
    assert result.status == 'ok'
    assert reads.count('source.pdf') == 0  # copied with its hash (freeze), reused by run_prepared and the seal
    # A snapshot changed after freezing is still caught by the re-read.
    from glossapi.stages.execution import remember_written_hash
    task2 = request(tmp_path/'b', source, repair_fonts=False)
    frozen = task2.work_directory/'inputs/0/source.pdf'
    with producer_hash_scope():
        from glossapi.stages.execution import freeze_inputs
        prepared = freeze_inputs(task2)
        frozen.write_bytes(frozen.read_bytes()+b'%tampered\n')
        result = ExtractStage().run_prepared(prepared, extractor=__import__('test_ledger_stages').prepared_extractor(), reporter=ReceiptCollector())
    assert result.status == 'failed' and result.reason_code == 'input_hash_changed'


def test_run_close_barrier_keeps_open_descriptors_bounded(tmp_path, monkeypatch):
    """debug=on keeps thousands of files: the barrier must not open them all at once (EMFILE on the cohort run)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    root = tmp_path
    with LedgerWriter(root/'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run({}, {}, {}, code_revision='bounded')
        files = [write(root/'.glossapi/artifacts/corpus/extract/a'/f'{index:04}.txt', str(index)) for index in range(300)]
        producer(writer, run, 'extract', [(path, 'attempt_diagnostic', 'intermediate') for path in files])
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
        try:
            writer.durable_barrier()
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
        assert len(writer._durable_files) == 300
