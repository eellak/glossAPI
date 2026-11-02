"""CP-K4a fix round 1 (reviewer 1 findings F1–F7 on bb411d0). Every test here fails on bb411d0.

Acquisition is replaced by deterministic in-memory PDF bodies (no sockets); scheduler, extraction, cleanup, metadata
merge, retention and export are the real code. Ported from the reviewer's probes (review_scratch/CP-K4a).
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from glossapi import Corpus
from glossapi.ledger.writer import LedgerWriter
from test_ledger_fixset_lg import pdf


@pytest.fixture(autouse=True)
def private_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


@pytest.fixture
def transport(monkeypatch):
    """Replace only acquisition transport/process with deterministic in-memory PDF bodies; no sockets."""
    from glossapi.stages import acquisition_pool
    from glossapi.stages.contracts import OutputArtifact
    from glossapi.stages.execution import seal_output_hashes, write_text_hashed
    bodies = {}

    class Kernel:
        async def download_file(self, row_index, url, semaphore, rate_limiter, retry_count=0, filename_base=None, referer=None):
            from glossapi.ledger.journal import observe_acquisition
            target = self.downloads_dir/(filename_base+'.pdf')
            target.write_bytes(bodies[url])
            observe_acquisition(url, target, source_observable=True)
            return True, target.name, 'pdf', '', 0

    class InlinePool:
        def __init__(self, capacity):
            self.capacity = capacity
            self.queue = []

        @property
        def active(self):
            return len(self.queue)

        def submit(self, stage, request, reporter=None):
            import asyncio, hashlib
            stage.downloader = Kernel()
            stage.kernel = stage.downloader.download_file
            stage.rate_limiter = None
            result = asyncio.run(stage.run_document_async(request, reporter=reporter))
            result = seal_output_hashes(result, request.work_directory)
            encoded = json.dumps(asdict(result), ensure_ascii=False)
            write_text_hashed(request.work_directory/'worker-result.json', encoded)
            receipt = OutputArtifact('worker_receipt', 'worker-result.json', None, 'evidence', False,
                                     hashlib.sha256(encoded.encode()).hexdigest(), len(encoded.encode()))
            self.queue.append((request, replace(result, outputs=(*result.outputs, receipt))))

        def poll(self):
            completed, self.queue = self.queue, []
            return completed

        def close(self):
            self.queue.clear()

    monkeypatch.setattr(acquisition_pool, 'AsyncAcquisitionPool', InlinePool)
    return bodies


SETTINGS = {'download': {'rate_limit': 0}, 'extract': {'repair_fonts': False}}
STAGES = ('extract', 'clean', 'export')


def records(root):
    return [json.loads(line) for line in (root/'export.jsonl').read_text().splitlines()]


# ---------------------------------------------------------------- F1 provenance ---

def test_f1_remote_then_local_same_basename_does_not_inherit_removed_url(tmp_path, transport):
    root = tmp_path/'out'
    url = 'http://fixture.invalid/remote.pdf'
    transport[url] = pdf(tmp_path/'remote.pdf', 'REMOTE document text').read_bytes()
    assert Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=SETTINGS)['state'] == 'completed'
    base = records(root)[0]['filename']
    source = pdf(tmp_path/'local'/(base+'.pdf'), 'LOCAL different document text')
    second = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    observed = records(root)
    assert second['state'] == 'completed' and len(observed) == 1
    assert 'LOCAL different' in observed[0]['document']
    assert observed[0]['url'] is None, 'new local text exported under a removed remote source URL'
    assert observed[0]['filename'] != base  # the remote source's name stays reserved for it


def test_f1_local_then_remote_same_basename_does_not_inherit_local_row(tmp_path, transport):
    root = tmp_path/'out'
    local = pdf(tmp_path/'local'/'AAA_000.pdf', 'LOCAL first document text')
    assert Corpus(tmp_path/'empty', root).run([local], stages=STAGES, settings=SETTINGS)['state'] == 'completed'
    assert records(root)[0]['filename'] == 'AAA_000'
    url = 'http://fixture.invalid/AAA_000.pdf'
    transport[url] = pdf(tmp_path/'remote.pdf', 'REMOTE later document text').read_bytes()
    second = Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=SETTINGS)
    observed = records(root)
    assert second['state'] == 'completed' and len(observed) == 1
    assert observed[0]['url'] == url and 'REMOTE later' in observed[0]['document']
    assert observed[0]['filename'] != 'AAA_000'


def test_f1_reintroduced_sources_keep_their_own_names_and_rows(tmp_path, transport):
    """remote A → local (same basename) → remote A again → the local again: every run exports only its own source."""
    root = tmp_path/'out'
    url = 'http://fixture.invalid/doc.pdf'
    transport[url] = pdf(tmp_path/'remote.pdf', 'REMOTE document text').read_bytes()
    Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=SETTINGS)
    remote_name = records(root)[0]['filename']
    local = pdf(tmp_path/'local'/(remote_name+'.pdf'), 'LOCAL document text')
    names = {}
    for label, source in (('local', local), ('remote', url), ('local-again', local), ('remote-again', url)):
        result = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
        observed = records(root)
        assert result['state'] == 'completed' and len(observed) == 1, label
        row = observed[0]
        names[label] = row['filename']
        if label.startswith('local'):
            assert row['url'] is None and 'LOCAL' in row['document'], label
        else:
            assert row['url'] == url and 'REMOTE' in row['document'], label
    assert names['remote'] == names['remote-again'] == remote_name
    assert names['local'] == names['local-again'] != remote_name


def test_f1_export_refuses_a_planned_row_that_names_another_source(tmp_path):
    from glossapi.stages.export import _borrowed_metadata_rows
    import pandas as pd
    path = tmp_path/'metadata.parquet'
    pd.DataFrame({'filename': ['AAA_000.pdf', 'BBB_000.pdf', 'CCC_000.pdf', 'DDD_000.pdf', 'EEE_000.pdf'],
                  'url': ['http://x/removed.pdf', None, 'http://x/c.pdf', 'https://origin.example/d.pdf', 'http://x/other.pdf'],
                  'resource_identity': ['url:v1:http://x/removed.pdf', None, 'url:v1:http://x/c.pdf', None, None]}).to_parquet(path, index=False)
    planned = {'AAA_000': dict(url=None, foreign_urls=['http://x/removed.pdf'], foreign_keys=['url:v1:http://x/removed.pdf']),
               'BBB_000': dict(url=None, foreign_urls=[], foreign_keys=[]),
               'CCC_000': dict(url='http://x/c.pdf', foreign_urls=[], foreign_keys=[]),
               'DDD_000': dict(url=None, foreign_urls=[], foreign_keys=[]),
               'EEE_000': dict(url='http://x/e.pdf', foreign_urls=['http://x/other.pdf'], foreign_keys=['url:v1:http://x/other.pdf'])}
    wrong = _borrowed_metadata_rows(path, planned)
    # AAA: a local input on a removed download's row; EEE: a download on another planned URL's row. DDD: a local input
    # whose caller metadata names its origin URL (no frozen plan gave the name to another source) keeps it, as before.
    assert [w['compatibility_filename'] for w in wrong] == ['AAA_000.pdf', 'EEE_000.pdf']
    assert _borrowed_metadata_rows(path, None) == []


def test_f1_local_run_over_legacy_download_rows_keeps_their_url(tmp_path):
    """A corpus whose download_results came from a legacy download (url, download_success, no frozen plan claim) and a
    run over those local files: the rows describe these very files, so the export keeps their URL (kill-matrix seed)."""
    import pandas as pd
    from glossapi.parquet_schema import ParquetSchema
    root = tmp_path/'out'
    source = pdf(tmp_path/'data'/'doc.pdf', 'LEGACY downloaded document text')
    (root/'download_results').mkdir(parents=True)
    ParquetSchema().write_metadata_parquet(pd.DataFrame([dict(filename='doc.pdf', filename_base='doc', url='https://origin.example/doc.pdf',
        download_success=True, file_ext='pdf', processing_stage='', extract_status='')]), root/'download_results/download_results.parquet')
    result = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    assert result['state'] == 'completed'
    assert records(root)[0]['url'] == 'https://origin.example/doc.pdf'




# ------------------------------------------------------- F2 extreme retirement ---

def _late_reader_plan(root, monkeypatch, reader_document='same'):
    """download -> extract (role source_payload) plus a later task reading the same download under role original_bytes."""
    from glossapi.stages.scheduler import StreamingScheduler, ScheduledOperation
    from glossapi.stages.contracts import DocumentSnapshot, StageSpecification
    from glossapi.stages.execution import sha256_file
    from glossapi.stages import disk_budget
    from test_ledger_retention import settings
    monkeypatch.setattr(disk_budget, 'require_disk_guard', lambda *a: None)
    source = root/'owned.pdf'
    source.write_bytes(b'owned PDF bytes')
    digest = sha256_file(source)
    writer = LedgerWriter(root/'corpus.sqlite3')
    with writer.transaction():
        doc = writer.document('url:v1:http://fixture.invalid/a.pdf')
        version = writer.version(doc, source_sha256=digest, stored_sha256=digest)
        other = writer.document('url:v1:http://fixture.invalid/b.pdf')
        other_version = writer.version(other, source_sha256='0'*64, stored_sha256='0'*64)
    document = DocumentSnapshot(doc, version, 'http://fixture.invalid/a.pdf')
    late_document = document if reader_document == 'same' else DocumentSnapshot(other, other_version, 'http://fixture.invalid/b.pdf')
    operations = [
        ScheduledOperation('download', 'download', document, StageSpecification('download.acquire', required_output_kinds=('download_payload',), settings={'filename_base': 'AAA_000'})),
        ScheduledOperation('extract', 'extract', document, StageSpecification('extract.document'), inputs=(('download', 'download_payload', 'source_payload'),)),
        ScheduledOperation('late', 'clean', late_document, StageSpecification('clean.custom_pdf_reader'), inputs=(('download', 'download_payload', 'original_bytes'),), after=('extract',)),
    ]
    engine = StreamingScheduler(writer, root)
    run = engine.freeze(operations, settings=settings(memory_saving='extreme'))
    _, tasks, _ = engine._load(run)
    with writer.transaction():
        down = writer.attempt(tasks['download'], worker_id='synthetic', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        writer.finish(run, tasks['download'], down, status='ok', reason_code='download_completed', outputs=[{
            'kind': 'download_payload', 'uri': source.as_uri(), 'sha256': digest, 'bytes': source.stat().st_size, 'retention_class': 'initial'}])
        file = writer.connection.execute('SELECT file_id FROM files WHERE producer_attempt_id=?', (down,)).fetchone()[0]
        extract = writer.attempt(tasks['extract'], worker_id='synthetic', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        writer.input(extract, file, 'source_payload', digest)
        output = root/'text.md'
        output.write_text('derived text')
        writer.finish(run, tasks['extract'], extract, status='ok', reason_code='extract_completed', outputs=[{
            'kind': 'raw_markdown', 'uri': output.as_uri(), 'sha256': sha256_file(output), 'bytes': output.stat().st_size, 'retention_class': 'intermediate'}])
    return writer, run, tasks, source, file, digest, settings(memory_saving='extreme')


@pytest.mark.parametrize('reader_document', ['same', 'other'])
def test_f2_extreme_waits_for_a_reader_with_another_role_spelling(tmp_path, monkeypatch, reader_document):
    from glossapi.stages.retention import retire_downloaded_sources
    writer, run, tasks, source, file, digest, settings = _late_reader_plan(tmp_path, monkeypatch, reader_document)
    try:
        retire_downloaded_sources(writer, tmp_path, run, settings)
        assert source.exists(), 'download deleted before a frozen reader (role original_bytes) ran'
        assert writer.task_state(tasks['late'])['status'] == 'planned'
        # The late reader consumes the exact bytes and succeeds: now the download goes.
        with writer.transaction():
            late = writer.attempt(tasks['late'], worker_id='synthetic', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            writer.input(late, file, 'original_bytes', digest)
            out = tmp_path/'late.md'
            out.write_text('late text')
            from glossapi.stages.execution import sha256_file
            writer.finish(run, tasks['late'], late, status='ok', reason_code='clean_completed', outputs=[{
                'kind': 'clean_markdown', 'uri': out.as_uri(), 'sha256': sha256_file(out), 'bytes': out.stat().st_size, 'retention_class': 'intermediate'}])
        writer.__dict__.pop('_streaming_retirement', None)
        retire_downloaded_sources(writer, tmp_path, run, settings)
        assert not source.exists()
    finally:
        writer.close()


def test_f2_reader_without_exact_hash_consumption_blocks_deletion(tmp_path, monkeypatch):
    from glossapi.stages.retention import retire_downloaded_sources
    writer, run, tasks, source, file, digest, settings = _late_reader_plan(tmp_path, monkeypatch)
    try:
        with writer.transaction():
            late = writer.attempt(tasks['late'], worker_id='synthetic', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
            out = tmp_path/'late.md'
            out.write_text('late text')
            from glossapi.stages.execution import sha256_file
            writer.finish(run, tasks['late'], late, status='ok', reason_code='clean_completed', outputs=[{
                'kind': 'clean_markdown', 'uri': out.as_uri(), 'sha256': sha256_file(out), 'bytes': out.stat().st_size, 'retention_class': 'intermediate'}])
        retire_downloaded_sources(writer, tmp_path, run, settings)
        assert source.exists(), 'a reader that never bound the exact bytes cannot authorize deletion'
    finally:
        writer.close()


# ------------------------------------------------- F3/F4 interruption epoch + why ---

def _attempt_start(writer, run, task, operations):
    with writer.transaction():
        attempt = writer.attempt(task, worker_id='simulated', device_id='cpu', fingerprint={}, inputs={}, settings={}, tools={})
        writer.event(run, event_kind='lifecycle', reason_code='stage_started', status='running', task_id=task, attempt_id=attempt,
                     version_id=operations[0].document.version_id)
    return attempt


def _interrupted_causes(writer):
    return {attempt: json.loads(details) for attempt, details in writer.connection.execute(
        f"SELECT attempt_id,details_json FROM {writer.connection.event_table} WHERE reason_code='attempt_interrupted' AND event_kind='lifecycle'")}


@pytest.mark.parametrize('evidence', ['ledger_event', 'record_only'])
def test_f3_an_old_disk_full_never_explains_a_later_process_death(tmp_path, evidence):
    from test_ledger_scheduler import graph, scheduler
    from glossapi.stages.interruption import allocate, record
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine = scheduler(writer, tmp_path)
        run = engine.freeze(graph(writer, tmp_path, count=1))
        operations, tasks, _ = engine._load(run)
        task = tasks['produce:0']
        first = _attempt_start(writer, run, task, operations)
        allocate(tmp_path)
        record(tmp_path, run, OSError(28, 'first interruption only'))
        if evidence == 'ledger_event':
            with writer.transaction():
                writer.event(run, event_kind='run', reason_code='disk_full', details={'errno': 28})
        engine._recover(run, tasks, operations)
        second = _attempt_start(writer, run, task, operations)  # replacement attempt; killed with NO new disk error
        engine._recover(run, tasks, operations)
        causes = _interrupted_causes(writer)
        assert causes[first].get('interruption_cause') == 'disk_full'
        assert causes[second].get('interruption_cause') is None, 'an old ENOSPC was attributed to a later process death'


def test_f4_report_why_shows_the_verified_record_like_status(tmp_path):
    from glossapi.ledger.reader import report_why, status_view, select_run
    from glossapi.stages.interruption import allocate, record
    from test_ledger_scheduler import graph, scheduler
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine = scheduler(writer, tmp_path)
        operations = graph(writer, tmp_path, 1)
        run = engine.freeze(operations)
        _, tasks, _ = engine._load(run)
        _attempt_start(writer, run, tasks['produce:0'], operations)
        allocate(tmp_path)
        record(tmp_path, run, OSError(28, 'ENOSPC that cannot be inserted in ledger'))
        selected = select_run(writer.connection, run)
        status = status_view(writer.connection, run=selected)
        why = report_why(writer.connection, operations[0].document.document_id, run=selected)
        assert status['state'] == 'interrupted' and status['interruption']['basis'] == 'record allocated at run start'
        assert why['interruption']['cause'] == 'disk_full' and why['interruption']['basis'] == status['interruption']['basis']
        assert why['interruption']['run_id'] == run
        # The CLI's default run (the latest) is the interrupted one.
        latest = select_run(writer.connection)
        assert report_why(writer.connection, operations[0].document.document_id, run=latest)['interruption']['run_id'] == run


def test_f4_report_why_has_no_interruption_once_a_replacement_started(tmp_path):
    from glossapi.ledger.reader import report_why, select_run
    from glossapi.stages.interruption import allocate, record
    from test_ledger_scheduler import graph, scheduler
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        engine = scheduler(writer, tmp_path)
        operations = graph(writer, tmp_path, 1)
        run = engine.freeze(operations)
        _, tasks, _ = engine._load(run)
        _attempt_start(writer, run, tasks['produce:0'], operations)
        allocate(tmp_path)
        record(tmp_path, run, OSError(28, 'full'))
        engine._recover(run, tasks, engine._load(run)[0])
        _attempt_start(writer, run, tasks['produce:0'], operations)
        why = report_why(writer.connection, operations[0].document.document_id, run=select_run(writer.connection, run))
        assert 'interruption' not in why


# ------------------------------------------------------------- F5 run-close ENOSPC ---

@pytest.mark.parametrize('where', ['barrier', 'vacuum'])
def test_f5_enospc_at_run_close_is_recorded_reported_and_resumable(tmp_path, monkeypatch, where):
    import sqlite3
    from glossapi.stages.run import DiskFullError
    from glossapi.stages.interruption import read
    from glossapi.ledger.reader import status_view, select_run, report_why
    from glossapi.ledger.writer import _OPEN_WRITERS, open_readonly
    source = pdf(tmp_path/'doc.pdf', 'Close interruption body text')
    root = tmp_path/'out'
    fired = []
    if where == 'barrier':
        original = LedgerWriter.durable_barrier
        def full(self, unlinking=(), **kwargs):
            if not unlinking and self.barrier_context and not fired:
                fired.append(self)
                raise sqlite3.OperationalError('database or disk is full')
            return original(self, unlinking, **kwargs)
        monkeypatch.setattr(LedgerWriter, 'durable_barrier', full)
    else:
        original = LedgerWriter._vacuum_at_run_close
        def full(self):
            if not fired:
                fired.append(self)
                raise sqlite3.OperationalError('database or disk is full')
            return original(self)
        monkeypatch.setattr(LedgerWriter, '_vacuum_at_run_close', full)
    with pytest.raises(DiskFullError) as raised:
        Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    run = raised.value.details['run_id']
    assert str(root/'corpus.sqlite3') not in _OPEN_WRITERS and fired[0].connection is None and fired[0].lock is None
    assert read(root, run) is not None, 'run-close ENOSPC escaped after the interruption record was cleared'
    monkeypatch.undo()
    with open_readonly(root/'corpus.sqlite3') as connection:
        selected = select_run(connection, run)
        status = status_view(connection, run=selected)
        assert status['state'] == 'interrupted' and status['interruption']['cause'] == 'disk_full'
        document = connection.execute("SELECT document_id FROM documents").fetchone()[0]
        assert report_why(connection, document, run=selected)['interruption']['cause'] == 'disk_full'
    resumed = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings={**SETTINGS, 'resume': run})
    assert resumed['state'] == 'completed' and read(root, run) is None
    with open_readonly(root/'corpus.sqlite3') as connection:
        assert status_view(connection, run=select_run(connection, run))['state'] == 'completed'


# ----------------------------------------------------------- F6 post-close edits ---

def test_f6_copy_edited_just_after_close_is_not_overwritten(tmp_path):
    import os
    from test_k4c2_fixes import published_copy
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path)
    closed = copy.stat().st_mtime_ns
    copy.write_text('my edit after the run closed\n')
    os.utime(copy, ns=(closed+1_000_000_000, closed+1_000_000_000))  # inside the old two-second window
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == []
        assert [entry['file_id'] for entry in writer.ambiguous_copies] == [copy_id]
    assert copy.read_text() == 'my edit after the run closed\n'
    assert source.read_bytes() == b'{"a": 1}\n'*1000  # both kept


def test_f6_edit_with_mtime_restored_is_still_not_overwritten(tmp_path):
    import os
    from test_k4c2_fixes import published_copy
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path)
    info = copy.stat()
    with copy.open('r+b') as stream: stream.truncate(100)  # same-ish edit, then the old times put back
    os.utime(copy, ns=(info.st_atime_ns, info.st_mtime_ns))
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == []
    assert copy.stat().st_size == 100


def test_f6_genuinely_torn_copy_is_still_republished(tmp_path):
    from test_k4c2_fixes import published_copy
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, source, copy, source_id, copy_id = published_copy(writer, tmp_path, torn=True)
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == [copy_id]
    assert copy.read_bytes() == source.read_bytes()


# ---------------------------------------------------- F7 exact product preflight ---

@pytest.mark.parametrize('where', ['manifest', 'none'])
def test_f7_large_producer_metadata_is_charged_before_publication(tmp_path, monkeypatch, transport, where):
    from glossapi.stages import final_products, disk_budget
    from glossapi.stages.disk_budget import DiskBudgetError
    monkeypatch.setattr(disk_budget, 'require_disk_guard', lambda root, settings: (Path(root)/'.glossapi/staging').mkdir(parents=True, exist_ok=True))
    # Room for the run's charge without producer metadata (CP-K4c3 R3: every publication, measured 1.36 MB for this
    # one-document run) but not for 2 x 300 KB of producer metadata on top of it (measured 2.30 MB).
    monkeypatch.setattr(final_products, 'corpus_room', lambda *a: 1800*1024)
    checks = []
    original = final_products.require_room
    def checked(root, settings, need, **kwargs):
        checks.append({'need': need, **kwargs})
        return original(root, settings, need, **kwargs)
    monkeypatch.setattr(final_products, 'require_room', checked)
    url = 'http://fixture.invalid/metadata.pdf'
    transport[url] = pdf(tmp_path/'doc.pdf', 'Small document body').read_bytes()
    description = 'metadata value '*20000 if where == 'manifest' else 'short'
    settings = {**SETTINGS, 'source_manifest': {'documents': [{'url': url, 'description': description}]}, 'memory_saving': 'extreme',
                'disk_budget_bytes': 1 << 30, 'max_source_bytes': 8192, 'max_attempt_bytes': 16384}
    root = tmp_path/'out'
    if where == 'manifest':
        with pytest.raises(DiskBudgetError, match='exact, before the publications'):
            Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=settings)
        assert not (root/'export.jsonl').exists()
        exact = [check for check in checks if 'exact' in check.get('when', '')]
        assert exact and exact[0]['need'] > 2*len(description)
    else:
        result = Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=settings)
        assert result['state'] == 'completed'
        exact = next(check for check in checks if 'exact' in check.get('when', ''))
        assert exact['need'] >= 2*(root/'export.jsonl').stat().st_size  # generation + published copy, at least


def test_f7_exact_export_need_counts_escaped_text_and_metadata(tmp_path):
    import pandas as pd
    from glossapi.stages.final_products import exact_export_need
    text = tmp_path/'a.md'
    text.write_text('line "one"\n\tline two\n'*100)
    metadata = tmp_path/'meta.parquet'
    pd.DataFrame({'filename': ['a.pdf', 'b.pdf'], 'description': ['x'*5000, 'y'*9000]}).to_parquet(metadata, index=False)
    need = exact_export_need([text], [metadata], {'a'})
    escaped = len(json.dumps(text.read_text(), ensure_ascii=False).encode())
    from glossapi.stages.final_products import RECORD_OVERHEAD_BYTES
    assert need >= 2*(escaped + 5000) + text.stat().st_size + metadata.stat().st_size
    assert need < 2*(escaped + RECORD_OVERHEAD_BYTES + 9000) + text.stat().st_size + metadata.stat().st_size  # the unplanned row is not charged


# ============================================================ CP-K4b (reviewer 2) ===

def _next_tick():
    """Let the filesystem clock advance: Linux stamps mtime/ctime from a coarse clock (one jiffy, a few ms), so a
    rewrite inside the same tick and of the same size is indistinguishable by stat identity. That same-tick window is
    the declared residual (CP-K4b F3); these tests exercise everything outside it."""
    import time
    time.sleep(0.1)


def _bad_snapshot(path):
    import sqlite3
    _next_tick()
    raw = sqlite3.connect(path)
    raw.execute('INSERT INTO snapshots VALUES(?,?,?,?)', (bytes.fromhex('f'*64), 'settings', 1, '{"kind":"settings","schema_version":1}'))
    raw.commit()
    raw.close()


def _fresh_ledger(path):
    with LedgerWriter(path) as writer:
        with writer.transaction():
            writer.run({}, {}, {}, code_revision='probe')


def test_k4b_f1_failed_open_never_installs_trust(tmp_path):
    from glossapi.ledger import writer as W
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    _bad_snapshot(path)
    for _ in range(2):  # the retry must run the full check again and refuse again
        with pytest.raises(W.LedgerError, match='checksum'):
            LedgerWriter(path)
    assert str(path.resolve()) not in W._integrity_known


def test_k4b_f1_external_commit_while_open_discards_trust(tmp_path):
    from glossapi.ledger import writer as W
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    with LedgerWriter(path):
        _bad_snapshot(path)  # another connection commits while the glossAPI writer is open
    assert str(path.resolve()) not in W._integrity_known
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)


def test_k4b_f1_clean_close_still_installs_trust(tmp_path):
    from glossapi.ledger import writer as W
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    assert str(path.resolve()) in W._integrity_known
    calls = []
    real = W.integrity
    try:
        W.integrity = lambda connection: (calls.append(1), real(connection))
        with LedgerWriter(path):
            pass
    finally:
        W.integrity = real
    assert calls == []


def test_k4b_f3_external_edit_preserving_mtime_forces_a_full_check(tmp_path):
    import os
    from glossapi.ledger import writer as W
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    before = path.stat()
    _bad_snapshot(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)


def test_k4b_f3_caller_files_are_always_byte_hashed(tmp_path):
    import hashlib, os
    from glossapi.ledger.encoding import file_hash, remember_own_hash, run_hash_scope
    own, alias = tmp_path/'own.md', tmp_path/'caller-alias.md'
    own.write_bytes(b'AAAA')
    os.link(own, alias)
    with run_hash_scope():
        before = own.stat()
        remember_own_hash(own, hashlib.sha256(b'AAAA').hexdigest())
        _next_tick()
        alias.write_bytes(b'BBBB')
        os.utime(alias, ns=(before.st_atime_ns, before.st_mtime_ns))
        actual = hashlib.sha256(b'BBBB').hexdigest()
        assert file_hash(alias) == actual  # caller-supplied: bytes, never the registry
        assert file_hash(own, own=True) == actual  # own: the ctime change voids the registration


def test_k4b_f3_original_rewrite_preserving_mtime_is_caught(tmp_path):
    import hashlib, os
    from glossapi.ledger import journal as J
    original = tmp_path/'original.pdf'
    original.write_bytes(b'AAAA')
    J.observed_original_hash('extract', original)
    before = original.stat()
    _next_tick()
    original.write_bytes(b'BBBB')
    os.utime(original, ns=(before.st_atime_ns, before.st_mtime_ns))
    cached = []
    assert J.observed_original_hash('clean', original, cached) == hashlib.sha256(b'BBBB').hexdigest()
    assert cached == []  # ctime differs: no cache hit


def test_k4b_f3_cached_original_is_byte_verified_before_destruction(tmp_path):
    from glossapi.ledger import journal as J
    original = tmp_path/'original.pdf'
    original.write_bytes(b'AAAA')
    J.observed_original_hash('extract', original)
    cached = []
    digest = J.observed_original_hash('clean', original, cached)
    assert cached == [(original, digest)]
    holder = type('Journal', (), {})()
    holder.cached_originals, holder.versions = cached, {}
    J.ResultJournal.verify_cached_originals(holder)  # unchanged: passes
    J._original_hashes[str(original)] = (J._original_hashes[str(original)][0], '0'*64)
    holder.cached_originals = [(original, '0'*64)]  # what a stale same-stat cache would have bound
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        J.ResultJournal.verify_cached_originals(holder)


def test_k4b_f4_a_forked_child_runs_the_full_check(tmp_path):
    import os, signal
    from glossapi.ledger import writer as W
    from glossapi.ledger import journal as J
    from glossapi.ledger import encoding as E
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    assert str(path.resolve()) in W._integrity_known
    J._original_hashes['x'] = ((0, 0, 0, 0, 0), 'h')
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        try:
            signal.alarm(20)
            os.close(read)
            calls = []
            real = W.integrity
            W.integrity = lambda connection: (calls.append(1), real(connection))
            with E.run_hash_scope():
                with LedgerWriter(path):
                    pass
            os.write(write, json.dumps({'checks': len(calls), 'originals': len(J._original_hashes)}).encode())
        finally:
            os._exit(0)
    os.close(write)
    os.waitpid(pid, 0)
    child = json.loads(os.read(read, 4096))
    os.close(read)
    J._original_hashes.pop('x', None)
    assert child == {'checks': 1, 'originals': 0}


def test_k4b_f6_failed_v4_migration_can_be_retried_with_the_same_backup(tmp_path):
    import hashlib, sqlite3
    from glossapi.ledger.schema import FORMAT_SCHEMA_SQL
    from glossapi.ledger.writer import LedgerError, open_readonly, integrity
    path, backup = tmp_path/'corpus.sqlite3', tmp_path/'backup.sqlite3'
    checksum = hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest()
    with LedgerWriter(path, storage_version=3) as writer:
        def deny(action, arg1, arg2, database, trigger):
            return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_INDEX and arg1 == 'files_export_kind' else sqlite3.SQLITE_OK
        writer.connection.set_authorizer(deny)
        with pytest.raises(sqlite3.DatabaseError):
            writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=checksum, code_revision='test', backup_path=backup)
        writer.connection.set_authorizer(None)
        assert backup.exists() and writer.connection.execute('PRAGMA user_version').fetchone()[0] == 3
        writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=checksum, code_revision='test', backup_path=backup)  # retry, same path
    with open_readonly(path) as db:
        integrity(db)
        assert db.execute('PRAGMA user_version').fetchone()[0] == 4
    with open_readonly(backup) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3


def test_k4b_f6_a_foreign_file_at_the_backup_path_is_kept_and_refused(tmp_path):
    import hashlib
    from glossapi.ledger.schema import FORMAT_SCHEMA_SQL
    from glossapi.ledger.writer import LedgerError
    path, backup = tmp_path/'corpus.sqlite3', tmp_path/'backup.sqlite3'
    backup.write_bytes(b'not a backup')
    with LedgerWriter(path, storage_version=3) as writer:
        with pytest.raises(LedgerError, match='not a verified backup'):
            writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest(), code_revision='test', backup_path=backup)
    assert backup.read_bytes() == b'not a backup'
