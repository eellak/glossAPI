"""Fix round 6 (ASTRA-FIXES2 F1-F3 on 70fc2c0).

F3 [HIGH] A full disk in the extraction-state metadata write (``_ProcessingStateManager.save``, including the
   ``ensure_metadata_parquet`` it calls first) or in the cleaner's noise-metric merge was swallowed: the run completed
   with no failure (state) or recorded ``clean_noise_scoring_failed``/``clean_exception`` (noise), and
   ``--retry-failed`` never repeated the lost write. Now the exhausted filesystem propagates with its original
   exception as the cause; the legacy wrapper attempt is typed ``disk_full`` (``ResultJournal.finish``) and the run
   stops with ``DiskFullError``; the resume repeats the barrier.
   Reviewer probes (review_scratch/ASTRA-FIXES2/test_adversarial_new.py): six direct injections (ENOSPC, EDQUOT, SQLite
   "database or disk is full" x the two sites) and two public-run probes. Each is ported here and extended: the
   failure is typed disk_full, then the retry runs, then the products equal an uninterrupted run byte for byte.
F1 [MEDIUM] (K2E-F3) The processing-state append concatenated with plain ``pd.concat``: an int64 foreign value
   2**53+1 became float64 and read back as 9007199254740992. The reviewer's original test is copied unchanged.
F2 [MEDIUM] (CP-K4c3 R4 / K4F1-G3) A freshly hashed original (excluded from the hash cache by the racy-file rule) was
   never re-checked at a destructive frontier. The reviewer's original fresh-source probe is copied unchanged (no
   ``RACY_MARGIN_NS`` override); the forced-cache control stays in test_k4fix3.py.
"""
from __future__ import annotations

import errno
import sqlite3

import pandas as pd
import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import writer as W
from glossapi.ledger.writer import open_readonly
from glossapi.stages.execution import disk_full
from test_k4fix1 import pdf, _next_tick
from test_k4g_fix4 import products

CODES = [errno.ENOSPC, errno.EDQUOT, 'sqlite']


def storage_error(code, what):
    """The injected exhaustion: ENOSPC / EDQUOT as OSError, or SQLite's disk-full error."""
    if code == 'sqlite':
        return sqlite3.OperationalError('database or disk is full')
    return OSError(code, 'injected storage failure: ' + what)


# ======================================================================== F3: direct injections, state site ===

def _state_corpus(root):
    """The reviewer's state fixture: a one-row metadata parquet and a markdown directory."""
    path = root / 'download_results' / 'download_results.parquet'
    path.parent.mkdir(parents=True)
    pd.DataFrame({'filename': ['alpha.pdf']}).to_parquet(path, index=False)
    (root / 'markdown').mkdir()
    return path


def _save(root):
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    _ProcessingStateManager(root / 'markdown' / '.processing_state.pkl').save({'alpha'}, set())


@pytest.mark.parametrize('site', ['metadata-writer', 'state-update'])
@pytest.mark.parametrize('code', CODES)
def test_processing_state_storage_error_reaches_the_caller_then_retry_matches(tmp_path, monkeypatch, code, site):
    """Reviewer probe test_processing_state_storage_error_reaches_the_caller (site metadata-writer: every metadata
    write fails, which ``ensure_metadata_parquet`` meets first), plus the state's own locked update (site
    state-update). The caller sees the exhaustion; a second save without the fault writes exactly the bytes an
    uninterrupted save writes."""
    import glossapi.corpus.corpus_state as cs
    reference = _state_corpus(tmp_path / 'reference')
    _save(tmp_path / 'reference')
    target = _state_corpus(tmp_path / 'faulted')
    hits = []

    def fail(*args, **kwargs):
        hits.append(site)
        raise storage_error(code, site)
    if site == 'metadata-writer':
        monkeypatch.setattr(ps, '_write_metadata_parquet_locked', fail)
    else:
        monkeypatch.setattr(cs, 'update_metadata_parquet', fail)
    with pytest.raises(Exception) as caught:
        _save(tmp_path / 'faulted')
    assert hits and disk_full(caught.value), repr(caught.value)
    monkeypatch.undo()
    _save(tmp_path / 'faulted')
    assert target.read_bytes() == reference.read_bytes()


# ======================================================================== F3: direct injections, noise site ===

def _noise_corpus(root):
    corpus = Corpus(root / 'in', root / 'out')
    (corpus.markdown_dir / 'alpha.md').write_text('Κείμενο με ελληνικά γράμματα και λίγα λατινικά words here.\n' * 20)
    return corpus


@pytest.mark.parametrize('code', CODES)
def test_clean_noise_metadata_storage_error_fails_clean_then_retry_matches(tmp_path, monkeypatch, code):
    """Reviewer probe test_clean_noise_metadata_storage_error_fails_publication: the noise-metric merge's storage error
    reaches the clean caller (no ``clean_noise_scoring_failed`` continuation); a repeated clean gives the products of
    an uninterrupted clean."""
    reference = _noise_corpus(tmp_path / 'reference')
    reference.clean(num_threads=1, drop_bad=False)
    corpus = _noise_corpus(tmp_path / 'faulted')
    real, hits = ps.update_metadata_parquet, []

    def inject(path, callback, *args, **kwargs):
        if callback.__name__ == '_merge_noise':
            hits.append(callback.__name__)
            raise storage_error(code, 'noise metric merge')
        return real(path, callback, *args, **kwargs)
    monkeypatch.setattr(ps, 'update_metadata_parquet', inject)
    with pytest.raises(Exception) as caught:
        corpus.clean(num_threads=1, drop_bad=False)
    assert hits and disk_full(caught.value), repr(caught.value)
    monkeypatch.undo()
    corpus.clean(num_threads=1, drop_bad=False)
    assert products(tmp_path / 'faulted' / 'out') == products(tmp_path / 'reference' / 'out')


# ============================================================================ F3: public runs (streaming) ===

def _sources(root):
    return [pdf(root / 'input' / 'alpha.pdf', 'A simple text for an extraction storage error probe'),
            pdf(root / 'input' / 'beta.pdf', 'Another native document so the publication covers two sources')]


def _public_events(output):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        return [tuple(row) for row in connection.execute(
            f"SELECT event_kind, reason_code, status FROM {connection.event_table} "
            "WHERE reason_code='disk_full' OR status='failed' ORDER BY event_id")]


SETTINGS6 = {'debug': 'off', 'extract': {'repair_fonts': False}}


@pytest.mark.parametrize('where', ['state', 'noise'])
def test_public_run_metadata_disk_full_is_typed_retried_and_matches(tmp_path, monkeypatch, where):
    """Reviewer probe test_streaming_publication_disk_full_is_typed_and_retried, extended: ENOSPC once in the
    extraction-state metadata write (state) or the noise-metric merge (noise) of a streaming run. The run stops with
    the typed DiskFullError, the wrapper attempt is typed disk_full and the run records disk_full; the resume with
    --retry-failed completes and every product equals an uninterrupted run's."""
    import inspect
    from glossapi.stages.run import DiskFullError
    sources = _sources(tmp_path)
    reference = Corpus(tmp_path / 'empty', tmp_path / 'reference').run(sources, stages=('extract', 'clean', 'export'),
                                                                       settings=SETTINGS6)
    assert reference['state'] == 'completed'
    output = tmp_path / 'faulted'
    original_write, original_update, hits = ps._write_metadata_parquet_locked, ps.update_metadata_parquet, []

    def write(*args, **kwargs):
        if where == 'state' and not hits and any(frame.function == 'save' and frame.filename.endswith('corpus_state.py')
                                                 for frame in inspect.stack()):
            hits.append('state')
            raise OSError(errno.ENOSPC, 'injected state metadata write ENOSPC')
        return original_write(*args, **kwargs)

    def update(path, callback, *args, **kwargs):
        if where == 'noise' and not hits and callback.__name__ == '_merge_noise':
            hits.append('noise')
            raise OSError(errno.ENOSPC, 'injected noise metadata write ENOSPC')
        return original_update(path, callback, *args, **kwargs)
    monkeypatch.setattr(ps, '_write_metadata_parquet_locked', write)
    monkeypatch.setattr(ps, 'update_metadata_parquet', update)
    with pytest.raises(DiskFullError) as caught:
        Corpus(tmp_path / 'empty', output).run(sources, stages=('extract', 'clean', 'export'), settings=SETTINGS6)
    monkeypatch.undo()
    assert hits == [where] and disk_full(caught.value)
    stage = 'extract' if where == 'state' else 'clean'
    events = _public_events(output)
    assert ('lifecycle', 'disk_full', 'failed') in events and ('run', 'disk_full', None) in events, events
    assert not any(reason == stage + '_exception' or reason == 'clean_noise_scoring_failed' for _, reason, _ in events), events
    run_id = caught.value.details['run_id']
    resumed = Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True)
    assert resumed['state'] == 'completed', resumed
    assert resumed.get('document_counts') == reference.get('document_counts')
    assert products(output) == products(tmp_path / 'reference')


# ================================================================== F1: reviewer's original test, unchanged ===

BIG = 2**53 + 1


def seed(root, frame=None):
    path = root / 'download_results/download_results.parquet'
    path.parent.mkdir(parents=True)
    if frame is None:
        frame = pd.DataFrame({'filename': ['old.pdf'], 'url': ['u'], 'foreign_id': pd.Series([BIG], dtype='int64')})
    frame.to_parquet(path, index=False)
    return path


def test_processing_state_append_keeps_exact_foreign_integer(tmp_path):
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    path = seed(tmp_path)
    (tmp_path / 'markdown').mkdir()
    _ProcessingStateManager(tmp_path / 'markdown/.processing_state.pkl').save({'new'}, set())
    stored = pd.read_parquet(path)
    value = stored.loc[stored.filename == 'old.pdf', 'foreign_id'].iloc[0]
    print('STATE_EXACT', repr(value), str(stored.foreign_id.dtype), stored.filename.tolist())
    assert int(value) == BIG and pd.api.types.is_integer_dtype(stored.foreign_id.dtype)


def test_processing_state_append_keeps_other_columns_as_before(tmp_path):
    """Only the column a float64 cannot hold changes dtype (Int64); an int64 column of small values, strings and the
    added row's nulls are written as the legacy concatenation wrote them (float64 with a missing cell)."""
    from glossapi.corpus.corpus_state import _ProcessingStateManager
    path = seed(tmp_path, pd.DataFrame({'filename': ['old.pdf'], 'url': ['u'], 'small': pd.Series([7], dtype='int64'),
                                        'foreign_id': pd.Series([-BIG], dtype='int64')}))
    (tmp_path / 'markdown').mkdir()
    _ProcessingStateManager(tmp_path / 'markdown/.processing_state.pkl').save({'new'}, set())
    stored = pd.read_parquet(path).set_index('filename')
    assert str(stored['foreign_id'].dtype) == 'Int64' and int(stored.at['old.pdf', 'foreign_id']) == -BIG
    assert pd.isna(stored.at['new.pdf', 'foreign_id'])
    assert str(stored['small'].dtype) == 'float64' and stored.at['old.pdf', 'small'] == 7.0
    assert stored.at['old.pdf', 'url'] == 'u' and pd.isna(stored.at['new.pdf', 'url'])


# ========================================================== F2: reviewer's original fresh-source probe, unchanged ===

def test_second_deletion_rechecks_changed_original(tmp_path, monkeypatch):
    from glossapi.stages import retention
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE','cutover')
    root=tmp_path/'out'
    sources=[pdf(root/'downloads'/f'{name}.pdf','Native text enough for cleaning '+name) for name in ['alpha','beta','gamma']]
    c=Corpus(tmp_path/'empty',root)
    c.extract(file_paths=sources,repair_fonts=False,accel_type='CPU',num_threads=1)
    c.clean(drop_bad=False,num_threads=1)
    shards=root/'shards';initial=c.jsonl_sharded(shards,shard_size_bytes=1,compression='none')
    real=retention.RetentionController.delete
    observed=[]
    def deletion(self,file_id,**kwargs):
        row=self.writer.file_catalog()[file_id]
        result=real(self,file_id,**kwargs)
        if row['kind']=='document_jsonl_shard' and result:
            observed.append(row['uri'])
            if len(observed)==1:
                _next_tick()
                sources[0].write_bytes(sources[0].read_bytes()+b'\nEXTERNAL EDIT BETWEEN DELETIONS\n')
                print('EDIT_AFTER_FIRST_DELETION',row['uri'])
        return result
    monkeypatch.setattr(retention.RetentionController,'delete',deletion)
    err=None
    try: Corpus(tmp_path/'empty',root).jsonl_sharded(shards,shard_size_bytes=1000000,compression='none')
    except Exception as e: err=str(e)
    with W.open_readonly(root/'corpus.sqlite3') as connection:
        states=[r[0] for r in connection.execute("SELECT status FROM current_task_state WHERE operation='export.jsonl_sharded'")]
    print('BETWEEN_DELETIONS',dict(error=err,deleted=observed,remaining=[p.name for p in initial if p.exists()],wrapper_states=states))
    assert err is not None and len(observed)==1, 'changed original was not rehashed at the next destructive boundary or before outcome'


def test_fresh_original_changed_after_last_deletion_fails_outcome(tmp_path, monkeypatch):
    """Fresh-source variant of the pre-outcome frontier (test_k4fix3 forces the cache): an edit after the last
    deletion is caught before the outcome; the wrapper records the failure and both deletions were the old shards."""
    from test_k4fix3 import _three_shards, _shard_deletions, _wrapper_states
    root, sources, shards, initial = _three_shards(tmp_path, monkeypatch)

    def edit(count):
        if count == 2:
            _next_tick()
            sources[1].write_bytes(sources[1].read_bytes() + b'\nEDIT AFTER THE LAST DELETION\n')
    observed = _shard_deletions(monkeypatch, edit)
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        Corpus(tmp_path / 'empty', root).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert len(observed) == 2
    assert _wrapper_states(root)[-1] == 'failed'


def test_fresh_unchanged_originals_hash_once_per_phase(tmp_path, monkeypatch):
    """Cost with fresh originals: the first frontier hashes each relied-on original once (a real verification); later
    frontiers re-stat it and hash nothing more."""
    from glossapi.ledger import journal as J
    from test_k4fix3 import _three_shards, _shard_deletions
    root, sources, shards, initial = _three_shards(tmp_path, monkeypatch)
    observed = _shard_deletions(monkeypatch, lambda count: None)
    hashed, verifying = [], []
    real_hash = J.file_hash

    def counting(path, *args, **kwargs):
        if verifying:
            hashed.append(str(path))
        return real_hash(path, *args, **kwargs)
    monkeypatch.setattr(J, 'file_hash', counting)
    real_verify = J.ResultJournal.verify_cached_originals

    def verify(self, rows=None):
        verifying.append(1)
        try:
            return real_verify(self, rows)
        finally:
            verifying.pop()
    monkeypatch.setattr(J.ResultJournal, 'verify_cached_originals', verify)
    Corpus(tmp_path / 'empty', root).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert len(observed) == 2
    assert hashed and all(hashed.count(str(source)) == 1 for source in sources if str(source) in hashed), hashed


# ============================================================== F3 audit: the legacy cleaner subprocess ===

@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EDQUOT, errno.EACCES])
def test_legacy_cleaner_subprocess_disk_full_is_raised_typed(tmp_path, monkeypatch, code):
    """The legacy profile runs the native cleaner in a subprocess; a full disk there was only a non-zero exit code
    (``clean_process_failed``, "proceeding with existing metrics"). ENOSPC/EDQUOT now reach the caller as OSError with
    their errno; any other native error keeps the old continuation."""
    import os
    fault = tmp_path / 'fault'
    fault.mkdir()
    (fault / 'sitecustomize.py').write_text(
        'import errno, glossapi_rs_cleaner\n'
        f'def full(*args, **kwargs): raise OSError({code}, "injected native cleaner write failure")\n'
        'glossapi_rs_cleaner.run_complete_pipeline = full\n')
    monkeypatch.setenv('PYTHONPATH', str(fault) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'legacy')
    corpus = _noise_corpus(tmp_path / 'corpus')
    if code == errno.EACCES:
        corpus.clean(num_threads=1, drop_bad=False)  # unchanged: logged, existing metrics
        return
    with pytest.raises(OSError) as caught:
        corpus.clean(num_threads=1, drop_bad=False)
    assert caught.value.errno == code and disk_full(caught.value), repr(caught.value)


# ================================================== F3 audit: the multi-GPU extract worker (in process, no GPU) ===

def test_multi_gpu_worker_reports_disk_full_and_stops(tmp_path, monkeypatch):
    """A full disk inside a multi-GPU worker's batch was a "problematic" batch (then skip-listed by the controller) and
    the worker went on with the queue. It now reports a typed disk_full event (no problematic items) and stops with
    WORKER_DISK_FULL_EXIT; the controller raises the typed error once every worker stopped (phase_extract)."""
    import os
    import queue
    from glossapi.corpus import corpus_orchestrator as orchestrator
    for name in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'GLOSSAPI_WORKER_LOG_DIR', 'GLOSSAPI_WORKER_LOG_VERBOSE'):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(Corpus, 'prime_extractor', lambda self, *args, **kwargs: None)

    def extract(self, *args, **kwargs):
        calls.append(kwargs.get('file_paths'))
        raise OSError(errno.ENOSPC, 'No space left on device (injected batch write)')
    monkeypatch.setattr(Corpus, 'extract', extract)
    (tmp_path / 'in').mkdir()
    work, results = queue.Queue(), queue.Queue()
    for name in ('a.pdf', 'b.pdf'):
        work.put(name)
    with pytest.raises(SystemExit) as stopped:
        orchestrator.gpu_extract_worker_queue(0, 0, 'gpu0-w0', str(tmp_path / 'in'), str(tmp_path / 'out'), work, False,
                                              False, False, False, False, 'pdf', 1, False, False, False, 'safe', results,
                                              None, None, False, False)
    events = []
    while not results.empty():
        events.append(results.get())
    assert stopped.value.code == orchestrator.WORKER_DISK_FULL_EXIT
    assert len(calls) == 1 and work.qsize() == 1, (calls, work.qsize())
    kinds = [event['event'] for event in events]
    assert 'disk_full' in kinds and not any(event.get('problematic') for event in events), events
    report = next(event for event in events if event['event'] == 'disk_full')
    assert report['errno'] == errno.ENOSPC


# ================================ FIX6-G1: any failed publication barrier, retried, equals the uninterrupted run ===

def test_failed_clean_barrier_retry_matches_uninterrupted_run(tmp_path, monkeypatch):
    """Found while testing F3, pre-existing on 70fc2c0 for any failed barrier: a generic exception in the clean barrier
    after its wrapper rewrote download_results.parquet made the --retry-failed resume refuse with post_close_edit; past
    that, the retried run kept public raw markdown and the temporary cleaning_report.parquet. The failed barrier now
    rolls the uncommitted metadata back to its committed projection, and retention releases what the abandoned wrapper
    attempt pinned: every product equals the uninterrupted run's."""
    sources = _sources(tmp_path)
    reference = Corpus(tmp_path / 'empty', tmp_path / 'reference').run(sources, stages=('extract', 'clean', 'export'),
                                                                       settings=SETTINGS6)
    real, hits = ps.update_metadata_parquet, []

    def update(path, callback, *args, **kwargs):
        if callback.__name__ == '_finalize' and not hits:
            hits.append(1)
            raise RuntimeError('generic failure in the clean finalize')
        return real(path, callback, *args, **kwargs)
    monkeypatch.setattr(ps, 'update_metadata_parquet', update)
    first = Corpus(tmp_path / 'empty', tmp_path / 'faulted').run(sources, stages=('extract', 'clean', 'export'), settings=SETTINGS6)
    monkeypatch.undo()
    assert hits and first['state'] == 'completed_with_failures', first
    resumed = Corpus(tmp_path / 'empty', tmp_path / 'faulted').run(resume=first['run_id'], retry_failed=True)
    assert resumed['state'] == 'completed', resumed
    assert products(tmp_path / 'faulted') == products(tmp_path / 'reference')


# ======================================================= F3 audit: the batched extractor (fake Docling adapter) ===

from test_batched_extract import FakeAdapter, inputs as batched_inputs, exporter, core as batched_core  # noqa: E402,F401


class FullDiskExportAdapter(FakeAdapter):
    """The export child meets a full disk for ``export_bad`` (module level: the spawned export child imports it)."""

    def export(self, result, output_dir):
        if result.input.file.stem == 'export_bad':
            raise OSError(errno.ENOSPC, 'No space left on device (injected export write)')
        return super().export(result, output_dir)


def test_batched_export_disk_full_crosses_the_child_and_stops_extract(tmp_path, exporter):  # noqa: F811
    """The export child's ENOSPC was a receipt of (name, text), re-raised as RuntimeError and recorded as an ordinary
    problematic document. The receipt now carries the errno; the owner raises OSError(ENOSPC) and the extraction stops
    typed instead of carrying on."""
    runner = batched_core.BatchedExtractor(FullDiskExportAdapter(exporter), workers=2)
    with pytest.raises(OSError) as caught:
        runner.extract(batched_inputs(tmp_path, ['export_bad', 'healthy']), tmp_path / 'out' / 'markdown')
    assert caught.value.errno == errno.ENOSPC and disk_full(caught.value), repr(caught.value)


def test_batched_failure_recording_disk_full_stops_extract(tmp_path, exporter):  # noqa: F811
    """Recording a failed document (metadata, problematic copy) on a full disk was appended to the document's error
    text; it now stops the extraction typed. Any other recording error keeps the old continuation (test_batched_extract
    test_failure_sidecar_error_does_not_stop_other_documents)."""
    adapter = FakeAdapter(exporter)

    def cannot_record(path, out, message):
        raise OSError(errno.ENOSPC, 'No space left on device (injected failure sidecar)')
    adapter.failure = cannot_record
    runner = batched_core.BatchedExtractor(adapter, workers=2)
    with pytest.raises(OSError) as caught:
        runner.extract(batched_inputs(tmp_path, ['export_bad', 'healthy']), tmp_path / 'out' / 'markdown')
    assert caught.value.errno == errno.ENOSPC
