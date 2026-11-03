"""Fix round 7 (ASTRA-FIXES2c2 R1 on e742d12): recovery after a sustained full disk.

R1 [HIGH] A cutover wrapper rewrites the public compatibility metadata (``download_results/*.parquet``, and
   ``cleaning_report.parquet``) in place before its publication barrier commits. When the barrier then fails on a
   full disk, fix 6 rolled those bytes back by copying the committed projection over them; on a disk that is still
   full that copy fails too (it needs a temporary file), the failure was swallowed, the run close named the file as
   an unflushed copy, and after the operator freed space ``--retry-failed`` refused the run's own uncommitted bytes as
   a user edit (``post_close_edit``).
   Design (record the intent first): every in-place replacement of a metadata target inside a cutover wrapper first
   binds the hash of the new bytes in the ledger, in its own synchronous commit, BEFORE the replace
   (``parquet_schema._replace_in_place`` -> ``LedgerWriter.record_uncommitted_write``); if the ledger cannot take
   that record the replacement does not happen. The next start rolls back exactly those bytes to the committed
   projection (``verify_excluded_copies``), before the post-close-edit classification. A user's edit (other bytes)
   and a newer committed projection (a registration newer than the record) are never matched.
   Reviewer probes (review_scratch/ASTRA-FIXES2c2/test_scoped_adversarial.py, the three
   ``test_storage_remains_full_during_rollback_then_retry`` cases) are ported here unchanged in substance (no receipt
   file); the original file is also run unchanged (receipts in fix7/mac/).
"""
from __future__ import annotations

import errno
import hashlib
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, PostCloseEditError, open_readonly
from glossapi.stages.execution import disk_full
from glossapi.stages.run import DiskFullError
from test_fix6_astra import _sources, SETTINGS6, storage_error
from test_k4g_fix4 import products

STAGES = ('extract', 'clean', 'export')
PLAN = ('extract', 'clean', 'sections', 'annotate', 'export')
METADATA = Path('download_results') / 'download_results.parquet'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _intents(output):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        return [row[0] for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE reason_code='uncommitted_projection_write'")]


def _rolled_back(output):
    with open_readonly(output / 'corpus.sqlite3') as connection:
        rows = [__import__('json').loads(row[0]) for row in connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE event_kind='artifact' AND reason_code='present'")]
    return [row for row in rows if row.get('observation') == 'uncommitted_write_rolled_back']


def ledger_matches_files(output):
    """Every present managed registration of a public corpus path (outside .glossapi) names the bytes the file holds:
    after recovery, no rolled-back or abandoned registration still claims a file."""
    from urllib.parse import unquote, urlsplit
    columns = ['uri', 'sha256', 'kind']
    with open_readonly(output / 'corpus.sqlite3') as connection:
        rows = [dict(zip(columns, row)) for row in connection.execute(
            'SELECT ' + ','.join(columns) + " FROM current_file_state WHERE presence='present' AND ownership='managed'")]
    by_path = {}
    for row in rows:
        path = Path(unquote(urlsplit(row['uri']).path))
        if '.glossapi' not in path.parts:
            by_path.setdefault(path, set()).add(row['sha256'])
    return {str(path.relative_to(output)) if path.is_relative_to(output) else str(path): sorted(digests)
            for path, digests in by_path.items() if not path.is_file() or _sha(path) not in digests}


def _fail_noise_and_rollback(monkeypatch, code):
    """The reviewer's injection: the noise merge meets the full disk, and so does every later copy (the rollback)."""
    real_update, real_copy = ps.update_metadata_parquet, ps._copy_file_locked
    full, hits = [], []

    def update(path, callback, *args, **kwargs):
        if callback.__name__ == '_merge_noise':
            full.append(True)
            hits.append('noise')
            raise storage_error(code, 'noise merge')
        return real_update(path, callback, *args, **kwargs)

    def copy(source, destination, *args, **kwargs):
        if full:
            hits.append('rollback-copy')
            raise OSError(errno.EDQUOT if code == errno.EDQUOT else errno.ENOSPC, 'filesystem remains full during rollback')
        return real_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(ps, 'update_metadata_parquet', update)
    monkeypatch.setattr(ps, '_copy_file_locked', copy)
    return hits


# ============================================ R1: the reviewer's three sustained-full-storage cases (ported) ===

@pytest.mark.parametrize('code', [errno.ENOSPC, errno.EDQUOT, 'sqlite'])
def test_storage_remains_full_during_rollback_then_retry(tmp_path, monkeypatch, code):
    """Reviewer probe, unchanged in substance: the noise merge fails (ENOSPC / EDQUOT / SQLite full) and the barrier
    rollback copy fails too; the operator frees space (faults removed; neither metadata nor ledger touched by hand);
    ``retry_failed=True`` completes and every product equals the uninterrupted run's."""
    sources = _sources(tmp_path)
    reference = tmp_path / 'reference'
    Corpus(tmp_path / 'empty', reference).run(sources, stages=STAGES, settings=SETTINGS6)
    output = tmp_path / 'faulted'
    hits = _fail_noise_and_rollback(monkeypatch, code)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    left = _sha(output / METADATA)  # the run's own uncommitted metadata, still in place: the rollback could not write
    resumed = Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert 'rollback-copy' in hits
    assert resumed['state'] == 'completed'
    assert products(output) == products(reference)
    # How: those bytes were recorded as the run's own before they were written, and the resume start rolled them back.
    assert left in {__import__('json').loads(item)['sha256'] for item in _intents(output)}
    assert _rolled_back(output), 'the resume start rolled the uncommitted write back'
    assert ledger_matches_files(output) == {}


# ======================================== R1: every publication barrier, rollback impossible, then retry ===

@pytest.mark.parametrize('corpus', ['fresh', 'second_run'])
@pytest.mark.parametrize('stage', PLAN)
def test_every_barrier_full_disk_with_failing_rollback_retries_to_identical_products(tmp_path, monkeypatch, stage, corpus):
    """The same pattern at every wrapper barrier of the longest plan: the barrier's own commit (sealing the metadata
    projection into a new immutable artifact, which needs space) meets a full disk after the wrapper rewrote its
    compatibility products in place, and the rollback cannot write either. After space is freed the retry completes
    and every product equals the uninterrupted run's. ``second_run``: the corpus already holds a completed run (one
    source), so every wrapper of the faulted second run (two sources) rewrites projections committed before it."""
    from glossapi.stages import metadata as M
    sources = _sources(tmp_path)
    reference = tmp_path / 'reference'
    output = tmp_path / 'faulted'
    for root in (reference, output) if corpus == 'second_run' else ():
        first = Corpus(tmp_path / 'empty', root).run(sources[:1], stages=PLAN, settings=SETTINGS6)
        assert first['state'] == 'completed', first
    result = Corpus(tmp_path / 'empty', reference).run(sources, stages=PLAN, settings=SETTINGS6)
    assert result['state'] == 'completed', result
    real_seal, real_copy = M.seal_metadata_projections, ps._copy_file_locked
    full, hits = [], []

    def seal(journal):
        if journal.stage == stage and not full:
            full.append(True)
            hits.append('seal')
            raise OSError(errno.ENOSPC, 'injected: no space to seal the ' + stage + ' metadata projection')
        return real_seal(journal)

    def copy(source, destination, *args, **kwargs):
        if full:
            hits.append('rollback-copy')
            raise OSError(errno.ENOSPC, 'filesystem remains full during rollback')
        return real_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(M, 'seal_metadata_projections', seal)
    monkeypatch.setattr(ps, '_copy_file_locked', copy)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=PLAN, settings=SETTINGS6)
    monkeypatch.undo()
    assert hits[0] == 'seal'
    resumed = Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert resumed['state'] == 'completed', resumed
    assert products(output) == products(reference)
    assert ledger_matches_files(output) == ledger_matches_files(reference) == {}


# =============================================================== R1: a full ledger blocks the in-place write ===

@pytest.mark.parametrize('code', [errno.ENOSPC, 'sqlite'])
def test_full_ledger_at_the_intent_record_leaves_metadata_untouched(tmp_path, monkeypatch, code):
    """If the ledger cannot take the record of an uncommitted write, the write does not happen: the public metadata
    keeps its bytes, the run stops typed disk_full, and the retry completes with the uninterrupted run's products."""
    sources = _sources(tmp_path)
    reference = tmp_path / 'reference'
    Corpus(tmp_path / 'empty', reference).run(sources, stages=STAGES, settings=SETTINGS6)
    output = tmp_path / 'faulted'
    real_record, real_copy = LedgerWriter.record_uncommitted_write, ps._copy_file_locked
    seen = {}

    def record(self, run_id, path, sha256, size, *, wrapper_stage, **kwargs):
        if wrapper_stage == 'clean' and 'before' not in seen and Path(path).name == METADATA.name:
            seen['before'] = _sha(path)
            raise storage_error(code, 'ledger full at the uncommitted-write record')
        return real_record(self, run_id, path, sha256, size, wrapper_stage=wrapper_stage, **kwargs)

    def copy(source, destination, *args, **kwargs):
        if 'before' in seen:
            raise OSError(errno.ENOSPC, 'filesystem remains full during rollback')
        return real_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(LedgerWriter, 'record_uncommitted_write', record)
    monkeypatch.setattr(ps, '_copy_file_locked', copy)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    assert disk_full(stopped.value) and 'before' in seen
    assert _sha(output / METADATA) == seen['before'], 'the unrecorded write must not have replaced the metadata'
    assert not list((output / 'download_results').glob('*.tmp')), 'no temporary file left behind'
    resumed = Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert resumed['state'] == 'completed'
    assert products(output) == products(reference)


def test_intent_is_committed_before_the_replace(tmp_path, monkeypatch):
    """Order at the choke point: the record is committed (visible to another connection) before ``os.replace``."""
    sources = _sources(tmp_path)
    output = tmp_path / 'out'
    real_replace, checked = ps.os.replace, []

    def replace(source, target, *args, **kwargs):
        if Path(target).name == METADATA.name and Path(source).name.endswith('.tmp'):
            with open_readonly(output / 'corpus.sqlite3') as connection:
                digests = {__import__('json').loads(row[0])['sha256'] for row in connection.execute(
                    f"SELECT details_json FROM {connection.event_table} WHERE reason_code='uncommitted_projection_write'")}
            checked.append(_sha(source) in digests)
        return real_replace(source, target, *args, **kwargs)
    monkeypatch.setattr(ps.os, 'replace', replace)
    result = Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    assert result['state'] == 'completed'
    assert checked and all(checked), checked


# ================================================ R1: protections (user edits, newer committed projections) ===

def test_user_edit_after_the_failure_is_kept_and_refused(tmp_path, monkeypatch):
    """After the failure (rollback impossible) the user rewrites the metadata: the start refuses (post_close_edit) and
    keeps the user's bytes; the recorded uncommitted write never matches them."""
    sources = _sources(tmp_path)
    output = tmp_path / 'faulted'
    _fail_noise_and_rollback(monkeypatch, errno.ENOSPC)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    path = output / METADATA
    frame = pd.read_parquet(path)
    frame['user_note'] = 'kept'
    frame.to_parquet(path, index=False)
    edited = _sha(path)
    with pytest.raises(PostCloseEditError):
        Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert _sha(path) == edited
    with pytest.raises(PostCloseEditError):  # still refused at the next start, still kept
        Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert _sha(path) == edited


def test_user_edit_with_the_same_timestamps_is_still_kept(tmp_path, monkeypatch):
    """The user's edit keeps mtime and ctime equal to the close's record (a write within the same clock tick): the
    bytes still decide, and they match no recorded uncommitted write."""
    import os
    sources = _sources(tmp_path)
    output = tmp_path / 'faulted'
    _fail_noise_and_rollback(monkeypatch, errno.ENOSPC)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    path = output / METADATA
    info = path.stat()
    frame = pd.read_parquet(path)
    frame['user_note'] = 'same tick'
    frame.to_parquet(path, index=False)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    edited = _sha(path)
    real_lstat = Path.lstat
    # Present the recorded identity exactly (ctime cannot be set from user space): the same-tick case.
    with open_readonly(output / 'corpus.sqlite3') as connection:
        import json
        details = json.loads(connection.execute(
            f"SELECT details_json FROM {connection.event_table} WHERE reason_code='durability_barrier' "
            "ORDER BY event_id DESC LIMIT 1").fetchone()[0])
    recorded = details['run_close_excluded_copy_identities'][0]

    class Same:
        def __init__(self, real):
            self._real = real
        def __getattr__(self, name):
            if name == 'st_mtime_ns':
                return recorded[0]
            if name == 'st_ctime_ns':
                return recorded[1]
            return getattr(self._real, name)
    monkeypatch.setattr(Path, 'lstat', lambda self: Same(real_lstat(self)) if self == path else real_lstat(self))
    with pytest.raises(PostCloseEditError):
        Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    monkeypatch.undo()
    assert _sha(path) == edited


def test_newer_committed_projection_is_never_rolled_back_by_an_older_record(tmp_path, monkeypatch):
    """The wrapper records an uncommitted write, then a newer projection is sealed (committed) at the same path, then
    the disk fills and the rollback cannot write. The start keeps the newer committed bytes: the older record predates
    the current registration and never matches."""
    import json
    from glossapi.stages.metadata import seal_metadata_projections
    from glossapi.stages.products import active_journal, current_product
    from glossapi.stages.execution import sha256_file
    real_update, real_copy = ps.update_metadata_parquet, ps._copy_file_locked
    output = tmp_path / 'out'
    path = output / METADATA
    state = {}

    def update(target, callback, *args, **kwargs):
        if callback.__name__ == '_merge_noise' and 'committed' not in state:
            def append(frame):
                row = {column: pd.NA for column in frame.columns}
                row['filename'] = 'foreign-committed.pdf'
                return ps.lossless_concat([frame, pd.DataFrame([row])])
            real_update(target, append)
            journal = active_journal()
            seal_metadata_projections(journal)
            state['committed'] = current_product(journal.writer.connection, path)['sha256']
            assert state['committed'] == sha256_file(path)
            raise OSError(errno.ENOSPC, 'failure after a newer metadata projection committed')
        return real_update(target, callback, *args, **kwargs)

    def copy(source, destination, *args, **kwargs):
        if 'committed' in state:
            raise OSError(errno.ENOSPC, 'filesystem remains full during rollback')
        return real_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(ps, 'update_metadata_parquet', update)
    monkeypatch.setattr(ps, '_copy_file_locked', copy)
    with pytest.raises(DiskFullError):
        Corpus(tmp_path / 'empty', output).run(_sources(tmp_path), stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    assert _sha(path) == state['committed']
    assert state['committed'] in {json.loads(item)['sha256'] for item in _intents(output)}  # it was recorded first
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        assert writer.verify_excluded_copies() == []
        assert not getattr(writer, 'ambiguous_copies', [])
    assert _sha(path) == state['committed']
    assert 'foreign-committed.pdf' in pd.read_parquet(path).filename.tolist()


def test_record_older_than_the_registration_is_ignored(tmp_path):
    """Unit: an uncommitted-write record matches only if it is newer than the copy's current registration."""
    path = tmp_path / 'download_results' / 'download_results.parquet'
    path.parent.mkdir()
    pd.DataFrame({'filename': ['a.pdf']}).to_parquet(path, index=False)
    digest = _sha(path)
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run(SETTINGS6, {}, {}, code_revision='fix7')
        writer.record_uncommitted_write(run, ps.resolve_metadata_target(path), digest, path.stat().st_size,
                                        wrapper_stage='clean')
        with writer.transaction():
            later = writer.event(run, event_kind='run', reason_code='legacy_observed', message='later registration')
        assert writer._own_uncommitted_write({'event_id': later}, path) is None
        assert writer._own_uncommitted_write({'event_id': later - 2}, path) is not None
        path.write_bytes(path.read_bytes() + b'user')
        assert writer._own_uncommitted_write({'event_id': later - 2}, path) is None


def test_intent_record_is_its_own_synchronous_commit(tmp_path):
    """The record refuses to ride inside an open observation (it must be durable before the replace) and commits with
    ``synchronous=FULL``."""
    with LedgerWriter(tmp_path / 'corpus.sqlite3') as writer:
        with writer.transaction():
            run = writer.run(SETTINGS6, {}, {}, code_revision='fix7')
            with pytest.raises(W.LedgerError):
                writer.record_uncommitted_write(run, tmp_path / 'x.parquet', '0' * 64, 1, wrapper_stage='clean')
        modes = []
        real = writer.connection.execute

        class Spy:
            def __init__(self, connection):
                self.connection = connection
            def __getattr__(self, name):
                return getattr(self.connection, name)
            def execute(self, sql, *args):
                if 'synchronous' in sql:
                    modes.append(sql)
                return real(sql, *args)
        writer.connection = Spy(writer.connection)
        try:
            writer.record_uncommitted_write(run, tmp_path / 'x.parquet', '0' * 64, 1, wrapper_stage='clean')
        finally:
            writer.connection = writer.connection.connection
        assert modes == ['PRAGMA synchronous=FULL', 'PRAGMA synchronous=NORMAL']


# ====================================== R1: the OCR plan's barriers (stub engine; spawned workers need a child) ===

OCR_PLAN = ('extract', 'clean', 'ocr', 'export')
OCR_CHILD = '''
import errno, json, os, sys
from pathlib import Path
from glossapi import Corpus
import glossapi.parquet_schema as ps
from glossapi.stages import metadata as M
from glossapi.stages.run import DiskFullError

def main():
    root, step = Path(sys.argv[1]), sys.argv[2]
    stages = tuple(os.environ['FIX7_STAGES'].split(','))
    settings = {'debug': os.environ.get('FIX7_DEBUG', 'off'), 'extract': {'repair_fonts': False}}
    corpus = Corpus(root / 'empty', root / 'output')
    if step == 'start':
        stage, real_seal, real_copy, full, hits = os.environ.get('FIX7_FAULT'), M.seal_metadata_projections, ps._copy_file_locked, [], []
        def seal(journal):
            if journal.stage == stage and not full:
                full.append(True); hits.append('seal')
                raise OSError(errno.ENOSPC, 'injected: no space to seal the ' + stage + ' metadata projection')
            return real_seal(journal)
        def copy(source, destination, *args, **kwargs):
            if full:
                hits.append('rollback-copy')
                raise OSError(errno.ENOSPC, 'filesystem remains full during rollback')
            return real_copy(source, destination, *args, **kwargs)
        real_update, merges = ps.update_metadata_parquet, []
        def update(path, callback, *args, **kwargs):
            # The OCR publication's after-OCR clean merges the noise metrics again (the first merge is the clean's).
            if callback.__name__ == '_merge_noise' and not full:
                merges.append(True)
                if len(merges) == 2:
                    full.append(True); hits.append('ocr-noise')
                    raise OSError(errno.ENOSPC, 'injected: the OCR publication noise merge meets a full disk')
            return real_update(path, callback, *args, **kwargs)
        from glossapi.ledger.writer import LedgerWriter
        real_artifact, registrations = LedgerWriter.artifact, []
        def artifact(self, run_id, stage_name, kind, uri, *args, **kwargs):
            # The OCR publication replaces markdown/<scan>.md (registered by the extraction in this run) and then
            # registers it: the ledger is full at that registration.
            if '/.glossapi/' not in uri and uri.endswith('/output/markdown/polytonic_scan.md') and not full:
                registrations.append(True)
                if len(registrations) == 2:
                    full.append(True); hits.append('ocr-register')
                    import sqlite3
                    raise sqlite3.OperationalError('database or disk is full')
            return real_artifact(self, run_id, stage_name, kind, uri, *args, **kwargs)
        if stage == 'ocr-register':
            LedgerWriter.artifact, ps._copy_file_locked = artifact, copy
        elif stage == 'ocr-register-once':  # the ledger is full once; the rollback's copy has space
            LedgerWriter.artifact = artifact
        elif stage == 'ocr-noise':
            ps.update_metadata_parquet, ps._copy_file_locked = update, copy
        elif stage:
            M.seal_metadata_projections, ps._copy_file_locked = seal, copy
        try:
            result = corpus.run([Path(p) for p in sys.argv[3:]], stages=stages, settings=settings)
            print(json.dumps(dict(state=result['state'], hits=hits)))
        except DiskFullError as error:
            (root / 'runid').write_text(error.details['run_id'])
            print(json.dumps(dict(state='disk_full', hits=hits)))
    else:
        try:
            result = corpus.run(resume=(root / 'runid').read_text(), retry_failed=True)
            print(json.dumps(dict(state=result['state'])))
        except Exception as error:
            print(json.dumps(dict(state='error', error=repr(error))))

if __name__ == '__main__':  # spawned stage workers re-import this module
    main()
'''


def _ocr_child(root, step, sources=(), fault=None, debug='off'):
    import json
    import os
    import subprocess
    import sys
    from test_ocr_same_basename import STUB
    root.mkdir(parents=True, exist_ok=True)
    (root / 'stub').mkdir(exist_ok=True)
    (root / 'stub' / 'sitecustomize.py').write_text(STUB)
    (root / 'child.py').write_text(OCR_CHILD)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'GLOSSAPI_TEST_STUB_OCR': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'FIX7_STAGES': ','.join(OCR_PLAN),
           'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([str(root / 'stub'), *filter(None, [os.environ.get('PYTHONPATH')]),
                                          str(repo / 'src'), str(repo / 'tests')])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    env.pop('FIX7_FAULT', None)
    env['FIX7_DEBUG'] = debug
    if fault:
        env['FIX7_FAULT'] = fault
    done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root), step, *map(str, sources)],
                          capture_output=True, text=True, env=env, cwd=root, timeout=900)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module', params=['off', 'on'], ids=['debug-off', 'debug-on'])
def ocr_reference(tmp_path_factory, request):
    import shutil
    from test_k4g_fix4 import PDFS, SCAN
    if not SCAN.is_file():
        pytest.skip(f'scanned fixture missing: {SCAN}')
    inputs = tmp_path_factory.mktemp('fix7-ocr-inputs')
    shutil.copyfile(PDFS / 'alpha.pdf', inputs / 'alpha.pdf')
    shutil.copyfile(SCAN, inputs / 'polytonic_scan.pdf')
    sources = [inputs / 'alpha.pdf', inputs / 'polytonic_scan.pdf']
    root = tmp_path_factory.mktemp('fix7-ocr-reference')
    result = _ocr_child(root, 'start', sources, debug=request.param)
    assert result['state'] == 'completed', result
    assert ledger_matches_files(root / 'output') == {}
    return sources, products(root / 'output'), request.param


@pytest.mark.parametrize('stage', ['ocr-noise', 'ocr-register', 'ocr-register-once', 'export'])
def test_ocr_plan_barrier_full_disk_with_failing_rollback_retries_to_identical_products(tmp_path, ocr_reference, stage):
    """The OCR plan. ``ocr-noise``: the OCR publication (a caller kernel whose legacy callback rewrites the metadata the
    clean barrier committed in the same run) meets a full disk in its noise merge; ``ocr-register``: the ledger is full
    at the registration of the mirrors it has just replaced (markdown/<scan>.md, published by the extraction earlier in
    the run); ``export``: the export barrier's seal meets a full disk. The rollback cannot write (``ocr-register-once``:
    it can). The retry completes with the uninterrupted run's products."""
    sources, expected, debug = ocr_reference
    started = _ocr_child(tmp_path, 'start', sources, fault=stage, debug=debug)
    assert started['state'] == 'disk_full' and started['hits'][0] in ('seal', 'ocr-noise', 'ocr-register'), started
    resumed = _ocr_child(tmp_path, 'resume', debug=debug)
    assert resumed['state'] == 'completed', resumed
    assert products(tmp_path / 'output') == expected
    assert ledger_matches_files(tmp_path / 'output') == {}


# ============== R1 regression found by the node kill matrix (publish_extract_link): caller-seeded metadata survives ===

def seed_metadata(output, sources):
    """The node kill matrix's seed (kill_matrix.py seed()): one metadata row per source before the run, with its
    remote URL, and the harness's direct-paths marker in downloads/ (ParquetSchema writer)."""
    from glossapi.parquet_schema import ParquetSchema
    (output / 'downloads').mkdir(parents=True, exist_ok=True)
    (output / 'downloads' / '.harness-direct-paths').touch()
    ParquetSchema().write_metadata_parquet(pd.DataFrame([dict(filename=Path(path).name, filename_base=Path(path).stem,
        url='https://example.org/archive/' + Path(path).name, download_success=True, file_ext='pdf', processing_stage='',
        extract_status='') for path in sources]), output / METADATA)


def test_seeded_metadata_survives_a_failed_extract_barrier(tmp_path, monkeypatch):
    """A caller-seeded metadata table is rewritten in place by the extract wrapper, whose barrier then fails on a full
    disk (its seal; the rollback cannot write either). There is no committed generation to restore: the rollback must
    leave the file (never remove a file the failed attempt did not create). The retry equals the uninterrupted run."""
    from glossapi.stages import metadata as M
    sources = _sources(tmp_path)
    reference, output = tmp_path / 'reference', tmp_path / 'faulted'
    for root in (reference, output):
        seed_metadata(root, sources)
    assert Corpus(tmp_path / 'empty', reference).run(sources, stages=STAGES, settings=SETTINGS6)['state'] == 'completed'
    real_seal, real_copy, full = M.seal_metadata_projections, ps._copy_file_locked, []

    def seal(journal):
        if journal.stage == 'extract' and not full:
            full.append(True)
            raise OSError(errno.ENOSPC, 'injected: no space to seal the extract metadata projection')
        return real_seal(journal)

    def copy(source, destination, *args, **kwargs):
        if full:
            raise OSError(errno.ENOSPC, 'filesystem remains full during rollback')
        return real_copy(source, destination, *args, **kwargs)
    monkeypatch.setattr(M, 'seal_metadata_projections', seal)
    monkeypatch.setattr(ps, '_copy_file_locked', copy)
    with pytest.raises(DiskFullError) as stopped:
        Corpus(tmp_path / 'empty', output).run(sources, stages=STAGES, settings=SETTINGS6)
    monkeypatch.undo()
    assert (output / METADATA).is_file(), 'the seeded metadata the failed wrapper rewrote must not be removed'
    resumed = Corpus(tmp_path / 'empty', output).run(resume=stopped.value.details['run_id'], retry_failed=True)
    assert resumed['state'] == 'completed', resumed
    assert products(output) == products(reference)


KILL_CHILD = '''
import json, os, signal, sys, traceback
from pathlib import Path
from glossapi import Corpus
from glossapi.stages import execution as EX
from test_fix7_astra import seed_metadata

def main():
    root, step = Path(sys.argv[1]), sys.argv[2]
    sources = [Path(p) for p in sys.argv[3:]]
    settings = {'debug': 'off', 'extract': {'repair_fonts': False}}
    if step == 'start' and not (root / 'output').exists():
        (root / 'output' / 'download_results').mkdir(parents=True)
        seed_metadata(root / 'output', sources)  # before the Corpus, as the node kill matrix's seed()
    corpus = Corpus(root / 'empty', root / 'output')
    corpus._shadow_ledger_enabled = True  # as the node kill matrix's child
    if step == 'start':
        kill_at = int(os.environ.get('FIX7_KILL_LINK', '0'))
        count = {'n': 0}
        def counted(original):
            def publish(*args, **kwargs):
                value = original(*args, **kwargs)
                if kill_at and any(f.name == 'extract_compatibility_document' for f in traceback.extract_stack()):
                    count['n'] += 1
                    if count['n'] == kill_at:
                        os.kill(os.getpid(), signal.SIGKILL)
                return value
            return publish
        for name in ('link_published', 'copy_hashed'):
            setattr(EX, name, counted(getattr(EX, name)))
        result = corpus.run(sources, workers={'extract': 2}, stages=('extract', 'clean', 'export'), settings=settings)
        print(json.dumps(dict(state=result['state'])))
    else:
        from glossapi.ledger.compact import connect
        with connect(f'file:{root}/output/corpus.sqlite3?mode=ro', uri=True) as db:
            run = db.execute("SELECT run_id FROM tasks WHERE operation='extract.compatibility_publish' ORDER BY rowid DESC LIMIT 1").fetchone()[0]
        result = corpus.run(resume=run, retry_failed=True)
        print(json.dumps(dict(state=result['state'])))

if __name__ == '__main__':
    main()
'''


def _kill_child(root, step, sources=(), kill=0):
    import json, os, subprocess, sys
    root.mkdir(parents=True, exist_ok=True)
    (root / 'child.py').write_text(KILL_CHILD)
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'FIX7_KILL_LINK': str(kill),
           'GLOSSAPI_CONVERSION_LOG_FILE': str(root / 'conversion.log'),
           'PYTHONPATH': os.pathsep.join([*filter(None, [os.environ.get('PYTHONPATH')]), str(repo / 'src'), str(repo / 'tests')])}
    env.pop('GLOSSAPI_STAGE_PROFILE', None)
    done = subprocess.run([sys.executable, '-B', str(root / 'child.py'), str(root), step, *map(str, sources)],
                          capture_output=True, text=True, env=env, cwd=root, timeout=900)
    return done.returncode, done.stdout.strip().splitlines()[-1:] , done.stderr[-3000:]


def test_kill_at_an_extract_link_publication_with_seeded_metadata_resumes_to_identical_products(tmp_path):
    """The node kill matrix's publish_extract_link scenario, on the Mac: seeded metadata, 7 sources (extraction batches
    of 5), SIGKILL right after the 13th per-document extract publication (two per document: the 7th document, 2nd batch, after the 1st batch's in-place
    state save); the resume with --retry-failed completes with the uninterrupted run's products."""
    from test_k4fix1 import pdf
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    sources = [pdf(inputs / f'doc{index}.pdf', f'Native document number {index} for the extract link kill probe')
               for index in range(7)]
    rc, out, err = _kill_child(tmp_path / 'reference', 'start', sources)
    assert rc == 0, err
    rc, out, err = _kill_child(tmp_path / 'killed', 'start', sources, kill=13)
    assert rc == -9, (rc, out, err)
    rc, out, err = _kill_child(tmp_path / 'killed', 'resume')
    assert rc == 0, err
    assert products(tmp_path / 'killed' / 'output') == products(tmp_path / 'reference' / 'output')
    assert ledger_matches_files(tmp_path / 'killed' / 'output') == {}
