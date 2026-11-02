"""CP-K4 fix round 3 (CP-K4c3 R1–R4 and K4F2-G1 on 9b6eb6f).

Every reviewer probe of review_scratch/CP-K4c3/test_certify.py is ported here: the eight that failed on 9b6eb6f are
the acceptance set, the passing ones stay as regression guards. Each finding is fixed at its invariant and tested over
its domain (k4fix3/notes.md). Acquisition is the deterministic in-memory transport of test_k4fix1 (no sockets);
everything else is the real code.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

import pytest

from glossapi import Corpus
from glossapi.ledger import writer as W, journal as J
from glossapi.ledger.writer import LedgerWriter
from test_k4fix1 import transport, private_log, pdf, records, SETTINGS, STAGES, _fresh_ledger, _next_tick  # noqa: F401
from test_k4fix2 import _extreme, _sequential_download, _unique_bytes

THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')


def _bad_commit_checkpoint(path):
    """A real external connection commits a checksum-invalid snapshot, checkpoints and closes."""
    connection = sqlite3.connect(path, timeout=2)
    try:
        connection.execute('INSERT INTO snapshots VALUES(?,?,?,?)', (bytes.fromhex('f'*64), 'settings', 1, '{"kind":"settings","schema_version":1}'))
        connection.commit()
        return tuple(connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone())
    finally:
        connection.close()


def _accepted(path):
    """Does the next writer open the ledger (trusting it), and would an explicit full check reject it?"""
    try:
        with LedgerWriter(path) as writer:
            try:
                W.integrity(writer.connection)
                return True, None
            except W.LedgerError as error:
                return True, str(error)
    except W.LedgerError as error:
        return False, str(error)


# ===================================================== R2: close-time trust ===

@pytest.mark.parametrize('opening', ['full', 'cache'])
def test_r2_external_commit_after_close_version_read_never_trusted(tmp_path, monkeypatch, opening):
    """Reviewer reproducer (probe test_external_commit_after_close_version_read): the external commit and checkpoint
    land at the close's identity observation. The identity is now read before the last data_version comparison, so
    the commit changes data_version and nothing is trusted."""
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    if opening == 'full':
        W._integrity_known.pop(str(path.resolve()), None)
    writer = LedgerWriter(path)
    identity = W._ledger_identity
    def at_close(path_arg, *, allow_wal=False):
        if allow_wal and Path(path_arg) == path and not getattr(at_close, 'done', False):
            at_close.done = True
            _next_tick()
            _bad_commit_checkpoint(path)
        return identity(path_arg, allow_wal=allow_wal)
    monkeypatch.setattr(W, '_ledger_identity', at_close)
    writer.close()
    monkeypatch.setattr(W, '_ledger_identity', identity)
    assert at_close.done
    assert str(path.resolve()) not in W._integrity_known
    accepted, error = _accepted(path)
    assert not accepted and 'checksum' in error


@pytest.mark.parametrize('opening', ['full', 'cache'])
def test_r2_external_commit_after_version_comparison_never_trusted(tmp_path, monkeypatch, opening):
    """The other side of the fence: the commit lands after the data_version comparison, just before the post-close
    identity. The identity observed inside the fence no longer matches: nothing is trusted."""
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    if opening == 'full':
        W._integrity_known.pop(str(path.resolve()), None)
    writer = LedgerWriter(path)
    identity = W._ledger_identity
    def after_close(path_arg, *, allow_wal=False):
        if not allow_wal and Path(path_arg) == path and writer.connection is None and not getattr(after_close, 'done', False):
            after_close.done = True
            _next_tick()
            _bad_commit_checkpoint(path)
        return identity(path_arg, allow_wal=allow_wal)
    monkeypatch.setattr(W, '_ledger_identity', after_close)
    writer.close()
    monkeypatch.setattr(W, '_ledger_identity', identity)
    assert after_close.done
    assert str(path.resolve()) not in W._integrity_known
    accepted, error = _accepted(path)
    assert not accepted and 'checksum' in error


def test_r2_incomplete_checkpoint_never_trusted(tmp_path, monkeypatch):
    """An open reader keeps the WAL: the owner's checkpoint is incomplete, so no identity is a trust candidate."""
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    writer = LedgerWriter(path)
    writer.connection.execute('PRAGMA busy_timeout=50')  # the blocked checkpoint returns quickly
    reader = sqlite3.connect(path, timeout=0.05)
    reader.execute('BEGIN')
    reader.execute('SELECT COUNT(*) FROM snapshots').fetchone()
    with writer.transaction():
        writer.run({'after_reader': True}, {}, {}, code_revision='probe')
    try:
        writer.close()
    finally:
        reader.rollback()
        reader.close()
    assert str(path.resolve()) not in W._integrity_known
    assert _accepted(path) == (True, None)  # a genuine ledger still opens (full check)


def test_swap_during_writer(tmp_path):
    """Guard (probe): an ordinary file replacement while a writer is open installs no trust and is rejected."""
    path = tmp_path/'corpus.sqlite3'
    other = tmp_path/'other'/'corpus.sqlite3'
    other.parent.mkdir()
    _fresh_ledger(path)
    _fresh_ledger(other)
    _bad_commit_checkpoint(other)
    writer = LedgerWriter(path)
    os.replace(other, path)
    try:
        writer.close()
    except Exception:
        pass
    assert str(path.resolve()) not in W._integrity_known
    with pytest.raises(W.LedgerError):
        LedgerWriter(path)


# =========================================== R4: original re-verification lifetime ===

def _three_shards(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    sources = [pdf(root/'downloads'/f'{name}.pdf', 'Native text enough for cleaning '+name) for name in ['alpha', 'beta', 'gamma']]
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=sources, repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    shards = root/'shards'
    initial = corpus.jsonl_sharded(shards, shard_size_bytes=1, compression='none')
    return root, sources, shards, initial


def _shard_deletions(monkeypatch, after):
    """Run the real RetentionController.delete; call ``after(n)`` once the n-th shard deletion has returned."""
    from glossapi.stages import retention
    real = retention.RetentionController.delete
    observed = []
    def deletion(self, file_id, **kwargs):
        row = self.writer.file_catalog()[file_id]
        result = real(self, file_id, **kwargs)
        if row['kind'] == 'document_jsonl_shard' and result:
            observed.append(row['uri'])
            after(len(observed))
        return result
    monkeypatch.setattr(retention.RetentionController, 'delete', deletion)
    return observed


def _wrapper_states(root):
    with W.open_readonly(root/'corpus.sqlite3') as connection:
        return [r[0] for r in connection.execute("SELECT status FROM current_task_state WHERE operation='export.jsonl_sharded'")]


def test_r4_second_deletion_rechecks_changed_original(tmp_path, monkeypatch):
    """Reviewer reproducer (probe test_second_deletion_rechecks_changed_original): the original changes (100 ms later,
    larger) right after the first shard deletion. The second deletion re-stats it, re-hashes the changed file and
    refuses; the second old shard stays and the wrapper records the failure."""
    root, sources, shards, initial = _three_shards(tmp_path, monkeypatch)
    def edit(count):
        if count == 1:
            _next_tick()
            sources[0].write_bytes(sources[0].read_bytes()+b'\nEXTERNAL EDIT BETWEEN DELETIONS\n')
    observed = _shard_deletions(monkeypatch, edit)
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        Corpus(tmp_path/'empty', root).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert len(observed) == 1
    assert sum(1 for path in initial if path.exists()) == 2
    assert _wrapper_states(root)[-1] == 'failed'


def test_r4_pre_outcome_rechecks_changed_original(tmp_path, monkeypatch):
    """An edit after the last deletion is caught at the pre-outcome frontier: the wrapper outcome is the failure."""
    root, sources, shards, initial = _three_shards(tmp_path, monkeypatch)
    def edit(count):
        if count == 2:
            _next_tick()
            sources[1].write_bytes(sources[1].read_bytes()+b'\nEDIT AFTER THE LAST DELETION\n')
    observed = _shard_deletions(monkeypatch, edit)
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        Corpus(tmp_path/'empty', root).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert len(observed) == 2
    assert _wrapper_states(root)[-1] == 'failed'


def test_r4_unchanged_original_hashed_once_across_frontiers(tmp_path, monkeypatch):
    """Cost: the first verification of an original hashes it; later frontiers re-stat it and hash nothing."""
    root, sources, shards, initial = _three_shards(tmp_path, monkeypatch)
    observed = _shard_deletions(monkeypatch, lambda count: None)
    hashed, verifying = [], []
    real_hash = J.file_hash
    def counting(path, *args, **kwargs):
        if verifying:  # only the guard's own byte verification (the wrapper inventory hashes the output tree anyway)
            hashed.append(str(path))
        return real_hash(path, *args, **kwargs)
    monkeypatch.setattr(J, 'file_hash', counting)
    verifications = []
    real_verify = J.ResultJournal.verify_cached_originals
    def verify(self, rows=None):
        verifications.append(1)
        verifying.append(1)
        try:
            return real_verify(self, rows)
        finally:
            verifying.pop()
    monkeypatch.setattr(J.ResultJournal, 'verify_cached_originals', verify)
    Corpus(tmp_path/'empty', root).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert len(observed) == 2 and len(verifications) >= 3  # two deletions and the pre-outcome frontier
    assert hashed and all(hashed.count(str(source)) <= 1 for source in sources), hashed


# ============================================================ K4F2-G1 thread variables ===

def test_k4f2g1_same_process_resume_after_extraction(tmp_path, monkeypatch):
    """Reviewer probe (test_same_process_resume_environment): with the thread variables unset, a run leaves them
    unset and a same-process resume passes the frozen-environment comparison."""
    for name in THREAD_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    source = pdf(tmp_path/'doc.pdf', 'Native document to demonstrate environment mutation')
    root = tmp_path/'out'
    result = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES)
    Corpus(tmp_path/'empty', root).run(resume=result['run_id'])
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    Corpus(tmp_path/'empty', tmp_path/'seq').extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1)
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES)


def test_k4f2g1_user_thread_settings_kept(tmp_path, monkeypatch):
    """The user's own values survive extraction and are the recorded (frozen) environment."""
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, '3')
    source = pdf(tmp_path/'doc.pdf', 'Native document with user thread settings')
    root = tmp_path/'out'
    result = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES, '3')
    with W.open_readonly(root/'corpus.sqlite3') as connection:
        digest = connection.execute('SELECT environment_sha256 FROM runs WHERE run_id=?', (result['run_id'],)).fetchone()[0]
        frozen = json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?', (digest,)).fetchone()[0])
    assert all(frozen['environment'][name] == '3' for name in THREAD_VARIABLES)
    Corpus(tmp_path/'empty', root).run(resume=result['run_id'])


# ================================================================= R1: sequential provenance ===

def _sequential_export(root, corpus, source):
    corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1, skip_existing=False, batched=False)
    corpus.clean(drop_bad=False, num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    return records(root)


def test_r1_sequential_remote_then_local_same_basename_refused(tmp_path, monkeypatch):
    """Reviewer sequence (probe test_sequential_provenance_sequence). A different local input named like the earlier
    download is refused with a typed error before anything is written: the remote text, its products and its URL row
    stay exactly as they were, and no exported row pairs the local text with the remote URL. The rest of the sequence
    (changed bytes at the same URL, another URL, re-adding the URL) keeps each source's name and URL."""
    from glossapi.stages.compatibility import PublicationSourceConflictError
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    url, url2 = 'http://fixture.invalid/doc.pdf', 'http://fixture.invalid/second.pdf'
    corpus, remote = _sequential_download(tmp_path, monkeypatch, root, url, 'REMOTE first content')
    rows = _sequential_export(root, corpus, remote)
    remote_row = next(r for r in rows if 'REMOTE first' in r['document'])
    assert remote_row['url'] == url
    products = {path: path.read_bytes() for path in [*root.glob('markdown/*'), *root.glob('clean_markdown/*'), *root.glob('sidecars/**/*')] if path.is_file()}
    local = pdf(tmp_path/'local'/remote.name, 'LOCAL same basename content')
    with pytest.raises(PublicationSourceConflictError, match='another source'):
        Corpus(tmp_path/'empty', root).extract(file_paths=[local], repair_fonts=False, accel_type='CPU', num_threads=1, skip_existing=False, batched=False)
    assert {path: path.read_bytes() for path in products} == products  # nothing replaced
    corpus = Corpus(tmp_path/'empty', root)
    corpus.clean(drop_bad=False, num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    assert not any('LOCAL same' in r['document'] for r in records(root))
    assert next(r for r in records(root) if 'REMOTE first' in r['document'])['url'] == url
    history = []
    for label, address, body in [('changed', url, 'REMOTE changed bytes'), ('other', url2, 'OTHER remote text'), ('readd', url, 'REMOTE readded text')]:
        corpus, source = _sequential_download(tmp_path, monkeypatch, root, address, body)
        pdf(source, body)  # a sequential download resumes a successful URL by design; give the acquired file new bytes
        # (bodies stay short: the fixture PDF wraps longer lines)
        exported = _sequential_export(root, corpus, source)
        found = next((r for r in exported if body in r['document']), None)
        assert found is not None, (label, source.name, [(r['filename'], r['url'], r['document'][:40]) for r in exported])
        history.append((label, found['filename'], found['url']))
        assert found['url'] == address
        if address == url:
            assert found['filename'] == remote_row['filename']


def test_r1_export_refuses_a_foreign_metadata_row(tmp_path, monkeypatch):
    """Export-side enforcement on its own: with the extraction refusal bypassed (the pre-fix extraction), the local
    text reaches clean_markdown under the remote name; the sequential export binds it to its ledger source and refuses
    the remote URL row with export_metadata_identity_mismatch."""
    from glossapi.stages import compatibility
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    url = 'http://fixture.invalid/doc.pdf'
    corpus, remote = _sequential_download(tmp_path, monkeypatch, root, url, 'REMOTE first content')
    _sequential_export(root, corpus, remote)
    monkeypatch.setattr(compatibility, 'refuse_foreign_publication', lambda *args, **kwargs: None)
    local = pdf(tmp_path/'local'/remote.name, 'LOCAL same basename content')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[local], repair_fonts=False, accel_type='CPU', num_threads=1, skip_existing=False, batched=False)
    corpus.clean(drop_bad=False, num_threads=1)
    with pytest.raises(ValueError, match='name a different source'):
        corpus.jsonl(root/'export.jsonl')
    assert 'LOCAL same' not in (root/'export.jsonl').read_text()


def test_r1_caller_url_on_local_input_is_kept(tmp_path, monkeypatch):
    """Caller-supplied URLs on local input are not cleared: no download ever gave that name to another source."""
    import pandas as pd
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    local = pdf(tmp_path/'local'/'caller.pdf', 'LOCAL text with caller metadata')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[local], repair_fonts=False, accel_type='CPU', num_threads=1, skip_existing=False, batched=False)
    corpus.clean(drop_bad=False, num_threads=1)
    metadata = tmp_path/'caller.parquet'
    pd.DataFrame({'filename': ['caller.pdf'], 'url': ['http://caller.invalid/origin.pdf']}).to_parquet(metadata, index=False)
    corpus.jsonl(root/'export.jsonl', metadata_path=metadata)
    row, = records(root)
    assert row['url'] == 'http://caller.invalid/origin.pdf'


@pytest.mark.parametrize('mutation', ['url', 'resource_identity', 'missing_identity'])
def test_fold_identity_protection(tmp_path, mutation):
    """Guard (probe): the download fold never replaces another source's URL or identity."""
    import pandas as pd
    from glossapi.parquet_schema import fold_download_rows
    directory = tmp_path/'download_results'
    directory.mkdir()
    canonical, incoming = directory/'download_results.parquet', directory/'download_results_input.parquet'
    old = {'filename': 'same.pdf', 'url': 'http://fixture.invalid/a.pdf', 'resource_identity': 'url:v1:http://fixture.invalid/a.pdf',
           'download_success': True, 'description': 'original'}
    new = {**old, 'description': 'replacement'}
    if mutation == 'url':
        new['url'] = 'http://fixture.invalid/b.pdf'
    elif mutation == 'resource_identity':
        new['resource_identity'] = 'other-identity'
    else:
        new['url'] = None
        new['resource_identity'] = None
    pd.DataFrame([old]).to_parquet(canonical, index=False)
    pd.DataFrame([new]).to_parquet(incoming, index=False)
    fold_download_rows(tmp_path, incoming)
    row = pd.read_parquet(canonical).iloc[0].to_dict()
    assert row['url'] == old['url'] and row['resource_identity'] == old['resource_identity']
    if mutation != 'missing_identity':
        assert row == old


def test_sequential_readd_and_changed_bytes(tmp_path, monkeypatch):
    """Guard (probe): changed acquired bytes at the same URL keep the URL and the name."""
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    url = 'http://fixture.invalid/doc.pdf'
    history = []
    for index, body in enumerate(['REMOTE first version', 'REMOTE changed version', 'REMOTE third version']):
        corpus, source = _sequential_download(tmp_path, monkeypatch, root, url, body)
        if index:
            pdf(source, body)
        row = next(r for r in _sequential_export(root, corpus, source) if body in r['document'])
        history.append(row['filename'])
        assert row['url'] == url and row['filename'] == history[0]


def test_streaming_amended_provenance_and_changed_url_bytes(tmp_path, transport):
    """Guard (probe): streaming local and remote same-basename re-adds keep their own names, URLs and new text."""
    root = tmp_path/'out'
    url = 'http://fixture.invalid/doc.pdf'
    transport[url] = pdf(tmp_path/'remote.pdf', 'REMOTE version first').read_bytes()
    Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=SETTINGS)
    name = records(root)[0]['filename']
    local = pdf(tmp_path/'local'/(name+'.pdf'), 'LOCAL separate identity')
    Corpus(tmp_path/'empty', root).run([local], stages=STAGES, settings=SETTINGS)
    local_name = records(root)[0]['filename']
    assert local_name != name and records(root)[0]['url'] is None
    for body in ['REMOTE version second', 'REMOTE version third']:
        transport[url] = pdf(tmp_path/'remote.pdf', body).read_bytes()
        Corpus(tmp_path/'empty', root).run([url], stages=STAGES, settings=SETTINGS)
        row = records(root)[0]
        assert row['filename'] == name and row['url'] == url and body in row['document']
    Corpus(tmp_path/'empty', root).run([local], stages=STAGES, settings=SETTINGS)
    assert records(root)[0]['filename'] == local_name and records(root)[0]['url'] is None


# ============================================================== R3: publication charge ===

def test_r3_unselected_metrics_not_copied_and_charged(tmp_path, monkeypatch):
    """Reviewer layout 1 (probe test_large_unselected_metrics_breaks_export_bound): a valid 1.1 MB per-page metrics
    sidecar of an unselected document is no longer an export input (nothing reads it), and the bytes the export adds
    while it copies its real inputs stay within the export re-check's charge."""
    from glossapi.stages import run as RUN, export as EX
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 1 << 30)
    root = tmp_path/'out'
    root.mkdir()
    metrics = root/'json'/'metrics'
    metrics.mkdir(parents=True)
    unselected = metrics/'unselected.per_page.metrics.json'
    unselected.write_text(json.dumps({'pages': [{'page_no': i, 'char_count': 4000, 'word_count': 1000, 'non_whitespace_chars': 3900} for i in range(6000)]}))
    observed = {}
    real_barrier, real_finalize = RUN.publish_barrier, EX.ExportStage._finalize
    def barrier(corpus, run, stage, callback, **kwargs):
        if stage == 'export':
            observed['base'] = _unique_bytes(root)
            observed['charge'] = checks[-1]['need']
        return real_barrier(corpus, run, stage, callback, **kwargs)
    def finalize(stage, request, view_root):
        observed['inputs'] = [Path(relative).name for relative in request.settings['input_paths']]
        real_copy = shutil.copyfile
        def copy(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(dst).is_relative_to(view_root):
                observed['peak'] = max(observed.get('peak', 0), _unique_bytes(root)-observed['base'])
            return result
        with monkeypatch.context() as patch:
            patch.setattr(shutil, 'copyfile', copy)
            return real_finalize(stage, request, view_root)
    monkeypatch.setattr(RUN, 'publish_barrier', barrier)
    monkeypatch.setattr(EX.ExportStage, '_finalize', finalize)
    result = Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'doc.pdf', 'Short document in the selected corpus')], stages=STAGES, settings=settings)
    assert result['state'] == 'completed'
    assert unselected.name not in observed['inputs']
    assert observed['charge'] >= observed['peak'] > 0
    assert observed['charge'] < unselected.stat().st_size  # the unselected sidecar is neither copied nor charged


@pytest.mark.parametrize('stages,n', [(('extract', 'clean', 'sections'), 1), (('extract', 'clean', 'sections', 'annotate', 'export'), 8)],
                         ids=['sections-1', 'both-products-8'])
def test_r3_small_documents_bound(tmp_path, monkeypatch, stages, n):
    """Reviewer layouts 2 and 3 (probe test_small_documents_sections_and_both_products_bound): the charge before the
    publications covers what every publication adds (fixed, per-row and schema overhead included), and each later
    re-check covers what remains."""
    from glossapi.stages import run as RUN
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 1 << 30)
    root = tmp_path/'out'
    measured = {}
    real = RUN.publish_barrier
    def barrier(corpus, run, stage, callback, **kwargs):
        before = _unique_bytes(root)
        measured.setdefault('base', before)
        measured.setdefault('rechecks', []).append((stage, before, checks[-1]))
        try:
            return real(corpus, run, stage, callback, **kwargs)
        finally:
            measured[stage] = _unique_bytes(root)-measured['base']
    monkeypatch.setattr(RUN, 'publish_barrier', barrier)
    sources = [pdf(tmp_path/f'doc{i}.pdf', f'Document {i} small text sufficient for native cleaning and short section.') for i in range(n)]
    result = Corpus(tmp_path/'empty', root).run(sources, stages=stages, settings=settings)
    assert result['state'] == 'completed'
    charge = next(check['need'] for check in checks if check.get('when') == '(exact, before the publications)')
    peak = max(value for key, value in measured.items() if key in stages)
    assert charge >= peak
    final = _unique_bytes(root)
    for stage, before, check in measured['rechecks']:
        assert check['when'] == f'(exact, at the {stage} publication)'
        assert check['need'] >= final-before  # each re-check covers every publication still to come


def test_r3_each_publication_rechecked(tmp_path, monkeypatch):
    """A check runs before the first publication and again before every publication, in publication order."""
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 1 << 30)
    stages = ('extract', 'clean', 'sections', 'annotate', 'export')
    Corpus(tmp_path/'empty', tmp_path/'out').run([pdf(tmp_path/'doc.pdf', 'Document for every publication')], stages=stages, settings=settings)
    exact = [check['when'] for check in checks if 'exact' in check.get('when', '')]
    assert exact == ['(exact, before the publications)', *(f'(exact, at the {stage} publication)' for stage in stages)]
    needs = [check['need'] for check in checks if 'exact' in check.get('when', '')]
    assert needs[-1] > 0


def test_r3_export_inputs_are_the_selected_documents(tmp_path, monkeypatch):
    """One input list for the writer and the charge: only the sidecars the serializer reads for the selected texts."""
    from glossapi.stages.export import export_inputs
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    source = pdf(tmp_path/'kept.pdf', 'Selected document text for the export')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    metrics = root/'json'/'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    for name in ('kept.metrics.json', 'kept.per_page.metrics.json', 'other.per_page.metrics.json', 'other.metrics.json'):
        (metrics/name).write_text(json.dumps({'pages': []}))
    (root/'json'/'other.latex_map.jsonl').write_text('')
    inputs, relative, texts = export_inputs(corpus, {})
    assert [stem for stem, _ in texts] == ['kept']
    assert sorted(Path(name).name for name in relative if name.startswith('json/')) == ['kept.metrics.json', 'kept.per_page.metrics.json']
    corpus.jsonl(root/'export.jsonl')
    with W.open_readonly(root/'corpus.sqlite3') as connection:
        snapshots = [Path(row[0]).name for row in connection.execute(
            "SELECT f.uri FROM files f JOIN attempts a ON a.attempt_id=f.producer_attempt_id JOIN tasks t ON t.task_id=a.task_id "
            "WHERE t.operation='export.publish' AND f.kind='source_snapshot'")]
    assert 'other.per_page.metrics.json' not in snapshots and 'kept.per_page.metrics.json' in snapshots


def test_first_run_refusal_minimum(tmp_path, monkeypatch):
    """Guard (probe): a first run with 16 KiB of room is refused before the publications, need >= 32,318 B."""
    from glossapi.stages.disk_budget import DiskBudgetError
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 16384)
    root = tmp_path/'out'
    with pytest.raises(DiskBudgetError, match='exact, before the publications'):
        Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'doc.pdf', 'A short text document')], stages=STAGES, settings=settings)
    exact = next(check for check in checks if check.get('when') == '(exact, before the publications)')
    assert exact['need'] >= 32318 and not (root/'export.jsonl').exists()


# ======================================================= guards: post-close edit refusals ===

@pytest.mark.parametrize('phase', ['download', 'extract', 'clean', 'section', 'jsonl', 'jsonl_sharded'])
def test_each_sequential_start_refuses_repeatedly(tmp_path, monkeypatch, phase):
    """Guard (probe): an unresolved post-close edit refuses each sequential start twice and keeps the edited bytes."""
    import pandas as pd
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    source = pdf(root/'downloads'/'doc.pdf', 'Preserve edited bytes between all sequential phases')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    _next_tick()
    edited = b'USER BYTES REMAIN\n'
    (root/'export.jsonl').write_bytes(edited)
    manifest = tmp_path/'manifest.parquet'
    pd.DataFrame({'url': ['http://fixture.invalid/a.pdf']}).to_parquet(manifest, index=False)
    calls = {'download': lambda c: c.download(manifest, rate_limit=0),
             'extract': lambda c: c.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1),
             'clean': lambda c: c.clean(drop_bad=False, num_threads=1), 'section': lambda c: c.section(),
             'jsonl': lambda c: c.jsonl(root/'export.jsonl'), 'jsonl_sharded': lambda c: c.jsonl_sharded(root/'shards', compression='none')}
    for _ in range(2):
        with pytest.raises(W.PostCloseEditError):
            calls[phase](Corpus(tmp_path/'empty', root))
        assert (root/'export.jsonl').read_bytes() == edited


def test_remaining_sequential_phases_refuse(tmp_path, monkeypatch):
    """Guard (probe): every remaining sequential entry refuses an unresolved post-close edit, twice."""
    import inspect
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    source = pdf(root/'downloads'/'doc.pdf', 'Edited copy refuses remaining sequential entries')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    _next_tick()
    (root/'export.jsonl').write_bytes(b'KEPT USER BYTES\n')
    phases = ['ocr', 'annotate', 'triage_math', 'formula_enrich_from_json', 'clean_ocr', 'clean_ocr_debug', 'clean_ocr_numeric_debug',
              'clean_token_category_debug', 'clean_ocr_numeric_word_debug_docs', 'clean_ocr_hybrid_debug',
              'clean_ocr_latex_slot_progression_debug', 'clean_ocr_latex_debug']
    for phase in phases:
        for _ in range(2):
            function = getattr(Corpus(tmp_path/'empty', root), phase)
            arguments = {name: tmp_path/('required-'+name) for name, parameter in inspect.signature(function).parameters.items()
                         if parameter.default is inspect.Parameter.empty and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}
            with pytest.raises(W.PostCloseEditError):
                function(**arguments)
            assert (root/'export.jsonl').read_bytes() == b'KEPT USER BYTES\n'


def test_relative_migration_alias(tmp_path, monkeypatch):
    """Guard (probe): a relative alias of the live ledger is refused as its own migration backup."""
    from glossapi.ledger.schema import FORMAT_SCHEMA_SQL
    path = tmp_path/'corpus.sqlite3'
    (tmp_path/'sub').mkdir()
    monkeypatch.chdir(tmp_path)
    with LedgerWriter(path, storage_version=3) as writer:
        with pytest.raises(W.LedgerError, match='live ledger itself'):
            writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest(), code_revision='test',
                           backup_path=Path('sub/../corpus.sqlite3'))
        assert writer.connection.execute('PRAGMA user_version').fetchone()[0] == 3
