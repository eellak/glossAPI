"""CP-K4 fix round 2 (CP-K4ac2 R1–R3, CP-K4bc2 R1–R4 on 15afe32). The reviewer reproducers are ported here.

Each finding is tested on every entry path it touches (streaming new/over/resume, sequential calls, recovery, bulk
release): see the entry-path × boundary table in reviews/CP-K4_reply_2.md. Acquisition is the deterministic in-memory
transport of test_k4fix1 (no sockets); everything else is the real code.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from glossapi import Corpus
from glossapi.ledger import writer as W, journal as J
from glossapi.ledger.writer import LedgerWriter
from test_k4fix1 import transport, private_log, pdf, records, SETTINGS, STAGES, _bad_snapshot, _fresh_ledger, _next_tick  # noqa: F401


# ------------------------------------------------- A-R1 sequential download provenance ---

def _sequential_download(tmp_path, monkeypatch, root, url, text):
    import pandas as pd
    from glossapi.gloss_downloader import GlossDownloader
    from glossapi.ledger.journal import observe_acquisition
    body = pdf(tmp_path/(text.split()[0].lower()+'.pdf'), text).read_bytes()

    async def download_file(self, row_index, url, semaphore, rate_limiter, retry_count=0, filename_base=None, referer=None):
        target = self.downloads_dir/(filename_base+'.pdf')
        target.write_bytes(body)
        observe_acquisition(url, target, source_observable=True)
        return True, target.name, 'pdf', '', 0
    monkeypatch.setattr(GlossDownloader, 'download_file', download_file)
    manifest = tmp_path/'manifest.parquet'
    pd.DataFrame({'url': [url]}).to_parquet(manifest, index=False)
    corpus = Corpus(tmp_path/'empty', root)
    downloaded = corpus.download(manifest, rate_limit=0)
    return corpus, root/'downloads'/downloaded[downloaded['url'] == url].iloc[0]['filename']


def test_r1_sequential_local_then_remote_keeps_url(tmp_path, monkeypatch):
    """Reviewer reproducer: local run, then public sequential download → extract → clean → jsonl."""
    root = tmp_path/'out'
    Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'local'/'AAA_000.pdf', 'LOCAL sequential text')], stages=STAGES, settings=SETTINGS)
    url = 'http://fixture.invalid/remote.pdf'
    corpus, source = _sequential_download(tmp_path, monkeypatch, root, url, 'REMOTE sequential text')
    corpus.extract(file_paths=[source], input_format='pdf', phase1_backend='safe', accel_type='CPU',
                   num_threads=1, skip_existing=False, batched=False, repair_fonts=False)
    corpus.clean(num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    remote, = [r for r in records(root) if 'REMOTE sequential' in r['document']]
    assert remote['url'] == url, 'sequential download row never reached the canonical table'


def test_r1_sequential_fold_never_overwrites_another_url(tmp_path):
    """Identity-safe: a canonical row that already records a different URL for the filename is kept as it is."""
    import pandas as pd
    from glossapi.parquet_schema import fold_download_rows
    directory = tmp_path/'download_results'
    directory.mkdir()
    pd.DataFrame({'filename': ['AAA_000.pdf', 'AAB_001.pdf'], 'url': ['http://x/first.pdf', None]}).to_parquet(directory/'download_results.parquet', index=False)
    published = directory/'download_results_sources.parquet'
    pd.DataFrame({'filename': ['AAA_000.pdf', 'AAB_001.pdf', 'AAC_002.pdf'], 'url': ['http://x/other.pdf', 'http://x/b.pdf', 'http://x/c.pdf'],
                  'download_success': [True, True, True]}).to_parquet(published, index=False)
    conflicts = fold_download_rows(tmp_path, published)
    table = pd.read_parquet(directory/'download_results.parquet').set_index('filename')
    assert conflicts == ['AAA_000.pdf']
    assert table.loc['AAA_000.pdf', 'url'] == 'http://x/first.pdf'  # never replaced by another URL
    assert table.loc['AAB_001.pdf', 'url'] == 'http://x/b.pdf' and table.loc['AAC_002.pdf', 'url'] == 'http://x/c.pdf'


# ----------------------------------------------------- A-R2 unresolved post-close edits ---

@pytest.mark.parametrize('mode', ['new_run', 'resume'])
def test_r2_post_close_edit_refuses_and_keeps_copy(tmp_path, mode):
    """Reviewer reproducer, both entries: a new run over the corpus and a resume."""
    root = tmp_path/'out'
    source = pdf(tmp_path/'doc.pdf', 'Preserve postclose edit text')
    first = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    copy = root/'export.jsonl'
    edited = b'USER EDIT AFTER CLOSE\n'
    _next_tick()  # outside the declared same-tick window (Linux coarse mtime clock)
    copy.write_bytes(edited)
    corpus = Corpus(tmp_path/'empty', root)
    with pytest.raises(W.PostCloseEditError, match='post_close_edit'):
        corpus.run(resume=first['run_id']) if mode == 'resume' else corpus.run([source], stages=STAGES, settings=SETTINGS)
    assert copy.read_bytes() == edited


def test_r2_sequential_phase_refuses_and_keeps_copy(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    root = tmp_path/'out'
    source = pdf(root/'downloads'/'doc.pdf', 'Sequential postclose edit text')
    corpus = Corpus(tmp_path/'empty', root)
    corpus.extract(file_paths=[source], repair_fonts=False, accel_type='CPU', num_threads=1)
    corpus.clean(drop_bad=False, num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    copy = root/'export.jsonl'
    _next_tick()  # outside the declared same-tick window (Linux coarse mtime clock)
    copy.write_bytes(b'USER EDIT\n')
    with pytest.raises(W.PostCloseEditError):
        Corpus(tmp_path/'empty', root).jsonl(root/'export.jsonl')
    assert copy.read_bytes() == b'USER EDIT\n'


def test_r2_refusal_persists_until_resolved(tmp_path):
    root = tmp_path/'out'
    source = pdf(tmp_path/'doc.pdf', 'Persistent refusal text')
    Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    copy = root/'export.jsonl'
    _next_tick()  # outside the declared same-tick window (Linux coarse mtime clock)
    copy.write_bytes(b'USER EDIT\n')
    for _ in range(2):  # a refused start (and its writer close) never reclassifies the edit
        with pytest.raises(W.PostCloseEditError):
            Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
        assert copy.read_bytes() == b'USER EDIT\n'
    copy.unlink()  # the caller resolves it: the copy is republished from the managed source
    result = Corpus(tmp_path/'empty', root).run([source], stages=STAGES, settings=SETTINGS)
    assert result['state'] == 'completed' and 'Persistent refusal' in records(root)[0]['document']


# --------------------------------------------- A-R3 metadata created by first-run publication ---

def _extreme(tmp_path, monkeypatch, room):
    from glossapi.stages import final_products, disk_budget
    # The extract stage pins these thread variables in the process; preset them so that a resume in the same test
    # process sees the frozen environment whether or not an earlier test ran extraction (test order independence).
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(disk_budget, 'require_disk_guard', lambda root, settings: (Path(root)/'.glossapi/staging').mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(final_products, 'corpus_room', lambda *a: room())
    checks = []
    original = final_products.require_room
    def checked(root, settings, need, **kwargs):
        checks.append({'need': need, **kwargs})
        return original(root, settings, need, **kwargs)
    monkeypatch.setattr(final_products, 'require_room', checked)
    return checks, {**SETTINGS, 'memory_saving': 'extreme', 'disk_budget_bytes': 1 << 30, 'max_source_bytes': 8192, 'max_attempt_bytes': 16384}


def test_r3_first_run_counts_the_future_metadata_table(tmp_path, monkeypatch):
    """Reviewer reproducer: 16 KiB of room, a fresh corpus. The table publication creates (measured 29,343 bytes) and
    its export working copy are charged before the publications, so the run is refused instead of completing."""
    from glossapi.stages.disk_budget import DiskBudgetError
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: 16384)
    root = tmp_path/'out'
    with pytest.raises(DiskBudgetError, match='exact, before the publications'):
        Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'doc.pdf', 'A short text document')], stages=STAGES, settings=settings)
    assert not (root/'export.jsonl').exists()
    exact = next(check for check in checks if 'exact' in check.get('when', ''))
    assert exact['need'] >= 32318  # the reviewer's measured minimum: 2 x 1,477 export + 29,343 table copy + 21 text


def test_r3_export_barrier_recheck_uses_the_actual_table(tmp_path, monkeypatch):
    """With the allowance removed, the pre-publication check passes; the re-check at the export publication reads the
    table that now exists and refuses before the export writes. Resume after enlarging the room completes."""
    from glossapi.stages import final_products
    from glossapi.stages.disk_budget import DiskBudgetError
    monkeypatch.setattr(final_products, 'METADATA_TABLE_FIXED_BYTES', 0)
    monkeypatch.setattr(final_products, 'METADATA_TABLE_ROW_BYTES', 0)
    room = {'bytes': 20000}
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: room['bytes'])
    root = tmp_path/'out'
    with pytest.raises(DiskBudgetError, match='at the export publication'):
        Corpus(tmp_path/'empty', root).run([pdf(tmp_path/'doc.pdf', 'A short text document')], stages=STAGES, settings=settings)
    assert [check['when'] for check in checks if 'exact' in check['when']] == ['(exact, before the publications)', '(exact, at the export publication)']
    assert not (root/'export.jsonl').exists()
    room['bytes'] = 1 << 30
    from glossapi.ledger.reader import select_run
    with W.open_readonly(root/'corpus.sqlite3') as connection:
        run = select_run(connection)['run_id']
    assert Corpus(tmp_path/'empty', root).run(resume=run)['state'] == 'completed'
    assert (root/'export.jsonl').exists()


def _unique_bytes(root):
    """Bytes of the corpus directory, hard links once; the ledger is charged to the emergency reserve, not here."""
    seen, total = set(), 0
    for directory, _, names in os.walk(root):
        for name in names:
            if name.startswith('corpus.sqlite3'):
                continue
            state = os.lstat(os.path.join(directory, name))
            if (state.st_dev, state.st_ino) not in seen:
                seen.add((state.st_dev, state.st_ino))
                total += state.st_size
    return total


def test_r3_export_recheck_counts_each_text_once(tmp_path, monkeypatch):
    """Node gate reproducer (corpus > budget, extreme): at the export publication the clean publication's compatibility
    copies are present rows of the same texts; counting them again over-charged the export (1,166,291,727 needed vs
    1,147,092,992 available; the run fits with a 0.98 GB high-water). Room between the export's actual need and the
    double count: the run completes, and the bytes the export adds stay within the charged bound."""
    from glossapi.stages import final_products, run as RUN
    measured = {}
    real_need = final_products.exact_export_need
    def need(texts, metadata, stems):
        texts = list(texts)
        value = real_need(texts, metadata, stems)
        if 'export publication' in phase.get('when', ''):  # the export re-check, after the clean publication
            copies = sorted((tmp_path/'out'/'clean_markdown').glob('*.md'))
            single, doubled = real_need(copies, metadata, stems), real_need(copies*2, metadata, stems)
            measured.update(texts=len(texts), copies=len(copies), need=value, single=single, doubled=doubled)
            room['bytes'] = (single + doubled)//2  # fixed by the test, not by the code under test
        return value
    monkeypatch.setattr(final_products, 'exact_export_need', need)
    phase = {}
    real_check = final_products.check_exact
    def check(corpus, result, stages, *, when):
        phase['when'] = when
        return real_check(corpus, result, stages, when=when)
    monkeypatch.setattr(final_products, 'check_exact', check)
    room = {'bytes': 1 << 40}
    checks, settings = _extreme(tmp_path, monkeypatch, lambda: room['bytes'])
    real_barrier = RUN.publish_barrier
    def barrier(corpus, run, stage, callback, **kwargs):
        before = _unique_bytes(tmp_path/'out')
        try:
            return real_barrier(corpus, run, stage, callback, **kwargs)
        finally:
            if stage == 'export':
                measured['added'] = _unique_bytes(tmp_path/'out') - before
    monkeypatch.setattr(RUN, 'publish_barrier', barrier)
    sources = [pdf(tmp_path/'src'/f'D{index:03d}.pdf', f'Document {index} export recheck text') for index in range(3)]
    result = Corpus(tmp_path/'empty', tmp_path/'out').run(sources, stages=STAGES, settings=settings)  # 15afe32+r2: refused
    assert result['state'] == 'completed'
    assert measured['texts'] == 3 and measured['copies'] == 3  # one text per document, although both copies are present
    assert measured['need'] == measured['single'] < room['bytes'] < measured['doubled']
    assert 0 < measured['added'] <= measured['need']  # still an upper bound of what the export publication adds
    assert any('export publication' in check.get('when', '') for check in checks)


# ------------------------------------------- B-R1 external commits over the whole lifetime ---

def test_k4b_r1_commit_after_full_check_never_trusted(tmp_path, monkeypatch):
    """Reviewer reproducer: an external commit lands right after the full check passes, before initialisation ends."""
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    W._integrity_known.pop(str(path.resolve()), None)
    real = W.integrity
    def checked_then_edit(connection):
        real(connection)
        if not getattr(checked_then_edit, 'done', False):
            checked_then_edit.done = True
            _bad_snapshot(path)
    monkeypatch.setattr(W, 'integrity', checked_then_edit)
    with pytest.raises(W.LedgerError, match='checksum'):  # the end-of-init epoch check validates again
        LedgerWriter(path)
    assert str(path.resolve()) not in W._integrity_known
    monkeypatch.setattr(W, 'integrity', real)
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)


def test_k4b_r1_commit_after_cache_hit_never_trusted(tmp_path, monkeypatch):
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    assert str(path.resolve()) in W._integrity_known  # the next open reuses trust
    real = LedgerWriter._durable_after_last_run_close
    def late_init(self):
        _bad_snapshot(path)  # an external commit during initialisation, after the trust decision
        return real(self)
    monkeypatch.setattr(LedgerWriter, '_durable_after_last_run_close', late_init)
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)
    monkeypatch.setattr(LedgerWriter, '_durable_after_last_run_close', real)
    assert str(path.resolve()) not in W._integrity_known
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)


def test_k4b_r1_path_swapped_after_stat_never_trusted(tmp_path, monkeypatch):
    """The cache-hit observation is fenced against the database actually opened."""
    import shutil
    path = tmp_path/'corpus.sqlite3'
    _fresh_ledger(path)
    corrupt = tmp_path/'corrupt.sqlite3'
    shutil.copy2(path, corrupt)
    _bad_snapshot(corrupt)
    real = W.connect
    def swap_then_connect(target, *args, **kwargs):
        if Path(target) == path and not getattr(swap_then_connect, 'done', False):
            swap_then_connect.done = True
            os.replace(corrupt, path)  # another file now sits at the path the identity was taken from
        return real(target, *args, **kwargs)
    monkeypatch.setattr(W, 'connect', swap_then_connect)
    with pytest.raises(W.LedgerError, match='checksum'):
        LedgerWriter(path)


def test_k4b_r1_two_corpus_objects_reject_an_external_corrupt_commit(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    first = Corpus(tmp_path/'empty', tmp_path/'out')
    (first.markdown_dir/'source.md').write_text('Enough readable text for a clean document and scoring.\n')
    first.clean(drop_bad=False, num_threads=1)
    _bad_snapshot(first.output_dir/'corpus.sqlite3')
    second = Corpus(tmp_path/'empty', tmp_path/'out')
    for corpus in (second, first):
        with pytest.raises(W.LedgerError, match='checksum'):
            corpus.clean(drop_bad=False, num_threads=1)
    assert str((first.output_dir/'corpus.sqlite3').resolve()) not in W._integrity_known


# ------------------------------------- B-R2 originals verified before the first deletion ---

def _two_documents(tmp_path, monkeypatch, debug='off'):
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    output = tmp_path/'out'
    sources = [pdf(output/'downloads'/f'{name}.pdf', 'Long enough native document text for '+name) for name in ('alpha', 'beta')]
    corpus = Corpus(tmp_path/'inputs', output)
    corpus.extract(file_paths=sources, repair_fonts=False, accel_type='CPU', num_threads=1)
    return corpus, output, sources


def test_k4b_r2_original_verified_before_shard_retirement(tmp_path, monkeypatch):
    """Reviewer reproducer: an external edit after aggregate generation, before publication retires a shard."""
    from glossapi.stages import export as export_module, retention
    corpus, output, sources = _two_documents(tmp_path, monkeypatch)
    corpus.clean(drop_bad=False, num_threads=1)
    shards = output/'shards'
    initial = corpus.jsonl_sharded(shards, shard_size_bytes=1, compression='none')
    retired = initial[-1]
    real = export_module.ExportStage.finalize
    bindings = []
    def finalize(stage, request, **kwargs):
        result = real(stage, request, **kwargs)
        bindings.extend(J._active_journal.get().cached_originals)
        _next_tick()
        sources[0].write_bytes(sources[0].read_bytes()+b'\nexternal edit during export\n')
        return result
    monkeypatch.setattr(export_module.ExportStage, 'finalize', finalize)
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        Corpus(tmp_path/'inputs', output).jsonl_sharded(shards, shard_size_bytes=1000000, compression='none')
    assert bindings and retired.exists(), 'a user-visible shard was deleted before the original was re-verified'


def test_k4b_r2_original_verified_before_outcome_and_retention(tmp_path, monkeypatch):
    """The verification precedes the wrapper's outcome: a changed original is the recorded failure, nothing deleted."""
    from glossapi.stages import retention
    corpus, output, sources = _two_documents(tmp_path, monkeypatch)
    deleted = []
    monkeypatch.setattr(retention.RetentionController, 'delete_many', lambda self, ids: deleted.extend(ids))
    real = retention.retention_candidates
    def edit_then_list(*args, **kwargs):
        _next_tick()
        sources[1].write_bytes(sources[1].read_bytes()+b'\nexternal edit\n')
        return real(*args, **kwargs)
    monkeypatch.setattr(retention, 'retention_candidates', edit_then_list)
    with pytest.raises(ValueError, match='original_changed_since_observed'):
        Corpus(tmp_path/'inputs', output).clean(drop_bad=False, num_threads=1)
    assert deleted == []
    with W.open_readonly(output/'corpus.sqlite3') as connection:
        states = [row[0] for row in connection.execute("SELECT s.status FROM current_task_state s WHERE s.operation='clean.clean'")]
    assert states == ['failed']  # the clean wrapper's recorded outcome


def _rehashes(monkeypatch, sources):
    """Originals re-read by the destructive-boundary verification (only inside verify_cached_originals)."""
    counted = []
    real_verify, real_hash = J.ResultJournal.verify_cached_originals, J.file_hash
    def counting(path, *, own=False):
        if Path(path) in sources:
            counted.append(Path(path).name)
        return real_hash(path, own=own)
    def verify(journal, rows=None):
        J.file_hash = counting
        try:
            return real_verify(journal, rows)
        finally:
            J.file_hash = real_hash
    monkeypatch.setattr(J.ResultJournal, 'verify_cached_originals', verify)
    return counted


def test_k4b_r2_no_rehash_without_pending_deletions(tmp_path, monkeypatch):
    """debug=on keeps the intermediates: no deletion relies on the cached originals, so none is re-read (cost)."""
    corpus, output, sources = _two_documents(tmp_path, monkeypatch)
    counted = _rehashes(monkeypatch, sources)
    keeping = Corpus(tmp_path/'inputs', output)
    keeping._retention_settings = {'debug': 'on'}
    keeping.clean(drop_bad=False, num_threads=1)
    assert counted == []


def test_k4b_r2_each_original_rehashed_once_per_phase(tmp_path, monkeypatch):
    corpus, output, sources = _two_documents(tmp_path, monkeypatch)
    counted = _rehashes(monkeypatch, sources)
    Corpus(tmp_path/'inputs', output).clean(drop_bad=False, num_threads=1)
    assert sorted(counted) == ['alpha.pdf', 'beta.pdf']  # once each, shared by every destructive boundary of the phase


class _Guard:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def verify_cached_originals(self, rows=None):
        self.calls.append([row['file_id'] for row in rows])
        if self.fail:
            raise ValueError('original_changed_since_observed: guard')


@pytest.mark.parametrize('boundary', ['delete', 'delete_many', 'recover', 'bulk_release'])
def test_k4b_r2_every_destructive_boundary_passes_the_guard(tmp_path, boundary):
    from test_ledger_retention import fixture, consumer, settings
    from glossapi.stages.retention import RetentionController, release_private_scratch
    with LedgerWriter(tmp_path/'corpus.sqlite3') as writer:
        run, _, version, file, path = fixture(writer, tmp_path)
        task, _ = consumer(writer, run, version, file)
        RetentionController(writer, tmp_path, run).declare(file, [task])
        if boundary == 'recover':
            def crash(point, *_):
                if point == 'after_intent':
                    raise RuntimeError('simulated death')
            with pytest.raises(RuntimeError):
                RetentionController(writer, tmp_path, run, fault=crash).delete_many([file])
        guard = _Guard(fail=True)
        writer.original_guards = [guard]
        controller = RetentionController(writer, tmp_path, run, settings())
        with pytest.raises(ValueError, match='original_changed_since_observed'):
            if boundary == 'delete':
                controller.delete(file)
            elif boundary == 'delete_many':
                controller.delete_many([file])
            elif boundary == 'recover':
                controller.recover()
            else:
                release_private_scratch(controller, [writer.file_catalog()[file]])
        assert guard.calls and file in guard.calls[0] and path.exists()


# ---------------------------------------------- B-R3 producer ownership decides own=True ---

def test_k4b_r3_score_only_caller_text_is_never_own(tmp_path, monkeypatch):
    """Reviewer reproducer: clean(write_cleaned_files=False) selects the caller's own text."""
    from glossapi.ledger import encoding as E
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    given = corpus.markdown_dir/'source.md'
    given.write_text('Caller supplied text with a long enough native body for cleaning metrics.\n')
    seen = []
    real = E.file_hash
    def observed(path, *, own=False):
        if Path(path).absolute() == given.absolute():
            seen.append(own)
        return real(path, own=own)
    monkeypatch.setattr(E, 'file_hash', observed)
    corpus.clean(write_cleaned_files=False, drop_bad=False, num_threads=1)
    assert seen and not any(seen), 'clean.selection declared caller-supplied score-only text an owned file'


def test_k4b_r3_caller_hard_link_is_byte_hashed(tmp_path, monkeypatch):
    """A caller text that is a hard link of a registered own file is still byte-hashed (never the registry)."""
    from glossapi.ledger import encoding as E
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    owned = tmp_path/'owned.md'
    owned.write_text('Caller hard link text with a long enough native body for cleaning metrics.\n')
    given = corpus.markdown_dir/'source.md'
    os.link(owned, given)
    seen = []
    real = E.file_hash
    def observed(path, *, own=False):
        if Path(path).absolute() == given.absolute():
            seen.append(own)
        return real(path, own=own)
    monkeypatch.setattr(E, 'file_hash', observed)
    with E.run_hash_scope():
        E.remember_own_hash(owned, hashlib.sha256(owned.read_bytes()).hexdigest())
        corpus.clean(write_cleaned_files=False, drop_bad=False, num_threads=1)
    assert seen and not any(seen)


def test_k4b_r3_managed_selection_keeps_its_reuse(tmp_path, monkeypatch):
    """A managed clean product (write_cleaned_files=True) is still checked with own=True."""
    from glossapi.ledger import encoding as E
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    (corpus.markdown_dir/'source.md').write_text('Managed product text with a long enough native body for cleaning metrics.\n')
    seen = []
    real = E.file_hash
    def observed(path, *, own=False):
        if Path(path).parent == corpus.cleaned_markdown_dir:
            seen.append(own)
        return real(path, own=own)
    monkeypatch.setattr(E, 'file_hash', observed)
    corpus.clean(drop_bad=False, num_threads=1)
    assert True in seen


def test_k4b_r3_section_restore_queries_ownership(tmp_path, monkeypatch):
    from glossapi.ledger import encoding as E
    monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'cutover')
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    given = corpus.markdown_dir/'source.md'
    given.write_text('# Caller text\n\nCaller supplied text with a long enough native body for cleaning metrics.\n')
    corpus.clean(write_cleaned_files=False, drop_bad=False, num_threads=1)
    seen = []
    real = E.file_hash
    def observed(path, *, own=False):
        if Path(path).absolute() == given.absolute():
            seen.append(own)
        return real(path, own=own)
    monkeypatch.setattr(E, 'file_hash', observed)
    Corpus(tmp_path/'empty', tmp_path/'out').section()  # a fresh object restores the selection from the ledger
    assert seen and not any(seen)


# ---------------------------------------------- B-R4 the live ledger is never its backup ---

@pytest.mark.parametrize('alias', ['same_path', 'hard_link', 'symlink'])
def test_k4b_r4_live_ledger_is_never_its_own_backup(tmp_path, alias):
    """Reviewer reproducer plus the symlink case."""
    from glossapi.ledger.schema import FORMAT_SCHEMA_SQL
    path = tmp_path/'corpus.sqlite3'
    backup = path if alias == 'same_path' else tmp_path/'backup.sqlite3'
    with LedgerWriter(path, storage_version=3) as writer:
        if alias == 'hard_link':
            os.link(path, backup)
        elif alias == 'symlink':
            backup.symlink_to(path)
        with pytest.raises(W.LedgerError, match='live ledger itself'):
            writer.migrate(4, FORMAT_SCHEMA_SQL, checksum=hashlib.sha256(FORMAT_SCHEMA_SQL.encode()).hexdigest(),
                           code_revision='test', backup_path=backup)
        assert writer.connection.execute('PRAGMA user_version').fetchone()[0] == 3
