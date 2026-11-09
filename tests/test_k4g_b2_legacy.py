"""K4g batch B2: the legacy pipeline (``Corpus`` phases, metadata mergers, legacy branches).

- LS6-G04: metadata merges never crash on, and never conflate, rows that have no filename (a skipped duplicate URL and
  a failed URL both have ``filename == ""``); they are occurrence rows, carried through unchanged.
- T2-G02: a resumed ``Corpus.download`` keeps every source column and the preassigned ``filename_base``.
- K2E-F3: the clean metric merge never rounds an integer (values above 2**53 survive a row union).
- K2E-F4: a lossless union refuses to drop a categorical/extension dtype family.
"""
import json
from pathlib import Path
import threading

import pandas as pd
import pytest

from glossapi import Corpus


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def _pdf(path, text):
    from test_ledger_fixset_lg import pdf
    return pdf(path, text)


class Server:
    """Loopback server over a mutable ``path -> bytes`` table; an absent path answers 404."""
    def __init__(self, files):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.files, self.requests = dict(files), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(self.path)
                body = outer.files.get(self.path)
                if body is None:
                    self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers(); return
                self.send_response(200); self.send_header('Content-Type', 'application/pdf')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

            def log_message(self, *args):
                pass
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def url(self, path):
        return f'http://127.0.0.1:{self.server.server_port}{path}'

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


DOWNLOAD = dict(rate_limit=0, sleep=0, max_retries=1, retry_interval=0, pre_ping_domains=False)


# ------------------------------------------------------------------ LS6-G04 ---

def test_ls6g04_ensure_metadata_merges_two_blank_filename_rows(tmp_path):
    """Reviewer probe: existing rows ``[AAA_001.pdf, "", "", AAA_002.pdf]`` (a skipped duplicate and a failed URL)."""
    from glossapi.parquet_schema import ParquetSchema
    results = tmp_path/'download_results'
    results.mkdir()
    existing = pd.DataFrame({
        'url': ['http://x/1.pdf', 'http://x/1.pdf?dup', 'http://x/missing.pdf', 'http://x/2.pdf'],
        'filename': ['AAA_001.pdf', '', '', 'AAA_002.pdf'],
        'download_success': [True, False, False, True],
        'download_error': ['', 'Duplicate URL (skipped)', 'HTTP 404', ''],
        'title': ['one', 'one again', 'gone', 'two'],
    })
    existing.to_parquet(results/'download_results_manifest.parquet', index=False)
    (tmp_path/'markdown').mkdir()
    (tmp_path/'markdown'/'AAA_001.md').write_text('one')
    path = ParquetSchema({'url_column': 'url'}).ensure_metadata_parquet(tmp_path)
    table = pd.read_parquet(path)
    assert len(table) == 4
    blanks = table[table['filename'] == ''].sort_values('url')
    # Both occurrence rows survive with their own values (never merged into each other or into a keyed row).
    assert blanks['url'].tolist() == ['http://x/1.pdf?dup', 'http://x/missing.pdf']
    assert blanks['download_error'].tolist() == ['Duplicate URL (skipped)', 'HTTP 404']
    assert blanks['title'].tolist() == ['one again', 'gone']
    keyed = table.set_index('filename')
    assert keyed.loc['AAA_001.pdf', 'title'] == 'one' and 'extract' in keyed.loc['AAA_001.pdf', 'processing_stage']
    assert keyed.loc['AAA_002.pdf', 'title'] == 'two'


def test_ls6g04_ensure_metadata_keeps_every_occurrence_of_a_repeated_filename(tmp_path):
    """Neighbour variant: the same non-blank filename on two rows gets the artifact evidence on each occurrence."""
    from glossapi.parquet_schema import ParquetSchema
    results = tmp_path/'download_results'
    results.mkdir()
    pd.DataFrame({'url': ['http://x/a', 'http://x/b', None], 'filename': ['AAA_001.pdf', 'AAA_001.pdf', None],
                  'note': ['first', 'second', 'no name']}).to_parquet(results/'download_results_m.parquet', index=False)
    (tmp_path/'markdown').mkdir()
    (tmp_path/'markdown'/'AAA_001.md').write_text('one')
    table = pd.read_parquet(ParquetSchema({'url_column': 'url'}).ensure_metadata_parquet(tmp_path))
    assert len(table) == 3
    repeated = table[table['filename'] == 'AAA_001.pdf'].sort_values('note')
    assert repeated['note'].tolist() == ['first', 'second']
    assert repeated['processing_stage'].map(lambda s: 'extract' in s.split(',')).all()
    assert table['note'].tolist().count('no name') == 1


def test_ls6g04_metric_merge_never_updates_blank_filename_rows(tmp_path):
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    base = pd.DataFrame({'filename': ['AAA_001.pdf', '', '', 'AAA_002.pdf'], 'x': [1, 2, 3, 4]})
    updates = pd.DataFrame({'filename': ['AAA_001.pdf', 'AAA_003.pdf'], 'm': [0.5, 0.25]})
    merged = CleanPhaseMixin._merge_metric_dataframe(base, updates)
    assert len(merged) == 5
    assert sorted(merged.loc[merged['filename'] == '', 'x'].tolist()) == [2, 3]
    assert merged.loc[merged['filename'] == '', 'm'].isna().all()
    keyed = merged[merged['filename'] != ''].set_index('filename')
    assert keyed.loc['AAA_001.pdf', 'm'] == 0.5 and keyed.loc['AAA_003.pdf', 'm'] == 0.25


def test_ls6g04_fold_keeps_each_failed_url_as_its_own_row(tmp_path):
    """The download fold never collapses blank-filename rows into one ``""`` key (or reports ``""`` as a conflict)."""
    from glossapi.parquet_schema import fold_download_rows
    results = tmp_path/'download_results'
    results.mkdir()
    pd.DataFrame({'filename': ['AAA_000.pdf', ''], 'url': ['http://x/0', 'http://x/old-failure'],
                  'download_success': [True, False]}).to_parquet(results/'download_results.parquet', index=False)
    published = results/'download_results_m.parquet'
    pd.DataFrame({'filename': ['AAB_001.pdf', '', ''], 'url': ['http://x/1', 'http://x/dup', 'http://x/404'],
                  'download_success': [True, False, False]}).to_parquet(published, index=False)
    assert fold_download_rows(tmp_path, published) == []
    assert fold_download_rows(tmp_path, published) == []  # idempotent: a second fold adds nothing
    table = pd.read_parquet(results/'download_results.parquet')
    assert sorted(table['url']) == ['http://x/0', 'http://x/1', 'http://x/404', 'http://x/dup', 'http://x/old-failure']


def test_ls6g04_duplicate_plus_failed_url_download_then_clean_and_export(tmp_path):
    """End to end (legacy sequential API): a skipped duplicate URL plus a failed URL, then extract, clean, export."""
    pytest.importorskip('aiohttp')
    pytest.importorskip('glossapi_rs_cleaner')
    good = _pdf(tmp_path/'src'/'good.pdf', 'Greek legacy duplicate text').read_bytes()
    root = tmp_path/'out'
    with Server({'/good.pdf': good}) as server:
        manifest = tmp_path/'manifest.parquet'
        pd.DataFrame({'url': [server.url('/good.pdf'), server.url('/good.pdf'), server.url('/missing.pdf')]}).to_parquet(manifest, index=False)
        corpus = Corpus(tmp_path/'empty', root)
        downloaded = corpus.download(manifest, **DOWNLOAD)
    assert (downloaded['filename'] == '').sum() == 2
    files = sorted((root/'downloads').glob('*.pdf'))
    corpus.extract(file_paths=files, input_format='pdf', phase1_backend='safe', accel_type='CPU',
                   num_threads=1, skip_existing=False, batched=False, repair_fonts=False)
    corpus.clean(num_threads=1)
    corpus.jsonl(root/'export.jsonl')
    documents = [json.loads(line)['document'] for line in (root/'export.jsonl').read_text().splitlines()]
    assert any('Greek legacy duplicate text' in d for d in documents)


# ------------------------------------------------------------------ T2-G02 ---

@pytest.mark.parametrize('cell', ['scalar', 'list', 'list-fallback'])
def test_t2g02_resumed_download_keeps_source_columns_and_preassigned_name(tmp_path, monkeypatch, cell):
    pytest.importorskip('aiohttp')
    a = _pdf(tmp_path/'src'/'a.pdf', 'First resumed source').read_bytes()
    b = _pdf(tmp_path/'src'/'b.pdf', 'Second resumed source').read_bytes()
    root = tmp_path/'out'
    with Server({'/a.pdf': a}) as server:
        urls = [server.url('/a.pdf'), server.url('/b.pdf')]
        manifest = tmp_path/'manifest.parquet'
        pd.DataFrame({'url': urls if cell == 'scalar' else [json.dumps([u]) for u in urls],
                      'filename_base': ['MINE_001', 'MINE_002'], 'title': ['Alpha', 'Beta']}).to_parquet(manifest, index=False)
        first = Corpus(tmp_path/'empty', root).download(manifest, **DOWNLOAD)
        assert first.set_index('filename_base').loc['MINE_002', 'download_success'] == False  # noqa: E712
        server.files['/b.pdf'] = b
        if cell == 'list-fallback':  # the resume's own expansion fails once: the basic JSON fallback expands the rows
            from glossapi.gloss_downloader import GlossDownloader
            real, calls = GlossDownloader._expand_and_mark_duplicates, []

            def flaky(self, df):
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError('expansion unavailable')
                return real(self, df)
            monkeypatch.setattr(GlossDownloader, '_expand_and_mark_duplicates', flaky)
        resumed = Corpus(tmp_path/'empty', root).download(manifest, **DOWNLOAD)
    row = resumed[resumed['url'] == urls[1]].iloc[0]
    assert bool(row['download_success'])
    assert row['filename_base'] == 'MINE_002' and row['filename'] == 'MINE_002.pdf'
    assert row['title'] == 'Beta'
    assert (root/'downloads'/'MINE_002.pdf').is_file()
    kept = resumed[resumed['url'] == urls[0]].iloc[0]
    assert kept['title'] == 'Alpha' and kept['filename'] == 'MINE_001.pdf'


# ------------------------------------------------------------------ K2E-F3 ---

def test_k2ef3_metric_merge_keeps_integers_above_2_53_on_a_row_union():
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    big = 2**53 + 1
    base = pd.DataFrame({'filename': ['a.pdf'], 'big': [big], 'n': [3]})
    updates = pd.DataFrame({'filename': ['b.pdf'], 'big': [5], 'n': [4]})
    merged = CleanPhaseMixin._merge_metric_dataframe(base, updates).set_index('filename')
    assert int(merged.loc['a.pdf', 'big']) == big and int(merged.loc['b.pdf', 'big']) == 5
    # The other direction: the update holds the large value, the base has rows the update lacks.
    merged = CleanPhaseMixin._merge_metric_dataframe(updates, base).set_index('filename')
    assert int(merged.loc['a.pdf', 'big']) == big


def test_k2ef3_metric_merge_is_unchanged_for_ordinary_values():
    """Output stability: values a float64 holds exactly keep the dtypes and values of the previous merge."""
    from glossapi.corpus.phase_clean import CleanPhaseMixin
    base = pd.DataFrame({'filename': ['a.pdf', 'c.pdf'], 'n': [3, 7], 's': ['x', 'y'], 'f': [0.5, pd.NA]})
    updates = pd.DataFrame({'filename': ['a.pdf', 'b.pdf'], 'n': [9, 4], 'f': [1.5, 2.5]})

    def legacy(base, updates, key='filename'):
        base_idx = base.set_index(key, drop=False)
        update_idx = updates.set_index(key, drop=False)
        base_idx = base_idx.combine_first(update_idx)
        base_idx.update(update_idx)
        return base_idx.reset_index(drop=True)
    for b, u in ((base, updates), (updates, base), (base, updates.iloc[:1]), (base, base)):
        pd.testing.assert_frame_equal(CleanPhaseMixin._merge_metric_dataframe(b, u), legacy(b, u))


# ------------------------------------------------------------------ K2E-F4 ---

def test_k2ef4_lossless_concat_refuses_a_categorical_family_change():
    from glossapi.parquet_schema import MetadataSchemaError, lossless_concat
    with pytest.raises(MetadataSchemaError, match='category'):
        lossless_concat([pd.DataFrame({'c': pd.Categorical(['u'])}), pd.DataFrame({'c': ['v']})])


def test_k2ef4_lossless_concat_keeps_identical_categoricals_and_plain_unions():
    from glossapi.parquet_schema import lossless_concat
    same = lossless_concat([pd.DataFrame({'c': pd.Categorical(['u'])}), pd.DataFrame({'c': pd.Categorical(['u'])})])
    assert isinstance(same['c'].dtype, pd.CategoricalDtype)
    partial = lossless_concat([pd.DataFrame({'c': pd.Categorical(['u']), 'k': [1]}), pd.DataFrame({'k': [2]})])
    assert isinstance(partial['c'].dtype, pd.CategoricalDtype) and partial['c'].isna().tolist() == [False, True]
    plain = lossless_concat([pd.DataFrame({'k': [1], 's': ['a']}), pd.DataFrame({'k': [2], 's': ['b']})])
    assert plain['k'].tolist() == [1, 2]
    # Two categoricals with different categories stay categorical (the categories are unified), values exact.
    mixed = lossless_concat([pd.DataFrame({'c': pd.Categorical(['u'])}), pd.DataFrame({'c': pd.Categorical(['v'])})])
    assert isinstance(mixed['c'].dtype, pd.CategoricalDtype) and mixed['c'].tolist() == ['u', 'v']


def test_k2ef4_lossless_concat_refuses_a_time_zone_loss():
    from glossapi.parquet_schema import MetadataSchemaError, lossless_concat
    aware = pd.DataFrame({'t': pd.to_datetime(['2026-11-02T10:00:00']).tz_localize('Europe/Athens')})
    with pytest.raises(MetadataSchemaError, match='time zone'):
        lossless_concat([aware, pd.DataFrame({'t': ['2026-11-02 10:00:00+02:00']})])


# ------------------------------------------------------------------ T2-G05 ---

def test_t2g05_items_no_worker_attempted_never_enter_the_fatal_skiplist(tmp_path, monkeypatch):
    """Legacy multi-GPU branch: every worker exits before taking an item; a later extract must still see the files."""
    import multiprocessing
    import queue
    from glossapi.corpus import phase_extract as phase
    from glossapi.corpus.corpus_skiplist import _SkiplistManager, _resolve_skiplist_path

    class State:
        def __init__(self, *args, **kwargs): pass
        def load(self): return set(), set()
        def save(self, *args): pass

    class Process:  # exits cleanly at once without taking a work item
        pid, exitcode = None, 0
        def __init__(self, target, args, kwargs=None): self.args = args
        def start(self): pass
        def is_alive(self): return False
        def join(self, **kwargs): pass
    context = type('Context', (), dict(Manager=staticmethod(lambda: type('M', (), dict(dict=dict, shutdown=lambda self: None))()),
                                       Queue=queue.Queue, Process=Process))()
    monkeypatch.setattr(multiprocessing, 'get_context', lambda method: context)
    monkeypatch.setattr(phase, '_ProcessingStateManager', State)
    monkeypatch.setattr(Corpus, '_finalize_font_repair_metadata', lambda *args: None)
    monkeypatch.setattr(phase, '_estimate_extract_work_pages', lambda path: 1)
    source = tmp_path/'downloads'/'healthy.pdf'
    source.parent.mkdir()
    source.write_bytes(b'input')
    Corpus(tmp_path, tmp_path).extract(file_paths=[source], phase1_backend='docling', use_gpus='multi', devices=[0],
                                       batched=False, skip_existing=False)
    import logging
    assert 'healthy' not in _SkiplistManager(_resolve_skiplist_path(tmp_path, logging.getLogger('t')), logging.getLogger('t')).load()


# ------------------------------------------------------------------ K2F-G2 ---

def _ocr_merge_inputs(tmp_path):
    master = tmp_path/'master.parquet'
    pd.DataFrame({'filename': ['a.pdf', 'b.pdf'], 'ocr_success': [False, False]}).to_parquet(master, index=False)
    shard = tmp_path/'shard.parquet'
    pd.DataFrame({'filename': ['a.pdf'], 'ocr_success': [True]}).to_parquet(shard, index=False)
    return master, shard


def test_k2fg2_in_place_ocr_merge_runs_under_the_metadata_lock(tmp_path, monkeypatch):
    """Output = master: the read, merge and replace happen inside one metadata lock (never an unlocked rewrite)."""
    from glossapi import parquet_schema
    from glossapi.scripts import openarchives_ocr_merge
    master, shard = _ocr_merge_inputs(tmp_path)
    locked, real = [], parquet_schema._parquet_lock

    def spy(path):
        locked.append(Path(path).name)
        return real(path)
    monkeypatch.setattr(parquet_schema, '_parquet_lock', spy)
    assert openarchives_ocr_merge.main(['--master-parquet', str(master), '--shard-parquets', str(shard),
                                        '--output-parquet', str(master)]) == 0
    assert locked == ['master.parquet']
    merged = pd.read_parquet(master).set_index('filename')
    assert bool(merged.loc['a.pdf', 'ocr_success']) and not bool(merged.loc['b.pdf', 'ocr_success'])


def test_k2fg2_ocr_merge_refuses_a_hard_linked_output_of_the_master(tmp_path):
    import os
    from glossapi.parquet_schema import MetadataAliasError
    from glossapi.scripts import openarchives_ocr_merge
    master, shard = _ocr_merge_inputs(tmp_path)
    alias = tmp_path/'alias.parquet'
    os.link(master, alias)
    before = master.read_bytes()
    with pytest.raises(MetadataAliasError):
        openarchives_ocr_merge.main(['--master-parquet', str(master), '--shard-parquets', str(shard),
                                     '--output-parquet', str(alias)])
    assert master.read_bytes() == before


# ------------------------------------------------------------------ T2-G11 ---

def test_t2g11_math_results_update_the_parquet_and_project_an_empty_success(tmp_path, monkeypatch):
    """The documented behaviour: the consolidated parquet is updated whatever GLOSSAPI_PARQUET_COMPACTOR says, and
    an enrichment with no formula is projected as enriched with ``math_items == 0`` (a compatibility projection)."""
    from glossapi.ocr.utils import triage
    for compactor in ('1', '0'):
        monkeypatch.setenv('GLOSSAPI_PARQUET_COMPACTOR', compactor)
        parquet = tmp_path/compactor/'download_results'/'download_results.parquet'
        parquet.parent.mkdir(parents=True)
        pd.DataFrame({'filename': ['doc.pdf']}).to_parquet(parquet, index=False)
        triage.update_math_enrich_results(parquet, 'doc', items=0, accepted=0, time_sec=0.5)
        row = pd.read_parquet(parquet).iloc[0]
        assert bool(row['enriched_math']) and bool(row['math_enriched']) and int(row['math_items']) == 0
        assert (tmp_path/compactor/'sidecars'/'math'/'doc.json').is_file() == (compactor == '1')
    doc = triage.update_math_enrich_results.__doc__
    assert 'updated in every case' in doc and 'compatibility projection' in doc


# ------------------------------------------------------------------ T2-G12 ---

def test_t2g12_section_schema_constants_are_the_emitted_schemas():
    pa = pytest.importorskip('pyarrow')
    from glossapi.gloss_section import section_schema
    from glossapi.parquet_schema import ParquetSchema
    schema = ParquetSchema()
    assert schema.SECTION_SCHEMA.equals(section_schema())
    assert schema.CLASSIFIED_SCHEMA.equals(section_schema().append(pa.field('predicted_section', pa.string())))
    emitted = section_schema().empty_table().to_pandas()
    assert schema.validate_schema(emitted, 'section') == (True, [])
    emitted['predicted_section'] = pd.Series(dtype=object)
    assert schema.validate_schema(emitted, 'classified') == (True, [])


# ------------------------------------------------------------------ LS6-G06 ---

def test_ls6g06_runtime_conversion_log_is_not_source():
    root = Path(__file__).resolve().parents[1]
    ignore = root/'.gitignore'
    if not ignore.is_file():
        pytest.skip('not a source checkout')
    # The runtime log a standalone script appends to in the working directory is ignored (and no longer tracked),
    # so a source-tree receipt never includes it.
    assert '/conversion.log' in {line.strip() for line in ignore.read_text().splitlines()}


# ------------------------------------------------------------------ T2-G03 ---

def test_t2g03_the_extractor_has_no_third_metadata_search_order():
    from glossapi.gloss_extract import GlossExtract
    assert not hasattr(GlossExtract, '_find_metadata_parquet') and not hasattr(GlossExtract, '_ensure_metadata_parquet')


def _two_metadata_parquets(tmp_path):
    source, out = tmp_path/'in', tmp_path/'out'
    (out/'download_results').mkdir(parents=True)
    source.mkdir()
    pd.DataFrame({'filename': ['a.pdf'], 'url': ['http://x/a']}).to_parquet(source/'metadata.parquet', index=False)
    pd.DataFrame({'filename': ['a.pdf'], 'url': ['http://x/a']}).to_parquet(out/'download_results'/'download_results.parquet', index=False)
    return source, out


def test_t2g03_documented_read_precedence(tmp_path):
    """The two live read rules, as documented (docs text in k4g/b2/reply.md): the clean metrics resolver prefers a
    metadata parquet in ``input_dir``; the other phases (sections, OCR, math) prefer the output's canonical table.
    One Corpus object resolves once and every later phase reuses that choice."""
    from glossapi.parquet_schema import ParquetSchema
    source, out = _two_metadata_parquets(tmp_path)
    schema = ParquetSchema({'url_column': 'url'})
    assert Corpus(source, out)._resolve_clean_metrics_parquet(schema) == source/'metadata.parquet'
    assert Corpus(source, out)._resolve_metadata_parquet(schema) == out/'download_results'/'download_results.parquet'
    corpus = Corpus(source, out)
    first = corpus._resolve_clean_metrics_parquet(schema)
    assert corpus._resolve_metadata_parquet(schema) == first


# ------------------------------------------------------------------ LS4-G01 ---

def _clean_corpus(root, **kwargs):
    import warnings
    pytest.importorskip('glossapi_rs_cleaner')
    (root/'markdown').mkdir(parents=True)
    (root/'markdown'/'doc.md').write_text('Η ελληνική γλώσσα είναι μία από τις αρχαιότερες γλώσσες. ' * 20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        Corpus(root, root).clean(num_threads=1, **kwargs)
    table = pd.read_parquet(root/'download_results'/'download_results.parquet')
    return [w for w in caught if issubclass(w.category, DeprecationWarning) and 'threshold' in str(w.message)], table


def test_ls4g01_clean_threshold_is_deprecated_and_changes_nothing(tmp_path):
    import sqlite3
    quiet, default = _clean_corpus(tmp_path/'default')
    assert quiet == []
    warned, passed = _clean_corpus(tmp_path/'passed', threshold=0.95)
    assert len(warned) == 1 and warned[0].filename == __file__  # attributed to the caller, so default filters show it
    columns = [c for c in default.columns if c not in ('processing_stage',)]
    pd.testing.assert_frame_equal(default[columns], passed[columns])
    # The recorded call parameters keep the value the signature always had: a later run passing 0.10 explicitly
    # records the same settings snapshot as the default run.
    import warnings
    root = tmp_path/'default'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        Corpus(root, root).clean(num_threads=1, threshold=0.10)
    with sqlite3.connect(root/'corpus.sqlite3') as connection:
        settings = [row[0] for row in connection.execute('SELECT settings_sha256 FROM runs ORDER BY rowid')]
    assert len(settings) == 2 and settings[0] == settings[1]


# ------------------------------------------------------------------ FG-G03 ---

def _quality_report(name, total, routed):
    """A page-quality report as ``page_quality().to_dict()`` shapes the fields both consumers read: ``run``'s OCR stage
    reads ``pages[*].page_needs_ocr``; the API reads the sidecar summary (``ocr_page_list``) that clean attaches."""
    ranges, start = [], None
    for page in range(1, total + 2):
        if page in routed and start is None:
            start = page
        elif page not in routed and start is not None:
            ranges.append(f'{start}-{page - 1}' if page - 1 > start else str(start)); start = None
    return dict(filename=name, pages_total=total, pages_needing_ocr=len(routed), ocr_page_list=','.join(ranges),
                pages=[dict(page=page, page_needs_ocr=page in routed) for page in range(1, total + 1)])


def _run_pages(report):
    # stages/ocr.py (the run DAG's OCR stage): the pages page quality routes; none -> the document is not OCRed.
    return {int(page['page']) for page in report['pages'] if page['page_needs_ocr'] is True}


def _api_pages(tmp_path, monkeypatch, documents, **ocr_kwargs):
    """``Corpus.ocr`` on metadata whose routes clean attached from the same reports: document -> selected pages."""
    from glossapi.corpus.ocr import dispatch as runner
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    from glossapi.page_quality import parse_page_ranges
    from test_t1_4_pipeline import make_pdf
    metrics = tmp_path/'json'/'metrics'
    metrics.mkdir(parents=True)
    rows = []
    for name, (flagged, total, routed) in documents.items():
        make_pdf(tmp_path/name, total)
        (metrics/f'{Path(name).stem}.page_quality.json').write_text(json.dumps(_quality_report(name, total, routed)))
        rows.append(dict(filename=name, needs_ocr=flagged, filter='ok', page_count=total))
    target = tmp_path/'download_results'/'download_results.parquet'
    target.parent.mkdir(parents=True)
    _attach_page_quality_routes(pd.DataFrame(rows), metrics).to_parquet(target, index=False)
    corpus = Corpus(tmp_path, tmp_path)
    monkeypatch.setattr(corpus, 'clean', lambda **kw: None)
    calls = []
    monkeypatch.setattr(runner, 'run_plugin_ocr', lambda self, files, **kw: calls.append((list(files), kw)))
    corpus.ocr(mode='ocr_bad', math_enhance=False, **ocr_kwargs)
    files, kwargs = calls[0] if calls else ([], {})
    ranges = {}
    for entry in kwargs.get('page_ranges') or []:
        name, first, last = entry.rsplit(':', 2)
        ranges.setdefault(name, set()).update(range(int(first), int(last) + 1))
    return {name: (ranges.get(name) or set(range(1, documents[name][1] + 1))) if name in files else set()
            for name in documents}


PAIRED = {'flagged_routed.pdf': (True, 4, {2, 3}), 'unflagged_routed.pdf': (False, 3, {1, 3}),
          'unflagged_unrouted.pdf': (False, 2, set())}


def test_fgg03_corpus_ocr_default_selects_the_pages_run_selects(tmp_path, monkeypatch):
    api = _api_pages(tmp_path, monkeypatch, PAIRED)
    run = {name: _run_pages(_quality_report(name, total, routed)) for name, (_flag, total, routed) in PAIRED.items()}
    assert api == run == {'flagged_routed.pdf': {2, 3}, 'unflagged_routed.pdf': {1, 3}, 'unflagged_unrouted.pdf': set()}


def test_fgg03_flagged_mode_is_still_available_explicitly(tmp_path, monkeypatch):
    api = _api_pages(tmp_path, monkeypatch, PAIRED, page_routing='flagged')
    assert api == {'flagged_routed.pdf': {2, 3}, 'unflagged_routed.pdf': set(), 'unflagged_unrouted.pdf': set()}


@pytest.mark.parametrize('page_routing', [None, 'flagged'])
def test_fgg03_flagged_unrouted_document_matches_run(tmp_path, monkeypatch, page_routing):
    """FG-G03 residual, resolved by K5 quality policy P4d (K5-A2b): a cleaner-flagged document whose page quality
    routes no page is not OCRed by Corpus.ocr, in both page_routing modes, as run leaves it (paired with run's rule)."""
    documents = {'flagged_unrouted.pdf': (True, 2, set()), 'flagged_routed.pdf': (True, 3, {2})}
    kwargs = {'page_routing': page_routing} if page_routing else {}
    api = _api_pages(tmp_path, monkeypatch, documents, **kwargs)
    run = {name: _run_pages(_quality_report(name, total, routed)) for name, (_flag, total, routed) in documents.items()}
    assert api == run == {'flagged_unrouted.pdf': set(), 'flagged_routed.pdf': {2}}
