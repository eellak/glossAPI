"""Offline witnesses for the policy defaults and the conservative language gate."""
from types import SimpleNamespace
from pathlib import Path
import json

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.corpus import phase_extract
from glossapi.corpus.phase_clean import _latin_word_shape_rate, _readable_non_greek_mask
from glossapi.page_quality import page_quality
from glossapi.scripts import page_quality_batch
from test_page_quality import COMMON_GREEK, quality_pdf


@pytest.mark.parametrize('cuda,device,expected', [
    (True, 'CUDA', True), (True, 'cuda:1', True), (True, 'Auto', True),
    (False, 'CUDA', False), (False, 'Auto', False), (True, 'CPU', False),
    (True, 'cpu:0', False), (True, 'MPS', False),
])
def test_auto_cuda(cuda, device, expected, monkeypatch):
    monkeypatch.setattr(phase_extract, '_maybe_import_torch',
                        lambda **kw: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)))
    assert phase_extract._resolve_batched_docling(None, accel_type=device) is expected


@pytest.mark.parametrize('kwargs', [dict(resolved_backend='safe'), dict(use_gpus='multi'),
    dict(formula_enrichment=True), dict(code_enrichment=True)])
def test_auto_keeps_other_extraction_paths(kwargs, monkeypatch):
    monkeypatch.setattr(phase_extract, '_maybe_import_torch', lambda **kw: pytest.fail('no CUDA probe needed'))
    assert not phase_extract._resolve_batched_docling(None, **kwargs)


@pytest.mark.parametrize('batched', [True, False])
def test_explicit_batched_overrides_auto(batched, monkeypatch):
    monkeypatch.setattr(phase_extract, '_maybe_import_torch', lambda **kw: pytest.fail('explicit override'))
    assert phase_extract._resolve_batched_docling(batched, accel_type='CPU') is batched


def test_public_extract_auto_dispatch(tmp_path, monkeypatch):
    from glossapi.corpus import batched_extract
    monkeypatch.setattr(phase_extract, '_maybe_import_torch',
                        lambda **kw: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    calls = []
    monkeypatch.setattr(batched_extract, 'corpus_batched_extract',
                        lambda self, original, **kw: calls.append(kw))
    corpus = Corpus(tmp_path, tmp_path)
    corpus.extract(phase1_backend='docling', num_threads=3)
    assert calls[0]['_auto_cuda'] is True and calls[0]['batched'] is False
    corpus.extract(batched=True)
    assert calls[1]['_auto_cuda'] is False
    corpus.extractor = SimpleNamespace(pipeline_options=SimpleNamespace(
        accelerator_options=SimpleNamespace(device='cpu')))
    # A prepared CPU converter must override an available CUDA device.
    monkeypatch.setattr(phase_extract, '_resolve_batched_docling',
                        lambda value, **kw: calls.append(kw) or True)
    corpus.extract(phase1_backend='docling', _prepared=True)
    assert calls[-2]['accel_type'] == 'cpu'
    assert calls[-2]['resolved_backend'] == 'docling'


@pytest.mark.parametrize('force,formula,code,expected', [
    (False, False, False, 'safe'), (True, False, False, 'safe'),
    (False, True, False, 'docling'), (False, False, True, 'docling'),
])
def test_auto_backend_policy_is_unchanged(tmp_path, force, formula, code, expected):
    corpus = Corpus(tmp_path, tmp_path)
    assert corpus._resolve_phase1_backend('auto', force_ocr=force,
        formula_enrichment=formula, code_enrichment=code) == expected


@pytest.mark.parametrize('changes,expected', [
    ({}, True), ({'mojibake_badness_score': .101}, False),
    ({'latin_percentage': 74.9}, False), ({'percentage_greek': 6}, False),
    ({'char_count_no_comments': 199}, False), ({'latin_word_shape_rate': .949}, False),
    ({'latin_word_shape_rate': None}, False), ({'greek_badness_score': 60}, False),
    ({'mojibake_badness_score': .1, 'char_count_no_comments': 200,
      'latin_word_shape_rate': .95}, True),
])
def test_readable_gate_thresholds(changes, expected):
    row = dict(greek_badness_score=974, mojibake_badness_score=0,
               latin_percentage=80, percentage_greek=1, char_count_no_comments=500,
               latin_word_shape_rate=1)
    assert bool(_readable_non_greek_mask(pd.DataFrame([row | changes])).iloc[0]) is expected
    assert not _readable_non_greek_mask(pd.DataFrame([{}])).iloc[0]


def test_raw_latin_shape_is_not_dictionary_validity():
    assert _latin_word_shape_rate('short title') is None
    assert _latin_word_shape_rate('Readable words with ordinary vowel balance. ' * 10) == 1
    assert _latin_word_shape_rate('bcdfghjklmn ' * 25) == 0


@pytest.mark.parametrize('text', [
    'Readable ordinary academic research words. ' * 20 + 'a b cd ef ' * 100,
    'Readable ordinary academic research words. ' * 20 + '! / \\ | 7 ' * 1000,
])
def test_short_or_raw_fragments_do_not_establish_readability(text):
    rate = _latin_word_shape_rate(text)
    assert rate is None
    frame = pd.DataFrame([dict(greek_badness_score=974, mojibake_badness_score=0,
        latin_percentage=90, percentage_greek=0, char_count_no_comments=1000,
        latin_word_shape_rate=rate)])
    assert not _readable_non_greek_mask(frame).iloc[0]


@pytest.mark.parametrize('batched,expected', [(None, False), (False, False), (True, True)])
def test_multi_gpu_orchestrator_through_real_worker(tmp_path, monkeypatch, batched, expected):
    import multiprocessing
    import queue
    from glossapi.corpus import batched_extract

    monkeypatch.setenv('GLOSSAPI_WORKER_LOG_VERBOSE', '0')
    # The real worker changes these process-local values; restore after inline execution.
    for key in ('CUDA_VISIBLE_DEVICES', 'GLOSSAPI_DOCLING_DEVICE'):
        monkeypatch.setenv(key, '')
    monkeypatch.setattr(phase_extract, '_maybe_import_torch',
        lambda **kw: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    calls, worker_args = [], []
    def prime(self, **kw):
        self.extractor = SimpleNamespace(pipeline_options=SimpleNamespace(
            accelerator_options=SimpleNamespace(device='cuda:0')),
            enable_accel=lambda **kw: None,
            extract_path=lambda paths, out, **kw: calls.append(('sequential', list(paths))))
    monkeypatch.setattr(Corpus, 'prime_extractor', prime)
    monkeypatch.setattr(batched_extract, 'corpus_batched_extract',
        lambda self, original, **kw: calls.append(('batched', kw)))
    monkeypatch.setattr(Corpus, '_finalize_font_repair_metadata', lambda *args: None)
    monkeypatch.setattr(phase_extract, '_estimate_extract_work_pages', lambda path: 1)

    class InlineProcess:
        pid = 12345
        exitcode = 0
        def __init__(self, target, args):
            self.target, self.args = target, args
            self.pid += len(worker_args)
            worker_args.append(args)
        def start(self):
            # Keep the worker's stdout/stderr redirection out of this test process.
            with monkeypatch.context() as worker_patch:
                worker_patch.delenv('GLOSSAPI_WORKER_LOG_DIR', raising=False)
                try:
                    self.target(*self.args)
                except SystemExit as exc:
                    self.exitcode = exc.code
        def join(self, **kw):
            pass
        def is_alive(self):
            return False
    context = SimpleNamespace(Manager=lambda: SimpleNamespace(dict=dict, shutdown=lambda: None),
                              Queue=queue.Queue, Process=InlineProcess)
    monkeypatch.setattr(multiprocessing, 'get_context', lambda method: context)
    source = tmp_path / 'downloads/doc.pdf'
    source.parent.mkdir()
    source.write_bytes(b'mocked preparation')
    Corpus(tmp_path, tmp_path).extract(phase1_backend='docling', use_gpus='multi',
        devices=[0, 1], batched=batched, skip_existing=False)
    assert len(worker_args) == 2
    assert all(args[-1] is expected for args in worker_args)
    assert [call[0] for call in calls] == ['batched' if expected else 'sequential']
    if expected:
        assert calls[0][1]['_auto_cuda'] is False


def test_safe_cuda_extract_stays_sequential(tmp_path, monkeypatch):
    from glossapi.corpus import batched_extract
    monkeypatch.setattr(phase_extract, '_maybe_import_torch',
        lambda **kw: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(batched_extract, 'corpus_batched_extract',
        lambda *args, **kw: pytest.fail('safe extraction must not batch'))
    calls = []
    def prime(**kw):
        calls.append(kw)
        corpus.extractor = SimpleNamespace(extract_path=lambda *args, **kw: None)
    corpus = Corpus(tmp_path, tmp_path)
    monkeypatch.setattr(corpus, 'prime_extractor', prime)
    monkeypatch.setattr(corpus, '_finalize_font_repair_metadata', lambda *args: None)
    source = tmp_path / 'downloads/doc.pdf'
    source.parent.mkdir()
    source.write_bytes(b'mocked preparation')
    corpus.extract(phase1_backend='auto')
    assert calls[0]['phase1_backend'] == 'safe'


def test_conversion_log_destination_override(tmp_path, monkeypatch):
    from glossapi import gloss_extract
    destination = tmp_path / 'conversion.log'
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(destination))
    monkeypatch.setattr(gloss_extract, 'LOGGING_INITIALIZED', False)
    extractor = gloss_extract.GlossExtract()
    assert extractor.log_file == destination
    assert destination.is_file()


@pytest.mark.parametrize('flagged', [False, True])
def test_replay_rejects_missing_join_even_for_flagged_pages(tmp_path, monkeypatch, flagged):
    import replay_t1_7_policy_defaults as replay_module
    archive = tmp_path / 'glossapi_improvement_20261006/node/archive/r_run'
    archive.mkdir(parents=True)
    (archive / 'page_quality.jsonl').write_text(json.dumps(dict(filename='known.pdf', pages=[])))
    monkeypatch.setattr(replay_module.pd, 'read_csv', lambda *args, **kw:
        pd.DataFrame([dict(doc='missing', page=1, doc_flagged=flagged)]))
    with pytest.raises(ValueError, match='Missing page join'):
        replay_module.replay(tmp_path)


def mock_recorded_non_greek_score(corpus, monkeypatch, stem):
    # Exercise routing with the archived score scale independently of the locally
    # installed noise extension, which may score pure-Latin synthetic text as 0.
    original = corpus._load_rust_extension
    def load(name, manifest, **kwargs):
        if name == 'glossapi_rs_noise':
            return SimpleNamespace(score_markdown_directory_detailed=lambda directory, threads:
                [(str(Path(directory) / f'{stem}.md'), 974, 99, 0, 0)])
        return original(name, manifest, **kwargs)
    monkeypatch.setattr(corpus, '_load_rust_extension', load)


@pytest.mark.parametrize('failed', [False, True])
def test_clean_readable_non_greek_keeps_tags_without_language_ocr(tmp_path, failed, monkeypatch):
    corpus = Corpus(tmp_path, tmp_path)
    mock_recorded_non_greek_score(corpus, monkeypatch, 'english')
    text = 'This document contains readable English prose about ordinary academic research. ' * 30
    (corpus.markdown_dir / 'english.md').write_text(text)
    if failed:
        path = tmp_path / 'sidecars/extract/english.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(extract_status='partial', no_partial_output=True)))
    corpus.clean()
    row = pd.read_parquet(tmp_path / 'download_results/download_results.parquet').iloc[0]
    assert {'non_greek_text', 'non_greek_readable'} <= set(row['filter'].split(';'))
    assert bool(row.needs_ocr) is failed
    if failed:
        assert 'extract_failed' in row['filter'].split(';')


def test_clean_garbage_non_greek_still_requires_ocr(tmp_path, monkeypatch):
    corpus = Corpus(tmp_path, tmp_path)
    mock_recorded_non_greek_score(corpus, monkeypatch, 'garbage')
    (corpus.markdown_dir / 'garbage.md').write_text('bcdfghjklmn ' * 100)
    corpus.clean()
    row = pd.read_parquet(tmp_path / 'download_results/download_results.parquet').iloc[0]
    assert 'non_greek_text' in row['filter'].split(';')
    assert 'non_greek_readable' not in row['filter'].split(';')
    assert row.needs_ocr


def test_default_v4_sidecar_cli_and_pipeline_metadata(quality_pdf, tmp_path):
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    report = page_quality(quality_pdf, wordlist=COMMON_GREEK, output_dir=tmp_path)
    assert report.rule_profile == 'v4' and report.fold_validity is False
    frame = _attach_page_quality_routes(pd.DataFrame([dict(filename='quality.pdf', needs_ocr=False)]),
                                        tmp_path / 'json/metrics')
    assert frame.iloc[0].page_rule_profile == 'v4'
    assert not frame.iloc[0].page_fold_validity
    output = tmp_path / 'batch.jsonl'
    assert page_quality_batch.main(['--pdfs', str(quality_pdf), '--out', str(output)]) == 0
    record = json.loads(output.read_text())
    assert record['rule_profile'] == 'v4' and record['fold_validity'] is False
