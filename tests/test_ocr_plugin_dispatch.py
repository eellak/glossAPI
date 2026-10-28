"""External-engine routing contracts; no OCR checkpoint needed for unit tests."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.corpus.ocr import dispatch
from glossapi.ocr import page_artifacts


def make_corpus(tmp_path, monkeypatch, pages=4):
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=300)
    source = tmp_path / 'source.pdf'
    writer.write(source)
    corpus = Corpus(tmp_path, tmp_path)
    parquet = tmp_path / 'download_results/download_results.parquet'
    parquet.parent.mkdir(exist_ok=True)
    pd.DataFrame([dict(filename=source.name, needs_ocr=True, ocr_pages='2-3')]).to_parquet(parquet)
    monkeypatch.setattr(corpus, '_refresh_metrics_after_ocr_rerun', lambda: None)
    return corpus, parquet


def test_missing_plugin_has_install_error_and_preserves_pending_state(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch, 'entry_points', lambda **kw: [])
    with pytest.raises(RuntimeError, match='Install an OCR plugin'):
        corpus.ocr(mode='ocr_bad')
    assert bool(pd.read_parquet(parquet).iloc[0].needs_ocr)
    assert not (tmp_path / 'markdown/source.md').exists()


def test_installed_plugin_requires_explicit_configuration(tmp_path, monkeypatch):
    corpus, _ = make_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch, 'entry_points', lambda **kw: [SimpleNamespace(name='glossapi')])
    with pytest.raises(ValueError, match='Configure an installed OCR plugin'):
        corpus.ocr(mode='ocr_bad')


def test_partial_ranges_keep_native_text_and_export_each_physical_page(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    page_artifacts._write_outputs(tmp_path, 'source', page_artifacts._join_page_outputs(
        ['# Native heading', 'obsolete two', 'obsolete three', '| Native | Table |']), 4)
    # Stale shards must be removed before conversion, while canonical/native text survives.
    page_artifacts._write_outputs(tmp_path, 'source__p00002-00004', 'STALE', 3)
    calls = []
    options = SimpleNamespace(kind='external-test')
    def build(request):
        assert request.ocr_options is options
        def convert(source, *, page_range):
            assert not (tmp_path / 'markdown/source__p00002-00004.md').exists()
            calls.append((source, page_range))
            return SimpleNamespace(status='success', document=SimpleNamespace(
                export_to_markdown=lambda *, page_no: f'OCR page {page_no}'))
        return SimpleNamespace(convert=convert)
    monkeypatch.setattr(dispatch, '_build_converter', build)
    corpus.ocr(mode='ocr_bad', ocr_options=options)
    assert calls == [(tmp_path / 'source.pdf', (2, 3))]
    assert page_artifacts._split_page_outputs((tmp_path / 'markdown/source.md').read_text()) == [
        '# Native heading', 'OCR page 2', 'OCR page 3', '| Native | Table |']
    row = pd.read_parquet(parquet).iloc[0]
    assert row.ocr_success and not row.needs_ocr and row.extraction_mode == 'docling'
    assert row.text == (tmp_path / 'markdown/source.md').read_text()


def test_partial_dispatch_requires_native_before_model_load(tmp_path, monkeypatch):
    corpus, _ = make_corpus(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatch, '_build_converter', lambda _: pytest.fail('must validate native content first'))
    with pytest.raises(FileNotFoundError, match='retained native Markdown'):
        corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))


@pytest.mark.parametrize('status', ['failure', 'partial_success'])
def test_failed_engine_does_not_mark_success(tmp_path, monkeypatch, status):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    frame = pd.read_parquet(parquet); frame['ocr_pages'] = ''; frame.to_parquet(parquet)
    monkeypatch.setattr(dispatch, '_build_converter', lambda _: SimpleNamespace(
        convert=lambda *a, **kw: SimpleNamespace(status=status)))
    with pytest.raises(RuntimeError, match='OCR plugin failed'):
        corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    assert bool(pd.read_parquet(parquet).iloc[0].needs_ocr)
    assert not (tmp_path / 'markdown/source.md').exists()


def test_whole_document_routes_empty_range(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    frame = pd.read_parquet(parquet); frame['ocr_pages'] = ''; frame.to_parquet(parquet)
    calls = []
    def convert(source, *, page_range):
        calls.append(page_range)
        return SimpleNamespace(status='success', document=SimpleNamespace(
            export_to_markdown=lambda *, page_no: str(page_no)))
    monkeypatch.setattr(dispatch, '_build_converter', lambda _: SimpleNamespace(convert=convert))
    corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    assert calls == [(1, 4)]
    assert page_artifacts._split_page_outputs((tmp_path / 'markdown/source.md').read_text()) == ['1', '2', '3', '4']


def test_math_only_needs_no_ocr_plugin(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    frame = pd.read_parquet(parquet); frame['needs_ocr'] = False; frame.to_parquet(parquet)
    (tmp_path / 'json').mkdir(exist_ok=True)
    (tmp_path / 'json/source.docling.json').write_text('{}')
    calls = []
    monkeypatch.setattr(corpus, 'formula_enrich_from_json', lambda **kw: calls.append(kw))
    monkeypatch.setattr(dispatch, 'entry_points', lambda **kw: pytest.fail('math must not inspect OCR plugins'))
    corpus.ocr(mode='math_only', device='cpu')
    assert calls[0]['files'] == ['source']


def test_downloaded_source_and_chunk_metadata_resolve_to_original_pdf(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    downloads = tmp_path / 'downloads'
    downloads.mkdir(exist_ok=True)
    (tmp_path / 'source.pdf').replace(downloads / 'source.pdf')
    frame = pd.read_parquet(parquet)
    frame['filename'] = 'source__p0001-0004.pdf'
    frame['ocr_pages'] = ''
    frame.to_parquet(parquet)
    calls = []
    def convert(source, *, page_range):
        calls.append((source, page_range))
        return SimpleNamespace(status='success', document=SimpleNamespace(
            export_to_markdown=lambda *, page_no: str(page_no)))
    monkeypatch.setattr(dispatch, '_build_converter', lambda _: SimpleNamespace(convert=convert))
    corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    assert calls == [(downloads / 'source.pdf', (1, 4))]
    assert pd.read_parquet(parquet).iloc[0].ocr_success


@pytest.mark.parametrize('pages', [['page one', '', 'page three'], ['', 'page two'], ['', '', '']])
def test_numbered_blank_pages_roundtrip(pages, tmp_path):
    markdown = page_artifacts._join_page_outputs(pages)
    page_artifacts._write_outputs(tmp_path, 'blank-pages', markdown, len(pages))
    assert page_artifacts._split_page_outputs((tmp_path / 'markdown/blank-pages.md').read_text()) == pages
