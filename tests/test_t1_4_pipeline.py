"""Offline regression contracts for the CP-R2 pipeline defects."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pypdf import PdfWriter
import pytest

from glossapi import Corpus
from glossapi.gloss_extract import GlossExtract, ConversionStatus
from glossapi.page_quality import page_ocr_decision, page_quality


def make_pdf(path, pages):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    writer.write(path)
    return path


def seed(root, rows):
    target = root / 'download_results/download_results.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target


def route(root, name, ranges='2', total=3, count=1):
    target = root / 'json/metrics' / f'{Path(name).stem}.page_quality.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(filename=name, ocr_page_list=ranges,
                                     pages_total=total, pages_needing_ocr=count)))


@pytest.mark.parametrize('status,has_md', [('problematic', False), ('timeout', True),
    ('success', False), ('failure', True), ('error', True)])
def test_clean_failed_extraction_is_whole_document_fallback(tmp_path, status, has_md):
    corpus = Corpus(tmp_path, tmp_path)
    parquet = seed(tmp_path, [dict(filename='failed.pdf', extract_status=status, needs_ocr=False,
                                  ocr_success=True)])
    if has_md:
        (corpus.markdown_dir / 'failed.md').write_text('Plain readable content.')
    route(tmp_path, 'failed.pdf')
    corpus.clean()
    row = pd.read_parquet(parquet).iloc[0]
    assert 'extract_failed' in row['filter'].split(';')
    assert row.needs_ocr
    assert row.ocr_pages == '' and row.ocr_pages_count == 0
    assert row.extract_status == status


def test_clean_reads_sidecar_status_and_counts_raw_markdown(tmp_path):
    corpus = Corpus(tmp_path, tmp_path)
    parquet = seed(tmp_path, [dict(filename='good.pdf', extract_status='problematic'),
                              dict(filename='bad.pdf', extract_status='success')])
    text = 'Readable text.\n\t\r' + '\ue000\uf8ff\ud7ff\uf900' + '\x00\x1f\x7f\x85\x9f'
    (corpus.markdown_dir / 'good.md').write_text(text)
    sc = tmp_path / 'sidecars/extract'
    sc.mkdir(parents=True)
    (sc / 'good.json').write_text(json.dumps(dict(extract_status='ok', page_count=3)))
    (sc / 'bad.json').write_text(json.dumps(dict(extract_status='timeout', page_count=7)))
    corpus.clean(write_cleaned_files=False)
    rows = pd.read_parquet(parquet).set_index('filename')
    assert rows.loc['good.pdf', 'extract_status'] == 'ok'
    assert rows.loc['bad.pdf', 'extract_status'] == 'timeout'
    assert rows.loc['good.pdf', 'pua_char_count'] == 2
    assert rows.loc['good.pdf', 'control_char_count'] == 5
    assert pd.isna(rows.loc['bad.pdf', 'pua_char_count'])
    assert rows.loc['good.pdf', 'page_count'] == 3


@pytest.mark.parametrize('status,no_partial_output', [
    ('partial', True), ('partial', False), ('pending', False), ('ok', True)])
def test_clean_incomplete_sidecar_cannot_accept_stale_markdown(tmp_path, status, no_partial_output):
    corpus = Corpus(tmp_path, tmp_path)
    parquet = seed(tmp_path, [dict(filename='x.pdf', extract_status='ok', needs_ocr=False)])
    (corpus.markdown_dir / 'x.md').write_text('Readable Markdown left by a prior run.')
    sidecar = tmp_path / 'sidecars/extract/x.json'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps(dict(extract_status=status, no_partial_output=no_partial_output)))
    route(tmp_path, 'x.pdf')  # Stale page routes must not restrict the fallback.
    corpus.clean()
    row = pd.read_parquet(parquet).iloc[0]
    assert row.extract_status == status
    assert 'extract_failed' in row['filter'].split(';')
    assert row.needs_ocr
    assert row.ocr_pages == '' and row.ocr_pages_count == 0


def test_safe_extract_populates_parquet_page_count_and_status(tmp_path):
    downloads = tmp_path / 'downloads'
    downloads.mkdir()
    make_pdf(downloads / 'safe.pdf', 3)
    corpus = Corpus(tmp_path, tmp_path)
    corpus.extract(phase1_backend='safe', repair_fonts=False, accel_type='CPU')
    row = pd.read_parquet(tmp_path / 'download_results/download_results.parquet').iloc[0]
    assert row.page_count == 3
    assert row.extract_status == 'ok'
    sc = json.loads((tmp_path / 'sidecars/extract/safe.json').read_text())
    assert sc['extract_status'] == 'ok' and sc['page_count'] == 3


def test_safe_long_pdf_never_uses_docling_chunks(tmp_path, monkeypatch):
    source = make_pdf(tmp_path / 'long.pdf', 601)
    out = tmp_path / 'markdown'
    out.mkdir()
    extractor = GlossExtract()
    extractor.repair_fonts = False
    extractor.configure_batch_policy('safe', prefer_safe_backend=True)
    monkeypatch.setattr(extractor, '_process_file_chunked', lambda *a, **kw: pytest.fail('safe must stay on PDFium'))
    assert extractor._process_batch([source], out) == (['long.pdf'], [])
    assert json.loads((tmp_path / 'sidecars/extract/long.json').read_text())['page_count'] == 601


def test_docling_export_page_count_reaches_parquet(tmp_path):
    source = make_pdf(tmp_path / 'docling.pdf', 4)
    corpus = Corpus(tmp_path, tmp_path)
    seed(tmp_path, [dict(filename=source.name, page_count=None)])
    result = SimpleNamespace(status=ConversionStatus.SUCCESS, input=SimpleNamespace(file=source),
                             document=SimpleNamespace(export_to_markdown=lambda: 'Greek prose'))
    extractor = GlossExtract()
    assert extractor._export_documents([result], corpus.markdown_dir) == (1, 0, 0)
    corpus._finalize_font_repair_metadata([source])
    row = pd.read_parquet(tmp_path / 'download_results/download_results.parquet').iloc[0]
    assert row.page_count == 4 and row.extract_status == 'success'


@pytest.mark.parametrize('mode,expected', [('flagged', ['bad.pdf', 'whole.pdf']),
                                          ('all', ['bad.pdf', 'whole.pdf', 'kept.pdf'])])
def test_ocr_routing_modes_and_unflagged_count(tmp_path, monkeypatch, mode, expected):
    from glossapi.ocr.deepseek import runner
    corpus = Corpus(tmp_path, tmp_path)
    monkeypatch.setattr(corpus, 'clean', lambda **kw: None)
    rows = [dict(filename='bad.pdf', needs_ocr=True, ocr_pages='2-3'),
            dict(filename='whole.pdf', needs_ocr=True, ocr_pages=''),
            dict(filename='kept.pdf', needs_ocr=False, ocr_pages='1,3'),
            dict(filename='blank.pdf', needs_ocr=False, ocr_pages='')]
    parquet = seed(tmp_path, rows)
    for row in rows:
        make_pdf(tmp_path / row['filename'], 3)
    calls = []
    monkeypatch.setattr(runner, 'run_for_files', lambda self, files, **kw: calls.append((files, kw)))
    corpus.ocr(mode='ocr_bad', math_enhance=False, page_routing=mode)
    assert calls[0][0] == expected
    ranges = ['bad.pdf:2:3'] + (['kept.pdf:1:1', 'kept.pdf:3:3'] if mode == 'all' else [])
    assert calls[0][1]['page_ranges'] == ranges
    assert pd.read_parquet(parquet).ocr_pages_unflagged_count.tolist() == [0, 0, 2, 0]


def test_ocr_routing_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match='page_routing'):
        Corpus(tmp_path, tmp_path).ocr(page_routing='typo')


def test_clean_retains_unflagged_page_evidence(tmp_path):
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    route(tmp_path, 'kept.pdf', '1,3', count=2)
    rows = _attach_page_quality_routes(pd.DataFrame([dict(filename='kept.pdf', needs_ocr=False)]), tmp_path / 'json/metrics')
    assert not rows.iloc[0].needs_ocr
    assert rows.iloc[0].ocr_pages == '1,3'
    assert rows.iloc[0].ocr_pages_count == rows.iloc[0].ocr_pages_unflagged_count == 2


def test_sparse_pua_page_routes_before_blank_gate():
    assert page_ocr_decision(chars=20, greek_chars=0, latin_chars=0, other_chars=20,
        pua_char_count=20, control_char_count=0, valid_greek_word_rate=None,
        image_area_ratio=0, words=0) == (True, 'symbol_font_garbage')


def test_page_quality_exposes_contamination_counts(tmp_path):
    report = page_quality(make_pdf(tmp_path / 'blank.pdf', 1))
    assert report.pages[0].pua_char_count == report.pages[0].control_char_count == 0


def test_report_dedupes_membership_counts_pdf_pages_and_failures(tmp_path):
    from glossapi.scripts.sample_report import build_report
    pdfs = tmp_path / 'pdfs'
    pdfs.mkdir()
    for name, pages in [('overlap.pdf', 3), ('failed.pdf', 5), ('kept.pdf', 4)]:
        make_pdf(pdfs / name, pages)
    manifest = tmp_path / 'manifest.csv'
    pd.DataFrame([dict(filename='overlap.pdf', stratum='accepted_control', pages=999),
                  dict(filename='overlap.pdf', stratum='non_greek_mid', pages=999),
                  dict(filename='failed.pdf', stratum='long_gt250', pages=None),
                  dict(filename='kept.pdf', stratum='accepted_control', pages=999)]).to_csv(manifest, index=False)
    parquet = seed(tmp_path, [dict(filename='overlap.pdf', needs_ocr=True, extract_status='success'),
                              dict(filename='failed.pdf', needs_ocr=False, extract_status='problematic'),
                              dict(filename='kept.pdf', needs_ocr=False, extract_status='success')])
    md = tmp_path / 'markdown'
    md.mkdir()
    for name in ['overlap', 'kept']:
        (md / f'{name}.md').write_text('Readable text')
    route(tmp_path, 'overlap.pdf')
    route(tmp_path, 'failed.pdf', total=5)
    route(tmp_path, 'kept.pdf', '1-2', total=4, count=2)
    result = build_report(manifest=manifest, parquet=parquet, pdf_dir=pdfs, output_dir=tmp_path)
    total = result['total']
    assert total['docs'] == 3 and total['source_pages'] == 12
    assert total['extract_failed'] == 1 and total['extract_failed_pages'] == 5
    assert total['dispatch_flagged'] == 6 and total['dispatch_all'] == 8
    assert total['ocr_pages_unflagged_count'] == 2
    assert next(r for r in result['strata'] if r['stratum'] == 'accepted_control')['docs'] == 1
    assert result['duplicate_memberships'] == ['overlap.pdf']


@pytest.mark.parametrize('mode,expected_ranges', [('flagged', []), ('all', ['done.pdf:2:2'])])
def test_completed_unflagged_rerun_keeps_default_whole_document_policy(tmp_path, monkeypatch, mode, expected_ranges):
    from glossapi.ocr.deepseek import runner
    corpus = Corpus(tmp_path, tmp_path)
    make_pdf(tmp_path / 'done.pdf', 3)
    seed(tmp_path, [dict(filename='done.pdf', needs_ocr=False, ocr_success=True, ocr_pages='2')])
    monkeypatch.setattr(corpus, 'clean', lambda **kw: None)
    calls = []
    monkeypatch.setattr(runner, 'run_for_files', lambda self, files, **kw: calls.append((files, kw)))
    corpus.ocr(mode='ocr_bad', math_enhance=False, reprocess_completed=True, page_routing=mode)
    assert calls[0][0] == ['done.pdf']
    assert calls[0][1].get('page_ranges', []) == expected_ranges
