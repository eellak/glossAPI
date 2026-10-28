"""Generated PDFs and offline contracts for page-level OCR routing."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
import pytest

from glossapi.page_quality import compress_page_ranges, page_ocr_decision, page_quality
from glossapi.pdf_repair.tounicode import _cmap_entries, extract_text_stats
from glossapi.scripts.page_quality_batch import batch, compare_doc_rule
from test_pdf_font_repair import _cmap  # Reuse E1's independent fixture CMap helper.


COMMON_GREEK = set("και το η ο σε με για από είναι ένα μία που δεν θα να του της στο στην ως αλλά αυτό αυτή ήταν έχει μπορεί μας σας τους τις".split())


@pytest.fixture
def quality_pdf(tmp_path, monkeypatch):
    """Build independent valid-text, image, corrupt-ToUnicode and blank pages."""
    fitz = pytest.importorskip("fitz", reason="PDF fixture generation requires optional PyMuPDF")
    monkeypatch.delenv("GLOSSAPI_GREEK_WORDLIST", raising=False)
    candidates = [Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                  Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
                  Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font = next((p for p in candidates if p.exists()), None)
    if font is None:
        pytest.skip("No local Greek TrueType fixture font")
    words = sorted(COMMON_GREEK)
    text = '\n'.join(' '.join(words[i:i + 6]) for i in range(0, len(words), 6))
    source = tmp_path / 'text.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_font(fontname='Greek', fontfile=str(font))
        page.insert_text((72, 72), text, fontname='Greek', fontsize=14)
        doc.save(source)
    writer = PdfWriter(clone_from=source)
    fontobj = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    mapping = _cmap_entries(fontobj['/ToUnicode'].get_data())
    shifted = {code: chr(ord(value) + 1) if len(value) == 1 and '\u0370' <= value <= '\u03ff' else value
               for code, value in mapping.items()}
    stream = DecodedStreamObject()
    stream.set_data(_cmap(shifted))
    fontobj[NameObject('/ToUnicode')] = writer._add_object(stream)
    corrupt = tmp_path / 'corrupt.pdf'
    writer.write(corrupt)
    target = tmp_path / 'quality.pdf'
    with fitz.open() as doc:
        with fitz.open(source) as good:
            doc.insert_pdf(good)
        page = doc.new_page()
        image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
        image.clear_with(128)
        page.insert_image(page.rect, stream=image.tobytes('png'))
        with fitz.open(corrupt) as broken:
            doc.insert_pdf(broken)
        doc.new_page()
        doc.save(target)
    return target


def test_four_page_decisions_and_sidecar(quality_pdf, tmp_path):
    report = page_quality(quality_pdf, wordlist=COMMON_GREEK, output_dir=tmp_path / 'out', dpi_probe=True)
    assert [p.page_needs_ocr for p in report.pages] == [False, True, True, False]
    assert [p.decision for p in report.pages] == ['text_layer', 'scanned_page', 'garbage_greek_text', 'blank_or_figure']
    assert report.pages[0].valid_greek_word_rate == 1
    assert report.pages[1].image_area_ratio == pytest.approx(1)
    assert report.pages[1].min_image_dpi is not None
    assert report.pages[2].valid_greek_word_rate < .4
    assert report.pages[3].chars == 0
    assert (report.pages_total, report.pages_needing_ocr, report.ocr_page_list) == (4, 2, '2-3')
    assert report.doc_needs_ocr_by_page_rule and report.share_pages_needing_ocr == .5
    sidecar = tmp_path / 'out/json/metrics/quality.page_quality.json'
    assert json.loads(sidecar.read_text()) == report.to_dict()
    e1 = extract_text_stats(quality_pdf, word_set=COMMON_GREEK)
    for page, expected in zip(report.pages, e1['pages']):
        assert (page.chars, page.greek_chars, page.latin_chars, page.words, page.valid_greek_word_rate) == (
            expected['chars'], expected['greek_chars'], expected['latin_chars'], expected['words'], expected['dictionary_valid_rate'])


def test_ranges():
    assert compress_page_ranges([22, 3, 4, 5, 6, 7, 8, 9, 14, 20, 21, 3]) == '3-9,14,20-22'
    assert compress_page_ranges([]) == ''
    with pytest.raises(ValueError):
        compress_page_ranges([0])


@pytest.mark.parametrize('changes,expected', [
    ({'chars': 49, 'image_area_ratio': .3}, (False, 'blank_or_figure')),
    ({'chars': 49, 'image_area_ratio': .31}, (True, 'scanned_page')),
    ({'valid_greek_word_rate': .4}, (False, 'text_layer')),
    ({'valid_greek_word_rate': .39}, (True, 'garbage_greek_text')),
    ({'valid_greek_word_rate': None}, (False, 'text_layer')),
    ({'greek_chars': 25, 'latin_chars': 25, 'valid_greek_word_rate': 0}, (False, 'text_layer')),
    ({'other_chars': 16}, (True, 'symbol_font_garbage')),
    ({'other_chars': 15}, (False, 'text_layer')),
])
def test_strict_thresholds(changes, expected):
    values = dict(chars=50, greek_chars=40, latin_chars=0, other_chars=0,
                  valid_greek_word_rate=1, image_area_ratio=0)
    assert page_ocr_decision(**(values | changes), rule_profile="v2") == expected


def test_wordlist_path_and_unknown_rate(quality_pdf, tmp_path):
    wordlist = tmp_path / 'words.txt'
    wordlist.write_text('\n'.join(COMMON_GREEK), encoding='utf-8')
    assert page_quality(quality_pdf, wordlist=wordlist).pages[0].valid_greek_word_rate == 1
    report = page_quality(quality_pdf)
    assert not report.dictionary_loaded
    assert all(p.valid_greek_word_rate is None for p in report.pages)


def test_batch_cli_errors_comparison_and_sidecars(quality_pdf, tmp_path):
    corrupt = tmp_path / 'bad.pdf'
    corrupt.write_bytes(b'not a PDF')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('filename,pages\nquality.pdf,4.0\nbad.pdf,3\nmissing.pdf,10\n')
    wordlist = tmp_path / 'words.txt'
    wordlist.write_text('\n'.join(COMMON_GREEK))
    pdf_list = tmp_path / 'pdfs.txt'
    pdf_list.write_text('quality.pdf\nbad.pdf\n')
    result = subprocess.run([sys.executable, '-m', 'glossapi.scripts.page_quality_batch',
        '--pdfs', str(pdf_list), '--out', str(tmp_path / 'report.jsonl'), '--workers', '2',
        '--wordlist', str(wordlist), '--compare-doc-rule', str(manifest),
        '--output-dir', str(tmp_path / 'sidecars')], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1, result.stderr
    records = [json.loads(line) for line in (tmp_path / 'report.jsonl').read_text().splitlines()]
    assert len(records) == 2
    assert next(r for r in records if r['filename'] == 'bad.pdf')['error']
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert (summary['pages_under_doc_rule'], summary['pages_under_page_rule'], summary['pages_saved']) == (17, 15, 2)
    assert summary['percent_saved'] == pytest.approx(200 / 17)
    assert summary['manifest_documents_missing'] == ['missing.pdf']
    assert summary['errors_retained_under_doc_rule'] == 1
    assert json.loads(result.stdout) == summary
    assert (tmp_path / 'sidecars/json/metrics/quality.page_quality.json').exists()
    next(r for r in records if r['filename'] == 'quality.pdf')['pages_total'] = 999
    with pytest.raises(ValueError, match='mismatch'):
        compare_doc_rule(records, manifest)


def test_batch_deadline_is_per_file(quality_pdf):
    records = list(batch([quality_pdf, quality_pdf], workers=2, timeout=.001))
    assert len(records) == 2
    assert all('Timeout' in row['error'] for row in records)


def test_clean_sidecar_routes_keep_document_flag(quality_pdf, tmp_path, caplog):
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    page_quality(quality_pdf, wordlist=COMMON_GREEK, output_dir=tmp_path)
    frame = pd.DataFrame([{'filename': 'quality.pdf', 'needs_ocr': True},
                          {'filename': 'kept.pdf', 'needs_ocr': False},
                          {'filename': 'absent.pdf', 'needs_ocr': True}])
    routed = _attach_page_quality_routes(frame, tmp_path / 'json/metrics')
    assert routed.needs_ocr.tolist() == frame.needs_ocr.tolist()
    assert routed.ocr_pages.tolist() == ['2-3', '', '']
    assert routed.ocr_pages_count.tolist() == [2, 0, 0]
    (tmp_path / 'json/metrics/quality.page_quality.json').write_text('{bad')
    assert _attach_page_quality_routes(routed, tmp_path / 'json/metrics').ocr_pages.tolist() == ['', '', '']
    assert 'Ignoring page-quality' in caplog.text


def test_corpus_forwards_parquet_ranges(tmp_path, monkeypatch):
    from glossapi import Corpus
    from glossapi.ocr.deepseek import runner
    corpus = Corpus(input_dir=tmp_path, output_dir=tmp_path)
    monkeypatch.setattr(corpus, 'clean', lambda **kwargs: None)
    for name in ['routed.pdf', 'whole.pdf', 'kept.pdf']:
        (tmp_path / name).write_bytes(b'%PDF-stub')
    parquet = tmp_path / 'download_results/download_results.parquet'
    parquet.parent.mkdir(exist_ok=True)
    pd.DataFrame([
        {'filename': 'routed.pdf', 'needs_ocr': True, 'ocr_pages': '3-9,14,20-22'},
        {'filename': 'whole.pdf', 'needs_ocr': True, 'ocr_pages': ''},
        {'filename': 'kept.pdf', 'needs_ocr': False, 'ocr_pages': '1'},
    ]).to_parquet(parquet, index=False)
    calls = []
    monkeypatch.setattr(runner, 'run_for_files', lambda self_ref, files, **kw: calls.append((files, kw)))
    corpus.ocr(backend='deepseek', mode='ocr_bad', math_enhance=False)
    assert calls[0][0] == ['routed.pdf', 'whole.pdf']
    assert calls[0][1]['page_ranges'] == ['routed.pdf:3:9', 'routed.pdf:14:14', 'routed.pdf:20:22']


@pytest.mark.parametrize('runtime', ['transformers', 'vllm'])
@pytest.mark.parametrize('multi', [False, True])
def test_runner_forwards_ranges_and_retains_existing_pages(quality_pdf, tmp_path, monkeypatch, runtime, multi):
    from types import SimpleNamespace
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs, _join_page_outputs, _split_page_outputs
    output = tmp_path / 'ocr-output'
    (output / 'markdown').mkdir(parents=True)
    # Existing canonical output must not mask new selected-range outputs.
    (output / 'markdown/quality.md').write_text(_join_page_outputs([
        '# Kept Greek heading\n' + ' '.join(sorted(COMMON_GREEK)),
        'stale text', 'stale text', '| kept | table |',
    ]))
    (output / 'json/metrics').mkdir(parents=True)
    (output / 'json/metrics/quality.metrics.json').write_text('{}')
    model = tmp_path / 'model'
    model.mkdir()
    monkeypatch.setattr(runner, 'resolve_deepseek_python', lambda **kw: Path(sys.executable))
    monkeypatch.setattr(runner, '_resolve_lane_devices', lambda **kw: [0])
    captured = {}

    def fake_cli(**kwargs):
        captured.update(kwargs)
        if multi:
            assert kwargs['file_list'] == ['quality.pdf']
        else:
            assert kwargs['files'] == []
        _write_outputs(output_dir=output, stem='quality__p00002-00003',
                       markdown=_join_page_outputs(['OCR image', 'OCR corrupted']), page_count=2,
                       extra_metrics={'source_start_page': 2, 'source_end_page': 3})

    monkeypatch.setattr(runner, '_run_multi_cli' if multi else '_run_cli', fake_cli)
    result = runner.run_for_files(SimpleNamespace(input_dir=quality_pdf.parent, output_dir=output),
                                 ['quality.pdf'], model_dir=model, runtime_backend=runtime,
                                 use_gpus='multi' if multi else 'single', page_ranges=['quality.pdf:2:3'])
    assert captured['page_ranges'] == ['quality.pdf:2:3']
    text = (output / 'markdown/quality.md').read_text()
    assert 'OCR image' in text and 'stale text' not in text
    assert all(word in text for word in COMMON_GREEK)
    assert _split_page_outputs(text)[1:3] == ['OCR image', 'OCR corrupted']
    assert result['quality']['page_count'] == 4


@pytest.mark.parametrize('planner', ['_plan_work_batches', '_plan_lane_batches'])
@pytest.mark.parametrize('scheduler', ['whole_doc', 'fixed_shard', 'exact_fill'])
def test_scheduling_intersects_selected_ranges(quality_pdf, planner, scheduler):
    from glossapi.ocr.deepseek import runner
    plans = getattr(runner, planner)(file_list=['quality.pdf'], input_root=quality_pdf.parent,
        max_pages=None, runtime_backend='vllm', scheduler=scheduler, lane_devices=[0, 1],
        workers_per_gpu=1, target_batch_pages=2, shard_pages=2, shard_threshold_pages=0,
        page_ranges=['quality.pdf:2:3'])
    batches = plans if planner == '_plan_work_batches' else [b for lane in plans for b in lane['batches']]
    items = [item for b in batches for item in b['items']]
    assert sum(item['pages'] for item in items) == 2
    assert {p for item in items for p in range(item['start_page'], item['end_page'] + 1)} == {2, 3}


def test_parquet_schema_defaults(tmp_path):
    from glossapi.parquet_schema import ParquetSchema
    schema = ParquetSchema()
    (tmp_path / 'downloads').mkdir()
    (tmp_path / 'downloads/a.pdf').write_bytes(b'%PDF-stub')
    path = schema.ensure_metadata_parquet(tmp_path)
    row = pd.read_parquet(path).iloc[0]
    assert row['ocr_pages'] == '' and row['ocr_pages_count'] == 0
    assert 'ocr_pages' in schema.METADATA_SCHEMA.names


def test_reassembly_rejects_missing_selected_pages(quality_pdf, tmp_path):
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs
    _write_outputs(output_dir=tmp_path, stem='quality__p00002-00002', markdown='OCR image',
                   page_count=1, extra_metrics={'source_start_page': 2, 'source_end_page': 2})
    with pytest.raises(RuntimeError, match='Incomplete selected OCR pages'):
        runner._ensure_canonical_outputs(out_root=tmp_path, pdf_root=quality_pdf.parent,
                                         file_list=['quality.pdf'], page_ranges=['quality.pdf:2:3'])
    assert not (tmp_path / 'markdown/quality.md').exists()


def test_all_pages_selection_can_write_canonical_directly(quality_pdf, tmp_path):
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs
    _write_outputs(output_dir=tmp_path, stem='quality', markdown='All pages OCR', page_count=4)
    runner._ensure_canonical_outputs(out_root=tmp_path, pdf_root=quality_pdf.parent,
                                     file_list=['quality.pdf'], page_ranges=['quality.pdf:1:4'])
    assert (tmp_path / 'markdown/quality.md').read_text().strip() == 'All pages OCR'


def test_reassembly_keeps_existing_page_markdown(quality_pdf, tmp_path):
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs, _join_page_outputs, _split_page_outputs
    original = ['# Existing heading', 'old scan', 'old garbage', '| kept | table |']
    _write_outputs(output_dir=tmp_path, stem='quality', markdown=_join_page_outputs(original), page_count=4)
    _write_outputs(output_dir=tmp_path, stem='quality__p00002-00003',
                   markdown=_join_page_outputs(['new scan', 'new text']), page_count=2,
                   extra_metrics={'source_start_page': 2, 'source_end_page': 3})
    runner._ensure_canonical_outputs(out_root=tmp_path, pdf_root=quality_pdf.parent,
                                     file_list=['quality.pdf'], page_ranges=['quality.pdf:2:3'])
    assert _split_page_outputs((tmp_path / 'markdown/quality.md').read_text()) == [original[0], 'new scan', 'new text', original[3]]
