"""Requirement probes adapted from ASTRA-final/test_astra_independent.py."""

import json
from io import BytesIO
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw
import pytest


def outputs(tmp_path, original=None):
    fitz = pytest.importorskip("fitz", reason="PDF fixture generation requires optional PyMuPDF")
    from glossapi.ocr.page_artifacts import _write_outputs
    source = tmp_path / 'case.pdf'
    with fitz.open() as doc:
        for index in range(3):
            doc.new_page().insert_text((50, 70), f'Native page {index + 1}')
        doc.save(source)
    out = tmp_path / 'out'
    if original is not None:
        _write_outputs(output_dir=out, stem='case', markdown=original, page_count=3)

    def shard(first, last, text):
        _write_outputs(output_dir=out, stem=f'case__p{first:05}-{last:05}',
                       markdown=text, page_count=last-first+1,
                       extra_metrics={'source_start_page': first, 'source_end_page': last})

    def assemble(ranges):
        from glossapi.ocr import page_artifacts as runner
        return runner._reassemble_canonical_output_for_source(
            out_root=out, pdf_path=source, source_name='case.pdf', selected_ranges=ranges)
    return out, shard, assemble


def test_repeated_partial_ocr_replaces_obsolete_supplement(tmp_path):
    from glossapi.ocr.page_artifacts import _join_page_outputs
    original = '# Kept heading\n\n| Field | Value |\n|---|---|\n| x | y |\n\nOld appendix'
    out, shard, assemble = outputs(tmp_path, original)
    native = (out / 'markdown/case.md').read_bytes()
    shard(2, 3, _join_page_outputs(['OCR FIRST WRONG TWO', 'OCR FIRST WRONG THREE']))
    assert assemble([(2, 3)])
    first = (out / 'markdown/case.md').read_text()
    shard(2, 3, _join_page_outputs(['OCR SECOND CORRECT TWO', 'OCR SECOND CORRECT THREE']))
    assert assemble([(2, 3)])
    actual = (out / 'markdown/case.md').read_text()
    assert 'OCR FIRST WRONG' not in actual, 'Canonical rerun still contains obsolete OCR'
    assert 'OCR SECOND CORRECT TWO' in actual and 'OCR SECOND CORRECT THREE' in actual
    assert actual.startswith(native.decode())
    assert (out / 'sidecars/ocr_native/case.md').read_bytes() == native
    assert actual.count('<!-- ocr-pages:') == 1
    begin = re.search(r'<!-- ocr-supplement-begin: pages=2-3 run_id=(\w+) -->', actual)
    assert begin and f'<!-- ocr-supplement-end: pages=2-3 run_id={begin[1]} -->' in actual
    assert begin[1] not in first, 'Each replacement needs a fresh run id'


def test_missing_kept_markdown_must_not_succeed(tmp_path):
    out, shard, assemble = outputs(tmp_path)
    shard(3, 3, 'OCR PAGE THREE')
    with pytest.raises(FileNotFoundError, match='retained native Markdown'):
        assemble([(3, 3)])
    assert not (out / 'markdown/case.md').exists()
    assert not (out / 'json/metrics/case.metrics.json').exists()
    assert (out / 'markdown/case__p00003-00003.md').exists()


def test_scan_with_text_overlay_must_not_lose_raster_body(tmp_path):
    fitz = pytest.importorskip("fitz", reason="PDF fixture generation requires optional PyMuPDF")
    from glossapi.page_quality import page_quality
    image = Image.new('RGB', (550, 700), 'white')
    draw = ImageDraw.Draw(image)
    for y in range(50, 640, 25):
        draw.text((40, y), 'Important source text exists only in the scanned body.', fill='black')
    buf = BytesIO()
    image.save(buf, format='PNG')
    source = tmp_path / 'overlay_scan.pdf'
    with fitz.open() as doc:
        for index in range(2):
            page = doc.new_page(width=600, height=800)
            page.insert_image(fitz.Rect(20, 20, 580, 740), stream=buf.getvalue())
            if index == 0:
                page.insert_text((20, 780), 'ARCHIVE HEADER ' + ('copyright notice ' * 5), fontsize=5)
        doc.save(source)
    report = page_quality(source)
    assert report.pages[0].chars == 100
    assert all(p.image_area_ratio == pytest.approx(.8315151774) for p in report.pages)
    assert report.pages_needing_ocr == 2, 'Footer cannot certify representation of the raster body'
    assert report.ocr_page_list == '1-2'


def test_changed_selections_keep_only_current_ocr_per_page(tmp_path):
    from glossapi.ocr.page_artifacts import _join_page_outputs
    out, shard, assemble = outputs(tmp_path, '# Native heading\n\n| a | b |\n|---|---|\n| c | d |')
    native = (out / 'markdown/case.md').read_bytes()
    shard(2, 3, _join_page_outputs(['KEEP CURRENT TWO', 'OBSOLETE THREE']))
    assert assemble([(2, 3)])
    # Rebuild from native and saved generated pages even if canonical is absent.
    (out / 'markdown/case.md').unlink()
    shard(3, 3, 'CURRENT THREE')
    assert assemble([(3, 3)])
    actual = (out / 'markdown/case.md').read_text()
    assert actual.startswith(native.decode())
    assert 'KEEP CURRENT TWO' in actual and 'CURRENT THREE' in actual
    assert 'OBSOLETE THREE' not in actual
    assert (out / 'sidecars/ocr_native/case.md').read_bytes() == native


def test_marked_page_reruns_rebuild_from_native_and_keep_other_ocr(tmp_path):
    from glossapi.ocr.page_artifacts import _join_page_outputs, _split_page_outputs
    out, shard, assemble = outputs(tmp_path, _join_page_outputs(['Native one', 'Native two', 'Native three']))
    native = (out / 'markdown/case.md').read_bytes()
    shard(2, 2, 'CURRENT TWO')
    assert assemble([(2, 2)])
    (out / 'markdown/case.md').write_text('Accidentally changed canonical')
    shard(3, 3, 'CURRENT THREE')
    assert assemble([(3, 3)])
    assert _split_page_outputs((out / 'markdown/case.md').read_text()) == ['Native one', 'CURRENT TWO', 'CURRENT THREE']
    assert (out / 'sidecars/ocr_native/case.md').read_bytes() == native


@pytest.mark.parametrize('original', ['', '\n\n', 'Native\n<!-- ocr-pages: 3 -->\nold OCR\n<!-- /ocr-pages -->'])
def test_partial_ocr_refuses_unusable_native_baseline(tmp_path, original):
    out, shard, assemble = outputs(tmp_path, original)
    before = (out / 'markdown/case.md').read_bytes()
    shard(3, 3, 'CURRENT THREE')
    with pytest.raises(FileNotFoundError, match='retained native Markdown'):
        assemble([(3, 3)])
    assert (out / 'markdown/case.md').read_bytes() == before
    assert not (out / 'sidecars/ocr_native/case.md').exists()


def test_all_page_selection_needs_no_native_markdown(tmp_path):
    from glossapi.ocr.page_artifacts import _join_page_outputs, _split_page_outputs
    out, shard, assemble = outputs(tmp_path)
    shard(1, 3, _join_page_outputs(['OCR ONE', 'OCR TWO', 'OCR THREE']))
    assert assemble([(1, 3)])
    assert _split_page_outputs((out / 'markdown/case.md').read_text()) == ['OCR ONE', 'OCR TWO', 'OCR THREE']


@pytest.mark.parametrize('chars,coverage,parameters,needs', [
    (100, .499, {}, False),
    (100, .5, {}, True),
    (599, .5, {}, True),
    (600, .5, {}, False),
    (100, .5, {'image_dominant_ratio': .6}, False),
    (100, .5, {'image_dominant_min_chars': 100}, False),
    (600, .5, {'image_dominant_min_chars': 601}, True),
])
def test_image_density_thresholds_are_parameterized(chars, coverage, parameters, needs):
    from glossapi.page_quality import page_ocr_decision
    result, _ = page_ocr_decision(chars=chars, greek_chars=0, latin_chars=chars,
                                 other_chars=0, valid_greek_word_rate=None,
                                 image_area_ratio=coverage, rule_profile="v2", **parameters)
    assert result is needs
