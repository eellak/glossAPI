"""CP-K5b regressions: guarded PDFium cleanup and effective OCR budgets."""
from __future__ import annotations

import ast
import gc
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pypdfium2 as pdfium
import pytest

from glossapi.pdfium_guard import pdfium_guard
from glossapi.stages.contracts import ArtifactReference
from glossapi.stages.execution import sha256_file
from pdf_fixtures import FixturePdf


def _ref(kind, path):
    return ArtifactReference(kind, kind, path, sha256_file(path), path.stat().st_size)


def _raster_module():
    # Test the exact retained implementation without importing optional Docling.
    path = Path(__file__).parents[1] / 'src/glossapi/ocr/math/enrich.py'
    tree = ast.parse(path.read_text())
    names = {'PageRasterCache', '_close_raster', 'enrich_from_docling_json'}
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = {'__name__': 'glossapi.ocr.math.enrich', '__package__': 'glossapi.ocr.math', 'pdfium': pdfium}
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), scope)
    return scope


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / 'source.pdf'
    doc = FixturePdf()
    for _ in range(2):
        page = doc.new_page(width=72, height=72)
        doc.insert_text(page, (5, 20), 'Hello')
        doc.insert_image(page, (5, 30, 20, 45), (2, 2, (100, 120, 140)))
    doc.save(path)
    gc.collect()  # builders are outside the production audit
    return path


@pytest.fixture
def trace(monkeypatch, pdf):
    import pypdfium2.internal.bases as bases
    events = []
    original = bases._close_template
    close = bases.AutoCloseable.close
    def traced(*args, **kwargs):
        events.append(('finalizer', args[2], pdfium_guard()._is_owned()))
        return original(*args, **kwargs)
    def traced_close(self, *args, **kwargs):
        events.append(('close', type(self).__name__, pdfium_guard()._is_owned()))
        return close(self, *args, **kwargs)
    monkeypatch.setattr(bases, '_close_template', traced)
    monkeypatch.setattr(bases.AutoCloseable, 'close', traced_close)
    yield events
    gc.collect()
    assert events
    assert all(owned for _, _, owned in events), events


@pytest.mark.parametrize('failure', [None, 'render', 'to_pil', 'copy'])
def test_raster_handles_close_under_guard_even_on_failure(pdf, trace, monkeypatch, failure):
    scope = _raster_module()
    cache = scope['PageRasterCache'](pdf, max_cached_pages=1)
    def fail(*args, **kwargs):
        raise RuntimeError('injected')
    if failure == 'render':
        monkeypatch.setattr(pdfium.PdfPage, 'render', fail)
    elif failure == 'to_pil':
        monkeypatch.setattr(pdfium.PdfBitmap, 'to_pil', fail)
    elif failure == 'copy':
        from PIL import Image
        monkeypatch.setattr(Image.Image, 'copy', fail)
    try:
        if failure:
            with pytest.raises(RuntimeError, match='injected'):
                cache.get_pil(1, 72)
        else:
            image = cache.get_pil(1, 72)
            pixels = image.tobytes()
            gc.collect()  # exercise cyclic finalizers while the document is still open
            assert cache.get_pil(1, 72) is image
            cache.get_pil(2, 72)  # eviction cannot retain a PDFium buffer
            cache.close()
            assert image.tobytes() == pixels
    finally:
        cache.close()
    assert not any(kind == 'finalizer' and 'PdfPage' in name and not owned for kind, name, owned in trace)


def test_enrichment_closes_document_before_roi_or_options_failure(pdf, trace, monkeypatch, tmp_path):
    from glossapi.ocr.utils import json_io
    scope = _raster_module()
    scope['_code_formula_backend'] = lambda: (None,) * 5
    scope['_backend_identity'] = lambda _: {}
    monkeypatch.setattr(json_io, 'load_docling_json', lambda _: SimpleNamespace())
    with pytest.raises(RuntimeError, match='iterate_items'):
        scope['enrich_from_docling_json'](tmp_path/'doc.json', pdf, tmp_path/'out.md', tmp_path/'out.jsonl')
    assert any('PdfDocument' in name for _, name, _ in trace)


@pytest.mark.parametrize('site', ['legacy', 'native_fallback', 'quality', 'phase_count', 'extract_count',
    'repair_stats', 'repair_sample', 'repair_unicode', 'ocr_count', 'batched_count', 'cost_count', 'checkpoint_count', 'sample_count'])
@pytest.mark.parametrize('failure', [False, True])
def test_every_pdfium_site_closes_under_guard(pdf, trace, monkeypatch, tmp_path, site, failure):
    def fail(*args, **kwargs):
        raise RuntimeError('injected')
    if site in ('legacy', 'native_fallback', 'quality', 'repair_stats', 'repair_sample', 'repair_unicode'):
        if failure:
            monkeypatch.setattr(pdfium.PdfTextPage, 'get_text_range', fail)
        if site == 'repair_stats':
            from glossapi.pdf_repair.text_stats import extract_text_stats
            action = lambda: extract_text_stats(pdf)
        elif site == 'repair_sample':
            from glossapi.pdf_repair.rust_backend import _sample_json
            action = lambda: _sample_json(str(pdf))
        elif site == 'repair_unicode':
            from glossapi.pdf_repair.tounicode import _sample_text
            action = lambda: _sample_text(pdf)
        elif site == 'quality':
            from glossapi.page_quality import page_quality
            action = lambda: page_quality(pdf)
        else:
            from glossapi.markdown_layout.reader import LegacySource, NativeSource
            class Native:
                def layout_v2_read_document(self, *args, **kwargs):
                    return 2, [(1, None, None, 'fallback')]
            source = LegacySource(pdf, chars=True) if site == 'legacy' else NativeSource(Native(), pdf)
            def action():
                try:
                    source.text(0)
                    source.reread_chars(0)
                finally:
                    source.close()
    else:
        if failure:
            monkeypatch.setattr(pdfium.PdfDocument, '__len__', fail)
        if site == 'phase_count':
            from glossapi.corpus.phase_extract import _estimate_extract_work_pages
            action = lambda: _estimate_extract_work_pages(pdf)
        elif site == 'extract_count':
            from glossapi.gloss_extract import GlossExtract
            action = lambda: GlossExtract._get_pdf_page_count(SimpleNamespace(_log=logging.getLogger(__name__)), pdf)
        elif site == 'ocr_count':
            from glossapi.ocr.page_artifacts import _page_count
            action = lambda: _page_count(pdf)
        elif site == 'batched_count':
            from glossapi.scripts.batched_extract_benchmark import page_count
            action = lambda: page_count(pdf)
        elif site == 'cost_count':
            from glossapi.scripts.docling_cost_benchmark import _count_pages
            action = lambda: _count_pages(pdf)
        elif site == 'checkpoint_count':
            from glossapi.scripts.extract_checkpoint_benchmark import _count_pdf_pages
            action = lambda: _count_pdf_pages(pdf)
        else:
            import pandas as pd
            from glossapi.scripts import sample_report as module
            manifest = tmp_path/'manifest.csv'
            manifest.write_text('filename,stratum\nsource.pdf,accepted_control\n')
            monkeypatch.setattr(module.pd, 'read_parquet', lambda _: pd.DataFrame([
                dict(filename='source.pdf', ocr_pages='', needs_ocr=False, extract_status='ok', pua_char_count=0, control_char_count=0)]))
            monkeypatch.setattr(module, '_enrich_extraction_metrics', lambda frame, *args: frame)
            monkeypatch.setattr(module, '_attach_page_quality_routes', lambda frame, *args: frame)
            action = lambda: module.build_report(manifest=manifest, parquet=tmp_path/'input.parquet', output_dir=tmp_path, pdf_dir=pdf.parent)
    try:
        action()
    except RuntimeError as error:
        assert failure and str(error) == 'injected'


def test_reader_constructor_closes_on_count_failure(pdf, trace, monkeypatch):
    from glossapi.markdown_layout.reader import LegacySource
    def fail(_):
        raise RuntimeError('injected')
    monkeypatch.setattr(pdfium.PdfDocument, '__len__', fail)
    with pytest.raises(RuntimeError, match='injected'):
        LegacySource(pdf, chars=False)


@pytest.mark.parametrize('maximum,expected', [(None, 20), (15, 15), (3, 3)])
def test_glyph_budget_matches_effective_selection(tmp_path, maximum, expected):
    from glossapi.ocr.deadline import attempt_budget
    from glossapi.ocr.selection import effective_selection
    path = tmp_path/'quality.json'
    path.write_text(json.dumps({'pages_total':20, 'pages':[
        {'page':page, 'page_needs_ocr':False} for page in range(1,21)]}))
    native = tmp_path/'native.md'
    native.write_text(' '.join(f'GLYPH<{n}>' for n in range(100)))
    refs = [_ref('page_quality', path), _ref('native_markdown', native)]
    settings = {} if maximum is None else {'max_pages':maximum}
    selected, glyph = effective_selection(refs, settings)
    assert glyph and selected == list(range(1, expected+1))
    seconds, receipt = attempt_budget(refs, settings)
    assert receipt['pages'] == len(selected)
    assert seconds == max(600, 120+45*expected)
    assert attempt_budget(refs, {**settings, 'timeout_seconds':12})[0] == 12
    native.write_text('changed')
    with pytest.raises(ValueError, match='input_hash_changed'):
        effective_selection(refs, settings)
    assert attempt_budget(refs, settings)[1]['pages'] is None


def test_real_twenty_page_glyph_report_gets_1020_seconds(tmp_path):
    from glossapi.page_quality import page_quality
    from glossapi.ocr.deadline import attempt_budget
    from glossapi.ocr.selection import effective_selection
    path = tmp_path/'glyphs.pdf'
    doc = FixturePdf()
    tokens = ' '.join(f'GLYPH<{n}>' for n in range(10))
    for _ in range(20):
        page = doc.new_page()
        doc.insert_text(page, (20, 40), '\n'.join([tokens]*12), fontsize=8)
    doc.save(path)
    report = page_quality(path).to_dict()
    assert sum(row['page_needs_ocr'] for row in report['pages']) == 0
    quality = tmp_path/'quality.json'
    quality.write_text(json.dumps(report))
    native = tmp_path/'native.md'
    native.write_text('\n'.join([tokens]*20))
    refs = [_ref('source_payload', path), _ref('page_quality', quality), _ref('native_markdown', native)]
    assert effective_selection(refs, {}) == (list(range(1, 21)), True)
    assert attempt_budget(refs, {})[0] == 1020
