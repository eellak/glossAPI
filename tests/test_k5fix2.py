"""CP-K5b-A: setup failures own cleanup; invalid selections cannot pass routing."""
from dataclasses import replace
import gc
import json
import threading
from types import SimpleNamespace

import pypdfium2 as pdfium
import pytest

from glossapi.pdfium_guard import pdfium_guard
from glossapi.ocr.selection import effective_selection
from glossapi.stages.execution import ReceiptCollector
from glossapi.stages.ocr import OcrStage
from test_k5fix1 import _ref
from test_ledger_stages import request, simple_pdf


@pytest.mark.parametrize('entry', ['extract', 'label_pdf'])
@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_fallback_setup_failure_closes_before_traceback_release(tmp_path, monkeypatch, entry, failure):
    from glossapi import markdown_layout as ml
    from glossapi.markdown_layout import reader
    from glossapi.gloss_extract import GlossExtract
    import pypdfium2.internal.bases as bases

    path = simple_pdf(tmp_path/'source.pdf')
    gc.collect()
    native = reader.native_module()[0]
    assert native is not None
    def unavailable(*args, **kwargs):
        raise native.LayoutV2ReaderUnavailable('injected unavailable reader')
    monkeypatch.setattr(native, 'layout_v2_read_document', unavailable)
    monkeypatch.setenv(ml.ENVIRONMENT, 'layout_v2')
    events, retained, documents = [], [], []
    original_close, original_finalizer = bases.AutoCloseable.close, bases._close_template
    def close(self, *args, **kwargs):
        events.append(('close', type(self).__name__, pdfium_guard()._is_owned()))
        return original_close(self, *args, **kwargs)
    def finalize(*args, **kwargs):
        events.append(('finalizer', args[2], pdfium_guard()._is_owned()))
        return original_finalizer(*args, **kwargs)
    monkeypatch.setattr(bases.AutoCloseable, 'close', close)
    monkeypatch.setattr(bases, '_close_template', finalize)
    original_open = ml.open_page_source
    def opened(*args, **kwargs):
        source = original_open(*args, **kwargs)
        assert isinstance(source, reader.LegacySource) and source.reader == 'python'
        documents.append(source.doc)
        return source
    monkeypatch.setattr(ml, 'open_page_source', opened)
    def labeller(*args, **kwargs):
        try:
            raise failure('injected labeller setup')
        except BaseException as error:
            retained.append(error)  # the traceback keeps the real fallback source alive
            raise
    monkeypatch.setattr(ml, 'DocumentLabeller', labeller)
    if entry == 'extract':
        extractor = GlossExtract()
        monkeypatch.setattr(extractor, '_prepare_font_repair', lambda path, _: path)
        output = tmp_path/'markdown'
        output.mkdir()
        action = lambda: extractor._process_single_document_pypdfium(path, output_dir=output)
    else:
        action = lambda: ml.label_pdf(path)
    if entry == 'extract' and failure is RuntimeError:
        assert action() is False
    else:
        with pytest.raises(failure, match='labeller setup'):
            action()
    assert retained and documents and all(doc.raw is None for doc in documents)
    documents.clear()
    held, release = threading.Event(), threading.Event()
    def holder():
        with pdfium_guard():
            held.set()
            assert release.wait(10)
    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(10)
        retained.clear()
        gc.collect()  # current thread does not own the guard; another thread does
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert events and all(owned for _, _, owned in events), events


BAD_CAPS = [0, -1, 1.5, '2', float('nan'), float('inf'), True, False]


@pytest.mark.parametrize('maximum', BAD_CAPS)
@pytest.mark.parametrize('route', ['glyph', 'empty', 'whole', 'ranges'])
def test_invalid_cap_rejected_before_any_selection_return(tmp_path, maximum, route):
    refs = []
    settings = {'max_pages': maximum}
    if route == 'ranges':
        settings['page_ranges'] = [[1, 2]]
    if route in ('glyph', 'empty'):
        native = tmp_path/'native.md'
        native.write_text(' '.join(f'GLYPH<{i}>' for i in range(100)) if route == 'glyph' else 'Native text')
        quality = tmp_path/'quality.json'
        quality.write_text(json.dumps({'pages_total': 2, 'pages': [
            {'page': i, 'page_needs_ocr': False} for i in (1, 2)]}))
        refs = [_ref('page_quality', quality), _ref('native_markdown', native)]
    with pytest.raises(ValueError, match='max_pages'):
        effective_selection(refs, settings)


@pytest.mark.parametrize('maximum', BAD_CAPS)
@pytest.mark.parametrize('glyph', [False, True])
def test_stage_invalid_cap_is_selection_failure(tmp_path, monkeypatch, maximum, glyph):
    from glossapi.corpus.ocr import dispatch
    monkeypatch.setattr(dispatch, 'entry_points', lambda **_: [SimpleNamespace(name='glossapi')])
    monkeypatch.setattr(dispatch, 'run_plugin_ocr', lambda *a, **k: pytest.fail('invalid cap reached dispatch'))
    source = simple_pdf(tmp_path/'source.pdf')
    native = tmp_path/'native.md'
    native.write_text(' '.join(f'GLYPH<{i}>' for i in range(100)) if glyph else 'Native text')
    quality = tmp_path/'quality.json'
    quality.write_text(json.dumps({'pages_total': 1, 'pages': [{'page': 1, 'page_needs_ocr': False}]}))
    given = request(tmp_path, source, operation='ocr.document', max_pages=maximum)
    given = replace(given, inputs=(*given.inputs, _ref('native_markdown', native), _ref('page_quality', quality)))
    result = OcrStage().run_document(given, reporter=ReceiptCollector())
    assert (result.status, result.reason_code) == ('failed', 'ocr_page_selection_invalid')


@pytest.mark.parametrize('ranges', [[[0, 1]], [[2, 1]], [[True, 2]], [[1.5, 2]], [['1', 2]], [[1]], '1:2', False, 0])
def test_invalid_ranges_rejected_even_with_empty_report(ranges):
    with pytest.raises(ValueError, match='page range'):
        effective_selection([], {'page_ranges': ranges}, report={'pages_total': 2, 'pages': []})


@pytest.mark.parametrize('maximum,expected', [(None, [1, 2, 3]), (1, [1]), (2, [1, 2]), (9, [1, 2, 3])])
def test_valid_caps_keep_physical_page_ceiling(maximum, expected):
    assert effective_selection([], {'max_pages': maximum, 'page_ranges': [[1, 3]]}) == (expected, False)


@pytest.mark.parametrize('report', [
    [], {'pages_total': float('nan'), 'pages': []},
    {'pages_total': True, 'pages': []}, {'pages_total': -1, 'pages': []},
    {'pages_total': 1, 'pages': [{'page': 0, 'page_needs_ocr': False}]},
    {'pages_total': 1, 'pages': [{'page': 2, 'page_needs_ocr': False}]},
    {'pages_total': 1, 'pages': [{'page': True, 'page_needs_ocr': False}]},
    {'pages_total': 1, 'pages': [{'page': 1, 'page_needs_ocr': 0}]},
])
def test_invalid_quality_selection_rejected_before_empty_return(report):
    with pytest.raises(ValueError, match='page range'):
        effective_selection([], {}, report=report)
