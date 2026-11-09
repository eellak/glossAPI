"""OCRDEF-C2-G03: the plugin cache override is scoped to one OCR call.

The conversion must see glossAPI's cache root, and the process environment must
be exactly as before afterwards, on success and on failure. A subprocess started
after the OCR call (as the standalone smoke does) must inherit the caller's value.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from glossapi.corpus.ocr import dispatch
from test_ocr_plugin_dispatch import make_corpus

NAME = 'GREEKOCR_CACHE_DIR'


def whole_document(parquet):
    frame = pd.read_parquet(parquet)
    frame['ocr_pages'] = ''
    frame.to_parquet(parquet)


def recording_converter(seen, status='success'):
    def convert(source, *, page_range):
        seen.append(os.environ.get(NAME))
        return SimpleNamespace(status=status, document=SimpleNamespace(
            export_to_markdown=lambda *, page_no: f'page {page_no}'))
    return lambda request: SimpleNamespace(convert=convert)


@pytest.mark.parametrize('previous', ['warm', None])
def test_ocr_call_restores_cache_environment(tmp_path, monkeypatch, previous):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    whole_document(parquet)
    cold = tmp_path / 'glossapi-cache'
    monkeypatch.setenv('GLOSSAPI_CACHE_DIR', str(cold))
    warm = str(tmp_path / 'warm-plugin-cache')
    if previous:
        monkeypatch.setenv(NAME, warm)
    else:
        monkeypatch.delenv(NAME, raising=False)
    before = dict(os.environ)
    seen = []
    monkeypatch.setattr(dispatch, '_build_converter', recording_converter(seen))
    corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    # The conversion saw glossAPI's cache root; the caller's environment is untouched.
    assert seen == [str(cold.resolve())]
    assert dict(os.environ) == before
    assert os.environ.get(NAME) == (warm if previous else None)


def test_failed_ocr_call_restores_cache_environment(tmp_path, monkeypatch):
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    whole_document(parquet)
    monkeypatch.setenv('GLOSSAPI_CACHE_DIR', str(tmp_path / 'glossapi-cache'))
    monkeypatch.setenv(NAME, str(tmp_path / 'warm-plugin-cache'))
    before = dict(os.environ)
    seen = []
    monkeypatch.setattr(dispatch, '_build_converter', recording_converter(seen, status='failure'))
    with pytest.raises(RuntimeError, match='OCR plugin failed'):
        corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    assert seen == [str((tmp_path / 'glossapi-cache').resolve())]
    assert dict(os.environ) == before


def test_later_subprocess_inherits_caller_cache_not_ocr_cache(tmp_path, monkeypatch):
    """The ordered-suite failure: a later smoke subprocess must keep the warm cache."""
    corpus, parquet = make_corpus(tmp_path, monkeypatch)
    whole_document(parquet)
    monkeypatch.setenv('GLOSSAPI_CACHE_DIR', str(tmp_path / 'cold-glossapi-cache'))
    warm = str(tmp_path / 'warm-plugin-cache')
    monkeypatch.setenv(NAME, warm)
    monkeypatch.setattr(dispatch, '_build_converter', recording_converter([]))
    corpus.ocr(mode='ocr_bad', ocr_options=SimpleNamespace(kind='test'))
    child = subprocess.run([sys.executable, '-c', f'import os; print(os.environ.get({NAME!r}))'],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == warm
