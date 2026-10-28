"""Optional stacks must stay outside local text processing."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def isolated(code):
    return subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                          env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')})


def test_local_text_setup_never_attempts_optional_imports(tmp_path):
    result = isolated(f'''
import importlib.abc, sys
class ForbidOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'sklearn','joblib','dask','aiohttp','aiofiles','tenacity','yaml','PIL','playwright','docling','docling_core','torch','pypdf','fontTools'}}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, ForbidOptional())
from glossapi import Corpus
from glossapi.gloss_extract import GlossExtract
Corpus({str(tmp_path)!r}, {str(tmp_path / 'out')!r})
GlossExtract()
''')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(('module', 'missing', 'extra'), [
    ('glossapi.gloss_section_classifier', 'joblib', 'annotate'),
    ('glossapi.gloss_section_classifier', 'sklearn', 'annotate'),
    ('glossapi.gloss_section_classifier', 'dask', 'annotate'),
    ('glossapi.gloss_downloader', 'aiohttp', 'download'),
    ('glossapi.gloss_downloader', 'aiofiles', 'download'),
    ('glossapi.gloss_downloader', 'tenacity', 'download'),
    ('glossapi.gloss_browser_downloader', 'PIL', 'browser'),
    ('glossapi.ocr.docling.pipeline', 'docling', 'layout'),
    ('glossapi.ocr.math.enrich', 'docling_core', 'layout'),
    ('glossapi.pdf_repair.tounicode', 'fontTools', 'font-repair'),
    ('glossapi.pdf_repair.tounicode', 'pypdf', 'font-repair'),
])
def test_missing_optional_dependency_names_installable_package(module, missing, extra):
    result = isolated(f'''
import importlib, importlib.abc, sys
class MissingOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == {missing!r}:
            raise ModuleNotFoundError('injected absent package', name={missing!r})
sys.meta_path.insert(0, MissingOptional())
try:
    importlib.import_module({module!r})
except ImportError as exc:
    assert {missing!r} in str(exc), str(exc)
    assert {('glossapi[' + extra + ']')!r} in str(exc), str(exc)
else:
    raise AssertionError('optional package unexpectedly imported')
''')
    assert result.returncode == 0, result.stderr


def test_annotation_missing_extra_fails_before_input_checks(tmp_path):
    result = isolated(f'''
import importlib.abc, sys
class MissingAnnotate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'joblib','dask','sklearn'}}:
            raise ModuleNotFoundError('absent', name=fullname)
sys.meta_path.insert(0, MissingAnnotate())
from glossapi import Corpus
try:
    Corpus({str(tmp_path)!r}, {str(tmp_path / 'out')!r}).annotate()
except ImportError as exc:
    assert 'glossapi[annotate]' in str(exc), str(exc)
else:
    raise AssertionError('missing annotation extra was swallowed')
''')
    assert result.returncode == 0, result.stderr


def test_page_quality_uses_metrics_without_python_font_reference():
    result = isolated("""
import importlib.abc, sys
class MissingReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pypdf','fontTools','PIL'}:
            raise ModuleNotFoundError('absent', name=fullname)
sys.meta_path.insert(0, MissingReference())
from glossapi.page_quality import text_contamination_counts
from glossapi.pdf_repair import get_backend
assert get_backend().name == 'rust'
assert 'glossapi.pdf_repair.tounicode' not in sys.modules
""")
    assert result.returncode == 0, result.stderr


def test_math_missing_docling_after_core_types_names_layout_extra():
    result = isolated("""
import importlib.abc, sys
from pathlib import Path
from types import ModuleType
for name in ('docling_core','docling_core.types','docling_core.types.doc'):
    module=ModuleType(name);module.__path__=[];sys.modules[name]=module
sys.modules['docling_core.types.doc'].DocItemLabel=object
class MissingLayout(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0]=='docling':
            raise ModuleNotFoundError('absent', name=fullname)
sys.meta_path.insert(0, MissingLayout())
from glossapi.ocr.math.enrich import enrich_from_docling_json
try:
    enrich_from_docling_json(*[Path('unused')]*4)
except ImportError as exc:
    assert 'glossapi[layout]' in str(exc),str(exc)
else:
    raise AssertionError('missing layout dependency was swallowed')
""")
    assert result.returncode == 0, result.stderr


def test_layout_model_dependency_names_layout_extra(monkeypatch):
    import importlib
    from glossapi.scripts.docling_cost_benchmark import fake_docling
    with fake_docling('baseline', {}):
        pipeline=importlib.import_module('glossapi.ocr.docling.pipeline')
        backend=importlib.import_module('docling.pipeline.standard_pdf_pipeline')
        def missing_model(options):
            raise ModuleNotFoundError('absent model dependency',name='torch')
        monkeypatch.setattr(backend,'StandardPdfPipeline',missing_model)
        with pytest.raises(ImportError,match=r'glossapi\[layout\]'):
            pipeline.build_layout_pipeline(device='cpu')
