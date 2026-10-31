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
    ('glossapi.gloss_downloader', 'aiohttp', 'download'),
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


@pytest.mark.parametrize(('module', 'replaced', 'needs'), [
    ('glossapi.gloss_section_classifier', 'dask', ()),
    ('glossapi.gloss_section_classifier', 'sklearn', ()),
    ('glossapi.gloss_section_classifier', 'joblib', ()),
    ('glossapi.section_model', 'sklearn', ()),
    ('glossapi.gloss_downloader', 'aiofiles', ('aiohttp',)),
    ('glossapi.gloss_downloader', 'tenacity', ('aiohttp',)),
    ('glossapi.gloss_browser_downloader', 'aiofiles', ('aiohttp', 'PIL')),
])
def test_replaced_packages_are_not_imported(module, replaced, needs):
    """K2: dask, aiofiles, tenacity and (W2b) scikit-learn/joblib inference were replaced by own code; their
    absence must not matter."""
    for name in needs:
        pytest.importorskip(name)
    result = isolated(f'''
import importlib, importlib.abc, sys
class Absent(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == {replaced!r}:
            raise ModuleNotFoundError('injected absent package', name={replaced!r})
sys.meta_path.insert(0, Absent())
importlib.import_module({module!r})
assert {replaced!r} not in sys.modules
''')
    assert result.returncode == 0, result.stderr


BLOCK_LEGACY = '''
import importlib.abc, sys
class MissingLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'joblib','dask','sklearn','scipy','threadpoolctl'}:
            raise ModuleNotFoundError('absent', name=fullname)
sys.meta_path.insert(0, MissingLegacy())
'''


def _sections(path):
    import pandas as pd
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(filename='doc', id=i, header=h, section=s, section_length=len(s), section_propo=len(s),
                       has_table=t, has_list=False) for i, (h, s, t) in enumerate([
        ('Εισαγωγή', 'Μικρό κείμενο για τον έλεγχο της ταξινόμησης.', False),
        ('Περιεχόμενα', '\n'.join(f'| Κεφάλαιο {i} | {i * 7} |' for i in range(1, 12)), True),
        ('Βιβλιογραφία', 'Παπαδόπουλος, Α. (2001). Ένα βιβλίο. Αθήνα.', False)], start=1)]).to_parquet(path, index=False)


def test_default_annotation_runs_with_sklearn_joblib_and_scipy_blocked(tmp_path):
    """K2 W2b: the packaged neutral model annotates with the legacy stack unimportable, and never loads it."""
    from glossapi import Corpus
    probe = Corpus(tmp_path, tmp_path / 'out')
    _sections(probe.sections_parquet)
    result = isolated(BLOCK_LEGACY + f'''
from glossapi import Corpus
import pandas as pd
corpus = Corpus({str(tmp_path)!r}, {str(tmp_path / 'out')!r})
corpus.annotate()
labels = pd.read_parquet(corpus.classified_parquet).sort_values('id').predicted_section.tolist()
assert labels == ['άλλο', 'π', 'β'], labels
assert corpus.fully_annotated_parquet.is_file()
loaded = sorted(n for n in sys.modules if n.split('.')[0] in ('sklearn', 'joblib', 'scipy', 'dask', 'threadpoolctl'))
assert not loaded, loaded
''')
    assert result.returncode == 0, result.stderr


def test_legacy_joblib_model_without_extra_names_annotate_legacy(tmp_path):
    """An explicit *.joblib model needs scikit-learn 1.6.1; without it the error names glossapi[annotate-legacy]."""
    from glossapi import Corpus
    import glossapi
    model = Path(glossapi.__file__).resolve().parent / 'models/section_classifier.joblib'
    probe = Corpus(tmp_path, tmp_path / 'out', section_classifier_model_path=model)
    _sections(probe.sections_parquet)
    result = isolated(BLOCK_LEGACY + f'''
from glossapi import Corpus
from glossapi.gloss_section_classifier import GlossSectionClassifier
try:
    GlossSectionClassifier().load_model({str(model)!r})
except ImportError as exc:
    assert 'glossapi[annotate-legacy]' in str(exc), str(exc)
else:
    raise AssertionError('legacy model loaded without scikit-learn')
try:
    GlossSectionClassifier().build_pipeline()
except ImportError as exc:
    assert 'glossapi[annotate-legacy]' in str(exc), str(exc)
else:
    raise AssertionError('training pipeline built without scikit-learn')
try:
    Corpus({str(tmp_path)!r}, {str(tmp_path / 'out')!r}, section_classifier_model_path={str(model)!r}).annotate()
except (ImportError, ValueError) as exc:
    assert 'glossapi[annotate-legacy]' in str(exc), str(exc)
else:
    raise AssertionError('missing annotate-legacy extra was swallowed')
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
