"""K5a: no duplicated work in safe extraction's font-repair pre-pass.

- The default (Rust) path decides text presence natively: the Rust -> Python -> PDFium ``_sample_json`` callback
  must never run on it.
- The font analysis runs once per document and is handed to ``repair_pdf`` (no second analysis).
- The stage preflight opens the source for its page count once; the font-repair budget reuses that count and the
  resulting timeouts are unchanged.
"""
import json
import shutil
import time
import uuid
from pathlib import Path

import pytest

native = pytest.importorskip('glossapi_rs_pdfrepair')

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / 'rust/glossapi_rs_pdfrepair/fixtures/review'
REPAIRABLE = REVIEW / 'encoded_pdf_name.pdf'      # category repairable, one repaired font
UNSUPPORTED = REVIEW / 'lowercase_agl.pdf'        # analysed, never repaired


def _require_analysis_api():
    # A stale extension built before K5a still has the callback path; this test is about the current build.
    assert hasattr(native, 'analyze_for_repair'), (
        'glossapi_rs_pdfrepair lacks analyze_for_repair: a stale build is first on the path')


@pytest.fixture
def forbid_callback(monkeypatch):
    """Any use of the PDFium callback on the default path fails the test (and is counted)."""
    from glossapi.pdf_repair import rust_backend
    calls = []

    def forbidden(path):
        calls.append(path)
        raise AssertionError(f'_sample_json callback reintroduced on the default path: {path}')

    monkeypatch.setattr(rust_backend, '_sample_json', forbidden)
    return calls


@pytest.fixture
def native_calls(monkeypatch):
    """Count the extension entry points the selector reaches (it looks them up on the module per call)."""
    calls = {'analyze_pdf': [], 'analyze_for_repair': [], 'repair_pdf': []}
    for name in calls:
        original = getattr(native, name)

        def counted(*args, _name=name, _original=original):
            calls[_name].append(args)
            return _original(*args)

        monkeypatch.setattr(native, name, counted)
    return calls


def _source(tmp_path, fixture):
    root = tmp_path / 'run'
    (root / 'downloads').mkdir(parents=True)
    source = root / 'downloads' / 'doc.pdf'
    shutil.copyfile(fixture, source)
    return root, source


def test_default_repair_path_analyses_once_without_the_callback(tmp_path, forbid_callback, native_calls):
    _require_analysis_api()
    from glossapi.pdf_repair.extract import repair_for_extract
    root, source = _source(tmp_path, REPAIRABLE)
    selected, data = repair_for_extract(source, root, enabled=True, deadline=time.monotonic() + 30)
    assert data['font_repair_status'] == 'repaired', data
    assert data['font_repair_backend'] == 'rust' and data['font_repair_fonts'] == 1
    assert selected == root / 'repaired_pdfs' / 'doc.pdf' and selected.is_file()
    assert forbid_callback == []
    # One analysis, and the repair received it (4th positional argument) instead of analysing again.
    assert len(native_calls['analyze_for_repair']) == 1 and native_calls['analyze_pdf'] == []
    assert len(native_calls['repair_pdf']) == 1
    args = native_calls['repair_pdf'][0]
    assert len(args) == 4 and isinstance(args[3], native.PdfAnalysis) and args[3].path == str(source)


def test_not_repaired_document_is_analysed_once_and_never_repaired(tmp_path, forbid_callback, native_calls):
    _require_analysis_api()
    from glossapi.pdf_repair.extract import repair_for_extract
    root, source = _source(tmp_path, UNSUPPORTED)
    selected, data = repair_for_extract(source, root, enabled=True, deadline=time.monotonic() + 30)
    assert selected == source and data['font_repair_status'] == 'unsupported'
    assert forbid_callback == []
    assert len(native_calls['analyze_for_repair']) == 1
    assert native_calls['analyze_pdf'] == [] and native_calls['repair_pdf'] == []


def test_public_analyze_and_repair_do_not_call_back(tmp_path, forbid_callback):
    """Standalone analyze_pdf / repair_pdf (CLI, typed API) use the native sample too."""
    from glossapi.pdf_repair import analyze_pdf, repair_pdf
    report = analyze_pdf(REPAIRABLE)
    assert report.backend == 'rust' and report.category == 'repairable' and report.pages == 1
    result = repair_pdf(REPAIRABLE, tmp_path / 'out.pdf')
    assert result.written and result.repaired_fonts == 1
    assert forbid_callback == []


def test_repair_from_analysis_equals_standalone_repair(tmp_path):
    _require_analysis_api()
    from glossapi.pdf_repair.selector import get_backend
    backend = get_backend('rust')
    report, analysis = backend.analyze_for_repair(REPAIRABLE)
    assert report == backend.analyze_pdf(REPAIRABLE)
    once = backend.repair_pdf(REPAIRABLE, tmp_path / 'once.pdf', analysis=analysis)
    alone = backend.repair_pdf(REPAIRABLE, tmp_path / 'alone.pdf')
    assert (tmp_path / 'once.pdf').read_bytes() == (tmp_path / 'alone.pdf').read_bytes()
    once.dst = alone.dst = None
    assert once == alone


def test_repair_from_analysis_refuses_a_changed_source(tmp_path):
    _require_analysis_api()
    from glossapi.pdf_repair.selector import get_backend
    backend = get_backend('rust')
    source = tmp_path / 'src.pdf'
    shutil.copyfile(REPAIRABLE, source)
    _, analysis = backend.analyze_for_repair(source)
    source.write_bytes(source.read_bytes() + b'%changed\n')
    result = backend.repair_pdf(source, tmp_path / 'out.pdf', analysis=analysis)
    assert not result.written and 'changed since its analysis' in result.error
    assert not (tmp_path / 'out.pdf').exists()


def test_stale_extension_without_the_analysis_api_keeps_two_passes(monkeypatch):
    from glossapi.pdf_repair import selector
    monkeypatch.delattr(native, 'analyze_for_repair', raising=False)
    assert selector.get_backend('rust').analyze_for_repair is None
    assert selector.get_backend('python').analyze_for_repair is None


def _stage_request(source, work):
    from glossapi.stages.contracts import ArtifactReference, StageRequest
    from glossapi.stages.execution import sha256_file
    ref = ArtifactReference(str(uuid.uuid4()), 'source_payload', source, sha256_file(source), source.stat().st_size)
    return StageRequest(*(str(uuid.uuid4()) for _ in range(5)), 'extract.document', (ref,),
                        {'phase1_backend': 'safe', 'repair_fonts': True}, {}, 'cpu', work,
                        time.monotonic_ns() + 600_000_000_000)


def test_safe_stage_opens_the_source_for_its_page_count_once(tmp_path, monkeypatch, forbid_callback):
    _require_analysis_api()
    from glossapi.gloss_extract import GlossExtract
    from glossapi.stages.execution import ReceiptCollector, producer_hash_scope, seal_output_hashes
    import glossapi.stages.extract as stage_module
    source = tmp_path / 'doc.pdf'
    shutil.copyfile(REPAIRABLE, source)
    work = tmp_path / 'work'
    (work / 'scratch').mkdir(parents=True)
    counted, budgets = [], []
    original_count = GlossExtract._get_pdf_page_count
    original_budget = GlossExtract._calculate_text_timeout

    def count(self, file_path):
        counted.append(Path(file_path).resolve())
        return original_count(self, file_path)

    def budget(self, **kwargs):
        value = original_budget(self, **kwargs)
        budgets.append((kwargs.get('page_count'), value))
        return value

    monkeypatch.setattr(GlossExtract, '_get_pdf_page_count', count)
    monkeypatch.setattr(GlossExtract, '_calculate_text_timeout', budget)
    request = _stage_request(source, work)
    with producer_hash_scope():
        result = seal_output_hashes(stage_module.ExtractStage()._run_document(request, reporter=ReceiptCollector()), work)
    assert result.status == 'ok', result
    # The stage stages its input copy; that file's page count is read exactly once (run_prepared's preflight).
    assert len(counted) == 1 and counted[0].name == 'doc.pdf', counted
    sidecar = json.loads((work / 'sidecars' / 'extract' / 'doc.json').read_text())
    assert sidecar['font_repair_status'] == 'repaired' and sidecar['page_count'] == 1
    # Budgets: the font-repair budget (from the reused count) equals the per-page rule; the cache is cleared.
    per_page = 20
    assert budgets and all(value == max(1 * per_page, per_page) for _, value in budgets)
    extractor = next(iter(stage_module._worker_extractors.values()))
    assert extractor._preflight_page_counts == {}
    assert forbid_callback == []


def test_text_timeout_is_identical_with_and_without_the_reused_count(tmp_path):
    from glossapi.gloss_extract import GlossExtract
    extractor = GlossExtract.__new__(GlossExtract)
    extractor.per_page_timeout_s = 20
    extractor.processing_timeout = 600
    extractor._log = __import__('logging').getLogger('k5a-test')
    pdf = REVIEW / 'encoded_pdf_name.pdf'
    missing = tmp_path / 'missing.pdf'
    extractor._preflight_page_counts = {}
    fresh = [extractor._calculate_text_timeout(file_path=p) for p in (pdf, missing)]
    fresh.append(extractor._calculate_text_timeout(file_path=missing, fallback=7))
    extractor._preflight_page_counts = {str(pdf.resolve()): extractor._get_pdf_page_count(pdf),
                                        str(missing.resolve()): None}
    reused = [extractor._calculate_text_timeout(file_path=p) for p in (pdf, missing)]
    reused.append(extractor._calculate_text_timeout(file_path=missing, fallback=7))
    assert fresh == reused == [20, 600, 7]
