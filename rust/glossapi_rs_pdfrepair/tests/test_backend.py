"""Run the reference's generated PDF fixtures against the native entry points.

The module is loaded read-only. Imported private mapping helpers still generate
fixtures; all public analysis and repair calls in these tests use Rust.
"""

from contextlib import closing
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'rust/glossapi_rs_pdfrepair/target/python'), str(ROOT / 'src')]
spec = importlib.util.spec_from_file_location('pdfrepair_native_fixtures', ROOT / 'tests/test_pdf_font_repair.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
from glossapi.pdf_repair import rust_backend
from glossapi.pdf_repair.tounicode import FontReport, PdfFontReport, RepairResult


def semantic_fields(report, backend):
    # Backend provenance is expected to differ; all decision fields must match.
    assert report["backend"] == backend
    return {key: value for key, value in report.items() if key != "backend"}


def analyze(path):
    d = rust_backend.analyze_pdf(path)
    d['fonts'] = [FontReport(**f) for f in d['fonts']]
    return PdfFontReport(**d)


def repair(src, dst, **kwargs):
    d = rust_backend.repair_pdf(src, dst, **kwargs)
    d['fonts'] = [FontReport(**f) for f in d['fonts']]
    return RepairResult(**d)


reference.analyze_pdf = analyze
reference.repair_pdf = repair
greek_fixture = reference.greek_fixture
for name in (
    'test_shifted_greek_map_is_repaired',
    'test_no_text_layer_and_bad_pdf',
    'test_type3_is_unsupported',
    'test_simple_font_mapping',
    'test_real_pdf_repair',
    'test_explicit_cid_to_gid_mapping',
    'test_broken_embedded_program_is_reported',
    'test_no_encoding_guesses_for_simple_font',
    'test_post_names_fallback',
    'test_explicit_encoding_codespace',
    'test_repair_protects_other_selected_source',
    'test_no_candidate_retains_original_tounicode',
    'test_direct_font_is_not_repair_eligible',
    'test_repairable_font_with_not_embedded_font',
):
    globals()[name] = getattr(reference, name)


def test_malformed_older_xref_uses_updated_font_dictionaries(tmp_path):
    """BPZ has an older in-use entry at offset zero: do not trigger recovery."""
    import pytest
    from pypdf import PdfReader
    from glossapi.pdf_repair.tounicode import analyze_pdf as python_analyze
    source = Path.home() / 'Projects/glossapi-ocr-local/experiments/compute_reduction_20261006/sample_pdfs/BPZ_307.pdf'
    if not source.exists():
        pytest.skip('Read-only corpus fixture is not present')
    output = tmp_path / 'bpz.pdf'
    r = rust_backend.repair_pdf(source, output)
    assert r['written'] and r['repaired_fonts'] == 4
    assert semantic_fields(python_analyze(output).to_dict(), "python") == semantic_fields(rust_backend.analyze_pdf(output), "rust")
    reader = PdfReader(output)
    assert reader.xref_index == 0
    for font in r['fonts']:
        if font['repair_possible']:
            ref = reader.get_object(font['object_id']).raw_get('/ToUnicode')
            assert ref.idnum >= PdfReader(source).trailer['/Size']


def test_review_high_reproductions_match_python(tmp_path):
    """CP-R1 fixtures: decisions, repaired text and raw font dictionary bytes."""
    from glossapi.pdf_repair.tounicode import analyze_pdf as py_analyze, repair_pdf as py_repair
    import pdf_invariants as equivalence
    for name in ('lowercase_agl', 'lowercase_u_agl', 'duplicate_post_names',
                 'format3_duplicate_synth_names', 'format3_collision_wrong_gid', 'encoded_pdf_name',
                 'encoded_pdf_name_compressed', 'encoded_pdf_name_missing_tounicode'):
        source = ROOT / 'rust/glossapi_rs_pdfrepair/fixtures/review' / (name + '.pdf')
        assert semantic_fields(py_analyze(source).to_dict(), "python") == semantic_fields(rust_backend.analyze_pdf(source), "rust"), name
        a, b = tmp_path / (name + '-python.pdf'), tmp_path / (name + '-rust.pdf')
        py, rs = py_repair(source, a).to_dict(), rust_backend.repair_pdf(source, b)
        assert {**semantic_fields(py, 'python'), 'dst': ''} == {**semantic_fields(rs, 'rust'), 'dst': ''}, name
        assert rs['written'] and not rs['error'], name
        assert equivalence.extracted(a) == equivalence.extracted(b), name
        if name.startswith('lowercase'):
            assert rs['repaired_fonts'] == 0 and b.read_bytes() == source.read_bytes(), name
            assert equivalence.extracted(b) == ['B'], name
        else:
            ids = {f['object_id'] for f in rs['fonts'] if f['repair_possible']}
            assert equivalence.object_invariants(source, b, ids) is None, name


def test_cycle3_name_decoding_matches_reference(tmp_path):
    """CP-R1c2 raw-byte collision and valid UTF-8 names retain B, without errors."""
    from glossapi.pdf_repair.tounicode import analyze_pdf as py_analyze, repair_pdf as py_repair
    from glossapi.pdf_repair.tounicode import _page_text
    import pypdfium2 as pdfium
    for name in ('name_decoding_different_raw_byte', 'name_decoding_valid_utf8'):
        source = ROOT / 'rust/glossapi_rs_pdfrepair/fixtures/review' / (name + '.pdf')
        assert semantic_fields(py_analyze(source).to_dict(), "python") == semantic_fields(rust_backend.analyze_pdf(source), "rust"), name
        a, b = tmp_path / (name + '-python.pdf'), tmp_path / (name + '-rust.pdf')
        py, rs = py_repair(source, a).to_dict(), rust_backend.repair_pdf(source, b)
        assert {**semantic_fields(py, 'python'), 'dst': ''} == {**semantic_fields(rs, 'rust'), 'dst': ''}, name
        assert rs['written'] and rs['error'] is None, name
        assert rs['repaired_fonts'] == 0, name
        assert a.read_bytes() == b.read_bytes() == source.read_bytes(), name
        with closing(pdfium.PdfDocument(str(b))) as pdf:
            assert _page_text(pdf, 0) == 'B', name
