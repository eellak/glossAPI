"""K2 batch 2a: the production font-repair boundary (B2 fontTools/pypdf row).

- The default production path (Corpus extract with its default repair setting) selects the Rust backend and
  imports neither fontTools nor pypdf, in the parent or in any worker process.
- Certified repair fixtures, native against the Python reference (pinned fontTools/pypdf oracle): decisions,
  repair reports, repaired text and repaired-page rasters for TrueType and CFF programs, AGL/legacy and
  lowercase names, duplicate/synthesized post names (aliases), subsets, encoded names and unknown glyphs.
- The missing-extension fallback and protected originals.
"""
from contextlib import closing
from dataclasses import asdict
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pypdfium2 as pdfium
import pytest

from reference_oracles import oracle

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / 'rust/glossapi_rs_pdfrepair/fixtures/review'

pytest.importorskip('glossapi_rs_pdfrepair')
oracle('fontTools')
oracle('pypdf')
from test_pdf_font_repair import greek_fixture, _cmap  # noqa: E402,F401  (fixture reuse)


GUARD = '''import importlib.abc, os, sys
with open(os.environ['K2_GUARD_PIDS'], 'a') as pids:
    pids.write(f'{os.getpid()}\\n')
class _K2ReferenceGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('fontTools', 'pypdf', 'PyPDF2', 'fitz') or fullname == 'glossapi.pdf_repair.tounicode':
            with open(os.environ['K2_IMPORT_LOG'], 'a') as log:
                log.write(f'{os.getpid()} {fullname}\\n')
sys.meta_path.insert(0, _K2ReferenceGuard())
'''


@pytest.mark.parametrize('control', [False, True], ids=['default', 'negative-control-python-backend'])
def test_default_extract_path_imports_neither_fonttools_nor_pypdf(greek_fixture, tmp_path, control):
    """Import trace of the default production path, workers included (sitecustomize is inherited by spawn)."""
    site = tmp_path / 'site'
    site.mkdir()
    (site / 'sitecustomize.py').write_text(GUARD)
    if control:  # the guard must see the reference when it is used: make the extension unimportable everywhere
        (site / 'glossapi_rs_pdfrepair.py').write_text("raise ImportError('negative control')\n")
    log, modules, pids = tmp_path / 'imports.log', tmp_path / 'modules.json', tmp_path / 'guarded.pids'
    log.write_text('')
    root = tmp_path / 'corpus'
    (root / 'downloads').mkdir(parents=True)
    (root / 'downloads' / 'paper.pdf').write_bytes(greek_fixture['broken'].read_bytes())
    code = f'''
import json, sys
from glossapi import Corpus
Corpus({str(root)!r}, {str(root)!r}).extract(phase1_backend="safe", accel_type="CPU")
import pandas as pd
row = pd.read_parquet({str(root / 'download_results/download_results.parquet')!r}).iloc[0]
json.dump(dict(status=row.font_repair_status, backend=row.font_repair_backend, fonts=int(row.font_repair_fonts),
               modules=sorted({{name.split('.')[0] for name in sys.modules}})), open({str(modules)!r}, 'w'))
'''
    env = {**os.environ, 'K2_IMPORT_LOG': str(log), 'K2_GUARD_PIDS': str(pids), 'PYTHONPATH': os.pathsep.join([str(site), *sys.path])}
    env.pop('GLOSSAPI_GREEK_WORDLIST', None)
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    report = json.loads(modules.read_text())
    if control:
        assert (report['status'], report['backend']) == ('repaired', 'python')
        assert 'fontTools' in log.read_text() and 'pypdf' in log.read_text()
        return
    assert (report['status'], report['backend']) == ('repaired', 'rust') and report['fonts'] >= 1
    assert log.read_text() == '', 'reference modules were imported on the default path:\n' + log.read_text()
    assert not {'fontTools', 'pypdf', 'PyPDF2', 'fitz'} & set(report['modules'])
    assert len(set(pids.read_text().split())) >= 1  # every interpreter of the run carried the guard
    text = (root / 'markdown' / 'paper.md').read_text()
    assert greek_fixture['text'] in text


def _render(path):
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        out = []
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            out.append((textpage.get_text_range(), page.render(scale=2).to_numpy().copy()))
            textpage.close(); page.close()
        return out


def _semantic(report):
    data = asdict(report) if not isinstance(report, dict) else dict(report)
    for key in ('path', 'src', 'dst', 'backend'):
        data.pop(key, None)
    return data


def _differential(source, tmp_path):
    from glossapi.pdf_repair import selector, tounicode
    python = selector.get_backend('python')
    rust = selector.get_backend('rust')
    assert (python.name, rust.name) == ('python', 'rust')
    reference_analysis, native_analysis = python.analyze_pdf(source), rust.analyze_pdf(source)
    assert _semantic(reference_analysis) == _semantic(native_analysis)
    a, b = tmp_path / (source.stem + '-python.pdf'), tmp_path / (source.stem + '-rust.pdf')
    reference_repair, native_repair = python.repair_pdf(source, a), rust.repair_pdf(source, b)
    assert _semantic(reference_repair) == _semantic(native_repair)
    if native_repair.written:
        left, right = _render(a), _render(b)
        assert [text for text, _ in left] == [text for text, _ in right]
        assert all(np.array_equal(x, y) for (_, x), (_, y) in zip(left, right))
    return native_analysis, native_repair


REVIEW_CASES = sorted(path.stem for path in REVIEW.glob('*.pdf'))


@pytest.mark.parametrize('name', REVIEW_CASES)
def test_certified_review_fixtures_native_matches_reference(name, tmp_path):
    """CP-R1 fixtures: lowercase AGL / uniXXXX names, duplicate and format-3 synthesized post names (aliases),
    format-3 collisions, encoded PDF names (plain, compressed, missing ToUnicode), raw-byte/UTF-8 name decoding."""
    assert len(REVIEW_CASES) == 10
    _differential(REVIEW / f'{name}.pdf', tmp_path)


@pytest.mark.parametrize('which', ['broken', 'original'])
def test_truetype_subset_fixture_native_matches_reference(greek_fixture, tmp_path, which):
    analysis, repair = _differential(greek_fixture[which], tmp_path)
    assert analysis.category == ('repairable' if which == 'broken' else 'consistent')
    assert repair.repaired_fonts == (1 if which == 'broken' else 0)


def _cff_pdf(path, *, unknown, composite):
    """An embedded OpenType-CFF program with a shifted ToUnicode, either under Type0/Identity-H with a
    CIDFontType0 descendant, or as a simple font whose /Differences name the glyphs (legacy/AGL names)."""
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject as A, DecodedStreamObject as S, DictionaryObject as D, NameObject as N, NumberObject as I
    order = ['.notdef', 'alpha', 'beta', 'gamma', 'uni03B4', 'glyph5' if unknown else 'epsilon']
    builder = FontBuilder(1000, isTTF=False)
    builder.setupGlyphOrder(order)
    builder.setupCharacterMap({0x3B1: 'alpha', 0x3B2: 'beta', 0x3B3: 'gamma', **({} if unknown else {0x3B5: 'epsilon'})})
    strings = {}
    for name in order:
        pen = T2CharStringPen(500, None)
        pen.moveTo((0, 0)); pen.lineTo((400, 0)); pen.lineTo((400, 600)); pen.closePath()
        strings[name] = pen.getCharString()
    builder.setupCFF('Synthetic', {'FullName': 'Synthetic', 'FamilyName': 'Synthetic', 'Weight': 'Regular'}, strings, {})
    builder.setupHorizontalMetrics({name: (500, 0) for name in order})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(dict(familyName='Synthetic', styleName='Regular'))
    builder.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    builder.setupPost(keepGlyphNames=True)
    program = BytesIO(); builder.save(program)
    writer = PdfWriter(); page = writer.add_blank_page(300, 200)
    def stream(data, extra=None):
        obj = S(); obj.set_data(data)
        for key, value in (extra or {}).items():
            obj[N(key)] = value
        return writer._add_object(obj)
    descriptor = D({N('/Type'): N('/FontDescriptor'), N('/FontName'): N('/Synthetic'), N('/Flags'): I(4 if composite else 32),
                    N('/FontBBox'): A([I(v) for v in (0, 0, 1000, 1000)]), N('/Ascent'): I(800), N('/Descent'): I(-200),
                    N('/CapHeight'): I(800), N('/StemV'): I(80), N('/ItalicAngle'): I(0),
                    N('/FontFile3'): stream(program.getvalue(), {'/Subtype': N('/OpenType')})})
    if composite:
        child = D({N('/Type'): N('/Font'), N('/Subtype'): N('/CIDFontType0'), N('/BaseFont'): N('/Synthetic'),
                   N('/FontDescriptor'): writer._add_object(descriptor),
                   N('/CIDSystemInfo'): D({N('/Registry'): N('/Adobe'), N('/Ordering'): N('/Identity'), N('/Supplement'): I(0)})})
        font = D({N('/Type'): N('/Font'), N('/Subtype'): N('/Type0'), N('/BaseFont'): N('/Synthetic'),
                  N('/DescendantFonts'): A([writer._add_object(child)]), N('/Encoding'): N('/Identity-H')})
        font[N('/ToUnicode')] = stream(_cmap({1: 'β', 2: 'γ', 3: 'δ', 4: 'ε', 5: 'ζ'}))
        content = b'BT /F1 20 Tf 20 100 Td <00010002000300040005> Tj ET'
    else:
        font = D({N('/Type'): N('/Font'), N('/Subtype'): N('/TrueType'), N('/BaseFont'): N('/Synthetic'),
                  N('/FontDescriptor'): writer._add_object(descriptor), N('/FirstChar'): I(65), N('/LastChar'): I(69),
                  N('/Widths'): A([I(500)] * 5),
                  N('/Encoding'): D({N('/Differences'): A([I(65), *[N('/' + name) for name in order[1:]]])})})
        font[N('/ToUnicode')] = stream(_cmap({65: 'β', 66: 'γ', 67: 'δ', 68: 'ε', 69: 'ζ'}, width=1))
        content = b'BT /F1 20 Tf 20 100 Td (ABCDE) Tj ET'
    page[N('/Resources')] = D({N('/Font'): D({N('/F1'): writer._add_object(font)})})
    page[N('/Contents')] = stream(content)
    writer.write(path)
    return path


@pytest.mark.parametrize('unknown', [False, True])
def test_cff_program_with_agl_names_and_unknown_glyph_native_matches_reference(tmp_path, unknown):
    analysis, repair = _differential(_cff_pdf(tmp_path / 'cff.pdf', unknown=unknown, composite=False), tmp_path)
    font = analysis.fonts[0]
    assert font.font_program_kind == 'FontFile3/OpenType'
    assert analysis.category in ('repairable', 'mixed') and font.repair_possible and repair.repaired_fonts == 1
    # uni03B4 is a legacy AGL name with no cmap entry; 'glyph5' has neither (unknown: its ToUnicode entry is kept).
    repaired_text = _render(tmp_path / 'cff-rust.pdf')[0][0]
    assert repaired_text == ('αβγδζ' if unknown else 'αβγδε')


def test_cff_cidfonttype0_without_cid_charset_is_unsupported_in_both(tmp_path):
    analysis, repair = _differential(_cff_pdf(tmp_path / 'cff0.pdf', unknown=False, composite=True), tmp_path)
    assert analysis.category == 'unsupported' and not analysis.fonts[0].repair_possible
    assert analysis.fonts[0].reason == 'CFF/OpenType lacks a usable CID mapping' and repair.repaired_fonts == 0


def test_missing_extension_falls_back_to_the_reference_with_provenance(greek_fixture, tmp_path):
    code = f'''
import importlib.abc, json, sys
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'glossapi_rs_pdfrepair':
            raise ImportError('injected: native extension missing')
sys.meta_path.insert(0, NoNative())
from glossapi.pdf_repair import get_backend, repair_pdf
backend = get_backend()
result = repair_pdf({str(greek_fixture['broken'])!r}, {str(tmp_path / 'fallback.pdf')!r})
print(json.dumps(dict(backend=backend.name, receipt=result.backend, repaired=result.repaired_fonts,
                      fonttools='fontTools' in sys.modules, pypdf='pypdf' in sys.modules)))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == dict(backend='python', receipt='python', repaired=1,
                                                              fonttools=True, pypdf=True)
    from glossapi.pdf_repair import selector
    with pytest.raises(ValueError):
        selector.get_backend('pdfium')


@pytest.mark.parametrize('backend', ['python', 'rust'])
def test_protected_original_is_never_overwritten(greek_fixture, tmp_path, backend):
    from glossapi.pdf_repair import selector
    from glossapi.pdf_repair.extract import repair_for_extract
    implementation = selector.get_backend(backend)
    source = tmp_path / 'source.pdf'
    source.write_bytes(greek_fixture['broken'].read_bytes())
    original = source.read_bytes()
    alias = tmp_path / 'alias.pdf'
    os.link(source, alias)
    for target in (source, alias):
        result = implementation.repair_pdf(source, target, protected_sources=(source,))
        assert not result.written and result.error and 'overwrite' in result.error
    assert source.read_bytes() == original
    # The extract pre-pass refuses a repaired_pdfs/ path that resolves into downloads/ and keeps the source.
    root = tmp_path / 'run'
    (root / 'downloads').mkdir(parents=True)
    downloaded = root / 'downloads' / 'doc.pdf'
    downloaded.write_bytes(original)
    (root / 'repaired_pdfs').symlink_to(root / 'downloads')
    import time
    selected, data = repair_for_extract(downloaded, root, enabled=True, deadline=time.monotonic() + 30, backend_name=backend)
    assert selected == downloaded and data['font_repair_status'] == 'error'
    assert downloaded.read_bytes() == original
