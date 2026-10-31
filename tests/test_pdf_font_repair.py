"""Generated-font and real-PDF checks for conservative ToUnicode repair."""

from __future__ import annotations

from contextlib import closing

import hashlib
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from fontTools import subset
from fontTools.ttLib import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject, DecodedStreamObject, DictionaryObject, FloatObject,
    NameObject, NumberObject,
)
import pypdfium2 as pdfium
import pytest

from glossapi.pdf_repair import analyze_pdf, extract_text_stats, repair_pdf
from glossapi.pdf_repair.tounicode import _cmap_entries, _glyph_unicode, _inspect_font, _make_cmap
from glossapi.scripts.pdf_font_repair import _batch, main, summarize
from pdf_fixtures import FixturePdf


def _cmap(mapping: dict[int, str], width: int = 2) -> bytes:
    """Create fixture streams independently of the repair CMap writer."""
    entries = '\n'.join(f'<{code:0{width * 2}x}> <{text.encode("utf-16-be").hex()}>' for code, text in mapping.items())
    return (f'/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n'
            f'/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n'
            f'/CMapName /Fixture def\n/CMapType 2 def\n1 begincodespacerange\n'
            f'<{"00" * width}> <{"ff" * width}>\nendcodespacerange\n'
            f'{len(mapping)} beginbfchar\n{entries}\nendbfchar\n'
            'endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n').encode()


def _text(path: Path, page_index: int = 0) -> str:
    """Extract with explicit native-handle cleanup."""
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        page = pdf[page_index]
        try:
            text = page.get_textpage()
            try:
                return text.get_text_range()
            finally:
                text.close()
        finally:
            page.close()


# TrueType fonts for the Greek fixture, in preference order: macOS first (the fixture the Mac runs have always
# used), then fonts present on Linux nodes and CI images. GLOSSAPI_TEST_GREEK_FONT names one explicitly.
GREEK_FIXTURE_FONTS = (
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
    '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Debian/Ubuntu fonts-dejavu-core
    '/usr/share/fonts/dejavu/DejaVuSans.ttf',  # Fedora/RHEL dejavu-sans-fonts
    '/usr/share/fonts/TTF/DejaVuSans.ttf',  # Arch ttf-dejavu
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
)
GREEK_FIXTURE_TEXT = 'αβγδε θή'


def fixture_fonts_required() -> bool:
    """CI mode (CI or GLOSSAPI_REFERENCE_REQUIRED set): a missing covering font fails instead of skipping."""
    return any(os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes'} for name in ('CI', 'GLOSSAPI_REFERENCE_REQUIRED'))


def greek_fixture_font(text: str = GREEK_FIXTURE_TEXT):
    """The first candidate TrueType font whose cmap covers every character of ``text`` -> (path, TTFont)."""
    explicit = os.environ.get('GLOSSAPI_TEST_GREEK_FONT')
    tried = []
    for path in ([explicit] if explicit else []) + list(GREEK_FIXTURE_FONTS):
        if not Path(path).is_file():
            continue
        candidate = TTFont(path)
        cmap = candidate.getBestCmap() or {}
        missing = sorted({char for char in text if ord(char) not in cmap})
        if not missing and 'glyf' in candidate:
            return path, candidate
        tried.append(f'{path} (missing {"".join(missing)!r})' if missing else f'{path} (not TrueType outlines)')
        candidate.close()
        if path == explicit:
            raise AssertionError(f'GLOSSAPI_TEST_GREEK_FONT={explicit} does not cover {text!r}: {tried[-1]}')
    message = (f'No TrueType font covering {text!r} for the Greek fixture; tried {tried or "none present"} of '
               f'{list(GREEK_FIXTURE_FONTS)}. Install fonts-dejavu-core or set GLOSSAPI_TEST_GREEK_FONT.')
    if fixture_fonts_required():
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


@pytest.fixture(scope="module")
def greek_fixture(tmp_path_factory):
    text = GREEK_FIXTURE_TEXT
    _, font = greek_fixture_font(text)
    directory = tmp_path_factory.mktemp('greek-font')
    options = subset.Options()
    options.glyph_names = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    fontpath = directory / 'subset.ttf'
    # Arial Unicode ships post format 3: retain the derived names explicitly
    # so symbol-only and post-only fixtures have an independent Unicode oracle.
    font['post'].formatType = 2.0
    font['post'].extraNames = []
    font['post'].mapping = {}
    font.save(fontpath)
    expected = {font.getGlyphID(name): chr(code) for code, name in font.getBestCmap().items()}
    widths = [int(font['hmtx'][name][0] * 1000 / font['head'].unitsPerEm) for name in font.getGlyphOrder()]
    font.close()
    original = directory / 'original.pdf'
    document = FixturePdf()
    document.insert_text(document.new_page(), (72, 72), text, fontsize=20, font=document.load_truetype(fontpath))
    document.save(directory / 'pdfium.pdf')
    # PDFium shifts /W past the subset's unmapped 'tonos' glyph; restore the hmtx advances (as PyMuPDF wrote
    # them), so glyph positions and renders equal the former fixture's. Text and ToUnicode are unaffected.
    writer = PdfWriter(clone_from=directory / 'pdfium.pdf')
    descendant = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()['/DescendantFonts'][0].get_object()
    descendant[NameObject('/W')] = ArrayObject([NumberObject(0), ArrayObject([NumberObject(w) for w in widths])])
    writer.write(original)
    assert _text(original).strip() == text
    writer = PdfWriter(clone_from=original)
    fontobj = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    corrupt = {code: chr(ord(value) + 1) if '\u0370' <= value <= '\u03ff' else value for code, value in expected.items()}
    stream = DecodedStreamObject()
    stream.set_data(_cmap(corrupt))
    fontobj[NameObject('/ToUnicode')] = writer._add_object(stream)
    broken = directory / 'broken.pdf'
    writer.write(broken)
    return {'original': original, 'broken': broken, 'text': text,
            'expected': expected, 'fontpath': fontpath,
            'affected': sum(value != corrupt[code] for code, value in expected.items())}


def _assert_only_tounicode_changed(src: Path, dst: Path) -> None:
    """Compare every original indirect object, including encoded stream bytes."""
    assert dst.read_bytes().startswith(src.read_bytes())
    before, after = PdfReader(src), PdfReader(dst)
    changed_fonts = 0
    for generation, entries in before.xref.items():
        for object_id in entries:
            if object_id == 0:
                continue
            old = before.get_object(object_id)
            new = after.get_object(object_id)
            if isinstance(old, DictionaryObject) and old.get('/Type') == '/Font':
                old_items = DictionaryObject({key: value for key, value in old.items() if key != '/ToUnicode'})
                new_items = DictionaryObject({key: value for key, value in new.items() if key != '/ToUnicode'})
                old_buffer, new_buffer = BytesIO(), BytesIO()
                old_items.write_to_stream(old_buffer)
                new_items.write_to_stream(new_buffer)
                assert old_buffer.getvalue() == new_buffer.getvalue()
                if '/ToUnicode' in old and '/ToUnicode' in new:
                    old_ref, new_ref = old.raw_get('/ToUnicode'), new.raw_get('/ToUnicode')
                    changed_fonts += (old_ref.idnum, old_ref.generation) != (new_ref.idnum, new_ref.generation)
            else:
                # Serialize without decoding: compressed images/content/fonts must match.
                old_buffer, new_buffer = BytesIO(), BytesIO()
                old.write_to_stream(old_buffer)
                new.write_to_stream(new_buffer)
                assert old_buffer.getvalue() == new_buffer.getvalue(), object_id
    assert changed_fonts > 0


def test_shifted_greek_map_is_repaired(greek_fixture, tmp_path):
    source = greek_fixture['broken']
    original_bytes = source.read_bytes()
    report = analyze_pdf(source)
    assert report.error is None
    assert report.category == 'repairable'
    assert len(report.fonts) == 1
    font = report.fonts[0]
    assert font.category == 'repairable'
    assert font.embedded and font.font_program_kind == 'FontFile2'
    assert font.disagreeing_codes == greek_fixture['affected']
    assert font.affected_greek_codes == greek_fixture['affected']
    destination = tmp_path / 'repaired.pdf'
    result = repair_pdf(source, destination)
    assert result.error is None
    assert result.repaired_fonts == 1
    assert result.written
    assert _text(destination).strip() == greek_fixture['text']
    assert analyze_pdf(destination).category == 'consistent'
    assert source.read_bytes() == original_bytes
    _assert_only_tounicode_changed(source, destination)


def test_no_text_layer_and_bad_pdf(tmp_path):
    blank = tmp_path / 'blank.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(blank)
    assert analyze_pdf(blank).category == 'no_text_layer'
    bad = tmp_path / 'bad.pdf'
    bad.write_bytes(b'not a PDF')
    assert analyze_pdf(bad).error
    assert not repair_pdf(bad, tmp_path / 'output.pdf').written
    assert repair_pdf(blank, blank).error
    link = tmp_path / 'hardlink.pdf'
    link.hardlink_to(blank)
    assert repair_pdf(blank, link).error


def test_type3_is_unsupported(tmp_path):
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    glyph = DecodedStreamObject()
    glyph.set_data(b'600 0 0 0 600 700 d1 0 0 400 600 re f')
    font = DictionaryObject({
        NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type3'),
        NameObject('/Name'): NameObject('/FixtureType3'),
        NameObject('/FontBBox'): ArrayObject([NumberObject(x) for x in (0, 0, 600, 700)]),
        NameObject('/FontMatrix'): ArrayObject([FloatObject(x) for x in (.001, 0, 0, .001, 0, 0)]),
        NameObject('/CharProcs'): DictionaryObject({NameObject('/A'): writer._add_object(glyph)}),
        NameObject('/Encoding'): DictionaryObject({NameObject('/Differences'): ArrayObject([NumberObject(65), NameObject('/A')])}),
        NameObject('/FirstChar'): NumberObject(65), NameObject('/LastChar'): NumberObject(65),
        NameObject('/Widths'): ArrayObject([NumberObject(600)]),
        NameObject('/Resources'): DictionaryObject(),
    })
    stream = DecodedStreamObject()
    stream.set_data(_cmap({65: 'A'}, 1))
    font[NameObject('/ToUnicode')] = writer._add_object(stream)
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(b'BT /F1 20 Tf 20 100 Td (A) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(content)
    source = tmp_path / 'type3.pdf'
    writer.write(source)
    assert _text(source).strip() == 'A'
    report = analyze_pdf(source)
    assert report.error is None
    assert report.category == 'unsupported'
    assert report.fonts[0].category == 'unsupported'


@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('encoding_kind', ['differences', 'symbol'])
def test_simple_font_mapping(greek_fixture, tmp_path, missing, encoding_kind):
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    program = DecodedStreamObject()
    with TTFont(greek_fixture['fontpath']) as tt:
        name = tt.getBestCmap()[ord('α')]
        if encoding_kind == 'symbol':
            from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
            table = CmapSubtable.newSubtable(4)
            table.platformID, table.platEncID, table.language = 3, 0, 0
            table.cmap = {0xF041: name}
            tt['cmap'].tables = [table]
        buffer = BytesIO()
        tt.save(buffer)
        program.set_data(buffer.getvalue())
    descriptor = DictionaryObject({NameObject('/Type'): NameObject('/FontDescriptor'),
        NameObject('/FontName'): NameObject('/GreekSimple'), NameObject('/Flags'): NumberObject(32),
        NameObject('/FontBBox'): ArrayObject([NumberObject(x) for x in (0, 0, 1000, 1000)]),
        NameObject('/ItalicAngle'): NumberObject(0), NameObject('/Ascent'): NumberObject(1000),
        NameObject('/Descent'): NumberObject(-200), NameObject('/CapHeight'): NumberObject(700),
        NameObject('/StemV'): NumberObject(80), NameObject('/FontFile2'): writer._add_object(program)})
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/TrueType'), NameObject('/BaseFont'): NameObject('/GreekSimple'),
        NameObject('/FontDescriptor'): writer._add_object(descriptor),
        NameObject('/Encoding'): DictionaryObject({NameObject('/Differences'): ArrayObject([NumberObject(65), NameObject('/' + name)])}),
        NameObject('/FirstChar'): NumberObject(65), NameObject('/LastChar'): NumberObject(65),
        NameObject('/Widths'): ArrayObject([NumberObject(600)])})
    if encoding_kind == 'symbol':
        del font['/Encoding']
        descriptor[NameObject('/Flags')] = NumberObject(4)
    if not missing:
        stream = DecodedStreamObject()
        stream.set_data(_cmap({65: 'β'}, 1))
        font[NameObject('/ToUnicode')] = writer._add_object(stream)
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    content = DecodedStreamObject()
    content.set_data(b'BT /F1 20 Tf 20 100 Td (A) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(content)
    src = tmp_path / 'simple.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    assert report.fonts[0].repair_possible
    assert report.fonts[0].category == ('no_tounicode' if missing else 'repairable')
    dst = tmp_path / 'simple-fixed.pdf'
    result = repair_pdf(src, dst)
    assert result.error is None
    assert _text(dst).strip() == 'α'


def test_cmap_ranges_surrogates_and_blocks():
    assert _cmap_entries(b'1 beginbfrange <0001> <0003> [<03B1> <03B2> <D83DDE00>] endbfrange') == {1: 'α', 2: 'β', 3: '😀'}
    assert _cmap_entries(b'1 beginbfrange <0001> <0003> <03B1> endbfrange') == {1: 'α', 2: 'β', 3: 'γ'}
    mapping = {index: '😀' if index == 0 else 'α' for index in range(205)}
    stream = _make_cmap(mapping)
    assert stream.count(b'100 beginbfchar') == 2
    assert b'5 beginbfchar' in stream
    assert _cmap_entries(stream) == mapping


def test_dictionary_stats(greek_fixture, tmp_path, monkeypatch):
    dictionary = tmp_path / 'words.txt'
    dictionary.write_text('ΑΒΓΔΕ\nθή\n', encoding='utf-8')
    monkeypatch.setenv('GLOSSAPI_GREEK_WORDLIST', str(dictionary))
    stats = extract_text_stats(greek_fixture['original'], 1)
    assert stats['totals']['dictionary_valid_rate'] == 1
    assert stats['totals']['greek_chars'] == 7
    assert stats['totals']['words'] == 2
    assert len(stats['pages']) == 1
    assert extract_text_stats(greek_fixture['original'], 0)['totals']['chars'] == 0
    assert extract_text_stats(greek_fixture['original'], word_set={'αβγδε'})['totals']['dictionary_valid_rate'] == .5
    monkeypatch.delenv('GLOSSAPI_GREEK_WORDLIST')
    assert extract_text_stats(greek_fixture['original'])['totals']['dictionary_valid_rate'] is None


def test_cli_repair_and_summary(greek_fixture, tmp_path):
    report = tmp_path / 'report.jsonl'
    out = tmp_path / 'fixed'
    wordlist = tmp_path / 'words.txt'
    wordlist.write_text('αβγδε\nθή\n', encoding='utf-8')
    command = [sys.executable, '-m', 'glossapi.scripts.pdf_font_repair', 'repair',
               '--pdfs', str(greek_fixture['broken']), '--out-dir', str(out),
               '--report', str(report), '--workers', '2', '--wordlist', str(wordlist)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    record = json.loads(report.read_text())
    assert record['before_category'] == 'repairable'
    assert record['after_category'] == 'consistent'
    assert record['after']['totals']['dictionary_valid_rate'] == 1
    assert _text(out / 'broken.pdf').strip() == greek_fixture['text']
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('filename,pages\nbroken.pdf,1\n', encoding='utf-8')
    completed = subprocess.run([sys.executable, '-m', 'glossapi.scripts.pdf_font_repair', 'summarize', '--report', str(report), '--manifest', str(manifest)], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result['projected_ocr_pages_removed'] == 1
    assert result['docs_after_rate_ge_085'] == 1
    assert result['categories']['repairable'] == {'docs': 1, 'pages': 1}


def test_batch_timeout_does_not_stall(greek_fixture):
    started = time.monotonic()
    records = list(_batch([greek_fixture['broken']] * 2, 'classify', None, 2, .001, None))
    assert len(records) == 2
    assert all('Timeout' in record['error'] for record in records)
    assert time.monotonic() - started < 10


# CP0 F3 coordinator amendment: the brief's ABO < .3 was an unverified guess.
# These are measured first-20-page dictionary baselines under this token metric.
ACH_787_MEASURED_BASELINE = 0.06096681096681097
ABO_768_MEASURED_BASELINE = 0.5212494687632809


@pytest.mark.parametrize('filename,needle,baseline_bound,measured_baseline', [
    ('ACH_787.pdf', 'ευχαριστήσω', .10, ACH_787_MEASURED_BASELINE),
    ('ABO_768.pdf', 'κατοίκησης', .60, ABO_768_MEASURED_BASELINE),
])
def test_real_pdf_repair(tmp_path, filename, needle, baseline_bound, measured_baseline):
    root = Path.home() / 'Projects/glossapi-ocr-local'
    src = root / 'reports/artifacts-20260407-ocr-box/pdfs' / filename
    dictionary = root / 'experiments/compute_reduction_20261006/data/dict/el_GR_words_utf8.txt'
    if not src.exists():
        pytest.skip(f'Read-only integration PDF unavailable: {src}')
    if not dictionary.exists():
        pytest.skip(f'Integration Greek word list unavailable: {dictionary}')
    original_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    assert analyze_pdf(src).category == 'repairable'
    dst = tmp_path / filename
    result = repair_pdf(src, dst)
    assert result.error is None
    assert result.repaired_fonts > 0
    assert needle in _text(dst, 5)
    before = extract_text_stats(src, 20, wordlist_path=dictionary)['totals']['dictionary_valid_rate']
    after = extract_text_stats(dst, 20, wordlist_path=dictionary)['totals']['dictionary_valid_rate']
    assert before == pytest.approx(measured_baseline, abs=1e-12)
    assert before < baseline_bound
    assert after > .95
    assert after - before > .4
    assert hashlib.sha256(src.read_bytes()).hexdigest() == original_hash
    _assert_only_tounicode_changed(src, dst)


def test_explicit_cid_to_gid_mapping(greek_fixture, tmp_path):
    """CIDs are not assumed to be GIDs when CIDToGIDMap is a stream."""
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    descendant = font['/DescendantFonts'][0].get_object()
    gid_alpha = next(code for code, value in greek_fixture['expected'].items() if value == 'α')
    gid_beta = next(code for code, value in greek_fixture['expected'].items() if value == 'β')
    mapping = list(range(max(greek_fixture['expected']) + 1))
    mapping[gid_alpha], mapping[gid_beta] = gid_beta, gid_alpha
    cidmap = DecodedStreamObject()
    cidmap.set_data(b''.join(gid.to_bytes(2, 'big') for gid in mapping))
    descendant[NameObject('/CIDToGIDMap')] = writer._add_object(cidmap)
    src, dst = tmp_path / 'cidmap.pdf', tmp_path / 'cidmap-fixed.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    assert report.category == 'repairable'
    result = repair_pdf(src, dst)
    assert result.error is None
    assert _text(dst).strip() == 'βαγδε θή'
    _assert_only_tounicode_changed(src, dst)


def test_broken_embedded_program_is_reported(greek_fixture, tmp_path):
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    program = DecodedStreamObject()
    program.set_data(b'broken TrueType data')
    descriptor = font['/DescendantFonts'][0]['/FontDescriptor']
    descriptor[NameObject('/FontFile2')] = writer._add_object(program)
    src, dst = tmp_path / 'bad-font.pdf', tmp_path / 'bad-font-copy.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    assert report.fonts[0].embedded
    assert report.fonts[0].category == 'unsupported'
    assert report.fonts[0].error
    result = repair_pdf(src, dst)
    assert result.error is None
    assert result.repaired_fonts == 0
    assert dst.read_bytes() == src.read_bytes()


def test_no_encoding_guesses_for_simple_font(greek_fixture, tmp_path):
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    descriptor = font['/DescendantFonts'][0]['/FontDescriptor']
    font[NameObject('/Subtype')] = NameObject('/TrueType')
    font[NameObject('/FontDescriptor')] = descriptor.indirect_reference
    font[NameObject('/Encoding')] = NameObject('/WinAnsiEncoding')
    del font['/DescendantFonts']
    src = tmp_path / 'unsafe-simple.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    assert not report.fonts[0].repair_possible
    assert 'no encoding guesses' in report.fonts[0].reason


def test_post_names_fallback(greek_fixture, tmp_path):
    """Glyph names recover Greek even when the embedded cmap table is absent."""
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    with TTFont(greek_fixture['fontpath']) as tt:
        del tt['cmap']
        buffer = BytesIO()
        tt.save(buffer)
    program = DecodedStreamObject()
    program.set_data(buffer.getvalue())
    descriptor = font['/DescendantFonts'][0]['/FontDescriptor']
    descriptor[NameObject('/FontFile2')] = writer._add_object(program)
    src, dst = tmp_path / 'post-only.pdf', tmp_path / 'post-only-fixed.pdf'
    writer.write(src)
    assert analyze_pdf(src).category == 'repairable'
    result = repair_pdf(src, dst)
    assert result.error is None
    assert _text(dst).strip() == greek_fixture['text']


@pytest.mark.parametrize('mixed_width', [False, True])
def test_explicit_encoding_codespace(greek_fixture, tmp_path, mixed_width):
    """Accept explicit two-byte CID encoding, refuse every mixed-width codespace."""
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    encoding = DecodedStreamObject()
    spaces = '2 begincodespacerange <00> <7F> <0000> <FFFF>' if mixed_width else '1 begincodespacerange <0000> <FFFF>'
    encoding.set_data((f'/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n'
        '/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> def\n'
        f'/CMapName /FixtureEncoding def /CMapType 1 def /WMode 0 def\n{spaces}\nendcodespacerange\n'
        '1 begincidrange <0000> <00FF> 0 endcidrange\n'
        'endcmap CMapName currentdict /CMap defineresource pop end end').encode())
    font[NameObject('/Encoding')] = writer._add_object(encoding)
    src, dst = tmp_path / 'encoded.pdf', tmp_path / 'encoded-fixed.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    if mixed_width:
        assert report.fonts[0].category == 'unsupported'
        assert not report.fonts[0].repair_possible
        assert 'two-byte codespace' in report.fonts[0].reason
    else:
        assert report.category == 'repairable'
        result = repair_pdf(src, dst)
        assert result.error is None
        assert _text(dst).strip() == greek_fixture['text']


def _sleeping_worker_stub(result_path, path, mode, destination, wordlist, sources=()):
    """A real spawned worker stalls on the first file and completes the next."""
    from glossapi.scripts.pdf_font_repair import _write_record
    if Path(path).name.startswith('slow'):
        if Path(path).name == 'slow-ignore.pdf':
            import signal
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Leave an incomplete temporary record to prove it is never consumed.
        Path(result_path + '.tmp').write_text('{')
        time.sleep(5)
    _write_record(Path(result_path), {'filename': Path(path).name,
                                    'category': 'consistent', 'error': None})


@pytest.mark.parametrize('slow_name', ['slow.pdf', 'slow-ignore.pdf'])
def test_sleeping_worker_deadline_and_next_file(tmp_path, monkeypatch, slow_name):
    import glossapi.scripts.pdf_font_repair as cli
    monkeypatch.setattr(cli, '_worker', _sleeping_worker_stub)
    timeout = .6
    started = time.monotonic()
    records = list(cli._batch([tmp_path / slow_name, tmp_path / 'next.pdf'],
                             'classify', None, 1, timeout, None))
    assert time.monotonic() - started < timeout + 1
    assert records[0]['category'] == 'timeout'
    assert 'Timeout' in records[0]['error']
    assert records[1]['filename'] == 'next.pdf'
    assert records[1]['error'] is None


@pytest.mark.parametrize('target_kind', ['classification', 'repair_report', 'repaired_pdf', 'summary'])
def test_cli_refuses_source_hardlinks(greek_fixture, tmp_path, target_kind, capsys):
    source = greek_fixture['broken']
    original = source.read_bytes()
    out_dir = tmp_path / 'repaired'
    out_dir.mkdir()
    report = tmp_path / 'report.jsonl'
    if target_kind == 'classification':
        os.link(source, report)
        argv = ['classify', '--pdfs', str(source), '--out', str(report)]
    elif target_kind in ('repair_report', 'repaired_pdf'):
        os.link(source, report if target_kind == 'repair_report' else out_dir / source.name)
        argv = ['repair', '--pdfs', str(source), '--out-dir', str(out_dir), '--report', str(report)]
    else:
        report.write_text(json.dumps({'filename': source.name, 'path': str(source),
                                     'category': 'repairable', 'pages': 1}) + '\n')
        manifest = tmp_path / 'manifest.csv'
        manifest.write_text('filename,pages\n' + source.name + ',1\n')
        os.link(source, tmp_path / 'summary.json')
        argv = ['summarize', '--report', str(report), '--manifest', str(manifest)]
    with pytest.raises(SystemExit, match='2'):
        main(argv)
    assert 'same inode/hard link' in capsys.readouterr().err
    assert source.read_bytes() == original


def test_cli_refuses_input_directory(greek_fixture, tmp_path, capsys):
    source = greek_fixture['broken']
    with pytest.raises(SystemExit, match='2'):
        main(['repair', '--pdfs', str(source), '--out-dir', str(source.parent),
              '--report', str(tmp_path / 'report.jsonl')])
    assert 'every input PDF directory' in capsys.readouterr().err
    assert not (tmp_path / 'report.jsonl').exists()


def test_repair_protects_other_selected_source(greek_fixture, tmp_path):
    other_source = tmp_path / 'other.pdf'
    other_source.write_bytes(greek_fixture['original'].read_bytes())
    target = tmp_path / 'target.pdf'
    os.link(other_source, target)
    result = repair_pdf(greek_fixture['broken'], target, protected_sources=[other_source])
    assert 'same inode/hard link' in result.error
    assert not result.written
    assert other_source.read_bytes() == greek_fixture['original'].read_bytes()


@pytest.mark.parametrize('candidates,expected', [
    (['A', 'α', 'Α'], 'Α'),  # Greek-block letters precede lower Latin codepoints.
    (['\u037e', 'β', 'α'], 'α'),  # Greek-block punctuation is not preferred.
    (['B', 'A'], 'A'),  # No Greek-block letter: choose the lowest codepoint.
    (['\u1f00', 'α'], 'α'),  # The requested preferred block is U+0370–U+03FF.
])
def test_nameless_glyph_alias_choice(greek_fixture, candidates, expected):
    with TTFont(greek_fixture['fontpath']) as tt:
        name = tt.getBestCmap()[ord('α')]
        gid = tt.getGlyphID(name)
        for table in tt['cmap'].tables:
            if table.isUnicode():
                table.cmap = {cp: glyph for cp, glyph in table.cmap.items() if glyph != name}
                table.cmap.update({ord(value): name for value in reversed(candidates)})
        tt['post'].formatType = 3.0
        buffer = BytesIO()
        tt.save(buffer)
    mapping, *_ = _glyph_unicode(buffer.getvalue(), 'FontFile2')
    assert mapping[gid] == expected


def test_no_candidate_retains_original_tounicode(greek_fixture, tmp_path):
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    with TTFont(greek_fixture['fontpath']) as tt:
        name = tt.getBestCmap()[ord('γ')]
        gid = tt.getGlyphID(name)
        for table in tt['cmap'].tables:
            if table.isUnicode():
                table.cmap = {cp: glyph for cp, glyph in table.cmap.items() if glyph != name}
        tt['post'].formatType = 3.0
        buffer = BytesIO()
        tt.save(buffer)
    mapping, *_ = _glyph_unicode(buffer.getvalue(), 'FontFile2')
    assert gid not in mapping
    program = DecodedStreamObject()
    program.set_data(buffer.getvalue())
    font['/DescendantFonts'][0]['/FontDescriptor'][NameObject('/FontFile2')] = writer._add_object(program)
    report, merged = _inspect_font(font)
    assert report.unmapped_codes >= 1
    assert merged[gid] == 'δ'  # Preserve the original shifted γ entry without an oracle.
    src, dst = tmp_path / 'unknown.pdf', tmp_path / 'unknown-fixed.pdf'
    writer.write(src)
    result = repair_pdf(src, dst)
    assert result.error is None
    assert _text(dst).strip() == greek_fixture['text'].replace('γ', 'δ')


@pytest.mark.parametrize('collision', ['planned', 'staging'])
def test_repair_report_reserves_output_paths(greek_fixture, tmp_path, collision, capsys):
    out_dir = tmp_path / 'repaired'
    report = (out_dir / greek_fixture['broken'].name if collision == 'planned'
              else out_dir / '.e1-staging' / 'report.jsonl')
    with pytest.raises(SystemExit, match='2'):
        main(['repair', '--pdfs', str(greek_fixture['broken']), '--out-dir', str(out_dir),
              '--report', str(report)])
    message = capsys.readouterr().err
    assert ('collides with planned PDF' if collision == 'planned' else 'inside the staging') in message
    assert not report.exists()
    assert not out_dir.exists()


@pytest.mark.parametrize('collision', ['planned', 'staging', 'report'])
def test_summary_reserves_output_paths(tmp_path, collision, capsys):
    report = tmp_path / ('summary.json' if collision == 'report' else 'report.jsonl')
    record = {'filename': 'fixture.pdf', 'category': 'consistent', 'pages': 1}
    if collision == 'planned':
        record['planned_output'] = str(tmp_path / 'summary.json')
    elif collision == 'staging':
        record['staging_dir'] = str(tmp_path)
    payload = json.dumps(record) + '\n'
    report.write_text(payload)
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('filename,pages\nfixture.pdf,1\n')
    with pytest.raises(SystemExit, match='2'):
        main(['summarize', '--report', str(report), '--manifest', str(manifest)])
    assert ('Summary would overwrite' if collision == 'report' else
            'collides with planned PDF' if collision == 'planned' else 'inside the staging') in capsys.readouterr().err
    assert report.read_text() == payload


@pytest.mark.parametrize('missing_tounicode', [False, True])
def test_direct_font_is_not_repair_eligible(greek_fixture, tmp_path, missing_tounicode):
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    resources = writer.pages[0]['/Resources']['/Font']
    key = next(iter(resources))
    direct = DictionaryObject(dict(resources[key].get_object().items()))
    if missing_tounicode:
        del direct['/ToUnicode']
    resources[key] = direct
    src, dst = tmp_path / 'direct.pdf', tmp_path / 'direct-copy.pdf'
    writer.write(src)
    report = analyze_pdf(src)
    assert report.error is None
    assert report.category == 'unsupported'
    assert report.fonts[0].object_id is None
    assert report.fonts[0].category == 'unsupported_direct'
    assert not report.fonts[0].repair_possible
    assert 'Direct font dictionaries' in report.fonts[0].reason
    result = repair_pdf(src, dst)
    assert result.error is None
    assert result.repaired_fonts == 0
    assert result.fonts[0].category == 'unsupported_direct'
    assert dst.read_bytes() == src.read_bytes()


@pytest.mark.parametrize('with_manifest', [False, True])
def test_summary_float_pages_and_optional_manifest(tmp_path, with_manifest, capsys):
    report = tmp_path / 'report.jsonl'
    report.write_text(json.dumps({'filename': 'fixture.pdf', 'category': 'mixed',
                                 'pages': 169, 'repair': {'repaired_fonts': 1},
                                 'before': {'totals': {'dictionary_valid_rate': .2}},
                                 'after': {'totals': {'dictionary_valid_rate': .99}}}) + '\n')
    argv = ['summarize', '--report', str(report)]
    if with_manifest:
        manifest = tmp_path / 'manifest.csv'
        manifest.write_text('filename,pages\nfixture.pdf,169.0\n')
        argv += ['--manifest', str(manifest)]
    assert main(argv) == 0
    result = json.loads((tmp_path / 'summary.json').read_text())
    assert result == json.loads(capsys.readouterr().out)
    assert result['categories']['mixed'] == {'docs': 1, 'pages': 169}
    assert result['projected_ocr_pages_removed'] == 169
    assert result['page_count_source'] == ('manifest' if with_manifest else 'report')


@pytest.mark.parametrize('latin_oracle', [False, True])
def test_repairable_font_with_not_embedded_font(greek_fixture, tmp_path, latin_oracle):
    """Eligibility is per font, including an actually mixed document category."""
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    fonts = writer.pages[0]['/Resources']['/Font']
    fonts[NameObject('/Unembedded')] = writer._add_object(DictionaryObject({
        NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica')}))
    expected = greek_fixture['text']
    if latin_oracle:
        # Keep Greek-shaped glyphs but give the fixture a non-Greek cmap oracle.
        # This creates a real mixed category (non-Greek repairable + not_embedded)
        # and reproduces the old category-based worker skip without mocking it.
        font = next(iter(fonts.values())).get_object()
        letters = sorted(set(greek_fixture['text'].replace(' ', '')))
        replacements = {char: chr(ord('A') + index) for index, char in enumerate(letters)}
        with TTFont(greek_fixture['fontpath']) as tt:
            for table in tt['cmap'].tables:
                if table.isUnicode():
                    table.cmap = {ord(replacements.get(chr(cp), chr(cp))): name
                                  for cp, name in table.cmap.items()}
            # Discard post names so unused Greek component glyphs do not supply
            # additional Greek mappings and change the document category.
            tt['post'].formatType = 3.0
            buffer = BytesIO()
            tt.save(buffer)
        program = DecodedStreamObject()
        program.set_data(buffer.getvalue())
        font['/DescendantFonts'][0]['/FontDescriptor'][NameObject('/FontFile2')] = writer._add_object(program)
        expected = ''.join(replacements.get(char, char) for char in expected)
    source = tmp_path / 'mixed.pdf'
    writer.write(source)
    before_bytes = source.read_bytes()
    report = analyze_pdf(source)
    assert report.category == ('mixed' if latin_oracle else 'repairable')
    assert sorted(font.category for font in report.fonts) == ['not_embedded', 'repairable']
    out_dir, report_path = tmp_path / 'fixed', tmp_path / 'report.jsonl'
    assert main(['repair', '--pdfs', str(source), '--out-dir', str(out_dir),
                 '--report', str(report_path)]) == 0
    record = json.loads(report_path.read_text())
    assert record['before_category'] == report.category
    assert record['after_category'] == 'mixed'
    assert record['repair']['repaired_fonts'] == 1
    assert 'skip_reason' not in record
    destination = out_dir / source.name
    assert _text(destination).strip() == expected
    assert sorted(font.category for font in analyze_pdf(destination).fonts) == ['consistent', 'not_embedded']
    assert source.read_bytes() == before_bytes
    _assert_only_tounicode_changed(source, destination)


def test_worker_warning_counts_and_debug_routing(tmp_path, monkeypatch, caplog, capsys):
    import glossapi.scripts.pdf_font_repair as cli
    from glossapi.pdf_repair import PdfFontReport
    font_logger = logging.getLogger('fontTools.ttLib.tables._p_o_s_t')
    pdf_logger = logging.getLogger('pypdf._reader')
    saved = [(logging.getLogger(name).handlers[:], logging.getLogger(name).level,
              logging.getLogger(name).propagate) for name in ('fontTools', 'pypdf')]

    def noisy_analyze(path):
        if Path(path).name == 'noisy.pdf':
            font_logger.warning('7 extra bytes in post.stringData array')
            pdf_logger.warning('Ignoring wrong pointing object 12 0 (offset 17)')
            pdf_logger.warning('Ignoring wrong pointing object 13 0 (offset 18)')
        return PdfFontReport(str(path), category='consistent', pages=1)

    monkeypatch.setattr(cli, 'analyze_pdf', noisy_analyze)
    result_path = tmp_path / 'result.json'
    with caplog.at_level(logging.DEBUG, logger='glossapi.pdf_repair.diagnostics'):
        cli._worker(str(result_path), str(tmp_path / 'noisy.pdf'), 'classify', None, None)
    record = json.loads(result_path.read_text())
    assert record['warning_count'] == 3
    assert record['warning_counts'] == {'fontTools': 1, 'pypdf': 2}
    assert 'extra bytes in post.stringData' in caplog.text
    assert 'Ignoring wrong pointing object' in caplog.text
    assert all(entry.levelno == logging.DEBUG for entry in caplog.records)
    assert capsys.readouterr().err == ''
    for name, state in zip(('fontTools', 'pypdf'), saved):
        logger = logging.getLogger(name)
        assert (logger.handlers, logger.level, logger.propagate) == state
    cli._worker(str(result_path), str(tmp_path / 'quiet.pdf'), 'classify', None, None)
    assert json.loads(result_path.read_text())['warning_count'] == 0


def test_warning_counts_survive_worker_error(tmp_path, monkeypatch, capsys):
    import glossapi.scripts.pdf_font_repair as cli

    def noisy_failure(path):
        logging.getLogger('pypdf._reader').warning('Ignoring wrong pointing object')
        raise ValueError('fixture parse failure')

    monkeypatch.setattr(cli, 'analyze_pdf', noisy_failure)
    result_path = tmp_path / 'result.json'
    cli._worker(str(result_path), 'bad.pdf', 'classify', None, None)
    record = json.loads(result_path.read_text())
    assert record['warning_count'] == 1
    assert record['warning_counts'] == {'fontTools': 0, 'pypdf': 1}
    assert record['error'] == 'ValueError: fixture parse failure'
    assert capsys.readouterr().err == ''


def _crashing_worker_stub(result_path, path, mode, destination, wordlist, sources=()):
    from glossapi.scripts.pdf_font_repair import _write_record
    if Path(path).name == 'crash.pdf':
        Path(result_path + '.tmp').write_text('{')
        os._exit(3)
    _write_record(Path(result_path), {'filename': Path(path).name, 'category': 'consistent', 'error': None})


def test_crashed_worker_does_not_consume_partial_json(tmp_path, monkeypatch):
    import glossapi.scripts.pdf_font_repair as cli
    monkeypatch.setattr(cli, '_worker', _crashing_worker_stub)
    records = list(cli._batch([tmp_path / 'crash.pdf', tmp_path / 'next.pdf'],
                             'classify', None, 1, 2, None))
    assert 'exitcode=3' in records[0]['error']
    assert records[0]['warning_count'] is None
    assert records[1]['filename'] == 'next.pdf'
    assert records[1]['error'] is None


@pytest.mark.parametrize('temporary', [False, True])
def test_task_record_protects_staging_hardlinks(greek_fixture, tmp_path, temporary):
    from glossapi.scripts.pdf_font_repair import _write_record
    source = greek_fixture['broken']
    original = source.read_bytes()
    result_path = tmp_path / 'result.json'
    os.link(source, Path(str(result_path) + '.tmp') if temporary else result_path)
    with pytest.raises(ValueError, match='same inode/hard link'):
        _write_record(result_path, {'category': 'consistent'}, (str(source),))
    assert source.read_bytes() == original


def test_real_fonttools_warning_is_quiet_and_counted(greek_fixture, tmp_path):
    from fontTools.ttLib.tables.DefaultTable import DefaultTable
    writer = PdfWriter(clone_from=greek_fixture['broken'])
    font = next(iter(writer.pages[0]['/Resources']['/Font'].values())).get_object()
    with TTFont(greek_fixture['fontpath']) as tt:
        # Append trailing bytes to a real post table to exercise fontTools' own
        # warning, rather than a patched log call.
        post = DefaultTable('post')
        post.data = tt['post'].compile(tt) + b'\x00\x00'
        tt['post'] = post
        buffer = BytesIO()
        tt.save(buffer)
    program = DecodedStreamObject()
    program.set_data(buffer.getvalue())
    font['/DescendantFonts'][0]['/FontDescriptor'][NameObject('/FontFile2')] = writer._add_object(program)
    source = tmp_path / 'noisy-font.pdf'
    writer.write(source)
    # The warning capture belongs to the Python reference: run the CLI (and its spawned workers) with
    # the native extension unimportable, which is the selector's documented fallback. The native
    # backend parses fonts in Rust and so records no fontTools warnings (pdfrepair README).
    stub = tmp_path / 'no-native'
    stub.mkdir()
    (stub / 'glossapi_rs_pdfrepair.py').write_text("raise ImportError('reference backend requested by the test')\n")
    records = {}
    import importlib.util
    native_available = importlib.util.find_spec('glossapi_rs_pdfrepair') is not None
    for backend, extra_path in (('python', [str(stub)]), ('native', []))[:2 if native_available else 1]:
        report = tmp_path / f'report-{backend}.jsonl'
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([*extra_path, *sys.path])}
        completed = subprocess.run([sys.executable, '-m', 'glossapi.scripts.pdf_font_repair',
                                    'classify', '--pdfs', str(source), '--out', str(report)],
                                   capture_output=True, text=True, timeout=30, env=env)
        assert completed.returncode == 0, completed.stderr
        assert 'post.stringData' not in completed.stderr
        records[backend] = json.loads(report.read_text())
    record = records['python']
    assert record['warning_counts']['fontTools'] >= 1
    assert record['warning_count'] == sum(record['warning_counts'].values())
    if native_available:
        native = records['native']
        assert native['category'] == record['category']
        assert native['warning_counts']['fontTools'] == 0
        assert native['warning_count'] == sum(native['warning_counts'].values())


def test_greek_fixture_font_choice_coverage_and_ci_failure(monkeypatch, tmp_path):
    """K2 batch 3: Linux fonts (DejaVuSans) are candidates; coverage is checked per character; in CI mode a
    missing covering font fails clearly instead of skipping the Greek-fixture tests."""
    import _pytest.outcomes
    monkeypatch.delenv('GLOSSAPI_TEST_GREEK_FONT', raising=False)
    assert any('DejaVuSans.ttf' in path for path in GREEK_FIXTURE_FONTS)
    found = [path for path in GREEK_FIXTURE_FONTS if Path(path).is_file()]
    if found:
        path, font = greek_fixture_font()
        assert path in found and all(ord(c) in font.getBestCmap() for c in GREEK_FIXTURE_TEXT)
        font.close()
        with pytest.raises((_pytest.outcomes.Skipped, _pytest.outcomes.Failed), match='missing'):
            greek_fixture_font('αβ\u0378')  # an unassigned code point: no font maps it (DejaVuSans maps emoji)
    monkeypatch.setattr(sys.modules[__name__], 'GREEK_FIXTURE_FONTS', (str(tmp_path / 'absent.ttf'),))
    for name in ('CI', 'GLOSSAPI_REFERENCE_REQUIRED'):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(_pytest.outcomes.Skipped, match='No TrueType font covering'):
        greek_fixture_font()
    for name in ('CI', 'GLOSSAPI_REFERENCE_REQUIRED'):
        monkeypatch.setenv(name, 'true' if name == 'CI' else '1')
        with pytest.raises(_pytest.outcomes.Failed, match='Install fonts-dejavu-core'):
            greek_fixture_font()
        monkeypatch.delenv(name)
