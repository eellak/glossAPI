"""K2 batch 2a: the PDFium page-count surfaces that replaced the PyMuPDF/pypdf/PyPDF2 fallbacks, and the
PDFium fixture builder that replaced PyMuPDF in the tests (B2: text, pixels, colour/alpha, geometry,
rotation/crop, malformed/encrypted/empty PDFs, page ranges, lifetimes and fallback behaviour).

pypdf is the independent oracle (reference group). The retired fallbacks are declared, not emulated:
- the extract work estimator returns 1 when PDFium cannot open a PDF (it used to try pypdf, then PyPDF2);
- the strict counters (checkpoint, Docling cost, batched benchmark, OCR reassembly) raise PDFium's error
  (the checkpoint counter used to try PyMuPDF, then PDFium, then pypdf, then PyPDF2).
"""
from contextlib import closing
import os
from pathlib import Path
import subprocess
import sys

import pypdfium2 as pdfium
import pytest

from pdf_fixtures import FixturePdf
from reference_oracles import oracle

REPO = Path(__file__).resolve().parents[1]


def _counters():
    from glossapi.corpus.phase_extract import _estimate_extract_work_pages
    from glossapi.ocr.page_artifacts import _page_count
    from glossapi.scripts.batched_extract_benchmark import page_count
    from glossapi.scripts.docling_cost_benchmark import _count_pages
    from glossapi.scripts.extract_checkpoint_benchmark import _count_pdf_pages
    return {'phase_extract._estimate_extract_work_pages': _estimate_extract_work_pages,
            'extract_checkpoint_benchmark._count_pdf_pages': _count_pdf_pages,
            'docling_cost_benchmark._count_pages': _count_pages,
            'batched_extract_benchmark.page_count': page_count,
            'ocr.page_artifacts._page_count': _page_count}


ESTIMATOR, STRICT = 'phase_extract._estimate_extract_work_pages', ('extract_checkpoint_benchmark._count_pdf_pages',
    'docling_cost_benchmark._count_pages', 'batched_extract_benchmark.page_count', 'ocr.page_artifacts._page_count')


def _marked(path, sizes, *, rotations=None, crops=None):
    """One page per size, each carrying the text 'Page N' (page order is checked through the text)."""
    doc = FixturePdf()
    for index, (width, height) in enumerate(sizes):
        page = doc.new_page(width, height)
        doc.insert_text(page, (20, 40), f'Page {index + 1}', fontsize=12)
        if rotations:
            page.set_rotation(rotations[index])
        if crops and crops[index]:
            page.set_cropbox(*crops[index])
    return doc.save(path)


def _oracle_pages(path, password=None):
    pypdf = oracle('pypdf')
    reader = pypdf.PdfReader(str(path))
    if password is not None:
        reader.decrypt(password)
    return reader


def _fds():
    return len(os.listdir('/dev/fd'))


@pytest.mark.parametrize('pages', [1, 3, 7])
def test_page_counts_and_order_match_the_pypdf_oracle(tmp_path, pages):
    sizes = [(300 + 10 * i, 400 + 7 * i) for i in range(pages)]
    path = _marked(tmp_path / 'doc.pdf', sizes)
    reader = _oracle_pages(path)
    assert {name: count(path) for name, count in _counters().items()} == {name: len(reader.pages) for name in _counters()}
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            assert textpage.get_text_range().strip() == f'Page {index + 1}'
            assert f'Page {index + 1}' in reader.pages[index].extract_text()
            assert tuple(round(v) for v in page.get_size()) == sizes[index] == tuple(int(v) for v in reader.pages[index].mediabox[2:])
            textpage.close(); page.close()
    from glossapi.page_quality import compress_page_ranges
    assert compress_page_ranges(list(range(1, _counters()['ocr.page_artifacts._page_count'](path) + 1))) == \
        (f'1-{pages}' if pages > 1 else '1')


def test_rotation_and_crop_keep_counts_and_match_oracle_geometry(tmp_path):
    rotations = [0, 90, 180, 270]
    crops = [None, (10, 20, 200, 300), None, (0, 0, 150, 150)]
    path = _marked(tmp_path / 'rotated.pdf', [(400, 500)] * 4, rotations=rotations, crops=crops)
    reader = _oracle_pages(path)
    assert {count(path) for count in _counters().values()} == {4}
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        for index in range(4):
            page, expected = pdf[index], reader.pages[index]
            assert page.get_rotation() == expected.rotation == rotations[index]
            assert tuple(page.get_mediabox()) == tuple(float(v) for v in expected.mediabox)
            assert tuple(page.get_cropbox()) == tuple(float(v) for v in expected.cropbox)
            page.close()


def test_empty_pdf(tmp_path):
    pypdf = oracle('pypdf')
    path = tmp_path / 'empty.pdf'
    pypdf.PdfWriter().write(path)
    assert len(pypdf.PdfReader(path).pages) == 0
    counters = _counters()
    # pypdfium2 4.30 refuses a zero-page document (it raises when FPDF_GetPageCount < 1), so the estimator
    # falls back to 1 and every strict counter raises; docling_cost's "no pages" ValueError is never reached.
    assert counters[ESTIMATOR](path) == 1
    for name in STRICT:
        with pytest.raises(pdfium.PdfiumError):
            counters[name](path)


@pytest.mark.xfail(strict=True, reason='pypdfium2 4.30 _open_pdf raises on a zero-page document without closing '
                   'the loaded FPDF_DOCUMENT, leaking one descriptor per open (K2 batch 2a gap; recheck at U1)')
def test_zero_page_pdf_releases_its_handle(tmp_path):
    pypdf = oracle('pypdf')
    path = tmp_path / 'empty.pdf'
    pypdf.PdfWriter().write(path)
    estimate = _counters()[ESTIMATOR]
    before = _fds()
    assert estimate(path) == 1
    assert _fds() == before


@pytest.mark.parametrize('kind', ['garbage', 'truncated', 'zero_bytes'])
def test_malformed_pdfs(tmp_path, kind):
    good = _marked(tmp_path / 'good.pdf', [(300, 300)] * 3)
    path = tmp_path / f'{kind}.pdf'
    path.write_bytes({'garbage': b'not a PDF at all', 'zero_bytes': b'',
                      'truncated': good.read_bytes()[:len(good.read_bytes()) // 3]}[kind])
    counters = _counters()
    before = _fds()
    assert counters[ESTIMATOR](path) == 1
    for name in STRICT:
        with pytest.raises(pdfium.PdfiumError):
            counters[name](path)
    assert _fds() == before
    # The pypdf oracle cannot count these either (declared: no parser-specific recovery is lost here).
    pypdf = oracle('pypdf')
    with pytest.raises(Exception):
        len(pypdf.PdfReader(str(path), strict=False).pages)


def test_encrypted_pdfs(tmp_path):
    pypdf = oracle('pypdf')
    good = _marked(tmp_path / 'good.pdf', [(300, 300)] * 3)
    locked, owner_only = tmp_path / 'locked.pdf', tmp_path / 'owner_only.pdf'
    for path, user in ((locked, 'secret'), (owner_only, '')):
        writer = pypdf.PdfWriter(clone_from=good)
        writer.encrypt(user_password=user, owner_password='owner')
        writer.write(path)
    counters = _counters()
    # An empty user password opens without a password in both PDFium and the oracle.
    assert {count(owner_only) for count in counters.values()} == {len(pypdf.PdfReader(owner_only).pages)} == {3}
    # A user password: PDFium refuses without it; the estimator falls back to 1, strict counters raise.
    assert len(_oracle_pages(locked, 'secret').pages) == 3
    assert counters[ESTIMATOR](locked) == 1
    for name in STRICT:
        with pytest.raises(pdfium.PdfiumError):
            counters[name](locked)


def test_handles_are_closed_and_counts_are_not_cached(tmp_path):
    path = _marked(tmp_path / 'doc.pdf', [(300, 300)] * 2)
    counters = _counters()
    for count in counters.values():
        count(path)  # warm imports
    before = _fds()
    for _ in range(25):
        for count in counters.values():
            assert count(path) == 2
    assert _fds() == before
    path.unlink()
    _marked(path, [(300, 300)] * 5)  # same name, new content
    assert {count(path) for count in counters.values()} == {5}
    assert _fds() == before


def test_without_pdfium_no_retired_fallback_is_attempted(tmp_path):
    """PyPDF2 and the pypdf/PyMuPDF count fallbacks are retired: none is imported even when installed."""
    path = _marked(tmp_path / 'doc.pdf', [(300, 300)] * 3)
    code = f'''
import importlib.abc, json, sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_k2_pdf_surfaces import _counters
counters = _counters()
attempted = []
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in ('pypdf', 'PyPDF2', 'fitz', 'pymupdf'):
            attempted.append(fullname)
        if root in ('pypdfium2', 'pypdfium2_raw'):
            raise ModuleNotFoundError('injected: PDFium unavailable', name=root)
for name in [name for name in sys.modules if name.split('.')[0] in ('pypdfium2', 'pypdfium2_raw', 'pypdf', 'PyPDF2', 'fitz')]:
    del sys.modules[name]
sys.meta_path.insert(0, Guard())
out = {{}}
for name, count in counters.items():
    try:
        out[name] = count(__import__('pathlib').Path({str(path)!r}))
    except ImportError as exc:
        out[name] = 'ImportError'
print(json.dumps(dict(out=out, attempted=attempted)))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                            env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)})
    assert result.returncode == 0, result.stderr
    report = __import__('json').loads(result.stdout.strip().splitlines()[-1])
    assert report['attempted'] == []
    assert report['out'] == {ESTIMATOR: 1, **{name: 'ImportError' for name in STRICT}}


def test_no_retired_pdf_library_in_source_or_declarations():
    """Callsite inventory: production source names only PDFium; pypdf/fontTools only in the reference backend."""
    import re
    retired = re.compile(r'^\s*(import|from)\s+(PyPDF2|fitz|pymupdf)\b', re.M)
    reference = re.compile(r'^\s*(import|from)\s+(pypdf|fontTools)\b', re.M)
    for path in sorted((REPO / 'src').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        assert not retired.search(text), path
        if reference.search(text):
            assert path.relative_to(REPO).as_posix() == 'src/glossapi/pdf_repair/tounicode.py', path
    declarations = (REPO / 'pyproject.toml').read_text() + ''.join(
        p.read_text() for p in sorted((REPO / 'dependency_setup').glob('*.txt')))
    assert not any(name.lower() in declarations.lower() for name in ('pypdf2', 'pymupdf'))


# ---- the fixture builder that replaced PyMuPDF in the tests -------------------------------------------------

def test_builder_text_round_trips_through_pdfium_and_the_oracle(tmp_path):
    font = next((p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                            Path('/System/Library/Fonts/Supplemental/Arial.ttf')] if p.exists()), None)
    if font is None:
        pytest.skip('No Greek fixture font')
    doc = FixturePdf()
    page = doc.new_page()
    doc.insert_text(page, (72, 72), 'Native heading\nA B', fontsize=11)
    doc.insert_text(page, (72, 200), 'και άνθρωπος', fontsize=14, font=doc.load_truetype(font))
    path = doc.save(tmp_path / 'text.pdf')
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        page = pdf[0]
        textpage = page.get_textpage()
        assert textpage.get_text_range().split('\r\n') == ['Native heading', 'A B', 'και άνθρωπος']
        # PyMuPDF's line advance: 11 pt Helvetica x 1.374 = 15.114 pt between baselines.
        first, second = textpage.get_charbox(0), textpage.get_charbox(len('Native heading') + 2)
        assert first[1] - second[1] == pytest.approx(15.114, abs=0.01)
        textpage.close(); page.close()
    pypdf = oracle('pypdf')
    reader = pypdf.PdfReader(path)
    # PDFium's generated ToUnicode maps the space glyph to U+0020 and, in an overlapping bfrange, U+00A0;
    # pypdf takes the latter. Compare words, not the separator.
    extracted = reader.pages[0].extract_text().replace('\xa0', ' ')
    assert 'Native heading' in extracted and 'και άνθρωπος' in extracted
    fonts = {str(f.get_object()['/Subtype']): f.get_object() for f in reader.pages[0]['/Resources']['/Font'].values()}
    assert set(fonts) == {'/Type1', '/Type0'}
    assert fonts['/Type0']['/Encoding'] == '/Identity-H'
    descendant = fonts['/Type0']['/DescendantFonts'][0].get_object()
    assert descendant['/Subtype'] == '/CIDFontType2'
    assert descendant['/FontDescriptor']['/FontFile2'].get_data() == font.read_bytes()


def test_builder_images_pixels_colour_alpha_and_geometry(tmp_path):
    PIL = pytest.importorskip('PIL.Image')
    doc = FixturePdf()
    page = doc.new_page(600, 800)
    doc.insert_image(page, (0, 0, 600, 800), (20, 20, (200, 10, 30)))
    doc.insert_image(page, (20, 20, 580, 740), PIL.new('RGB', (550, 700), (5, 120, 250)), fit=True)
    path = doc.save(tmp_path / 'images.pdf')
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        page = pdf[0]
        solid, photo = page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])
        assert solid.get_pos() == pytest.approx((0, 0, 600, 800))
        scaled = 560 * 700 / 550
        assert photo.get_pos() == pytest.approx((20, 800 - 20 - (720 + scaled) / 2, 580, 800 - 20 - (720 - scaled) / 2), abs=1e-3)
        for obj, size, bgr in ((solid, (20, 20), (30, 10, 200)), (photo, (550, 700), (250, 120, 5))):
            assert tuple(obj.get_size()) == size
            pixels = obj.get_bitmap(render=False).to_numpy()
            assert pixels.shape == (size[1], size[0], 3) and (pixels == bgr).all()
        page.close()
    pypdf = oracle('pypdf')
    images = pypdf.PdfReader(path).pages[0].images
    assert [image.image.getpixel((0, 0)) for image in images] == [(200, 10, 30), (5, 120, 250)]
    for image in images:
        assert '/SMask' not in image.indirect_reference.get_object()  # no alpha, as the PyMuPDF fixtures (alpha=False)


def test_builder_page_import_and_blank_pages(tmp_path):
    first = _marked(tmp_path / 'first.pdf', [(300, 300), (310, 310)])
    doc = FixturePdf()
    doc.import_pdf(first)
    doc.new_page(400, 400)
    doc.import_pdf(first)
    path = doc.save(tmp_path / 'joined.pdf')
    reader = _oracle_pages(path)
    assert [int(p.mediabox[2]) for p in reader.pages] == [300, 310, 400, 300, 310]
    assert [p.extract_text().strip() for p in reader.pages] == ['Page 1', 'Page 2', '', 'Page 1', 'Page 2']
