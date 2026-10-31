"""Measured scan-layer quality, unchanged legacy profiles, and source identity."""
import hashlib
from pathlib import Path

import pytest

from glossapi.page_quality import page_ocr_decision, page_quality, page_rule_selection
from test_page_quality import COMMON_GREEK
from pdf_fixtures import FixturePdf


def _page_text(path):
    """The first page's PDFium text with PyMuPDF-style line breaks (formerly fitz get_text())."""
    from contextlib import closing
    import pypdfium2 as pdfium
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        page = pdf[0]
        textpage = page.get_textpage()
        try:
            return textpage.get_text_range().replace('\r\n', '\n')
        finally:
            textpage.close(); page.close()


BASE = dict(chars=300, greek_chars=200, latin_chars=10, other_chars=0,
            valid_greek_word_rate=None, image_area_ratio=1, words=60, junk_token_share=0)


@pytest.fixture
def scanner_layer_pdf(tmp_path):
    """Independent scan plus text; does not use the legacy corrupt-CMap fixture."""
    font = next((p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                            Path('/System/Library/Fonts/Supplemental/Arial.ttf')]
                 if p.exists()), None)
    if font is None:
        pytest.skip('No Greek fixture font')
    target = tmp_path / 'scanner_layer.pdf'
    doc = FixturePdf()
    page = doc.new_page()
    words = sorted(COMMON_GREEK)
    text = '\n'.join(' '.join(words[i:i+6]) for i in range(0,len(words),6))
    doc.insert_image(page, (0, 0, 595, 842), (20, 20, (200, 200, 200)))  # underlay (overlay=False)
    doc.insert_text(page, (72,72), text, fontsize=14, font=doc.load_truetype(font))
    doc.save(target)
    return target


@pytest.mark.parametrize("changes,expected", [
    ({}, (True, "image_dominant_unknown_validity")),
    ({"chars": 2000}, (True, "image_dominant_unknown_validity")),
    ({"image_area_ratio": .5}, (True, "image_dominant_unknown_validity")),
    ({"valid_greek_word_rate": .399}, (True, "image_dominant_bad_text")),
    ({"valid_greek_word_rate": .4}, (True, "image_dominant_bad_text")),
    ({"valid_greek_word_rate": .9}, (False, "text_layer")),
    ({"valid_greek_word_rate": 1, "chars": 49}, (True, "scanned_page")),
    ({"valid_greek_word_rate": .9, "chars": 50}, (False, "text_layer")),
    ({"valid_greek_word_rate": 1, "pua_char_count": 1}, (True, "image_dominant_contaminated_text")),
    ({"valid_greek_word_rate": 1, "control_char_count": 1}, (True, "image_dominant_contaminated_text")),
    ({"valid_greek_word_rate": .6, "min_valid_word_rate": .7}, (True, "image_dominant_bad_text")),
    ({"valid_greek_word_rate": float("nan")}, (True, "image_dominant_bad_text")),
])
def test_v4_image_quality(changes, expected):
    values = BASE | changes
    assert page_ocr_decision(**values) == expected
    assert page_rule_selection("scanned_page" if expected[1] == "scanned_page" else "image_dominant_sparse_text", **{
        k: v for k, v in values.items() if k in {
            'valid_greek_word_rate', 'words', 'pua_char_count', 'control_char_count',
            'image_area_ratio', 'min_valid_word_rate', 'greek_chars', 'latin_chars',
            'junk_token_share'}}) == expected[0]


@pytest.mark.parametrize("rate", [None, 0, .39, .4, .49, .5, 1])
@pytest.mark.parametrize("coverage", [0, .3, .499])
@pytest.mark.parametrize("changes", [{}, {"chars": 0, "words": 0}, {"pua_char_count": 1},
                                       {"other_chars": 200}, {"words": 10}])
def test_non_image_v4_is_v3(rate, coverage, changes):
    values = BASE | dict(valid_greek_word_rate=rate, image_area_ratio=coverage) | changes
    assert page_ocr_decision(**values, rule_profile="v4") == page_ocr_decision(**values, rule_profile="v3")


def test_custom_image_threshold_and_legacy_profiles():
    values = BASE | dict(image_area_ratio=.6, image_dominant_ratio=.7)
    assert page_ocr_decision(**values) == page_ocr_decision(**values, rule_profile="v3")
    assert page_ocr_decision(**BASE, rule_profile="v3") == (False, "image_dominant_sparse_text")
    assert page_ocr_decision(**BASE, rule_profile="v2") == (True, "image_dominant_sparse_text")


def test_default_dictionary_integrity_and_membership(monkeypatch):
    import importlib
    quality = importlib.import_module('glossapi.page_quality')
    words = quality._default_words()
    assert {'και', 'άνθρωποσ', 'ελλάδα'} <= words
    assert 'jveg' not in words
    assert len(words) > 700_000
    quality._default_words.cache_clear()
    monkeypatch.setattr(quality.gzip, 'decompress', lambda data: b'1\ncorrupt\n')
    with pytest.raises(ValueError, match='SHA256'):
        quality._default_words()
    quality._default_words.cache_clear()


def test_report_page_profiles_and_dictionary_precedence(scanner_layer_pdf, monkeypatch, tmp_path):
    monkeypatch.delenv('GLOSSAPI_GREEK_WORDLIST', raising=False)
    default = page_quality(scanner_layer_pdf)
    assert default.rule_profile == 'v4' and default.dictionary_loaded
    assert all(p.rule_profile == 'v4' for p in default.pages)
    assert default.dictionary_identity['sha256'] == 'f08daefb302600beb1b345e4fd77f4ecf6617aa080a72efe6ae7eec0ad5b2ac7'
    configured = tmp_path / 'words.txt'
    configured.write_text('\n'.join(COMMON_GREEK))
    monkeypatch.setenv('GLOSSAPI_GREEK_WORDLIST', str(configured))
    report = page_quality(scanner_layer_pdf)
    assert report.dictionary_identity['sha256'] == hashlib.sha256(configured.read_bytes()).hexdigest()
    assert report.pages[0].valid_greek_word_rate == 1
    assert page_quality(scanner_layer_pdf, wordlist=set()).dictionary_identity is None


def test_real_scan_layer_with_unknown_bad_and_good_validity(scanner_layer_pdf):
    target = scanner_layer_pdf
    assert not page_quality(target, wordlist=set(), rule_profile='v3').pages[0].page_needs_ocr
    unknown = page_quality(target, wordlist=set()).pages[0]
    assert unknown.page_needs_ocr and unknown.decision == 'image_dominant_unknown_validity'
    bad = page_quality(target, wordlist={'ασυναρτησία'}).pages[0]
    assert bad.page_needs_ocr and bad.decision == 'image_dominant_bad_text'
    good = page_quality(target, wordlist=COMMON_GREEK).pages[0]
    assert not good.page_needs_ocr and good.decision == 'text_layer'


def test_default_non_image_pdf_keeps_v3_routing(scanner_layer_pdf, tmp_path, monkeypatch):
    target = tmp_path / 'non_image.pdf'
    font = next(p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                           Path('/System/Library/Fonts/Supplemental/Arial.ttf')] if p.exists())
    doc = FixturePdf()
    doc.insert_text(doc.new_page(), (72,72), _page_text(scanner_layer_pdf), fontsize=14, font=doc.load_truetype(font))
    doc.save(target)
    monkeypatch.delenv('GLOSSAPI_GREEK_WORDLIST', raising=False)
    old, new = page_quality(target, rule_profile='v3'), page_quality(target)
    assert new.dictionary_identity['scope'] == 'image-dominant pages'
    assert new.pages[0].valid_greek_word_rate is None
    assert new.pages[0].image_area_ratio < .5
    assert (new.pages[0].page_needs_ocr, new.pages[0].decision) == (old.pages[0].page_needs_ocr, old.pages[0].decision)


def test_v4_includes_supplementary_pua_without_changing_legacy_counts():
    from glossapi.page_quality import text_contamination_counts
    assert text_contamination_counts('\U000f0000')['pua_char_count'] == 0
    assert text_contamination_counts('\U000f0000', all_pua=True)['pua_char_count'] == 1


LATIN = dict(chars=631, greek_chars=0, latin_chars=413, other_chars=65,
             words=91, valid_greek_word_rate=None, image_area_ratio=1)


@pytest.mark.parametrize('rate', [None, 0, .399, .4, 1])
@pytest.mark.parametrize('changes', [{}, {'greek_chars': 200, 'latin_chars': 200},
                                    {'chars': 600}, {'other_chars': 0},
                                    {'chars': 1000, 'other_chars': 300}])
def test_clean_latin_layer_retains_v3(rate, changes):
    values = LATIN | {'valid_greek_word_rate': rate} | changes
    result = page_ocr_decision(**values)
    assert result == page_ocr_decision(**values, rule_profile='v3') == (False, 'text_layer')
    assert not page_rule_selection(result[1], valid_greek_word_rate=rate,
                                   words=values['words'], image_area_ratio=1,
                                   validity_not_applicable=True)


@pytest.mark.parametrize('changes,expected', [
    ({'greek_chars': 201, 'latin_chars': 200}, 'image_dominant_unknown_validity'),
    ({'greek_chars': 201, 'latin_chars': 200, 'valid_greek_word_rate': 0}, 'image_dominant_bad_text'),
    ({'greek_chars': 0, 'latin_chars': 0}, 'image_dominant_unknown_validity'),
    ({'other_chars': 190}, 'image_dominant_unknown_validity'),
    ({'chars': 599}, 'image_dominant_unknown_validity'),
    ({'chars': 49}, 'scanned_page'),
    ({'pua_char_count': 1}, 'image_dominant_contaminated_text'),
    ({'control_char_count': 1}, 'image_dominant_contaminated_text'),
    ({'min_chars': 632}, 'scanned_page'),
    ({'image_dominant_min_chars': 632}, 'image_dominant_unknown_validity'),
    ({'min_greek_share': .2, 'greek_chars': 101, 'latin_chars': 400}, 'image_dominant_unknown_validity'),
    ({'max_other_share': .1}, 'image_dominant_unknown_validity'),
])
def test_latin_exception_keeps_v4_safety_gates(changes, expected):
    assert page_ocr_decision(**(LATIN | changes)) == (True, expected)


def test_sparse_latin_caption_cannot_certify_image_body():
    # Sheet 03 is a URL over a broken layer, not the clean prose in sheet 05.
    assert page_ocr_decision(**(LATIN | dict(chars=54, latin_chars=30,
        other_chars=17, words=7))) == (True, 'image_dominant_unknown_validity')


def test_latin_pdf_records_validity_not_applicable(tmp_path):
    target = tmp_path / 'english_figure.pdf'
    doc = FixturePdf()
    page = doc.new_page()
    text = 'First of all, we compute the minimum bounding rectangle. '
    doc.insert_image(page, (0, 0, 595, 842), (20, 20, (200, 200, 200)))  # underlay (overlay=False)
    doc.insert_text(page, (72, 72), '\n'.join([text] * 14), fontsize=10)
    doc.save(target)
    old = page_quality(target, rule_profile='v3').pages[0]
    report = page_quality(target)
    new = report.pages[0]
    assert new.chars >= 600 and new.greek_chars == 0 and new.image_area_ratio == 1
    assert new.valid_greek_word_rate is None and new.validity_not_applicable
    assert (new.page_needs_ocr, new.decision) == (old.page_needs_ocr, old.decision) == (False, 'text_layer')
    assert report.to_dict()['pages'][0]['validity_not_applicable'] is True
    assert not old.validity_not_applicable


# ---- CP-OCRDEF F5: every effective routing threshold travels with the page row ----

def _image_text_pdf(path, coverage=0.6, text='The archive stores printed books reports and articles for readers'):
    """Raw PDF (no fitz): a 1x1 raster covering `coverage` of the page plus a Latin text layer."""
    content = f'q 200 0 0 {200*coverage:g} 0 0 cm /Im1 Do Q BT /F1 8 Tf 5 190 Td ({text}) Tj ET'.encode()
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> '
        b'/XObject << /Im1 6 0 R >> >> /Contents 5 0 R >>',
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        b'<< /Length '+str(len(content)).encode()+b' >>\nstream\n'+content+b'\nendstream',
        b'<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /ColorSpace /DeviceGray /BitsPerComponent 8 '
        b'/Length 1 >>\nstream\n\xc8\nendstream']
    payload = b'%PDF-1.4\n'; offsets = []
    for number, value in enumerate(objects, 1):
        offsets.append(len(payload)); payload += str(number).encode()+b' 0 obj\n'+value+b'\nendobj\n'
    xref = len(payload)
    payload += f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode()
    payload += b''.join(f'{offset:010} 00000 n \n'.encode() for offset in offsets)
    payload += f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    path.write_bytes(payload)
    return path


def _signature_replay(record):
    """Archived-sidecar replay protocol: forward every page_rule_selection parameter the row records."""
    import inspect
    names = [n for n in inspect.signature(page_rule_selection).parameters if n != 'decision']
    return page_rule_selection(record['decision'], **{n: record[n] for n in names if n in record})


@pytest.mark.parametrize('settings', [
    {'image_dominant_ratio': .7},
    {'image_dominant_ratio': .7, 'min_valid_word_rate': .6, 'image_dominant_min_chars': 900},
])
def test_nondefault_thresholds_round_trip_through_the_page_sidecar(tmp_path, settings):
    import json
    source = _image_text_pdf(tmp_path/'threshold.pdf')
    default = page_quality(source).pages[0]
    report = page_quality(source, output_dir=tmp_path, **settings)
    page = report.pages[0]
    assert page.image_area_ratio == pytest.approx(.6) and default.page_needs_ocr != page.page_needs_ocr
    saved = json.loads((tmp_path/'json/metrics/threshold.page_quality.json').read_text())
    assert _signature_replay(saved['pages'][0]) is page.page_needs_ocr is False, 'the sidecar omits an effective threshold'
    for record in (saved, saved['pages'][0]):
        assert {k: record[k] for k in settings} == settings
        assert {'min_chars', 'min_image_ratio', 'min_greek_share', 'max_other_share'} <= set(record)
    row = saved['pages'][0]
    assert _signature_replay(row) is page.page_needs_ocr is False, 'the sidecar omits an effective threshold'
    from glossapi.page_quality import replay_page_selection
    assert replay_page_selection(row) is replay_page_selection(page) is page.page_needs_ocr


def test_nondefault_image_threshold_is_replayable_from_page(tmp_path):
    """Reviewer probe (F5): real Greek text layer over a 0.6-coverage raster, inspected at 0.7."""
    from dataclasses import asdict
    font = next((p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                            Path('/System/Library/Fonts/Supplemental/Arial.ttf')] if p.exists()), None)
    if font is None:
        pytest.skip('No Greek fixture font')
    source = tmp_path/'threshold.pdf'
    d = FixturePdf()
    p = d.new_page(400, 500)
    d.insert_image(p, (0, 0, 400, 300), (20, 20, (200, 200, 200)))  # underlay, keep_proportion=False
    d.insert_text(p, (20, 30), '\n'.join(['και άνθρωπος']*20), fontsize=10, font=d.load_truetype(font))
    d.save(source)
    page = page_quality(source, image_dominant_ratio=.7).pages[0]
    record = asdict(page)
    assert record['image_dominant_ratio'] == .7
    assert _signature_replay(record) == page.page_needs_ocr, 'The sidecar omits the effective image coverage threshold'
