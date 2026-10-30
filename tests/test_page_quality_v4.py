"""Measured scan-layer quality, unchanged legacy profiles, and source identity."""
import hashlib
from pathlib import Path

import pytest

from glossapi.page_quality import page_ocr_decision, page_quality, page_rule_selection
from test_page_quality import COMMON_GREEK


BASE = dict(chars=300, greek_chars=200, latin_chars=10, other_chars=0,
            valid_greek_word_rate=None, image_area_ratio=1, words=60)


@pytest.fixture
def scanner_layer_pdf(tmp_path):
    """Independent scan plus text; does not use the legacy corrupt-CMap fixture."""
    fitz = pytest.importorskip('fitz')
    font = next((p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                            Path('/System/Library/Fonts/Supplemental/Arial.ttf')]
                 if p.exists()), None)
    if font is None:
        pytest.skip('No Greek fixture font')
    target = tmp_path / 'scanner_layer.pdf'
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_font(fontname='Greek', fontfile=str(font))
        words = sorted(COMMON_GREEK)
        text = '\n'.join(' '.join(words[i:i+6]) for i in range(0,len(words),6))
        page.insert_text((72,72), text, fontname='Greek', fontsize=14)
        image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0,0,20,20), False)
        image.clear_with(200)
        page.insert_image(page.rect, stream=image.tobytes('png'), overlay=False)
        doc.save(target)
    return target


@pytest.mark.parametrize("changes,expected", [
    ({}, (True, "image_dominant_unknown_validity")),
    ({"chars": 2000}, (True, "image_dominant_unknown_validity")),
    ({"image_area_ratio": .5}, (True, "image_dominant_unknown_validity")),
    ({"valid_greek_word_rate": .399}, (True, "image_dominant_bad_text")),
    ({"valid_greek_word_rate": .4}, (False, "text_layer")),
    ({"valid_greek_word_rate": 1, "chars": 49}, (True, "scanned_page")),
    ({"valid_greek_word_rate": .4, "chars": 50}, (False, "text_layer")),
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
            'image_area_ratio', 'min_valid_word_rate'}}) == expected[0]


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
    fitz = pytest.importorskip('fitz')
    target = tmp_path / 'non_image.pdf'
    font = next(p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                           Path('/System/Library/Fonts/Supplemental/Arial.ttf')] if p.exists())
    with fitz.open(scanner_layer_pdf) as source, fitz.open() as doc:
        page = doc.new_page()
        page.insert_font(fontname='Greek', fontfile=str(font))
        page.insert_text((72,72), source[0].get_text(), fontname='Greek', fontsize=14)
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
