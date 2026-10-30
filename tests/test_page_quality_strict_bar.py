"""Strict Greek scan bar, measured junk, per-page settings and CLI wiring."""
import json
import subprocess
import sys
import unicodedata

import pytest

from glossapi.page_quality import (
    fold_validity_word, junk_token_stats, page_ocr_decision, page_quality,
    page_rule_selection,
)
from test_page_quality import COMMON_GREEK
from test_page_quality_v4 import BASE, LATIN, scanner_layer_pdf


@pytest.mark.parametrize('rate,junk,selected', [
    (.899999, 0, True), (.9, .099999, False), (.9, .1, True),
    (.9, .100001, True), (1, 0, False), (1, None, True),
    (1, float('nan'), True), (float('nan'), 0, True),
    (.78, 0, True), (.5, .2, True),
])
def test_strict_bar_and_sidecar_replay(rate, junk, selected):
    values = BASE | dict(valid_greek_word_rate=rate, junk_token_share=junk)
    assert page_ocr_decision(**values) == (selected, 'image_dominant_bad_text' if selected else 'text_layer')
    replay = {k: values[k] for k in ('valid_greek_word_rate', 'words', 'greek_chars',
              'latin_chars', 'image_area_ratio', 'junk_token_share')}
    assert page_rule_selection('text_layer', **replay) == selected


def test_configurable_bar_is_separate_from_non_image_validity():
    values = BASE | dict(valid_greek_word_rate=.8, junk_token_share=.12)
    assert page_ocr_decision(**values)[0]
    assert page_ocr_decision(**values, image_dominant_min_valid_word_rate=.8,
                             image_dominant_max_junk_share=.13) == (False, 'text_layer')
    assert page_ocr_decision(**values, min_valid_word_rate=0)[0]
    assert page_ocr_decision(**(values | {'image_area_ratio': .499}),
                             image_dominant_min_valid_word_rate=1,
                             image_dominant_max_junk_share=0) == (False, 'text_layer')
    assert page_ocr_decision(**LATIN, image_dominant_min_valid_word_rate=1,
                             image_dominant_max_junk_share=0) == (False, 'text_layer')


@pytest.mark.parametrize('key,value', [
    ('image_dominant_min_valid_word_rate', -.01),
    ('image_dominant_min_valid_word_rate', 1.01),
    ('image_dominant_max_junk_share', float('nan')),
    ('image_dominant_max_junk_share', float('inf')),
])
def test_invalid_bar_fails_before_reading_pdf(key, value, tmp_path):
    with pytest.raises(ValueError, match='thresholds'):
        page_quality(tmp_path/'absent.pdf', **{key: value})


def test_junk_tokens_union_digit_intrusions_and_script_mixing():
    stats = junk_token_stats('και άνθρωπος αβ aa JVeg 2Vcg ελλ2άδα ελλaδα λοgoς 123 2026',
                             frozenset({'και', 'άνθρωποσ'}))
    assert stats['junk_token_total'] == 9  # Pure numbers are not words.
    assert stats['junk_token_count'] == 7
    assert stats['junk_token_share'] == pytest.approx(7/9)
    assert stats['junk_token_categories'] == dict(short_non_dictionary=3,
        latin_fragment=3, mixed_script=2, digit_fragment=2)


def test_junk_normalization_unknown_and_polytonic_marks():
    text = 'ᾧ ἄνθρωπος και'
    dictionary = frozenset(unicodedata.normalize('NFC', w).casefold() for w in text.split())
    assert junk_token_stats(unicodedata.normalize('NFD', text), dictionary)['junk_token_share'] == 0
    folded = frozenset(fold_validity_word(w) for w in text.split())
    assert junk_token_stats(text, folded, True)['junk_token_share'] == 0
    assert junk_token_stats(text, None)['junk_token_share'] is None
    assert junk_token_stats('123', dictionary)['junk_token_share'] is None


def test_real_pdf_records_effective_bar_and_measured_counts(scanner_layer_pdf, tmp_path):
    report = page_quality(scanner_layer_pdf, wordlist=COMMON_GREEK,
        image_dominant_min_valid_word_rate=.95, image_dominant_max_junk_share=.02,
        output_dir=tmp_path/'out')
    page = report.pages[0]
    assert page.greek_image_bar_applies and not page.page_needs_ocr
    assert page.valid_greek_word_rate == 1 and page.junk_token_share == 0
    assert page.junk_token_total > 20 and page.junk_token_count == 0
    assert page.image_dominant_min_valid_word_rate == .95
    assert page.image_dominant_max_junk_share == .02
    sidecar = tmp_path/'out/json/metrics/scanner_layer.page_quality.json'
    assert json.loads(sidecar.read_text()) == report.to_dict()
    assert page_quality(scanner_layer_pdf, wordlist=COMMON_GREEK,
                        image_dominant_max_junk_share=0).pages[0].page_needs_ocr


def test_batch_cli_accepts_and_records_bar(scanner_layer_pdf, tmp_path):
    wordlist = tmp_path/'words.txt'; wordlist.write_text('\n'.join(COMMON_GREEK))
    target = tmp_path/'pages.jsonl'
    result = subprocess.run([sys.executable, '-m', 'glossapi.scripts.page_quality_batch',
        '--pdfs', str(scanner_layer_pdf), '--wordlist', str(wordlist), '--out', str(target),
        '--image-dominant-min-valid-word-rate', '.95',
        '--image-dominant-max-junk-share', '0', '--output-dir', str(tmp_path/'sidecars')],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    row = json.loads(target.read_text()); page = row['pages'][0]
    assert row['image_dominant_min_valid_word_rate'] == .95
    assert page['image_dominant_max_junk_share'] == 0
    assert page['page_needs_ocr'] and page['decision'] == 'image_dominant_bad_text'
