"""Experimental European vocabularies, script/length guards and replay."""
from dataclasses import asdict
import importlib

import pytest

from glossapi.latin_validity import measure_latin_validity, vocabulary_key
from glossapi.page_quality import _latin_words, page_quality, replay_page_selection
from pdf_fixtures import FixturePdf


SENTENCES = {
    'en': 'The structure of the human body is described in this chapter of the book',
    'fr': 'La langue française est une langue qui possède une longue histoire et une littérature très riche',
    'de': 'Die deutsche Sprache ist eine Sprache mit einer langen Geschichte und einer reichen Literatur',
    'it': 'La lingua italiana è una lingua con una lunga storia e una ricca letteratura',
    'pt': 'A língua portuguesa é uma língua com uma longa história e uma literatura muito rica',
    'es': 'La lengua española es una lengua con una larga historia y una literatura muy rica',
}


@pytest.mark.parametrize('language', SENTENCES)
def test_dictionary_selection_and_accented_surface_entries(language):
    measured = measure_latin_validity(SENTENCES[language], _latin_words())
    assert measured.language == language
    assert measured.rate >= 0.4
    assert measured.rate == max(measured.rates.values())
    assert measured.tokens >= 10


def _layer(tmp_path, text, *, repeat=1):
    pdf = FixturePdf()
    page = pdf.new_page()
    for i in range(repeat):
        pdf.insert_text(page, (25, 35 + 25*i), text, fontsize=8)
    return pdf.save(tmp_path/'layer.pdf')


@pytest.mark.parametrize('language', SENTENCES)
def test_v6_clean_languages_keep_text_and_record_replay(tmp_path, language):
    path = _layer(tmp_path, SENTENCES[language], repeat=12)
    report = page_quality(path, rule_profile='v6')
    page = report.pages[0]
    assert not page.page_needs_ocr
    assert page.latin_validity_applies
    assert page.latin_language == language
    assert not replay_page_selection(asdict(page))
    assert set(report.latin_dictionary_identity['dictionaries']) == set(SENTENCES)
    assert page_quality(path).rule_profile == 'v4'


def test_v6_garbage_routes_without_a_language_model(tmp_path):
    path = _layer(tmp_path, 'xqzvkj pxxdqn zrjkwv qzvpxn lkqzwv jxxvqr', repeat=12)
    old = page_quality(path)
    new = page_quality(path, rule_profile='v6')
    assert not old.pages[0].page_needs_ocr
    assert new.pages[0].decision == 'bad_latin_text'
    assert replay_page_selection(new.pages[0])
    assert new.pages[0].valid_latin_word_rate == 0


@pytest.mark.parametrize('text', [
    'xqzvkj pxxdqn',
    'xqzvkjpxxdqnzrjkwvqzvpxnlkqzwvjxxvqr xqzvkjpxxdqnzrjkwvqzvpxnlkqzwvjxxvqr',
    'I am in it as we go to do so he is on an at by or if no up us',
])
def test_short_pages_and_real_short_words_are_not_added(tmp_path, text):
    page = page_quality(_layer(tmp_path, text), rule_profile='v6').pages[0]
    assert not page.page_needs_ocr
    assert not replay_page_selection(page)


def test_empty_and_non_latin_text_have_unknown_latin_measurement():
    measured = measure_latin_validity('Ελληνικό κείμενο 123', _latin_words())
    assert measured.tokens == 0 and measured.rate is None and measured.language is None
    assert vocabulary_key('größer', 'de') == 'groesser'
    assert vocabulary_key('ÉTUDE', 'fr') == 'etude'
    assert vocabulary_key('española', 'es') == 'espanola'


def test_mixed_script_guard_does_not_add_a_route(tmp_path, monkeypatch):
    from test_k5a2b_routing_metrics import _greek_font
    pdf = FixturePdf()
    font = pdf.load_truetype(_greek_font())
    p = pdf.new_page()
    for i in range(8):
        pdf.insert_text(p, (20,35+25*i), 'Αυτή είναι μία ελληνική παράγραφος xqzvkj pxxdqn zrjkwv', font=font, fontsize=9)
    path = pdf.save(tmp_path/'mixed.pdf')
    old = page_quality(path).pages[0]
    new = page_quality(path, rule_profile='v6').pages[0]
    assert not new.latin_validity_applies
    assert new.page_needs_ocr == old.page_needs_ocr
    assert replay_page_selection(new) == old.page_needs_ocr


@pytest.mark.parametrize('language', ['fr', 'de', 'it', 'es', 'pt'])
def test_bundled_wordlists_verify_integrity(language, monkeypatch):
    validity = importlib.import_module('glossapi.latin_validity')
    validity._words.cache_clear()
    assert len(validity._words(language)) > (20000 if language == "pt" else 60000)
    identities = {k: dict(v) for k, v in validity.vocabulary_identities().items()}
    identities[language]['sha256'] = '0'*64
    monkeypatch.setattr(validity, 'vocabulary_identities', lambda: identities)
    validity._words.cache_clear()
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        validity._words(language)
    validity._words.cache_clear()
