"""Opt-in validity folding, default v3 selection, and routing provenance."""

import json
import subprocess
import sys
import unicodedata

import pandas as pd
import pytest

from glossapi.page_quality import (
    _page_text_stats, fold_validity_word, page_ocr_decision,
    page_quality, page_rule_selection,
)
from glossapi.pdf_repair.tounicode import _text_stats
from test_page_quality import COMMON_GREEK, quality_pdf


@pytest.mark.parametrize("word,expected", [
    ("ΆΝΘΡΩΠΟΣ", "ανθρωποσ"), ("άνθρωπος", "ανθρωποσ"),
    ("α\u0301νθρωπος", "ανθρωποσ"), ("ΜΑΪΟΣ", "μαιοσ"),
    ("ἄνθρωπος", "ανθρωποσ"), ("ᾳ", "α"),
])
def test_fold_order_and_spellings(word, expected):
    assert fold_validity_word(word) == expected


def test_folded_lookup_preserves_original_tokens_and_counts():
    text = "ΆΝΘΡΩΠΟΣ άνθρωπος α\u0301νθρωπος ἄνθρωπος ΜΑΪΟΣ XYZ 123"
    raw = frozenset({"ανθρωποσ", "μαιοσ"})
    old = _text_stats(text, raw)
    assert _page_text_stats(text, raw) == old
    folded = _page_text_stats(text, raw, True)
    assert old["dictionary_valid_rate"] == 0
    assert folded["dictionary_valid_rate"] == 1
    for key in ("chars", "greek_chars", "latin_chars", "words", "greek_words", "greek_share"):
        assert folded[key] == old[key]
    assert folded["words"] == 6 and folded["greek_words"] == 5


@pytest.mark.parametrize("dictionary,text", [(None, "ΕΛΛΑΔΑ"), (frozenset(), "English only"), (frozenset(), "")])
def test_folded_unknown_is_null(dictionary, text):
    assert _page_text_stats(text, dictionary, True)["dictionary_valid_rate"] is None


@pytest.mark.parametrize("source", ["set", "frozenset", "path", "environment"])
def test_opt_in_lookup_sources_and_sidecar(quality_pdf, tmp_path, monkeypatch, source):
    # Dictionary capitals/polytonic marks and page tonos must fold on both sides.
    words = {unicodedata.normalize("NFD", w.upper()) for w in COMMON_GREEK}
    words = {"".join(c for c in w if not unicodedata.combining(c)) for w in words}
    path = tmp_path / "words.txt"
    path.write_text("\n".join(words), encoding="utf-8")
    wordlist = words if source == "set" else frozenset(words) if source == "frozenset" else path
    if source == "environment":
        monkeypatch.setenv("GLOSSAPI_GREEK_WORDLIST", str(path))
        wordlist = None
    default = page_quality(quality_pdf, wordlist=wordlist)
    explicit = page_quality(quality_pdf, wordlist=wordlist, fold_validity=False, rule_profile="v3")
    assert default.pages == explicit.pages
    assert default.fold_validity is False and default.rule_profile == "v3"
    assert default.pages[0].valid_greek_word_rate < 1
    folded = page_quality(quality_pdf, wordlist=wordlist, fold_validity=True,
                          rule_profile="v3", output_dir=tmp_path)
    assert folded.pages[0].valid_greek_word_rate == 1
    payload = json.loads((tmp_path / "json/metrics/quality.page_quality.json").read_text())
    assert payload["fold_validity"] is True and payload["rule_profile"] == "v3"


@pytest.mark.parametrize("decision,rate,words,pua,control,expected", [
    ("scanned_page", None, 0, 0, 0, True),
    ("garbage_greek_text", .499, 11, 0, 0, True),
    ("garbage_greek_text", .5, 11, 0, 0, False),
    ("garbage_greek_text", 1, 10, 0, 0, True),
    ("symbol_font_garbage", .1, 100, 0, 0, True),
    ("symbol_font_garbage", 1, 100, 0, 0, False),
    ("symbol_font_garbage", None, 11, 0, 0, False),
    ("symbol_font_garbage", None, 10, 0, 0, True),
    ("image_dominant_sparse_text", 0, 0, 0, 0, False),
    ("image_dominant_sparse_text", 1, 100, 1, 0, True),
    ("text_layer", 1, 100, 0, 1, True),
    ("blank_or_figure", None, 0, 1, 0, True),
    ("text_layer", 0, 1, 0, 0, False),
])
def test_profile_selection(decision, rate, words, pua, control, expected):
    values = dict(valid_greek_word_rate=rate, words=words, pua_char_count=pua, control_char_count=control)
    assert page_rule_selection(decision, rule_profile="v3", **values) is expected
    assert page_rule_selection(decision, rule_profile="v2", **values) is (decision in {
        "scanned_page", "garbage_greek_text", "symbol_font_garbage", "image_dominant_sparse_text"})


@pytest.mark.parametrize("changes,expected", [
    ({"image_area_ratio": 1, "chars": 49}, (True, "scanned_page")),
    ({"image_area_ratio": 1, "chars": 100, "valid_greek_word_rate": 0}, (False, "image_dominant_sparse_text")),
    ({"other_chars": 200}, (False, "symbol_font_garbage")),
    ({"other_chars": 200, "words": 10}, (True, "symbol_font_garbage")),
    ({"other_chars": 200, "valid_greek_word_rate": .49}, (True, "symbol_font_garbage")),
    ({"other_chars": 200, "valid_greek_word_rate": .5}, (False, "symbol_font_garbage")),
    ({"valid_greek_word_rate": .39}, (True, "garbage_greek_text")),
    ({"chars": 20, "pua_char_count": 1}, (True, "symbol_font_garbage")),
    ({"control_char_count": 1}, (True, "symbol_font_garbage")),
])
def test_v3_decision_integration(changes, expected):
    base = dict(chars=600, greek_chars=400, latin_chars=0, other_chars=0,
                valid_greek_word_rate=1, image_area_ratio=0, words=100)
    assert page_ocr_decision(**(base | changes), rule_profile="v3") == expected


def test_invalid_profile_rejected_before_reading(tmp_path):
    with pytest.raises(ValueError, match="rule_profile"):
        page_quality(tmp_path / "absent.pdf", rule_profile="typo")
    with pytest.raises(ValueError, match="words"):
        page_ocr_decision(chars=0, greek_chars=0, latin_chars=0, other_chars=0,
                          valid_greek_word_rate=None, image_area_ratio=0, rule_profile="v3")


def test_v3_folding_filters_legacy_candidates_at_0_5(quality_pdf):
    selected = sorted(COMMON_GREEK)[:len(COMMON_GREEK) // 2 - 1]
    dictionary = {w[0] + "\u0300" + w[1:] for w in selected}
    legacy = page_quality(quality_pdf, wordlist=dictionary)
    assert legacy.pages[0].decision == "garbage_greek_text"
    folded_v2 = page_quality(quality_pdf, wordlist=dictionary, fold_validity=True, rule_profile="v2")
    assert .4 <= folded_v2.pages[0].valid_greek_word_rate < .5
    assert not folded_v2.pages[0].page_needs_ocr
    folded_v3 = page_quality(quality_pdf, wordlist=dictionary, fold_validity=True, rule_profile="v3")
    assert folded_v3.pages[0].decision == "garbage_greek_text"
    assert folded_v3.pages[0].page_needs_ocr
    assert folded_v3.pages[0].valid_greek_word_rate == folded_v2.pages[0].valid_greek_word_rate


def test_v3_candidate_short_word_override():
    values = dict(chars=100, greek_chars=80, latin_chars=0, other_chars=0,
                  image_area_ratio=0, valid_greek_word_rate=1, words=10)
    assert page_ocr_decision(**values, rule_profile="v3",
                             candidate_decision="garbage_greek_text") == (True, "garbage_greek_text")


def test_cli_and_parquet_profile_provenance(quality_pdf, tmp_path):
    from glossapi.corpus.phase_clean import _attach_page_quality_routes
    target = tmp_path / "report.jsonl"
    result = subprocess.run([sys.executable, "-m", "glossapi.scripts.page_quality_batch",
        "--pdfs", str(quality_pdf), "--out", str(target), "--output-dir", str(tmp_path),
        "--fold-validity", "--rule-profile", "v3"], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    record = json.loads(target.read_text())
    assert record["fold_validity"] is True and record["rule_profile"] == "v3"
    assert record["dictionary_loaded"] is False
    frame = pd.DataFrame([dict(filename="quality.pdf", needs_ocr=False),
                          dict(filename="missing.pdf", needs_ocr=True)])
    routed = _attach_page_quality_routes(frame, tmp_path / "json/metrics")
    parquet = tmp_path / "routes.parquet"
    routed.to_parquet(parquet, index=False)
    restored = pd.read_parquet(parquet)
    assert restored.loc[0, "page_rule_profile"] == "v3"
    assert restored.loc[0, "page_fold_validity"]
    assert pd.isna(restored.loc[1, "page_rule_profile"])
    assert pd.isna(restored.loc[1, "page_fold_validity"])
    # Legacy sidecars are v2/unfolded; deleted or invalid sidecars clear stale provenance.
    sidecar = tmp_path / "json/metrics/quality.page_quality.json"
    record.pop("fold_validity")
    record.pop("rule_profile")
    sidecar.write_text(json.dumps(record))
    legacy = _attach_page_quality_routes(routed, sidecar.parent)
    assert legacy.loc[0, "page_rule_profile"] == "v2"
    assert not legacy.loc[0, "page_fold_validity"]
    record["rule_profile"] = "typo"
    sidecar.write_text(json.dumps(record))
    invalid = _attach_page_quality_routes(routed, sidecar.parent)
    assert pd.isna(invalid.loc[0, "page_rule_profile"]) and invalid.loc[0, "ocr_pages"] == ""
    sidecar.unlink()
    assert _attach_page_quality_routes(routed, sidecar.parent).page_rule_profile.isna().all()
