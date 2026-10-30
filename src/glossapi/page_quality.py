"""Text-layer page routing; PDFium objects are inspected without rendering.

Counts and dictionary normalization share E1's helpers. ``other_chars`` counts
non-whitespace characters other than Greek/Latin letters (including symbols,
digits and punctuation); ``chars`` includes whitespace, as in E1. A missing
dictionary rate is unknown. Default v4 requires measured good text on scans.
"""

from __future__ import annotations

from contextlib import closing

from dataclasses import asdict, dataclass
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from typing import Iterable

import pypdfium2 as pdfium

from glossapi.pdf_repair.text_stats import _is_greek, _load_words, _text_stats


@dataclass(frozen=True)
class PageQuality:
    """One immutable row, with a 1-based source page number and nullable rate."""

    page: int
    chars: int
    greek_chars: int
    latin_chars: int
    other_chars: int
    words: int
    valid_greek_word_rate: float | None
    image_area_ratio: float
    page_needs_ocr: bool
    decision: str
    min_image_dpi: float | None = None
    pua_char_count: int = 0
    control_char_count: int = 0
    rule_profile: str = "v4"
    validity_not_applicable: bool = False
    junk_token_count: int = 0
    junk_token_total: int = 0
    junk_token_share: float | None = None
    junk_token_categories: dict | None = None
    image_dominant_min_valid_word_rate: float = 0.9
    image_dominant_max_junk_share: float = 0.1
    greek_image_bar_applies: bool = False


@dataclass(frozen=True)
class PageQualityReport:
    """Summary counts and ranges describe exactly the rows in ``pages``."""

    path: str
    filename: str
    pages: list[PageQuality]
    pages_total: int
    pages_needing_ocr: int
    ocr_page_list: str
    doc_needs_ocr_by_page_rule: bool
    share_pages_needing_ocr: float
    dictionary_loaded: bool
    elapsed_seconds: float
    fold_validity: bool = False
    rule_profile: str = "v4"
    dictionary_identity: dict | None = None
    image_dominant_min_valid_word_rate: float = 0.9
    image_dominant_max_junk_share: float = 0.1

    def to_dict(self) -> dict:
        """Return a JSON-compatible copy, retaining null unknown rates."""
        return asdict(self)


_WORDLIST_DIR = Path(__file__).with_name("data") / "wordlists"
_WORDLIST_SHA256 = "f08daefb302600beb1b345e4fd77f4ecf6617aa080a72efe6ae7eec0ad5b2ac7"
_WORDLIST_REVISION = "8cfea406b505e4d7df52d5a19bce525df98c54ab"


def default_wordlist_identity() -> dict:
    """Identity of the unmodified, compressed Hunspell surface-entry source."""
    return {"name": "el_GR 0.9", "revision": _WORDLIST_REVISION,
            "sha256": _WORDLIST_SHA256,
            "source": f"https://raw.githubusercontent.com/wooorm/dictionaries/{_WORDLIST_REVISION}/dictionaries/el/index.dic",
            "license": "MPL-1.1 OR GPL-2.0-or-later OR LGPL-2.1-or-later",
            "lookup": "Hunspell entries without affix expansion"}


@lru_cache(maxsize=1)
def _default_words() -> frozenset[str]:
    """Verify the packaged source, discard count/flags, and preserve accents."""
    data = gzip.decompress((_WORDLIST_DIR / "el_GR.dic.gz").read_bytes())
    if hashlib.sha256(data).hexdigest() != _WORDLIST_SHA256:
        raise ValueError("Packaged Greek wordlist SHA256 mismatch")
    lines = data.decode("utf-8").splitlines()
    if int(lines[0]) != len(lines) - 1:
        raise ValueError("Packaged Greek wordlist entry count mismatch")
    return frozenset(unicodedata.normalize("NFC", line.split("/", 1)[0]).casefold()
                     for line in lines[1:] if line)


def fold_validity_word(word: str) -> str:
    """Fold dictionary keys: NFD, remove marks, lowercase, final sigma to sigma."""
    return "".join(c for c in unicodedata.normalize("NFD", word)
                   if not unicodedata.category(c).startswith("M")).lower().replace("ς", "σ")


@lru_cache(maxsize=4)
def _load_folded_words(path: str) -> frozenset[str]:
    # Read original spellings: casefold can expand a Greek iota subscript before
    # NFD has the opportunity to strip it, unlike the requested folding order.
    with open(path, encoding="utf-8") as handle:
        return frozenset(fold_validity_word(w.strip()) for w in handle if w.strip())


def _page_text_stats(text: str, dictionary: frozenset[str] | None,
                     fold_validity: bool = False) -> dict:
    """Keep E1's original token denominator and counts in both lookup modes.

    In folded mode the caller supplies a folded dictionary. Only the validity
    lookup changes; accents, case and marks are preserved in all other metrics.
    """
    stats = _text_stats(text, None if fold_validity else dictionary)
    if fold_validity and dictionary is not None:
        tokens = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text), flags=re.UNICODE)
        greek = [token for token in tokens if any(_is_greek(c) for c in token)]
        valid = sum(fold_validity_word(token) in dictionary for token in greek)
        stats["dictionary_valid_words"] = valid
        stats["dictionary_valid_rate"] = valid / len(greek) if greek else None
    return stats


def junk_token_stats(text: str, dictionary: frozenset[str] | None,
                     fold_validity: bool = False) -> dict:
    """Measure junk once per letter-bearing alphanumeric run (not page numbers).

    A token is junk if it mixes letter scripts, mixes Greek/Latin letters with
    digits, is an unknown word of <=3 letters, or an unknown Latin fragment of
    <=4 letters. Categories may overlap; the numerator is their union. Lookup
    uses the validity dictionary and normalization. No dictionary/no tokens is
    unknown, never a measured zero. NFC plus attached marks preserves polytonic
    tokens; the legacy validity denominator is unchanged.
    """
    tokens = re.findall(r'[^\W_]+(?:[\u0300-\u036f]+[^\W_]*)*',
                        unicodedata.normalize('NFC', text))
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    counts = dict(short_non_dictionary=0, latin_fragment=0,
                  mixed_script=0, digit_fragment=0)
    junk = 0
    for token in tokens:
        letters = [c for c in token if c.isalpha()]
        greek = any(_is_greek(c) for c in letters)
        scripts = {unicodedata.name(c, '').split(' ')[0] for c in letters}
        latin = 'LATIN' in scripts
        key = fold_validity_word(token) if fold_validity else token.casefold()
        unknown = dictionary is not None and key not in dictionary
        flags = (len(letters) <= 3 and unknown,
                 latin and not greek and len(letters) <= 4 and unknown,
                 len(scripts) > 1,
                 any(c.isdigit() for c in token) and (greek or latin))
        for name, flag in zip(counts, flags):
            counts[name] += flag
        junk += any(flags)
    return dict(junk_token_count=junk, junk_token_total=len(tokens),
                junk_token_share=junk / len(tokens) if tokens and dictionary is not None else None,
                junk_token_categories=counts)


def _validate_image_text_bar(validity: float, junk: float) -> None:
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (validity, junk)):
        raise ValueError('image-dominant validity and junk thresholds must be finite rates in [0, 1]')


def page_rule_selection(decision: str, *, valid_greek_word_rate: float | None,
                        words: int, pua_char_count: int = 0,
                        control_char_count: int = 0,
                        rule_profile: str = "v4",
                        image_area_ratio: float | None = None,
                        image_dominant_ratio: float = 0.5,
                        min_valid_word_rate: float = 0.4,
                        validity_not_applicable: bool = False,
                        greek_chars: int = 0, latin_chars: int = 0,
                        min_greek_share: float = 0.5,
                        junk_token_share: float | None = None,
                        image_dominant_min_valid_word_rate: float = 0.9,
                        image_dominant_max_junk_share: float = 0.1) -> bool:
    """Select existing decision labels; useful for archived-sidecar replay too.

    v4 additionally requires image_area_ratio, and trusts image-dominant text
    only with measured validity and no contamination. Greek-dominant image
    layers require validity >= image_dominant_min_valid_word_rate AND junk
    share < image_dominant_max_junk_share. Unknown junk cannot certify a layer.
    A recorded validity_not_applicable exception replays the v3 selection.
    v3 keeps scans and any PUA/control contamination. Garbage candidates need
    known validity < 0.5 or at most 10 words. Unknown validity is not zero.
    Image-dominant sparse text alone is excluded. Validity uses the report's
    explicit lookup mode; choosing v3 does not implicitly enable folding.
    """
    if rule_profile not in ("v2", "v3", "v4"):
        raise ValueError("rule_profile must be 'v2', 'v3' or 'v4'")
    if rule_profile == "v4":
        _validate_image_text_bar(image_dominant_min_valid_word_rate, image_dominant_max_junk_share)
        if image_area_ratio is None:
            raise ValueError("image_area_ratio is required for rule_profile='v4'")
        if image_area_ratio >= image_dominant_ratio and not validity_not_applicable:
            letters = greek_chars + latin_chars
            if letters and greek_chars / letters > min_greek_share:
                return (decision == "scanned_page" or valid_greek_word_rate is None
                        or not valid_greek_word_rate >= image_dominant_min_valid_word_rate
                        or junk_token_share is None
                        or not junk_token_share < image_dominant_max_junk_share
                        or pua_char_count > 0 or control_char_count > 0)
            return (decision == "scanned_page" or valid_greek_word_rate is None
                    or not valid_greek_word_rate >= min_valid_word_rate
                    or pua_char_count > 0 or control_char_count > 0)
    if rule_profile == "v2":
        return decision in {"scanned_page", "garbage_greek_text",
                            "symbol_font_garbage", "image_dominant_sparse_text"}
    return (decision == "scanned_page" or pua_char_count > 0 or control_char_count > 0
            or (decision in {"garbage_greek_text", "symbol_font_garbage"}
                and (words <= 10 or (valid_greek_word_rate is not None
                                    and valid_greek_word_rate < 0.5))))


def text_contamination_counts(text: str, *, all_pua: bool = False) -> dict[str, int]:
    """Count PUA and C0/C1 controls except TAB/LF/CR; legacy uses BMP only."""
    return {
        "pua_char_count": len(re.findall(
            r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]"
            if all_pua else r"[\ue000-\uf8ff]", text)),
        "control_char_count": len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", text)),
    }


def _validity_not_applicable(*, chars: int, greek_chars: int, latin_chars: int,
                             other_chars: int, image_area_ratio: float,
                             pua_char_count: int, control_char_count: int,
                             min_chars: int = 50, min_greek_share: float = 0.5,
                             max_other_share: float = 0.3,
                             image_dominant_ratio: float = 0.5,
                             image_dominant_min_chars: int = 600) -> bool:
    """Greek validity cannot qualify a sufficiently dense, clean Latin layer."""
    letters = greek_chars + latin_chars
    return (image_area_ratio >= image_dominant_ratio
            and chars > 0 and chars >= max(min_chars, image_dominant_min_chars)
            and latin_chars > 0 and letters > 0
            and greek_chars / letters <= min_greek_share
            and other_chars / chars <= max_other_share
            and pua_char_count == 0 and control_char_count == 0)


def page_ocr_decision(*, chars: int, greek_chars: int, latin_chars: int,
                      other_chars: int, valid_greek_word_rate: float | None,
                      image_area_ratio: float, pua_char_count: int = 0,
                      control_char_count: int = 0, min_chars: int = 50,
                      min_image_ratio: float = 0.3, min_greek_share: float = 0.5,
                      min_valid_word_rate: float = 0.4,
                      max_other_share: float = 0.3,
                      image_dominant_ratio: float = 0.5,
                      image_dominant_min_chars: int = 600,
                      words: int | None = None,
                      rule_profile: str = "v4",
                      candidate_decision: str | None = None,
                      junk_token_share: float | None = None,
                      image_dominant_min_valid_word_rate: float = 0.9,
                      image_dominant_max_junk_share: float = 0.1) -> tuple[bool, str]:
    """Classify pages with v4, or unchanged selectable legacy v2/v3 routing.

    v4 trusts an image-dominant text layer only with known validity at least
    the configured image validity bar and junk below its configured maximum
    for Greek-dominant pages, and no PUA/control contamination. Density does not
    establish quality. The existing chars < min_chars scan gate is retained:
    a valid short caption does not certify the raster body. Below the image
    threshold it delegates to v3. Clean Latin-majority layers with at least
    image_dominant_min_chars characters also retain v3's decision: Greek
    validity is not applicable. Sparse captions cannot use this exception.

    Legacy v2 labels image coverage >= image_dominant_ratio with
    chars < image_dominant_min_chars as an OCR candidate: a footer/caption does not establish that a raster body is present
    in the text layer. Defaults are 0.5 and 600; coverage is inclusive and the
    character bound is strict. This is a conservative heuristic, not a fidelity
    guarantee. A scan also has chars < min_chars and image ratio > min_image_ratio.
    PUA/control dominance above max_other_share also routes sparse text before
    the blank-page gate (normal TAB/LF/CR are excluded). Otherwise dense Greek text with known validity below min_valid_word_rate, or dense
    text with other_chars/chars > max_other_share, needs OCR. Greek share uses
    Greek + Latin letters, not all characters. Unknown validity is not zero.

    v3 filters these labels with page_rule_selection, dropping image-only
    sparse-text candidates and requiring weak validity/short text for garbage
    candidates while retaining scans and any PUA/control. words is required for v3. Rejected candidates retain their diagnostic decision label.
    candidate_decision can supply a legacy label before validity folding; this
    keeps v3 a filter of v2 candidates without a second validity cutoff.
    """
    if rule_profile not in ("v2", "v3", "v4"):
        raise ValueError("rule_profile must be 'v2', 'v3' or 'v4'")
    if rule_profile == "v4":
        _validate_image_text_bar(image_dominant_min_valid_word_rate, image_dominant_max_junk_share)
        if words is None:
            raise ValueError("words is required for rule_profile='v4'")
        if image_area_ratio >= image_dominant_ratio:
            if chars < min_chars:
                return True, "scanned_page"
            if _validity_not_applicable(
                chars=chars, greek_chars=greek_chars, latin_chars=latin_chars,
                other_chars=other_chars, image_area_ratio=image_area_ratio,
                pua_char_count=pua_char_count, control_char_count=control_char_count,
                min_chars=min_chars, min_greek_share=min_greek_share,
                max_other_share=max_other_share,
                image_dominant_ratio=image_dominant_ratio,
                image_dominant_min_chars=image_dominant_min_chars,
            ):
                return page_ocr_decision(
                    chars=chars, greek_chars=greek_chars, latin_chars=latin_chars,
                    other_chars=other_chars, valid_greek_word_rate=valid_greek_word_rate,
                    image_area_ratio=image_area_ratio, words=words,
                    min_chars=min_chars, min_image_ratio=min_image_ratio,
                    min_greek_share=min_greek_share, min_valid_word_rate=min_valid_word_rate,
                    max_other_share=max_other_share, image_dominant_ratio=image_dominant_ratio,
                    image_dominant_min_chars=image_dominant_min_chars,
                    candidate_decision=candidate_decision, rule_profile="v3")
            needs = page_rule_selection(
                "text_layer", valid_greek_word_rate=valid_greek_word_rate,
                words=words, pua_char_count=pua_char_count,
                control_char_count=control_char_count, rule_profile="v4",
                image_area_ratio=image_area_ratio,
                image_dominant_ratio=image_dominant_ratio,
                min_valid_word_rate=min_valid_word_rate,
                greek_chars=greek_chars, latin_chars=latin_chars,
                min_greek_share=min_greek_share, junk_token_share=junk_token_share,
                image_dominant_min_valid_word_rate=image_dominant_min_valid_word_rate,
                image_dominant_max_junk_share=image_dominant_max_junk_share)
            if pua_char_count or control_char_count:
                return needs, "image_dominant_contaminated_text"
            if valid_greek_word_rate is None:
                return needs, "scanned_page" if chars < min_chars else "image_dominant_unknown_validity"
            return needs, "image_dominant_bad_text" if needs else "text_layer"
        rule_profile = "v3"
    if rule_profile == "v3":
        if words is None:
            raise ValueError("words is required for rule_profile='v3'")
        _, decision = page_ocr_decision(
            chars=chars, greek_chars=greek_chars, latin_chars=latin_chars,
            other_chars=other_chars, valid_greek_word_rate=valid_greek_word_rate,
            image_area_ratio=image_area_ratio, pua_char_count=pua_char_count,
            control_char_count=control_char_count, min_chars=min_chars,
            min_image_ratio=min_image_ratio, min_greek_share=min_greek_share,
            min_valid_word_rate=min_valid_word_rate, max_other_share=max_other_share,
            image_dominant_ratio=image_dominant_ratio,
            image_dominant_min_chars=image_dominant_min_chars,
            rule_profile="v2",
        )
        if candidate_decision is not None:
            if candidate_decision not in {"text_layer", "blank_or_figure", "scanned_page",
                                          "garbage_greek_text", "symbol_font_garbage",
                                          "image_dominant_sparse_text"}:
                raise ValueError("Invalid candidate_decision")
            decision = candidate_decision
        needs = page_rule_selection(decision, valid_greek_word_rate=valid_greek_word_rate,
                                    words=words, pua_char_count=pua_char_count,
                                    control_char_count=control_char_count,
                                    rule_profile=rule_profile)
        if needs and decision in {"text_layer", "blank_or_figure"}:
            decision = "symbol_font_garbage"
        return needs, decision
    if image_area_ratio >= image_dominant_ratio and chars < image_dominant_min_chars:
        return True, "scanned_page" if chars < min_chars else "image_dominant_sparse_text"
    if chars and (pua_char_count + control_char_count) / chars > max_other_share:
        return True, "symbol_font_garbage"
    if chars < min_chars:
        return (True, "scanned_page") if image_area_ratio > min_image_ratio else (False, "blank_or_figure")
    letters = greek_chars + latin_chars
    if (letters and greek_chars / letters > min_greek_share
            and valid_greek_word_rate is not None
            and valid_greek_word_rate < min_valid_word_rate):
        return True, "garbage_greek_text"
    if chars and other_chars / chars > max_other_share:
        return True, "symbol_font_garbage"
    return False, "text_layer"


def compress_page_ranges(pages: Iterable[int]) -> str:
    """Return sorted, unique positive 1-based pages as maximal closed ranges."""
    ordered = sorted(set(pages))
    if any(page < 1 for page in ordered):
        raise ValueError("Page numbers must be positive")
    ranges: list[str] = []
    for page in ordered:
        if ranges and page == end + 1:
            end = page
            ranges[-1] = f"{start}-{end}"
        else:
            start = end = page
            ranges.append(str(page))
    return ",".join(ranges)


def parse_page_ranges(value: str) -> list[tuple[int, int]]:
    """Parse nonoverlapping, ascending 1-based ranges; reject ambiguous input."""
    result: list[tuple[int, int]] = []
    for token in value.split(",") if value.strip() else []:
        bounds = token.strip().split("-")
        if len(bounds) not in (1, 2):
            raise ValueError(f"Invalid page range: {token}")
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 1 or end < start or (result and start <= result[-1][1]):
            raise ValueError(f"Invalid page range: {token}")
        result.append((start, end))
    return result


def _image_coverage(page: pdfium.PdfPage, dpi_probe: bool) -> tuple[float, float | None]:
    """Sum image bbox areas and clamp coverage to [0,1]; never decode pixels."""
    width, height = page.get_size()
    area = width * height
    image_area = 0.0
    dpis: list[float] = []
    for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
        # PDFium 4.x exposes the same bounds operation as get_pos().
        # Keep the OCR inference pin (4.30) usable for raster-page routing.
        bounds = getattr(obj, 'get_bounds', None) or obj.get_pos
        left, bottom, right, top = bounds()
        w, h = max(0.0, right - left), max(0.0, top - bottom)
        image_area += w * h
        if dpi_probe and w and h:
            pixel_size = getattr(obj, 'get_px_size', None) or obj.get_size
            px, py = pixel_size()
            dpis.append(min(px * 72 / w, py * 72 / h))
    return min(1.0, image_area / area) if area > 0 else 0.0, min(dpis) if dpis else None


def page_quality(pdf_path: str | Path, *,
                 wordlist: str | Path | set[str] | frozenset[str] | None = None,
                 dpi_probe: bool = False,
                 output_dir: str | Path | None = None,
                 image_dominant_ratio: float = 0.5,
                 image_dominant_min_chars: int = 600,
                 min_valid_word_rate: float = 0.4,
                 image_dominant_min_valid_word_rate: float = 0.9,
                 image_dominant_max_junk_share: float = 0.1,
                 fold_validity: bool = False,
                 rule_profile: str = "v4") -> PageQualityReport:
    """Inspect every page once, closing all native handles; never render pages.

    A word set takes precedence over the environment; a path uses E1's cached
    loader. Optional DPI probes read image dimensions only. When output_dir is
    supplied, publish json/metrics/<stem>.page_quality.json atomically.
    The image-dominant density bounds are forwarded to page_ocr_decision.
    Defaults are rule_profile="v4" and fold_validity=False. Only v4 loads the
    packaged wordlist automatically; v2/v3 retain their previous lookup defaults.
    The bundled fallback measures image-dominant pages only, preserving v3's
    unconfigured non-image behavior. Explicit dictionaries retain v3's all-page
    scope. An explicit set or GLOSSAPI_GREEK_WORDLIST overrides it. An empty set disables
    lookup (unknown validity), useful for offline routing-policy comparisons.
    v3 selects legacy v2 candidates using the requested validity lookup. With
    folding enabled, candidate labels still use legacy validity so the old
    0.4 cutoff does not silently override v3's 0.5 filter or short-word rule.
    """
    if rule_profile not in ("v2", "v3", "v4"):
        raise ValueError("rule_profile must be 'v2', 'v3' or 'v4'")
    if rule_profile == "v4":
        _validate_image_text_bar(image_dominant_min_valid_word_rate, image_dominant_max_junk_share)
    started = time.perf_counter()
    path = Path(pdf_path)
    identity = None
    bundled_dictionary = False
    if isinstance(wordlist, (set, frozenset)):
        legacy_dictionary = frozenset(unicodedata.normalize("NFC", w).casefold() for w in wordlist)
        dictionary = frozenset(fold_validity_word(w) for w in wordlist) if fold_validity else legacy_dictionary
        if dictionary:
            identity = {"name": "explicit words", "sha256": hashlib.sha256(
                "\n".join(sorted(legacy_dictionary)).encode("utf-8")).hexdigest()}
        elif rule_profile == "v4":
            dictionary = None
    else:
        dictionary_path = wordlist or os.environ.get("GLOSSAPI_GREEK_WORDLIST")
        loader = _load_folded_words if fold_validity else _load_words
        dictionary = loader(str(Path(dictionary_path).expanduser().resolve())) if dictionary_path else None
        legacy_dictionary = (_load_words(str(Path(dictionary_path).expanduser().resolve()))
                             if dictionary_path and fold_validity and rule_profile in ("v3", "v4") else dictionary)
        if dictionary_path:
            identity = {"name": "configured wordlist", "path": str(Path(dictionary_path).expanduser().resolve()),
                        "sha256": hashlib.sha256(Path(dictionary_path).expanduser().read_bytes()).hexdigest()}
        elif rule_profile == "v4":
            bundled_dictionary = True
            legacy_dictionary = _default_words()
            dictionary = (frozenset(fold_validity_word(w) for w in legacy_dictionary)
                          if fold_validity else legacy_dictionary)
            identity = default_wordlist_identity()
    if identity is not None:
        identity["scope"] = "image-dominant pages" if bundled_dictionary else "all pages"
        identity["normalization"] = ("NFD strip marks, lowercase, final sigma to sigma"
                                     if fold_validity else "NFC casefold")
    pages: list[PageQuality] = []
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        for index in range(len(pdf)):
            page = pdf[index]
            try:
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()
                ratio, dpi = _image_coverage(page, dpi_probe)
                use_dictionary = not bundled_dictionary or ratio >= image_dominant_ratio
                stats = _page_text_stats(text, dictionary if use_dictionary else None, fold_validity)
                other = stats["chars"] - stats["greek_chars"] - stats["latin_chars"] - sum(c.isspace() for c in text)
                values = {key: stats[key] for key in ("chars", "greek_chars", "latin_chars", "words")}
                values.update(**text_contamination_counts(text, all_pua=rule_profile == "v4" and ratio >= image_dominant_ratio), other_chars=other, valid_greek_word_rate=stats["dictionary_valid_rate"], image_area_ratio=ratio)
                greek_image = (rule_profile == "v4" and ratio >= image_dominant_ratio
                               and stats["greek_chars"] > stats["latin_chars"])
                junk = junk_token_stats(text, dictionary if use_dictionary else None,
                                        fold_validity) if greek_image else {}
                candidate = None
                if fold_validity and rule_profile in ("v3", "v4"):
                    legacy_rate = _text_stats(text, legacy_dictionary if use_dictionary else None)["dictionary_valid_rate"]
                    candidate = page_ocr_decision(
                        **(values | {"valid_greek_word_rate": legacy_rate}),
                        image_dominant_ratio=image_dominant_ratio,
                        image_dominant_min_chars=image_dominant_min_chars,
                        min_valid_word_rate=min_valid_word_rate,
                        rule_profile="v2",
                    )[1]
                needs, reason = page_ocr_decision(
                    **values,
                    image_dominant_ratio=image_dominant_ratio,
                    image_dominant_min_chars=image_dominant_min_chars,
                    min_valid_word_rate=min_valid_word_rate,
                    rule_profile=rule_profile,
                    candidate_decision=candidate,
                    junk_token_share=junk.get("junk_token_share"),
                    image_dominant_min_valid_word_rate=image_dominant_min_valid_word_rate,
                    image_dominant_max_junk_share=image_dominant_max_junk_share,
                )
                not_applicable = rule_profile == "v4" and _validity_not_applicable(
                    **{key: values[key] for key in ("chars", "greek_chars", "latin_chars",
                        "other_chars", "image_area_ratio", "pua_char_count", "control_char_count")},
                    image_dominant_ratio=image_dominant_ratio,
                    image_dominant_min_chars=image_dominant_min_chars)
                pages.append(PageQuality(page=index + 1, **values, page_needs_ocr=needs, decision=reason,
                                         min_image_dpi=dpi, rule_profile=rule_profile,
                                         validity_not_applicable=not_applicable, **junk,
                                         greek_image_bar_applies=greek_image,
                                         image_dominant_min_valid_word_rate=image_dominant_min_valid_word_rate,
                                         image_dominant_max_junk_share=image_dominant_max_junk_share))
            finally:
                page.close()
    selected = [p.page for p in pages if p.page_needs_ocr]
    report = PageQualityReport(str(path), path.name, pages, len(pages), len(selected),
                               compress_page_ranges(selected), bool(selected),
                               len(selected) / len(pages) if pages else 0.0,
                               dictionary is not None, time.perf_counter() - started,
                               fold_validity=fold_validity, rule_profile=rule_profile,
                               dictionary_identity=identity,
                               image_dominant_min_valid_word_rate=image_dominant_min_valid_word_rate,
                               image_dominant_max_junk_share=image_dominant_max_junk_share)
    if output_dir is not None:
        write_page_quality(report, output_dir)
    return report


def write_page_quality(report: PageQualityReport, output_dir: str | Path) -> Path:
    """Publish a complete sidecar atomically; never leave a partial JSON file."""
    target = Path(output_dir) / "json" / "metrics" / f"{Path(report.filename).stem}.page_quality.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(report.to_dict(), handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return target
