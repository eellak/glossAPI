"""Text-layer page routing; PDFium objects are inspected without rendering.

Counts and dictionary normalization share E1's helpers. ``other_chars`` counts
non-whitespace characters other than Greek/Latin letters (including symbols,
digits and punctuation); ``chars`` includes whitespace, as in E1. A missing
dictionary rate is unknown and cannot trigger the Greek-garbage rule.
"""

from __future__ import annotations

from contextlib import closing

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
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
    rule_profile: str = "v3"

    def to_dict(self) -> dict:
        """Return a JSON-compatible copy, retaining null unknown rates."""
        return asdict(self)


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


def page_rule_selection(decision: str, *, valid_greek_word_rate: float | None,
                        words: int, pua_char_count: int = 0,
                        control_char_count: int = 0,
                        rule_profile: str = "v3") -> bool:
    """Select existing decision labels; useful for archived-sidecar replay too.

    v3 keeps scans and any PUA/control contamination. Garbage candidates need
    known validity < 0.5 or at most 10 words. Unknown validity is not zero.
    Image-dominant sparse text alone is excluded. Validity uses the report's
    explicit lookup mode; choosing v3 does not implicitly enable folding.
    """
    if rule_profile not in ("v2", "v3"):
        raise ValueError("rule_profile must be 'v2' or 'v3'")
    if rule_profile == "v2":
        return decision in {"scanned_page", "garbage_greek_text",
                            "symbol_font_garbage", "image_dominant_sparse_text"}
    return (decision == "scanned_page" or pua_char_count > 0 or control_char_count > 0
            or (decision in {"garbage_greek_text", "symbol_font_garbage"}
                and (words <= 10 or (valid_greek_word_rate is not None
                                    and valid_greek_word_rate < 0.5))))


def text_contamination_counts(text: str) -> dict[str, int]:
    """Count BMP PUA and C0/C1 controls except Markdown whitespace (TAB/LF/CR)."""
    return {
        "pua_char_count": sum("\ue000" <= c <= "\uf8ff" for c in text),
        "control_char_count": sum((ord(c) < 32 or 127 <= ord(c) <= 159)
                                  and c not in "\t\n\r" for c in text),
    }


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
                      rule_profile: str = "v3",
                      candidate_decision: str | None = None) -> tuple[bool, str]:
    """Classify pages, then apply default v3 or selectable legacy v2 routing.

    Legacy v2 labels image coverage >= image_dominant_ratio with
    chars < image_dominant_min_chars as an OCR candidate: a footer/caption does not establish that a raster body is present
    in the text layer. Defaults are 0.5 and 600; coverage is inclusive and the
    character bound is strict. This is a conservative heuristic, not a fidelity
    guarantee. A scan also has chars < min_chars and image ratio > min_image_ratio.
    PUA/control dominance above max_other_share also routes sparse text before
    the blank-page gate (normal TAB/LF/CR are excluded). Otherwise dense Greek text with known validity below min_valid_word_rate, or dense
    text with other_chars/chars > max_other_share, needs OCR. Greek share uses
    Greek + Latin letters, not all characters. Unknown validity is not zero.

    v3 (default) filters these labels with page_rule_selection, dropping image-only
    sparse-text candidates and requiring weak validity/short text for garbage
    candidates while retaining scans and any PUA/control. words is required for v3. Rejected candidates retain their diagnostic decision label.
    candidate_decision can supply a legacy label before validity folding; this
    keeps v3 a filter of v2 candidates without a second validity cutoff.
    """
    if rule_profile not in ("v2", "v3"):
        raise ValueError("rule_profile must be 'v2' or 'v3'")
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
        left, bottom, right, top = obj.get_bounds()
        w, h = max(0.0, right - left), max(0.0, top - bottom)
        image_area += w * h
        if dpi_probe and w and h:
            px, py = obj.get_px_size()
            dpis.append(min(px * 72 / w, py * 72 / h))
    return min(1.0, image_area / area) if area > 0 else 0.0, min(dpis) if dpis else None


def page_quality(pdf_path: str | Path, *,
                 wordlist: str | Path | set[str] | frozenset[str] | None = None,
                 dpi_probe: bool = False,
                 output_dir: str | Path | None = None,
                 image_dominant_ratio: float = 0.5,
                 image_dominant_min_chars: int = 600,
                 fold_validity: bool = False,
                 rule_profile: str = "v3") -> PageQualityReport:
    """Inspect every page once, closing all native handles; never render pages.

    A word set takes precedence over the environment; a path uses E1's cached
    loader. Optional DPI probes read image dimensions only. When output_dir is
    supplied, publish json/metrics/<stem>.page_quality.json atomically.
    The image-dominant density bounds are forwarded to page_ocr_decision.
    Defaults are rule_profile="v3" and fold_validity=False; v2 remains selectable.
    v3 selects legacy v2 candidates using the requested validity lookup. With
    folding enabled, candidate labels still use legacy validity so the old
    0.4 cutoff does not silently override v3's 0.5 filter or short-word rule.
    """
    if rule_profile not in ("v2", "v3"):
        raise ValueError("rule_profile must be 'v2' or 'v3'")
    started = time.perf_counter()
    path = Path(pdf_path)
    if isinstance(wordlist, (set, frozenset)):
        legacy_dictionary = frozenset(unicodedata.normalize("NFC", w).casefold() for w in wordlist)
        dictionary = frozenset(fold_validity_word(w) for w in wordlist) if fold_validity else legacy_dictionary
    else:
        dictionary_path = wordlist or os.environ.get("GLOSSAPI_GREEK_WORDLIST")
        loader = _load_folded_words if fold_validity else _load_words
        dictionary = loader(str(Path(dictionary_path).expanduser().resolve())) if dictionary_path else None
        legacy_dictionary = (_load_words(str(Path(dictionary_path).expanduser().resolve()))
                             if dictionary_path and fold_validity and rule_profile == "v3" else dictionary)
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
                stats = _page_text_stats(text, dictionary, fold_validity)
                other = stats["chars"] - stats["greek_chars"] - stats["latin_chars"] - sum(c.isspace() for c in text)
                ratio, dpi = _image_coverage(page, dpi_probe)
                values = {key: stats[key] for key in ("chars", "greek_chars", "latin_chars", "words")}
                values.update(**text_contamination_counts(text), other_chars=other, valid_greek_word_rate=stats["dictionary_valid_rate"], image_area_ratio=ratio)
                candidate = None
                if fold_validity and rule_profile == "v3":
                    legacy_rate = _text_stats(text, legacy_dictionary)["dictionary_valid_rate"]
                    candidate = page_ocr_decision(
                        **(values | {"valid_greek_word_rate": legacy_rate}),
                        image_dominant_ratio=image_dominant_ratio,
                        image_dominant_min_chars=image_dominant_min_chars,
                        rule_profile="v2",
                    )[1]
                needs, reason = page_ocr_decision(
                    **values,
                    image_dominant_ratio=image_dominant_ratio,
                    image_dominant_min_chars=image_dominant_min_chars,
                    rule_profile=rule_profile,
                    candidate_decision=candidate,
                )
                pages.append(PageQuality(page=index + 1, **values, page_needs_ocr=needs, decision=reason, min_image_dpi=dpi))
            finally:
                page.close()
    selected = [p.page for p in pages if p.page_needs_ocr]
    report = PageQualityReport(str(path), path.name, pages, len(pages), len(selected),
                               compress_page_ranges(selected), bool(selected),
                               len(selected) / len(pages) if pages else 0.0,
                               dictionary is not None, time.perf_counter() - started,
                               fold_validity=fold_validity, rule_profile=rule_profile)
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
