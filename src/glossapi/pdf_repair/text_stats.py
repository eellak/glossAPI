"""Text-layer diagnostics using only PDFium and the standard library."""
from contextlib import closing
from functools import lru_cache
import os
from pathlib import Path
import re
import unicodedata
from typing import Any
import pypdfium2 as pdfium


def _is_greek(char: str) -> bool:
    """Count Greek letters, excluding punctuation in the Greek Unicode blocks."""
    return char.isalpha() and "GREEK" in unicodedata.name(char, "")


def _page_text(pdf: Any, index: int) -> str:
    """Close page/text handles explicitly, including on extraction failure."""
    page = pdf[index]
    try:
        text = page.get_textpage()
        try:
            return text.get_text_range()
        finally:
            text.close()
    finally:
        page.close()


@lru_cache(maxsize=4)
def _load_words(path: str) -> frozenset[str]:
    """Load each requested word list lazily, preserving tonos and normalizing case."""
    with open(path, encoding="utf-8") as handle:
        return frozenset(unicodedata.normalize("NFC", word.strip()).casefold() for word in handle if word.strip())


def _text_stats(text: str, words: frozenset[str] | None) -> dict[str, Any]:
    """Use Greek-containing word tokens as the dictionary-rate denominator."""
    tokens = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text), flags=re.UNICODE)
    greek = [token for token in tokens if any(_is_greek(char) for char in token)]
    valid = sum(token.casefold() in words for token in greek) if words is not None else None
    chars = len(text)
    greek_chars = sum(_is_greek(char) for char in text)
    return {"chars": chars, "greek_chars": greek_chars,
            "latin_chars": sum(char.isalpha() and "LATIN" in unicodedata.name(char, "") for char in text),
            "words": len(tokens), "greek_words": len(greek), "dictionary_valid_words": valid,
            "dictionary_valid_rate": valid / len(greek) if greek and valid is not None else None,
            "greek_share": greek_chars / chars if chars else 0.0}


def extract_text_stats(path: str | Path, max_pages: int | None = None, *,
                       wordlist_path: str | Path | None = None,
                       word_set: set[str] | frozenset[str] | None = None) -> dict[str, Any]:
    """Extract the first max_pages with PDFium; absent dictionary rates are null.

    Explicit word_set takes precedence over wordlist_path and the environment.
    Totals are word-weighted counts, never an average of page rates.
    """
    if max_pages is not None and max_pages < 0:
        raise ValueError("max_pages must be nonnegative")
    dictionary = wordlist_path or os.environ.get("GLOSSAPI_GREEK_WORDLIST")
    words = (frozenset(unicodedata.normalize("NFC", word).casefold() for word in word_set)
             if word_set is not None else _load_words(str(Path(dictionary).expanduser().resolve())) if dictionary else None)
    pages = []
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        count = len(pdf) if max_pages is None else min(max_pages, len(pdf))
        for index in range(count):
            stats = _text_stats(_page_text(pdf, index), words)
            pages.append({"page": index + 1, **stats})
    keys = ("chars", "greek_chars", "latin_chars", "words", "greek_words")
    totals = {key: sum(page[key] for page in pages) for key in keys}
    totals["dictionary_valid_words"] = sum(page["dictionary_valid_words"] for page in pages) if words is not None else None
    totals["dictionary_valid_rate"] = totals["dictionary_valid_words"] / totals["greek_words"] if words is not None and totals["greek_words"] else None
    totals["greek_share"] = totals["greek_chars"] / totals["chars"] if totals["chars"] else 0.0
    return {"path": str(path), "pages": pages, "totals": totals, "dictionary_loaded": words is not None}
