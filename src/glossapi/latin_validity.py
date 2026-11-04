"""Small, exact surface-vocabulary measurements for experimental routing v6.

These are dictionary coverage scores, not language-model probabilities. Ties
keep the first language in the fixed order; no union of languages can validate
a page. The English vocabulary and v5 normalization remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata


WORDLIST_DIR = Path(__file__).with_name("data") / "wordlists"
LANGUAGES = ("en", "fr", "de", "it", "es", "pt")
MIN_LATIN_TOKENS = 10
MIN_LATIN_SHARE = 0.8


def latin_tokens(text: str) -> list[str]:
    """Use the same all-Latin letter token denominator as routing v5."""
    return [token.casefold() for token in re.findall(
        r"[^\W\d_]+", unicodedata.normalize("NFC", text))
        if all("LATIN" in unicodedata.name(char, "") for char in token)]


def vocabulary_key(word: str, language: str) -> str:
    """Match Moby's ASCII accent notation without guessing accented spellings."""
    if language == "en":
        return unicodedata.normalize("NFC", word).casefold()
    word = unicodedata.normalize("NFC", word).casefold()
    if language == "de":
        word = word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return "".join(char for char in unicodedata.normalize("NFD", word)
                   if "a" <= char <= "z")


@lru_cache(maxsize=1)
def vocabulary_identities() -> dict[str, dict]:
    return json.loads((WORDLIST_DIR / "moby_manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=5)
def _words(language: str) -> frozenset[str]:
    identity = vocabulary_identities()[language]
    data = gzip.decompress((WORDLIST_DIR / identity["file"]).read_bytes())
    if hashlib.sha256(data).hexdigest() != identity["sha256"]:
        raise ValueError(f"Packaged {language} wordlist SHA256 mismatch")
    words = frozenset(data.decode("ascii").splitlines())
    if len(words) != identity["entries"]:
        raise ValueError(f"Packaged {language} wordlist entry count mismatch")
    return words


@dataclass(frozen=True)
class LatinValidity:
    language: str | None
    rate: float | None
    rates: dict[str, float]
    tokens: int


def measure_latin_validity(text: str, english: frozenset[str]) -> LatinValidity:
    """Pick the single dictionary with the greatest token-weighted hit rate."""
    tokens = latin_tokens(text)
    if not tokens:
        return LatinValidity(None, None, {}, 0)
    rates = {}
    for language in LANGUAGES:
        words = english if language == "en" else _words(language)
        rates[language] = sum(vocabulary_key(token, language) in words
                              for token in tokens) / len(tokens)
    best = max(LANGUAGES, key=lambda language: rates[language])
    return LatinValidity(best, rates[best], rates, len(tokens))
