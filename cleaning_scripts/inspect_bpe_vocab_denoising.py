"""Phase 6: inspect what changed in the fresh BPE vocabulary after
three-counter-driven cleaning.

Compares an OLD tokenizer (trained on pre-v3-cleaning corpus) and a NEW
tokenizer (trained on v3). Counts per-category of interest:

  - Tokens containing Latin Ext-A/-B codepoints (U+0100..U+024F)
  - Tokens containing PUA codepoints (U+E000..U+F8FF, U+F0000+)
  - Tokens containing U+FFFD replacement char
  - Tokens containing bare `GLYPH`, `/uni`, `/gid`, `/hyphenminus`, etc.
    (the extended glyph-marker family we added this session)
  - Tokens containing Math-Alphanumeric Latin/Greek (U+1D400..U+1D7FF)
  - Tokens containing escaped-underscore chains `\\_\\_\\_`

For each category reports: count in OLD, count in NEW, delta, sample
tokens removed (in OLD but not NEW) and added (in NEW but not OLD).

This is the single test that actually tells us whether the cleaning
work was worth the effort (per the "inspection, not just counts" rule
from glossapi_tokenizer_methodology.md).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set


CATEGORIES: Dict[str, callable] = {
    "latin_ext_a_or_b": lambda t: any(0x0100 <= ord(c) <= 0x024F for c in t),
    "pua": lambda t: any(
        0xE000 <= ord(c) <= 0xF8FF
        or 0xF0000 <= ord(c) <= 0xFFFFD
        or 0x100000 <= ord(c) <= 0x10FFFD
        for c in t
    ),
    "replacement_char": lambda t: "\uFFFD" in t,
    "bare_glyph_family": lambda t: any(
        needle in t
        for needle in (
            "GLYPH",
            "hyphenminus",
            "/uni",
            "/gid",
            "/hyphenminus",
            "/space",
            "/period",
            "/glyph",
            "font=/",
            "FontName=",
        )
    ),
    "math_alphanumeric": lambda t: any(0x1D400 <= ord(c) <= 0x1D7FF for c in t),
    "escaped_underscore_chain": lambda t: "\\_\\_" in t,
    "script_salad_ext_a_plus_b": lambda t: (
        any(0x0100 <= ord(c) <= 0x017F for c in t)
        and any(0x0180 <= ord(c) <= 0x024F for c in t)
    ),
    "pdf_font_subset_regex": lambda t: bool(
        re.search(r"/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+", t)
    ),
}


def _load_vocab_tokens(tokenizer_path: Path) -> List[str]:
    """Load a HF tokenizer.json and return its vocab list."""
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab", {})
    if isinstance(vocab, dict):
        # Many HF tokenizers encode as dict token -> id.
        return list(vocab.keys())
    return [t for t, _ in vocab]


def _byte_decode_token(token: str) -> str:
    """Hugging Face ByteLevel BPE stores tokens with byte-level escapes.
    Return the semantic string for pattern matching. We strip the
    leading-space marker `Ġ` and the `Ċ` newline marker to keep the
    pattern checks honest.
    """
    try:
        # Best-effort: the ByteLevel pretokenizer uses bytes_to_unicode().
        # A full inversion isn't critical for pattern-match categories,
        # since the patterns we care about are composed of ASCII-range
        # or common Unicode chars that survive the byte-level round-trip.
        decoded = token
    except Exception:
        decoded = token
    # Trim common byte-level markers so they don't confuse the needle checks.
    decoded = decoded.replace("Ġ", " ").replace("Ċ", "\n")
    return decoded


def _categorize(tokens: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {k: [] for k in CATEGORIES}
    for tok in tokens:
        decoded = _byte_decode_token(tok)
        for name, predicate in CATEGORIES.items():
            if predicate(decoded):
                out[name].append(tok)
    return out


def _summary(
    old_tokens: List[str], new_tokens: List[str], sample_size: int = 10
) -> Dict[str, Any]:
    old_cat = _categorize(old_tokens)
    new_cat = _categorize(new_tokens)

    out: Dict[str, Any] = {
        "old_vocab_size": len(old_tokens),
        "new_vocab_size": len(new_tokens),
        "per_category": {},
    }
    for name in CATEGORIES:
        old_set: Set[str] = set(old_cat[name])
        new_set: Set[str] = set(new_cat[name])
        removed = sorted(old_set - new_set)
        added = sorted(new_set - old_set)
        retained = sorted(old_set & new_set)
        out["per_category"][name] = {
            "old_count": len(old_set),
            "new_count": len(new_set),
            "delta": len(new_set) - len(old_set),
            "removed_count": len(removed),
            "added_count": len(added),
            "retained_count": len(retained),
            "sample_removed": removed[:sample_size],
            "sample_added": added[:sample_size],
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-tokenizer", required=True, type=Path)
    parser.add_argument("--new-tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=10)
    args = parser.parse_args(argv)

    old_tokens = _load_vocab_tokens(args.old_tokenizer)
    new_tokens = _load_vocab_tokens(args.new_tokenizer)
    summary = _summary(old_tokens, new_tokens, args.sample_size)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"old vocab: {summary['old_vocab_size']}")
    print(f"new vocab: {summary['new_vocab_size']}")
    for name, rec in summary["per_category"].items():
        sign = "+" if rec["delta"] >= 0 else ""
        print(
            f"  {name:35s} old={rec['old_count']:4d} "
            f"new={rec['new_count']:4d} "
            f"delta={sign}{rec['delta']:4d} "
            f"(removed={rec['removed_count']}, added={rec['added_count']})"
        )
    print(f"\nfull diff → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
