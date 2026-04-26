"""
Python integration smoke test for the GlossAPI Rust extensions.

Exercises the entry points production code paths actually hit:
  - glossapi_rs_cleaner.clean_text()           — basic clean
  - glossapi_rs_cleaner.clean_text_with_stats() — clean + accounting
  - glossapi_rs_cleaner.analyze_charset()      — charset metrics
  - glossapi_rs_cleaner.non_empty_line_stats() — line accounting
  - glossapi_rs_noise.match_token_category_debug_text() — restored matcher PyO3
  - glossapi_rs_noise.evaluate_page_character_noise()  — OCR-side noise scoring

Run via:
    .venv-hplt-review/bin/python3 /tmp/cleaner-smoke-test.py
"""
import json
import sys
from pathlib import Path

import glossapi_rs_cleaner as cleaner
import glossapi_rs_noise as noise

SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]
GREEK_CLEAN = "Καλημέρα κόσμε! Τι όμορφη μέρα.\n"
GREEK_NOISY = (
    "Καλημέρα /uni0301/uni0302/uni0303/uni0304/uni0305/uni0306"
    "/uni0307/uni0308/uni0309/uni030A/uni030B/uni030C\n"
    "\n"  # blank line: keeps Επίλογος in its own paragraph after reflow
    "Επίλογος.\n"
)
LATIN1_TEST = "Hello µ-test © 2026 €\n"
PILOT_B_INPUT = "# Heading\n\nParagraph one.\n\nParagraph two with 𝑥 = 1.\n"

failures: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


print("=" * 72)
print("1. clean_text()")
print("=" * 72)
out = cleaner.clean_text(GREEK_CLEAN, SCRIPTS, None, True, 30, 3, "parser_surgical_verified")
check("clean_text returns string", isinstance(out, str))
check("clean_text preserves Greek chars", "Καλημέρα" in out)
check("clean_text accepts phase_a_mode arg", True, "no exception")

# LineBased opt-in path
out_lb = cleaner.clean_text(GREEK_CLEAN, SCRIPTS, None, True, 30, 3, "line_based")
check("clean_text accepts line_based", isinstance(out_lb, str))

print()
print("=" * 72)
print("2. clean_text_with_stats()")
print("=" * 72)
cleaned, stats = cleaner.clean_text_with_stats(
    GREEK_NOISY, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
)
check("returns (str, dict)", isinstance(cleaned, str) and isinstance(stats, dict))

# Required stats fields per the cleaner's contract
required = [
    "content_chars_kept",
    "chars_dropped_by_line_drop",
    "chars_dropped_by_normalization",
    "chars_dropped_by_per_char_filter",
    "lines_dropped_count",
    "marker_chars_passthrough",
    "marker_chars_added",
    "original_chars_for_badness",
    "sum_kept_line_content_chars",
    "phase_a_fallback_reason",
    "phase_a_dialect_ambiguous_input",
    "rule_a_match_count",
    "rule_b_match_count",
    "residue_line_drop_count",
]
for k in required:
    check(f"stats has '{k}'", k in stats)

# Concrete behaviour: dense /uniXXXX line should hit Rule B count+coverage gate
check(
    "Rule B fires on dense /uniXXXX line",
    stats["rule_b_match_count"] >= 12,
    f"rule_b_match_count={stats['rule_b_match_count']}",
)
check(
    "noisy input drops ≥1 line",
    stats["lines_dropped_count"] >= 1,
    f"lines_dropped_count={stats['lines_dropped_count']}",
)
check(
    "Greek prose on non-noisy line preserved",
    "Επίλογος" in cleaned,
    "(Καλημέρα is on the noisy line that drops correctly)",
)

# µ→μ fold (Group 2)
cleaned_latin1, _ = cleaner.clean_text_with_stats(
    LATIN1_TEST, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
)
check(
    "U+00B5 (µ) folds to U+03BC (μ)",
    "μ" in cleaned_latin1 and "µ" not in cleaned_latin1,
    f"output={cleaned_latin1!r}",
)

# Pilot B fallback signal exposed
cleaned_pb, stats_pb = cleaner.clean_text_with_stats(
    PILOT_B_INPUT, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
)
check(
    "phase_a_fallback_reason is None or str (not raise)",
    stats_pb["phase_a_fallback_reason"] is None or isinstance(stats_pb["phase_a_fallback_reason"], str),
)

print()
print("=" * 72)
print("3. analyze_charset()")
print("=" * 72)
cs = cleaner.analyze_charset(GREEK_CLEAN)
check("analyze_charset returns dict", isinstance(cs, dict))
check("greek_letter_ratio > 0.5", cs.get("greek_letter_ratio", 0) > 0.5)

print()
print("=" * 72)
print("4. non_empty_line_stats()")
print("=" * 72)
total, ne_lines, ne_chars = cleaner.non_empty_line_stats(GREEK_CLEAN)
check("returns 3-tuple", isinstance(total, int) and isinstance(ne_lines, int) and isinstance(ne_chars, int))
check("counts include the one Greek line", ne_lines >= 1 and ne_chars > 0)

print()
print("=" * 72)
print("5. noise.evaluate_page_character_noise() — OCR-side smoke")
print("=" * 72)
res = noise.evaluate_page_character_noise(GREEK_CLEAN)
check("returns dict", isinstance(res, dict))
check("has total_chars / bad_char_ratio", "total_chars" in res and "bad_char_ratio" in res)

print()
print("=" * 72)
print("6. noise.match_token_category_debug_text() — RESTORED matcher PyO3")
print("=" * 72)
# Build a tiny single-category spec on the fly (was three_counter_spec_*)
specs = [
    {
        "category": "glyph_font_like",
        "pattern_family": "uni_glyph",
        "match_kind": "regex",
        "pattern": "/uni[0-9A-Fa-f]{4,6}",
    },
]
specs_path = Path("/tmp/cleaner-smoke-test-spec.json")
specs_path.write_text(json.dumps(specs), encoding="utf-8")
out_dir = Path("/tmp/cleaner-smoke-test-matcher-out")
out_dir.mkdir(parents=True, exist_ok=True)
rows = noise.match_token_category_debug_text(
    GREEK_NOISY,
    str(out_dir),
    str(specs_path),
    "smoke_source",
    "smoke_stem",
    "smoke_base",
    write_files=False,
)
check("matcher returns list of pages", isinstance(rows, list))
check("matcher non-empty", len(rows) > 0)
import json as _json
# per_category_match_count tallies MERGED spans, not raw matches —
# 12 contiguous /uniXXXX hits merge into one span. Count the raw
# matches inside matches_json to verify the matcher saw all 12.
total_raw = 0
for row in rows:
    matches = _json.loads(row.get("matches_json") or "[]")
    for m in matches:
        total_raw += len(m.get("raw_texts") or [m.get("matched_text", "")])
check(
    "matcher saw 12 /uniXXXX raw hits inside the merged span",
    total_raw >= 12,
    f"total_raw={total_raw}",
)

# Bug 1 verification: char offsets, not byte offsets, on Greek-prefixed input
import json as _json
matches_json = rows[0].get("matches_json", "[]")
matches = _json.loads(matches_json)
if matches:
    first = matches[0]
    page_text = rows[0].get("page_text", "")
    sliced = page_text[first["start"]:first["end"]]
    check(
        "Bug 1: char-offset slice == matched_text on Greek prefix",
        sliced == first.get("matched_text"),
        f"sliced={sliced!r} expected={first.get('matched_text')!r}",
    )

print()
print("=" * 72)
print("Summary")
print("=" * 72)
if failures:
    print(f"FAILED: {len(failures)} checks")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL PASS")
