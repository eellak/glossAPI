"""Formal verification pass over the top-altered BEFORE/AFTER sample.

For each pair in the sample dir, runs
`verify_md_preview_equivalent(before_body, after_body)` (pulldown-cmark
reference parser) and reports pass-rate + failure details. If Phase A
truly preserves preview, every pair passes strict equivalence.

Reads the body by stripping the metadata header (everything up to and
including the first `---` HR line after the `## Post-Phase-A output`
or `## Raw input (pre-Phase-A)` section marker).

Usage:
  python3 verify_phase_a_sample_pairs.py \
      --sample-dir /home/foivos/data/phase_a_audit/top100_review \
      --output /home/foivos/data/phase_a_audit/verify_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import glossapi_rs_cleaner as c


# Anchor for the header-body divider. The sampler emits either of
# these section headings immediately before the `---\n` that opens
# the body, so finding the FIRST `---\n` after this anchor is the
# correct way to locate the body — "last `---` in the file" is
# WRONG for HR-dominated docs where the AFTER body contains many
# canonical `---` HR lines after Phase A's normalization.
HEADER_ANCHOR = re.compile(
    r"^##\s+(?:Raw input \(pre-Phase-A\) — BEFORE|Post-Phase-A output — AFTER)\s*$",
    re.MULTILINE,
)


def extract_body(md_path: Path) -> str:
    """Return the body (content below the `---` HR that opens the
    body, identified by anchoring on the `## Raw input … BEFORE` /
    `## Post-Phase-A output — AFTER` section heading that the sampler
    writes immediately above the divider).
    """
    text = md_path.read_text(encoding="utf-8")
    anchor = HEADER_ANCHOR.search(text)
    if not anchor:
        return text
    rest = text[anchor.end():]
    # First `---\n` (standalone HR line) after the anchor is the
    # body divider.
    divider = re.search(r"^---\s*\n", rest, re.MULTILINE)
    if not divider:
        return rest.lstrip("\n")
    return rest[divider.end():].lstrip("\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    befores = sorted(args.sample_dir.glob("*_BEFORE.md"))
    pairs_total = 0
    strict_pass = 0
    strict_fail = []
    per_lens_stats = Counter()
    for b in befores:
        after = args.sample_dir / b.name.replace("_BEFORE.md", "_AFTER.md")
        if not after.is_file():
            print(f"  [skip] missing AFTER for {b.name}", file=sys.stderr)
            continue
        pairs_total += 1
        before_body = extract_body(b)
        after_body = extract_body(after)
        r = c.verify_md_preview_equivalent_py(before_body, after_body)
        if r.get("is_strict_equivalent"):
            strict_pass += 1
        else:
            strict_fail.append({
                "pair": b.stem.replace("_BEFORE", ""),
                "html_render_equal": r.get("html_render_equal"),
                "block_sequence_equal": r.get("block_sequence_equal"),
                "paragraph_text_equal": r.get("paragraph_text_equal"),
                "table_cells_equal": r.get("table_cells_equal"),
                "first_diff": (r.get("first_diff") or "")[:400],
            })

    report = {
        "pairs_total": pairs_total,
        "strict_pass": strict_pass,
        "strict_fail_count": len(strict_fail),
        "strict_pass_pct": (
            100.0 * strict_pass / pairs_total if pairs_total else 0.0
        ),
        "fails": strict_fail,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"pairs: {pairs_total}")
    print(f"strict_pass: {strict_pass} ({report['strict_pass_pct']:.1f}%)")
    print(f"strict_fail: {len(strict_fail)}")
    if strict_fail:
        print("first 5 failure pairs:")
        for f in strict_fail[:5]:
            print(f"  - {f['pair']}: {f['first_diff'][:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
