"""Extract the N most interesting reflow-join segments from a
BEFORE/AFTER pair in the Phase-A review sample.

A "reflow segment" is a contiguous diff region where BEFORE had
multiple consecutive lines joined by Phase A's reflow into one (or
fewer) lines in AFTER. Other Phase A transforms (HR / GFM / blank-
line collapse) are deliberately ignored — user directive: focus on
reflow behaviour specifically.

Output: JSON list of `{before_lines, after_lines, before_snippet,
after_snippet}` objects, sorted by the size of the reflow region
descending (largest collapses first).

Usage:
  python3 extract_reflow_segments.py \\
      --sample-dir /home/foivos/data/phase_a_audit/top100_review \\
      --pair 001_R0425418_H000000_G0012849_pct0164_greek_phd_xxx \\
      --top 20 --output /tmp/segments.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

HDR = re.compile(
    r"^##\s+(?:Raw input \(pre-Phase-A\) — BEFORE|Post-Phase-A output — AFTER)\s*$",
    re.MULTILINE,
)


def extract_body(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    anc = HDR.search(text)
    if not anc:
        return text
    rest = text[anc.end():]
    div = re.search(r"^---\s*\n", rest, re.MULTILINE)
    if not div:
        return rest.lstrip("\n")
    return rest[div.end():].lstrip("\n")


def reflow_segments(before: str, after: str, max_context: int = 2):
    """Return list of reflow regions.

    A region is a hunk in the unified diff where BEFORE contributes
    ≥ 2 consecutive removed lines and AFTER contributes ≥ 1 replaced
    line (i.e. 2+ lines were joined into 1+). HR-like regions (single
    dash-run line on one side) are excluded since those are HR rewrite,
    not reflow.
    """
    bl = before.split("\n")
    al = after.split("\n")
    sm = difflib.SequenceMatcher(None, bl, al, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op not in ("replace", "delete"):
            continue
        before_n = i2 - i1
        after_n = j2 - j1
        # Reflow signature: ≥ 2 lines replaced with fewer.
        if before_n < 2:
            continue
        if after_n >= before_n:
            continue
        before_lines = bl[i1:i2]
        after_lines = al[j1:j2]
        # Skip if the "collapse" is just stripping blank lines.
        if all(not l.strip() for l in before_lines):
            continue
        # Skip if a full line on either side is an HR-candidate
        # (dashes/underscores/asterisks only) — that's HR, not reflow.
        def is_hr(s): return bool(re.fullmatch(r"\s*[-_*]{3,}\s*", s))
        if any(is_hr(l) for l in before_lines) or any(
            is_hr(l) for l in after_lines
        ):
            continue
        # Skip GFM table sep rows.
        if any(
            re.match(r"\s*\|[-:\s|]+\|?\s*$", l)
            for l in before_lines
        ):
            continue
        # Accumulate context
        ctx_before_start = max(0, i1 - max_context)
        ctx_after_start = max(0, j1 - max_context)
        before_ctx = bl[ctx_before_start:i1]
        after_ctx = al[ctx_after_start:j1]
        ctx_before_end = min(len(bl), i2 + max_context)
        ctx_after_end = min(len(al), j2 + max_context)
        before_post = bl[i2:ctx_before_end]
        after_post = al[j2:ctx_after_end]
        out.append({
            "before_count": before_n,
            "after_count": after_n,
            "collapsed": before_n - after_n,
            "before_snippet": "\n".join(before_ctx + ["<<<"] + before_lines + [">>>"] + before_post),
            "after_snippet": "\n".join(after_ctx + ["<<<"] + after_lines + [">>>"] + after_post),
        })
    out.sort(key=lambda r: r["collapsed"], reverse=True)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--pair", required=True,
                   help="Pair base name (without _BEFORE.md / _AFTER.md).")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-snippet-chars", type=int, default=2000,
                   help="Truncate each snippet to this many chars.")
    args = p.parse_args()

    b = args.sample_dir / f"{args.pair}_BEFORE.md"
    a = args.sample_dir / f"{args.pair}_AFTER.md"
    if not b.is_file() or not a.is_file():
        print(f"missing pair: {b} or {a}", file=sys.stderr)
        return 1
    before = extract_body(b)
    after = extract_body(a)
    segs = reflow_segments(before, after)[: args.top]
    # Truncate snippet chars.
    for s in segs:
        for k in ("before_snippet", "after_snippet"):
            if len(s[k]) > args.max_snippet_chars:
                s[k] = s[k][: args.max_snippet_chars] + "\n…[truncated]…"
    payload = {
        "pair": args.pair,
        "before_lines": len(before.split("\n")),
        "after_lines": len(after.split("\n")),
        "total_reflow_segments_found": len(reflow_segments(before, after)),
        "top_segments": segs,
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"wrote {len(segs)} segments → {args.output}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
