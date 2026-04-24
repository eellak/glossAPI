"""Run `format_parsed` on every BEFORE doc in the sample dir,
verify each (input, output) pair via the cmark-gfm subprocess
oracle, and emit a pass-rate summary.

cmark-gfm is the C reference renderer GitHub actually uses — so
"input and output render to the same HTML under cmark-gfm" is the
strongest preview-preservation signal available.

Usage:
  python3 verify_md_format_via_cmark_gfm.py \\
      --sample-dir ~/data/phase_a_audit/top100_review \\
      --output ~/data/phase_a_audit/cmark_verify_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import glossapi_rs_cleaner as c

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
    return rest[div.end():].lstrip("\n") if div else rest.lstrip("\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-docs", type=int, default=0,
                   help="If >0, process only the first N BEFORE files.")
    p.add_argument("--formatter", default="format_parsed_py",
                   choices=["format_parsed_py", "format_surgical_py"],
                   help="Pilot A (round-trip) vs Pilot B (surgical).")
    args = p.parse_args()

    # Confirm cmark-gfm is actually usable before starting.
    probe = c.cmark_gfm_verify_py("a", "a")
    if not probe.get("is_available"):
        print(f"cmark-gfm not available: {probe.get('error')}", file=sys.stderr)
        return 2

    befores = sorted(args.sample_dir.glob("*_BEFORE.md"))
    if args.max_docs > 0:
        befores = befores[: args.max_docs]
    total = len(befores)
    n_pass = 0
    fails = []
    formatter = getattr(c, args.formatter)
    for i, b in enumerate(befores):
        body = extract_body(b)
        try:
            out = formatter(body)
        except Exception as err:
            fails.append({
                "pair": b.stem.replace("_BEFORE", ""),
                "stage": "format_parsed",
                "error": str(err),
            })
            continue
        r = c.cmark_gfm_verify_py(body, out)
        if r.get("identical"):
            n_pass += 1
        else:
            fails.append({
                "pair": b.stem.replace("_BEFORE", ""),
                "stage": "cmark_gfm",
                "first_diff": (r.get("first_diff") or "")[:500],
            })
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] pass so far: {n_pass}", flush=True)

    report = {
        "total": total,
        "pass": n_pass,
        "pass_pct": (100.0 * n_pass / total) if total else 0.0,
        "fails_count": len(fails),
        "fails": fails[:100],  # keep report small
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"total: {total}  pass: {n_pass} ({report['pass_pct']:.1f}%)")
    print(f"fails: {len(fails)}")
    if fails:
        print("first 3 failure pairs:")
        for f in fails[:3]:
            snippet = f.get('first_diff') or f.get('error', '')
            print(f"  {f['pair']}: {snippet[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
