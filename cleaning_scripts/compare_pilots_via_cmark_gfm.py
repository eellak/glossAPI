"""Run both Pilot A (format_parsed, full round-trip) and Pilot B
(format_surgical, targeted rewrite) on every BEFORE doc in the
sample, verify each via cmark-gfm, and report side-by-side pass
rates.

Usage:
  python3 compare_pilots_via_cmark_gfm.py \\
      --sample-dir ~/data/phase_a_audit/top100_review \\
      --output ~/data/phase_a_audit/pilot_comparison.json
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


def run_pilot(fn_name: str, body: str):
    try:
        out = getattr(c, fn_name)(body)
    except Exception as err:
        return None, f"{fn_name} raised: {err}"
    r = c.cmark_gfm_verify_py(body, out)
    return r.get("preview_identical"), r.get("first_diff")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    probe = c.cmark_gfm_verify_py("a", "a")
    if not probe.get("is_available"):
        print("cmark-gfm not available", file=sys.stderr)
        return 2

    befores = sorted(args.sample_dir.glob("*_BEFORE.md"))
    total = len(befores)
    a_pass = 0
    b_pass = 0
    only_b = []
    only_a = []
    both_fail = []
    for i, b in enumerate(befores):
        body = extract_body(b)
        a_ok, a_diff = run_pilot("format_parsed_py", body)
        b_ok, b_diff = run_pilot("format_surgical_py", body)
        if a_ok:
            a_pass += 1
        if b_ok:
            b_pass += 1
        pair = b.stem.replace("_BEFORE", "")
        if b_ok and not a_ok:
            only_b.append({"pair": pair, "a_diff": (a_diff or "")[:300]})
        elif a_ok and not b_ok:
            only_a.append({"pair": pair, "b_diff": (b_diff or "")[:300]})
        elif not a_ok and not b_ok:
            both_fail.append({
                "pair": pair,
                "a_diff": (a_diff or "")[:200],
                "b_diff": (b_diff or "")[:200],
            })
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] pilot_a_pass={a_pass} pilot_b_pass={b_pass}",
                  flush=True)

    report = {
        "total": total,
        "pilot_a_pass": a_pass,
        "pilot_b_pass": b_pass,
        "pilot_a_pct": 100 * a_pass / total if total else 0.0,
        "pilot_b_pct": 100 * b_pass / total if total else 0.0,
        "b_recovers_failures_a": len(only_b),
        "a_pass_b_fail": len(only_a),
        "both_fail": len(both_fail),
        "only_b_samples": only_b[:10],
        "only_a_samples": only_a[:10],
        "both_fail_samples": both_fail[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print()
    print(f"TOTAL: {total}")
    print(f"  Pilot A (full round-trip):  {a_pass}/{total} = {report['pilot_a_pct']:.1f}%")
    print(f"  Pilot B (surgical rewrite): {b_pass}/{total} = {report['pilot_b_pct']:.1f}%")
    print(f"  B recovers A's failures: {len(only_b)}")
    print(f"  B breaks A's successes:  {len(only_a)}")
    print(f"  Both still fail:         {len(both_fail)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
