"""Quick band-comparison analyzer for the char_strip_ratio smoke test.

Input:  verdicts.jsonl from gemini_broken_text_reviewer.py
Output: band_summary.md — contingency tables per verdict axis, split
        by low/high band. Plus a sample of short_reasons per band.

This is NOT the full zone-analyzer (analyze_broken_text_verdicts.py) —
that one does quantile-zone breakdowns. For a 2-band smoke test we just
want the low-vs-high contrast.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


BINARY_AXES = [
    "subject_clear_end_to_end",
    "has_broken_words_mid_token",
    "has_narrative_jumps_from_line_drops",
    "has_mid_thought_sentences",
    "is_too_short_to_be_useful",
]
ENUM_AXES = ["defect_rate_estimate", "text_partition"]


def _load(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("verdict"):
                out.append(d)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts-path", required=True, type=Path)
    parser.add_argument("--sample-path", required=True, type=Path,
                        help="Original sample.jsonl (has 'band' field for joining)")
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = _load(args.verdicts_path)
    # Join band info by source_path.
    band_by_path = {}
    with args.sample_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            band_by_path[d["source_path"]] = d.get("band", "?")
    for r in rows:
        r["band"] = band_by_path.get(r.get("source_path"), "?")

    by_band: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_band[r["band"]].append(r)

    out = [f"# Band-comparison smoke test — {len(rows)} verdicts", ""]
    for band, docs in sorted(by_band.items()):
        out.append(f"## Band: `{band}` (N={len(docs)})")
        out.append("")
        out.append("### Enum axes (distribution)")
        out.append("")
        for axis in ENUM_AXES:
            c = Counter(d["verdict"].get(axis, "MISSING") for d in docs)
            out.append(f"**{axis}**: {dict(c)}")
        out.append("")
        out.append("### Binary axes (yes-rate)")
        out.append("")
        out.append("| axis | yes | no | uncertain | yes_rate |")
        out.append("|---|---:|---:|---:|---:|")
        for axis in BINARY_AXES:
            c = Counter(d["verdict"].get(axis, "MISSING") for d in docs)
            yr = c["yes"] / max(len(docs), 1)
            out.append(f"| {axis} | {c['yes']} | {c['no']} | {c['uncertain']} | {yr:.2f} |")
        out.append("")
        out.append("### Sample short_reasons")
        out.append("")
        for r in docs[:5]:
            rsn = r["verdict"].get("short_reason", "")
            ds = r.get("source_dataset", "")
            did = str(r.get("source_doc_id", ""))[:30]
            out.append(f"- [{ds}/{did}] {rsn}")
        out.append("")

    out.append("## Low-vs-high contrast (binary yes-rate delta)")
    out.append("")
    out.append("| axis | low yes_rate | high yes_rate | delta (high - low) |")
    out.append("|---|---:|---:|---:|")
    for axis in BINARY_AXES:
        for band_pair in ["low", "high"]:
            pass
        low_docs = by_band.get("low", [])
        high_docs = by_band.get("high", [])
        low_yes = sum(1 for d in low_docs if d["verdict"].get(axis) == "yes")
        high_yes = sum(1 for d in high_docs if d["verdict"].get(axis) == "yes")
        low_rate = low_yes / max(len(low_docs), 1)
        high_rate = high_yes / max(len(high_docs), 1)
        out.append(f"| {axis} | {low_rate:.2f} | {high_rate:.2f} | {high_rate - low_rate:+.2f} |")
    out.append("")
    out.append("## Enum axis contrast")
    out.append("")
    for axis in ENUM_AXES:
        out.append(f"### {axis}")
        out.append("")
        out.append("| value | low count | high count |")
        out.append("|---|---:|---:|")
        low_c = Counter(d["verdict"].get(axis) for d in by_band.get("low", []))
        high_c = Counter(d["verdict"].get(axis) for d in by_band.get("high", []))
        all_values = sorted(set(low_c.keys()) | set(high_c.keys()), key=str)
        for v in all_values:
            out.append(f"| {v} | {low_c.get(v, 0)} | {high_c.get(v, 0)} |")
        out.append("")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote → {args.output_path}")
    # Also print the contrast table for quick viewing.
    print()
    print("=== low-vs-high contrast ===")
    for line in [l for l in out if l.startswith("|")][-6:]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
