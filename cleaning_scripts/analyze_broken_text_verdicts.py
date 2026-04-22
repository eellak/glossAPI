"""Analyze Gemini broken-text verdicts → per-metric zone yes-rates +
correlation-ready summary.

Purpose: look at SHAPE of label-vs-stats distributions BEFORE setting
any cutoff (per user direction 2026-04-22 — no pre-committed thresholds).

Input:
  --verdicts-path  verdicts.jsonl from gemini_broken_text_reviewer.py

Output in --output-dir:
  quality_distributions.md  — zone yes-rates per (metric, question) pair
                              + overall distribution tables
  quality_summary.json      — structured version of the same for
                              downstream cutoff calibration

What the report shows:
  1. Overall distributions of each verdict axis (defect_rate_estimate,
     text_partition, and five yes/no/uncertain axes).
  2. Per-metric zone breakdowns: for each of the four stratification
     metrics (char_strip_ratio, line_drop_ratio, non_empty_chars_out,
     pct_chars_removed_non_empty), bucket docs into quantile zones and
     show yes-rate of each binary verdict axis per zone.
  3. Correlation table: which metric-zone most strongly predicts each
     yes/no axis (simple: pick the zone with the highest yes-rate).

No threshold is set. This is a LOOK-AT-SHAPE report; the user picks
cutoffs by reading the zone tables.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


METRICS = ["char_strip_ratio", "line_drop_ratio", "non_empty_chars_out",
           "pct_chars_removed_non_empty"]

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


def _quantile_cuts(values: List[float], n_zones: int = 10) -> List[float]:
    if not values:
        return []
    xs = sorted(values)
    return [xs[int(len(xs) * q / n_zones)] for q in range(1, n_zones)]


def _zone_of(value: float, cuts: List[float]) -> int:
    for i, c in enumerate(cuts):
        if value < c:
            return i
    return len(cuts)


def _overall_distribution(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dist = {}
    for axis in ENUM_AXES + BINARY_AXES:
        c = Counter(r["verdict"].get(axis, "MISSING") for r in rows)
        dist[axis] = dict(c)
    return dist


def _zone_rates(
    rows: List[Dict[str, Any]],
    metric: str,
    axis: str,
    n_zones: int = 10,
) -> List[Dict[str, Any]]:
    values = [r.get("metrics", {}).get(metric, 0.0) for r in rows]
    cuts = _quantile_cuts(values, n_zones=n_zones)
    zones: Dict[int, List[str]] = defaultdict(list)
    for r in rows:
        v = r.get("metrics", {}).get(metric, 0.0)
        z = _zone_of(v, cuts)
        zones[z].append(r["verdict"].get(axis, "uncertain"))
    out = []
    for z in range(n_zones):
        labels = zones.get(z, [])
        if not labels:
            out.append({"zone": z, "n": 0, "yes": 0, "no": 0, "uncertain": 0,
                        "yes_rate": None, "zone_min": None, "zone_max": None})
            continue
        c = Counter(labels)
        yes_rate = c["yes"] / len(labels) if labels else None
        # Zone boundaries for display.
        zone_min = cuts[z - 1] if z > 0 else min(values, default=0)
        zone_max = cuts[z] if z < len(cuts) else max(values, default=0)
        out.append({
            "zone": z, "n": len(labels),
            "yes": c["yes"], "no": c["no"], "uncertain": c["uncertain"],
            "yes_rate": round(yes_rate, 3) if yes_rate is not None else None,
            "zone_min": round(zone_min, 4),
            "zone_max": round(zone_max, 4),
        })
    return out


def _enum_zone_rates(
    rows: List[Dict[str, Any]],
    metric: str,
    axis: str,
    n_zones: int = 10,
) -> List[Dict[str, Any]]:
    """For enum axes (defect_rate_estimate, text_partition): per zone,
    show distribution across categories rather than yes-rate."""
    values = [r.get("metrics", {}).get(metric, 0.0) for r in rows]
    cuts = _quantile_cuts(values, n_zones=n_zones)
    zones: Dict[int, List[str]] = defaultdict(list)
    for r in rows:
        v = r.get("metrics", {}).get(metric, 0.0)
        z = _zone_of(v, cuts)
        zones[z].append(r["verdict"].get(axis, "MISSING"))
    out = []
    for z in range(n_zones):
        labels = zones.get(z, [])
        zone_min = cuts[z - 1] if z > 0 else min(values, default=0)
        zone_max = cuts[z] if z < len(cuts) else max(values, default=0)
        out.append({
            "zone": z, "n": len(labels),
            "distribution": dict(Counter(labels)),
            "zone_min": round(zone_min, 4),
            "zone_max": round(zone_max, 4),
        })
    return out


def _format_binary_zone_table(zone_rates: List[Dict[str, Any]], metric: str, axis: str) -> str:
    lines = [f"### {axis} vs {metric}"]
    lines.append("| zone | range | n | yes | no | unc | yes_rate |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for z in zone_rates:
        rng = (f"{z['zone_min']}–{z['zone_max']}"
               if z['zone_min'] is not None else "-")
        yr = f"{z['yes_rate']:.2f}" if z['yes_rate'] is not None else "-"
        lines.append(
            f"| {z['zone']} | {rng} | {z['n']} | {z['yes']} | {z['no']} | "
            f"{z['uncertain']} | {yr} |"
        )
    return "\n".join(lines) + "\n"


def _format_enum_zone_table(zone_rates: List[Dict[str, Any]], metric: str, axis: str) -> str:
    lines = [f"### {axis} vs {metric}"]
    lines.append("| zone | range | n | distribution |")
    lines.append("|---:|---|---:|---|")
    for z in zone_rates:
        rng = (f"{z['zone_min']}–{z['zone_max']}"
               if z['zone_min'] is not None else "-")
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(z["distribution"].items()))
        lines.append(f"| {z['zone']} | {rng} | {z['n']} | {dist_str} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-zones", type=int, default=10)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(args.verdicts_path)
    print(f"{len(rows)} verdicts loaded")
    if not rows:
        print("no verdicts — empty input")
        return 1

    overall = _overall_distribution(rows)

    binary_tables = {
        axis: {metric: _zone_rates(rows, metric, axis, n_zones=args.n_zones)
               for metric in METRICS}
        for axis in BINARY_AXES
    }
    enum_tables = {
        axis: {metric: _enum_zone_rates(rows, metric, axis, n_zones=args.n_zones)
               for metric in METRICS}
        for axis in ENUM_AXES
    }

    summary = {
        "n_verdicts": len(rows),
        "overall_distribution": overall,
        "binary_yes_rates": binary_tables,
        "enum_distributions": enum_tables,
        "contract": (
            "No pre-committed thresholds. Read the per-metric zone "
            "yes-rates to identify natural cutoffs where yes-rate "
            "crosses ~50% or shows a step-change. Set policy post-hoc."
        ),
    }
    (args.output_dir / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Broken-text quality review — distributions",
        "",
        f"Verdicts analyzed: **{len(rows)}**",
        "",
        "**Contract**: no pre-committed thresholds. This report shows the "
        "SHAPE of each label-vs-metric distribution. Read the zone "
        "yes-rate columns below to identify natural cutoffs (where "
        "yes-rate crosses ~50% or shows a step-change) and set "
        "rejection policy post-hoc.",
        "",
        "## Overall verdict distribution",
        "",
    ]
    for axis, dist in overall.items():
        md_lines.append(f"- **{axis}**: {dist}")
    md_lines.append("")

    md_lines.append("## Binary axes (yes-rate per metric zone)")
    md_lines.append("")
    for axis in BINARY_AXES:
        for metric in METRICS:
            md_lines.append(_format_binary_zone_table(
                binary_tables[axis][metric], metric, axis))
        md_lines.append("")

    md_lines.append("## Enum axes (distribution per metric zone)")
    md_lines.append("")
    for axis in ENUM_AXES:
        for metric in METRICS:
            md_lines.append(_format_enum_zone_table(
                enum_tables[axis][metric], metric, axis))
        md_lines.append("")

    (args.output_dir / "quality_distributions.md").write_text(
        "\n".join(md_lines), encoding="utf-8",
    )
    print(f"summary → {args.output_dir}/quality_summary.json")
    print(f"report  → {args.output_dir}/quality_distributions.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
