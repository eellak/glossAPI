"""Post-run analyzer for the 3-rule charset filter. Reports:

- total docs, kept / dropped counts by reason
- per-rule exclusion counts (moji / punct / greek_low) and overlaps
- per-dataset breakdown
- distribution of each ratio across the kept pool (for post-hoc
  threshold recalibration)

Run as:
  python3 analyze_charset_filter_impact.py \
    --stats-glob '<run_dir>/stats/*.jsonl' \
    --output-dir <report_dir>
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _iter_stats(stats_glob: str):
    for path in sorted(globmod.glob(stats_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    drop_counts: Dict[str, int] = defaultdict(int)
    per_dataset_drops: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_dataset_total: Dict[str, int] = defaultdict(int)
    kept_greek_ratios: List[float] = []
    kept_moji_ratios: List[float] = []
    kept_punct_ratios: List[float] = []

    for d in _iter_stats(args.stats_glob):
        total += 1
        ds = d.get("source_dataset", "unknown")
        per_dataset_total[ds] += 1
        reason = d.get("drop_reason") or ""
        if reason:
            drop_counts[reason] += 1
            per_dataset_drops[ds][reason] += 1
        else:
            if "charset_greek_ratio" in d:
                kept_greek_ratios.append(float(d["charset_greek_ratio"]))
                kept_moji_ratios.append(float(d["charset_moji_ratio"]))
                kept_punct_ratios.append(float(d["charset_punct_ratio"]))

    kept = total - sum(drop_counts.values())

    # Narrative.
    md = [f"# Charset filter impact", "",
          f"Total rows: **{total}**",
          f"Kept after all filters: **{kept}** ({100*kept/max(total,1):.2f}%)",
          "",
          "## Drop reasons", "",
          "| reason | count | % of total |", "|---|---:|---:|"]
    for reason, n in sorted(drop_counts.items(), key=lambda x: -x[1]):
        md.append(f"| {reason} | {n} | {100*n/max(total,1):.2f}% |")

    md.extend(["", "## Per-dataset breakdown (drops only)", "",
               "| dataset | total | kept | charset_moji | charset_punct | charset_greek_low | counter drops | other |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for ds in sorted(per_dataset_total.keys(), key=lambda k: -per_dataset_total[k]):
        n_total = per_dataset_total[ds]
        drops = per_dataset_drops[ds]
        kept_ds = n_total - sum(drops.values())
        moji = drops.get("charset_moji", 0)
        punct = drops.get("charset_punct", 0)
        greek_low = drops.get("charset_greek_low", 0)
        counters = sum(v for k, v in drops.items() if k.startswith("counter:"))
        other = sum(drops.values()) - moji - punct - greek_low - counters
        md.append(f"| {ds} | {n_total} | {kept_ds} | {moji} | {punct} | {greek_low} | {counters} | {other} |")

    # Distribution of ratios on kept docs (post-hoc threshold sanity).
    md.extend(["", "## Charset-ratio distribution on KEPT docs", ""])
    for name, vals in [("greek_letter_ratio", kept_greek_ratios),
                       ("moji_residue_ratio", kept_moji_ratios),
                       ("ascii_punct_ratio", kept_punct_ratios)]:
        if not vals:
            md.append(f"- **{name}**: 0 kept docs with this field")
            continue
        a = np.array(vals)
        md.append(f"- **{name}**: N={len(a)} "
                  f"p25={np.quantile(a,0.25):.3f} p50={np.quantile(a,0.5):.3f} "
                  f"p75={np.quantile(a,0.75):.3f} p95={np.quantile(a,0.95):.3f} "
                  f"p99={np.quantile(a,0.99):.3f} max={a.max():.3f}")

    report_path = args.output_dir / "charset_filter_impact.md"
    report_path.write_text("\n".join(md), encoding="utf-8")

    summary = {
        "total": total,
        "kept": kept,
        "drop_counts": dict(drop_counts),
        "per_dataset_total": dict(per_dataset_total),
        "per_dataset_drops": {k: dict(v) for k, v in per_dataset_drops.items()},
        "kept_ratio_stats": {
            name: {
                "n": len(vals),
                "p25": float(np.quantile(vals, 0.25)) if vals else None,
                "p50": float(np.quantile(vals, 0.5)) if vals else None,
                "p75": float(np.quantile(vals, 0.75)) if vals else None,
                "p95": float(np.quantile(vals, 0.95)) if vals else None,
                "p99": float(np.quantile(vals, 0.99)) if vals else None,
                "max": float(max(vals)) if vals else None,
            }
            for name, vals in [("greek_letter_ratio", kept_greek_ratios),
                               ("moji_residue_ratio", kept_moji_ratios),
                               ("ascii_punct_ratio", kept_punct_ratios)]
        },
    }
    (args.output_dir / "charset_filter_impact.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"report → {report_path}")
    print(f"summary → {args.output_dir / 'charset_filter_impact.json'}")
    print()
    print(f"total={total} kept={kept} ({100*kept/max(total,1):.1f}%)")
    for r, n in sorted(drop_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
