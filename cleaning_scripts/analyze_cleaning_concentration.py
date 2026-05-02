"""Analyze per-dataset + per-doc cleaning concentration.

Reads the per-doc stats.jsonl files produced by clean_and_stats_full.py
and reports:

  1. Per-dataset rollup (source_dataset field):
     - n_docs total / kept / dropped (by reason)
     - chars_before total
     - chars_after total
     - chars_removed total (strip only — dropped docs are separate)
     - chars_removed_by_drop (from doc drops)
     - pct_removed_by_strip
     - pct_removed_total (strip + drop)

  2. Top-N docs by pct_removed within each dataset (non-dropped docs only,
     to see where per-line strip concentrates in surviving docs).

  3. Top-N docs by absolute chars_removed (cross-dataset).

  4. Counter-value histograms per dataset — which counter fires most for
     each source.

Output: analysis.json + analysis.md at the stats dir root.

Run:
  python3 analyze_cleaning_concentration.py \\
    --stats-dir /home/foivos/runs/raw_clean_stats_20260422/stats \\
    --output-dir /home/foivos/runs/raw_clean_stats_20260422/analysis
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Dict, List, Optional


@dataclass
class DatasetRoll:
    source_dataset: str
    n_docs: int = 0
    n_kept: int = 0
    n_dropped: Dict[str, int] = field(default_factory=dict)
    chars_before: int = 0
    chars_after: int = 0
    chars_removed_strip: int = 0   # per-line strip on surviving docs
    chars_removed_drop: int = 0    # full doc chars for dropped docs
    pct_removed_per_doc: List[float] = field(default_factory=list)
    counter_font: List[int] = field(default_factory=list)
    counter_glyph: List[int] = field(default_factory=list)
    counter_script: List[int] = field(default_factory=list)


def _load_stats(stats_dir: Path) -> Dict[str, DatasetRoll]:
    rolls: Dict[str, DatasetRoll] = {}
    for p in sorted(stats_dir.glob("*.stats.jsonl")):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                ds = d.get("source_dataset") or p.stem.split(".")[0]
                roll = rolls.setdefault(ds, DatasetRoll(source_dataset=ds))
                roll.n_docs += 1
                drop_reason = d.get("drop_reason") or ""
                chars_before = int(d.get("chars_before", 0) or 0)
                chars_after = int(d.get("chars_after", 0) or 0)
                chars_removed = int(d.get("chars_removed", 0) or 0)
                roll.counter_font.append(int(d.get("counter_font_marker", 0) or 0))
                roll.counter_glyph.append(int(d.get("counter_glyph_marker", 0) or 0))
                roll.counter_script.append(int(d.get("counter_script_residue", 0) or 0))
                if drop_reason:
                    roll.n_dropped[drop_reason] = roll.n_dropped.get(drop_reason, 0) + 1
                    roll.chars_removed_drop += chars_before
                else:
                    roll.n_kept += 1
                    roll.chars_before += chars_before
                    roll.chars_after += chars_after
                    roll.chars_removed_strip += chars_removed
                    pct = float(d.get("pct_removed", 0.0) or 0.0)
                    roll.pct_removed_per_doc.append(pct)
    return rolls


def _quantiles_safe(data: List[float], n: int = 4) -> List[float]:
    if len(data) < n:
        return []
    return quantiles(sorted(data), n=n)


def _summarize(rolls: Dict[str, DatasetRoll]) -> Dict[str, Any]:
    out_per_ds: Dict[str, Any] = {}
    total = DatasetRoll(source_dataset="__total__")
    for ds, r in rolls.items():
        qs = _quantiles_safe(r.pct_removed_per_doc, n=4)
        out_per_ds[ds] = {
            "n_docs": r.n_docs,
            "n_kept": r.n_kept,
            "n_dropped_by_reason": r.n_dropped,
            "chars_before_strip_kept": r.chars_before,
            "chars_after_strip_kept": r.chars_after,
            "chars_removed_by_strip": r.chars_removed_strip,
            "chars_removed_by_drop": r.chars_removed_drop,
            "chars_removed_total": r.chars_removed_strip + r.chars_removed_drop,
            "pct_removed_by_strip": round(
                100.0 * r.chars_removed_strip / max(r.chars_before, 1), 3),
            "pct_removed_total": round(
                100.0 * (r.chars_removed_strip + r.chars_removed_drop)
                / max(r.chars_before + r.chars_removed_drop, 1), 3),
            "per_doc_pct_removed_quartiles": [round(q, 3) for q in qs] if qs else None,
            "per_doc_pct_removed_mean": round(mean(r.pct_removed_per_doc), 3)
                if r.pct_removed_per_doc else None,
            "counter_font_mean": round(mean(r.counter_font), 3)
                if r.counter_font else 0,
            "counter_font_max": max(r.counter_font) if r.counter_font else 0,
            "counter_glyph_mean": round(mean(r.counter_glyph), 3)
                if r.counter_glyph else 0,
            "counter_glyph_max": max(r.counter_glyph) if r.counter_glyph else 0,
            "counter_script_mean": round(mean(r.counter_script), 3)
                if r.counter_script else 0,
            "counter_script_max": max(r.counter_script) if r.counter_script else 0,
        }
        total.n_docs += r.n_docs
        total.n_kept += r.n_kept
        total.chars_before += r.chars_before
        total.chars_after += r.chars_after
        total.chars_removed_strip += r.chars_removed_strip
        total.chars_removed_drop += r.chars_removed_drop
        for k, v in r.n_dropped.items():
            total.n_dropped[k] = total.n_dropped.get(k, 0) + v

    out_per_ds["__total__"] = {
        "n_docs": total.n_docs,
        "n_kept": total.n_kept,
        "n_dropped_by_reason": total.n_dropped,
        "chars_before_strip_kept": total.chars_before,
        "chars_after_strip_kept": total.chars_after,
        "chars_removed_by_strip": total.chars_removed_strip,
        "chars_removed_by_drop": total.chars_removed_drop,
        "chars_removed_total": total.chars_removed_strip + total.chars_removed_drop,
        "pct_removed_by_strip": round(
            100.0 * total.chars_removed_strip / max(total.chars_before, 1), 3),
        "pct_removed_total": round(
            100.0 * (total.chars_removed_strip + total.chars_removed_drop)
            / max(total.chars_before + total.chars_removed_drop, 1), 3),
    }
    return out_per_ds


def _top_docs(stats_dir: Path, n: int = 30) -> Dict[str, List[Dict[str, Any]]]:
    """Cross-dataset top-N docs by absolute chars_removed AND by pct_removed."""
    top_abs: List[Dict[str, Any]] = []
    top_pct: List[Dict[str, Any]] = []
    for p in sorted(stats_dir.glob("*.stats.jsonl")):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("drop_reason"):
                    continue   # only non-dropped docs — strip only
                chars_removed = int(d.get("chars_removed", 0) or 0)
                chars_before = int(d.get("chars_before", 0) or 0)
                pct = float(d.get("pct_removed", 0) or 0)
                row = {
                    "source_path": d.get("source_path"),
                    "source_dataset": d.get("source_dataset"),
                    "source_doc_id": d.get("source_doc_id"),
                    "chars_before": chars_before,
                    "chars_removed": chars_removed,
                    "pct_removed": pct,
                    "counter_font": d.get("counter_font_marker"),
                    "counter_glyph": d.get("counter_glyph_marker"),
                    "counter_script": d.get("counter_script_residue"),
                }
                top_abs.append(row)
                # Only include in pct ranking if chars_before >= 1000 to avoid
                # trivial small docs where 100% removal is uninformative.
                if chars_before >= 1000:
                    top_pct.append(row)
    top_abs.sort(key=lambda r: -r["chars_removed"])
    top_pct.sort(key=lambda r: -r["pct_removed"])
    return {
        "top_by_abs_chars_removed": top_abs[:n],
        "top_by_pct_removed_min_1kchar": top_pct[:n],
    }


def _format_md(per_ds: Dict[str, Any], top: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Cleaning concentration — per-dataset + per-doc\n")
    lines.append("## Per-dataset summary\n")
    lines.append("| dataset | docs | kept | dropped | chars_before | chars_after | strip % | total % | glyph_mean | script_mean | font_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    # Sort by pct_removed_total descending
    items = sorted(
        [(k, v) for k, v in per_ds.items() if k != "__total__"],
        key=lambda kv: -kv[1].get("pct_removed_total", 0),
    )
    for ds, s in items:
        lines.append(
            f"| {ds} | {s['n_docs']} | {s['n_kept']} | "
            f"{sum(s['n_dropped_by_reason'].values())} | "
            f"{s['chars_before_strip_kept']:,} | {s['chars_after_strip_kept']:,} | "
            f"{s['pct_removed_by_strip']}% | {s['pct_removed_total']}% | "
            f"{s.get('counter_glyph_mean', 0)} | "
            f"{s.get('counter_script_mean', 0)} | "
            f"{s.get('counter_font_mean', 0)} |"
        )
    t = per_ds["__total__"]
    lines.append(
        f"| **TOTAL** | **{t['n_docs']}** | **{t['n_kept']}** | "
        f"**{sum(t['n_dropped_by_reason'].values())}** | "
        f"**{t['chars_before_strip_kept']:,}** | **{t['chars_after_strip_kept']:,}** | "
        f"**{t['pct_removed_by_strip']}%** | **{t['pct_removed_total']}%** | | | |"
    )

    lines.append("\n## Top 30 docs by absolute chars_removed (strip only, non-dropped)\n")
    lines.append("| rank | dataset | doc_id | chars_before | chars_removed | pct | glyph | script | font |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(top["top_by_abs_chars_removed"], 1):
        lines.append(
            f"| {i} | {r['source_dataset']} | {r['source_doc_id']} | "
            f"{r['chars_before']:,} | {r['chars_removed']:,} | "
            f"{r['pct_removed']}% | {r['counter_glyph']} | "
            f"{r['counter_script']} | {r['counter_font']} |"
        )
    lines.append("\n## Top 30 docs by pct_removed (min 1k chars, non-dropped)\n")
    lines.append("| rank | dataset | doc_id | chars_before | chars_removed | pct | glyph | script | font |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(top["top_by_pct_removed_min_1kchar"], 1):
        lines.append(
            f"| {i} | {r['source_dataset']} | {r['source_doc_id']} | "
            f"{r['chars_before']:,} | {r['chars_removed']:,} | "
            f"{r['pct_removed']}% | {r['counter_glyph']} | "
            f"{r['counter_script']} | {r['counter_font']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args(argv)

    rolls = _load_stats(args.stats_dir)
    per_ds = _summarize(rolls)
    top = _top_docs(args.stats_dir, n=args.top_n)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps({"per_dataset": per_ds, **top}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "analysis.md").write_text(_format_md(per_ds, top), encoding="utf-8")
    print(f"wrote {args.output_dir / 'analysis.json'}")
    print(f"wrote {args.output_dir / 'analysis.md'}")
    t = per_ds["__total__"]
    print(
        f"\nTOTAL: docs={t['n_docs']} kept={t['n_kept']} "
        f"dropped={sum(t['n_dropped_by_reason'].values())}   "
        f"chars_removed_strip={t['chars_removed_by_strip']:,} ({t['pct_removed_by_strip']}%)  "
        f"chars_removed_drop={t['chars_removed_by_drop']:,}  "
        f"total_pct_removed={t['pct_removed_total']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
