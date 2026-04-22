"""Stratified sampler for post-cleaning quality review.

Input:
  --stats-glob   glob of *.stats.jsonl files produced by
                 clean_and_stats_rowsharded.py
  --parquet-glob glob of original parquet files (for text re-extraction,
                 so we can clean with newlines preserved; the
                 .txt.gz shards from clean_and_stats_* flatten the doc
                 onto a single line which loses the line-boundary signal
                 Gemini needs to judge narrative jumps)
  --output-dir   output dir for sample.jsonl + per-metric zone breakdown
  --sample-size  default 150 — 10 stratified zones × 4 axes, approximate

Stratifies INDEPENDENTLY on four metrics (per
`feedback_multi_axis_gemini_prompts.md` + `feedback_stratified_sampling.md`):

  1. char_strip_ratio = chars_dropped_by_per_char_filter / non_empty_chars_in
     (correlates with Gemini Q4 has_broken_words_mid_token)
  2. line_drop_ratio  = lines_dropped_by_cleaner / non_empty_lines_in
     (correlates with Q5 has_narrative_jumps_from_line_drops)
  3. non_empty_chars_out  (absolute surviving chars — correlates with Q7
     is_too_short_to_be_useful)
  4. pct_chars_removed_non_empty  (combined signal, already emitted)

Each axis → 10 zones over its observed distribution quantiles → ~4 docs
per zone per axis → ~160 raw samples → dedup by doc_id → target 150.

NO HARD THRESHOLDS anywhere. The sampler is distribution-driven: zone
boundaries come from quantiles over the corpus, not from a fixed
value. We look at shape before setting any cutoff (user direction
2026-04-22).

Output per sampled doc: {source_path, source_dataset, source_doc_id,
stats snapshot, axis_zones, cleaned_text (with newlines preserved)}.
"""
from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _load_stats(stats_glob: str, altered_only: bool = True) -> List[Dict[str, Any]]:
    """Load all per-doc stats records and filter to the target population.

    Target population (per 2026-04-22 design note):
      docs that actually experienced cleaning damage — i.e. non-zero
      chars removed OR at least one line dropped. Docs that passed
      through untouched don't help calibrate cutoffs because there's
      nothing to judge.
    """
    out = []
    for path in sorted(globmod.glob(stats_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("drop_reason"):
                    # Skip pre-cleaner-drop docs (empty / counter-triggered
                    # doc-level drop) — we want to review docs that SURVIVED
                    # cleaning, not docs that were doc-rejected.
                    continue
                # Only keep docs that have the new four-way stats — older
                # runs won't.
                if "content_chars_kept" not in d:
                    continue
                if altered_only:
                    pct_chars = float(d.get("pct_chars_removed_non_empty", 0) or 0)
                    lines_dropped = int(d.get("lines_dropped_by_cleaner", 0) or 0)
                    if pct_chars <= 0 and lines_dropped <= 0:
                        continue
                out.append(d)
    return out


def _compute_metrics(rec: Dict[str, Any]) -> Dict[str, float]:
    non_empty_chars_in = max(int(rec.get("non_empty_chars_in", 0) or 0), 1)
    non_empty_lines_in = max(int(rec.get("non_empty_lines_in", 0) or 0), 1)
    return {
        "char_strip_ratio": float(rec.get("chars_dropped_by_per_char_filter", 0) or 0) / non_empty_chars_in,
        "line_drop_ratio": float(rec.get("lines_dropped_by_cleaner", 0) or 0) / non_empty_lines_in,
        "non_empty_chars_out": float(rec.get("non_empty_chars_out", 0) or 0),
        "pct_chars_removed_non_empty": float(rec.get("pct_chars_removed_non_empty", 0) or 0),
    }


def _quantile_zones(values: List[float], n_zones: int = 10) -> List[float]:
    if not values:
        return []
    xs = sorted(values)
    # n_zones-1 internal cut-points at quantiles 1/n..(n-1)/n.
    return [xs[int(len(xs) * q / n_zones)] for q in range(1, n_zones)]


def _zone_of(value: float, cuts: List[float]) -> int:
    for i, c in enumerate(cuts):
        if value < c:
            return i
    return len(cuts)


def _load_parquet_index(parquet_glob: str) -> Dict[str, str]:
    """Map parquet-path (as stored in stats.source_path prefix) → absolute
    path. `source_path` is stored as `<parquet>#<doc_id>`."""
    index = {}
    for p in sorted(globmod.glob(parquet_glob)):
        index[p] = p
    return index


def _find_row(parquet_path: str, doc_id: str, doc_id_column: str, text_column: str) -> Optional[Dict[str, Any]]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            if str(row.get(doc_id_column)) == str(doc_id):
                return {"text": row.get(text_column) or "", "row": row}
    return None


def _clean_with_newlines(text: str, scripts_to_keep: List[str]) -> str:
    """Run the cleaner and return output with newlines preserved. This
    is the representation Gemini needs — the .txt.gz shards flatten
    newlines to spaces."""
    import glossapi_rs_cleaner as cleaner
    return cleaner.clean_text(text, scripts_to_keep)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--parquet-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Total target sample (default 500 — gives ~30/zone across "
                             "10 zones for each of 4 marginal-axis estimates at ~±17%% CI)")
    parser.add_argument("--min-per-dataset", type=int, default=15,
                        help="Per-dataset floor — every dataset gets at least this many "
                             "samples regardless of stratification. Default 15.")
    parser.add_argument("--altered-only", action="store_true", default=True,
                        help="Only sample docs with pct_chars_removed_non_empty > 0 OR "
                             "lines_dropped_by_cleaner > 0 (default on).")
    parser.add_argument("--include-unaltered", dest="altered_only", action="store_false",
                        help="Include docs with zero cleaning damage.")
    parser.add_argument("--doc-id-column", default="doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--scripts", nargs="+", default=DEFAULT_SCRIPTS)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--n-zones", type=int, default=10)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"loading stats from {args.stats_glob} "
          f"(altered_only={args.altered_only}) ...", flush=True)
    stats = _load_stats(args.stats_glob, altered_only=args.altered_only)
    print(f"  population size: {len(stats)} docs")
    if not stats:
        print("  nothing to sample. did you run clean_and_stats_rowsharded "
              "after the 2026-04-22 four-way-accounting change?",
              file=sys.stderr)
        return 2

    # Compute metrics + quantile cut-points per axis.
    axes = ["char_strip_ratio", "line_drop_ratio", "non_empty_chars_out",
            "pct_chars_removed_non_empty"]
    per_doc_metrics = {i: _compute_metrics(rec) for i, rec in enumerate(stats)}
    axis_cuts: Dict[str, List[float]] = {}
    for axis in axes:
        values = [per_doc_metrics[i][axis] for i in range(len(stats))]
        axis_cuts[axis] = _quantile_zones(values, n_zones=args.n_zones)

    # Per axis: partition docs into zones and sample ~per_axis_budget / n_zones
    # per zone.
    per_axis_budget = args.sample_size // len(axes) + 5
    per_zone_budget = max(per_axis_budget // args.n_zones, 2)
    sample_ids = set()
    per_zone_picks = defaultdict(list)  # (axis, zone) → [doc_idx]
    for axis in axes:
        zones: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(stats)):
            z = _zone_of(per_doc_metrics[i][axis], axis_cuts[axis])
            zones[z].append(i)
        for z, idxs in zones.items():
            rng.shuffle(idxs)
            picks = idxs[:per_zone_budget]
            per_zone_picks[(axis, z)].extend(picks)
            sample_ids.update(picks)

    # Per-dataset floor: every dataset gets at least --min-per-dataset picks.
    # Supplements stratified zone picks so rare datasets don't get under-represented.
    by_dataset: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(stats):
        by_dataset[str(rec.get("source_dataset", "unknown"))].append(i)
    dataset_picks = defaultdict(list)
    for ds, idxs in by_dataset.items():
        rng.shuffle(idxs)
        existing = [i for i in idxs if i in sample_ids]
        need = max(args.min_per_dataset - len(existing), 0)
        extra = [i for i in idxs if i not in sample_ids][:need]
        dataset_picks[ds].extend(existing + extra)
        sample_ids.update(extra)

    # Trim to exactly sample_size while preserving per-dataset floor.
    floor_ids = set()
    for ds_list in dataset_picks.values():
        floor_ids.update(ds_list[: args.min_per_dataset])
    remaining_budget = max(args.sample_size - len(floor_ids), 0)
    extras = [i for i in sample_ids if i not in floor_ids]
    rng.shuffle(extras)
    picks_list = list(floor_ids) + extras[:remaining_budget]
    rng.shuffle(picks_list)
    print(f"stratified to {len(picks_list)} unique docs "
          f"({len(per_zone_picks)} zones across {len(axes)} axes, "
          f"{len(by_dataset)} datasets covered at ≥{args.min_per_dataset} each)")

    # Build output. Per pick: re-load parquet row, re-run cleaner with newlines.
    output_path = args.output_dir / "sample.jsonl"
    n_written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for i in picks_list:
            rec = stats[i]
            source_path = rec.get("source_path", "")
            parquet_path, _, doc_id = source_path.partition("#")
            if not Path(parquet_path).is_file():
                continue
            row_data = _find_row(parquet_path, doc_id,
                                 args.doc_id_column, args.text_column)
            if row_data is None:
                continue
            raw_text = row_data["text"]
            cleaned = _clean_with_newlines(raw_text, args.scripts)
            zones_membership = {
                axis: _zone_of(per_doc_metrics[i][axis], axis_cuts[axis])
                for axis in axes
            }
            out.write(json.dumps({
                "source_path": source_path,
                "source_dataset": rec.get("source_dataset"),
                "source_doc_id": rec.get("source_doc_id"),
                "metrics": per_doc_metrics[i],
                "zones": zones_membership,
                "stats_snapshot": {k: rec.get(k) for k in [
                    "chars_before", "chars_after", "chars_removed",
                    "pct_removed",
                    "content_chars_kept",
                    "chars_dropped_by_line_drop",
                    "chars_dropped_by_normalization",
                    "chars_dropped_by_per_char_filter",
                    "lines_dropped_by_cleaner",
                    "marker_chars_passthrough",
                    "marker_chars_added",
                    "non_empty_lines_in", "non_empty_lines_out",
                    "non_empty_chars_in", "non_empty_chars_out",
                    "pct_chars_removed_non_empty",
                    "pct_lines_removed_non_empty",
                ]},
                "cleaned_text": cleaned,
            }, ensure_ascii=False) + "\n")
            n_written += 1
            if n_written % 25 == 0:
                print(f"  wrote {n_written}/{len(picks_list)}", flush=True)

    # Zone breakdown for the log.
    zone_summary_path = args.output_dir / "zone_breakdown.md"
    with zone_summary_path.open("w", encoding="utf-8") as fh:
        fh.write("# Stratification zone breakdown\n\n")
        for axis in axes:
            fh.write(f"## {axis}\n\n")
            fh.write(f"Quantile cuts: {axis_cuts[axis]}\n\n")
            fh.write("| zone | picks |\n|---|---:|\n")
            for z in range(args.n_zones):
                picks = per_zone_picks.get((axis, z), [])
                fh.write(f"| {z} | {len(picks)} |\n")
            fh.write("\n")
    print(f"wrote {n_written} to {output_path}")
    print(f"zone breakdown → {zone_summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
