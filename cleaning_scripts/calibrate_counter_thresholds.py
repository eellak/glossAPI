"""Calibrate per-counter drop thresholds from Gemini verdicts.

Input:
  A `verdicts.jsonl` from gemini_three_counter_reviewer.py. Each row carries
  `meta.counters` (the three counter values for that page) and
  `verdict.keep_or_drop`.

For each counter (font_marker, glyph_marker, script_residue) we:
  1. Group cases by zone of that counter (10 zones matching the sampler).
  2. Compute per-zone drop-rate = drops / total_verdicts.
  3. Find the threshold := the smallest counter value at which drop-rate
     >= TARGET_DROP_RATE (default 0.80). Both zone-edge and per-case
     interpolation are reported.

Also surfaces "unknown-signal detection":
  Fraction of cases where the model answered `noise_character ==
  garbled_text_other` or `dominant_signal == other_unknown`. If high,
  we're missing a counter.

Output:
  `thresholds.json` at the sample-dir root, consumable by the cleaner's
  `drop_low_salvage_pages` (or a new `drop_pages_by_counters`) config.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


COUNTERS = ("font_marker", "glyph_marker", "script_residue")


def _load_aggregate(path: Path) -> Dict[str, Any]:
    return json.loads((path / "aggregate.json").read_text(encoding="utf-8"))


def _load_verdicts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with (path / "verdicts.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def _zone_for_value(v: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if (i < len(edges) - 2 and lo <= v < hi) or (i == len(edges) - 2 and lo <= v <= hi):
            return i
    return len(edges) - 2


def _calibrate_one(
    counter_name: str,
    verdicts: List[Dict[str, Any]],
    zone_edges: List[float],
    target_drop_rate: float,
) -> Dict[str, Any]:
    # Zone-level aggregation
    zone_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"drop": 0, "keep": 0, "uncertain": 0})
    zone_values: Dict[int, List[int]] = defaultdict(list)
    per_case: List[Dict[str, Any]] = []
    for row in verdicts:
        meta = row.get("meta") or {}
        if meta.get("primary_counter") != counter_name:
            continue
        verdict = row.get("verdict") or {}
        decision = verdict.get("keep_or_drop", "uncertain")
        counters = meta.get("counters") or {}
        counter_value = int(counters.get(counter_name, 0))
        zone = _zone_for_value(counter_value, zone_edges)
        zone_counts[zone][decision] += 1
        zone_values[zone].append(counter_value)
        per_case.append(
            {
                "counter_value": counter_value,
                "zone": zone,
                "keep_or_drop": decision,
                "noise_character": verdict.get("noise_character"),
                "dominant_signal": verdict.get("dominant_signal"),
                "short_reason": verdict.get("short_reason"),
            }
        )

    zone_rates: List[Dict[str, Any]] = []
    threshold_value: Optional[int] = None
    for i in range(len(zone_edges) - 1):
        c = zone_counts.get(i, {})
        total = c.get("drop", 0) + c.get("keep", 0) + c.get("uncertain", 0)
        drop_rate = (c.get("drop", 0) / total) if total else None
        zone_rates.append(
            {
                "zone": i,
                "edge_lo": zone_edges[i],
                "edge_hi": zone_edges[i + 1],
                "min_value_in_zone": min(zone_values.get(i, [0]), default=None),
                "max_value_in_zone": max(zone_values.get(i, [0]), default=None),
                "drop_count": c.get("drop", 0),
                "keep_count": c.get("keep", 0),
                "uncertain_count": c.get("uncertain", 0),
                "total": total,
                "drop_rate": drop_rate,
            }
        )
        if threshold_value is None and drop_rate is not None and drop_rate >= target_drop_rate:
            threshold_value = int(math.ceil(zone_edges[i]))

    # Per-case threshold: lowest counter value at which `drop` decisions
    # empirically exceed `target_drop_rate` in a rolling window.
    per_case.sort(key=lambda r: r["counter_value"])
    rolling_threshold: Optional[int] = None
    window = 10
    for i in range(len(per_case) - window + 1):
        sub = per_case[i : i + window]
        drops = sum(1 for x in sub if x["keep_or_drop"] == "drop")
        if drops / window >= target_drop_rate:
            rolling_threshold = sub[0]["counter_value"]
            break

    return {
        "counter": counter_name,
        "target_drop_rate": target_drop_rate,
        "zone_rates": zone_rates,
        "zone_edge_threshold": threshold_value,
        "rolling_window_threshold": rolling_threshold,
        "n_cases": len(per_case),
        "n_drop": sum(1 for r in per_case if r["keep_or_drop"] == "drop"),
        "n_keep": sum(1 for r in per_case if r["keep_or_drop"] == "keep"),
        "n_uncertain": sum(1 for r in per_case if r["keep_or_drop"] == "uncertain"),
        "n_unknown_noise_char": sum(
            1 for r in per_case if r.get("noise_character") == "garbled_text_other"
        ),
        "n_unknown_dominant_signal": sum(
            1 for r in per_case if r.get("dominant_signal") == "other_unknown"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--target-drop-rate", type=float, default=0.80)
    args = parser.parse_args(argv)

    aggregate = _load_aggregate(args.sample_dir)
    verdicts = _load_verdicts(args.sample_dir)
    print(f"loaded {len(verdicts)} verdicts from {args.sample_dir / 'verdicts.jsonl'}")

    results: Dict[str, Any] = {
        "target_drop_rate": args.target_drop_rate,
        "verdict_count": len(verdicts),
        "per_counter": {},
    }
    for counter_name in COUNTERS:
        counter_meta = aggregate["counters"].get(counter_name) or {}
        zone_edges = counter_meta.get("zone_edges") or []
        if not zone_edges:
            print(f"  {counter_name}: skipped (no zone edges in aggregate)")
            continue
        calib = _calibrate_one(counter_name, verdicts, zone_edges, args.target_drop_rate)
        results["per_counter"][counter_name] = calib
        print(
            f"  {counter_name}: "
            f"n={calib['n_cases']} "
            f"drop/keep/unc={calib['n_drop']}/{calib['n_keep']}/{calib['n_uncertain']} "
            f"zone_thr={calib['zone_edge_threshold']} "
            f"rolling_thr={calib['rolling_window_threshold']} "
            f"unknown_noise={calib['n_unknown_noise_char']} "
            f"unknown_sig={calib['n_unknown_dominant_signal']}"
        )

    # Also surface a single consolidated threshold-set the cleaner can use.
    results["suggested_thresholds"] = {
        name: (
            (results["per_counter"].get(name) or {}).get("rolling_window_threshold")
            or (results["per_counter"].get(name) or {}).get("zone_edge_threshold")
        )
        for name in COUNTERS
    }

    out_path = args.sample_dir / "thresholds.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"suggested thresholds: {json.dumps(results['suggested_thresholds'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
