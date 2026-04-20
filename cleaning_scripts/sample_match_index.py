#!/usr/bin/env python3
"""Stream a large match_index.jsonl and reservoir-sample per category, then
emit a filtered page_metrics.jsonl containing only the pages the sampled
matches belong to. Output sizes drop by ~5 orders of magnitude, making the
existing bundler usable on corpus-scale matcher output.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _cats(row: Dict[str, Any]) -> List[str]:
    raw = row.get("categories")
    if isinstance(raw, (list, tuple)) and raw:
        return [str(x) for x in raw]
    cat = row.get("category")
    if isinstance(cat, str) and cat:
        return [cat]
    return []


def _page_key_tuple(row: Dict[str, Any]) -> tuple:
    return (
        str(row.get("source_stem", "")),
        str(row.get("page_kind", "")),
        int(row.get("page_number", 0) or 0),
        int(row.get("page_index_in_file", 0) or 0),
    )


def reservoir_sample_match_index(
    in_path: Path,
    *,
    per_category: int,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Dict[str, int] = defaultdict(int)
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            for cat in _cats(row):
                seen[cat] += 1
                bucket = buckets[cat]
                if len(bucket) < per_category:
                    bucket.append(row)
                else:
                    j = rng.randrange(seen[cat])
                    if j < per_category:
                        bucket[j] = row
    for cat, total in sorted(seen.items(), key=lambda kv: -kv[1]):
        print(f"category {cat!r}: sampled {len(buckets[cat]):>5} of {total:>12,}", file=sys.stderr)
    return buckets


def collect_page_keys(buckets: Dict[str, List[Dict[str, Any]]]) -> set:
    keys = set()
    for rows in buckets.values():
        for row in rows:
            keys.add(_page_key_tuple(row))
    return keys


def filter_page_metrics(in_path: Path, page_keys: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if _page_key_tuple(row) in page_keys:
                out.append(row)
    print(f"page_metrics filter: kept {len(out):>8} of (streamed)", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-category", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = args.input_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    match_in = in_dir / "match_index.jsonl"
    page_in = in_dir / "page_metrics.jsonl"
    summary_in = in_dir / "summary.json"

    buckets = reservoir_sample_match_index(match_in, per_category=args.per_category, seed=args.seed)
    page_keys = collect_page_keys(buckets)
    page_rows = filter_page_metrics(page_in, page_keys)

    with (out_dir / "match_index.jsonl").open("w", encoding="utf-8") as f:
        for rows in buckets.values():
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "page_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in page_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Empty manifest — bundler regenerates it per-sample.
    (out_dir / "manifest.jsonl").write_text("", encoding="utf-8")
    if summary_in.exists():
        (out_dir / "summary.json").write_text(summary_in.read_text(encoding="utf-8"), encoding="utf-8")

    per_cat = {cat: len(rows) for cat, rows in buckets.items()}
    (out_dir / "sampler_summary.json").write_text(
        json.dumps(
            {
                "source": str(in_dir),
                "per_category_cap": args.per_category,
                "seed": args.seed,
                "per_category_sampled": per_cat,
                "page_metrics_kept": len(page_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
