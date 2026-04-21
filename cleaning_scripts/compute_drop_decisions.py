"""Lightweight Phase-5 alternative — compute drop decisions per doc
without writing the full cleaned corpus out.

Rationale: v2 corpus is 804 GB; writing v3 alongside it would need ~200+
GB of free disk we don't have. Instead we emit a compact
`drop_decisions.jsonl` keyed by source_doc_id with per-counter values
and the drop verdict. Phase-6 BPE training then streams v2 → matcher →
cleaner → trainer iterator in one pass, filtering by the drop decisions
on the fly.

Output schema (one row per doc):
{
  "source_path": "<parquet>#<doc_id>",
  "source_doc_id": "...",
  "source_dataset": "...",
  "counter_font_marker": 3,
  "counter_glyph_marker": 412,
  "counter_script_residue": 2,
  "drop_reason": "counter:glyph_font_like"  |  "" (kept)
}

Parallelizes over parquet files by spawning N worker processes
(round-robin sharding, same idea as run_matcher_parallel.py).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


def _load_thresholds(path: Path) -> Dict[str, Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suggested = data.get("suggested_thresholds") or {}
    return {
        "font_name_literal": suggested.get("font_marker"),
        "glyph_font_like": suggested.get("glyph_marker"),
        "script_residue_restricted": suggested.get("script_residue"),
    }


def _drop_reason(
    counters: Dict[str, int], thresholds: Dict[str, Optional[int]]
) -> str:
    for name, threshold in thresholds.items():
        if threshold is None:
            continue
        if counters.get(name, 0) >= threshold:
            return f"counter:{name}"
    return ""


def _process_one_parquet(
    parquet_path: str,
    output_path: str,
    category_specs_path: str,
    thresholds: Dict[str, Optional[int]],
    text_column: str,
    doc_id_column: str,
    dataset_column: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Process one parquet entirely; called in a subprocess."""
    import glossapi_rs_noise as noise  # local import in worker

    # Use a stable subdir next to the output file so the matcher's
    # side-effect `.md` writes don't land in a tempdir that tempfile
    # auto-cleans. The matcher writes small per-match .md files there
    # we don't actually consume — they get deleted at the end of the
    # call.
    output_stem = Path(output_path).stem
    output_dir_root = Path(output_path).parent
    scratch = str(output_dir_root / f"_scratch_{output_stem}")
    Path(scratch).mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows_seen = 0
    drops_by_reason: Dict[str, int] = {}
    kept = 0

    with open(output_path, "w", encoding="utf-8") as out_fh:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            cols = batch.to_pylist()
            for row in cols:
                rows_seen += 1
                text = row.get(text_column) or ""
                if not text.strip():
                    out_fh.write(json.dumps({
                        "source_path": f"{parquet_path}#{row.get(doc_id_column)}",
                        "source_doc_id": row.get(doc_id_column),
                        "source_dataset": row.get(dataset_column),
                        "counter_font_marker": 0,
                        "counter_glyph_marker": 0,
                        "counter_script_residue": 0,
                        "drop_reason": "empty",
                    }) + "\n")
                    drops_by_reason["empty"] = drops_by_reason.get("empty", 0) + 1
                    continue

                source_doc_id = str(row.get(doc_id_column) or f"row-{rows_seen}")
                source_dataset = str(row.get(dataset_column) or Path(parquet_path).stem)
                source_path = f"{parquet_path}#{source_doc_id}"
                # Sanitize: some datasets (e.g. HPLT) have `/` in their
                # name — that breaks the matcher which uses stems in
                # filesystem paths for debug .md writes. Replace any
                # char that can't appear in a file-name with `_`.
                def _safe(s: str) -> str:
                    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)
                source_stem = _safe((f"{source_dataset}__{source_doc_id}")[:200])
                base_stem = _safe(source_dataset)

                pages = noise.match_token_category_debug_text(
                    text,
                    scratch,
                    category_specs_path,
                    source_path,
                    source_stem,
                    base_stem,
                )
                counters: Dict[str, int] = {
                    "font_name_literal": 0,
                    "glyph_font_like": 0,
                    "script_residue_restricted": 0,
                }
                for page in pages:
                    for match in json.loads(page.get("matches_json") or "[]"):
                        for category in list(match.get("categories") or []):
                            if category in counters:
                                counters[category] += 1

                reason = _drop_reason(counters, thresholds)
                out_fh.write(json.dumps({
                    "source_path": source_path,
                    "source_doc_id": source_doc_id,
                    "source_dataset": source_dataset,
                    "counter_font_marker": counters["font_name_literal"],
                    "counter_glyph_marker": counters["glyph_font_like"],
                    "counter_script_residue": counters["script_residue_restricted"],
                    "drop_reason": reason,
                }) + "\n")
                if reason:
                    drops_by_reason[reason] = drops_by_reason.get(reason, 0) + 1
                else:
                    kept += 1

    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    # Also prune any .md debug files the matcher wrote at the root
    # (belt-and-suspenders — some builds of the matcher write to the
    # output_dir not to scratch).
    for stale in output_dir_root.glob("*.md"):
        try:
            stale.unlink()
        except Exception:
            pass
    return {
        "parquet": parquet_path,
        "output": output_path,
        "rows_seen": rows_seen,
        "rows_kept": kept,
        "drops_by_reason": drops_by_reason,
        "elapsed_seconds": time.time() - start,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--dataset-column", default="source_dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    from glob import glob as globmod

    paths: List[Path] = []
    for pattern in args.input_glob:
        paths.extend(sorted(Path(p).resolve() for p in globmod(pattern)))
    paths = sorted(dict.fromkeys(paths))
    print(f"{len(paths)} parquets to process with {args.workers} workers")

    thresholds = _load_thresholds(args.thresholds)
    print(f"thresholds: {thresholds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.jsonl"
    specs = str(args.category_specs.resolve())

    tasks = [
        (
            str(p),
            str(args.output_dir / (p.stem + ".drop_decisions.jsonl")),
            specs,
            thresholds,
            args.text_column,
            args.doc_id_column,
            args.dataset_column,
            args.batch_size,
        )
        for p in paths
    ]

    start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(_process_one_parquet, tasks)
    elapsed = time.time() - start

    with summary_path.open("w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res) + "\n")

    totals_drops: Dict[str, int] = {}
    total_rows = 0
    total_kept = 0
    for r in results:
        total_rows += r["rows_seen"]
        total_kept += r["rows_kept"]
        for k, v in r["drops_by_reason"].items():
            totals_drops[k] = totals_drops.get(k, 0) + v
    print(f"\n== {elapsed:.1f}s wall ==")
    print(f"  rows seen: {total_rows}")
    print(f"  rows kept: {total_kept}")
    print(f"  drops by reason: {totals_drops}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
