"""Full clean+stats driver for the from-scratch run.

For each doc of each input parquet:

  1. Run the matcher to get per-counter values (font_name_literal,
     glyph_font_like, script_residue_restricted).
  2. If any counter ≥ its threshold → DROP (record chars_removed=chars_before,
     drop_reason=counter:<name>).
  3. Else: run the Rust cleaner; record chars_before/after/removed/pct
     and write the cleaned text (one doc per line, gzip compressed) to an
     output text shard for downstream BPE training.

Emits per-doc stats JSONL + per-parquet cleaned-text shard.

Parallelized via mp.Pool over parquet files. Each worker:
  - loads its own matcher + cleaner bindings (PyO3).
  - opens one parquet and one .txt.gz shard.
  - streams rows through matcher → threshold decision → cleaner.
"""
from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)


def _load_thresholds(path: Path) -> Dict[str, Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suggested = data.get("suggested_thresholds") or {}
    return {
        "font_name_literal": suggested.get("font_marker"),
        "glyph_font_like": suggested.get("glyph_marker"),
        "script_residue_restricted": suggested.get("script_residue"),
    }


def _drop_reason(counters, thresholds):
    for name, threshold in thresholds.items():
        if threshold is None:
            continue
        if counters.get(name, 0) >= threshold:
            return f"counter:{name}"
    return ""


def _process_one_parquet(
    parquet_path: str,
    stats_path: str,
    text_gz_path: str,
    category_specs_path: str,
    thresholds: Dict[str, Optional[int]],
    scripts_to_keep: List[str],
    text_column: str,
    doc_id_column: str,
    dataset_column: str,
    batch_size: int,
) -> Dict[str, Any]:
    import glossapi_rs_cleaner as cleaner
    import glossapi_rs_noise as noise

    scratch_root = Path(stats_path).parent
    scratch = scratch_root / f"_scratch_{Path(stats_path).stem}"
    scratch.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows_seen = 0
    rows_kept = 0
    rows_dropped: Dict[str, int] = {}
    total_chars_before = 0
    total_chars_after = 0
    total_chars_dropped_by_drop = 0

    with open(stats_path, "w", encoding="utf-8") as stats_fh, \
         gzip.open(text_gz_path, "wt", encoding="utf-8") as text_fh:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                rows_seen += 1
                text = row.get(text_column) or ""
                chars_before = len(text)
                if not text.strip():
                    rows_dropped["empty"] = rows_dropped.get("empty", 0) + 1
                    stats_fh.write(json.dumps({
                        "source_path": f"{parquet_path}#{row.get(doc_id_column)}",
                        "source_doc_id": row.get(doc_id_column),
                        "source_dataset": row.get(dataset_column),
                        "chars_before": 0,
                        "chars_after": 0,
                        "chars_removed": 0,
                        "pct_removed": 0.0,
                        "counter_font_marker": 0,
                        "counter_glyph_marker": 0,
                        "counter_script_residue": 0,
                        "drop_reason": "empty",
                    }) + "\n")
                    continue

                source_doc_id = str(row.get(doc_id_column) or f"row-{rows_seen}")
                source_dataset = str(row.get(dataset_column) or Path(parquet_path).stem)
                source_path = f"{parquet_path}#{source_doc_id}"
                source_stem = _safe(f"{source_dataset}__{source_doc_id}")[:200]
                base_stem = _safe(source_dataset)

                pages = noise.match_token_category_debug_text(
                    text, str(scratch), category_specs_path,
                    source_path, source_stem, base_stem,
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
                if reason:
                    rows_dropped[reason] = rows_dropped.get(reason, 0) + 1
                    total_chars_dropped_by_drop += chars_before
                    stats_fh.write(json.dumps({
                        "source_path": source_path,
                        "source_doc_id": source_doc_id,
                        "source_dataset": source_dataset,
                        "chars_before": chars_before,
                        "chars_after": 0,
                        "chars_removed": chars_before,
                        "pct_removed": 100.0,
                        "counter_font_marker": counters["font_name_literal"],
                        "counter_glyph_marker": counters["glyph_font_like"],
                        "counter_script_residue": counters["script_residue_restricted"],
                        "drop_reason": reason,
                    }) + "\n")
                    continue

                cleaned = cleaner.clean_text(text, scripts_to_keep)
                if not cleaned.strip() or cleaned.strip().startswith("<!-- text-missing"):
                    rows_dropped["cleaner_empty"] = rows_dropped.get("cleaner_empty", 0) + 1
                    total_chars_dropped_by_drop += chars_before
                    stats_fh.write(json.dumps({
                        "source_path": source_path,
                        "source_doc_id": source_doc_id,
                        "source_dataset": source_dataset,
                        "chars_before": chars_before,
                        "chars_after": 0,
                        "chars_removed": chars_before,
                        "pct_removed": 100.0,
                        "counter_font_marker": counters["font_name_literal"],
                        "counter_glyph_marker": counters["glyph_font_like"],
                        "counter_script_residue": counters["script_residue_restricted"],
                        "drop_reason": "cleaner_empty",
                    }) + "\n")
                    continue

                chars_after = len(cleaned)
                chars_removed = chars_before - chars_after
                pct_removed = (chars_removed / chars_before * 100.0) if chars_before else 0.0
                total_chars_before += chars_before
                total_chars_after += chars_after
                rows_kept += 1
                # One doc per line — internal newlines → spaces
                text_fh.write(cleaned.replace("\r", " ").replace("\n", " ") + "\n")
                stats_fh.write(json.dumps({
                    "source_path": source_path,
                    "source_doc_id": source_doc_id,
                    "source_dataset": source_dataset,
                    "chars_before": chars_before,
                    "chars_after": chars_after,
                    "chars_removed": chars_removed,
                    "pct_removed": round(pct_removed, 3),
                    "counter_font_marker": counters["font_name_literal"],
                    "counter_glyph_marker": counters["glyph_font_like"],
                    "counter_script_residue": counters["script_residue_restricted"],
                    "drop_reason": "",
                }) + "\n")

    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    for stale in scratch_root.glob("*.md"):
        try: stale.unlink()
        except Exception: pass

    return {
        "parquet": parquet_path,
        "stats_out": stats_path,
        "text_gz_out": text_gz_path,
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "rows_dropped": rows_dropped,
        "chars_before_total_kept_docs": total_chars_before,
        "chars_after_total_kept_docs": total_chars_after,
        "chars_removed_by_per_line_strip": total_chars_before - total_chars_after,
        "chars_removed_by_drop": total_chars_dropped_by_drop,
        "elapsed_seconds": time.time() - start,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--stats-dir", required=True, type=Path)
    parser.add_argument("--text-shards-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--dataset-column", default="source_dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pat in args.input_glob:
        paths.extend(Path(p).resolve() for p in globmod.glob(pat))
    paths = sorted(dict.fromkeys(paths))
    print(f"{len(paths)} parquets to process with {args.workers} workers")

    thresholds = _load_thresholds(args.thresholds)
    print(f"thresholds: {thresholds}")

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    args.text_shards_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            str(p),
            str(args.stats_dir / (p.stem + ".stats.jsonl")),
            str(args.text_shards_dir / (p.stem + ".txt.gz")),
            str(args.category_specs.resolve()),
            thresholds,
            args.scripts_to_keep,
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

    total_seen = sum(r["rows_seen"] for r in results)
    total_kept = sum(r["rows_kept"] for r in results)
    total_drop_counts: Dict[str, int] = {}
    total_chars_before = 0
    total_chars_after = 0
    total_chars_removed_strip = 0
    total_chars_removed_drop = 0
    for r in results:
        total_chars_before += r["chars_before_total_kept_docs"]
        total_chars_after += r["chars_after_total_kept_docs"]
        total_chars_removed_strip += r["chars_removed_by_per_line_strip"]
        total_chars_removed_drop += r["chars_removed_by_drop"]
        for k, v in r["rows_dropped"].items():
            total_drop_counts[k] = total_drop_counts.get(k, 0) + v

    summary = {
        "elapsed_seconds": elapsed,
        "rows_seen": total_seen,
        "rows_kept": total_kept,
        "rows_dropped": total_drop_counts,
        "chars_before_total_kept_docs": total_chars_before,
        "chars_after_total_kept_docs": total_chars_after,
        "chars_removed_by_per_line_strip": total_chars_removed_strip,
        "chars_removed_by_drop": total_chars_removed_drop,
        "pct_chars_removed_by_strip": round(
            100.0 * total_chars_removed_strip / max(total_chars_before, 1), 3),
    }
    (args.stats_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
