"""Parallel preclean to text shards — feeds Phase-6 BPE training.

Reads parquets + drop_decisions, applies the Rust cleaner in parallel
worker processes, writes cleaned text to newline-separated `.txt.gz`
shards — one per parquet. Each cleaned doc becomes one line of text
(newlines in the doc are replaced with `\\n` escape? — no, we simply
strip newlines since the BPE trainer treats each yielded line
independently anyway).

Actually: BPE from `train_new_from_iterator` doesn't care about
document boundaries as long as each chunk is a list of strings. We
write one doc per line (no internal line breaks — newlines within
docs replaced by spaces) so downstream streaming reads naturally.

Output: `<output_dir>/<parquet_stem>.txt.gz` per parquet. The BPE
trainer iterates all these gzipped text files.
"""
from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Set

import pyarrow.parquet as pq


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _load_dropped_for_parquet(drop_dir: Path, parquet_name: str) -> Set[str]:
    p = drop_dir / (parquet_name.replace(".parquet", ".drop_decisions.jsonl"))
    bad: Set[str] = set()
    if not p.exists():
        return bad
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("drop_reason"):
                bad.add(str(d.get("source_doc_id")))
    return bad


def _process_one_parquet(
    parquet_path: str,
    drop_dir: str,
    output_dir: str,
    scripts_to_keep: List[str],
    text_column: str,
    doc_id_column: str,
    batch_size: int,
) -> Dict:
    import glossapi_rs_cleaner as cleaner

    parquet_name = Path(parquet_path).name
    bad_ids = _load_dropped_for_parquet(Path(drop_dir), parquet_name)

    out_file = Path(output_dir) / (Path(parquet_path).stem + ".txt.gz")
    rows_seen = 0
    rows_kept = 0
    rows_dropped = 0
    rows_empty_after_clean = 0

    start = time.time()
    with gzip.open(out_file, "wt", encoding="utf-8") as out_fh:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column, doc_id_column]):
            for row in batch.to_pylist():
                rows_seen += 1
                doc_id = str(row.get(doc_id_column) or f"row-{rows_seen}")
                if doc_id in bad_ids:
                    rows_dropped += 1
                    continue
                text = row.get(text_column) or ""
                if not text.strip():
                    rows_empty_after_clean += 1
                    continue
                cleaned = cleaner.clean_text(text, scripts_to_keep)
                if not cleaned.strip() or cleaned.strip().startswith("<!-- text-missing"):
                    rows_empty_after_clean += 1
                    continue
                # Replace internal newlines with spaces so one-doc-per-line
                one_line = cleaned.replace("\r", " ").replace("\n", " ")
                out_fh.write(one_line + "\n")
                rows_kept += 1

    return {
        "parquet": parquet_path,
        "out": str(out_file),
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "rows_dropped": rows_dropped,
        "rows_empty_after_clean": rows_empty_after_clean,
        "elapsed": time.time() - start,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--drop-decisions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pattern in args.input_glob:
        paths.extend(Path(p).resolve() for p in globmod.glob(pattern))
    paths = sorted(dict.fromkeys(paths))
    print(f"{len(paths)} parquets × {args.workers} workers")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            str(p),
            str(args.drop_decisions_dir.resolve()),
            str(args.output_dir.resolve()),
            args.scripts_to_keep,
            args.text_column,
            args.doc_id_column,
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
    total_dropped = sum(r["rows_dropped"] for r in results)
    total_empty = sum(r["rows_empty_after_clean"] for r in results)

    (args.output_dir / "precleaning_summary.json").write_text(
        json.dumps(
            {
                "total_seen": total_seen,
                "total_kept": total_kept,
                "total_dropped": total_dropped,
                "total_empty_after_clean": total_empty,
                "elapsed_seconds": elapsed,
                "n_parquets": len(paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"seen={total_seen} kept={total_kept} dropped={total_dropped} empty_after={total_empty} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
