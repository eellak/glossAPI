"""Compute Phase A (MD-syntax) alteration stats per doc across a
parquet corpus and write one jsonl record per doc.

Phase A transforms tracked separately:
- hr_lines_normalized / hr_chars_saved       — HR minimization.
- gfm_rows_normalized / gfm_chars_saved      — GFM table sep minimization.
- reflow_joins                               — paragraph reflow joins.
- total_chars_saved                          — input_chars - output_chars.

Output: one jsonl row per doc with the above + source identity +
relative metrics (chars_saved_pct, joins_per_1k_chars).

Usage:
  python3 compute_phase_a_stats_per_doc.py \
      --parquet-dir /home/foivos/data/glossapi_work/unified_corpus/data \
      --output /home/foivos/data/phase_a_audit/phase_a_stats.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

import glossapi_rs_cleaner as c


def process_parquet(path: Path, out_fh, batch_size: int = 1000) -> int:
    """Stream parquet rows, compute Phase A stats, write jsonl.
    Returns number of rows processed.

    The hot per-doc path is entirely in Rust via
    `phase_a_stats_jsonl_line` — no Python dict build, no
    `json.dumps`. The Python driver here just iterates parquet
    batches (pyarrow is a C++ backend) and writes the Rust-formatted
    line.
    """
    pf = pq.ParquetFile(path)
    n = 0
    path_name = str(path.name)
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["source_dataset", "source_doc_id", "text"],
    ):
        datasets = batch.column(0).to_pylist()
        doc_ids = batch.column(1).to_pylist()
        texts = batch.column(2).to_pylist()
        for ds, did, text in zip(datasets, doc_ids, texts):
            if text is None or not isinstance(text, str):
                continue
            # No length floor — the user explicitly wants the FULL
            # corpus. An empty or very short doc produces trivial
            # stats (zeros); those records are preserved because
            # downstream percentile / distribution math needs them
            # to reflect the actual corpus size.
            line = c.phase_a_stats_jsonl_line(ds or "", did or "", path_name, text)
            out_fh.write(line)
            out_fh.write("\n")
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--glob", default="*.parquet",
                   help="Parquet filename glob within --parquet-dir. "
                        "Default: *.parquet.")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--limit-files", type=int, default=0,
                   help="If >0, stop after processing this many parquet "
                        "files. For smoke-tests.")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    parquets = sorted(args.parquet_dir.glob(args.glob))
    if args.limit_files > 0:
        parquets = parquets[: args.limit_files]
    if not parquets:
        print(f"no parquets matched {args.parquet_dir}/{args.glob}",
              file=sys.stderr)
        return 1

    total_rows = 0
    t0 = time.time()
    with args.output.open("w") as fh:
        for i, pfile in enumerate(parquets):
            t1 = time.time()
            n = process_parquet(pfile, fh, batch_size=args.batch_size)
            total_rows += n
            dt = time.time() - t1
            print(f"  [{i+1}/{len(parquets)}] {pfile.name}: "
                  f"{n:,} docs in {dt:.1f}s "
                  f"({(n/dt if dt else 0):.0f} docs/s)",
                  flush=True)
    dt_total = time.time() - t0
    print(f"done: {total_rows:,} rows in {dt_total:.1f}s "
          f"→ {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
