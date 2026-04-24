"""Benchmark Phase A throughput vs worker count.

Loads a fixed set of docs into memory once, then runs
`phase_a_stats_jsonl_line` over them at various worker counts using
multiprocessing.Pool. Each worker is a separate process (no GIL
contention); the text list is pickled once per worker via Pool's
initializer.

Reports docs/s, MB/s, wall time, speedup vs 1-worker baseline, and
efficiency (speedup / workers).

Usage:
  python3 bench_phase_a_workers.py \
      --parquet /home/foivos/data/glossapi_work/unified_corpus/data/eurlex-greek-legislation.parquet \
      --workers 1 2 4 6 8 12 \
      --n-docs 10000
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pyarrow.parquet as pq


# Module-global set by the pool initializer. Avoids re-pickling a
# potentially-gigabyte text list once per task.
_TEXTS: List[str] | None = None
_DATASETS: List[str] | None = None
_DOC_IDS: List[str] | None = None
_PATH_NAME: str | None = None


def _pool_init(texts: List[str], datasets: List[str], doc_ids: List[str],
               path_name: str, mode: str) -> None:
    global _TEXTS, _DATASETS, _DOC_IDS, _PATH_NAME, _MODE
    _TEXTS = texts
    _DATASETS = datasets
    _DOC_IDS = doc_ids
    _PATH_NAME = path_name
    _MODE = mode
    # Import the Rust lib inside the worker so each worker has its own
    # cached copy. Assigning to a module global avoids import-per-call
    # overhead.
    import glossapi_rs_cleaner as c
    global _C
    _C = c  # type: ignore


_MODE = "phase_a"


def _process_range(r: Tuple[int, int]) -> int:
    """Run the configured pipeline on texts[start:end]; return bytes
    processed (a usage-count proxy to prevent dead-code elimination)."""
    start, end = r
    total = 0
    if _MODE == "phase_a":
        for i in range(start, end):
            line = _C.phase_a_stats_jsonl_line(
                _DATASETS[i] or "", _DOC_IDS[i] or "",
                _PATH_NAME, _TEXTS[i],
            )
            total += len(line)
    elif _MODE == "full_cleaner":
        scripts = ["greek", "latin", "french", "spanish",
                   "punctuation", "numbers", "common_symbols"]
        for i in range(start, end):
            out, _stats = _C.clean_text_with_stats(
                _TEXTS[i], scripts,
                None,   # min_chars_for_comment
                True,   # enable_latex_repetition_crop
                30, 3,
            )
            total += len(out)
    else:
        raise ValueError(f"unknown mode {_MODE}")
    return total


def load_corpus(parquet: Path, n_docs: int) -> Tuple[List[str], List[str], List[str]]:
    print(f"loading {parquet} ...")
    t0 = time.time()
    pf = pq.ParquetFile(parquet)
    texts: List[str] = []
    datasets: List[str] = []
    doc_ids: List[str] = []
    for batch in pf.iter_batches(
        batch_size=2000,
        columns=["source_dataset", "source_doc_id", "text"],
    ):
        for ds, did, txt in zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            batch.column(2).to_pylist(),
        ):
            if txt is None or not isinstance(txt, str):
                continue
            texts.append(txt)
            datasets.append(ds or "")
            doc_ids.append(did or "")
            if n_docs > 0 and len(texts) >= n_docs:
                break
        if n_docs > 0 and len(texts) >= n_docs:
            break
    dt = time.time() - t0
    total_chars = sum(len(t) for t in texts)
    print(f"  loaded {len(texts):,} docs, {total_chars/1e6:.1f} MB text "
          f"in {dt:.1f}s")
    return texts, datasets, doc_ids


def run_once(texts: List[str], datasets: List[str], doc_ids: List[str],
             path_name: str, n_workers: int, mode: str,
             chunk_size: int = 200) -> float:
    """Pool-map the configured pipeline over all texts with `n_workers`
    workers; return wall time."""
    ranges: List[Tuple[int, int]] = []
    for start in range(0, len(texts), chunk_size):
        ranges.append((start, min(start + chunk_size, len(texts))))
    t0 = time.time()
    if n_workers == 1:
        _pool_init(texts, datasets, doc_ids, path_name, mode)
        for r in ranges:
            _process_range(r)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=n_workers,
            initializer=_pool_init,
            initargs=(texts, datasets, doc_ids, path_name, mode),
        ) as pool:
            for _ in pool.imap_unordered(_process_range, ranges, chunksize=1):
                pass
    return time.time() - t0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, type=Path)
    p.add_argument("--workers", nargs="+", type=int,
                   default=[1, 2, 4, 6, 8, 12])
    p.add_argument("--n-docs", type=int, default=10000,
                   help="Cap the number of docs loaded. 0 = all.")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Docs per Pool task. Smaller = more tasks, more "
                        "scheduling overhead; larger = less parallelism "
                        "granularity. 200 is a reasonable default.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Times to run each worker count; median reported.")
    p.add_argument("--mode", choices=["phase_a", "full_cleaner"],
                   default="phase_a",
                   help="phase_a: just the MD-module Phase A transforms "
                        "(cheap, safe to benchmark on any host). "
                        "full_cleaner: full `clean_text_with_stats` "
                        "pipeline — heavier; run this on gcloud, not "
                        "locally.")
    args = p.parse_args()

    texts, datasets, doc_ids = load_corpus(args.parquet, args.n_docs)
    if not texts:
        print("no docs loaded", file=sys.stderr)
        return 1
    path_name = args.parquet.name
    n_docs = len(texts)
    total_chars = sum(len(t) for t in texts)

    print(f"mode={args.mode!r}")
    # Warm-up: prime disk caches / Rust library fork state.
    print("warm-up (1 worker, discarded) ...")
    run_once(texts, datasets, doc_ids, path_name, 1, args.mode,
             chunk_size=args.chunk_size)

    results: List[dict] = []
    for w in args.workers:
        samples: List[float] = []
        for _ in range(args.repeats):
            dt = run_once(texts, datasets, doc_ids, path_name, w, args.mode,
                          chunk_size=args.chunk_size)
            samples.append(dt)
        median_dt = statistics.median(samples)
        docs_per_s = n_docs / median_dt
        mb_per_s = (total_chars / 1e6) / median_dt
        results.append({
            "workers": w,
            "wall_s": median_dt,
            "samples_s": samples,
            "docs_per_s": docs_per_s,
            "mb_per_s": mb_per_s,
        })
        print(f"  w={w:>2}  wall={median_dt:>5.2f}s  "
              f"{docs_per_s:>8,.0f} docs/s  "
              f"{mb_per_s:>5.1f} MB/s  "
              f"samples={[f'{s:.2f}' for s in samples]}")

    # Speedup table (relative to workers=1 baseline).
    base = next((r for r in results if r["workers"] == 1), None)
    if base is not None:
        print("")
        print(f"{'workers':>8} {'wall(s)':>8} {'speedup':>8} "
              f"{'efficiency':>11} {'docs/s':>10} {'MB/s':>7}")
        for r in results:
            sp = base["wall_s"] / r["wall_s"]
            eff = sp / r["workers"]
            print(f"{r['workers']:>8} {r['wall_s']:>8.2f} "
                  f"{sp:>7.2f}x {eff*100:>9.1f}% "
                  f"{r['docs_per_s']:>10,.0f} {r['mb_per_s']:>7.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
