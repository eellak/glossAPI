"""Phase 6: train a fresh BPE tokenizer on the cleaned corpus.

Streams (parquet row, drop_decision) pairs — for each kept row applies
the Rust cleaner on-the-fly and feeds the cleaned text to
`tokenizer.train_new_from_iterator`. Dropped rows (per Phase-5
drop_decisions.jsonl) are skipped entirely. No full cleaned corpus is
materialized on disk — the BPE trainer sees cleaned text as a stream.

Reuses the base tokenizer contract from
`glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/
scripts/train_discovery_tokenizer.py`.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set

import pyarrow.parquet as pq

try:
    from transformers import AutoTokenizer
except ImportError:
    print("transformers required. `pip install transformers`", file=sys.stderr)
    sys.exit(1)

try:
    import glossapi_rs_cleaner as cleaner
except ImportError:
    print("glossapi_rs_cleaner (Rust binding) required.", file=sys.stderr)
    sys.exit(1)


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _load_drop_decisions(drop_dir: Path) -> Dict[str, Set[str]]:
    """Return {parquet_basename: set of dropped source_doc_id}."""
    dropped: Dict[str, Set[str]] = {}
    for p in sorted(drop_dir.glob("*.drop_decisions.jsonl")):
        parquet_basename = p.name.replace(".drop_decisions.jsonl", ".parquet")
        bad: Set[str] = set()
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("drop_reason"):
                    bad.add(str(d.get("source_doc_id")))
        dropped[parquet_basename] = bad
    return dropped


def _expand_inputs(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(p).resolve() for p in globmod.glob(pattern))
    return sorted(dict.fromkeys(paths))


def iter_cleaned_text(
    paths: List[Path],
    dropped_by_parquet: Dict[str, Set[str]],
    scripts_to_keep: List[str],
    text_column: str,
    doc_id_column: str,
    batch_size: int,
    progress_every: int,
) -> Iterator[List[str]]:
    """Yield lists-of-strings (HF tokenizer trainer expects a list iterator)."""
    total_docs = 0
    kept_docs = 0
    dropped_docs = 0
    empty_docs = 0
    cleaner_removed_docs = 0
    start = time.time()
    for path in paths:
        parquet_basename = path.name
        bad_ids = dropped_by_parquet.get(parquet_basename, set())
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[text_column, doc_id_column]):
            chunk: List[str] = []
            for row in batch.to_pylist():
                total_docs += 1
                doc_id = str(row.get(doc_id_column) or f"row-{total_docs}")
                if doc_id in bad_ids:
                    dropped_docs += 1
                    continue
                text = row.get(text_column) or ""
                if not text.strip():
                    empty_docs += 1
                    continue
                cleaned = cleaner.clean_text(text, scripts_to_keep)
                if not cleaned.strip() or cleaned.strip().startswith("<!-- text-missing"):
                    cleaner_removed_docs += 1
                    continue
                kept_docs += 1
                chunk.append(cleaned)
            if chunk:
                yield chunk
            if progress_every and total_docs % progress_every == 0:
                elapsed = time.time() - start
                rate = total_docs / elapsed if elapsed else 0
                print(
                    f"[iter] total={total_docs} kept={kept_docs} dropped={dropped_docs} "
                    f"empty={empty_docs} cleaner_removed={cleaner_removed_docs} "
                    f"rate={rate:.0f}/s elapsed={elapsed:.0f}s",
                    flush=True,
                )
    elapsed = time.time() - start
    print(
        f"[iter-done] total={total_docs} kept={kept_docs} dropped={dropped_docs} "
        f"empty={empty_docs} cleaner_removed={cleaner_removed_docs} elapsed={elapsed:.0f}s",
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--drop-decisions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--base-tokenizer", default="swiss-ai/Apertus-8B-2509")
    parser.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    paths = _expand_inputs(args.input_glob)
    print(f"input parquets: {len(paths)}")
    dropped = _load_drop_decisions(args.drop_decisions_dir)
    total_dropped = sum(len(s) for s in dropped.values())
    print(f"drop_decisions parquets loaded: {len(dropped)}; total dropped docs: {total_dropped}")

    base = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    assert getattr(base, "is_fast", False), "base tokenizer must be fast"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_row_estimate = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    print(f"total rows (pre-filter): {total_row_estimate}")

    iterator = iter_cleaned_text(
        paths, dropped,
        args.scripts_to_keep,
        args.text_column, args.doc_id_column,
        args.batch_size, args.progress_every,
    )
    start = time.time()
    trained = base.train_new_from_iterator(
        iterator, vocab_size=args.vocab_size, length=total_row_estimate,
    )
    elapsed = time.time() - start
    trained.save_pretrained(args.output_dir)

    summary = {
        "base_tokenizer": args.base_tokenizer,
        "vocab_size": args.vocab_size,
        "input_parquets": len(paths),
        "total_rows_pre_filter": total_row_estimate,
        "total_dropped_docs": total_dropped,
        "elapsed_seconds": elapsed,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"\ntraining done in {elapsed:.1f}s")
    print(f"output tokenizer saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
