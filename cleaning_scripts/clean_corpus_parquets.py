#!/usr/bin/env python3
"""Apply `glossapi_rs_cleaner.clean_text` to every row of every parquet in an input tree.

Intended use: corpus-scale re-clean after a cleaner behavior change, without
the round-trip through markdown files that `Corpus.clean` requires.

Design decisions:
- Per-row invocation — the Rust `clean_text` holds the GIL but each call is
  microseconds; per-row is simpler than batching, and any real speedup
  comes from `--workers` (one process per parquet).
- Skip existing outputs: the output dir acts as an idempotent checkpoint so
  interrupted runs can resume without redoing work.
- Preserve schema: the input parquet's columns, dtypes, and row order are
  preserved; only the `text` column is rewritten.
- Failure mode: if a row raises during cleaning (shouldn't happen with
  valid utf-8), the original text is kept and a counter is logged. A full
  parquet failure (unreadable file, missing column) is logged and the
  file is skipped; the rest of the run continues.

Typical invocation:

    python cleaning_scripts/clean_corpus_parquets.py \\
        --input-dir  ~/data/glossapi_work/hf_release_publish_working/data \\
        --output-dir ~/data/glossapi_work/hf_release_publish_cleaned_v2/data \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


LOGGER = logging.getLogger("clean_corpus_parquets")

DEFAULT_SCRIPTS: List[str] = [
    "greek",
    "latin",
    "punctuation",
    "numbers",
    "common_symbols",
]


def _clean_text(text: Any, scripts: List[str]) -> Any:
    import glossapi_rs_cleaner as _gc

    if text is None:
        return text
    if isinstance(text, float) and pd.isna(text):
        return text
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    try:
        return _gc.clean_text(text, scripts, None)
    except Exception as exc:  # pragma: no cover - defensive, keep row intact
        LOGGER.warning("clean_text raised on one row: %s", exc)
        return text


def process_parquet(
    in_path: Path,
    out_path: Path,
    *,
    text_col: str,
    scripts: List[str],
) -> Dict[str, Any]:
    t0 = time.monotonic()
    try:
        df = pd.read_parquet(in_path)
    except Exception as exc:
        return {
            "path": str(in_path),
            "status": "read_failed",
            "error": repr(exc),
            "elapsed_s": time.monotonic() - t0,
        }
    row_count = len(df)
    if text_col not in df.columns:
        return {
            "path": str(in_path),
            "status": "no_text_column",
            "row_count": row_count,
            "columns": list(df.columns),
            "elapsed_s": time.monotonic() - t0,
        }

    before_chars = int(df[text_col].fillna("").astype(str).map(len).sum())
    df[text_col] = df[text_col].apply(_clean_text, scripts=scripts)
    after_chars = int(df[text_col].fillna("").astype(str).map(len).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, index=False)
    except Exception as exc:
        return {
            "path": str(in_path),
            "status": "write_failed",
            "error": repr(exc),
            "elapsed_s": time.monotonic() - t0,
        }

    return {
        "path": str(in_path),
        "out_path": str(out_path),
        "status": "ok",
        "row_count": row_count,
        "chars_before": before_chars,
        "chars_after": after_chars,
        "chars_delta": after_chars - before_chars,
        "elapsed_s": time.monotonic() - t0,
    }


def _process_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    return process_parquet(
        Path(args["in_path"]),
        Path(args["out_path"]),
        text_col=args["text_col"],
        scripts=list(args["scripts"]),
    )


def _collect_inputs(root: Path, pattern: str) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--pattern", default="*.parquet")
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=DEFAULT_SCRIPTS,
        help="Script keys to preserve (see SCRIPT_SETS in cleaning_module.rs)",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSONL report path; defaults to <output-dir>/report.jsonl",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_root = args.input_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = (args.report_path or (output_root / "report.jsonl")).expanduser().resolve()

    inputs = _collect_inputs(input_root, args.pattern)
    if not inputs:
        print(f"No files matching {args.pattern!r} under {input_root}", file=sys.stderr)
        return 2

    jobs: List[Dict[str, Any]] = []
    for in_path in inputs:
        rel = in_path.relative_to(input_root) if input_root.is_dir() else in_path.name
        out_path = output_root / rel
        if out_path.exists():
            LOGGER.info("skip %s (output exists)", rel)
            continue
        jobs.append(
            {
                "in_path": str(in_path),
                "out_path": str(out_path),
                "text_col": args.text_col,
                "scripts": list(args.scripts),
            }
        )

    print(
        f"inputs={len(inputs)}  pending={len(jobs)}  workers={args.workers}  "
        f"output_root={output_root}  report={report_path}",
        file=sys.stderr,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode so reruns accumulate diagnostics.
    report_fp = report_path.open("a", encoding="utf-8")

    wall_t0 = time.monotonic()
    done = 0
    failed = 0
    try:
        if args.workers <= 1:
            for job in jobs:
                result = _process_worker(job)
                report_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                report_fp.flush()
                done += 1
                if result["status"] != "ok":
                    failed += 1
                print(
                    f"[{done}/{len(jobs)}] {result['status']:>12}  "
                    f"rows={result.get('row_count','?'):>7}  "
                    f"{result.get('elapsed_s', 0):.1f}s  "
                    f"{Path(job['in_path']).name}",
                    file=sys.stderr,
                )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_process_worker, job): job for job in jobs}
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "path": job["in_path"],
                            "status": "worker_exception",
                            "error": repr(exc),
                        }
                    report_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
                    report_fp.flush()
                    done += 1
                    if result["status"] != "ok":
                        failed += 1
                    print(
                        f"[{done}/{len(jobs)}] {result['status']:>12}  "
                        f"rows={result.get('row_count','?'):>7}  "
                        f"{result.get('elapsed_s', 0):.1f}s  "
                        f"{Path(job['in_path']).name}",
                        file=sys.stderr,
                    )
    finally:
        report_fp.close()

    wall = time.monotonic() - wall_t0
    print(
        f"\nDone. ok={done - failed}  failed={failed}  total={done}  wall={wall:.1f}s",
        file=sys.stderr,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
