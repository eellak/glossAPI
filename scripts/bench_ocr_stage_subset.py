#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark clean / clean_ocr stages on a fixed raw-markdown subset."
    )
    parser.add_argument(
        "--stage",
        choices=("clean", "clean_ocr", "clean_ocr_debug"),
        required=True,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Text file listing absolute markdown paths to benchmark, one per line.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of raw markdown files. Used with --limit when --manifest is omitted.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Number of markdown files to take from --input-dir when --manifest is omitted.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def load_source_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest is not None:
        return [
            Path(line.strip())
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if args.input_dir is None:
        raise SystemExit("Either --manifest or --input-dir must be provided.")
    return sorted(args.input_dir.glob("*.md"))[: int(args.limit)]


def hash_dir(md_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for md in sorted(md_dir.glob("*.md")):
        hashes[md.name] = hashlib.sha256(md.read_bytes()).hexdigest()
    return hashes


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from glossapi import Corpus
    import pandas as pd

    source_paths = load_source_paths(args)
    out_root = args.out_root
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    corpus = Corpus(input_dir=out_root / "input", output_dir=out_root / "output")
    corpus.markdown_dir.mkdir(parents=True, exist_ok=True)
    for src in source_paths:
        shutil.copy2(src, corpus.markdown_dir / src.name)

    start = time.perf_counter()
    if args.stage == "clean":
        corpus.clean(write_cleaned_files=True, drop_bad=False)
    elif args.stage == "clean_ocr":
        corpus.clean_ocr(write_cleaned_files=True, write_debug_files=False, drop_bad=False)
    else:
        corpus.clean_ocr(write_cleaned_files=True, write_debug_files=True, drop_bad=False)
    elapsed = time.perf_counter() - start

    clean_hashes = hash_dir(corpus.cleaned_markdown_dir)
    debug_hashes: dict[str, str] = {}
    debug_dir = corpus.output_dir / "debug"
    if debug_dir.exists():
        debug_hashes = hash_dir(debug_dir)

    parquet_path = corpus.output_dir / "download_results" / "download_results.parquet"
    df = pd.read_parquet(parquet_path).copy()
    metric_columns = [
        "filename",
        "char_count_no_comments",
        "is_empty",
        "percentage_greek",
        "latin_percentage",
        "polytonic_ratio",
        "greek_badness_score",
        "mojibake_badness_score",
        "needs_ocr",
        "filter",
        "ocr_noise_suspect",
        "ocr_noise_flags",
        "ocr_repeat_phrase_run_max",
        "ocr_repeat_line_run_max",
        "ocr_repeat_suspicious_line_count",
        "ocr_repeat_suspicious_line_ratio",
    ]
    cols = [c for c in metric_columns if c in df.columns]
    metrics = df[cols].sort_values("filename").to_dict("records")

    summary = {
        "label": args.label,
        "stage": args.stage,
        "doc_count": len(source_paths),
        "elapsed_seconds": elapsed,
        "clean_file_count": len(clean_hashes),
        "debug_file_count": len(debug_hashes),
        "clean_hashes": clean_hashes,
        "debug_hashes": debug_hashes,
        "metrics": metrics,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "label": args.label,
                "stage": args.stage,
                "elapsed_seconds": elapsed,
                "clean_file_count": len(clean_hashes),
                "debug_file_count": len(debug_hashes),
                "summary_path": str(summary_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
