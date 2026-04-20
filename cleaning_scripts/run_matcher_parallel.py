#!/usr/bin/env python3
"""Run `glossapi.scripts.export_token_category_debug_parquet` in parallel shards.

The upstream script processes parquets sequentially in one Python process with
the matcher held in that one GIL-heavy process — leaves 63 of 64 cores idle on
a megamem instance. This wrapper splits the input glob into N shards (one per
worker), runs the upstream script per shard with its own output subdirectory,
then concatenates each shard's `match_index.jsonl` / `page_metrics.jsonl` /
`manifest.jsonl` / `summary.json` into a single merged output.

Per-match debug `.md` files are produced by the upstream script in each
shard's subdirectory and left there — they are only needed if we run the
legacy bundler, not the 2026-04-20 per-task bundlers that read from
`source_path` directly.

Typical invocation on the gcloud instance:

    python cleaning_scripts/run_matcher_parallel.py \\
        --input-glob '/home/foivos/data/glossapi_work/hf_release_publish_cleaned_v2/data/*.parquet' \\
        --output-dir ~/data/glossapi_work/hf_release_publish_cleaned_v2/token_category_debug \\
        --category-specs ~/token_noise_review_workspace/corpus_clean_normalization/specs/first_pass_glossapi_review.json \\
        --workers 24
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _collect(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        out.extend(Path(item).expanduser().resolve() for item in globmod.glob(pattern))
    return sorted(dict.fromkeys(out))


def _shard(paths: List[Path], n: int) -> List[List[Path]]:
    n = max(1, min(n, len(paths)))
    shards: List[List[Path]] = [[] for _ in range(n)]
    # Round-robin across shards — balances mixed file sizes better than
    # contiguous slicing for our corpus (small sources followed by many
    # identically-sized HPLT parts).
    for i, p in enumerate(paths):
        shards[i % n].append(p)
    return shards


def _concat_jsonl(src_files: List[Path], dst: Path) -> int:
    n = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        for src in src_files:
            if not src.exists():
                continue
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def _merge_summary(shard_summaries: List[Path], dst: Path) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "shards": [],
        "per_category_match_counts": {},
        "per_pattern_family_match_counts": {},
        "total_matches": 0,
        "total_pages": 0,
        "total_docs": 0,
    }
    for sp in shard_summaries:
        if not sp.exists():
            continue
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception as exc:
            merged["shards"].append({"path": str(sp), "error": repr(exc)})
            continue
        merged["shards"].append({"path": str(sp), "data": data})
        for k in ("total_matches", "total_pages", "total_docs"):
            if isinstance(data.get(k), (int, float)):
                merged[k] += int(data[k])
        for key in ("per_category_match_counts", "per_pattern_family_match_counts"):
            for cat, cnt in (data.get(key) or {}).items():
                merged[key][cat] = merged[key].get(cat, 0) + int(cnt)
    dst.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--source-dataset-column", default="source_dataset")
    parser.add_argument("--source-doc-id-column", default="source_doc_id")
    parser.add_argument("--synthetic-page-target-chars", type=int, default=4000)
    parser.add_argument("--synthetic-page-min-header-chars", type=int, default=1200)
    parser.add_argument("--synthetic-page-hard-max-chars", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument(
        "--upstream-module",
        default="glossapi.scripts.export_token_category_debug_parquet",
        help="Upstream matcher entry module run per shard",
    )
    args = parser.parse_args()

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    inputs = _collect(args.input_glob)
    if not inputs:
        print(f"No files matched {args.input_glob!r}", file=sys.stderr)
        return 2
    shards = _shard(inputs, args.workers)
    shards_dir = output_root / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"files={len(inputs)}  workers={args.workers}  shards={len(shards)}  "
        f"output_root={output_root}",
        file=sys.stderr,
    )
    for i, shard in enumerate(shards):
        print(f"  shard {i:02d}: {len(shard)} file(s)", file=sys.stderr)

    wall_t0 = time.monotonic()
    procs: List[subprocess.Popen] = []
    shard_dirs: List[Path] = []
    for i, shard in enumerate(shards):
        shard_dir = shards_dir / f"shard_{i:02d}"
        shard_dirs.append(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        # Each shard gets a glob constructed from explicit paths via ?(pat) —
        # portable: pass one --input-glob per file. Upstream accepts repeats.
        upstream_cmd: List[str] = [
            sys.executable,
            "-m",
            args.upstream_module,
            "--output-dir",
            str(shard_dir),
            "--category-specs",
            str(args.category_specs),
            "--text-column",
            args.text_column,
            "--source-dataset-column",
            args.source_dataset_column,
            "--source-doc-id-column",
            args.source_doc_id_column,
            "--synthetic-page-target-chars",
            str(args.synthetic_page_target_chars),
            "--synthetic-page-min-header-chars",
            str(args.synthetic_page_min_header_chars),
            "--synthetic-page-hard-max-chars",
            str(args.synthetic_page_hard_max_chars),
        ]
        for path in shard:
            upstream_cmd.extend(["--input-glob", str(path)])
        log_path = shard_dir / "shard.log"
        log_fp = log_path.open("w", encoding="utf-8")
        procs.append(
            subprocess.Popen(
                upstream_cmd,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        )

    failed: List[int] = []
    for i, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append(i)
            print(f"  shard {i:02d} EXIT {rc} (see {shard_dirs[i]}/shard.log)", file=sys.stderr)
        else:
            print(f"  shard {i:02d} ok", file=sys.stderr)

    wall = time.monotonic() - wall_t0
    print(f"\nAll shards done in {wall:.1f}s. Merging outputs …", file=sys.stderr)

    n_matches = _concat_jsonl(
        [d / "match_index.jsonl" for d in shard_dirs],
        output_root / "match_index.jsonl",
    )
    n_page_metrics = _concat_jsonl(
        [d / "page_metrics.jsonl" for d in shard_dirs],
        output_root / "page_metrics.jsonl",
    )
    n_manifest = _concat_jsonl(
        [d / "manifest.jsonl" for d in shard_dirs],
        output_root / "manifest.jsonl",
    )
    summary = _merge_summary(
        [d / "summary.json" for d in shard_dirs],
        output_root / "summary.json",
    )
    print(
        f"  match_index.jsonl: {n_matches:,} rows\n"
        f"  page_metrics.jsonl: {n_page_metrics:,} rows\n"
        f"  manifest.jsonl: {n_manifest:,} rows",
        file=sys.stderr,
    )
    print(
        f"  total_matches (per-shard summary sum): {summary['total_matches']:,}\n"
        f"  total_pages:  {summary['total_pages']:,}\n"
        f"  total_docs:   {summary['total_docs']:,}",
        file=sys.stderr,
    )

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
