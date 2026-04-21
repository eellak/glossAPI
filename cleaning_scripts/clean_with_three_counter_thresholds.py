"""Phase 5 cleaner driver — runs the matcher + cleaner + threshold-based
doc drop across the full corpus, emitting a new cleaned parquet per source.

For each row of each input parquet:

  1. Run the matcher on the row's text to get per-category counts
     (font_name_literal, glyph_font_like, script_residue_restricted).
  2. If ANY of the three per-doc counts exceeds its threshold (from
     thresholds.json), mark the row as `dropped_by_counter` and write
     a minimal skip-row to the output (source_doc_id + drop reason).
  3. Otherwise call the Rust cleaner on the text and write the cleaned
     row to the output parquet.

Scale: with the existing clean_corpus_parquets.py as the reference, each
parquet is streamed with pyarrow batches and processed by a rayon-style
ThreadPoolExecutor.

Run on the GCP worker. Output goes to a new `hf_release_publish_cleaned_v3`
tree sibling to the v2 one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import glossapi_rs_cleaner as cleaner
    import glossapi_rs_noise as noise
except ImportError as exc:
    print(f"Rust bindings not importable: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _load_thresholds(path: Path) -> Dict[str, Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suggested = data.get("suggested_thresholds") or {}
    return {
        "font_name_literal": suggested.get("font_marker"),
        "glyph_font_like": suggested.get("glyph_marker"),
        "script_residue_restricted": suggested.get("script_residue"),
    }


def _row_counter_counts(
    text: str,
    category_specs_path: str,
    scratch_dir: str,
    source_path: str,
    source_stem: str,
    base_stem: str,
) -> Dict[str, int]:
    """Sum per-category matches across the synthetic pages of one document."""
    pages = noise.match_token_category_debug_text(
        text,
        scratch_dir,
        category_specs_path,
        source_path,
        source_stem,
        base_stem,
    )
    totals: Dict[str, int] = {
        "font_name_literal": 0,
        "glyph_font_like": 0,
        "script_residue_restricted": 0,
    }
    for page in pages:
        matches_json = page.get("matches_json") or "[]"
        for match in json.loads(matches_json):
            for category in list(match.get("categories") or []):
                if category in totals:
                    totals[category] += 1
    return totals


def _drops_by_threshold(
    counters: Dict[str, int], thresholds: Dict[str, Optional[int]]
) -> Optional[str]:
    for name, threshold in thresholds.items():
        if threshold is None:
            continue
        if counters.get(name, 0) >= threshold:
            return name
    return None


def process_parquet(
    parquet_path: Path,
    out_dir: Path,
    category_specs_path: Path,
    thresholds: Dict[str, Optional[int]],
    scratch_dir: Path,
    scripts_to_keep: List[str],
    text_column: str = "text",
    doc_id_column: str = "source_doc_id",
    dataset_column: str = "source_dataset",
    batch_size: int = 256,
) -> Dict[str, Any]:
    start = time.time()
    rows_seen = 0
    rows_dropped: Dict[str, int] = {}
    rows_kept = 0
    out_path = out_dir / parquet_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(parquet_path)
    writer: Optional[pq.ParquetWriter] = None
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size)):
            records = batch.to_pylist()
            out_records: List[Dict[str, Any]] = []
            for row in records:
                rows_seen += 1
                text = row.get(text_column) or ""
                if not text.strip():
                    row["cleaned_text"] = ""
                    row["drop_reason"] = "empty"
                    rows_dropped["empty"] = rows_dropped.get("empty", 0) + 1
                    out_records.append(row)
                    continue
                source_doc_id = str(row.get(doc_id_column) or f"row-{rows_seen}")
                source_dataset = str(row.get(dataset_column) or parquet_path.stem)
                source_path = f"{parquet_path}#{source_doc_id}"
                source_stem = f"{source_dataset}__{source_doc_id}"[:200]
                base_stem = source_dataset

                counters = _row_counter_counts(
                    text,
                    str(category_specs_path),
                    str(scratch_dir),
                    source_path,
                    source_stem,
                    base_stem,
                )
                drop_reason = _drops_by_threshold(counters, thresholds)
                if drop_reason:
                    row["cleaned_text"] = ""
                    row["drop_reason"] = f"counter:{drop_reason}"
                    row["counter_font_marker"] = counters["font_name_literal"]
                    row["counter_glyph_marker"] = counters["glyph_font_like"]
                    row["counter_script_residue"] = counters["script_residue_restricted"]
                    rows_dropped[drop_reason] = rows_dropped.get(drop_reason, 0) + 1
                    out_records.append(row)
                    continue
                cleaned_text = cleaner.clean_text(text, scripts_to_keep)
                row["cleaned_text"] = cleaned_text
                row["drop_reason"] = ""
                row["counter_font_marker"] = counters["font_name_literal"]
                row["counter_glyph_marker"] = counters["glyph_font_like"]
                row["counter_script_residue"] = counters["script_residue_restricted"]
                rows_kept += 1
                out_records.append(row)

            out_table = pa.Table.from_pylist(out_records)
            if writer is None:
                writer = pq.ParquetWriter(out_path, out_table.schema)
            writer.write_table(out_table)
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.time() - start
    return {
        "parquet": str(parquet_path),
        "out": str(out_path),
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "rows_dropped_counts": rows_dropped,
        "elapsed_seconds": elapsed,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--dataset-column", default="source_dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)

    from glob import glob as globmod

    inputs: List[Path] = []
    for pattern in args.input_glob:
        inputs.extend(sorted(Path(p).resolve() for p in globmod(pattern)))
    inputs = sorted(dict.fromkeys(inputs))
    print(f"{len(inputs)} parquets to process")

    thresholds = _load_thresholds(args.thresholds)
    print(f"thresholds: {thresholds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="clean_scratch_"))

    summary: List[Dict[str, Any]] = []
    for pq_path in inputs:
        res = process_parquet(
            pq_path,
            args.output_dir,
            args.category_specs,
            thresholds,
            scratch,
            args.scripts_to_keep,
            text_column=args.text_column,
            doc_id_column=args.doc_id_column,
            dataset_column=args.dataset_column,
            batch_size=args.batch_size,
        )
        summary.append(res)
        print(
            f"  {pq_path.name}: seen={res['rows_seen']} "
            f"kept={res['rows_kept']} "
            f"dropped={res['rows_dropped_counts']} "
            f"elapsed={res['elapsed_seconds']:.1f}s"
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
