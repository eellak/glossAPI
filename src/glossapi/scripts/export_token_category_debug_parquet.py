from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pyarrow.parquet as pq

from glossapi.scripts.token_category_debug_common import (
    build_token_category_match_index_rows,
    build_token_category_page_metric_row,
    load_rust_extension,
)


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "doc"


def _expand_inputs(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for pattern in patterns:
        out.extend(Path(item).expanduser().resolve() for item in glob.glob(pattern))
    deduped = sorted(dict.fromkeys(out))
    if not deduped:
        raise SystemExit("No parquet inputs matched the provided --input-glob values.")
    return deduped


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream parquet rows through the token-category debug matcher.",
    )
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--source-dataset-column", default="source_dataset")
    parser.add_argument("--source-doc-id-column", default="source_doc_id")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--synthetic-page-target-chars", type=int, default=4000)
    parser.add_argument("--synthetic-page-min-header-chars", type=int, default=1200)
    parser.add_argument("--synthetic-page-hard-max-chars", type=int, default=6000)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    input_paths = _expand_inputs(args.input_glob)
    output_dir = args.output_dir.expanduser().resolve()
    category_specs = args.category_specs.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.md"):
        stale.unlink()
    for stale_name in ("manifest.jsonl", "page_metrics.jsonl", "match_index.jsonl", "summary.json"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()

    noise_mod = load_rust_extension(
        project_root=Path(__file__).resolve().parents[3],
        module_name="glossapi_rs_noise",
        manifest_relative="rust/glossapi_rs_noise/Cargo.toml",
        required_attrs=("match_token_category_debug_text",),
    )

    manifest_rows: List[Dict[str, Any]] = []
    page_metric_rows: List[Dict[str, Any]] = []
    match_index_rows: List[Dict[str, Any]] = []
    category_page_counter: Counter[str] = Counter()
    category_match_counter: Counter[str] = Counter()
    pattern_family_page_counter: Counter[str] = Counter()
    pattern_family_match_counter: Counter[str] = Counter()
    page_kind_counter: Counter[str] = Counter()
    doc_count_scanned = 0
    matched_doc_keys: set[str] = set()

    columns = [
        args.source_dataset_column,
        args.source_doc_id_column,
        args.text_column,
    ]

    for parquet_path in input_paths:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for raw_row in batch.to_pylist():
                if args.max_docs is not None and doc_count_scanned >= int(args.max_docs):
                    break
                doc_count_scanned += 1
                source_dataset = str(raw_row.get(args.source_dataset_column) or parquet_path.stem)
                source_doc_id = str(raw_row.get(args.source_doc_id_column) or f"row-{doc_count_scanned:08d}")
                text = str(raw_row.get(args.text_column) or "")
                if not text.strip():
                    continue
                source_stem = _safe_stem(f"{source_dataset}__{source_doc_id}")
                base_stem = _safe_stem(source_dataset)
                source_path = f"{parquet_path}#{source_doc_id}"
                rows = list(
                    noise_mod.match_token_category_debug_text(
                        text,
                        str(output_dir),
                        str(category_specs),
                        source_path,
                        source_stem,
                        base_stem,
                        1,
                        int(args.synthetic_page_target_chars),
                        int(args.synthetic_page_min_header_chars),
                        int(args.synthetic_page_hard_max_chars),
                    )
                )
                if not rows:
                    continue
                matched_doc_keys.add(source_path)
                for raw_page_row in rows:
                    row = dict(raw_page_row)
                    page_text = str(row.pop("page_text", ""))
                    matches = json.loads(str(row.pop("matches_json", "[]")))
                    manifest_rows.append(row)
                    page_metric_rows.append(build_token_category_page_metric_row(row, matches))
                    match_index_rows.extend(
                        build_token_category_match_index_rows(page_text, matches, page_row=row)
                    )
                    page_kind_counter[str(row.get("page_kind", ""))] += 1
                    for category in str(row.get("match_categories", "")).split(","):
                        if category:
                            category_page_counter[category] += 1
                    for family in str(row.get("match_pattern_families", "")).split(","):
                        if family:
                            pattern_family_page_counter[family] += 1
                    for match in matches:
                        for category in list(match.get("categories") or []):
                            category_match_counter[str(category)] += 1
                        for family in list(match.get("pattern_families") or []):
                            pattern_family_match_counter[str(family)] += 1
            if args.max_docs is not None and doc_count_scanned >= int(args.max_docs):
                break
        if args.max_docs is not None and doc_count_scanned >= int(args.max_docs):
            break

    for path, rows in (
        (output_dir / "manifest.jsonl", manifest_rows),
        (output_dir / "page_metrics.jsonl", page_metric_rows),
        (output_dir / "match_index.jsonl", match_index_rows),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    summary = {
        "input_globs": list(args.input_glob),
        "input_files": [str(path) for path in input_paths],
        "output_dir": str(output_dir),
        "category_specs_path": str(category_specs),
        "doc_count_scanned": int(doc_count_scanned),
        "matched_doc_count": int(len(matched_doc_keys)),
        "page_count": int(len(manifest_rows)),
        "match_count": int(len(match_index_rows)),
        "category_page_counts": dict(category_page_counter),
        "category_match_counts": dict(category_match_counter),
        "pattern_family_page_counts": dict(pattern_family_page_counter),
        "pattern_family_match_counts": dict(pattern_family_match_counter),
        "page_kind_counts": dict(page_kind_counter),
        "synthetic_page_target_chars": int(args.synthetic_page_target_chars),
        "synthetic_page_min_header_chars": int(args.synthetic_page_min_header_chars),
        "synthetic_page_hard_max_chars": int(args.synthetic_page_hard_max_chars),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
