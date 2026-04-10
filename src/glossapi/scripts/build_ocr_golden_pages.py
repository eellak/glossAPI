from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAGE_SPLIT_MARKER = "<--- Page Split --->"


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_sort_rows(rows: Sequence[Dict[str, object]], seed: str) -> List[Dict[str, object]]:
    def _key(row: Dict[str, object]) -> str:
        basis = f"{seed}|{row['source_stem']}|{row['page_number']}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    return sorted(rows, key=_key)


def _take_rows(
    rows: Sequence[Dict[str, object]],
    selected_keys: set[Tuple[str, int]],
    *,
    limit: int,
    seed: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in _stable_sort_rows(rows, seed):
        key = (str(row["source_stem"]), int(row["page_number"]))
        if key in selected_keys:
            continue
        out.append(row)
        selected_keys.add(key)
        if len(out) >= limit:
            break
    return out


def _split_pages(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").split(PAGE_SPLIT_MARKER)


def build_ocr_goldens(
    *,
    run_dir: Path,
    source_dir: Path,
    output_dir: Path,
    seed: str = "ocr-golden-v1",
) -> Dict[str, object]:
    page_metrics = _read_jsonl(run_dir / "page_metrics.jsonl")
    manifest_rows = _read_jsonl(run_dir / "manifest.jsonl")
    source_by_stem = {Path(str(row["source_path"])).stem: Path(str(row["source_path"])) for row in manifest_rows}
    output_by_stem = {Path(str(row["output_path"])).stem: Path(str(row["output_path"])) for row in manifest_rows}

    for target in (output_dir / "inputs", output_dir / "expected"):
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.iterdir():
            if stale.is_file():
                stale.unlink()
    for stale_name in ("manifest.jsonl", "summary.json"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()

    source_pages_cache: Dict[str, List[str]] = {}
    output_pages_cache: Dict[str, List[str]] = {}

    rows_with_features: List[Dict[str, object]] = []
    for row in page_metrics:
        stem = str(row["source_stem"])
        source_path = source_by_stem.get(stem)
        output_path = output_by_stem.get(stem)
        if source_path is None or output_path is None:
            continue
        if stem not in source_pages_cache:
            source_pages_cache[stem] = _split_pages(source_path)
            output_pages_cache[stem] = _split_pages(output_path)
        page_idx = int(row["page_number"]) - 1
        source_page = source_pages_cache[stem][page_idx]
        output_page = output_pages_cache[stem][page_idx]
        feature_row = dict(row)
        feature_row["has_table_html"] = "<table" in source_page.lower()
        feature_row["source_page"] = source_page
        feature_row["expected_page"] = output_page
        positive_categories = [
            category
            for category in ("table", "numeric", "latex", "hybrid", "word")
            if int(row.get(f"{category}_match_count", 0)) > 0
        ]
        feature_row["positive_categories"] = positive_categories
        rows_with_features.append(feature_row)

    selected_keys: set[Tuple[str, int]] = set()
    selected_rows: List[Tuple[str, Dict[str, object]]] = []

    def add_bucket(label: str, candidates: Iterable[Dict[str, object]], limit: int) -> None:
        bucket = _take_rows(list(candidates), selected_keys, limit=limit, seed=f"{seed}:{label}")
        for item in bucket:
            selected_rows.append((label, item))

    add_bucket(
        "hybrid_positive",
        [row for row in rows_with_features if int(row.get("hybrid_match_count", 0)) > 0],
        9999,
    )
    add_bucket(
        "latex_positive",
        [row for row in rows_with_features if int(row.get("latex_match_count", 0)) > 0],
        9999,
    )
    add_bucket(
        "mixed_positive",
        [row for row in rows_with_features if len(list(row.get("positive_categories", []))) >= 2],
        120,
    )
    add_bucket(
        "numeric_positive",
        [row for row in rows_with_features if int(row.get("numeric_match_count", 0)) > 0],
        140,
    )
    add_bucket(
        "word_positive",
        [row for row in rows_with_features if int(row.get("word_match_count", 0)) > 0],
        140,
    )
    add_bucket(
        "table_positive",
        [row for row in rows_with_features if int(row.get("table_match_count", 0)) > 0],
        180,
    )
    add_bucket(
        "table_kept_conversion",
        [
            row
            for row in rows_with_features
            if row.get("has_table_html")
            and all(int(row.get(f"{category}_match_count", 0)) == 0 for category in ("table", "numeric", "latex", "hybrid", "word"))
        ],
        60,
    )
    add_bucket(
        "negative_plain",
        [
            row
            for row in rows_with_features
            if not row.get("has_table_html")
            and all(int(row.get(f"{category}_match_count", 0)) == 0 for category in ("table", "numeric", "latex", "hybrid", "word"))
        ],
        60,
    )

    manifest_out = output_dir / "manifest.jsonl"
    summary_out = output_dir / "summary.json"
    written_rows: List[Dict[str, object]] = []
    category_counts: Dict[str, int] = {}

    for idx, (label, row) in enumerate(selected_rows, start=1):
        stem = str(row["source_stem"])
        page_number = int(row["page_number"])
        base_name = f"{idx:04d}__{stem}__page_{page_number:05d}"
        input_path = output_dir / "inputs" / f"{base_name}.md"
        expected_path = output_dir / "expected" / f"{base_name}.md"
        input_path.write_text(str(row["source_page"]), encoding="utf-8")
        expected_path.write_text(str(row["expected_page"]), encoding="utf-8")

        category_counts[label] = category_counts.get(label, 0) + 1
        written_rows.append(
            {
                "case_id": base_name,
                "label": label,
                "source_stem": stem,
                "page_number": page_number,
                "input_path": str(input_path),
                "expected_path": str(expected_path),
                "source_path": str(source_by_stem[stem]),
                "output_path": str(output_by_stem[stem]),
                "match_counts": {
                    category: int(row.get(f"{category}_match_count", 0))
                    for category in ("table", "numeric", "latex", "hybrid", "word")
                },
                "has_table_html": bool(row.get("has_table_html")),
            }
        )

    with manifest_out.open("w", encoding="utf-8") as handle:
        for row in written_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "run_dir": str(run_dir),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "case_count": len(written_rows),
        "category_counts": category_counts,
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OCR golden page fixtures from a combined debug run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="ocr-golden-v1")
    args = parser.parse_args()

    summary = build_ocr_goldens(
        run_dir=args.run_dir,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
