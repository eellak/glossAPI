from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import zstandard as zstd

from glossapi.scripts.openarchives_ocr_enrich import _resolve_jsonl_path


PIPELINE_FIELDS = (
    "greek_badness_score",
    "mojibake_badness_score",
    "latin_percentage",
    "polytonic_ratio",
    "char_count_no_comments",
    "is_empty",
    "filter",
    "needs_ocr",
    "ocr_success",
    "quality_method",
    "reevaluated_at",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_hf_refresh",
        description=(
            "Refresh the canonical OpenArchives HF jsonl.zst shards in place from a refreshed "
            "document-level parquet and update the dataset card counts."
        ),
    )
    p.add_argument("--dataset-root", required=True, help="Local clone/snapshot root of the HF dataset repo.")
    p.add_argument("--metadata-parquet", required=True, help="Refreshed document-level parquet with source_jsonl/doc ids.")
    p.add_argument("--output-root", default="", help="Optional separate output root. Defaults to in-place dataset-root.")
    p.add_argument("--readme-path", default="README.md", help="Dataset card path relative to dataset-root/output-root.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _normalize_source_key(dataset_root: Path, recorded_path: str) -> str:
    resolved = _resolve_jsonl_path(dataset_root, recorded_path)
    return str(resolved.relative_to(dataset_root))


def _clean_value(value: object) -> object:
    if pd.isna(value):  # type: ignore[arg-type]
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _build_update_index(metadata_df: pd.DataFrame, *, dataset_root: Path) -> Dict[str, Dict[str, dict]]:
    required = {"source_doc_id", "source_jsonl"}
    missing = sorted(required - set(metadata_df.columns))
    if missing:
        raise SystemExit(f"Metadata parquet missing required column(s): {', '.join(missing)}")
    updates: Dict[str, Dict[str, dict]] = {}
    work = metadata_df.copy()
    work["_source_key"] = work["source_jsonl"].astype(str).map(lambda p: _normalize_source_key(dataset_root, p))
    for _, row in work.iterrows():
        source_key = str(row["_source_key"])
        doc_id = str(row["source_doc_id"] or "")
        payload = {field: _clean_value(row[field]) for field in PIPELINE_FIELDS if field in row.index}
        updates.setdefault(source_key, {})[doc_id] = payload
    return updates


def _iter_jsonl_rows(path: Path) -> Iterable[dict]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
        for line in text_reader:
            yield json.loads(line)


def _write_jsonl_rows(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    count = 0
    with path.open("wb") as fh:
        with cctx.stream_writer(fh) as writer:
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                writer.write(payload)
                count += 1
    return count


def _refresh_readme(readme_text: str, *, total_docs: int, needs_ocr_docs: int) -> str:
    title_text = f"OpenArchives.gr {total_docs:,} docs".replace(",", ",")
    percent = (100.0 * needs_ocr_docs / total_docs) if total_docs else 0.0
    pct_text = f"{percent:.2f}%"

    replacements = [
        (r"pretty_name:\s*OpenArchives\.gr [^\n]+", f"pretty_name: {title_text}"),
        (r"# OpenArchives\.gr [^\n]+", f"# {title_text}"),
        (
            r"- Σύνολο markdown αρχείων: \*\*[0-9,]+\*\* from openarchives\.gr",
            f"- Σύνολο markdown αρχείων: **{total_docs:,}** from openarchives.gr",
        ),
        (
            r"- Total markdown files: \*\*[0-9,]+\*\* from openarchives\.gr",
            f"- Total markdown files: **{total_docs:,}** from openarchives.gr",
        ),
        (
            r"- Τα χαμηλής ποιότητας αρχεία που ενδέχεται να χρειάζονται OCR επεξεργασία επισημαίνονται με τη στήλη `needs_ocr`: \*\*[0-9,]+ / [0-9,]+ \([0-9.]+%\)\*\*",
            f"- Τα χαμηλής ποιότητας αρχεία που ενδέχεται να χρειάζονται OCR επεξεργασία επισημαίνονται με τη στήλη `needs_ocr`: **{needs_ocr_docs:,} / {total_docs:,} ({pct_text})**",
        ),
        (
            r"- Lower-quality files that may require OCR reprocessing are marked by the `needs_ocr` indicator: \*\*[0-9,]+ / [0-9,]+ \([0-9.]+%\)\*\*",
            f"- Lower-quality files that may require OCR reprocessing are marked by the `needs_ocr` indicator: **{needs_ocr_docs:,} / {total_docs:,} ({pct_text})**",
        ),
    ]
    updated = readme_text
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)
    return updated


def _refresh_shard(
    *,
    input_path: Path,
    output_path: Path,
    updates: Dict[str, dict],
    dry_run: bool,
) -> dict:
    total = 0
    matched = 0
    needs_ocr = 0
    unmatched_doc_ids: list[str] = []
    rows_out: list[dict] = []

    for row in _iter_jsonl_rows(input_path):
        total += 1
        doc_id = str(row.get("doc_id") or "")
        payload = updates.get(doc_id)
        if payload is not None:
            pipeline = dict(row.get("pipeline_metadata") or {})
            pipeline.update({k: v for k, v in payload.items() if v is not None})
            row["pipeline_metadata"] = pipeline
            matched += 1
        else:
            unmatched_doc_ids.append(doc_id)
        pipeline = row.get("pipeline_metadata") or {}
        if bool(pipeline.get("needs_ocr")):
            needs_ocr += 1
        rows_out.append(row)

    if not dry_run:
        _write_jsonl_rows(output_path, rows_out)

    return {
        "path": str(input_path),
        "total_rows": total,
        "matched_rows": matched,
        "unmatched_rows": total - matched,
        "needs_ocr_rows": needs_ocr,
        "sample_unmatched_doc_ids": unmatched_doc_ids[:5],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_parquet).expanduser().resolve()

    metadata_df = pd.read_parquet(metadata_path).copy()
    updates_by_shard = _build_update_index(metadata_df, dataset_root=dataset_root)

    summaries: list[dict] = []
    total_rows = 0
    matched_rows = 0
    needs_ocr_rows = 0
    shard_root = dataset_root / "data" / "openarchives"
    for rel_key, updates in sorted(updates_by_shard.items()):
        input_path = dataset_root / rel_key
        output_path = output_root / rel_key
        summary = _refresh_shard(
            input_path=input_path,
            output_path=output_path,
            updates=updates,
            dry_run=bool(args.dry_run),
        )
        summaries.append(summary)
        total_rows += int(summary["total_rows"])
        matched_rows += int(summary["matched_rows"])
        needs_ocr_rows += int(summary["needs_ocr_rows"])

    readme_rel = Path(args.readme_path)
    readme_in = dataset_root / readme_rel
    readme_out = output_root / readme_rel
    if readme_in.exists() and not args.dry_run:
        readme_text = readme_in.read_text(encoding="utf-8")
        readme_out.write_text(
            _refresh_readme(readme_text, total_docs=matched_rows, needs_ocr_docs=int(metadata_df["needs_ocr"].fillna(False).sum())),
            encoding="utf-8",
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "metadata_parquet": str(metadata_path),
        "shards_touched": len(summaries),
        "total_rows_seen": total_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": total_rows - matched_rows,
        "needs_ocr_rows_after_refresh": needs_ocr_rows,
        "metadata_rows": int(len(metadata_df)),
        "metadata_needs_ocr_rows": int(metadata_df["needs_ocr"].fillna(False).sum()) if "needs_ocr" in metadata_df.columns else None,
        "sample_shards": summaries[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
