from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd
import zstandard as zstd


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_ocr_enrich",
        description="Enrich OpenArchives OCR routing rows with page counts and PDF URLs from raw JSONL shards.",
    )
    p.add_argument("--parquet", required=True, help="Canonical parquet after OpenArchives cleaning/fill.")
    p.add_argument("--raw-repo-root", required=True, help="Local root of the raw HF OpenArchives dataset.")
    p.add_argument("--output-parquet", required=True, help="Where the enriched parquet will be written.")
    p.add_argument("--filename-column", default="filename")
    p.add_argument("--doc-id-column", default="source_doc_id")
    p.add_argument("--source-jsonl-column", default="source_jsonl")
    p.add_argument("--needs-ocr-column", default="needs_ocr")
    p.add_argument(
        "--allow-threshold-derive",
        action="store_true",
        help="If needs_ocr is missing, derive targets from greek/mojibake thresholds.",
    )
    p.add_argument("--greek-threshold", type=float, default=60.0)
    p.add_argument("--mojibake-threshold", type=float, default=0.1)
    return p.parse_args(argv)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _resolve_targets(
    df: pd.DataFrame,
    *,
    needs_ocr_column: str,
    allow_threshold_derive: bool,
    greek_threshold: float,
    mojibake_threshold: float,
) -> pd.Series:
    if needs_ocr_column in df.columns:
        return _coerce_bool_series(df[needs_ocr_column])
    if not allow_threshold_derive:
        raise SystemExit(
            f"Column '{needs_ocr_column}' not found and threshold derivation is disabled."
        )
    greek = pd.to_numeric(df.get("greek_badness_score"), errors="coerce")
    moj = pd.to_numeric(df.get("mojibake_badness_score"), errors="coerce")
    if greek is None and moj is None:
        raise SystemExit(
            "Cannot derive OCR targets: neither needs_ocr nor greek/mojibake badness columns are present."
        )
    greek_mask = (greek > float(greek_threshold)).fillna(False) if greek is not None else False
    moj_mask = (moj > float(mojibake_threshold)).fillna(False) if moj is not None else False
    return greek_mask | moj_mask


def _resolve_jsonl_path(raw_repo_root: Path, recorded_path: str) -> Path:
    candidate = Path(recorded_path)
    if candidate.exists():
        return candidate

    marker = "data/openarchives/"
    text = str(recorded_path)
    idx = text.find(marker)
    if idx != -1:
        rel = Path(text[idx:])
        rewritten = raw_repo_root / rel
        if rewritten.exists():
            return rewritten

    name = Path(recorded_path).name
    matches = list((raw_repo_root / "data" / "openarchives").glob(f"**/{name}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"could not resolve JSONL path for {recorded_path}")


def _pick_pdf_url(source_meta: dict) -> str:
    for key in ("refined_pdf_links_json", "pdf_links_json"):
        value = source_meta.get(key)
        url = _normalize_pdf_link(value)
        if url:
            return url
    for key in ("external_link", "handle_url", "url"):
        value = source_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_pdf_link(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith("http://") or text.startswith("https://"):
            return text
        try:
            parsed = json.loads(text)
        except Exception:
            return text
        return _normalize_pdf_link(parsed)
    if isinstance(value, list):
        for item in value:
            normalized = _normalize_pdf_link(item)
            if normalized:
                return normalized
        return ""
    if isinstance(value, dict):
        for key in ("url", "href", "pdf_url", "link"):
            if key in value:
                normalized = _normalize_pdf_link(value[key])
                if normalized:
                    return normalized
        return ""
    return ""


def _coerce_page_count(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(1, int(float(value)))
    except Exception:
        return None


def _enrich_targets(
    targets: pd.DataFrame,
    *,
    raw_repo_root: Path,
    doc_id_column: str,
    source_jsonl_column: str,
) -> pd.DataFrame:
    work = targets.copy()
    work["_resolved_jsonl"] = work[source_jsonl_column].map(
        lambda p: str(_resolve_jsonl_path(raw_repo_root, str(p)))
    )
    grouped: Dict[str, Dict[str, int]] = {}
    for row_index, row in work[[doc_id_column, "_resolved_jsonl"]].iterrows():
        grouped.setdefault(str(row["_resolved_jsonl"]), {})[str(row[doc_id_column])] = int(row_index)

    dctx = zstd.ZstdDecompressor()
    for jsonl_path, doc_map in grouped.items():
        with Path(jsonl_path).open("rb") as fh, dctx.stream_reader(fh) as reader:
            text_reader = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_reader:
                record = json.loads(line)
                doc_id = str(record.get("doc_id") or "")
                row_index = doc_map.get(doc_id)
                if row_index is None:
                    continue
                pipeline = record.get("pipeline_metadata") or {}
                source_meta = record.get("source_metadata") or {}
                page_count = _coerce_page_count(pipeline.get("page_count"))
                pages_total = _coerce_page_count(pipeline.get("pages_total"))
                if page_count is None:
                    page_count = pages_total
                if pages_total is None:
                    pages_total = page_count
                work.at[row_index, "page_count_source"] = page_count
                work.at[row_index, "pages_total_source"] = pages_total
                work.at[row_index, "pdf_url"] = _pick_pdf_url(source_meta)
                work.at[row_index, "source_collection_slug"] = source_meta.get("collection_slug") or ""
                work.at[row_index, "source_language_code"] = source_meta.get("language_code") or ""

    return work.drop(columns=["_resolved_jsonl"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    parquet_path = Path(args.parquet).expanduser().resolve()
    raw_repo_root = Path(args.raw_repo_root).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    for required in (args.filename_column, args.doc_id_column, args.source_jsonl_column):
        if required not in df.columns:
            raise SystemExit(f"Required column '{required}' not found in parquet.")

    target_mask = _resolve_targets(
        df,
        needs_ocr_column=str(args.needs_ocr_column),
        allow_threshold_derive=bool(args.allow_threshold_derive),
        greek_threshold=float(args.greek_threshold),
        mojibake_threshold=float(args.mojibake_threshold),
    )
    targets = df.loc[target_mask].copy()
    if targets.empty:
        raise SystemExit("No OCR target rows selected; enriched parquet was not created.")

    enriched_targets = _enrich_targets(
        targets,
        raw_repo_root=raw_repo_root,
        doc_id_column=str(args.doc_id_column),
        source_jsonl_column=str(args.source_jsonl_column),
    )

    enriched_targets.to_parquet(output_path, index=False)
    summary = {
        "source_parquet": str(parquet_path),
        "output_parquet": str(output_path),
        "target_docs": int(len(enriched_targets)),
        "page_count_source_non_null": int(enriched_targets["page_count_source"].notna().sum()),
        "pdf_url_non_empty": int(enriched_targets["pdf_url"].fillna("").astype(str).str.len().gt(0).sum()),
        "pages_total_sum": int(pd.to_numeric(enriched_targets["page_count_source"], errors="coerce").fillna(0).sum()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
