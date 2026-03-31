from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from glossapi.scripts.openarchives_ocr_shards import (
    PAGE_COLUMN_CANDIDATES,
    _assign_rows,
    _coerce_bool_series,
    _resolve_page_column,
    _resolve_targets,
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_ocr_cutoff_shards",
        description=(
            "Build OCR shard manifests from the materialized local PDFs available at a cutoff, "
            "plus residual manifests for missing OCR targets."
        ),
    )
    p.add_argument("--parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--local-download-root", action="append", default=[])
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--pages-per-hour-per-node", type=float, default=50700.0)
    p.add_argument("--filename-column", default="filename")
    p.add_argument("--needs-ocr-column", default="needs_ocr")
    p.add_argument("--page-column", default=None)
    p.add_argument("--allow-threshold-derive", action="store_true")
    p.add_argument("--greek-threshold", type=float, default=60.0)
    p.add_argument("--mojibake-threshold", type=float, default=0.1)
    p.add_argument("--key-column", default="source_doc_id")
    p.add_argument("--cutoff-id", default="")
    return p.parse_args(argv)


def _canonical_stem_from_row(row: pd.Series, filename_column: str) -> str:
    if "filename_base" in row.index and str(row.get("filename_base") or "").strip():
        return str(row.get("filename_base")).strip()
    return Path(str(row.get(filename_column) or "")).stem


def _scan_local_pdfs(roots: Sequence[Path]) -> Dict[str, Tuple[Path, Path]]:
    available: Dict[str, Tuple[Path, Path]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for pdf in sorted(p for p in root.rglob("*.pdf") if p.is_file()):
            stem = pdf.stem
            if stem not in available:
                available[stem] = (root, pdf)
    return available


def _stable_item_id(cutoff_id: str, key_value: str, stem: str) -> str:
    payload = f"{cutoff_id}|{key_value}|{stem}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    parquet_path = Path(args.parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    local_roots = [Path(p).expanduser().resolve() for p in (args.local_download_root or [])]
    if not local_roots:
        raise SystemExit("Pass at least one --local-download-root.")

    df = pd.read_parquet(parquet_path).copy()
    if args.filename_column not in df.columns:
        raise SystemExit(f"Filename column '{args.filename_column}' not found in parquet.")

    page_column = _resolve_page_column(df, args.page_column)
    target_mask = _resolve_targets(
        df,
        needs_ocr_column=str(args.needs_ocr_column),
        allow_threshold_derive=bool(args.allow_threshold_derive),
        greek_threshold=float(args.greek_threshold),
        mojibake_threshold=float(args.mojibake_threshold),
    )
    target_df = df.loc[target_mask].copy()
    if target_df.empty:
        raise SystemExit("No OCR target rows selected at cutoff.")

    cutoff_id = str(args.cutoff_id or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ"))
    target_df["filename_base"] = target_df.apply(
        lambda row: _canonical_stem_from_row(row, str(args.filename_column)),
        axis=1,
    )
    available = _scan_local_pdfs(local_roots)

    rows_available: List[Dict[str, object]] = []
    rows_missing: List[Dict[str, object]] = []
    key_column = str(args.key_column)
    preserve_columns = [c for c in target_df.columns if c not in {"filename_base"}]

    for row in target_df.to_dict(orient="records"):
        stem = str(row.get("filename_base") or "")
        key_value = str(row.get(key_column) or stem or row.get(args.filename_column) or "")
        base = {col: row.get(col) for col in preserve_columns}
        item_id = _stable_item_id(cutoff_id, key_value, stem)
        if stem in available:
            root, pdf_path = available[stem]
            rel_path = pdf_path.relative_to(root)
            out = dict(base)
            out["source_filename"] = str(row.get(args.filename_column) or "")
            out["filename"] = pdf_path.name
            out["md_filename"] = f"{stem}.md"
            out["filename_base"] = stem
            out["ocr_item_id"] = item_id
            out["ocr_cutoff_id"] = cutoff_id
            out["local_pdf_path"] = str(pdf_path)
            out["local_pdf_root"] = str(root)
            out["local_pdf_relpath"] = str(rel_path)
            out["available_at_cutoff"] = True
            rows_available.append(out)
        else:
            out = dict(base)
            out["filename_base"] = stem
            out["ocr_item_id"] = item_id
            out["ocr_cutoff_id"] = cutoff_id
            out["available_at_cutoff"] = False
            rows_missing.append(out)

    available_df = pd.DataFrame(rows_available)
    missing_df = pd.DataFrame(rows_missing)
    available_path = output_dir / "openarchives_ocr_available_at_cutoff.parquet"
    missing_path = output_dir / "openarchives_ocr_missing_at_cutoff.parquet"
    if not available_df.empty:
        bins = _assign_rows(available_df, page_column=page_column, node_count=int(args.nodes))
    else:
        bins = []

    summaries: List[Dict[str, object]] = []
    total_pages = 0
    total_docs = 0
    for node in bins:
        node_id = int(node["node_id"])
        node_df = pd.DataFrame(list(node["rows"]))
        if "_pages_int" in node_df.columns:
            node_df = node_df.drop(columns=["_pages_int"])
        node_df["shard_id"] = f"node-{node_id:02d}"
        node_df["node_id"] = node_id
        out_path = output_dir / f"openarchives_ocr_shard_node_{node_id:02d}.parquet"
        node_df.to_parquet(out_path, index=False)
        node_pages = int(node["pages_total"])
        node_docs = int(node["docs_total"])
        total_pages += node_pages
        total_docs += node_docs
        summaries.append(
            {
                "node_id": node_id,
                "manifest_path": str(out_path),
                "docs_total": node_docs,
                "pages_total": node_pages,
                "eta_hours_at_validated_speed": float(node_pages / float(args.pages_per_hour_per_node)),
            }
        )

    available_df.to_parquet(available_path, index=False)
    missing_df.to_parquet(missing_path, index=False)
    overall = {
        "source_parquet": str(parquet_path),
        "cutoff_id": cutoff_id,
        "nodes": int(args.nodes),
        "key_column": key_column,
        "filename_column": str(args.filename_column),
        "page_column": str(page_column),
        "available_docs_total": int(len(available_df)),
        "available_pages_total": int(total_pages),
        "missing_docs_total": int(len(missing_df)),
        "missing_pages_total": int(pd.to_numeric(missing_df.get(page_column, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not missing_df.empty else 0,
        "pages_per_hour_per_node": float(args.pages_per_hour_per_node),
        "eta_hours_one_node": float(total_pages / float(args.pages_per_hour_per_node)) if total_pages else 0.0,
        "eta_hours_all_nodes": float(total_pages / (float(args.pages_per_hour_per_node) * max(1, int(args.nodes)))) if total_pages else 0.0,
        "available_manifest_path": str(available_path),
        "missing_manifest_path": str(missing_path),
        "node_summaries": summaries,
    }
    (output_dir / "openarchives_ocr_cutoff_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
