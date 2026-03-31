from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd


PAGE_COLUMN_CANDIDATES: Sequence[str] = (
    "pages_total",
    "page_count",
    "total_pages",
    "num_pages",
    "pages",
)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m glossapi.scripts.openarchives_ocr_shards",
        description="Create page-balanced OCR shard manifests from a canonical GlossAPI parquet.",
    )
    p.add_argument("--parquet", required=True, help="Canonical download_results parquet with needs_ocr flags.")
    p.add_argument("--output-dir", required=True, help="Directory where shard manifests and summaries will be written.")
    p.add_argument("--nodes", type=int, default=4, help="Number of OCR nodes to shard across.")
    p.add_argument(
        "--pages-per-hour-per-node",
        type=float,
        default=50700.0,
        help="Validated throughput per OCR node, used for ETA calculations.",
    )
    p.add_argument("--filename-column", default="filename")
    p.add_argument("--needs-ocr-column", default="needs_ocr")
    p.add_argument(
        "--page-column",
        default=None,
        help="Explicit page-count column. If omitted, the script searches common page columns.",
    )
    p.add_argument(
        "--copy-columns",
        default="",
        help="Comma-separated extra metadata columns to preserve in every shard manifest.",
    )
    p.add_argument(
        "--allow-threshold-derive",
        action="store_true",
        help="If needs_ocr is missing, derive the target set from greek/mojibake thresholds.",
    )
    p.add_argument("--greek-threshold", type=float, default=60.0)
    p.add_argument("--mojibake-threshold", type=float, default=0.1)
    return p.parse_args(argv)


def _resolve_page_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"--page-column '{explicit}' not found in parquet.")
        return explicit
    for candidate in PAGE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise SystemExit(
        "No page-count column found. Expected one of: "
        + ", ".join(PAGE_COLUMN_CANDIDATES)
        + " or pass --page-column."
    )


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


def _page_int(value: object) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _make_node_bins(node_count: int) -> List[Dict[str, object]]:
    return [
        {
            "node_id": idx,
            "pages_total": 0,
            "docs_total": 0,
            "rows": [],
        }
        for idx in range(max(1, int(node_count)))
    ]


def _assign_rows(df: pd.DataFrame, *, page_column: str, node_count: int) -> List[Dict[str, object]]:
    ordered = df.copy()
    ordered["_pages_int"] = ordered[page_column].map(_page_int)
    ordered = ordered.sort_values(["_pages_int"], ascending=[False]).reset_index(drop=True)
    bins = _make_node_bins(node_count)
    for row in ordered.to_dict(orient="records"):
        node = min(bins, key=lambda item: (int(item["pages_total"]), int(item["node_id"])))
        row["node_id"] = int(node["node_id"])
        node["rows"].append(row)
        node["docs_total"] = int(node["docs_total"]) + 1
        node["pages_total"] = int(node["pages_total"]) + int(row["_pages_int"])
    return bins


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    parquet_path = Path(args.parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
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
    shard_df = df.loc[target_mask].copy()
    if shard_df.empty:
        raise SystemExit("No OCR target rows selected; shard manifests were not created.")

    copy_columns = [c.strip() for c in str(args.copy_columns or "").split(",") if c.strip()]
    selected_columns = [args.filename_column, page_column]
    for optional in [
        "needs_ocr",
        "greek_badness_score",
        "mojibake_badness_score",
        "ocr_success",
        "source_row",
        "document_type",
    ] + copy_columns:
        if optional in shard_df.columns and optional not in selected_columns:
            selected_columns.append(optional)
    shard_df = shard_df[selected_columns].copy()

    bins = _assign_rows(shard_df, page_column=page_column, node_count=int(args.nodes))
    summaries: List[Dict[str, object]] = []
    total_pages = 0
    total_docs = 0
    for node in bins:
        node_id = int(node["node_id"])
        rows = list(node["rows"])
        node_df = pd.DataFrame(rows)
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

    overall = {
        "source_parquet": str(parquet_path),
        "nodes": int(args.nodes),
        "filename_column": str(args.filename_column),
        "page_column": str(page_column),
        "docs_total": int(total_docs),
        "pages_total": int(total_pages),
        "pages_per_hour_per_node": float(args.pages_per_hour_per_node),
        "eta_hours_one_node": float(total_pages / float(args.pages_per_hour_per_node)),
        "eta_hours_all_nodes": float(total_pages / (float(args.pages_per_hour_per_node) * max(1, int(args.nodes)))),
        "node_summaries": summaries,
    }
    (output_dir / "openarchives_ocr_shard_summary.json").write_text(
        json.dumps(overall, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
