"""Backend-independent OpenArchives download manifest preparation."""
from pathlib import Path
import pandas as pd
from glossapi.parquet_schema import ParquetSchema

DEFAULT_DOWNLOAD_CONCURRENCY = 24
DEFAULT_DOWNLOAD_TIMEOUT = 60


def _prepare_download_input(df: pd.DataFrame) -> pd.DataFrame:
    required = {"filename", "pdf_url"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Shard parquet missing required column(s): {', '.join(missing)}")
    out = df.copy()
    out["url"] = out["pdf_url"].astype(str)
    out["filename_base"] = out["filename"].astype(str).map(lambda s: Path(s).stem)
    return out


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def _normalize_download_results(
    *,
    shard_df: pd.DataFrame,
    download_results_df: pd.DataFrame,
    url_column: str = "url",
) -> pd.DataFrame:
    shard = shard_df.copy()
    if "filename_base" not in shard.columns:
        shard["filename_base"] = shard["filename"].astype(str).map(lambda s: Path(s).stem)

    dl = download_results_df.copy()
    if "filename_base" not in dl.columns:
        dl["filename_base"] = dl["filename"].astype(str).map(lambda s: Path(s).stem)

    merged = dl.merge(
        shard,
        on="filename_base",
        how="left",
        suffixes=("", "_shard"),
    )
    if "filename_shard" in merged.columns:
        merged["filename"] = merged["filename_shard"].fillna(merged["filename"])
        merged = merged.drop(columns=["filename_shard"])
    if "pdf_url" in merged.columns and url_column in merged.columns:
        merged[url_column] = merged["pdf_url"].fillna(merged[url_column])
    elif "pdf_url" in merged.columns and url_column not in merged.columns:
        merged[url_column] = merged["pdf_url"]
    if "download_success" not in merged.columns:
        merged["download_success"] = False
    if "download_error" not in merged.columns:
        merged["download_error"] = ""
    if "ocr_success" not in merged.columns:
        merged["ocr_success"] = False
    if "needs_ocr" not in merged.columns:
        merged["needs_ocr"] = True
    return merged


def _write_canonical_metadata(work_root: Path, df: pd.DataFrame) -> Path:
    schema = ParquetSchema({"url_column": "url"})
    canonical = work_root / "download_results" / "download_results.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    normalized = schema.normalize_metadata_frame(df)
    schema.write_metadata_parquet(normalized, canonical)
    return canonical
