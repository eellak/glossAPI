"""OCR result persistence helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..._naming import canonical_stem
from .context import CorpusOcrContext


def build_ocr_stage_artifact_update(
    *,
    markdown_dir: Path,
    metrics_dir: Path,
    stem: str,
) -> Optional[Dict[str, object]]:
    """Return direct OCR-owned artifact fields for one canonical OCR document."""

    markdown_path = Path(markdown_dir) / f"{stem}.md"
    if not markdown_path.exists():
        return None
    text_payload = markdown_path.read_text(encoding="utf-8")
    metrics_path = Path(metrics_dir) / f"{stem}.metrics.json"
    return {
        "text": text_payload,
        "ocr_markdown_relpath": str(Path("markdown") / markdown_path.name),
        "ocr_metrics_relpath": (
            str(Path("json") / "metrics" / metrics_path.name) if metrics_path.exists() else None
        ),
        "ocr_text_sha256": hashlib.sha256(text_payload.encode("utf-8")).hexdigest(),
    }


def apply_ocr_success_updates(
    df_meta: pd.DataFrame,
    *,
    filenames: List[str],
    markdown_dir: Path,
    metrics_dir: Path,
    backend_norm: str,
) -> pd.DataFrame:
    """Apply direct OCR-owned metadata updates to parquet rows."""

    if "filename" not in df_meta.columns:
        return df_meta

    if "filter" not in df_meta.columns:
        df_meta["filter"] = "ok"
    if "needs_ocr" not in df_meta.columns:
        df_meta["needs_ocr"] = False
    if "ocr_success" not in df_meta.columns:
        df_meta["ocr_success"] = False
    if "extraction_mode" not in df_meta.columns:
        df_meta["extraction_mode"] = None

    direct_columns = ("text", "ocr_markdown_relpath", "ocr_metrics_relpath", "ocr_text_sha256")
    for column in direct_columns:
        if column not in df_meta.columns:
            df_meta[column] = None

    filename_series = df_meta["filename"].astype(str)
    stem_series = filename_series.map(canonical_stem)

    for fname in filenames:
        stem = canonical_stem(fname)
        mask = stem_series == stem
        if not bool(mask.any()):
            continue
        artifact_update = build_ocr_stage_artifact_update(
            markdown_dir=markdown_dir,
            metrics_dir=metrics_dir,
            stem=stem,
        )
        df_meta.loc[mask, "filter"] = "ok"
        df_meta.loc[mask, "needs_ocr"] = False
        df_meta.loc[mask, "ocr_success"] = True
        if backend_norm == "deepseek":
            df_meta.loc[mask, "extraction_mode"] = "deepseek"
        if artifact_update is None:
            continue
        for column, value in artifact_update.items():
            df_meta.loc[mask, column] = value

    return df_meta


def persist_ocr_success(
    context: CorpusOcrContext,
    *,
    filenames: List[str],
    backend_norm: str,
) -> List[str]:
    from ...parquet_schema import ParquetSchema

    success_files: List[str] = []
    for fname in filenames:
        stem = canonical_stem(fname)
        if (context.markdown_dir / f"{stem}.md").exists():
            success_files.append(fname)

    if not success_files:
        return success_files

    parquet_schema = ParquetSchema({"url_column": context.url_column})
    parquet_path = context._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
    if parquet_path and parquet_path.exists():
        df_meta = pd.read_parquet(parquet_path)
        df_meta = apply_ocr_success_updates(
            df_meta,
            filenames=success_files,
            markdown_dir=context.markdown_dir,
            metrics_dir=context.output_dir / "json" / "metrics",
            backend_norm=backend_norm,
        )
        context._cache_metadata_parquet(parquet_path)
        parquet_schema.write_metadata_parquet(df_meta, parquet_path)

    stems = [canonical_stem(name) for name in success_files]
    if hasattr(context, "good_files"):
        for stem in stems:
            if stem not in getattr(context, "good_files", []):
                context.good_files.append(stem)

    return success_files


def refresh_cleaner_after_ocr(context: CorpusOcrContext) -> None:
    """Refresh cleaner metrics after OCR reruns rewrite markdown outputs."""

    refresh = getattr(context, "_refresh_metrics_after_ocr_rerun", None)
    if callable(refresh):
        refresh()
        return

    context.logger.info("Re-running Rust cleaner after OCR rerun to refresh metrics")
    context.clean(
        input_dir=context.markdown_dir,
        drop_bad=False,
    )
