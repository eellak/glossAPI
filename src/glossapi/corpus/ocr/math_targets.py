"""Math-target selection helpers for corpus OCR orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Set

import pandas as pd

from ..._naming import canonical_stem
from ...parquet_schema import ParquetSchema
from .context import CorpusOcrContext


def discover_docling_json_stems(output_dir: Path) -> List[str]:
    json_dir = Path(output_dir) / "json"
    if not json_dir.exists():
        return []
    return sorted({canonical_stem(path) for path in json_dir.glob("*.docling.json*")})


def filter_math_only_stems(
    *,
    stems: Sequence[str],
    bad_files: Sequence[str],
    math_done_stems: Set[str],
    reprocess_completed: bool,
    logger,
) -> List[str]:
    kept = list(stems)
    if bad_files:
        before = len(kept)
        bad_set = {canonical_stem(name) for name in bad_files}
        kept = [stem for stem in kept if stem not in bad_set]
        removed = before - len(kept)
        if removed:
            logger.info("Math-only: skipping %d document(s) flagged for OCR", removed)
    if not reprocess_completed and kept and math_done_stems:
        before = len(kept)
        kept = [stem for stem in kept if stem not in math_done_stems]
        removed = before - len(kept)
        if removed:
            logger.info(
                "Math enrichment: skipping %d already enriched document(s) (reprocess_completed=False).",
                removed,
            )
    return kept


def select_followup_math_stems(
    *,
    stems: Sequence[str],
    bad_files: Sequence[str],
    math_done_stems: Set[str],
    reprocess_completed: bool,
    reran_ocr: bool,
    logger,
) -> List[str]:
    kept = list(stems)
    bad_set = {canonical_stem(name) for name in bad_files}
    if kept and not reran_ocr:
        kept = [stem for stem in kept if stem not in bad_set]
    if not reprocess_completed and math_done_stems:
        before = len(kept)
        kept = [stem for stem in kept if stem not in math_done_stems]
        removed = before - len(kept)
        if removed:
            logger.info(
                "Math enrichment: skipping %d already enriched document(s) (reprocess_completed=False).",
                removed,
            )
    return kept


def filter_stems_by_parquet_math_signals(
    context: CorpusOcrContext,
    *,
    stems: Sequence[str],
) -> List[str]:
    parquet_schema = ParquetSchema({"url_column": context.url_column})
    pq = context._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
    if pq is None or not pq.exists():
        return list(stems)
    try:
        df = pd.read_parquet(pq)
    except Exception:
        return list(stems)

    df["stem"] = df["filename"].astype(str).str.replace(r"\.pdf$", "", regex=True)
    phase_mask = (
        df["phase_recommended"].astype(str) == "2A"
        if "phase_recommended" in df.columns
        else ((df["filename"] == df["filename"]) & False)
    )
    formula_mask = (
        df["formula_total"].fillna(0).astype("float") > 0
        if "formula_total" in df.columns
        else ((df["filename"] == df["filename"]) & False)
    )
    detected_mask = (
        df["math_equations_detected"].fillna(0).astype("float") > 0
        if "math_equations_detected" in df.columns
        else ((df["filename"] == df["filename"]) & False)
    )
    parq_stems = set(df.loc[phase_mask | formula_mask | detected_mask, "stem"].dropna().astype(str).tolist())
    if parq_stems:
        try:
            context.logger.info("Phase-2: parquet-selected stems: %s", ",".join(sorted(parq_stems)))
        except Exception:
            pass
        return [stem for stem in stems if stem in parq_stems]
    return list(stems)


def ensure_math_placeholder_sidecars(
    context: CorpusOcrContext,
    *,
    stems: Iterable[str],
    include_parquet_signals: bool = False,
) -> None:
    sidecar_stems = set(stems)
    if include_parquet_signals:
        parquet_schema = ParquetSchema({"url_column": context.url_column})
        pq = context._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
        if pq and pq.exists():
            try:
                df = pd.read_parquet(pq)
                if "filename" in df.columns:
                    df["stem"] = df["filename"].astype(str).str.replace(r"\.pdf$", "", regex=True)
                    phase_mask = (
                        df["phase_recommended"].astype(str) == "2A"
                        if "phase_recommended" in df.columns
                        else ((df["filename"] == df["filename"]) & False)
                    )
                    formula_mask = (
                        df["formula_total"].fillna(0).astype("float") > 0
                        if "formula_total" in df.columns
                        else ((df["filename"] == df["filename"]) & False)
                    )
                    detected_mask = (
                        df["math_equations_detected"].fillna(0).astype("float") > 0
                        if "math_equations_detected" in df.columns
                        else ((df["filename"] == df["filename"]) & False)
                    )
                    sidecar_stems |= set(
                        df.loc[phase_mask | formula_mask | detected_mask, "stem"]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )
            except Exception:
                pass

    sc_dir = context.output_dir / "sidecars" / "math"
    sc_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"items": 0, "accepted": 0, "time_sec": 0.0}, ensure_ascii=False)
    for stem in sorted(sidecar_stems):
        path = sc_dir / f"{stem}.json"
        if not path.exists():
            path.write_text(payload, encoding="utf-8")
