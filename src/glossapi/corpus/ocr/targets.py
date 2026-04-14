"""Target selection helpers for corpus OCR orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from ..._naming import canonical_stem
from ...parquet_schema import ParquetSchema
from .context import CorpusOcrContext
from ..corpus_skiplist import _SkiplistManager, _resolve_skiplist_path


@dataclass(slots=True)
class OcrSelection:
    bad_files: List[str]
    ocr_candidates_initial: int
    skipped_completed: int
    skipped_skiplist: int
    parquet_meta: Optional[pd.DataFrame]
    ocr_done_files: List[str]
    ocr_done_stems: Set[str]
    math_done_stems: Set[str]
    skip_mgr: _SkiplistManager
    skiplist_path: Path


def normalize_ocr_target_filenames(*, filenames: List[str], input_dir: Path) -> List[str]:
    """Collapse chunk-like metadata rows back to real OCR source files when possible."""

    source_by_stem = {}
    try:
        for path in sorted(Path(input_dir).glob("*.pdf")):
            source_by_stem.setdefault(canonical_stem(path.name), path.name)
    except Exception:
        source_by_stem = {}

    normalized: List[str] = []
    seen: Set[str] = set()
    for fname in filenames:
        resolved = source_by_stem.get(canonical_stem(fname), str(fname))
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)
    return normalized


def build_ocr_selection(
    context: CorpusOcrContext,
    *,
    mode: str,
    reprocess_completed: bool,
) -> OcrSelection:
    bad_files: List[str] = []
    skipped_completed = 0
    skipped_skiplist = 0
    parquet_meta: Optional[pd.DataFrame] = None
    ocr_done_files: List[str] = []
    ocr_done_stems: Set[str] = set()
    math_done_stems: Set[str] = set()

    parquet_schema = ParquetSchema({"url_column": context.url_column})
    parquet_path = context._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
    if parquet_path and parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "filename" in df.columns and "needs_ocr" in df.columns:
            bad_files = df.loc[df["needs_ocr"] == True, "filename"].dropna().astype(str).tolist()
        if "ocr_success" in df.columns:
            ocr_done_files = df.loc[df["ocr_success"].fillna(False), "filename"].dropna().astype(str).tolist()
            ocr_done_stems = {canonical_stem(name) for name in ocr_done_files}
        math_done_files: List[str] = []
        if "math_enriched" in df.columns:
            math_done_files = df.loc[df["math_enriched"].fillna(False), "filename"].dropna().astype(str).tolist()
        elif "enriched_math" in df.columns:
            math_done_files = df.loc[df["enriched_math"].fillna(False), "filename"].dropna().astype(str).tolist()
        if math_done_files:
            math_done_stems = {canonical_stem(name) for name in math_done_files}
        if not reprocess_completed and ocr_done_stems:
            before = len(bad_files)
            bad_files = [name for name in bad_files if canonical_stem(name) not in ocr_done_stems]
            skipped_completed = before - len(bad_files)
            if skipped_completed:
                context.logger.info(
                    "OCR: skipping %d already completed document(s) (reprocess_completed=False).",
                    skipped_completed,
                )
        if reprocess_completed and mode in {"ocr_bad", "ocr_bad_then_math"} and ocr_done_files:
            pending = {str(name) for name in bad_files}
            for fname in ocr_done_files:
                if fname not in pending:
                    bad_files.append(fname)
                    pending.add(fname)
        parquet_meta = df

    ocr_candidates_initial = len(bad_files)
    skiplist_path = _resolve_skiplist_path(context.output_dir, context.logger)
    skip_mgr = _SkiplistManager(skiplist_path, context.logger)
    skip_stems = skip_mgr.load()
    if skip_stems:
        before = len(bad_files)
        bad_files = [name for name in bad_files if canonical_stem(name) not in skip_stems]
        skipped_skiplist = before - len(bad_files)
        if skipped_skiplist:
            context.logger.warning(
                "Skip-list %s filtered %d document(s) from Phase-3 OCR.",
                skiplist_path,
                skipped_skiplist,
            )

    normalized_bad_files = normalize_ocr_target_filenames(
        filenames=bad_files,
        input_dir=Path(context.input_dir),
    )
    if len(normalized_bad_files) != len(bad_files):
        context.logger.info(
            "OCR: collapsed %d metadata-selected row(s) onto %d real source PDF(s) by canonical stem.",
            len(bad_files),
            len(normalized_bad_files),
        )
    bad_files = normalized_bad_files
    context.logger.info(
        "OCR targets: total=%d kept=%d skipped_completed=%d skipped_skiplist=%d",
        ocr_candidates_initial,
        len(bad_files),
        skipped_completed,
        skipped_skiplist,
    )

    return OcrSelection(
        bad_files=bad_files,
        ocr_candidates_initial=ocr_candidates_initial,
        skipped_completed=skipped_completed,
        skipped_skiplist=skipped_skiplist,
        parquet_meta=parquet_meta,
        ocr_done_files=ocr_done_files,
        ocr_done_stems=ocr_done_stems,
        math_done_stems=math_done_stems,
        skip_mgr=skip_mgr,
        skiplist_path=skiplist_path,
    )
