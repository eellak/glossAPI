"""Persistence helpers for corpus processing state."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

from .._naming import canonical_stem
from ..parquet_schema import ParquetSchema


class _ProcessingStateManager:
    """
    Maintain resume checkpoints using the canonical pipeline metadata parquet.

    This replaces legacy pickle checkpoints with structured, typed rows in
    ``download_results.parquet`` so the pipeline can resume cleanly and surface
    state to downstream consumers without ad-hoc files.
    """

    def __init__(self, state_file: Path, *, url_column: str = "url") -> None:
        # `state_file` may be a legacy `.processing_state.pkl`; we derive the
        # pipeline root from its parent (e.g., markdown/) to avoid breaking
        # call sites that still pass that path.
        self.state_file = Path(state_file)
        self.url_column = url_column
        self.logger = logging.getLogger(__name__)
        self.base_dir = self._infer_base_dir(self.state_file)
        self.schema = ParquetSchema({"url_column": self.url_column})

    @staticmethod
    def _infer_base_dir(state_path: Path) -> Path:
        parent = state_path.parent
        # Common layout: <root>/markdown/.processing_state.pkl
        if parent.name in {"markdown", "clean_markdown", "downloads", "sections"}:
            return parent.parent
        if parent.name == "download_results":
            return parent.parent
        # Fallback to parent when nothing matches.
        return parent

    @staticmethod
    def _canonical_set(values: Iterable[str]) -> Set[str]:
        stems: Set[str] = set()
        for value in values:
            try:
                stems.add(canonical_stem(value))
            except Exception:
                continue
        return stems

    def _metadata_path(self) -> Optional[Path]:
        try:
            return self.schema.ensure_metadata_parquet(self.base_dir)
        except Exception as exc:
            self.logger.warning("Failed to ensure metadata parquet under %s: %s", self.base_dir, exc)
            return None

    def _mark_stage(self, current: str, stage: str) -> str:
        parts = [p for p in (current or "").split(",") if p]
        if stage and stage not in parts:
            parts.append(stage)
        return ",".join(parts)

    def _guess_filename(self, stem: str) -> tuple[str, bool]:
        """Return a plausible filename for a stem and whether it exists on disk."""
        for root in (self.base_dir / "downloads", self.base_dir):
            for ext in ("pdf", "docx", "xml", "html", "pptx", "csv", "md"):
                candidate = root / f"{stem}.{ext}"
                if candidate.exists():
                    return candidate.name, True
        return f"{stem}.pdf", False

    def load(self) -> Tuple[Set[str], Set[str]]:
        """
        Return processed/problematic stems inferred from the pipeline parquet.

        - Processed: rows with a successful extract status or a recorded extract
          stage.
        - Problematic: rows marked as failed/timeout/errored for extraction.
        """
        parquet_path = self._metadata_path()
        if parquet_path is None or not parquet_path.exists():
            return set(), set()

        try:
            df = self.schema.normalize_metadata_frame(pd.read_parquet(parquet_path))
        except Exception as exc:
            self.logger.warning("Unable to read metadata parquet %s: %s", parquet_path, exc)
            return set(), set()

        if df.empty:
            return set(), set()

        def _stem_for(row: Dict) -> str:
            try:
                return canonical_stem(row.get("filename", ""))
            except Exception:
                return ""

        processed: Set[str] = set()
        problematic: Set[str] = set()
        success_markers = {"success", "ok", "complete", "completed", "partial", "chunked"}
        failure_markers = {"failed", "failure", "error", "timeout", "problematic", "skipped"}

        for _, row in df.iterrows():
            stem = _stem_for(row)
            if not stem:
                continue
            status_raw = str(row.get("extract_status") or "").strip().lower()
            stage_raw = str(row.get("processing_stage") or "")
            if status_raw in success_markers or "extract" in stage_raw.split(","):
                processed.add(stem)
                continue
            if status_raw in failure_markers:
                problematic.add(stem)

        # Fallback: trust existing markdown files if metadata is missing flags
        try:
            markdown_dir = self.base_dir / "markdown"
            if markdown_dir.exists():
                for md_path in markdown_dir.glob("*.md"):
                    processed.add(canonical_stem(md_path))
        except Exception:
            pass

        return processed, problematic

    def save(self, processed: Set[str], problematic: Set[str]) -> None:
        """
        Persist extraction state into the metadata parquet.

        Args:
            processed: Canonical stems that completed extraction.
            problematic: Canonical stems that failed extraction.
        """
        processed_stems = self._canonical_set(processed)
        problematic_stems = self._canonical_set(problematic)
        if not processed_stems and not problematic_stems:
            return

        parquet_path = self._metadata_path()
        if parquet_path is None:
            return

        try:
            df = self.schema.normalize_metadata_frame(pd.read_parquet(parquet_path))
        except Exception as exc:
            self.logger.warning("Unable to read metadata parquet %s: %s", parquet_path, exc)
            return

        if "extract_status" not in df.columns:
            df["extract_status"] = pd.Series([pd.NA] * len(df))
        if "extract_error" not in df.columns:
            df["extract_error"] = pd.Series([""] * len(df))
        if "processing_stage" not in df.columns:
            df["processing_stage"] = pd.Series([""] * len(df))

        df["__stem__"] = df["filename"].astype(str).map(canonical_stem)

        def _update_rows(target_stems: Set[str], status: str, error: str = "") -> None:
            nonlocal df
            if not target_stems:
                return
            mask = df["__stem__"].isin(target_stems)
            existing_stems = set(df.loc[mask, "__stem__"].tolist())
            missing_stems = target_stems.difference(existing_stems)

            if missing_stems:
                rows: list[dict] = []
                for stem in sorted(missing_stems):
                    filename, exists = self._guess_filename(stem)
                    file_ext = Path(filename).suffix.lstrip(".")
                    row = {col: pd.NA for col in df.columns}
                    row[self.url_column] = row.get(self.url_column, "") or ""
                    row["filename"] = filename
                    row["file_ext"] = file_ext
                    row["filename_base"] = stem
                    row["download_success"] = bool(exists)
                    row["extract_status"] = status
                    row["extract_error"] = error
                    row["processing_stage"] = self._mark_stage("", "extract")
                    row["__stem__"] = stem
                    rows.append(row)
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                mask = df["__stem__"].isin(target_stems)

            df.loc[mask, "extract_status"] = status
            if error:
                df.loc[mask, "extract_error"] = error
            df.loc[mask, "processing_stage"] = df.loc[mask, "processing_stage"].apply(
                lambda current: self._mark_stage(current, "extract")
            )

        # Prioritize problematic markers over processed ones for overlapping stems
        overlap = processed_stems & problematic_stems
        if overlap:
            processed_stems.difference_update(overlap)

        _update_rows(processed_stems, "success")
        _update_rows(problematic_stems, "problematic")

        try:
            df = df.drop(columns=["__stem__"])
        except KeyError:
            pass

        try:
            self.schema.write_metadata_parquet(df, parquet_path)
        except Exception as exc:
            self.logger.warning("Failed to persist processing state to %s: %s", parquet_path, exc)
