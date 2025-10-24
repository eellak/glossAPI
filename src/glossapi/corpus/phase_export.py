"""Export helpers split from Corpus."""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._naming import canonical_stem
from ..gloss_downloader import GlossDownloader
from ..gloss_section import GlossSection
from ..gloss_section_classifier import GlossSectionClassifier
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch


class ExportPhaseMixin:
    def jsonl(self, output_path: Union[str, Path]) -> Path:
        """Export cleaned markdown and metadata into a JSONL corpus."""

        output_path = Path(output_path)
        download_dir = self.output_dir / "download_results"
        metadata_path = download_dir / "download_results.parquet"
        if not metadata_path.exists():
            candidates = sorted(download_dir.glob("*.parquet")) if download_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(f"Metadata parquet not found in {download_dir}")
            preferred = [p for p in candidates if p.name.startswith("download_results_")]
            metadata_path = preferred[0] if preferred else candidates[0]

        df = pd.read_parquet(metadata_path)
        if df.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            return output_path

        def _stem_for(value: str) -> str:
            if not value:
                return ""
            return canonical_stem(value)

        df["__stem__"] = df["filename"].astype(str).map(_stem_for)
        metadata_by_stem: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            data = row.to_dict()
            stem = str(data.pop("__stem__", ""))
            if stem:
                metadata_by_stem[stem] = data

        markdown_root = self.cleaned_markdown_dir if any(self.cleaned_markdown_dir.glob("*.md")) else self.markdown_dir

        def _load_metrics(stem: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
            metrics_dir = self.output_dir / "json" / "metrics"
            candidates = [
                metrics_dir / f"{stem}.metrics.json",
                metrics_dir / f"{stem}.per_page.metrics.json",
            ]
            data = None
            for candidate in candidates:
                if candidate.exists():
                    try:
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        continue
            if not data:
                return None, None, None
            page_count = data.get("page_count")
            pages = data.get("pages") or []
            try:
                formula_total = sum(int(p.get("formula_count", 0) or 0) for p in pages)
            except Exception:
                formula_total = None
            try:
                code_total = sum(int(p.get("code_count", 0) or 0) for p in pages)
            except Exception:
                code_total = None
            try:
                if page_count is not None:
                    page_count = int(page_count)
            except Exception:
                page_count = None
            return page_count, formula_total, code_total

        def _load_math_accepts(stem: str) -> tuple[bool, int]:
            latex_map = self.output_dir / "json" / f"{stem}.latex_map.jsonl"
            if not latex_map.exists():
                return False, 0
            accepted = 0
            try:
                with latex_map.open("r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        score = payload.get("accept_score")
                        try:
                            if score is not None and float(score) >= 1.0:
                                accepted += 1
                        except Exception:
                            continue
            except Exception:
                return True, 0
            return True, accepted

        def _normalize_value(value: Any) -> Any:
            if value is None:
                return None
            try:
                if isinstance(value, float) and math.isnan(value):
                    return None
            except Exception:
                pass
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            return value

        output_path.parent.mkdir(parents=True, exist_ok=True)
        records_written = 0
        with output_path.open("w", encoding="utf-8") as fp:
            for md_path in sorted(markdown_root.glob("*.md")):
                stem = canonical_stem(md_path)
                metadata = metadata_by_stem.get(stem)
                if metadata is None:
                    continue
                record = {k: _normalize_value(v) for k, v in metadata.items()}
                record["document"] = md_path.read_text(encoding="utf-8")

                filetype = record.get("filetype") or record.get("file_ext")
                if not filetype:
                    filename_value = record.get("filename")
                    if isinstance(filename_value, str):
                        filetype = Path(filename_value).suffix.lstrip(".")
                record["filetype"] = filetype or None

                metrics_page_count, metrics_formula, metrics_code = _load_metrics(stem)
                if metrics_page_count is not None:
                    record["page_count"] = metrics_page_count

                existing_formula = record.get("formula_total")
                if metrics_formula is not None:
                    record["formula_total"] = metrics_formula
                else:
                    try:
                        record["formula_total"] = int(existing_formula)
                    except Exception:
                        record["formula_total"] = 0

                existing_code = record.get("code_total")
                if metrics_code is not None:
                    record["code_total"] = metrics_code
                else:
                    try:
                        record["code_total"] = int(existing_code)
                    except Exception:
                        record["code_total"] = 0

                math_enriched, math_accepted = _load_math_accepts(stem)
                record["math_enriched"] = bool(math_enriched)
                record["math_accepted"] = int(math_accepted)

                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

        if records_written == 0:
            output_path.write_text("", encoding="utf-8")
        return output_path

    def process_all(self, input_format: str = "pdf", fully_annotate: bool = True, annotation_type: str = "auto", download_first: bool = False) -> None:
        """
        Run the complete processing pipeline: extract, section, and annotate.

        Args:
            input_format: Input format (default: "pdf")
            fully_annotate: Whether to perform full annotation after classification (default: True)
            annotation_type: Annotation method to use (default: "auto")
            download_first: Whether to run the downloader before extraction (default: False)
        """
        if download_first:
            try:
                self.download()
                self.logger.info("Download step completed, proceeding with extraction...")
            except Exception as e:
                self.logger.error(f"Error during download step: {e}")
                self.logger.warning("Continuing with extraction of already downloaded files...")

        self.extract(input_format=input_format)
        self.section()
        self.annotate(fully_annotate=fully_annotate, annotation_type=annotation_type)

        self.logger.info("Complete processing pipeline finished successfully.")
