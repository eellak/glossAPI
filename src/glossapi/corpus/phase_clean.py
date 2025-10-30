"""Cleaning and filtering helpers split from Corpus."""
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
# Avoid importing section/classifier here; cleaning phase does not use them.
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch


class CleanPhaseMixin:
    @staticmethod
    def _project_root() -> Path:
        """Locate the repository root that houses the Rust crates."""
        here = Path(__file__).resolve()
        for candidate in here.parents:
            rust_dir = candidate / "rust"
            if rust_dir.exists() and rust_dir.is_dir():
                return candidate
        return here.parents[2]

    def _load_rust_extension(self, module_name: str, manifest_relative: str):
        """Import a Rust extension, building it with maturin if necessary."""
        import importlib

        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            self.logger.warning(
                "Rust extension %s missing; attempting in-place build via maturin …",
                module_name,
            )
            root_dir = self._project_root()
            manifest = root_dir / manifest_relative
            if not manifest.exists():
                raise RuntimeError(
                    f"Cannot locate Cargo manifest for {module_name} at {manifest}"
                )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "maturin",
                        "develop",
                        "--release",
                        "--manifest-path",
                        str(manifest),
                    ],
                    check=True,
                )
                return importlib.import_module(module_name)
            except Exception as build_err:
                raise RuntimeError(
                    f"Automatic build of {module_name} failed: {build_err}"
                )

    def _load_metrics_dataframe(
        self, parquet_path: Path, filenames: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Load an analytics parquet or seed an empty frame keyed by filename."""
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        names: List[str] = []
        if filenames is not None:
            seen: Set[str] = set()
            for item in filenames:
                if item is None:
                    continue
                name = str(item)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return pd.DataFrame({"filename": names})

    @staticmethod
    def _ensure_metric_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> None:
        """Ensure metric columns exist with provided defaults."""
        for column, default in defaults.items():
            if column not in df.columns:
                df[column] = default

    @staticmethod
    def _merge_metric_dataframe(
        base: pd.DataFrame, updates: pd.DataFrame, *, key: str = "filename"
    ) -> pd.DataFrame:
        """Overlay scorer output onto the authoritative metrics dataframe."""
        if updates.empty:
            return base
        base_idx = base.set_index(key, drop=False)
        update_idx = updates.set_index(key, drop=False)
        base_idx = base_idx.combine_first(update_idx)
        base_idx.update(update_idx)
        return base_idx.reset_index(drop=True)

    def clean(
        self,
        input_dir: Union[str, Path] = None,
        threshold: float = 0.10,
        num_threads: int = None,
        drop_bad: bool = True,
        *,
        write_cleaned_files: bool = True,
        ocr_model_dir: Union[str, Path, None] = None,
        force_ocr_fallback: bool = False,
        empty_char_threshold: int = 0,
        empty_min_pages: int = 0,
    ) -> None:
        """Clean markdown files and evaluate badness using the Rust extension.

        Args:
            input_dir: Folder with `.md` files to process (defaults to `self.markdown_dir`).
            threshold: Badness threshold for optional dropping.
            num_threads: Rayon thread-count to pass to Rust.
            drop_bad: If True, files with badness_score > threshold are removed from downstream processing. Set to False to keep all files and only record the score.
            write_cleaned_files: Set False to skip writing cleaned markdown files; metrics and parquet updates still occur.
            ocr_model_dir: [DEPRECATED – no effect] Use Corpus.ocr(model_dir=...) instead.
            force_ocr_fallback: [DEPRECATED – no effect] Use Corpus.ocr(fix_bad=True) instead.
            empty_char_threshold: Character threshold (after stripping comments and whitespace) that flags markdown as nearly empty. Default 0 only enforces the zero-character safeguard.
            empty_min_pages: Minimum page count for a low-character document to trigger an OCR rerun recommendation.
        """
        from pathlib import Path
        import shutil
        import pandas as pd
        from glossapi.parquet_schema import ParquetSchema

        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        # Handle OCR model directory override
        if ocr_model_dir is not None:
            self.ocr_model_dir = Path(ocr_model_dir)

        self._load_rust_extension(
            "glossapi_rs_cleaner", "rust/glossapi_rs_cleaner/Cargo.toml"
        )
        self.logger.info("Using compiled glossapi_rs_cleaner extension for fast cleaning")

        # Ensure cleaned directory exists and is empty (idempotent runs)
        if write_cleaned_files:
            if self.cleaned_markdown_dir.exists():
                shutil.rmtree(self.cleaned_markdown_dir)
            self.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)

        # Prepare parquet helper
        parquet_schema = ParquetSchema({"url_column": self.url_column})
        parquet_path: Optional[Path] = self._get_cached_metadata_parquet()
        if parquet_path is None:
            existing_metadata = parquet_schema.find_metadata_parquet(self.input_dir)
            if existing_metadata is not None:
                parquet_path = self._cache_metadata_parquet(existing_metadata)
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.output_dir)
            if ensured is not None:
                parquet_path = self._cache_metadata_parquet(ensured)
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.input_dir)
            if ensured is not None:
                parquet_path = self._cache_metadata_parquet(ensured)
        if parquet_path is None:
            metadata_target = self.output_dir / "download_results" / "download_results.parquet"
            self.logger.info(
                "Cleaner: no metadata parquet found; will bootstrap %s when metrics become available.",
                metadata_target,
            )
        else:
            metadata_target = parquet_path
        parquet_path = self._cache_metadata_parquet(metadata_target)

        import os
        records: list = []  # will hold metrics for parquet merge
        metrics_dir = self.output_dir / "json" / "metrics"

        def _page_count_for(stem: str) -> Optional[int]:
            candidates = [
                metrics_dir / f"{stem}.metrics.json",
                metrics_dir / f"{stem}.per_page.metrics.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    pc = data.get("page_count")
                    if pc is not None:
                        try:
                            return int(pc)
                        except Exception:
                            pass
                    pages = data.get("pages")
                    if isinstance(pages, list):
                        return len(pages)
            return None

        # ----- Call Rust high-level pipeline once -----
        scripts_to_keep = ["greek", "latin"]  # keep common alphabetic scripts; numbers/punctuation are added internally
        report_parquet_path = self.cleaned_markdown_dir.parent / "cleaning_report.parquet"

        md_files = sorted(input_dir.glob("*.md"))
        total_files = len(md_files)

        self.logger.info(
            "Invoking glossapi_rs_cleaner.run_complete_pipeline on %d markdown files…",
            total_files,
        )

        class _CleanerProgress:
            def __init__(self, logger: logging.Logger, total: int) -> None:
                self.logger = logger
                self.total = total
                self.processed: set[str] = set()
                self.buffer = ""
                if total > 0:
                    step = max(1, math.ceil(total * 0.02))
                else:
                    step = 1
                self.step = step
                self.next_target = step
                self.logged_full = False
                self.last_message: Optional[str] = None
                self.direct_updates = False
                self.last_processed = 0

            def write(self, text: str) -> int:
                if not text:
                    return 0
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    self._handle_line(line.strip())
                return len(text)

            def flush(self) -> None:  # pragma: no cover - required by IO interface
                return

            def handle_line(self, line: str) -> None:
                self._handle_line(line.strip())

            def _handle_line(self, line: str) -> None:
                if not line:
                    return
                direct = re.search(
                    r"Rust cleaning progress:\s*(\d+)%\s*\((\d+)/(\d+)\)", line
                )
                if direct:
                    try:
                        percent = int(direct.group(1))
                        processed = int(direct.group(2))
                        total_reported = int(direct.group(3))
                    except (TypeError, ValueError):
                        percent = processed = 0
                        total_reported = self.total
                    else:
                        if total_reported > 0 and total_reported != self.total:
                            self.total = total_reported
                            self.step = max(1, math.ceil(self.total * 0.02))
                            self.next_target = self.step
                    self.direct_updates = True
                    self.last_processed = processed
                    self.logger.info(
                        "Rust cleaning progress: %d%% (%d/%d)",
                        percent,
                        processed,
                        self.total or total_reported,
                    )
                    if percent >= 100 or (
                        self.total and processed >= self.total
                    ):
                        self.logged_full = True
                    return
                match = re.search(r"Processing file:\s*(.+)", line)
                if match:
                    path = match.group(1).strip()
                    stem = Path(path).stem if path else None
                    if stem and stem not in self.processed:
                        self.processed.add(stem)
                        self._log_progress()
                    return
                if "complete pipeline finished successfully" in line or "Parquet report written successfully" in line:
                    self.last_message = line

            def _log_progress(self) -> None:
                if self.direct_updates:
                    return
                if self.total <= 0:
                    return
                processed = len(self.processed)
                while self.next_target <= self.total and processed >= self.next_target:
                    percent = min(100, int(round(self.next_target * 100 / self.total)))
                    self.logger.info(
                        "Rust cleaning progress: %d%% (%d/%d)", percent, processed, self.total
                    )
                    if percent >= 100:
                        self.logged_full = True
                    self.next_target += self.step

            def finalize(self) -> None:
                if self.total == 0:
                    self.logger.info("Rust cleaning progress: 100%% (0/0)")
                elif not self.logged_full:
                    processed = self.last_processed or len(self.processed)
                    self.logger.info(
                        "Rust cleaning progress: 100%% (%d/%d)", processed, self.total
                    )
                if self.last_message:
                    self.logger.debug(self.last_message)

        progress = _CleanerProgress(self.logger, total_files)
        cmd = (
            "import glossapi_rs_cleaner\n"
            f"glossapi_rs_cleaner.run_complete_pipeline({repr(str(input_dir))}, "
            f"{repr(str(self.cleaned_markdown_dir))}, {repr(str(report_parquet_path))}, "
            f"{repr(scripts_to_keep)}, {int(num_threads or os.cpu_count() or 4)}, "
            f"{'True' if write_cleaned_files else 'False'})\n"
        )

        process = subprocess.Popen(
            [sys.executable, "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                progress.handle_line(line)
            return_code = process.wait()
        except Exception:
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            progress.finalize()

        if return_code != 0:
            # Do not abort the entire cleaning pass – proceed to evaluate gates
            # using existing metrics on disk. If the Rust report is available,
            # it will be merged below as usual.
            self.logger.error("Rust cleaning pipeline failed (code=%s); proceeding with existing metrics", return_code)

        # ----- Parse metrics Parquet produced by Rust -----
        if report_parquet_path.exists():
            try:
                df_metrics_parquet = pd.read_parquet(report_parquet_path)
                for _, row in df_metrics_parquet.iterrows():
                    records.append(
                        {
                            "filename": f"{Path(row['file_name']).stem}.pdf",  # match original PDF filename
                            "badness_score": row.get("badness_score_all_chars", 0.0),
                            "percentage_greek": row.get("percentage_greek_cleaned"),
                            "percentage_latin": row.get("percentage_latin_cleaned"),
                            "char_count_no_comments": row.get("char_count_no_comments"),
                            "is_empty": row.get("is_empty", False),
                        }
                    )
            except Exception as e:
                self.logger.warning("Failed to parse cleaning report %s: %s", report_parquet_path, e)
        else:
            self.logger.warning("Cleaning report Parquet not found: %s", report_parquet_path)

        # ---- Delete cleaning report to avoid retaining it ----
        try:
            if report_parquet_path.exists():
                report_parquet_path.unlink(missing_ok=True)
                self.logger.debug("Deleted temporary cleaning report %s", report_parquet_path)
        except Exception as e:
            self.logger.warning("Could not delete cleaning report %s: %s", report_parquet_path, e)

        self.logger.info(f"Cleaned {len(records)} markdown files → {self.cleaned_markdown_dir}")

        # ------------------------------------------------------------------
        # Update parquet with Mojibake metrics (single authoritative schema)
        # ------------------------------------------------------------------
        if records:
            df_metrics = pd.DataFrame(records).rename(
                columns={
                    "badness_score": "mojibake_badness_score",
                    "percentage_latin": "mojibake_latin_percentage",
                }
            )

            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            df = self._load_metrics_dataframe(parquet_path, df_metrics.get("filename"))
            self._ensure_metric_columns(
                df,
                {
                    "mojibake_badness_score": pd.NA,
                    "mojibake_latin_percentage": pd.NA,
                    "percentage_greek": pd.NA,
                    "greek_badness_score": pd.NA,
                    "greek_latin_percentage": pd.NA,
                    "rejection_reason": pd.NA,
                    "char_count_no_comments": pd.NA,
                    "is_empty": pd.NA,
                },
            )

            df = self._merge_metric_dataframe(
                df,
                df_metrics[
                    [
                        "filename",
                        "mojibake_badness_score",
                        "mojibake_latin_percentage",
                        "percentage_greek",
                        "char_count_no_comments",
                        "is_empty",
                    ]
                ],
            )
            parquet_schema.write_metadata_parquet(df, parquet_path)
            self.logger.info("Mojibake metrics updated in %s", parquet_path)

        # ----- Noise-metrics scoring (Rust) -----
        try:
            self.logger.info("Scoring cleaned markdown files with glossapi_rs_noise …")
            noise_mod = self._load_rust_extension(
                "glossapi_rs_noise", "rust/glossapi_rs_noise/Cargo.toml"
            )
            results = noise_mod.score_markdown_directory_detailed(
                str(self.cleaned_markdown_dir), os.cpu_count()
            )
            if results:
                rows = []
                for row in results:
                    try:
                        path, score, latin_pct, _table_ratio, poly_ratio = row[:5]
                    except Exception:
                        continue
                    rows.append((path, float(score), float(latin_pct), float(poly_ratio)))

                df_scores = pd.DataFrame(
                    rows,
                    columns=[
                        "filepath",
                        "greek_badness_score",
                        "greek_latin_percentage",
                        "polytonic_ratio",
                    ],
                )
                df_scores["polytonic_ratio"] = df_scores["polytonic_ratio"].round(2)
                df_scores["stem"] = df_scores["filepath"].apply(lambda p: Path(p).name)
                df_scores["stem"] = df_scores["stem"].str.replace(r"\.md$", "", regex=True)
                df_scores["filename"] = df_scores["stem"] + ".pdf"
                df_scores["rejection_reason"] = np.select(
                    [df_scores["greek_badness_score"] > 60],
                    ["greek>60"],
                    default="ok",
                )
                if not parquet_path.exists():
                    self.logger.error(
                        "Expected parquet %s not found when adding noise metrics",
                        parquet_path,
                    )
                else:
                    df = self._load_metrics_dataframe(parquet_path)
                    self._ensure_metric_columns(
                        df,
                        {
                            "greek_badness_score": pd.NA,
                            "greek_latin_percentage": pd.NA,
                            "polytonic_ratio": pd.NA,
                            "rejection_reason": pd.NA,
                        },
                    )
                    updates = df_scores[
                        [
                            "filename",
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "polytonic_ratio",
                            "rejection_reason",
                        ]
                    ]
                    df = self._merge_metric_dataframe(df, updates)
                    parquet_schema.write_metadata_parquet(df, parquet_path)
                    self.logger.info("Noise metrics filled in %s", parquet_path)
        except Exception as e:
            self.logger.warning("Noise-metrics scoring failed: %s", e)

        # Determine good / bad list based on enriched metrics
        if parquet_path.exists():
            df_final = pd.read_parquet(parquet_path)
            self._ensure_metric_columns(
                df_final,
                {
                    "mojibake_badness_score": pd.NA,
                    "mojibake_latin_percentage": pd.NA,
                    "percentage_greek": pd.NA,
                    "greek_badness_score": pd.NA,
                    "greek_latin_percentage": pd.NA,
                    "char_count_no_comments": pd.NA,
                    "is_empty": pd.NA,
                },
            )
            # --- tidy schema ---
            df_final.rename(columns={
                "badness_score": "mojibake_badness_score",
                "percentage_latin": "mojibake_latin_percentage",
                "mojibake_latin_percentage": "latin_percentage",  # ADD THIS
                "rejection_reason": "filter"                      # ADD THIS
            }, inplace=True, errors="ignore")

            # drop duplicate pandas merge suffixes and keep clean names
            df_final = df_final.loc[:, ~df_final.columns.str.endswith('_x')]
            df_final.columns = df_final.columns.str.replace('_y$','', regex=True)

            # round Greek scores for readability
            for _col in ("greek_badness_score", "greek_latin_percentage"):
                if _col in df_final.columns:
                    df_final[_col] = pd.to_numeric(df_final[_col], errors="coerce").round(3)
            if "polytonic_ratio" in df_final.columns:
                df_final["polytonic_ratio"] = df_final["polytonic_ratio"].round(2)

            # drop any leftover placeholder columns to avoid duplicates
            df_final.drop(columns=["badness_score", "percentage_latin"], errors="ignore", inplace=True)
            # ADD: Drop unwanted columns
            df_final.drop(columns=["greek_latin_percentage", "badness_before", "badness_after"], errors="ignore", inplace=True)

            # ensure no duplicate column names
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]

            def _collapse_measure(df: pd.DataFrame, base: str) -> None:
                cols = [col for col in df.columns if col == base or col.startswith(f"{base}_")]
                if not cols:
                    return
                collapsed = None
                for col in cols:
                    values = pd.to_numeric(df[col], errors="coerce")
                    collapsed = values if collapsed is None else collapsed.combine_first(values)
                df[base] = collapsed
                for col in cols:
                    if col != base:
                        df.drop(columns=col, inplace=True, errors="ignore")

            _collapse_measure(df_final, "char_count_no_comments")
            _collapse_measure(df_final, "page_count")

            if "char_count_no_comments" in df_final.columns:
                df_final["char_count_no_comments"] = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce")
            if "page_count" in df_final.columns:
                df_final["page_count"] = pd.to_numeric(df_final["page_count"], errors="coerce")

            df_final["filter"] = "ok"
            df_final["needs_ocr"] = False
            if "is_empty" in df_final.columns:
                df_final["is_empty"] = df_final["is_empty"].fillna(False).astype(bool)
            else:
                df_final["is_empty"] = False

            filename_series = df_final.get("filename")
            if filename_series is None:
                pdf_mask = pd.Series(False, index=df_final.index)
            else:
                pdf_mask = filename_series.astype(str).str.lower().str.endswith(".pdf")
                pdf_mask = pdf_mask.fillna(False)

            def _append_reason(mask: pd.Series, reason: str, *, requires_ocr: bool) -> None:
                if df_final.empty:
                    return
                if not isinstance(mask, pd.Series):
                    mask = pd.Series(mask, index=df_final.index)
                mask = mask.fillna(False)
                applicable = (mask & pdf_mask).fillna(False)
                if not bool(applicable.any()):
                    return
                current = df_final.loc[applicable, "filter"].astype(str)

                def _merge_reason(value: str) -> str:
                    if value == "ok" or not value:
                        return reason
                    parts = [part for part in value.split(";") if part]
                    if reason not in parts:
                        parts.append(reason)
                    return ";".join(parts)

                df_final.loc[applicable, "filter"] = current.apply(_merge_reason)
                if requires_ocr:
                    needs_targets = applicable
                    if "ocr_success" in df_final.columns:
                        success_mask = df_final["ocr_success"].fillna(False)
                        needs_targets = needs_targets & ~success_mask
                    df_final.loc[needs_targets, "needs_ocr"] = True

            try:
                empty_threshold_int = int(empty_char_threshold) if empty_char_threshold is not None else 0
            except Exception:
                empty_threshold_int = 0
            if empty_threshold_int < 0:
                empty_threshold_int = 0
            try:
                min_pages = int(empty_min_pages) if empty_min_pages is not None else 0
            except Exception:
                min_pages = 0
            if min_pages < 0:
                min_pages = 0

            raw_moj = df_final.get("mojibake_badness_score")
            if isinstance(raw_moj, pd.Series):
                mojibake_series = pd.to_numeric(raw_moj, errors="coerce")
            else:
                mojibake_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
            if mojibake_series.notna().any():
                # Token policy: every OCR-trigger writes a filter tag.
                # Keep original label for compatibility with tests and downstream tools.
                _append_reason(mojibake_series > 0.1, "mojibake>0.1", requires_ocr=True)

            raw_gr = df_final.get("greek_badness_score")
            if isinstance(raw_gr, pd.Series):
                greek_series = pd.to_numeric(raw_gr, errors="coerce")
            else:
                greek_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
            if greek_series.notna().any():
                # Greek script gate: keep threshold (>60) as-is.
                # Use canonical token expected by tests and downstream tools.
                _append_reason(greek_series > 60, "non_greek_text", requires_ocr=True)

            if "char_count_no_comments" in df_final.columns:
                # Preserve NaN to avoid treating unknown counts as zero
                char_series = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce")
                page_series_raw = df_final.get("page_count")
                if page_series_raw is not None:
                    page_series = pd.to_numeric(page_series_raw, errors="coerce")
                else:
                    page_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
                page_series = page_series.fillna(min_pages if min_pages else 0)

                zero_mask = char_series <= 0
                zero_pdf = (zero_mask & pdf_mask).fillna(False)
                if bool(zero_pdf.any()):
                    df_final.loc[zero_pdf.index, "is_empty"] = df_final.loc[zero_pdf.index, "is_empty"] | zero_pdf
                if empty_threshold_int == 0:
                    zeros = int(zero_pdf.sum())
                    if zeros:
                        self.logger.info("Empty text check: %d files have zero characters", zeros)
                    # Strict-empty safeguard: rename token to "is_empty" and trigger OCR.
                    _append_reason(zero_pdf, "is_empty", requires_ocr=True)
                elif empty_threshold_int > 0:
                    low_mask = char_series < empty_threshold_int
                    long_mask = page_series >= max(1, min_pages)
                    _append_reason(low_mask & long_mask, f"empty_text<{empty_threshold_int}", requires_ocr=True)
                    if min_pages > 0:
                        _append_reason(low_mask & ~long_mask, f"empty_text<{empty_threshold_int}_short", requires_ocr=False)
                    total_low = int(low_mask.fillna(False).sum())
                    long_low = int((low_mask & long_mask).fillna(False).sum())
                    self.logger.info(
                        "Empty text check: %d files below %d chars; %d have >= %d pages",
                        total_low,
                        empty_threshold_int,
                        long_low,
                        min_pages,
                    )

            df_final["needs_ocr"] = df_final["needs_ocr"].fillna(False).astype(bool)

            # persist cleaned parquet
            parquet_schema.write_metadata_parquet(df_final, parquet_path)
            if drop_bad:
                good_df = df_final[df_final["needs_ocr"] == False]
                filenames = good_df.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
                self.logger.info(f"After filtering, {len(self.good_files)} good files remain")
            else:
                filenames = df_final.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
        else:
            self.good_files = []

        # After cleaning, point markdown_dir to cleaned files for downstream stages
        if write_cleaned_files:
            self.markdown_dir = self.cleaned_markdown_dir

    def filter(self, *args, **kwargs):  # type: ignore[override]
        """Deprecated: use :py:meth:`clean` instead.  Retained for one release."""
        self.logger.warning("Corpus.filter() is deprecated – calling clean() instead")
        self.clean(*args, **kwargs)
