"""
Standardized Parquet Schema definitions for GlossAPI pipeline.

This module defines standard schemas for parquet files used throughout the GlossAPI
pipeline, ensuring consistency between different pipeline stages.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import pandas as pd

try:  # Optional dependency – lazily confirmed via _ensure_pyarrow()
    import pyarrow as pa  # type: ignore[import]
    import pyarrow.parquet as pq  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when pyarrow missing
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - optional helper
    FileLock = None

from ._naming import canonical_stem

_BOOLEAN_METADATA_COLUMNS: Tuple[str, ...] = (
    "download_success",
    "is_duplicate",
    "needs_ocr",
    "ocr_success",
    "math_enriched",
    "enriched_math",
    "is_empty",
)


def _ensure_pyarrow() -> Tuple[Any, Any]:
    """Import pyarrow on demand, surfacing a friendly error if missing."""

    global pa, pq  # noqa: PLW0603 - shared cache for optional dependency
    if pa is not None and pq is not None:
        return pa, pq
    try:
        import pyarrow as pa_mod  # type: ignore[import]
        import pyarrow.parquet as pq_mod  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "pyarrow is required to read or write metadata parquets. "
            "Install it with `pip install pyarrow` inside the active environment."
        ) from exc
    pa = pa_mod  # type: ignore[assignment]
    pq = pq_mod  # type: ignore[assignment]
    return pa, pq


def _coerce_bool_value(value: Any) -> object:
    """Normalize common truthy/falsey representations to pandas boolean domain."""

    if value is None:
        return pd.NA
    if value is pd.NA:
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            if pd.isna(value):
                return pd.NA
        except Exception:
            pass
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
        if lowered in {"", "none", "null"}:
            return pd.NA
        return pd.NA
    try:
        return bool(value)
    except Exception:
        return pd.NA


def _boolean_series(series: pd.Series) -> pd.Series:
    values = [_coerce_bool_value(v) for v in series.tolist()]
    return pd.Series(pd.array(values, dtype="boolean"), index=series.index)


def _nullable_or(lhs: object, rhs: object) -> object:
    lhs_norm = _coerce_bool_value(lhs)
    rhs_norm = _coerce_bool_value(rhs)
    if lhs_norm is True or rhs_norm is True:
        return True
    if lhs_norm is pd.NA or rhs_norm is pd.NA:
        return pd.NA
    return False


def _prepare_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized boolean columns and legacy aliases aligned."""

    if df.empty:
        return df.copy()
    result = df.copy()

    math_series: Optional[pd.Series] = None
    if "math_enriched" in result.columns:
        math_series = _boolean_series(result["math_enriched"])
    if "enriched_math" in result.columns:
        enriched_series = _boolean_series(result["enriched_math"])
        if math_series is None:
            math_series = enriched_series
        else:
            math_series = math_series.combine(enriched_series, _nullable_or)
    if math_series is not None:
        math_series = math_series.astype("boolean")
        result["math_enriched"] = math_series
        result["enriched_math"] = math_series

    for column in _BOOLEAN_METADATA_COLUMNS:
        if column not in result.columns:
            continue
        result[column] = _boolean_series(result[column])

    return result


@contextmanager
def _parquet_lock(path: Path) -> Iterator[None]:
    """Serialize writers via file-based lock, falling back to atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    if FileLock is not None:
        lock = FileLock(str(lock_path))
        try:
            with lock:
                yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        return

    acquired = False
    handle: Optional[int] = None
    try:
        while not acquired:
            try:
                handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                acquired = True
            except FileExistsError:
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            try:
                os.close(handle)
            except Exception:
                pass
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _write_metadata_parquet(df: pd.DataFrame, parquet_path: Path) -> None:
    prepared = _prepare_metadata_frame(df)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pa_mod, pq_mod = _ensure_pyarrow()
    table = pa_mod.Table.from_pandas(prepared, preserve_index=False)
    tmp_path = parquet_path.with_suffix(parquet_path.suffix + f".{uuid.uuid4().hex}.tmp")
    with _parquet_lock(parquet_path):
        try:
            pq_mod.write_table(table, tmp_path)
            os.replace(tmp_path, parquet_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass


class ParquetSchema:
    """
    Defines standardized schema for parquet files in the GlossAPI pipeline.
    
    This class provides methods to validate, read, and write parquet files
    with consistent schemas for different pipeline stages.
    
    The pipeline uses two distinct types of parquet files:
    
    1. Metadata Parquet:
       - Each row represents a file (one-to-one relationship with files)
       - Essential columns: filename, URL column (configurable), extraction quality
       - Used by: downloader, extractor, and filter stages
       - Example: download_results.parquet
       - Typical location: {output_dir}/download_results/
       - Schema: METADATA_SCHEMA, DOWNLOAD_SCHEMA
    
    2. Sections Parquet:
       - Each row represents a section from a file (many-to-one relationship with files)
       - Essential columns: filename, title, content, section, predicted_section
       - Used by: section and annotation stages
       - Examples: sections_for_annotation.parquet, classified_sections.parquet
       - Typical location: {output_dir}/sections/
       - Schema: SECTION_SCHEMA, CLASSIFIED_SCHEMA
    
    When the pipeline runs, it first creates and populates a metadata parquet,
    then uses it to filter files, and finally creates section parquets from the
    filtered files.
    """
    
    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ParquetSchema with optional pipeline configuration.
        
        Args:
            pipeline_config: Configuration dictionary with settings such as
                url_column, which will be used throughout the pipeline
        """
        # TODO: Add more robust configuration options for each parquet type from input metadata and downloder, to section, and two phases of annotaiton.
        # TODO: Add support for consolidated sections parquet handling
        # TODO: Add methods to find the latest sections parquet in a pipeline
        self.config = pipeline_config or {}
        self.url_column = self.config.get('url_column', 'url')
    
    # Basic schema with common fields used across all parquet files
    if pa is not None:
        COMMON_SCHEMA = pa.schema(
            [
                ("id", pa.string()),
                ("row_id", pa.int64()),
                ("filename", pa.string()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        COMMON_SCHEMA = None
    
    # Metadata schema for files used by downloader and quality assessment
    if pa is not None:
        METADATA_SCHEMA = pa.schema(
            [
                ("filename", pa.string()),
                ("url", pa.string()),  # Can be customized with url_column parameter
                ("download_success", pa.bool_()),
                ("download_error", pa.string()),
                ("trigrams", pa.string()),  # Values: "natural", "unnatural", "unknown"
                ("processing_stage", pa.string()),  # Tracks progress through pipeline
                ("badness_score", pa.float64()),
                ("percentage_greek", pa.float64()),
                ("percentage_latin", pa.float64()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        METADATA_SCHEMA = None
    
    # Additional schemas for specific pipeline stages
    if pa is not None:
        DOWNLOAD_SCHEMA = pa.schema(
            [
                ("url", pa.string()),  # Will be replaced with the actual url_column
                ("download_success", pa.bool_()),
                ("download_error", pa.string()),
                ("download_retry_count", pa.int32()),
                ("filename", pa.string()),
                ("file_ext", pa.string()),
                ("is_duplicate", pa.bool_()),
                ("duplicate_of", pa.string()),
                ("source_row", pa.int32()),
                ("url_index", pa.int32()),
                ("filename_base", pa.string()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        DOWNLOAD_SCHEMA = None
    
    if pa is not None:
        SECTION_SCHEMA = pa.schema(
            [
                ("id", pa.string()),
                ("row_id", pa.int64()),
                ("filename", pa.string()),
                ("title", pa.string()),
                ("content", pa.string()),
                ("section", pa.string()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        SECTION_SCHEMA = None
    
    if pa is not None:
        CLASSIFIED_SCHEMA = pa.schema(
            [
                ("id", pa.string()),
                ("row_id", pa.int64()),
                ("filename", pa.string()),
                ("title", pa.string()),
                ("content", pa.string()),
                ("section", pa.string()),
                ("predicted_section", pa.string()),
                ("probability", pa.float64()),
            ]
        )
    else:  # pragma: no cover - pyarrow not installed
        CLASSIFIED_SCHEMA = None
    
    def get_required_metadata(self) -> Dict[str, str]:
        """
        Get required metadata fields for GlossAPI parquet files.
        
        Returns:
            Dict[str, str]: Dictionary of required metadata fields and their descriptions
        """
        return {
            'pipeline_version': 'GlossAPI pipeline version',
            'created_at': 'ISO format timestamp when the file was created',
            'source_file': 'Original source file that generated this parquet',
            'processing_stage': 'Pipeline processing stage (download, extract, section, etc)'
        }
    
    def validate_schema(self, df: pd.DataFrame, schema_type: str = 'common') -> Tuple[bool, List[str]]:
        """
        Validate that a DataFrame conforms to the specified schema.
        
        Args:
            df: DataFrame to validate
            schema_type: Type of schema to validate against ('common', 'download', 'section', 'classified', 'metadata')
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_columns)
        """
        if schema_type.lower() == 'download':
            required_columns = [field.name for field in self.DOWNLOAD_SCHEMA]
            # Make sure to use the configured url_column
            if self.url_column != 'url' and 'url' in required_columns:
                required_columns.remove('url')
                required_columns.append(self.url_column)
        elif schema_type.lower() == 'section':
            required_columns = [field.name for field in self.SECTION_SCHEMA]
        elif schema_type.lower() == 'classified':
            required_columns = [field.name for field in self.CLASSIFIED_SCHEMA]
        elif schema_type.lower() == 'metadata':
            required_columns = ['filename']
            # Make sure to use the configured url_column
            required_columns.append(self.url_column)  
        else:  # Default to common schema
            required_columns = [field.name for field in self.COMMON_SCHEMA]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        return len(missing_columns) == 0, missing_columns
    
    def add_metadata(self, table: pa.Table, metadata: Dict[str, str]) -> pa.Table:
        """
        Add metadata to a PyArrow Table.
        
        Args:
            table: PyArrow Table to add metadata to
            metadata: Dictionary of metadata to add
            
        Returns:
            pa.Table: Table with added metadata
        """
        # Add pipeline configuration to metadata
        if self.config:
            for key, value in self.config.items():
                if key not in metadata:
                    metadata[f'config_{key}'] = str(value)
        # Convert all metadata values to strings
        metadata_bytes = {k.encode(): str(v).encode() for k, v in metadata.items()}
        
        # Add required metadata if missing
        required_metadata = self.get_required_metadata()
        for key in required_metadata:
            if key not in metadata:
                metadata_bytes[key.encode()] = f"MISSING: {required_metadata[key]}".encode()
        
        return table.replace_schema_metadata(metadata_bytes)
    
    def read_parquet(self, file_path: Union[str, Path], validate: bool = True, schema_type: str = 'common') -> pd.DataFrame:
        """
        Read a parquet file with validation.
        
        Args:
            file_path: Path to parquet file
            validate: Whether to validate the schema
            schema_type: Type of schema to validate against
            
        Returns:
            pd.DataFrame: DataFrame from parquet file
        """
        df = pd.read_parquet(file_path)
        
        if validate:
            is_valid, missing_columns = self.validate_schema(df, schema_type)
            if not is_valid:
                print(f"Warning: Parquet file {file_path} is missing required columns: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error', 'file_ext']:
                        df[col] = ''
                    elif col in ['row_id', 'download_retry_count', 'source_row', 'url_index']:
                        df[col] = 0
                    elif col == 'download_success':
                        df[col] = False
                    elif col == 'is_duplicate':
                        df[col] = False
                    elif col == 'probability':
                        df[col] = 0.0
                    elif col == 'filename_base':
                        df[col] = ''
        
        return df
    
    def find_metadata_parquet(self, directory: Union[str, Path], require_url_column: bool = False) -> Optional[Path]:
        """
        Find the first valid metadata parquet file in a directory.
        
        Looks for parquet files that don't have section-specific columns
        like 'title' and 'header', and prioritizes files with the url_column.
        
        Args:
            directory: Directory to search for parquet files
            require_url_column: If True, require the URL column to be present; if False, only require filename column
            
        Returns:
            Optional[Path]: Path to the first valid metadata parquet, or None if not found
        """
        import logging
        logger = logging.getLogger(__name__)
        
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Directory {directory} does not exist")
            return None
            
        # Get all parquet files in the directory
        parquet_files = [
            f for f in directory.glob('**/*.parquet')
            if '.partial.' not in f.name and not f.name.endswith('.partial.parquet')
        ]
        if not parquet_files:
            logger.debug(f"No parquet files found in {directory}")
            return None
            
        # Check for download_results files first
        download_files = [f for f in parquet_files if 'download_results' in str(f)]
        if download_files:
            logger.debug(f"Found {len(download_files)} download_results files")
        
        # If any download_results parquets exist, check them first
        search_order = download_files + [f for f in parquet_files if f not in download_files]
        metrics_signature = {"filter", "mojibake_badness_score", "greek_badness_score"}
        downloader_signature = {"download_success", "download_error", "filename_base"}

        # First pass: prefer metrics parquet if present anywhere in the tree
        for file_path in search_order:
            try:
                df = pd.read_parquet(file_path)
                columns = df.columns.tolist()
                if 'section' in columns:
                    continue
                if metrics_signature.issubset(set(columns)):
                    logger.info(f"Found metrics parquet with cleaner columns: {file_path}")
                    return file_path
            except Exception as e:
                logger.debug(f"Error reading parquet {file_path}: {e}")
                continue

        # Second pass: look for downloader metadata parquet
        for file_path in search_order:
            try:
                df = pd.read_parquet(file_path)
                columns = df.columns.tolist()

                # Skip section-level parquets – identified by a 'section' column
                if 'section' in columns:
                    logger.debug(f"Skipping section-level parquet: {file_path}")
                    continue

                col_set = set(columns)

                if downloader_signature.issubset(col_set):
                    if require_url_column and self.url_column not in col_set:
                        logger.warning(
                            "Downloader parquet missing required %s column: %s",
                            self.url_column,
                            file_path,
                        )
                        logger.debug(f"Available columns: {columns}")
                        continue
                    logger.info(f"Found downloader metadata parquet: {file_path}")
                    return file_path

                # For other metadata parquets - they don't have title/header but have filename
                if 'filename' in columns:
                    if require_url_column:
                        # Check if required URL column exists
                        if self.url_column in columns:
                            logger.info(f"Found metadata parquet with filename and {self.url_column}: {file_path}")
                            return file_path
                        else:
                            # Missing URL column
                            logger.warning(f"Found parquet with filename column but no {self.url_column} column: {file_path}")
                            logger.debug(f"Available columns: {columns}")
                    else:
                        # URL not required, filename is enough
                        logger.info(f"Found metadata parquet with filename (URL not required): {file_path}")
                        return file_path
                else:
                    logger.debug(f"Found parquet without filename column: {file_path}")
            except Exception as e:
                logger.debug(f"Error reading parquet {file_path}: {e}")
                continue
                
        logger.warning(f"No suitable metadata parquet found in {directory}")
        return None
    
    def ensure_metadata_parquet(self, base_dir: Union[str, Path]) -> Optional[Path]:
        """
        Ensure that a consolidated metadata parquet exists for a pipeline run.

        When the canonical ``download_results/download_results.parquet`` file is
        missing, this method gathers the available artifacts (downloads, markdown,
        metrics, math sidecars, etc.) and synthesises a best-effort metadata table
        so downstream stages always have a consistent entry point.

        Returns:
            Optional[Path]: Path to the ensured parquet, or ``None`` when no rows
            could be inferred.
        """
        import logging
        import math

        base_dir = Path(base_dir)
        download_results_dir = base_dir / "download_results"
        parquet_path = download_results_dir / "download_results.parquet"
        if parquet_path.exists():
            return parquet_path

        download_results_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)

        existing_df = pd.DataFrame()
        target_path: Optional[Path] = None
        existing_candidates = sorted(
            f
            for f in download_results_dir.glob("*.parquet")
            if ".partial." not in f.name and f.name != parquet_path.name
        )
        if existing_candidates:
            candidate = existing_candidates[0]
            try:
                existing_df = pd.read_parquet(candidate)
                target_path = candidate
            except Exception as exc:
                logger.debug("Failed to read existing metadata parquet %s: %s", candidate, exc)
                existing_df = pd.DataFrame()
                target_path = candidate

        defaults: Dict[str, Any] = {
            self.url_column: "",
            "filename": "",
            "file_ext": "",
            "document_type": pd.NA,
            "download_success": False,
            "download_error": "",
            "download_retry_count": 0,
            "is_duplicate": False,
            "duplicate_of": "",
            "source_row": pd.NA,
            "url_index": pd.NA,
            "filename_base": "",
            "filter": "ok",
            "extract_status": "",
            "extract_error": "",
            "needs_ocr": False,
            "ocr_success": False,
            "processing_stage": "",
            "mojibake_badness_score": pd.NA,
            "mojibake_latin_percentage": pd.NA,
            "percentage_greek": pd.NA,
            "char_count_no_comments": pd.NA,
            "is_empty": False,
            "page_count": pd.NA,
            "pages_total": pd.NA,
            "pages_with_formula": pd.NA,
            "formula_total": pd.NA,
            "formula_avg_pp": pd.NA,
            "formula_p90_pp": pd.NA,
            "phase_recommended": pd.NA,
            "math_enriched": False,
            "math_items": pd.NA,
            "math_accepted": pd.NA,
            "math_accept_rate": pd.NA,
            "math_time_sec": pd.NA,
        }

        records: Dict[str, Dict[str, Any]] = {}

        def _append_stage(row: Dict[str, Any], stage: str) -> None:
            if not stage:
                return
            current = row.get("processing_stage") or ""
            stages = [part for part in current.split(",") if part]
            if stage not in stages:
                stages.append(stage)
            row["processing_stage"] = ",".join(stages)

        def _row_for(stem: str) -> Dict[str, Any]:
            canonical = canonical_stem(stem)
            row = records.get(canonical)
            if row is None:
                row = {key: value for key, value in defaults.items()}
                row["filename"] = f"{canonical}.pdf"
                row["file_ext"] = "pdf"
                row["filename_base"] = canonical
                records[canonical] = row
            return row

        # 1) Original downloads (PDFs)
        downloads_dir = base_dir / "downloads"
        pdf_paths: List[Path] = []
        if downloads_dir.exists():
            pdf_paths.extend(sorted(p for p in downloads_dir.rglob("*.pdf") if p.is_file()))
        pdf_paths.extend(sorted(p for p in base_dir.glob("*.pdf") if p.is_file()))

        for pdf_path in pdf_paths:
            stem = canonical_stem(pdf_path)
            row = _row_for(stem)
            if not row.get(self.url_column):
                row[self.url_column] = ""
            row["filename"] = pdf_path.name
            row["file_ext"] = pdf_path.suffix.lstrip(".").lower()
            row["filename_base"] = stem
            row["download_success"] = True
            row["download_error"] = ""
            _append_stage(row, "download")

        # 2) Markdown outputs (extraction + cleaning)
        markdown_sources = [
            (base_dir / "markdown", "extract"),
            (base_dir / "clean_markdown", "clean"),
        ]
        for md_dir, stage in markdown_sources:
            if not md_dir.exists():
                continue
            for md_path in md_dir.glob("*.md"):
                if not md_path.is_file():
                    continue
                stem = canonical_stem(md_path)
                row = _row_for(stem)
                if not row["filename"]:
                    row["filename"] = f"{stem}.pdf"
                row["filename_base"] = stem
                _append_stage(row, "extract")
                if stage == "clean":
                    _append_stage(row, "clean")

        # 3) Latex maps indicate successful math enrichment
        json_dir = base_dir / "json"
        if json_dir.exists():
            for latex_map in json_dir.glob("*.latex_map.jsonl"):
                if not latex_map.is_file():
                    continue
                stem = canonical_stem(latex_map)
                row = _row_for(stem)
                row["math_enriched"] = True
                _append_stage(row, "math")

        # 4) Metrics JSON (page counts, formula counts)
        def _ingest_metrics_payload(stem: str, payload: Dict[str, Any]) -> None:
            row = _row_for(stem)
            page_count = payload.get("page_count")
            if page_count is not None:
                try:
                    row["page_count"] = int(page_count)
                except Exception:
                    row["page_count"] = page_count
            pages = payload.get("pages") or []
            if isinstance(pages, list) and pages:
                row["pages_total"] = len(pages)
                try:
                    formula_counts = [int(p.get("formula_count", 0) or 0) for p in pages]
                except Exception:
                    formula_counts = []
                if formula_counts:
                    total_formula = sum(formula_counts)
                    if pd.isna(row["formula_total"]):
                        row["formula_total"] = total_formula
                    row["formula_avg_pp"] = (
                        float(total_formula) / max(1, len(formula_counts))
                    )
                    formula_counts_sorted = sorted(formula_counts)
                    idx = max(0, math.ceil(0.9 * len(formula_counts_sorted)) - 1)
                    row["formula_p90_pp"] = float(formula_counts_sorted[idx])
                    row["pages_with_formula"] = sum(1 for c in formula_counts if c > 0)

        metrics_dir = json_dir / "metrics" if json_dir.exists() else None
        if metrics_dir and metrics_dir.exists():
            for metrics_file in metrics_dir.glob("*.metrics.json"):
                if not metrics_file.is_file():
                    continue
                stem = canonical_stem(metrics_file)
                try:
                    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    _ingest_metrics_payload(stem, payload)
            for per_page in metrics_dir.glob("*.per_page.metrics.json"):
                if not per_page.is_file():
                    continue
                name = per_page.name
                if not name.endswith(".per_page.metrics.json"):
                    continue
                stem = canonical_stem(per_page)
                try:
                    payload = json.loads(per_page.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(payload, dict):
                    _ingest_metrics_payload(stem, payload)

        # 5) Triage sidecars (phase recommendation, formula stats)
        triage_dir = base_dir / "sidecars" / "triage"
        if triage_dir.exists():
            for triage_file in triage_dir.glob("*.json"):
                if not triage_file.is_file():
                    continue
                try:
                    payload = json.loads(triage_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem = canonical_stem(triage_file)
                row = _row_for(stem)
                for key in ("formula_total", "formula_avg_pp", "formula_p90_pp", "pages_total", "pages_with_formula"):
                    if key in payload and pd.isna(row.get(key)):
                        row[key] = payload.get(key)
                if "phase_recommended" in payload and pd.isna(row.get("phase_recommended")):
                    row["phase_recommended"] = payload.get("phase_recommended")

        # 6) Math sidecars (items, acceptance rate, duration)
        math_dir = base_dir / "sidecars" / "math"
        if math_dir.exists():
            for math_file in math_dir.glob("*.json"):
                if not math_file.is_file():
                    continue
                try:
                    payload = json.loads(math_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem = canonical_stem(math_file)
                row = _row_for(stem)
                items = payload.get("items")
                accepted = payload.get("accepted")
                time_sec = payload.get("time_sec")
                if items is not None:
                    row["math_items"] = items
                if accepted is not None:
                    row["math_accepted"] = accepted
                if items:
                    try:
                        row["math_accept_rate"] = float(accepted) / float(items) if accepted is not None else row["math_accept_rate"]
                    except Exception:
                        pass
                if time_sec is not None:
                    row["math_time_sec"] = time_sec
                row["math_enriched"] = True
                _append_stage(row, "math")

        df_records = pd.DataFrame(list(records.values())) if records else pd.DataFrame()
        # Ensure column order is stable for readability
        ordered_columns = list(defaults.keys())
        if "filename" not in ordered_columns:
            ordered_columns.insert(0, "filename")
        if self.url_column not in ordered_columns:
            ordered_columns.insert(0, self.url_column)
        df_records = df_records.reindex(columns=ordered_columns, fill_value=pd.NA)
        if not df_records.empty:
            df_records.sort_values(by="filename", inplace=True)

        if target_path is not None:
            combined = df_records
            if combined.empty and not existing_df.empty:
                combined = existing_df.copy()
            elif not combined.empty and not existing_df.empty and "filename" in existing_df.columns:
                base_idx = combined.set_index("filename", drop=False)
                existing_idx = existing_df.set_index("filename", drop=False)
                base_idx = base_idx.combine_first(existing_idx)
                base_idx.update(existing_idx)
                combined = base_idx.reset_index(drop=True)
            elif combined.empty:
                combined = existing_df.copy()

            if combined.empty:
                logger.warning(
                    "Unable to update metadata parquet under %s – no artifacts discovered",
                    base_dir,
                )
                return target_path

            for column, default in defaults.items():
                if column not in combined.columns:
                    combined[column] = default
            combined = combined.reindex(columns=ordered_columns, fill_value=pd.NA)
            combined.sort_values(by="filename", inplace=True)
            _write_metadata_parquet(combined, target_path)
            logger.info(
                "Updated existing metadata parquet with %d row(s): %s",
                len(combined),
                target_path,
            )
            return target_path

        if df_records.empty:
            logger.warning(
                "Unable to synthesise metadata parquet under %s – no artifacts discovered",
                base_dir,
            )
            return None

        df_records.sort_values(by="filename", inplace=True)
        _write_metadata_parquet(df_records, parquet_path)
        logger.info(
            "Synthesised metadata parquet with %d row(s): %s",
            len(df_records),
            parquet_path,
        )
        return parquet_path

    @staticmethod
    def normalize_metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Expose metadata normalisation for callers who need in-memory updates."""

        return _prepare_metadata_frame(df)

    def write_metadata_parquet(self, df: pd.DataFrame, parquet_path: Union[str, Path]) -> None:
        """Persist metadata updates with schema/dtype normalisation and locking."""

        target = Path(parquet_path)
        _write_metadata_parquet(df, target)

    def is_valid_metadata_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file conforms to the metadata schema used by downloader.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has the required metadata fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for url_column (which might be custom) and filename
            required_fields = [self.url_column, 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def create_basic_metadata_parquet(self, markdown_dir: Union[str, Path], output_dir: Union[str, Path]) -> Union[Path, None]:
        """
        Create a simple metadata parquet file from a directory of markdown files.
        This is used when there is no existing parquet file to update.
        
        Args:
            markdown_dir: Directory containing markdown files
            output_dir: Directory where to create the parquet file
            
        Returns:
            Path: Path to the created parquet file, or None if creation failed
        """
        pipeline_root = Path(output_dir)
        ensured = self.ensure_metadata_parquet(pipeline_root)
        if ensured is not None:
            return ensured

        try:
            markdown_dir = Path(markdown_dir)
            
            # Create output directory if it doesn't exist
            download_results_dir = pipeline_root / "download_results"
            os.makedirs(download_results_dir, exist_ok=True)
            
            # Get all markdown files in the input directory
            markdown_files = list(markdown_dir.glob("*.md"))
            if not markdown_files:
                print(f"No markdown files found in {markdown_dir}")
                return None
                
            # Create a DataFrame with just filenames
            data = []
            for md_file in markdown_files:
                entry = {
                    'filename': md_file.name,
                    self.url_column: ""  # Minimal URL placeholder
                }
                data.append(entry)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Set output path for the parquet file
            output_path = download_results_dir / "download_results.parquet"
            
            # Write to parquet without adding complex metadata
            pq.write_table(pa.Table.from_pandas(df), output_path)
            
            print(f"Created new metadata parquet file at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating metadata parquet file: {e}")
            return None
            
    def is_download_result_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains download results with success/error information.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has download result fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for download result fields
            required_fields = ['download_success', 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def is_sections_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains section data from extracted files.
        This identifies the second type of parquet in the pipeline - the sections parquet.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has section data fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for required section fields
            required_fields = ['filename', 'title', 'content', 'section']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
        
    def add_processing_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Add or update processing stage column in a DataFrame.
        
        Args:
            df: Input DataFrame to update
            stage: Processing stage value to set (e.g., 'downloaded', 'extracted', 'classified')
            
        Returns:
            pd.DataFrame: Updated DataFrame with processing_stage column
        """
        df['processing_stage'] = stage
        return df
        
    def verify_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if a DataFrame contains all required columns and return missing ones.
        
        Args:
            df: DataFrame to check
            required_columns: List of column names that should be present
            
        Returns:
            Tuple containing:
            - bool: True if all required columns are present
            - List[str]: List of missing columns (empty if all present)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        return (len(missing_columns) == 0, missing_columns)
    
    def write_parquet(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        schema_type: str = 'common',
        validate: bool = True
    ) -> None:
        """
        Write a DataFrame to parquet with standard schema and metadata.
        
        Args:
            df: DataFrame to write
            file_path: Path to write parquet file
            metadata: Dictionary of metadata to include
            schema_type: Type of schema to use
            validate: Whether to validate the schema before writing
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Validate and fix schema if needed
        if validate:
            is_valid, missing_columns = self.validate_schema(df_copy, schema_type)
            if not is_valid:
                print(f"Adding missing columns to DataFrame: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error', 'file_ext']:
                        df_copy[col] = ''
                    elif col in ['row_id', 'download_retry_count', 'source_row', 'url_index']:
                        df_copy[col] = 0
                    elif col == 'download_success':
                        df_copy[col] = False
                    elif col == 'is_duplicate':
                        df_copy[col] = False
                    elif col == 'probability':
                        df_copy[col] = 0.0
                    elif col == 'filename_base':
                        df_copy[col] = ''
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df_copy)
        
        # Add metadata if provided
        if metadata:
            table = self.add_metadata(table, metadata)
        
        # Write to parquet
        pq.write_table(table, file_path)
        print(f"Parquet file written to {file_path} with schema type '{schema_type}'")
