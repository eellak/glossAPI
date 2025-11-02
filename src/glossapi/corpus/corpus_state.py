"""Persistence helpers for corpus processing state using parquet metadata files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

from .._naming import canonical_stem
from ..parquet_schema import ParquetSchema


class _ProcessingStateManager:
    """Maintain the resume checkpoints using parquet metadata files instead of pickle.

    This replaces the previous pickle-based state management with parquet-based tracking.
    State is now stored in the metadata parquet file with stage flags (stage_extract, etc.)
    to track which files have been processed in each stage.
    """

    def __init__(self, metadata_parquet_path: Optional[Path] = None, base_dir: Optional[Path] = None) -> None:
        """
        Initialize the state manager.
        
        Args:
            metadata_parquet_path: Direct path to metadata parquet file (optional)
            base_dir: Base directory where metadata parquet should be located (optional)
                      If metadata_parquet_path is not provided, will search in base_dir
        """
        self.logger = logging.getLogger(__name__)
        self.metadata_parquet_path = metadata_parquet_path
        self.base_dir = base_dir
        self.schema = ParquetSchema()
        
        # Determine metadata parquet path
        if self.metadata_parquet_path is None and self.base_dir is not None:
            # Try to find or ensure metadata parquet in base_dir
            download_results_dir = Path(base_dir) / "download_results"
            candidate = download_results_dir / "download_results.parquet"
            if candidate.exists():
                self.metadata_parquet_path = candidate
            else:
                # Ensure metadata parquet exists
                ensured = self.schema.ensure_metadata_parquet(base_dir)
                if ensured:
                    self.metadata_parquet_path = ensured

    def _load_metadata_df(self) -> pd.DataFrame:
        """Load the metadata parquet file as a DataFrame."""
        if self.metadata_parquet_path is None or not self.metadata_parquet_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(self.metadata_parquet_path)
            return df
        except Exception as exc:
            self.logger.warning("Failed to load metadata parquet %s: %s", self.metadata_parquet_path, exc)
            return pd.DataFrame()

    def _save_metadata_df(self, df: pd.DataFrame) -> None:
        """Save the metadata DataFrame to parquet."""
        if self.metadata_parquet_path is None:
            if self.base_dir is None:
                self.logger.error("Cannot save state: no metadata parquet path or base_dir provided")
                return
            # Ensure metadata parquet exists
            download_results_dir = Path(self.base_dir) / "download_results"
            download_results_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_parquet_path = download_results_dir / "download_results.parquet"
        
        try:
            self.schema.write_metadata_parquet(df, self.metadata_parquet_path)
        except Exception as exc:
            self.logger.warning("Failed to save metadata parquet %s: %s", self.metadata_parquet_path, exc)

    def load(self) -> Tuple[Set[str], Set[str]]:
        """
        Load processed and problematic files from metadata parquet.
        
        Returns:
            Tuple of (processed_files, problematic_files) sets
        """
        df = self._load_metadata_df()
        if df.empty:
            return set(), set()
        
        # Files that have successfully completed extract stage are considered processed
        processed = set()
        problematic = set()
        
        if "filename" in df.columns:
            for _, row in df.iterrows():
                filename = str(row.get("filename", ""))
                if not filename:
                    continue
                
                # Check extract_success flag (processed) or extract_error (problematic)
                extract_success = row.get("extract_success", False)
                extract_error = row.get("extract_error", "")
                stage_extract = row.get("stage_extract", False)
                
                # A file is processed if extract succeeded
                if extract_success or stage_extract:
                    processed.add(filename)
                
                # A file is problematic if it has an extract error
                if extract_error and str(extract_error).strip():
                    problematic.add(filename)
        
        return processed, problematic

    def save(self, processed: Set[str], problematic: Set[str]) -> None:
        """
        Save processed and problematic files to metadata parquet.
        
        This updates the stage_extract, extract_success, and extract_error columns
        in the metadata parquet file.
        """
        df = self._load_metadata_df()
        
        # Create mapping from filename to status
        filename_to_status = {}
        for filename in processed:
            filename_to_status[filename] = {"processed": True, "problematic": False}
        for filename in problematic:
            filename_to_status[filename] = {"processed": False, "problematic": True}
        
        # Update DataFrame
        if df.empty:
            # Create new DataFrame if empty
            data = []
            for filename in processed | problematic:
                canonical = canonical_stem(filename)
                row = {
                    "filename": filename,
                    self.schema.url_column: "",
                    "stage_extract": filename in processed,
                    "extract_success": filename in processed,
                    "extract_error": "" if filename not in problematic else "Processing failed",
                }
                data.append(row)
            if data:
                df = pd.DataFrame(data)
            else:
                return  # Nothing to save
        else:
            # Update existing rows
            for idx, row in df.iterrows():
                filename = str(row.get("filename", ""))
                if not filename:
                    continue
                
                if filename in processed:
                    df.at[idx, "stage_extract"] = True
                    df.at[idx, "extract_success"] = True
                    df.at[idx, "extract_error"] = ""
                elif filename in problematic:
                    df.at[idx, "stage_extract"] = False
                    df.at[idx, "extract_success"] = False
                    if pd.isna(df.at[idx, "extract_error"]) or not str(df.at[idx, "extract_error"]).strip():
                        df.at[idx, "extract_error"] = "Processing failed"
            
            # Add new rows for files not in DataFrame
            existing_filenames = set(df["filename"].astype(str))
            for filename in processed | problematic:
                if filename not in existing_filenames:
                    canonical = canonical_stem(filename)
                    new_row = {
                        "filename": filename,
                        self.schema.url_column: "",
                        "stage_extract": filename in processed,
                        "extract_success": filename in processed,
                        "extract_error": "" if filename not in problematic else "Processing failed",
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        self._save_metadata_df(df)
