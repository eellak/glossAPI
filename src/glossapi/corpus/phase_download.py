"""Download phase helpers split from Corpus."""
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


class DownloadPhaseMixin:
    def download(
        self,
        input_parquet: Optional[Union[str, Path]] = None,
        url_column: str = 'url',
        verbose: Optional[bool] = None,
        *,
        parallelize_by: Optional[str] = None,
        links_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download files from URLs in a parquet file.

        If input_parquet is not specified, it will automatically look for any .parquet file
        in the input_dir and use the first one found.

        Args:
            input_parquet: Path to input parquet file with URLs (optional)
                           If not provided, will search input_dir for parquet files
            url_column: Name of column containing URLs (defaults to 'url')
            verbose: Whether to enable verbose logging (overrides instance setting if provided)
            **kwargs: Additional parameters to override default downloader config

        Returns:
            pd.DataFrame: DataFrame with download results
        """
        # If input_parquet not specified, find parquet files in input_dir
        if input_parquet is None:
            parquet_files = list(self.input_dir.glob('*.parquet'))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {self.input_dir}")
            input_parquet = parquet_files[0]
            self.logger.info(f"Using parquet file: {input_parquet}")
        else:
            input_parquet = Path(input_parquet)

        # Load the input file with URLs to download
        original_input_filename = Path(input_parquet).name
        input_df = pd.read_parquet(input_parquet)
        total_urls = len(input_df)
        self.logger.info(f"Total URLs in input file: {total_urls}")

        # Respect links_column override early so resume filter uses correct column name
        if links_column:
            url_column = links_column

        # Look for existing download results file by the specific input filename first
        input_filename = Path(input_parquet).name
        download_results_dir = Path(self.output_dir) / "download_results"
        specific_results_path = download_results_dir / f"download_results_{input_filename}"
        partial_results_path = download_results_dir / f"download_results_{input_filename}.partial.parquet"

        existing_results = None
        existing_results_path = None
        found_existing = False

        # Check for specific download results file
        if os.path.exists(specific_results_path):
            self.logger.info(f"Found existing download results: {specific_results_path}")
            try:
                existing_results = pd.read_parquet(specific_results_path)
                existing_results_path = specific_results_path
                found_existing = True
            except Exception as e:
                self.logger.warning(f"Failed to read specific download results: {e}")
        elif os.path.exists(partial_results_path):
            self.logger.info(f"Found partial download checkpoint: {partial_results_path}")
            try:
                existing_results = pd.read_parquet(partial_results_path)
                existing_results_path = partial_results_path
                found_existing = True
            except Exception as e:
                self.logger.warning(f"Failed to read partial results: {e}")

        # If specific results not found, look in the directory for any download results
        if not found_existing and os.path.exists(download_results_dir):
            result_files = list(download_results_dir.glob('*.parquet'))
            for file in result_files:
                try:
                    test_df = pd.read_parquet(file)
                    if url_column in test_df.columns and 'download_success' in test_df.columns:
                        self.logger.info(f"Found alternative download results: {file}")
                        existing_results = test_df
                        existing_results_path = file
                        found_existing = True
                        break
                except Exception:
                    continue

        # Filter out already downloaded URLs and prepare to download only remaining ones
        if found_existing and url_column in existing_results.columns:
            # Find filenames that have already been assigned (whether download succeeded or not)
            # to ensure we don't reuse the same filenames and overwrite files
            existing_filenames = []
            if 'filename' in existing_results.columns:
                existing_filenames = existing_results['filename'].dropna().tolist()
                self.logger.info(f"Found {len(existing_filenames)} existing filenames to avoid")

            # Build the set of successful URLs from checkpoint/results
            successful_urls = []
            if 'download_success' in existing_results.columns:
                successful_urls = existing_results[
                    existing_results['download_success'] == True
                ][url_column].dropna().astype(str).tolist()

            if successful_urls:
                self.logger.info(f"Found {len(successful_urls)} previously successful downloads")

                # If input uses list/JSON URLs, expand to one-URL-per-row before filtering
                def _looks_like_list(s: str) -> bool:
                    try:
                        t = str(s).strip()
                        return t.startswith('[') or t.startswith('{')
                    except Exception:
                        return False

                need_expand = False
                try:
                    sample = input_df[url_column].dropna().astype(str).head(50).tolist()
                    need_expand = any(_looks_like_list(x) for x in sample)
                except Exception:
                    need_expand = False

                if need_expand:
                    try:
                        # Reuse downloader's expansion to mirror runtime behavior
                        dl_tmp = GlossDownloader(url_column=url_column, output_dir=str(self.output_dir))
                        expanded_df = dl_tmp._expand_and_mark_duplicates(input_df.copy())  # type: ignore[attr-defined]
                        # Keep URL and provenance columns if present
                        keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug") if c in expanded_df.columns]
                        remaining_df = expanded_df[~expanded_df[url_column].isin(successful_urls)][keep_cols]
                        self.logger.info(
                            f"Expanded list/JSON URLs to {len(expanded_df)} rows; pending {len(remaining_df)}"
                        )
                    except Exception:
                        # Fallback: basic JSON parse expansion
                        import json as _json
                        rows = []
                        for _, row in input_df.iterrows():
                            val = row.get(url_column)
                            if isinstance(val, str) and _looks_like_list(val):
                                try:
                                    arr = _json.loads(val)
                                    if isinstance(arr, list):
                                        rows.extend([str(u) for u in arr if isinstance(u, (str,))])
                                    elif isinstance(arr, dict):
                                        u = arr.get('url') or arr.get('href') or arr.get('link')
                                        if u:
                                            rows.append(str(u))
                                except Exception:
                                    pass
                            elif isinstance(val, str) and val.strip():
                                rows.append(val.strip())
                        import pandas as _pd
                        expanded_df = _pd.DataFrame({url_column: rows})
                        keep_cols = [url_column]
                        remaining_df = expanded_df[~expanded_df[url_column].isin(successful_urls)][keep_cols]
                        self.logger.info(
                            f"Expanded (fallback) to {len(expanded_df)} rows; pending {len(remaining_df)}"
                        )
                else:
                    # Simple string URLs: filter directly and keep provenance if present
                    keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug") if c in input_df.columns]
                    remaining_df = input_df[~input_df[url_column].isin(successful_urls)][keep_cols]

                # If all URLs already downloaded, return existing results
                if len(remaining_df) == 0:
                    self.logger.info("All files already successfully downloaded")
                    return existing_results

                self.logger.info(
                    f"Processing {len(remaining_df)} remaining URLs after skipping successes"
                )

                # Save filtered per-URL input to a temporary file for the downloader
                temp_input = self.output_dir / "temp_download_input.parquet"
                remaining_df.to_parquet(temp_input, index=False)
                input_parquet = temp_input
        else:
            self.logger.info("No existing download results found or usable")
            existing_results = pd.DataFrame()

        # Initialize downloader configuration (kwargs take precedence)
        dl_cfg = dict(self.downloader_config)
        dl_cfg.update(kwargs)
        # Allow caller to override which column holds links
        if links_column:
            url_column = links_column
        # Allow caller to choose grouping for scheduler (e.g., 'collection_slug' or 'base_domain')
        if parallelize_by:
            dl_cfg['scheduler_group_by'] = parallelize_by
        # Build used filename bases set to avoid collisions on resume
        used_bases = set()
        try:
            used_bases |= {canonical_stem(str(fn)) for fn in existing_filenames if isinstance(fn, str)}
        except Exception:
            pass
        try:
            # Also include on-disk stems
            downloads_dir = Path(self.output_dir) / 'downloads'
            if downloads_dir.exists():
                used_bases |= {canonical_stem(p) for p in downloads_dir.glob('*') if p.is_file()}
        except Exception:
            pass

        downloader = GlossDownloader(
            url_column=url_column,
            output_dir=str(self.output_dir),
            log_level=self.logger.level,
            verbose=verbose if verbose is not None else self.verbose,
            **{k: v for k, v in dl_cfg.items() if k not in {'input_parquet'}},
            _used_filename_bases=used_bases
        )

        # Download files
        self.logger.info(f"Downloading files from URLs in {input_parquet}...")
        new_results = downloader.download_files(input_parquet=str(input_parquet))

        # Merge with existing results
        if not existing_results.empty:
            # Filter out rows from existing_results that are in new_results (based on URL)
            if url_column in new_results.columns and url_column in existing_results.columns:
                processed_urls = new_results[url_column].tolist()
                existing_filtered = existing_results[~existing_results[url_column].isin(processed_urls)]

                # Combine existing and new results
                final_results = pd.concat([existing_filtered, new_results], ignore_index=True)
                self.logger.info(f"Merged {len(existing_filtered)} existing results with {len(new_results)} new results")
        else:
            final_results = new_results

        # Ensure we have a download_results directory
        os.makedirs(download_results_dir, exist_ok=True)

        # Save results using the input filename pattern
        output_parquet = download_results_dir / f"download_results_{original_input_filename}"
        final_results.to_parquet(output_parquet, index=False)
        self.logger.info(f"Saved download results to {output_parquet}")

        # Clean up temporary files if created
        temp_path = self.output_dir / "temp_download_input.parquet"
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Report download completion
        success_count = len(final_results[final_results['download_success'] == True]) if 'download_success' in final_results.columns else 0
        self.logger.info(f"Download complete. {success_count} files downloaded to {self.output_dir / 'downloads'}")

        return final_results
