"""Download phase helpers split from Corpus."""
from __future__ import annotations

from ..ledger.journal import shadow_stage, observe, observe_store, observe_source_rows

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
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._naming import canonical_stem
# Avoid importing section/classifier here; download phase does not use them.
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch


class DownloadPhaseMixin:
    @shadow_stage("download", "download.wrapper")
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

        Public targets only: a URL with userinfo or a credential query/fragment
        field is refused before any request (``credentials_unsupported``), and so
        is a credential setting. Recorded results (``download_results*.parquet`` and
        the partial checkpoint) hold the URLs as given plus ``resource_identity``;
        a resume matches previous rows by resource identity.

        Returns:
            pd.DataFrame: DataFrame with download results
        """
        from ..gloss_downloader import (GlossDownloader, RESOURCE_IDENTITY_COLUMN, frame_identities,
                                        persisted_download_frame)
        from ..parquet_schema import lossless_concat, pandas_parquet, same_metadata_file, update_metadata_parquet
        BrowserGlossDownloader = None

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

        replay = getattr(self, '_streaming_download_replays', None) is not None

        # Look for existing download results file by the specific input filename first
        input_filename = Path(input_parquet).name
        download_results_dir = Path(self.output_dir) / "download_results"
        specific_results_path = download_results_dir / f"download_results_{input_filename}"
        partial_results_path = download_results_dir / f"download_results_{input_filename}.partial.parquet"

        existing_results = None
        existing_results_path = None
        found_existing = False
        input_frame = None

        # Check for specific download results file. A streaming run's publication replays every committed
        # acquisition of its frozen plan (K4BP-G1/G4): earlier results never filter it (a later run re-acquires a
        # source; a killed publication restarts); its rows replace earlier rows of the same identity below.
        if replay:
            pass
        elif os.path.exists(specific_results_path):
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
        if not found_existing and not replay and os.path.exists(download_results_dir):
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

            # Previously successful rows, matched by resource identity (recorded,
            # or computed from the URL for results without the column).
            successful_urls = []
            successful_identities = set()
            if 'download_success' in existing_results.columns:
                done = existing_results[existing_results['download_success'] == True]
                successful_urls = done[url_column].dropna().astype(str).tolist()
                successful_identities = {identity for identity in frame_identities(done, url_column) if identity}
            def _pending(frame):
                identities = frame_identities(frame, url_column)
                return [not ((identity in successful_identities) if identity else (str(url) in successful_urls))
                        for identity, url in zip(identities, frame[url_column].tolist())]

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
                        keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug", "source_id", RESOURCE_IDENTITY_COLUMN) if c in expanded_df.columns]
                        remaining_df = expanded_df[_pending(expanded_df)][keep_cols]
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
                        remaining_df = expanded_df[_pending(expanded_df)][keep_cols]
                        self.logger.info(
                            f"Expanded (fallback) to {len(expanded_df)} rows; pending {len(remaining_df)}"
                        )
                else:
                    # Simple string URLs: filter directly and keep provenance if present
                    keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug", "source_id", RESOURCE_IDENTITY_COLUMN) if c in input_df.columns]
                    remaining_df = input_df[_pending(input_df)][keep_cols]

                # If all URLs already downloaded, return existing results
                if len(remaining_df) == 0:
                    self.logger.info("All files already successfully downloaded")
                    recorded = persisted_download_frame(existing_results, url_column)
                    if existing_results_path is not None and not recorded.equals(existing_results):
                        # An older results file has no resource_identity column: rewrite it with one, applied to
                        # the file as it is now, inside the metadata lock (CP-K2a F1): rows or columns another
                        # writer committed since the read above are kept, and a removed file is not recreated.
                        rewritten = []
                        def _with_identity(current):
                            if not Path(existing_results_path).exists():
                                return None
                            current_recorded = persisted_download_frame(current, url_column)
                            rewritten.append(current_recorded)
                            return None if current_recorded.equals(current) else current_recorded
                        update_metadata_parquet(existing_results_path, _with_identity, serialize=pandas_parquet)
                        recorded = rewritten[0] if rewritten else recorded
                    return recorded

                self.logger.info(
                    f"Processing {len(remaining_df)} remaining URLs after skipping successes"
                )

                # The remaining URLs go to the downloader in memory; the
                # temporary name only keeps the checkpoint name unchanged.
                temp_input = self.output_dir / "temp_download_input.parquet"
                input_frame = remaining_df
                input_parquet = temp_input
        else:
            self.logger.info("No existing download results found or usable")
            existing_results = pd.DataFrame()

        # Initialize downloader configuration (kwargs take precedence)
        dl_cfg = dict(self.downloader_config)
        dl_cfg.update(kwargs)
        browser_mode = dl_cfg.pop('browser_mode', None)
        if browser_mode is not None and 'download_mode' not in dl_cfg:
            dl_cfg['download_mode'] = 'browser' if browser_mode else 'standard'
        download_mode = str(dl_cfg.pop('download_mode', 'standard')).strip().lower()
        policy_requested = bool(dl_cfg.get('download_policy_file') or dl_cfg.get('download_policy'))
        if download_mode in {'standard', 'default', 'http'} and not policy_requested:
            downloader_cls = GlossDownloader
            default_download_route = 'standard'
        elif download_mode in {'browser', 'browser_protected'} or policy_requested:
            from ..gloss_browser_downloader import BrowserGlossDownloader
            downloader_cls = BrowserGlossDownloader
            default_download_route = 'browser' if download_mode in {'browser', 'browser_protected'} else 'standard'
        elif download_mode in {'auto', 'browser_fallback'}:
            from ..gloss_browser_downloader import BrowserGlossDownloader
            downloader_cls = BrowserGlossDownloader
            default_download_route = 'auto'
        else:
            raise ValueError(f"Unsupported download_mode: {download_mode}")
        # Allow caller to override which column holds links
        if links_column:
            url_column = links_column
        # Allow caller to choose grouping for scheduler (e.g., 'collection_slug' or 'base_domain')
        if parallelize_by:
            dl_cfg['scheduler_group_by'] = parallelize_by
        # Build used filename bases set to avoid collisions on resume
        used_bases = set()
        if not replay:  # a replaying publication reproduces the frozen names exactly (K4BP-G1); nothing on disk shifts them
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

        downloader_kwargs = {
            "url_column": url_column,
            "output_dir": str(self.output_dir),
            "log_level": self.logger.level,
            "verbose": verbose if verbose is not None else self.verbose,
            **{k: v for k, v in dl_cfg.items() if k not in {'input_parquet'}},
            "_used_filename_bases": used_bases,
        }
        if downloader_cls is BrowserGlossDownloader:
            downloader_kwargs["default_download_route"] = default_download_route

        downloader = downloader_cls(**downloader_kwargs)
        from ..stages.profile import adapters_enabled
        if adapters_enabled(self):
            from ..stages.download_compatibility import install_download_adapter
            install_download_adapter(self,downloader,{k:v for k,v in downloader_kwargs.items() if k!='output_dir'},download_mode)

        # Download files
        self.logger.info(f"Downloading files from URLs in {input_parquet}...")
        try:
            new_results = downloader.download_files(input_parquet=str(input_parquet), input_frame=input_frame)
        finally:
            # This call's request/progress logs are complete; no later
            # downloader or Corpus in this process may append to them.
            close_logs = getattr(downloader, 'close_logs', None)
            if callable(close_logs):
                close_logs()

        # Ensure we have a download_results directory
        os.makedirs(download_results_dir, exist_ok=True)

        # Save results using the input filename pattern
        output_parquet = download_results_dir / f"download_results_{original_input_filename}"

        def _merge(base):
            # Merge with existing results
            if base.empty:
                return new_results
            # Filter out rows from the existing results that are in new_results (based on URL)
            if url_column in new_results.columns and url_column in base.columns:
                # Replace previous rows of the same resource identity.
                processed = set(frame_identities(new_results, url_column)) - {None}
                processed_urls = set(new_results[url_column].astype(str).tolist())
                existing_filtered = base[[
                    (identity not in processed) if identity else (str(url) not in processed_urls)
                    for identity, url in zip(frame_identities(base, url_column), base[url_column].tolist())]]

                # Combine existing and new results
                self.logger.info(f"Merged {len(existing_filtered)} existing results with {len(new_results)} new results")
                return lossless_concat([existing_filtered, new_results], what='download results')
            # Rows without the URL column cannot be matched: keep them (before CP-K2a F1 this was an unbound name).
            return lossless_concat([base, new_results], what='download results')

        # Publication (CP-K2a F1): the network work above ran without the lock. Under the shared metadata lock the
        # results file is re-read as it is now, this acquisition's rows replace only earlier rows of the same
        # resource identity, every other row and column (another writer's commit since the read above) is kept,
        # and the file is replaced atomically. With nothing concurrent the bytes are those of the previous
        # read-then-write (the same rows, the same ``to_parquet`` serializer).
        published = []
        def _publish(current):
            base = current
            if (existing_results_path is not None and not same_metadata_file(existing_results_path, output_parquet)
                    and not existing_results.empty):
                # Rows resumed from the partial checkpoint or another results file belong to this publication.
                if base.empty:
                    base = existing_results
                else:
                    present = set(frame_identities(base, url_column)) - {None}
                    resumed = existing_results[[identity not in present for identity in frame_identities(existing_results, url_column)]]
                    if len(resumed):
                        base = lossless_concat([base, resumed], what='resumed download results')
            final = _merge(base)
            published.append(final)
            return persisted_download_frame(final, url_column)

        update_metadata_parquet(output_parquet, _publish, serialize=pandas_parquet)
        # Streaming publication and sequential download() alike: the canonical metadata table, when the corpus has
        # one, receives these rows, identity-safe (CP-K4a F1, CP-K4ac2 R1).
        from ..parquet_schema import fold_download_rows
        fold_download_rows(self.output_dir, output_parquet)
        final_results = published[0]
        self.logger.info(f"Saved download results to {output_parquet}")

        # Clean up temporary files if created
        temp_path = self.output_dir / "temp_download_input.parquet"
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Report download completion
        success_count = len(final_results[final_results['download_success'] == True]) if 'download_success' in final_results.columns else 0
        self.logger.info(f"Download complete. {success_count} files downloaded to {self.output_dir / 'downloads'}")

        return final_results
