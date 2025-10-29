"""Export helpers split from Corpus."""
from __future__ import annotations

import hashlib
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
import zstandard as zstd

from .._naming import canonical_stem
from ..gloss_downloader import GlossDownloader
from ..gloss_section import GlossSection
# Avoid importing classifier at import time; export phase does not require it.
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch


class ExportPhaseMixin:
    def _iter_jsonl_records(
        self,
        *,
        text_key: str,
        metadata_key: Optional[str],
        metadata_fields: Optional[Iterable[str]],
        include_remaining_metadata: bool,
        metadata_path: Optional[Union[str, Path]],
        source_metadata_key: Optional[str],
        source_metadata_fields: Optional[Iterable[str]],
        source_metadata_path: Optional[Union[str, Path]],
    ) -> Iterable[dict[str, Any]]:
        download_dir = self.output_dir / "download_results"

        chunk_suffix_re = re.compile(r"__p\d{4}-\d{4}$")
        page_range_re = re.compile(r"__p(\d+)-(\d+)")

        def _strip_chunk_suffix(stem: str) -> str:
            if not stem:
                return stem
            return chunk_suffix_re.sub("", stem)

        def _strip_chunk_from_filename(value: str) -> str:
            if not value:
                return value
            candidate = value.strip()
            try:
                path = Path(candidate)
            except Exception:
                path = None
            if path is not None:
                suffix = path.suffix
                base = _strip_chunk_suffix(path.stem)
                if suffix:
                    return f"{base}{suffix}"
                return base
            return _strip_chunk_suffix(candidate)

        def _chunk_sort_key(path: Path) -> tuple[int, int, str]:
            match = page_range_re.search(path.stem)
            if match:
                try:
                    start = int(match.group(1))
                    end = int(match.group(2))
                except Exception:
                    start, end = sys.maxsize, sys.maxsize
                return start, end, path.stem
            return sys.maxsize, sys.maxsize, path.stem

        if metadata_path is not None:
            metadata_path = Path(metadata_path)
        else:
            metadata_path = download_dir / "download_results.parquet"
            if not metadata_path.exists():
                candidates = sorted(download_dir.glob("*.parquet")) if download_dir.exists() else []
                if not candidates:
                    raise FileNotFoundError(f"Metadata parquet not found in {download_dir}")
                preferred = [p for p in candidates if p.name.startswith("download_results_")]
                metadata_path = preferred[0] if preferred else candidates[0]

        df = pd.read_parquet(metadata_path)
        if df.empty:
            return

        def _stem_for(value: str) -> str:
            if not value:
                return ""
            return canonical_stem(value)

        df["__stem__"] = df["filename"].astype(str).map(_stem_for)
        metadata_by_stem: dict[str, dict[str, Any]] = {}
        metadata_is_chunk: dict[str, bool] = {}
        for _, row in df.iterrows():
            data = row.to_dict()
            stem = str(data.pop("__stem__", ""))
            if not stem:
                continue
            base_stem = _strip_chunk_suffix(stem)
            key = base_stem or stem
            is_chunk = bool(chunk_suffix_re.search(stem))
            existing_is_chunk = metadata_is_chunk.get(key)
            if key not in metadata_by_stem or (existing_is_chunk and not is_chunk):
                metadata_by_stem[key] = data
                metadata_is_chunk[key] = is_chunk

        source_metadata_by_stem: dict[str, dict[str, Any]] = {}
        source_metadata_is_chunk: dict[str, bool] = {}
        source_fields_order: Optional[List[str]] = None
        if source_metadata_key:
            assert source_metadata_path is not None  # guarded upstream
            source_metadata_path = Path(source_metadata_path)
            if not source_metadata_path.exists():
                raise FileNotFoundError(f"Source metadata parquet not found at {source_metadata_path}")
            source_df = pd.read_parquet(source_metadata_path)
            if source_df.empty:
                raise ValueError(f"Source metadata parquet {source_metadata_path} is empty")
            if source_metadata_fields is not None:
                source_fields_order = [str(field) for field in source_metadata_fields]
                missing_columns = [field for field in source_fields_order if field not in source_df.columns]
                if missing_columns:
                    raise KeyError(f"Source metadata columns {missing_columns} not found in {source_metadata_path}")
            for _, row in source_df.iterrows():
                data = row.to_dict()
                filename_raw = data.get("filename")
                if not isinstance(filename_raw, str) or not filename_raw.strip():
                    continue
                stem = canonical_stem(str(filename_raw))
                if not stem:
                    continue
                base_stem = _strip_chunk_suffix(stem)
                key = base_stem or stem
                is_chunk = bool(chunk_suffix_re.search(stem))
                existing_is_chunk = source_metadata_is_chunk.get(key)
                if key not in source_metadata_by_stem or (existing_is_chunk and not is_chunk):
                    source_metadata_by_stem[key] = data
                    source_metadata_is_chunk[key] = is_chunk

        markdown_root = self.cleaned_markdown_dir
        chunks_root = markdown_root / "chunks"
        if not markdown_root.exists():
            markdown_root = self.markdown_dir
            chunks_root = markdown_root / "chunks"
        else:
            has_root_markdown = any(markdown_root.glob("*.md"))
            has_chunk_markdown = chunks_root.exists() and any(chunks_root.rglob("*.md"))
            if not has_root_markdown and not has_chunk_markdown:
                markdown_root = self.markdown_dir
                chunks_root = markdown_root / "chunks"

        document_entries: dict[str, dict[str, Any]] = {}

        def _document_entry(key: str) -> dict[str, Any]:
            return document_entries.setdefault(key, {"base_path": None, "chunk_paths": []})

        for md_path in sorted(markdown_root.glob("*.md")):
            stem = canonical_stem(md_path)
            if not stem:
                continue
            base_stem = _strip_chunk_suffix(stem)
            key = base_stem or stem
            entry = _document_entry(key)
            entry["base_path"] = md_path

        if chunks_root.exists():
            for chunk_path in sorted(chunks_root.rglob("*.md")):
                stem = canonical_stem(chunk_path)
                if not stem:
                    continue
                base_stem = _strip_chunk_suffix(stem)
                key = base_stem or stem
                entry = _document_entry(key)
                entry.setdefault("chunk_paths", []).append(chunk_path)

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

        metadata_fields_filter: Optional[Set[str]] = None
        metadata_fields_order: Optional[List[str]] = None
        if metadata_fields is not None:
            metadata_fields_order = [str(field) for field in metadata_fields]
            metadata_fields_filter = set(metadata_fields_order)

        for stem in sorted(document_entries):
            entry = document_entries[stem]
            chunk_paths: List[Path] = entry.get("chunk_paths", []) or []
            base_path: Optional[Path] = entry.get("base_path")
            representative_path: Optional[Path] = base_path
            if representative_path is None and chunk_paths:
                representative_path = sorted(chunk_paths, key=_chunk_sort_key)[0]
            metadata = metadata_by_stem.get(stem)
            if metadata is None:
                continue
            metadata = {k: _normalize_value(v) for k, v in metadata.items()}
            original_filename_value = metadata.get("filename")
            if chunk_paths:
                ordered_chunks = sorted(chunk_paths, key=_chunk_sort_key)
                parts: List[str] = []
                for path in ordered_chunks:
                    parts.append(path.read_text(encoding="utf-8"))
                document_text = "\n".join(parts)
            elif representative_path is not None:
                document_text = representative_path.read_text(encoding="utf-8")
            else:
                continue

            filetype = metadata.get("filetype") or metadata.get("file_ext")
            if not filetype:
                filename_candidate = original_filename_value or metadata.get("filename")
                if isinstance(filename_candidate, str):
                    filetype = Path(filename_candidate).suffix.lstrip(".")
            metadata["filetype"] = filetype or None

            filename_value = metadata.get("filename")
            filename_base: Optional[str] = None
            if isinstance(filename_value, str) and filename_value.strip():
                filename_base = _strip_chunk_suffix(Path(filename_value).stem)
            if not filename_base and representative_path is not None:
                filename_base = _strip_chunk_suffix(representative_path.stem)
            if not filename_base:
                filename_base = stem
            metadata["filename"] = filename_base

            filename_label = None
            if isinstance(original_filename_value, str) and original_filename_value.strip():
                filename_label = _strip_chunk_from_filename(original_filename_value)
            if not filename_label and representative_path is not None:
                filename_label = _strip_chunk_from_filename(representative_path.name)
            if not filename_label:
                filename_label = stem
            doc_id = hashlib.sha256(filename_label.encode("utf-8")).hexdigest()
            metadata["doc_id"] = doc_id

            metrics_page_count, metrics_formula, metrics_code = _load_metrics(stem)
            if metrics_page_count is not None:
                metadata["page_count"] = metrics_page_count

            existing_formula = metadata.get("formula_total")
            if metrics_formula is not None:
                metadata["formula_total"] = metrics_formula
            else:
                try:
                    metadata["formula_total"] = int(existing_formula)
                except Exception:
                    metadata["formula_total"] = 0

            existing_code = metadata.get("code_total")
            if metrics_code is not None:
                metadata["code_total"] = metrics_code
            else:
                try:
                    metadata["code_total"] = int(existing_code)
                except Exception:
                    metadata["code_total"] = 0

            math_enriched, math_accepted = _load_math_accepts(stem)
            metadata["math_enriched"] = bool(math_enriched)
            metadata["math_accepted"] = int(math_accepted)

            record: dict[str, Any] = {text_key: document_text, "doc_id": doc_id, "chunk_id": 0}

            if metadata_key:
                if metadata_fields_order is None:
                    filtered = dict(metadata)
                else:
                    filtered = {k: metadata[k] for k in metadata_fields_order if k in metadata}
                record[metadata_key] = filtered
                if include_remaining_metadata:
                    for key, value in metadata.items():
                        if metadata_fields_filter is not None and key in metadata_fields_filter:
                            continue
                        record[key] = value
            else:
                record.update(metadata)

            if source_metadata_key:
                source_entry = source_metadata_by_stem.get(stem)
                fallback_name = representative_path.name if representative_path is not None else stem
                source_lookup_label = original_filename_value or metadata.get("filename") or fallback_name
                source_lookup_label = _strip_chunk_from_filename(str(source_lookup_label))
                if source_entry is None:
                    raise KeyError(f"Missing source metadata for filename '{source_lookup_label}'")
                assert source_fields_order is not None
                filtered_source: dict[str, Any] = {}
                for key in source_fields_order:
                    if key not in source_entry:
                        raise KeyError(f"Missing source metadata column '{key}' for filename '{source_lookup_label}'")
                    raw_value = source_entry[key]
                    if pd.isna(raw_value):
                        raise ValueError(
                            f"Source metadata field '{key}' has no value for filename '{source_lookup_label}'. "
                            "Use 'NA' explicitly if the value is unavailable."
                        )
                    if isinstance(raw_value, str):
                        if not raw_value.strip():
                            raise ValueError(
                                f"Source metadata field '{key}' is blank for filename '{source_lookup_label}'. "
                                "Use 'NA' explicitly if the value is unavailable."
                            )
                    filtered_source[key] = _normalize_value(raw_value)
                for key, value in filtered_source.items():
                    if value is None:
                        raise ValueError(
                            f"Source metadata field '{key}' resolved to null for filename '{source_lookup_label}'. "
                            "Use 'NA' explicitly if the value is unavailable."
                        )
                record[source_metadata_key] = filtered_source

            record["filename"] = filename_base
            yield record

    def jsonl(
        self,
        output_path: Union[str, Path],
        *,
        text_key: str = "document",
        metadata_key: Optional[str] = None,
        metadata_fields: Optional[Iterable[str]] = None,
        include_remaining_metadata: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        source_metadata_key: Optional[str] = None,
        source_metadata_fields: Optional[Iterable[str]] = None,
        source_metadata_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Export cleaned markdown and metadata into a JSONL corpus.

        Args:
            output_path: Destination JSONL file.
            text_key: Key that should hold the markdown text in each record.
            metadata_key: Optional key under which to nest metadata. When omitted,
                metadata columns are written at the top level (legacy behaviour).
            metadata_fields: Optional iterable of metadata column names to include
                under ``metadata_key``. When ``None`` all metadata columns are used.
            include_remaining_metadata: When ``metadata_key`` is provided, control
                whether any metadata columns not captured by ``metadata_fields`` are
                written at the top level.
            metadata_path: Optional explicit parquet file containing the metadata.
                Defaults to ``download_results/download_results.parquet`` (or the
                first matching parquet file in that directory).
            source_metadata_key: Optional key under which to nest additional source
                metadata sourced from a separate parquet.
            source_metadata_fields: Optional iterable of source metadata column names
                to include under ``source_metadata_key``.
            source_metadata_path: Optional parquet file that holds the source metadata.
        """

        if source_metadata_key and source_metadata_path is None:
            raise ValueError("source_metadata_path must be provided when source_metadata_key is set")
        if source_metadata_key and source_metadata_fields is None:
            raise ValueError("source_metadata_fields must be provided when source_metadata_key is set")
        if source_metadata_path and not source_metadata_key:
            raise ValueError("source_metadata_key must be provided when source_metadata_path is set")
        if source_metadata_fields and not source_metadata_key:
            raise ValueError("source_metadata_key must be provided when source_metadata_fields are set")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records_written = 0
        with output_path.open("w", encoding="utf-8") as fp:
            for record in self._iter_jsonl_records(
                text_key=text_key,
                metadata_key=metadata_key,
                metadata_fields=metadata_fields,
                include_remaining_metadata=include_remaining_metadata,
                metadata_path=metadata_path,
                source_metadata_key=source_metadata_key,
                source_metadata_fields=source_metadata_fields,
                source_metadata_path=source_metadata_path,
            ):
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

        if records_written == 0:
            output_path.write_text("", encoding="utf-8")
        return output_path

    def jsonl_sharded(
        self,
        output_dir: Union[str, Path],
        *,
        shard_size_bytes: int = 500 * 1024 * 1024,
        shard_prefix: str = "train",
        compression: str = "zstd",
        compression_level: int = 3,
        text_key: str = "document",
        metadata_key: Optional[str] = None,
        metadata_fields: Optional[Iterable[str]] = None,
        include_remaining_metadata: bool = True,
        metadata_path: Optional[Union[str, Path]] = None,
        source_metadata_key: Optional[str] = None,
        source_metadata_fields: Optional[Iterable[str]] = None,
        source_metadata_path: Optional[Union[str, Path]] = None,
    ) -> List[Path]:
        """Export sharded JSONL files with optional compression."""

        if shard_size_bytes <= 0:
            raise ValueError("shard_size_bytes must be positive")
        codec = compression.lower()
        if codec not in {"zstd", "none"}:
            raise ValueError(f"Unsupported compression codec '{compression}'")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_paths: List[Path] = []
        shard_index = -1
        bytes_written = 0
        writer = None

        def _close_writer() -> None:
            nonlocal writer
            if writer is None:
                return
            if codec == "zstd":
                writer.flush(zstd.FLUSH_FRAME)
            writer.close()
            writer = None

        for record in self._iter_jsonl_records(
            text_key=text_key,
            metadata_key=metadata_key,
            metadata_fields=metadata_fields,
            include_remaining_metadata=include_remaining_metadata,
            metadata_path=metadata_path,
            source_metadata_key=source_metadata_key,
            source_metadata_fields=source_metadata_fields,
            source_metadata_path=source_metadata_path,
        ):
            line = json.dumps(record, ensure_ascii=False) + "\n"
            encoded = line.encode("utf-8")
            if writer is None or bytes_written + len(encoded) > shard_size_bytes:
                _close_writer()
                shard_index += 1
                bytes_written = 0
                shard_stem = f"{shard_prefix}-{shard_index:06d}.jsonl"
                if codec == "zstd":
                    shard_path = output_dir / f"{shard_stem}.zst"
                    compressor = zstd.ZstdCompressor(level=compression_level)
                    writer = compressor.stream_writer(shard_path.open("wb"))
                else:
                    shard_path = output_dir / shard_stem
                    writer = shard_path.open("wb")
                shard_paths.append(shard_path)
            assert writer is not None
            writer.write(encoded)
            bytes_written += len(encoded)

        _close_writer()
        return shard_paths

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
