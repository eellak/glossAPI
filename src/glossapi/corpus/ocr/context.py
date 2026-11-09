"""Shared typing contracts for corpus OCR helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class CorpusOcrContext(Protocol):
    logger: Any
    input_dir: Path
    output_dir: Path
    markdown_dir: Path
    logs_dir: Path
    url_column: str
    good_files: list[str]

    def _resolve_metadata_parquet(self, *args: Any, **kwargs: Any) -> Path | None: ...

    def _cache_metadata_parquet(self, path: Path | None) -> None: ...

    def _get_cached_metadata_parquet(self) -> Path | None: ...

    def clean(self, *args: Any, **kwargs: Any) -> None: ...

    def formula_enrich_from_json(self, *args: Any, **kwargs: Any) -> None: ...
