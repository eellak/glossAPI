"""Helpers for normalising file identifiers across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Union

_KNOWN_SUFFIXES = (
    ".docling.json.zst",
    ".docling.json",
    ".latex_map.jsonl",
    ".per_page.metrics.json",
    ".metrics.json",
    ".jsonl",
    ".json",
    ".md",
    ".pdf",
    ".html",
    ".htm",
)


def canonical_stem(value: Union[str, Path]) -> str:
    """Return a normalised stem for any pipeline artefact."""

    name = Path(value).name
    working = name
    stripped = True
    while stripped and working:
        stripped = False
        for suffix in _KNOWN_SUFFIXES:
            if working.endswith(suffix):
                working = working[: -len(suffix)]
                stripped = True
                break
    if working:
        return working
    fallback = Path(name).stem
    return fallback or name
