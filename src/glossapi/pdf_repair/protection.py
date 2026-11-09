"""Protect all selected sources and reserve PDF/staging output pathnames."""

import os
from pathlib import Path
from typing import Iterable


def protect_sources(target: str | Path, sources: Iterable[str | Path]) -> None:
    """Refuse resolved aliases and hard links before opening an output."""
    target = Path(target)
    resolved = target.resolve()
    try:
        stat = os.stat(target)
        identity = (stat.st_dev, stat.st_ino)
    except FileNotFoundError:
        identity = None
    for source in sources:
        source = Path(source)
        if resolved == source.resolve():
            raise ValueError(f"Output would overwrite source PDF: {target} aliases {source}")
        try:
            stat = os.stat(source)
        except FileNotFoundError:
            continue
        if identity == (stat.st_dev, stat.st_ino):
            raise ValueError(f"Output would overwrite source PDF (same inode/hard link): {target} aliases {source}")


def protect_artifact(target: Path, sources: Iterable[str | Path],
                     planned: Iterable[Path] = (), staging: Iterable[Path] = ()) -> None:
    """Reports/summaries must not occupy a planned PDF or staging pathname."""
    protect_sources(target, sources)
    resolved = target.resolve()
    if any(resolved == path.resolve() for path in planned):
        raise ValueError(f"Report/summary collides with planned PDF output: {target}")
    if any(resolved.is_relative_to(path.resolve()) for path in staging):
        raise ValueError(f"Report/summary is inside the staging directory: {target}")
