"""Skip-list helpers for fatal corpus errors."""

from __future__ import annotations

from ..ledger.journal import observe, observe_store

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

from .._naming import canonical_stem


def _disk_full(error: BaseException) -> bool:
    from ..stages.execution import disk_full
    return disk_full(error)


@dataclass
class _SkiplistManager:
    """Single-writer helper around the on-disk fatal skip-list."""

    path: Path
    logger: logging.Logger
    _cache: Optional[Set[str]] = None

    @staticmethod
    def _normalize(entry: Optional[str]) -> Optional[str]:
        if not entry:
            return None
        stem = canonical_stem(entry.strip())
        return stem or None

    def load(self) -> Set[str]:
        if self._cache is not None:
            return set(self._cache)
        stems: Set[str] = set()
        try:
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    norm = self._normalize(line)
                    if norm:
                        stems.add(norm)
        except Exception as exc:
            observe("unknown_not_recorded", path=self.path, error=exc, operation="skiplist_read", warning=True)
            self.logger.warning("Failed to read skip-list %s: %s", self.path, exc)
        observe("legacy_observed", path=self.path, store="fatal_skiplist", stems=sorted(stems), provenance="unknown_not_recorded")
        self._cache = stems
        return set(stems)

    def add(self, new_entries: Iterable[str]) -> Set[str]:
        current = self.load()
        to_add = {stem for stem in (self._normalize(val) for val in new_entries) if stem}
        if not to_add or to_add.issubset(current):
            return current
        merged = current | to_add
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text("\n".join(sorted(merged)) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
            observe_store(self.path, kind="fatal_skiplist")
            self._cache = merged
            self.logger.warning(
                "Skip-list updated (%d new stem%s): %s",
                len(to_add),
                "s" if len(to_add) != 1 else "",
                ", ".join(sorted(to_add)),
            )
        except Exception as exc:
            if _disk_full(exc):
                raise  # a lost skip-list write on a full disk is not a logged warning (fix 6, F3 audit)
            observe("unknown_not_recorded", path=self.path, error=exc, operation="skiplist_write", warning=True)
            self.logger.error("Failed to update skip-list %s: %s", self.path, exc)
        return self.load()

    def reload(self) -> Set[str]:
        self._cache = None
        return self.load()


def _resolve_skiplist_path(output_dir: Path, logger: logging.Logger) -> Path:
    """Return the directory for the fatal skip-list, respecting env overrides."""

    env_override = os.environ.get("GLOSSAPI_SKIPLIST_PATH")
    if env_override:
        return Path(env_override)

    candidate = output_dir / "skiplists" / "fatal_skip.txt"
    legacy = output_dir.parent / "aws_bundle" / "skiplists" / "fatal_skip.txt"

    for option in (candidate, legacy):
        try:
            option.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            if _disk_full(exc):
                raise  # fix 6, F3 audit
            logger.debug("Skip-list path %s could not be prepared: %s", option, exc)
        if option.exists():
            return option

    return candidate
