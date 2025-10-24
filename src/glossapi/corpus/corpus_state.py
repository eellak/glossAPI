"""Persistence helpers for corpus processing state."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Set, Tuple


class _ProcessingStateManager:
    """Maintain the resume checkpoints for multi-GPU extraction."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.logger = logging.getLogger(__name__)

    def load(self) -> Tuple[Set[str], Set[str]]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "rb") as handle:
                    state = pickle.load(handle)
                processed = set(state.get("processed", set()))
                problematic = set(state.get("problematic", set()))
                return processed, problematic
            except Exception as exc:
                self.logger.warning("Failed to load processing state %s: %s", self.state_file, exc)
        return set(), set()

    def save(self, processed: Set[str], problematic: Set[str]) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "wb") as handle:
                pickle.dump({"processed": set(processed), "problematic": set(problematic)}, handle)
        except Exception as exc:
            self.logger.warning("Failed to persist processing state %s: %s", self.state_file, exc)

