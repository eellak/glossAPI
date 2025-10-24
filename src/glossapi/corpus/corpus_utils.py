"""Support utilities shared across the corpus pipeline modules."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional


def _maybe_import_torch(*, force: bool = False) -> Optional[ModuleType]:
    """Return the cached torch module, importing on demand when requested."""

    if not force:
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            return torch_mod
    try:
        return importlib.import_module("torch")  # type: ignore[import]
    except Exception:
        return None
