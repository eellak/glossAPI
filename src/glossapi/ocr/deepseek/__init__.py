"""DeepSeek OCR backend."""

from .runner import run_for_files
from . import preflight

__all__ = ["run_for_files", "preflight"]
