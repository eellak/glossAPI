"""Conservative, CPU-only repair of PDF font Unicode maps."""

from .reports import (
    FontReport,
    PdfFontReport,
    RepairResult,
)
from .text_stats import extract_text_stats
from .selector import analyze_pdf, get_backend, repair_pdf

__all__ = [
    "FontReport", "PdfFontReport", "RepairResult", "analyze_pdf",
    "extract_text_stats", "repair_pdf", "get_backend",
]
