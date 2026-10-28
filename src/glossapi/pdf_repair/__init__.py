"""Conservative, CPU-only repair of PDF font Unicode maps."""

from .tounicode import (
    FontReport,
    PdfFontReport,
    RepairResult,
    extract_text_stats,
)
from .selector import analyze_pdf, get_backend, repair_pdf

__all__ = [
    "FontReport", "PdfFontReport", "RepairResult", "analyze_pdf",
    "extract_text_stats", "repair_pdf", "get_backend",
]
