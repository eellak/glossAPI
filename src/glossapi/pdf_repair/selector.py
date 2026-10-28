"""Select Rust on successful import; only ImportError selects the reference.

Native execution errors retain Rust provenance and are never silently retried
with Python. Selection is lazy and uncached so installing an optional extension
does not require importing heavy extraction components.
"""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable

from . import reports


@dataclass(frozen=True)
class RepairBackend:
    name: str
    analyze_pdf: Callable
    repair_pdf: Callable


def get_backend() -> RepairBackend:
    try:
        native = import_module("glossapi_rs_pdfrepair")
    except ImportError:
        from . import tounicode
        return RepairBackend("python", tounicode.analyze_pdf, tounicode.repair_pdf)

    def report(kind, payload):
        data = dict(payload)
        data["fonts"] = [reports.FontReport(**font) for font in data.get("fonts", [])]
        data["backend"] = "rust"
        return kind(**data)

    def analyze(path):
        return report(reports.PdfFontReport, native.analyze_pdf(str(path)))

    def repair(src, dst, *, protected_sources=()):
        return report(reports.RepairResult, native.repair_pdf(
            str(src), str(dst), [str(p) for p in protected_sources]))

    return RepairBackend("rust", analyze, repair)


def analyze_pdf(path: str | Path) -> reports.PdfFontReport:
    """Return a typed report whose backend field identifies the implementation."""
    return get_backend().analyze_pdf(path)


def repair_pdf(src: str | Path, dst: str | Path, *, protected_sources=()) -> reports.RepairResult:
    """Repair with the selected implementation and retain its backend receipt."""
    return get_backend().repair_pdf(src, dst, protected_sources=protected_sources)
