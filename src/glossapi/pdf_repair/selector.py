"""Select Rust on successful import; only ImportError selects the reference.

Native execution errors retain Rust provenance and are never silently retried
with Python. Selection is lazy and uncached so installing an optional extension
does not require importing heavy extraction components.
"""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Callable, Optional

from . import reports


@dataclass(frozen=True)
class RepairBackend:
    name: str
    analyze_pdf: Callable
    repair_pdf: Callable
    # Optional analyse-once path: returns (report, analysis); repair_pdf(..., analysis=analysis) then neither
    # re-inspects fonts nor re-samples text. None when the backend has no such path (the Python reference).
    analyze_for_repair: Optional[Callable] = None


def get_backend(name: str | None = None) -> RepairBackend:
    if name not in (None, "rust", "python"):
        raise ValueError("Repair backend must be rust or python")
    if name == "python":
        from . import tounicode
        return RepairBackend("python", tounicode.analyze_pdf, tounicode.repair_pdf)
    try:
        native = import_module("glossapi_rs_pdfrepair")
    except ImportError:
        if name == "rust":
            raise
        from . import tounicode
        return RepairBackend("python", tounicode.analyze_pdf, tounicode.repair_pdf)

    def report(kind, payload):
        data = dict(payload)
        data["fonts"] = [reports.FontReport(**font) for font in data.get("fonts", [])]
        data["backend"] = "rust"
        return kind(**data)

    def analyze(path):
        return report(reports.PdfFontReport, native.analyze_pdf(str(path)))

    def analyze_for_repair(path):
        payload, analysis = native.analyze_for_repair(str(path))
        return report(reports.PdfFontReport, payload), analysis

    def repair(src, dst, *, protected_sources=(), analysis=None):
        args = (str(src), str(dst), [str(p) for p in protected_sources])
        return report(reports.RepairResult,
                      native.repair_pdf(*args) if analysis is None else native.repair_pdf(*args, analysis))

    # An extension built before the analyse-once API keeps the two-pass path.
    once = analyze_for_repair if hasattr(native, "analyze_for_repair") else None
    return RepairBackend("rust", analyze, repair, once)


def analyze_pdf(path: str | Path) -> reports.PdfFontReport:
    """Return a typed report whose backend field identifies the implementation."""
    return get_backend().analyze_pdf(path)


def repair_pdf(src: str | Path, dst: str | Path, *, protected_sources=()) -> reports.RepairResult:
    """Repair with the selected implementation and retain its backend receipt."""
    return get_backend().repair_pdf(src, dst, protected_sources=protected_sources)
