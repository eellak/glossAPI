"""PDF repair reports shared by native and reference implementations."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FontReport:
    """Keep font diagnostics independent of PDF reader object lifetimes."""

    font_name: str
    subtype: str
    embedded: bool = False
    font_program_kind: str | None = None
    has_tounicode: bool = False
    codes: int = 0
    disagreeing_codes: int = 0
    greek_codes: int = 0
    affected_greek_codes: int = 0
    unmapped_codes: int = 0
    category: str = "unsupported"
    repair_possible: bool = False
    reason: str | None = None
    error: str | None = None
    object_id: int | None = None


@dataclass
class PdfFontReport:
    """Keep malformed documents reportable rather than raising from analysis."""

    path: str
    category: str = "unsupported"
    pages: int = 0
    sampled_pages: list[int] = field(default_factory=list)
    fonts: list[FontReport] = field(default_factory=list)
    error: str | None = None
    backend: str = "python"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot without PDF objects."""
        return asdict(self)


@dataclass
class RepairResult:
    """Describe exactly which font dictionaries received a new ToUnicode."""

    src: str
    dst: str
    category: str = "unsupported"
    repaired_fonts: int = 0
    fonts: list[FontReport] = field(default_factory=list)
    written: bool = False
    error: str | None = None
    backend: str = "python"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot without PDF objects."""
        return asdict(self)


