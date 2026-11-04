"""Layout v2 labels -> Markdown on the plain order (rendering rules ``glossapi-layout-md/1``, K5-A2).

Nothing is re-ordered, added to or removed from the text; the rules only add markup around blocks.

Document
  D1. Pages appear in page order. Each page's text is the safe extractor's page text (``get_text_range()``, CRLF/CR
      -> LF, ``strip()``), exactly as before K5-A2. Blank pages are skipped and the remaining pages are joined by one
      blank line (``"\\n\\n"``): the page separator is unchanged. Page boundaries are recorded as offsets in the
      metadata, not as text markers (a ``<!-- page:N -->`` marker would switch partial OCR from an appended supplement
      to in-place page replacement, an OCR change outside this unit).
  D2. A page without labels (no char table, input rejected by the K5A1-G3 contract, no text layer, labeller
      unavailable or failed) is its page text unchanged.

Labelled page
  P1. The page text is partitioned by the labeller into blocks: block texts plus their original separators equal the
      page text byte for byte, and every separator is whitespace (a newline, or the whitespace where PDFium joined two
      visual rows). The blocks of a page are joined by one blank line (``"\\n\\n"``) instead of their separator.
  P2. ``title``: ``"# " + text``; ``section_header``: ``"#" * min(6, level + 1) + " " + text`` (level 1 = the
      largest heading size of the document -> ``##``). Newlines inside a heading are rendered as spaces, so a 2- or
      3-row heading stays one ATX heading.
  P3. ``list_item``: the text, prefixed with ``"- "`` unless it already starts with a Markdown list marker
      (``-``, ``*``, ``+`` or ``1.``/``1)`` followed by whitespace). The printed bullet or number is kept.
  P4. Marked blocks: one comment line, then the text. ``caption`` -> ``<!-- caption -->``, ``footnote`` ->
      ``<!-- footnote -->``, ``table`` -> ``<!-- table -->`` (cell text in PDFium order; no Markdown table),
      ``picture`` -> ``<!-- picture -->`` (text inside a figure), ``formula`` -> ``<!-- formula -->``,
      ``page_header`` -> ``<!-- page-header -->``, ``page_footer`` -> ``<!-- page-footer -->``. Furniture (page
      header/footer) is marked and kept.
  P5. ``paragraph``: the text unchanged.
  P6. Text is never escaped: a line of the PDF that starts with ``#`` or ``1.`` reads as Markdown, as it did before.

Invariant (tested): deleting the markup that P2-P4 add, turning heading spaces that came from P2 back into
newlines and replacing each block separator ``"\\n\\n"`` by the block's original separator gives the page text byte
for byte; the metadata block offsets point at each block's text inside the Markdown. Under the bake-off scorer's
normalisation (``score.norm``: comments, ``#`` and list markers removed, whitespace collapsed) Markdown equals the
plain text (the K5g invariant).

Metadata (extract sidecar key ``markdown_layout``; offsets are code points of the Markdown file):
  ``pages``: ``[page, start, end, status]`` for every page present in the Markdown (status ``labelled`` or the reason
  it is not); ``blocks``: ``[page, start, end, label, level]`` with start/end around the block's text (after any
  prefix); ``label_counts``; ``warnings`` (kind -> pages); ``rejected``: ``[page, reason]``; ``context``: pages,
  body size, heading sizes, furniture signatures.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

RULES = "glossapi-layout-md/1"
SCHEMA = "glossapi.markdown_layout.v1"
MARKS = {"caption": "caption", "footnote": "footnote", "table": "table", "picture": "picture", "formula": "formula",
         "page_header": "page-header", "page_footer": "page-footer"}
_LIST_MARK = re.compile(r"(?:[-*+]|\d{1,9}[.)])(?:\s|$)")
LABELS = ("title", "section_header", "paragraph", "list_item", "caption", "footnote", "page_header", "page_footer",
          "table", "picture", "formula")


def block_prefix(label: str, level: Optional[int], text: str) -> tuple[str, str]:
    """(prefix, rendered text) of one block (rules P2-P5)."""
    if label == "title":
        return "# ", text.replace("\n", " ")
    if label == "section_header":
        return "#" * min(6, (level or 1) + 1) + " ", text.replace("\n", " ")
    if label == "list_item":
        return ("" if _LIST_MARK.match(text) else "- "), text
    mark = MARKS.get(label)
    if mark is not None:
        return f"<!-- {mark} -->\n", text
    return "", text


def render_page(text: str, blocks: Sequence[tuple[int, int, str, Optional[int]]]) -> tuple[str, list]:
    """Markdown of one labelled page and its block spans ``[(start, end, label, level)]`` relative to it.
    ``blocks``: ``(start, end, label, level)`` code-point offsets into ``text`` in plain order."""
    parts: list[str] = []
    spans: list = []
    offset = 0
    for start, end, label, level in blocks:
        if parts:
            parts.append("\n\n")
            offset += 2
        prefix, body = block_prefix(label, level, text[start:end])
        parts.append(prefix)
        offset += len(prefix)
        spans.append((offset, offset + len(body), label, level))
        parts.append(body)
        offset += len(body)
    return "".join(parts), spans


class DocumentMarkdown:
    """Accumulates rendered pages (rule D1) and the metadata."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.pages: list = []
        self.blocks: list = []
        self.label_counts: dict[str, int] = {}
        self.warnings: dict[str, int] = {}

    def add_page(self, page_no: int, page_markdown: str, status: str, spans: Iterable = ()) -> None:
        if not page_markdown:
            return
        if self.parts:
            self.parts.append("\n\n")
            self.length += 2
        start = self.length
        self.parts.append(page_markdown)
        self.length += len(page_markdown)
        self.pages.append([page_no, start, self.length, status])
        for s, e, label, level in spans:
            self.blocks.append([page_no, start + s, start + e, label, level])
            self.label_counts[label] = self.label_counts.get(label, 0) + 1

    def add_warnings(self, warnings: Iterable[dict]) -> None:
        for warning in warnings:
            kind = str(warning.get("kind"))
            self.warnings[kind] = self.warnings.get(kind, 0) + 1

    @property
    def markdown(self) -> str:
        return "".join(self.parts)

