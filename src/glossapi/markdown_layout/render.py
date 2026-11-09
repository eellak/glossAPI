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

Strip record (K5-A2b, K5A2A-G2): ``metadata["strip"]`` carries the edits that undo the markup -
``[start, end, original]`` code-point spans of the Markdown whose replacement by ``original`` (applied in order) gives
the plain text: each P2-P4 prefix -> ``""``, each block separator ``"\\n\\n"`` -> the block's original separator, each
P2 heading space that was a newline -> ``"\\n"``. With ``markdown_chars``/``markdown_sha256`` (the Markdown it
describes) and ``plain_sha256`` (the plain pages, ``"\\n\\n"``-joined, as the plain mode writes them), ``markup_free``
gives back the plain text and checks it; the cleaner measures its text metrics on that text, so layout markup never
moves a metric or a routing decision (the markup rules stay here, in one place).

Metadata (extract sidecar key ``markdown_layout``; offsets are code points of the Markdown file):
  ``pages``: ``[page, start, end, status]`` for every page present in the Markdown (status ``labelled`` or the reason
  it is not); ``blocks``: ``[page, start, end, label, level]`` with start/end around the block's text (after any
  prefix); ``label_counts``; ``warnings`` (kind -> pages); ``rejected``: ``[page, reason]``; ``context``: pages,
  body size, heading sizes, furniture signatures.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

RULES = "glossapi-layout-md/1"
SCHEMA = "glossapi.markdown_layout.v1"
STRIP_SCHEMA = "glossapi.markdown_layout.strip.v1"
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
    markdown, spans, _ = render_page_with_edits(text, blocks)
    return markdown, spans


def render_page_with_edits(text: str, blocks: Sequence[tuple[int, int, str, Optional[int]]]) -> tuple[str, list, list]:
    """``render_page`` plus the edits that undo its markup: ``[start, end, original]`` spans of the page Markdown whose
    replacement gives ``text`` back (the strip invariant)."""
    parts: list[str] = []
    spans: list = []
    edits: list = []
    offset = 0
    previous_end = 0
    for start, end, label, level in blocks:
        separator = text[previous_end:start]
        if parts:
            parts.append("\n\n")
            if separator != "\n\n":
                edits.append([offset, offset + 2, separator])
            offset += 2
        elif separator:
            edits.append([offset, offset, separator])
        original = text[start:end]
        prefix, body = block_prefix(label, level, original)
        if prefix:
            edits.append([offset, offset + len(prefix), ""])
        parts.append(prefix)
        offset += len(prefix)
        if body != original:   # P2: a heading's newlines became spaces
            edits.extend([offset + i, offset + i + 1, "\n"] for i, ch in enumerate(original) if ch == "\n")
        spans.append((offset, offset + len(body), label, level))
        parts.append(body)
        offset += len(body)
        previous_end = end
    if text[previous_end:]:
        edits.append([offset, offset, text[previous_end:]])
    return "".join(parts), spans, edits


def apply_edits(markdown: str, edits: Iterable[Sequence[Any]]) -> str:
    """Apply ``[start, end, replacement]`` edits (ordered, non-overlapping code-point spans) to ``markdown``."""
    out: list[str] = []
    position = 0
    for start, end, replacement in edits:
        start, end = int(start), int(end)
        if start < position or end < start or end > len(markdown):
            raise ValueError("strip edits are not ordered, non-overlapping spans of the Markdown")
        out.append(markdown[position:start])
        out.append(str(replacement))
        position = end
    out.append(markdown[position:])
    return "".join(out)


def edits_between(source: str, target: str) -> list:
    """``[start, end, replacement]`` edits (line granularity, code points of ``source``) that turn ``source`` into
    ``target``: the strip record of a text derived from a labelled Markdown (the OCR cleaner's output, K5-A2b) whose
    markup-free counterpart was derived from the plain text. Correct for any pair (``strip_record`` checks it)."""
    import difflib
    a, b = source.splitlines(keepends=True), target.splitlines(keepends=True)
    offsets = [0]
    for line in a:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            edits.append([offsets[i1], offsets[i2], "".join(b[j1:j2])])
    return edits


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def strip_record(markdown: str, edits: list, plain: str) -> Optional[dict]:
    """The ``strip`` metadata of a Markdown file whose plain text is ``plain``; None when the edits do not give
    ``plain`` back (never expected: the invariant is tested; the cleaner then measures the Markdown as before)."""
    if apply_edits(markdown, edits) != plain:
        return None
    return {"schema": STRIP_SCHEMA, "markdown_chars": len(markdown), "markdown_sha256": _sha256(markdown),
            "plain_sha256": _sha256(plain), "edits": edits}


def markup_free(text: str, layout: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The text under the layout markup (K5-A2b; the one shared strip function), or None when ``layout`` (an extract
    sidecar's ``markdown_layout`` value) does not describe ``text``.

    ``text`` is the Markdown the record describes, or that Markdown followed by more text (an OCR supplement appended
    after the native pages): the described prefix is replaced by its plain text and the rest is kept. The prefix is
    checked by length and sha256 before, and the plain text by sha256 after, so a changed file or a record of another
    file is never applied."""
    strip = layout.get("strip") if isinstance(layout, Mapping) else None
    if not isinstance(strip, Mapping) or strip.get("schema") != STRIP_SCHEMA:
        return None
    try:
        count = int(strip["markdown_chars"])
        if count < 0 or len(text) < count or _sha256(text[:count]) != strip["markdown_sha256"]:
            return None
        plain = apply_edits(text[:count], strip["edits"])
    except (KeyError, TypeError, ValueError):
        return None
    if _sha256(plain) != strip.get("plain_sha256"):
        return None
    return plain + text[count:]


class DocumentMarkdown:
    """Accumulates rendered pages (rule D1) and the metadata."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.pages: list = []
        self.blocks: list = []
        self.label_counts: dict[str, int] = {}
        self.warnings: dict[str, int] = {}
        self.edits: list = []

    def add_page(self, page_no: int, page_markdown: str, status: str, spans: Iterable = (),
                 edits: Iterable = ()) -> None:
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
        self.edits.extend([start + s, start + e, original] for s, e, original in edits)

    def add_warnings(self, warnings: Iterable[dict]) -> None:
        for warning in warnings:
            kind = str(warning.get("kind"))
            self.warnings[kind] = self.warnings.get(kind, 0) + 1

    @property
    def markdown(self) -> str:
        return "".join(self.parts)

