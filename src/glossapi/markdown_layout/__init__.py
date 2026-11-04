"""Layout-aware Markdown from Rust layout v2, Docling-free (K5-A2; K5 quality policy P2, P3, P7).

The safe extractor's Markdown is the PDFium plain text in plain order, with layout v2 labels (title/section headers
with levels, list items, captions, footnotes, page header/footer, table, picture, formula) rendered by the rules in
``render.py``. Labels never change the text: deleting the markup gives the plain pages back (``render.py``
invariant). Everything needed is in core (pypdfium2 and the ``glossapi_rs_pdfrepair`` extension); the ``layout``
extra (Docling) is not involved.

``GLOSSAPI_MARKDOWN_LAYOUT`` selects the Markdown of the safe extractor:
  - ``layout_v2`` (default): labelled Markdown, extract stage contract 2;
  - ``plain``: the Markdown of extract contract 1 (the plain pages, byte for byte as before K5-A2), read through the
    same one-load-per-page reader.
The value is part of the extract stage's identity (``ExtractStage.contract_version``), so a corpus never reuses the
other mode's extraction.

Modules: ``reader`` (page text + char tables, native or Python reader, PDFium coordination), ``labels`` (context
selection, document pass, page labels), ``render`` (rules, metadata).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from ..pdfium_guard import pdfium_guard
from .labels import DocumentLabeller, LabelledDocument, context_pages
from .reader import NO_SHA, native_module, open_page_source
from .render import RULES, SCHEMA, DocumentMarkdown, render_page

MODES = ("layout_v2", "plain")
ENVIRONMENT = "GLOSSAPI_MARKDOWN_LAYOUT"


def mode() -> str:
    """The safe extractor's Markdown mode (``GLOSSAPI_MARKDOWN_LAYOUT``; default ``layout_v2``)."""
    value = (os.environ.get(ENVIRONMENT) or "layout_v2").strip().lower()
    if value not in MODES:
        raise ValueError(f"{ENVIRONMENT} must be one of {', '.join(MODES)}, not {value!r}")
    return value


def page_text(content: str) -> str:
    """The safe extractor's page text: CRLF/CR -> LF, then ``strip()`` (unchanged since before K5-A2)."""
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def label_pdf(path: Path, *, targets: Optional[Iterable[int]] = None, sha: bytes = NO_SHA):
    """Labels of ``targets`` (default: every page) of one PDF through the production reader, context selection and
    document pass. Returns ``(source kind, page count, LabelledDocument)``; for the exactness receipts and tests."""
    native, why = native_module()
    if native is None:
        raise RuntimeError(why)
    with pdfium_guard():
        source = open_page_source(Path(path), chars=True, sha=sha)
        try:
            count = source.page_count
            wanted = set(range(1, count + 1)) if targets is None else {int(p) for p in targets}
            labeller = DocumentLabeller(native, targets=None if targets is None else wanted)
            for page in context_pages(count, wanted):
                try:
                    text = page_text(source.text(page - 1))
                except Exception:
                    text = None
                table = source.chars(page - 1)
                labeller.add(page, text if page in wanted else None, table)
            return source.kind, count, labeller.finish(lambda page: source.reread_chars(page - 1))
        finally:
            source.close()


def render_document(document: LabelledDocument, *, reader: Optional[str]) -> tuple[str, dict]:
    """Markdown of a labelled document (rules D1-D2, P1-P6) and its ``markdown_layout`` metadata."""
    out = DocumentMarkdown()
    for page in document.pages:
        if page.status == "labelled":
            md, spans = render_page(page.text, [(b[0], b[1], b[2], b[3]) for b in page.blocks])
            out.add_page(page.page, md, "labelled", spans)
            out.add_warnings(page.warnings)
        elif page.text:
            out.add_page(page.page, page.text, page.status)
    labelled = sum(1 for page in document.pages if page.status == "labelled")
    texted = sum(1 for page in document.pages if page.text)
    metadata = {"schema": SCHEMA, "rules": RULES, "reader": reader,
                "status": "labelled" if labelled == texted else "partial" if labelled else "unlabelled",
                "pages": out.pages, "blocks": out.blocks, "label_counts": dict(sorted(out.label_counts.items())),
                "warnings": dict(sorted(out.warnings.items())), "rejected": document.rejected,
                "reread_pages": document.reread_pages, "context": document.context}
    return out.markdown, metadata


def unavailable_metadata(reason: str) -> dict:
    """``markdown_layout`` metadata of a document whose Markdown could not be labelled at all."""
    return {"schema": SCHEMA, "rules": RULES, "reader": None, "status": "unavailable", "reason": reason}


__all__ = ["DocumentLabeller", "ENVIRONMENT", "MODES", "context_pages", "label_pdf", "mode", "native_module",
           "open_page_source", "page_text", "render_document", "unavailable_metadata"]
