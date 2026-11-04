"""Layout v2 document pass and page labels over the safe extractor's pages (K5-A2; K5A1-G1, K5A1-G3, K5A1R2-G1).

Context sourcing (the frozen K5c/K5g rule, ``k5_prep/layout/freeze_chars.py``):
  - the target pages are the pages being labelled (production: every page of the document);
  - the context pages are the targets, the pages ``p-3 .. p+3`` around every target, and 8 evenly spaced pages
    (``1 + round(i * (n - 1) / 7)``, ``i = 0..7``, when the document has more than one page), in ascending
    page-number order;
  - a target page enters the document pass only if it loaded and has at least one char PDFium did not generate (the
    fixture rule); any other context page enters if it loaded; a page outside the K5A1-G3 input contract never
    enters.
  The document pass (``LayoutV2ContextBuilder``) is fed in that order and equals ``layout_v2_context`` over the same
  pages, so the held-out evidence (whose targets were the page-set pages of each document) is reproduced exactly by
  ``label_pdf(path, targets=...)``, and a whole document (targets = all pages) uses every page with a text layer.

Labels: ``layout_v2_label_page(plain, table, ctx, page_no)`` per target page with text, on the page text the
extractor produced (no re-ordering; the labeller only cuts and labels). A page is left unlabelled, with its reason,
when it has no char table, no text layer, a table outside the input contract (``layout_input_rejected``) or the
labeller raises for it (``layout_page_failed``).

Memory: the char tables of the pages still to be labelled are kept up to ``TABLE_BUDGET`` bytes; beyond that a page's
table is read again for labelling (a second load, counted in ``reread_pages``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .reader import real_chars

CONTEXT_RADIUS = 3
CONTEXT_EVEN = 8
TABLE_BUDGET = 128 * 1024 * 1024


def context_pages(page_count: int, targets: Iterable[int]) -> list[int]:
    """The frozen context selection: targets, their ``p-3 .. p+3`` neighbours and 8 evenly spaced pages, sorted."""
    pages: set[int] = set()
    for page in targets:
        pages.update(range(max(1, page - CONTEXT_RADIUS), min(page_count, page + CONTEXT_RADIUS) + 1))
        pages.add(page)
    if page_count > 1:
        pages.update(1 + round(i * (page_count - 1) / (CONTEXT_EVEN - 1)) for i in range(CONTEXT_EVEN))
    return sorted(page for page in pages if 1 <= page <= page_count)


@dataclass
class PageLabel:
    page: int
    text: str
    status: str                     # "labelled" or the reason the page is unlabelled
    blocks: list = field(default_factory=list)   # [(start, end, label, level, rotated, why, cut, first, last, bbox)]
    rotation: int = 0
    warnings: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    markdown: str = ""              # the labeller's own page_markdown (K5g form), for the exactness receipts


@dataclass
class LabelledDocument:
    pages: list
    context: dict
    rejected: list
    reread_pages: int


class DocumentLabeller:
    """Feed pages in ascending page order with ``add``; ``finish`` builds the document context and labels."""

    def __init__(self, native, *, targets: Optional[set[int]] = None, budget: int = TABLE_BUDGET):
        self.native = native
        self.targets = targets
        self.budget = budget
        self.builder = native.LayoutV2ContextBuilder()
        self.pages: list[tuple[int, str, Optional[str]]] = []   # (page, text, unlabelled reason or None)
        self.tables: dict[int, bytes] = {}
        self.dropped: set[int] = set()
        self.kept = 0
        self.last = 0
        self.rejected: list = []
        self.context_pages: list[int] = []
        self.failed: Optional[str] = None

    def is_target(self, page: int) -> bool:
        return self.targets is None or page in self.targets

    def add(self, page: int, text: Optional[str], table: Optional[bytes], *, in_context: bool = True) -> None:
        """One page in ascending order. ``text``: the page text (None: not a target or the page failed); ``table``:
        its ``.k5chars`` bytes (None: no table). ``in_context``: the page is a context page (the frozen selection)."""
        if page <= self.last:
            raise ValueError("layout pages must be added in ascending page order")
        self.last = page
        target = self.is_target(page) and text is not None
        reason = None
        problem = None
        if table is None:
            reason = "no_char_table"
        else:
            problem = self.native.layout_v2_input_problem(table)
            if problem is not None:
                reason = "layout_input_rejected"
                self.rejected.append([page, problem])
            elif target and real_chars(table) == 0:
                reason = "no_text_layer"
        if in_context and table is not None and problem is None and (reason is None or not target):
            if self.failed is None:
                try:
                    self.builder.add(page, table)
                    self.context_pages.append(page)
                except ValueError as error:      # outside the parity contract despite the checks: no labels at all
                    self.failed = str(error)
        if not target or text is None:
            return
        if reason is None and text and table is not None:
            if self.kept + len(table) <= self.budget:
                self.tables[page] = table
                self.kept += len(table)
            else:
                self.dropped.add(page)
        self.pages.append((page, text, reason))

    def finish(self, reread: Optional[Callable[[int], Optional[bytes]]] = None) -> LabelledDocument:
        context: dict = {"pages": len(self.context_pages)}
        ctx = None
        if self.failed is None:
            try:
                ctx = self.builder.finish()
            except ValueError as error:
                self.failed = str(error)
        if ctx is not None:
            context.update(body_size=ctx.body, heading_sizes=list(ctx.heading_sizes), text_pages=ctx.pages,
                           furniture=[[f["band"], f["signature"], f["y"], f["pages"]] for f in ctx.furniture])
        else:
            context["error"] = self.failed
        out, rereads = [], 0
        for page, text, reason in self.pages:
            if not text:
                out.append(PageLabel(page, text, "blank"))
                continue
            if reason is not None:
                out.append(PageLabel(page, text, reason))
                continue
            if ctx is None:
                out.append(PageLabel(page, text, "layout_context_failed"))
                continue
            table = self.tables.pop(page, None)
            if table is None and page in self.dropped and reread is not None:
                table = reread(page)
                rereads += 1
                if table is not None and self.native.layout_v2_input_problem(table) is not None:
                    table = None
            if table is None:
                out.append(PageLabel(page, text, "no_char_table"))
                continue
            try:
                rotation, warnings, lines, blocks, markdown = self.native.layout_v2_label_page(text, table, ctx, page)
            except ValueError:
                out.append(PageLabel(page, text, "layout_page_failed"))
                continue
            out.append(PageLabel(page, text, "labelled", list(blocks), rotation, list(warnings), list(lines),
                                 markdown))
        return LabelledDocument(out, context, self.rejected, rereads)
