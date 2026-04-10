"""Table-specific OCR cleaning helpers.

This module isolates HTML-table handling from the broader OCR repetition logic.

That separation is intentional:
- some table decisions are repetition-based, like repeated rows
- others are structural cleanups, like sentence-shell tables or near-empty shells

Keeping table logic together makes the policy easier to understand and keeps the
main OCR page pipeline focused on ordering and span ownership.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..scripts.table_markdown_audit import (
    _expand_rows as _audit_expand_table_rows,
    _parse_table_rows as _audit_parse_table_rows,
    audit_table as _audit_table_html,
)

HTML_TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table\s*>")
HTML_TABLE_LINE_RE = re.compile(r"(?i)</?(?:table|thead|tbody|tfoot|tr|td|th)\b")
HTML_TABLE_ROW_RE = re.compile(r"(?is)<tr\b.*?>.*?</tr\s*>")
HTML_TABLE_CELL_RE = re.compile(r"(?is)<t[dh]\b.*?>(.*?)</t[dh]\s*>")
HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")

TABLE_EMPTY_MIN_ROWS = 6
TABLE_EMPTY_MIN_CELLS = 18
TABLE_EMPTY_MAX_NONEMPTY_RATIO = 0.15
TABLE_REPEAT_MIN_ROWS = 4
TABLE_REPEAT_MIN_NONEMPTY_CELLS = 2
TABLE_REPEAT_MIN_ROW_TEXT_CHARS = 6
TABLE_REPEAT_MIN_DUPLICATE_ROWS = 2
TABLE_SENTENCE_SHELL_MIN_WORDS = 6
TABLE_SENTENCE_SHELL_MIN_CHARS = 40


def _normalize_table_cell_text(cell_html: str) -> str:
    text = HTML_TAG_RE.sub(" ", cell_html)
    text = html.unescape(text)
    return " ".join(text.split())


def _table_cell_has_content(cell_text: str) -> bool:
    return any(ch.isalnum() for ch in cell_text)


def _extract_html_table_rows(table_text: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for row_match in HTML_TABLE_ROW_RE.finditer(table_text):
        cells = [
            _normalize_table_cell_text(cell_match.group(1))
            for cell_match in HTML_TABLE_CELL_RE.finditer(row_match.group(0))
        ]
        if cells:
            rows.append(cells)
    return rows


@lru_cache(maxsize=2048)
def _extract_html_table_rows_cached(table_text: str) -> Tuple[Tuple[str, ...], ...]:
    """Cache repeated table shells by exact HTML string.

    The OCR corpus contains many duplicated HTML fragments, so exact-string
    memoization pays off without changing behavior.
    """
    return tuple(tuple(row) for row in _extract_html_table_rows(table_text))


def _flatten_html_table_nonempty_cells(table_text: str) -> List[str]:
    parsed_rows, _ = _audit_parse_table_rows(table_text)
    grid, _ = _audit_expand_table_rows(parsed_rows)
    if not grid:
        return []
    nonempty: List[str] = []
    for row in grid:
        for cell in row:
            normalized = " ".join(cell.split())
            if any(ch.isalnum() for ch in normalized):
                nonempty.append(normalized)
    return nonempty


@lru_cache(maxsize=2048)
def _flatten_html_table_nonempty_cells_cached(table_text: str) -> Tuple[str, ...]:
    return tuple(_flatten_html_table_nonempty_cells(table_text))


def _extract_sentence_shell_table_text(table_text: str) -> Optional[str]:
    """Return prose text when a table is only a layout shell around one cell.

    This is intentionally not a repetition rule. OCR and VLM extraction often
    emit a normal sentence inside a tiny one-cell table shell; when that
    happens, the table structure is noise and the prose cell is the content.
    """
    nonempty_cells = _flatten_html_table_nonempty_cells_cached(table_text)
    if len(nonempty_cells) != 1:
        return None
    candidate = nonempty_cells[0].strip()
    if len(candidate) < TABLE_SENTENCE_SHELL_MIN_CHARS:
        return None
    if len(re.findall(r"[^\W\d_]+", candidate, re.UNICODE)) < TABLE_SENTENCE_SHELL_MIN_WORDS:
        return None
    return candidate


@lru_cache(maxsize=2048)
def _render_table_html_for_output_cached(table_text: str, match_kind: Optional[str]) -> str:
    sentence_shell = _extract_sentence_shell_table_text(table_text)
    if sentence_shell and match_kind == "sentence_shell_table":
        return sentence_shell

    audit = _audit_table_html(Path("/tmp/table_fragment.md"), 0, 0, table_text)
    if audit.markdown:
        return audit.markdown
    return table_text


def render_table_html_for_output(table_text: str, *, match_kind: Optional[str] = None) -> str:
    """Render one HTML table for human review/debug output."""
    return _render_table_html_for_output_cached(table_text, match_kind)


def replace_html_tables_with_markdown(text: str) -> str:
    """Normalize kept HTML tables into GitHub-style Markdown in page text."""
    if "<table" not in text.lower():
        return text
    return HTML_TABLE_BLOCK_RE.sub(
        lambda match: render_table_html_for_output(match.group(0)),
        text,
    )


def render_table_html_for_clean(table_text: str, *, match_kind: Optional[str] = None) -> str:
    """Render a table in clean mode.

    Clean mode drops tables whose structure is the problem:
    - sentence-shell tables
    - empty shell tables
    - repeated-row tables
    """
    if match_kind in {"sentence_shell_table", "empty_table_collapse", "repeated_rows"}:
        return ""
    return render_table_html_for_output(table_text, match_kind=match_kind)


def find_table_repeat_spans(page_text: str, *, match_category: str) -> List[Dict[str, Any]]:
    """Classify OCR table problems on a page.

    Table handling is intentionally broader than repetition:
    - sentence-shell tables are removed because they are layout shells around prose
    - empty table collapse removes sparse structural noise
    - repeated rows is the actual repetition-oriented table rule
    """
    if "<table" not in page_text.lower():
        return []

    spans: List[Dict[str, Any]] = []
    for table_match in HTML_TABLE_BLOCK_RE.finditer(page_text):
        raw_table = page_text[table_match.start() : table_match.end()]
        sentence_shell = _extract_sentence_shell_table_text(raw_table)
        if sentence_shell is not None:
            spans.append(
                {
                    "start": table_match.start(),
                    "end": table_match.end(),
                    "match_types": ["table_repeat"],
                    "category": match_category,
                    "kind": "sentence_shell_table",
                    "word_count": len(re.findall(r"[^\W\d_]+", sentence_shell, re.UNICODE)),
                    "char_count": len(sentence_shell),
                }
            )
            continue

        rows = _extract_html_table_rows_cached(raw_table)
        if not rows:
            continue

        row_count = len(rows)
        cell_count = sum(len(row) for row in rows)
        nonempty_cells = sum(
            1 for row in rows for cell in row if _table_cell_has_content(cell)
        )
        nonempty_ratio = (nonempty_cells / cell_count) if cell_count else 0.0

        if (
            row_count >= TABLE_EMPTY_MIN_ROWS
            and cell_count >= TABLE_EMPTY_MIN_CELLS
            and nonempty_ratio <= TABLE_EMPTY_MAX_NONEMPTY_RATIO
        ):
            spans.append(
                {
                    "start": table_match.start(),
                    "end": table_match.end(),
                    "match_types": ["table_repeat"],
                    "category": match_category,
                    "kind": "empty_table_collapse",
                    "row_count": row_count,
                    "cell_count": cell_count,
                    "nonempty_ratio": round(nonempty_ratio, 3),
                }
            )
            continue

        row_keys: List[Tuple[str, ...]] = []
        for row in rows:
            nonempty_cells_in_row = [cell for cell in row if _table_cell_has_content(cell)]
            if len(nonempty_cells_in_row) < TABLE_REPEAT_MIN_NONEMPTY_CELLS:
                continue
            row_text = " ".join(nonempty_cells_in_row)
            if len(row_text) < TABLE_REPEAT_MIN_ROW_TEXT_CHARS:
                continue
            row_keys.append(tuple(cell.casefold() for cell in row))

        if row_count < TABLE_REPEAT_MIN_ROWS or not row_keys:
            continue

        row_counts = Counter(row_keys)
        duplicate_rows = sum(freq - 1 for freq in row_counts.values() if freq >= 2)
        if duplicate_rows >= TABLE_REPEAT_MIN_DUPLICATE_ROWS:
            spans.append(
                {
                    "start": table_match.start(),
                    "end": table_match.end(),
                    "match_types": ["table_repeat"],
                    "category": match_category,
                    "kind": "repeated_rows",
                    "row_count": row_count,
                    "duplicate_rows": duplicate_rows,
                }
            )

    return spans
