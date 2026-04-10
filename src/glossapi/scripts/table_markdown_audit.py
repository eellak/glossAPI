from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table\s*>")
ROW_RE = re.compile(r"(?is)<tr\b.*?>.*?</tr\s*>")
CELL_RE = re.compile(r"(?is)<(td|th)\b(.*?)>(.*?)</\1\s*>")
ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(".*?"|\'.*?\'|[^\s>]+)', re.S)
TAG_RE = re.compile(r"(?is)<[^>]+>")
DISALLOWED_TAG_RE = re.compile(r"(?is)</?(?!sub\b|sup\b)[a-zA-Z][^>]*>")
BREAK_TAG_RE = re.compile(r"(?is)<br\s*/?>")


@dataclass
class ParsedCell:
    tag: str
    text: str
    rowspan: int
    colspan: int


@dataclass
class TableAudit:
    source_path: str
    source_stem: str
    table_index_in_doc: int
    global_index: int
    html: str
    status: str
    convertible: bool
    broken: bool
    reasons: List[str]
    row_count: int
    col_count: int
    nonempty_ratio: float
    duplicate_rows: int
    header_mode: str
    spans_present: bool
    markdown: Optional[str]


class _CellHTMLNormalizer(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.link_stack: List[Optional[str]] = []

    def _append_break(self) -> None:
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        if tag == "br":
            self._append_break()
            return
        if tag in {"p", "div", "li"}:
            self._append_break()
            if tag == "li":
                self.parts.append("- ")
            return
        if tag in {"sub", "sup"}:
            self.parts.append(f"<{tag}>")
            return
        if tag == "img":
            alt = " ".join(attr_map.get("alt", "").split())
            if alt:
                self.parts.append(alt)
            return
        if tag == "a":
            href = attr_map.get("href", "").strip()
            self.link_stack.append(href or None)
            self.parts.append("[")
            return

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"p", "div", "li"}:
            self._append_break()
            return
        if tag in {"sub", "sup"}:
            self.parts.append(f"</{tag}>")
            return
        if tag == "a":
            href = self.link_stack.pop() if self.link_stack else None
            if href:
                self.parts.append(f"]({href})")
            else:
                self.parts.append("]")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def _parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key, raw_value in ATTR_RE.findall(attr_text):
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        attrs[key.lower()] = html.unescape(value)
    return attrs


def _normalize_cell_html(cell_html: str) -> str:
    parser = _CellHTMLNormalizer()
    parser.feed(cell_html)
    parser.close()
    text = parser.get_text()
    text = BREAK_TAG_RE.sub("\n", text)
    text = DISALLOWED_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _parse_table_rows(table_html: str) -> Tuple[List[List[ParsedCell]], List[str]]:
    reasons: List[str] = []
    if re.search(r"(?is)<table\b", table_html[6:]):
        reasons.append("nested_table")

    rows: List[List[ParsedCell]] = []
    for row_match in ROW_RE.finditer(table_html):
        raw_row = row_match.group(0)
        cells: List[ParsedCell] = []
        for tag, attr_text, inner_html in CELL_RE.findall(raw_row):
            attrs = _parse_attrs(attr_text)
            try:
                rowspan = max(1, int(attrs.get("rowspan", "1")))
            except ValueError:
                rowspan = 1
                reasons.append("invalid_rowspan")
            try:
                colspan = max(1, int(attrs.get("colspan", "1")))
            except ValueError:
                colspan = 1
                reasons.append("invalid_colspan")
            cells.append(
                ParsedCell(
                    tag=tag.lower(),
                    text=_normalize_cell_html(inner_html),
                    rowspan=rowspan,
                    colspan=colspan,
                )
            )
        if cells:
            rows.append(cells)

    if not rows:
        reasons.append("no_rows")
    return rows, reasons


def _expand_rows(parsed_rows: Sequence[Sequence[ParsedCell]]) -> Tuple[Optional[List[List[str]]], List[str]]:
    reasons: List[str] = []
    active_rowspans: Dict[int, int] = {}
    expanded_rows: List[List[str]] = []
    max_cols = 0

    for parsed_row in parsed_rows:
        row: List[str] = []
        col_idx = 0

        def fill_active_until_free() -> None:
            nonlocal col_idx
            while active_rowspans.get(col_idx, 0) > 0:
                row.append("")
                active_rowspans[col_idx] -= 1
                if active_rowspans[col_idx] <= 0:
                    del active_rowspans[col_idx]
                col_idx += 1

        fill_active_until_free()
        for cell in parsed_row:
            fill_active_until_free()
            row.append(cell.text)
            if cell.rowspan > 1:
                active_rowspans[col_idx] = max(active_rowspans.get(col_idx, 0), cell.rowspan - 1)
            start_col = col_idx
            col_idx += 1
            for extra in range(1, cell.colspan):
                row.append("")
                if cell.rowspan > 1:
                    active_rowspans[start_col + extra] = max(
                        active_rowspans.get(start_col + extra, 0), cell.rowspan - 1
                    )
                col_idx += 1
            fill_active_until_free()

        max_cols = max(max_cols, len(row))
        expanded_rows.append(row)

    while active_rowspans:
        row: List[str] = []
        col_idx = 0
        max_active_col = max(active_rowspans)
        while col_idx <= max_active_col:
            if active_rowspans.get(col_idx, 0) > 0:
                row.append("")
                active_rowspans[col_idx] -= 1
                if active_rowspans[col_idx] <= 0:
                    del active_rowspans[col_idx]
            else:
                row.append("")
            col_idx += 1
        max_cols = max(max_cols, len(row))
        expanded_rows.append(row)

    if max_cols == 0 or not expanded_rows:
        reasons.append("empty_grid")
        return None, reasons

    for row in expanded_rows:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
    return expanded_rows, reasons


def _markdown_escape(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_markdown_row(values: Sequence[str], widths: Sequence[int]) -> str:
    padded = [value.ljust(width) for value, width in zip(values, widths)]
    return "| " + " | ".join(padded) + " |"


def _should_infer_header_row(grid: Sequence[Sequence[str]]) -> bool:
    if len(grid) < 2:
        return False
    first_row = grid[0]
    if not first_row:
        return False
    return all(any(ch.isalnum() for ch in cell) for cell in first_row)


def _grid_to_markdown(grid: Sequence[Sequence[str]], header_mode: str) -> str:
    if not grid:
        return ""
    cols = len(grid[0])
    if header_mode in {"explicit_first_row", "inferred_first_row"}:
        header = [_markdown_escape(cell) for cell in grid[0]]
        data_rows = list(grid[1:])
    else:
        header = [""] * cols
        data_rows = list(grid)
    escaped_rows = [[_markdown_escape(cell) for cell in row] for row in data_rows]
    sep = ["---"] * cols
    widths = [
        max(
            len(header[idx]),
            len(sep[idx]),
            *(len(row[idx]) for row in escaped_rows),
        )
        for idx in range(cols)
    ]

    lines = [
        _format_markdown_row(header, widths),
        _format_markdown_row(sep, widths),
    ]
    for row in escaped_rows:
        lines.append(_format_markdown_row(row, widths))
    return "\n".join(lines)


def _assess_content(
    grid: Sequence[Sequence[str]],
    *,
    spans_present: bool,
) -> Tuple[bool, List[str], float, int]:
    total_cells = sum(len(row) for row in grid)
    nonempty_cells = sum(1 for row in grid for cell in row if any(ch.isalnum() for ch in cell))
    nonempty_ratio = (nonempty_cells / total_cells) if total_cells else 0.0

    row_keys = []
    for row in grid:
        normalized = tuple(" ".join(cell.split()).casefold() for cell in row)
        nonempty_in_row = sum(1 for cell in normalized if any(ch.isalnum() for ch in cell))
        if nonempty_in_row >= 2:
            row_keys.append(normalized)
    duplicate_rows = sum(freq - 1 for freq in Counter(row_keys).values() if freq >= 2)

    reasons: List[str] = []
    broken = False
    if total_cells >= 18 and nonempty_ratio <= 0.15:
        broken = True
        reasons.append("near_empty_table")
    if spans_present and total_cells >= 4 and nonempty_ratio <= 0.34:
        broken = True
        reasons.append("sparse_span_shell")
    if len(grid) >= 4 and duplicate_rows >= 2:
        broken = True
        reasons.append("repeated_rows")
    return broken, reasons, round(nonempty_ratio, 4), duplicate_rows


def audit_table(source_path: Path, table_index_in_doc: int, global_index: int, table_html: str) -> TableAudit:
    parsed_rows, parse_reasons = _parse_table_rows(table_html)
    spans_present = any(cell.rowspan > 1 or cell.colspan > 1 for row in parsed_rows for cell in row)
    explicit_header = bool(parsed_rows and any(cell.tag == "th" for cell in parsed_rows[0]))
    grid, expand_reasons = _expand_rows(parsed_rows)
    reasons = list(dict.fromkeys(parse_reasons + expand_reasons))

    if grid is None:
        return TableAudit(
            source_path=str(source_path),
            source_stem=source_path.stem,
            table_index_in_doc=table_index_in_doc,
            global_index=global_index,
            html=table_html,
            status="broken_or_ambiguous",
            convertible=False,
            broken=True,
            reasons=reasons or ["parse_failure"],
            row_count=0,
            col_count=0,
            nonempty_ratio=0.0,
            duplicate_rows=0,
            header_mode="none",
            spans_present=spans_present,
            markdown=None,
        )

    broken, content_reasons, nonempty_ratio, duplicate_rows = _assess_content(
        grid,
        spans_present=spans_present,
    )
    reasons = list(dict.fromkeys(reasons + content_reasons))
    if explicit_header:
        header_mode = "explicit_first_row"
    elif _should_infer_header_row(grid):
        header_mode = "inferred_first_row"
    else:
        header_mode = "blank_first_row"
    markdown = _grid_to_markdown(grid, header_mode=header_mode)

    if any(reason in {"nested_table", "invalid_rowspan", "invalid_colspan"} for reason in reasons):
        status = "broken_or_ambiguous"
        convertible = False
        markdown = None
        broken = True
    else:
        status = "convertible_but_broken" if broken else "convertible_clean"
        convertible = True

    return TableAudit(
        source_path=str(source_path),
        source_stem=source_path.stem,
        table_index_in_doc=table_index_in_doc,
        global_index=global_index,
        html=table_html,
        status=status,
        convertible=convertible,
        broken=broken,
        reasons=reasons,
        row_count=len(grid),
        col_count=len(grid[0]) if grid else 0,
        nonempty_ratio=nonempty_ratio,
        duplicate_rows=duplicate_rows,
        header_mode=header_mode,
        spans_present=spans_present,
        markdown=markdown,
    )


def iter_tables(markdown_dir: Path):
    global_index = 0
    for source_path in sorted(markdown_dir.glob("*.md")):
        text = source_path.read_text(encoding="utf-8", errors="ignore")
        table_index = 0
        for match in TABLE_BLOCK_RE.finditer(text):
            table_index += 1
            global_index += 1
            yield source_path, table_index, global_index, match.group(0)


def write_review_file(output_dir: Path, audit: TableAudit) -> str:
    filename = f"{audit.global_index:05d}__{audit.source_stem}__table_{audit.table_index_in_doc:03d}.txt"
    output_path = output_dir / filename
    lines = [
        f"SOURCE_PATH: {audit.source_path}",
        f"SOURCE_STEM: {audit.source_stem}",
        f"TABLE_INDEX_IN_DOC: {audit.table_index_in_doc}",
        f"GLOBAL_INDEX: {audit.global_index}",
        f"STATUS: {audit.status}",
        f"CONVERTIBLE: {audit.convertible}",
        f"BROKEN: {audit.broken}",
        f"REASONS: {', '.join(audit.reasons) if audit.reasons else 'none'}",
        f"ROWS: {audit.row_count}",
        f"COLS: {audit.col_count}",
        f"NONEMPTY_RATIO: {audit.nonempty_ratio}",
        f"DUPLICATE_ROWS: {audit.duplicate_rows}",
        f"HEADER_MODE: {audit.header_mode}",
        f"SPANS_PRESENT: {audit.spans_present}",
        "",
        "=== HTML ===",
        audit.html,
        "",
        "=== GITHUB_MD ===",
        audit.markdown if audit.markdown is not None else "UNAVAILABLE",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def write_clean_markdown_file(output_dir: Path, audit: TableAudit) -> Optional[str]:
    if audit.markdown is None:
        return None
    filename = f"{audit.global_index:05d}__{audit.source_stem}__table_{audit.table_index_in_doc:03d}.md"
    output_path = output_dir / filename
    output_path.write_text(
        "\n".join(
            [
                "## ORIGINAL_HTML",
                "",
                audit.html,
                "",
                "## GITHUB_MD",
                "",
                audit.markdown,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HTML tables and export GitHub Markdown conversions.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tables", type=int, default=1000)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    clean_md_dir = output_dir / "github_md_tables"
    clean_md_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    if manifest_path.exists():
        manifest_path.unlink()
    if summary_path.exists():
        summary_path.unlink()
    for stale in tables_dir.glob("*.txt"):
        stale.unlink()
    for stale in clean_md_dir.glob("*.md"):
        stale.unlink()

    rows = []
    audited = 0
    for source_path, table_index, global_index, table_html in iter_tables(args.input_dir):
        audited += 1
        audit = audit_table(source_path, table_index, global_index, table_html)
        output_path = write_review_file(tables_dir, audit)
        markdown_path = write_clean_markdown_file(clean_md_dir, audit)
        row = {
            "source_path": audit.source_path,
            "source_stem": audit.source_stem,
            "table_index_in_doc": audit.table_index_in_doc,
            "global_index": audit.global_index,
            "status": audit.status,
            "convertible": audit.convertible,
            "broken": audit.broken,
            "reasons": audit.reasons,
            "row_count": audit.row_count,
            "col_count": audit.col_count,
            "nonempty_ratio": audit.nonempty_ratio,
            "duplicate_rows": audit.duplicate_rows,
            "header_mode": audit.header_mode,
            "spans_present": audit.spans_present,
            "output_path": output_path,
            "markdown_output_path": markdown_path,
        }
        rows.append(row)
        if audited >= args.max_tables:
            break

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    reason_counts = Counter(reason for row in rows for reason in row["reasons"])
    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "github_md_dir": str(clean_md_dir),
        "audited_table_count": len(rows),
        "convertible_count": sum(1 for row in rows if row["convertible"]),
        "broken_count": sum(1 for row in rows if row["broken"]),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
