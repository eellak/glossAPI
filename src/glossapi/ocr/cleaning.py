from __future__ import annotations

import html
import re
from typing import Iterable, List, Optional, Tuple

# Common patterns and utilities for OCR text post-processing.


FULL_WIDTH_BAR = "\uFF5C"
FULL_WIDTH_LT = "\uFF1C"

ORPHAN_META_FRAGMENT_PATTERN = re.compile(
    rf"<[|{FULL_WIDTH_BAR}][^>]{0,64}[|{FULL_WIDTH_BAR}]>", re.DOTALL
)
LEFTOVER_META_PATTERN = re.compile(rf"(?:<|{FULL_WIDTH_LT})(?:\||{FULL_WIDTH_BAR})")
REFDET_BLOCK_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>", re.DOTALL | re.IGNORECASE
)
BOUNDING_BOX_PREFIX_PATTERN = re.compile(r"^\s*[A-Za-z_]+\[\[.*?\]\]\s*")
PLACEHOLDER_CELL_PATTERN = re.compile(
    r"(<td\b[^>]*>)(.*?)(</td>)", re.IGNORECASE | re.DOTALL
)
TABLE_BLOCK_PATTERN = re.compile(
    r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL
)
NBSP_PATTERN = re.compile(r"(?:&nbsp;|\u00A0)+", re.IGNORECASE)
DEHYPHEN_PATTERN = re.compile(r"(?<=\w)-\n(?=[a-zα-ωά-ώ])", re.IGNORECASE)
INLINE_LATEX_PATTERN = re.compile(r"\\\((.+?)\\\)")
BLOCK_LATEX_PATTERN = re.compile(r"\\\[\s*(.+?)\s*\\\]", re.DOTALL)
SIMPLE_SUP_PATTERN = re.compile(r"\$\^\{?([A-Za-z0-9+\-]+)\}?\$")
SIMPLE_SUB_PATTERN = re.compile(r"\$_\{?([A-Za-z0-9+\-]+)\}?\$")
CITATION_SUP_PATTERN = re.compile(r"<sup>(\d{2,}(?:[A-Za-z]{1,2})?)</sup>")


def strip_prompt_echo(text: str, prompt: str) -> str:
    lines = [
        line.strip()
        for line in prompt.splitlines()
        if line.strip() and line.strip() != "<image>"
    ]
    for line in lines:
        escaped = re.escape(line)
        pattern = re.compile(rf"(?:{escaped})(?:\s+|$)")
        while True:
            new_text = pattern.sub("", text, count=1)
            if new_text == text:
                break
            text = new_text.strip()
    return text


def _normalize_cell_text(fragment: str) -> str:
    text = html.unescape(fragment)
    text = NBSP_PATTERN.sub(" ", text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PLACEHOLDER_VALUES = {
    "none",
    "n/a",
    "na",
    "null",
    "--",
    "—",
    "–",
    "−",
    "-",
    "•",
    "·",
    "[[blank page]]",
    "[blank]",
    "(blank)",
}


def _is_placeholder_content(fragment: str) -> bool:
    normalized = _normalize_cell_text(fragment)
    if not normalized:
        return True
    lowered = normalized.lower()
    compact = re.sub(r"[\s._\-\\/]+", "", lowered)
    return lowered in PLACEHOLDER_VALUES or compact in PLACEHOLDER_VALUES


def prune_placeholder_cells(html_text: str, metrics: Optional[dict] = None) -> str:
    def replacer(match: re.Match[str]) -> str:
        opening, body, closing = match.groups()
        if _is_placeholder_content(body):
            if metrics is not None:
                metrics["placeholder_cells_pruned"] = (
                    metrics.get("placeholder_cells_pruned", 0) + 1
                )
            return f"{opening}{closing}"
        return f"{opening}{body}{closing}"

    return PLACEHOLDER_CELL_PATTERN.sub(replacer, html_text)


def drop_empty_tables(html_text: str, metrics: Optional[dict] = None) -> str:
    def replace_table(match: re.Match[str]) -> str:
        table_html = match.group(0)
        # If there are <td> with non-placeholder content, keep the table
        for cell_match in PLACEHOLDER_CELL_PATTERN.finditer(table_html):
            cell_body = cell_match.group(2)
            if not _is_placeholder_content(cell_body):
                return table_html
        # If headers with content exist, keep
        for header_match in re.finditer(r"<th\b[^>]*>(.*?)</th>", table_html, re.IGNORECASE | re.DOTALL):
            header_content = _normalize_cell_text(header_match.group(1))
            if header_content:
                return table_html
        if metrics is not None:
            metrics["tables_dropped"] = metrics.get("tables_dropped", 0) + 1
        return ""

    return TABLE_BLOCK_PATTERN.sub(replace_table, html_text)


def _default_special_token_pattern(keep_refdet: bool) -> re.Pattern:
    tokens = {
        "<|User|>",
        "<|Assistant|>",
        "<|grounding|>",
        "<image>",
        "</s>",
        "<s>",
        # ref/det tokens are optionally kept
        "<|ref|>",
        "<|/ref|>",
        "<|det|>",
        "<|/det|>",
    }
    if keep_refdet:
        tokens.difference_update({"<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>"})
    variants = set()
    for tok in tokens:
        variants.add(re.escape(tok))
        if "|" in tok:
            variants.add(re.escape(tok.replace("|", FULL_WIDTH_BAR)))
    return re.compile("|".join(sorted(variants, key=len, reverse=True)))


def clean_output(text: str, *, keep_refdet: bool, metrics: Optional[dict] = None) -> str:
    # Remove common special tokens and artifacts
    pattern = _default_special_token_pattern(keep_refdet)
    text = text.replace("<s>", "").replace("</s>", "")
    text = pattern.sub("", text)
    text = ORPHAN_META_FRAGMENT_PATTERN.sub("", text)

    # Drop heavy LaTeX/TikZ blocks to avoid noise in Markdown
    text = re.sub(r"(?:\\draw\s*\([^;]*\);\s*){10,}", "", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "[[Figure omitted; refer to original page image]]", text, flags=re.DOTALL)
    text = re.sub(r"(?:\[\s*\\begin\{array\}.*?\\end\{array\}\s*\]){3,}", "[[Matrix omitted; refer to original page image]]", text, flags=re.DOTALL)

    # Line-wise cleanup and bounding box prefixes
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = BOUNDING_BOX_PREFIX_PATTERN.sub("", raw)
        line = line.replace("<center>", "").replace("</center>", "")
        stripped = line.strip()
        if not stripped:
            if out_lines and out_lines[-1] == "":
                continue
            out_lines.append("")
            continue
        out_lines.append(stripped)

    cleaned = "\n".join(out_lines).strip()
    # Replace simple LaTeX super/subscripts
    cleaned = SIMPLE_SUP_PATTERN.sub(r"<sup>\1</sup>", cleaned)
    cleaned = SIMPLE_SUB_PATTERN.sub(r"<sub>\1</sub>", cleaned)
    # Prune placeholder cells and empty tables
    cleaned = prune_placeholder_cells(cleaned, metrics)
    cleaned = drop_empty_tables(cleaned, metrics)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def canonicalize_markdown(text: str) -> str:
    text = NBSP_PATTERN.sub(" ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = DEHYPHEN_PATTERN.sub("", text)
    text = prune_placeholder_cells(text)
    text = drop_empty_tables(text)
    text = CITATION_SUP_PATTERN.sub(lambda m: f"[^{m.group(1)}]", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

