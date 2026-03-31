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


# --- Early termination guards (lightweight O(n) scans) ---

def _detect_repeated_char_cut(text: str, *, threshold: int = 200) -> Optional[int]:
    """Return the index at which to cut if any single character repeats >= threshold.

    The returned index is at the start of the offending run (exclusive of the run).
    Runs reset across newlines. Complexity: O(n) time, O(1) space.
    """
    if threshold <= 1:
        return 0
    last_char: Optional[str] = None
    run_len = 0
    run_start = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            last_char = None
            run_len = 0
            continue
        if ch == last_char:
            run_len += 1
            if run_len >= threshold:
                # Keep up to `threshold` repeated characters, cut after that
                return run_start + threshold
        else:
            last_char = ch
            run_len = 1
            run_start = i
    return None


def _detect_repeated_lines_cut(text: str, *, threshold: int = 10) -> Optional[int]:
    """Return the index at which to cut if the same line repeats >= threshold times.

    Comparison uses ``strip()`` normalization. The cut position is at the start of
    the repeated run (the first repetition). Complexity: O(n) time, O(1) space.
    """
    if threshold <= 1:
        return 0
    prev_norm: Optional[str] = None
    prev_start: int = 0
    run_count: int = 0
    run_start_idx: Optional[int] = None
    i = 0
    n = len(text)
    while i <= n:
        # Find end of current line (or end of text)
        j = i
        while j < n and text[j] != "\n":
            j += 1
        line = text[i:j]
        norm = line.strip()
        if prev_norm is not None and norm == prev_norm:
            if run_count == 1:
                # Start of repetition run spans from previous line start
                run_start_idx = prev_start
            run_count += 1
            # Allow the first `threshold` identical lines, cut at the start of the (threshold+1)-th
            if run_count > threshold:
                return i
        else:
            prev_norm = norm
            prev_start = i
            run_count = 1
            run_start_idx = None
        # Advance to next line (skip newline char if present)
        i = j + 1
    return None


def _is_private_use_char(ch: str) -> bool:
    codepoint = ord(ch)
    return (
        0xE000 <= codepoint <= 0xF8FF
        or 0xF0000 <= codepoint <= 0xFFFFD
        or 0x100000 <= codepoint <= 0x10FFFD
    )


def _is_symbol_garbage_char(ch: str) -> bool:
    if _is_private_use_char(ch):
        return True
    return ch in {
        "•",
        "",
        "·",
        "◦",
        "▪",
        "▫",
        "‣",
        "∙",
        "⋅",
        "●",
        "○",
        "◉",
        "◌",
        "◆",
        "◇",
        "■",
        "□",
        "▲",
        "△",
        "▼",
        "▽",
        "►",
        "◄",
        "◊",
        "",
        "",
        "",
        "",
        "",
        "",
    }


def _detect_symbol_garbage_cut(text: str, *, threshold: int = 16) -> Optional[int]:
    """Cut on long runs of isolated bullet/dingbat/private-use symbols.

    This targets the common DeepSeek garbage mode where the model emits long
    whitespace-separated runs of bullets or private-use glyphs instead of text.
    """
    if threshold <= 1:
        return 0
    run_count = 0
    run_start: Optional[int] = None
    last_non_ws = -10_000
    for index, ch in enumerate(text):
        if ch.isspace():
            continue
        if _is_symbol_garbage_char(ch):
            if run_count == 0 or (index - last_non_ws) > 3:
                run_start = index
                run_count = 1
            else:
                run_count += 1
            last_non_ws = index
            if run_count >= threshold:
                return run_start
            continue
        run_count = 0
        run_start = None
        last_non_ws = index
    return None


NUMERIC_LIST_TOKEN_PATTERN = re.compile(r"(?<!\S)(\d{1,3})[.)](?=\s|$)")


def _detect_numeric_list_garbage_cut(text: str, *, threshold: int = 12) -> Optional[int]:
    """Cut on degenerate `1. 2. 3. ...` style list output."""
    if threshold <= 1:
        return 0
    matches = list(NUMERIC_LIST_TOKEN_PATTERN.finditer(text))
    if len(matches) < threshold:
        return None
    run_start = matches[0].start()
    run_count = 1
    prev_value = int(matches[0].group(1))
    prev_end = matches[0].end()
    for match in matches[1:]:
        current_value = int(match.group(1))
        gap = text[prev_end : match.start()]
        if current_value == prev_value + 1 and len(gap) <= 4 and gap.strip() == "":
            run_count += 1
        else:
            run_start = match.start()
            run_count = 1
        if run_count >= threshold:
            return run_start
        prev_value = current_value
        prev_end = match.end()
    return None


class StreamingGarbageDetector:
    """Incremental detector for common OCR garbage generation modes.

    This is designed for hot decode loops: feed only newly decoded text chunks
    and keep O(1) mutable state instead of rescanning the whole suffix.
    """

    def __init__(
        self,
        *,
        symbol_threshold: int = 16,
        numeric_list_threshold: int = 12,
    ) -> None:
        self.symbol_threshold = int(symbol_threshold)
        self.numeric_list_threshold = int(numeric_list_threshold)
        self._symbol_run = 0
        self._numeric_run = 0
        self._expected_next_number: Optional[int] = None
        self._digits_buffer: str = ""
        self.triggered_reason: Optional[str] = None

    def reset(self) -> None:
        self._symbol_run = 0
        self._numeric_run = 0
        self._expected_next_number = None
        self._digits_buffer = ""
        self.triggered_reason = None

    def _reset_numeric(self) -> None:
        self._numeric_run = 0
        self._expected_next_number = None
        self._digits_buffer = ""

    def _feed_symbol_char(self, ch: str) -> bool:
        if ch.isspace():
            return False
        if _is_symbol_garbage_char(ch):
            self._symbol_run += 1
            if self._symbol_run >= self.symbol_threshold:
                self.triggered_reason = "symbol_garbage"
                return True
            return False
        self._symbol_run = 0
        return False

    def _feed_numeric_char(self, ch: str) -> bool:
        if ch.isspace():
            if self._digits_buffer:
                self._reset_numeric()
            return False
        if "0" <= ch <= "9":
            self._digits_buffer += ch
            return False
        if ch in {".", ")"} and self._digits_buffer:
            value = int(self._digits_buffer)
            self._digits_buffer = ""
            if self._expected_next_number is None:
                if value == 1:
                    self._numeric_run = 1
                    self._expected_next_number = 2
                else:
                    self._reset_numeric()
            else:
                if value == self._expected_next_number:
                    self._numeric_run += 1
                    self._expected_next_number += 1
                elif value == 1:
                    self._numeric_run = 1
                    self._expected_next_number = 2
                else:
                    self._reset_numeric()
            if self._numeric_run >= self.numeric_list_threshold:
                self.triggered_reason = "numeric_list_garbage"
                return True
            return False
        self._reset_numeric()
        return False

    def feed(self, text: str) -> bool:
        if self.triggered_reason is not None:
            return True
        for ch in str(text or ""):
            if self._feed_symbol_char(ch):
                return True
            if self._feed_numeric_char(ch):
                return True
        return False


def detect_early_stop_index(
    text: str,
    *,
    line_repeat_threshold: int = 10,
    char_repeat_threshold: int = 200,
    symbol_garbage_threshold: int = 16,
    numeric_list_threshold: int = 12,
) -> Optional[int]:
    """Find earliest cut index based on repetition heuristics.

    Returns the smallest index where content should be cut to avoid degenerate
    loops. If no condition triggers, returns ``None``.
    """
    idx_char = _detect_repeated_char_cut(text, threshold=char_repeat_threshold)
    idx_line = _detect_repeated_lines_cut(text, threshold=line_repeat_threshold)
    idx_symbol = _detect_symbol_garbage_cut(text, threshold=symbol_garbage_threshold)
    idx_numeric = _detect_numeric_list_garbage_cut(text, threshold=numeric_list_threshold)
    candidates = [idx for idx in (idx_char, idx_line, idx_symbol, idx_numeric) if idx is not None]
    if not candidates:
        return None
    return min(candidates)


def apply_early_stop(
    text: str,
    *,
    content_debug: bool = False,
    line_repeat_threshold: int = 10,
    char_repeat_threshold: int = 200,
    symbol_garbage_threshold: int = 16,
    numeric_list_threshold: int = 12,
    metrics: Optional[dict] = None,
) -> str:
    """Apply early termination heuristics to ``text`` and optionally append notice.

    - Cuts when a single character repeats ``char_repeat_threshold`` times.
    - Cuts when the same line repeats ``line_repeat_threshold`` times.
    - Appends a debug marker only when ``content_debug`` is True.
    Complexity: O(n).
    """
    cut = detect_early_stop_index(
        text,
        line_repeat_threshold=line_repeat_threshold,
        char_repeat_threshold=char_repeat_threshold,
        symbol_garbage_threshold=symbol_garbage_threshold,
        numeric_list_threshold=numeric_list_threshold,
    )
    if cut is None:
        return text
    if metrics is not None:
        metrics["early_stops"] = metrics.get("early_stops", 0) + 1
    truncated = text[:cut].rstrip()
    if content_debug:
        notice = "[[Early stop: repetition detected]]"
        return f"{truncated}\n\n{notice}" if truncated else notice
    return truncated
