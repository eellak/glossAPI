"""Shared OCR span rendering and match-index helpers.

This module owns the last stage of OCR page handling:
- merge raw detector spans into one reviewed span plan
- render that exact plan in `debug` or `clean` mode
- serialize match metadata for later inspection

Keeping this logic out of `phase_clean.py` is intentional. The analyzer should
answer *what* spans exist and in what ownership order; this module controls
*how* those exact spans become page text and debug sidecars.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .ocr_table import (
    render_table_html_for_clean as _render_table_html_for_clean,
    render_table_html_for_output as _render_table_html_for_output,
    replace_html_tables_with_markdown as _replace_html_tables_with_markdown,
)

# Neighboring same-category spans may merge when the visible separator is still
# short enough to read as one corrupted region rather than two separate
# failures. This is intentionally more permissive than the older 10-char rule.
WORD_REPEAT_MERGE_MAX_NONWHITESPACE_GAP = 40


def _gap_has_at_most_n_nonwhitespace_chars(text: str, start: int, end: int, limit: int) -> bool:
    if start >= end:
        return True
    count = 0
    for ch in text[start:end]:
        if ch.isspace():
            continue
        count += 1
        if count > limit:
            return False
    return True


def _clean_fill_for_removed_span(page_text: str, start: int, end: int) -> str:
    """Choose the smallest filler that keeps surrounding text readable.

    Clean mode removes non-table matches from the page surface. We keep this
    filler policy centralized because tiny whitespace decisions directly affect
    byte-for-byte debug/clean parity and downstream markdown readability.
    """
    removed = page_text[start:end]
    prev_char = page_text[start - 1] if start > 0 else ""
    next_char = page_text[end] if end < len(page_text) else ""
    if "\n" in removed:
        if prev_char == "\n" or next_char == "\n":
            return ""
        return "\n"
    if prev_char and next_char and not prev_char.isspace() and not next_char.isspace():
        return " "
    return ""


def _merge_labeled_raw_spans(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clamp, sort, and merge raw detector spans into one render plan.

    The merge layer is the final contract between detectors and renderers.
    It must be defensive about detector offsets and explicit about when nearby
    spans of the same category should collapse into one visible debug match.
    """
    if not spans:
        return []

    text_len = len(text)
    sanitized_spans: List[Dict[str, Any]] = []
    for span in spans:
        start = max(0, int(span["start"]))
        end = min(text_len, int(span["end"]))
        if start >= text_len or end <= start:
            continue
        sanitized = dict(span)
        sanitized["start"] = start
        sanitized["end"] = end
        sanitized_spans.append(sanitized)
    if not sanitized_spans:
        return []

    ordered_spans = sorted(sanitized_spans, key=lambda item: (item["start"], item["end"]))
    merged: List[Dict[str, Any]] = []
    for span in ordered_spans:
        if not merged:
            merged.append(dict(span))
            continue

        previous = merged[-1]
        overlaps = span["start"] <= previous["end"]
        close_gap = (
            not overlaps
            and previous["category"] == span["category"]
            and previous["category"] != "table"
            and _gap_has_at_most_n_nonwhitespace_chars(
                text,
                previous["end"],
                span["start"],
                WORD_REPEAT_MERGE_MAX_NONWHITESPACE_GAP,
            )
        )
        if overlaps or close_gap:
            same_single_type = previous.get("match_types", []) == span.get("match_types", [])
            same_kind = previous.get("kind") == span.get("kind")
            previous["start"] = min(previous["start"], span["start"])
            previous["end"] = max(previous["end"], span["end"])
            previous["match_types"] = sorted(
                set(previous.get("match_types", [])) | set(span.get("match_types", []))
            )
            if (
                previous.get("kind") is None
                and span.get("kind") is not None
                and previous.get("match_types", []) == span.get("match_types", [])
            ):
                previous["kind"] = span.get("kind")
            if "period" in span:
                previous["period"] = min(previous.get("period", span["period"]), span["period"])
            if "repetitions" in span:
                previous["repetitions"] = max(
                    previous.get("repetitions", span["repetitions"]),
                    span["repetitions"],
                )
            if "tail_chars" in span:
                previous["tail_chars"] = max(
                    previous.get("tail_chars", 0),
                    span.get("tail_chars", 0),
                )
            if (
                same_single_type
                and same_kind
                and previous.get("item_count") is not None
                and span.get("item_count") is not None
            ):
                previous["item_count"] = int(previous["item_count"]) + int(span["item_count"])
            continue
        merged.append(dict(span))
    return merged


def _summarize_merged_labeled_spans(
    merged_spans: List[Dict[str, Any]],
) -> Tuple[List[str], int, int, int, int, int]:
    seen_types: Set[str] = set()
    numeric_count = 0
    word_count = 0
    latex_count = 0
    table_count = 0
    hybrid_count = 0
    for span in merged_spans:
        seen_types.update(span.get("match_types", []))
        if span["category"] == "numeric":
            numeric_count += 1
        elif span["category"] == "word":
            word_count += 1
        elif span["category"] == "latex":
            latex_count += 1
        elif span["category"] == "table":
            table_count += 1
        elif span["category"] == "hybrid":
            hybrid_count += 1
    return (
        sorted(seen_types),
        numeric_count,
        word_count,
        latex_count,
        table_count,
        hybrid_count,
    )


def _build_debug_match_open_tag(span: Dict[str, Any]) -> str:
    match_types = list(span.get("match_types", []))
    open_tag = f'<match of type {",".join(match_types)}'
    if match_types == ["word_repeat"]:
        open_tag += f' period={span.get("period", 0)} reps={span.get("repetitions", 0)}'
    elif match_types == ["latex_repeat"]:
        if span.get("kind"):
            open_tag += f' kind={span.get("kind")}'
        if span.get("item_count") is not None:
            open_tag += f' items={span.get("item_count")}'
    elif match_types == ["table_repeat"]:
        if span.get("kind"):
            open_tag += f' kind={span.get("kind")}'
        if span.get("row_count") is not None:
            open_tag += f' rows={span.get("row_count")}'
        if span.get("duplicate_rows") is not None:
            open_tag += f' dup_rows={span.get("duplicate_rows")}'
        if span.get("nonempty_ratio") is not None:
            open_tag += f' nonempty_ratio={span.get("nonempty_ratio")}'
        if span.get("word_count") is not None:
            open_tag += f' words={span.get("word_count")}'
        if span.get("char_count") is not None:
            open_tag += f' chars={span.get("char_count")}'
    elif match_types == ["hybrid_repeat"]:
        if span.get("kind"):
            open_tag += f' kind={span.get("kind")}'
        if span.get("item_count") is not None:
            open_tag += f' items={span.get("item_count")}'
        if span.get("cycle_len") is not None:
            open_tag += f' cycle={span.get("cycle_len")}'
    return open_tag + ">"


def _render_page_from_merged_labeled_spans(
    page_text: str,
    merged_spans: List[Dict[str, Any]],
    *,
    mode: str,
) -> str:
    if not merged_spans:
        return _replace_html_tables_with_markdown(page_text)

    parts: List[str] = []
    pos = 0
    for span in merged_spans:
        start = span["start"]
        end = span["end"]
        if start > pos:
            parts.append(_replace_html_tables_with_markdown(page_text[pos:start]))
        match_types = list(span.get("match_types", []))
        if mode == "debug":
            parts.append(_build_debug_match_open_tag(span))
            if match_types == ["table_repeat"]:
                parts.append(
                    _render_table_html_for_output(
                        page_text[start:end],
                        match_kind=span.get("kind"),
                    )
                )
            else:
                parts.append(page_text[start:end])
            parts.append("</match>")
        else:
            if match_types == ["table_repeat"]:
                parts.append(
                    _render_table_html_for_clean(
                        page_text[start:end],
                        match_kind=span.get("kind"),
                    )
                )
            else:
                parts.append(_clean_fill_for_removed_span(page_text, start, end))
        pos = end
    if pos < len(page_text):
        parts.append(_replace_html_tables_with_markdown(page_text[pos:]))
    return "".join(parts)


def _render_page_with_labeled_spans_result(
    page_text: str,
    spans: List[Dict[str, Any]],
    *,
    mode: str = "debug",
) -> Dict[str, Any]:
    if mode not in {"debug", "clean"}:
        raise ValueError(f"Unsupported OCR render mode: {mode}")
    merged_spans = _merge_labeled_raw_spans(page_text, spans)
    (
        page_types,
        numeric_count,
        word_count,
        latex_count,
        table_count,
        hybrid_count,
    ) = _summarize_merged_labeled_spans(merged_spans)
    rendered_page = _render_page_from_merged_labeled_spans(
        page_text,
        merged_spans,
        mode=mode,
    )
    return {
        "rendered_page": rendered_page,
        "merged_spans": merged_spans,
        "page_types": page_types,
        "page_numeric_count": numeric_count,
        "page_word_count": word_count,
        "page_latex_count": latex_count,
        "page_table_count": table_count,
        "page_hybrid_count": hybrid_count,
    }


def _render_page_with_labeled_spans(
    page_text: str,
    spans: List[Dict[str, Any]],
    *,
    mode: str = "debug",
) -> Tuple[str, List[str], int, int, int, int, int]:
    """Render one page from a shared span plan.

    `debug` and `clean` intentionally share the exact same merged span plan.
    The only difference is how that plan is rendered:
    - debug wraps the matched source surface in `<match ...>` tags
    - clean removes or rewrites the matched surface according to policy

    Keeping both modes on one renderer prevents the real cleaner from drifting
    away from the reviewed debug output.
    """
    result = _render_page_with_labeled_spans_result(page_text, spans, mode=mode)
    return (
        str(result["rendered_page"]),
        list(result["page_types"]),
        int(result["page_numeric_count"]),
        int(result["page_word_count"]),
        int(result["page_latex_count"]),
        int(result["page_table_count"]),
        int(result["page_hybrid_count"]),
    )


def _annotate_page_with_labeled_spans(
    page_text: str,
    spans: List[Dict[str, Any]],
) -> Tuple[str, List[str], int, int, int, int, int]:
    return _render_page_with_labeled_spans(page_text, spans, mode="debug")


def _utf8_prefix_byte_offsets(text: str) -> List[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def _span_repeat_count(span: Dict[str, Any]) -> Optional[int]:
    if span.get("repetitions") is not None:
        return int(span["repetitions"])
    if span.get("item_count") is not None:
        return int(span["item_count"])
    if span.get("duplicate_rows") is not None:
        return int(span["duplicate_rows"])
    return None


def _build_match_index_rows(
    page_text: str,
    merged_spans: List[Dict[str, Any]],
    *,
    source_path: Path,
    page_number: int,
    debug_output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Serialize the exact merged spans that drove the visible debug page.

    Match offsets are recorded *after* span clamping and merging so index rows
    refer to the same boundaries a reviewer sees in `<match ...>` output.
    """
    if not merged_spans:
        return []
    byte_offsets = _utf8_prefix_byte_offsets(page_text)
    rows: List[Dict[str, Any]] = []
    for match_index, span in enumerate(merged_spans, start=1):
        start = int(span["start"])
        end = int(span["end"])
        match_text = page_text[start:end]
        rows.append(
            {
                "match_id": f"{source_path.stem}:page:{page_number}:match:{match_index}",
                "source_path": str(source_path),
                "source_stem": source_path.stem,
                "debug_output_path": None if debug_output_path is None else str(debug_output_path),
                "page_number": int(page_number),
                "page_index_in_file": int(page_number),
                "match_index_in_page": int(match_index),
                "start_char": start,
                "end_char": end,
                "start_byte": int(byte_offsets[start]),
                "end_byte": int(byte_offsets[end]),
                "match_length_chars": int(end - start),
                "match_length_bytes": int(byte_offsets[end] - byte_offsets[start]),
                "match_types": list(span.get("match_types", [])),
                "match_type": ",".join(span.get("match_types", [])),
                "category": str(span.get("category", "")),
                "kind": span.get("kind"),
                "repeat_count": _span_repeat_count(span),
                "period": span.get("period"),
                "repetitions": span.get("repetitions"),
                "tail_chars": span.get("tail_chars"),
                "item_count": span.get("item_count"),
                "cycle_len": span.get("cycle_len"),
                "row_count": span.get("row_count"),
                "duplicate_rows": span.get("duplicate_rows"),
                "nonempty_ratio": span.get("nonempty_ratio"),
                "word_count": span.get("word_count"),
                "char_count": span.get("char_count"),
                "matched_text": match_text,
            }
        )
    return rows


__all__ = [
    "WORD_REPEAT_MERGE_MAX_NONWHITESPACE_GAP",
    "_annotate_page_with_labeled_spans",
    "_build_match_index_rows",
    "_merge_labeled_raw_spans",
    "_render_page_with_labeled_spans",
    "_render_page_with_labeled_spans_result",
    "_summarize_merged_labeled_spans",
]
