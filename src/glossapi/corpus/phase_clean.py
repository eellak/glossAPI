"""Cleaning and filtering helpers split from Corpus.

This module now primarily owns OCR orchestration:
- page-level analyzer ordering
- shared clean/debug rendering
- worker/process orchestration

Specialized policy modules, like HTML-table handling, live alongside it so the
main pipeline can stay focused on span ownership and mode selection.
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._naming import canonical_stem
from ..gloss_downloader import GlossDownloader
# Avoid importing section/classifier here; cleaning phase does not use them.
from .ocr_table import (
    HTML_TABLE_BLOCK_RE,
    HTML_TABLE_LINE_RE,
    find_table_repeat_spans as _find_table_repeat_spans_impl,
    render_table_html_for_clean as _render_table_html_for_clean,
    render_table_html_for_output as _render_table_html_for_output,
    replace_html_tables_with_markdown as _replace_html_tables_with_markdown,
)
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch
from .ocr_render import (
    _gap_has_at_most_n_nonwhitespace_chars,
    _clean_fill_for_removed_span,
    _merge_labeled_raw_spans,
    _summarize_merged_labeled_spans,
    _render_page_from_merged_labeled_spans,
    _render_page_with_labeled_spans_result,
    _render_page_with_labeled_spans,
    _annotate_page_with_labeled_spans,
    _utf8_prefix_byte_offsets,
    _span_repeat_count,
    _build_match_index_rows,
)
from .text_surface_metrics import sanitized_char_count

PAGE_SPLIT_MARKER = "<--- Page Split --->"
WORD_REPEAT_HASH_MASK = (1 << 64) - 1
WORD_REPEAT_HASH_BASE = 1469598103934665603
# Neighboring same-category spans may be merged when the visible separator is
# still small enough to read as one corrupted region rather than two separate
# failures. This is intentionally more permissive than the older 10-char rule.
WORD_REPEAT_MERGE_MAX_NONWHITESPACE_GAP = 40
EXISTING_MATCH_BLOCK_RE = re.compile(r"(?is)<match\b[^>]*>.*?</match\s*>")
LATEX_BLOCK_RE = re.compile(r"(?is)\$\$.*?\$\$")
LATEX_BRACKET_RE = re.compile(r"(?is)\\\[.*?\\\]")
LATEX_BEGIN_END_RE = re.compile(r"(?is)\\begin\{([^\n{}]+)\}.*?\\end\{\1\}")
LATEX_INLINE_PAREN_RE = re.compile(r"(?is)\\\(.*?\\\)")
LATEX_INLINE_DOLLAR_RE = re.compile(r"(?s)(?<!\$)\$(?!\$)(?:\\.|[^$\n])+\$(?!\$)")
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
LATEX_TEXT_WRAPPER_BODY_RE = re.compile(
    r"\\(?:mathrm|text|operatorname|mathit|mathbf)\{([^{}]*)\}"
)
LATEX_TEXT_WRAPPER_OPEN_BODY_RE = re.compile(
    r"\\(?:mathrm|text|operatorname|mathit|mathbf)\{([^\}\n]*)"
)
HTML_MATH_MARKUP_CLUSTER_RE = re.compile(
    r"(?:[A-Za-zΑ-Ωα-ω](?:<sub>[^<]{1,16}</sub>|<sup>[^<]{1,16}</sup>)){8,}"
)
WORD_CONFUSABLE_FOLD_MAP = {
    "ο": "o",
    "κ": "k",
}
LATEX_SEGMENT_PATTERNS = [
    ("begin_end", LATEX_BEGIN_END_RE),
    ("display_dollar", LATEX_BLOCK_RE),
    ("display_bracket", LATEX_BRACKET_RE),
    ("inline_paren", LATEX_INLINE_PAREN_RE),
    ("inline_dollar", LATEX_INLINE_DOLLAR_RE),
]
LATEX_TEXT_WRAPPER_MACROS = (
    r"\mathrm{",
    r"\text{",
    r"\operatorname{",
    r"\mathit{",
    r"\mathbf{",
)
LATEX_INTERNAL_REPEAT_COMMANDS = {
    r"\frac",
    r"\left",
    r"\right",
    r"\sqrt",
    r"\begin",
    r"\end",
    r"\quad",
    r"\qquad",
    r"\cdots",
    r"\ldots",
    r"\mathrm",
    r"\text",
    r"\operatorname",
    r"\mathit",
    r"\mathbf",
    r"\hat",
    r"\tilde",
    r"\bar",
}
LATEX_INTERNAL_SMALL_VOCAB_COMMANDS = {
    r"\cdots",
    r"\ldots",
    r"\vdots",
    r"\ddots",
}
LATEX_SHORT_REPEAT_ATOM_COMMANDS = {
    r"\Delta",
    r"\hat",
    r"\tilde",
    r"\bar",
}
LATEX_SHORT_ATOM_BLOCK_BASE_COMMANDS = {
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\epsilon",
    r"\varepsilon",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\omega",
    r"\Delta",
}
LATEX_SHORT_ATOM_BLOCK_DECORATOR_COMMANDS = {
    r"\hat",
    r"\tilde",
    r"\bar",
}
LATEX_SEGMENT_LOCAL_NONWHITESPACE_GAP = 12
LATEX_SEGMENT_EXACT_RUN_MIN = 4
LATEX_SEGMENT_SKELETON_RUN_MIN = 4
LATEX_SEGMENT_ALTERNATING_RUN_MIN = 6
LATEX_SEGMENT_SLOT_PROGRESS_RUN_MIN = 4
LATEX_SHORT_ATOM_BLOCK_REPEAT_MIN_ITEMS = 12
LATEX_SHORT_ATOM_EXACT_SEGMENT_MIN_TOKENS = 2
LATEX_SHORT_ATOM_CHAIN_MIN_TOKENS = 6
LATEX_INTERNAL_SMALL_VOCAB_RUN_MIN_COMMANDS = 24
LATEX_SHORT_SEGMENT_MAX_NORM = 32
LATEX_LONG_SEGMENT_MIN_NORM = 24
LATEX_INTERNAL_REPEAT_MIN_COMMAND_DUP = 3
LATEX_SMALL_DEFINITION_FAMILY_MAX_RUN = 6
HYBRID_PREFIX_RE = re.compile(
    r"(?<!\d)(?P<prefix>\d+\)|\d+\.(?:\d+\.)*\d*\.?)(?=\s*[^\W\d_])",
    re.UNICODE,
)
HYBRID_MARKUP_BODY_RE = re.compile(r"(?i)(<[^>]+>|src=|alt=|image_|\.png\b|\.jpg\b|\.jpeg\b|\.gif\b)")
HYBRID_REPEAT_MIN_ITEMS = 4
HYBRID_REPEAT_MIN_BODY_ALNUM = 6
HYBRID_REPEAT_MAX_CYCLE = 6
HYBRID_REPEAT_MIN_CYCLE_ITEMS = 8
HYBRID_INLINE_CLAUSE_DELIMITER_RE = re.compile(r"[;\n]|,(?!\d)")
HYBRID_INLINE_TOKEN_RE = re.compile(r"[0-9]+(?:[.,/][0-9]+)*|[^\W\d_]+", re.UNICODE)
HYBRID_INLINE_CONTEXT_WORDS = 2
HYBRID_INLINE_CONTEXT_MIN_ALPHA_WORDS = 2
HYBRID_INLINE_CONTEXT_MIN_CHARS = 8
HYBRID_INLINE_REPEAT_MIN_ITEMS = 6
LATEX_SYMBOL_SLOT_COMMANDS = (
    r"\mu",
    r"\nu",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\lambda",
    r"\tau",
    r"\omega",
)
MATCH_CATEGORY_BY_TYPE = {
    "ascending_numeric_sequence": "numeric",
    "repeat_numeric_run": "numeric",
    "same_digit_numeric_run": "numeric",
    "numeric_page_collapse": "numeric",
    "numeric_block_collapse": "numeric",
    "numeric_repeat": "numeric",
    "word_repeat": "word",
    "latex_repeat": "latex",
    "hybrid_repeat": "hybrid",
    "table_repeat": "table",
}

_WORD_REPEAT_RUST_MOD: Optional[Any] = None
_WORD_REPEAT_RUST_IMPORT_ATTEMPTED = False
_RUST_EXTENSION_PREBUILD_ATTEMPTED: Set[str] = set()
_COMBINED_OCR_WORKER_NOISE_MOD: Optional[Any] = None
_COMBINED_OCR_WORKER_REQUIRED_ATTRS = (
    "find_numeric_debug_page_spans",
    "evaluate_page_character_noise",
)


def _blank_non_newlines(text: str) -> str:
    return "".join("\n" if ch == "\n" else " " for ch in text)


def _init_combined_ocr_worker() -> None:
    global _COMBINED_OCR_WORKER_NOISE_MOD, _WORD_REPEAT_RUST_MOD, _WORD_REPEAT_RUST_IMPORT_ATTEMPTED
    noise_mod = importlib.import_module("glossapi_rs_noise")
    missing = [
        attr for attr in _COMBINED_OCR_WORKER_REQUIRED_ATTRS if not hasattr(noise_mod, attr)
    ]
    if missing:
        raise ImportError(
            "glossapi_rs_noise missing required attrs for OCR worker: "
            + ", ".join(missing)
        )
    _COMBINED_OCR_WORKER_NOISE_MOD = noise_mod
    _WORD_REPEAT_RUST_IMPORT_ATTEMPTED = True
    _WORD_REPEAT_RUST_MOD = noise_mod if hasattr(noise_mod, "find_word_repeat_spans") else None


def _get_combined_ocr_worker_noise_mod() -> Any:
    global _COMBINED_OCR_WORKER_NOISE_MOD
    if _COMBINED_OCR_WORKER_NOISE_MOD is None:
        _init_combined_ocr_worker()
    return _COMBINED_OCR_WORKER_NOISE_MOD


def _prime_word_repeat_rust_module(module_name: str, module: Any) -> Any:
    global _WORD_REPEAT_RUST_MOD, _WORD_REPEAT_RUST_IMPORT_ATTEMPTED
    if module_name == "glossapi_rs_noise":
        _WORD_REPEAT_RUST_IMPORT_ATTEMPTED = True
        _WORD_REPEAT_RUST_MOD = module if hasattr(module, "find_word_repeat_spans") else None
    return module


def _can_use_combined_ocr_process_pool(noise_mod: Any, render_workers: int) -> bool:
    return (
        render_workers > 1
        and os.name != "nt"
        and getattr(noise_mod, "__name__", "") == "glossapi_rs_noise"
    )


def _default_combined_ocr_render_workers(
    *,
    noise_mod: Any,
    requested_workers: Optional[int],
    max_workers: int,
) -> int:
    if requested_workers is not None:
        return max(1, int(requested_workers))
    host_workers = max(1, int(max_workers))
    if _can_use_combined_ocr_process_pool(noise_mod, host_workers):
        return host_workers
    return min(4, host_workers)


@contextmanager
def _combined_ocr_process_pool_warning_ctx() -> Iterable[None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"This process .* is multi-threaded, use of fork\(\) may lead to deadlocks in the child\.",
            category=DeprecationWarning,
            module=r"multiprocessing\.popen_fork",
        )
        yield


def _blank_regex_matches_preserve_layout(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda match: _blank_non_newlines(match.group(0)), text)


def _filter_tables_preserve_layout(text: str) -> str:
    lowered = text.lower()
    if "<table" not in lowered and "|" not in text and "<tr" not in lowered and "<td" not in lowered and "<th" not in lowered:
        return text
    text = _blank_regex_matches_preserve_layout(text, HTML_TABLE_BLOCK_RE)
    kept: List[str] = []
    for segment in text.splitlines(keepends=True):
        trimmed = segment.strip()
        if trimmed and trimmed.startswith("|") and trimmed.endswith("|"):
            kept.append(_blank_non_newlines(segment))
            continue
        if trimmed and HTML_TABLE_LINE_RE.search(trimmed):
            kept.append(_blank_non_newlines(segment))
            continue
        kept.append(segment)
    return "".join(kept)


def _filter_latex_preserve_layout(text: str) -> str:
    if "$" not in text and "\\" not in text:
        return text
    for pattern in (
        LATEX_BEGIN_END_RE,
        LATEX_BLOCK_RE,
        LATEX_BRACKET_RE,
        LATEX_INLINE_PAREN_RE,
        LATEX_INLINE_DOLLAR_RE,
    ):
        text = _blank_regex_matches_preserve_layout(text, pattern)
    return text


def _blank_existing_match_regions_preserve_layout(text: str) -> str:
    if "<match" not in text:
        return text
    return _blank_regex_matches_preserve_layout(text, EXISTING_MATCH_BLOCK_RE)


def _blank_raw_spans_preserve_layout(text: str, spans: List[Dict[str, Any]]) -> str:
    if not spans:
        return text

    chars = list(text)
    for span in spans:
        start = max(0, int(span["start"]))
        end = min(len(chars), int(span["end"]))
        for idx in range(start, end):
            if chars[idx] != "\n":
                chars[idx] = " "
    return "".join(chars)


def _extract_latex_segments(text: str) -> List[Dict[str, Any]]:
    raw: List[Tuple[int, int, str, str]] = []
    for name, pattern in LATEX_SEGMENT_PATTERNS:
        for match in pattern.finditer(text):
            raw.append((match.start(), match.end(), name, match.group(0)))

    raw.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    segments: List[Dict[str, Any]] = []
    last_end = -1
    for start, end, kind, body in raw:
        if segments and start >= segments[-1]["start"] and end <= segments[-1]["end"]:
            continue
        if start < last_end:
            continue
        segments.append({"start": start, "end": end, "kind": kind, "text": body})
        last_end = end
    return segments


def _find_table_repeat_spans(page_text: str) -> List[Dict[str, Any]]:
    """Keep phase_clean's old call shape while table policy lives in ocr_table."""
    return _find_table_repeat_spans_impl(
        page_text,
        match_category=MATCH_CATEGORY_BY_TYPE["table_repeat"],
    )


def _normalize_latex_repeat_with_map(text: str) -> Tuple[str, List[int]]:
    normalized: List[str] = []
    raw_map: List[int] = []
    for raw_idx, ch in enumerate(text):
        if ch.isspace():
            continue
        normalized.append(ch.casefold())
        raw_map.append(raw_idx)
    return "".join(normalized), raw_map


def _normalize_latex_segment_exact(text: str) -> str:
    return "".join(ch.casefold() for ch in text if not ch.isspace())


def _normalize_latex_segment_skeleton(text: str) -> str:
    normalized = _normalize_latex_segment_exact(text)
    normalized = re.sub(r"\d+", "#", normalized)
    for command in LATEX_SYMBOL_SLOT_COMMANDS:
        normalized = normalized.replace(command.casefold(), r"\sym")
    normalized = re.sub(r"dr(?:_?\*|_?\\ast)?", "dr@", normalized)
    return normalized


def _is_short_latex_repeat_atom(raw_segment: str) -> bool:
    normalized = _normalize_latex_segment_exact(raw_segment)
    if len(normalized) > LATEX_SHORT_SEGMENT_MAX_NORM:
        return False
    command_tokens = LATEX_COMMAND_RE.findall(raw_segment)
    if not command_tokens:
        return False
    return set(command_tokens).issubset(LATEX_SHORT_REPEAT_ATOM_COMMANDS)


def _strip_latex_outer_delimiters(raw_segment: str) -> str:
    stripped = raw_segment.strip()
    wrappers = (
        (r"\(", r"\)"),
        (r"\[", r"\]"),
        ("$$", "$$"),
        ("$", "$"),
    )
    for left, right in wrappers:
        if stripped.startswith(left) and stripped.endswith(right) and len(stripped) >= len(left) + len(right):
            return stripped[len(left) : len(stripped) - len(right)].strip()
    return stripped


def _latex_short_atom_block_key(raw_segment: str) -> Optional[str]:
    body = "".join(ch for ch in _strip_latex_outer_delimiters(raw_segment) if not ch.isspace())
    if not body or len(body) > LATEX_SHORT_SEGMENT_MAX_NORM:
        return None

    plain_pattern = (
        r"^(?P<base>"
        + "|".join(re.escape(token) for token in sorted(LATEX_SHORT_ATOM_BLOCK_BASE_COMMANDS))
        + r")(?P<primes>\'+)?$"
    )
    match = re.fullmatch(plain_pattern, body)
    if match:
        base = match.group("base") or ""
        primes = match.group("primes") or ""
        return f"{base}{primes}"

    decorated_pattern = (
        r"^(?P<decorator>"
        + "|".join(re.escape(token) for token in sorted(LATEX_SHORT_ATOM_BLOCK_DECORATOR_COMMANDS))
        + r")\{(?P<base>"
        + "|".join(re.escape(token) for token in sorted(LATEX_SHORT_ATOM_BLOCK_BASE_COMMANDS))
        + r")\}(?P<primes>\'+)?$"
    )
    match = re.fullmatch(decorated_pattern, body)
    if match:
        decorator = match.group("decorator") or ""
        base = match.group("base") or ""
        primes = match.group("primes") or ""
        return f"{decorator}{{{base}}}{primes}"

    return None


def _consume_latex_short_atom_script(body: str, pos: int) -> Optional[int]:
    while pos < len(body) and body[pos] in "_^":
        pos += 1
        if pos >= len(body):
            return None
        if body[pos] == "{":
            end = body.find("}", pos + 1)
            if end == -1 or end == pos + 1:
                return None
            content = body[pos + 1 : end]
            if any(ch.isspace() for ch in content) or "{" in content or "}" in content:
                return None
            pos = end + 1
            continue
        if body[pos] == "\\":
            match = re.match(r"\\[A-Za-z]+", body[pos:])
            if match is None:
                return None
            pos += len(match.group(0))
            continue
        if body[pos].isalnum():
            pos += 1
            continue
        return None
    return pos


def _latex_short_atom_sequence_tokens(
    raw_segment: str,
    *,
    allow_truncated_tail: bool = False,
) -> Optional[List[str]]:
    body = "".join(ch for ch in _strip_latex_outer_delimiters(raw_segment) if not ch.isspace())
    if not body:
        return None

    base_commands = sorted(LATEX_SHORT_ATOM_BLOCK_BASE_COMMANDS, key=len, reverse=True)
    decorator_commands = sorted(LATEX_SHORT_ATOM_BLOCK_DECORATOR_COMMANDS, key=len, reverse=True)
    tokens: List[str] = []
    pos = 0
    while pos < len(body):
        token: Optional[str] = None
        for decorator in decorator_commands:
            prefix = decorator + "{"
            if not body.startswith(prefix, pos):
                continue
            inner_pos = pos + len(prefix)
            base = next((candidate for candidate in base_commands if body.startswith(candidate, inner_pos)), None)
            if base is None:
                continue
            end_pos = inner_pos + len(base)
            if end_pos >= len(body) or body[end_pos] != "}":
                continue
            token = f"{decorator}{{{base}}}"
            pos = end_pos + 1
            break

        if token is None:
            base = next((candidate for candidate in base_commands if body.startswith(candidate, pos)), None)
            if base is not None:
                token = base
                pos += len(base)

        if token is None:
            remaining = body[pos:]
            if allow_truncated_tail and tokens and len(remaining) >= 4 and any(command.startswith(remaining) for command in base_commands):
                break
            return None

        while pos < len(body) and body[pos] == "'":
            token += "'"
            pos += 1

        script_end = _consume_latex_short_atom_script(body, pos)
        if script_end is None:
            return None
        token += body[pos:script_end]
        pos = script_end

        while pos < len(body) and body[pos] == "'":
            token += "'"
            pos += 1

        tokens.append(token)

    return tokens or None


def _is_short_latex_whitelist_segment(raw_segment: str) -> bool:
    normalized = _normalize_latex_segment_exact(raw_segment)
    if len(normalized) > LATEX_SHORT_SEGMENT_MAX_NORM:
        return False
    tokens = _latex_short_atom_sequence_tokens(raw_segment)
    return tokens is not None and len(tokens) >= LATEX_SHORT_ATOM_EXACT_SEGMENT_MIN_TOKENS


def _is_latex_short_atom_chain_segment(raw_segment: str) -> bool:
    tokens = _latex_short_atom_sequence_tokens(raw_segment, allow_truncated_tail=True)
    if tokens is None or len(tokens) < LATEX_SHORT_ATOM_CHAIN_MIN_TOKENS:
        return False
    counts = Counter(tokens)
    return max(counts.values(), default=0) >= LATEX_SEGMENT_EXACT_RUN_MIN and len(counts) <= 3


def _is_suspicious_internal_latex_repeat(raw_segment: str) -> bool:
    if not raw_segment:
        return False
    if "<sub>" in raw_segment or "<sup>" in raw_segment:
        return True
    if _is_latex_short_atom_chain_segment(raw_segment):
        return True

    command_tokens = LATEX_COMMAND_RE.findall(raw_segment)
    if any(wrapper in raw_segment for wrapper in LATEX_TEXT_WRAPPER_MACROS):
        return len(command_tokens) >= 8 or len(raw_segment) >= 60

    counts = Counter(command_tokens)
    if set(command_tokens).issubset(LATEX_INTERNAL_SMALL_VOCAB_COMMANDS):
        if len(command_tokens) >= LATEX_INTERNAL_SMALL_VOCAB_RUN_MIN_COMMANDS and len(counts) <= 3:
            return True
    if any(command in LATEX_INTERNAL_REPEAT_COMMANDS for command in counts):
        return max(counts.values(), default=0) >= LATEX_INTERNAL_REPEAT_MIN_COMMAND_DUP

    return False


def _extract_latex_lhs_key(raw_segment: str) -> Optional[str]:
    normalized = _normalize_latex_segment_exact(raw_segment)
    if "=" not in normalized:
        return None
    lhs = normalized.split("=", 1)[0]
    return lhs or None


def _is_latex_symbol_inventory_segment(raw_segment: str) -> bool:
    normalized = _normalize_latex_segment_exact(raw_segment)
    if not normalized or len(normalized) > 96:
        return False
    if any(token in normalized for token in ("=", "+", "-", r"\sum", r"\prod", r"\int", r"\frac")):
        return False
    if _is_short_latex_repeat_atom(raw_segment):
        return False
    command_tokens = LATEX_COMMAND_RE.findall(raw_segment)
    return bool(command_tokens)


def _is_small_parameterized_definition_family(run: List[Dict[str, Any]]) -> bool:
    if len(run) > LATEX_SMALL_DEFINITION_FAMILY_MAX_RUN:
        return False
    lhs_keys = [_extract_latex_lhs_key(str(item["text"])) for item in run]
    if any(key is None for key in lhs_keys):
        return False
    if any(
        key is not None and any(token in key for token in (r"\frac", r"\sum", r"\prod", r"\int", "+", "-", "="))
        for key in lhs_keys
    ):
        return False
    return len(set(lhs_keys)) == len(lhs_keys)


def _is_symbol_inventory_run(run: List[Dict[str, Any]]) -> bool:
    return all(_is_latex_symbol_inventory_segment(str(item["text"])) for item in run)


def _short_atom_run_has_clean_gaps(page_text: str, run: List[Dict[str, Any]]) -> bool:
    if len(run) < 2:
        return True
    for left, right in zip(run, run[1:]):
        gap = page_text[int(left["end"]) : int(right["start"])]
        if any(ch.isalnum() for ch in gap):
            return False
    return True


def _extract_latex_numeric_slots(raw_segment: str) -> Optional[List[float]]:
    slots: List[float] = []
    for token in re.findall(r"[0-9]+(?:[.,/][0-9]+)*", raw_segment):
        if "/" in token:
            if token.count("/") != 1:
                return None
            lhs, rhs = token.split("/", 1)
            if not lhs.isdigit() or not rhs.isdigit() or int(rhs) == 0:
                return None
            slots.append(float(int(lhs) / int(rhs)))
            continue
        if token.count(".") + token.count(",") > 1:
            return None
        normalized = token.replace(",", ".", 1)
        if "." in normalized:
            lhs, rhs = normalized.split(".", 1)
            if not lhs.isdigit() or not rhs.isdigit():
                return None
            slots.append(float(normalized))
            continue
        if token.isdigit():
            slots.append(float(int(token)))
            continue
        return None
    return slots or None


def _latex_slot_progress_position(values: List[float]) -> bool:
    if len(values) < LATEX_SEGMENT_SLOT_PROGRESS_RUN_MIN:
        return False

    diffs: List[float] = []
    tolerance = 1e-9
    for left, right in zip(values, values[1:]):
        diff = right - left
        if diff < -tolerance:
            return False
        if diff > tolerance:
            diffs.append(diff)

    if not diffs:
        return False

    baseline = diffs[0]
    return all(abs(diff - baseline) <= max(tolerance, abs(baseline) * 1e-6) for diff in diffs[1:])


def _is_latex_slot_progression_run(run: List[Dict[str, Any]]) -> bool:
    if len(run) < LATEX_SEGMENT_SLOT_PROGRESS_RUN_MIN:
        return False
    if _is_small_parameterized_definition_family(run):
        return False
    if _is_symbol_inventory_run(run):
        return False
    if _is_short_latex_repeat_atom(str(run[0]["text"])):
        return False

    slot_lists = [item.get("numeric_slots") for item in run]
    if any(not slots for slots in slot_lists):
        return False
    slot_count = len(slot_lists[0] or [])
    if slot_count == 0 or any(len(slots or []) != slot_count for slots in slot_lists):
        return False

    varying_positions = 0
    for slot_idx in range(slot_count):
        values = [float(slots[slot_idx]) for slots in slot_lists if slots is not None]
        if len({round(value, 9) for value in values}) > 1:
            varying_positions += 1
    if varying_positions == 0 or varying_positions > 2:
        return False

    for slot_idx in range(slot_count):
        values = [float(slots[slot_idx]) for slots in slot_lists if slots is not None]
        if _latex_slot_progress_position(values):
            return True
    return False


def _normalize_alnum_with_map_skip_tags(text: str) -> Tuple[str, List[int]]:
    norm_chars: List[str] = []
    raw_char_indices: List[int] = []
    in_tag = False
    for raw_idx, ch in enumerate(text):
        if in_tag:
            if ch == ">":
                in_tag = False
            continue
        if ch == "<":
            in_tag = True
            continue
        folded = unicodedata.normalize("NFD", ch.casefold())
        for sub in folded:
            category = unicodedata.category(sub)
            if category.startswith("L") or category.startswith("N"):
                sub = WORD_CONFUSABLE_FOLD_MAP.get(sub, sub)
                norm_chars.append(sub)
                raw_char_indices.append(raw_idx)
    return "".join(norm_chars), raw_char_indices


def _normalize_hybrid_body(text: str) -> str:
    norm_chars: List[str] = []
    for ch in text:
        folded = unicodedata.normalize("NFD", ch.casefold())
        for sub in folded:
            category = unicodedata.category(sub)
            if category.startswith("L") or category.startswith("N"):
                norm_chars.append(WORD_CONFUSABLE_FOLD_MAP.get(sub, sub))
    return "".join(norm_chars)


def _classify_hybrid_numeric_field(prefix: str) -> Optional[Dict[str, Any]]:
    token = prefix.strip()
    if not token:
        return None

    trailing_paren = token.endswith(")")
    trailing_dot = token.endswith(".")
    stripped = token[:-1] if trailing_paren or trailing_dot else token
    if not stripped:
        return None

    if "/" in stripped:
        return {"field_kind": "numeric_value", "raw": token}

    parts = stripped.split(".")
    if not all(part.isdigit() for part in parts):
        return None

    numbers = [int(part) for part in parts]
    shape = ".".join("#" for _ in numbers)
    if trailing_paren:
        shape += ")"
    elif trailing_dot:
        shape += "."

    if trailing_paren or trailing_dot:
        field_kind = "header_counter"
    elif len(numbers) >= 3:
        field_kind = "header_counter"
    elif len(numbers) == 2 and len(parts[-1]) <= 2:
        field_kind = "header_counter"
    else:
        field_kind = "numeric_value"

    return {
        "field_kind": field_kind,
        "numbers": numbers,
        "shape": shape,
        "raw": token,
    }


def _classify_hybrid_inline_numeric_field(token: str) -> Optional[Dict[str, Any]]:
    stripped = token.strip()
    if not stripped:
        return None

    if re.fullmatch(r"[0-9]+", stripped):
        return {"field_kind": "numeric_value", "raw": stripped}

    if stripped.count("/") == 1:
        lhs, rhs = stripped.split("/", 1)
        if re.fullmatch(r"[0-9]+", lhs) and re.fullmatch(r"[0-9]+", rhs) and int(rhs) != 0:
            return {"field_kind": "numeric_value", "raw": stripped}
        return None

    decimal_candidate = stripped.replace(",", ".", 1)
    if decimal_candidate.count(".") == 1:
        lhs, rhs = decimal_candidate.split(".", 1)
        if re.fullmatch(r"[0-9]+", lhs) and re.fullmatch(r"[0-9]+", rhs):
            return {"field_kind": "numeric_value", "raw": stripped}

    return None


def _parse_hybrid_numeric_value(token: str) -> Optional[float]:
    stripped = token.strip()
    if not stripped:
        return None

    if re.fullmatch(r"[0-9]+", stripped):
        return float(int(stripped))

    if stripped.count("/") == 1:
        lhs, rhs = stripped.split("/", 1)
        if re.fullmatch(r"[0-9]+", lhs) and re.fullmatch(r"[0-9]+", rhs) and int(rhs) != 0:
            return float(int(lhs) / int(rhs))
        return None

    decimal_candidate = stripped.replace(",", ".", 1)
    if decimal_candidate.count(".") == 1:
        lhs, rhs = decimal_candidate.split(".", 1)
        if re.fullmatch(r"[0-9]+", lhs) and re.fullmatch(r"[0-9]+", rhs):
            return float(decimal_candidate)

    return None


def _prepare_hybrid_analysis_text(
    page_text: str,
    *,
    blocked_spans: List[Dict[str, Any]],
) -> str:
    analysis_text = _filter_tables_preserve_layout(page_text)
    analysis_text = _filter_latex_preserve_layout(analysis_text)
    analysis_text = _blank_existing_match_regions_preserve_layout(analysis_text)
    analysis_text = _blank_raw_spans_preserve_layout(analysis_text, blocked_spans)
    return analysis_text


def _extract_hybrid_numbered_items_from_analysis_text(analysis_text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for match in HYBRID_PREFIX_RE.finditer(analysis_text):
        field = _classify_hybrid_numeric_field(match.group("prefix"))
        if field is None:
            continue
        candidates.append(
            {
                "prefix_start": match.start("prefix"),
                "prefix_end": match.end("prefix"),
                **field,
            }
        )

    items: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        next_start = (
            int(candidates[idx + 1]["prefix_start"]) if idx + 1 < len(candidates) else len(analysis_text)
        )
        body_raw = analysis_text[int(candidate["prefix_end"]) : next_start].strip()
        if HYBRID_MARKUP_BODY_RE.search(body_raw):
            continue
        body_key = _normalize_hybrid_body(body_raw)
        has_alpha = any(ch.isalpha() for ch in body_key)
        if not has_alpha:
            continue
        body_is_full = len(body_key) >= HYBRID_REPEAT_MIN_BODY_ALNUM
        items.append(
            {
                "start": int(candidate["prefix_start"]),
                "end": next_start,
                "prefix_end": int(candidate["prefix_end"]),
                "field_kind": str(candidate["field_kind"]),
                "numbers": list(candidate.get("numbers", [])),
                "shape": str(candidate.get("shape", "")),
                "body_raw": body_raw,
                "body_key": body_key,
                "body_is_full": body_is_full,
            }
        )

    return items


def _extract_hybrid_inline_numeric_items_from_analysis_text(analysis_text: str) -> List[Dict[str, Any]]:
    clause_ranges: List[Tuple[int, int]] = []
    clause_start = 0
    for match in HYBRID_INLINE_CLAUSE_DELIMITER_RE.finditer(analysis_text):
        clause_ranges.append((clause_start, match.start()))
        clause_start = match.end()
    clause_ranges.append((clause_start, len(analysis_text)))

    items: List[Dict[str, Any]] = []
    for clause_index, (raw_start, raw_end) in enumerate(clause_ranges):
        clause = analysis_text[raw_start:raw_end]
        if not clause.strip():
            continue

        leading_ws = len(clause) - len(clause.lstrip())
        trailing_ws = len(clause) - len(clause.rstrip())
        clause_start_abs = raw_start + leading_ws
        clause_end_abs = raw_end - trailing_ws
        clause_text = analysis_text[clause_start_abs:clause_end_abs]
        if not clause_text or HYBRID_MARKUP_BODY_RE.search(clause_text):
            continue

        working_offset = clause_start_abs
        working_text = clause_text
        prefix_match = HYBRID_PREFIX_RE.match(working_text)
        if prefix_match:
            working_offset += prefix_match.end()
            working_text = working_text[prefix_match.end() :].lstrip()
            working_offset = clause_end_abs - len(working_text)
        if not working_text:
            continue

        tokens: List[Dict[str, Any]] = []
        numeric_token_positions: List[int] = []
        for match in HYBRID_INLINE_TOKEN_RE.finditer(working_text):
            token = match.group(0)
            abs_start = working_offset + match.start()
            abs_end = working_offset + match.end()
            if token and token[0].isdigit():
                numeric_info = _classify_hybrid_inline_numeric_field(token)
                if numeric_info is None:
                    continue
                parsed_value = _parse_hybrid_numeric_value(token)
                if parsed_value is None:
                    continue
                numeric_token_positions.append(len(tokens))
                tokens.append(
                    {
                        "kind": "numeric",
                        "start": abs_start,
                        "end": abs_end,
                        "raw": token,
                        "numeric_value": parsed_value,
                    }
                )
                continue
            token_key = _normalize_hybrid_body(token)
            if not token_key:
                continue
            tokens.append(
                {
                    "kind": "alpha",
                    "start": abs_start,
                    "end": abs_end,
                    "raw": token,
                    "token_key": token_key,
                }
            )

        if len(numeric_token_positions) != 1:
            continue

        numeric_pos = numeric_token_positions[0]
        numeric_token = tokens[numeric_pos]
        left_alpha = [token for token in tokens[:numeric_pos] if token.get("kind") == "alpha"]
        right_alpha = [token for token in tokens[numeric_pos + 1 :] if token.get("kind") == "alpha"]
        left_context = left_alpha[-HYBRID_INLINE_CONTEXT_WORDS:]
        right_context = right_alpha[:HYBRID_INLINE_CONTEXT_WORDS]
        alpha_word_count = len(left_context) + len(right_context)
        if alpha_word_count < HYBRID_INLINE_CONTEXT_MIN_ALPHA_WORDS:
            continue

        context_parts = [str(token.get("token_key", "")) for token in left_context]
        context_parts.append("num")
        context_parts.extend(str(token.get("token_key", "")) for token in right_context)
        context_key = _normalize_hybrid_body(" ".join(context_parts))
        if len(context_key) < HYBRID_INLINE_CONTEXT_MIN_CHARS:
            continue

        item_start = int(left_context[0]["start"]) if left_context else int(numeric_token["start"])
        item_end = int(right_context[-1]["end"]) if right_context else int(numeric_token["end"])
        items.append(
            {
                "start": item_start,
                "end": item_end,
                "clause_index": clause_index,
                "field_kind": "numeric_value",
                "inline_context_key": context_key,
                "numeric_value": float(numeric_token["numeric_value"]),
            }
        )

    return items


def _hybrid_partial_body_matches(candidate_body_key: str, target_body_key: str) -> bool:
    if not candidate_body_key or not target_body_key:
        return False
    if candidate_body_key == target_body_key:
        return False
    if not target_body_key.startswith(candidate_body_key):
        return False
    min_chars = min(4, len(target_body_key))
    min_ratio_chars = max(1, math.ceil(len(target_body_key) * 0.5))
    return len(candidate_body_key) >= min(min_chars, min_ratio_chars)


def _extend_hybrid_tail_span_end(
    items: List[Dict[str, Any]],
    *,
    run_start: int,
    run_end: int,
    expected_body_key: str,
) -> int:
    span_end = int(items[run_end - 1]["end"])
    if run_end >= len(items):
        return span_end

    tail = items[run_end]
    if tail.get("field_kind") != "header_counter":
        return span_end
    if str(tail.get("shape", "")) != str(items[run_start].get("shape", "")):
        return span_end
    if not _hybrid_header_progresses(items[run_end - 1], tail):
        return span_end
    if not _hybrid_partial_body_matches(str(tail.get("body_key", "")), expected_body_key):
        return span_end
    return int(tail["end"])


def _hybrid_header_progresses(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    if previous.get("field_kind") != "header_counter" or current.get("field_kind") != "header_counter":
        return False
    prev_numbers = list(previous.get("numbers", []))
    curr_numbers = list(current.get("numbers", []))
    if len(prev_numbers) != len(curr_numbers) or not prev_numbers:
        return False
    return prev_numbers[:-1] == curr_numbers[:-1] and curr_numbers[-1] == prev_numbers[-1] + 1


def _hybrid_header_is_parent(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    if previous.get("field_kind") != "header_counter" or current.get("field_kind") != "header_counter":
        return False
    prev_numbers = list(previous.get("numbers", []))
    curr_numbers = list(current.get("numbers", []))
    if not prev_numbers or len(prev_numbers) + 1 != len(curr_numbers):
        return False
    return curr_numbers[:-1] == prev_numbers


def _hybrid_inline_step(previous: Dict[str, Any], current: Dict[str, Any]) -> Optional[float]:
    if previous.get("field_kind") != "numeric_value" or current.get("field_kind") != "numeric_value":
        return None
    if int(current.get("clause_index", -1)) != int(previous.get("clause_index", -1)) + 1:
        return None
    if str(previous.get("inline_context_key", "")) != str(current.get("inline_context_key", "")):
        return None

    previous_value = float(previous.get("numeric_value", 0.0))
    current_value = float(current.get("numeric_value", 0.0))
    step = current_value - previous_value
    if step <= 0:
        return None
    return step


def _hybrid_inline_step_matches(expected_step: float, actual_step: float) -> bool:
    tolerance = max(1e-9, abs(expected_step) * 1e-6)
    return abs(expected_step - actual_step) <= tolerance


def _find_hybrid_same_body_progression_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(items):
        item = items[idx]
        if item.get("field_kind") != "header_counter" or not bool(item.get("body_is_full")):
            idx += 1
            continue

        end_idx = idx + 1
        while (
            end_idx < len(items)
            and items[end_idx].get("field_kind") == "header_counter"
            and bool(items[end_idx].get("body_is_full"))
            and str(items[end_idx].get("body_key", "")) == str(item.get("body_key", ""))
            and str(items[end_idx].get("shape", "")) == str(item.get("shape", ""))
            and _hybrid_header_progresses(items[end_idx - 1], items[end_idx])
        ):
            end_idx += 1

        run_length = end_idx - idx
        if run_length >= HYBRID_REPEAT_MIN_ITEMS:
            start_idx = idx
            if idx > 0:
                previous = items[idx - 1]
                if (
                    bool(previous.get("body_is_full"))
                    and
                    str(previous.get("body_key", "")) == str(item.get("body_key", ""))
                    and _hybrid_header_is_parent(previous, item)
                ):
                    start_idx = idx - 1

            span_end = _extend_hybrid_tail_span_end(
                items,
                run_start=idx,
                run_end=end_idx,
                expected_body_key=str(item.get("body_key", "")),
            )
            spans.append(
                {
                    "start": int(items[start_idx]["start"]),
                    "end": span_end,
                    "match_types": ["hybrid_repeat"],
                    "category": MATCH_CATEGORY_BY_TYPE["hybrid_repeat"],
                    "kind": "same_body_progression",
                    "item_count": end_idx - start_idx,
                }
            )
            idx = end_idx
            continue

        idx += 1

    return spans


def _find_hybrid_cycle_progression_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    n_items = len(items)
    for cycle_len in range(2, HYBRID_REPEAT_MAX_CYCLE + 1):
        idx = 0
        while idx + 2 * cycle_len <= n_items:
            run = items[idx : idx + 2 * cycle_len]
            if any(item.get("field_kind") != "header_counter" or not bool(item.get("body_is_full")) for item in run):
                idx += 1
                continue
            shapes = {str(item.get("shape", "")) for item in run}
            if len(shapes) != 1:
                idx += 1
                continue
            if not all(_hybrid_header_progresses(run[pos - 1], run[pos]) for pos in range(1, len(run))):
                idx += 1
                continue

            template = [str(item.get("body_key", "")) for item in run[:cycle_len]]
            if len(set(template)) < 2:
                idx += 1
                continue

            if any(str(run[pos].get("body_key", "")) != template[pos % cycle_len] for pos in range(cycle_len, len(run))):
                idx += 1
                continue

            end_idx = idx + 2 * cycle_len
            while (
                end_idx < n_items
                and items[end_idx].get("field_kind") == "header_counter"
                and bool(items[end_idx].get("body_is_full"))
                and str(items[end_idx].get("shape", "")) == str(items[idx].get("shape", ""))
                and _hybrid_header_progresses(items[end_idx - 1], items[end_idx])
                and str(items[end_idx].get("body_key", "")) == template[(end_idx - idx) % cycle_len]
            ):
                end_idx += 1

            item_count = end_idx - idx
            if item_count >= HYBRID_REPEAT_MIN_CYCLE_ITEMS:
                span_end = _extend_hybrid_tail_span_end(
                    items,
                    run_start=idx,
                    run_end=end_idx,
                    expected_body_key=template[(end_idx - idx) % cycle_len],
                )
                spans.append(
                    {
                        "start": int(items[idx]["start"]),
                        "end": span_end,
                        "match_types": ["hybrid_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["hybrid_repeat"],
                        "kind": "body_cycle_progression",
                        "item_count": item_count,
                        "cycle_len": cycle_len,
                    }
                )
                idx = end_idx
                continue

            idx += 1

    return spans


def _find_hybrid_inline_progression_spans(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    idx = 0
    while idx + HYBRID_INLINE_REPEAT_MIN_ITEMS <= len(items):
        first = items[idx]
        second = items[idx + 1]
        expected_step = _hybrid_inline_step(first, second)
        if expected_step is None:
            idx += 1
            continue

        end_idx = idx + 2
        while end_idx < len(items):
            actual_step = _hybrid_inline_step(items[end_idx - 1], items[end_idx])
            if actual_step is None or not _hybrid_inline_step_matches(expected_step, actual_step):
                break
            end_idx += 1

        item_count = end_idx - idx
        if item_count >= HYBRID_INLINE_REPEAT_MIN_ITEMS:
            spans.append(
                {
                    "start": int(items[idx]["start"]),
                    "end": int(items[end_idx - 1]["end"]),
                    "match_types": ["hybrid_repeat"],
                    "category": MATCH_CATEGORY_BY_TYPE["hybrid_repeat"],
                    "kind": "inline_numeric_progression",
                    "item_count": item_count,
                }
            )
            idx = end_idx
            continue

        idx += 1

    return spans


def _find_hybrid_numbered_repeat_spans(
    page_text: str,
    *,
    blocked_spans: List[Dict[str, Any]],
    analysis_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not any(ch.isdigit() for ch in page_text):
        return []
    if analysis_text is None:
        analysis_text = _prepare_hybrid_analysis_text(page_text, blocked_spans=blocked_spans)
    else:
        analysis_text = _blank_raw_spans_preserve_layout(analysis_text, blocked_spans)
    rust_mod = _get_word_repeat_rust_module()
    if rust_mod is not None and hasattr(rust_mod, "find_hybrid_repeat_spans"):
        return [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "match_types": list(item["match_types"]),
                "category": str(item["category"]),
                "kind": str(item["kind"]),
                "item_count": int(item["item_count"]),
                **({"cycle_len": int(item["cycle_len"])} if "cycle_len" in item else {}),
            }
            for item in rust_mod.find_hybrid_repeat_spans(analysis_text)
        ]
    items = _extract_hybrid_numbered_items_from_analysis_text(analysis_text)
    spans = _find_hybrid_same_body_progression_spans(items)
    spans.extend(_find_hybrid_cycle_progression_spans(items))
    inline_items = _extract_hybrid_inline_numeric_items_from_analysis_text(analysis_text)
    spans.extend(_find_hybrid_inline_progression_spans(inline_items))
    spans.sort(key=lambda item: (int(item["start"]), -(int(item["end"]) - int(item["start"]))))

    deduped: List[Dict[str, Any]] = []
    for span in spans:
        if deduped and int(span["start"]) >= int(deduped[-1]["start"]) and int(span["end"]) <= int(deduped[-1]["end"]):
            continue
        deduped.append(span)
    return deduped


def _build_word_repeat_hash(text: str) -> Tuple[List[int], List[int]]:
    pref = [0] * (len(text) + 1)
    pw = [1] * (len(text) + 1)
    for idx, ch in enumerate(text):
        code = ord(ch) + 1
        pref[idx + 1] = (pref[idx] * WORD_REPEAT_HASH_BASE + code) & WORD_REPEAT_HASH_MASK
        pw[idx + 1] = (pw[idx] * WORD_REPEAT_HASH_BASE) & WORD_REPEAT_HASH_MASK
    return pref, pw


def _word_repeat_hash_slice(pref: List[int], pw: List[int], start: int, end: int) -> int:
    return (pref[end] - ((pref[start] * pw[end - start]) & WORD_REPEAT_HASH_MASK)) & WORD_REPEAT_HASH_MASK


def _word_repeat_blocks_equal(
    text: str,
    pref: List[int],
    pw: List[int],
    lhs: int,
    rhs: int,
    period: int,
) -> bool:
    return (
        _word_repeat_hash_slice(pref, pw, lhs, lhs + period)
        == _word_repeat_hash_slice(pref, pw, rhs, rhs + period)
        and text[lhs : lhs + period] == text[rhs : rhs + period]
    )


def _get_word_repeat_rust_module() -> Optional[Any]:
    global _WORD_REPEAT_RUST_MOD, _WORD_REPEAT_RUST_IMPORT_ATTEMPTED
    if _WORD_REPEAT_RUST_IMPORT_ATTEMPTED:
        return _WORD_REPEAT_RUST_MOD
    _WORD_REPEAT_RUST_IMPORT_ATTEMPTED = True
    try:
        module = importlib.import_module("glossapi_rs_noise")
    except Exception:
        _WORD_REPEAT_RUST_MOD = None
        return None
    if hasattr(module, "find_word_repeat_spans"):
        _WORD_REPEAT_RUST_MOD = module
    else:
        _WORD_REPEAT_RUST_MOD = None
    return _WORD_REPEAT_RUST_MOD


def _find_word_repeat_spans_python(
    normalized_text: str,
    *,
    rep_threshold: int,
    min_period: int,
    window: int,
) -> List[Dict[str, int]]:
    n_chars = len(normalized_text)
    if n_chars < rep_threshold * min_period:
        return []

    pref, pw = _build_word_repeat_hash(normalized_text)
    max_period = min(max(min_period, window // rep_threshold), n_chars // rep_threshold)
    spans: List[Dict[str, int]] = []

    for period in range(min_period, max_period + 1):
        idx = 0
        while idx + rep_threshold * period <= n_chars:
            is_repeat = True
            for multiple in range(1, rep_threshold):
                if not _word_repeat_blocks_equal(
                    normalized_text,
                    pref,
                    pw,
                    idx,
                    idx + multiple * period,
                    period,
                ):
                    is_repeat = False
                    break
            if not is_repeat:
                idx += 1
                continue

            left = idx
            while left - period >= 0 and _word_repeat_blocks_equal(
                normalized_text,
                pref,
                pw,
                left - period,
                left,
                period,
            ):
                left -= period

            right = idx + rep_threshold * period
            while right + period <= n_chars and _word_repeat_blocks_equal(
                normalized_text,
                pref,
                pw,
                right - period,
                right,
                period,
            ):
                right += period

            pattern = normalized_text[left : left + period]
            tail_chars = 0
            while (
                right + tail_chars < n_chars
                and tail_chars < period
                and normalized_text[right + tail_chars] == pattern[tail_chars]
            ):
                tail_chars += 1

            spans.append(
                {
                    "start": left,
                    "end": right + tail_chars,
                    "period": period,
                    "repetitions": (right - left) // period,
                    "tail_chars": tail_chars,
                }
            )
            idx = right

    spans.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["period"]))
    deduped: List[Dict[str, int]] = []
    for span in spans:
        if deduped and span["start"] >= deduped[-1]["start"] and span["end"] <= deduped[-1]["end"]:
            continue
        deduped.append(span)
    return deduped


def _find_word_repeat_spans(
    normalized_text: str,
    *,
    rep_threshold: int,
    min_period: int,
    window: int,
) -> List[Dict[str, int]]:
    rust_mod = _get_word_repeat_rust_module()
    if rust_mod is None:
        return _find_word_repeat_spans_python(
            normalized_text,
            rep_threshold=rep_threshold,
            min_period=min_period,
            window=window,
        )
    return [
        {
            "start": int(item["start"]),
            "end": int(item["end"]),
            "period": int(item["period"]),
            "repetitions": int(item["repetitions"]),
            "tail_chars": int(item["tail_chars"]),
        }
        for item in rust_mod.find_word_repeat_spans(
            normalized_text,
            int(rep_threshold),
            int(min_period),
            int(window),
        )
    ]


def _gap_has_fewer_than_n_nonwhitespace_chars(text: str, start: int, end: int, limit: int) -> bool:
    if start >= end:
        return True
    count = 0
    for ch in text[start:end]:
        if not ch.isspace():
            count += 1
            if count >= limit:
                return False
    return True


def _latex_segments_are_local(page_text: str, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return _gap_has_fewer_than_n_nonwhitespace_chars(
        page_text,
        int(left["end"]),
        int(right["start"]),
        LATEX_SEGMENT_LOCAL_NONWHITESPACE_GAP,
    )


def _latex_local_groups(page_text: str, segments: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not segments:
        return []

    groups: List[List[Dict[str, Any]]] = [[segments[0]]]
    for segment in segments[1:]:
        if _latex_segments_are_local(page_text, groups[-1][-1], segment):
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def _find_local_latex_segment_block_spans(
    page_text: str,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labeled_spans: List[Dict[str, Any]] = []
    for group in _latex_local_groups(page_text, segments):
        if len(group) < LATEX_SEGMENT_EXACT_RUN_MIN:
            continue

        idx = 0
        while idx < len(group):
            exact_key = str(group[idx]["exact_key"])
            end_idx = idx + 1
            while end_idx < len(group) and str(group[end_idx]["exact_key"]) == exact_key:
                end_idx += 1

            run_length = end_idx - idx
            exact_run = group[idx:end_idx]
            is_short_repeat_atom = _is_short_latex_repeat_atom(str(group[idx]["text"]))
            is_short_whitelist_segment = _is_short_latex_whitelist_segment(str(group[idx]["text"]))
            if run_length >= LATEX_SEGMENT_EXACT_RUN_MIN and (
                len(exact_key) >= LATEX_LONG_SEGMENT_MIN_NORM
                or (is_short_repeat_atom and _short_atom_run_has_clean_gaps(page_text, exact_run))
                or (is_short_whitelist_segment and _short_atom_run_has_clean_gaps(page_text, exact_run))
            ):
                span = {
                    "start": int(exact_run[0]["start"]),
                    "end": int(exact_run[-1]["end"]),
                    "match_types": ["latex_repeat"],
                    "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                }
                if is_short_whitelist_segment and not is_short_repeat_atom:
                    span["kind"] = "short_atom_segment_repeat"
                    span["item_count"] = len(exact_run)
                labeled_spans.append(span)
            idx = end_idx

        idx = 0
        while idx < len(group):
            skeleton_key = str(group[idx]["skeleton_key"])
            end_idx = idx + 1
            while end_idx < len(group) and str(group[end_idx]["skeleton_key"]) == skeleton_key:
                end_idx += 1

            run = group[idx:end_idx]
            exact_vocab = {str(item["exact_key"]) for item in run}
            if (
                len(run) >= LATEX_SEGMENT_SKELETON_RUN_MIN
                and len(skeleton_key) >= LATEX_LONG_SEGMENT_MIN_NORM
                and not _is_short_latex_repeat_atom(str(run[0]["text"]))
                and len(exact_vocab) >= 2
                and not _is_small_parameterized_definition_family(run)
                and not _is_symbol_inventory_run(run)
            ):
                labeled_spans.append(
                    {
                        "start": int(run[0]["start"]),
                        "end": int(run[-1]["end"]),
                        "match_types": ["latex_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                    }
                )
            idx = end_idx

        exact_sequence = [str(item["exact_key"]) for item in group]
        exact_counts = Counter(exact_sequence)
        if (
            len(group) >= LATEX_SEGMENT_ALTERNATING_RUN_MIN
            and len(exact_counts) <= 2
            and min(exact_counts.values()) >= 2
        ):
            avg_length = sum(len(item) for item in exact_sequence) / len(exact_sequence)
            if avg_length >= LATEX_LONG_SEGMENT_MIN_NORM and not all(
                _is_short_latex_repeat_atom(str(item["text"])) for item in group
            ):
                labeled_spans.append(
                    {
                        "start": int(group[0]["start"]),
                        "end": int(group[-1]["end"]),
                        "match_types": ["latex_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                    }
                )

    return labeled_spans


def _find_short_atom_block_repeat_bounds(
    atom_keys: List[str],
) -> Optional[Tuple[int, int, int, int]]:
    n_items = len(atom_keys)
    if n_items < LATEX_SHORT_ATOM_BLOCK_REPEAT_MIN_ITEMS:
        return None

    best: Optional[Tuple[int, int, int, int]] = None
    for period in range(n_items // 2, 1, -1):
        for start in range(0, n_items - (2 * period) + 1):
            pattern = atom_keys[start : start + period]
            if atom_keys[start + period : start + (2 * period)] != pattern:
                continue
            if len(set(pattern)) < 2:
                continue

            left = start
            while left - period >= 0 and atom_keys[left - period : left] == pattern:
                left -= period

            right = start + (2 * period)
            while right + period <= n_items and atom_keys[right : right + period] == pattern:
                right += period

            repeated_items = right - left
            repetitions = repeated_items // period
            if repeated_items < LATEX_SHORT_ATOM_BLOCK_REPEAT_MIN_ITEMS or repetitions < 2:
                continue

            candidate = (left, right, period, repetitions)
            if best is None:
                best = candidate
                continue

            best_span_len = best[1] - best[0]
            candidate_span_len = candidate[1] - candidate[0]
            if candidate_span_len > best_span_len:
                best = candidate
                continue
            if candidate_span_len == best_span_len and candidate[2] > best[2]:
                best = candidate
    return best


def _extend_latex_short_atom_block_partial_tail(
    page_text: str,
    run: List[Dict[str, Any]],
    repeated_bounds: Tuple[int, int, int, int],
) -> int:
    if not run:
        return 0

    left, _, period, _ = repeated_bounds
    if period <= 0 or left >= len(run):
        return int(run[-1]["end"])

    expected_idx = left + ((len(run) - left) % period)
    if expected_idx >= len(run):
        return int(run[-1]["end"])

    expected_text = str(run[expected_idx]["text"])
    segment_end = int(run[-1]["end"])
    cursor = segment_end
    while cursor < len(page_text) and page_text[cursor].isspace():
        cursor += 1
    if cursor >= len(page_text):
        return segment_end

    prefix_len = 0
    while (
        cursor + prefix_len < len(page_text)
        and prefix_len < len(expected_text)
        and page_text[cursor + prefix_len] == expected_text[prefix_len]
    ):
        prefix_len += 1

    if prefix_len == 0 or prefix_len >= len(expected_text):
        return segment_end
    return cursor + prefix_len


def _find_local_latex_short_atom_block_spans(
    page_text: str,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labeled_spans: List[Dict[str, Any]] = []
    for group in _latex_local_groups(page_text, segments):
        idx = 0
        while idx < len(group):
            if not group[idx].get("short_atom_block_key"):
                idx += 1
                continue

            end_idx = idx + 1
            while end_idx < len(group) and group[end_idx].get("short_atom_block_key"):
                end_idx += 1

            run = group[idx:end_idx]
            atom_keys = [str(item["short_atom_block_key"]) for item in run]
            repeated_bounds = _find_short_atom_block_repeat_bounds(atom_keys)
            if repeated_bounds is not None:
                _, _, period_items, repetitions = repeated_bounds
                span_end = _extend_latex_short_atom_block_partial_tail(page_text, run, repeated_bounds)
                labeled_spans.append(
                    {
                        "start": int(run[0]["start"]),
                        "end": int(span_end),
                        "match_types": ["latex_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                        "kind": "short_atom_block_repeat",
                        "item_count": len(run),
                        "period_items": int(period_items),
                        "repetitions": int(repetitions),
                    }
                )

            idx = end_idx
    return labeled_spans


def _find_raw_latex_small_vocab_command_spans(page_text: str) -> List[Dict[str, Any]]:
    labeled_spans: List[Dict[str, Any]] = []
    command_matches = list(LATEX_COMMAND_RE.finditer(page_text))
    run_start: Optional[int] = None
    run_end: Optional[int] = None
    run_commands: List[str] = []
    previous_end = 0

    def flush_run() -> None:
        if run_start is None or run_end is None or not run_commands:
            return
        counts = Counter(run_commands)
        if (
            len(run_commands) >= LATEX_INTERNAL_SMALL_VOCAB_RUN_MIN_COMMANDS
            and len(counts) <= 3
            and max(counts.values(), default=0) >= LATEX_SEGMENT_EXACT_RUN_MIN
        ):
            labeled_spans.append(
                {
                    "start": run_start,
                    "end": run_end,
                    "match_types": ["latex_repeat"],
                    "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                    "kind": "internal_small_vocab_command_run",
                    "item_count": len(run_commands),
                }
            )

    for command_match in command_matches:
        command = command_match.group(0)
        gap = page_text[previous_end : command_match.start()]
        can_extend_run = not any(ch.isalnum() for ch in gap)
        if command in LATEX_INTERNAL_SMALL_VOCAB_COMMANDS and (not run_commands or can_extend_run):
            if not run_commands:
                run_start = command_match.start()
            run_end = command_match.end()
            run_commands.append(command)
        else:
            flush_run()
            run_start = None
            run_end = None
            run_commands = []
            if command in LATEX_INTERNAL_SMALL_VOCAB_COMMANDS:
                run_start = command_match.start()
                run_end = command_match.end()
                run_commands = [command]
        previous_end = command_match.end()
    flush_run()

    return labeled_spans


def _find_internal_latex_small_vocab_command_spans(
    page_text: str,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labeled_spans: List[Dict[str, Any]] = []
    for segment in segments:
        raw_text = str(segment["text"])
        command_matches = list(LATEX_COMMAND_RE.finditer(raw_text))
        run_start: Optional[int] = None
        run_end: Optional[int] = None
        run_commands: List[str] = []
        previous_end = 0

        def flush_run() -> None:
            if run_start is None or run_end is None or not run_commands:
                return
            counts = Counter(run_commands)
            if (
                len(run_commands) >= LATEX_INTERNAL_SMALL_VOCAB_RUN_MIN_COMMANDS
                and len(counts) <= 3
                and max(counts.values(), default=0) >= LATEX_SEGMENT_EXACT_RUN_MIN
            ):
                labeled_spans.append(
                    {
                        "start": int(segment["start"]) + run_start,
                        "end": int(segment["start"]) + run_end,
                        "match_types": ["latex_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                        "kind": "internal_small_vocab_command_run",
                        "item_count": len(run_commands),
                    }
                )

        for command_match in command_matches:
            command = command_match.group(0)
            gap = raw_text[previous_end : command_match.start()]
            can_extend_run = not any(ch.isalnum() for ch in gap)
            if command in LATEX_INTERNAL_SMALL_VOCAB_COMMANDS and (not run_commands or can_extend_run):
                if not run_commands:
                    run_start = command_match.start()
                run_end = command_match.end()
                run_commands.append(command)
            else:
                flush_run()
                run_start = None
                run_end = None
                run_commands = []
                if command in LATEX_INTERNAL_SMALL_VOCAB_COMMANDS:
                    run_start = command_match.start()
                    run_end = command_match.end()
                    run_commands = [command]
            previous_end = command_match.end()
        flush_run()

    return labeled_spans


def _find_local_latex_slot_progression_spans(
    page_text: str,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    labeled_spans: List[Dict[str, Any]] = []
    for group in _latex_local_groups(page_text, segments):
        if len(group) < LATEX_SEGMENT_SLOT_PROGRESS_RUN_MIN:
            continue

        idx = 0
        while idx < len(group):
            skeleton_key = str(group[idx]["skeleton_key"])
            end_idx = idx + 1
            while end_idx < len(group) and str(group[end_idx]["skeleton_key"]) == skeleton_key:
                end_idx += 1

            run = group[idx:end_idx]
            exact_vocab = {str(item["exact_key"]) for item in run}
            if (
                len(run) >= LATEX_SEGMENT_SLOT_PROGRESS_RUN_MIN
                and len(skeleton_key) >= LATEX_LONG_SEGMENT_MIN_NORM
                and len(exact_vocab) >= 2
                and _is_latex_slot_progression_run(run)
            ):
                labeled_spans.append(
                    {
                        "start": int(run[0]["start"]),
                        "end": int(run[-1]["end"]),
                        "match_types": ["latex_repeat"],
                        "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                        "kind": "slot_progression",
                        "item_count": len(run),
                    }
                )
            idx = end_idx

    return labeled_spans


def _find_latex_repeat_spans(
    page_text: str,
    *,
    blocked_spans: List[Dict[str, Any]],
    rep_threshold: int,
    min_period: int,
    window: int,
    analysis_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if analysis_text is None:
        analysis_text = _filter_tables_preserve_layout(page_text)
        analysis_text = _blank_existing_match_regions_preserve_layout(analysis_text)
    if (
        "$" not in analysis_text
        and "\\" not in analysis_text
        and "<sub>" not in analysis_text
        and "<sup>" not in analysis_text
    ):
        return []
    analysis_text = _blank_raw_spans_preserve_layout(analysis_text, blocked_spans)

    labeled_spans: List[Dict[str, Any]] = []

    for wrapper_pattern in (LATEX_TEXT_WRAPPER_BODY_RE, LATEX_TEXT_WRAPPER_OPEN_BODY_RE):
        for match in wrapper_pattern.finditer(analysis_text):
            body = match.group(1)
            command_tokens = LATEX_COMMAND_RE.findall(body)
            if len(command_tokens) < 16:
                continue
            if len(set(command_tokens)) > 4:
                continue
            labeled_spans.append(
                {
                    "start": match.start(1),
                    "end": match.end(1),
                    "match_types": ["latex_repeat"],
                    "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
                }
            )

    for match in HTML_MATH_MARKUP_CLUSTER_RE.finditer(analysis_text):
        labeled_spans.append(
            {
                "start": match.start(),
                "end": match.end(),
                "match_types": ["latex_repeat"],
                "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
            }
        )

    labeled_spans.extend(_find_raw_latex_small_vocab_command_spans(analysis_text))

    segments = _extract_latex_segments(analysis_text)
    for segment in segments:
        raw_text = str(segment["text"])
        segment["exact_key"] = _normalize_latex_segment_exact(raw_text)
        segment["skeleton_key"] = _normalize_latex_segment_skeleton(raw_text)
        segment["short_atom_block_key"] = _latex_short_atom_block_key(raw_text)

    labeled_spans.extend(_find_local_latex_segment_block_spans(page_text, segments))
    labeled_spans.extend(_find_local_latex_short_atom_block_spans(page_text, segments))
    labeled_spans.extend(_find_internal_latex_small_vocab_command_spans(page_text, segments))

    for segment in segments:
        normalized_text, raw_map = _normalize_latex_repeat_with_map(segment["text"])
        normalized_spans = _find_word_repeat_spans(
            normalized_text,
            rep_threshold=rep_threshold,
            min_period=min_period,
            window=window,
        )
        for span in normalized_spans:
            if span["end"] <= span["start"] or span["start"] >= len(raw_map):
                continue
            start = segment["start"] + raw_map[span["start"]]
            end = segment["start"] + raw_map[span["end"] - 1] + 1
            raw_span = page_text[start:end]
            if not _is_suspicious_internal_latex_repeat(raw_span):
                continue
            labeled_span = {
                "start": start,
                "end": end,
                "period": span["period"],
                "repetitions": span["repetitions"],
                "tail_chars": span["tail_chars"],
                "match_types": ["latex_repeat"],
                "category": MATCH_CATEGORY_BY_TYPE["latex_repeat"],
            }
            if _is_latex_short_atom_chain_segment(raw_span):
                labeled_span["kind"] = "short_atom_chain_segment"
            labeled_spans.append(labeled_span)
    return labeled_spans


def _find_latex_slot_progression_spans(
    page_text: str,
    *,
    blocked_spans: List[Dict[str, Any]],
    analysis_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if analysis_text is None:
        analysis_text = _filter_tables_preserve_layout(page_text)
        analysis_text = _blank_existing_match_regions_preserve_layout(analysis_text)
    analysis_text = _blank_raw_spans_preserve_layout(analysis_text, blocked_spans)

    segments = _extract_latex_segments(analysis_text)
    for segment in segments:
        raw_text = str(segment["text"])
        segment["exact_key"] = _normalize_latex_segment_exact(raw_text)
        segment["skeleton_key"] = _normalize_latex_segment_skeleton(raw_text)
        segment["numeric_slots"] = _extract_latex_numeric_slots(raw_text)

    return _find_local_latex_slot_progression_spans(page_text, segments)


def _shared_repeat_match_type(segment: str) -> Optional[str]:
    if not segment:
        return None
    has_letter = any(ch.isalpha() for ch in segment)
    has_digit = any(ch.isdigit() for ch in segment)
    if has_letter:
        return "word_repeat"
    if has_digit:
        return "numeric_repeat"
    return None


def _count_hybrid_matches_in_page(page_text: str, spans: List[Dict[str, Any]]) -> int:
    merged_spans = _merge_labeled_raw_spans(page_text, spans)
    return sum(1 for span in merged_spans if span.get("category") == "hybrid")


def _build_token_category_page_metric_row(
    page_row: Dict[str, Any],
    matches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    category_counts: Counter[str] = Counter()
    pattern_family_counts: Counter[str] = Counter()
    for match in matches:
        for category in list(match.get("categories") or []):
            category_counts[str(category)] += 1
        for family in list(match.get("pattern_families") or []):
            pattern_family_counts[str(family)] += 1

    page_char_count = int(page_row.get("page_char_count", 0) or 0)
    match_count = int(page_row.get("match_count", 0) or 0)
    return {
        "source_path": str(page_row.get("source_path", "")),
        "source_stem": str(page_row.get("source_stem", "")),
        "base_stem": str(page_row.get("base_stem", "")),
        "debug_output_path": str(page_row.get("output_path", "")),
        "page_kind": str(page_row.get("page_kind", "")),
        "page_number": int(page_row.get("page_number", 0) or 0),
        "page_index_in_file": int(page_row.get("page_index_in_file", 0) or 0),
        "page_char_count": page_char_count,
        "match_count": match_count,
        "match_density_per_1k_chars": (
            float(match_count) * 1000.0 / float(page_char_count)
            if page_char_count > 0
            else 0.0
        ),
        "match_categories": str(page_row.get("match_categories", "")),
        "match_pattern_families": str(page_row.get("match_pattern_families", "")),
        "category_match_counts": dict(category_counts),
        "pattern_family_match_counts": dict(pattern_family_counts),
    }


def _build_token_category_match_index_rows(
    page_text: str,
    matches: List[Dict[str, Any]],
    *,
    page_row: Dict[str, Any],
    context_window_chars: int = 240,
) -> List[Dict[str, Any]]:
    if not matches:
        return []

    byte_offsets = _utf8_prefix_byte_offsets(page_text)
    rows: List[Dict[str, Any]] = []
    source_stem = str(page_row.get("source_stem", ""))
    page_kind = str(page_row.get("page_kind", ""))
    page_number = int(page_row.get("page_number", 0) or 0)
    page_index_in_file = int(page_row.get("page_index_in_file", 0) or 0)
    page_char_count = int(page_row.get("page_char_count", 0) or 0)
    output_path = str(page_row.get("output_path", ""))
    for fallback_index, match in enumerate(matches, start=1):
        start = int(match.get("start", 0) or 0)
        end = int(match.get("end", 0) or 0)
        if start < 0 or end < start or end > len(page_text):
            continue
        match_index = int(match.get("match_index_in_page", fallback_index) or fallback_index)
        categories = [str(item) for item in list(match.get("categories") or []) if str(item)]
        pattern_families = [
            str(item) for item in list(match.get("pattern_families") or []) if str(item)
        ]
        excerpt_start = max(0, start - int(context_window_chars))
        excerpt_end = min(len(page_text), end + int(context_window_chars))
        rows.append(
            {
                "match_id": f"{source_stem}:{page_kind}:{page_number}:match:{match_index}",
                "source_path": str(page_row.get("source_path", "")),
                "source_stem": source_stem,
                "base_stem": str(page_row.get("base_stem", "")),
                "debug_output_path": output_path,
                "page_kind": page_kind,
                "page_number": page_number,
                "page_index_in_file": page_index_in_file,
                "page_char_count": page_char_count,
                "match_index_in_page": match_index,
                "start_char": start,
                "end_char": end,
                "start_byte": int(byte_offsets[start]),
                "end_byte": int(byte_offsets[end]),
                "match_length_chars": int(end - start),
                "match_length_bytes": int(byte_offsets[end] - byte_offsets[start]),
                "start_line": int(page_text.count("\n", 0, start) + 1),
                "end_line": int(page_text.count("\n", 0, max(start, end - 1)) + 1),
                "categories": categories,
                "category": ",".join(categories),
                "pattern_families": pattern_families,
                "pattern_family": ",".join(pattern_families),
                "matched_text": page_text[start:end],
                "raw_texts": [str(item) for item in list(match.get("raw_texts") or [])],
                "context_before": page_text[excerpt_start:start],
                "context_after": page_text[end:excerpt_end],
                "context_excerpt": page_text[excerpt_start:excerpt_end],
            }
        )
    return rows


def _find_labeled_shared_repeat_spans(
    page_text: str,
    *,
    blocked_spans: List[Dict[str, Any]],
    rep_threshold: int,
    min_period: int,
    window: int,
    analysis_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if analysis_text is None:
        analysis_text = _filter_tables_preserve_layout(page_text)
        analysis_text = _filter_latex_preserve_layout(analysis_text)
        analysis_text = _blank_existing_match_regions_preserve_layout(analysis_text)
    analysis_text = _blank_raw_spans_preserve_layout(analysis_text, blocked_spans)
    rust_mod = _get_word_repeat_rust_module()
    if rust_mod is not None and hasattr(rust_mod, "find_labeled_shared_repeat_spans"):
        return [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "period": int(item["period"]),
                "repetitions": int(item["repetitions"]),
                "tail_chars": int(item["tail_chars"]),
                "match_types": [str(item["match_type"])],
                "category": MATCH_CATEGORY_BY_TYPE[str(item["match_type"])],
            }
            for item in rust_mod.find_labeled_shared_repeat_spans(
                analysis_text,
                int(rep_threshold),
                int(min_period),
                int(window),
            )
        ]
    normalized_text, raw_map = _normalize_alnum_with_map_skip_tags(analysis_text)
    normalized_spans = _find_word_repeat_spans(
        normalized_text,
        rep_threshold=rep_threshold,
        min_period=min_period,
        window=window,
    )
    labeled_spans: List[Dict[str, Any]] = []
    for span in normalized_spans:
        if span["end"] <= span["start"] or span["start"] >= len(raw_map):
            continue
        match_type = _shared_repeat_match_type(normalized_text[span["start"] : span["end"]])
        if match_type is None:
            continue
        start = raw_map[span["start"]]
        end = raw_map[span["end"] - 1] + 1
        labeled_spans.append(
            {
                "start": start,
                "end": end,
                "period": span["period"],
                "repetitions": span["repetitions"],
                "tail_chars": span["tail_chars"],
                "match_types": [match_type],
                "category": MATCH_CATEGORY_BY_TYPE[match_type],
            }
        )
    return labeled_spans


def _analyze_combined_ocr_page(
    page_text: str,
    *,
    noise_mod: Any,
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
) -> Dict[str, Any]:
    page_start = time.perf_counter()

    char_eval_start = time.perf_counter()
    page_noise_metrics = dict(noise_mod.evaluate_page_character_noise(page_text))
    char_eval_elapsed = time.perf_counter() - char_eval_start

    table_start = time.perf_counter()
    table_spans = _find_table_repeat_spans(page_text)
    table_elapsed = time.perf_counter() - table_start

    # Reuse progressively filtered page views so later passes do not rebuild the
    # same blanked surfaces repeatedly.
    page_without_tables = _filter_tables_preserve_layout(page_text)
    page_without_tables_existing = _blank_existing_match_regions_preserve_layout(page_without_tables)
    page_without_tables_latex = _filter_latex_preserve_layout(page_without_tables)
    page_without_tables_latex_existing = _blank_existing_match_regions_preserve_layout(
        page_without_tables_latex
    )

    numeric_start = time.perf_counter()
    numeric_spans = [
        {
            "start": int(item["start"]),
            "end": int(item["end"]),
            "match_types": [str(item["match_type"])],
            "category": MATCH_CATEGORY_BY_TYPE[str(item["match_type"])],
        }
        for item in noise_mod.find_numeric_debug_page_spans(
            page_without_tables_latex,
            int(min_progress_steps),
            int(min_repeat_steps),
            int(min_same_digit_steps),
        )
    ]
    numeric_elapsed = time.perf_counter() - numeric_start

    latex_start = time.perf_counter()
    latex_spans = _find_latex_repeat_spans(
        page_text,
        blocked_spans=table_spans + numeric_spans,
        rep_threshold=int(word_rep_threshold),
        min_period=int(word_min_period),
        window=int(word_window),
        analysis_text=page_without_tables_existing,
    )
    latex_elapsed = time.perf_counter() - latex_start

    hybrid_start = time.perf_counter()
    hybrid_spans = _find_hybrid_numbered_repeat_spans(
        page_text,
        blocked_spans=table_spans + numeric_spans + latex_spans,
        analysis_text=page_without_tables_latex_existing,
    )
    hybrid_elapsed = time.perf_counter() - hybrid_start

    shared_start = time.perf_counter()
    shared_spans = _find_labeled_shared_repeat_spans(
        page_text,
        blocked_spans=table_spans + numeric_spans + latex_spans + hybrid_spans,
        rep_threshold=int(word_rep_threshold),
        min_period=int(word_min_period),
        window=int(word_window),
        analysis_text=page_without_tables_latex_existing,
    )
    shared_elapsed = time.perf_counter() - shared_start

    page_total_time = time.perf_counter() - page_start
    return {
        "spans": table_spans + numeric_spans + latex_spans + hybrid_spans + shared_spans,
        "page_noise_metrics": page_noise_metrics,
        "char_eval_seconds": char_eval_elapsed,
        "table_seconds": table_elapsed,
        "numeric_seconds": numeric_elapsed,
        "latex_seconds": latex_elapsed,
        "hybrid_seconds": hybrid_elapsed,
        "shared_repeat_seconds": shared_elapsed,
        "total_page_seconds": page_total_time,
    }


def _render_combined_ocr_page(
    page_text: str,
    *,
    noise_mod: Any,
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
    mode: str = "debug",
) -> Dict[str, Any]:
    """Analyze one OCR page in the shared ownership order.

    The ordering is a policy decision, not an implementation accident:
    1. tables first, because table shells distort every later text pass
    2. numeric second, because numeric progressions should not be stolen by
       generic word repetition
    3. LaTeX and hybrid structural passes next, because they operate on more
       specialized local structure
    4. shared text repetition last, on the remaining visible surface only

    That ownership model keeps the matcher family specific and reduces the
    false positives that appear when a single fuzzy text matcher sees
    everything at once.
    """
    analysis = _analyze_combined_ocr_page(
        page_text,
        noise_mod=noise_mod,
        min_progress_steps=min_progress_steps,
        min_repeat_steps=min_repeat_steps,
        min_same_digit_steps=min_same_digit_steps,
        word_rep_threshold=word_rep_threshold,
        word_min_period=word_min_period,
        word_window=word_window,
    )
    render_result = _render_page_with_labeled_spans_result(
        page_text,
        list(analysis["spans"]),
        mode=mode,
    )
    return {
        "annotated_page": render_result["rendered_page"],
        "merged_spans": render_result["merged_spans"],
        "page_types": render_result["page_types"],
        "page_numeric_count": render_result["page_numeric_count"],
        "page_word_count": render_result["page_word_count"],
        "page_latex_count": render_result["page_latex_count"],
        "page_table_count": render_result["page_table_count"],
        "page_hybrid_count": render_result["page_hybrid_count"],
        **analysis,
    }


def _render_combined_ocr_page_modes(
    page_text: str,
    *,
    noise_mod: Any,
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
    modes: Iterable[str],
) -> Dict[str, Any]:
    analysis = _analyze_combined_ocr_page(
        page_text,
        noise_mod=noise_mod,
        min_progress_steps=min_progress_steps,
        min_repeat_steps=min_repeat_steps,
        min_same_digit_steps=min_same_digit_steps,
        word_rep_threshold=word_rep_threshold,
        word_min_period=word_min_period,
        word_window=word_window,
    )
    merged_spans = _merge_labeled_raw_spans(page_text, list(analysis["spans"]))
    (
        page_types,
        page_numeric_count,
        page_word_count,
        page_latex_count,
        page_table_count,
        page_hybrid_count,
    ) = _summarize_merged_labeled_spans(merged_spans)
    rendered_pages = {
        str(mode): _render_page_from_merged_labeled_spans(page_text, merged_spans, mode=str(mode))
        for mode in modes
    }
    return {
        "rendered_pages": rendered_pages,
        "merged_spans": merged_spans,
        "page_types": page_types,
        "page_numeric_count": page_numeric_count,
        "page_word_count": page_word_count,
        "page_latex_count": page_latex_count,
        "page_table_count": page_table_count,
        "page_hybrid_count": page_hybrid_count,
        **analysis,
    }


def _render_combined_ocr_debug_page(
    page_text: str,
    *,
    noise_mod: Any,
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
) -> Dict[str, Any]:
    return _render_combined_ocr_page(
        page_text,
        noise_mod=noise_mod,
        min_progress_steps=min_progress_steps,
        min_repeat_steps=min_repeat_steps,
        min_same_digit_steps=min_same_digit_steps,
        word_rep_threshold=word_rep_threshold,
        word_min_period=word_min_period,
        word_window=word_window,
        mode="debug",
    )


def _process_combined_ocr_document(
    source_path: Path,
    *,
    clean_output_path: Optional[Path],
    debug_output_path: Optional[Path],
    noise_mod: Optional[Any],
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
    include_page_metrics: bool,
    include_match_index: bool,
) -> Dict[str, Any]:
    if noise_mod is None:
        noise_mod = _get_combined_ocr_worker_noise_mod()
    text = source_path.read_text(encoding="utf-8")
    pages = text.split(PAGE_SPLIT_MARKER)
    cleaned_pages: List[str] = []
    debug_pages: List[str] = []
    matched_page_count = 0
    table_match_count = 0
    numeric_match_count = 0
    latex_match_count = 0
    hybrid_match_count = 0
    word_match_count = 0
    doc_match_types: Set[str] = set()
    page_metric_rows: List[Dict[str, Any]] = []
    match_index_rows: List[Dict[str, Any]] = []

    for page_index, page in enumerate(pages, start=1):
        if clean_output_path is not None and debug_output_path is not None:
            page_result = _render_combined_ocr_page_modes(
                page,
                noise_mod=noise_mod,
                min_progress_steps=int(min_progress_steps),
                min_repeat_steps=int(min_repeat_steps),
                min_same_digit_steps=int(min_same_digit_steps),
                word_rep_threshold=int(word_rep_threshold),
                word_min_period=int(word_min_period),
                word_window=int(word_window),
                modes=("clean", "debug"),
            )
            cleaned_page = str(page_result["rendered_pages"]["clean"])
            debug_page = str(page_result["rendered_pages"]["debug"])
        elif debug_output_path is not None:
            page_result = _render_combined_ocr_page(
                page,
                noise_mod=noise_mod,
                min_progress_steps=int(min_progress_steps),
                min_repeat_steps=int(min_repeat_steps),
                min_same_digit_steps=int(min_same_digit_steps),
                word_rep_threshold=int(word_rep_threshold),
                word_min_period=int(word_min_period),
                word_window=int(word_window),
                mode="debug",
            )
            cleaned_page = ""
            debug_page = str(page_result["annotated_page"])
        else:
            page_result = _render_combined_ocr_page(
                page,
                noise_mod=noise_mod,
                min_progress_steps=int(min_progress_steps),
                min_repeat_steps=int(min_repeat_steps),
                min_same_digit_steps=int(min_same_digit_steps),
                word_rep_threshold=int(word_rep_threshold),
                word_min_period=int(word_min_period),
                word_window=int(word_window),
                mode="clean",
            )
            cleaned_page = str(page_result["annotated_page"])
            debug_page = ""

        merged_spans = list(page_result.get("merged_spans", []))
        page_types = list(page_result["page_types"])
        page_numeric_count = int(page_result["page_numeric_count"])
        page_word_count = int(page_result["page_word_count"])
        page_latex_count = int(page_result["page_latex_count"])
        page_table_count = int(page_result["page_table_count"])
        page_hybrid_count = int(page_result["page_hybrid_count"])
        page_noise_metrics = dict(page_result["page_noise_metrics"])
        char_eval_elapsed = float(page_result["char_eval_seconds"])
        table_elapsed = float(page_result["table_seconds"])
        numeric_elapsed = float(page_result["numeric_seconds"])
        latex_elapsed = float(page_result["latex_seconds"])
        hybrid_elapsed = float(page_result["hybrid_seconds"])
        shared_elapsed = float(page_result["shared_repeat_seconds"])
        page_total_time = float(page_result["total_page_seconds"])

        if clean_output_path is not None:
            cleaned_pages.append(cleaned_page)
        if debug_output_path is not None:
            debug_pages.append(debug_page)

        page_match_total = (
            page_table_count + page_numeric_count + page_word_count + page_latex_count + page_hybrid_count
        )
        if page_match_total:
            matched_page_count += 1
        table_match_count += page_table_count
        numeric_match_count += page_numeric_count
        latex_match_count += page_latex_count
        hybrid_match_count += page_hybrid_count
        word_match_count += page_word_count
        doc_match_types.update(page_types)

        if include_page_metrics:
            page_metric_rows.append(
                {
                    "source_path": str(source_path),
                    "source_stem": source_path.stem,
                    "page_number": page_index,
                    "page_index_in_file": page_index,
                    "total_chars": int(page_noise_metrics.get("total_chars", 0)),
                    "bad_char_count": int(page_noise_metrics.get("bad_char_count", 0)),
                    "bad_char_ratio": float(page_noise_metrics.get("bad_char_ratio", 0.0)),
                    "control_count": int(page_noise_metrics.get("control_count", 0)),
                    "private_use_count": int(page_noise_metrics.get("private_use_count", 0)),
                    "cjk_count": int(page_noise_metrics.get("cjk_count", 0)),
                    "replacement_count": int(page_noise_metrics.get("replacement_count", 0)),
                    "table_match_count": page_table_count,
                    "numeric_match_count": page_numeric_count,
                    "latex_match_count": page_latex_count,
                    "hybrid_match_count": page_hybrid_count,
                    "word_match_count": page_word_count,
                    "match_types": ",".join(page_types),
                    "char_eval_seconds": char_eval_elapsed,
                    "table_seconds": table_elapsed,
                    "numeric_seconds": numeric_elapsed,
                    "latex_seconds": latex_elapsed,
                    "hybrid_seconds": hybrid_elapsed,
                    "shared_repeat_seconds": shared_elapsed,
                    "total_page_seconds": page_total_time,
                }
            )

        if include_match_index:
            match_index_rows.extend(
                _build_match_index_rows(
                    page,
                    merged_spans,
                    source_path=source_path,
                    page_number=page_index,
                    debug_output_path=debug_output_path,
                )
            )

    if clean_output_path is not None:
        clean_output_path.write_text(PAGE_SPLIT_MARKER.join(cleaned_pages), encoding="utf-8")
    if debug_output_path is not None:
        debug_output_path.write_text(PAGE_SPLIT_MARKER.join(debug_pages), encoding="utf-8")

    output_path = debug_output_path or clean_output_path
    row = {
        "source_path": str(source_path),
        "output_path": None if output_path is None else str(output_path),
        "clean_output_path": None if clean_output_path is None else str(clean_output_path),
        "debug_output_path": None if debug_output_path is None else str(debug_output_path),
        "source_stem": source_path.stem,
        "base_stem": canonical_stem(source_path.stem),
        "page_count": len(pages),
        "matched_page_count": matched_page_count,
        "table_match_count": table_match_count,
        "numeric_match_count": numeric_match_count,
        "latex_match_count": latex_match_count,
        "hybrid_match_count": hybrid_match_count,
        "word_match_count": word_match_count,
        "match_count": int(len(match_index_rows)),
        "match_types": ",".join(sorted(doc_match_types)),
    }
    return {
        "row": row,
        "page_metric_rows": page_metric_rows,
        "match_index_rows": match_index_rows,
    }


def _process_combined_ocr_debug_document(
    source_path: Path,
    output_path: Path,
    *,
    noise_mod: Optional[Any],
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
) -> Dict[str, Any]:
    return _process_combined_ocr_document(
        source_path,
        clean_output_path=None,
        debug_output_path=output_path,
        noise_mod=noise_mod,
        min_progress_steps=min_progress_steps,
        min_repeat_steps=min_repeat_steps,
        min_same_digit_steps=min_same_digit_steps,
        word_rep_threshold=word_rep_threshold,
        word_min_period=word_min_period,
        word_window=word_window,
        include_page_metrics=True,
        include_match_index=True,
    )


def _process_combined_ocr_clean_document(
    source_path: Path,
    output_path: Path,
    *,
    noise_mod: Optional[Any],
    min_progress_steps: int,
    min_repeat_steps: int,
    min_same_digit_steps: int,
    word_rep_threshold: int,
    word_min_period: int,
    word_window: int,
) -> None:
    _process_combined_ocr_document(
        source_path,
        clean_output_path=output_path,
        debug_output_path=None,
        noise_mod=noise_mod,
        min_progress_steps=min_progress_steps,
        min_repeat_steps=min_repeat_steps,
        min_same_digit_steps=min_same_digit_steps,
        word_rep_threshold=word_rep_threshold,
        word_min_period=word_min_period,
        word_window=word_window,
        include_page_metrics=False,
        include_match_index=False,
    )


def _process_combined_ocr_debug_document_job(
    job: Tuple[str, str, int, int, int, int, int, int]
) -> Dict[str, Any]:
    (
        source_path_str,
        output_path_str,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
        word_rep_threshold,
        word_min_period,
        word_window,
    ) = job
    return _process_combined_ocr_debug_document(
        Path(source_path_str),
        Path(output_path_str),
        noise_mod=None,
        min_progress_steps=int(min_progress_steps),
        min_repeat_steps=int(min_repeat_steps),
        min_same_digit_steps=int(min_same_digit_steps),
        word_rep_threshold=int(word_rep_threshold),
        word_min_period=int(word_min_period),
        word_window=int(word_window),
    )


def _process_combined_ocr_clean_document_job(
    job: Tuple[str, str, int, int, int, int, int, int]
) -> None:
    (
        source_path_str,
        output_path_str,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
        word_rep_threshold,
        word_min_period,
        word_window,
    ) = job
    _process_combined_ocr_clean_document(
        Path(source_path_str),
        Path(output_path_str),
        noise_mod=None,
        min_progress_steps=int(min_progress_steps),
        min_repeat_steps=int(min_repeat_steps),
        min_same_digit_steps=int(min_same_digit_steps),
        word_rep_threshold=int(word_rep_threshold),
        word_min_period=int(word_min_period),
        word_window=int(word_window),
    )


def _process_combined_ocr_dual_document_job(
    job: Tuple[str, str, str, int, int, int, int, int, int]
) -> Dict[str, Any]:
    (
        source_path_str,
        clean_output_path_str,
        debug_output_path_str,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
        word_rep_threshold,
        word_min_period,
        word_window,
    ) = job
    return _process_combined_ocr_document(
        Path(source_path_str),
        clean_output_path=Path(clean_output_path_str),
        debug_output_path=Path(debug_output_path_str),
        noise_mod=None,
        min_progress_steps=int(min_progress_steps),
        min_repeat_steps=int(min_repeat_steps),
        min_same_digit_steps=int(min_same_digit_steps),
        word_rep_threshold=int(word_rep_threshold),
        word_min_period=int(word_min_period),
        word_window=int(word_window),
        include_page_metrics=True,
        include_match_index=True,
    )


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    array = np.array(values, dtype=float)
    return {
        "count": int(array.size),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


class CleanPhaseMixin:
    @staticmethod
    def _project_root() -> Path:
        """Locate the repository root that houses the Rust crates."""
        here = Path(__file__).resolve()
        for candidate in here.parents:
            rust_dir = candidate / "rust"
            if rust_dir.exists() and rust_dir.is_dir():
                return candidate
        return here.parents[2]

    def _load_rust_extension(
        self,
        module_name: str,
        manifest_relative: str,
        *,
        required_attrs: Optional[Iterable[str]] = None,
    ):
        """Import a Rust extension, building it with maturin if necessary.

        The load path is intentionally import-first:
        - fast path: import an already-built extension and return immediately
        - fallback: build in place only if the module is missing or incomplete

        That keeps ordinary OCR runs from paying a `maturin develop` startup tax
        in every fresh process while still letting a developer bootstrap a local
        checkout without separate setup steps.
        """
        import importlib

        required = tuple(required_attrs or ())

        def _missing_attrs(module: Any) -> List[str]:
            return [attr for attr in required if not hasattr(module, attr)]

        def _build_extension_once() -> None:
            if module_name in _RUST_EXTENSION_PREBUILD_ATTEMPTED:
                return
            _RUST_EXTENSION_PREBUILD_ATTEMPTED.add(module_name)
            root_dir = self._project_root()
            manifest = root_dir / manifest_relative
            if not manifest.exists():
                return
            build_env = os.environ.copy()
            if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
                build_env.setdefault("VIRTUAL_ENV", sys.prefix)
                venv_bin = str(Path(sys.prefix) / "bin")
                build_env["PATH"] = (
                    f"{venv_bin}:{build_env['PATH']}"
                    if build_env.get("PATH")
                    else venv_bin
                )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"],
                    check=True,
                    env=build_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "maturin",
                        "develop",
                        "--release",
                        "--manifest-path",
                        str(manifest),
                    ],
                    check=True,
                    env=build_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                importlib.invalidate_caches()
            except Exception as build_err:
                self.logger.debug(
                    "Rust prebuild for %s skipped or failed: %s",
                    module_name,
                    build_err,
                )

        def _import_module_with_fallback():
            candidates = [module_name]
            if "." not in module_name:
                candidates.append(f"{module_name}.{module_name}")

            last_error: Optional[Exception] = None
            for candidate in candidates:
                try:
                    return importlib.import_module(candidate)
                except Exception as err:  # pragma: no cover - import surface varies by wheel layout
                    last_error = err
            if last_error is not None:
                raise last_error
            raise ModuleNotFoundError(module_name)

        needs_build = False
        try:
            module = _import_module_with_fallback()
            missing = _missing_attrs(module)
            if not missing:
                return _prime_word_repeat_rust_module(module_name, module)
            self.logger.warning(
                "Rust extension %s is missing required attributes %s; attempting in-place build via maturin …",
                module_name,
                ", ".join(missing),
            )
            needs_build = True
        except ModuleNotFoundError:
            self.logger.warning(
                "Rust extension %s missing; attempting in-place build via maturin …",
                module_name,
            )
            needs_build = True

        if needs_build:
            _build_extension_once()
            try:
                module = _import_module_with_fallback()
                missing = _missing_attrs(module)
                if not missing:
                    return _prime_word_repeat_rust_module(module_name, module)
            except ModuleNotFoundError:
                pass

        if not needs_build:
            raise RuntimeError(f"Unexpected load state for Rust extension {module_name}")

        root_dir = self._project_root()
        manifest = root_dir / manifest_relative
        if not manifest.exists():
            raise RuntimeError(
                f"Cannot locate Cargo manifest for {module_name} at {manifest}"
            )
        try:
            build_env = os.environ.copy()
            if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
                build_env.setdefault("VIRTUAL_ENV", sys.prefix)
                venv_bin = str(Path(sys.prefix) / "bin")
                build_env["PATH"] = (
                    f"{venv_bin}:{build_env['PATH']}"
                    if build_env.get("PATH")
                    else venv_bin
                )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"],
                check=True,
                env=build_env,
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "maturin",
                    "develop",
                    "--release",
                    "--manifest-path",
                    str(manifest),
                ],
                check=True,
                env=build_env,
            )
            importlib.invalidate_caches()
            sys.modules.pop(module_name, None)
            if "." not in module_name:
                sys.modules.pop(f"{module_name}.{module_name}", None)
            module = _import_module_with_fallback()
            missing = _missing_attrs(module)
            if missing:
                raise RuntimeError(
                    f"Built {module_name} but it is still missing required attributes: {missing}"
                )
            return _prime_word_repeat_rust_module(module_name, module)
        except Exception as build_err:
            raise RuntimeError(
                f"Automatic build of {module_name} failed: {build_err}"
            )

    def _load_metrics_dataframe(
        self, parquet_path: Path, filenames: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Load an analytics parquet or seed an empty frame keyed by filename."""
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        names: List[str] = []
        if filenames is not None:
            seen: Set[str] = set()
            for item in filenames:
                if item is None:
                    continue
                name = str(item)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return pd.DataFrame({"filename": names})

    @staticmethod
    def _ensure_metric_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> None:
        """Ensure metric columns exist with provided defaults."""
        for column, default in defaults.items():
            if column not in df.columns:
                df[column] = default

    @staticmethod
    def _merge_metric_dataframe(
        base: pd.DataFrame, updates: pd.DataFrame, *, key: str = "filename"
    ) -> pd.DataFrame:
        """Overlay scorer output onto the authoritative metrics dataframe."""
        if updates.empty:
            return base
        base_idx = base.set_index(key, drop=False)
        update_idx = updates.set_index(key, drop=False)
        base_idx = base_idx.combine_first(update_idx)
        base_idx.update(update_idx)
        return base_idx.reset_index(drop=True)

    def _resolve_clean_metrics_parquet(self, parquet_schema) -> Path:
        parquet_path: Optional[Path] = self._get_cached_metadata_parquet()
        if parquet_path is None:
            existing_metadata = parquet_schema.find_metadata_parquet(self.input_dir)
            if existing_metadata is not None:
                parquet_path = self._cache_metadata_parquet(existing_metadata)
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.output_dir)
            if ensured is not None:
                parquet_path = self._cache_metadata_parquet(ensured)
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.input_dir)
            if ensured is not None:
                parquet_path = self._cache_metadata_parquet(ensured)
        if parquet_path is None:
            metadata_target = self.output_dir / "download_results" / "download_results.parquet"
            self.logger.info(
                "Cleaner: no metadata parquet found; will bootstrap %s when metrics become available.",
                metadata_target,
            )
        else:
            metadata_target = parquet_path
        return self._cache_metadata_parquet(metadata_target)

    def clean(
        self,
        input_dir: Union[str, Path] = None,
        threshold: float = 0.10,
        num_threads: int = None,
        drop_bad: bool = True,
        *,
        write_cleaned_files: bool = True,
        ocr_model_dir: Union[str, Path, None] = None,
        force_ocr_fallback: bool = False,
        empty_char_threshold: int = 0,
        empty_min_pages: int = 0,
    ) -> None:
        """Clean markdown files and evaluate badness using the Rust extension.

        Args:
            input_dir: Folder with `.md` files to process (defaults to `self.markdown_dir`).
            threshold: Badness threshold for optional dropping.
            num_threads: Rayon thread-count to pass to Rust.
            drop_bad: If True, files with badness_score > threshold are removed from downstream processing. Set to False to keep all files and only record the score.
            write_cleaned_files: Set False to skip writing cleaned markdown files; metrics and parquet updates still occur.
            ocr_model_dir: [DEPRECATED – no effect] Use Corpus.ocr(model_dir=...) instead.
            force_ocr_fallback: [DEPRECATED – no effect] Use Corpus.ocr(fix_bad=True) instead.
            empty_char_threshold: Character threshold (after stripping comments and whitespace) that flags markdown as nearly empty. Default 0 only enforces the zero-character safeguard.
            empty_min_pages: Minimum page count for a low-character document to trigger an OCR rerun recommendation.
        """
        from pathlib import Path
        import shutil
        import pandas as pd
        from glossapi.parquet_schema import ParquetSchema

        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        # Handle OCR model directory override
        if ocr_model_dir is not None:
            self.ocr_model_dir = Path(ocr_model_dir)

        self._load_rust_extension(
            "glossapi_rs_cleaner",
            "rust/glossapi_rs_cleaner/Cargo.toml",
            required_attrs=("run_complete_pipeline",),
        )
        self.logger.info("Using compiled glossapi_rs_cleaner extension for fast cleaning")

        # Ensure cleaned directory exists and is empty (idempotent runs)
        if write_cleaned_files:
            if self.cleaned_markdown_dir.exists():
                shutil.rmtree(self.cleaned_markdown_dir)
            self.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)

        # Prepare parquet helper
        parquet_schema = ParquetSchema({"url_column": self.url_column})
        parquet_path = self._resolve_clean_metrics_parquet(parquet_schema)

        import os
        records: list = []  # will hold metrics for parquet merge
        metrics_dir = self.output_dir / "json" / "metrics"

        def _page_count_for(stem: str) -> Optional[int]:
            candidates = [
                metrics_dir / f"{stem}.metrics.json",
                metrics_dir / f"{stem}.per_page.metrics.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    pc = data.get("page_count")
                    if pc is not None:
                        try:
                            return int(pc)
                        except Exception:
                            pass
                    pages = data.get("pages")
                    if isinstance(pages, list):
                        return len(pages)
            return None

        # ----- Call Rust high-level pipeline once -----
        scripts_to_keep = ["greek", "latin"]  # keep common alphabetic scripts; numbers/punctuation are added internally
        report_parquet_path = self.cleaned_markdown_dir.parent / "cleaning_report.parquet"

        md_files = sorted(input_dir.glob("*.md"))
        total_files = len(md_files)

        self.logger.info(
            "Invoking glossapi_rs_cleaner.run_complete_pipeline on %d markdown files…",
            total_files,
        )

        class _CleanerProgress:
            def __init__(self, logger: logging.Logger, total: int) -> None:
                self.logger = logger
                self.total = total
                self.processed: set[str] = set()
                self.buffer = ""
                if total > 0:
                    step = max(1, math.ceil(total * 0.02))
                else:
                    step = 1
                self.step = step
                self.next_target = step
                self.logged_full = False
                self.last_message: Optional[str] = None
                self.direct_updates = False
                self.last_processed = 0

            def write(self, text: str) -> int:
                if not text:
                    return 0
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    self._handle_line(line.strip())
                return len(text)

            def flush(self) -> None:  # pragma: no cover - required by IO interface
                return

            def handle_line(self, line: str) -> None:
                self._handle_line(line.strip())

            def _handle_line(self, line: str) -> None:
                if not line:
                    return
                direct = re.search(
                    r"Rust cleaning progress:\s*(\d+)%\s*\((\d+)/(\d+)\)", line
                )
                if direct:
                    try:
                        percent = int(direct.group(1))
                        processed = int(direct.group(2))
                        total_reported = int(direct.group(3))
                    except (TypeError, ValueError):
                        percent = processed = 0
                        total_reported = self.total
                    else:
                        if total_reported > 0 and total_reported != self.total:
                            self.total = total_reported
                            self.step = max(1, math.ceil(self.total * 0.02))
                            self.next_target = self.step
                    self.direct_updates = True
                    self.last_processed = processed
                    self.logger.info(
                        "Rust cleaning progress: %d%% (%d/%d)",
                        percent,
                        processed,
                        self.total or total_reported,
                    )
                    if percent >= 100 or (
                        self.total and processed >= self.total
                    ):
                        self.logged_full = True
                    return
                match = re.search(r"Processing file:\s*(.+)", line)
                if match:
                    path = match.group(1).strip()
                    stem = Path(path).stem if path else None
                    if stem and stem not in self.processed:
                        self.processed.add(stem)
                        self._log_progress()
                    return
                if "complete pipeline finished successfully" in line or "Parquet report written successfully" in line:
                    self.last_message = line

            def _log_progress(self) -> None:
                if self.direct_updates:
                    return
                if self.total <= 0:
                    return
                processed = len(self.processed)
                while self.next_target <= self.total and processed >= self.next_target:
                    percent = min(100, int(round(self.next_target * 100 / self.total)))
                    self.logger.info(
                        "Rust cleaning progress: %d%% (%d/%d)", percent, processed, self.total
                    )
                    if percent >= 100:
                        self.logged_full = True
                    self.next_target += self.step

            def finalize(self) -> None:
                if self.total == 0:
                    self.logger.info("Rust cleaning progress: 100%% (0/0)")
                elif not self.logged_full:
                    processed = self.last_processed or len(self.processed)
                    self.logger.info(
                        "Rust cleaning progress: 100%% (%d/%d)", processed, self.total
                    )
                if self.last_message:
                    self.logger.debug(self.last_message)

        progress = _CleanerProgress(self.logger, total_files)
        cmd = (
            "import glossapi_rs_cleaner\n"
            f"glossapi_rs_cleaner.run_complete_pipeline({repr(str(input_dir))}, "
            f"{repr(str(self.cleaned_markdown_dir))}, {repr(str(report_parquet_path))}, "
            f"{repr(scripts_to_keep)}, {int(num_threads or os.cpu_count() or 4)}, "
            f"{'True' if write_cleaned_files else 'False'})\n"
        )

        process = subprocess.Popen(
            [sys.executable, "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                progress.handle_line(line)
            return_code = process.wait()
        except Exception:
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            progress.finalize()

        if return_code != 0:
            # Do not abort the entire cleaning pass – proceed to evaluate gates
            # using existing metrics on disk. If the Rust report is available,
            # it will be merged below as usual.
            self.logger.error("Rust cleaning pipeline failed (code=%s); proceeding with existing metrics", return_code)

        # ----- Parse metrics Parquet produced by Rust -----
        if report_parquet_path.exists():
            try:
                df_metrics_parquet = pd.read_parquet(report_parquet_path)
                for _, row in df_metrics_parquet.iterrows():
                    records.append(
                        {
                            "filename": f"{Path(row['file_name']).stem}.pdf",  # match original PDF filename
                            "badness_score": row.get("badness_score_all_chars", 0.0),
                            "percentage_greek": row.get("percentage_greek_cleaned"),
                            "percentage_latin": row.get("percentage_latin_cleaned"),
                            "char_count_no_comments": row.get("char_count_no_comments"),
                            "is_empty": row.get("is_empty", False),
                        }
                    )
            except Exception as e:
                self.logger.warning("Failed to parse cleaning report %s: %s", report_parquet_path, e)
        else:
            self.logger.warning("Cleaning report Parquet not found: %s", report_parquet_path)

        # ---- Delete cleaning report to avoid retaining it ----
        try:
            if report_parquet_path.exists():
                report_parquet_path.unlink(missing_ok=True)
                self.logger.debug("Deleted temporary cleaning report %s", report_parquet_path)
        except Exception as e:
            self.logger.warning("Could not delete cleaning report %s: %s", report_parquet_path, e)

        self.logger.info(f"Cleaned {len(records)} markdown files → {self.cleaned_markdown_dir}")

        # ------------------------------------------------------------------
        # Update parquet with Mojibake metrics (single authoritative schema)
        # ------------------------------------------------------------------
        if records:
            df_metrics = pd.DataFrame(records).rename(
                columns={
                    "badness_score": "mojibake_badness_score",
                    "percentage_latin": "mojibake_latin_percentage",
                }
            )

            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            df = self._load_metrics_dataframe(parquet_path, df_metrics.get("filename"))
            self._ensure_metric_columns(
                df,
                {
                    "mojibake_badness_score": pd.NA,
                    "mojibake_latin_percentage": pd.NA,
                    "percentage_greek": pd.NA,
                    "greek_badness_score": pd.NA,
                    "greek_latin_percentage": pd.NA,
                    "rejection_reason": pd.NA,
                    "char_count_no_comments": pd.NA,
                    "is_empty": pd.NA,
                },
            )

            df = self._merge_metric_dataframe(
                df,
                df_metrics[
                    [
                        "filename",
                        "mojibake_badness_score",
                        "mojibake_latin_percentage",
                        "percentage_greek",
                        "char_count_no_comments",
                        "is_empty",
                    ]
                ],
            )
            parquet_schema.write_metadata_parquet(df, parquet_path)
            self.logger.info("Mojibake metrics updated in %s", parquet_path)

        # ----- Noise-metrics scoring (Rust) -----
        try:
            self.logger.info("Scoring cleaned markdown files with glossapi_rs_noise …")
            noise_mod = self._load_rust_extension(
                "glossapi_rs_noise",
                "rust/glossapi_rs_noise/Cargo.toml",
                required_attrs=("score_markdown_directory_detailed",),
            )
            results = noise_mod.score_markdown_directory_detailed(
                str(self.cleaned_markdown_dir), os.cpu_count()
            )
            if results:
                rows = []
                for row in results:
                    try:
                        path, score, latin_pct, _table_ratio, poly_ratio = row[:5]
                    except Exception:
                        continue
                    rows.append((path, float(score), float(latin_pct), float(poly_ratio)))

                df_scores = pd.DataFrame(
                    rows,
                    columns=[
                        "filepath",
                        "greek_badness_score",
                        "greek_latin_percentage",
                        "polytonic_ratio",
                    ],
                )
                df_scores["polytonic_ratio"] = df_scores["polytonic_ratio"].round(2)
                df_scores["stem"] = df_scores["filepath"].apply(lambda p: Path(p).name)
                df_scores["stem"] = df_scores["stem"].str.replace(r"\.md$", "", regex=True)
                df_scores["filename"] = df_scores["stem"] + ".pdf"
                df_scores["rejection_reason"] = np.select(
                    [df_scores["greek_badness_score"] > 60],
                    ["greek>60"],
                    default="ok",
                )
                if not parquet_path.exists():
                    self.logger.error(
                        "Expected parquet %s not found when adding noise metrics",
                        parquet_path,
                    )
                else:
                    df = self._load_metrics_dataframe(parquet_path)
                    self._ensure_metric_columns(
                        df,
                        {
                            "greek_badness_score": pd.NA,
                            "greek_latin_percentage": pd.NA,
                            "polytonic_ratio": pd.NA,
                            "rejection_reason": pd.NA,
                        },
                    )
                    updates = df_scores[
                        [
                            "filename",
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "polytonic_ratio",
                            "rejection_reason",
                        ]
                    ]
                    df = self._merge_metric_dataframe(df, updates)
                    parquet_schema.write_metadata_parquet(df, parquet_path)
                    self.logger.info("Noise metrics filled in %s", parquet_path)
        except Exception as e:
            self.logger.warning("Noise-metrics scoring failed: %s", e)

        # Determine good / bad list based on enriched metrics
        if parquet_path.exists():
            df_final = pd.read_parquet(parquet_path)
            self._ensure_metric_columns(
                df_final,
                {
                    "mojibake_badness_score": pd.NA,
                    "mojibake_latin_percentage": pd.NA,
                    "percentage_greek": pd.NA,
                    "greek_badness_score": pd.NA,
                    "greek_latin_percentage": pd.NA,
                    "char_count_no_comments": pd.NA,
                    "is_empty": pd.NA,
                },
            )
            # --- tidy schema ---
            df_final.rename(columns={
                "badness_score": "mojibake_badness_score",
                "percentage_latin": "mojibake_latin_percentage",
                "mojibake_latin_percentage": "latin_percentage",  # ADD THIS
                "rejection_reason": "filter"                      # ADD THIS
            }, inplace=True, errors="ignore")

            # drop duplicate pandas merge suffixes and keep clean names
            df_final = df_final.loc[:, ~df_final.columns.str.endswith('_x')]
            df_final.columns = df_final.columns.str.replace('_y$','', regex=True)

            # round Greek scores for readability
            for _col in ("greek_badness_score", "greek_latin_percentage"):
                if _col in df_final.columns:
                    df_final[_col] = pd.to_numeric(df_final[_col], errors="coerce").round(3)
            if "polytonic_ratio" in df_final.columns:
                df_final["polytonic_ratio"] = df_final["polytonic_ratio"].round(2)

            # drop any leftover placeholder columns to avoid duplicates
            df_final.drop(columns=["badness_score", "percentage_latin"], errors="ignore", inplace=True)
            # ADD: Drop unwanted columns
            df_final.drop(columns=["greek_latin_percentage", "badness_before", "badness_after"], errors="ignore", inplace=True)

            # ensure no duplicate column names
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]

            def _collapse_measure(df: pd.DataFrame, base: str) -> None:
                cols = [col for col in df.columns if col == base or col.startswith(f"{base}_")]
                if not cols:
                    return
                collapsed = None
                for col in cols:
                    values = pd.to_numeric(df[col], errors="coerce")
                    collapsed = values if collapsed is None else collapsed.combine_first(values)
                df[base] = collapsed
                for col in cols:
                    if col != base:
                        df.drop(columns=col, inplace=True, errors="ignore")

            _collapse_measure(df_final, "char_count_no_comments")
            _collapse_measure(df_final, "page_count")

            if "char_count_no_comments" in df_final.columns:
                df_final["char_count_no_comments"] = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce")
            if "page_count" in df_final.columns:
                df_final["page_count"] = pd.to_numeric(df_final["page_count"], errors="coerce")

            df_final["filter"] = "ok"
            df_final["needs_ocr"] = False
            if "is_empty" in df_final.columns:
                df_final["is_empty"] = df_final["is_empty"].fillna(False).astype(bool)
            else:
                df_final["is_empty"] = False

            filename_series = df_final.get("filename")
            if filename_series is None:
                pdf_mask = pd.Series(False, index=df_final.index)
            else:
                pdf_mask = filename_series.astype(str).str.lower().str.endswith(".pdf")
                pdf_mask = pdf_mask.fillna(False)

            def _append_reason(mask: pd.Series, reason: str, *, requires_ocr: bool) -> None:
                if df_final.empty:
                    return
                if not isinstance(mask, pd.Series):
                    mask = pd.Series(mask, index=df_final.index)
                mask = mask.fillna(False)
                applicable = (mask & pdf_mask).fillna(False)
                if not bool(applicable.any()):
                    return
                current = df_final.loc[applicable, "filter"].astype(str)

                def _merge_reason(value: str) -> str:
                    if value == "ok" or not value:
                        return reason
                    parts = [part for part in value.split(";") if part]
                    if reason not in parts:
                        parts.append(reason)
                    return ";".join(parts)

                df_final.loc[applicable, "filter"] = current.apply(_merge_reason)
                if requires_ocr:
                    needs_targets = applicable
                    if "ocr_success" in df_final.columns:
                        success_mask = df_final["ocr_success"].fillna(False)
                        needs_targets = needs_targets & ~success_mask
                    df_final.loc[needs_targets, "needs_ocr"] = True

            try:
                empty_threshold_int = int(empty_char_threshold) if empty_char_threshold is not None else 0
            except Exception:
                empty_threshold_int = 0
            if empty_threshold_int < 0:
                empty_threshold_int = 0
            try:
                min_pages = int(empty_min_pages) if empty_min_pages is not None else 0
            except Exception:
                min_pages = 0
            if min_pages < 0:
                min_pages = 0

            raw_moj = df_final.get("mojibake_badness_score")
            if isinstance(raw_moj, pd.Series):
                mojibake_series = pd.to_numeric(raw_moj, errors="coerce")
            else:
                mojibake_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
            if mojibake_series.notna().any():
                # Token policy: every OCR-trigger writes a filter tag.
                # Keep original label for compatibility with tests and downstream tools.
                _append_reason(mojibake_series > 0.1, "mojibake>0.1", requires_ocr=True)

            raw_gr = df_final.get("greek_badness_score")
            if isinstance(raw_gr, pd.Series):
                greek_series = pd.to_numeric(raw_gr, errors="coerce")
            else:
                greek_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
            if greek_series.notna().any():
                # Greek script gate: keep threshold (>60) as-is.
                # Use canonical token expected by tests and downstream tools.
                _append_reason(greek_series > 60, "non_greek_text", requires_ocr=True)

            if "char_count_no_comments" in df_final.columns:
                # Preserve NaN to avoid treating unknown counts as zero
                char_series = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce")
                page_series_raw = df_final.get("page_count")
                if page_series_raw is not None:
                    page_series = pd.to_numeric(page_series_raw, errors="coerce")
                else:
                    page_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
                page_series = page_series.fillna(min_pages if min_pages else 0)

                zero_mask = char_series <= 0
                zero_pdf = (zero_mask & pdf_mask).fillna(False)
                if bool(zero_pdf.any()):
                    df_final.loc[zero_pdf.index, "is_empty"] = df_final.loc[zero_pdf.index, "is_empty"] | zero_pdf
                if empty_threshold_int == 0:
                    zeros = int(zero_pdf.sum())
                    if zeros:
                        self.logger.info("Empty text check: %d files have zero characters", zeros)
                    # Strict-empty safeguard: rename token to "is_empty" and trigger OCR.
                    _append_reason(zero_pdf, "is_empty", requires_ocr=True)
                elif empty_threshold_int > 0:
                    low_mask = char_series < empty_threshold_int
                    long_mask = page_series >= max(1, min_pages)
                    _append_reason(low_mask & long_mask, f"empty_text<{empty_threshold_int}", requires_ocr=True)
                    if min_pages > 0:
                        _append_reason(low_mask & ~long_mask, f"empty_text<{empty_threshold_int}_short", requires_ocr=False)
                    total_low = int(low_mask.fillna(False).sum())
                    long_low = int((low_mask & long_mask).fillna(False).sum())
                    self.logger.info(
                        "Empty text check: %d files below %d chars; %d have >= %d pages",
                        total_low,
                        empty_threshold_int,
                        long_low,
                        min_pages,
                    )

            df_final["needs_ocr"] = df_final["needs_ocr"].fillna(False).astype(bool)

            # persist cleaned parquet
            parquet_schema.write_metadata_parquet(df_final, parquet_path)
            if drop_bad:
                good_df = df_final[df_final["needs_ocr"] == False]
                filenames = good_df.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
                self.logger.info(f"After filtering, {len(self.good_files)} good files remain")
            else:
                filenames = df_final.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
        else:
            self.good_files = []

        # After cleaning, point markdown_dir to cleaned files for downstream stages
        if write_cleaned_files:
            self.markdown_dir = self.cleaned_markdown_dir

    def clean_ocr(
        self,
        input_dir: Union[str, Path] = None,
        num_threads: int = None,
        drop_bad: bool = False,
        *,
        min_repeat_run: int = 6,
        write_cleaned_files: bool = True,
        write_debug_files: bool = False,
        debug_output_dir: Union[str, Path, None] = None,
        min_progress_steps: int = 10,
        min_repeat_steps: int = 8,
        min_same_digit_steps: int = 10,
        word_rep_threshold: int = 4,
        word_min_period: int = 3,
        word_window: int = 96,
    ) -> None:
        """Clean OCR markdown with the shared page loop and update OCR-noise metrics.

        The OCR profile keeps the existing canonical script metrics columns
        (`percentage_greek`, `latin_percentage`, `polytonic_ratio`) and adds
        OCR-specific noise diagnostics. The same combined page analyzer drives
        both clean and debug outputs:
        - clean mode writes pipeline-ready markdown to ``self.cleaned_markdown_dir``
        - debug mode writes annotated markdown and a structured match index under
          ``debug_output_dir`` (default: ``self.output_dir / "debug"``)
        """
        from glossapi.parquet_schema import ParquetSchema

        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        parquet_schema = ParquetSchema({"url_column": self.url_column})
        parquet_path = self._resolve_clean_metrics_parquet(parquet_schema)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        noise_mod = self._load_rust_extension(
            "glossapi_rs_noise",
            "rust/glossapi_rs_noise/Cargo.toml",
            required_attrs=(
                "score_markdown_directory_ocr_profile",
                "find_numeric_debug_page_spans",
                "evaluate_page_character_noise",
            ),
        )
        n_threads = int(num_threads or os.cpu_count() or 4)
        render_workers = _default_combined_ocr_render_workers(
            noise_mod=noise_mod,
            requested_workers=None,
            max_workers=n_threads,
        )
        md_files = sorted(input_dir.glob("*.md"))
        debug_dir: Optional[Path] = None
        debug_manifest_path: Optional[Path] = None
        debug_page_metrics_path: Optional[Path] = None
        debug_match_index_path: Optional[Path] = None
        debug_summary_path: Optional[Path] = None
        if write_debug_files:
            debug_dir = Path(debug_output_dir) if debug_output_dir is not None else (self.output_dir / "debug")
            if debug_dir.exists():
                shutil.rmtree(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_manifest_path = debug_dir / "manifest.jsonl"
            debug_page_metrics_path = debug_dir / "page_metrics.jsonl"
            debug_match_index_path = debug_dir / "match_index.jsonl"
            debug_summary_path = debug_dir / "summary.json"

        if write_cleaned_files:
            if self.cleaned_markdown_dir.exists():
                shutil.rmtree(self.cleaned_markdown_dir)
            self.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)

        if write_cleaned_files or write_debug_files:
            mode_label = "clean+debug" if write_cleaned_files and write_debug_files else ("debug" if write_debug_files else "clean")
            self.logger.info(
                "Running shared OCR %s loop over %d markdown files (workers=%d)…",
                mode_label,
                len(md_files),
                render_workers,
            )

            if write_debug_files:
                rows: List[Dict[str, Any]] = []
                total_page_times: List[float] = []
                table_page_times: List[float] = []
                numeric_page_times: List[float] = []
                latex_page_times: List[float] = []
                shared_page_times: List[float] = []
                hybrid_page_times: List[float] = []
                char_eval_times: List[float] = []
                bad_char_ratios: List[float] = []

                def _consume_debug_doc_result(
                    doc_result: Dict[str, Any],
                    *,
                    page_metrics_handle: Any,
                    match_index_handle: Any,
                ) -> None:
                    rows.append(dict(doc_result["row"]))
                    for page_row in doc_result["page_metric_rows"]:
                        page_metrics_handle.write(json.dumps(page_row, ensure_ascii=False))
                        page_metrics_handle.write("\n")
                        total_page_times.append(float(page_row["total_page_seconds"]))
                        table_page_times.append(float(page_row["table_seconds"]))
                        numeric_page_times.append(float(page_row["numeric_seconds"]))
                        latex_page_times.append(float(page_row["latex_seconds"]))
                        hybrid_page_times.append(float(page_row["hybrid_seconds"]))
                        shared_page_times.append(float(page_row["shared_repeat_seconds"]))
                        char_eval_times.append(float(page_row["char_eval_seconds"]))
                        bad_char_ratios.append(float(page_row["bad_char_ratio"]))
                    for match_row in doc_result["match_index_rows"]:
                        match_index_handle.write(json.dumps(match_row, ensure_ascii=False))
                        match_index_handle.write("\n")

                if _can_use_combined_ocr_process_pool(noise_mod, render_workers):
                    if write_cleaned_files:
                        jobs = [
                            (
                                str(source_path),
                                str(self.cleaned_markdown_dir / source_path.name),
                                str(debug_dir / source_path.name),
                                int(min_progress_steps),
                                int(min_repeat_steps),
                                int(min_same_digit_steps),
                                int(word_rep_threshold),
                                int(word_min_period),
                                int(word_window),
                            )
                            for source_path in md_files
                        ]
                    else:
                        jobs = [
                            (
                                str(source_path),
                                str(debug_dir / source_path.name),
                                int(min_progress_steps),
                                int(min_repeat_steps),
                                int(min_same_digit_steps),
                                int(word_rep_threshold),
                                int(word_min_period),
                                int(word_window),
                            )
                            for source_path in md_files
                        ]
                    with debug_page_metrics_path.open("w", encoding="utf-8") as page_metrics_handle, debug_match_index_path.open("w", encoding="utf-8") as match_index_handle:
                        with _combined_ocr_process_pool_warning_ctx():
                            with ProcessPoolExecutor(
                                max_workers=render_workers,
                                mp_context=mp.get_context("fork"),
                                initializer=_init_combined_ocr_worker,
                            ) as executor:
                                if write_cleaned_files:
                                    iterator = executor.map(_process_combined_ocr_dual_document_job, jobs)
                                else:
                                    iterator = executor.map(_process_combined_ocr_debug_document_job, jobs)
                                for doc_result in iterator:
                                    _consume_debug_doc_result(
                                        doc_result,
                                        page_metrics_handle=page_metrics_handle,
                                        match_index_handle=match_index_handle,
                                    )
                else:
                    if write_cleaned_files:
                        def _run_dual_doc(source_path: Path) -> Dict[str, Any]:
                            return _process_combined_ocr_document(
                                source_path,
                                clean_output_path=self.cleaned_markdown_dir / source_path.name,
                                debug_output_path=debug_dir / source_path.name,
                                noise_mod=noise_mod,
                                min_progress_steps=int(min_progress_steps),
                                min_repeat_steps=int(min_repeat_steps),
                                min_same_digit_steps=int(min_same_digit_steps),
                                word_rep_threshold=int(word_rep_threshold),
                                word_min_period=int(word_min_period),
                                word_window=int(word_window),
                                include_page_metrics=True,
                                include_match_index=True,
                            )
                        run_doc = _run_dual_doc
                    else:
                        def _run_debug_doc(source_path: Path) -> Dict[str, Any]:
                            return _process_combined_ocr_debug_document(
                                source_path,
                                debug_dir / source_path.name,
                                noise_mod=noise_mod,
                                min_progress_steps=int(min_progress_steps),
                                min_repeat_steps=int(min_repeat_steps),
                                min_same_digit_steps=int(min_same_digit_steps),
                                word_rep_threshold=int(word_rep_threshold),
                                word_min_period=int(word_min_period),
                                word_window=int(word_window),
                            )
                        run_doc = _run_debug_doc

                    with debug_page_metrics_path.open("w", encoding="utf-8") as page_metrics_handle, debug_match_index_path.open("w", encoding="utf-8") as match_index_handle:
                        with ThreadPoolExecutor(max_workers=render_workers) as executor:
                            for doc_result in executor.map(run_doc, md_files):
                                _consume_debug_doc_result(
                                    doc_result,
                                    page_metrics_handle=page_metrics_handle,
                                    match_index_handle=match_index_handle,
                                )

                with debug_manifest_path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False))
                        handle.write("\n")

                debug_summary = {
                    "doc_count": len(rows),
                    "matched_doc_count": sum(1 for row in rows if int(row["matched_page_count"]) > 0),
                    "matched_page_count": int(sum(int(row["matched_page_count"]) for row in rows)),
                    "match_count": int(sum(int(row.get("match_count", 0)) for row in rows)),
                    "table_match_count": int(sum(int(row["table_match_count"]) for row in rows)),
                    "numeric_match_count": int(sum(int(row["numeric_match_count"]) for row in rows)),
                    "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
                    "hybrid_match_count": int(sum(int(row["hybrid_match_count"]) for row in rows)),
                    "word_match_count": int(sum(int(row["word_match_count"]) for row in rows)),
                    "word_rep_threshold": int(word_rep_threshold),
                    "word_min_period": int(word_min_period),
                    "word_window": int(word_window),
                    "total_page_seconds": _summarize_metric(total_page_times),
                    "table_seconds": _summarize_metric(table_page_times),
                    "numeric_seconds": _summarize_metric(numeric_page_times),
                    "latex_seconds": _summarize_metric(latex_page_times),
                    "hybrid_seconds": _summarize_metric(hybrid_page_times),
                    "shared_repeat_seconds": _summarize_metric(shared_page_times),
                    "char_eval_seconds": _summarize_metric(char_eval_times),
                    "bad_char_ratio": _summarize_metric(bad_char_ratios),
                }
                debug_summary_path.write_text(json.dumps(debug_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                if _can_use_combined_ocr_process_pool(noise_mod, render_workers):
                    jobs = [
                        (
                            str(source_path),
                            str(self.cleaned_markdown_dir / source_path.name),
                            int(min_progress_steps),
                            int(min_repeat_steps),
                            int(min_same_digit_steps),
                            int(word_rep_threshold),
                            int(word_min_period),
                            int(word_window),
                        )
                        for source_path in md_files
                    ]
                    with _combined_ocr_process_pool_warning_ctx():
                        with ProcessPoolExecutor(
                            max_workers=render_workers,
                            mp_context=mp.get_context("fork"),
                            initializer=_init_combined_ocr_worker,
                        ) as executor:
                            list(executor.map(_process_combined_ocr_clean_document_job, jobs))
                else:
                    def _run_clean_doc(source_path: Path) -> None:
                        _process_combined_ocr_clean_document(
                            source_path,
                            self.cleaned_markdown_dir / source_path.name,
                            noise_mod=noise_mod,
                            min_progress_steps=int(min_progress_steps),
                            min_repeat_steps=int(min_repeat_steps),
                            min_same_digit_steps=int(min_same_digit_steps),
                            word_rep_threshold=int(word_rep_threshold),
                            word_min_period=int(word_min_period),
                            word_window=int(word_window),
                        )

                    with ThreadPoolExecutor(max_workers=render_workers) as executor:
                        list(executor.map(_run_clean_doc, md_files))


        self.logger.info(
            "Scoring OCR markdown files with glossapi_rs_noise OCR profile on %d markdown files…",
            len(md_files),
        )

        results = noise_mod.score_markdown_directory_ocr_profile(
            str(input_dir),
            n_threads,
            int(min_repeat_run),
        )
        df_updates = pd.DataFrame(list(results))
        if df_updates.empty:
            self.good_files = []
            self.logger.info("OCR cleaning found no markdown files under %s", input_dir)
            return

        df_updates["filename"] = df_updates["path"].apply(
            lambda value: f"{Path(str(value)).stem}.pdf"
        )
        df_updates["polytonic_ratio"] = pd.to_numeric(
            df_updates["polytonic_ratio"], errors="coerce"
        ).round(2)
        df_updates["percentage_greek"] = pd.to_numeric(
            df_updates["percentage_greek"], errors="coerce"
        ).round(3)
        df_updates["latin_percentage"] = pd.to_numeric(
            df_updates["latin_percentage"], errors="coerce"
        ).round(3)
        df_updates["ocr_repeat_suspicious_line_ratio"] = pd.to_numeric(
            df_updates["ocr_repeat_suspicious_line_ratio"], errors="coerce"
        ).round(4)
        df_updates["ocr_noise_flags"] = (
            df_updates["ocr_noise_flags"].fillna("").astype(str)
        )

        update_columns = [
            "filename",
            "percentage_greek",
            "latin_percentage",
            "polytonic_ratio",
            "ocr_noise_suspect",
            "ocr_noise_flags",
            "ocr_repeat_phrase_run_max",
            "ocr_repeat_line_run_max",
            "ocr_repeat_suspicious_line_count",
            "ocr_repeat_suspicious_line_ratio",
        ]

        df = self._load_metrics_dataframe(parquet_path, df_updates.get("filename"))
        self._ensure_metric_columns(
            df,
            {
                "filter": "ok",
                "percentage_greek": pd.NA,
                "latin_percentage": pd.NA,
                "polytonic_ratio": pd.NA,
                "ocr_noise_suspect": False,
                "ocr_noise_flags": "",
                "ocr_repeat_phrase_run_max": pd.NA,
                "ocr_repeat_line_run_max": pd.NA,
                "ocr_repeat_suspicious_line_count": pd.NA,
                "ocr_repeat_suspicious_line_ratio": pd.NA,
            },
        )
        df = self._merge_metric_dataframe(df, df_updates[update_columns])

        if "filter" not in df.columns:
            df["filter"] = "ok"
        else:
            df["filter"] = df["filter"].fillna("ok").astype(str)

        suspect_mask = df["ocr_noise_suspect"].fillna(False).astype(bool)
        if bool(suspect_mask.any()):
            current = df.loc[suspect_mask, "filter"].astype(str)

            def _append_ocr_noise(value: str) -> str:
                if value == "ok" or not value:
                    return "ocr_noise"
                tokens = [token for token in value.split(";") if token]
                if "ocr_noise" not in tokens:
                    tokens.append("ocr_noise")
                return ";".join(tokens)

            df.loc[suspect_mask, "filter"] = current.apply(_append_ocr_noise)

        parquet_schema.write_metadata_parquet(df, parquet_path)
        self.logger.info("OCR metrics updated in %s", parquet_path)

        filenames = df.get("filename", pd.Series(dtype=str))
        if drop_bad:
            good_df = df[~df["ocr_noise_suspect"].fillna(False).astype(bool)]
            filenames = good_df.get("filename", pd.Series(dtype=str))
            self.logger.info(
                "After OCR filtering, %d good files remain",
                len(filenames.dropna()),
            )
        self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
        if write_cleaned_files:
            self.markdown_dir = self.cleaned_markdown_dir

    def clean_ocr_debug(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        num_threads: int = None,
        *,
        min_repeat_run: int = 6,
        max_pages: Optional[int] = 1000,
        sample_seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Export page-level OCR debug files for repeated-pattern matches.

        Only pages that contain OCR repetition matches are exported. Each output page
        contains inline `<match of type ...>...</match>` tags around the matched spans.
        """
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        noise_mod = self._load_rust_extension(
            "glossapi_rs_noise",
            "rust/glossapi_rs_noise/Cargo.toml",
            required_attrs=("export_ocr_match_debug_pages",),
        )
        n_threads = int(num_threads or os.cpu_count() or 4)
        self.logger.info(
            "Exporting OCR debug matches from %s into %s with glossapi_rs_noise…",
            input_dir,
            output_dir,
        )

        rows = list(
            noise_mod.export_ocr_match_debug_pages(
                str(input_dir),
                str(output_dir),
                n_threads,
                int(min_repeat_run),
                None if max_pages is None else int(max_pages),
                int(sample_seed),
            )
        )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False))
                handle.write("\n")

        self.logger.info(
            "Exported %d OCR debug pages with matches to %s",
            len(rows),
            output_dir,
        )
        return [dict(row) for row in rows]

    def clean_ocr_numeric_debug(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        num_threads: int = None,
        *,
        min_progress_steps: int = 10,
        min_repeat_steps: int = 8,
        min_same_digit_steps: int = 10,
        max_pages: Optional[int] = 1000,
        sample_seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Export page-level OCR debug files for numeric-only collapse patterns."""
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        noise_mod = self._load_rust_extension(
            "glossapi_rs_noise",
            "rust/glossapi_rs_noise/Cargo.toml",
            required_attrs=("export_numeric_match_debug_pages",),
        )
        n_threads = int(num_threads or os.cpu_count() or 4)
        self.logger.info(
            "Exporting OCR numeric debug matches from %s into %s with glossapi_rs_noise…",
            input_dir,
            output_dir,
        )

        rows = list(
            noise_mod.export_numeric_match_debug_pages(
                str(input_dir),
                str(output_dir),
                n_threads,
                int(min_progress_steps),
                int(min_repeat_steps),
                int(min_same_digit_steps),
                None if max_pages is None else int(max_pages),
                int(sample_seed),
            )
        )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False))
                handle.write("\n")

        self.logger.info(
            "Exported %d OCR numeric debug pages with matches to %s",
            len(rows),
            output_dir,
        )
        return [dict(row) for row in rows]

    def clean_token_category_debug(
        self,
        output_dir: Union[str, Path],
        category_specs_path: Union[str, Path],
        input_dir: Union[str, Path] = None,
        num_threads: int = None,
        *,
        max_pages: Optional[int] = 1000,
        sample_seed: int = 0,
        synthetic_page_target_chars: int = 4000,
        synthetic_page_min_header_chars: int = 1200,
        synthetic_page_hard_max_chars: int = 6000,
    ) -> List[Dict[str, Any]]:
        """Export synthetic-page debug files for token/category review experiments.

        This is the debug substrate for token-noise and normalization review work.
        It mirrors the OCR debug workflow style: Rust-backed matching, annotated
        debug pages, manifest output, and a compact summary for later review steps.
        """
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        page_metrics_path = output_dir / "page_metrics.jsonl"
        if page_metrics_path.exists():
            page_metrics_path.unlink()
        match_index_path = output_dir / "match_index.jsonl"
        if match_index_path.exists():
            match_index_path.unlink()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

        category_specs_path = Path(category_specs_path)
        noise_mod = self._load_rust_extension(
            "glossapi_rs_noise",
            "rust/glossapi_rs_noise/Cargo.toml",
            required_attrs=("export_token_category_debug_pages",),
        )
        n_threads = int(num_threads or os.cpu_count() or 4)
        self.logger.info(
            "Exporting token category debug pages from %s into %s using specs %s with glossapi_rs_noise…",
            input_dir,
            output_dir,
            category_specs_path,
        )

        rows = list(
            noise_mod.export_token_category_debug_pages(
                str(input_dir),
                str(output_dir),
                str(category_specs_path),
                n_threads,
                None if max_pages is None else int(max_pages),
                int(sample_seed),
                int(synthetic_page_target_chars),
                int(synthetic_page_min_header_chars),
                int(synthetic_page_hard_max_chars),
            )
        )

        manifest_rows: List[Dict[str, Any]] = []
        page_metric_rows: List[Dict[str, Any]] = []
        match_index_rows: List[Dict[str, Any]] = []
        category_page_counter: Counter[str] = Counter()
        category_match_counter: Counter[str] = Counter()
        pattern_family_page_counter: Counter[str] = Counter()
        pattern_family_match_counter: Counter[str] = Counter()
        page_kind_counter: Counter[str] = Counter()

        for raw_row in rows:
            row = dict(raw_row)
            page_text = str(row.pop("page_text", ""))
            matches = json.loads(str(row.pop("matches_json", "[]")))
            manifest_rows.append(row)
            page_metric_rows.append(_build_token_category_page_metric_row(row, matches))
            match_index_rows.extend(
                _build_token_category_match_index_rows(page_text, matches, page_row=row)
            )
            page_kind_counter[str(row.get("page_kind", ""))] += 1
            for category in str(row.get("match_categories", "")).split(","):
                if category:
                    category_page_counter[category] += 1
            for family in str(row.get("match_pattern_families", "")).split(","):
                if family:
                    pattern_family_page_counter[family] += 1
            for match in matches:
                for category in list(match.get("categories") or []):
                    category_match_counter[str(category)] += 1
                for family in list(match.get("pattern_families") or []):
                    pattern_family_match_counter[str(family)] += 1

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        with page_metrics_path.open("w", encoding="utf-8") as handle:
            for row in page_metric_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
        with match_index_path.open("w", encoding="utf-8") as handle:
            for row in match_index_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        summary = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "category_specs_path": str(category_specs_path),
            "page_count": len(manifest_rows),
            "match_count": int(len(match_index_rows)),
            "category_page_counts": dict(category_page_counter),
            "category_match_counts": dict(category_match_counter),
            "pattern_family_page_counts": dict(pattern_family_page_counter),
            "pattern_family_match_counts": dict(pattern_family_match_counter),
            "page_kind_counts": dict(page_kind_counter),
            "synthetic_page_target_chars": int(synthetic_page_target_chars),
            "synthetic_page_min_header_chars": int(synthetic_page_min_header_chars),
            "synthetic_page_hard_max_chars": int(synthetic_page_hard_max_chars),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.logger.info(
            "Exported %d token category debug pages with matches to %s",
            len(manifest_rows),
            output_dir,
        )
        return manifest_rows

    def clean_ocr_numeric_word_debug_docs(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        *,
        max_docs: Optional[int] = 100,
        doc_offset: int = 0,
        doc_workers: Optional[int] = None,
        min_progress_steps: int = 10,
        min_repeat_steps: int = 8,
        min_same_digit_steps: int = 10,
        word_rep_threshold: int = 4,
        word_min_period: int = 3,
        word_window: int = 96,
    ) -> List[Dict[str, Any]]:
        """Annotate complete markdown documents with table, numeric, LaTeX, hybrid, then shared-repeat matches.

        Default repetition threshold for both word and LaTeX repeat detection is 4.
        """
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        page_metrics_path = output_dir / "page_metrics.jsonl"
        if page_metrics_path.exists():
            page_metrics_path.unlink()
        match_index_path = output_dir / "match_index.jsonl"
        if match_index_path.exists():
            match_index_path.unlink()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

        noise_mod = self._load_rust_extension(
            "glossapi_rs_noise",
            "rust/glossapi_rs_noise/Cargo.toml",
            required_attrs=("find_numeric_debug_page_spans", "evaluate_page_character_noise"),
        )

        all_source_paths = sorted(input_dir.glob("*.md"))
        doc_offset = max(0, int(doc_offset))
        if max_docs is not None:
            source_paths = all_source_paths[doc_offset : doc_offset + int(max_docs)]
        else:
            source_paths = all_source_paths[doc_offset:]
        render_workers = _default_combined_ocr_render_workers(
            noise_mod=noise_mod,
            requested_workers=doc_workers,
            max_workers=int(os.cpu_count() or 1),
        )

        self.logger.info(
            "Exporting combined OCR table+numeric+latex+hybrid+word debug docs from %s into %s for %d documents (offset=%d, workers=%d)",
            input_dir,
            output_dir,
            len(source_paths),
            doc_offset,
            render_workers,
        )

        rows: List[Dict[str, Any]] = []
        total_page_times: List[float] = []
        table_page_times: List[float] = []
        numeric_page_times: List[float] = []
        latex_page_times: List[float] = []
        shared_page_times: List[float] = []
        hybrid_page_times: List[float] = []
        char_eval_times: List[float] = []
        bad_char_ratios: List[float] = []
        def _consume_doc_result(
            doc_result: Dict[str, Any],
            *,
            page_metrics_handle: Any,
            match_index_handle: Any,
        ) -> None:
            rows.append(dict(doc_result["row"]))
            for page_row in doc_result["page_metric_rows"]:
                page_metrics_handle.write(json.dumps(page_row, ensure_ascii=False))
                page_metrics_handle.write("\n")
                total_page_times.append(float(page_row["total_page_seconds"]))
                table_page_times.append(float(page_row["table_seconds"]))
                numeric_page_times.append(float(page_row["numeric_seconds"]))
                latex_page_times.append(float(page_row["latex_seconds"]))
                hybrid_page_times.append(float(page_row["hybrid_seconds"]))
                shared_page_times.append(float(page_row["shared_repeat_seconds"]))
                char_eval_times.append(float(page_row["char_eval_seconds"]))
                bad_char_ratios.append(float(page_row["bad_char_ratio"]))
            for match_row in doc_result["match_index_rows"]:
                match_index_handle.write(json.dumps(match_row, ensure_ascii=False))
                match_index_handle.write("\n")
        if _can_use_combined_ocr_process_pool(noise_mod, render_workers):
            jobs = [
                (
                    str(source_path),
                    str(output_dir / source_path.name),
                    int(min_progress_steps),
                    int(min_repeat_steps),
                    int(min_same_digit_steps),
                    int(word_rep_threshold),
                    int(word_min_period),
                    int(word_window),
                )
                for source_path in source_paths
            ]
            iterator: Iterable[Dict[str, Any]]
            with page_metrics_path.open("w", encoding="utf-8") as page_metrics_handle, match_index_path.open("w", encoding="utf-8") as match_index_handle:
                with _combined_ocr_process_pool_warning_ctx():
                    with ProcessPoolExecutor(
                        max_workers=render_workers,
                        # Match the clean-mode executor policy so debug and
                        # clean keep the same performance shape and worker init.
                        mp_context=mp.get_context("fork"),
                        initializer=_init_combined_ocr_worker,
                    ) as executor:
                        iterator = executor.map(_process_combined_ocr_debug_document_job, jobs)
                        for doc_result in iterator:
                            _consume_doc_result(
                                doc_result,
                                page_metrics_handle=page_metrics_handle,
                                match_index_handle=match_index_handle,
                            )
        else:
            def _run_debug_doc(source_path: Path) -> Dict[str, Any]:
                return _process_combined_ocr_debug_document(
                    source_path,
                    output_dir / source_path.name,
                    noise_mod=noise_mod,
                    min_progress_steps=int(min_progress_steps),
                    min_repeat_steps=int(min_repeat_steps),
                    min_same_digit_steps=int(min_same_digit_steps),
                    word_rep_threshold=int(word_rep_threshold),
                    word_min_period=int(word_min_period),
                    word_window=int(word_window),
                )

            with page_metrics_path.open("w", encoding="utf-8") as page_metrics_handle, match_index_path.open("w", encoding="utf-8") as match_index_handle:
                with ThreadPoolExecutor(max_workers=render_workers) as executor:
                    for doc_result in executor.map(_run_debug_doc, source_paths):
                        _consume_doc_result(
                            doc_result,
                            page_metrics_handle=page_metrics_handle,
                            match_index_handle=match_index_handle,
                        )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        summary = {
            "doc_count": len(rows),
            "matched_doc_count": sum(1 for row in rows if int(row["matched_page_count"]) > 0),
            "matched_page_count": int(sum(int(row["matched_page_count"]) for row in rows)),
            "match_count": int(sum(int(row.get("match_count", 0)) for row in rows)),
            "table_match_count": int(sum(int(row["table_match_count"]) for row in rows)),
            "numeric_match_count": int(sum(int(row["numeric_match_count"]) for row in rows)),
            "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
            "hybrid_match_count": int(sum(int(row["hybrid_match_count"]) for row in rows)),
            "word_match_count": int(sum(int(row["word_match_count"]) for row in rows)),
            "word_rep_threshold": int(word_rep_threshold),
            "word_min_period": int(word_min_period),
            "word_window": int(word_window),
            "total_page_seconds": _summarize_metric(total_page_times),
            "table_seconds": _summarize_metric(table_page_times),
            "numeric_seconds": _summarize_metric(numeric_page_times),
            "latex_seconds": _summarize_metric(latex_page_times),
            "hybrid_seconds": _summarize_metric(hybrid_page_times),
            "shared_repeat_seconds": _summarize_metric(shared_page_times),
            "char_eval_seconds": _summarize_metric(char_eval_times),
            "bad_char_ratio": _summarize_metric(bad_char_ratios),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info(
            "Exported %d combined OCR debug docs to %s",
            len(rows),
            output_dir,
        )
        return rows

    def clean_ocr_hybrid_debug(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        *,
        max_docs: Optional[int] = 100,
        doc_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Export only matched pages for local hybrid numbered repetitions."""
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

        all_source_paths = sorted(list(Path(input_dir).glob("*.md")) + list(Path(input_dir).glob("*.txt")))
        doc_offset = max(0, int(doc_offset))
        if max_docs is not None:
            source_paths = all_source_paths[doc_offset : doc_offset + int(max_docs)]
        else:
            source_paths = all_source_paths[doc_offset:]

        self.logger.info(
            "Exporting hybrid OCR debug pages from %s into %s for %d documents (offset=%d)",
            input_dir,
            output_dir,
            len(source_paths),
            doc_offset,
        )

        rows: List[Dict[str, Any]] = []
        page_times: List[float] = []

        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            pages = text.split(PAGE_SPLIT_MARKER)
            for page_index, page in enumerate(pages, start=1):
                page_start = time.perf_counter()
                hybrid_spans = _find_hybrid_numbered_repeat_spans(page, blocked_spans=[])
                page_elapsed = time.perf_counter() - page_start
                page_times.append(page_elapsed)
                if not hybrid_spans:
                    continue

                annotated_page, page_types, _, _, _, _, _ = _annotate_page_with_labeled_spans(
                    page,
                    hybrid_spans,
                )
                hybrid_count = _count_hybrid_matches_in_page(page, hybrid_spans)
                output_name = f"{source_path.stem}__debug_page_{page_index:05d}.md"
                output_path = output_dir / output_name
                output_path.write_text(annotated_page, encoding="utf-8")
                rows.append(
                    {
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "source_stem": source_path.stem,
                        "base_stem": canonical_stem(source_path.stem),
                        "page_number": page_index,
                        "page_index_in_file": page_index,
                        "hybrid_match_count": hybrid_count,
                        "match_types": ",".join(page_types),
                        "page_seconds": page_elapsed,
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        summary = {
            "doc_count": len(source_paths),
            "matched_page_count": len(rows),
            "hybrid_match_count": int(sum(int(row["hybrid_match_count"]) for row in rows)),
            "page_seconds": _summarize_metric(page_times),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info(
            "Exported %d hybrid OCR debug pages to %s",
            len(rows),
            output_dir,
        )
        return rows

    def clean_ocr_latex_slot_progression_debug(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        *,
        max_docs: Optional[int] = 1000,
        doc_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Export only matched pages for local LaTeX slot-progression runs."""
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

        all_source_paths = sorted(list(Path(input_dir).glob("*.md")) + list(Path(input_dir).glob("*.txt")))
        doc_offset = max(0, int(doc_offset))
        if max_docs is not None:
            source_paths = all_source_paths[doc_offset : doc_offset + int(max_docs)]
        else:
            source_paths = all_source_paths[doc_offset:]

        self.logger.info(
            "Exporting LaTeX slot-progression debug pages from %s into %s for %d documents (offset=%d)",
            input_dir,
            output_dir,
            len(source_paths),
            doc_offset,
        )

        rows: List[Dict[str, Any]] = []
        page_times: List[float] = []

        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            pages = text.split(PAGE_SPLIT_MARKER)
            for page_index, page in enumerate(pages, start=1):
                page_start = time.perf_counter()
                latex_spans = _find_latex_slot_progression_spans(page, blocked_spans=[])
                page_elapsed = time.perf_counter() - page_start
                page_times.append(page_elapsed)
                if not latex_spans:
                    continue

                annotated_page, page_types, _, _, latex_count, _, _ = _annotate_page_with_labeled_spans(
                    page,
                    latex_spans,
                )
                output_name = f"{source_path.stem}__debug_page_{page_index:05d}.md"
                output_path = output_dir / output_name
                output_path.write_text(annotated_page, encoding="utf-8")
                rows.append(
                    {
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "source_stem": source_path.stem,
                        "base_stem": canonical_stem(source_path.stem),
                        "page_number": page_index,
                        "page_index_in_file": page_index,
                        "latex_match_count": latex_count,
                        "match_types": ",".join(page_types),
                        "page_seconds": page_elapsed,
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        summary = {
            "doc_count": len(source_paths),
            "matched_page_count": len(rows),
            "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
            "page_seconds": _summarize_metric(page_times),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info(
            "Exported %d LaTeX slot-progression debug pages to %s",
            len(rows),
            output_dir,
        )
        return rows

    def clean_ocr_latex_debug(
        self,
        output_dir: Union[str, Path],
        input_dir: Union[str, Path] = None,
        *,
        max_docs: Optional[int] = 1000,
        doc_offset: int = 0,
        word_rep_threshold: int = 4,
        word_min_period: int = 3,
        word_window: int = 96,
    ) -> List[Dict[str, Any]]:
        """Export only matched pages for all LaTeX repeat classes."""
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in output_dir.glob("*.md"):
            stale.unlink()
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary_path.unlink()

        all_source_paths = sorted(list(Path(input_dir).glob("*.md")) + list(Path(input_dir).glob("*.txt")))
        doc_offset = max(0, int(doc_offset))
        if max_docs is not None:
            source_paths = all_source_paths[doc_offset : doc_offset + int(max_docs)]
        else:
            source_paths = all_source_paths[doc_offset:]

        self.logger.info(
            "Exporting LaTeX debug pages from %s into %s for %d documents (offset=%d)",
            input_dir,
            output_dir,
            len(source_paths),
            doc_offset,
        )

        rows: List[Dict[str, Any]] = []
        page_times: List[float] = []

        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            pages = text.split(PAGE_SPLIT_MARKER)
            for page_index, page in enumerate(pages, start=1):
                page_start = time.perf_counter()
                latex_spans = _find_latex_repeat_spans(
                    page,
                    blocked_spans=[],
                    rep_threshold=int(word_rep_threshold),
                    min_period=int(word_min_period),
                    window=int(word_window),
                )
                page_elapsed = time.perf_counter() - page_start
                page_times.append(page_elapsed)
                if not latex_spans:
                    continue

                annotated_page, page_types, _, _, latex_count, _, _ = _annotate_page_with_labeled_spans(
                    page,
                    latex_spans,
                )
                output_name = f"{source_path.stem}__debug_page_{page_index:05d}.md"
                output_path = output_dir / output_name
                output_path.write_text(annotated_page, encoding="utf-8")
                rows.append(
                    {
                        "source_path": str(source_path),
                        "output_path": str(output_path),
                        "source_stem": source_path.stem,
                        "base_stem": canonical_stem(source_path.stem),
                        "page_number": page_index,
                        "page_index_in_file": page_index,
                        "latex_match_count": latex_count,
                        "match_types": ",".join(page_types),
                        "page_seconds": page_elapsed,
                    }
                )

        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

        summary = {
            "doc_count": len(source_paths),
            "matched_page_count": len(rows),
            "latex_match_count": int(sum(int(row["latex_match_count"]) for row in rows)),
            "word_rep_threshold": int(word_rep_threshold),
            "word_min_period": int(word_min_period),
            "word_window": int(word_window),
            "page_seconds": _summarize_metric(page_times),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        self.logger.info(
            "Exported %d LaTeX debug pages to %s",
            len(rows),
            output_dir,
        )
        return rows

    def filter(self, *args, **kwargs):  # type: ignore[override]
        """Deprecated: use :py:meth:`clean` instead.  Retained for one release."""
        self.logger.warning("Corpus.filter() is deprecated – calling clean() instead")
        self.clean(*args, **kwargs)
