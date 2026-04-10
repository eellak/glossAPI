from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PAGE_SPLIT_MARKER = "<--- Page Split --->"
TABLE_BLOCK_RE = re.compile(r"(?is)<table\b.*?</table\s*>")
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


_TABLE_AUDIT_PATH = Path(__file__).with_name("table_markdown_audit.py")
_TABLE_AUDIT_SPEC = importlib.util.spec_from_file_location("table_markdown_audit_local", _TABLE_AUDIT_PATH)
assert _TABLE_AUDIT_SPEC and _TABLE_AUDIT_SPEC.loader
_TABLE_AUDIT_MODULE = importlib.util.module_from_spec(_TABLE_AUDIT_SPEC)
sys.modules[_TABLE_AUDIT_SPEC.name] = _TABLE_AUDIT_MODULE
_TABLE_AUDIT_SPEC.loader.exec_module(_TABLE_AUDIT_MODULE)
_expand_rows = _TABLE_AUDIT_MODULE._expand_rows
_parse_table_rows = _TABLE_AUDIT_MODULE._parse_table_rows


def _extract_review_html(review_text: str) -> str:
    return review_text.split("=== HTML ===\n", 1)[1].split("\n\n=== GITHUB_MD ===", 1)[0]


def _flatten_nonempty_cells(table_html: str) -> List[str]:
    parsed_rows, _ = _parse_table_rows(table_html)
    grid, _ = _expand_rows(parsed_rows)
    if not grid:
        return []
    nonempty: List[str] = []
    for row in grid:
        for cell in row:
            normalized = " ".join(cell.split())
            if any(ch.isalnum() for ch in normalized):
                nonempty.append(normalized)
    return nonempty


def _is_sentence_shell_candidate(review_row: Dict[str, object], table_html: str) -> Tuple[bool, Dict[str, int]]:
    nonempty_cells = _flatten_nonempty_cells(table_html)
    word_count = sum(len(WORD_RE.findall(cell)) for cell in nonempty_cells)
    max_cell_len = max((len(cell) for cell in nonempty_cells), default=0)
    metrics = {
        "nonempty_cell_count": len(nonempty_cells),
        "word_count": word_count,
        "max_cell_len": max_cell_len,
    }
    is_candidate = (
        bool(review_row.get("broken"))
        and "sparse_span_shell" in list(review_row.get("reasons", []))
        and len(nonempty_cells) == 1
        and word_count >= 6
        and max_cell_len >= 40
    )
    return is_candidate, metrics


def _find_table_page_context(
    source_path: Path,
    table_index_in_doc: int,
) -> Tuple[int, int, int, int, str, str, str]:
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    pages = text.split(PAGE_SPLIT_MARKER)
    seen = 0
    for page_idx, page in enumerate(pages):
        matches = list(TABLE_BLOCK_RE.finditer(page))
        if seen + len(matches) < table_index_in_doc:
            seen += len(matches)
            continue
        local_idx = table_index_in_doc - seen - 1
        match = matches[local_idx]
        prev_page = pages[page_idx - 1] if page_idx > 0 else ""
        curr_page = page
        next_page = pages[page_idx + 1] if page_idx + 1 < len(pages) else ""
        return page_idx, match.start(), match.end(), len(pages), prev_page, curr_page, next_page
    raise ValueError(f"Could not find table {table_index_in_doc} in {source_path}")


def _smart_join(before_text: str, inline_text: str, after_text: str) -> str:
    left = before_text.rstrip()
    right = after_text.lstrip()
    insertion = inline_text.strip()

    if left and not left.endswith(("\n", " ", "(", "[", "{", "“", "\"", "'")):
        if left[-1].isalnum() and insertion and insertion[0].isalnum():
            left += " "
    if right and not right.startswith(("\n", " ", ".", ",", ";", ":", "!", "?", ")", "]", "}", "”", "\"", "'")):
        if insertion and insertion[-1].isalnum() and right[0].isalnum():
            insertion += " "
    return left + insertion + right


def _context_fit_guess(before_text: str, inline_text: str, after_text: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    word_count = len(WORD_RE.findall(inline_text))
    if word_count < 6:
        reasons.append("short_inline_text")
    left_window = before_text[-4:]
    right_window = after_text[:4]
    left_blockish = (not before_text) or ("\n" in left_window) or before_text.endswith((" ", "\t"))
    right_blockish = (not after_text) or ("\n" in right_window) or after_text.startswith((" ", "\t"))
    if not left_blockish:
        reasons.append("not_block_isolated_left")
    if not right_blockish:
        reasons.append("not_block_isolated_right")
    fit = word_count >= 6 and left_blockish and right_blockish
    return fit, reasons


def _format_three_page_context(
    prev_page: str,
    curr_page: str,
    next_page: str,
    start: int,
    end: int,
    inline_text: str,
) -> Tuple[str, str]:
    tagged_current = curr_page[:start] + "[[[TABLE_START]]]" + curr_page[start:end] + "[[[TABLE_END]]]" + curr_page[end:]
    replaced_current = (
        curr_page[:start]
        + "[[[INLINE_TEXT_START]]]"
        + inline_text
        + "[[[INLINE_TEXT_END]]]"
        + curr_page[end:]
    )
    original_context = (
        f"=== PAGE -1 ===\n{prev_page}\n\n"
        f"=== PAGE 0 ===\n{tagged_current}\n\n"
        f"=== PAGE +1 ===\n{next_page}\n"
    )
    replaced_context = (
        f"=== PAGE -1 ===\n{prev_page}\n\n"
        f"=== PAGE 0 ===\n{replaced_current}\n\n"
        f"=== PAGE +1 ===\n{next_page}\n"
    )
    return original_context, replaced_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3-page context review files for sentence-in-table shells.")
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    audit_dir = args.audit_dir
    manifest_path = audit_dir / "manifest.jsonl"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir = output_dir / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.json"
    review_manifest_path = output_dir / "manifest.jsonl"
    if summary_path.exists():
        summary_path.unlink()
    if review_manifest_path.exists():
        review_manifest_path.unlink()
    for stale in contexts_dir.glob("*.txt"):
        stale.unlink()

    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    review_rows: List[Dict[str, object]] = []

    for row in rows:
        review_text = Path(str(row["output_path"])).read_text(encoding="utf-8")
        table_html = _extract_review_html(review_text)
        is_candidate, metrics = _is_sentence_shell_candidate(row, table_html)
        if not is_candidate:
            continue

        inline_text = _flatten_nonempty_cells(table_html)[0]
        page_idx, start, end, page_count, prev_page, curr_page, next_page = _find_table_page_context(
            Path(str(row["source_path"])),
            int(row["table_index_in_doc"]),
        )
        fit_guess, fit_reasons = _context_fit_guess(curr_page[:start], inline_text, curr_page[end:])
        original_context, replaced_context = _format_three_page_context(
            prev_page,
            curr_page,
            next_page,
            start,
            end,
            inline_text,
        )
        filename = f"{int(row['global_index']):05d}__{row['source_stem']}__table_{int(row['table_index_in_doc']):03d}.txt"
        output_path = contexts_dir / filename
        output_path.write_text(
            "\n".join(
                [
                    f"SOURCE_PATH: {row['source_path']}",
                    f"SOURCE_STEM: {row['source_stem']}",
                    f"TABLE_INDEX_IN_DOC: {row['table_index_in_doc']}",
                    f"GLOBAL_INDEX: {row['global_index']}",
                    f"PAGE_INDEX_ZERO_BASED: {page_idx}",
                    f"PAGE_NUMBER_ONE_BASED: {page_idx + 1}",
                    f"PAGE_COUNT: {page_count}",
                    f"FIT_GUESS: {fit_guess}",
                    f"FIT_REASONS: {', '.join(fit_reasons) if fit_reasons else 'none'}",
                    f"INLINE_TEXT_WORDS: {metrics['word_count']}",
                    f"INLINE_TEXT_CHARS: {metrics['max_cell_len']}",
                    "",
                    "=== INLINE_TEXT ===",
                    inline_text,
                    "",
                    "=== ORIGINAL_CONTEXT_3P ===",
                    original_context,
                    "",
                    "=== REPLACED_CONTEXT_3P ===",
                    replaced_context,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        review_rows.append(
            {
                "source_path": row["source_path"],
                "source_stem": row["source_stem"],
                "table_index_in_doc": row["table_index_in_doc"],
                "global_index": row["global_index"],
                "page_number": page_idx + 1,
                "page_count": page_count,
                "fit_guess": fit_guess,
                "fit_reasons": fit_reasons,
                "inline_text_words": metrics["word_count"],
                "inline_text_chars": metrics["max_cell_len"],
                "output_path": str(output_path),
            }
        )

    with review_manifest_path.open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    fit_counter = Counter(bool(row["fit_guess"]) for row in review_rows)
    reason_counter = Counter(reason for row in review_rows for reason in row["fit_reasons"])
    summary = {
        "audit_dir": str(audit_dir),
        "output_dir": str(output_dir),
        "candidate_count": len(review_rows),
        "fit_guess_count": fit_counter.get(True, 0),
        "fit_guess_rate": round((fit_counter.get(True, 0) / len(review_rows)), 4) if review_rows else 0.0,
        "fit_reason_counts": dict(reason_counter),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
