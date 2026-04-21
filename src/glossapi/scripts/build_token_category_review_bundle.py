from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


_SAFE_LABEL_RE = re.compile(r"[^a-z0-9._-]+")

# ---------------------------------------------------------------------------
# 2026-04-20 task-specific case-body renderers
# ---------------------------------------------------------------------------
#
# Each new review_mode declared in
# `src/glossapi/scripts/review_token_category_with_gemini.py` is paired with a
# renderer here that produces the case body the prompt-builder wraps with the
# cached preamble and the literal question block. The rendered body is a
# single string written to disk by `build_token_category_review_bundle`.
#
# Shared structure:
#   [MATCH_META]   — review_case_id, source, line indices, matched text
#   [EXISTING_CLEANER_PATTERNS] (optional, cleaning-ish modes)
#   [CONTEXT_BEFORE]   — up to N lines preceding the match, verbatim
#   [MATCH]            — the match line with the matched span wrapped in
#                        `<match type="..." family="...">…</match>`
#   [CONTEXT_AFTER]    — up to N lines following the match, verbatim
#   [CONTEXT_AFTER_NORMALIZATION] (optional, normalization-ish modes)
#                      — same window, matched span replaced with canonical

NEW_TASK_CASE_LINE_WINDOW = 100


_SEPARATOR_EXISTING_CLEANER_NOTE = (
    "The cleaner currently normalizes standalone separator lines via the Rust\n"
    "regex SEPARATOR_LINE_REGEX in glossapi_rs_cleaner::normalize:\n"
    "  ^[ \\t]*(?:-{4,}|_{4,}|\\*{4,}|={4,}|\\u{2014}{3,}|\\u{2015}{3,}|\\u{2500}{3,}|\\u{2550}{3,})[ \\t]*$\n"
    "Mixed-char runs (e.g. `---___`) are intentionally NOT matched. Dot-leader\n"
    "lines have a separate rule (target `.....`). This review confirms that the\n"
    "current regex + target `---` capture this case correctly."
)

NEW_TASK_CANONICAL_TARGETS: Dict[str, str] = {
    "separator_normalization": "---",
    # md_table_audit canonical is per-cell (GFM parser driven); rendered in
    # the TABLE_AFTER block rather than as a single string replacement.
    # slash_dash_classification is not a normalization; no canonical target.
    # page_noise_detection is per-page; no canonical target.
}

NEW_TASK_EXISTING_CLEANER: Dict[str, str] = {
    "separator_normalization": _SEPARATOR_EXISTING_CLEANER_NOTE,
}


def _read_source_lines(source_path: Path, max_bytes: int = 200 * 1024 * 1024) -> List[str]:
    """Read a markdown source file and return it split into lines.

    Guards against enormous files by size cap (default 200 MB); over-cap
    files return an empty list and the caller falls back to the matcher's
    char-based context.
    """
    try:
        if not source_path.is_file():
            return []
        if source_path.stat().st_size > max_bytes:
            return []
        return source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _extract_line_window(
    lines: Sequence[str],
    match_line_1indexed: int,
    *,
    context_lines: int = NEW_TASK_CASE_LINE_WINDOW,
) -> Tuple[List[str], str, List[str]]:
    """Return (before_lines, match_line_text, after_lines) for a 1-indexed line.

    If the line index is out of range returns ([], "", []).
    """
    zero = match_line_1indexed - 1
    if zero < 0 or zero >= len(lines):
        return ([], "", [])
    before = list(lines[max(0, zero - context_lines) : zero])
    after = list(lines[zero + 1 : min(len(lines), zero + 1 + context_lines)])
    return (before, lines[zero], after)


def _wrap_match_in_line(
    line: str,
    matched_text: str,
    *,
    category: str,
    pattern_family: str,
    replacement: Optional[str] = None,
) -> str:
    """Wrap the first occurrence of ``matched_text`` inside ``line`` with a
    `<match kind="…">…</match>` tag. The tag uses the compact `kind`
    attribute (see `_short_kind_for_category`) rather than the verbose
    matcher-internal `type`/`family` pair; keeps the line readable.

    When ``replacement`` is set, the tagged span carries the replacement
    text (shadow-normalized view). When ``matched_text`` can't be located
    in ``line`` the whole line is wrapped so the model always sees a tag.
    """
    kind = _short_kind_for_category(category)
    tag_attr = f' kind="{kind}"' if kind else ""
    open_tag = f"<match{tag_attr}>"
    close_tag = "</match>"
    payload = replacement if replacement is not None else matched_text
    if matched_text and matched_text in line:
        idx = line.index(matched_text)
        return f"{line[:idx]}{open_tag}{payload}{close_tag}{line[idx + len(matched_text):]}"
    return f"{open_tag}{payload if payload else line}{close_tag}"


def _build_v2_case_body(
    row: Mapping[str, Any],
    *,
    review_mode: str,
    canonical_target: Optional[str],
    existing_cleaner_note: Optional[str],
) -> str:
    """Render the 2026-04-20 line-windowed case body.

    Works for any per-match review mode. The caller supplies ``canonical_target``
    (to emit a shadow-normalized view) or ``existing_cleaner_note`` (to inject
    the existing cleaner inventory), or both, or neither.

    If the source file can't be read for line-windowed context, falls back to
    the matcher's char-based ``context_before`` / ``context_after`` fields so
    the case is still reviewable.
    """
    category = str(row.get("category", ""))
    pattern_family = str(row.get("pattern_family", ""))
    matched_text = str(row.get("matched_text", ""))
    source_path = Path(str(row.get("source_path", "")))
    source_stem = str(row.get("source_stem", ""))
    match_id = str(row.get("match_id", ""))
    start_line = int(row.get("start_line", 0) or 0)

    lines = _read_source_lines(source_path) if start_line > 0 else []
    before_lines, match_line_text, after_lines = (
        _extract_line_window(lines, start_line) if lines else ([], "", [])
    )
    fallback_mode = not lines or not match_line_text

    parts: List[str] = []
    parts.append("[MATCH_META]")
    parts.append(f"review_case_id: {category}::{match_id}")
    parts.append(f"source: {source_stem}")
    parts.append(f"matched_text: {json.dumps(matched_text, ensure_ascii=False)}")
    if canonical_target is not None:
        parts.append(f"canonical_target: {json.dumps(canonical_target, ensure_ascii=False)}")
    if fallback_mode:
        parts.append("note: source file unavailable; using matcher-supplied char windows")

    if existing_cleaner_note:
        parts.extend(["", "[EXISTING_CLEANER_PATTERNS]", existing_cleaner_note])

    # --- 2026-04-21 change: emit a single continuous passage with the
    # match wrapped inline via <match>…</match>, not three separate
    # CONTEXT_BEFORE / MATCH / CONTEXT_AFTER blocks. Reads as flowing
    # text the way the model would see it in the original document.
    if fallback_mode:
        before = str(row.get("context_before", ""))
        after = str(row.get("context_after", ""))
        tagged_match = _wrap_match_tag_only(matched_text, category=category)
        passage = f"{before}{tagged_match}{after}"
        parts.extend(
            ["", "[CONTEXT]  (continuous; match wrapped inline in <match>…</match>)",
             passage]
        )
        if canonical_target is not None:
            tagged_canonical = _wrap_match_tag_only(canonical_target, category=category)
            shadow_passage = f"{before}{tagged_canonical}{after}"
            parts.extend(
                ["", "[CONTEXT_AFTER_NORMALIZATION]  (same passage; match replaced by canonical target)",
                 shadow_passage]
            )
        return "\n".join(parts).rstrip() + "\n"

    # Line-windowed continuous passage. The match line sits in its
    # natural position between before_lines and after_lines.
    tagged_match_line = _wrap_match_in_line(
        match_line_text, matched_text, category=category, pattern_family=pattern_family
    )
    passage_lines = list(before_lines) + [tagged_match_line] + list(after_lines)
    passage = "\n".join(passage_lines)
    parts.extend(
        ["", f"[CONTEXT]  ({len(before_lines)} lines before + match line + "
             f"{len(after_lines)} lines after; continuous; match wrapped inline)",
         passage]
    )

    if canonical_target is not None:
        shadow_match_line = _wrap_match_in_line(
            match_line_text,
            matched_text,
            category=category,
            pattern_family=pattern_family,
            replacement=canonical_target,
        )
        shadow_passage_lines = list(before_lines) + [shadow_match_line] + list(after_lines)
        shadow_passage = "\n".join(shadow_passage_lines)
        parts.extend(
            ["", "[CONTEXT_AFTER_NORMALIZATION]  (same passage; match replaced by canonical target)",
             shadow_passage]
        )

    return "\n".join(parts).rstrip() + "\n"


def _wrap_match_tag_only(matched_text: str, *, category: str) -> str:
    """Compact match tag for fallback paths where we don't have a full line
    to show. Uses the short form `<match kind="...">…</match>` per the
    2026-04-21 steering note on match-tag verbosity.
    """
    kind = _short_kind_for_category(category)
    tag_attr = f' kind="{kind}"' if kind else ""
    return f"<match{tag_attr}>{matched_text}</match>"


def _short_kind_for_category(category: str) -> str:
    """Map matcher category to a compact `kind` label for the <match> tag.
    Keeps tag attributes legible (no 60-char pattern_family strings).
    """
    return {
        "separator_run_like": "separator",
        "markdown_table_separator_like": "md_table_sep",
        "table_border_ascii_art": "table_border",
        "dot_leader_like": "dot_leader",
        "glyph_font_like": "glyph",
        "control_private_use_replacement": "invisible",
        "short_nonascii_latin_like": "latin_ext",
    }.get(category, category)


NEW_TASK_MODES = {
    "separator_normalization",
    "slash_dash_classification",
    # md_table_audit and page_noise_detection need a different data pipeline
    # (table-before/after synthesis, synthetic-page content injection).
    # Their prompt preambles + schemas are registered in the prompt-builder;
    # bundler support here is a follow-up. Falling through to the legacy
    # renderer with a clear note keeps them reviewable manually if needed.
}


DEFAULT_REVIEW_SPECS: Dict[str, Dict[str, Any]] = {
    "glyph_font_like": {
        "review_mode": "cleaning",
        "goal": (
            "We are reviewing likely PDF extractor/font residue that survived the current cleaning "
            "pipeline. Decide whether the matched content is semantic noise and whether it suggests "
            "an extension to existing glyph/font cleaning."
        ),
        "questions": [
            "Is the matched content noise?",
            "Is the noisy span larger than the matched anchor?",
            "Is there adjacent noise of a different type in the same local context?",
            "Should an existing glyph/font regex be extended, or is a new regex family needed?",
            "If regex-matchable, what regex or regex family would capture the bad span safely?",
        ],
    },
    "control_private_use_replacement": {
        "review_mode": "cleaning",
        "goal": (
            "We are reviewing control characters, private-use glyphs, replacement characters, and "
            "similar Unicode residue that survived the current cleaning pipeline."
        ),
        "questions": [
            "Is the matched content noise?",
            "Is the noisy span larger than the matched anchor?",
            "Is there adjacent noise of a different type in the same local context?",
            "Should this be removed with broader Unicode sanitation, a targeted regex family, or kept?",
            "If regex-matchable, what regex or regex family would capture the bad span safely?",
        ],
    },
    "short_nonascii_latin_like": {
        "review_mode": "cleaning",
        "goal": (
            "We are reviewing short suspicious non-Greek fragments that may indicate mojibake, broken "
            "glyph decoding, or broader page corruption. Use the page metrics to judge whether this is "
            "isolated residue or part of a noisier synthetic page."
        ),
        "questions": [
            "Is the matched content noise?",
            "Is the noisy span larger than the matched anchor?",
            "Is there adjacent noise of a different type in the same local context?",
            "Does the page-level density suggest this page is broadly noisy beyond the matched span?",
            "If regex-matchable, what regex or regex family would capture the bad span safely?",
        ],
    },
    "dot_leader_like": {
        "review_mode": "normalization",
        "goal": (
            "We are reviewing likely TOC/layout leader runs for tokenizer-oriented normalization. The "
            "goal is to minimize redundant separator classes without losing semantics."
        ),
        "questions": [
            "Is this acting as a layout/TOC leader rather than ordinary prose punctuation?",
            "Is it safely interchangeable with the canonical leader form `.....` in this context?",
            "Would that normalization preserve local semantics?",
            "Does any surrounding context indicate this should be treated as noise instead of normalized structure?",
        ],
    },
}


def _slugify_label(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    text = _SAFE_LABEL_RE.sub("_", text).strip("._-")
    return text or "unlabeled"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _row_categories(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("category", "")
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item)]
    return [part for part in str(raw).split(",") if part]


def _stable_sample(rows: Sequence[Dict[str, Any]], limit: int, seed: str) -> List[Dict[str, Any]]:
    def _key(row: Dict[str, Any]) -> str:
        basis = f"{seed}|{row.get('match_id', '')}|{row.get('source_stem', '')}|{row.get('page_number', 0)}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    return sorted(rows, key=_key)[: max(int(limit), 0)]


def _page_key(row: Mapping[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(row.get("source_stem", "")),
        str(row.get("page_kind", "")),
        int(row.get("page_number", 0) or 0),
        int(row.get("page_index_in_file", 0) or 0),
    )


def _load_review_specs(config_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if config_path is None:
        return dict(DEFAULT_REVIEW_SPECS)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Review config must be a JSON object: {config_path}")
    merged = dict(DEFAULT_REVIEW_SPECS)
    for category, spec in payload.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Review config entry must be an object for {category}")
        merged[str(category)] = dict(spec)
    return merged


def _format_case_text(
    row: Mapping[str, Any],
    *,
    page_metrics: Optional[Mapping[str, Any]],
    review_spec: Mapping[str, Any],
    debug_page_text: str,
    include_full_debug_page: bool,
) -> str:
    review_mode = str(review_spec.get("review_mode", "unknown"))
    if review_mode in NEW_TASK_MODES:
        return _build_v2_case_body(
            row,
            review_mode=review_mode,
            canonical_target=NEW_TASK_CANONICAL_TARGETS.get(review_mode),
            existing_cleaner_note=NEW_TASK_EXISTING_CLEANER.get(review_mode),
        )
    metadata_lines = [
        f"REVIEW_MODE: {review_spec.get('review_mode', 'unknown')}",
        f"CATEGORY: {row.get('category', '')}",
        f"PATTERN_FAMILY: {row.get('pattern_family', '')}",
        f"MATCH_ID: {row.get('match_id', '')}",
        f"SOURCE_PATH: {row.get('source_path', '')}",
        f"SOURCE_STEM: {row.get('source_stem', '')}",
        f"DEBUG_OUTPUT_PATH: {row.get('debug_output_path', '')}",
        f"PAGE_KIND: {row.get('page_kind', '')}",
        f"PAGE_NUMBER: {row.get('page_number', 0)}",
        f"PAGE_INDEX_IN_FILE: {row.get('page_index_in_file', 0)}",
        f"PAGE_CHAR_COUNT: {row.get('page_char_count', 0)}",
        f"MERGED_MATCHED_TEXT: {row.get('matched_text', '')}",
        f"RAW_TEXTS: {json.dumps(list(row.get('raw_texts') or []), ensure_ascii=False)}",
        f"CONTEXT_BEFORE: {row.get('context_before', '')}",
        f"CONTEXT_AFTER: {row.get('context_after', '')}",
    ]
    if page_metrics is not None:
        metadata_lines.extend(
            [
                f"PAGE_MATCH_COUNT: {page_metrics.get('match_count', 0)}",
                f"PAGE_MATCH_DENSITY_PER_1K: {page_metrics.get('match_density_per_1k_chars', 0.0)}",
                f"PAGE_CATEGORY_MATCH_COUNTS: {json.dumps(page_metrics.get('category_match_counts', {}), ensure_ascii=False)}",
                f"PAGE_PATTERN_FAMILY_MATCH_COUNTS: {json.dumps(page_metrics.get('pattern_family_match_counts', {}), ensure_ascii=False)}",
            ]
        )

    questions = "\n".join(
        f"{idx}. {question}" for idx, question in enumerate(list(review_spec.get("questions") or []), start=1)
    )
    parts = [
        "\n".join(metadata_lines),
        "=== REVIEW_GOAL ===",
        str(review_spec.get("goal", "")),
        "=== FIELD_NOTES ===",
        (
            "MERGED_MATCHED_TEXT is the merged surface span on the synthetic page. "
            "RAW_TEXTS are the original literal anchors that triggered the category. "
            "If they differ, trust RAW_TEXTS for the original trigger identity and use "
            "MERGED_MATCHED_TEXT plus context to judge surrounding noise or structure."
        ),
        "=== CATEGORY_QUESTIONS ===",
        questions,
        "=== MATCH_CONTEXT_EXCERPT ===",
        str(row.get("context_excerpt", "")),
    ]
    if include_full_debug_page:
        parts.extend(
            [
                "=== DEBUG_PAGE_WITH_MATCH_TAGS ===",
                debug_page_text,
            ]
        )
    return "\n\n".join(parts).rstrip() + "\n"


def build_token_category_review_bundle(
    *,
    run_dir: Path,
    output_dir: Path,
    sample_size_per_category: int = 300,
    categories: Optional[Sequence[str]] = None,
    seed: str = "token-category-review-v1",
    review_config: Optional[Path] = None,
    include_full_debug_page: bool = False,
) -> Dict[str, Any]:
    match_rows = _read_jsonl(run_dir / "match_index.jsonl")
    page_metric_rows = _read_jsonl(run_dir / "page_metrics.jsonl")
    review_specs = _load_review_specs(review_config)

    if categories is None:
        selected_categories = [
            name for name in review_specs if any(name in _row_categories(row) for row in match_rows)
        ]
    else:
        selected_categories = [str(name) for name in categories]

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for stale in cases_dir.rglob("*.txt"):
        stale.unlink()
    for stale_name in ("manifest.jsonl", "summary.json"):
        stale = output_dir / stale_name
        if stale.exists():
            stale.unlink()

    page_metrics_by_key: Dict[Tuple[str, str, int, int], Dict[str, Any]] = {
        _page_key(row): row for row in page_metric_rows
    }

    manifest_rows: List[Dict[str, Any]] = []
    summary_rows: MutableMapping[str, Dict[str, Any]] = {}

    for category in selected_categories:
        category_rows = [row for row in match_rows if category in _row_categories(row)]
        if not category_rows:
            continue
        sample = _stable_sample(category_rows, sample_size_per_category, f"{seed}:{category}")
        review_spec = review_specs.get(category, {"review_mode": "unknown", "goal": "", "questions": []})
        category_slug = _slugify_label(category)
        category_dir = cases_dir / category_slug
        category_dir.mkdir(parents=True, exist_ok=True)
        pattern_counts: Dict[str, int] = {}
        for idx, row in enumerate(sample, start=1):
            page_key = _page_key(row)
            page_metrics = page_metrics_by_key.get(page_key)
            debug_output_path = Path(str(row.get("debug_output_path", "")))
            debug_page_text = (
                debug_output_path.read_text(encoding="utf-8", errors="ignore")
                if include_full_debug_page and debug_output_path.exists()
                else ""
            )
            pattern_family = str(row.get("pattern_family", ""))
            pattern_counts[pattern_family] = pattern_counts.get(pattern_family, 0) + 1
            case_name = (
                f"{idx:04d}__{category_slug}__{_slugify_label(pattern_family)}__"
                f"{_slugify_label(row.get('source_stem', 'source'))}__p{int(row.get('page_number', 0) or 0):05d}.txt"
            )
            case_path = category_dir / case_name
            manifest_row = dict(row)
            manifest_row.update(
                {
                    "category": category,
                    "matched_row_categories": _row_categories(row),
                    "review_mode": review_spec.get("review_mode", "unknown"),
                    "case_path": str(case_path),
                    "page_metrics": page_metrics or {},
                    "include_full_debug_page": include_full_debug_page,
                }
            )
            case_path.write_text(
                _format_case_text(
                    manifest_row,
                    page_metrics=page_metrics,
                    review_spec=review_spec,
                    debug_page_text=debug_page_text,
                    include_full_debug_page=include_full_debug_page,
                ),
                encoding="utf-8",
            )
            manifest_rows.append(manifest_row)

        summary_rows[category] = {
            "review_mode": review_spec.get("review_mode", "unknown"),
            "available_matches": len(category_rows),
            "sampled_matches": len(sample),
            "pattern_family_counts_in_sample": pattern_counts,
            "cases_dir": str(category_dir),
        }

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "sample_size_per_category": int(sample_size_per_category),
        "seed": seed,
        "include_full_debug_page": include_full_debug_page,
        "categories": selected_categories,
        "sampled_case_count": len(manifest_rows),
        "category_summaries": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sampled review bundles from token-category debug exports.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size-per-category", type=int, default=300)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--seed", default="token-category-review-v1")
    parser.add_argument("--review-config", type=Path, default=None)
    parser.add_argument("--include-full-debug-page", action="store_true")
    args = parser.parse_args()

    summary = build_token_category_review_bundle(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        sample_size_per_category=args.sample_size_per_category,
        categories=args.category,
        seed=args.seed,
        review_config=args.review_config,
        include_full_debug_page=bool(args.include_full_debug_page),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
