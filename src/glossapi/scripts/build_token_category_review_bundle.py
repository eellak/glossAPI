from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


_SAFE_LABEL_RE = re.compile(r"[^a-z0-9._-]+")


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
