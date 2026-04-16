from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def _safe_rate(counter: Mapping[str, int], value: str, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(counter.get(value, 0)) / float(total)


def _normalize_candidate_regex(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
    if not text:
        return None
    return text


def _compile_regex(pattern: str) -> Optional[re.Pattern[str]]:
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _regex_hits_case(pattern: re.Pattern[str], bundle_row: Mapping[str, Any]) -> bool:
    fields = [
        str(bundle_row.get("matched_text", "")),
        str(bundle_row.get("context_excerpt", "")),
        str(bundle_row.get("context_before", "")),
        str(bundle_row.get("context_after", "")),
    ]
    for raw in list(bundle_row.get("raw_texts") or []):
        fields.append(str(raw))
    expanded_fields: List[str] = []
    for field in fields:
        if not field:
            continue
        expanded_fields.append(field)
        unescaped = html.unescape(field)
        if unescaped != field:
            expanded_fields.append(unescaped)
    return any(bool(pattern.search(field)) for field in expanded_fields if field)


def _union_regex(patterns: Sequence[str]) -> Optional[str]:
    normalized = [pattern for pattern in patterns if pattern]
    if not normalized:
        return None
    unique = list(dict.fromkeys(normalized))
    if len(unique) == 1:
        return unique[0]
    return "|".join(f"(?:{pattern})" for pattern in unique)


def _decision_counts_cleaning(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    fields = [
        "is_noise",
        "larger_bad_span_than_anchor",
        "adjacent_different_noise",
        "match_update_type",
    ]
    out: Dict[str, Dict[str, int]] = {}
    for field in fields:
        counter = Counter(str((row.get("parsed") or {}).get(field, "missing")) for row in rows)
        out[field] = dict(counter)
    return out


def _decision_counts_normalization(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    fields = [
        "is_normalizable_structure",
        "structure_role",
        "interchangeable_with_target",
        "preserves_semantics",
        "preserves_markdown_structure",
        "canonical_target",
    ]
    out: Dict[str, Dict[str, int]] = {}
    for field in fields:
        counter = Counter(str((row.get("parsed") or {}).get(field, "missing")) for row in rows)
        out[field] = dict(counter)
    return out


def _family_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    return (
        str(row.get("category", "")),
        str(row.get("pattern_family", "")),
    )


def aggregate_token_category_reviews(
    *,
    bundle_dir: Path,
    review_dir: Path,
    output_dir: Path,
    min_reviewed_per_family: int = 10,
    cleaning_noise_yes_threshold: float = 0.85,
    cleaning_uncertain_max: float = 0.15,
    cleaning_match_update_threshold: float = 0.5,
    min_regex_votes: int = 2,
    normalization_interchangeable_threshold: float = 0.85,
    normalization_semantics_threshold: float = 0.90,
    markdown_preserve_threshold: float = 0.90,
) -> Dict[str, Any]:
    bundle_rows = _read_jsonl(bundle_dir / "manifest.jsonl")
    review_rows = _read_jsonl(review_dir / "results.jsonl")
    bundle_by_match_id = {str(row.get("match_id", "")): row for row in bundle_rows}

    joined_rows: List[Dict[str, Any]] = []
    for review_row in review_rows:
        match_id = str(review_row.get("match_id", ""))
        bundle_row = bundle_by_match_id.get(match_id)
        merged = dict(bundle_row or {})
        merged.update(review_row)
        joined_rows.append(merged)

    output_dir.mkdir(parents=True, exist_ok=True)

    category_summary_rows: List[Dict[str, Any]] = []
    family_summary_rows: List[Dict[str, Any]] = []
    regex_inventory_rows: List[Dict[str, Any]] = []
    candidate_cleaning_rules: List[Dict[str, Any]] = []
    candidate_normalization_rules: List[Dict[str, Any]] = []
    manual_review_queue: List[Dict[str, Any]] = []

    rows_by_category: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    rows_by_family: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        rows_by_category[str(row.get("category", ""))].append(row)
        rows_by_family[_family_key(row)].append(row)

    for category, rows in sorted(rows_by_category.items()):
        review_mode = str(rows[0].get("review_mode", "unknown")) if rows else "unknown"
        reviewed_count = len(rows)
        if review_mode == "cleaning":
            decisions = _decision_counts_cleaning(rows)
            category_summary_rows.append(
                {
                    "category": category,
                    "review_mode": review_mode,
                    "reviewed_count": reviewed_count,
                    "decision_counts": decisions,
                    "noise_yes_rate": _safe_rate(decisions["is_noise"], "yes", reviewed_count),
                    "noise_uncertain_rate": _safe_rate(decisions["is_noise"], "uncertain", reviewed_count),
                }
            )
        elif review_mode == "normalization":
            decisions = _decision_counts_normalization(rows)
            category_summary_rows.append(
                {
                    "category": category,
                    "review_mode": review_mode,
                    "reviewed_count": reviewed_count,
                    "decision_counts": decisions,
                    "normalizable_yes_rate": _safe_rate(decisions["is_normalizable_structure"], "yes", reviewed_count),
                    "interchangeable_yes_rate": _safe_rate(decisions["interchangeable_with_target"], "yes", reviewed_count),
                    "semantics_yes_rate": _safe_rate(decisions["preserves_semantics"], "yes", reviewed_count),
                }
            )
        else:
            category_summary_rows.append(
                {
                    "category": category,
                    "review_mode": review_mode,
                    "reviewed_count": reviewed_count,
                    "decision_counts": {},
                }
            )

    for (category, pattern_family), rows in sorted(rows_by_family.items()):
        if not rows:
            continue
        review_mode = str(rows[0].get("review_mode", "unknown"))
        reviewed_count = len(rows)
        decision_counts = (
            _decision_counts_cleaning(rows)
            if review_mode == "cleaning"
            else _decision_counts_normalization(rows)
            if review_mode == "normalization"
            else {}
        )
        family_row: Dict[str, Any] = {
            "category": category,
            "pattern_family": pattern_family,
            "review_mode": review_mode,
            "reviewed_count": reviewed_count,
            "decision_counts": decision_counts,
        }

        if review_mode == "cleaning":
            is_noise = Counter(decision_counts.get("is_noise", {}))
            noise_yes_rate = _safe_rate(is_noise, "yes", reviewed_count)
            noise_uncertain_rate = _safe_rate(is_noise, "uncertain", reviewed_count)
            family_row["noise_yes_rate"] = noise_yes_rate
            family_row["noise_uncertain_rate"] = noise_uncertain_rate

            match_update_counts = Counter(decision_counts.get("match_update_type", {}))
            top_match_update_type = (
                max(match_update_counts.items(), key=lambda item: item[1])[0]
                if match_update_counts
                else "missing"
            )
            top_match_update_rate = _safe_rate(match_update_counts, top_match_update_type, reviewed_count)
            family_row["top_match_update_type"] = top_match_update_type
            family_row["top_match_update_rate"] = top_match_update_rate

            regex_counter: Counter[str] = Counter()
            regex_examples: MutableMapping[str, List[str]] = defaultdict(list)
            validation_counter: Counter[str] = Counter()
            for row in rows:
                parsed = dict(row.get("parsed") or {})
                normalized_regex = _normalize_candidate_regex(parsed.get("candidate_regex"))
                if not normalized_regex:
                    continue
                regex_counter[normalized_regex] += 1
                if len(regex_examples[normalized_regex]) < 5:
                    regex_examples[normalized_regex].append(str(row.get("match_id", "")))
                compiled = _compile_regex(normalized_regex)
                validation_counter[normalized_regex] += int(
                    compiled is not None and _regex_hits_case(compiled, row)
                )
            regex_rows = []
            for pattern, count in regex_counter.most_common():
                hit_count = validation_counter.get(pattern, 0)
                regex_rows.append(
                    {
                        "category": category,
                        "pattern_family": pattern_family,
                        "candidate_regex": pattern,
                        "vote_count": count,
                        "vote_rate": float(count) / float(reviewed_count) if reviewed_count else 0.0,
                        "validated_hit_count": hit_count,
                        "validated_hit_rate_over_votes": float(hit_count) / float(count) if count else 0.0,
                        "example_match_ids": regex_examples.get(pattern, []),
                    }
                )
            regex_inventory_rows.extend(regex_rows)
            family_row["regex_candidate_count"] = sum(regex_counter.values())
            family_row["regex_inventory"] = regex_rows
            supported_regex_rows = [
                row
                for row in regex_rows
                if int(row["vote_count"]) >= int(min_regex_votes)
                and float(row["validated_hit_rate_over_votes"]) > 0.0
            ]
            family_row["supported_regex_inventory"] = supported_regex_rows

            promote = (
                reviewed_count >= int(min_reviewed_per_family)
                and noise_yes_rate >= float(cleaning_noise_yes_threshold)
                and noise_uncertain_rate <= float(cleaning_uncertain_max)
                and top_match_update_type in {"existing_regex_extension", "new_regex"}
                and top_match_update_rate >= float(cleaning_match_update_threshold)
                and bool(supported_regex_rows)
            )
            union_regex = _union_regex([row["candidate_regex"] for row in supported_regex_rows])
            top_regex = supported_regex_rows[0]["candidate_regex"] if supported_regex_rows else None
            if promote:
                candidate_cleaning_rules.append(
                    {
                        "category": category,
                        "pattern_family": pattern_family,
                        "reviewed_count": reviewed_count,
                        "noise_yes_rate": noise_yes_rate,
                        "noise_uncertain_rate": noise_uncertain_rate,
                        "top_match_update_type": top_match_update_type,
                        "top_match_update_rate": top_match_update_rate,
                        "top_candidate_regex": top_regex,
                        "candidate_regex_union": union_regex,
                        "regex_inventory": supported_regex_rows,
                        "recommended_action": "promote_cleaning_rule",
                    }
                )
            if (
                noise_yes_rate < float(cleaning_noise_yes_threshold)
                or noise_uncertain_rate > float(cleaning_uncertain_max)
                or top_match_update_type not in {"existing_regex_extension", "new_regex"}
                or top_match_update_rate < float(cleaning_match_update_threshold)
                or not supported_regex_rows
            ):
                for row in rows:
                    parsed = dict(row.get("parsed") or {})
                    if str(parsed.get("is_noise", "missing")) == "uncertain" or not regex_rows:
                        manual_review_queue.append(
                            {
                                "category": category,
                                "pattern_family": pattern_family,
                                "review_mode": review_mode,
                                "match_id": row.get("match_id", ""),
                                "case_path": row.get("case_path", ""),
                                "reason": "uncertain_or_no_regex_consensus",
                                "parsed": parsed,
                            }
                        )

        elif review_mode == "normalization":
            normalizable = Counter(decision_counts.get("is_normalizable_structure", {}))
            interchangeable = Counter(decision_counts.get("interchangeable_with_target", {}))
            semantics = Counter(decision_counts.get("preserves_semantics", {}))
            markdown = Counter(decision_counts.get("preserves_markdown_structure", {}))
            target_counter = Counter(
                str((row.get("parsed") or {}).get("canonical_target", "missing")) for row in rows
            )

            normalizable_yes_rate = _safe_rate(normalizable, "yes", reviewed_count)
            interchangeable_yes_rate = _safe_rate(interchangeable, "yes", reviewed_count)
            semantics_yes_rate = _safe_rate(semantics, "yes", reviewed_count)
            markdown_yes = markdown.get("yes", 0)
            markdown_not_applicable = markdown.get("not_applicable", 0)
            markdown_denominator = max(reviewed_count - markdown_not_applicable, 0)
            markdown_yes_rate = (
                float(markdown_yes) / float(markdown_denominator)
                if markdown_denominator > 0
                else 1.0
            )

            family_row.update(
                {
                    "normalizable_yes_rate": normalizable_yes_rate,
                    "interchangeable_yes_rate": interchangeable_yes_rate,
                    "semantics_yes_rate": semantics_yes_rate,
                    "markdown_yes_rate": markdown_yes_rate,
                    "canonical_target_counts": dict(target_counter),
                }
            )

            top_target = max(target_counter.items(), key=lambda item: item[1])[0] if target_counter else None
            promote = (
                reviewed_count >= int(min_reviewed_per_family)
                and normalizable_yes_rate >= float(normalization_interchangeable_threshold)
                and interchangeable_yes_rate >= float(normalization_interchangeable_threshold)
                and semantics_yes_rate >= float(normalization_semantics_threshold)
                and markdown_yes_rate >= float(markdown_preserve_threshold)
                and top_target not in {None, "missing", "null"}
            )
            if promote:
                candidate_normalization_rules.append(
                    {
                        "category": category,
                        "pattern_family": pattern_family,
                        "reviewed_count": reviewed_count,
                        "normalizable_yes_rate": normalizable_yes_rate,
                        "interchangeable_yes_rate": interchangeable_yes_rate,
                        "semantics_yes_rate": semantics_yes_rate,
                        "markdown_yes_rate": markdown_yes_rate,
                        "canonical_target": top_target,
                        "recommended_action": "promote_normalization_rule",
                    }
                )
            if not promote:
                for row in rows:
                    parsed = dict(row.get("parsed") or {})
                    if (
                        str(parsed.get("interchangeable_with_target", "missing")) == "uncertain"
                        or str(parsed.get("preserves_semantics", "missing")) == "uncertain"
                    ):
                        manual_review_queue.append(
                            {
                                "category": category,
                                "pattern_family": pattern_family,
                                "review_mode": review_mode,
                                "match_id": row.get("match_id", ""),
                                "case_path": row.get("case_path", ""),
                                "reason": "uncertain_normalization_judgment",
                                "parsed": parsed,
                            }
                        )

        family_summary_rows.append(family_row)

    _write_jsonl(output_dir / "category_summary.jsonl", category_summary_rows)
    _write_jsonl(output_dir / "family_summary.jsonl", family_summary_rows)
    _write_jsonl(output_dir / "regex_inventory.jsonl", regex_inventory_rows)
    _write_jsonl(output_dir / "candidate_cleaning_rules.jsonl", candidate_cleaning_rules)
    _write_jsonl(output_dir / "candidate_normalization_rules.jsonl", candidate_normalization_rules)
    _write_jsonl(output_dir / "manual_review_queue.jsonl", manual_review_queue)

    summary = {
        "bundle_dir": str(bundle_dir),
        "review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "reviewed_case_count": len(joined_rows),
        "min_reviewed_per_family": int(min_reviewed_per_family),
        "category_count": len(category_summary_rows),
        "family_count": len(family_summary_rows),
        "candidate_cleaning_rule_count": len(candidate_cleaning_rules),
        "candidate_normalization_rule_count": len(candidate_normalization_rules),
        "manual_review_count": len(manual_review_queue),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate structured token-category review results into rule candidates.")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-reviewed-per-family", type=int, default=10)
    parser.add_argument("--cleaning-noise-yes-threshold", type=float, default=0.85)
    parser.add_argument("--cleaning-uncertain-max", type=float, default=0.15)
    parser.add_argument("--cleaning-match-update-threshold", type=float, default=0.5)
    parser.add_argument("--min-regex-votes", type=int, default=2)
    parser.add_argument("--normalization-interchangeable-threshold", type=float, default=0.85)
    parser.add_argument("--normalization-semantics-threshold", type=float, default=0.90)
    parser.add_argument("--markdown-preserve-threshold", type=float, default=0.90)
    args = parser.parse_args()

    summary = aggregate_token_category_reviews(
        bundle_dir=args.bundle_dir,
        review_dir=args.review_dir,
        output_dir=args.output_dir,
        min_reviewed_per_family=args.min_reviewed_per_family,
        cleaning_noise_yes_threshold=args.cleaning_noise_yes_threshold,
        cleaning_uncertain_max=args.cleaning_uncertain_max,
        cleaning_match_update_threshold=args.cleaning_match_update_threshold,
        min_regex_votes=args.min_regex_votes,
        normalization_interchangeable_threshold=args.normalization_interchangeable_threshold,
        normalization_semantics_threshold=args.normalization_semantics_threshold,
        markdown_preserve_threshold=args.markdown_preserve_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
