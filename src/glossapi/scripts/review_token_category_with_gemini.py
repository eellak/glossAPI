from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

COMMON_SYSTEM_INSTRUCTION = (
    "You are reviewing tokenizer-oriented corpus cleaning and normalization cases. "
    "Use only the evidence in the provided case file. Be conservative. If you are not "
    "confident that something is noise or safely normalizable, return uncertain. "
    "Important: MERGED_MATCHED_TEXT is a merged page span, while RAW_TEXTS are the original "
    "literal anchors that triggered the category. If they differ, use RAW_TEXTS for anchor identity. "
    "Return only JSON that matches the schema."
)

REVIEW_MODE_INSTRUCTIONS = {
    "cleaning": (
        "Decide whether the matched content is noise, whether the noisy span is larger "
        "than the anchor, whether there is adjacent different noise, and whether a safe "
        "regex update is possible. If you propose a regex, keep it narrow."
    ),
    "normalization": (
        "Decide whether the matched content is a normalizable structural separator/layout form, "
        "whether it is interchangeable with the target canonical form, and whether semantics "
        "or markdown structure are preserved."
    ),
}

YES_NO_UNCERTAIN = ["yes", "no", "uncertain"]

SCHEMA_BY_REVIEW_MODE: Dict[str, Dict[str, Any]] = {
    "cleaning": {
        "type": "object",
        "properties": {
            "is_noise": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether the matched content is semantic noise with no meaningful language or structural purpose.",
            },
            "larger_bad_span_than_anchor": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether the actual bad span extends beyond the matched anchor.",
            },
            "adjacent_different_noise": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether nearby context also contains a different type of noise.",
            },
            "match_update_type": {
                "type": "string",
                "enum": [
                    "existing_regex_extension",
                    "new_regex",
                    "no_regex_needed",
                    "keep",
                    "uncertain",
                ],
                "description": "Best next action for matching or cleaning this noise.",
            },
            "candidate_regex": {
                "type": ["string", "null"],
                "description": "Candidate regex if a safe regex update is possible, otherwise null.",
            },
            "regex_scope_note": {
                "type": "string",
                "description": "Short note on how broadly the regex should apply or why no regex should be used.",
            },
            "short_reason": {
                "type": "string",
                "description": "Short explanation grounded in the provided context.",
            },
        },
        "required": [
            "is_noise",
            "larger_bad_span_than_anchor",
            "adjacent_different_noise",
            "match_update_type",
            "candidate_regex",
            "regex_scope_note",
            "short_reason",
        ],
        "additionalProperties": False,
    },
    "normalization": {
        "type": "object",
        "properties": {
            "is_normalizable_structure": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether the matched content is valid structure to normalize instead of noise to remove.",
            },
            "structure_role": {
                "type": "string",
                "enum": [
                    "layout_leader",
                    "separator_line",
                    "markdown_table_separator",
                    "noise",
                    "other",
                    "uncertain",
                ],
                "description": "Best structural role for the matched content in context.",
            },
            "interchangeable_with_target": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether the content is safely interchangeable with the chosen canonical target.",
            },
            "preserves_semantics": {
                "type": "string",
                "enum": YES_NO_UNCERTAIN,
                "description": "Whether the normalization preserves local semantics.",
            },
            "preserves_markdown_structure": {
                "type": "string",
                "enum": ["yes", "no", "not_applicable", "uncertain"],
                "description": "Whether the normalization preserves markdown structure when relevant.",
            },
            "canonical_target": {
                "type": ["string", "null"],
                "description": "Canonical form if normalization is appropriate, otherwise null.",
            },
            "short_reason": {
                "type": "string",
                "description": "Short explanation grounded in the provided context.",
            },
        },
        "required": [
            "is_normalizable_structure",
            "structure_role",
            "interchangeable_with_target",
            "preserves_semantics",
            "preserves_markdown_structure",
            "canonical_target",
            "short_reason",
        ],
        "additionalProperties": False,
    },
}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _iter_selected_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    categories: Optional[Sequence[str]],
    max_cases_per_category: Optional[int],
) -> List[Dict[str, Any]]:
    allowed = None if categories is None else {str(item) for item in categories}
    per_category: MutableMapping[str, int] = defaultdict(int)
    selected: List[Dict[str, Any]] = []
    for row in rows:
        category = str(row.get("category", ""))
        if allowed is not None and category not in allowed:
            continue
        if max_cases_per_category is not None and per_category[category] >= int(max_cases_per_category):
            continue
        per_category[category] += 1
        selected.append(row)
    return selected


def _build_prompt(row: Mapping[str, Any], case_text: str) -> str:
    review_mode = str(row.get("review_mode", "unknown"))
    mode_instruction = REVIEW_MODE_INSTRUCTIONS.get(review_mode, "")
    return "\n\n".join(
        [
            f"REVIEW_MODE: {review_mode}",
            mode_instruction,
            "Use the following case file exactly as provided.",
            case_text,
        ]
    )


def _call_gemini_structured(
    *,
    api_key: str,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    url = GEMINI_API_URL.format(model=urllib.parse.quote(model, safe=""))
    payload = {
        "system_instruction": {
            "parts": [{"text": COMMON_SYSTEM_INSTRUCTION}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
            "responseJsonSchema": schema,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url}?key={urllib.parse.quote(api_key)}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API error {exc.code}: {error_body}") from exc
    payload = json.loads(body)
    candidates = list(payload.get("candidates") or [])
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {body}")
    parts = list((((candidates[0] or {}).get("content") or {}).get("parts") or []))
    if not parts:
        raise RuntimeError(f"Gemini API returned no content parts: {body}")
    text = str(parts[0].get("text", "")).strip()
    if not text:
        raise RuntimeError(f"Gemini API returned empty text: {body}")
    return {
        "parsed": json.loads(text),
        "raw_text": text,
        "raw_response": payload,
    }


def review_token_category_bundle_with_gemini(
    *,
    bundle_dir: Path,
    output_dir: Path,
    api_key: Optional[str],
    model: str = "gemini-2.5-flash",
    categories: Optional[Sequence[str]] = None,
    max_cases_per_category: Optional[int] = None,
    temperature: float = 0.2,
    dry_run: bool = False,
    sleep_seconds: float = 0.0,
    max_workers: int = 1,
) -> Dict[str, Any]:
    manifest_rows = _read_jsonl(bundle_dir / "manifest.jsonl")
    selected_rows = _iter_selected_rows(
        manifest_rows,
        categories=categories,
        max_cases_per_category=max_cases_per_category,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    results_path = output_dir / "results.jsonl"
    errors_path = output_dir / "errors.jsonl"
    summary_path = output_dir / "summary.json"
    existing_request_rows = _read_jsonl(requests_path)
    existing_result_rows = _read_jsonl(results_path)
    existing_request_ids = {str(row.get("match_id", "")) for row in existing_request_rows}
    existing_result_by_id = {str(row.get("match_id", "")): row for row in existing_result_rows}

    per_category_counts: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    error_rows: List[Dict[str, Any]] = []
    for row in existing_result_rows:
        review_mode = str(row.get("review_mode", "unknown"))
        category = str(row.get("category", ""))
        parsed = dict(row.get("parsed") or {})
        key_field = "is_noise" if review_mode == "cleaning" else "preserves_semantics"
        per_category_counts[category][str(parsed.get(key_field, "missing"))] += 1

    requests_handle = requests_path.open("a", encoding="utf-8")
    results_handle = results_path.open("a", encoding="utf-8")
    errors_handle = errors_path.open("a", encoding="utf-8")
    try:
        pending_calls: List[Dict[str, Any]] = []
        for row in selected_rows:
            match_id = str(row.get("match_id", ""))
            review_mode = str(row.get("review_mode", "unknown"))
            schema = SCHEMA_BY_REVIEW_MODE.get(review_mode)
            if schema is None:
                raise ValueError(f"Unsupported review_mode {review_mode!r} for {row.get('match_id', '')}")
            case_path = Path(str(row["case_path"]))
            case_text = case_path.read_text(encoding="utf-8", errors="ignore")
            prompt = _build_prompt(row, case_text)
            request_row = {
                "match_id": match_id,
                "category": row.get("category", ""),
                "review_mode": review_mode,
                "case_path": str(case_path),
                "prompt_char_count": len(prompt),
                "schema": schema,
                "model": model,
            }
            if match_id not in existing_request_ids:
                requests_handle.write(json.dumps(request_row, ensure_ascii=False))
                requests_handle.write("\n")
                requests_handle.flush()
                existing_request_ids.add(match_id)

            if match_id in existing_result_by_id:
                continue

            if dry_run:
                continue
            if not api_key:
                raise RuntimeError("Gemini API key is required unless --dry-run is set.")
            pending_calls.append(
                {
                    "row": row,
                    "match_id": match_id,
                    "category": str(row.get("category", "")),
                    "review_mode": review_mode,
                    "case_path": str(case_path),
                    "prompt": prompt,
                    "schema": schema,
                }
            )

        if not dry_run and pending_calls:
            worker_count = max(1, int(max_workers))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _call_gemini_structured,
                        api_key=api_key,
                        model=model,
                        prompt=item["prompt"],
                        schema=item["schema"],
                        temperature=temperature,
                    ): item
                    for item in pending_calls
                }
                for future in as_completed(future_map):
                    item = future_map[future]
                    try:
                        response = future.result()
                    except Exception as exc:
                        error_row = {
                            "match_id": item["match_id"],
                            "category": item["category"],
                            "review_mode": item["review_mode"],
                            "case_path": item["case_path"],
                            "error": str(exc),
                        }
                        error_rows.append(error_row)
                        errors_handle.write(json.dumps(error_row, ensure_ascii=False))
                        errors_handle.write("\n")
                        errors_handle.flush()
                        continue

                    parsed = response["parsed"]
                    category = item["category"]
                    key_field = "is_noise" if item["review_mode"] == "cleaning" else "preserves_semantics"
                    per_category_counts[category][str(parsed.get(key_field, "missing"))] += 1
                    result_row = {
                        "match_id": item["match_id"],
                        "category": category,
                        "review_mode": item["review_mode"],
                        "case_path": item["case_path"],
                        "parsed": parsed,
                        "raw_text": response["raw_text"],
                    }
                    results_handle.write(json.dumps(result_row, ensure_ascii=False))
                    results_handle.write("\n")
                    results_handle.flush()
                    existing_result_by_id[item["match_id"]] = result_row
                    if sleep_seconds > 0:
                        time.sleep(float(sleep_seconds))
    finally:
        requests_handle.close()
        results_handle.close()
        errors_handle.close()

    summary = {
        "bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "model": model,
        "dry_run": bool(dry_run),
        "selected_case_count": len(selected_rows),
        "completed_case_count": sum(1 for row in selected_rows if str(row.get("match_id", "")) in existing_result_by_id),
        "error_count": len(_read_jsonl(errors_path)),
        "categories": sorted({str(row.get("category", "")) for row in selected_rows}),
        "per_category_decision_counts": {category: dict(counter) for category, counter in per_category_counts.items()},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Review sampled token-category cases with Gemini structured output.")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--max-cases-per-category", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = None if args.dry_run else os.environ.get(args.api_key_env)
    summary = review_token_category_bundle_with_gemini(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        api_key=api_key,
        model=args.model,
        categories=args.category,
        max_cases_per_category=args.max_cases_per_category,
        temperature=args.temperature,
        dry_run=bool(args.dry_run),
        sleep_seconds=args.sleep_seconds,
        max_workers=args.max_workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
