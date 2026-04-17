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
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    genai = None
    genai_types = None


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)
DEFAULT_BATCH_INLINE_MAX_BYTES = 18_000_000
TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
SUCCESSFUL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}
RETRYABLE_BATCH_STATES = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

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


def _require_google_genai() -> None:
    if genai is None or genai_types is None:
        raise RuntimeError(
            "google-genai is required for --execution-mode live-sdk or batch. "
            "Install the project dependencies or `pip install google-genai`."
        )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


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


def _safe_batch_state(value: object) -> str:
    raw = str(value or "")
    if raw.startswith("JobState."):
        return raw.split(".", 1)[1]
    return raw or "JOB_STATE_UNSPECIFIED"


def _json_safe(value: object) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json", exclude_none=True))
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)


def _build_sdk_generation_config(
    *,
    schema: Mapping[str, Any],
    temperature: float,
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_json_schema": dict(schema),
    }
    thinking_config: Dict[str, Any] = {}
    if thinking_budget is not None:
        thinking_config["thinking_budget"] = int(thinking_budget)
    if thinking_level:
        thinking_config["thinking_level"] = str(thinking_level)
    if thinking_config:
        config["thinking_config"] = thinking_config
    return config


def _build_rest_generation_config(
    *,
    schema: Mapping[str, Any],
    temperature: float,
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "temperature": temperature,
        "responseMimeType": "application/json",
        "responseJsonSchema": dict(schema),
    }
    thinking_config: Dict[str, Any] = {}
    if thinking_budget is not None:
        thinking_config["thinkingBudget"] = int(thinking_budget)
    if thinking_level:
        thinking_config["thinkingLevel"] = str(thinking_level)
    if thinking_config:
        config["thinkingConfig"] = thinking_config
    return config


def _extract_text_from_generate_payload(payload: Mapping[str, Any]) -> str:
    candidates = list(payload.get("candidates") or [])
    if not candidates:
        raise RuntimeError(f"Google API returned no candidates: {json.dumps(payload, ensure_ascii=False)}")
    parts = list((((candidates[0] or {}).get("content") or {}).get("parts") or []))
    if not parts:
        raise RuntimeError(f"Google API returned no content parts: {json.dumps(payload, ensure_ascii=False)}")
    text = str(parts[0].get("text", "")).strip()
    if not text:
        raise RuntimeError(f"Google API returned empty text: {json.dumps(payload, ensure_ascii=False)}")
    return text


def _extract_text_from_sdk_response(response: object) -> str:
    text = str(getattr(response, "text", "") or "").strip()
    if text:
        return text
    payload = _json_safe(response)
    if isinstance(payload, dict):
        return _extract_text_from_generate_payload(payload)
    raise RuntimeError("Google SDK response did not contain text.")


def _build_http_options(
    *,
    api_version: str,
    request_timeout_ms: int,
    retry_attempts: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    retry_status_codes: Sequence[int],
) -> Dict[str, Any]:
    return {
        "api_version": api_version,
        "timeout": int(request_timeout_ms),
        "retry_options": {
            "attempts": int(retry_attempts),
            "initial_delay": float(retry_initial_delay),
            "max_delay": float(retry_max_delay),
            "exp_base": 2.0,
            "jitter": 1.0,
            "http_status_codes": [int(code) for code in retry_status_codes],
        },
    }


def _new_google_client(
    *,
    api_key: str,
    api_version: str,
    request_timeout_ms: int,
    retry_attempts: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    retry_status_codes: Sequence[int],
):
    _require_google_genai()
    return genai.Client(
        api_key=api_key,
        http_options=_build_http_options(
            api_version=api_version,
            request_timeout_ms=request_timeout_ms,
            retry_attempts=retry_attempts,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
            retry_status_codes=retry_status_codes,
        ),
    )


def _call_google_structured_rest(
    *,
    api_key: str,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    temperature: float,
    request_timeout_ms: int,
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
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
        "generationConfig": _build_rest_generation_config(
            schema=schema,
            temperature=temperature,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        ),
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
    timeout_seconds = max(float(request_timeout_ms) / 1000.0, 1.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Google API error {exc.code}: {error_body}") from exc
    payload = json.loads(body)
    text = _extract_text_from_generate_payload(payload)
    return {
        "parsed": json.loads(text),
        "raw_text": text,
        "raw_response": payload,
    }


def _call_google_structured_sdk(
    *,
    api_key: str,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    temperature: float,
    api_version: str,
    request_timeout_ms: int,
    retry_attempts: int,
    retry_initial_delay: float,
    retry_max_delay: float,
    retry_status_codes: Sequence[int],
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
) -> Dict[str, Any]:
    client = _new_google_client(
        api_key=api_key,
        api_version=api_version,
        request_timeout_ms=request_timeout_ms,
        retry_attempts=retry_attempts,
        retry_initial_delay=retry_initial_delay,
        retry_max_delay=retry_max_delay,
        retry_status_codes=retry_status_codes,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=_build_sdk_generation_config(
            schema=schema,
            temperature=temperature,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        ),
    )
    text = _extract_text_from_sdk_response(response)
    return {
        "parsed": json.loads(text),
        "raw_text": text,
        "raw_response": _json_safe(response),
    }


def _batch_request_metadata(item: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "match_id": str(item["match_id"]),
        "category": str(item["category"]),
        "review_mode": str(item["review_mode"]),
        "case_path": str(item["case_path"]),
    }


def _batch_request_payload(
    *,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    temperature: float,
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
    metadata: Mapping[str, str],
) -> Dict[str, Any]:
    return {
        "model": model,
        "contents": prompt,
        "config": _build_sdk_generation_config(
            schema=schema,
            temperature=temperature,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        ),
        "metadata": dict(metadata),
    }


def _estimate_batch_request_bytes(item: Mapping[str, Any], *, model: str, temperature: float, thinking_budget: Optional[int], thinking_level: Optional[str]) -> int:
    payload = _batch_request_payload(
        model=model,
        prompt=str(item["prompt"]),
        schema=dict(item["schema"]),
        temperature=temperature,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        metadata=_batch_request_metadata(item),
    )
    return len(json.dumps(payload, ensure_ascii=False))


def _shard_pending_calls_for_batch(
    pending_calls: Sequence[Mapping[str, Any]],
    *,
    model: str,
    temperature: float,
    thinking_budget: Optional[int],
    thinking_level: Optional[str],
    max_inline_bytes: int,
) -> List[List[Mapping[str, Any]]]:
    shards: List[List[Mapping[str, Any]]] = []
    current: List[Mapping[str, Any]] = []
    current_bytes = 0
    limit = max(int(max_inline_bytes), 1024)
    for item in pending_calls:
        estimated_bytes = _estimate_batch_request_bytes(
            item,
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
        )
        if current and current_bytes + estimated_bytes > limit:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += estimated_bytes
    if current:
        shards.append(current)
    return shards


def _serialize_batch_job(
    *,
    shard_index: int,
    request_count: int,
    request_bytes: int,
    match_ids: Sequence[str],
    batch_job: object,
) -> Dict[str, Any]:
    return {
        "shard_index": int(shard_index),
        "request_count": int(request_count),
        "request_bytes": int(request_bytes),
        "match_ids": [str(match_id) for match_id in match_ids],
        "job_name": str(getattr(batch_job, "name", "") or ""),
        "state": _safe_batch_state(getattr(batch_job, "state", None)),
        "display_name": str(getattr(batch_job, "display_name", "") or ""),
        "model": str(getattr(batch_job, "model", "") or ""),
        "src": _json_safe(getattr(batch_job, "src", None)),
        "dest": _json_safe(getattr(batch_job, "dest", None)),
        "error": _json_safe(getattr(batch_job, "error", None)),
        "completion_stats": _json_safe(getattr(batch_job, "completion_stats", None)),
        "create_time": _json_safe(getattr(batch_job, "create_time", None)),
        "start_time": _json_safe(getattr(batch_job, "start_time", None)),
        "end_time": _json_safe(getattr(batch_job, "end_time", None)),
        "update_time": _json_safe(getattr(batch_job, "update_time", None)),
    }


def _load_existing_batch_jobs(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    return [dict(row) for row in rows if str(row.get("job_name", "")).strip()]


def _extract_batch_response_text(response: object) -> str:
    return _extract_text_from_sdk_response(response)


def _ingest_batch_job_rows(
    *,
    batch_job_row: Mapping[str, Any],
    batch_job: object,
    results_handle,
    errors_handle,
    existing_result_by_id: MutableMapping[str, Dict[str, Any]],
    existing_error_keys: MutableMapping[Tuple[str, str], bool],
    per_category_counts: MutableMapping[str, Counter[str]],
) -> Tuple[int, int]:
    ingested_results = 0
    ingested_errors = 0
    dest = getattr(batch_job, "dest", None)
    inlined_responses = list(getattr(dest, "inlined_responses", None) or [])
    for inlined_response in inlined_responses:
        metadata = dict(getattr(inlined_response, "metadata", None) or {})
        match_id = str(metadata.get("match_id", ""))
        category = str(metadata.get("category", ""))
        review_mode = str(metadata.get("review_mode", ""))
        case_path = str(metadata.get("case_path", ""))
        if match_id and match_id in existing_result_by_id:
            continue

        error_obj = getattr(inlined_response, "error", None)
        if error_obj is not None:
            error_row = {
                "match_id": match_id,
                "category": category,
                "review_mode": review_mode,
                "case_path": case_path,
                "error": _json_safe(error_obj),
                "batch_job_name": str(batch_job_row.get("job_name", "")),
            }
            error_key = (match_id, json.dumps(error_row["error"], ensure_ascii=False, sort_keys=True))
            if error_key not in existing_error_keys:
                errors_handle.write(json.dumps(error_row, ensure_ascii=False))
                errors_handle.write("\n")
                errors_handle.flush()
                existing_error_keys[error_key] = True
                ingested_errors += 1
            continue

        response_obj = getattr(inlined_response, "response", None)
        if response_obj is None:
            error_row = {
                "match_id": match_id,
                "category": category,
                "review_mode": review_mode,
                "case_path": case_path,
                "error": "Batch response was missing both response and error.",
                "batch_job_name": str(batch_job_row.get("job_name", "")),
            }
            error_key = (match_id, str(error_row["error"]))
            if error_key not in existing_error_keys:
                errors_handle.write(json.dumps(error_row, ensure_ascii=False))
                errors_handle.write("\n")
                errors_handle.flush()
                existing_error_keys[error_key] = True
                ingested_errors += 1
            continue

        try:
            raw_text = _extract_batch_response_text(response_obj)
            parsed = json.loads(raw_text)
        except Exception as exc:
            error_row = {
                "match_id": match_id,
                "category": category,
                "review_mode": review_mode,
                "case_path": case_path,
                "error": f"Failed to parse batch response: {exc}",
                "batch_job_name": str(batch_job_row.get("job_name", "")),
            }
            error_key = (match_id, str(error_row["error"]))
            if error_key not in existing_error_keys:
                errors_handle.write(json.dumps(error_row, ensure_ascii=False))
                errors_handle.write("\n")
                errors_handle.flush()
                existing_error_keys[error_key] = True
                ingested_errors += 1
            continue

        key_field = "is_noise" if review_mode == "cleaning" else "preserves_semantics"
        per_category_counts[category][str(parsed.get(key_field, "missing"))] += 1
        result_row = {
            "match_id": match_id,
            "category": category,
            "review_mode": review_mode,
            "case_path": case_path,
            "parsed": parsed,
            "raw_text": raw_text,
            "batch_job_name": str(batch_job_row.get("job_name", "")),
        }
        results_handle.write(json.dumps(result_row, ensure_ascii=False))
        results_handle.write("\n")
        results_handle.flush()
        existing_result_by_id[match_id] = result_row
        ingested_results += 1
    return ingested_results, ingested_errors


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
    execution_mode: str = "live-sdk",
    api_version: str = "v1beta",
    request_timeout_ms: int = 300_000,
    retry_attempts: int = 5,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    retry_status_codes: Sequence[int] = DEFAULT_RETRY_STATUS_CODES,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    batch_poll_seconds: float = 30.0,
    batch_inline_max_bytes: int = DEFAULT_BATCH_INLINE_MAX_BYTES,
    batch_display_name: Optional[str] = None,
    batch_no_wait: bool = False,
) -> Dict[str, Any]:
    if execution_mode not in {"live-rest", "live-sdk", "batch"}:
        raise ValueError(f"Unsupported execution_mode {execution_mode!r}")

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
    batch_jobs_path = output_dir / "batch_jobs.jsonl"
    batch_status_path = output_dir / "batch_status.json"

    existing_request_rows = _read_jsonl(requests_path)
    existing_result_rows = _read_jsonl(results_path)
    existing_error_rows = _read_jsonl(errors_path)
    existing_request_ids = {str(row.get("match_id", "")) for row in existing_request_rows}
    existing_result_by_id = {str(row.get("match_id", "")): row for row in existing_result_rows}
    existing_error_keys: MutableMapping[Tuple[str, str], bool] = {}
    for row in existing_error_rows:
        error_key = (
            str(row.get("match_id", "")),
            json.dumps(row.get("error"), ensure_ascii=False, sort_keys=True),
        )
        existing_error_keys[error_key] = True

    per_category_counts: MutableMapping[str, Counter[str]] = defaultdict(Counter)
    for row in existing_result_rows:
        review_mode = str(row.get("review_mode", "unknown"))
        category = str(row.get("category", ""))
        parsed = dict(row.get("parsed") or {})
        key_field = "is_noise" if review_mode == "cleaning" else "preserves_semantics"
        per_category_counts[category][str(parsed.get(key_field, "missing"))] += 1

    requests_handle = requests_path.open("a", encoding="utf-8")
    results_handle = results_path.open("a", encoding="utf-8")
    errors_handle = errors_path.open("a", encoding="utf-8")
    batch_jobs = _load_existing_batch_jobs(batch_jobs_path)
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
                "execution_mode": execution_mode,
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
                raise RuntimeError("Google API key is required unless --dry-run is set.")
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

        if execution_mode in {"live-rest", "live-sdk"} and not dry_run and pending_calls:
            worker_count = max(1, int(max_workers))
            call_fn = _call_google_structured_rest if execution_mode == "live-rest" else _call_google_structured_sdk
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        call_fn,
                        api_key=api_key,
                        model=model,
                        prompt=item["prompt"],
                        schema=item["schema"],
                        temperature=temperature,
                        request_timeout_ms=request_timeout_ms,
                        thinking_budget=thinking_budget,
                        thinking_level=thinking_level,
                        **(
                            {}
                            if execution_mode == "live-rest"
                            else {
                                "api_version": api_version,
                                "retry_attempts": retry_attempts,
                                "retry_initial_delay": retry_initial_delay,
                                "retry_max_delay": retry_max_delay,
                                "retry_status_codes": retry_status_codes,
                            }
                        ),
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
                        error_key = (
                            str(error_row["match_id"]),
                            json.dumps(error_row["error"], ensure_ascii=False, sort_keys=True),
                        )
                        if error_key not in existing_error_keys:
                            errors_handle.write(json.dumps(error_row, ensure_ascii=False))
                            errors_handle.write("\n")
                            errors_handle.flush()
                            existing_error_keys[error_key] = True
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

        if execution_mode == "batch" and not dry_run:
            client = _new_google_client(
                api_key=api_key,
                api_version=api_version,
                request_timeout_ms=request_timeout_ms,
                retry_attempts=retry_attempts,
                retry_initial_delay=retry_initial_delay,
                retry_max_delay=retry_max_delay,
                retry_status_codes=retry_status_codes,
            )

            if pending_calls:
                existing_batch_match_ids = {
                    match_id
                    for row in batch_jobs
                    if str(row.get("state", "")) not in RETRYABLE_BATCH_STATES
                    for match_id in list(row.get("match_ids") or [])
                }
                unsubmitted_pending = [
                    item for item in pending_calls if str(item["match_id"]) not in existing_batch_match_ids
                ]
            else:
                unsubmitted_pending = []

            if unsubmitted_pending:
                shards = _shard_pending_calls_for_batch(
                    unsubmitted_pending,
                    model=model,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                    thinking_level=thinking_level,
                    max_inline_bytes=batch_inline_max_bytes,
                )
                submitted_jobs: List[Dict[str, Any]] = []
                for shard_offset, shard in enumerate(shards, start=len(batch_jobs)):
                    metadata_rows = [_batch_request_metadata(item) for item in shard]
                    request_bytes = sum(
                        _estimate_batch_request_bytes(
                            item,
                            model=model,
                            temperature=temperature,
                            thinking_budget=thinking_budget,
                            thinking_level=thinking_level,
                        )
                        for item in shard
                    )
                    batch_job = client.batches.create(
                        model=model,
                        src=[
                            {
                                "contents": str(item["prompt"]),
                                "config": _build_sdk_generation_config(
                                    schema=dict(item["schema"]),
                                    temperature=temperature,
                                    thinking_budget=thinking_budget,
                                    thinking_level=thinking_level,
                                ),
                                "metadata": metadata,
                            }
                            for item, metadata in zip(shard, metadata_rows)
                        ],
                        config={
                            "display_name": batch_display_name or output_dir.name,
                        },
                    )
                    submitted_jobs.append(
                        _serialize_batch_job(
                            shard_index=shard_offset,
                            request_count=len(shard),
                            request_bytes=request_bytes,
                            match_ids=[str(item["match_id"]) for item in shard],
                            batch_job=batch_job,
                        )
                    )
                batch_jobs.extend(submitted_jobs)
                _write_jsonl(batch_jobs_path, batch_jobs)

            if batch_jobs:
                while True:
                    refreshed_jobs: List[Dict[str, Any]] = []
                    all_terminal = True
                    for batch_job_row in batch_jobs:
                        batch_job = client.batches.get(name=str(batch_job_row["job_name"]))
                        refreshed_jobs.append(
                            _serialize_batch_job(
                                shard_index=int(batch_job_row.get("shard_index", 0)),
                                request_count=int(batch_job_row.get("request_count", 0)),
                                request_bytes=int(batch_job_row.get("request_bytes", 0)),
                                match_ids=list(batch_job_row.get("match_ids") or []),
                                batch_job=batch_job,
                            )
                        )
                        if _safe_batch_state(getattr(batch_job, "state", None)) not in TERMINAL_BATCH_STATES:
                            all_terminal = False
                    batch_jobs = refreshed_jobs
                    _write_jsonl(batch_jobs_path, batch_jobs)
                    batch_status_path.write_text(
                        json.dumps(
                            {
                                "output_dir": str(output_dir),
                                "model": model,
                                "job_count": len(batch_jobs),
                                "states": Counter(str(row.get("state", "")) for row in batch_jobs),
                                "completed_case_count": len(existing_result_by_id),
                                "pending_case_count": max(len(selected_rows) - len(existing_result_by_id), 0),
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    if all_terminal or batch_no_wait:
                        break
                    time.sleep(max(float(batch_poll_seconds), 1.0))

                if not batch_no_wait:
                    for batch_job_row in batch_jobs:
                        if str(batch_job_row.get("state", "")) not in SUCCESSFUL_BATCH_STATES:
                            continue
                        batch_job = client.batches.get(name=str(batch_job_row["job_name"]))
                        _ingest_batch_job_rows(
                            batch_job_row=batch_job_row,
                            batch_job=batch_job,
                            results_handle=results_handle,
                            errors_handle=errors_handle,
                            existing_result_by_id=existing_result_by_id,
                            existing_error_keys=existing_error_keys,
                            per_category_counts=per_category_counts,
                        )
            elif batch_no_wait:
                batch_status_path.write_text(
                    json.dumps(
                        {
                            "output_dir": str(output_dir),
                            "model": model,
                            "job_count": 0,
                            "states": {},
                            "completed_case_count": len(existing_result_by_id),
                            "pending_case_count": max(len(selected_rows) - len(existing_result_by_id), 0),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    finally:
        requests_handle.close()
        results_handle.close()
        errors_handle.close()

    batch_state_counts = Counter(str(row.get("state", "")) for row in batch_jobs)
    if execution_mode != "batch":
        batch_completed = True
    elif dry_run:
        batch_completed = True
    elif not batch_jobs:
        batch_completed = True
    else:
        batch_completed = all(
            str(row.get("state", "")) in TERMINAL_BATCH_STATES for row in batch_jobs
        )
    summary = {
        "bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "model": model,
        "execution_mode": execution_mode,
        "api_version": api_version,
        "dry_run": bool(dry_run),
        "selected_case_count": len(selected_rows),
        "completed_case_count": sum(1 for row in selected_rows if str(row.get("match_id", "")) in existing_result_by_id),
        "error_count": len(_read_jsonl(errors_path)),
        "categories": sorted({str(row.get("category", "")) for row in selected_rows}),
        "per_category_decision_counts": {category: dict(counter) for category, counter in per_category_counts.items()},
        "batch_job_count": len(batch_jobs),
        "batch_state_counts": dict(batch_state_counts),
        "batch_completed": bool(batch_completed),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review sampled token-category cases with Google structured output (Gemini/Gemma) in live or batch mode."
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--execution-mode", choices=["live-rest", "live-sdk", "batch"], default="live-sdk")
    parser.add_argument("--api-version", default="v1beta")
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--max-cases-per-category", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--request-timeout-ms", type=int, default=300_000)
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-initial-delay", type=float, default=1.0)
    parser.add_argument("--retry-max-delay", type=float, default=30.0)
    parser.add_argument(
        "--retry-status-code",
        action="append",
        dest="retry_status_codes",
        type=int,
        default=None,
        help="HTTP status code eligible for SDK retry; repeat to add multiple codes.",
    )
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--thinking-level", choices=["low", "medium", "high"], default=None)
    parser.add_argument("--batch-poll-seconds", type=float, default=30.0)
    parser.add_argument("--batch-inline-max-bytes", type=int, default=DEFAULT_BATCH_INLINE_MAX_BYTES)
    parser.add_argument("--batch-display-name", default=None)
    parser.add_argument("--batch-no-wait", action="store_true")
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
        execution_mode=args.execution_mode,
        api_version=args.api_version,
        request_timeout_ms=args.request_timeout_ms,
        retry_attempts=args.retry_attempts,
        retry_initial_delay=args.retry_initial_delay,
        retry_max_delay=args.retry_max_delay,
        retry_status_codes=args.retry_status_codes or list(DEFAULT_RETRY_STATUS_CODES),
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        batch_poll_seconds=args.batch_poll_seconds,
        batch_inline_max_bytes=args.batch_inline_max_bytes,
        batch_display_name=args.batch_display_name,
        batch_no_wait=bool(args.batch_no_wait),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
