"""Submit the three-counter page-level samples to Gemini for review.

Reads a sample folder (produced by stratified_three_counter_sampler.py) with
per-counter subdirectories of Markdown case files. Calls gemini-2.5-flash
with a strict JSON response schema matching Task type 2 (keep_or_drop /
noise_character / dominant_signal / short_reason). Writes verdicts to
`verdicts.jsonl` at the root of the sample folder, one row per case.

Client-side join: case files carry an `<!-- case_meta={...} -->` HTML
comment with `match_id`, counter values, zone, etc. We copy that meta into
the verdict row so post-hoc threshold calibration can correlate Gemini's
`keep_or_drop` with our stored counter values WITHOUT ever sending those
values to the model (per `feedback_data_inspection_format.md`).

Concurrency: ThreadPoolExecutor with `--workers` parallel requests. Each
request sets a max-retry loop for transient 429/500s. Total wall-clock for
~150 cases at 10 workers ≈ 3–5 minutes.

Model: gemini-2.5-flash (cheapest production tier as of 2026-04; ~$0.30/1M
input tokens, ~$1.20/1M output tokens). Expected spend for 150 cases at
~4000 input + ~100 output tokens each: ~$0.20.

Run on the laptop; requires `google-genai` and `GEMINI_API_KEY` env var.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("google-genai is required. `pip install google-genai`", file=sys.stderr)
    sys.exit(1)


META_RE = re.compile(r"<!--\s*case_meta=(\{.*?\})\s*-->", re.DOTALL)
CONTEXT_RE = re.compile(r"\[CONTEXT\]\s*(.*?)\s*\[QUESTIONS\]", re.DOTALL)


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": ["keep_or_drop", "noise_character", "dominant_signal", "short_reason"],
    "properties": {
        "keep_or_drop": {"type": "STRING", "enum": ["keep", "drop", "uncertain"]},
        "noise_character": {
            "type": "STRING",
            "enum": [
                "clean",
                "mojibake",
                "glyph_corruption",
                "script_salad",
                "garbled_text_other",
                "mixed",
                "unclear",
            ],
        },
        "dominant_signal": {
            "type": "STRING",
            "enum": [
                "font_names",
                "glyph_tags",
                "script_residue",
                "other_unknown",
                "none",
                "multiple",
            ],
        },
        "short_reason": {"type": "STRING"},
    },
}


SYSTEM_INSTRUCTION = (
    "You are reviewing a passage of Greek-language text extracted from a "
    "document corpus. Judge whether the page is RECOVERABLE signal for "
    "tokenizer training, or corrupted beyond recovery.\n\n"
    "Answer all four questions. If the page is clean Greek prose, answer "
    "`keep_or_drop=keep`, `noise_character=clean`, `dominant_signal=none`. "
    "If the page is dominated by PDF-extraction artefacts, rendering "
    "corruption, or mixed-script garble, answer `keep_or_drop=drop` with "
    "the most fitting noise_character and dominant_signal. Use "
    "`garbled_text_other` for noise patterns that don't fit the named "
    "categories, and `other_unknown` for dominant signals we don't list — "
    "`short_reason` should briefly describe what you saw."
)


def _parse_case_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    meta_match = META_RE.search(text)
    meta: Dict[str, Any] = {}
    if meta_match:
        try:
            meta = json.loads(meta_match.group(1))
        except json.JSONDecodeError:
            pass
    ctx_match = CONTEXT_RE.search(text)
    context = ctx_match.group(1).strip() if ctx_match else text
    # prompt sent to the model = the full file body WITHOUT the meta comment
    prompt_body = text
    if meta_match:
        prompt_body = text.replace(meta_match.group(0) + "\n\n", "", 1).replace(
            meta_match.group(0), "", 1
        )
    prompt_body = prompt_body.strip()
    return {
        "case_file": str(path),
        "meta": meta,
        "context_preview": context[:200],
        "prompt": prompt_body,
    }


def _call_gemini(
    client: "genai.Client",
    model: str,
    prompt: str,
    max_retries: int = 4,
) -> Dict[str, Any]:
    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.0,
        candidate_count=1,
    )
    attempt = 0
    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            text = resp.text or "{}"
            return json.loads(text)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def _process_case(
    client: "genai.Client", model: str, case_path: Path
) -> Dict[str, Any]:
    parsed = _parse_case_file(case_path)
    try:
        verdict = _call_gemini(client, model, parsed["prompt"])
        return {
            "case_file": parsed["case_file"],
            "meta": parsed["meta"],
            "context_preview": parsed["context_preview"],
            "verdict": verdict,
            "error": None,
        }
    except Exception as exc:
        return {
            "case_file": parsed["case_file"],
            "meta": parsed["meta"],
            "context_preview": parsed["context_preview"],
            "verdict": None,
            "error": str(exc),
        }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument(
        "--limit", type=int, default=None, help="optional cap for testing"
    )
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"no API key in ${args.api_key_env}", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)

    case_files: List[Path] = []
    for counter_dir in sorted(p for p in args.sample_dir.iterdir() if p.is_dir()):
        for case in sorted(counter_dir.glob("*.md")):
            case_files.append(case)
    if args.limit:
        case_files = case_files[: args.limit]

    print(f"reviewing {len(case_files)} cases with {args.model} × {args.workers} workers")

    verdicts_path = args.sample_dir / "verdicts.jsonl"
    errors = 0
    start = time.time()
    with verdicts_path.open("w", encoding="utf-8") as out, cf.ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:
        futures = [
            ex.submit(_process_case, client, args.model, case) for case in case_files
        ]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            result = fut.result()
            if result["error"]:
                errors += 1
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed else 0
                print(
                    f"  {i}/{len(case_files)} done   "
                    f"elapsed={elapsed:.0f}s   "
                    f"rate={rate:.1f}/s   "
                    f"errors={errors}"
                )

    elapsed = time.time() - start
    print(
        f"\n{len(case_files)} cases, {errors} errors, {elapsed:.1f}s wall clock"
    )
    print(f"verdicts written to {verdicts_path}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
