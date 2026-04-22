"""Gemini broken-text quality reviewer — drives sample.jsonl through the
multi-axis quality prompt (see prompt_drafts/05_broken_text_quality_
review_prompt.md).

NO MASTER BINARY — seven orthogonal axes + short_reason. Rejection
policy is set post-hoc from label-vs-stats distributions (see
analyze_broken_text_verdicts.py).

Input:  sample.jsonl from sample_broken_text_candidates.py
Output: verdicts.jsonl (one row per sampled doc)

Same ThreadPoolExecutor shape as sample_and_review_line_vs_span.py.
Retries on transient failures, 10 concurrent workers by default.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("google-genai required. `pip install google-genai`", file=sys.stderr)
    sys.exit(1)


SYSTEM_INSTRUCTION = """\
You are reviewing a document drawn from a Greek-language pretraining
corpus. The document has been through a deterministic cleaner that:
- strips PDF-extraction residue (PostScript glyph names like /hyphenminus,
  /uni03B1, /g302; Adobe font-subset markers like /XQDMQS+CenturyGothic)
- removes individual lines dominated by PDF glyph IDs
- canonicalizes repeated special chars (dots ........ → ....., separator
  lines of ---/___/\\_\\_\\_ → ---)
- strips characters from scripts other than Greek, Latin, French-Spanish
  diacritics, punctuation, digits, common symbols
- collapses multi-space runs to bucketed canonical lengths

Where cleaning removed whole lines, the marker <!-- line-removed -->
appears inline. Where cleaning stripped a partial line's content, you may
see <!-- text-missing --> at line end. These markers are SIGNAL — they
tell you a discontinuity comes from cleaning, not from the source.

Some statistical variation is acceptable for pretraining: up to roughly
5% of the document can have isolated minor defects (missing punctuation,
small word-internal breaks, one-off awkward phrasings) AS LONG AS the
text is mostly sensible — the main subject can be understood end-to-end
and only minor details are lost. What is NOT acceptable is text where
the subject itself becomes unclear: pervasive incomplete sentences,
broken words mid-token, bad syntax, or disconnected fragments that leave
the reader unable to follow what the text is actually about.

Answer the structured questions below independently. There is no single
master accept/reject label — orthogonal signals are collected so the
rejection policy can be set later from the distribution shape.
"""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": [
        "defect_rate_estimate",
        "text_partition",
        "subject_clear_end_to_end",
        "has_broken_words_mid_token",
        "has_narrative_jumps_from_line_drops",
        "has_mid_thought_sentences",
        "is_too_short_to_be_useful",
        "short_reason",
    ],
    "properties": {
        "defect_rate_estimate": {
            "type": "STRING",
            "enum": [
                "≤5% (sensible, subject clear)",
                "5-20%",
                "20-50%",
                ">50%",
            ],
        },
        "text_partition": {
            "type": "STRING",
            "enum": [
                "uniformly_good",
                "uniformly_bad",
                "half_half",
                "mostly_good_with_bad_patches",
                "mostly_bad_with_good_patches",
            ],
        },
        "subject_clear_end_to_end": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "has_broken_words_mid_token": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "has_narrative_jumps_from_line_drops": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "has_mid_thought_sentences": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "is_too_short_to_be_useful": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "short_reason": {"type": "STRING"},
    },
}


def _smart_sample_context(text: str, target_chars: int = 500_000, n_chunks: int = 4) -> str:
    """Build the [CONTEXT] payload for Gemini.

    - Docs ≤ target_chars: pass through in full. Gemini 2.5 Flash has a
      1M-token context window; Greek is ~3 chars/token so 500k chars ≈
      170k tokens — well inside the standard <200k input-price tier.
      Passing the whole doc lets Gemini judge text_partition accurately
      (a multi-chunk sample of the same doc would hide defects in the
      gaps).
    - Docs > target_chars (rare — p99 doc length is 1.48M chars, only
      top ~1% exceed 500k): take n_chunks evenly-spaced chunks. This
      preserves beginning / middle-early / middle-late / end coverage.

    Prepends a metadata line so Gemini knows the doc's full length
    and sampling strategy — affects judgment of too_short_to_be_useful
    (a doc shown in full that reads short IS short; a chunk of a long
    doc shown short may not be).
    """
    total = len(text)
    if total <= target_chars:
        return f"[doc length: {total} chars, shown in full]\n\n{text}"
    chunk_size = target_chars // n_chunks
    positions = [i * total // n_chunks for i in range(n_chunks)]
    chunks = [text[p : p + chunk_size] for p in positions]
    sampled = "\n[...]\n".join(chunks)
    return (
        f"[doc length: {total} chars; shown: {n_chunks} evenly-spaced "
        f"samples of ~{chunk_size} chars each from positions "
        f"{[round(p / max(total, 1), 2) for p in positions]}]\n\n"
        + sampled
    )


def _build_prompt(row: Dict[str, Any], target_chars: int = 8000) -> str:
    cleaned = _smart_sample_context(row.get("cleaned_text") or "", target_chars=target_chars)
    body = (
        "[CONTEXT]\n"
        f"{cleaned}\n\n"
        "[QUESTIONS]\n"
        "1. defect_rate_estimate — one of:\n"
        '   "≤5% (sensible, subject clear)", "5-20%", "20-50%", ">50%"\n'
        "2. text_partition — one of:\n"
        '   "uniformly_good", "uniformly_bad", "half_half",\n'
        '   "mostly_good_with_bad_patches", "mostly_bad_with_good_patches"\n'
        "3. subject_clear_end_to_end (yes / no / uncertain)\n"
        "4. has_broken_words_mid_token (yes / no / uncertain)\n"
        "5. has_narrative_jumps_from_line_drops (yes / no / uncertain)\n"
        "6. has_mid_thought_sentences (yes / no / uncertain)\n"
        "7. is_too_short_to_be_useful (yes / no / uncertain)\n"
        "8. short_reason (≤ 40 words)\n"
    )
    return body


def _call_gemini(client, model, prompt, max_retries=4) -> Dict[str, Any]:
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
                model=model, contents=prompt, config=config,
            )
            return json.loads(resp.text or "{}")
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--context-chars", type=int, default=500_000,
                        help="Pass through in full if ≤ N chars (default 500k ≈ 170k "
                             "tokens — comfortable inside Flash's 1M context + standard "
                             "price tier). Docs > N get multi-chunk sampled as fallback.")
    parser.add_argument("--context-chunks", type=int, default=4,
                        help="Number of evenly-spaced chunks for long docs. Default 4.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, review only the first N docs (smoke test).")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"no API key in ${args.api_key_env}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verdicts_path = args.output_dir / "verdicts.jsonl"

    rows = []
    with args.sample_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"{len(rows)} sample docs to review "
          f"(context: {args.context_chars} chars × up to {args.context_chunks} chunks)")

    client = genai.Client(api_key=api_key)

    errors = 0
    start = time.time()
    ctx_chars = args.context_chars
    ctx_chunks = args.context_chunks
    def _task(r):
        prompt = _build_prompt(r, target_chars=ctx_chars)
        return (r, _call_gemini(client, args.model, prompt))
    with verdicts_path.open("w", encoding="utf-8") as out, \
         cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_task, r) for r in rows]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                row, verdict = fut.result()
                out.write(json.dumps({
                    "source_path": row.get("source_path"),
                    "source_dataset": row.get("source_dataset"),
                    "source_doc_id": row.get("source_doc_id"),
                    "metrics": row.get("metrics"),
                    "zones": row.get("zones"),
                    "stats_snapshot": row.get("stats_snapshot"),
                    "verdict": verdict,
                }, ensure_ascii=False) + "\n")
                out.flush()
            except Exception as exc:
                errors += 1
                out.write(json.dumps({"error": str(exc)}) + "\n")
            if i % 25 == 0:
                el = time.time() - start
                rate = i / el if el else 0
                print(
                    f"  {i}/{len(rows)} elapsed={el:.0f}s rate={rate:.1f}/s errors={errors}",
                    flush=True,
                )
    print(f"done: {len(rows)} cases, {errors} errors, "
          f"{time.time() - start:.0f}s wall. verdicts → {verdicts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
