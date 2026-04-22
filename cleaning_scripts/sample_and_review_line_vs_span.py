"""Sample lines that hit the 2 unreviewed rule sets, send to Gemini
asking whether the whole line or just the match should drop.

Input:
  Per-dataset `*.line_matches.jsonl` files from
  `extract_ps_glyph_line_matches.py`.

For each of the two rule sets (A = PS-glyph literals, B = PS-glyph
regex), draw a random sample of `--sample-size` lines (default 1000),
compose a prompt per the locked inspection format:

  [CONTEXT]
  <context_before + line_with_match_tagged + context_after>
  (inline <match kind="ps_glyph_literal">...</match> or <match kind="ps_glyph_regex">...</match>)

  [QUESTIONS]
  1. is_match_noise              (yes / no / uncertain)
  2. should_drop_whole_line       (yes / no / uncertain)
  3. surrounding_prose_is_legitimate_greek (yes / no / uncertain)
  4. short_reason                 (≤ 40 words)

Concurrent submission via ThreadPoolExecutor, same pattern as
`gemini_three_counter_reviewer.py`. Writes verdicts.jsonl per rule set.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    print("google-genai required. `pip install google-genai`", file=sys.stderr)
    sys.exit(1)


SYSTEM_INSTRUCTION = (
    "You are reviewing a LINE of text from a Greek-language corpus. The "
    "cleaner has detected a pattern match (PostScript glyph-name residue "
    "from broken PDF extraction). Your job: decide whether (a) the match "
    "is actual noise we should strip, and (b) whether the WHOLE LINE "
    "should be dropped, or JUST THE MATCH SPAN. Judge conservatively — "
    "if the match is a small span inside otherwise-legitimate Greek "
    "prose, preferring `should_drop_whole_line=no` preserves that prose."
)


RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "required": ["is_match_noise", "should_drop_whole_line",
                 "surrounding_prose_is_legitimate_greek", "short_reason"],
    "properties": {
        "is_match_noise": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "should_drop_whole_line": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "surrounding_prose_is_legitimate_greek": {"type": "STRING", "enum": ["yes", "no", "uncertain"]},
        "short_reason": {"type": "STRING"},
    },
}


def _build_prompt(row: Dict[str, Any], rule_set: str) -> str:
    """rule_set ∈ {'A', 'B'}. Highlights match type in the inline tag."""
    kind = "ps_glyph_literal" if rule_set == "A" else "ps_glyph_regex"
    line = row["line_text"]
    context_before = "\n".join(row.get("context_before", []))
    context_after = "\n".join(row.get("context_after", []))

    # Insert inline match tags around the actual span(s). For rule A we
    # have exact literal strings; for rule B we have regex captures as
    # strings.
    matches = [m[0] if isinstance(m, (list, tuple)) else m
               for m in row.get(f"matches_rule_{rule_set.lower()}", [])]
    tagged = line
    for m in sorted(set(matches), key=len, reverse=True):
        tagged = tagged.replace(m, f'<match kind="{kind}">{m}</match>')

    passage = ""
    if context_before:
        passage += context_before + "\n"
    passage += tagged + "\n"
    if context_after:
        passage += context_after
    body = (
        f"[CONTEXT]\n{passage}\n\n"
        f"[QUESTIONS]\n"
        f"1. is_match_noise (yes / no / uncertain)\n"
        f"2. should_drop_whole_line (yes / no / uncertain)\n"
        f"3. surrounding_prose_is_legitimate_greek (yes / no / uncertain)\n"
        f"4. short_reason (≤ 40 words)\n"
    )
    return body


def _sample_rows(stats_dir: Path, rule: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """rule ∈ {'A', 'B'}. Returns random n rows where the given rule has matches."""
    assert rule in ("A", "B")
    key = "match_count_rule_a" if rule == "A" else "match_count_rule_b"
    all_rows: List[Dict[str, Any]] = []
    for p in sorted(stats_dir.glob("*.line_matches.jsonl")):
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get(key, 0) > 0:
                    all_rows.append(d)
    rng = random.Random(seed)
    rng.shuffle(all_rows)
    return all_rows[:n]


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
    parser.add_argument("--line-matches-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--rules", nargs="+", default=["A", "B"])
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"no API key in ${args.api_key_env}", file=sys.stderr)
        return 2

    client = genai.Client(api_key=api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for rule in args.rules:
        rows = _sample_rows(args.line_matches_dir, rule, args.sample_size, args.seed)
        print(f"RULE {rule}: {len(rows)} cases (requested {args.sample_size})")
        verdicts_path = args.output_dir / f"verdicts_rule_{rule}.jsonl"
        errors = 0
        start = time.time()
        with verdicts_path.open("w", encoding="utf-8") as out, \
             cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    lambda r=r:(r, _call_gemini(client, args.model, _build_prompt(r, rule)))
                )
                for r in rows
            ]
            for i, fut in enumerate(cf.as_completed(futures), 1):
                try:
                    row, verdict = fut.result()
                    out.write(json.dumps({
                        "rule_set": rule,
                        "source_path": row.get("source_path"),
                        "source_dataset": row.get("source_dataset"),
                        "source_doc_id": row.get("source_doc_id"),
                        "line_number": row.get("line_number"),
                        "line_char_count": row.get("line_char_count"),
                        "match_count": row.get(f"match_count_rule_{rule.lower()}"),
                        "matches": row.get(f"matches_rule_{rule.lower()}"),
                        "verdict": verdict,
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                except Exception as exc:
                    errors += 1
                    out.write(json.dumps({
                        "rule_set": rule, "error": str(exc),
                    }) + "\n")
                if i % 50 == 0:
                    el = time.time() - start
                    rate = i / el if el else 0
                    print(
                        f"  rule {rule} {i}/{len(rows)}  "
                        f"elapsed={el:.0f}s rate={rate:.1f}/s errors={errors}",
                        flush=True,
                    )
        print(f"RULE {rule} done: {len(rows)} cases, {errors} errors, "
              f"{time.time() - start:.0f}s wall. verdicts → {verdicts_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
