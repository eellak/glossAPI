"""Classify cmark-gfm verifier failures by root cause + index each
failure back to its source MD line number, so a reviewer can jump
directly to the problematic span instead of scanning the full doc.

For each failing pair from cmark_verify_report.json:

1. Parse the `first_diff` message to extract the HTML-byte position
   and surrounding text snippet on both sides.
2. Strip HTML tags from the snippet to get literal text content.
3. Grep that content in the BEFORE (raw) source MD to find the line
   number — that's the diff-→source index.
4. Classify by signature: URL-escape diff, hard-break loss, table
   boundary shift, bracket-escape diff, other.

Output: per-pair JSON record with {pair, bucket, before_source_line,
html_context, source_context}.

Usage:
  python3 classify_cmark_failures.py \\
      --report ~/data/phase_a_audit/cmark_verify_report.json \\
      --sample-dir ~/data/phase_a_audit/top100_review \\
      --output ~/data/phase_a_audit/cmark_failures_indexed.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HDR = re.compile(
    r"^##\s+(?:Raw input \(pre-Phase-A\) — BEFORE|Post-Phase-A output — AFTER)\s*$",
    re.MULTILINE,
)


def extract_body(md_path: Path) -> tuple[str, int]:
    """Return (body, starting_line_number_in_file)."""
    text = md_path.read_text(encoding="utf-8")
    anc = HDR.search(text)
    if not anc:
        return text, 1
    rest = text[anc.end():]
    div = re.search(r"^---\s*\n", rest, re.MULTILINE)
    if not div:
        return rest.lstrip("\n"), 1
    body_start_in_rest = div.end()
    body_start_in_text = anc.end() + body_start_in_rest
    # How many newlines before body starts.
    line_of_body_start = text[:body_start_in_text].count("\n") + 1
    body = rest[body_start_in_rest:].lstrip("\n")
    return body, line_of_body_start


def strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def classify(first_diff: str) -> str:
    """Bucket each failure by signature patterns in the first_diff."""
    if "%5C_" in first_diff or "\\_" in first_diff:
        if "%5C_" in first_diff and "\\_" in first_diff:
            return "url_escape_decoded"  # comrak decodes %5C to \ in URLs
    if "<br" in first_diff and "<br" not in first_diff.split("out:", 1)[-1]:
        return "hard_break_lost_input_had_br"
    if "<table" in first_diff and "<table" not in first_diff.split("in:", 1)[-1].split("out:", 1)[0]:
        return "table_detection_output_has_table"
    if "<table" in first_diff.split("in:", 1)[-1].split("out:", 1)[0]:
        if "<table" not in first_diff.split("out:", 1)[-1]:
            return "table_detection_input_had_table"
    if "<pre>" in first_diff or "<code>" in first_diff:
        return "code_block_boundary_shift"
    if "&amp;" in first_diff:
        return "html_entity_diff"
    if "[" in first_diff and "\\[" in first_diff:
        return "bracket_escape_diff"
    return "other"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--sample-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    report = json.load(args.report.open())
    fails = report.get("fails", [])
    out_rows = []
    buckets = Counter()
    for f in fails:
        pair = f["pair"]
        bucket = classify(f.get("first_diff", ""))
        buckets[bucket] += 1
        # Parse first_diff to extract the "in:" text snippet.
        m = re.search(r"in:\s+(.+?)\n\s+out:", f.get("first_diff", ""), re.DOTALL)
        html_in_snippet = m.group(1).strip() if m else ""
        # Extract a letter-only signature that survives ALL render
        # differences (HTML entity encoding, URL escape, backslash
        # escape, CM escape). Letters + digits + spaces are preserved
        # verbatim; symbols / punctuation can differ.
        text_snippet = strip_tags(html_in_snippet)
        # Replace common HTML entities with their literal form for
        # matching (amp/lt/gt/quot suffice for our corpus).
        text_snippet = (
            text_snippet
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
        )
        text_snippet = re.sub(r"\s+", " ", text_snippet).strip()
        # Find the longest contiguous letter/digit/space run (≥ 20
        # chars) — this is the robust signature.
        letter_runs = re.findall(
            r"[\w ]+", text_snippet, flags=re.UNICODE
        )
        letter_runs = [r.strip() for r in letter_runs if len(r.strip()) >= 20]
        letter_runs.sort(key=len, reverse=True)
        search_key = letter_runs[0][:80] if letter_runs else text_snippet[:50]

        before_md = args.sample_dir / f"{pair}_BEFORE.md"
        source_line = None
        source_context = None
        if before_md.is_file():
            body, body_line_start = extract_body(before_md)
            # First try full search_key direct substring.
            idx = -1
            if search_key:
                idx = body.find(search_key)
            # Fall back: try whitespace-flexible match (src may have
            # internal `\n` where html has ` `).
            if idx < 0 and search_key:
                parts = search_key.split()
                if parts:
                    pattern = r"\s+".join(re.escape(p) for p in parts)
                    m = re.search(pattern, body)
                    if m:
                        idx = m.start()
            # Final fallback: find the FIRST distinctive word (≥ 6
            # chars) from the snippet in the body. Broad but still
            # gives an approximate jump point.
            if idx < 0:
                words = re.findall(r"\w{6,}", text_snippet)
                for w in words:
                    i = body.find(w)
                    if i >= 0:
                        idx = i
                        search_key = w
                        break
            if idx >= 0:
                source_line = body[:idx].count("\n") + body_line_start
                ctx_start = max(0, idx - 60)
                ctx_end = min(len(body), idx + len(search_key) + 60)
                source_context = body[ctx_start:ctx_end]
        out_rows.append({
            "pair": pair,
            "bucket": bucket,
            "html_in_snippet": html_in_snippet[:300],
            "source_key": search_key,
            "source_line_in_file": source_line,
            "source_context": source_context[:300] if source_context else None,
        })

    out_rows.sort(key=lambda r: (r["bucket"], r["pair"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "total_fails": len(fails),
                "buckets": dict(buckets.most_common()),
                "failures": out_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"classified {len(fails)} failures")
    print("buckets:")
    for b, n in buckets.most_common():
        print(f"  {n:>3}  {b}")
    n_indexed = sum(1 for r in out_rows if r["source_line_in_file"])
    print(f"source-line index resolved: {n_indexed} / {len(fails)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
