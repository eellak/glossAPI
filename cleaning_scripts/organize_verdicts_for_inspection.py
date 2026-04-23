"""Organize a verdicts.jsonl + sample.jsonl pair into a browsable
directory tree for manual inspection.

Layout:
  <output-dir>/
    band_<low_or_high>/
      subject_clear_<yes|no|uncertain>/
        <zero-padded-char-strip-pct>_<defect-bucket>_<dataset>_<doc-id>.md
        ...
      INDEX.md          per-band-per-subject-clear roll-up
    INDEX.md            top-level roll-up with counts per cell

Each leaf .md contains:
  - Source metadata (dataset, doc_id, bands, metrics)
  - Full Gemini verdict (all 7 axes + short_reason)
  - Cleaned text body (from sample.jsonl, truncated if long)

Filename prefix = zero-padded char_strip_ratio (rounded %) so `ls` orders
docs by their actual stripping damage within each cell — per
feedback_metric_prefix_in_sample_filenames.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _safe_fname(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))
    return s[:maxlen].strip("_") or "x"


def _defect_bucket_code(v: str) -> str:
    m = {
        "≤5% (sensible, subject clear)": "le5",
        "5-20%": "lo20",
        "20-50%": "lo50",
        ">50%": "gt50",
    }
    return m.get(v, "unk")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts-path", required=True, type=Path)
    parser.add_argument("--sample-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--body-chars", type=int, default=8000,
                        help="Truncate cleaned body to N chars in the .md "
                             "(show head+tail). 0 = show full.")
    args = parser.parse_args(argv)

    # Load sample (has cleaned_text + band).
    sample_by_path: Dict[str, Dict[str, Any]] = {}
    with args.sample_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            sample_by_path[d["source_path"]] = d

    # Load verdicts.
    verdicts = []
    with args.verdicts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("verdict"):
                verdicts.append(d)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    for v in verdicts:
        sp = v.get("source_path")
        s = sample_by_path.get(sp)
        if s is None:
            print(f"[skip] no sample for {sp}")
            continue
        band = s.get("band", "unk")
        verdict = v["verdict"]
        sc = verdict.get("subject_clear_end_to_end", "unknown")
        defect = verdict.get("defect_rate_estimate", "unknown")
        ds = v.get("source_dataset", "unk")
        did = str(v.get("source_doc_id", "unk"))[:40]

        char_strip = float(s.get("metrics", {}).get("char_strip_ratio", 0.0))
        prefix = f"{int(round(char_strip)):03d}"

        folder = args.output_dir / f"band_{band}" / f"subject_clear_{sc}"
        folder.mkdir(parents=True, exist_ok=True)

        fname = (
            f"{prefix}_{_defect_bucket_code(defect)}"
            f"_{_safe_fname(ds, 20)}_{_safe_fname(did, 30)}.md"
        )

        # Build file body.
        body_lines = [f"# {ds} / {did}", ""]
        body_lines.append(f"- **band**: {band}")
        body_lines.append(f"- **char_strip_ratio**: {char_strip:.2f}%")
        for k in [
            "pct_chars_removed_non_empty", "non_empty_chars_in",
            "non_empty_chars_out",
        ]:
            val = s.get("metrics", {}).get(k)
            body_lines.append(f"- **{k}**: {val}")
        body_lines.append("")
        body_lines.append("## Gemini verdict")
        body_lines.append("")
        for k in [
            "defect_rate_estimate", "text_partition",
            "subject_clear_end_to_end", "has_broken_words_mid_token",
            "has_narrative_jumps_from_line_drops",
            "has_mid_thought_sentences", "is_too_short_to_be_useful",
        ]:
            body_lines.append(f"- **{k}**: `{verdict.get(k, '?')}`")
        body_lines.append("")
        body_lines.append(f"**short_reason**: {verdict.get('short_reason', '')}")
        body_lines.append("")
        body_lines.append("## Cleaned text")
        body_lines.append("")
        body_lines.append("```")
        text = s.get("cleaned_text") or ""
        if args.body_chars > 0 and len(text) > args.body_chars:
            half = args.body_chars // 2
            body_lines.append(text[:half])
            body_lines.append(f"\n[...truncated {len(text) - args.body_chars} chars...]\n")
            body_lines.append(text[-half:])
        else:
            body_lines.append(text)
        body_lines.append("```")

        (folder / fname).write_text("\n".join(body_lines), encoding="utf-8")
        key = f"band_{band}/subject_clear_{sc}"
        counts[key] = counts.get(key, 0) + 1

    # Top-level INDEX.
    index_lines = ["# Smoke-test verdict inspection index", "",
                   "## Counts per cell", "",
                   "| band | subject_clear | N |",
                   "|---|---|---:|"]
    for key, n in sorted(counts.items()):
        band, _, sc = key.partition("/")
        index_lines.append(f"| {band} | {sc} | {n} |")
    index_lines.extend(["", "## Layout", "",
                        "```",
                        str(args.output_dir) + "/",
                        "  band_low/",
                        "    subject_clear_yes/  — docs Gemini judged as Greek-coherent",
                        "    subject_clear_no/   — docs Gemini judged as not-coherent",
                        "    subject_clear_uncertain/",
                        "  band_high/",
                        "    (same structure)",
                        "```",
                        "",
                        "**Filenames** are prefixed with zero-padded `char_strip_ratio` (%), "
                        "so `ls` in each leaf folder orders docs by their actual cleaning "
                        "damage. Second segment is the `defect_rate_estimate` bucket Gemini "
                        "chose (`le5` / `lo20` / `lo50` / `gt50`). Last segments are "
                        "dataset/doc_id.",
                        ""])
    (args.output_dir / "INDEX.md").write_text("\n".join(index_lines),
                                               encoding="utf-8")
    print(f"wrote {sum(counts.values())} docs across {len(counts)} cells")
    print(f"index → {args.output_dir}/INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
