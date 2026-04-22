"""Extract corpus lines that match the two unreviewed line-drop rule sets
from the raw glossapi corpus, with per-line match counts.

Rule set A — PS glyph-name LITERALS:
    /hyphenminus /space /period /comma /colon /semicolon /slash
    /backslash /parenleft /parenright /bracketleft /bracketright
    /braceleft /braceright /quotesingle /quotedbl /exclam /question
    /asterisk /plus /minus /equal /less /greater /ampersand /percent
    /at /dollar /numbersign /underscore /asciitilde /asciicircum
    /endash /emdash /hyphen /bullet /copyright /registered /trademark
    /degree /plusminus /multiply /divide /section /paragraph /dagger
    /daggerdbl /ellipsis /glyph CID+

Rule set B — PS glyph REGEX:
    /uni[0-9A-Fa-f]{4,6}   (Unicode-codepoint PS glyph name)
    /g(?:id)?\d+           (glyph-ID PS name)

For each matching line, record:
    {
      source_path, source_doc_id, source_dataset,
      line_number, line_text, line_char_count,
      match_count_rule_a, match_count_rule_b,
      matches_rule_a: [...], matches_rule_b: [...],
      context_before: [N lines], context_after: [N lines],
    }

Context window: 5 lines above + 5 lines below by default. Outputs one
JSONL per dataset plus a global totals.json.

Run on the laptop (glossapi corpus is at /home/foivos/data/glossapi_work/
unified_corpus/data/*.parquet).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq


RULE_A_LITERALS = [
    "/hyphenminus", "/space", "/period", "/comma", "/colon", "/semicolon",
    "/slash", "/backslash", "/parenleft", "/parenright",
    "/bracketleft", "/bracketright", "/braceleft", "/braceright",
    "/quotesingle", "/quotedbl", "/exclam", "/question",
    "/asterisk", "/plus", "/minus", "/equal", "/less", "/greater",
    "/ampersand", "/percent", "/at", "/dollar", "/numbersign",
    "/underscore", "/asciitilde", "/asciicircum",
    "/endash", "/emdash", "/hyphen", "/bullet",
    "/copyright", "/registered", "/trademark", "/degree",
    "/plusminus", "/multiply", "/divide", "/section",
    "/paragraph", "/dagger", "/daggerdbl", "/ellipsis",
    "/glyph", "CID+",
]

RULE_B_REGEX_STR = r"(/uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+)"
RULE_B_REGEX = re.compile(RULE_B_REGEX_STR)

# For rule A, compile a single alternation regex for efficiency
RULE_A_REGEX = re.compile(
    "|".join(re.escape(lit) for lit in sorted(RULE_A_LITERALS, key=len, reverse=True))
)


def extract_from_text(
    text: str,
    source_path: str,
    source_doc_id: str,
    source_dataset: str,
    context_lines: int,
) -> List[Dict[str, Any]]:
    """Return one row per line that has at least one match in rule A or B."""
    lines = text.split("\n")
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        matches_a = RULE_A_REGEX.findall(line)
        matches_b = RULE_B_REGEX.findall(line)
        if not matches_a and not matches_b:
            continue
        cb_lo = max(0, i - context_lines)
        cb_hi = i
        ca_lo = i + 1
        ca_hi = min(len(lines), i + 1 + context_lines)
        out.append({
            "source_path": source_path,
            "source_doc_id": source_doc_id,
            "source_dataset": source_dataset,
            "line_number": i,
            "line_text": line,
            "line_char_count": len(line),
            "match_count_rule_a": len(matches_a),
            "match_count_rule_b": len(matches_b),
            "matches_rule_a": list(Counter(matches_a).most_common()),
            "matches_rule_b": list(Counter(matches_b).most_common()),
            "context_before": lines[cb_lo:cb_hi],
            "context_after": lines[ca_lo:ca_hi],
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--dataset-column", default="source_dataset")
    parser.add_argument("--context-lines", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pat in args.input_glob:
        paths.extend(Path(p).resolve() for p in globmod.glob(pat))
    paths = sorted(dict.fromkeys(paths))
    print(f"{len(paths)} parquets")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "lines_with_rule_a_matches": 0,
        "lines_with_rule_b_matches": 0,
        "lines_with_either": 0,
        "total_match_count_rule_a": 0,
        "total_match_count_rule_b": 0,
        "per_dataset": {},
        "per_literal_rule_a": {},
        "per_pattern_rule_b": Counter(),
    }
    per_ds_counter: Counter = Counter()

    for p in paths:
        print(f"[scan] {p.name}")
        rows: List[Dict[str, Any]] = []
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(
            batch_size=args.batch_size,
            columns=[args.text_column, args.doc_id_column, args.dataset_column],
        ):
            for row in batch.to_pylist():
                text = row.get(args.text_column) or ""
                if not text:
                    continue
                doc_id = str(row.get(args.doc_id_column) or "")
                ds = str(row.get(args.dataset_column) or p.stem)
                source_path = f"{p}#{doc_id}"
                rows.extend(extract_from_text(
                    text, source_path, doc_id, ds, args.context_lines
                ))
        out_path = args.output_dir / f"{p.stem}.line_matches.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Aggregate totals
        ds_name = rows[0]["source_dataset"] if rows else p.stem
        for r in rows:
            if r["match_count_rule_a"]:
                totals["lines_with_rule_a_matches"] += 1
                totals["total_match_count_rule_a"] += r["match_count_rule_a"]
                for lit, n in r["matches_rule_a"]:
                    totals["per_literal_rule_a"][lit] = totals["per_literal_rule_a"].get(lit, 0) + n
            if r["match_count_rule_b"]:
                totals["lines_with_rule_b_matches"] += 1
                totals["total_match_count_rule_b"] += r["match_count_rule_b"]
                for pat, n in r["matches_rule_b"]:
                    totals["per_pattern_rule_b"][pat] += n
            totals["lines_with_either"] += 1
        per_ds_counter[ds_name] += len(rows)
        print(f"  {p.name}: {len(rows)} matching lines")

    totals["per_dataset"] = dict(per_ds_counter.most_common())
    totals["per_pattern_rule_b"] = dict(totals["per_pattern_rule_b"])
    (args.output_dir / "totals.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTOTALS:")
    print(f"  rule A (literals) lines: {totals['lines_with_rule_a_matches']:,}   matches: {totals['total_match_count_rule_a']:,}")
    print(f"  rule B (regex)    lines: {totals['lines_with_rule_b_matches']:,}   matches: {totals['total_match_count_rule_b']:,}")
    print(f"  per dataset: {totals['per_dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
