"""One-off focused sampler: pick N docs from two char_strip_ratio bands
for the first Gemini smoke test.

Rationale (2026-04-22): `pct_chars_removed_non_empty` is dominated by
whitespace bucketing (normalization) and is not a content-loss signal.
`char_strip_ratio = chars_dropped_by_per_char_filter / non_empty_chars_in`
is where we actually stripped content. User wants to first review docs
at two contrast points — moderate (~30%) and heavy (~70%) content-loss
— to see if Gemini's verdicts discriminate cleanly and to debug the
review format before running the full wave.

Filters:
  - cleaning-altered: chars_dropped_by_per_char_filter > 0 OR
    lines_dropped_by_cleaner > 0 (not normalization-only)
  - non_empty_chars_in >= 500 (skip trivially short docs)

Bands (char_strip_ratio %):
  - --low-band (default 25..35)
  - --high-band (default 50..100)

For each sampled doc: re-load the parquet row, re-run the cleaner
(producing newline-preserved output), write the sample.jsonl record
compatible with gemini_broken_text_reviewer.py.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation",
                   "numbers", "common_symbols"]


def _load_and_filter(stats_glob: str,
                     low_band: Tuple[float, float],
                     high_band: Tuple[float, float],
                     min_chars: int) -> Tuple[List[Dict], List[Dict]]:
    low, high = [], []
    for path in sorted(globmod.glob(stats_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("drop_reason"):
                    continue
                if "content_chars_kept" not in d:
                    continue
                pc = int(d.get("chars_dropped_by_per_char_filter", 0) or 0)
                ld = int(d.get("lines_dropped_by_cleaner", 0) or 0)
                inp = int(d.get("non_empty_chars_in", 0) or 0)
                if pc == 0 and ld == 0:
                    continue
                if inp < min_chars:
                    continue
                ratio = 100.0 * pc / max(inp, 1)
                if low_band[0] <= ratio <= low_band[1]:
                    low.append((ratio, d))
                elif high_band[0] <= ratio <= high_band[1]:
                    high.append((ratio, d))
    return [d for _, d in low], [d for _, d in high]


def _find_row(parquet_path: str, doc_id: str,
              doc_id_column: str, text_column: str) -> Optional[str]:
    # Slow fallback (single-doc lookup). Prefer _bulk_find when picking
    # >1 doc from the same parquet.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            if str(row.get(doc_id_column)) == str(doc_id):
                return row.get(text_column) or ""
    return None


def _bulk_find(parquet_path: str, doc_ids: List[str],
               doc_id_column: str, text_column: str) -> Dict[str, str]:
    """Scan a parquet ONCE and return {doc_id: text} for all requested ids
    found. Orders of magnitude faster than _find_row per doc when multiple
    picks come from the same parquet file."""
    import pyarrow.parquet as pq
    want = set(str(d) for d in doc_ids)
    out: Dict[str, str] = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=5000):
        for row in batch.to_pylist():
            did = str(row.get(doc_id_column))
            if did in want:
                out[did] = row.get(text_column) or ""
                want.discard(did)
                if not want:
                    return out
    return out


def _clean_with_newlines(text: str, scripts_to_keep: List[str]) -> str:
    import glossapi_rs_cleaner as cleaner
    return cleaner.clean_text(text, scripts_to_keep)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--parquet-dir", required=True, type=Path,
                        help="Directory holding the *.parquet files referenced in "
                             "source_path (e.g. .../unified_corpus/data/).")
    parser.add_argument("--output-path", required=True, type=Path,
                        help="Output sample.jsonl path")
    parser.add_argument("--per-band", type=int, default=25)
    parser.add_argument("--low-band", default="25,35",
                        help="Low char_strip_ratio band 'lo,hi' in percent")
    parser.add_argument("--high-band", default="50,100",
                        help="High char_strip_ratio band 'lo,hi' in percent")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--scripts", nargs="+", default=DEFAULT_SCRIPTS)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args(argv)

    lo_band = tuple(float(x) for x in args.low_band.split(","))
    hi_band = tuple(float(x) for x in args.high_band.split(","))
    rng = random.Random(args.seed)

    print(f"loading stats; bands low={lo_band}, high={hi_band} ...", flush=True)
    low, high = _load_and_filter(args.stats_glob, lo_band, hi_band,
                                  args.min_chars)
    print(f"  low band population:  {len(low)}")
    print(f"  high band population: {len(high)}")

    rng.shuffle(low)
    rng.shuffle(high)
    picks = low[: args.per_band] + high[: args.per_band]
    print(f"  picking {min(args.per_band, len(low))} from low + "
          f"{min(args.per_band, len(high))} from high = {len(picks)} total")

    # Group picks by parquet file so we can do ONE scan per parquet and
    # find all relevant doc_ids in that pass. Previous per-doc-scan loop
    # was O(P × N_docs) with large openarchives parquets (~42k rows) —
    # this is O(P) where P = number of parquets touched.
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    by_parquet: Dict[Path, List[Dict]] = defaultdict(list)
    for rec in picks:
        parquet_name, _, _ = rec.get("source_path", "").partition("#")
        by_parquet[args.parquet_dir / Path(parquet_name).name].append(rec)

    text_cache: Dict[str, Dict[str, str]] = {}
    for parquet_file, recs in by_parquet.items():
        if not parquet_file.is_file():
            print(f"  [skip] parquet not found: {parquet_file}", file=sys.stderr)
            continue
        doc_ids = [rec["source_path"].partition("#")[2] for rec in recs]
        print(f"  bulk-finding {len(doc_ids)} docs in {parquet_file.name} ...",
              flush=True)
        text_cache[str(parquet_file)] = _bulk_find(
            str(parquet_file), doc_ids, args.doc_id_column, args.text_column,
        )

    with args.output_path.open("w", encoding="utf-8") as out:
        for i, rec in enumerate(picks, 1):
            source_path = rec.get("source_path", "")
            parquet_name, _, doc_id = source_path.partition("#")
            parquet_file = args.parquet_dir / Path(parquet_name).name
            if not parquet_file.is_file():
                continue
            raw_text = text_cache.get(str(parquet_file), {}).get(doc_id)
            if raw_text is None:
                print(f"  [skip] doc_id not in parquet: {doc_id}", file=sys.stderr)
                continue
            cleaned = _clean_with_newlines(raw_text, args.scripts)
            pc = int(rec.get("chars_dropped_by_per_char_filter", 0) or 0)
            inp = int(rec.get("non_empty_chars_in", 0) or 0)
            band = "low" if pc / max(inp, 1) * 100 <= lo_band[1] else "high"
            out.write(json.dumps({
                "source_path": source_path,
                "source_dataset": rec.get("source_dataset"),
                "source_doc_id": rec.get("source_doc_id"),
                "band": band,
                "metrics": {
                    "char_strip_ratio": 100.0 * pc / max(inp, 1),
                    "pct_chars_removed_non_empty": rec.get("pct_chars_removed_non_empty"),
                    "non_empty_chars_in": inp,
                    "non_empty_chars_out": rec.get("non_empty_chars_out"),
                },
                "stats_snapshot": {k: rec.get(k) for k in [
                    "content_chars_kept",
                    "chars_dropped_by_line_drop",
                    "chars_dropped_by_normalization",
                    "chars_dropped_by_per_char_filter",
                    "lines_dropped_by_cleaner",
                    "marker_chars_passthrough",
                    "marker_chars_added",
                    "non_empty_lines_in", "non_empty_lines_out",
                    "pct_chars_removed_non_empty",
                    "pct_lines_removed_non_empty",
                    "counter_font_marker",
                    "counter_glyph_marker",
                    "counter_script_residue",
                ]},
                "cleaned_text": cleaned,
            }, ensure_ascii=False) + "\n")
            if i % 10 == 0:
                print(f"  wrote {i}/{len(picks)}", flush=True)
    print(f"done → {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
