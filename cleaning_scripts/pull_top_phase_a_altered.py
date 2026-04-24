"""Pull the docs most heavily altered by Phase A (MD-syntax) transforms.

Input: the per-doc jsonl produced by `compute_phase_a_stats_per_doc.py`.

Picks the union of top-N from four metric lenses:
  - reflow_joins              (absolute — big or heavily soft-wrapped docs)
  - hr_chars_saved            (absolute — long separator runs collapsed)
  - gfm_chars_saved           (absolute — long table separator rows collapsed)
  - joins_per_1k_chars        (density — aggressive reflow fraction)

Dedups across lenses → ~100 unique docs. Each pick produces TWO
files in the output dir:

  - {base}_BEFORE.md : raw parquet text (no cleaner run)
  - {base}_AFTER.md  : Phase-A output (apply_phase_a)

Both files carry the same metadata header so either file is
self-contained for review; body below the `---` HR is the raw/
transformed text rendered as markdown (no fencing), so preview
renders whatever MD structure the text actually has.

Filename base format:
  {rank:03d}_R{reflow:07d}_H{hr:06d}_G{gfm:07d}_pct{chars_saved_pct×10:04d}_{dataset}_{did}

Usage:
  python3 pull_top_phase_a_altered.py \
      --stats-jsonl /home/foivos/data/phase_a_audit/phase_a_stats.jsonl \
      --parquet-dir /home/foivos/data/glossapi_work/unified_corpus/data \
      --output-dir /home/foivos/data/phase_a_audit/top100_review \
      --per-lens 25

Restrict to PDF-extracted sources (where formatting matters most):
      --pdf-sources-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq

import glossapi_rs_cleaner as c


LENSES = [
    # (jsonl field name, short label for logging)
    ("reflow_joins",       "reflow"),
    ("hr_chars_saved",     "hr"),
    ("gfm_chars_saved",    "gfm"),
    ("joins_per_1k_chars", "reflow_density"),
]

# Sources where the corpus-MD was Docling-extracted from PDFs. These
# are the datasets where the cleaner's Phase A transforms are most
# likely to be surfacing real alterations (PDF column-wrap → soft-
# wrapped paragraphs, separator-run artifacts, table-like regions).
# Digital-born sources (wikisource, openbook_gr, etc.) rarely have
# these shapes. Kept in sync with `pull_deletion_band_samples.py`.
PDF_SOURCES = {
    "openarchives.gr",
    "greek_phd",
    "Apothetirio_Pergamos",
    "Apothetirio_Kallipos",
    "eurlex-greek-legislation",
    "ellinika_dedomena_europaikou_koinovouliou",
    "opengov.gr-diaboyleuseis",
}


def _safe(s: str, n: int = 24) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))[:n].strip("_") or "x"


def _fetch_one(parquet_path: str, doc_id: str) -> str:
    """Fetch the text for exactly one doc_id from one parquet. Used
    in a one-at-a-time loop so we never hold more than one doc's
    text in memory — important because the corpus has 10M+ char
    docs that OOM when bundled.
    """
    want = str(doc_id)
    try:
        t = pq.read_table(
            parquet_path,
            columns=["source_doc_id", "text"],
            filters=[("source_doc_id", "=", want)],
        )
        if t.num_rows:
            txt = t.column(1).to_pylist()[0]
            return txt or ""
    except Exception:
        pass
    # Fallback: scan batches (pyarrow filter-pushdown can fail on
    # some parquet encodings).
    pf = pq.ParquetFile(parquet_path)
    for b in pf.iter_batches(batch_size=5000, columns=["source_doc_id", "text"]):
        sids = b.column(0).to_pylist()
        for i, sid in enumerate(sids):
            if str(sid) == want:
                return b.column(1).to_pylist()[i] or ""
    return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stats-jsonl", required=True, type=Path)
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--per-lens", type=int, default=25,
                   help="Top-N docs pulled from each metric lens before "
                        "dedup. With 4 lenses and per-lens=25 → ~100 "
                        "unique picks.")
    p.add_argument("--pdf-sources-only", action="store_true",
                   help="Restrict the pool to PDF-extracted datasets "
                        "(openarchives / greek_phd / Apothetirio_* / "
                        "eurlex / ellinika_dedomena_europaikou / "
                        "opengov). Non-PDF sources rarely exhibit "
                        "the extractor-noise shapes the cleaner's "
                        "Phase A addresses, so including them in a "
                        "top-K audit dilutes signal.")
    args = p.parse_args()

    # Streaming top-K-heap per lens — avoids materializing 160k rows
    # just to take 25 from each end (the OOM we hit when combined
    # with a 1.9M-char doc in memory).
    import heapq
    # Each heap stores (metric_value, tie_break_str, row_dict). Using a
    # min-heap of size `per_lens` → pop smallest when full and new is
    # larger. Result: the lens's top-N by value.
    heaps: Dict[str, List] = {k: [] for k, _ in LENSES}
    # Track corpus-wide max per metric for combined-score normalization.
    maxes: Dict[str, float] = {k: 0.0 for k, _ in LENSES}
    n_rows = 0
    n_filtered = 0
    with args.stats_jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            n_rows += 1
            if args.pdf_sources_only and r.get("source_dataset") not in PDF_SOURCES:
                n_filtered += 1
                continue
            for metric, _ in LENSES:
                v = r.get(metric, 0) or 0
                if v > maxes[metric]:
                    maxes[metric] = float(v)
                h = heaps[metric]
                key = (v, r["source_doc_id"])
                if len(h) < args.per_lens:
                    heapq.heappush(h, (key, r))
                elif key > h[0][0]:
                    heapq.heapreplace(h, (key, r))
    if args.pdf_sources_only:
        print(f"  --pdf-sources-only: dropped {n_filtered:,} non-PDF rows "
              f"(kept {n_rows - n_filtered:,} / {n_rows:,})")
    print(f"  streamed {n_rows:,} rows")
    print("per-lens max:")
    for k, label in LENSES:
        print(f"  {label:>14} ({k}): {maxes[k]}")
    # Zero-guard the maxes so the combined score doesn't div-zero.
    for k in maxes:
        if maxes[k] == 0:
            maxes[k] = 1.0

    # Dedup across lenses by (dataset, doc_id).
    picks_by_key: Dict[tuple, Dict] = {}
    for metric, label in LENSES:
        h_rows = sorted(heaps[metric], reverse=True)
        n_before = len(picks_by_key)
        for _, r in h_rows:
            key = (r["source_dataset"], r["source_doc_id"])
            picks_by_key[key] = r
        print(f"  lens {label!r}: top-{args.per_lens} → "
              f"+{len(picks_by_key) - n_before} unique "
              f"(top value = {h_rows[0][0][0] if h_rows else 0})")

    picks = list(picks_by_key.values())

    # Combined rank: weighted sum of normalized metrics. Weights are
    # equal across the three absolute metrics; density is not in the
    # combined score (it's a ratio and already indirectly captured by
    # reflow_joins + input_chars). Used only for ordering the output
    # filenames.
    def combined_score(r: Dict) -> float:
        return (
            (r.get("reflow_joins", 0) or 0) / maxes["reflow_joins"]
            + (r.get("hr_chars_saved", 0) or 0) / maxes["hr_chars_saved"]
            + (r.get("gfm_chars_saved", 0) or 0) / maxes["gfm_chars_saved"]
        )
    picks.sort(key=lambda r: (combined_score(r), r["source_doc_id"]),
               reverse=True)
    print(f"total unique picks: {len(picks)}")

    # Write the review MDs — one doc at a time so we never hold
    # more than one parquet-row's text in RAM. The corpus has docs
    # up to ~11M chars, which multiplied by 90 picks was OOM.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for rank, r in enumerate(picks):
        ds = r["source_dataset"]
        did = r["source_doc_id"]
        pfile = args.parquet_dir / r["source_path"]
        if not pfile.is_file():
            print(f"  [skip] {pfile} missing", file=sys.stderr)
            continue
        text = _fetch_one(str(pfile), did)
        if not text:
            continue
        # Phase-A-only output (no per-char filter, no line-drop). This
        # is what the review is actually about: the shape change the
        # MD-syntax normalizer imposes. `apply_phase_a` is a thin
        # wrapper around `normalize_md_syntax` — fast and memory-safe
        # even on 1.9M-char docs.
        try:
            post_text = c.apply_phase_a(text)
        except Exception as err:
            post_text = f"[apply_phase_a raised: {err}]\n\n{text}"

        reflow = int(r.get("reflow_joins", 0) or 0)
        hr = int(r.get("hr_chars_saved", 0) or 0)
        gfm = int(r.get("gfm_chars_saved", 0) or 0)
        pct = float(r.get("chars_saved_pct", 0) or 0)
        inp = int(r.get("input_chars", 0) or 0)
        out = int(r.get("output_chars", 0) or 0)

        # Widths chosen so corpus-max values (reflow 425k, hr 630k,
        # gfm 1.9M on 2026-04-24 run) fit without truncation.
        base = (
            f"{rank:03d}"
            f"_R{reflow:07d}"
            f"_H{hr:06d}"
            f"_G{gfm:07d}"
            f"_pct{int(round(pct*10)):04d}"
            f"_{_safe(ds, 16)}_{_safe(did, 22)}"
        )
        # Shared metadata header — each file is self-contained so the
        # reviewer can diff BEFORE and AFTER in sibling tabs / panes
        # without having to hunt for the stats block.
        header = [
            f"# {ds} / {did}",
            "",
            f"## Phase A alterations (per-transform)",
            "",
            f"- **reflow_joins**: {reflow:,} "
            f"(density {r.get('joins_per_1k_chars', 0):.2f} / 1k chars)",
            f"- **hr_lines_normalized**: "
            f"{r.get('hr_lines_normalized', 0)} "
            f"→ **hr_chars_saved**: {hr:,}",
            f"- **gfm_rows_normalized**: "
            f"{r.get('gfm_rows_normalized', 0)} "
            f"→ **gfm_chars_saved**: {gfm:,}",
            f"- **total_chars_saved**: "
            f"{r.get('total_chars_saved', 0):,} "
            f"(**{pct:.3f}%** of input_chars={inp:,})",
            f"- output_chars: {out:,}",
            "",
            f"## Size",
            "",
            f"- raw input: **{len(text):,}** chars",
            f"- Phase-A output (apply_phase_a): **{len(post_text):,}** chars",
            "",
        ]
        before_lines = header + [
            "## Raw input (pre-Phase-A) — BEFORE",
            "",
            "<!-- raw parquet text rendered as markdown (no fencing), "
            "so the preview shows whatever shape the source actually has -->",
            "",
            "---",
            "",
            text,
        ]
        after_lines = header + [
            "## Post-Phase-A output — AFTER",
            "",
            "<!-- output of apply_phase_a — only MD-syntax transforms "
            "(GFM sep min, HR min, reflow). No per-char filter, no "
            "line-drop. -->",
            "",
            "---",
            "",
            post_text,
        ]
        (args.output_dir / f"{base}_BEFORE.md").write_text(
            "\n".join(before_lines), encoding="utf-8")
        (args.output_dir / f"{base}_AFTER.md").write_text(
            "\n".join(after_lines), encoding="utf-8")
        n_written += 2
    print(f"wrote {n_written} files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
