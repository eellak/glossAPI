"""Pull top-N PDF docs by all three noise counters in ONE pass.

Optimization over calling `pull_deletion_band_samples.py` three
times (once per counter):
- Upstream jsonl loaded ONCE, with PDF-source prefilter (substring
  check before json.loads — skips ~70% of rows on a PDF-only run).
- Stats jsonl scanned ONCE, with parquet-name prefilter.
- Three top-K heaps maintained in parallel.
- Bulk parquet fetch deduplicated across all three picks.

Output layout (matches v7):
  <output-dir>/top500_by_counter_font_marker/
  <output-dir>/top500_by_counter_glyph_marker/
  <output-dir>/top500_by_counter_script_residue/

Each .md file has the v7 header (deletion metrics, charset ratios,
three-counter scores, upstream scores, body).

Usage:
  python3 pull_top_three_counters_pdf.py \
    --stats-glob '...stats/*.stats.jsonl' \
    --upstream-path .../upstream_scores.jsonl \
    --parquet-dir .../hf_release_publish_working/data \
    --output-dir .../pdf_three_counters \
    --top-n 500 \
    --upstream-greek-badness-max 60 --upstream-mojibake-badness-max 0.1
"""
from __future__ import annotations

import argparse
import glob as globmod
import heapq
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq

PDF_SOURCES = {
    "openarchives.gr", "greek_phd", "Apothetirio_Pergamos",
    "Apothetirio_Kallipos", "eurlex-greek-legislation",
    "ellinika_dedomena_europaikou_koinovouliou",
    "opengov.gr-diaboyleuseis",
}

COUNTERS = ("counter_font_marker", "counter_glyph_marker",
            "counter_script_residue")


def load_upstream(path: Path) -> Dict[Tuple[str, str], Dict]:
    needles = [f'"source_dataset": "{n}"' for n in PDF_SOURCES]
    out: Dict[Tuple[str, str], Dict] = {}
    n_total = n_kept = 0
    with path.open() as fh:
        for line in fh:
            n_total += 1
            if not any(n in line for n in needles):
                continue
            d = json.loads(line)
            out[(d["source_dataset"], d["source_doc_id"])] = d
            n_kept += 1
    print(f"  upstream: {n_total:,} → {n_kept:,} (PDF only)", flush=True)
    return out


def stats_iter_pdf(stats_glob: str):
    pdf_prefixes = tuple(PDF_SOURCES)
    for p in sorted(globmod.glob(stats_glob)):
        base = Path(p).name
        if not base.startswith(pdf_prefixes):
            continue
        with open(p) as fh:
            for line in fh:
                yield json.loads(line)


def _safe(s: str, n: int = 32) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))[:n].strip("_") or "x"


def _bulk_find(parquet_path: str, doc_ids: List[str]) -> Dict[str, str]:
    want = set(map(str, doc_ids))
    out: Dict[str, str] = {}
    pf = pq.ParquetFile(parquet_path)
    for b in pf.iter_batches(batch_size=5000,
                              columns=["source_doc_id", "text"]):
        for sid, txt in zip(b.column(0).to_pylist(),
                            b.column(1).to_pylist()):
            if str(sid) in want:
                out[str(sid)] = txt or ""
                want.discard(str(sid))
                if not want:
                    return out
    return out


def render_doc_md(rec: Dict, up: Dict, text: str, counter_field: str) -> str:
    pct = float(rec.get("pct_chars_removed_non_empty", 0) or 0)
    chars_in = int(rec.get("non_empty_chars_in", 0) or 0)
    line_drop = int(rec.get("chars_dropped_by_line_drop", 0) or 0)
    per_char = int(rec.get("chars_dropped_by_per_char_filter", 0) or 0)
    cleaning_pct = 100.0 * (line_drop + per_char) / max(chars_in, 1)
    moji = float(rec.get("charset_moji_ratio") or 0)
    punct = float(rec.get("charset_punct_ratio") or 0)
    head = [
        f"# {rec.get('source_dataset')} / {rec.get('source_doc_id')}",
        "",
        f"## Sample selection",
        "",
        f"- selected by: **{counter_field}**",
        f"- value: **{rec.get(counter_field)}**",
        "",
        "## Deletion metrics",
        "",
        f"- **cleaning_only_deletion_pct** (line_drop + per_char_filter, excludes normalization): **{cleaning_pct:.3f}%**",
        f"- pct_chars_removed_non_empty (total incl normalization): {pct}%",
        f"- non_empty_chars_in: {rec.get('non_empty_chars_in')}",
        f"- non_empty_chars_out: {rec.get('non_empty_chars_out')}",
        f"- content_chars_kept: {rec.get('content_chars_kept')}",
        "",
        "## Three-counter matcher scores (doc-level totals)",
        "",
        f"- counter_font_marker (font_name_literal): {rec.get('counter_font_marker')}",
        f"- counter_glyph_marker (glyph_font_like): {rec.get('counter_glyph_marker')}",
        f"- counter_script_residue (script_residue_restricted): {rec.get('counter_script_residue')}",
        f"- pages_dropped_script_residue (page-level rule): {rec.get('pages_dropped_script_residue', 0)}",
        f"- chars_dropped_script_residue_pages: {rec.get('chars_dropped_script_residue_pages', 0)}",
        "",
        "## Charset ratios",
        "",
        f"- charset_greek_ratio: {rec.get('charset_greek_ratio')}",
        f"- charset_moji_ratio: {moji}",
        f"- charset_punct_ratio: {punct}",
        f"- mojibake_noise_ratio (moji + punct): {round(moji + punct, 4)}",
        "",
        "## Upstream pre-existing scores",
        "",
        f"- greek_badness_score: {up.get('greek_badness_score')}",
        f"- mojibake_badness_score: {up.get('mojibake_badness_score')}",
        f"- greek_percentage: {up.get('greek_percentage')}",
        f"- latin_percentage: {up.get('latin_percentage')}",
        "",
        "## Doc text — POST-cleaner output",
        "",
        f"**Input size: {len(text):,} chars.** Body shown in full (no truncation).",
        "",
        "<!-- body below this HR is the cleaner output (POST) — unfenced so MD renders -->",
        "",
        "---",
        "",
    ]
    return "\n".join(head) + text + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-glob", required=True)
    ap.add_argument("--upstream-path", required=True, type=Path)
    ap.add_argument("--parquet-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--upstream-greek-badness-max", type=float, default=60.0)
    ap.add_argument("--upstream-mojibake-badness-max", type=float, default=0.1)
    args = ap.parse_args()

    t0 = time.time()
    print("loading upstream (PDF-only) ...", flush=True)
    upstream = load_upstream(args.upstream_path)
    print(f"  done in {time.time()-t0:.1f}s; {len(upstream):,} rows", flush=True)

    # Three top-K min-heaps — pop smallest when full and new is bigger.
    heaps: Dict[str, List] = {c: [] for c in COUNTERS}
    n_seen = n_pdf = n_passed_upstream = 0
    t1 = time.time()
    print("scanning stats (PDF-only) ...", flush=True)
    seen_keys = set()
    for d in stats_iter_pdf(args.stats_glob):
        n_seen += 1
        # Note: do NOT skip drop_reason rows — we WANT high-counter
        # docs that hit the rejection rule. Their stats jsonl row
        # already carries the counter values.
        if d.get("source_dataset") not in PDF_SOURCES:
            continue
        # Dedup: rowsharded shards can emit duplicate stats lines for
        # the same (dataset, doc_id) when a doc straddles shard
        # boundaries (rare). Keep the first occurrence.
        key = (d.get("source_dataset"), d.get("source_doc_id"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        n_pdf += 1
        ds_key = (d.get("source_dataset"), d.get("source_doc_id"))
        up = upstream.get(ds_key, {})
        gb = up.get("greek_badness_score")
        mb = up.get("mojibake_badness_score")
        if gb is not None and float(gb) > args.upstream_greek_badness_max:
            continue
        if mb is not None and float(mb) > args.upstream_mojibake_badness_max:
            continue
        n_passed_upstream += 1
        for c in COUNTERS:
            v = int(d.get(c, 0) or 0)
            if v <= 0:
                continue
            h = heaps[c]
            key = (v, str(d.get("source_doc_id")))
            if len(h) < args.top_n:
                heapq.heappush(h, (key, d))
            elif key > h[0][0]:
                heapq.heapreplace(h, (key, d))
    print(f"  scanned {n_seen:,} rows; PDF {n_pdf:,}; upstream-pass {n_passed_upstream:,}", flush=True)
    print(f"  scan in {time.time()-t1:.1f}s", flush=True)
    for c in COUNTERS:
        if heaps[c]:
            top_key = max(h[0] for h in [heaps[c]])
            n = len(heaps[c])
            print(f"  {c}: {n} picks; top value = {sorted(heaps[c], reverse=True)[0][0][0]}", flush=True)

    # Dedup + bulk-fetch text per parquet.
    all_picks: Dict[Tuple[str, str], Dict] = {}
    pick_counters: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for c in COUNTERS:
        for _, rec in heaps[c]:
            key = (rec["source_dataset"], rec["source_doc_id"])
            all_picks[key] = rec
            pick_counters[key].append(c)

    by_parquet: Dict[str, List[str]] = defaultdict(list)
    for (ds, did), _ in all_picks.items():
        # source_path field has full path; fall back to dataset-name.parquet
        src = all_picks[(ds, did)].get("source_path", "")
        pname, _, _ = src.partition("#")
        if pname:
            by_parquet[Path(pname).name].append(did)
        else:
            by_parquet[f"{ds}.parquet"].append(did)
    print(f"fetching text from {len(by_parquet)} parquets ...", flush=True)
    texts: Dict[Tuple[str, str], str] = {}
    for pname, dids in by_parquet.items():
        pfile = args.parquet_dir / pname
        if not pfile.is_file():
            print(f"  [skip] {pfile} missing", file=sys.stderr)
            continue
        found = _bulk_find(str(pfile), dids)
        for (ds_, did_), _ in all_picks.items():
            if did_ in found:
                texts[(ds_, did_)] = found[did_]
        print(f"  {pname}: {len(dids)} requested, got {len(found)}", flush=True)

    # Write samples per counter folder.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for c in COUNTERS:
        sub = args.output_dir / f"top500_by_{c}"
        sub.mkdir(parents=True, exist_ok=True)
        ranked = sorted(heaps[c], reverse=True)  # by metric desc
        n_max = max((k[0] for k, _ in ranked), default=0)
        # Zero-padded width based on n_max digits
        width = max(6, len(str(n_max)))
        n_written = 0
        for rank, ((value, _doc_id), rec) in enumerate(ranked):
            ds = rec["source_dataset"]; did = rec["source_doc_id"]
            text = texts.get((ds, did), "")
            if not text:
                continue
            up = upstream.get((ds, did), {})
            prefix = f"{value:0{width}d}_{rank:03d}"
            slug = f"{_safe(ds, 24)}_{_safe(did, 24)}"
            fname = f"{prefix}_{slug}.md"
            (sub / fname).write_text(render_doc_md(rec, up, text, c),
                                     encoding="utf-8")
            n_written += 1
        print(f"  wrote {n_written} files to {sub}", flush=True)

    print(f"total elapsed: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
