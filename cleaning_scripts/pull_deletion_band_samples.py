"""Pull a stratified sample of KEPT docs around the 20% deletion boundary
for manual review. Output: .md files named by zero-padded deletion%
so `ls` orders them ascending.

Buckets:
- low band  (pct_chars_removed_non_empty < 20%): 0-5%, 5-10%, 10-15%, 15-20%
- high band (pct_chars_removed_non_empty >= 20%): 20-40%, 40-60%, 60-100%

Each .md shows: all upstream scores, all our new charset ratios, and
a ~8k-char sample of the cleaned text (head + tail for long docs).
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq


def _safe(s: str, n: int = 24) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))[:n].strip("_") or "x"


def _iter_stats(glob_pat: str):
    for p in sorted(globmod.glob(glob_pat)):
        with open(p) as fh:
            for line in fh:
                yield json.loads(line)


def _load_upstream(path: Path) -> Dict[tuple, Dict]:
    out = {}
    with path.open() as fh:
        for line in fh:
            d = json.loads(line)
            out[(d["source_dataset"], d["source_doc_id"])] = d
    return out


def _bulk_find(parquet_path: str, doc_ids: List[str]) -> Dict[str, str]:
    want = set(str(d) for d in doc_ids)
    out = {}
    try:
        t = pq.read_table(parquet_path, columns=["source_doc_id", "text"],
                          filters=[("source_doc_id", "in", list(want))])
        if t.num_rows:
            for sid, txt in zip(t.column(0).to_pylist(), t.column(1).to_pylist()):
                out[str(sid)] = txt or ""
    except Exception:
        pass
    if set(want) - set(out):
        pf = pq.ParquetFile(parquet_path)
        missing = set(want) - set(out)
        for b in pf.iter_batches(batch_size=5000, columns=["source_doc_id", "text"]):
            for sid, txt in zip(b.column(0).to_pylist(), b.column(1).to_pylist()):
                if str(sid) in missing:
                    out[str(sid)] = txt or ""
                    missing.discard(str(sid))
                    if not missing: return out
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats-glob", required=True)
    p.add_argument("--upstream-path", required=True, type=Path)
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--per-bucket", type=int, default=8)
    p.add_argument("--n-low", type=int, default=0,
                   help="If >0, uniform-random N from <20% overriding "
                        "per-bucket logic on the low side.")
    p.add_argument("--n-high", type=int, default=0,
                   help="If >0, uniform-random N from >=20%.")
    p.add_argument("--pdf-sources-only", action="store_true",
                   help="Restrict LOW-band docs to PDF-extracted sources "
                        "(openarchives / greek_phd / Apothetirio_* / eurlex "
                        "/ ellinika_dedomena_eu / opengov). Non-PDF sources "
                        "don't exhibit the Docling mojibake we're hunting "
                        "and don't need review at low deletion levels. HIGH "
                        "band is unfiltered (all sources).")
    p.add_argument("--seed", type=int, default=20260423)
    p.add_argument("--max-text-chars", type=int, default=8000)
    args = p.parse_args()

    buckets = [
        ("low",  0.0,  5.0),
        ("low",  5.0, 10.0),
        ("low", 10.0, 15.0),
        ("low", 15.0, 20.0),
        ("hi",  20.0, 40.0),
        ("hi",  40.0, 60.0),
        ("hi",  60.0, 101.0),
    ]

    print(f"loading upstream ...")
    upstream = _load_upstream(args.upstream_path)

    print(f"bucketing stats ...")
    by_bucket: Dict[tuple, List[Dict]] = defaultdict(list)
    for d in _iter_stats(args.stats_glob):
        if d.get("drop_reason"):
            continue
        pct = d.get("pct_chars_removed_non_empty")
        if pct is None:
            continue
        for (band, lo, hi) in buckets:
            if lo <= pct < hi:
                by_bucket[(band, lo, hi)].append(d)
                break

    rng = random.Random(args.seed)
    picks: List[Dict] = []
    # PDF-extracted sources where Docling artifacts appear (font-subst,
    # LaTeX-escape, |-table noise). Clean digital sources are skipped
    # from the LOW band review — they don't exhibit these classes.
    PDF_SOURCES = {
        "openarchives.gr", "greek_phd", "Apothetirio_Pergamos",
        "Apothetirio_Kallipos", "eurlex-greek-legislation",
        "ellinika_dedomena_europaikou_koinovouliou",
        "opengov.gr-diaboyleuseis",
    }
    if args.n_low or args.n_high:
        # Uniform-random across each side, ignoring sub-buckets.
        low_pool, high_pool = [], []
        for (band, lo, hi), items in by_bucket.items():
            if band == "low":
                low_pool.extend(items)
            else:
                high_pool.extend(items)
        if args.pdf_sources_only:
            before = len(low_pool)
            low_pool = [r for r in low_pool
                        if r.get("source_dataset") in PDF_SOURCES]
            print(f"  low-pool PDF-source filter: {before} → {len(low_pool)}")
        rng.shuffle(low_pool); rng.shuffle(high_pool)
        if args.n_low:
            picks.extend(low_pool[: args.n_low])
        if args.n_high:
            picks.extend(high_pool[: args.n_high])
        print(f"  low-pool={len(low_pool)} → sampling {min(args.n_low, len(low_pool))}")
        print(f"  high-pool={len(high_pool)} → sampling {min(args.n_high, len(high_pool))}")
    else:
        for k, v in by_bucket.items():
            rng.shuffle(v)
        for (band, lo, hi), items in by_bucket.items():
            picks.extend(items[: args.per_bucket])

    # Group picks by parquet for bulk extraction.
    by_parquet = defaultdict(list)
    for rec in picks:
        pname, _, did = rec["source_path"].partition("#")
        by_parquet[args.parquet_dir / Path(pname).name].append(did)

    # Can't bulk-extract on laptop (parquet lives on instance). Fallback:
    # instance has already computed cleaned shards — but text isn't indexed
    # by doc_id. Easiest: run this script on the instance against the
    # parquets there. Here we assume the parquet_dir is reachable.
    print(f"extracting text from {len(by_parquet)} parquets ...")
    texts: Dict[str, str] = {}
    for pfile, doc_ids in by_parquet.items():
        if not pfile.is_file():
            print(f"  [skip] {pfile} missing")
            continue
        print(f"  {pfile.name}: {len(doc_ids)} docs")
        texts.update(_bulk_find(str(pfile), doc_ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    low_dir = args.output_dir / "lt_20pct"
    high_dir = args.output_dir / "ge_20pct"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for rec in picks:
        pname, _, did = rec["source_path"].partition("#")
        text = texts.get(did)
        if not text:
            continue
        pct = float(rec.get("pct_chars_removed_non_empty", 0))
        ds = rec.get("source_dataset", "unk")
        up = upstream.get((ds, did), {})
        target_dir = low_dir if pct < 20.0 else high_dir
        # Filename prefix = mojibake_noise_ratio (moji + punct, ×10000
        # zero-padded) so ls within each subfolder orders by the combined
        # mojibake signal ascending. Deletion % is split via subfolders.
        moji = float(rec.get("charset_moji_ratio") or 0)
        punct = float(rec.get("charset_punct_ratio") or 0)
        mojibake_noise = moji + punct
        # 4-digit prefix, 1/10000th precision (0.0000 - 2.0000 range)
        prefix = f"{int(round(mojibake_noise * 10000)):05d}"
        fname = f"{prefix}_pct{int(round(pct*10)):04d}_{_safe(ds, 16)}_{_safe(did, 22)}.md"
        lines = [
            f"# {ds} / {did}",
            "",
            f"## Deletion metrics",
            "",
            f"- **pct_chars_removed_non_empty**: {pct}%",
            f"- **non_empty_chars_in**: {rec.get('non_empty_chars_in')}",
            f"- **non_empty_chars_out**: {rec.get('non_empty_chars_out')}",
            f"- **content_chars_kept**: {rec.get('content_chars_kept')}",
            "",
            f"## Four-way drop attribution",
            "",
            f"- chars_dropped_by_line_drop: {rec.get('chars_dropped_by_line_drop')}",
            f"- chars_dropped_by_normalization: {rec.get('chars_dropped_by_normalization')}",
            f"- chars_dropped_by_per_char_filter: {rec.get('chars_dropped_by_per_char_filter')}",
            f"- lines_dropped_by_cleaner: {rec.get('lines_dropped_by_cleaner')}",
            "",
            f"## Charset ratios (new)",
            "",
            f"- charset_greek_ratio: {rec.get('charset_greek_ratio')}",
            f"- charset_moji_ratio: {rec.get('charset_moji_ratio')}",
            f"- charset_punct_ratio: {rec.get('charset_punct_ratio')}",
            f"- **mojibake_noise_ratio (moji + punct)**: "
            f"{rec.get('mojibake_noise_ratio', round((float(rec.get('charset_moji_ratio') or 0) + float(rec.get('charset_punct_ratio') or 0)), 4))}",
            "",
            f"## Upstream pre-existing scores (preserve, do not overwrite)",
            "",
            f"- greek_badness_score: {up.get('greek_badness_score')}",
            f"- mojibake_badness_score: {up.get('mojibake_badness_score')}",
            f"- greek_percentage: {up.get('greek_percentage')}",
            f"- latin_percentage: {up.get('latin_percentage')}",
            "",
            f"## Text sample (rendered as markdown)",
            "",
            f"<!-- the metadata above is cleaner stats; everything below "
            f"this HR is the cleaned doc text, shown unfenced so tables / "
            f"headings render in preview -->",
            "",
            "---",
            "",
        ]
        if args.max_text_chars > 0 and len(text) > args.max_text_chars:
            half = args.max_text_chars // 2
            lines.append(text[:half])
            lines.append(f"\n*[...truncated {len(text) - args.max_text_chars} chars...]*\n")
            lines.append(text[-half:])
        else:
            lines.append(text)
        (target_dir / fname).write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
    print(f"wrote {n_written} files to {args.output_dir}")


if __name__ == "__main__":
    main()
