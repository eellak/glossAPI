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


def _iter_stats(glob_pat: str, ds_prefix_filter: set = None):
    """Iterate stats records. If ds_prefix_filter is set, skip any stats
    file whose basename doesn't begin with one of those dataset prefixes
    — rowsharded stats are named `<dataset_stem>.shard_NofM.stats.jsonl`,
    so prefix matching is sufficient to skip irrelevant shards entirely
    without parsing any of their lines."""
    for p in sorted(globmod.glob(glob_pat)):
        if ds_prefix_filter:
            base = Path(p).name
            if not any(base.startswith(pref) for pref in ds_prefix_filter):
                continue
        with open(p) as fh:
            for line in fh:
                yield json.loads(line)


def _load_upstream(path: Path, ds_filter: set = None) -> Dict[tuple, Dict]:
    """Load upstream score lookup. If ds_filter is set, only retain rows
    whose source_dataset is in the set — saves ~75% memory + parse time
    when sampling a single dataset.

    Uses a substring pre-filter (literal `"source_dataset": "<name>"`) to
    skip irrelevant lines without calling json.loads on them."""
    out = {}
    needles = None
    if ds_filter:
        needles = [f'"source_dataset": "{name}"' for name in ds_filter]
    with path.open() as fh:
        for line in fh:
            if needles is not None and not any(n in line for n in needles):
                continue
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
    p.add_argument("--dataset-filter", default="",
                   help="Comma-separated dataset names. If set, restricts "
                        "BOTH bands to docs whose source_dataset is in this "
                        "list (overrides --pdf-sources-only). Use to focus "
                        "the sample on one dataset, e.g. for per-dataset "
                        "elbow review.")
    p.add_argument("--deletion-split-pct", type=float, default=20.0,
                   help="Cleaning-only deletion %% threshold that splits "
                        "lt vs ge subfolders. Default 20.0; pass e.g. 1.567 "
                        "for the openarchives.gr knee from the CDF analysis.")
    p.add_argument("--lt-subdir", default="",
                   help="Override the lt-subfolder name (default: "
                        "lt_{split_pct}pct).")
    p.add_argument("--ge-subdir", default="",
                   help="Override the ge-subfolder name (default: "
                        "ge_{split_pct}pct).")
    p.add_argument("--seed", type=int, default=20260423)
    p.add_argument("--max-text-chars", type=int, default=0,
                   help="Max chars to render in the body. 0 (default) = "
                        "no truncation; write the full doc. The whole "
                        "point of these samples is inspection — only set "
                        ">0 if you really want head+tail stubs.")
    p.add_argument("--top-by", default="",
                   help="Top-N selection mode. If set to a stats-record "
                        "field name (e.g. 'counter_script_residue', "
                        "'charset_punct_ratio', 'charset_moji_ratio'), "
                        "ignores deletion bucketing and picks the "
                        "--top-n records with the highest values of "
                        "that field. All picks land in a single subdir "
                        "named by the field (or override with "
                        "--top-subdir). --dataset-filter still applies. "
                        "drop_reason rows are still skipped.")
    p.add_argument("--top-n", type=int, default=500,
                   help="N for --top-by mode.")
    p.add_argument("--top-subdir", default="",
                   help="Override --top-by output subfolder name "
                        "(default: 'top{N}_by_{field}').")
    p.add_argument("--show-pre-cleaner", action="store_true",
                   help="Render the RAW parquet text in the body instead "
                        "of running the cleaner first. Off by default: "
                        "samples are evaluative (post-cleaner = what's "
                        "in the training corpus). Use this flag when "
                        "diagnosing which INPUT caused a specific "
                        "cleaner behavior.")
    p.add_argument("--scripts-to-keep", nargs="*",
                   default=["greek", "latin", "french", "spanish",
                            "punctuation", "numbers", "common_symbols"],
                   help="Script set passed to the cleaner when rendering "
                        "post-cleaner bodies. Match the v5 run's set.")
    p.add_argument("--upstream-greek-badness-max", type=float, default=None,
                   help="Filter out docs with upstream greek_badness_score "
                        "above this threshold (e.g. 60). Use to restrict "
                        "review to docs we'd otherwise KEEP — high upstream "
                        "badness docs are already in the rejection cone.")
    p.add_argument("--upstream-mojibake-badness-max", type=float, default=None,
                   help="Filter out docs with upstream mojibake_badness_score "
                        "above this threshold (e.g. 0.1). Same rationale "
                        "as --upstream-greek-badness-max.")
    args = p.parse_args()
    DS_FILTER = set(s.strip() for s in args.dataset_filter.split(",") if s.strip())
    SPLIT_PCT = float(args.deletion_split_pct)

    # Per-bucket logic always uses the canonical 0/5/10/15/SPLIT vs
    # SPLIT/40/60/100 layout, with the LOW/HI boundary at SPLIT_PCT.
    buckets = [
        ("low", 0.0,           SPLIT_PCT * 0.25),
        ("low", SPLIT_PCT*0.25, SPLIT_PCT * 0.50),
        ("low", SPLIT_PCT*0.50, SPLIT_PCT * 0.75),
        ("low", SPLIT_PCT*0.75, SPLIT_PCT),
        ("hi",  SPLIT_PCT,     max(SPLIT_PCT*2, 40.0)),
        ("hi",  max(SPLIT_PCT*2, 40.0), 60.0),
        ("hi",  60.0, 101.0),
    ]

    # Use DS_FILTER throughout — saves both upstream-load time
    # (~75% on single-dataset runs) and stats-iter time (~80% on
    # openarchives, which has 5/24 parquets).
    print(f"loading upstream ... (filter={sorted(DS_FILTER) if DS_FILTER else 'all'})")
    upstream = _load_upstream(args.upstream_path,
                              ds_filter=DS_FILTER if DS_FILTER else None)
    print(f"  loaded {len(upstream):,} upstream rows")

    print(f"bucketing stats ... split at {SPLIT_PCT}% cleaning-only")
    by_bucket: Dict[tuple, List[Dict]] = defaultdict(list)
    # In --top-by mode, collect all surviving (non-drop) records into a
    # flat pool; sort + slice happens after the iter loop.
    top_pool: List[Dict] = []
    n_filtered_upstream = 0
    for d in _iter_stats(args.stats_glob,
                         ds_prefix_filter=DS_FILTER if DS_FILTER else None):
        if d.get("drop_reason"):
            continue
        # Apply --dataset-filter early so the pool itself is restricted
        # (used for per-dataset elbow review, e.g. just openarchives).
        if DS_FILTER and d.get("source_dataset") not in DS_FILTER:
            continue
        # Upstream-score gate (Cases 12 / 13 review request, 2026-04-23):
        # restrict the pool to docs that would PASS standard rejection.
        # High upstream badness docs are already excluded by other rules
        # — including them in a top-by audit dilutes the signal.
        ds_key = (d.get("source_dataset"), d.get("source_doc_id"))
        if args.upstream_greek_badness_max is not None:
            up = upstream.get(ds_key, {})
            score = up.get("greek_badness_score")
            if score is not None and float(score) > args.upstream_greek_badness_max:
                n_filtered_upstream += 1
                continue
        if args.upstream_mojibake_badness_max is not None:
            up = upstream.get(ds_key, {})
            score = up.get("mojibake_badness_score")
            if score is not None and float(score) > args.upstream_mojibake_badness_max:
                n_filtered_upstream += 1
                continue
        # Use CLEANING-ONLY deletion pct (line_drop + per_char_filter),
        # excluding normalization effects. Normalization is mostly
        # whitespace-bucket / dot-leader / separator collapse — format
        # compression, not content loss. For the sample split at SPLIT_PCT
        # we want actual content removal.
        inp = int(d.get("non_empty_chars_in", 0) or 0)
        line_drop = int(d.get("chars_dropped_by_line_drop", 0) or 0)
        per_char = int(d.get("chars_dropped_by_per_char_filter", 0) or 0)
        pct = 100.0 * (line_drop + per_char) / max(inp, 1)
        d["__cleaning_only_deletion_pct"] = pct
        if args.top_by:
            top_pool.append(d)
            continue
        # Two-band assignment (low <SPLIT, hi >=SPLIT).
        band = "low" if pct < SPLIT_PCT else "hi"
        # Find sub-bucket; fall back to the first matching band's first bucket.
        placed = False
        for (b, lo, hi) in buckets:
            if b == band and lo <= pct < hi:
                by_bucket[(b, lo, hi)].append(d)
                placed = True
                break
        if not placed and band == "hi":
            # tail beyond all hi buckets — stuff into the last hi bucket
            for (b, lo, hi) in reversed(buckets):
                if b == "hi":
                    by_bucket[(b, lo, hi)].append(d); break

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
    if args.top_by:
        # Sort the flat pool by --top-by field descending; pick first N.
        # Tie-breaker: source_doc_id for determinism. Records missing
        # the field sort as 0.
        def _key(r):
            v = r.get(args.top_by, 0)
            try:
                return float(v) if v is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        top_pool.sort(key=lambda r: (_key(r), str(r.get("source_doc_id"))),
                      reverse=True)
        picks = top_pool[: args.top_n]
        print(f"  top-by={args.top_by!r}: pool={len(top_pool):,} → "
              f"picking top {len(picks)}")
        if n_filtered_upstream:
            print(f"  upstream-badness filter dropped: {n_filtered_upstream:,}")
        if picks:
            print(f"  top value: {_key(picks[0]):.4f}  "
                  f"bottom-of-top value: {_key(picks[-1]):.4f}")
    elif args.n_low or args.n_high:
        # Uniform-random across each side, ignoring sub-buckets.
        low_pool, high_pool = [], []
        for (band, lo, hi), items in by_bucket.items():
            if band == "low":
                low_pool.extend(items)
            else:
                high_pool.extend(items)
        if args.pdf_sources_only and not DS_FILTER:
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
    if args.top_by:
        # Single subfolder for top-N mode; name encodes field + N.
        top_name = args.top_subdir or f"top{args.top_n}_by_{args.top_by}"
        top_dir = args.output_dir / top_name
        top_dir.mkdir(parents=True, exist_ok=True)
        # Reuse low_dir / high_dir as aliases so the per-rec routing
        # below ends up in top_dir regardless of which side `pct < SPLIT`
        # falls on (top-N mode doesn't care about the split).
        low_dir = high_dir = top_dir
    else:
        # Subfolder names follow the actual split %; explicit overrides win.
        split_label = (f"{SPLIT_PCT:.3f}".rstrip("0").rstrip(".")).replace(".", "p")
        lt_name = args.lt_subdir or f"lt_{split_label}pct"
        ge_name = args.ge_subdir or f"ge_{split_label}pct"
        low_dir = args.output_dir / lt_name
        high_dir = args.output_dir / ge_name
        low_dir.mkdir(parents=True, exist_ok=True)
        high_dir.mkdir(parents=True, exist_ok=True)

    # Default: render POST-cleaner body (see feedback memory
    # `feedback_review_samples_post_cleaner_default.md`). Only skip
    # the cleaner pass if the user asked for raw input via
    # --show-pre-cleaner.
    cleaner = None
    if not args.show_pre_cleaner:
        try:
            import glossapi_rs_cleaner as cleaner  # type: ignore
        except ImportError as err:
            raise SystemExit(
                "glossapi_rs_cleaner wheel not installed; install it "
                "(maturin develop --release) or pass --show-pre-cleaner "
                f"to render raw parquet bodies. error: {err}"
            )

    n_written = 0
    # Per-pick row counter feeds into filename to guarantee uniqueness
    # (Case: top-by mode collisions when multiple picks share metric +
    # pct + did_22char triple — we silently overwrote files before).
    for pick_idx, rec in enumerate(picks):
        pname, _, did = rec["source_path"].partition("#")
        text = texts.get(did)
        if not text:
            continue
        # Body to render in the .md file. Post-cleaner by default;
        # matches the v6 run parameters (LaTeX repetition cropping ON)
        # so samples reflect what the training corpus actually contains.
        if cleaner is not None:
            try:
                body_text, _ = cleaner.clean_text_with_stats(
                    text, args.scripts_to_keep,
                    None,   # min_chars_for_comment
                    True,   # enable_latex_repetition_crop
                    30,     # latex_char_threshold
                    3,      # latex_line_threshold
                )
            except Exception as err:
                body_text = f"[cleaner raised: {err}]\n\n{text}"
        else:
            body_text = text
        # Use the cleaning-only deletion pct (excludes normalization)
        # for the SPLIT_PCT split, matching the bucket classification above.
        pct = float(rec.get("__cleaning_only_deletion_pct", 0))
        pct_total = float(rec.get("pct_chars_removed_non_empty", 0))
        ds = rec.get("source_dataset", "unk")
        up = upstream.get((ds, did), {})
        target_dir = low_dir if pct < SPLIT_PCT else high_dir
        moji = float(rec.get("charset_moji_ratio") or 0)
        punct = float(rec.get("charset_punct_ratio") or 0)
        mojibake_noise = moji + punct
        if args.top_by:
            # Prefix = ACTUAL metric value, zero-padded (NOT inverted —
            # so readers can eyeball the metric from the filename).
            # `ls` sorts ascending-by-value; use `ls -r` or `ls | sort
            # -rn` to view highest-first.
            try:
                v = float(rec.get(args.top_by) or 0)
            except (TypeError, ValueError):
                v = 0.0
            # Ratio fields → ×10000 → 5 digits (range 0..9999 → ratio
            # 0.0000..0.9999). Raw counters → as integer, 6 digits
            # (handles up to ~999,999 counter value).
            if "ratio" in args.top_by:
                prefix = f"{int(round(v * 10000)):05d}"
            else:
                prefix = f"{int(round(v)):06d}"
        else:
            # Filename prefix = mojibake_noise_ratio (moji + punct, ×10000
            # zero-padded) so ls within each subfolder orders by the
            # combined mojibake signal ascending. Deletion % is split via
            # subfolders.
            prefix = f"{int(round(mojibake_noise * 10000)):05d}"
        # Append the pick index (3-digit) to guarantee uniqueness — the
        # `_safe(did, 22)` truncation collided in top-by mode when many
        # picks shared the same metric value + pct rounding bucket.
        fname = f"{prefix}_{pick_idx:03d}_pct{int(round(pct*10)):04d}_{_safe(ds, 16)}_{_safe(did, 22)}.md"
        lines = [
            f"# {ds} / {did}",
            "",
            f"## Deletion metrics",
            "",
            f"- **cleaning_only_deletion_pct** (line_drop + per_char_filter, "
            f"excludes normalization): **{pct:.3f}%**",
            f"- pct_chars_removed_non_empty (total incl normalization): {pct_total}%",
            f"- non_empty_chars_in: {rec.get('non_empty_chars_in')}",
            f"- non_empty_chars_out: {rec.get('non_empty_chars_out')}",
            f"- content_chars_kept: {rec.get('content_chars_kept')}",
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
            f"## Doc text — rendered as markdown "
            f"({'POST-cleaner output' if not args.show_pre_cleaner else 'PRE-cleaner input'})",
            "",
            f"**Input size: {len(text):,} chars; body size: {len(body_text):,} chars.** "
            f"Body shown in full (no truncation).",
            "",
            f"<!-- body below this HR is the "
            f"{'cleaner output' if not args.show_pre_cleaner else 'raw parquet text'} "
            f"— unfenced so tables / headings render -->",
            "",
            "---",
            "",
        ]
        # Full doc, no truncation. --max-text-chars defaults to 0.
        # Only truncate if user explicitly opts in with a positive cap.
        if args.max_text_chars > 0 and len(body_text) > args.max_text_chars:
            half = args.max_text_chars // 2
            lines.append(body_text[:half])
            lines.append(f"\n*[...display truncated by --max-text-chars {args.max_text_chars}; body was {len(body_text):,} chars...]*\n")
            lines.append(body_text[-half:])
        else:
            lines.append(body_text)
        (target_dir / fname).write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
    print(f"wrote {n_written} files to {args.output_dir}")


if __name__ == "__main__":
    main()
