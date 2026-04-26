"""Row-sharded version of clean_and_stats_full.py.

Instead of 1 worker = 1 parquet (limits parallelism to N parquets), each
worker processes a ROW RANGE of a parquet. With --workers 48 and 250+
HPLT parquets each with 500k rows, we get 48-way parallelism uniformly
without any worker monopolizing one big parquet.

Layout:
  input-glob → list of parquets
  Each parquet is split into `--shards-per-parquet` equal row-chunks
  → N_parquets × shards_per_parquet tasks → mp.Pool(workers=...) starmap.

Per-task output:
  <stats_dir>/<stem>.shard_<i>of<n>.stats.jsonl
  <shards_dir>/<stem>.shard_<i>of<n>.txt.gz

Downstream consumers iterate all *.stats.jsonl / *.txt.gz — the shard
suffix is transparent.

Same matcher + cleaner + threshold logic as clean_and_stats_full.py.
"""
from __future__ import annotations

import argparse
import glob as globmod
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq


DEFAULT_SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)


def _load_thresholds(path: Path) -> Dict[str, Optional[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    suggested = data.get("suggested_thresholds") or {}
    return {
        "font_name_literal": suggested.get("font_marker"),
        "glyph_font_like": suggested.get("glyph_marker"),
        "script_residue_restricted": suggested.get("script_residue"),
    }


def _doc_drop_reason(doc_counters, thresholds):
    """Doc-level drop rules removed 2026-04-25.

    `font_name_literal` and `glyph_font_like` doc-drops were removed
    after the user observed that the line-drop rules in the Rust
    cleaner already cover the same noise patterns at finer
    granularity:

    - `PDF_FONT_SUBSET_REGEX` line-drop uses the IDENTICAL pattern
      `/[A-Z]{6}\\+[A-Z][A-Za-z0-9-]+` as the `font_name_literal`
      counter — line-drop is strictly less destructive.
    - `BAD_LINE_AC` + `GLYPH_FONT_TAG_REGEX` + `FONT_GLYPH_TAG_REGEX`
      line-drops cover the structural-PDF-residue subset of
      `glyph_font_like`. The 50 PostScript glyph names + `/uni`/`/g`
      regex part of `glyph_font_like` is more aggressive but no
      longer drives doc rejection.

    The script_residue_restricted page-level rule is unaffected and
    still applied separately in the main loop.
    """
    return ""


def _page_script_residue_count(page):
    """Pages now carry `per_category_match_count` from Rust — this is a
    cheap dict lookup. Accepts either the full page dict OR the old
    matches_json string for back-compat with callers not yet migrated."""
    if isinstance(page, dict):
        pc = page.get("per_category_match_count") or {}
        return int(pc.get("script_residue_restricted", 0) or 0)
    # Legacy path: matches_json string.
    try:
        matches = json.loads(page or "[]")
    except Exception:
        return 0
    return sum(1 for m in matches
               if "script_residue_restricted" in list(m.get("categories") or []))


_MARKER_LINES = {
    "<!-- line-removed -->",
    "<!-- text-missing -->",
    "<!-- table-removed -->",
}


def _non_empty_stats(text: str):
    """Return (line_count_total, non_empty_line_count, non_empty_char_count).

    A line is "non-empty" if its trimmed form is non-empty AND isn't
    one of our known marker comments. Character count sums only chars
    on non-empty, non-marker lines (newlines excluded).
    """
    total_lines = 0
    non_empty_lines = 0
    non_empty_chars = 0
    for line in text.split("\n"):
        total_lines += 1
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in _MARKER_LINES:
            continue
        non_empty_lines += 1
        non_empty_chars += len(line)
    return total_lines, non_empty_lines, non_empty_chars


def _process_row_shard(
    parquet_path: str,
    start_row: int,
    end_row: int,
    stats_path: str,
    text_gz_path: str,
    category_specs_path: str,
    thresholds: Dict[str, Optional[int]],
    scripts_to_keep: List[str],
    text_column: str,
    doc_id_column: str,
    dataset_column: str,
    batch_size: int,
) -> Dict[str, Any]:
    import glossapi_rs_cleaner as cleaner
    # CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25 Point 7: the matcher's
    # three-counter PyO3 surface (`match_token_category_debug_text`)
    # is gone. The cleaner now emits per-rule match counts directly
    # in `clean_stats` (`rule_a_match_count`, `rule_b_match_count`,
    # `residue_line_drop_count`). No separate matcher invocation.

    scratch_root = Path(stats_path).parent
    scratch = scratch_root / f"_scratch_{Path(stats_path).stem}"
    scratch.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows_seen = 0
    rows_kept = 0
    rows_dropped: Dict[str, int] = {}
    total_chars_before = 0
    total_chars_after = 0
    total_chars_dropped = 0
    phase_a_fallback_count = 0
    phase_a_dialect_ambiguous_count = 0

    global_row_idx = 0
    with open(stats_path, "w", encoding="utf-8") as stats_fh, \
         gzip.open(text_gz_path, "wt", encoding="utf-8") as text_fh:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            for row in rows:
                if global_row_idx < start_row:
                    global_row_idx += 1
                    continue
                if global_row_idx >= end_row:
                    break
                global_row_idx += 1
                rows_seen += 1
                text = row.get(text_column) or ""
                chars_before = len(text)
                if not text.strip():
                    rows_dropped["empty"] = rows_dropped.get("empty", 0) + 1
                    stats_fh.write(json.dumps({
                        "source_path": f"{parquet_path}#{row.get(doc_id_column)}",
                        "source_doc_id": row.get(doc_id_column),
                        "source_dataset": row.get(dataset_column),
                        "chars_before": 0, "chars_after": 0,
                        "chars_removed": 0, "pct_removed": 0.0,
                        "counter_font_marker": 0, "counter_glyph_marker": 0,
                        "counter_script_residue": 0, "drop_reason": "empty",
                    }) + "\n")
                    continue

                source_doc_id = str(row.get(doc_id_column) or f"row-{rows_seen}")
                source_dataset = str(row.get(dataset_column) or Path(parquet_path).stem)
                source_path = f"{parquet_path}#{source_doc_id}"
                source_stem = _safe(f"{source_dataset}__{source_doc_id}")[:200]
                base_stem = _safe(source_dataset)

                # Pre-clean charset analysis. As of 2026-04-25, NO charset
                # ratio drives a doc-level rejection here — the moji /
                # punct / greek-low / counter-{glyph,font_name} rules
                # were all removed after user review of v7 samples
                # showed they were too aggressive. The ratios + counters
                # are still emitted in the per-doc stats jsonl as
                # diagnostic / threshold-study levers, and the line-drop
                # rules in the Rust cleaner already cover the noise
                # patterns these doc-drops were targeting.
                cs = cleaner.analyze_charset(text)

                # Doc-level counters initialised; populated AFTER cleaning
                # from `clean_stats` per-rule fields (Point 7). Pre-cleaning
                # matcher invocation removed.
                doc_counters: Dict[str, int] = {
                    "font_name_literal": 0, "glyph_font_like": 0, "script_residue_restricted": 0,
                }

                # Attach the charset ratios computed above so kept-docs
                # records show them (distribution/threshold calibration).
                charset_greek_ratio = round(cs["greek_letter_ratio"], 4)
                charset_moji_ratio = round(cs["moji_residue_ratio"], 4)
                charset_punct_ratio = round(cs["ascii_punct_ratio"], 4)
                # Combined metric per 2026-04-23: additive, no weighting.
                # Interpretation: fraction of non-whitespace chars that look
                # like mojibake residue by either axis. NOT a rejection
                # signal on its own — user review needed before setting cutoff.
                mojibake_noise_ratio = round(
                    charset_moji_ratio + charset_punct_ratio, 4)

                # Doc-level drop: font + glyph only.
                reason = _doc_drop_reason(doc_counters, thresholds)
                if reason:
                    rows_dropped[reason] = rows_dropped.get(reason, 0) + 1
                    total_chars_dropped += chars_before
                    stats_fh.write(json.dumps({
                        "source_path": source_path, "source_doc_id": source_doc_id,
                        "source_dataset": source_dataset,
                        "chars_before": chars_before, "chars_after": 0,
                        "chars_removed": chars_before, "pct_removed": 100.0,
                        "counter_font_marker": doc_counters["font_name_literal"],
                        "counter_glyph_marker": doc_counters["glyph_font_like"],
                        "counter_script_residue": doc_counters["script_residue_restricted"],
                        "charset_greek_ratio": charset_greek_ratio,
                        "charset_moji_ratio": charset_moji_ratio,
                        "charset_punct_ratio": charset_punct_ratio,
                        "mojibake_noise_ratio": mojibake_noise_ratio,
                        "drop_reason": reason,
                    }) + "\n")
                    continue

                # Page-level script_residue rule REMOVED 2026-04-25.
                # Replaced by the LINE-level R1 ∪ R2 rule in the Rust
                # cleaner (`normalize::is_residue_mojibake_line` →
                # invoked from `cleaning_module::core_clean_text_with_stats`
                # alongside BAD_LINE_AC and has_decoded_glyph_font_artefact).
                # The line-level rule is finer-grained (drops only the
                # offending line, not a whole page) and was empirically
                # validated on the v7 sample
                # `top500_by_counter_script_residue` (2.6 M body lines).
                # Kept the per-doc accounting fields below for backward
                # compatibility with downstream consumers; they always
                # report 0 now.
                pages_dropped_sr = 0
                chars_dropped_sr_pages = 0
                text_for_cleaner = text

                # v6 wave-2 (2026-04-23): enable LaTeX repetition crop.
                # Default thresholds (char=30, line=3) tuned for typical
                # math-OCR `+ + + + + …` runs and `x = x = x =` lines.
                # See latex_module.rs::crop_latex_repetitions docs.
                cleaned, clean_stats = cleaner.clean_text_with_stats(
                    text_for_cleaner, scripts_to_keep,
                    None,           # min_chars_for_comment (default)
                    True,           # enable_latex_repetition_crop
                    30,             # latex_char_threshold
                    3,              # latex_line_threshold
                )
                if clean_stats.get("phase_a_fallback_reason") is not None:
                    phase_a_fallback_count += 1
                    if clean_stats.get("phase_a_dialect_ambiguous_input"):
                        phase_a_dialect_ambiguous_count += 1
                # Point 7 migration: populate doc_counters from cleaner's
                # per-rule match counts (replaces the deleted matcher).
                # Note Rule B is now ONE engine (not split into
                # font_name_literal vs glyph_font_like at cleaner-level).
                # We attribute Rule A to glyph_marker (PostScript glyph-name
                # literals) and Rule B to glyph_marker too (regex covers
                # GLYPH<…>, /uniXXXX, /gN, font subsets, ...). The
                # font_name_literal counter is no longer separately
                # measurable post-unification — kept at 0 for back-compat.
                doc_counters["font_name_literal"] = 0
                doc_counters["glyph_font_like"] = int(
                    clean_stats.get("rule_a_match_count", 0)
                    + clean_stats.get("rule_b_match_count", 0)
                )
                doc_counters["script_residue_restricted"] = int(
                    clean_stats.get("residue_line_drop_count", 0)
                )
                # Four-way char-drop attribution + line-drop count come from
                # the Rust side. Quality stats (non-empty lines/chars in/out)
                # and derived percentages computed here in Python.
                # Rust-side non_empty_line_stats — replaces the Python
                # _non_empty_stats helper. Iterates text once; orders of
                # magnitude faster on large docs.
                lines_in_total, non_empty_lines_in, non_empty_chars_in = (
                    cleaner.non_empty_line_stats(text_for_cleaner)
                )
                lines_out_total, non_empty_lines_out, non_empty_chars_out = (
                    cleaner.non_empty_line_stats(cleaned)
                )
                if not cleaned.strip() or cleaned.strip().startswith("<!-- text-missing"):
                    rows_dropped["cleaner_empty"] = rows_dropped.get("cleaner_empty", 0) + 1
                    total_chars_dropped += chars_before
                    stats_fh.write(json.dumps({
                        "source_path": source_path, "source_doc_id": source_doc_id,
                        "source_dataset": source_dataset,
                        "chars_before": chars_before, "chars_after": 0,
                        "chars_removed": chars_before, "pct_removed": 100.0,
                        "counter_font_marker": doc_counters["font_name_literal"],
                        "counter_glyph_marker": doc_counters["glyph_font_like"],
                        "counter_script_residue": doc_counters["script_residue_restricted"],
                        "charset_greek_ratio": charset_greek_ratio,
                        "charset_moji_ratio": charset_moji_ratio,
                        "charset_punct_ratio": charset_punct_ratio,
                        "mojibake_noise_ratio": mojibake_noise_ratio,
                        "pages_dropped_script_residue": pages_dropped_sr,
                        "chars_dropped_script_residue_pages": chars_dropped_sr_pages,
                        "lines_in_total": lines_in_total,
                        "non_empty_lines_in": non_empty_lines_in,
                        "non_empty_chars_in": non_empty_chars_in,
                        "lines_dropped_by_cleaner": clean_stats.get("lines_dropped_count", 0),
                        "chars_dropped_by_line_drop": clean_stats.get("chars_dropped_by_line_drop", 0),
                        "chars_dropped_by_normalization": clean_stats.get("chars_dropped_by_normalization", 0),
                        "chars_dropped_by_per_char_filter": clean_stats.get("chars_dropped_by_per_char_filter", 0),
                        "content_chars_kept": clean_stats.get("content_chars_kept", 0),
                        "drop_reason": "cleaner_empty",
                    }) + "\n")
                    continue

                # `chars_after` reports cleaned-output length INCLUDING marker
                # chars (legacy compat). The "content_chars_kept" field is the
                # per-user-spec char count that excludes comment markers.
                chars_after = len(cleaned)
                chars_removed = chars_before - chars_after
                pct_removed = (chars_removed / chars_before * 100.0) if chars_before else 0.0
                # Non-empty-based pcts (user spec: percentage against non-empty
                # baseline, so marker lines don't inflate the denominator).
                pct_chars_removed_non_empty = (
                    (1.0 - non_empty_chars_out / non_empty_chars_in) * 100.0
                    if non_empty_chars_in else 0.0
                )
                pct_lines_removed_non_empty = (
                    (1.0 - non_empty_lines_out / non_empty_lines_in) * 100.0
                    if non_empty_lines_in else 0.0
                )
                total_chars_before += chars_before
                total_chars_after += chars_after
                rows_kept += 1
                text_fh.write(cleaned.replace("\r", " ").replace("\n", " ") + "\n")
                stats_fh.write(json.dumps({
                    "source_path": source_path, "source_doc_id": source_doc_id,
                    "source_dataset": source_dataset,
                    "chars_before": chars_before, "chars_after": chars_after,
                    "chars_removed": chars_removed,
                    "pct_removed": round(pct_removed, 3),
                    "counter_font_marker": doc_counters["font_name_literal"],
                    "counter_glyph_marker": doc_counters["glyph_font_like"],
                    "counter_script_residue": doc_counters["script_residue_restricted"],
                    "charset_greek_ratio": charset_greek_ratio,
                    "charset_moji_ratio": charset_moji_ratio,
                    "charset_punct_ratio": charset_punct_ratio,
                    "pages_dropped_script_residue": pages_dropped_sr,
                    "chars_dropped_script_residue_pages": chars_dropped_sr_pages,
                    # Four-way per-doc char attribution (from Rust).
                    "content_chars_kept": clean_stats.get("content_chars_kept", 0),
                    "chars_dropped_by_line_drop": clean_stats.get("chars_dropped_by_line_drop", 0),
                    "chars_dropped_by_normalization": clean_stats.get("chars_dropped_by_normalization", 0),
                    "chars_dropped_by_per_char_filter": clean_stats.get("chars_dropped_by_per_char_filter", 0),
                    "lines_dropped_by_cleaner": clean_stats.get("lines_dropped_count", 0),
                    "marker_chars_passthrough": clean_stats.get("marker_chars_passthrough", 0),
                    "marker_chars_added": clean_stats.get("marker_chars_added", 0),
                    # Quality signals for broken-text flagging.
                    "lines_in_total": lines_in_total,
                    "non_empty_lines_in": non_empty_lines_in,
                    "non_empty_chars_in": non_empty_chars_in,
                    "lines_out_total": lines_out_total,
                    "non_empty_lines_out": non_empty_lines_out,
                    "non_empty_chars_out": non_empty_chars_out,
                    "pct_chars_removed_non_empty": round(pct_chars_removed_non_empty, 3),
                    "pct_lines_removed_non_empty": round(pct_lines_removed_non_empty, 3),
                    "drop_reason": "",
                }) + "\n")
            if global_row_idx >= end_row:
                break

    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    for stale in scratch_root.glob("*.md"):
        try: stale.unlink()
        except Exception: pass

    return {
        "parquet": parquet_path, "start_row": start_row, "end_row": end_row,
        "stats_out": stats_path, "text_gz_out": text_gz_path,
        "rows_seen": rows_seen, "rows_kept": rows_kept,
        "rows_dropped": rows_dropped,
        "chars_before_total_kept_docs": total_chars_before,
        "chars_after_total_kept_docs": total_chars_after,
        "chars_removed_by_per_line_strip": total_chars_before - total_chars_after,
        "chars_removed_by_drop": total_chars_dropped,
        "phase_a_fallback_count": phase_a_fallback_count,
        "phase_a_dialect_ambiguous_count": phase_a_dialect_ambiguous_count,
        "elapsed_seconds": time.time() - start,
    }


def _build_tasks(
    parquets: List[Path],
    shards_per_parquet: int,
    stats_dir: Path,
    text_shards_dir: Path,
    category_specs: str,
    thresholds: Dict[str, Optional[int]],
    scripts_to_keep: List[str],
    text_column: str,
    doc_id_column: str,
    dataset_column: str,
    batch_size: int,
) -> List[tuple]:
    tasks = []
    for p in parquets:
        n_rows = pq.ParquetFile(p).metadata.num_rows
        if n_rows == 0:
            continue
        n_shards = max(1, min(shards_per_parquet, n_rows))
        chunk = (n_rows + n_shards - 1) // n_shards
        for i in range(n_shards):
            start = i * chunk
            end = min(start + chunk, n_rows)
            if start >= end:
                continue
            suffix = f"shard_{i:03d}of{n_shards:03d}"
            stats_path = stats_dir / f"{p.stem}.{suffix}.stats.jsonl"
            text_gz_path = text_shards_dir / f"{p.stem}.{suffix}.txt.gz"
            tasks.append((
                str(p), start, end, str(stats_path), str(text_gz_path),
                category_specs, thresholds, scripts_to_keep,
                text_column, doc_id_column, dataset_column, batch_size,
            ))
    return tasks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--stats-dir", required=True, type=Path)
    parser.add_argument("--text-shards-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--doc-id-column", default="source_doc_id")
    parser.add_argument("--dataset-column", default="source_dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument(
        "--shards-per-parquet", type=int, default=8,
        help="Row-range shards per parquet — total tasks = N_parquets × this",
    )
    args = parser.parse_args(argv)

    paths: List[Path] = []
    for pat in args.input_glob:
        paths.extend(Path(p).resolve() for p in globmod.glob(pat))
    paths = sorted(dict.fromkeys(paths))

    thresholds = _load_thresholds(args.thresholds)
    print(f"{len(paths)} parquets, {args.shards_per_parquet} shards/parquet → "
          f"~{len(paths) * args.shards_per_parquet} tasks × {args.workers} workers")
    print(f"thresholds: {thresholds}")

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    args.text_shards_dir.mkdir(parents=True, exist_ok=True)

    tasks = _build_tasks(
        paths, args.shards_per_parquet,
        args.stats_dir, args.text_shards_dir,
        str(args.category_specs.resolve()), thresholds, args.scripts_to_keep,
        args.text_column, args.doc_id_column, args.dataset_column,
        args.batch_size,
    )
    print(f"{len(tasks)} row-shards to process")
    start = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(_process_row_shard, tasks)
    elapsed = time.time() - start

    total_seen = sum(r["rows_seen"] for r in results)
    total_kept = sum(r["rows_kept"] for r in results)
    total_drop_counts: Dict[str, int] = {}
    total_chars_before = 0
    total_chars_after = 0
    total_chars_removed_strip = 0
    total_chars_removed_drop = 0
    total_phase_a_fallback = 0
    total_phase_a_dialect_ambiguous = 0
    for r in results:
        total_chars_before += r["chars_before_total_kept_docs"]
        total_chars_after += r["chars_after_total_kept_docs"]
        total_chars_removed_strip += r["chars_removed_by_per_line_strip"]
        total_chars_removed_drop += r["chars_removed_by_drop"]
        total_phase_a_fallback += r.get("phase_a_fallback_count", 0)
        total_phase_a_dialect_ambiguous += r.get("phase_a_dialect_ambiguous_count", 0)
        for k, v in r["rows_dropped"].items():
            total_drop_counts[k] = total_drop_counts.get(k, 0) + v

    summary = {
        "elapsed_seconds": elapsed,
        "rows_seen": total_seen,
        "rows_kept": total_kept,
        "rows_dropped": total_drop_counts,
        "chars_before_total_kept_docs": total_chars_before,
        "chars_after_total_kept_docs": total_chars_after,
        "chars_removed_by_per_line_strip": total_chars_removed_strip,
        "chars_removed_by_drop": total_chars_removed_drop,
        "pct_chars_removed_by_strip": round(
            100.0 * total_chars_removed_strip / max(total_chars_before, 1), 3),
        "phase_a_fallback_count": total_phase_a_fallback,
        "phase_a_dialect_ambiguous_count": total_phase_a_dialect_ambiguous,
    }
    (args.stats_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
