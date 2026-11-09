"""Join per-doc cleaner stats with upstream parquet quality scores,
then show how the 4 quality metrics relate to deletion percentage:

- `greek_badness_score` (upstream)
- `mojibake_badness_score` (upstream)
- `ascii_punct_ratio` (our new charset metric)
- `moji_residue_ratio` (our new charset metric)

X-axes:
- `pct_chars_removed_non_empty` (total deletion, includes normalization)
- `char_strip_ratio` = chars_dropped_by_per_char_filter / non_empty_chars_in
  (cleaning-only deletion, the real content-loss signal)

Buckets of deletion %: [0, 1, 5, 10, 20, 40, 60, 100] → 7 bins.
For each bin: median / p25 / p75 / p95 of each of the 4 metrics.
Also: correlation coefficients (Spearman) between deletion % and each
metric.

Reports both KEPT docs (cleaner output has text) and the FULL SET
(including charset-dropped docs) so we can see the tail.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _iter_stats(stats_glob: str):
    for p in sorted(globmod.glob(stats_glob)):
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def _load_upstream(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            key = (d["source_dataset"], d["source_doc_id"])
            out[key] = d
    return out


def _f(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--upstream-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("loading upstream scores ...")
    upstream = _load_upstream(args.upstream_path)
    print(f"  {len(upstream)} rows")

    # Join + build per-doc records.
    rows: List[Dict[str, Any]] = []
    for s in _iter_stats(args.stats_glob):
        ds = s.get("source_dataset")
        did = str(s.get("source_doc_id") or "")
        up = upstream.get((ds, did), {})
        rows.append({
            "dataset": ds,
            "drop_reason": s.get("drop_reason") or "",
            "kept": not s.get("drop_reason"),
            "pct_removed": _f(s.get("pct_chars_removed_non_empty")),
            "char_strip_ratio":
                100.0 * (int(s.get("chars_dropped_by_per_char_filter", 0) or 0)
                         / max(int(s.get("non_empty_chars_in", 0) or 0), 1))
                if s.get("non_empty_chars_in") else None,
            "greek_badness": _f(up.get("greek_badness_score")),
            "mojibake_badness": _f(up.get("mojibake_badness_score")),
            "punct_ratio": _f(s.get("charset_punct_ratio")),
            "moji_ratio": _f(s.get("charset_moji_ratio")),
            "greek_pct_upstream": _f(up.get("greek_percentage")),
        })
    print(f"  joined: {len(rows)} docs")

    # Bucket ranges on pct_removed (for kept docs) and char_strip_ratio.
    buckets = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 101)]
    metrics = [
        ("greek_badness",   "greek_badness_score (upstream)"),
        ("mojibake_badness","mojibake_badness_score (upstream)"),
        ("punct_ratio",     "ascii_punct_ratio (new, as %)"),
        ("moji_ratio",      "moji_residue_ratio (new, as %)"),
    ]

    def bucket_report(field: str, label: str, filter_fn) -> List[str]:
        lines = [f"### {label} — bucketed by {field}", ""]
        header = "| bucket | N | " + " | ".join(f"{m[1]} median (p25–p75, p95)" for m in metrics) + " |"
        lines.append(header)
        lines.append("|---|---:|" + "---|" * len(metrics))
        for lo, hi in buckets:
            subset = [r for r in rows if filter_fn(r) and r[field] is not None and lo <= r[field] < hi]
            row = f"| {lo}–{hi}% | {len(subset)} |"
            for key, _ in metrics:
                vals = [r[key] for r in subset if r[key] is not None]
                if not vals:
                    row += " — |"
                    continue
                a = np.array(vals)
                scale = 100.0 if key in ("punct_ratio", "moji_ratio") else 1.0
                row += (f" {np.median(a)*scale:.2f} "
                        f"({np.quantile(a, 0.25)*scale:.2f}–{np.quantile(a, 0.75)*scale:.2f}, "
                        f"p95={np.quantile(a, 0.95)*scale:.2f}) |")
            lines.append(row)
        lines.append("")
        # Spearman correlations.
        lines.append(f"**Spearman ρ between {field} and each metric (kept docs only):**")
        lines.append("")
        subset = [r for r in rows if filter_fn(r) and r[field] is not None]
        xs = np.array([r[field] for r in subset])
        for key, name in metrics:
            ys_raw = [r[key] for r in subset]
            valid = [(x, y) for x, y in zip(xs, ys_raw) if y is not None]
            if len(valid) < 30:
                lines.append(f"- {name}: N too small")
                continue
            x_arr = np.array([v[0] for v in valid])
            y_arr = np.array([v[1] for v in valid])
            x_rank = np.argsort(np.argsort(x_arr))
            y_rank = np.argsort(np.argsort(y_arr))
            if np.std(x_rank) == 0 or np.std(y_rank) == 0:
                rho = float('nan')
            else:
                rho = float(np.corrcoef(x_rank, y_rank)[0, 1])
            lines.append(f"- {name}: ρ = {rho:+.3f} (N={len(valid)})")
        lines.append("")
        return lines

    md = ["# Quality metrics vs deletion % — all 168,078 docs", "",
          f"Kept docs: {sum(1 for r in rows if r['kept'])}",
          f"Dropped docs: {sum(1 for r in rows if not r['kept'])}",
          "",
          "The 4 quality signals:",
          "- **greek_badness_score** — pre-existing upstream score, higher = worse Greek",
          "- **mojibake_badness_score** — pre-existing upstream score, higher = more mojibake",
          "- **ascii_punct_ratio** — our new metric, high = font-substitution mojibake",
          "- **moji_residue_ratio** — our new metric, high = Latin-1/IPA/PUA mojibake",
          "",
          "The deletion X-axis:",
          "- **pct_chars_removed_non_empty** — total deletion incl. whitespace normalization",
          "- **char_strip_ratio** — per-char filter only (real content loss)",
          "",
          "## 1) Kept docs — bucketed by pct_chars_removed_non_empty",
          ""]
    md.extend(bucket_report("pct_removed",
        "pct_chars_removed_non_empty (kept docs only)",
        lambda r: r["kept"]))

    md.append("## 2) Kept docs — bucketed by char_strip_ratio")
    md.append("")
    md.extend(bucket_report("char_strip_ratio",
        "char_strip_ratio (kept docs only)",
        lambda r: r["kept"]))

    md.append("## 3) All docs (incl. charset-dropped) — bucketed by char_strip_ratio")
    md.append("")
    md.append("Note: docs dropped by the charset filter have no cleaner output, "
              "so their char_strip_ratio is 0 in the stats — they end up in the "
              "first bucket. Use this view mostly for the metric medians per bucket.")
    md.append("")
    md.extend(bucket_report("char_strip_ratio",
        "char_strip_ratio (all docs)",
        lambda r: True))

    (args.output_dir / "quality_vs_deletions.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"report → {args.output_dir / 'quality_vs_deletions.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
