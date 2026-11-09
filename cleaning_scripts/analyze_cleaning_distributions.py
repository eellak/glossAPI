"""Distribution analysis over the per-doc stats.jsonl from
clean_and_stats_rowsharded.py.

Purpose: BEFORE committing to any sampling strategy for the quality
review, look at the shape of the population — length distributions,
pct-change distributions, per-dataset variance, correlations between
drop buckets. Numbers + plots.

Output in --output-dir:
  distributions.md         — narrative summary with stats
  01_lengths.png           — non_empty_chars_in / _out histograms
  02_pct_changes.png       — pct_chars_removed / pct_lines_removed
  03_drop_attribution.png  — four-way drop bucket comparison
  04_per_dataset.png       — per-dataset pct_removed boxplots
  05_joint_density.png     — pct_chars vs pct_lines 2D density
  06_cdf.png               — cumulative doc count vs pct_removed
  summary.json             — structured stats for downstream scripts
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_stats(stats_glob: str) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(globmod.glob(stats_glob)):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                out.append(d)
    return out


def _kept_only(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Docs that survived cleaning (drop_reason is empty)."""
    return [d for d in stats if not d.get("drop_reason")]


def _altered_kept(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Kept docs that experienced non-zero cleaning damage."""
    out = []
    for d in _kept_only(stats):
        if float(d.get("pct_chars_removed_non_empty", 0) or 0) > 0:
            out.append(d)
            continue
        if int(d.get("lines_dropped_by_cleaner", 0) or 0) > 0:
            out.append(d)
    return out


def _quantile_summary(values: List[float], label: str) -> Dict[str, float]:
    if not values:
        return {"label": label, "n": 0}
    a = np.asarray(values, dtype=float)
    return {
        "label": label,
        "n": int(a.size),
        "min": float(a.min()),
        "p05": float(np.quantile(a, 0.05)),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def _plot_lengths(kept, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lens_in = [int(d.get("non_empty_chars_in", 0) or 0) for d in kept]
    lens_out = [int(d.get("non_empty_chars_out", 0) or 0) for d in kept]
    for ax, lens, title in ((axes[0], lens_in, "non_empty_chars_in"),
                             (axes[1], lens_out, "non_empty_chars_out")):
        # Log scale — these are highly skewed (Greek PDFs are huge, Wikipedia is small).
        data = [x for x in lens if x > 0]
        ax.hist(data, bins=np.logspace(np.log10(max(min(data), 1)),
                                       np.log10(max(data)), 60),
                color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title} (log-log, N={len(data)})")
        ax.set_xlabel("chars")
        ax.set_ylabel("doc count")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "01_lengths.png", dpi=120)
    plt.close(fig)


def _plot_pct_changes(altered, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, field, title in ((axes[0], "pct_chars_removed_non_empty", "% chars removed"),
                              (axes[1], "pct_lines_removed_non_empty", "% lines removed")):
        vals = [float(d.get(field, 0) or 0) for d in altered]
        # Clip to [0, 100] to handle any slight overflow from saturating math.
        vals = [max(0.0, min(100.0, v)) for v in vals]
        ax.hist(vals, bins=np.linspace(0, 100, 51), color="coral",
                edgecolor="black", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title} (altered docs only, N={len(vals)}, log-y)")
        ax.set_xlabel("%")
        ax.set_ylabel("doc count")
        ax.axvline(np.median(vals), color="red", linestyle="--", alpha=0.7,
                   label=f"median={np.median(vals):.2f}%")
        ax.axvline(np.mean(vals), color="blue", linestyle=":", alpha=0.7,
                   label=f"mean={np.mean(vals):.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "02_pct_changes.png", dpi=120)
    plt.close(fig)


def _plot_drop_attribution(altered, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = [
        ("chars_dropped_by_line_drop", "line-drop"),
        ("chars_dropped_by_normalization", "normalization"),
        ("chars_dropped_by_per_char_filter", "per-char filter"),
    ]
    for ax, (field, title) in zip(axes, labels):
        # Per-doc ratio of this bucket to the doc's non_empty_chars_in.
        ratios = []
        for d in altered:
            denom = max(int(d.get("non_empty_chars_in", 0) or 0), 1)
            ratios.append(100.0 * float(d.get(field, 0) or 0) / denom)
        ratios = [max(0.0, min(100.0, v)) for v in ratios]
        ax.hist(ratios, bins=np.linspace(0, 100, 51), color="seagreen",
                edgecolor="black", alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title} (% of non_empty_chars_in, N={len(ratios)}, log-y)")
        ax.set_xlabel("%")
        ax.set_ylabel("doc count")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "03_drop_attribution.png", dpi=120)
    plt.close(fig)


def _plot_per_dataset(altered, output_dir: Path):
    by_ds = defaultdict(list)
    for d in altered:
        by_ds[str(d.get("source_dataset", "unknown"))].append(d)
    datasets = sorted(by_ds.keys(),
                      key=lambda k: -np.median([float(x.get("pct_chars_removed_non_empty", 0) or 0)
                                                 for x in by_ds[k]]))
    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(datasets) * 0.9), 10))
    for ax, field, title in ((axes[0], "pct_chars_removed_non_empty", "% chars removed per dataset"),
                              (axes[1], "pct_lines_removed_non_empty", "% lines removed per dataset")):
        box_data = [[float(d.get(field, 0) or 0) for d in by_ds[ds]] for ds in datasets]
        bp = ax.boxplot(box_data, tick_labels=[ds[:30] for ds in datasets],
                        showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_title(f"{title} — altered docs only, flyers hidden")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        # Add n per dataset as annotation.
        for i, ds in enumerate(datasets, 1):
            ax.text(i, 95, f"n={len(by_ds[ds])}", ha="center", fontsize=7,
                    rotation=0, color="darkred")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "04_per_dataset.png", dpi=120)
    plt.close(fig)


def _plot_joint_density(altered, output_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    x = np.asarray([float(d.get("pct_chars_removed_non_empty", 0) or 0) for d in altered])
    y = np.asarray([float(d.get("pct_lines_removed_non_empty", 0) or 0) for d in altered])
    x = np.clip(x, 0, 100); y = np.clip(y, 0, 100)
    h = ax.hexbin(x, y, gridsize=50, cmap="viridis", bins="log",
                  extent=(0, 100, 0, 100))
    ax.set_xlabel("% chars removed")
    ax.set_ylabel("% lines removed")
    ax.set_title(f"Joint density (log count), N={len(x)}")
    plt.colorbar(h, ax=ax, label="log doc count")
    ax.plot([0, 100], [0, 100], "r--", alpha=0.4, label="y=x")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "05_joint_density.png", dpi=120)
    plt.close(fig)


def _plot_cdf(altered, output_dir: Path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for field, title, color in [
        ("pct_chars_removed_non_empty", "% chars removed", "coral"),
        ("pct_lines_removed_non_empty", "% lines removed", "steelblue"),
    ]:
        vals = sorted(float(d.get(field, 0) or 0) for d in altered)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=title, color=color, linewidth=1.5)
    ax.set_xlabel("% removed")
    ax.set_ylabel("cumulative fraction of altered docs")
    ax.set_title(f"CDF of cleaning damage (altered docs only, N={len(altered)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_cdf.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading stats from {args.stats_glob} ...")
    stats = _load_stats(args.stats_glob)
    kept = _kept_only(stats)
    altered = _altered_kept(stats)
    print(f"  total rows: {len(stats)}")
    print(f"  kept (survived cleaning): {len(kept)}")
    print(f"  altered (kept + non-zero damage): {len(altered)}")

    # Drop reasons.
    drop_reasons = defaultdict(int)
    for d in stats:
        r = d.get("drop_reason") or ""
        if r:
            drop_reasons[r] += 1

    summary = {
        "n_total": len(stats),
        "n_kept": len(kept),
        "n_altered": len(altered),
        "n_untouched": len(kept) - len(altered),
        "drop_reasons": dict(drop_reasons),
        "length_in": _quantile_summary(
            [int(d.get("non_empty_chars_in", 0) or 0) for d in kept],
            "non_empty_chars_in"),
        "length_out": _quantile_summary(
            [int(d.get("non_empty_chars_out", 0) or 0) for d in kept],
            "non_empty_chars_out"),
        "pct_chars_removed": _quantile_summary(
            [float(d.get("pct_chars_removed_non_empty", 0) or 0) for d in altered],
            "pct_chars_removed_non_empty (altered only)"),
        "pct_lines_removed": _quantile_summary(
            [float(d.get("pct_lines_removed_non_empty", 0) or 0) for d in altered],
            "pct_lines_removed_non_empty (altered only)"),
    }

    # Per-dataset summary.
    by_ds = defaultdict(list)
    for d in kept:
        by_ds[str(d.get("source_dataset", "unknown"))].append(d)
    per_ds = {}
    for ds, docs in by_ds.items():
        altered_ds = [d for d in docs if float(d.get("pct_chars_removed_non_empty", 0) or 0) > 0
                      or int(d.get("lines_dropped_by_cleaner", 0) or 0) > 0]
        per_ds[ds] = {
            "n_kept": len(docs),
            "n_altered": len(altered_ds),
            "pct_altered": round(100.0 * len(altered_ds) / max(len(docs), 1), 2),
            "pct_chars_removed_median": round(float(np.median(
                [float(d.get("pct_chars_removed_non_empty", 0) or 0) for d in altered_ds]
                or [0])), 3),
            "pct_lines_removed_median": round(float(np.median(
                [float(d.get("pct_lines_removed_non_empty", 0) or 0) for d in altered_ds]
                or [0])), 3),
        }
    summary["per_dataset"] = per_ds

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plots.
    print("plotting...")
    _plot_lengths(kept, args.output_dir)
    _plot_pct_changes(altered, args.output_dir)
    _plot_drop_attribution(altered, args.output_dir)
    _plot_per_dataset(altered, args.output_dir)
    _plot_joint_density(altered, args.output_dir)
    _plot_cdf(altered, args.output_dir)

    # Markdown narrative.
    md = ["# Cleaning distribution analysis", "",
          f"Source: `{args.stats_glob}`", "",
          "## Top-level counts", "",
          f"- Total rows: **{summary['n_total']}**",
          f"- Kept (survived cleaning): **{summary['n_kept']}**",
          f"- Altered (kept + non-zero damage): **{summary['n_altered']}**",
          f"- Untouched (kept + zero damage): **{summary['n_untouched']}**",
          "",
          "### Drop reasons",
          "",
          "| reason | count |",
          "|---|---:|"]
    for r, n in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        md.append(f"| {r} | {n} |")
    md.extend(["", "## Length distributions", ""])
    for key in ("length_in", "length_out"):
        qs = summary[key]
        md.append(f"### {qs['label']} (N={qs['n']})")
        md.append(f"- min/p25/p50/p75/p95/p99/max: "
                  f"{qs['min']:.0f} / {qs['p25']:.0f} / {qs['p50']:.0f} / "
                  f"{qs['p75']:.0f} / {qs['p95']:.0f} / {qs['p99']:.0f} / {qs['max']:.0f}")
        md.append(f"- mean: {qs['mean']:.0f}")
        md.append("")
    md.extend(["## Pct changes (altered docs only)", ""])
    for key in ("pct_chars_removed", "pct_lines_removed"):
        qs = summary[key]
        md.append(f"### {qs['label']} (N={qs['n']})")
        md.append(f"- p05/p25/p50/p75/p90/p95/p99/max: "
                  f"{qs['p05']:.2f}% / {qs['p25']:.2f}% / {qs['p50']:.2f}% / "
                  f"{qs['p75']:.2f}% / {qs['p90']:.2f}% / {qs['p95']:.2f}% / "
                  f"{qs['p99']:.2f}% / {qs['max']:.2f}%")
        md.append(f"- mean: {qs['mean']:.2f}%")
        md.append("")
    md.extend(["## Per-dataset summary", "",
               "| dataset | n_kept | n_altered | % altered | median %chars | median %lines |",
               "|---|---:|---:|---:|---:|---:|"])
    for ds, s in sorted(per_ds.items(), key=lambda kv: -kv[1]["pct_altered"]):
        md.append(f"| {ds} | {s['n_kept']} | {s['n_altered']} | {s['pct_altered']}% "
                  f"| {s['pct_chars_removed_median']}% | {s['pct_lines_removed_median']}% |")
    md.extend(["", "## Plots", "",
               "- `01_lengths.png` — doc length histograms (log-log)",
               "- `02_pct_changes.png` — % chars/lines removed histograms",
               "- `03_drop_attribution.png` — four-way drop bucket comparison",
               "- `04_per_dataset.png` — per-dataset boxplots",
               "- `05_joint_density.png` — pct_chars vs pct_lines 2D hexbin",
               "- `06_cdf.png` — cumulative distribution of cleaning damage",
               ""])
    (args.output_dir / "distributions.md").write_text("\n".join(md), encoding="utf-8")
    print(f"summary → {args.output_dir / 'summary.json'}")
    print(f"report  → {args.output_dir / 'distributions.md'}")
    print(f"plots   → {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
