#!/usr/bin/env python3
"""Corpus-sampling MD-equivalence verifier.

Samples N random documents from the canonical source parquets, runs the
full cleaner (Phase A + Phase B), then runs both verifiers:

- **STRICT** (`verify_md_preview_equivalent_py`): preview-render
  equivalence. Phase A transforms must preserve this. Expected to
  FAIL on most docs after full cleaner because Phase B deliberately
  deletes content — this is diagnostic, not a pass/fail gate.

- **STRUCTURAL** (`verify_md_structural_py`): output tokens are a
  monotone subsequence of input tokens, block structure preserved,
  table cells preserved. Phase B output MUST pass this. Failures
  point to real bugs (word fusion, dropped blocks, reordered content,
  added content).

Output:
- Per-doc: pass/fail per verification mode + `token_retention_pct`.
- Corpus-wide summary: % passing structural, mean retention,
  distribution over failure modes.
- Failure sample dir: up to `--save-failures N` structural-failing
  docs are written out for human inspection.

Usage:
  verify_md_equivalence.py \
      --parquet-dir /home/foivos/data/glossapi_work/hf_release_publish_working/data \
      --dataset-filter openarchives.gr \
      --n 100 \
      --out /home/foivos/runs/md_verify_v6_snapshot/

Pairs with the in-tree Rust unit tests: unit tests guarantee each
PHASE A transform preserves strict equivalence on synthetic fixtures;
this script quantifies Phase B real-world behavior on actual corpus.
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq

try:
    import glossapi_rs_cleaner as cleaner
except ImportError as err:
    raise SystemExit(
        "glossapi_rs_cleaner wheel not installed. Build with "
        "`maturin develop --release` first. error: " + str(err)
    )


DEFAULT_SCRIPTS = [
    "greek", "latin", "french", "spanish",
    "punctuation", "numbers", "common_symbols",
]


def _sample_docs(
    parquet_dir: Path,
    dataset_filter: Optional[List[str]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Pick `n` random `{source_dataset, source_doc_id, text}` rows."""
    files: List[Path] = []
    for pat in ("*.parquet",):
        files.extend(Path(p) for p in globmod.glob(str(parquet_dir / pat)))
    if dataset_filter:
        files = [
            f for f in files
            if any(prefix in f.name for prefix in dataset_filter)
        ]
    if not files:
        raise SystemExit(f"no parquet files in {parquet_dir}")
    files.sort()
    rng = random.Random(seed)
    # Read metadata to know row counts; sample (file_idx, row_idx) pairs.
    counts = []
    for f in files:
        pf = pq.ParquetFile(f)
        counts.append((f, pf.metadata.num_rows))
    total = sum(c for _, c in counts)
    if total == 0:
        raise SystemExit("0 rows across selected parquets")
    picks: List[int] = sorted(rng.sample(range(total), k=min(n, total)))
    # Map global row index to (file, in-file row index).
    samples: List[Dict[str, Any]] = []
    cursor = 0
    wanted = iter(picks)
    next_pick = next(wanted, None)
    for f, row_count in counts:
        if next_pick is None:
            break
        file_picks = []
        while next_pick is not None and next_pick < cursor + row_count:
            file_picks.append(next_pick - cursor)
            next_pick = next(wanted, None)
        if file_picks:
            pf = pq.ParquetFile(f)
            need = set(file_picks)
            idx = 0
            for batch in pf.iter_batches(
                batch_size=max(1, min(5000, row_count)),
                columns=["source_dataset", "source_doc_id", "text"],
            ):
                rows = batch.to_pylist()
                for r in rows:
                    if idx in need:
                        samples.append({
                            "source_dataset": r.get("source_dataset"),
                            "source_doc_id": r.get("source_doc_id"),
                            "text": r.get("text") or "",
                        })
                    idx += 1
                    if len(samples) >= n:
                        break
                if len(samples) >= n:
                    break
        cursor += row_count
    return samples


def _verify_one(
    input_md: str,
    output_md: str,
) -> Dict[str, Any]:
    """Run both verifier modes on one doc."""
    strict = cleaner.verify_md_preview_equivalent_py(input_md, output_md)
    structural = cleaner.verify_md_structural_py(input_md, output_md)
    return {
        "strict": strict,
        "structural": structural,
    }


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--dataset-filter", default="",
                   help="Comma-separated dataset name prefixes to include.")
    p.add_argument("--n", type=int, default=100,
                   help="Sample size.")
    p.add_argument("--seed", type=int, default=20260424)
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for report + failure samples.")
    p.add_argument("--save-failures", type=int, default=20,
                   help="Max structural-failing docs to write out for "
                        "inspection (per dataset).")
    p.add_argument("--scripts-to-keep", nargs="*", default=DEFAULT_SCRIPTS)
    args = p.parse_args()
    ds_filter = [s.strip() for s in args.dataset_filter.split(",") if s.strip()]

    args.out.mkdir(parents=True, exist_ok=True)
    failures_dir = args.out / "structural_failures"
    failures_dir.mkdir(exist_ok=True)

    print(f"sampling {args.n} docs from {args.parquet_dir}"
          + (f" (filter={ds_filter})" if ds_filter else ""))
    samples = _sample_docs(args.parquet_dir, ds_filter or None, args.n, args.seed)
    print(f"  got {len(samples)} docs")

    results: List[Dict[str, Any]] = []
    structural_failures: List[Dict[str, Any]] = []
    per_dataset_failure_counts: Counter = Counter()

    for i, s in enumerate(samples):
        text = s["text"]
        if not text.strip():
            continue
        try:
            cleaned, _ = cleaner.clean_text_with_stats(
                text, args.scripts_to_keep, None, False, 30, 3,
            )
        except Exception as err:
            results.append({
                "source_dataset": s["source_dataset"],
                "source_doc_id": s["source_doc_id"],
                "error": f"cleaner raised: {err}",
            })
            continue
        v = _verify_one(text, cleaned)
        rec = {
            "source_dataset": s["source_dataset"],
            "source_doc_id": s["source_doc_id"],
            "input_chars": len(text),
            "output_chars": len(cleaned),
            "strict_pass": bool(v["strict"]["is_strict_equivalent"]),
            "structural_pass": bool(v["structural"]["is_structural_equivalent"]),
            "token_retention_pct": float(
                v["structural"]["token_retention_pct"]
            ),
            "strict_first_diff": v["strict"].get("first_diff"),
            "structural_first_diff": v["structural"].get("first_diff"),
        }
        results.append(rec)
        if not rec["structural_pass"]:
            ds = s["source_dataset"] or "unk"
            if per_dataset_failure_counts[ds] < args.save_failures:
                fname = (
                    f"{ds}__{s['source_doc_id']}.md"
                )
                (failures_dir / fname).write_text(
                    "# INPUT\n\n" + text + "\n\n# CLEANER OUTPUT\n\n" + cleaned
                    + f"\n\n# STRUCTURAL FIRST DIFF\n\n"
                    + (rec["structural_first_diff"] or "(none)"),
                    encoding="utf-8",
                )
                per_dataset_failure_counts[ds] += 1
            structural_failures.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(samples)}")

    # Summary.
    ok = [r for r in results if "error" not in r]
    strict_passes = sum(1 for r in ok if r["strict_pass"])
    structural_passes = sum(1 for r in ok if r["structural_pass"])
    mean_retention = (
        sum(r["token_retention_pct"] for r in ok) / len(ok) if ok else 0.0
    )
    per_dataset: Dict[str, Dict[str, Any]] = {}
    for r in ok:
        ds = r["source_dataset"] or "unk"
        d = per_dataset.setdefault(ds, {
            "n": 0, "strict_pass": 0, "structural_pass": 0,
            "retention_sum": 0.0,
        })
        d["n"] += 1
        d["strict_pass"] += int(r["strict_pass"])
        d["structural_pass"] += int(r["structural_pass"])
        d["retention_sum"] += r["token_retention_pct"]
    for ds, d in per_dataset.items():
        d["strict_frac"] = d["strict_pass"] / d["n"]
        d["structural_frac"] = d["structural_pass"] / d["n"]
        d["mean_retention"] = d["retention_sum"] / d["n"]
        d.pop("retention_sum")

    summary = {
        "total_sampled": len(results),
        "errored": sum(1 for r in results if "error" in r),
        "ok": len(ok),
        "strict_pass_rate": strict_passes / max(len(ok), 1),
        "structural_pass_rate": structural_passes / max(len(ok), 1),
        "mean_token_retention_pct": mean_retention,
        "per_dataset": per_dataset,
        "n_structural_failures": len(structural_failures),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "per_doc.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results),
        encoding="utf-8",
    )
    print()
    print("=== summary ===")
    print(f"  total sampled:       {summary['total_sampled']:>6}")
    print(f"  errored:             {summary['errored']:>6}")
    print(f"  ok (scored):         {summary['ok']:>6}")
    print(f"  strict pass rate:       {summary['strict_pass_rate']:>6.3f}")
    print(f"  structural pass rate:   {summary['structural_pass_rate']:>6.3f}")
    print(f"  mean token retention:   {summary['mean_token_retention_pct']:>6.3f}")
    print(f"  structural failures saved: {summary['n_structural_failures']:,}")
    print()
    print("=== per-dataset ===")
    print(f"  {'dataset':<46} {'n':>6} {'strict':>8} {'struct':>8} {'retention':>10}")
    for ds, d in sorted(per_dataset.items()):
        print(
            f"  {ds:<46} {d['n']:>6} "
            f"{d['strict_frac']:>7.3f} {d['structural_frac']:>7.3f} "
            f"{d['mean_retention']:>9.3f}"
        )
    print()
    print(f"summary.json + per_doc.jsonl + structural_failures/ → {args.out}")


if __name__ == "__main__":
    main()
