"""Pull 150 NEW challenging Phase-A docs (disjoint from the 90
already in top100_review) and run Pilot B + cmark-gfm verify
against them. Emits a per-pair report so we can spot any new
failure categories.

Strategy — balance across lenses so we get diverse challenge
shapes rather than 150 copies of the same class:

- 40 docs by reflow_joins (rank 25..100 — the band BELOW the
  already-reviewed top 25).
- 40 docs by gfm_chars_saved (rank 25..100).
- 20 docs by hr_chars_saved (rank 5..50 — we deprioritized HR
  in the first pass; give it more coverage here).
- 30 docs by joins_per_1k_chars (reflow density, rank 25..100).
- 20 docs by a size-balanced random across all three signals —
  includes some MEDIUM-altered docs (previously none) to make
  sure Pilot B holds up on less-extreme inputs too.

All 150 picks EXCLUDE any doc_id already in top100_review.

Writes:
- `150_new_challenging_stats.jsonl` — per-doc stats of picks.
- Per-pair output + cmark-gfm verify → report.
"""
from __future__ import annotations

import argparse
import heapq
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

import glossapi_rs_cleaner as c


PDF_SOURCES = {
    "openarchives.gr",
    "greek_phd",
    "Apothetirio_Pergamos",
    "Apothetirio_Kallipos",
    "eurlex-greek-legislation",
    "ellinika_dedomena_europaikou_koinovouliou",
    "opengov.gr-diaboyleuseis",
}


def load_excluded_doc_ids(existing_sample_dir: Path) -> set[tuple[str, str]]:
    """Return (dataset, doc_id) tuples already in top100_review."""
    excluded = set()
    for f in existing_sample_dir.glob("*_BEFORE.md"):
        # filename encodes dataset + did at the end
        m = re.match(
            r"^\d+_R\d+_H\d+_G\d+_pct\d+_([A-Za-z0-9_-]+?)_([A-Za-z0-9_]{1,22})_BEFORE\.md$",
            f.name,
        )
        if m:
            excluded.add((m.group(1), m.group(2)))
    return excluded


def rank_range(rows, key, start, stop, excluded):
    """Return `rows` sorted desc by key, sliced [start, stop), with
    excluded (dataset, doc_id) pairs filtered out."""
    # Filter out excluded first.
    filtered = [
        r for r in rows
        if (r["source_dataset"][:16].replace(".", "_"),
            r["source_doc_id"][:22]) not in excluded
    ]
    filtered.sort(key=lambda r: (r.get(key, 0) or 0, r["source_doc_id"]),
                  reverse=True)
    return filtered[start:stop]


def _fetch_one(parquet_path: str, doc_id: str) -> str:
    want = str(doc_id)
    try:
        t = pq.read_table(
            parquet_path,
            columns=["source_doc_id", "text"],
            filters=[("source_doc_id", "=", want)],
        )
        if t.num_rows:
            return t.column(1).to_pylist()[0] or ""
    except Exception:
        pass
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
    p.add_argument("--excluded-sample-dir", required=True, type=Path,
                   help="Sample dir whose docs to EXCLUDE (the original 90).")
    p.add_argument("--parquet-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260425)
    args = p.parse_args()

    # Gate: cmark-gfm must be available; otherwise this test is weak.
    probe = c.cmark_gfm_verify_py("a", "a")
    if not probe.get("is_available"):
        print("cmark-gfm not available — abort", file=sys.stderr)
        return 2

    excluded = load_excluded_doc_ids(args.excluded_sample_dir)
    print(f"excluded {len(excluded)} existing sample pairs")

    # Load stats — only PDF sources, only docs with pilot-B-relevant
    # alterations (reflow_joins > 0 OR gfm_chars_saved > 0 OR
    # hr_chars_saved > 0).
    rows = []
    with args.stats_jsonl.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("source_dataset") not in PDF_SOURCES:
                continue
            if (r.get("reflow_joins", 0) or 0) == 0 \
               and (r.get("gfm_chars_saved", 0) or 0) == 0 \
               and (r.get("hr_chars_saved", 0) or 0) == 0:
                continue
            rows.append(r)
    print(f"loaded {len(rows):,} candidate stat rows (PDF sources, altered)")

    picks_keyed: dict[tuple[str, str], dict] = {}

    def add(bucket):
        for r in bucket:
            k = (r["source_dataset"], r["source_doc_id"])
            picks_keyed[k] = r

    add(rank_range(rows, "reflow_joins", 25, 65, excluded))
    add(rank_range(rows, "gfm_chars_saved", 25, 65, excluded))
    add(rank_range(rows, "hr_chars_saved", 5, 25, excluded))
    add(rank_range(rows, "joins_per_1k_chars", 25, 55, excluded))

    # Size-balanced random fill — aim to get to 150 total.
    rng = random.Random(args.seed)
    rest = [r for r in rows
            if (r["source_dataset"], r["source_doc_id"]) not in picks_keyed]
    rng.shuffle(rest)
    for r in rest:
        if len(picks_keyed) >= 150:
            break
        picks_keyed[(r["source_dataset"], r["source_doc_id"])] = r

    picks = list(picks_keyed.values())
    print(f"final pick count: {len(picks)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_pass = 0
    n_refuse = 0
    n_bug = 0
    fails = []
    refusals = []
    manifest = []
    for i, rec in enumerate(picks):
        pname = rec["source_path"]
        did = rec["source_doc_id"]
        pfile = args.parquet_dir / pname
        if not pfile.is_file():
            continue
        body = _fetch_one(str(pfile), did)
        if not body:
            continue
        # Run Pilot B checked wrapper — it's what production would use.
        r = c.format_surgical_checked_py(body)
        manifest.append({
            "source_dataset": rec["source_dataset"],
            "source_doc_id": did,
            "input_chars": len(body),
            "output_chars": len(r["output"]),
            "changed": r["changed"],
            "preview_identical": r["preview_identical"],
            "dialect_ambiguous_input": r["dialect_ambiguous_input"],
            "fallback_reason": r["fallback_reason"],
            "reflow_joins": rec.get("reflow_joins"),
            "hr_chars_saved": rec.get("hr_chars_saved"),
            "gfm_chars_saved": rec.get("gfm_chars_saved"),
        })
        if r["fallback_reason"] is None:
            n_pass += 1
        elif r["dialect_ambiguous_input"]:
            n_refuse += 1
            refusals.append(manifest[-1])
        else:
            # Refused because rewrite changed preview — might be a
            # real Phase A bug surfacing. Verify via cmark-gfm.
            vr = c.cmark_gfm_verify_py(body, c.format_surgical_py(body))
            if not vr.get("preview_identical"):
                n_bug += 1
                fails.append({
                    **manifest[-1],
                    "cmark_gfm_first_diff": (vr.get("first_diff") or "")[:500],
                })
        if (i + 1) % 20 == 0 or i == len(picks) - 1:
            print(f"  [{i+1}/{len(picks)}] pass={n_pass} refuse={n_refuse} bug={n_bug}",
                  flush=True)

    report = {
        "total": len(manifest),
        "pass": n_pass,
        "refuse_dialect_ambiguous": n_refuse,
        "preview_changing_bugs": n_bug,
        "pass_pct": (100.0 * n_pass / len(manifest)) if manifest else 0.0,
        "refusals_sample": refusals[:20],
        "bugs_sample": fails[:20],
    }
    (args.output_dir / "150_new_challenging_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "150_new_challenging_manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest),
        encoding="utf-8")
    print()
    print(f"TOTAL: {len(manifest)}")
    print(f"  pass (rewrite shipped):         {n_pass} ({report['pass_pct']:.1f}%)")
    print(f"  refused (dialect-ambiguous):    {n_refuse}")
    print(f"  preview-changing bugs:          {n_bug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
