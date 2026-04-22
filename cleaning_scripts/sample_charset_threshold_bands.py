"""Sample docs in bands around each charset threshold for sub-agent
manual review. Writes .md files per doc with stats + doc text so an
Agent can read them directly.

Thresholds:
  moji_residue_ratio  = 0.30  → band [0.20, 0.40] = 25 docs
  ascii_punct_ratio   = 0.30  → band [0.20, 0.40] = 25 docs
  greek_letter_ratio  = 0.05  → band [0.02, 0.15] = 25 docs (below vs
                                  above the cutoff)

Output layout:
  <out>/
    near_moji_threshold/    <prefix_ratio>_<dataset>_<doc>.md
    near_punct_threshold/
    near_greek_threshold/
    REVIEW_INDEX.md         overview + per-file doc path
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.parquet as pq


def _safe(s: str, max_len: int = 30) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))[:max_len].strip("_") or "x"


def _iter_stats(stats_glob: str):
    for p in sorted(globmod.glob(stats_glob)):
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def _load_band(stats_glob: str, ratio_field: str, lo: float, hi: float
               ) -> List[Dict[str, Any]]:
    out = []
    for d in _iter_stats(stats_glob):
        v = d.get(ratio_field)
        if v is None:
            continue
        vf = float(v)
        if lo <= vf <= hi:
            out.append(d)
    return out


def _bulk_find(parquet_path: str, doc_ids: List[str]) -> Dict[str, str]:
    want = set(str(d) for d in doc_ids)
    out = {}
    try:
        table = pq.read_table(
            parquet_path, columns=["source_doc_id", "text"],
            filters=[("source_doc_id", "in", list(want))],
        )
    except Exception:
        table = None
    if table is not None and table.num_rows:
        for sid, txt in zip(table.column("source_doc_id").to_pylist(),
                            table.column("text").to_pylist()):
            out[str(sid)] = txt or ""
    # Fallback scan.
    if set(want) - set(out):
        pf = pq.ParquetFile(parquet_path)
        missing = set(want) - set(out)
        for batch in pf.iter_batches(batch_size=5000, columns=["source_doc_id", "text"]):
            sids = batch.column(0).to_pylist()
            txts = batch.column(1).to_pylist()
            for sid, txt in zip(sids, txts):
                if str(sid) in missing:
                    out[str(sid)] = txt or ""
                    missing.discard(str(sid))
                    if not missing: return out
    return out


def _write_doc_md(target: Path, doc_id: str, record: Dict[str, Any],
                  text: str, band_name: str, max_chars: int = 8000) -> None:
    body = [f"# {record.get('source_dataset')} / {doc_id}", "",
            f"- **band**: {band_name}",
            f"- **chars_before**: {record.get('chars_before')}",
            f"- **drop_reason**: {record.get('drop_reason') or '(kept)'}",
            "",
            f"- **charset_greek_ratio**: {record.get('charset_greek_ratio')}",
            f"- **charset_moji_ratio**: {record.get('charset_moji_ratio')}",
            f"- **charset_punct_ratio**: {record.get('charset_punct_ratio')}",
            "", "## Text sample", ""]
    body.append("```")
    if len(text) > max_chars:
        half = max_chars // 2
        body.append(text[:half])
        body.append(f"\n[...truncated {len(text) - max_chars} chars...]\n")
        body.append(text[-half:])
    else:
        body.append(text)
    body.append("```")
    target.write_text("\n".join(body), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-glob", required=True)
    parser.add_argument("--parquet-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-band", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260422)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    bands = [
        ("near_moji_threshold",   "charset_moji_ratio",  0.20, 0.40),
        ("near_punct_threshold",  "charset_punct_ratio", 0.20, 0.40),
        ("near_greek_threshold",  "charset_greek_ratio", 0.02, 0.15),
    ]

    index_lines = ["# Charset threshold borderline review", "",
                   "Each `.md` file has source metadata, the three charset "
                   "ratios, and a sample of the raw text. For each band, skim "
                   "the docs and decide whether the filter's threshold is "
                   "discriminating correctly.", ""]

    for name, field, lo, hi in bands:
        docs = _load_band(args.stats_glob, field, lo, hi)
        rng.shuffle(docs)
        picks = docs[: args.per_band]
        band_dir = args.output_dir / name
        band_dir.mkdir(parents=True, exist_ok=True)

        by_parquet: Dict[Path, List[str]] = defaultdict(list)
        for rec in picks:
            parquet_name, _, doc_id = rec["source_path"].partition("#")
            by_parquet[args.parquet_dir / Path(parquet_name).name].append(doc_id)
        text_cache: Dict[str, Dict[str, str]] = {}
        for pfile, dids in by_parquet.items():
            if not pfile.is_file(): continue
            text_cache[str(pfile)] = _bulk_find(str(pfile), dids)

        index_lines.append(f"## {name} — band [{lo}, {hi}] on {field}")
        index_lines.append(f"Population: {len(docs)} docs, sampled {len(picks)}")
        index_lines.append("")

        for rec in picks:
            pname, _, doc_id = rec["source_path"].partition("#")
            text = text_cache.get(str(args.parquet_dir / Path(pname).name), {}).get(doc_id)
            if not text:
                continue
            val = float(rec.get(field, 0.0))
            ratio_prefix = f"{int(round(val * 100)):03d}"
            reason = rec.get("drop_reason") or "kept"
            ds_short = _safe(rec.get("source_dataset", ""), 18)
            did_short = _safe(doc_id, 24)
            fname = f"{ratio_prefix}_{reason}_{ds_short}_{did_short}.md"
            _write_doc_md(band_dir / fname, doc_id, rec, text, name)
            index_lines.append(f"- `{band_dir.name}/{fname}` — {field}={val:.3f}, {reason}")
        index_lines.append("")

    (args.output_dir / "REVIEW_INDEX.md").write_text(
        "\n".join(index_lines), encoding="utf-8")
    print(f"wrote borderline review tree → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
