"""Frozen small whole-corpus compatibility gate: selection, IDs and shards.

The prescribed filesystem enumeration is an INPUT to this fixture. This is
not a new sorting rule for production. Products come from public wrappers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch


FIXTURE = Path(__file__).with_name("manifests") / "whole_corpus_compatibility.json"
ENUMERATION = ["zeta.md", "alpha.md", "beta.md", "orphan__p0001-0002.md"]
TEXTS = {
    "zeta.md": "# Ζήτα\n\nΠρώτη παράγραφος.\n\n## Μέθοδος\n\nΔεύτερη παράγραφος.",
    "alpha.md": "# Άλφα\n\nΈνα μικρό κείμενο.",
    "beta.md": "# Βήτα\n\nΚείμενο με πίνακα.\n\n| Α | Β |\n| --- | --- |\n| 1 | 2 |",
    "orphan__p0001-0002.md": "# Μόνο τμήμα\n\nΧωρίς κανονικό γονικό αρχείο.",
}


def capture(root):
    import pandas as pd
    import pyarrow.parquet as pq
    from glossapi import Corpus
    result = dict(schema_version=1, input_enumeration=ENUMERATION, texts=TEXTS,
                  shard_size_bytes=550, compression="none", scenarios={})
    for scenario in ("selected_good_files", "metadata_selection", "zero_accepted_fallback"):
        output = Path(root)/scenario
        corpus = Corpus(Path(root)/"empty", output)
        corpus._shadow_ledger_enabled = False
        raw = corpus.markdown_dir
        raw.mkdir(parents=True, exist_ok=True)
        corpus.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)
        for filename in ENUMERATION:
            (raw/filename).write_text(TEXTS[filename])
            (corpus.cleaned_markdown_dir/filename).write_text(TEXTS[filename])
        rows = [dict(filename=Path(name).stem+".pdf", url="https://fixture.invalid/"+Path(name).stem,
                     filter="ok" if scenario == "metadata_selection" and name == "alpha.md" else "bad",
                     ocr_success=False, extraction="bad", marker=ordinal)
                for ordinal, name in enumerate(ENUMERATION)]
        metadata = output/"download_results/download_results.parquet"
        metadata.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_parquet(metadata, index=False)
        corpus.good_files = ["beta", "zeta"] if scenario == "selected_good_files" else []
        real_listdir = os.listdir
        def prescribed(path):
            return list(ENUMERATION) if Path(path) == raw else real_listdir(path)
        with patch("os.listdir", prescribed):
            corpus.section()
        section_path = corpus.sections_dir/"sections_for_annotation.parquet"
        table = pq.read_table(section_path)
        parquet = pq.ParquetFile(section_path)
        shards = corpus.jsonl_sharded(output/"shards", shard_size_bytes=550, compression="none",
                                      metadata_fields=["marker"], metadata_key="metadata")
        membership = []
        for shard in shards:
            records = [json.loads(line) for line in shard.read_text().splitlines()]
            membership.append(dict(name=shard.name, bytes=shard.stat().st_size,
                                   sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                                   filenames=[r["filename"] for r in records], doc_ids=[r["doc_id"] for r in records], records=records))
        result["scenarios"][scenario] = dict(
            section_schema=str(table.schema.remove_metadata()), section_rows=table.to_pylist(),
            row_groups=[parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)],
            shards=membership, omitted_export_stems=["orphan"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True)
    parser.add_argument("--expected", default=str(FIXTURE))
    parser.add_argument("--capture", help="Explicit one-time capture, never an automatic update")
    args = parser.parse_args()
    result = capture(args.work)
    if args.capture:
        Path(args.capture).write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    else:
        assert result == json.loads(Path(args.expected).read_text()), "Whole-corpus compatibility fixture changed"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
