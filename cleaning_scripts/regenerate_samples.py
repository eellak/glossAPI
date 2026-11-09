"""Regenerate /tmp/cleaner-100-test/cleaned_docs/ with FAITHFUL Pilot B
output (newlines preserved by calling `clean_text_with_stats` directly,
bypassing the driver's lossy newline→space squash).

Sampling filters applied:
  - empty body after stripping HTML comments and line-removed markers
  - Greek-letter ratio < 50% on the cleaned body
"""
import json, re, shutil
from pathlib import Path

import pyarrow.parquet as pq
import glossapi_rs_cleaner as cleaner

INPUT = Path("/tmp/cleaner-100-test/input/openarchives_100.parquet")
STATS_DIR = Path("/tmp/cleaner-100-test/stats")
OUT = Path("/tmp/cleaner-100-test/cleaned_docs")

SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

src = pq.read_table(INPUT)
ids = src.column("source_doc_id").to_pylist()
texts = src.column("text").to_pylist()

stats_by_id = {}
for sp in STATS_DIR.glob("*.stats.jsonl"):
    for line in sp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        stats_by_id[rec["source_doc_id"]] = rec

written = 0
skipped_empty = 0
skipped_low_greek = 0
for doc_id, raw in zip(ids, texts):
    cleaned, _ = cleaner.clean_text_with_stats(
        raw, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
    )
    body = COMMENT_RE.sub("", cleaned).strip()
    if not body:
        skipped_empty += 1
        continue
    letters = LETTER_RE.findall(body)
    if not letters:
        skipped_empty += 1
        continue
    greek_ratio = sum(1 for ch in letters if GREEK_RE.match(ch)) / len(letters)
    if greek_ratio < 0.50:
        skipped_low_greek += 1
        continue

    s = stats_by_id.get(doc_id, {})
    pct = s.get("pct_chars_removed_non_empty", 0.0)
    glyph = s.get("counter_glyph_marker", 0)
    residue = s.get("counter_script_residue", 0)
    header = (
        f"<!-- source_doc_id={doc_id} -->\n"
        f"<!-- chars_before={s.get('chars_before','?')} chars_after={s.get('chars_after','?')} "
        f"pct_removed_non_empty={pct:.2f}% greek_ratio={greek_ratio:.3f} -->\n"
        f"<!-- counters: glyph_marker={glyph} script_residue={residue} -->\n\n"
    )
    fname = f"{pct:06.2f}__{doc_id[:16]}.md"
    (OUT / fname).write_text(header + cleaned, encoding="utf-8")
    written += 1

print(f"wrote {written} docs to {OUT}")
print(f"skipped (empty body):    {skipped_empty}")
print(f"skipped (<50% Greek):    {skipped_low_greek}")
