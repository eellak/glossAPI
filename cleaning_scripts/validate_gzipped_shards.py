"""Validate the gzipped text-shard output is exactly the cleaner's output
under the documented newline→space transformation, with no other alteration.

For each kept doc:
  1. Read raw input from the source parquet.
  2. Run `cleaner.clean_text_with_stats(raw, …, "parser_surgical_verified")`
     — same call the driver makes — to get the canonical cleaned text.
  3. Apply the driver's documented squash:
        cleaned.replace("\\r", " ").replace("\\n", " ") + "\\n"
     This is verbatim from clean_and_stats_rowsharded.py.
  4. Read the corresponding line from the gzipped shard.
  5. Compare. If unequal, report char-level diff position + neighbourhood.

Expected outcome on a clean run: every doc's gzipped line is byte-identical
to step (3). Any divergence means an alteration is happening between the
cleaner output and the on-disk shard (e.g. an unintended driver edit, an
encoding mismatch, a row-ordering bug).
"""
import gzip, json
from pathlib import Path

import pyarrow.parquet as pq
import glossapi_rs_cleaner as cleaner

INPUT = Path("/tmp/cleaner-100-test/input/openarchives_100.parquet")
STATS_DIR = Path("/tmp/cleaner-100-test/stats")
TEXT_DIR = Path("/tmp/cleaner-100-test/text-shards")

SCRIPTS = ["greek", "latin", "french", "spanish", "punctuation", "numbers", "common_symbols"]


def squash(s: str) -> str:
    """The driver's documented transformation, byte-for-byte."""
    return s.replace("\r", " ").replace("\n", " ") + "\n"


# Map source_doc_id → raw input text
src = pq.read_table(INPUT)
raw_by_id = dict(zip(src.column("source_doc_id").to_pylist(), src.column("text").to_pylist()))

shards = sorted(STATS_DIR.glob("*.stats.jsonl"))
checked = 0
diverged: list[tuple[str, int, str]] = []
for sp in shards:
    recs = [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
    text_path = TEXT_DIR / f"{sp.stem.split('.stats')[0]}.txt.gz"
    with gzip.open(text_path, "rt", encoding="utf-8", newline="") as f:
        for rec in recs:
            if rec.get("drop_reason"):
                # Dropped docs don't go to the text shard.
                continue
            doc_id = rec["source_doc_id"]
            raw = raw_by_id[doc_id]
            cleaned, _ = cleaner.clean_text_with_stats(
                raw, SCRIPTS, None, True, 30, 3, "parser_surgical_verified"
            )
            expected = squash(cleaned)
            actual = f.readline()
            checked += 1
            if expected != actual:
                # Locate the first differing character.
                first = next(
                    (i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                    min(len(expected), len(actual)),
                )
                ctx_e = expected[max(0, first - 40): first + 40]
                ctx_a = actual[max(0, first - 40): first + 40]
                diverged.append((doc_id, first, f"expected={ctx_e!r}  actual={ctx_a!r}"))

print(f"checked: {checked} docs")
if not diverged:
    print(f"PASS: every gzipped line is byte-identical to "
          f"`squash(clean_text_with_stats(raw, …))` — no alteration detected.")
else:
    print(f"FAIL: {len(diverged)} docs diverged from expected:")
    for doc_id, pos, ctx in diverged[:5]:
        print(f"  {doc_id}  @char {pos}")
        print(f"    {ctx}")
