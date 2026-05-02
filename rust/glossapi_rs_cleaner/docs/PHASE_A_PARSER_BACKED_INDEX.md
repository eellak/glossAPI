# Phase A parser-backed rewrite — work index (2026-04-24)

Index of the main docs, Rust modules, Python scripts, and data
artifacts produced while turning the line-heuristic Phase A
(`md_module`) into a parser-backed pipeline verified against
GitHub's reference renderer.

## Headline result

### Pass rate iteration on 90 hardest-altered PDF corpus docs (cmark-gfm preview-identity)

| Approach | Pass rate |
|---|---|
| Line-based Phase A (original) | ~26% |
| Pilot A — comrak full round-trip | 26.7% |
| Pilot B v1 — serialize Paragraphs via comrak | 67.8% |
| Pilot B v2 — source-level SoftBreak unwrap | 82.2% |
| Pilot B v3 — delimiter-only table rewrite + blank-line pad | 87.8% |
| Pilot B v4 — targeted blank-line only before HR | 96.7% |
| **Pilot B v5 (current) — ASCII-only trim (NBSP preserved as content)** | **98.9% (89/90)** |

Zero regressions across all iterations.

### `format_surgical_checked` (production candidate) on 240 hardest-altered docs

`format_surgical_checked_with_oracles` — the checked wrapper that
production would use — exercised across two disjoint corpus
samples for a total of 240 docs:

| Sample | Shipped rewrite | Shipped input (no-op) | Refused (safety net fired) | Production bugs |
|---|---|---|---|---|
| 90 original (top100_review) | varies | varies | 1 | **0** |
| 150 new challenging (challenge150) | 87 | 61 | 2 | **0** |
| **Total: 240** | — | — | **3** | **0** |

**Zero preview-violations shipped to production output across all
240 docs.** The 3 refusals are multi-MB documents (2.1MB / 474KB
range) where the GFM table boundary is dialect-ambiguous; the
checked wrapper correctly emits input verbatim with a
`fallback_reason` rather than risking a preview change.

## Rust modules (all under `rust/glossapi_rs_cleaner/src/`)

- `md_format.rs` — Pilot A: parse with comrak, re-serialize via
  `format_commonmark`, dual-parser verifier
  (pulldown-cmark + comrak HTML agreement).
- `md_format_surgical.rs` — **Pilot B (current production
  candidate)**: walks the comrak AST, rewrites only
  Paragraph / Table / ThematicBreak spans, keeps everything else
  byte-exact from source. Paragraphs use a source-level SoftBreak
  unwrap. Tables use delimiter-row-only rewrite. HRs canonicalize
  to `---`.
- `cmark_gfm_oracle.rs` — cmark-gfm C subprocess oracle. Renders
  input and output via `/usr/bin/cmark-gfm` (GitHub's reference
  renderer) and compares HTML after preview-equivalent whitespace
  normalization. 9 ground-truth anchors encode CM-spec edge cases
  (escaped-underscore → literal, 2-space hard break, optional-pipe
  table, setext markers, etc.).

## Python scripts (all under `cleaning_scripts/`)

Corpus audit + sampling:

- `compute_phase_a_stats_per_doc.py` — runs Phase A on each doc in
  a parquet dir, emits per-doc jsonl of per-transform counters
  (reflow joins, HR chars saved, GFM chars saved, …). 168,078-doc
  run took ~7 min on the cleaning instance.
- `pull_top_phase_a_altered.py` — pulls top-N docs per metric lens
  (reflow / HR / GFM / density / reflow+tables composite), with a
  `--pdf-sources-only` filter. Emits `{rank}_R..._H..._G..._{did}_BEFORE.md`
  and `..._AFTER.md` pairs per pick.
- `extract_reflow_segments.py` — given a BEFORE/AFTER pair, emits
  JSON of the N largest reflow-caused diff regions with
  surrounding context, so reviewers can inspect reflow decisions
  without reading full docs.

Verification:

- `verify_md_format_via_cmark_gfm.py` — runs a formatter over a
  sample dir, verifies each pair via cmark-gfm. Takes
  `--formatter format_parsed_py|format_surgical_py` to switch
  Pilot A vs Pilot B.
- `compare_pilots_via_cmark_gfm.py` — runs BOTH pilots, reports
  side-by-side pass rates + which pilot recovers / breaks which
  failures.
- `verify_phase_a_sample_pairs.py` — older pulldown-cmark-only
  verifier. Superseded by the cmark-gfm version; kept for
  backward comparison.
- `classify_cmark_failures.py` — reads a verify report, classifies
  each failure by heuristic signature, and indexes each failure
  back to a source MD line number so a reviewer can jump directly
  to the problem spot instead of reading full docs.

## Documentation (all under `rust/glossapi_rs_cleaner/docs/`)

- `MD_MODULE_ARCHITECTURE.md` — live architecture doc for the
  md module. Current; reflects line-based Phase A plus the C11–C16
  review-response series.
- `MD_MODULE_ARCHITECTURE_IMPLEMENTATION_REVIEW_2026-04-24.md` —
  independent review of the C11-era implementation + our
  point-by-point responses and follow-up Q&A.
- `MD_LIBRARY_SURVEY_LEARNINGS_2026-04-24.md` — comparative survey
  of CommonMark/GFM parsers across Rust (comrak, pulldown-cmark),
  C (cmark-gfm), JS (remark, markdown-it), Python (mdformat), Go
  (goldmark), Pandoc. This is what drove the parser-backed pilot
  direction and the dual-parser-oracle approach.
- `PHASE_A_PARSER_BACKED_INDEX.md` — this file.

## Data artifacts (under `/home/foivos/data/phase_a_audit/`)

- `phase_a_stats.jsonl` — 168,078 rows, ~65 MB. Per-doc Phase A
  alteration stats across the unified corpus.
- `top100_review/` — 180 files (90 × BEFORE/AFTER), ~480 MB. The
  small iteration corpus used to drive Pilot B refinement.
- `cmark_pilot_b_report.json` — latest cmark-gfm verifier report
  on Pilot B. 87/90 pass, 3 residuals.
- `pilot_comparison.json` — side-by-side Pilot A vs B pass rates.
- `cmark_pilot_b_failures_indexed.json` — residual failures with
  source-line jump pointers.

## Coordination

- `/home/foivos/AGENT_COORDINATION.md` — shared file between
  Claude-Cleaner (cleaner + audits) and Claude-MD (md module +
  verifier). Ownership boundaries, planned/in-flight shared runs,
  Q&A. Updated whenever a long run goes on the cleaning instance.

## Reproduce

On the cleaning instance (`apertus-greek-tokenizer-20260408t160000z`,
europe-west4-b, CPU-only m3-megamem-64, taskset -c 0-31 for the
agent's 32-vCPU budget):

```
source ~/venvs/glossapi-corpus-clean/bin/activate
cd ~/data/phase_a_audit

# Corpus-wide Phase A alteration stats (~7 min):
taskset -c 0-31 python compute_phase_a_stats_per_doc.py \
    --parquet-dir ~/data/glossapi_work/unified_corpus/data \
    --output phase_a_stats.jsonl

# Pull top-90 most-altered PDF-only sample (~30 s):
taskset -c 0-31 python pull_top_phase_a_altered.py \
    --stats-jsonl phase_a_stats.jsonl \
    --parquet-dir ~/data/glossapi_work/unified_corpus/data \
    --output-dir top100_review \
    --pdf-sources-only

# Compare Pilot A and Pilot B against cmark-gfm (~30 s):
taskset -c 0-31 python compare_pilots_via_cmark_gfm.py \
    --sample-dir top100_review \
    --output pilot_comparison.json

# Pilot-B-only pass rate + residuals with source-line index:
taskset -c 0-31 python verify_md_format_via_cmark_gfm.py \
    --sample-dir top100_review \
    --output cmark_pilot_b_report.json \
    --formatter format_surgical_py
taskset -c 0-31 python classify_cmark_failures.py \
    --report cmark_pilot_b_report.json \
    --sample-dir top100_review \
    --output cmark_pilot_b_failures_indexed.json
```

Local wheel rebuild (both laptop and instance):

```
cd rust/glossapi_rs_cleaner
source <venv>/bin/activate
maturin develop --release
```

Rust test suite (entire repo):

```
cargo test --release --lib
```

## Git commits (chronological, all on `codex/three-counter-pipeline-20260421`)

- `c4716d8` — Pilot B v4: source-level SoftBreak unwrap + delimiter-
  only table + targeted blank-line pad → 96.7% pass rate.
- `a1cf8c1` — Pilot A + B v1 + cmark-gfm oracle + 29-fixture suite.
- `17fc14f` — Drop buggy `\_\_\_\_` HR rule, add blank-line collapse
  to line-based Phase A, bucket escaped-underscore runs.
- `b825fb2` — Phase A instrumentation: `PhaseAStats` +
  `normalize_md_syntax_with_stats` + JSONL emitter.
- `0649f3a`, `bfc1e03`, `f1b0f65`, `88609f6`, `10aaa3e`, `965a8fd`,
  `c6de5e5`, `f50ddab` — C11–C16 review-response series on the
  line-based Phase A (escaped underscores, CommonMark indentation
  awareness, hard-break preservation, shared canonicalization,
  structural verifier coverage).

Not pushed to `origin` yet.

## Status

- Core implementation: **done and tested** (Pilot B, 96.7% on the
  hardest 90 corpus docs, 29/29 synthetic fixtures).
- Verifier: **done and tested** (cmark-gfm subprocess oracle, 9
  ground-truth anchors, whitespace-normalized preview identity).
- Integration into the main cleaner pipeline: **not yet**. The
  line-based `md_module::normalize_md_syntax` is still the Phase A
  the cleaner invokes. `format_surgical_py` is exposed as a PyO3
  entry point; swapping it in is the next integration step, gated
  on a full-corpus scorecard run.
- Dialect-ambiguity refusal path: **not yet** — would take the 3
  residuals to a clean skip.
