# MD-module architecture + verification plan (2026-04-24)

Preserved from the design discussion between user and assistant.
Captures the goal, the architectural split, and the commit plan so a
fresh reader can pick up the work.

## Goal

Transform MD files so they are **dual-readable**:

- In **preview** (any spec-compliant renderer — GitHub, VS Code,
  pandoc, mdbook, etc.): visibly identical to the original MD. No
  lost paragraphs, headings, tables, lists, emphasis. User sees the
  same content.
- In **raw** (opening the file in a plain editor, or the tokenizer
  consuming it as a byte stream): reads as natural text. No PDF-
  column-wrap fragmentation, no 200-char dash separators, no
  multi-line soft-wrapped paragraphs. A reader can follow the text
  without relying on the preview renderer to reflow it.

This is the original Markdown design ethos: raw text should look
like what a human would write, AND render well. Most MD tooling
ignores this today; the cleaner's job is to restore it.

## Two transform classes

Cleaner passes split into two categories with distinct invariants:

### Phase A — MD-syntax-aware, preview-preserving

Transforms that require knowing CommonMark/GFM grammar to be correct.
Raw chars CHANGE by design — we want `&amp;` → `&` in the corpus —
but the preview render is unchanged.

Invariant: `pandoc-render(input) ≡ pandoc-render(output)` (modulo
whitespace). Any Phase A transform that violates this is a bug.

Members:
- **Paragraph linearization** (`reflow_paragraphs`): collapse
  soft-wrap `\n`s within a paragraph block into a single space.
  Guards: headings, blockquotes, list markers, tables, HRs, fenced
  code, sentence terminators, indented-code blocks.
- **GFM table separator minimization** (`scan_gfm_table_separators`):
  rewrite `| ---------- | ---------- |` to `| --- | --- |` while
  preserving alignment colons (`:---`, `---:`, `:---:`).
- **HR thematic-break minimization** (`normalize_separator_line`):
  rewrite `-------` / `___________` / `*****` runs (≥4 chars) and
  the markdown-escaped underscore form `\_\_\_\_` to `---`. Length
  threshold ≥4 for the *rewrite* rule (`---` is already canonical,
  so no-op). Rejected at ≥4 leading columns (indented code). A
  *separate* ≥3-char recognizer (`HR_HARD_BREAK_REGEX`) is used by
  the reflow hard-break detector so canonical `---` and setext
  headings (`===`, `---`) are still treated as block boundaries.

  **Intentionally NOT rewritten (would change preview):**
  `====` runs (setext heading marker / literal `=` paragraph);
  Unicode em-dash / horizontal-bar / box-drawing (`———`, `═══`,
  `───`) — these parse as literal paragraphs under CommonMark.
- **Fenced code detector** (`is_code_fence_marker`): predicate used
  by the other three (and by the cleaner) for code-fence state
  tracking. Takes the RAW line and rejects at ≥4 leading columns
  (indented `` ``` `` is literal code, not a fence opener).
- **CommonMark indentation helper** (`leading_columns`): returns
  the column of the first non-whitespace char under the CM tab rule
  (tab advances to next multiple of 4). Used by every Phase A
  detector to enforce the indented-code-block boundary.

### Non-destructive canonical form

A public function `non_destructive_canonicalize(md)` defines the
single canonical form that the cleaner produces if every pass were
non-destructive — entity decode + PUA decode + soft-hyphen strip +
per-line char-fold/dot-runs/whitespace-runs/ellipsis-runs +
orchestrator. The structural verifier (`md_verify`) delegates to
this function for its input-side canonicalization, so the verifier
baseline cannot drift from what the cleaner produces. A set of
`drift_cleaner_eq_canonicalize_*` tests in `cleaning_module::tests`
locks this equivalence in.

Preview-preserving recovery passes that conceptually belong to
Phase A but currently live in `normalize.rs` (historical reasons;
the module boundary is "requires CommonMark/GFM parser knowledge
to be correct" and entity / PUA decode don't strictly need that
to be safe — so they can stay outside `md_module.rs`):

- **HTML entity decode** (`normalize::decode_html_entities`):
  `&amp;` → `&`, `&#8212;` → `—`, etc. Preview-identical because
  the decoder applies exactly the mapping a spec-compliant
  renderer would.
- **Adobe Symbol PUA decode** (`normalize::decode_adobe_symbol_pua`):
  U+F061 → α etc. Preview-identical at the rendered-glyph layer;
  PUA codepoints with no font fallback render as `.notdef`
  boxes, so mapping them to the intended real char is at worst
  neutral and at best a fix.
- **Soft-hyphen strip** (`normalize::strip_soft_hyphens`): U+00AD
  deleted. U+00AD is zero-width (invisible) in every renderer, so
  its deletion is preview-identical.

All three are included in `non_destructive_canonicalize` and run
before the Phase A orchestrator in the cleaner.

### Phase B — content-modifying

Transforms that deliberately REMOVE content visible in preview.
Preview WILL differ after; that's the whole point.

Invariant (structural only): number and type of block elements
preserved; token sequence within each block is a monotone
subsequence of input (allows deletions, disallows reorderings, NO
fusions).

Members (implemented, scattered across modules):
- **GLYPH marker strip** (`strip_glyph_markers`): `GLYPH<\d+>` /
  `/uniXXXX` / `/gN` deleted. Runs BEFORE Phase A in the cleaner
  (markers would otherwise break pipe-count scanning inside table
  rows); see "Order of operations" below for why this destructive
  pass runs pre-Phase-A.
- **Per-char filter**: chars outside allowed-scripts set deleted.
- **Line-drop** (rule-A / rule-B / glyph-regex): whole lines
  replaced with `<!-- line-removed -->` marker.

## Order of operations

The actual pipeline in `core_clean_text_with_stats` is NOT a clean
"Phase A → Phase B" split. Earlier revisions of this doc described
Phase A as running first, but in practice a small set of
content-level passes runs before the Phase A orchestrator because
Phase A's structural scanners (pipe counting, run matching) need
to see post-decode, post-marker-strip text to be accurate.

```
  input MD
    ↓
  Pre-Phase-A — preview-preserving recovery:
    • decode_html_entities         (&amp; → &)
    • decode_adobe_symbol_pua      (U+F061 → α)
    ↓
  Pre-Phase-A — destructive strips that MUST run before Phase A:
    • strip_glyph_markers          (GLYPH<…> / /uniXXXX deleted;
                                    renders as literal text, so
                                    this IS destructive — but
                                    leaving it would make Phase A
                                    miscount pipes inside rows)
    • strip_soft_hyphens           (U+00AD invisible anyway)
    ↓
  Phase A — MD-syntax normalization (preview-preserving):
    • md_module::normalize_md_syntax
        - scan_gfm_table_separators
        - normalize_separator_line (HR min)
        - reflow_paragraphs
    ↓
  Phase B — per-line cleanup (destructive by design):
    • per-line char-fold / dot / whitespace / ellipsis normalize
    • per-char allowlist filter
    • rule-A / rule-B line-drop → <!-- line-removed --> markers
    ↓
  output MD
```

Notes on the ordering:

- **Entity + PUA decode run first** so Phase A's detectors see the
  same glyphs a renderer would. If Phase A ran first, a `&#8212;`
  (em-dash) would still look like an entity rather than `—`, and
  the structure Phase A sees could differ from what the renderer
  sees.
- **GLYPH-marker strip runs BEFORE Phase A** even though it IS
  destructive (the markers render as literal text). If GLYPH
  markers survived to Phase A, a `|GLYPH<7>|` inside a row would
  break `scan_gfm_table_separators`' pipe-count check and leak
  literal marker content into a canonicalized table row. The cost
  — a small class of Phase-A-structural decisions being made on
  content the renderer would have shown — is considered acceptable
  because GLYPH markers are themselves extractor noise, not
  author-intended text.
- **Soft-hyphen strip runs before Phase A** because U+00AD is
  zero-width: removing it is structurally preview-preserving.
- **Phase A runs in the middle** so subsequent destructive passes
  operate on canonicalized MD. Per-char filters and line-drops can
  safely leave structural chars (`|`, `#`, `---`) alone because
  the MD is already in compact canonical form.
- **Phase B runs last** as the per-line loop inside
  `core_clean_text_with_stats`. This is where content actually
  gets removed; it's what `verify_md_structural` bounds.

The non-destructive baseline `non_destructive_canonicalize` (used
by the verifier) applies entity-decode + PUA-decode + soft-hyphen
strip + per-line normalization + Phase A. It deliberately SKIPS
`strip_glyph_markers` (destructive) and Phase B. Drift-prevention
tests in `cleaning_module::tests` assert that on inputs without
GLYPH markers — and with a permissive allowed-chars so Phase B
deletes nothing — `core_clean_text_with_stats` and
`non_destructive_canonicalize` produce identical output.

## Key architectural constraints

1. **All deterministic per-doc work in Rust.** Per the
   `feedback_rust_for_corpus_pipelines` memory rule. Python is a
   thin driver.
2. **Co-locate per-text-type logic.** Per
   `feedback_group_cleaner_features_by_text_type`. MD-syntax-aware
   transforms live in `md_module.rs`. LaTeX transforms live in
   `latex_module.rs`. Char-level transforms (entity decode, PUA
   decode, etc.) live in `normalize.rs` or smaller specialized
   modules.
3. **No threshold rules without user request.** Per
   `feedback_no_threshold_rules_unprompted`. The MD module
   transforms text but doesn't make keep/reject decisions.
4. **Post-cleaner samples default.** Per
   `feedback_review_samples_post_cleaner_default`. Samples show
   cleaner output; verification runs against samples.
5. **Normalize runs after cleaning — but MD-syntax normalize is a
   pre-pass.** Per `feedback_normalize_after_cleaning`, other
   normalize passes (whitespace/dot/separator/ellipsis/entities)
   run AFTER cleaning. Phase A (MD-syntax) is an exception — it
   runs before cleaning so subsequent content passes see
   canonical structure. Phase A doesn't overlap semantically with
   the post-cleaning normalize step.

## Verification plan

Two verification modes.

### Strict mode (Phase A only)

Via `md_verify.rs` using `pulldown-cmark`.

Checks:
1. **HTML-render equality**: parse both input and cleaner-output via
   pulldown-cmark → render to HTML → whitespace-collapse → compare.
   Fails if any paragraph dropped, any heading level changed, any
   table structure changed, any list rearranged.
2. **Block-event sequence equality**: flatten both to block-level
   event streams (`Start(Paragraph)`, `Start(Heading)`,
   `Start(Table)`, `Start(List)`, etc.) → compare. Catches cases
   where HTML differs in whitespace but element sequence matches
   (stricter than HTML equality).
3. **Per-paragraph content tokens**: for each matched paragraph,
   extract all leaf `Text` events, tokenize on whitespace,
   compare token sequences.
4. **Per-table cell mapping**: for each matched Table, traverse
   Row×Cell, assert same cell count, same whitespace-collapsed
   cell text.

Used by: unit tests for every Phase A transform + orchestrator.

### Structural mode (full cleaner, Phase A + B)

Also via `md_verify.rs`, different entry point.

Checks:
- Number and type of top-level block elements match.
- Tokens in each block are a monotone subsequence of input tokens
  (permits deletions, disallows reorderings and fusions).
- Table cell count preserved per table.

Used by: corpus-sampling script (`cleaning_scripts/verify_md_equivalence.py`)
that takes N random docs, runs full cleaner, reports pass/fail
rate and failure modes.

### Why not run strict mode on the full cleaner

The cleaner's Phase B DELIBERATELY modifies content (deletes glyph
markers, drops lines). Strict mode always fails by design on Phase
B. The structural mode is the right contract: Phase B can remove
things but not add, reorder, or fuse.

### Why not run either mode on the full 956k corpus

Too slow (pulldown-cmark parse per doc × 956k). Sample-based
(100-500 docs) is sufficient for dev-time regression signal.
Measurement is a periodic scorecard, not a hot-path check.

## Commit history

The original 6-commit plan landed (C1–C6, March–April 2026). The
2026-04-24 independent review (`MD_MODULE_ARCHITECTURE_IMPLEMENTATION_
REVIEW_2026-04-24.md`) surfaced five gaps and they were addressed
across five follow-up commits:

- **C11** — failing tests first (RED): 6 expected-failure tests +
  2 property-green tests documenting current coverage.
- **C12** — CommonMark indentation awareness. Added
  `leading_columns` helper; HR / GFM / fenced-code detectors all
  bail at ≥4 leading columns (indented-code block per CM).
- **C13** — cleaner routes through `normalize_md_syntax` as single
  Phase A entry; reflow preserves CommonMark hard breaks (`  \n`
  and `\\n`); reflow recognizes canonical `---` and setext markers
  as hard breaks; orchestrator step 2 made fence-aware.
- **C14** — extracted `non_destructive_canonicalize` as shared
  source of truth for verifier and cleaner; five drift-prevention
  tests lock equivalence in.
- **C15** — structural verifier's token extractor broadened from
  `Paragraph` only to also cover `Heading` and `Item` (tight list
  items emit text directly inside `Item` under pulldown-cmark).

Not adopted (deferred): expanding reflow to join across sentence
terminators. The reviewer's M-1 suggestion is reasonable spec-wise
but would change raw training-corpus shape; gating on a corpus-
level scorecard comparison and user review before landing. Until
then, reflow remains conservative.

## Resolved bugs (formerly "known bugs to fix")

These are landed in the implementation; preserved here so future
readers understand why the architecture looks the way it does.

- **v6-11**: NBSP (U+00A0) stripped by per-char filter fusing
  words. Fixed by folding U+00A0 and other Unicode whitespace
  variants to U+0020 in the pre-filter char-fold pass. Regression
  test in `cleaning_module::tests`. Structural verifier catches any
  reintroduction as a "Fusion" subsequence failure.
- **Optional-pipe GFM table destruction** (H-2 from 2026-04-24
  review): cleaner's Phase A pre-pass used to call
  `reflow_paragraphs` directly, running before
  `scan_gfm_table_separators` saw the text. Optional-pipe tables
  like `a | b\n--- | ---\n1 | 2` weren't recognized as tables by
  reflow, so the separator row got fused with the first body row.
  Fixed in C13: cleaner now goes through `normalize_md_syntax` as
  a single entry, so tables are identified first and then their
  rows are treated as reflow hard breaks.
- **CommonMark hard-break destruction** (H-3): reflow's
  `can_join_lines` called `trim_end()` before checking for the
  two-trailing-space / trailing-backslash hard-break markers,
  silently dropping `<br>` renders. Fixed in C13: detection now
  happens on the raw line before trimming, with proper
  backslash-parity accounting.
- **Indented-code corruption** (H-1): Phase A detectors used
  `trim_start()` or whitespace-permissive regex, so a `----` or
  `| a | b |` at ≥4 leading columns was rewritten even though
  CommonMark parses it as literal code content. Fixed in C12:
  `leading_columns` helper + early returns at ≥4 in all three
  detectors.

## Still out of scope / future work

- **v6-07 / v6-10**: pseudo-tables (TOC wrapped as tables) —
  currently pass through unchanged. A future cleaner pass will
  unwrap them; that pass will fail strict equivalence by design
  (tables → prose) and needs a dedicated invariant (not the strict
  preview-equivalence one).
- **v6-03**: single-line `$$…$$` LaTeX not excluded from
  `charset_punct_ratio` — ratio-computation bug, orthogonal to MD
  syntax.
- **Reflow expansion across sentence terminators** (M-1 from
  2026-04-24 review): deferred pending scorecard comparison.

## Terminology corrections captured during discussion

- "Cosmetic vs. syntactic" was user's initial split; refined to
  "semiotic vs. syntactic" where semiotic = has meaning for a human
  reader (even if parser-invisible). HRs and dot-leaders are
  semiotic; only the former are also syntactic (parser recognizes
  `<hr/>`). So "cosmetic" ≠ "useless" — it still matters for raw
  readability. The module boundary is "requires MD-parser knowledge
  to be correct."
- "Content preserving" — I initially called Phase A this; user
  correctly pointed out that `&amp;` → `&` IS a content change from
  the model's perspective (model trains on raw chars). The correct
  invariant is "preview-render equivalent," not "content
  identical." Phase A raw chars change by design; only the
  rendered output is unchanged.
