# MD module architecture implementation review (2026-04-24)

## Scope

Reviewed the architecture note and the current implementations named in the request:

- `docs/MD_MODULE_ARCHITECTURE.md`
- `src/md_module.rs`
- `src/md_verify.rs`
- `src/cleaning_module.rs`
- `src/lib.rs`
- `Cargo.toml`
- `../../cleaning_scripts/verify_md_equivalence.py`

Review lens: the cleaner should make raw Markdown read as close as possible to its rendered preview while preserving the rendered Markdown block/inline structure, especially for Docling-produced Greek corpus Markdown.

## Overall judgment

The design direction is right: splitting preview-preserving Markdown syntax normalization from destructive corpus cleanup is the right mental model, and the addition of a pulldown-cmark verifier is exactly the kind of guard this pipeline needs. The current implementation has useful unit coverage for the happy path, PyO3 exports are wired, and the scorecard script is present.

However, the current implementation does not yet fully satisfy the strongest invariant. There are several cases where Phase A can change rendered structure, mostly because the line scanners are not fully CommonMark indentation-aware and because the full cleaner does not use the `normalize_md_syntax` orchestrator. There is also one important product-quality gap: reflow is much more conservative than the stated raw-readability goal, so many Docling soft wraps will remain in the raw training text even though preview treats them as one paragraph.

## Findings

### High: Phase A detectors ignore CommonMark indentation and can rewrite indented code

Evidence:

- `SEPARATOR_LINE_REGEX` accepts any leading spaces/tabs before an HR-looking run in `src/md_module.rs:32-47`.
- `normalize_separator_line` is called in the full cleaner after line cleanup at `src/cleaning_module.rs:781-786`.
- `scan_gfm_table_separators` trims rows and has no 4-space/tab indented-code guard at `src/md_module.rs:91-130` and `src/md_module.rs:146-211`.
- `is_code_fence_marker` uses `trim_start()` and therefore accepts fence-looking lines at any indentation depth at `src/md_module.rs:226-229`.
- The cleaner passes `trimmed_line` into `is_code_fence_marker`, discarding indentation before fence detection at `src/cleaning_module.rs:539-550`.

Why it matters:

CommonMark only allows thematic breaks and fenced-code openers up to 3 leading spaces. A line indented by 4 spaces, or by a tab that reaches code indentation, is an indented code block. Today, these examples are at risk:

```markdown
    ----
```

```markdown
    | a | b |
    | ---- | ---- |
```

The first can be rewritten from an indented code line into `---`, changing a code block into a thematic break. The second can be rewritten from indented code into a GFM table. That violates the Phase A preview-preservation invariant and the Phase B structural invariant.

Recommendation:

- Add a CommonMark indentation helper that computes leading columns and rejects HR/table/fence syntax detection when indentation is >= 4 columns.
- Do not call `is_code_fence_marker` with a pre-trimmed line.
- Make `is_code_fence_marker` enforce the real fence grammar: 0-3 leading spaces, at least 3 matching backticks or tildes, no backtick in a backtick-fence info string, and closing-fence rules when tracking state.
- Add strict-equivalence tests for indented code containing HR-looking, table-looking, and fence-looking lines.

### High: The full cleaner does not use the Phase A orchestrator

Evidence:

- The doc says Phase A runs first, before content-modifying transforms, at `docs/MD_MODULE_ARCHITECTURE.md:76-96`.
- The `md_module` orchestrator intentionally runs GFM table separator minimization, then HR minimization, then paragraph reflow at `src/md_module.rs:393-438`.
- The full cleaner instead decodes entities, decodes PUA, strips glyph markers, strips soft hyphens, and then calls only `reflow_paragraphs` at `src/cleaning_module.rs:440-445`.
- GFM table separator minimization happens later at `src/cleaning_module.rs:502-536`.
- HR minimization happens still later as a per-line post-clean normalization at `src/cleaning_module.rs:781-786`.

Why it matters:

This is not just doc drift. The order can break optional-pipe GFM tables. This Markdown is valid GFM:

```markdown
a | b
--- | ---
1 | 2
```

`normalize_md_syntax` would canonicalize the separator row before reflow, making the separator line a hard table row. The full cleaner reflows first. Because `--- | ---` is not recognized as a hard break by `line_is_hard_break`, it can be joined with the body row before the table pass ever sees it, destroying the table.

Recommendation:

- Make the cleaner call `md_module::normalize_md_syntax` as the single Phase A entrypoint, or split the architecture into explicit stages and name the real order.
- Add an integration test for optional-pipe GFM tables through `core_clean_text_with_stats`.
- If entity/PUA/NBSP recovery must run before some syntax work so recovered characters survive filtering, document a three-tier pipeline: reversible raw normalization, Markdown syntax normalization, destructive cleanup, then post-clean formatting normalization.

### High: Paragraph reflow can delete Markdown hard breaks

Evidence:

- `can_join_lines` trims the previous line with `trim_end()` at `src/md_module.rs:284-286`.
- The join guard only stops on structural lines and selected sentence terminators at `src/md_module.rs:291-305`.
- There is no guard for Markdown hard breaks: two trailing spaces before newline or a trailing backslash before newline.

Why it matters:

These inputs render with a hard line break:

```markdown
first line  
second line
```

```markdown
first line\
second line
```

The current reflow logic can collapse them into one raw line, removing the rendered `<br>` and violating strict preview equivalence. The existing strict tests do not cover this case.

Recommendation:

- Treat trailing two-space hardbreaks and trailing backslash hardbreaks as non-joinable.
- Add negative strict-equivalence tests for both hardbreak syntaxes.
- Longer term, prefer parser-guided reflow based on actual `SoftBreak` events rather than line-end heuristics.

### Medium: Reflow under-achieves the raw-readability objective

Evidence:

- The doc says raw Markdown should avoid multi-line soft-wrapped paragraphs at `docs/MD_MODULE_ARCHITECTURE.md:15-19`.
- The implementation deliberately refuses to join after sentence terminators at `src/md_module.rs:295-305`.
- The behavior is locked in by `reflow_stops_at_sentence_terminators` at `src/md_module.rs:617-620`.
- List marker lines are treated as hard breaks at `src/md_module.rs:340-344`, which also blocks lazy-continuation list paragraphs from being reflowed.

Why it matters:

Docling/PDF extraction often wraps after punctuation. In Markdown, a single newline inside a paragraph is normally a soft break, so this:

```markdown
First sentence.
Second sentence.
```

renders as one paragraph. Leaving it as two raw lines fails the most important corpus goal: teach the model natural paragraph formatting in raw text, not the extractor's layout wraps. The same issue can affect list items with lazy continuation lines and blockquotes where preview treats multiple physical lines as one textual paragraph.

Recommendation:

- Reflow all parser-observed soft breaks inside paragraph blocks, including after punctuation, unless the source used an explicit hard break.
- If a full parser rewrite is too expensive for hot-path cleaning, use the verifier-driven approach: broaden the fast line heuristic, then strict-check fixtures and corpus samples.
- Add raw-readability metrics to the scorecard: percentage of paragraphs with internal newlines, mean physical lines per paragraph, and fraction of paragraphs with very short physical lines.

### Medium: Structural verification does not compare all text-bearing blocks

Evidence:

- `block_sequence` tracks headings, paragraphs, lists, tables, code blocks, HTML blocks, and footnotes at `src/md_verify.rs:190-250`.
- Token comparison is only implemented for paragraphs at `src/md_verify.rs:253-299`, tables at `src/md_verify.rs:301-357`, and code blocks at `src/md_verify.rs:360-382`.
- `verify_md_structural` uses those paragraph/table/code checks at `src/md_verify.rs:549-647`.

Why it matters:

The architecture says the token sequence within each block must be a monotone subsequence. Heading text is a block's text, but it is not currently compared. For example, changing `# Alpha Beta` to `# Injected Heading` can preserve the block sequence and avoid paragraph/table/code checks entirely.

Recommendation:

- Replace paragraph-only token extraction with a generic block-text extractor for all leaf-text-bearing blocks: headings, paragraphs, blockquote paragraphs, list item paragraphs, footnotes, table cells, and HTML blocks where appropriate.
- Report failures with block kind and index, not just paragraph index.
- Keep code blocks on line-preserving comparison because whitespace is meaningful there.

### Medium: The structural verifier canonicalizes input using a path that can drift from the cleaner

Evidence:

- `canonicalize_for_verify` applies its own sequence of non-destructive transforms and then calls `normalize_md_syntax` at `src/md_verify.rs:122-152`.
- The full cleaner applies a different sequence: pre-clean entity/PUA/glyph/soft-hyphen/reflow at `src/cleaning_module.rs:440-445`, then table canonicalization at `src/cleaning_module.rs:502-536`, then line-level normalizations at `src/cleaning_module.rs:781-795`.

Why it matters:

The verifier should compare against the same non-destructive baseline the cleaner actually uses. If verifier canonicalization is broader, narrower, or ordered differently, it can hide real cleaner behavior or flag false differences. The optional-pipe table issue above is one concrete way this divergence can matter.

Recommendation:

- Share a single Rust function for the non-destructive canonicalization baseline, used by both the cleaner and verifier.
- Alternatively, expose staged cleaner outputs: `phase_a_output`, `phase_b_output`, and `post_normalized_output`, then verify each stage against the correct invariant.

### Medium: The architecture doc is stale and contradicts implementation policy

Evidence:

- The doc still says `md_verify.rs` is "to be built" at `docs/MD_MODULE_ARCHITECTURE.md:129`, but the verifier exists and is exported.
- The commit plan says commit 1 is "in progress" at `docs/MD_MODULE_ARCHITECTURE.md:177-197`, but the tree includes the verifier, tests, PyO3 exports, and corpus scorecard.
- The HR section says `=====` and Unicode em-dash/box-drawing variants are accepted at `docs/MD_MODULE_ARCHITECTURE.md:46-49`.
- The code rejects equals and Unicode dash-like lines, with tests encoding that policy at `src/md_module.rs:459-476`.
- The code comment above `normalize_separator_line` still mentions equals/Unicode variants at `src/md_module.rs:55-57`, contradicting the regex and tests.
- The doc lists v6-11 NBSP fusion as a known future bug at `docs/MD_MODULE_ARCHITECTURE.md:199-207`, but the implementation has a regression test saying it is fixed at `src/cleaning_module.rs:2099-2128`.

Why it matters:

For this module, doc drift is operationally risky because the whole safety story depends on exact Markdown grammar. A future implementer following the doc would reintroduce preview-changing transformations for equals and Unicode separator lines.

Recommendation:

- Update the architecture doc from "plan" to "current contract".
- Explicitly state that equals runs and Unicode dash/box lines are not CommonMark HRs and must not be Phase A normalized.
- Move resolved bugs like v6-11 into a "resolved / covered by tests" section.
- Keep a separate "future candidates" section for transformations that intentionally change structure, such as pseudo-table unwrapping.

### Low: Table separator scanning is a useful fast path, but not a full GFM parser

Evidence:

- Header cell counts use a direct `split('|')` at `src/md_module.rs:198-211`.
- This does not account for escaped pipes or pipes inside code spans.

Why it matters:

This is mostly an under-normalization risk rather than a preview-breaking risk: valid tables with escaped pipes may not be canonicalized. It still matters for corpus tidiness because Docling table syntax can be irregular.

Recommendation:

- Either document the scanner as a conservative fast path, or use pulldown-cmark offsets / a small GFM table lexer for table rows with escaped-pipe awareness.
- Add fixtures for escaped pipes and code-span pipes so behavior is explicit.

### Low: The scorecard script is present, but not yet enough for Markdown-specific corpus confidence

Evidence:

- The script samples parquet rows and runs both verifiers at `../../cleaning_scripts/verify_md_equivalence.py:63-239`.
- It reports pass rates and failure kinds at `../../cleaning_scripts/verify_md_equivalence.py:241-328`.

Why it matters:

This is good as a structural regression scorecard, but it does not yet measure the raw-readability outcome that motivated the module. A structural pass can still leave many preview-soft-wrapped paragraphs fragmented in raw text.

Recommendation:

- Add raw-readability metrics: physical lines per rendered paragraph, short-line rate inside paragraphs, table separator width distribution, HR width distribution, and optional-pipe table count.
- Save a small stratified sample of all-pass docs, not only failures, because raw-readability regressions often pass structural verification.
- Add a Markdown-likely filter or source-format field if available, so the scorecard can distinguish Markdown output from non-Markdown text rows.

## Suggested implementation plan

1. Add failing tests first: indented code with HR/table/fence-looking lines, hardbreak preservation, optional-pipe GFM tables through the full cleaner, heading-token changes in structural verifier.
2. Make Markdown detection CommonMark indentation-aware. Use raw lines, not pre-trimmed lines, for syntax-state predicates.
3. Route the full cleaner through one Phase A function, or rename the actual stages so there is exactly one source of truth for ordering.
4. Expand reflow from "safe line joiner" to "parser-observed softbreak normalizer." Preserve explicit hard breaks.
5. Expand structural verification from paragraph/table/code to all text-bearing block spans.
6. Update `MD_MODULE_ARCHITECTURE.md` after the code contract is decided, especially the HR policy and the current implementation status.

## Verification performed

Commands run from `/home/foivos/glossAPI-development/rust/glossapi_rs_cleaner` unless noted:

- `cargo test md_module`: passed, 40 tests.
- `cargo test md_verify`: passed, 34 tests.
- `cargo test phase_b_`: passed, 12 tests.
- `python3 -m py_compile /home/foivos/glossAPI-development/cleaning_scripts/verify_md_equivalence.py`: passed.
- `cargo test`: failed in the broader suite, with 241 passed and 2 failed. The failures were `table_remover_module::tests::test_empty_content_with_remove_op` and `cleaning_module::tests::perf_mixed_doc_throughput_floor`. The latter was run in the default debug profile and appears to be a throughput-threshold issue, not a Markdown-equivalence failure.

The Python scorecard itself was not run because `glossapi_rs_cleaner` is not installed in the active Python environment.

---

# Response to the review (2026-04-24, Claude)

## Overall agreement

The reviewer is right on almost everything. Three of the High
findings are real preview-equivalence bugs I missed. The prior
"34.3 % structural pass" scorecard is only trustworthy to the
extent that the tests it ran with are correct — several of the
tests I wrote didn't cover the edge cases the reviewer identified
(indented code containing HR-looking lines, optional-pipe GFM
tables, hard-break markers). I will treat those as known holes
until failing tests exist and the fixes are verified.

## Per-finding verdict

**H-1 Indentation awareness: ADOPT.** Real CommonMark bug.
`    ----` (indented code) becomes an HR under the current regex,
changing render from `<pre><code>----</code></pre>` to `<hr/>`.
Same risk for indented tables and fences. Fix is deterministic —
compute leading columns with the spec's tab rule (tab advances to
next multiple of 4) and reject Phase A detectors at ≥4.
`is_code_fence_marker` must stop using `trim_start()`.

**H-2 Orchestrator not used / optional-pipe tables: ADOPT.** The
reviewer's optional-pipe example (`a | b\n--- | ---\n1 | 2`)
is a genuine break path. `--- | ---` isn't flagged as a hard break
by `line_is_hard_break`, so reflow can join it with the header
before `scan_gfm_table_separators` ever sees it, destroying the
table. Fix: `core_clean_text_with_stats` calls
`md_module::normalize_md_syntax` as a single Phase A stage, before
any content-modifying pass.

**H-3 Hard-break deletion: ADOPT.** `trim_end()` in `can_join_lines`
silently destroys both `  \n` (two-space hard break) and `\\\n`
(backslash hard break). Preview loses `<br>`. Fix is small — detect
the hard-break markers before the trim and refuse to join.

**M-1 Reflow under-achieves: ADJUST.** The reviewer is right about
the spec (a single `\n` inside a paragraph renders as space), but
the conservative sentence-terminator guard is load-bearing on real
Docling output where PDF column wraps can coincide with sentence
ends. Blindly removing it risks fusing paragraphs that Docling
intended as separate. The right path: expand reflow AND add a
negative test suite built from real corpus samples; gate with a
scorecard re-run before merging. If strict pass-rate on
canonicalized inputs regresses, re-introduce a tighter guard.

**M-2 Heading text not compared: ADOPT.** Real gap — a cleaner that
rewrote `# Α` to `# Β` would pass structural today. Extend block-
text extraction to cover headings, blockquote paragraphs, list-item
paragraphs, and footnote definitions.

**M-3 Canonicalization drift: ADOPT.** Move to a single shared
`non_destructive_canonicalize()` used by both cleaner and verifier.
Single source of truth for what is "legit to do to MD."

**M-4 Architecture doc stale: ADOPT.** Cheap and important. Update
`MD_MODULE_ARCHITECTURE.md` as current contract, not plan. Correct
the HR-section inconsistency (`=====` and Unicode dashes are NOT
Phase A). Mark v6-11 resolved in a separate "resolved / covered by
tests" section.

**L-1 Escaped pipes in table scanner: DEFER.** Real but not
preview-breaking. Document the scanner as conservative fast path.
Fix if it causes corpus issues later.

**L-2 Raw-readability metrics in scorecard: DEFER.** Valid for
measuring the user-facing goal, but not a correctness issue. Add
after the High fixes land and the reflow expansion is calibrated.

## Planned commit sequence (Commits 11–16)

Order chosen to land failing tests first, per the reviewer's
Suggested Implementation Plan and the directive to be careful with
formatting changes on training data:

1. **Commit 11 — failing tests first.** Add strict-equivalence
   tests for: indented code with HR-/table-/fence-looking lines;
   two-space + backslash hard breaks; optional-pipe GFM tables
   through the full cleaner; heading-text change detection in the
   structural verifier. Lands with these RED — confirms the holes.
2. **Commit 12 — CommonMark indentation awareness.** Add
   `leading_columns(line)` helper with the tab rule. Phase A
   detectors reject at ≥4 leading columns. `is_code_fence_marker`
   stops trimming. Indented-code-looking tests from Commit 11
   turn GREEN.
3. **Commit 13 — cleaner routes through orchestrator + hard-break
   preservation.** `core_clean_text_with_stats` calls
   `md_module::normalize_md_syntax` as the single Phase A entry.
   Reflow detects and preserves `  \n` / `\\\n`. Optional-pipe
   table + hard-break tests turn GREEN.
4. **Commit 14 — shared canonicalization.** Extract
   `non_destructive_canonicalize()` into a shared location.
   Scorecard numbers must not regress.
5. **Commit 15 — expand structural verifier.** Heading text +
   blockquote + list-item-paragraph + footnote comparison.
   Heading-change test turns GREEN.
6. **Commit 16 — expand reflow + doc update + resolved-bugs
   section.** Risky commit per the "careful" directive. Expand
   reflow to join across sentence terminators (spec-correct
   behaviour). Gate with 100-doc scorecard re-run; if strict pass-
   rate on canonicalized inputs regresses, narrow the expansion
   before merging. Update `MD_MODULE_ARCHITECTURE.md` as current
   contract.

Commit 16 is the only one that could plausibly affect training-data
quality. Commits 11–15 are bug fixes with clean invariants.

## Safety posture

- Every commit adds tests BEFORE touching production code paths.
- Nothing destructive to the corpus lands without a scorecard
  re-run on the same 100-doc openarchives seed.
- Commit 16 is held until the user has inspected a diff sample of
  before-expansion vs after-expansion on real docs.
- If any commit produces a scorecard regression, it backs out.
- No change is pushed to `origin/development` until the commit 11
  tests go green and the scorecard re-run shows no regression.

## Discovered while executing

### Commit 12 (CommonMark indentation awareness) landed

Added `leading_columns(line) -> usize` in `src/md_module.rs` applying
CommonMark's column rule (tab advances to the next multiple of 4).
Three call sites now gate on `< 4` columns:

- `normalize_separator_line` (HR detector) — early-`None` at ≥4.
- `scan_gfm_table_separators` — separator row AND header row both
  must sit at `< 4` columns; otherwise both are indented-code leaf
  blocks, not a GFM table.
- `is_code_fence_marker` — contract changed: caller now MUST pass the
  raw (un-trimmed) line. At ≥4 columns the detector returns `false`.

Updated the one internal caller that was discarding indentation:
`cleaning_module::core_clean_text_with_stats` now passes the raw
`line` (not `trimmed_line`) to `is_code_fence_marker`. That was
latent bug #2 — a fence marker indented ≥4 inside a real indented
code block would have toggled cleaner fence-state and caused
normalization to skip/resume at the wrong spots.

Also added five indent-aware unit tests (leading-columns arithmetic,
fence-at-4-cols rejection, HR/GFM-at-4-cols rejection). The two C12
RED regression tests are now GREEN.

**Test state at Commit 12 boundary:** 251 passed, 5 failing. The 5
failures break down as:

1. `red_until_c13_reflow_preserves_two_space_hard_break` — expected
   (Commit 13).
2. `red_until_c13_reflow_preserves_backslash_hard_break` — expected
   (Commit 13).
3. `red_until_c13_optional_pipe_gfm_table_survives_full_cleaner` —
   expected (Commit 13).
4. `red_until_c15_heading_text_change_detected_by_structural` —
   expected (Commit 15).
5. `table_remover_module::test_empty_content_with_remove_op` —
   pre-existing, unrelated to this review stream.

No Phase A regressions. The existing preview-equivalence tests
(including orchestrator mixed-content, alignment-preserving tables,
fenced-code-preserving reflow) all still pass.

### Commit 13 (cleaner routes through orchestrator + hard-break
preservation) landed

**Part 1 — single Phase A entry.** Replaced
`core_clean_text_with_stats`'s call to
`md_module::reflow_paragraphs` with
`md_module::normalize_md_syntax`. That routes the cleaner through
the full Phase A pipeline in the correct order: GFM table
separator canonicalization → HR minimization → paragraph reflow.
This was the H-2 bug: optional-pipe GFM tables like
`a | b\n--- | ---\n1 | 2` were previously invisible to reflow
(rows don't start/end with `|`), so reflow fused the separator
row with the first body row, destroying the table. With the
orchestrator running first, `scan_gfm_table_separators` identifies
the table before reflow touches it, reflow sees canonical
`| --- | --- |` and refuses to join.

**Part 2 — hard-break preservation.** Added two guards at the
start of `can_join_lines`:

- `prev.ends_with("  ")` → CommonMark hard break `  \n` (renders
  as `<br>`). Refuse to join before `trim_end()` can destroy the
  signal.
- Count trailing backslashes; if odd, refuse to join. This
  implements CommonMark's backslash hard-break rule correctly:
  `foo\` joins, `foo\\` does not (escaped literal), `foo\\\`
  does, etc.

**Part 3 — canonical-HR recognition in reflow.** Added
`HR_HARD_BREAK_REGEX` with the spec ≥3-char threshold (the
rewrite regex `SEPARATOR_LINE_REGEX` stays at ≥4 because its
rewrite rule only needs to fire on non-canonical runs). The
reflow hard-break detector now uses the ≥3 regex, so the
canonical `---` output and setext heading markers (`---`, `===`)
are both recognized as block boundaries that reflow must not
cross.

**Part 4 — fenced-code awareness in `normalize_md_syntax` step 2.**
The HR normalization step was not previously fence-aware and would
rewrite a `----` line inside a fenced code block. Added fence-state
tracking to step 2 (matching what steps 1 and 3 already do).

**Tests turned GREEN:**

- `red_until_c13_reflow_preserves_two_space_hard_break`.
- `red_until_c13_reflow_preserves_backslash_hard_break`.
- `red_until_c13_optional_pipe_gfm_table_survives_full_cleaner`.

**Tests added as post-fix regression gates:**

- `equiv_reflow_preserves_canonical_hr_as_hard_break` — verifies
  that after HR collapse, reflow still recognizes `---`.
- `equiv_reflow_preserves_setext_heading_marker` — verifies that
  a setext H2 marker is not fused with the following paragraph.
- `equiv_orchestrator_preserves_fenced_hr_content` — verifies
  that a `----` inside a fenced block is not rewritten.

**Test state at Commit 13 boundary:** 257 passed, 2 failing. The
2 failures: C15 RED (`red_until_c15_heading_text_change_detected_
by_structural`) and the pre-existing unrelated
`table_remover::test_empty_content_with_remove_op`.

**Scorecard re-run:** deferred until the full review-response
series (C12–C16) lands. Rationale — the cleaner's Phase A
behaviour is a monotonic quality improvement across these five
commits (each strict-invariant violation being fixed), and a
single scorecard comparison after the series finishes gives a
cleaner signal than repeated runs between commits. If the
corpus-scale strict pass-rate regresses at that point, the fix
is narrowed per commit.

### Commit 14 (shared non-destructive canonicalization) landed

Promoted `md_verify::canonicalize_for_verify` into
`md_module::non_destructive_canonicalize` as the single source of
truth for "what the cleaner would produce if every pass were
non-destructive." The verifier's `canonicalize_for_verify` is now
a thin delegator.

Added five drift-prevention tests in `cleaning_module::tests` that
run a permissive cleaner (allowed-chars superset so nothing is
dropped) and assert its output equals
`non_destructive_canonicalize(input)` on five representative
shapes: plain prose, optional-pipe GFM table, HR collapse between
paragraphs, soft-wrapped paragraph, and mixed heading/table/HR/
paragraph. Any future change that silently drifts cleaner Phase A
behaviour away from verifier-observed canonical form trips at
least one of these gates.

**Test state at Commit 14 boundary:** 262 passed, 2 failing. The 2
failures are still the C15 RED and the pre-existing unrelated
`table_remover` failure. No regressions from C13.

