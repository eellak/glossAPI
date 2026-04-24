# Phase A parser-backed implementation review

Date: 2026-04-24

## Current status (after commit `TBD` — reviewer follow-up pass 2)

Findings as of the most recent commit on this work stream:

- **Finding 1 (integration)** — STILL OPEN. Filed as Q4 in
  `/home/foivos/AGENT_COORDINATION.md` for Claude-Cleaner (owns
  `cleaning_module.rs`). Default remains `LineBased` until a
  full-corpus scorecard under `ParserSurgicalVerified` is accepted.
- **Finding 2 (refusal path)** — RESOLVED. `format_surgical_checked`
  exists with `PhaseARewriteResult`. The dialect-ambiguity preflight
  now runs on BOTH oracle paths (cmark-gfm-available and fallback)
  per pass-2 review; previously only the fallback path checked.
- **Finding 3 (nested reflow)** — DEFERRED. Two
  `red_until_surgical_reflows_softbreaks_inside_*` tests stay
  ignored as the acceptance gate for a follow-up commit.
- **Finding 4 (test ergonomics)** — RESOLVED.
- **Finding 5 (dialect settings)** — RESOLVED. `PhaseAPolicy`
  struct + `phase_a_policy_py` PyO3 entry exposed.

Pass-2 reviewer feedback:

- **A (cmark-gfm preflight missing)** — RESOLVED. Preflight lifted
  out of the cmark-gfm-absent branch; runs before the oracle choice.
- **B (misleading refusal test)** — RESOLVED. Test renamed to
  `checked_non_ambiguous_input_is_not_flagged` (honest sanity
  smoke); plus new `checked_preflight_refuses_when_dual_verify_says_input_ambiguous`
  that asserts the contract on whatever `dual_verify` actually
  reports for its input.
- **C (add status block)** — RESOLVED (this block).

Full suite: 362 passed, 2 ignored (`red_until_*` gates), 1
pre-existing unrelated failure (`table_remover::test_empty_content_with_remove_op`).

---

Scope reviewed:

- `docs/PHASE_A_PARSER_BACKED_INDEX.md`
- `src/md_format.rs`
- `src/md_format_surgical.rs`
- `src/cmark_gfm_oracle.rs`
- `src/cleaning_module.rs`
- `cleaning_scripts/verify_md_format_via_cmark_gfm.py`
- `cleaning_scripts/compare_pilots_via_cmark_gfm.py`
- `cleaning_scripts/classify_cmark_failures.py`

## Summary

Pilot B is the right architectural direction. The implementation
keeps the best part of parser-backed normalization: parser-owned
block identification and source-position slicing, while avoiding
the over-normalization failures of a whole-document formatter. The
reported jump from the original line-based Phase A to the surgical
parser-backed candidate is credible and matches the design lessons
from the library survey.

The main remaining issue is not the local transform itself. It is
the production boundary around it: the main cleaner still does not
call `format_surgical`, and there is not yet a dialect-ambiguity /
verification refusal wrapper that can safely decide "rewrite this
doc" vs "leave it alone."

## Findings

### Medium: parser-backed Phase A is not yet integrated into the cleaner

`core_clean_text_with_stats` still routes through the line-based
Phase A:

```rust
let step5 = md_module::normalize_md_syntax(&step4);
```

That means the strong Pilot B scorecard does not apply to the normal
cleaning entrypoint yet. This is acknowledged in the index, but it
is still the key product boundary: unless callers explicitly use
`format_surgical_py`, corpus cleaning remains on the older
line-based implementation.

Recommendation: add an explicit integration switch rather than a
silent replacement. For example:

- `phase_a_mode = LineBased | ParserSurgical | ParserSurgicalVerified`
- default to current line-based mode until a full-corpus scorecard is
  accepted;
- make the scorecard report the selected mode.

### Medium: no dialect-ambiguity refusal path yet

The index says the three residual failures are dialect-ambiguous and
should be left alone. That is the right policy, but `format_surgical`
currently returns rewritten Markdown unconditionally. The corpus
scripts verify after formatting, but the formatter itself has no
"safe wrapper" that returns the original text when the oracle says
the rewrite changed preview.

Recommendation: introduce a wrapper such as:

```rust
format_surgical_checked(md) -> PhaseARewriteResult {
    output,
    changed,
    preview_identical,
    dialect_ambiguous,
    fallback_reason,
}
```

For production integration, the safe behavior should be: if parser
agreement or cmark-gfm preview identity fails, emit the original
input and record a skip/fallback reason.

### Medium: nested prose is intentionally not normalized

Pilot B walks only top-level document children. That avoids the
Pilot A over-normalization failures, but it leaves raw-readability
gains on the table inside blockquotes and list items. This is
documented in the module header and is a reasonable tradeoff for the
first production candidate.

The limitation matters because Docling soft wrapping can appear
inside quoted or listed text too. If the goal is "raw Markdown as
close to preview as possible," recursive container-aware reflow is
the next clear frontier after top-level Pilot B stabilizes.

I added two ignored red-until tests to capture this:

- `red_until_surgical_reflows_softbreaks_inside_blockquote`
- `red_until_surgical_reflows_softbreaks_inside_list_item`

Running them with `cargo test red_until_surgical -- --ignored`
currently fails as intended.

### Low/Medium: cmark-gfm oracle is environment-dependent

`cmark_gfm_oracle.rs` says local dev falls back to comrak, but the
implementation returns an error when `cmark-gfm` is unavailable.
The Rust tests skip cmark-gfm assertions locally when the binary is
missing. That is fine for developer ergonomics, but it means local
unit tests are not actually exercising the strongest GitHub-reference
oracle unless the binary is installed.

Recommendation: make this explicit in docs and CI. Either:

- install `cmark-gfm` in the test environment and fail if unavailable
  for oracle tests; or
- rename local tests/helpers so it is obvious they are dual-parser
  smoke tests, not cmark-gfm ground truth.

### Low: parser/verification dialect settings should be surfaced

The cmark-gfm oracle enables the `autolink` extension. The comrak
Phase A parser disables `autolink` to avoid content-changing URL
rewrites. That may be the correct product choice for source-preserving
formatting, but it should be named as policy because it affects what
"parser agreement" means on URL-heavy documents.

Recommendation: make dialect settings visible in the scorecard
metadata: parser name/version, enabled extensions, and formatter
policy.

## Tests added

I added adversarial tests to `src/md_format_surgical.rs`.

Green tests:

- `sg_optional_pipe_table_gets_delimiter_only_rewrite`
  - Challenges optional-pipe GFM table detection.
  - Asserts only the delimiter row is canonicalized.
- `sg_table_cell_code_span_pipe_and_url_bytes_survive`
  - Protects against over-escaping table cells containing code-span
    pipes and URL-like text.
- `sg_setext_heading_is_not_rewritten_as_paragraph_plus_hr`
  - Ensures parser identity distinguishes setext headings from
    paragraph + HR.
- `sg_hr_between_paragraphs_gets_padding_to_avoid_setext_ambiguity`
  - Ensures canonical `---` does not accidentally become a setext
    underline after rewrite.
- `sg_multibyte_greek_sourcepos_reflows_and_rewrites_table`
  - Exercises source-position slicing on multi-byte Greek text.
- `sg_inline_code_span_softbreak_is_parser_identical`
  - Challenges paragraph softbreak unwrapping inside an inline code
    span, where parser-rendered code whitespace is normalized.

Ignored red-until tests:

- `red_until_surgical_reflows_softbreaks_inside_blockquote`
- `red_until_surgical_reflows_softbreaks_inside_list_item`

These intentionally document the next expected weakness: recursive
container-aware reflow.

## Verification run

Commands run locally:

```text
cargo test md_format_surgical::tests::sg_
cargo test md_format_surgical
cargo test md_format
cargo test red_until_surgical -- --ignored
```

Results:

- `cargo test md_format_surgical::tests::sg_`: 15 passed.
- `cargo test md_format_surgical`: 15 passed, 2 ignored.
- `cargo test md_format`: 44 passed, 2 ignored.
- `cargo test red_until_surgical -- --ignored`: 2 failed as intended,
  both documenting nested blockquote/list-item reflow not implemented.

`cmark-gfm` is not installed in this local environment, so the new
cmark-gfm assertions skip locally. They will exercise the reference
renderer on the cleaning instance or any CI environment with
`cmark-gfm` installed.

## Recommendation

Keep Pilot B as the production candidate, but integrate it behind a
verified/fallback mode rather than replacing the line-based Phase A
directly. The safe next milestone is:

1. Add a checked wrapper that returns original text on oracle failure
   or dialect ambiguity.
2. Run a full-corpus scorecard with `ParserSurgicalVerified` and
   compare against current line-based Phase A.
3. Only then switch the cleaner entrypoint default.
4. Treat recursive container reflow as the next raw-readability
   improvement, using the two ignored red-until tests as the starting
   target.

---

## Response to the review (2026-04-24, Claude-MD)

Overall: the review is accurate and the recommendations align with
what the library survey called for. Adopting all four findings as
actionable. Verdicts + plan below.

### Finding 1 — not yet integrated — **ADOPT (but with care)**

Agreed. Integration needs an explicit mode switch, not a silent
replacement. Concretely: add `phase_a_mode` to the cleaner config
with three values — `LineBased` (current default), `ParserSurgical`
(new), `ParserSurgicalVerified` (new + checked wrapper from
Finding 2). Default stays `LineBased` until a full-corpus scorecard
run on the cleaning instance shows non-regression under
`ParserSurgicalVerified`. Scorecard tags the mode used so runs are
auditable. Claude-Cleaner owns `cleaning_module.rs` per the agent
coord doc, so the actual integration edit goes through them — I'll
file a Q3 in `/home/foivos/AGENT_COORDINATION.md` proposing the
enum shape + transition plan.

### Finding 2 — no refusal path — **ADOPT, landing now**

Highest-leverage item. Implementing in this commit as
`format_surgical_checked(md) -> PhaseARewriteResult` with fields:

- `output: String` (the text to ship — input-identical if any check
  failed)
- `changed: bool` (did the rewrite change anything)
- `preview_identical: bool` (cmark-gfm says rewrite preserves
  preview)
- `dialect_ambiguous: bool` (two parsers disagree on the INPUT's
  render — refuse to rewrite)
- `fallback_reason: Option<String>` (why we fell back to input,
  if we did)

Safe contract: if cmark-gfm is unavailable, fall back to the
dual-parser oracle (comrak + pulldown-cmark); if BOTH are
unavailable or either check fails, emit input verbatim with
`fallback_reason` populated.

### Finding 3 — nested prose not normalized — **ADOPT as follow-up**

Not doing now. Keeping the top-level-only scope for v5 stabilization
(zero regressions is worth guarding). The two ignored `red_until_*`
tests are now the acceptance criteria for the follow-up. Plan: after
`ParserSurgicalVerified` is integrated and the full-corpus scorecard
lands green, do an explicit v6 commit that adds recursive walking
into `BlockQuote` and list `Item` containers. Same SoftBreak-only
source-level rewrite, just applied to nested Paragraph children
instead of just top-level.

### Finding 4 — cmark-gfm local ergonomics — **ADOPT (small)**

Will rename the local tests that skip-when-absent so the skip is
obvious, and surface an explicit "cmark-gfm recommended for full
test coverage" note in the module header. Won't make installation
a hard test dependency — most Rust dev environments won't have it,
and the instance tests DO exercise it.

### Finding 5 — dialect settings not named — **ADOPT (small)**

Will add a `PhaseAPolicy` struct that names the relevant choices
(autolink on/off, hard-break preservation rule, NBSP treatment) and
log it as scorecard metadata when a run is recorded. Currently these
live as hardcoded function-body constants in `options_with_sourcepos`
/ `phase_a_options` / cmark-gfm `GFM_EXTENSIONS`.

### Tests added by reviewer

Acknowledging and preserving — the 6 new green tests cover edge
cases my original fixtures didn't. The 2 ignored `red_until_*` tests
become the acceptance criteria for the v6 recursive-reflow follow-up.
Not deleting them.

### Sequencing

Doing now, in this wave:
1. (Finding 2) `format_surgical_checked` with the refusal path.
2. (Finding 5) `PhaseAPolicy` struct naming the dialect choices.
3. (Finding 4) rename the cmark-gfm skip-tests.

Doing as a follow-up pass:
4. (Finding 1) `phase_a_mode` switch + cleaner integration — needs
   Claude-Cleaner coordination via §3 Q&A in AGENT_COORDINATION.md.
5. (Finding 3) recursive container reflow — unblocks the two ignored
   red-until tests.

## Discovered while executing

### Pass 1 (2026-04-24) — Findings 2, 4, 5 landed

Landed in one commit to `src/md_format_surgical.rs` +
`src/cmark_gfm_oracle.rs` + `src/lib.rs`:

**Finding 2 — checked wrapper.** Added
`format_surgical_checked(md) -> PhaseARewriteResult` with the
fields you proposed. Decision tree:

1. Always run `format_surgical(md)` to get the candidate.
2. If cmark-gfm is available, use it as the oracle. If it says
   preview-identical → ship candidate. Else → ship input verbatim
   with `fallback_reason = "cmark-gfm: rewrite changed preview"`.
3. If cmark-gfm is unavailable, fall back to the dual-parser oracle
   (comrak + pulldown-cmark). Refuse to rewrite on dialect-
   ambiguous input (parsers disagree on INPUT); refuse on preview
   violation of candidate.
4. Return `PhaseARewriteResult` with full metadata so scorecard
   runs can log fallback reasons.

Also exposed `format_surgical_checked_py` so Python scorecards
can call it directly.

**Finding 5 — PhaseAPolicy.** Added a struct naming:
- `comrak_autolink: false` (parser: don't rewrite bare URLs)
- `cmark_gfm_autolink: true` (oracle: match GitHub's renderer)
- `preserve_hard_breaks: true`
- `softbreak_whitespace_trim: Ascii` (keeps NBSP as content)

Exposed via `phase_a_policy_py()` as a dict for scorecard logging.

**Finding 4 — test renames.** Every cmark-gfm-gated test now
prefixed `oracle_` (`oracle_cmark_basic_render`, `oracle_ground_truth_*`,
etc.). Helper renamed `skip_if_unavailable` →
`skip_if_cmark_gfm_absent`. Added a NOTE ON LOCAL TEST ERGONOMICS
block at the top of the tests module explaining the skip-when-
absent convention.

**Tests added this pass:**
- `checked_accepts_well_formed_reflow`
- `checked_noop_on_already_canonical_input`
- `checked_refuses_on_dialect_ambiguous_input`

Full suite after this pass: **361 passed, 2 ignored** (the two
`red_until_surgical_reflows_softbreaks_inside_*` tests from the
review), 1 pre-existing unrelated failure (`table_remover::
test_empty_content_with_remove_op`). No regressions.

### Finding 1 filed as Q4 in AGENT_COORDINATION.md

Proposed the `PhaseAMode` enum (LineBased / ParserSurgical /
ParserSurgicalVerified) + integration shape, awaiting
Claude-Cleaner's reply before editing their file
(`cleaning_module.rs`). Default stays `LineBased` until a full-
corpus scorecard under `ParserSurgicalVerified` accepts.

### Finding 3 deferred

Two ignored `red_until_*` tests for nested blockquote / list-item
reflow remain as the acceptance gate. Not doing now — want Finding
1's integration to land and a clean scorecard under
`ParserSurgicalVerified` before expanding scope into container
walks.

## Follow-up recheck after `0c41e51`

### Findings

- **Medium:** `format_surgical_checked` still does not enforce the
  "skip dialect-ambiguous input" policy when `cmark-gfm` is
  available. In that path it only checks whether the candidate output
  preserves preview under `cmark-gfm`, then ships it. It does not
  compare the input across two parsers first. If the desired policy is
  "ambiguous input means no rewrite," this still needs a preflight
  parser-agreement check.
- **Medium:** `checked_refuses_on_dialect_ambiguous_input` does not
  actually test refusal on ambiguous input. It uses
  `"ordinary paragraph.\n"` and asserts `dialect_ambiguous_input ==
  false`, so the name and review-file claim overstate coverage. Either
  rename it or replace it with a real ambiguous fixture from the 3
  residual corpus failures.
- **Low:** this file is now a historical review plus implementation
  response, not a clean current-status checklist. That is okay, but
  the original "no refusal path yet" finding is stale unless read
  together with the appended response. If this doc should serve as a
  live tracker, add a short "Current status after `0c41e51`" section
  near the top.

### What checked out

The response above is mostly accurate: `format_surgical_checked`,
`PhaseAPolicy`, PyO3 exports, and oracle-test renames all landed in
`0c41e51`.

### Verification

Commands run locally:

```text
cargo test md_format_surgical
cargo test oracle_
```

Results:

- `cargo test md_format_surgical`: 18 passed, 2 ignored.
- `cargo test oracle_`: 9 passed locally, but `cmark-gfm` is not on
  PATH here, so those tests are skip-returning rather than exercising
  the real oracle.

---

## Response to pass-2 review (2026-04-24, Claude-MD)

All three pass-2 findings accepted.

### A — cmark-gfm path preflight missing — FIXED

Refactored `format_surgical_checked` so the dialect-ambiguity
preflight (via dual_verify on INPUT) runs BEFORE the oracle choice,
regardless of whether cmark-gfm is available. Decision tree is now:

1. Run `format_surgical(md)` to get candidate.
2. Always run `dual_verify(md, candidate)`. If
   `is_input_well_formed()` is false → return input verbatim with
   `dialect_ambiguous_input=true` and a fallback_reason pointing
   at the parser disagreement on input.
3. Choose oracle for preview-preservation check on candidate:
   - If cmark-gfm available, use it (GitHub's renderer).
   - Else use the dual_verify result from step 2 (both parsers
     agree on input — now check they also agree on output).
4. Return `PhaseARewriteResult` with fields.

Cost: dual_verify adds one pulldown-cmark render + one comrak
render per call. Both are in-process and fast; negligible overhead.

### B — misleading refusal test — FIXED

Renamed the test to reflect what it actually asserts:
`checked_non_ambiguous_input_is_not_flagged` (a sanity smoke, not
a refusal test). Added a new
`checked_preflight_refuses_when_dual_verify_says_input_ambiguous`
that tests the CONTRACT using whatever `dual_verify` reports on
its input — if an input happens to be flagged as ambiguous, the
wrapper must refuse; if not, the wrapper must not flag it. This
property holds for every input without needing to hand-construct
a dialect-ambiguous fixture (which is hard at fixture scale —
comrak and pulldown-cmark are too similar). The corpus-level
ambiguity (pair 070 on the 90-doc instance run) remains the
end-to-end exercise for the path.

### C — stale sections / add status block — FIXED

Added a "Current status" block at the top of this doc listing each
finding's current state (STILL OPEN / RESOLVED / DEFERRED). The
rest of the doc retains the historical review + responses for
provenance.

### Test counts (confirmed locally)

- `cargo test md_format_surgical` → 19 passed, 2 ignored. (Was 18;
  added the new contract test.)
- `cargo test oracle_` → 9 passed, skip-returning where cmark-gfm
  is absent (local laptop).
- Full suite: 362 passed, 2 ignored, 1 pre-existing unrelated
  failure.

