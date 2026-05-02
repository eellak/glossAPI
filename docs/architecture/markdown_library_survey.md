# Markdown Library Survey

This document captures the design rationale behind the cleaner's parser-backed Phase A. It surveys established Markdown parsers, formatters, and renderers across Rust, C, JavaScript/TypeScript, Python, Go, and Pandoc/Haskell, then maps the lessons onto the choices that landed in the production cleaner.

The goal was never to replace the cleaner with one of these tools wholesale. The goal was to extract design lessons for Docling-produced Markdown corpus cleanup: make raw Markdown closer to rendered preview while preserving the rendered element graph.

The conclusions of this survey shipped as **Pilot B** — the parser-backed `format_surgical_checked` path that is now the production Phase A. See [OCR Cleaning Runtime](ocr_cleaning_runtime.md) for the runtime architecture that consumes this layer, and `rust/glossapi_rs_cleaner/src/md_format_surgical.rs` for the implementation.

## Sources consulted

- [Comrak](https://comrak.ee/) and [comrak docs.rs](https://docs.rs/comrak/latest/comrak/)
- [pulldown-cmark guide](https://pulldown-cmark.github.io/pulldown-cmark/)
- [cmark-gfm](https://github.com/github/cmark-gfm)
- [remark](https://unifiedjs.com/explore/package/remark/)
- [markdown-it](https://github.com/markdown-it/markdown-it)
- [mdformat](https://mdformat.readthedocs.io/en/stable/)
- [Prettier options](https://prettier.io/docs/next/options)
- [goldmark](https://github.com/yuin/goldmark)
- [Pandoc manual](https://pandoc.org/MANUAL.html)
- [GitHub Flavored Markdown spec](https://github.github.com/gfm/)

## Executive takeaways

The main ecosystem lesson: do not hand-roll Markdown grammar with regexes when correctness matters. Mature projects either parse Markdown into an AST or event stream and then transform/serialize, or they make formatting choices explicit and configurable.

The original line-based cleaner moved in the right direction with `pulldown-cmark` verification, but the transforms themselves still relied on line heuristics — that is exactly where the highest-risk findings in the implementation review came from.

Direction adopted (status as of the cleaner integration):

1. ✅ Rust remained the production implementation language.
2. ✅ A parser-backed Markdown normalization path landed using `comrak` (Pilot B / `md_format_surgical`).
3. ✅ `pulldown-cmark` is kept as the fast independent verifier in `dual_verify`, not as the formatter.
4. ✅ cmark-gfm/Pandoc/mdformat are treated as differential oracles in sampled tests, not hot-path dependencies. cmark-gfm is the dev-only ground-truth oracle when installed; production uses `dual_verify` (comrak + pulldown-cmark) only.
5. ✅ Raw-readability is explicit: the corpus default unwraps soft-wrapped prose while preserving hard breaks (Pilot B's three target transforms — paragraph reflow, GFM separator minimization, HR canonicalization).

## Library notes

### Rust: pulldown-cmark

What it offers:

- Pull-parser event stream for CommonMark with GFM-style options such as tables, task lists, strikethrough, footnotes, admonitions, and math.
- Low memory, high performance, and a good fit for verification because it can render HTML from events.
- The guide highlights pull parsing and notes that consecutive text events can occur, with `TextMergeStream` available to smooth text iteration.

What we learned:

- Use it for strict/structural verification, especially as an independent parser from any future formatter.
- Use its `Event::SoftBreak` and `Event::HardBreak` semantics as a model for reflow: only rewrite soft breaks, never hard breaks.
- Avoid building production formatting around HTML output alone; event streams are excellent for checking, but source-preserving rewriting needs either source spans or an AST renderer.

### Rust: comrak

What it offers:

- CommonMark and GFM-compatible parser/renderer in Rust.
- Parses to an AST, allows AST manipulation, and supports `format_commonmark` / `markdown_to_commonmark`.
- Fine-grained parse/render options and custom formatter support to override rendering of node types.

What we learned:

- This was the best Rust candidate for a parser-backed Phase A. Pilot B uses comrak's AST + sourcepos to walk top-level block children and render only Paragraph/Table/ThematicBreak nodes, copying everything else verbatim from source.
- Full reserialization (the abandoned Pilot A approach) over-normalized list markers, link forms, escapes, and other syntax. The surgical approach — re-render only the three target node types — was needed to avoid those side effects.

### C: cmark-gfm

What it offers:

- GitHub's fork of the C CommonMark reference implementation, with GFM extensions.
- Parses to an AST, supports AST manipulation, and renders to HTML, CommonMark, XML, LaTeX, groff man, and more.
- Conformance, speed, fuzzing, and standardized behavior.

What we learned:

- Use cmark-gfm as a differential oracle when installed (development hosts only).
- Its XML AST output is useful for golden fixtures.
- Do not bind it into production. C FFI is extra operational surface area; comrak is already Rust-native and sufficient for `dual_verify`.

### JavaScript/TypeScript: remark / unified

What it offers:

- Unified processor that parses Markdown and serializes Markdown using `remark-parse` and `remark-stringify`.
- Uses mdast as the syntax tree.
- Lint plugins and configurable stringify settings for stylistic rules such as ordered-list markers and setext/ATX heading choices.

What we learned:

- Separate "parse", "transform", "lint", and "stringify" as distinct phases. That maps cleanly onto the Phase A / Phase B / verification split.
- Add lint-like diagnostics to the scorecard, not only pass/fail verification.
- Markdown dialect handling should be modular, not encoded as ad hoc checks scattered across the cleaner.

### JavaScript: markdown-it

What it offers:

- CommonMark-oriented parser with GFM tables/strikethrough, plugin rules, and configurable syntax.
- CommonMark support, extensibility, high speed, and safe rendering defaults.

What we learned:

- Treat dialect features as enabled rule sets. There should be a clear `MarkdownDialect::GfmDocling` profile instead of assuming every pipe-like line means one thing.
- Safety defaults matter. For us that means "do not normalize unknown syntax unless the parser confirms it is a known Markdown element."

### Python: mdformat

What it offers:

- CommonMark-compliant Markdown formatter, CLI and Python library.
- Opinionated style: consistent indentation/whitespace, ATX headings, sorted link references, fenced code instead of indented code, `1.` ordered-list markers.
- Intentionally does not change word wrapping by default to support semantic line breaks.
- Plugins for additional Markdown engines/dialects; escapes engine-specific syntax when it cannot safely understand it.

What we learned:

- Full-document formatters make many style decisions outside our scope. We should not blindly run mdformat over corpus rows.
- The plugin/dialect model is valuable. Unknown dialect syntax should be either preserved or escaped/diagnosed, not "cleaned" by regex.
- Default wrapping caution is relevant, but our objective differs: Docling line breaks are usually layout artifacts, not semantic line breaks. The policy is now explicit (see §Executive takeaways item 5).

### Prettier Markdown

What it offers:

- A `proseWrap` option with modes: `always`, `never`, and `preserve`.
- Default is `preserve` because some services have linebreak-sensitive renderers, but `never` intentionally unwraps prose blocks into single lines.

What we learned:

- The cleaner's corpus default is the equivalent of Prettier's `never` for prose paragraphs (Docling layout-induced soft breaks are unwrapped).
- Verification enforces that unwrap only rewrites parser-observed soft breaks, not hard breaks, code, HTML-sensitive blocks, tables, or line-block-like content.

### Go: goldmark

What it offers:

- CommonMark-compliant Go parser with AST, source positions, parser transformers, AST transformers, and renderers.
- Built-in GFM extension bundle.
- Extension APIs include block parsers, inline parsers, paragraph transformers, AST transformers, and renderers.

What we learned:

- Source positions are the missing primitive that comrak's `sourcepos: true` provides — it is what lets Pilot B map AST nodes back to byte ranges and rewrite only the paragraph/table/HR spans we intend to touch.
- Paragraph transformers are a strong pattern for the reflow pass: normalize paragraph text after block parsing, before final rendering.
- The architecture reinforces the need to separate block parsing from inline parsing.

### Pandoc

What it offers:

- Many Markdown variants and writer options.
- `--wrap=auto|none|preserve` controls source-level output wrapping.
- The Markdown philosophy that plain text should be readable; ordinary paragraph newlines are treated as spaces, while two trailing spaces or backslash create hard line breaks.

What we learned:

- Pandoc's `--wrap=none` is the closest existing user-facing behavior to the desired raw corpus mode.
- Pandoc is a good differential oracle for small samples but is too heavyweight for the hot path.

## Design changes that landed

### 1. Parser-backed Phase A — ✅ landed as Pilot B

Implemented in `rust/glossapi_rs_cleaner/src/md_format_surgical.rs`:

```text
input markdown
  → parse with comrak (GFM options + sourcepos)
  → walk top-level block children
  → re-render Paragraph / Table / ThematicBreak nodes
  → copy everything else verbatim from source
  → verify with dual_verify (comrak + pulldown-cmark HTML agreement on input + output)
  → on disagreement: ship input verbatim; record fallback_reason
```

The three target transformations:

- **Thematic break canonicalization** — render parser-confirmed `ThematicBreak` as `---`.
- **Table delimiter canonicalization** — render parser-confirmed table delimiter cells as `---`, `:---`, `---:`, or `:---:`.
- **Paragraph softbreak unwrap** — serialize parser-confirmed paragraph soft breaks as spaces, while preserving hard breaks.

### 2. Line-based path — ✅ removed entirely

The original "fast line-based path as conservative fallback" recommendation was reconsidered: with `dual_verify` providing per-doc safety, the line-based path's value as fallback was eclipsed by Pilot B's verbatim-on-disagreement behavior. The line-based code (`md_module::normalize_md_syntax`) was removed in the cleaner integration.

### 3. Multi-parser verification oracles — ✅ landed as dual_verify

Production cleaning runs `dual_verify(input, output)` on every Pilot B rewrite — comrak + pulldown-cmark HTML agreement. cmark-gfm is consulted when present (development hosts) as the ground-truth oracle.

### 4. Explicit dialect and formatting policy — ✅ landed as PhaseAPolicy

`md_format_surgical::PhaseAPolicy` carries the active dialect choices (comrak/cmark-gfm autolink behavior, hard-break preservation, softbreak whitespace trim policy). Surfaced through PyO3 as `phase_a_policy_py` so callers and scorecards can log what was in effect.

### 5. AST-preserving transformations as the strongest invariant — ✅ landed

The `dual_verify` check enforces preview-preservation under both parsers. When either parser sees the input rendering differently from the output, the rewrite is refused and input is shipped verbatim. The `phase_a_fallback_reason` field surfaces the cause for downstream sampling.

## Still open

These directions from the survey have NOT yet landed. They remain reasonable next steps if the cleaner's correctness or coverage is extended further.

### Pseudo-table unwrapping as a separate semantic transform

The ecosystem tools preserve tables because they are valid Markdown structures. If the corpus needs TOC pseudo-tables converted to prose, that is not formatting — it is a semantic structural rewrite. It needs its own module, its own classifier, a distinct invariant (table-to-list/prose mapping with preserved cell text order), and separate scorecard metrics. Do not mix pseudo-table unwrapping into Phase A.

### Raw-readability scorecard metrics beyond CleanStats

`CleanStats` already tracks per-rule counts and per-doc 4-way char accounting. The survey suggested extra readability-oriented metrics (`softbreaks_before/after`, `mean_physical_lines_per_paragraph_before/after`, `hr_run_width_before/after`, etc.) that would give a more direct answer to "did raw Markdown become closer to rendered Markdown?". These are not in the current `CleanStats` schema.

### Lint-style diagnostic categories

Pass/fail verification (`dual_verify` agreement) is in production. Per-category lint diagnostics (`hard_break_preserved`, `softbreak_unwrapped`, `table_delimiter_minimized`, etc.) would help when investigating regressions but are not implemented today.

## See also

- [OCR Cleaning Runtime](ocr_cleaning_runtime.md) — how the cleaner is split between analyzer and renderer responsibilities.
- `rust/glossapi_rs_cleaner/src/md_format_surgical.rs` — Pilot B implementation.
- `rust/glossapi_rs_cleaner/src/md_format.rs` — `dual_verify` (the in-process production oracle).
- `rust/glossapi_rs_cleaner/src/cmark_gfm_oracle.rs` — optional dev-only ground-truth oracle.
- `rust/glossapi_rs_cleaner/docs/PHASE_A_PARSER_BACKED_INDEX.md` — internal index of Phase A pilot reviews.
