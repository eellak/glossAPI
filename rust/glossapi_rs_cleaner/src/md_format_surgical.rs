//! Pilot B — surgical parser-backed Phase A rewriter.
//!
//! Unlike `md_format::format_parsed` (which round-trips the whole
//! document through comrak's `format_commonmark`), this module:
//!
//! 1. Parses the input with comrak to get an AST with source
//!    positions (line:col ranges per node).
//! 2. Walks ONLY the top-level block children of the Document node.
//! 3. For each top-level child, decides:
//!    - **Paragraph / Table / ThematicBreak** → emit comrak's
//!      canonical serialization for this single node. This is what
//!      gives us the three Phase A transformations (reflow unwrap,
//!      GFM sep minimization, HR canonicalization).
//!    - **Everything else** (Heading, List, BlockQuote, CodeBlock,
//!      HtmlBlock, ReferenceDef, FootnoteDef, …) → copy the source
//!      bytes for that node verbatim.
//! 4. Preserves the original text between consecutive top-level
//!    nodes (blank lines, trailing content) verbatim.
//!
//! Why: comrak's whole-doc round-trip over-normalizes things outside
//! our 3 target transforms — list markers (`1.` vs `1)`), link
//! forms, character escapes, URL encoding. On the 90-doc real-corpus
//! audit, 50 of 66 Pilot A failures traced to these non-target
//! normalizations. Pilot B keeps every non-target span byte-exact.
//!
//! Paragraphs NESTED inside BlockQuote / List / etc. are kept
//! verbatim (so their reflow is NOT performed) — this is a
//! conservative trade-off: the outer container (BlockQuote, List)
//! is preserved exactly, which means the non-target canonicalization
//! can't happen there. A future extension could walk into container
//! nodes and surgically reflow nested paragraphs, but only after the
//! top-level approach is proven safe.

use comrak::{
    nodes::{AstNode, NodeValue, TableAlignment},
    parse_document, Arena, Options,
};

/// Build comrak options matching what `md_format::format_parsed`
/// uses, so Pilot A and Pilot B operate under the same parser
/// assumptions. Sourcepos IS enabled here because we need the
/// line:col ranges.
fn options_with_sourcepos() -> Options<'static> {
    let mut opts = Options::default();
    opts.extension.table = true;
    opts.extension.strikethrough = true;
    opts.extension.tasklist = true;
    opts.extension.footnotes = true;
    opts.extension.autolink = false;
    opts.extension.tagfilter = false;
    opts.render.sourcepos = true;
    opts.render.unsafe_ = true;
    opts
}

/// Byte-offset table keyed by 1-based line number. Given a
/// CommonMark source string, `line_byte_offsets[i]` is the byte
/// offset at which the i-th line begins (line 1 → index 1).
/// Index 0 is unused (1-based convention to match Sourcepos).
fn build_line_offset_table(src: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(src.len() / 40 + 2);
    offsets.push(0); // index 0 unused
    offsets.push(0); // line 1 starts at byte 0
    let bytes = src.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            offsets.push(i + 1);
        }
    }
    offsets
}

/// Convert (line, column) to a byte offset in `src`. Columns are
/// 1-based CHARACTER positions per CommonMark; this function walks
/// chars from the start of the line to handle multi-byte UTF-8.
/// Caps at the length of the line / src if the column is past EOL.
fn line_col_to_byte(src: &str, line_offsets: &[usize], line: usize, col: usize) -> usize {
    if line == 0 || line >= line_offsets.len() {
        return src.len();
    }
    let line_start = line_offsets[line];
    let line_end = if line + 1 < line_offsets.len() {
        line_offsets[line + 1]
    } else {
        src.len()
    };
    // Column is 1-based char position within the line (stripped of
    // trailing `\n`). Walk chars until we've consumed `col-1` of
    // them, then return the byte offset of the next char.
    if col <= 1 {
        return line_start;
    }
    let line_text = &src[line_start..line_end];
    let mut b = 0usize;
    for (i, (byte_off, _)) in line_text.char_indices().enumerate() {
        if i + 1 == col {
            return line_start + byte_off;
        }
        b = byte_off;
    }
    // Past end of line — return end-of-line (excluding trailing `\n`).
    let _ = b;
    // If line ends with `\n`, exclude it from the byte range.
    if line_text.ends_with('\n') {
        line_start + line_text.len() - 1
    } else {
        line_end
    }
}

/// Serialize ONE comrak AST node (and its descendants) back to
/// CommonMark via `format_commonmark`. Used for ThematicBreak nodes
/// where the only transform we want IS comrak's canonical form.
///
/// NOT used for Paragraph / Table — those go through source-level
/// rewrites that preserve inline content byte-exact (see
/// `paragraph_source_with_softbreaks_unwrapped` and
/// `table_source_with_minimal_delimiter`).
fn serialize_node_only<'a>(node: &'a AstNode<'a>, opts: &Options) -> String {
    // Rewrite SoftBreak → Text(" ") in-place before serialization.
    let descendants: Vec<_> = node.descendants().collect();
    for desc in descendants {
        let needs_rewrite = matches!(desc.data.borrow().value, NodeValue::SoftBreak);
        if needs_rewrite {
            desc.data.borrow_mut().value = NodeValue::Text(" ".to_string());
        }
    }
    let mut out = Vec::with_capacity(256);
    comrak::format_commonmark(node, opts, &mut out).expect("format_commonmark");
    String::from_utf8(out).unwrap_or_default()
}

/// Source-level paragraph unwrap: take the paragraph's raw source
/// bytes and replace SoftBreak newlines with single spaces.
/// Everything else (URLs with `%XX`, escape forms like `\*`, inline
/// code with literal pipes, link markup, em/strong markers) stays
/// byte-exact from source.
///
/// CommonMark rule for identifying soft break vs hard break inside
/// a paragraph:
/// - `  \n` (two or more trailing spaces before newline) = hard
/// - `\\\n` (odd count of trailing backslashes) = hard
/// - any other single `\n` = soft break, unwrap to space
///
/// Important: this is line-level rewriting on the raw source of a
/// paragraph (everything between the paragraph's start and end
/// byte offsets). Works correctly even if the paragraph contains
/// inline HTML, autolinks, or escape sequences, because we only
/// touch whitespace around `\n`.
fn paragraph_source_with_softbreaks_unwrapped(para_src: &str) -> String {
    let lines: Vec<&str> = para_src.split('\n').collect();
    let mut out = String::with_capacity(para_src.len());
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            out.push_str(line);
            continue;
        }
        let prev = &lines[i - 1];
        let hard_break = prev_is_hard_break(prev);
        if hard_break {
            out.push('\n');
            out.push_str(line);
        } else {
            // Soft break: roll back trailing ASCII-whitespace from
            // `out`, emit one space, then append line with leading
            // ASCII-whitespace stripped. ASCII-ONLY because
            // Docling-extracted PDFs use U+00A0 (NBSP) as a
            // meaningful column-preservation marker — cmark-gfm
            // treats NBSP as content and so must we. Using Rust's
            // `trim_start()` / `trim_end()` here would be wrong;
            // those are Unicode-whitespace-aware and strip NBSP.
            let trimmed_len = trim_end_ascii_ws(&out).len();
            out.truncate(trimmed_len);
            out.push(' ');
            out.push_str(trim_start_ascii_ws(line));
        }
    }
    out
}

/// Return `s` with trailing ASCII whitespace (space, tab, `\r`)
/// removed. NBSP (U+00A0) and other Unicode whitespace are
/// preserved — cmark-gfm treats those as content.
fn trim_end_ascii_ws(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 {
        let c = bytes[end - 1];
        if c == b' ' || c == b'\t' || c == b'\r' {
            end -= 1;
        } else {
            break;
        }
    }
    &s[..end]
}

/// Return `s` with leading ASCII whitespace (space, tab, `\r`)
/// removed. NBSP (U+00A0) and other Unicode whitespace preserved.
fn trim_start_ascii_ws(s: &str) -> &str {
    let bytes = s.as_bytes();
    let mut start = 0;
    while start < bytes.len() {
        let c = bytes[start];
        if c == b' ' || c == b'\t' || c == b'\r' {
            start += 1;
        } else {
            break;
        }
    }
    &s[start..]
}

/// Return true if the last non-`\r` content of `prev` is a
/// CommonMark hard-break marker:
/// - 2+ trailing spaces, OR
/// - odd count of trailing backslashes.
fn prev_is_hard_break(prev: &str) -> bool {
    let s = prev.trim_end_matches('\r');
    if s.ends_with("  ") {
        return true;
    }
    let trailing_backslashes = s.chars().rev().take_while(|c| *c == '\\').count();
    trailing_backslashes % 2 == 1
}

/// Emit the canonical GFM separator row for a table with the given
/// per-column alignments. e.g. `| --- | :--- | ---: |`.
fn canonical_gfm_separator_row(alignments: &[TableAlignment]) -> String {
    let mut s = String::with_capacity(alignments.len() * 8);
    s.push('|');
    for a in alignments {
        s.push(' ');
        match a {
            TableAlignment::None => s.push_str("---"),
            TableAlignment::Left => s.push_str(":---"),
            TableAlignment::Right => s.push_str("---:"),
            TableAlignment::Center => s.push_str(":---:"),
        }
        s.push_str(" |");
    }
    s
}

/// Surgical rewrite of a Table node's source: keep header + body
/// bytes byte-exact, replace ONLY the delimiter row (second line)
/// with a canonical `| --- | :--- | …` form.
///
/// Why not re-serialize via `format_commonmark`: comrak's table
/// serialization adds `\` escapes to `_`, `[`, `]`, `#` etc. inside
/// URL text in cells, which cmark-gfm then percent-encodes — the
/// single biggest Pilot B residual-failure category.
fn table_source_with_delimiter_rewritten(table_src: &str, alignments: &[TableAlignment]) -> String {
    // Find the first and second `\n` in the table source: the
    // delimiter row is between them. Header row = bytes 0..first_nl.
    // Delimiter row = bytes first_nl+1..second_nl.
    let bytes = table_src.as_bytes();
    let mut first_nl = None;
    let mut second_nl = None;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            if first_nl.is_none() {
                first_nl = Some(i);
            } else {
                second_nl = Some(i);
                break;
            }
        }
    }
    let (Some(first), Some(second)) = (first_nl, second_nl) else {
        // Malformed / single-line "table" — pass through unchanged.
        return table_src.to_string();
    };
    // Sanity: the original delimiter row must start with `|` or a
    // digit-free dash sequence. If not, don't rewrite.
    let original_delim = &table_src[first + 1..second];
    let t = original_delim.trim();
    if !t.contains('-') {
        return table_src.to_string();
    }
    let canonical = canonical_gfm_separator_row(alignments);
    let mut out = String::with_capacity(table_src.len());
    out.push_str(&table_src[..first + 1]); // header row + its `\n`
    out.push_str(&canonical);
    out.push_str(&table_src[second..]); // `\n` + body
    out
}

/// Pilot B entry: surgical Phase A rewrite.
///
/// Walks top-level block children of the comrak Document. Emits
/// comrak's canonical form for Paragraph / Table / ThematicBreak;
/// everything else, including inter-node whitespace, is preserved
/// byte-exact from the source.
pub fn format_surgical(md: &str) -> String {
    let arena = Arena::new();
    let opts = options_with_sourcepos();
    let root = parse_document(&arena, md, &opts);
    let line_offsets = build_line_offset_table(md);

    let mut out = String::with_capacity(md.len());
    let mut cursor: usize = 0; // byte offset in source we've copied up to

    // Collect children so we can peek at the next sibling's kind
    // when deciding whether to inject a blank-line separator.
    let children: Vec<_> = root.children().collect();
    for (idx, child) in children.iter().enumerate() {
        let ast = child.data.borrow();
        let next_is_hr = children
            .get(idx + 1)
            .map(|n| matches!(n.data.borrow().value, NodeValue::ThematicBreak))
            .unwrap_or(false);
        let sp = ast.sourcepos;
        // Byte range for this node in the source.
        let start = line_col_to_byte(md, &line_offsets, sp.start.line, sp.start.column);
        // End column is inclusive — add 1 char width to get the
        // byte AFTER the last char.
        let mut end_exclusive = {
            let col_end_byte = line_col_to_byte(md, &line_offsets, sp.end.line, sp.end.column);
            if col_end_byte < md.len() {
                let rest = &md[col_end_byte..];
                col_end_byte + rest.chars().next().map_or(0, |c| c.len_utf8())
            } else {
                md.len()
            }
        };
        // Comrak sometimes reports block-node sourcepos.end on the
        // line AFTER the content (e.g. HR `3:1-4:0` includes the
        // blank line after). Trim trailing `\n` chars off the byte
        // range so blank lines fall into inter-node preservation,
        // not into the node's splice span.
        while end_exclusive > start && md.as_bytes()[end_exclusive - 1] == b'\n' {
            end_exclusive -= 1;
        }
        // Defensive: if the end still looks wrong (empty span or
        // crosses the cursor backward), skip this node — something
        // is off with the sourcepos. Better to lose a transform on
        // one node than to corrupt the output.
        if end_exclusive <= start || start < cursor {
            continue;
        }
        // Preserve inter-node source (blank lines, comments, etc.)
        if start > cursor {
            out.push_str(&md[cursor..start]);
        }
        let rewritten_block = match &ast.value {
            NodeValue::Paragraph => {
                // Source-level SoftBreak unwrap — preserves all
                // inline content (URLs, escapes, inline code, etc.)
                // byte-exact. Only whitespace around `\n` changes.
                let para_src = &md[start..end_exclusive];
                let rewritten = paragraph_source_with_softbreaks_unwrapped(para_src);
                out.push_str(&rewritten);
                true
            }
            NodeValue::Table(tbl) => {
                // Delimiter-only rewrite: keep every cell byte-exact,
                // replace ONLY the `|---|---|` row with canonical
                // `| --- | :--- | ---: | :---: |` form. This avoids
                // comrak's URL-escape injection inside cells that
                // cmark-gfm re-encodes.
                let table_src = &md[start..end_exclusive];
                let rewritten = table_source_with_delimiter_rewritten(table_src, &tbl.alignments);
                out.push_str(&rewritten);
                true
            }
            NodeValue::ThematicBreak => {
                // Canonical HR is just `---`.
                out.push_str("---");
                true
            }
            _ => {
                // Verbatim passthrough.
                out.push_str(&md[start..end_exclusive]);
                false
            }
        };
        cursor = end_exclusive;

        // If this rewritten block is a Paragraph immediately
        // followed by a ThematicBreak, the source might only have a
        // single `\n` between them (or the HR's canonical form
        // might land adjacent). That creates setext-heading
        // ambiguity — cmark-gfm would re-parse the paragraph as a
        // setext H2 with `---` as the underline. Force `\n\n`
        // separation in output, consuming source's `\n`s so we
        // don't double-up.
        //
        // For paragraph → any-other-block (including comrak's split
        // of what cmark-gfm sees as one soft-wrapped paragraph),
        // DO NOT inject extra blank line — let the source decide.
        // (Dialect-ambiguous input ≠ our bug.)
        let needs_forced_blank_line =
            matches!(&ast.value, NodeValue::Paragraph | NodeValue::ThematicBreak)
                && (next_is_hr || matches!(&ast.value, NodeValue::ThematicBreak));
        if needs_forced_blank_line {
            let mut consumed = 0usize;
            while cursor + consumed < md.len() && md.as_bytes()[cursor + consumed] == b'\n' {
                consumed += 1;
            }
            out.push_str("\n\n");
            cursor += consumed;
        }
        let _ = rewritten_block;
    }
    // Trailing source after last node (typically `\n` or blank).
    if cursor < md.len() {
        out.push_str(&md[cursor..]);
    }
    out
}

// ---------------------------------------------------------------------------
// PyO3.
// ---------------------------------------------------------------------------

use pyo3::prelude::*;

#[pyfunction]
pub fn format_surgical_py(md: &str) -> String {
    format_surgical(md)
}

// ---------------------------------------------------------------------------
// Tests — reuse the fixture shape from md_format but assert preview
// preservation via the same dual-parser verifier.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md_format::dual_verify;

    fn assert_surgical_preserves_preview(input: &str) {
        let out = format_surgical(input);
        let r = dual_verify(input, &out);
        assert!(
            r.is_preview_preserving_per_parser(),
            "\n=== INPUT ===\n{}\n=== OUTPUT ===\n{}\n=== DIFF ===\n{:?}\n",
            input,
            out,
            r.first_diff,
        );
    }

    fn assert_surgical_preserves_cmark_preview_if_available(input: &str) {
        let out = format_surgical(input);
        assert_surgical_preserves_preview(input);
        if !crate::cmark_gfm_oracle::is_available() {
            eprintln!("cmark-gfm not available — skipping cmark-gfm assertion");
            return;
        }
        let r = crate::cmark_gfm_oracle::verify(input, &out).expect("cmark-gfm verify");
        assert!(
            r.preview_identical,
            "\n=== INPUT ===\n{}\n=== OUTPUT ===\n{}\n=== CMARK DIFF ===\n{:?}\n",
            input, out, r.first_diff,
        );
    }

    #[test]
    fn sg_paragraph_reflow() {
        let input = "This is a\nsoft-wrapped\nparagraph.\n\nSecond\npara.\n";
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_paragraph_with_nbsp_preserves_nbsp_as_content() {
        // Docling-extracted PDFs use NBSP (U+00A0) as a meaningful
        // column-preservation marker on continuation lines. cmark-gfm
        // treats NBSP as content. Our reflow must NOT strip NBSP via
        // Unicode-aware trim — only ASCII space/tab/CR should be
        // trimmed at soft-break boundaries.
        let input = "Παρασκευή\t\n \u{00A0}των\t\n \u{00A0}δεκαδικών\n";
        let out = format_surgical(input);
        // NBSP chars must survive in the output (one per original
        // continuation line).
        let nbsp_count = out.chars().filter(|c| *c == '\u{00A0}').count();
        assert_eq!(
            nbsp_count, 2,
            "expected 2 NBSPs preserved, got {nbsp_count}. output={out:?}"
        );
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_table_minimization() {
        let input = "| a | b |\n| ---------- | ---------- |\n| 1 | 2 |\n";
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_hr_canonical() {
        let input = "before\n\n----------\n\nafter\n";
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_numeric_prefix_not_in_list_preserved() {
        // If the source has `28 «text»` as part of prose (not a
        // list item), surgical MUST NOT turn it into a list marker.
        // Comrak's full round-trip canonicalized whitespace around
        // it and that triggered spurious list detection; surgical
        // keeps the containing list / paragraph verbatim.
        let input = "previous sentence.\n\n«ΕΟΚ και ΝΑΤΟ το ίδιο συνδικάτο» 28 . Αυτό το κόμμα.\n";
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_list_markers_preserved_verbatim() {
        // A list should pass through with its original markers +
        // formatting — NOT be re-canonicalized. If the source uses
        // `1)` instead of `1.`, surgical keeps `1)`.
        let input = "1) alpha\n2) beta\n3) gamma\n";
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_fenced_code_byte_exact() {
        // Fenced code blocks are NOT in our target set; surgical
        // should pass them through byte-for-byte (no info-string
        // normalization, no trailing-newline mangling).
        let input = "```rust\nfn main() {}\n```\n\nprose.\n";
        let out = format_surgical(input);
        // Code block must survive verbatim.
        assert!(out.contains("```rust\nfn main() {}\n```"), "got: {out:?}");
    }

    #[test]
    fn sg_blockquote_preserved_verbatim() {
        // Blockquotes are outside our target — preserved verbatim,
        // including any soft-wrap inside (no reflow through `>`).
        let input = "> quoted line one\n> quoted line two\n\nafter.\n";
        let out = format_surgical(input);
        assert!(out.contains("> quoted line one\n> quoted line two"));
    }

    #[test]
    fn sg_mixed_preserves_list_but_reflows_paragraph() {
        let input = concat!(
            "# Heading\n\n",
            "A soft-wrapped\nparagraph.\n\n",
            "- item one\n- item two\n\n",
            "Another soft\nwrapped\nparagraph.\n\n",
            "---------\n\n",
            "After.\n",
        );
        let out = format_surgical(input);
        // Paragraph got reflowed.
        assert!(out.contains("A soft-wrapped paragraph."), "out: {out}");
        // List preserved verbatim.
        assert!(out.contains("- item one\n- item two"));
        // HR canonicalized to `-----` from 9 dashes → comrak emits
        // `-----` (or `---`). Either way, not the original 9-dash.
        // Preview preservation is what matters.
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_optional_pipe_table_gets_delimiter_only_rewrite() {
        // GFM tables do not require leading/trailing pipes. The
        // parser must identify the table and Pilot B should rewrite
        // only the delimiter row, leaving header/body bytes alone.
        let input = "a | b\n---------- | :----------:\n1 | 2\n";
        let out = format_surgical(input);
        assert!(
            out.contains("a | b\n| --- | :---: |\n1 | 2"),
            "delimiter should be canonical but cells byte-exact; out={out:?}"
        );
        assert_surgical_preserves_cmark_preview_if_available(input);
    }

    #[test]
    fn sg_table_cell_code_span_pipe_and_url_bytes_survive() {
        // Pipes inside code spans and URL-ish cell text are classic
        // places where table serializers over-escape. Surgical must
        // rely on the parser for the table span, but preserve cell
        // source bytes exactly.
        let input = concat!(
            "| expr | url |\n",
            "| ---------- | ---------- |\n",
            "| `a | b` | https://example.com/a_b?q=[x] |\n",
        );
        let out = format_surgical(input);
        assert!(
            out.contains("| `a | b` | https://example.com/a_b?q=[x] |"),
            "out={out:?}"
        );
        assert!(out.contains("| --- | --- |"), "out={out:?}");
        assert_surgical_preserves_cmark_preview_if_available(input);
    }

    #[test]
    fn sg_setext_heading_is_not_rewritten_as_paragraph_plus_hr() {
        // Parser identity matters here: `Title\n---` is a heading,
        // not a paragraph followed by a thematic break.
        let input = "Title\n---\n\nAfter.\n";
        let out = format_surgical(input);
        assert_eq!(out, input, "setext heading should pass through byte-exact");
        assert_surgical_preserves_preview(input);
    }

    #[test]
    fn sg_hr_between_paragraphs_gets_padding_to_avoid_setext_ambiguity() {
        // Canonicalizing `-----` to `---` next to paragraph text can
        // accidentally create a setext heading unless we force blank
        // line separation around the HR.
        let input = "alpha\n\n-----\nbeta\n";
        let out = format_surgical(input);
        assert_eq!(out, "alpha\n\n---\n\nbeta\n", "out={out:?}");
        assert_surgical_preserves_cmark_preview_if_available(input);
    }

    #[test]
    fn sg_multibyte_greek_sourcepos_reflows_and_rewrites_table() {
        // Source-position slicing must stay correct on multi-byte
        // Greek text, otherwise byte ranges will corrupt UTF-8 or
        // splice the wrong block.
        let input = concat!(
            "Αλφα\n",
            "βήτα\n\n",
            "| λέξη | τιμή |\n",
            "| ------------ | ------------ |\n",
            "| γάμμα | δέλτα |\n",
        );
        let out = format_surgical(input);
        assert!(out.contains("Αλφα βήτα"), "out={out:?}");
        assert!(out.contains("| --- | --- |"), "out={out:?}");
        assert_surgical_preserves_cmark_preview_if_available(input);
    }

    #[test]
    fn sg_inline_code_span_softbreak_is_parser_identical() {
        // CommonMark normalizes line endings inside code spans to
        // spaces in rendered code text. This challenges source-level
        // paragraph unwrap without letting a full formatter touch the
        // rest of the inline markup.
        let input = "Use `alpha\nbeta` inside code\nand continue.\n";
        let out = format_surgical(input);
        assert!(
            out.contains("Use `alpha beta` inside code and continue."),
            "out={out:?}"
        );
        assert_surgical_preserves_cmark_preview_if_available(input);
    }

    #[test]
    #[ignore = "current Pilot B only rewrites top-level paragraphs; recursive container rewrites are future work"]
    fn red_until_surgical_reflows_softbreaks_inside_blockquote() {
        let input = "> quoted line one\n> quoted line two\n\nAfter.\n";
        let out = format_surgical(input);
        assert!(
            out.contains("> quoted line one quoted line two"),
            "nested blockquote paragraph was not reflowed; out={out:?}"
        );
        assert_surgical_preserves_preview(input);
    }

    #[test]
    #[ignore = "current Pilot B only rewrites top-level paragraphs; recursive container rewrites are future work"]
    fn red_until_surgical_reflows_softbreaks_inside_list_item() {
        let input = "- This item is\n  soft wrapped.\n- Next item.\n";
        let out = format_surgical(input);
        assert!(
            out.contains("- This item is soft wrapped."),
            "nested list paragraph was not reflowed; out={out:?}"
        );
        assert_surgical_preserves_preview(input);
    }
}
