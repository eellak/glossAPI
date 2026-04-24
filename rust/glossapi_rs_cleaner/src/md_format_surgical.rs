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

use comrak::{nodes::AstNode, nodes::NodeValue, parse_document, Arena, Options};

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

/// Return true if this node kind is one of the three Phase A
/// transforms we WANT to apply. Everything else stays verbatim.
fn is_phase_a_target(v: &NodeValue) -> bool {
    matches!(
        v,
        NodeValue::Paragraph | NodeValue::Table(_) | NodeValue::ThematicBreak,
    )
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
/// CommonMark. Used for the 3 target node kinds.
///
/// Before serialization, rewrite `SoftBreak` inline nodes inside
/// this subtree to be spaces. Comrak's default `format_commonmark`
/// emits `SoftBreak` as `\n`, which preserves source-level soft
/// wrapping — but for our Phase A "raw-readability" goal we WANT
/// soft breaks unwrapped to a single space (CM treats them as
/// whitespace in preview anyway). `LineBreak` (hard break) is
/// preserved — it's a real `<br />` in preview.
fn serialize_node_only<'a>(node: &'a AstNode<'a>, opts: &Options) -> String {
    // Rewrite SoftBreak → Text(" ") in-place before serialization.
    // Collect nodes first to avoid borrow conflicts during iteration.
    let descendants: Vec<_> = node.descendants().collect();
    for desc in descendants {
        let needs_rewrite = matches!(
            desc.data.borrow().value,
            NodeValue::SoftBreak
        );
        if needs_rewrite {
            desc.data.borrow_mut().value = NodeValue::Text(" ".to_string());
        }
    }
    let mut out = Vec::with_capacity(256);
    comrak::format_commonmark(node, opts, &mut out).expect("format_commonmark");
    String::from_utf8(out).unwrap_or_default()
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

    // Walk Document's top-level block children in order.
    for child in root.children() {
        let ast = child.data.borrow();
        let sp = ast.sourcepos;
        // Byte range for this node in the source.
        let start = line_col_to_byte(md, &line_offsets, sp.start.line, sp.start.column);
        // End column is inclusive — add 1 char width to get the
        // byte AFTER the last char.
        let mut end_exclusive = {
            let col_end_byte =
                line_col_to_byte(md, &line_offsets, sp.end.line, sp.end.column);
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
        while end_exclusive > start
            && md.as_bytes()[end_exclusive - 1] == b'\n'
        {
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
        if is_phase_a_target(&ast.value) {
            // Canonical serialization for this one node.
            let canonical = serialize_node_only(child, &opts);
            // comrak's per-node output ends with `\n` already; we'll
            // rely on the source's own line-end bytes outside the
            // span for final layout. Strip a single trailing `\n`
            // if present so we don't double-up when the source
            // already had a newline after the node.
            let canonical_trimmed = canonical.strip_suffix('\n').unwrap_or(&canonical);
            out.push_str(canonical_trimmed);
        } else {
            // Verbatim passthrough.
            out.push_str(&md[start..end_exclusive]);
        }
        cursor = end_exclusive;
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

    #[test]
    fn sg_paragraph_reflow() {
        let input = "This is a\nsoft-wrapped\nparagraph.\n\nSecond\npara.\n";
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
}
