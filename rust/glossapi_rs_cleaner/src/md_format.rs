//! Parser-backed Phase A formatter (pilot).
//!
//! This module is the *minimal* parser-backed Phase A: parse MD with
//! a real CommonMark/GFM parser (`comrak`), then re-serialize with
//! `format_commonmark`. Correctness follows from the parser + the
//! dual-parser verifier in `dual_verify`. No line heuristics.
//!
//! Design per the 2026-04-24 MD library survey (§1 "Add parser-backed
//! Phase A as a pilot") and the user's direction that the right
//! solution is parser semantics, not heuristics.
//!
//! This module is DELIBERATELY SMALL. The hypothesis is that a full
//! round-trip through comrak already produces the corpus we want —
//! paragraphs unwrapped (soft breaks → spaces), tables minimized,
//! thematic breaks canonicalized, all by comrak's own formatter.
//! We verify with `dual_verify` (pulldown-cmark + comrak HTML
//! agreement across input and output). If the dual verifier passes,
//! the output is safe.
//!
//! If full round-trip over-normalizes (e.g. rewrites link forms or
//! heading styles we want preserved), we'll escalate to a surgical
//! rewrite — parse, keep most spans verbatim, re-serialize only the
//! paragraph / table / thematic-break nodes. Not done in this pilot.

use comrak::{nodes::AstNode, parse_document, Arena, Options};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pulldown_cmark::{html as pd_html, Options as PdOptions, Parser as PdParser};

/// Build the comrak options we use for Phase A. GFM extensions
/// enabled so tables / strikethrough / tasklists / footnotes parse
/// correctly. CommonMark-compatible output by default.
fn phase_a_options() -> Options<'static> {
    let mut opts = Options::default();
    opts.extension.table = true;
    opts.extension.strikethrough = true;
    opts.extension.tasklist = true;
    opts.extension.footnotes = true;
    // Autolink + tagfilter off — they alter content in ways we don't
    // want for a Phase A rewriter (autolink rewrites bare URLs to
    // `<url>` markdown; tagfilter escapes HTML tags). Both would show
    // up as diffs vs the input without being formatting.
    opts.extension.autolink = false;
    opts.extension.tagfilter = false;
    opts.render.unsafe_ = true; // don't filter HTML on HTML render
    opts
}

/// Parser-backed Phase A: parse with comrak, re-serialize to
/// CommonMark. Returns the rewritten markdown.
///
/// Panics / parse errors propagate — comrak is permissive, should
/// not normally error; if it does the caller should fall back to the
/// line-based path.
pub fn format_parsed(md: &str) -> String {
    let arena = Arena::new();
    let opts = phase_a_options();
    let root: &AstNode = parse_document(&arena, md, &opts);
    let mut out: Vec<u8> = Vec::with_capacity(md.len());
    comrak::format_commonmark(root, &opts, &mut out).expect("format_commonmark write");
    String::from_utf8(out).unwrap_or_else(|_| md.to_string())
}

// ---------------------------------------------------------------------------
// Dual-parser verifier: pulldown-cmark + comrak agreement.
// ---------------------------------------------------------------------------

/// Report from the dual-parser verifier.
#[derive(Debug, Clone, Default)]
pub struct DualVerifyReport {
    /// pulldown-cmark HTML of the INPUT, whitespace-collapsed.
    pub pd_input_html: String,
    /// pulldown-cmark HTML of the OUTPUT, whitespace-collapsed.
    pub pd_output_html: String,
    /// comrak HTML of the INPUT, whitespace-collapsed.
    pub cm_input_html: String,
    /// comrak HTML of the OUTPUT, whitespace-collapsed.
    pub cm_output_html: String,
    /// True if both parsers agree the INPUT renders to the same HTML
    /// (well-formedness signal: two independent parsers see the same
    /// structure — the doc is not dialect-ambiguous).
    pub input_parser_agreement: bool,
    /// True if both parsers agree the OUTPUT renders to the same HTML.
    pub output_parser_agreement: bool,
    /// True if INPUT and OUTPUT render to the same HTML under
    /// pulldown-cmark (render identity — the Phase A invariant).
    pub pd_identity: bool,
    /// True if INPUT and OUTPUT render to the same HTML under comrak
    /// (render identity under the second parser).
    pub cm_identity: bool,
    /// Short description of the first divergence, if any.
    pub first_diff: Option<String>,
}

impl DualVerifyReport {
    /// True iff all four agreement/identity checks pass — meaning:
    /// - two independent parsers agree on the input's render
    /// - two independent parsers agree on the output's render
    /// - input and output render identically under BOTH parsers
    ///
    /// This is the strongest signal we have that a rewrite preserved
    /// preview on a well-formed document.
    pub fn is_well_formed_and_identical(&self) -> bool {
        self.input_parser_agreement
            && self.output_parser_agreement
            && self.pd_identity
            && self.cm_identity
    }

    /// True iff input and output render to the same HTML under EACH
    /// parser individually (parsers may disagree with each other).
    ///
    /// This is the Phase A preview-preservation invariant strictly:
    /// the rewrite didn't change preview, but the document itself
    /// may be dialect-ambiguous (two parsers render it differently).
    /// Dialect-ambiguity is a property of the INPUT, not the rewrite.
    pub fn is_preview_preserving_per_parser(&self) -> bool {
        self.pd_identity && self.cm_identity
    }

    /// True iff the input is well-formed (two parsers agree). Used
    /// to classify a doc as dialect-safe for rewrite, independent of
    /// whether a rewrite has happened.
    pub fn is_input_well_formed(&self) -> bool {
        self.input_parser_agreement
    }
}

fn pulldown_render(md: &str) -> String {
    let mut opts = PdOptions::empty();
    opts.insert(PdOptions::ENABLE_TABLES);
    opts.insert(PdOptions::ENABLE_FOOTNOTES);
    opts.insert(PdOptions::ENABLE_STRIKETHROUGH);
    opts.insert(PdOptions::ENABLE_TASKLISTS);
    let parser = PdParser::new_ext(md, opts);
    let mut html = String::new();
    pd_html::push_html(&mut html, parser);
    collapse_ws(&html)
}

fn comrak_render(md: &str) -> String {
    let arena = Arena::new();
    let opts = phase_a_options();
    let root = parse_document(&arena, md, &opts);
    let mut html = Vec::new();
    comrak::format_html(root, &opts, &mut html).expect("format_html write");
    collapse_ws(&String::from_utf8_lossy(&html))
}

/// Collapse whitespace runs to a single space, trim, AND drop
/// whitespace between adjacent tags (`> <` → `><`). pulldown-cmark
/// and comrak emit differently-spaced HTML for the same input
/// (notably around `<table>` / `<tr>` / `<td>`), so inter-tag
/// whitespace MUST be normalized out before comparison — it's
/// invisible in any browser / renderer anyway.
fn collapse_ws(s: &str) -> String {
    let mut collapsed = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(c);
            prev_ws = false;
        }
    }
    // Iteratively strip `> <` → `><` so `<table> <thead>` → `<table><thead>`.
    // Run in a loop because collapsing can create new adjacencies.
    let mut prev = collapsed.trim().to_string();
    loop {
        let next = prev.replace("> <", "><");
        if next == prev {
            break;
        }
        prev = next;
    }
    // Strip trailing whitespace before a closing tag: `X </tag>` →
    // `X</tag>`. Pulldown-cmark preserves trailing whitespace inside
    // block content (e.g. `## heading\t` → `<h2>heading </h2>`);
    // comrak strips it on round-trip. Both render identically in any
    // preview since block-level trailing whitespace is invisible.
    let mut prev = prev;
    loop {
        let next = prev.replace(" </", "</");
        if next == prev {
            break;
        }
        prev = next;
    }
    // Strip comrak-specific `<!-- end list -->` boilerplate. Comrak
    // emits this marker between consecutive lists of different
    // marker types to disambiguate; pulldown-cmark doesn't. The
    // comment is invisible in any renderer so stripping is safe for
    // preview-equivalence comparison.
    let prev = prev.replace("<!-- end list -->", "");
    // Re-collapse any space runs created by the strip.
    let mut collapsed = String::with_capacity(prev.len());
    let mut prev_ws = false;
    for c in prev.chars() {
        if c == ' ' {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(c);
            prev_ws = false;
        }
    }
    let mut prev = collapsed.trim().to_string();
    loop {
        let next = prev.replace("> <", "><");
        if next == prev {
            break;
        }
        prev = next;
    }
    // Normalize GFM table alignment attribute encoding:
    //   pulldown-cmark emits `style="text-align: left"`
    //   comrak        emits `align="left"`
    // Both produce the same rendered alignment. Collapse to one form
    // (picking comrak's `align="X"` since it's shorter + HTML-legal
    // in all browsers for `<th>`/`<td>`).
    let prev = prev.replace(r#" style="text-align: left""#, r#" align="left""#);
    let prev = prev.replace(r#" style="text-align: right""#, r#" align="right""#);
    let prev = prev.replace(r#" style="text-align: center""#, r#" align="center""#);
    prev
}

/// Run the dual-parser verifier on `(input, output)`. Returns a
/// report; `is_well_formed_and_identical()` is the single-boolean
/// summary.
pub fn dual_verify(input: &str, output: &str) -> DualVerifyReport {
    let pd_in = pulldown_render(input);
    let pd_out = pulldown_render(output);
    let cm_in = comrak_render(input);
    let cm_out = comrak_render(output);
    let input_parser_agreement = pd_in == cm_in;
    let output_parser_agreement = pd_out == cm_out;
    let pd_identity = pd_in == pd_out;
    let cm_identity = cm_in == cm_out;
    let first_diff = if pd_identity && cm_identity && input_parser_agreement && output_parser_agreement {
        None
    } else {
        let (label, a, b) = if !input_parser_agreement {
            ("parsers disagree on input (dialect-ambiguous)",
             pd_in.as_str(), cm_in.as_str())
        } else if !output_parser_agreement {
            ("parsers disagree on output",
             pd_out.as_str(), cm_out.as_str())
        } else if !pd_identity {
            ("pulldown-cmark sees input != output",
             pd_in.as_str(), pd_out.as_str())
        } else {
            ("comrak sees input != output",
             cm_in.as_str(), cm_out.as_str())
        };
        let prefix = common_prefix_len(a, b);
        let mut snippet = String::new();
        snippet.push_str(label);
        snippet.push_str("\n  a: ");
        snippet.push_str(&a.chars().skip(prefix.saturating_sub(40)).take(200).collect::<String>());
        snippet.push_str("\n  b: ");
        snippet.push_str(&b.chars().skip(prefix.saturating_sub(40)).take(200).collect::<String>());
        Some(snippet)
    };
    DualVerifyReport {
        pd_input_html: pd_in,
        pd_output_html: pd_out,
        cm_input_html: cm_in,
        cm_output_html: cm_out,
        input_parser_agreement,
        output_parser_agreement,
        pd_identity,
        cm_identity,
        first_diff,
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut n = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        n += ca.len_utf8();
    }
    n
}

// ---------------------------------------------------------------------------
// PyO3 surface.
// ---------------------------------------------------------------------------

/// PyO3: apply parser-backed Phase A. Returns the rewritten markdown.
#[pyfunction]
pub fn format_parsed_py(md: &str) -> String {
    format_parsed(md)
}

/// PyO3: run the dual-parser verifier, return a Python dict.
#[pyfunction]
pub fn dual_verify_py(py: Python<'_>, input: &str, output: &str) -> PyResult<PyObject> {
    let r = dual_verify(input, output);
    let d = PyDict::new(py);
    d.set_item("input_parser_agreement", r.input_parser_agreement)?;
    d.set_item("output_parser_agreement", r.output_parser_agreement)?;
    d.set_item("pd_identity", r.pd_identity)?;
    d.set_item("cm_identity", r.cm_identity)?;
    d.set_item("is_well_formed_and_identical", r.is_well_formed_and_identical())?;
    d.set_item("is_preview_preserving_per_parser", r.is_preview_preserving_per_parser())?;
    d.set_item("is_input_well_formed", r.is_input_well_formed())?;
    d.set_item("first_diff", r.first_diff)?;
    // Expose the rendered HTMLs for diagnosis — they can be large
    // (millions of chars on corpus docs) but are cheap to skip on
    // the Python side if not needed.
    d.set_item("pd_input_html", r.pd_input_html)?;
    d.set_item("pd_output_html", r.pd_output_html)?;
    d.set_item("cm_input_html", r.cm_input_html)?;
    d.set_item("cm_output_html", r.cm_output_html)?;
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Tests — small curated fixture set covering every Phase A concern.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert that `format_parsed(input)` passes the dual
    /// verifier (parsers agree on input, parsers agree on output,
    /// both renders identical).
    fn assert_formatter_preserves_preview(input: &str) {
        let out = format_parsed(input);
        let r = dual_verify(input, &out);
        assert!(
            r.is_well_formed_and_identical(),
            "\n=== INPUT ===\n{}\n=== OUTPUT ===\n{}\n=== DIFF ===\n{:?}\n",
            input,
            out,
            r.first_diff,
        );
    }

    // --- Paragraph reflow: soft-wrap prose → one line ---

    #[test]
    fn fx_paragraph_soft_wrapped_prose_unwraps() {
        let input = "This is a\nsoft-wrapped\nparagraph of prose.\n\nSecond paragraph\non two lines.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_paragraph_with_hard_break_preserved() {
        // Two trailing spaces before newline = `<br>` in preview.
        let input = "first line  \nsecond line\n\nnext paragraph.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_paragraph_with_backslash_hard_break_preserved() {
        let input = "first line\\\nsecond line\n\nnext paragraph.\n";
        assert_formatter_preserves_preview(input);
    }

    // --- GFM tables ---

    #[test]
    fn fx_gfm_table_with_long_dash_separator() {
        let input = "| A | B |\n| ---------- | ---------- |\n| 1 | 2 |\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_gfm_table_with_alignment_colons() {
        let input = "| a | b | c | d |\n| :---- | -----: | :----: | ---- |\n| 1 | 2 | 3 | 4 |\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_gfm_optional_pipe_table() {
        // GFM allows tables without leading/trailing pipes.
        let input = "a | b\n--- | ---\n1 | 2\n";
        assert_formatter_preserves_preview(input);
    }

    // --- HR thematic breaks ---

    #[test]
    fn fx_hr_long_dash_run() {
        let input = "before\n\n----------\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_hr_underscore_run() {
        let input = "before\n\n__________\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_hr_asterisk_run() {
        let input = "before\n\n**********\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Escaped underscores (must NOT become HR) ---

    #[test]
    fn fx_escaped_underscores_stay_literal() {
        let input = "before para.\n\n\\_\\_\\_\\_\\_\\_\\_\\_\n\nafter para.\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Fenced code blocks ---

    #[test]
    fn fx_fenced_code_content_preserved() {
        let input = "before\n\n```\nsome code\nwith -----\nand ```` inline\n```\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_fenced_code_with_info_string() {
        let input = "```rust\nfn main() {}\n```\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Blockquotes ---

    #[test]
    fn fx_blockquote_single_line() {
        let input = "> quoted text\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_blockquote_multi_line() {
        let input = "> first line\n> second line\n> third line\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Lists ---

    #[test]
    fn fx_tight_unordered_list() {
        let input = "- alpha\n- beta\n- gamma\n\nafter.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_loose_unordered_list() {
        let input = "- alpha\n\n- beta\n\n- gamma\n\nafter.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_ordered_list() {
        let input = "1. first\n2. second\n3. third\n\nafter.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_nested_list() {
        let input = "- outer\n  - inner\n  - inner2\n- outer2\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Headings ---

    #[test]
    fn fx_atx_headings() {
        let input = "# H1\n\n## H2\n\n### H3\n\nbody.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_setext_heading_level1() {
        let input = "Heading text\n============\n\nbody.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_setext_heading_level2() {
        let input = "Heading text\n------------\n\nbody.\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Mixed / realistic Greek corpus shapes ---

    #[test]
    fn fx_greek_paragraph_soft_wrapped() {
        let input = "Το παρόν έργο\nαδειοδοτείται υπό\nτους όρους της άδειας\nCreative Commons.\n\nΤέλος κειμένου.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_mixed_heading_table_prose() {
        let input = concat!(
            "# Top heading\n\n",
            "A soft-wrapped\nparagraph of prose.\n\n",
            "| Col A | Col B |\n| ---------- | ---------- |\n| 1 | 2 |\n\n",
            "-----------\n\n",
            "## Section two\n\n",
            "- item alpha\n- item beta\n\n",
            "Final\nparagraph\nsoft-wrapped.\n",
        );
        assert_formatter_preserves_preview(input);
    }

    // --- Indented code block ---

    #[test]
    fn fx_indented_code_block() {
        let input = "paragraph\n\n    code line one\n    code line two\n\nafter\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Inline code with pipe (common GFM gotcha) ---

    #[test]
    fn fx_inline_code_with_pipe() {
        let input = "Use `foo | bar` to filter.\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Links and images ---

    #[test]
    fn fx_inline_link() {
        let input = "See [the docs](https://example.com) for details.\n";
        assert_formatter_preserves_preview(input);
    }

    #[test]
    fn fx_reference_link() {
        let input = "See [the docs][1] for details.\n\n[1]: https://example.com\n";
        assert_formatter_preserves_preview(input);
    }

    // --- Dual-verifier itself (negative controls) ---

    #[test]
    fn dual_verifier_catches_dropped_paragraph() {
        let input = "para1\n\npara2\n\npara3\n";
        let broken = "para1\n\npara3\n";
        let r = dual_verify(input, broken);
        assert!(!r.is_well_formed_and_identical());
        assert!(!r.pd_identity);
        assert!(!r.cm_identity);
    }

    #[test]
    fn dual_verifier_catches_fused_words() {
        let input = "alpha beta gamma\n";
        let broken = "alphabeta gamma\n";
        let r = dual_verify(input, broken);
        assert!(!r.is_well_formed_and_identical());
    }
}
