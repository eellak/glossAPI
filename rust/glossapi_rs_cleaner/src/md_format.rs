//! Dual-parser verifier (`dual_verify`) — used by Pilot B's checked
//! wrapper (`md_format_surgical::format_surgical_checked`) to confirm
//! that a Phase A rewrite preserved the rendered HTML preview.
//!
//! Historical note: this module also held Pilot A
//! (`format_parsed`, a whole-doc round-trip through comrak's
//! `format_commonmark`). Pilot A was abandoned per the 2026-04-25
//! cleanup plan — it over-normalized things outside the 3 target
//! transforms (50 of 66 audit failures traced to non-target
//! normalizations of list markers, link forms, escapes). Pilot B
//! (`md_format_surgical::format_surgical`) supersedes it; only the
//! dual-parser verifier remains here.
//!
//! What `dual_verify` checks:
//! - INPUT parser agreement: comrak and pulldown-cmark agree on what
//!   the input renders to (well-formedness signal — the doc isn't
//!   dialect-ambiguous).
//! - OUTPUT parser agreement: same for the rewrite output.
//! - Identity per parser: the rewrite preserves the rendered HTML
//!   under EACH parser independently.
//!
//! On any disagreement, `format_surgical_checked` ships the input
//! verbatim and records `phase_a_fallback_reason`.

use comrak::{nodes::AstNode, parse_document, Arena, Options};
use pulldown_cmark::{html as pd_html, Options as PdOptions, Parser as PdParser};

/// Build the comrak options the verifier uses to render input + output
/// to HTML for comparison. GFM extensions match what
/// `md_format_surgical` parses with so the comparison is on the same
/// dialect.
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
    let root: &AstNode = parse_document(&arena, md, &opts);
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
    let first_diff =
        if pd_identity && cm_identity && input_parser_agreement && output_parser_agreement {
            None
        } else {
            let (label, a, b) = if !input_parser_agreement {
                (
                    "parsers disagree on input (dialect-ambiguous)",
                    pd_in.as_str(),
                    cm_in.as_str(),
                )
            } else if !output_parser_agreement {
                (
                    "parsers disagree on output",
                    pd_out.as_str(),
                    cm_out.as_str(),
                )
            } else if !pd_identity {
                (
                    "pulldown-cmark sees input != output",
                    pd_in.as_str(),
                    pd_out.as_str(),
                )
            } else {
                (
                    "comrak sees input != output",
                    cm_in.as_str(),
                    cm_out.as_str(),
                )
            };
            let prefix = common_prefix_len(a, b);
            let mut snippet = String::new();
            snippet.push_str(label);
            snippet.push_str("\n  a: ");
            snippet.push_str(
                &a.chars()
                    .skip(prefix.saturating_sub(40))
                    .take(200)
                    .collect::<String>(),
            );
            snippet.push_str("\n  b: ");
            snippet.push_str(
                &b.chars()
                    .skip(prefix.saturating_sub(40))
                    .take(200)
                    .collect::<String>(),
            );
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
