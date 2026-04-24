//! MD-equivalence verification using pulldown-cmark as reference parser.
//!
//! Two verification modes:
//!
//! - **Strict (Phase A):** `verify_md_preview_equivalent` — asserts that
//!   an MD transform preserves preview rendering. `pandoc-render(input) ≡
//!   pandoc-render(output)`. Used for testing Phase A (md_module)
//!   transforms where this invariant MUST hold.
//!
//! - **Structural (Phase B):** `verify_md_structural` — asserts that a
//!   content-modifying transform preserves block structure and only
//!   deletes content (no reorderings, no fusions). Used for spot-
//!   checking the full cleaner on sample docs.
//!
//! Uses `pulldown-cmark` as the reference CommonMark/GFM parser —
//! battle-tested, used by `rustdoc`, streaming event API (low memory),
//! same spec as GitHub's renderer. HTML render via
//! `pulldown_cmark::html::push_html`.
//!
//! See `docs/MD_MODULE_ARCHITECTURE.md` for the full design context.

use pulldown_cmark::{html, Event, Options, Parser, Tag, TagEnd};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::md_module;
use crate::normalize;

/// Detailed report from a verification run. Boolean fields are what
/// tests normally assert on; the diagnostic strings exist to make
/// failures self-explanatory (copy-paste into an issue).
#[derive(Debug, Clone, Default)]
pub struct MdEquivalenceReport {
    /// HTML render of input vs output is identical (after whitespace
    /// normalization). Strongest preview-equivalence signal.
    pub html_render_equal: bool,
    /// Block-level event sequence (`Start(Heading)`, `Start(Paragraph)`,
    /// `Start(Table)`, etc.) matches in order. Catches cases where
    /// whitespace-trim made HTML match by coincidence.
    pub block_sequence_equal: bool,
    /// For each matched paragraph, the whitespace-tokenized text content
    /// is identical. Catches word fusion (the v6-11 NBSP bug).
    pub paragraph_text_equal: bool,
    /// For each matched table, row count and cell count match, and each
    /// cell's whitespace-tokenized content is identical.
    pub table_cells_equal: bool,
    /// Diagnostic: first mismatch description, if any.
    pub first_diff: Option<String>,
}

impl MdEquivalenceReport {
    /// All four checks passed — preview-render equivalence holds.
    pub fn is_strict_equivalent(&self) -> bool {
        self.html_render_equal
            && self.block_sequence_equal
            && self.paragraph_text_equal
            && self.table_cells_equal
    }
}

/// Structural verification report for Phase B outputs.
#[derive(Debug, Clone, Default)]
pub struct MdStructuralReport {
    /// Number and type of top-level block elements match.
    pub block_count_equal: bool,
    /// In each matched paragraph, output tokens are a MONOTONE
    /// SUBSEQUENCE of input tokens (permits deletions, disallows
    /// reorderings or fusions).
    pub paragraph_tokens_subsequence: bool,
    /// For each matched table, cell count per row matches AND each
    /// cell's output tokens are a subsequence of input tokens.
    pub table_cells_subsequence: bool,
    /// Per-CodeBlock: output lines are a subsequence of input lines
    /// (whitespace-preserved comparison, unlike paragraph tokens).
    pub code_blocks_preserved: bool,
    /// Percentage of input tokens retained in output (continuous
    /// metric; useful even when booleans pass).
    pub token_retention_pct: f64,
    pub first_diff: Option<String>,
    /// When `paragraph_tokens_subsequence` is false, categorizes the
    /// failure: fusion (v6-11 NBSP signature), injection (added
    /// content), reordering, or other.
    pub subsequence_failure_kind: Option<String>,
}

impl MdStructuralReport {
    pub fn is_structural_equivalent(&self) -> bool {
        self.block_count_equal
            && self.paragraph_tokens_subsequence
            && self.table_cells_subsequence
            && self.code_blocks_preserved
    }
}

/// Apply every NON-destructive cleaner transform to `md`. Used by the
/// structural verifier to canonicalize the INPUT before comparing
/// against cleaner output — otherwise cosmetic differences like
/// `&amp;` → `&`, `........(40 dots)` → `....(20)`, `-----` → `---`
/// etc. are misclassified as "injections."
///
/// Transforms applied (all token-semantic-preserving or invisible):
/// 1. HTML entity decode (`&amp;` → `&` etc.)
/// 2. Adobe Symbol PUA decode (U+F061 → α etc.)
/// 3. Soft-hyphen strip (U+00AD, invisible anyway)
/// 4. Per-line character fold (NBSP → space, ligatures → pairs,
///    Unicode whitespace variants → space, enclosed digits → ASCII)
/// 5. Dot-run normalization (tiered bucket collapse)
/// 6. Whitespace-run normalization (multi-space → tiered bucket)
/// 7. Ellipsis-run normalization
/// 8. HR thematic-break minimization (`-----` → `---`)
/// 9. GFM table separator minimization (`|-----|` → `|---|`)
/// 10. Paragraph reflow (soft-wrap `\n` → space within paragraphs)
///
/// NOT applied (destructive or content-removing):
/// - GLYPH strip
/// - Per-char allowlist filter
/// - Line-drop rules
/// - Rule-A/B filtering
///
/// The result is what the cleaner WOULD produce if every pass were
/// non-destructive.
fn canonicalize_for_verify(md: &str) -> String {
    // Step 1-3: content-level wave-2 preprocessing.
    let step1 = normalize::decode_html_entities(md);
    let step2 = normalize::decode_adobe_symbol_pua(&step1);
    let step3 = normalize::strip_soft_hyphens(&step2);

    // Step 4: per-line char fold + per-line normalizations.
    let mut per_line_out = String::with_capacity(step3.len());
    let lines: Vec<&str> = step3.split('\n').collect();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            per_line_out.push('\n');
        }
        let mut cur = line.to_string();
        if let Some(folded) = normalize::fold_line(&cur) {
            cur = folded;
        }
        if let Some(normed) = normalize::normalize_dot_runs(&cur) {
            cur = normed;
        }
        if let Some(normed) = normalize::normalize_whitespace_runs(&cur) {
            cur = normed;
        }
        if let Some(normed) = normalize::normalize_ellipsis_runs(&cur) {
            cur = normed;
        }
        per_line_out.push_str(&cur);
    }

    // Step 5: MD-syntax-aware Phase A (GFM sep min, HR min, reflow).
    md_module::normalize_md_syntax(&per_line_out)
}

fn gfm_options() -> Options {
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_FOOTNOTES);
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts
}

/// Collapse all runs of whitespace to a single space + trim ends. Used
/// to normalize HTML render output before equality check.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

fn render_html(md: &str) -> String {
    let parser = Parser::new_ext(md, gfm_options());
    let mut html = String::new();
    html::push_html(&mut html, parser);
    html
}

/// Block-level element kinds we compare. Inline emphasis / links are
/// folded into their parent block's text content — we don't enforce
/// that `**bold**` stays as `**bold**` specifically (different renderers
/// emit different inline detail; the HTML-equality check handles it).
#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockKind {
    Paragraph,
    Heading(u8),
    BlockQuote,
    CodeBlock,
    List(bool /* ordered */),
    Item,
    Table,
    TableHead,
    TableRow,
    TableCell,
    ThematicBreak,
    HtmlBlock,
    FootnoteDefinition,
}

fn tag_to_block_kind(tag: &Tag) -> Option<BlockKind> {
    match tag {
        Tag::Paragraph => Some(BlockKind::Paragraph),
        Tag::Heading { level, .. } => Some(BlockKind::Heading(*level as u8)),
        Tag::BlockQuote(_) => Some(BlockKind::BlockQuote),
        Tag::CodeBlock(_) => Some(BlockKind::CodeBlock),
        Tag::List(_) => {
            // List(Some(_)) is ordered; List(None) is unordered.
            if let Tag::List(start) = tag {
                Some(BlockKind::List(start.is_some()))
            } else {
                unreachable!()
            }
        }
        Tag::Item => Some(BlockKind::Item),
        Tag::Table(_) => Some(BlockKind::Table),
        Tag::TableHead => Some(BlockKind::TableHead),
        Tag::TableRow => Some(BlockKind::TableRow),
        Tag::TableCell => Some(BlockKind::TableCell),
        Tag::FootnoteDefinition(_) => Some(BlockKind::FootnoteDefinition),
        Tag::HtmlBlock => Some(BlockKind::HtmlBlock),
        _ => None,
    }
}

/// Flatten an MD doc to a linear sequence of block-kind starts.
fn block_sequence(md: &str) -> Vec<BlockKind> {
    let mut seq = Vec::new();
    for ev in Parser::new_ext(md, gfm_options()) {
        match ev {
            Event::Start(tag) => {
                if let Some(k) = tag_to_block_kind(&tag) {
                    seq.push(k);
                }
            }
            Event::Rule => seq.push(BlockKind::ThematicBreak),
            _ => {}
        }
    }
    seq
}

/// Extract one whitespace-tokenized vector per paragraph block, in
/// source order. Text inside a paragraph (including inline formatting)
/// is concatenated before tokenization. Link/image URLs ARE included
/// in the token stream so that a cleaner that silently rewrites a URL
/// is detected.
fn paragraph_tokens(md: &str) -> Vec<Vec<String>> {
    let mut paragraphs: Vec<Vec<String>> = Vec::new();
    let mut current: Option<String> = None;
    for ev in Parser::new_ext(md, gfm_options()) {
        match ev {
            Event::Start(Tag::Paragraph) => {
                current = Some(String::new());
            }
            Event::End(TagEnd::Paragraph) => {
                if let Some(buf) = current.take() {
                    paragraphs.push(
                        buf.split_whitespace()
                            .map(|s| s.to_string())
                            .collect(),
                    );
                }
            }
            Event::Text(t) | Event::Code(t) | Event::Html(t) => {
                if let Some(buf) = current.as_mut() {
                    buf.push_str(&t);
                    buf.push(' ');
                }
            }
            // Link / image URLs are meaningful content. Append them as
            // space-separated tokens so they show up in the token
            // sequence and a silent URL rewrite fails verification.
            Event::Start(Tag::Link { dest_url, title, .. })
            | Event::Start(Tag::Image { dest_url, title, .. }) => {
                if let Some(buf) = current.as_mut() {
                    buf.push_str(&dest_url);
                    buf.push(' ');
                    if !title.is_empty() {
                        buf.push_str(&title);
                        buf.push(' ');
                    }
                }
            }
            _ => {}
        }
    }
    paragraphs
}

/// Extract table structure: `Vec<Table>` where each Table is
/// `Vec<Row>` and each Row is `Vec<Cell tokens>`. Link/image URLs
/// inside cells are included in the token stream.
fn table_structure(md: &str) -> Vec<Vec<Vec<Vec<String>>>> {
    let mut tables: Vec<Vec<Vec<Vec<String>>>> = Vec::new();
    let mut cur_table: Option<Vec<Vec<Vec<String>>>> = None;
    let mut cur_row: Option<Vec<Vec<String>>> = None;
    let mut cur_cell_buf: Option<String> = None;
    for ev in Parser::new_ext(md, gfm_options()) {
        match ev {
            Event::Start(Tag::Table(_)) => cur_table = Some(Vec::new()),
            Event::End(TagEnd::Table) => {
                if let Some(t) = cur_table.take() {
                    tables.push(t);
                }
            }
            Event::Start(Tag::TableRow) | Event::Start(Tag::TableHead) => {
                cur_row = Some(Vec::new());
            }
            Event::End(TagEnd::TableRow) | Event::End(TagEnd::TableHead) => {
                if let (Some(row), Some(tbl)) = (cur_row.take(), cur_table.as_mut()) {
                    tbl.push(row);
                }
            }
            Event::Start(Tag::TableCell) => cur_cell_buf = Some(String::new()),
            Event::End(TagEnd::TableCell) => {
                if let (Some(buf), Some(row)) =
                    (cur_cell_buf.take(), cur_row.as_mut())
                {
                    row.push(
                        buf.split_whitespace()
                            .map(|s| s.to_string())
                            .collect(),
                    );
                }
            }
            Event::Text(t) | Event::Code(t) | Event::Html(t) => {
                if let Some(buf) = cur_cell_buf.as_mut() {
                    buf.push_str(&t);
                    buf.push(' ');
                }
            }
            Event::Start(Tag::Link { dest_url, title, .. })
            | Event::Start(Tag::Image { dest_url, title, .. }) => {
                if let Some(buf) = cur_cell_buf.as_mut() {
                    buf.push_str(&dest_url);
                    buf.push(' ');
                    if !title.is_empty() {
                        buf.push_str(&title);
                        buf.push(' ');
                    }
                }
            }
            _ => {}
        }
    }
    tables
}

/// Extract code block contents as lines (preserves indentation and
/// whitespace, unlike the paragraph tokenizer). One `Vec<String>` of
/// lines per `CodeBlock`, in source order.
fn code_block_lines(md: &str) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut current: Option<String> = None;
    for ev in Parser::new_ext(md, gfm_options()) {
        match ev {
            Event::Start(Tag::CodeBlock(_)) => current = Some(String::new()),
            Event::End(TagEnd::CodeBlock) => {
                if let Some(buf) = current.take() {
                    out.push(buf.lines().map(String::from).collect());
                }
            }
            Event::Text(t) => {
                if let Some(buf) = current.as_mut() {
                    buf.push_str(&t);
                }
            }
            _ => {}
        }
    }
    out
}

/// Classification of a paragraph-subsequence failure — distinguishes
/// the underlying cause so scorecard output is directly actionable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsequenceFailureKind {
    /// Output has a token not present in input (injection or fusion).
    Injection,
    /// Output has adjacent input tokens concatenated into one token
    /// (the v6-11 NBSP-strip signature). Detected when an output token
    /// is not in input but IS a concat of 2+ adjacent input tokens.
    Fusion,
    /// Input tokens are all present in output, but out of order.
    Reordering,
    /// Some other combination — couldn't be cleanly classified.
    Other,
}

impl SubsequenceFailureKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Injection => "injection",
            Self::Fusion => "fusion",
            Self::Reordering => "reordering",
            Self::Other => "other",
        }
    }
}

/// Classify WHY output tokens are not a monotone subsequence of input
/// tokens. Returns None if `output` IS a subsequence (no failure).
pub fn classify_subsequence_failure(
    input: &[String],
    output: &[String],
) -> Option<SubsequenceFailureKind> {
    if is_subsequence(output, input) {
        return None;
    }
    // Is every output token AT LEAST in the input set? If no, injection.
    let input_set: std::collections::HashSet<&String> = input.iter().collect();
    let missing_in_input: Vec<&String> =
        output.iter().filter(|t| !input_set.contains(t)).collect();
    if !missing_in_input.is_empty() {
        // Fusion signature: the out-of-input token is a concat of 2+
        // adjacent input tokens.
        for missing in &missing_in_input {
            if is_concat_of_adjacent_input_tokens(missing, input) {
                return Some(SubsequenceFailureKind::Fusion);
            }
        }
        return Some(SubsequenceFailureKind::Injection);
    }
    // All output tokens ARE in the input set, but not as a subsequence
    // → order changed.
    Some(SubsequenceFailureKind::Reordering)
}

/// Check if `needle` is exactly the concatenation of some window of
/// adjacent tokens from `input` (with empty separator — as NBSP-strip
/// would produce).
fn is_concat_of_adjacent_input_tokens(needle: &str, input: &[String]) -> bool {
    for start in 0..input.len() {
        let mut acc = String::new();
        for i in start..input.len() {
            acc.push_str(&input[i]);
            if acc.len() > needle.len() {
                break;
            }
            // Need at least 2 tokens to call it a "fusion".
            if acc == needle && i > start {
                return true;
            }
        }
    }
    false
}

/// Strict Phase A verification: preview render must be identical.
pub fn verify_md_preview_equivalent(input: &str, output: &str) -> MdEquivalenceReport {
    let mut r = MdEquivalenceReport::default();

    // 1. HTML render equality.
    let html_in = collapse_ws(&render_html(input));
    let html_out = collapse_ws(&render_html(output));
    r.html_render_equal = html_in == html_out;
    if !r.html_render_equal && r.first_diff.is_none() {
        r.first_diff = Some(format!(
            "html render differs\n  in:  {}\n  out: {}",
            html_in.chars().take(200).collect::<String>(),
            html_out.chars().take(200).collect::<String>(),
        ));
    }

    // 2. Block sequence equality.
    let seq_in = block_sequence(input);
    let seq_out = block_sequence(output);
    r.block_sequence_equal = seq_in == seq_out;
    if !r.block_sequence_equal && r.first_diff.is_none() {
        r.first_diff = Some(format!(
            "block sequence differs\n  in:  {:?}\n  out: {:?}",
            seq_in, seq_out
        ));
    }

    // 3. Per-paragraph text tokens.
    let par_in = paragraph_tokens(input);
    let par_out = paragraph_tokens(output);
    r.paragraph_text_equal = par_in == par_out;
    if !r.paragraph_text_equal && r.first_diff.is_none() {
        let idx = par_in
            .iter()
            .zip(par_out.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(par_in.len().min(par_out.len()));
        let a = par_in.get(idx).cloned().unwrap_or_default();
        let b = par_out.get(idx).cloned().unwrap_or_default();
        r.first_diff = Some(format!(
            "paragraph {} text differs\n  in:  {:?}\n  out: {:?}",
            idx, a, b
        ));
    }

    // 4. Table cells.
    let tbl_in = table_structure(input);
    let tbl_out = table_structure(output);
    r.table_cells_equal = tbl_in == tbl_out;
    if !r.table_cells_equal && r.first_diff.is_none() {
        r.first_diff = Some(format!(
            "table structure differs: {} tables in, {} tables out",
            tbl_in.len(),
            tbl_out.len()
        ));
    }

    r
}

/// Structural Phase B verification: output is a content-subset of input,
/// structure preserved.
///
/// **Input pre-canonicalization** (2026-04-24): before extracting
/// tokens, runs `canonicalize_for_verify(input)` so that entity decode,
/// HR/GFM-sep minimization, Unicode-whitespace folding, and other
/// non-destructive cleaner transforms don't produce misclassified
/// "injection" failures. Without this, a cleaner that decodes `&amp;`
/// to `&` would appear to have "injected" the token `&` because it
/// wasn't literally in the raw input. With this, both sides see `&`.
pub fn verify_md_structural(input: &str, output: &str) -> MdStructuralReport {
    let mut r = MdStructuralReport::default();

    // Canonicalize input — apply the cleaner's non-destructive
    // transforms so diffs reflect real content changes only.
    let input_canon_owned = canonicalize_for_verify(input);
    let input = input_canon_owned.as_str();

    let seq_in = block_sequence(input);
    let seq_out = block_sequence(output);
    r.block_count_equal = seq_in == seq_out;
    if !r.block_count_equal && r.first_diff.is_none() {
        r.first_diff = Some(format!(
            "block sequence differs ({} in vs {} out)",
            seq_in.len(),
            seq_out.len()
        ));
    }

    // Paragraph tokens: output must be a subsequence of input.
    // Classify failure kind (fusion / reordering / injection / other)
    // when the check fails — makes scorecard output directly actionable.
    let par_in = paragraph_tokens(input);
    let par_out = paragraph_tokens(output);
    let mut all_pass = true;
    let mut tokens_in_total = 0usize;
    let mut tokens_out_total = 0usize;
    for (i, (a, b)) in par_in.iter().zip(par_out.iter()).enumerate() {
        tokens_in_total += a.len();
        tokens_out_total += b.len();
        if !is_subsequence(b, a) {
            all_pass = false;
            let kind = classify_subsequence_failure(a, b)
                .unwrap_or(SubsequenceFailureKind::Other);
            if r.first_diff.is_none() {
                r.first_diff = Some(format!(
                    "paragraph {} subsequence failure ({}): in_len={} out_len={}",
                    i,
                    kind.as_str(),
                    a.len(),
                    b.len(),
                ));
                r.subsequence_failure_kind = Some(kind.as_str().to_string());
            }
        }
    }
    r.paragraph_tokens_subsequence = all_pass && par_in.len() == par_out.len();
    r.token_retention_pct = if tokens_in_total == 0 {
        1.0
    } else {
        tokens_out_total as f64 / tokens_in_total as f64
    };

    // Table cells: same structure + subsequence per cell.
    let tbl_in = table_structure(input);
    let tbl_out = table_structure(output);
    let mut tbl_pass = tbl_in.len() == tbl_out.len();
    if tbl_pass {
        'outer: for (table_in, table_out) in tbl_in.iter().zip(tbl_out.iter()) {
            if table_in.len() != table_out.len() {
                tbl_pass = false;
                if r.first_diff.is_none() {
                    r.first_diff = Some("table row count differs".to_string());
                }
                break;
            }
            for (row_in, row_out) in table_in.iter().zip(table_out.iter()) {
                if row_in.len() != row_out.len() {
                    tbl_pass = false;
                    if r.first_diff.is_none() {
                        r.first_diff = Some("table cell count differs".to_string());
                    }
                    break 'outer;
                }
                for (cell_in, cell_out) in row_in.iter().zip(row_out.iter()) {
                    if !is_subsequence(cell_out, cell_in) {
                        tbl_pass = false;
                        if r.first_diff.is_none() {
                            let kind = classify_subsequence_failure(cell_in, cell_out)
                                .unwrap_or(SubsequenceFailureKind::Other);
                            r.first_diff = Some(format!(
                                "table cell subsequence failure ({})",
                                kind.as_str()
                            ));
                        }
                        break 'outer;
                    }
                }
            }
        }
    }
    r.table_cells_subsequence = tbl_pass;

    // Code blocks: output lines are a subsequence of input lines.
    // Code is whitespace-sensitive; line-based check (not whitespace-
    // tokenized) catches accidental re-indentation.
    let code_in = code_block_lines(input);
    let code_out = code_block_lines(output);
    let mut code_pass = code_in.len() == code_out.len();
    if code_pass {
        for (a, b) in code_in.iter().zip(code_out.iter()) {
            if !is_subsequence(b, a) {
                code_pass = false;
                if r.first_diff.is_none() {
                    r.first_diff =
                        Some("code block lines NOT a subsequence".to_string());
                }
                break;
            }
        }
    } else if r.first_diff.is_none() {
        r.first_diff = Some(format!(
            "code block count differs ({} in vs {} out)",
            code_in.len(),
            code_out.len(),
        ));
    }
    r.code_blocks_preserved = code_pass;

    r
}

/// Test whether `needle` is a monotone subsequence of `haystack`.
fn is_subsequence(needle: &[String], haystack: &[String]) -> bool {
    let mut h_iter = haystack.iter();
    for tok in needle {
        let found = h_iter.any(|h| h == tok);
        if !found {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// PyO3 bindings — exposed so Python driver can spot-check docs.
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn verify_md_preview_equivalent_py(
    py: Python<'_>,
    input: &str,
    output: &str,
) -> PyResult<PyObject> {
    let r = verify_md_preview_equivalent(input, output);
    let d = PyDict::new(py);
    d.set_item("html_render_equal", r.html_render_equal)?;
    d.set_item("block_sequence_equal", r.block_sequence_equal)?;
    d.set_item("paragraph_text_equal", r.paragraph_text_equal)?;
    d.set_item("table_cells_equal", r.table_cells_equal)?;
    d.set_item("is_strict_equivalent", r.is_strict_equivalent())?;
    d.set_item("first_diff", r.first_diff)?;
    Ok(d.into())
}

#[pyfunction]
pub fn verify_md_structural_py(
    py: Python<'_>,
    input: &str,
    output: &str,
) -> PyResult<PyObject> {
    let r = verify_md_structural(input, output);
    let d = PyDict::new(py);
    d.set_item("block_count_equal", r.block_count_equal)?;
    d.set_item(
        "paragraph_tokens_subsequence",
        r.paragraph_tokens_subsequence,
    )?;
    d.set_item("table_cells_subsequence", r.table_cells_subsequence)?;
    d.set_item("code_blocks_preserved", r.code_blocks_preserved)?;
    d.set_item("token_retention_pct", r.token_retention_pct)?;
    d.set_item("is_structural_equivalent", r.is_structural_equivalent())?;
    d.set_item("first_diff", r.first_diff)?;
    d.set_item("subsequence_failure_kind", r.subsequence_failure_kind)?;
    Ok(d.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- identity: any doc verified against itself should pass strict ---

    #[test]
    fn identity_passes_strict_on_simple_prose() {
        let doc = "# Hello\n\nThis is a paragraph.\n\nAnd another.\n";
        let r = verify_md_preview_equivalent(doc, doc);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn identity_passes_strict_on_gfm_table() {
        let doc = "| a | b |\n| --- | --- |\n| 1 | 2 |\n";
        let r = verify_md_preview_equivalent(doc, doc);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn identity_passes_strict_on_headings_and_lists() {
        let doc = "# Title\n\n## Section\n\n- item one\n- item two\n- item three\n\nbody\n";
        let r = verify_md_preview_equivalent(doc, doc);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- Phase A transforms: assert equivalence ---

    #[test]
    fn reflow_preserves_strict_equivalence() {
        let input = "This is a\nparagraph that\nis soft-wrapped.\n";
        let output = "This is a paragraph that is soft-wrapped.\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn gfm_table_sep_min_preserves_strict_equivalence() {
        let input = "| a | b |\n| -------- | -------- |\n| 1 | 2 |\n";
        let output = "| a | b |\n| --- | --- |\n| 1 | 2 |\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn hr_min_preserves_strict_equivalence() {
        let input = "before\n\n----------\n\nafter\n";
        let output = "before\n\n---\n\nafter\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- Phase A violations: should be caught ---

    #[test]
    fn dropped_paragraph_fails_strict() {
        let input = "para1\n\npara2\n\npara3\n";
        let output = "para1\n\npara3\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(!r.is_strict_equivalent());
        assert!(!r.block_sequence_equal);
    }

    #[test]
    fn fused_words_fails_paragraph_token_check() {
        // The v6-11 NBSP-strip bug — words fused into one.
        let input = "Η εργασία αυτή\n";
        let output = "Ηεργασίααυτή\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(!r.paragraph_text_equal);
    }

    #[test]
    fn reordered_tokens_fails_paragraph_check() {
        let input = "alpha beta gamma\n";
        let output = "gamma alpha beta\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(!r.paragraph_text_equal);
    }

    #[test]
    fn heading_level_change_fails_block_sequence() {
        let input = "# title\n";
        let output = "## title\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(!r.block_sequence_equal);
    }

    // --- Phase B structural verifier ---

    #[test]
    fn structural_accepts_token_deletion() {
        // Removing some tokens (e.g. GLYPH markers) is allowed. Use a
        // clearly-tokenizable marker so the retention fraction is
        // parser-stable (GLYPH<216> can be parsed in parser-specific
        // ways depending on HTML-inline handling).
        let input = "alpha beta gamma delta\n";
        let output = "alpha gamma\n";
        let r = verify_md_structural(input, output);
        assert!(r.is_structural_equivalent(), "{:?}", r);
        assert!(r.token_retention_pct < 1.0);
        assert!(r.token_retention_pct >= 0.5);
    }

    #[test]
    fn structural_rejects_fusion() {
        let input = "Η εργασία αυτή\n";
        let output = "Ηεργασίααυτή\n";
        let r = verify_md_structural(input, output);
        assert!(!r.paragraph_tokens_subsequence);
    }

    #[test]
    fn structural_rejects_reordering() {
        let input = "alpha beta gamma delta\n";
        let output = "gamma delta alpha beta\n";
        let r = verify_md_structural(input, output);
        assert!(!r.paragraph_tokens_subsequence);
    }

    #[test]
    fn structural_rejects_added_content() {
        let input = "alpha beta\n";
        let output = "alpha injected beta\n";
        let r = verify_md_structural(input, output);
        assert!(!r.paragraph_tokens_subsequence);
    }

    #[test]
    fn structural_reports_retention_fraction() {
        // 3 of 6 input tokens retained.
        let input = "a b c d e f\n";
        let output = "a c e\n";
        let r = verify_md_structural(input, output);
        assert!(r.is_structural_equivalent());
        assert!((r.token_retention_pct - 0.5).abs() < 1e-9);
    }

    // --- helper: is_subsequence ---

    #[test]
    fn is_subsequence_basic() {
        let make = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(is_subsequence(&make(&["a", "c"]), &make(&["a", "b", "c"])));
        assert!(is_subsequence(&make(&[]), &make(&["a", "b", "c"])));
        assert!(!is_subsequence(&make(&["c", "a"]), &make(&["a", "b", "c"])));
        assert!(!is_subsequence(&make(&["d"]), &make(&["a", "b", "c"])));
    }

    // --- URL capture (wave-3 enhancement) ---

    #[test]
    fn url_change_detected_by_strict_verifier() {
        let input = "See [the site](https://example.com/a) for details.\n";
        let output = "See [the site](https://example.com/b) for details.\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(
            !r.is_strict_equivalent(),
            "URL change in link should be detected: {:?}",
            r
        );
    }

    #[test]
    fn image_url_change_detected_by_strict_verifier() {
        let input = "Image: ![alt](https://cdn.example.com/img/a.png)\n";
        let output = "Image: ![alt](https://cdn.example.com/img/b.png)\n";
        let r = verify_md_preview_equivalent(input, output);
        assert!(
            !r.is_strict_equivalent(),
            "image URL change should be detected: {:?}",
            r
        );
    }

    #[test]
    fn url_change_detected_by_structural_verifier() {
        // URL treated as a token; change = token injection.
        let input = "Visit [here](https://a.example.com).\n";
        let output = "Visit [here](https://b.example.com).\n";
        let r = verify_md_structural(input, output);
        assert!(
            !r.is_structural_equivalent(),
            "URL change should fail structural: {:?}",
            r
        );
    }

    // --- Subsequence failure classification ---

    #[test]
    fn classify_detects_fusion() {
        // NBSP-strip signature: adjacent input tokens concat'd.
        let input = vec!["Η".to_string(), "εργασία".to_string(), "αυτή".to_string()];
        let output = vec!["Ηεργασία".to_string(), "αυτή".to_string()];
        let kind = classify_subsequence_failure(&input, &output);
        assert_eq!(kind, Some(SubsequenceFailureKind::Fusion));
    }

    #[test]
    fn classify_detects_reordering() {
        let input = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let output = vec!["c".to_string(), "a".to_string(), "b".to_string()];
        let kind = classify_subsequence_failure(&input, &output);
        assert_eq!(kind, Some(SubsequenceFailureKind::Reordering));
    }

    #[test]
    fn classify_detects_injection() {
        let input = vec!["a".to_string(), "b".to_string()];
        let output = vec!["a".to_string(), "INJECTED".to_string(), "b".to_string()];
        let kind = classify_subsequence_failure(&input, &output);
        assert_eq!(kind, Some(SubsequenceFailureKind::Injection));
    }

    #[test]
    fn classify_returns_none_when_subsequence() {
        let input = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let output = vec!["a".to_string(), "c".to_string()];
        let kind = classify_subsequence_failure(&input, &output);
        assert_eq!(kind, None);
    }

    // --- Code block line-based comparison ---

    #[test]
    fn code_block_preserved_passes() {
        let md = "before\n\n```\nfn foo() {\n    42\n}\n```\n\nafter\n";
        let r = verify_md_structural(md, md);
        assert!(r.code_blocks_preserved, "{:?}", r);
        assert!(r.is_structural_equivalent());
    }

    #[test]
    fn code_block_line_dropped_allowed() {
        // Deleting a line in a code block counts as a subsequence.
        let input = "```\nline one\nline two\nline three\n```\n";
        let output = "```\nline one\nline three\n```\n";
        let r = verify_md_structural(input, output);
        assert!(r.code_blocks_preserved, "{:?}", r);
    }

    #[test]
    fn code_block_line_changed_flagged() {
        // Modifying a line breaks subsequence — new line not in input.
        let input = "```\nfn foo() {\n    42\n}\n```\n";
        let output = "```\nfn foo() {\n    43\n}\n```\n";
        let r = verify_md_structural(input, output);
        assert!(
            !r.code_blocks_preserved,
            "changed line should be caught: {:?}",
            r
        );
    }

    #[test]
    fn code_block_reindented_flagged() {
        // Re-indenting changes the line content → subsequence fails.
        let input = "```\n    indented line\n    another\n```\n";
        let output = "```\nindented line\nanother\n```\n";
        let r = verify_md_structural(input, output);
        assert!(
            !r.code_blocks_preserved,
            "reindent should be caught: {:?}",
            r
        );
    }

    // --- Structural report now exposes classification ---

    #[test]
    fn structural_report_exposes_failure_kind_on_fusion() {
        let input = "Η εργασία αυτή\n";
        let output = "Ηεργασία αυτή\n";
        let r = verify_md_structural(input, output);
        assert!(!r.is_structural_equivalent());
        assert_eq!(r.subsequence_failure_kind.as_deref(), Some("fusion"));
    }

    #[test]
    fn structural_report_exposes_failure_kind_on_injection() {
        let input = "alpha beta\n";
        let output = "alpha injected beta\n";
        let r = verify_md_structural(input, output);
        assert!(!r.is_structural_equivalent());
        assert_eq!(r.subsequence_failure_kind.as_deref(), Some("injection"));
    }

    // --- Input pre-canonicalization (wave-3 enhancement) ---

    #[test]
    fn canonicalization_makes_entity_decode_invisible() {
        // `&amp;` in input, `&` in output — cleaner did entity decode.
        // After pre-canonicalization of input, both see `&`.
        let input = "Text with &amp; entity.\n";
        let output = "Text with & entity.\n";
        let r = verify_md_structural(input, output);
        assert!(
            r.is_structural_equivalent(),
            "entity-decode should pass after input canonicalization: {:?}",
            r
        );
    }

    #[test]
    fn canonicalization_makes_hr_min_invisible() {
        let input = "before\n\n-----------\n\nafter\n";
        let output = "before\n\n---\n\nafter\n";
        let r = verify_md_structural(input, output);
        assert!(
            r.is_structural_equivalent(),
            "HR min should pass: {:?}",
            r
        );
    }

    #[test]
    fn canonicalization_makes_gfm_sep_min_invisible() {
        let input = "| a | b |\n| --------- | --------- |\n| 1 | 2 |\n";
        let output = "| a | b |\n| --- | --- |\n| 1 | 2 |\n";
        let r = verify_md_structural(input, output);
        assert!(
            r.is_structural_equivalent(),
            "GFM sep min should pass: {:?}",
            r
        );
    }

    #[test]
    fn canonicalization_makes_nbsp_fold_invisible() {
        // Input has NBSP between words; output has regular space.
        // Cleaner folded NBSP (post v6-11 fix). After canonicalization,
        // both sides see `Η εργασία`.
        let input = "Η\u{00A0}εργασία\u{00A0}αυτή\n";
        let output = "Η εργασία αυτή\n";
        let r = verify_md_structural(input, output);
        assert!(
            r.is_structural_equivalent(),
            "NBSP fold should pass: {:?}",
            r
        );
    }

    #[test]
    fn canonicalization_does_not_hide_real_content_changes() {
        // Regression: pre-canonicalization should NOT mask a genuinely
        // destructive cleaner change.
        let input = "alpha beta gamma delta epsilon\n";
        let output = "alpha gamma epsilon\n"; // dropped beta, delta — fine
        let r = verify_md_structural(input, output);
        assert!(r.is_structural_equivalent(), "deletion should pass: {:?}", r);

        // But ADDING content or REORDERING must still fail.
        let output_reorder = "gamma alpha beta delta epsilon\n";
        let r2 = verify_md_structural(input, output_reorder);
        assert!(!r2.is_structural_equivalent(), "reorder must fail: {:?}", r2);
    }
}
