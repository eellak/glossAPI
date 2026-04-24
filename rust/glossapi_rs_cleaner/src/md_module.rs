//! MD-syntax-aware transforms (Phase A).
//!
//! All transforms in this module share a single invariant:
//!
//!   `pandoc-render(input) == pandoc-render(output)`
//!
//! That is — a downstream reader viewing the MD via any spec-compliant
//! renderer sees the same preview before and after. Raw chars DO change
//! by design: we linearize soft-wrapped paragraphs, minimize redundant
//! separator runs, canonicalize GFM table separator rows, etc. Those
//! changes are not "content preserving" in the strict char-sequence
//! sense, but they are preview-preserving, which is the stronger
//! guarantee a pretraining corpus needs: a reader using a preview can't
//! see any difference, while the raw form used by the tokenizer is
//! shorter / less fragmented / more regular.
//!
//! Consumers:
//! - `cleaning_module::core_clean_text_with_stats` runs this as a
//!   pre-pass before any content-destructive transform.
//! - `md_verify` runs these transforms inside tests that assert the
//!   invariant above holds (using pulldown-cmark as the reference
//!   MD parser).
//!
//! Organization follows the `feedback_group_cleaner_features_by_text_type`
//! rule: all transforms whose correctness depends on CommonMark / GFM
//! grammar live here; ONE detector per concept, many consumers.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

use crate::normalize;

lazy_static! {
    /// Matches a standalone CommonMark horizontal-rule (`<hr/>` in the
    /// rendered output) — runs of `-` / `_` / `*` of length ≥4 on a line
    /// that contains only those chars (plus optional leading/trailing
    /// whitespace). Also matches the markdown-escaped underscore run
    /// `\_\_\_\_` that appears in EU legislative corpus docs — the
    /// escape preserves the same thematic-break render.
    ///
    /// Threshold is ≥4 (not ≥3) so the rewriter only fires when a
    /// collapse actually reduces characters — `---` / `___` / `***`
    /// are already canonical and produce no-op rewrites, which is fine,
    /// but keeping the threshold at ≥4 documents that intent.
    ///
    /// Intentionally does NOT match `=` runs (`====` is a setext heading
    /// marker, not an HR), Unicode em-dash / horizontal-bar / box-drawing
    /// (these parse as literal paragraphs, not HRs, under CommonMark).
    /// Transforming them to `---` would CHANGE preview rendering and
    /// violate the Phase A invariant — verifier catches it.
    pub static ref SEPARATOR_LINE_REGEX: Regex = Regex::new(
        r"^[ \t]*(?:-{4,}|_{4,}|(?:\\_){4,}|\*{4,})[ \t]*$",
    )
    .unwrap();

    /// CommonMark thematic-break recognizer used for reflow hard-break
    /// detection. Uses the spec threshold of ≥3 chars (different from
    /// `SEPARATOR_LINE_REGEX`, which only fires on ≥4 runs because it
    /// is the *rewrite* rule). Recognizing ≥3 here is required so:
    ///
    /// - Our own canonical output `---` (produced by
    ///   `normalize_separator_line`) is still detected as a hard break
    ///   by `reflow_paragraphs`, preventing the cleaner from fusing
    ///   the HR line with an adjacent paragraph.
    /// - Setext heading markers `---` / `===` are preserved — joining
    ///   `---` with a following paragraph would demote an H2 to a
    ///   regular paragraph, breaking preview.
    static ref HR_HARD_BREAK_REGEX: Regex = Regex::new(
        r"^[ \t]{0,3}(?:-{3,}|_{3,}|(?:\\_){3,}|\*{3,}|={3,})[ \t]*$",
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// CommonMark indentation helper.
// ---------------------------------------------------------------------------

/// Column width of the line's leading whitespace under CommonMark's
/// indentation rule.
///
/// Per CommonMark: a space advances the column by 1; a tab advances to
/// the next multiple of 4. `≥4` columns of leading whitespace triggers
/// an indented code block, which is a different leaf-block type than
/// any of the markers Phase A rewrites (thematic break, GFM table
/// separator, fenced code opener). Our detectors must bail out at that
/// threshold or they'll corrupt indented-code content.
///
/// Returns the column position of the first non-whitespace char (or
/// the total column width if the line is whitespace-only).
pub fn leading_columns(line: &str) -> usize {
    let mut col: usize = 0;
    for c in line.chars() {
        match c {
            ' ' => col += 1,
            '\t' => col = (col / 4 + 1) * 4,
            _ => return col,
        }
    }
    col
}

// ---------------------------------------------------------------------------
// HR (thematic break) minimization.
// ---------------------------------------------------------------------------

/// Collapse a standalone CommonMark thematic-break line (runs of
/// `-` / `_` / `*`, or the markdown-escaped-underscore form
/// `\_\_\_\_`) to the canonical `---`.
///
/// Per CommonMark: any run of ≥3 identical `-` / `_` / `*` characters
/// (optionally surrounded by whitespace, up to 3 leading spaces)
/// parses to `<hr/>`. Length and choice of char are irrelevant to the
/// parser — `-------` and `---` and `___` all produce identical HTML.
/// We canonicalize to `---` so the raw form doesn't bloat the training
/// corpus with 80-char dash runs.
///
/// Intentionally NOT rewritten (rewriting would CHANGE preview and
/// violate the Phase A invariant — verifier catches it):
///
/// - `====` runs: setext heading level-1 marker under CommonMark
///   (when preceded by a non-blank line), or a literal paragraph of
///   `=` chars otherwise. Never an HR.
/// - Unicode em-dash / horizontal-bar / box-drawing / double-dash
///   (`———`, `═══`, `───`): CommonMark renders these as a paragraph
///   of literal chars, not as an HR.
/// - Dot-leader lines (`..........`): parsed as paragraph text by
///   CommonMark; handled separately in the cosmetic-leader pass that
///   lives outside this module.
///
/// Also skips at `≥4` leading columns — that's indented code per
/// CommonMark, not a thematic break.
pub fn normalize_separator_line(line: &str) -> Option<String> {
    // Indented code block: `≥4` leading columns. CommonMark renders any
    // dash/underscore/asterisk run in this context as literal text, not
    // as an HR — rewriting it would change preview.
    if leading_columns(line) >= 4 {
        return None;
    }
    if !SEPARATOR_LINE_REGEX.is_match(line) {
        return None;
    }
    Some("---".to_string())
}

// ---------------------------------------------------------------------------
// GFM table separator pre-pass.
// ---------------------------------------------------------------------------

/// Scan the full text for GFM-compliant table separator rows. A row
/// qualifies when (a) the row itself parses as a separator (cells of
/// `:?-{3,}:?`, pipe-delimited) AND (b) the line immediately preceding
/// it is a pipe-delimited row with the same number of cells (a header
/// row).
///
/// Returns a map from `line_index` (0-based, as emitted by
/// `str::lines()`) to the canonical replacement line. The replacement
/// always uses the minimal `---` hyphen body per cell; alignment colons
/// (`:---` left / `---:` right / `:---:` center) are preserved. GFM
/// parser sees identical table; raw form is compact.
pub fn scan_gfm_table_separators(text: &str) -> HashMap<usize, String> {
    let mut replacements: HashMap<usize, String> = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    // Track code-fence state so we don't normalize `|----|`-shaped
    // lines that appear inside fenced code blocks (which must survive
    // intact).
    let mut in_code_fence = false;
    for (i, line) in lines.iter().enumerate() {
        if is_code_fence_marker(line) {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence {
            continue;
        }
        if i == 0 {
            continue;
        }
        // CommonMark: a leaf block starting at ≥4 leading columns is an
        // indented code block, not a GFM table. If either the separator
        // or its header lies at that indentation, leave both alone.
        if leading_columns(line) >= 4 {
            continue;
        }
        let sep = match parse_gfm_separator_row(line) {
            Some(s) => s,
            None => continue,
        };
        let header = lines[i - 1];
        if leading_columns(header) >= 4 {
            continue;
        }
        let header_cells = count_gfm_row_cells(header);
        if header_cells != sep.cells.len() {
            continue;
        }
        let canonical_cells: Vec<&str> = sep
            .cells
            .iter()
            .map(|a| match a {
                GfmAlign::Default => "---",
                GfmAlign::Left => ":---",
                GfmAlign::Center => ":---:",
                GfmAlign::Right => "---:",
            })
            .collect();
        replacements.insert(i, format!("| {} |", canonical_cells.join(" | ")));
    }
    replacements
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GfmAlign {
    Default,
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone)]
struct GfmSeparatorRow {
    cells: Vec<GfmAlign>,
}

fn parse_gfm_separator_row(line: &str) -> Option<GfmSeparatorRow> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    // A GFM table row MUST contain at least one pipe. Without this check
    // a bare `----` (standalone separator) would be (mis)-parsed as a
    // 1-cell table separator and then collapsed to `| --- |` whenever
    // the line above happened to be non-empty.
    if !trimmed.contains('|') {
        return None;
    }
    // Strip optional leading/trailing pipe.
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    if inner.is_empty() {
        return None;
    }
    let cells: Vec<&str> = inner.split('|').map(str::trim).collect();
    if cells.is_empty() {
        return None;
    }
    let mut parsed = Vec::with_capacity(cells.len());
    for cell in cells {
        if cell.is_empty() {
            return None;
        }
        let left = cell.starts_with(':');
        let right = cell.ends_with(':');
        // Strip leading/trailing colons to get the hyphen body.
        let body_start = if left { 1 } else { 0 };
        let body_end = if right { cell.len() - 1 } else { cell.len() };
        if body_end <= body_start {
            return None;
        }
        let body = &cell[body_start..body_end];
        if body.len() < 3 {
            return None;
        }
        if !body.chars().all(|c| c == '-') {
            return None;
        }
        let align = match (left, right) {
            (true, true) => GfmAlign::Center,
            (true, false) => GfmAlign::Left,
            (false, true) => GfmAlign::Right,
            (false, false) => GfmAlign::Default,
        };
        parsed.push(align);
    }
    Some(GfmSeparatorRow { cells: parsed })
}

fn count_gfm_row_cells(line: &str) -> usize {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return 0;
    }
    // A GFM table row MUST contain at least one pipe.
    if !trimmed.contains('|') {
        return 0;
    }
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    if inner.is_empty() {
        return 0;
    }
    inner.split('|').count()
}

// ---------------------------------------------------------------------------
// Fenced code block detector.
// ---------------------------------------------------------------------------

/// Line-level predicate for a fenced code block marker (opening or
/// closing). Used by consumers that need to track code-fence state —
/// this module's own `scan_gfm_table_separators` and
/// `reflow_paragraphs`, and `cleaning_module::core_clean_text_with_stats`.
///
/// Per CommonMark: `^ {0,3}(?:`{3,}|~{3,})[info]?$` roughly. Caller
/// MUST pass the raw (un-trimmed) line — at `≥4` leading columns the
/// same visual shape is an indented code block, not a fence opener,
/// and wrongly toggling fence state there would make the cleaner skip
/// normalization on real prose (or, symmetrically, normalize inside
/// a real fenced code block).
pub fn is_code_fence_marker(line: &str) -> bool {
    if leading_columns(line) >= 4 {
        return false;
    }
    let t = line.trim_start();
    // Require at least 3 backticks or 3 tildes.
    t.starts_with("```") || t.starts_with("~~~")
}

// ---------------------------------------------------------------------------
// Paragraph linearization (reflow soft-wrapped paragraphs onto one line).
// ---------------------------------------------------------------------------

/// Collapse soft-wrap line breaks inside a paragraph block into a
/// single space. PDF-extracted MD commonly fragments a single paragraph
/// across multiple short lines (PDF column-width wrap). CommonMark
/// treats single `\n` inside a paragraph as whitespace, so joining is
/// a preview no-op — and it makes the raw form read as actual
/// paragraphs instead of 60-char stubs.
///
/// Guards (hard breaks that halt the join):
/// - Blank line (paragraph break).
/// - `#` heading, `>` blockquote, list markers (`- `, `* `, `+ `,
///   `N. `, `N) `).
/// - GFM table rows (`|...|`).
/// - HR thematic-break lines (matches `SEPARATOR_LINE_REGEX`).
/// - Fenced-code markers (```, ~~~) — state-machine-tracked.
/// - Prior line ends with a sentence terminator.
/// - Next line is indented ≥4 spaces or a tab (= indented code block).
pub fn reflow_paragraphs(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    if lines.len() < 2 {
        return text.to_string();
    }
    let mut out_lines: Vec<String> = Vec::with_capacity(lines.len());
    let mut in_fenced_code = false;
    for line in &lines {
        // Track fenced code state so we don't reflow inside code blocks.
        if is_code_fence_marker(line) {
            in_fenced_code = !in_fenced_code;
            out_lines.push(line.to_string());
            continue;
        }
        if in_fenced_code {
            out_lines.push(line.to_string());
            continue;
        }
        // Can we append this line to the previous one in out_lines?
        if let Some(prev) = out_lines.last() {
            if can_join_lines(prev, line) {
                let joined = format!("{} {}", prev.trim_end(), line.trim_start());
                let idx = out_lines.len() - 1;
                out_lines[idx] = joined;
                continue;
            }
        }
        out_lines.push(line.to_string());
    }
    out_lines.join("\n")
}

fn can_join_lines(prev: &str, next: &str) -> bool {
    // CommonMark hard break #1: prev ends in two (or more) trailing
    // spaces → `<br>` in preview. Joining would strip the break.
    // Detect BEFORE `trim_end()` destroys the signal.
    if prev.ends_with("  ") {
        return false;
    }
    // CommonMark hard break #2: prev ends in an unescaped backslash.
    // An odd number of trailing backslashes means the last one escapes
    // the newline → `<br>`. An even count means the last backslash is
    // itself escaped and is a literal `\`, so no hard break.
    let trailing_backslashes = prev.chars().rev().take_while(|c| *c == '\\').count();
    if trailing_backslashes % 2 == 1 {
        return false;
    }
    let prev_trim = prev.trim_end();
    let next_trim = next.trim_start();
    // Both must be non-empty content.
    if prev_trim.is_empty() || next_trim.is_empty() {
        return false;
    }
    // Don't merge across structural lines.
    if line_is_hard_break(prev_trim) || line_is_hard_break(next_trim) {
        return false;
    }
    // Prior line's last non-whitespace char — sentence terminators
    // stop merging.
    let last = prev_trim.chars().next_back().unwrap();
    if matches!(
        last,
        '.' | '!' | '?' | ':' | ';' | '·' | '\u{037E}' /* Greek ; */
        | '"' | '\'' | ')' | ']' | '}' | '…'
        | '»' | '\u{201D}' | '\u{2019}'
    ) {
        return false;
    }
    // Next line's first char — must look like continuation (letter/
    // digit/opening-quote).
    let first = next_trim.chars().next().unwrap();
    if first.is_alphanumeric() || matches!(first, '«' | '(' | '\u{201C}' | '\u{2018}') {
        // Also guard: if the RAW `next` line (with leading whitespace)
        // is indented by 4+ spaces or a tab, it's an indented code
        // block in markdown — don't join.
        let raw_leading = next.len() - next.trim_start().len();
        let tab_or_4spaces = next.starts_with('\t')
            || (raw_leading >= 4 && next.chars().take(raw_leading).all(|c| c == ' '));
        if tab_or_4spaces {
            return false;
        }
        return true;
    }
    false
}

fn line_is_hard_break(line: &str) -> bool {
    if line.is_empty() {
        return true;
    }
    // Fenced code markers (`````` / `~~~`) are hard breaks too — the
    // outer reflow walker tracks fenced-code state, but if the prev/
    // next line itself IS a fence marker, joining it to the
    // surrounding prose is wrong.
    if is_code_fence_marker(line) {
        return true;
    }
    let first = line.chars().next().unwrap();
    // Headings, blockquotes.
    if matches!(first, '#' | '>') {
        return true;
    }
    // List markers at line start (`- item`, `* item`, `+ item`,
    // `1. item`) — preserve.
    if matches!(first, '-' | '*' | '+') && line.chars().nth(1) == Some(' ') {
        return true;
    }
    // Ordered list: `N.` or `N)`
    let mut digit_run = 0;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() {
            digit_run += 1;
        } else {
            if digit_run > 0 && (c == '.' || c == ')') {
                if chars.next() == Some(' ') {
                    return true;
                }
            }
            break;
        }
    }
    // Table rows.
    if line.starts_with('|') && line.ends_with('|') && line.matches('|').count() >= 2 {
        return true;
    }
    // HR thematic-break / setext heading marker lines. Uses the
    // ≥3-char CM threshold so the canonical `---` output of
    // `normalize_separator_line` is recognized, and so setext H1/H2
    // markers (`===`, `---`) are preserved as block boundaries.
    if HR_HARD_BREAK_REGEX.is_match(line) {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Phase A orchestrator — run all Phase A transforms in the correct order.
// ---------------------------------------------------------------------------

/// Run the full MD-syntax normalization phase in the correct order.
///
/// Order rationale:
///
/// 1. **GFM table separator minimization first.** Runs against raw
///    input lines so it can pair each separator row with its header
///    row. If reflow ran before this, a long `|-----|-----|` row
///    would pass through unchanged (table rows are hard-breaks for
///    reflow anyway, but any future subtle interaction is avoided by
///    running this first).
/// 2. **HR thematic-break minimization.** Per-line pass; order mostly
///    independent of the other two.
/// 3. **Paragraph reflow LAST.** Reflow depends on being able to
///    identify hard-break lines (including table rows and HRs, both
///    of which should already be in canonical form so the hard-break
///    detector is reliable).
///
/// Returns the rewritten text.
pub fn normalize_md_syntax(text: &str) -> String {
    // Step 1: GFM table separator minimization — rewrite matched rows.
    let replacements = scan_gfm_table_separators(text);
    let step1: String = if replacements.is_empty() {
        text.to_string()
    } else {
        let lines: Vec<&str> = text.lines().collect();
        let mut out = String::with_capacity(text.len());
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            if let Some(canonical) = replacements.get(&i) {
                out.push_str(canonical);
            } else {
                out.push_str(line);
            }
        }
        // Preserve trailing newline if the original text had one.
        if text.ends_with('\n') {
            out.push('\n');
        }
        out
    };

    // Step 2: HR thematic-break minimization — per-line, fence-aware.
    // A dash/underscore/asterisk run inside a fenced code block is
    // literal content (e.g., a horizontal ruler drawn by the code
    // author); rewriting it would change preview and corrupt the
    // code block.
    let step2: String = {
        let lines: Vec<&str> = step1.lines().collect();
        let mut out = String::with_capacity(step1.len());
        let mut in_code_fence = false;
        for (i, line) in lines.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            if is_code_fence_marker(line) {
                in_code_fence = !in_code_fence;
                out.push_str(line);
                continue;
            }
            if in_code_fence {
                out.push_str(line);
                continue;
            }
            match normalize_separator_line(line) {
                Some(canonical) => out.push_str(&canonical),
                None => out.push_str(line),
            }
        }
        if step1.ends_with('\n') {
            out.push('\n');
        }
        out
    };

    // Step 3: paragraph reflow — collapse soft-wraps within blocks.
    reflow_paragraphs(&step2)
}

// ---------------------------------------------------------------------------
// Non-destructive canonicalization — single source of truth for what
// the cleaner WOULD produce if every pass were non-destructive.
// ---------------------------------------------------------------------------

/// Apply every non-destructive cleaner transform to `md`, in the same
/// order the cleaner applies them.
///
/// Used as the shared baseline by:
/// - `md_verify::canonicalize_for_verify` — pre-canonicalizes INPUT
///   before comparing against cleaner OUTPUT in structural mode, so
///   cosmetic differences aren't misclassified as injections.
/// - (regression test in `cleaning_module`) — asserts that for any
///   input where the cleaner wouldn't delete anything, its output
///   equals this function's output. That test catches drift between
///   cleaner and verifier.
///
/// Transforms applied (all semantic- or preview-preserving):
/// 1. HTML entity decode (`&amp;` → `&`).
/// 2. Adobe Symbol PUA decode (U+F061 → α).
/// 3. Soft-hyphen strip (U+00AD is invisible anyway).
/// 4. Per-line char fold (NBSP → space, ligatures → pairs, Unicode
///    whitespace variants → space, enclosed digits → ASCII).
/// 5. Dot-run normalization (tiered bucket collapse).
/// 6. Whitespace-run normalization (multi-space → tiered bucket).
/// 7. Ellipsis-run normalization.
/// 8. Phase A orchestrator (GFM sep min, HR min, paragraph reflow).
///
/// NOT applied (destructive or content-removing — belong to Phase B):
/// - GLYPH-marker strip.
/// - Per-char allowlist filter.
/// - Line-drop rules.
/// - Rule-A/B filtering.
pub fn non_destructive_canonicalize(md: &str) -> String {
    // Steps 1-3: content-level preprocessing.
    let step1 = normalize::decode_html_entities(md);
    let step2 = normalize::decode_adobe_symbol_pua(&step1);
    let step3 = normalize::strip_soft_hyphens(&step2);

    // Step 4-7: per-line char fold + per-line normalizations.
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

    // Step 8: MD-syntax-aware Phase A (GFM sep min, HR min, reflow).
    normalize_md_syntax(&per_line_out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- HR minimization ---

    #[test]
    fn hr_minimization_collapses_long_ascii_runs() {
        assert_eq!(normalize_separator_line("----"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("______"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("****"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("  ----  "), Some("---".to_string()));
    }

    #[test]
    fn hr_minimization_does_not_touch_equals_runs() {
        // `====` is a setext heading level-1 marker in CommonMark (when
        // preceded by a non-blank line) or a paragraph of `=` chars
        // otherwise. NEVER an HR — transforming it would change render.
        assert_eq!(normalize_separator_line("===="), None);
        assert_eq!(normalize_separator_line("========"), None);
    }

    #[test]
    fn hr_minimization_does_not_touch_unicode_dash_like_chars() {
        // Em-dash, horizontal-bar, box-drawing are NOT CommonMark HRs.
        // CommonMark renders them as a paragraph of literal chars;
        // transforming to `---` would change render.
        assert_eq!(normalize_separator_line("———"), None);
        assert_eq!(normalize_separator_line("═══"), None);
        assert_eq!(normalize_separator_line("───"), None);
    }

    #[test]
    fn hr_minimization_preserves_non_hr() {
        // ASCII threshold is 4 chars; exactly 3 dashes unchanged.
        assert_eq!(normalize_separator_line("---"), None);
        assert_eq!(normalize_separator_line("hello ----"), None);
        assert_eq!(normalize_separator_line("----- x"), None);
        // Dot-leader runs are not HRs.
        assert_eq!(normalize_separator_line("......"), None);
        // Mixed chars not a valid HR.
        assert_eq!(normalize_separator_line("---___"), None);
    }

    #[test]
    fn hr_minimization_handles_escaped_underscores() {
        assert_eq!(normalize_separator_line(r"\_\_\_\_"), Some("---".to_string()));
        assert_eq!(
            normalize_separator_line(r"\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_"),
            Some("---".to_string())
        );
        assert_eq!(normalize_separator_line(r"  \_\_\_\_  "), Some("---".to_string()));
        // Below threshold (3 escaped pairs).
        assert_eq!(normalize_separator_line(r"\_\_\_"), None);
        assert_eq!(normalize_separator_line(r"\_\_\___"), None);
    }

    // --- GFM table separator minimization ---

    #[test]
    fn gfm_sep_minimizes_long_dash_body() {
        let text = "| a | b |\n| -------- | -------- |\n| 1 | 2 |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps.get(&1), Some(&"| --- | --- |".to_string()));
    }

    #[test]
    fn gfm_sep_preserves_alignment_colons() {
        let text = "| a | b | c | d |\n| :---- | -----: | :----: | ---- |\n| 1 | 2 | 3 | 4 |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(
            reps.get(&1),
            Some(&"| :--- | ---: | :---: | --- |".to_string())
        );
    }

    #[test]
    fn gfm_sep_ignores_lines_without_pipes() {
        // Standalone `----` (HR) must NOT be claimed as a 1-cell
        // GFM table separator.
        let text = "para\n----\nother\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_sep_ignores_lines_inside_fenced_code() {
        let text = "```\n| a | b |\n| --- | --- |\n```\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_sep_requires_matching_header_cell_count() {
        // header has 3 cells, separator has 2 → don't touch.
        let text = "| a | b | c |\n| --- | --- |\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_sep_rejects_sep_without_header_line() {
        // First-line separator (i=0) has no header before it.
        let text = "| --- | --- |\n| 1 | 2 |\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_sep_rejects_body_row_with_short_dashes() {
        // Body cells must be ≥3 hyphens.
        let text = "| a | b |\n| - | -- |\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_sep_handles_multiple_tables() {
        let text = "| a | b |\n| ------ | ------ |\n| 1 | 2 |\n\n| c | d |\n| ---- | ---- |\n| 3 | 4 |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 2);
    }

    // --- CommonMark indentation helper ---

    #[test]
    fn leading_columns_counts_spaces() {
        assert_eq!(leading_columns(""), 0);
        assert_eq!(leading_columns("abc"), 0);
        assert_eq!(leading_columns(" abc"), 1);
        assert_eq!(leading_columns("   abc"), 3);
        assert_eq!(leading_columns("    abc"), 4);
    }

    #[test]
    fn leading_columns_applies_tab_rule() {
        // A tab advances to the next multiple of 4.
        assert_eq!(leading_columns("\tabc"), 4);
        assert_eq!(leading_columns(" \tabc"), 4);
        assert_eq!(leading_columns("   \tabc"), 4);
        assert_eq!(leading_columns("    \tabc"), 8);
        // Two tabs.
        assert_eq!(leading_columns("\t\tabc"), 8);
    }

    #[test]
    fn leading_columns_ignores_non_leading_whitespace() {
        assert_eq!(leading_columns("abc   "), 0);
        assert_eq!(leading_columns("a\tb"), 0);
    }

    // --- Fenced code detection ---

    #[test]
    fn code_fence_detects_backticks_and_tildes() {
        assert!(is_code_fence_marker("```"));
        assert!(is_code_fence_marker("```python"));
        assert!(is_code_fence_marker("~~~"));
        assert!(is_code_fence_marker("~~~rust"));
        assert!(is_code_fence_marker("  ```"));
        assert!(is_code_fence_marker("   ```")); // 3 leading spaces
        assert!(!is_code_fence_marker("`inline`"));
        assert!(!is_code_fence_marker("``double``"));
        assert!(!is_code_fence_marker("no fence"));
    }

    #[test]
    fn code_fence_rejects_at_four_leading_columns() {
        // CommonMark: `≥4` leading columns = indented code block. The
        // same visual shape is NOT a fence opener in that context.
        assert!(!is_code_fence_marker("    ```"));
        assert!(!is_code_fence_marker("    ```python"));
        assert!(!is_code_fence_marker("    ~~~"));
        // Tab counts as 4 columns.
        assert!(!is_code_fence_marker("\t```"));
        // Mixed-whitespace cases that add up to ≥4 columns.
        assert!(!is_code_fence_marker("   \t```"));
    }

    #[test]
    fn hr_and_gfm_rejected_at_four_leading_columns() {
        // HR detector bails: `    ----` is indented code.
        assert_eq!(normalize_separator_line("    ----"), None);
        assert_eq!(normalize_separator_line("\t----"), None);
        // 3 leading spaces still fine.
        assert_eq!(normalize_separator_line("   ----"), Some("---".to_string()));

        // GFM scanner: both separator and header must be outside
        // indented-code range.
        let indented_table = "\
paragraph\n\n    | a | b |\n    | --- | --- |\n    | 1 | 2 |\n\nafter\n";
        let reps = scan_gfm_table_separators(indented_table);
        assert!(reps.is_empty(), "indented table must be left alone");
    }

    // --- Paragraph reflow ---

    #[test]
    fn reflow_joins_soft_wrapped_lines() {
        assert_eq!(reflow_paragraphs("word1\nword2\nword3"), "word1 word2 word3");
    }

    #[test]
    fn reflow_preserves_blank_line_breaks() {
        let input = "paragraph1.\n\nparagraph2.";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_preserves_headings() {
        let input = "body text\n# Heading\nmore text";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_preserves_table_rows() {
        let input = "intro\n| a | b |\n| - | - |\n| 1 | 2 |\nafter";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_preserves_list_items() {
        let input = "intro\n- item one\n- item two\nafter";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_stops_at_sentence_terminators() {
        let input = "First sentence.\nSecond starts here";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_stops_at_fenced_code() {
        let input = "before\n```\ncode line\n```\nafter";
        assert_eq!(reflow_paragraphs(input), input);
    }

    #[test]
    fn reflow_does_not_join_indented_code() {
        let input = "prose\n    code line\nprose again";
        let out = reflow_paragraphs(input);
        assert!(out.contains("    code line"));
    }

    #[test]
    fn reflow_joins_pdf_column_wrap_pattern() {
        let input = "word1\t\n  word2\t\n  word3";
        let out = reflow_paragraphs(input);
        assert_eq!(out, "word1 word2 word3");
    }

    // --- Phase A orchestrator ---

    #[test]
    fn normalize_md_syntax_applies_all_three_in_sequence() {
        let input = "| a | b |\n| ----------- | ----------- |\n| 1 | 2 |\n\n------------\n\nFirst word\nsecond word\nthird word\n";
        let out = normalize_md_syntax(input);
        // Table separator canonicalized.
        assert!(out.contains("| --- | --- |"));
        // HR minimized.
        assert!(out.contains("\n---\n"));
        // Paragraph reflowed.
        assert!(out.contains("First word second word third word"));
    }

    // -----------------------------------------------------------------
    // Preview-equivalence regression tests for Phase A transforms.
    //
    // Invariant asserted: for each transform, the cleaner OUTPUT renders
    // identically to the INPUT under a spec-compliant GFM parser. Any
    // future edit that breaks preview-preservation fails loudly here.
    //
    // Uses `md_verify::verify_md_preview_equivalent` (pulldown-cmark as
    // reference parser). See `docs/MD_MODULE_ARCHITECTURE.md`.
    // -----------------------------------------------------------------

    // --- HR minimization preserves preview ---

    #[test]
    fn equiv_hr_min_on_ascii_dash_run() {
        let input = "before\n\n----------\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_hr_min_on_underscore_run() {
        let input = "before\n\n__________\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_hr_min_on_asterisk_run() {
        let input = "before\n\n**********\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_hr_min_leaves_em_dash_paragraph_alone() {
        // `———` renders as a paragraph of em-dashes under CommonMark
        // (NOT an HR). Phase A must leave it unchanged so preview
        // stays bit-identical.
        let input = "before\n\n———\n\nafter\n";
        let out = normalize_md_syntax(input);
        assert_eq!(out, input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- GFM table separator minimization preserves preview ---

    #[test]
    fn equiv_gfm_sep_min_on_long_dash_body() {
        let input = "| a | b |\n| ---------- | ---------- |\n| 1 | 2 |\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_gfm_sep_min_preserves_all_alignments() {
        let input = "| a | b | c | d |\n| :-------- | --------: | :--------: | -------- |\n| 1 | 2 | 3 | 4 |\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_gfm_sep_min_on_single_column_table() {
        let input = "| Header |\n| ---------- |\n| cell |\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_gfm_sep_min_on_multiple_tables() {
        let input = concat!(
            "| a | b |\n| ---------- | ---------- |\n| 1 | 2 |\n\n",
            "| c | d | e |\n| :----- | :-----: | -----: |\n| x | y | z |\n"
        );
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- Paragraph reflow preserves preview ---

    #[test]
    fn equiv_reflow_on_soft_wrapped_paragraph() {
        let input = "This is a\nsoft-wrapped\nparagraph.\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_reflow_across_mixed_content() {
        let input = concat!(
            "# Heading\n\n",
            "First soft-wrapped\nparagraph of text.\n\n",
            "Second paragraph\nalso wrapped.\n\n",
            "- list item one\n- list item two\n\n",
            "Final paragraph\non multiple\nshort lines.\n"
        );
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_reflow_does_not_merge_across_blockquote() {
        let input = "First paragraph\nline two\n\n> quoted content\n> line two\n\nAfter.\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_reflow_does_not_merge_across_fenced_code() {
        let input = "Before paragraph\ntext.\n\n```\ncode line one\ncode line two\n```\n\nAfter paragraph\ntext.\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- Orchestrator equivalence on mixed-content docs ---

    #[test]
    fn equiv_orchestrator_on_heading_para_table_hr_mix() {
        let input = concat!(
            "# Top heading\n\n",
            "A soft-wrapped\nparagraph of text.\n\n",
            "| Col A | Col B |\n| ---------- | ---------- |\n| 1 | 2 |\n\n",
            "-----------\n\n",
            "## Section two\n\n",
            "- item alpha\n- item beta\n- item gamma\n\n",
            "Final\nparagraph\nsoft-wrapped.\n"
        );
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn equiv_orchestrator_idempotent_on_canonical_input() {
        // Already-canonical input should pass through unchanged.
        let input = concat!(
            "# Heading\n\n",
            "Already-joined paragraph.\n\n",
            "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n",
            "---\n\n",
            "After.\n"
        );
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- Negative controls: if equiv check is wrong, these would pass ---

    #[test]
    fn equiv_detects_an_incorrect_transform_that_drops_paragraph() {
        // This is NOT md_module's output — we manufacture a broken
        // transform to confirm the verifier would catch it.
        let input = "para1\n\npara2\n\npara3\n";
        let broken_output = "para1\n\npara3\n";
        let r = crate::md_verify::verify_md_preview_equivalent(input, broken_output);
        assert!(!r.is_strict_equivalent(), "verifier should catch dropped paragraph");
    }

    #[test]
    fn equiv_detects_an_incorrect_transform_that_fuses_words() {
        // Simulates the v6-11 NBSP-strip bug. Would-be Phase A violation.
        let input = "Η εργασία αυτή έχει σκοπό.\n";
        let broken_output = "Ηεργασίααυτήέχεισκοπό.\n";
        let r = crate::md_verify::verify_md_preview_equivalent(input, broken_output);
        assert!(!r.is_strict_equivalent(), "verifier should catch word fusion");
        assert!(!r.paragraph_text_equal);
    }

    // -----------------------------------------------------------------
    // Commit 11 — RED tests for the bugs identified in the
    // MD_MODULE_ARCHITECTURE_IMPLEMENTATION_REVIEW (2026-04-24).
    //
    // These tests expose Phase A preview-equivalence violations that
    // the current implementation commits. They are EXPECTED TO FAIL
    // on the commit-11 boundary; commits 12–15 turn them green by
    // adding CommonMark indentation awareness + hard-break guards
    // + orchestrator wiring + expanded structural comparison.
    //
    // Each test name ends in `_red_until_C<N>` to make the tracking
    // explicit.
    // -----------------------------------------------------------------

    // --- H-1 indentation awareness (CommonMark: ≥4 leading spaces /
    //                                tab = indented code, NOT an HR /
    //                                table / fence opener) ---

    #[test]
    fn red_until_c12_indented_code_with_hr_looking_line_preserved() {
        // CommonMark: lines indented by ≥4 spaces (after preceding blank
        // line) form an indented code block. `----` inside such a block
        // is literal text, NOT a thematic break. Phase A must not touch
        // it. Render: `<pre><code>----</code></pre>`.
        let input = "paragraph\n\n    ----\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(
            r.is_strict_equivalent(),
            "indented code containing HR-looking line was rewritten — \
             violates Phase A invariant. Fix in Commit 12: detector \
             must reject at ≥4 leading columns. {:?}",
            r
        );
    }

    #[test]
    fn red_until_c12_indented_code_with_table_looking_lines_preserved() {
        // Same CommonMark rule: `    | a | b |\n    | --- | --- |` is
        // indented code, NOT a GFM table. Phase A must not canonicalize
        // the separator row.
        let input = "paragraph\n\n    | a | b |\n    | --- | --- |\n    | 1 | 2 |\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(
            r.is_strict_equivalent(),
            "indented code containing table-looking lines was rewritten — \
             scan_gfm_table_separators must reject rows whose leading \
             columns >= 4. {:?}",
            r
        );
    }

    #[test]
    fn equiv_indented_code_with_fence_looking_line_passthrough() {
        // `    ` + ``` → indented code in CommonMark. Even though our
        // current is_code_fence_marker erroneously matches this (bug
        // per the review, fixed in Commit 12), normal inputs like this
        // one happen to still pass preview-equivalence because reflow's
        // 4-space guard already refuses to join these lines. The
        // deeper bug (state-machine toggling on indented `` ` `` inside
        // a real fenced code block) is exercised by the fence-grammar
        // tests we add in Commit 12.
        let input = "paragraph\n\n    ```\n    body\n    ```\n\nafter\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    // --- H-3 paragraph reflow destroys hard breaks ---

    #[test]
    fn red_until_c13_reflow_preserves_two_space_hard_break() {
        // CommonMark hard break: two trailing spaces before `\n`
        // produces `<br>` in preview. Current reflow `trim_end()`s
        // the prev line, destroying the hard-break marker.
        let input = "first line  \nsecond line\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(
            r.is_strict_equivalent(),
            "two-space hard break lost by reflow — preview `<br>` \
             removed. Fix in Commit 13: detect `  $` and refuse to \
             join. {:?}",
            r
        );
    }

    // --- Post-C13 regression tests ---

    #[test]
    fn equiv_reflow_preserves_canonical_hr_as_hard_break() {
        // After `normalize_separator_line` collapses `--------` to the
        // canonical `---`, reflow must still recognize `---` as a hard
        // break and NOT fuse it with adjacent paragraphs. This was the
        // silent Phase A violation that Commit 13 fixed.
        let input = "hello\n--------\nworld\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "canonical HR fused by reflow: {:?}", r);
        assert!(out.contains("\n---\n"), "HR should remain isolated, got {out:?}");
    }

    #[test]
    fn equiv_reflow_preserves_setext_heading_marker() {
        // Setext H2: `text\n---\n` renders as <h2>text</h2>. If reflow
        // joins `---` with the next paragraph, the H2 demotes to a
        // regular paragraph of "text --- more", changing preview.
        let input = "heading text\n---\n\nbody paragraph.\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "setext heading demoted: {:?}", r);
    }

    #[test]
    fn equiv_orchestrator_preserves_fenced_hr_content() {
        // A `----` line inside a fenced code block is literal content
        // (e.g., an author-drawn ruler). The HR normalizer must NOT
        // fire there.
        let input = "prose\n\n```\n----\n```\n\nafter\n";
        let out = normalize_md_syntax(input);
        assert!(out.contains("\n----\n"), "fenced ---- rewritten: {out:?}");
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(r.is_strict_equivalent(), "{:?}", r);
    }

    #[test]
    fn red_until_c13_reflow_preserves_backslash_hard_break() {
        // CommonMark hard break: trailing `\` before `\n` → `<br>`.
        let input = "first line\\\nsecond line\n";
        let out = normalize_md_syntax(input);
        let r = crate::md_verify::verify_md_preview_equivalent(input, &out);
        assert!(
            r.is_strict_equivalent(),
            "backslash hard break lost by reflow. Fix in Commit 13: \
             detect `\\$` and refuse to join. {:?}",
            r
        );
    }
}
