//! Layout / structure normalization helpers for `core_clean_text`.
//!
//! Design spec:
//! `Projects/glossapi-tokenizer-extension/corpus_clean_normalization/NORMALIZATION_DESIGN_20260420.md`
//!
//! The helpers in this module implement the deterministic normalize/strip rules:
//! - character fold (ligatures, enclosed/dingbat/math-alphanumeric digits,
//!   vulgar fractions, Unicode whitespace variants)
//! - line-level ellipsis / whitespace / separator-line normalization
//! - malformed HTML entity fallback (`&gt`, `&lt`, `&amp` without `;`)
//! - GFM table separator pre-pass (parser-validated)
//! - code-fence marker detection (used by callers to guard normalization)
//!
//! All helpers are pure functions. Wire-in happens inside `core_clean_text`.

use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

lazy_static! {
    /// Two or more U+2026 ellipsis chars.
    pub static ref ELLIPSIS_RUN_REGEX: Regex = Regex::new(r"…{2,}").unwrap();

    /// Two or more ASCII spaces or tabs.
    pub static ref WHITESPACE_RUN_REGEX: Regex = Regex::new(r"[ \t]{2,}").unwrap();

    /// `&gt`, `&lt`, `&amp` NOT followed by `;` or alphanumeric.
    ///
    /// Rust's `regex` crate has no look-ahead, so we capture the following
    /// context char (end-of-line, or a non-alphanumeric / non-`;` byte) and
    /// preserve it in the replacement closure.
    pub static ref MALFORMED_ENTITY_REGEX: Regex =
        Regex::new(r"&(gt|lt|amp)($|[^a-zA-Z0-9;])").unwrap();

    /// Standalone separator line: homogeneous runs of a single ASCII
    /// separator char (`-` / `_` / `*` / `=`) of length >=4, OR homogeneous
    /// runs of em-dash (U+2014), horizontal bar (U+2015), box-drawing light
    /// (U+2500), or box-drawing double (U+2550) of length >=3. Optional
    /// leading / trailing whitespace. Dots are handled separately (see
    /// `normalize_layout_leader_runs`).
    ///
    /// Mixed-char runs like `---___` are intentionally *not* matched — the
    /// design doc treats those as ASCII art, not thematic breaks.
    pub static ref SEPARATOR_LINE_REGEX: Regex = Regex::new(
        r"^[ \t]*(?:-{4,}|_{4,}|\*{4,}|={4,}|\u{2014}{3,}|\u{2015}{3,}|\u{2500}{3,}|\u{2550}{3,})[ \t]*$",
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Character fold
// ---------------------------------------------------------------------------

const ASCII_DIGITS: [&str; 10] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
];

/// Return `Some(replacement)` if `ch` should fold, `None` otherwise.
///
/// Policy (from the 2026-04-20 design):
/// - Fold enclosed / circled / dingbat / mathematical-alphanumeric digits to ASCII.
/// - Fold vulgar fractions to ASCII `a/b`.
/// - Fold ligatures (`ﬁ`, `ﬂ`, `ﬃ`, `ﬄ`, `ﬀ`, `ﬅ`, `ﬆ`) to ASCII pairs.
/// - Fold Unicode whitespace variants (U+2007 figure space, U+2009 thin space,
///   U+202F narrow NBSP) to a regular space.
/// - KEEP subscripts (U+2080–2089) and superscripts (U+2070, U+00B2, U+00B3,
///   U+00B9, U+2074–2079) as-is — they carry semantic weight.
pub fn fold_codepoint(ch: char) -> Option<&'static str> {
    // Ligatures first (they come before more numeric work, small set)
    match ch {
        '\u{FB00}' => return Some("ff"),
        '\u{FB01}' => return Some("fi"),
        '\u{FB02}' => return Some("fl"),
        '\u{FB03}' => return Some("ffi"),
        '\u{FB04}' => return Some("ffl"),
        '\u{FB05}' => return Some("st"),
        '\u{FB06}' => return Some("st"),
        _ => {}
    }

    // Unicode whitespace variants folded to regular space.
    match ch {
        '\u{2007}' | '\u{2009}' | '\u{202F}' => return Some(" "),
        _ => {}
    }

    // Vulgar fractions (Latin-1 Supplement + Number Forms block).
    match ch {
        '\u{00BC}' => return Some("1/4"),
        '\u{00BD}' => return Some("1/2"),
        '\u{00BE}' => return Some("3/4"),
        '\u{2150}' => return Some("1/7"),
        '\u{2151}' => return Some("1/9"),
        '\u{2152}' => return Some("1/10"),
        '\u{2153}' => return Some("1/3"),
        '\u{2154}' => return Some("2/3"),
        '\u{2155}' => return Some("1/5"),
        '\u{2156}' => return Some("2/5"),
        '\u{2157}' => return Some("3/5"),
        '\u{2158}' => return Some("4/5"),
        '\u{2159}' => return Some("1/6"),
        '\u{215A}' => return Some("5/6"),
        '\u{215B}' => return Some("1/8"),
        '\u{215C}' => return Some("3/8"),
        '\u{215D}' => return Some("5/8"),
        '\u{215E}' => return Some("7/8"),
        // U+215F FRACTION NUMERATOR ONE — numerator prefix, fold to "1/".
        '\u{215F}' => return Some("1/"),
        // U+2189 VULGAR FRACTION ZERO THIRDS.
        '\u{2189}' => return Some("0/3"),
        _ => {}
    }

    // Ranged patterns: match on the integer codepoint.
    let code = ch as u32;

    // Circled digits ①–⑨ (U+2460–U+2468), ⑩ (U+2469).
    if (0x2460..=0x2468).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x2460 + 1) as usize]);
    }
    if code == 0x2469 {
        return Some("10");
    }

    // Parenthesized digits ⑴–⑼ (U+2474–U+247C), ⑽ (U+247D).
    if (0x2474..=0x247C).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x2474 + 1) as usize]);
    }
    if code == 0x247D {
        return Some("10");
    }

    // Digits-with-full-stop ⒈–⒐ (U+2488–U+2490), ⒑ (U+2491).
    if (0x2488..=0x2490).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x2488 + 1) as usize]);
    }
    if code == 0x2491 {
        return Some("10");
    }

    // Dingbat negative circled ❶–❾ (U+2776–U+277E), ❿ (U+277F).
    if (0x2776..=0x277E).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x2776 + 1) as usize]);
    }
    if code == 0x277F {
        return Some("10");
    }

    // Dingbat negative sans-serif ➀–➈ (U+2780–U+2788), ➉ (U+2789).
    if (0x2780..=0x2788).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x2780 + 1) as usize]);
    }
    if code == 0x2789 {
        return Some("10");
    }

    // Dingbat sans-serif ➊–➒ (U+278A–U+2792), ➓ (U+2793).
    if (0x278A..=0x2792).contains(&code) {
        return Some(ASCII_DIGITS[(code - 0x278A + 1) as usize]);
    }
    if code == 0x2793 {
        return Some("10");
    }

    // Mathematical alphanumeric digit blocks (U+1D7CE–U+1D7FF).
    // Five blocks of 10: Bold, Double-struck, Sans-serif, Sans-serif bold, Monospace.
    if let 0x1D7CE..=0x1D7FF = code {
        let offset = (code - 0x1D7CE) % 10;
        return Some(ASCII_DIGITS[offset as usize]);
    }

    None
}

/// Fold every codepoint in `line` per `fold_codepoint`. Returns `None` if no
/// fold fired (allocation-free fast path for ASCII-only lines).
pub fn fold_line(line: &str) -> Option<String> {
    // Cheap fast path: every fold target is >= U+00BC, so an ASCII-only line
    // cannot fire any fold and we can skip the scan entirely.
    if line.is_ascii() {
        return None;
    }
    let mut out = String::with_capacity(line.len());
    let mut changed = false;
    for ch in line.chars() {
        if let Some(replacement) = fold_codepoint(ch) {
            out.push_str(replacement);
            changed = true;
        } else {
            out.push(ch);
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Line-level normalizations
// ---------------------------------------------------------------------------

/// Collapse runs of `…{2,}` to a single `…`.
pub fn normalize_ellipsis_runs(line: &str) -> Option<String> {
    if !line.contains('…') {
        return None;
    }
    if !ELLIPSIS_RUN_REGEX.is_match(line) {
        return None;
    }
    let out = ELLIPSIS_RUN_REGEX.replace_all(line, "…").into_owned();
    if out == line {
        None
    } else {
        Some(out)
    }
}

/// Collapse runs of `[ \t]{2,}` to a single space.
pub fn normalize_whitespace_runs(line: &str) -> Option<String> {
    if !WHITESPACE_RUN_REGEX.is_match(line) {
        return None;
    }
    let out = WHITESPACE_RUN_REGEX.replace_all(line, " ").into_owned();
    if out == line {
        None
    } else {
        Some(out)
    }
}

/// Replace malformed HTML entities (`&gt` / `&lt` / `&amp` without a trailing `;`)
/// with their decoded form.
pub fn normalize_malformed_entities(line: &str) -> Option<String> {
    // Cheap early-out: need at least one candidate substring.
    if !line.contains("&gt") && !line.contains("&lt") && !line.contains("&amp") {
        return None;
    }
    if !MALFORMED_ENTITY_REGEX.is_match(line) {
        return None;
    }
    let out = MALFORMED_ENTITY_REGEX
        .replace_all(line, |caps: &regex::Captures| {
            let entity = match &caps[1] {
                "gt" => ">",
                "lt" => "<",
                "amp" => "&",
                _ => unreachable!("regex constrained to gt|lt|amp"),
            };
            // caps[2] is the preserved context char (empty at end-of-line,
            // otherwise a single non-alphanumeric, non-`;` byte).
            format!("{}{}", entity, &caps[2])
        })
        .into_owned();
    if out == line {
        None
    } else {
        Some(out)
    }
}

/// Collapse a standalone separator line (hyphen / underscore / asterisk / equals
/// runs, or em-dash / horizontal-bar / box-drawing runs) to the canonical `---`.
///
/// Does NOT fire on dot-leader lines (those are handled by
/// `normalize_layout_leader_runs` in `cleaning_module.rs` with target `.....`).
pub fn normalize_separator_line(line: &str) -> Option<String> {
    if !SEPARATOR_LINE_REGEX.is_match(line) {
        return None;
    }
    Some("---".to_string())
}

// ---------------------------------------------------------------------------
// GFM table separator pre-pass
// ---------------------------------------------------------------------------

/// Scan the full text for GFM-compliant table separator rows. A row qualifies
/// when (a) the row itself parses as a separator (cells of `:?-{3,}:?`,
/// pipe-delimited) AND (b) the line immediately preceding it is a
/// pipe-delimited row with the same number of cells (a header row).
///
/// Returns a map from `line_index` (0-based, as emitted by `str::lines()`)
/// to the canonical replacement line.
pub fn scan_gfm_table_separators(text: &str) -> HashMap<usize, String> {
    let mut replacements: HashMap<usize, String> = HashMap::new();
    let lines: Vec<&str> = text.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let sep = match parse_gfm_separator_row(line) {
            Some(s) => s,
            None => continue,
        };
        let header = lines[i - 1];
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
    let inner = trimmed.trim_start_matches('|').trim_end_matches('|');
    if inner.is_empty() {
        return 0;
    }
    inner.split('|').count()
}

// ---------------------------------------------------------------------------
// Code fence detection
// ---------------------------------------------------------------------------

/// True if the line opens or closes a fenced code block (``` or ~~~).
/// Leading whitespace up to 3 spaces is allowed (CommonMark spec).
pub fn is_code_fence_marker(line: &str) -> bool {
    let t = line.trim_start();
    // Require at least 3 backticks or 3 tildes.
    t.starts_with("```") || t.starts_with("~~~")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_ligatures() {
        assert_eq!(fold_line("ﬁshing ﬂour"), Some("fishing flour".to_string()));
        assert_eq!(fold_line("eﬃcient"), Some("efficient".to_string()));
        assert_eq!(fold_line("baﬄe"), Some("baffle".to_string()));
    }

    #[test]
    fn fold_enclosed_digits() {
        assert_eq!(fold_line("①②③"), Some("123".to_string()));
        assert_eq!(fold_line("chapter ⑩"), Some("chapter 10".to_string()));
        assert_eq!(fold_line("❺"), Some("5".to_string()));
        assert_eq!(fold_line("➋"), Some("2".to_string()));
    }

    #[test]
    fn fold_math_alphanumeric_digits() {
        // U+1D7CE–U+1D7D7 bold
        assert_eq!(fold_line("𝟎𝟏𝟐𝟑𝟒𝟓𝟔𝟕𝟖𝟗"), Some("0123456789".to_string()));
        // U+1D7D8–U+1D7E1 double-struck
        assert_eq!(fold_line("𝟘𝟡"), Some("09".to_string()));
        // U+1D7EC–U+1D7F5 sans-serif bold
        assert_eq!(fold_line("𝟬𝟱"), Some("05".to_string()));
    }

    #[test]
    fn fold_fractions() {
        assert_eq!(fold_line("½ cup"), Some("1/2 cup".to_string()));
        assert_eq!(fold_line("¼ + ¾"), Some("1/4 + 3/4".to_string()));
        assert_eq!(fold_line("⅗"), Some("3/5".to_string()));
    }

    #[test]
    fn keep_subscripts_and_superscripts() {
        // Subscripts and superscripts are preserved per design decision.
        assert_eq!(fold_line("H₂O"), None);
        assert_eq!(fold_line("E=mc²"), None);
        assert_eq!(fold_line("x₁ + x₂"), None);
        assert_eq!(fold_line("10³"), None);
    }

    #[test]
    fn fold_unicode_whitespace_variants() {
        assert_eq!(fold_line("a\u{2007}b"), Some("a b".to_string())); // figure space
        assert_eq!(fold_line("a\u{2009}b"), Some("a b".to_string())); // thin space
        assert_eq!(fold_line("a\u{202F}b"), Some("a b".to_string())); // narrow NBSP
    }

    #[test]
    fn fold_ascii_fast_path() {
        // ASCII-only inputs should short-circuit and return None without allocating.
        assert_eq!(fold_line("plain ASCII text"), None);
        assert_eq!(fold_line(""), None);
        assert_eq!(fold_line("1 + 2 = 3"), None);
    }

    #[test]
    fn ellipsis_runs_collapse() {
        assert_eq!(
            normalize_ellipsis_runs("wait…… then"),
            Some("wait… then".to_string())
        );
        assert_eq!(
            normalize_ellipsis_runs("………"),
            Some("…".to_string())
        );
        assert_eq!(normalize_ellipsis_runs("single …"), None);
        assert_eq!(normalize_ellipsis_runs("no ellipsis"), None);
    }

    #[test]
    fn whitespace_runs_collapse() {
        assert_eq!(
            normalize_whitespace_runs("a    b"),
            Some("a b".to_string())
        );
        assert_eq!(
            normalize_whitespace_runs("a\t\tb"),
            Some("a b".to_string())
        );
        assert_eq!(normalize_whitespace_runs("a b"), None);
        assert_eq!(normalize_whitespace_runs(""), None);
    }

    #[test]
    fn malformed_entities_fallback() {
        assert_eq!(
            normalize_malformed_entities("x &gt y"),
            Some("x > y".to_string())
        );
        assert_eq!(
            normalize_malformed_entities("&lt tag"),
            Some("< tag".to_string())
        );
        assert_eq!(
            normalize_malformed_entities("a &amp b"),
            Some("a & b".to_string())
        );
        // Well-formed entities left alone (htmlentity handles them).
        assert_eq!(normalize_malformed_entities("x &gt; y"), None);
        assert_eq!(normalize_malformed_entities("x &lt; y"), None);
        // Alphanumeric following `&gt` means it's something else; don't fold.
        assert_eq!(normalize_malformed_entities("&gtfoo"), None);
    }

    #[test]
    fn separator_line_detection() {
        assert_eq!(normalize_separator_line("----"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("______"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("****"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("===="), Some("---".to_string()));
        assert_eq!(normalize_separator_line("  ----  "), Some("---".to_string()));
        // Em-dash / horizontal bar / box-drawing (>=3).
        assert_eq!(normalize_separator_line("———"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("═══"), Some("---".to_string()));
        assert_eq!(normalize_separator_line("───"), Some("---".to_string()));
        // Below threshold.
        assert_eq!(normalize_separator_line("---"), None); // ASCII threshold is 4
        assert_eq!(normalize_separator_line("==="), None);
        // Contains non-separator content.
        assert_eq!(normalize_separator_line("hello ----"), None);
        assert_eq!(normalize_separator_line("----- x"), None);
        // Dot leader stays out of this rule.
        assert_eq!(normalize_separator_line("......"), None);
        // Mixed separator chars on one line — not collapsed (keeps each family
        // distinct; only repeats of a single family qualify).
        assert_eq!(normalize_separator_line("---___"), None);
    }

    #[test]
    fn gfm_table_separator_basic() {
        let text = "| Header | Col2 |\n|-------|------|\n| a | b |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps.get(&1).map(String::as_str), Some("| --- | --- |"));
    }

    #[test]
    fn gfm_table_separator_with_alignment() {
        let text = "| A | B | C | D |\n| :--- | :---: | ---: | --- |\n| 1 | 2 | 3 | 4 |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 1);
        assert_eq!(
            reps.get(&1).map(String::as_str),
            Some("| :--- | :---: | ---: | --- |")
        );
    }

    #[test]
    fn gfm_table_separator_wider_cells_collapsed() {
        let text = "| A | B |\n| :---------- | -----------: |\n| 1 | 2 |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps.get(&1).map(String::as_str), Some("| :--- | ---: |"));
    }

    #[test]
    fn gfm_table_separator_no_header_does_not_match() {
        // `random prose` has 0 pipes; row-count mismatch => no replacement.
        let text = "random prose\n|-------|------|\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_table_separator_cell_count_mismatch_does_not_match() {
        // Header has 3 cells; separator has 2.
        let text = "| A | B | C |\n|-------|------|\n| 1 | 2 | 3 |\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_table_separator_rejects_non_hyphen_body() {
        // `---=` is not a valid separator cell body.
        let text = "| A | B |\n|---=|-----|\n| 1 | 2 |\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn code_fence_marker_detection() {
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
}
