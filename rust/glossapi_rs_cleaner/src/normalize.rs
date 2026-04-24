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

    /// Two or more ASCII spaces or tabs — cheap presence check before we
    /// run the tiered bucket rewriter.
    pub static ref WHITESPACE_RUN_REGEX: Regex = Regex::new(r"[ \t]{2,}").unwrap();

    /// Two or more ASCII dots — cheap presence check for dot-run tiered
    /// bucket rewriting.
    pub static ref DOT_RUN_2PLUS_REGEX: Regex = Regex::new(r"\.{2,}").unwrap();

    /// `&gt`, `&lt`, `&amp` NOT followed by `;` or alphanumeric.
    ///
    /// Rust's `regex` crate has no look-ahead, so we capture the following
    /// context char (end-of-line, or a non-alphanumeric / non-`;` byte) and
    /// preserve it in the replacement closure.
    pub static ref MALFORMED_ENTITY_REGEX: Regex =
        Regex::new(r"&(gt|lt|amp)($|[^a-zA-Z0-9;])").unwrap();

    // (SEPARATOR_LINE_REGEX moved to md_module.rs alongside
    // normalize_separator_line — it's MD-syntax-aware and lives with
    // the Phase A transforms.)
}

// ---------------------------------------------------------------------------
// Character fold
// ---------------------------------------------------------------------------

const ASCII_DIGITS: [&str; 10] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
];

const ASCII_UPPER: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

const ASCII_LOWER: [&str; 26] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
    "t", "u", "v", "w", "x", "y", "z",
];

/// Greek capitals in the order Math Alphanumeric Greek blocks use them:
/// Α Β Γ Δ Ε Ζ Η Θ Ι Κ Λ Μ Ν Ξ Ο Π Ρ ϴ Σ Τ Υ Φ Χ Ψ Ω
/// (Position 17 is the capital-theta variant ϴ (U+03F4), not regular Θ.)
const GREEK_CAPITAL_MATH_ORDER: [&str; 25] = [
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "ϴ",
    "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
];

/// Greek smalls in Math-block order:
/// α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ ς σ τ υ φ χ ψ ω
const GREEK_SMALL_MATH_ORDER: [&str; 25] = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "ς",
    "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
];

/// Math Alphanumeric Greek "variant symbols" block-tail (offsets 52..57):
/// ϵ ϑ ϰ ϕ ϱ ϖ (epsilon / theta / kappa / phi / rho / pi variants).
const GREEK_VARIANT_MATH_ORDER: [&str; 6] = [
    "\u{03F5}", "\u{03D1}", "\u{03F0}", "\u{03D5}", "\u{03F1}", "\u{03D6}",
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

    // Mathematical Alphanumeric Symbols — Latin letter blocks
    // (U+1D400..U+1D6A3). 13 style blocks of 52 codepoints each (A-Z then
    // a-z). Reserved "holes" in some blocks (Italic, Script, Fraktur,
    // Double-Struck) are actually encoded in Letterlike Symbols
    // (U+2100..U+214F) — handled immediately below. For valid codepoints
    // inside each style block the mapping is uniform: offset modulo 52
    // picks the ASCII letter regardless of style.
    if let 0x1D400..=0x1D6A3 = code {
        let block_offset = (code - 0x1D400) % 52;
        return Some(if block_offset < 26 {
            ASCII_UPPER[block_offset as usize]
        } else {
            ASCII_LOWER[(block_offset - 26) as usize]
        });
    }

    // Math italic dotless i/j (U+1D6A4, U+1D6A5) and reserved slots
    // U+1D6A6/U+1D6A7. The two assigned slots fold to plain i/j.
    if code == 0x1D6A4 {
        return Some("i");
    }
    if code == 0x1D6A5 {
        return Some("j");
    }

    // Mathematical Alphanumeric Symbols — Greek letter blocks
    // (U+1D6A8..U+1D7C9). Five style blocks of 58 codepoints each:
    //   1D6A8 Bold, 1D6E2 Italic, 1D71C Bold Italic,
    //   1D756 Sans-Serif Bold, 1D790 Sans-Serif Bold Italic.
    // Layout inside each block:
    //   0..24   capital letters Α..Ω (order per Math Alphanumeric spec)
    //   25      nabla ∇ (U+2207)
    //   26..50  small letters α..ω
    //   51      partial differential ∂ (U+2202)
    //   52..57  variant symbols ϵ ϑ ϰ ϕ ϱ ϖ
    // Policy (2026-04-21): these are Greek letters used in math with
    // semantic meaning; Apertus has single-token merges for regular
    // Greek, so we FOLD them into the regular Greek codepoint that
    // Apertus tokenizes efficiently, rather than stripping.
    if let 0x1D6A8..=0x1D7C9 = code {
        let off = (code - 0x1D6A8) % 58;
        return Some(match off {
            0..=24 => GREEK_CAPITAL_MATH_ORDER[off as usize],
            25 => "\u{2207}", // ∇ NABLA
            26..=50 => GREEK_SMALL_MATH_ORDER[(off - 26) as usize],
            51 => "\u{2202}", // ∂ PARTIAL DIFFERENTIAL
            52..=57 => GREEK_VARIANT_MATH_ORDER[(off - 52) as usize],
            _ => unreachable!(),
        });
    }

    // Mathematical Bold Capital/Small Digamma (U+1D7CA..U+1D7CB) fold to
    // regular Greek Digamma (U+03DC / U+03DD). U+1D7CC/U+1D7CD are reserved.
    if code == 0x1D7CA {
        return Some("\u{03DC}");
    }
    if code == 0x1D7CB {
        return Some("\u{03DD}");
    }

    // Letterlike Symbols that are the "hole" chars for the Math
    // Alphanumeric blocks (Script h, Fraktur H, Double-Struck H, etc.).
    // Fold to the matching ASCII Latin letter.
    match code {
        0x210A => return Some("g"), // ℊ SCRIPT SMALL G
        0x210B => return Some("H"), // ℋ SCRIPT CAPITAL H
        0x210C => return Some("H"), // ℌ BLACK-LETTER CAPITAL H
        0x210D => return Some("H"), // ℍ DOUBLE-STRUCK CAPITAL H
        0x210E => return Some("h"), // ℎ PLANCK CONSTANT (== math italic h)
        0x2110 => return Some("I"), // ℐ SCRIPT CAPITAL I
        0x2111 => return Some("I"), // ℑ BLACK-LETTER CAPITAL I
        0x2112 => return Some("L"), // ℒ SCRIPT CAPITAL L
        0x2113 => return Some("l"), // ℓ SCRIPT SMALL L
        0x2115 => return Some("N"), // ℕ DOUBLE-STRUCK N
        0x2119 => return Some("P"), // ℙ DOUBLE-STRUCK P
        0x211A => return Some("Q"), // ℚ DOUBLE-STRUCK Q
        0x211B => return Some("R"), // ℛ SCRIPT R
        0x211C => return Some("R"), // ℜ FRAKTUR R
        0x211D => return Some("R"), // ℝ DOUBLE-STRUCK R
        0x2124 => return Some("Z"), // ℤ DOUBLE-STRUCK Z
        0x2128 => return Some("Z"), // ℨ FRAKTUR Z
        0x212C => return Some("B"), // ℬ SCRIPT B
        0x212D => return Some("C"), // ℭ FRAKTUR C
        0x212F => return Some("e"), // ℯ SCRIPT SMALL E
        0x2130 => return Some("E"), // ℰ SCRIPT E
        0x2131 => return Some("F"), // ℱ SCRIPT F
        0x2133 => return Some("M"), // ℳ SCRIPT M
        0x2134 => return Some("o"), // ℴ SCRIPT SMALL O
        _ => {}
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

/// Tiered bucket for run-length normalization (2026-04-21 policy, v2):
///   1, 2    → 1  (single char / accidental double collapse to one)
///   3       → 3  (unchanged — natural prose triple, e.g. ellipsis)
///   4..=20  → 5  (medium run — canonical "short leader" form)
///   >20     → 20 (long run — canonical "long leader" form)
/// Target token vocabulary for dots: `.`, `...`, `.....`,
/// `....................` — four forms. Uniform across dots and
/// whitespace so the BPE sees a small fixed vocabulary of leader
/// tokens regardless of which fill a PDF used or how long the run was.
pub fn bucket_run_length(n: usize) -> usize {
    match n {
        0 | 1 => n,
        2 => 1,
        3 => 3,
        4..=20 => 5,
        _ => 20,
    }
}

/// Normalize runs of exactly `target` per the tiered bucket rule above.
pub fn normalize_char_runs_tiered(line: &str, target: char) -> Option<String> {
    if !line.contains(target) {
        return None;
    }
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut changed = false;
    let mut i = 0usize;
    while i < chars.len() {
        if chars[i] == target {
            let start = i;
            while i < chars.len() && chars[i] == target {
                i += 1;
            }
            let n = i - start;
            let m = bucket_run_length(n);
            if m != n {
                changed = true;
            }
            for _ in 0..m {
                out.push(target);
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    if changed {
        Some(out)
    } else {
        None
    }
}

/// Normalize dot runs per the tiered bucket rule (uniform with whitespace).
/// Legacy note: previously `\.{4,}` collapsed to `.....` (5 dots). Now:
///   4..=10 → `...` (3 dots), >10 → `....................` (20 dots).
pub fn normalize_dot_runs(line: &str) -> Option<String> {
    if !DOT_RUN_2PLUS_REGEX.is_match(line) {
        return None;
    }
    normalize_char_runs_tiered(line, '.')
}

/// Normalize runs of `[ \t]` per the tiered bucket rule. Mixed space+tab
/// runs are treated as one run and emitted as N spaces.
///
/// Leading whitespace at line start is preserved verbatim — protects
/// markdown indented code blocks (4-space indent), list indentation,
/// nested list indent, and verse indent. Only INTERIOR whitespace runs
/// (those that follow a non-whitespace character on the same line) are
/// bucketized.
pub fn normalize_whitespace_runs(line: &str) -> Option<String> {
    // Fire if there's a 2+ whitespace run OR any non-leading tab.
    if !WHITESPACE_RUN_REGEX.is_match(line) && !line.contains('\t') {
        return None;
    }
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut changed = false;
    let mut i = 0usize;

    // Copy leading whitespace verbatim (code-block / list-indent preservation).
    while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
        out.push(chars[i]);
        i += 1;
    }

    // Bucketize interior + trailing whitespace runs.
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' || c == '\t' {
            let start = i;
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                i += 1;
            }
            let n = i - start;
            let m = bucket_run_length(n);
            // A single interior tab collapsing to a single space is still a
            // change (tabs carry no semantic value in markdown prose).
            let original: String = chars[start..start + n].iter().collect();
            if m != n || original.contains('\t') {
                changed = true;
            }
            for _ in 0..m {
                out.push(' ');
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    if changed {
        Some(out)
    } else {
        None
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

// (normalize_separator_line, scan_gfm_table_separators, parse_gfm_separator_row,
// count_gfm_row_cells, GfmAlign, GfmSeparatorRow moved to md_module.rs —
// they're MD-syntax-aware transforms with the preview-render-preserving
// invariant, co-located with the other Phase A passes.)

// ---------------------------------------------------------------------------
// Page salvage (drop pages with too much content stripped)
// ---------------------------------------------------------------------------

/// Drop synthetic pages from `cleaned_text` whose retained non-whitespace
/// content falls below `min_retention_ratio` of the corresponding page in
/// `original_text`.
///
/// Synthetic pages are built from the ORIGINAL text's line structure:
/// each markdown header line (`^#+`) starts a new page; the first page
/// runs from line 0 to the first header (or end of text if no headers).
/// This matches the boundary heuristic used by the matcher's synthetic-page
/// builder for consistency between cleaner output and matcher re-audit.
///
/// Line-alignment assumption: `cleaned_text` must preserve one output line
/// per input line — `core_clean_text` already satisfies this. If line
/// counts diverge the function returns `cleaned_text` unmodified.
///
/// Design reference:
/// corpus_clean_normalization/NORMALIZATION_DESIGN_20260420.md §14.
pub fn drop_low_salvage_pages(
    original_text: &str,
    cleaned_text: &str,
    min_retention_ratio: f64,
) -> String {
    let orig_lines: Vec<&str> = original_text.lines().collect();
    let clean_lines: Vec<&str> = cleaned_text.lines().collect();

    // If line counts diverge we can't safely align per-page accounting.
    // Return cleaned_text unchanged so the caller sees a clear no-op.
    if orig_lines.len() != clean_lines.len() {
        return cleaned_text.to_string();
    }

    let page_ranges = synthetic_page_line_ranges(&orig_lines);
    let mut kept_lines: Vec<&str> = Vec::with_capacity(clean_lines.len());
    let mut dropped_any = false;

    for (start, end) in page_ranges {
        let orig_nonws = count_nonwhitespace_in_range(&orig_lines, start, end);
        let clean_nonws = count_nonwhitespace_in_range(&clean_lines, start, end);
        let retention = if orig_nonws > 0 {
            clean_nonws as f64 / orig_nonws as f64
        } else {
            // Empty original page — nothing to salvage, no ratio to check.
            1.0
        };
        if retention >= min_retention_ratio {
            kept_lines.extend_from_slice(&clean_lines[start..end]);
        } else {
            dropped_any = true;
        }
    }

    if !dropped_any {
        return cleaned_text.to_string();
    }

    let mut result = kept_lines.join("\n");
    // Preserve trailing newline behavior: if the input had one and we kept
    // any content, re-add a single trailing newline.
    if cleaned_text.ends_with('\n') && !result.is_empty() {
        result.push('\n');
    }
    result
}

fn synthetic_page_line_ranges(lines: &[&str]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if i > 0 && is_markdown_header_line(line) {
            ranges.push((start, i));
            start = i;
        }
    }
    ranges.push((start, lines.len()));
    ranges
}

fn is_markdown_header_line(line: &str) -> bool {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return false;
    }
    // Valid ATX header: one or more `#` followed by a space or end-of-line.
    // Rejects `####word` (no space) to avoid treating random `#`-led strings
    // as headers.
    let after_hashes = t.trim_start_matches('#');
    after_hashes.is_empty() || after_hashes.starts_with(' ') || after_hashes.starts_with('\t')
}

fn count_nonwhitespace_in_range(lines: &[&str], start: usize, end: usize) -> usize {
    lines[start..end]
        .iter()
        .flat_map(|l| l.chars())
        .filter(|c| !c.is_whitespace())
        .count()
}

// ---------------------------------------------------------------------------
// Code fence detection
// ---------------------------------------------------------------------------

/// True if the line opens or closes a fenced code block (``` or ~~~).
/// Leading whitespace up to 3 spaces is allowed (CommonMark spec).
// (is_code_fence_marker moved to md_module.rs.)

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
    fn fold_math_italic_latin_letters() {
        // Math Italic Latin letters (U+1D434..U+1D467) — the main form seen
        // in the tokenizer evidence for broken italic-variable extraction.
        assert_eq!(fold_line("𝑖"), Some("i".to_string()));
        assert_eq!(fold_line("𝑛"), Some("n".to_string()));
        assert_eq!(fold_line("𝑥"), Some("x".to_string()));
        assert_eq!(fold_line("𝐴"), Some("A".to_string()));
        assert_eq!(fold_line("𝑅"), Some("R".to_string()));
        assert_eq!(fold_line("𝑆"), Some("S".to_string()));
        // A typical math-variable cluster.
        assert_eq!(
            fold_line("𝑥 + 𝑦 = 𝑧"),
            Some("x + y = z".to_string())
        );
    }

    #[test]
    fn fold_math_bold_and_other_styles() {
        // Bold block (U+1D400..U+1D433).
        assert_eq!(fold_line("𝐀𝐁𝐂"), Some("ABC".to_string()));
        assert_eq!(fold_line("𝐚𝐛𝐜"), Some("abc".to_string()));
        // Bold Italic (U+1D468..U+1D49B).
        assert_eq!(fold_line("𝑨𝒂"), Some("Aa".to_string()));
        // Script (U+1D49C..U+1D4CF) — body chars fold; holes use Letterlike.
        assert_eq!(fold_line("𝒜𝒶"), Some("Aa".to_string()));
        // Fraktur (U+1D504..U+1D537).
        assert_eq!(fold_line("𝔄𝔞"), Some("Aa".to_string()));
        // Double-Struck (U+1D538..U+1D56B).
        assert_eq!(fold_line("𝔸𝕒"), Some("Aa".to_string()));
        // Sans-Serif (U+1D5A0..U+1D5D3).
        assert_eq!(fold_line("𝖠𝖺"), Some("Aa".to_string()));
        // Monospace (U+1D670..U+1D6A3).
        assert_eq!(fold_line("𝙰𝚊"), Some("Aa".to_string()));
    }

    #[test]
    fn fold_math_italic_dotless() {
        // U+1D6A4 dotless italic i, U+1D6A5 dotless italic j.
        assert_eq!(fold_line("𝚤"), Some("i".to_string()));
        assert_eq!(fold_line("𝚥"), Some("j".to_string()));
    }

    #[test]
    fn fold_letterlike_symbols_holes() {
        // The "holes" in Math Italic / Script / Fraktur / Double-Struck
        // blocks are encoded as separate codepoints in the Letterlike
        // Symbols block. They fold to the matching ASCII letter.
        assert_eq!(fold_line("ℎ"), Some("h".to_string())); // PLANCK CONSTANT
        assert_eq!(fold_line("ℓ"), Some("l".to_string())); // SCRIPT SMALL L
        assert_eq!(fold_line("ℝ"), Some("R".to_string())); // DOUBLE-STRUCK R
        assert_eq!(fold_line("ℕ"), Some("N".to_string())); // DOUBLE-STRUCK N
        assert_eq!(fold_line("ℤ"), Some("Z".to_string())); // DOUBLE-STRUCK Z
        assert_eq!(fold_line("ℚ"), Some("Q".to_string())); // DOUBLE-STRUCK Q
        assert_eq!(fold_line("ℙ"), Some("P".to_string())); // DOUBLE-STRUCK P
        assert_eq!(fold_line("ℂ"), None); // DOUBLE-STRUCK C at U+2102 — not in our fold (intentional)
        assert_eq!(fold_line("ℋ"), Some("H".to_string())); // SCRIPT CAPITAL H
        assert_eq!(fold_line("ℌ"), Some("H".to_string())); // BLACK-LETTER CAPITAL H
        assert_eq!(fold_line("ℒ"), Some("L".to_string())); // SCRIPT CAPITAL L
        assert_eq!(fold_line("ℯ"), Some("e".to_string())); // SCRIPT SMALL E
    }

    #[test]
    fn fold_math_alphanumeric_greek_to_regular_greek() {
        // Math Bold Greek (U+1D6A8..U+1D6E1): capitals.
        assert_eq!(fold_line("𝚨"), Some("Α".to_string())); // U+1D6A8
        assert_eq!(fold_line("𝛀"), Some("Ω".to_string())); // U+1D6C0
        // Math Bold Greek: nabla at offset 25.
        assert_eq!(fold_line("𝛁"), Some("\u{2207}".to_string())); // ∇
        // Math Bold Greek: smalls.
        assert_eq!(fold_line("𝛂"), Some("α".to_string())); // U+1D6C2
        assert_eq!(fold_line("𝛚"), Some("ω".to_string())); // U+1D6DA
        assert_eq!(fold_line("𝛓"), Some("ς".to_string())); // final sigma (offset 43)
        // Math Bold Greek: partial differential at offset 51.
        assert_eq!(fold_line("𝛛"), Some("\u{2202}".to_string())); // ∂
        // Math Bold Greek: variant symbols.
        assert_eq!(fold_line("𝛜"), Some("\u{03F5}".to_string())); // ϵ

        // Math Italic Greek (U+1D6E2..).
        assert_eq!(fold_line("𝛼"), Some("α".to_string())); // U+1D6FC
        assert_eq!(fold_line("𝛽"), Some("β".to_string())); // U+1D6FD
        assert_eq!(fold_line("𝛾"), Some("γ".to_string())); // U+1D6FE

        // Math Bold Digamma.
        assert_eq!(fold_line("𝟊"), Some("Ϝ".to_string())); // U+1D7CA
        assert_eq!(fold_line("𝟋"), Some("ϝ".to_string())); // U+1D7CB

        // Composite: math-Greek sentence folds to plain Greek.
        assert_eq!(
            fold_line("𝛼 + 𝛽 = 𝛾"),
            Some("α + β = γ".to_string())
        );
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
    fn bucket_run_length_tiered_v2() {
        // {0, 1} → unchanged
        assert_eq!(bucket_run_length(0), 0);
        assert_eq!(bucket_run_length(1), 1);
        // {2} → 1
        assert_eq!(bucket_run_length(2), 1);
        // {3} → 3 (natural prose triple, e.g. ellipsis)
        assert_eq!(bucket_run_length(3), 3);
        // {4..=20} → 5
        for n in 4..=20 {
            assert_eq!(bucket_run_length(n), 5, "n={n}");
        }
        // > 20 → 20
        assert_eq!(bucket_run_length(21), 20);
        assert_eq!(bucket_run_length(42), 20);
        assert_eq!(bucket_run_length(200), 20);
    }

    #[test]
    fn dot_runs_tiered_v2() {
        // 2 dots → 1
        assert_eq!(
            normalize_dot_runs("word..here"),
            Some("word.here".to_string())
        );
        // 3 dots unchanged — natural prose ellipsis
        assert_eq!(normalize_dot_runs("wait... next"), None);
        // 4 dots → 5 (rounding up to canonical short-leader form)
        assert_eq!(
            normalize_dot_runs("Chapter 1 .... 5"),
            Some("Chapter 1 ..... 5".to_string())
        );
        // 10 dots → 5
        assert_eq!(
            normalize_dot_runs("..........heads"),
            Some(".....heads".to_string())
        );
        // 20 dots → 5 (edge of the short-leader bucket)
        let twenty = format!("x{}y", ".".repeat(20));
        let expected = format!("x{}y", ".".repeat(5));
        assert_eq!(normalize_dot_runs(&twenty), Some(expected));
        // >20 dots → 20
        let long = "x".to_string() + &".".repeat(42) + "y";
        let expected_long = "x".to_string() + &".".repeat(20) + "y";
        assert_eq!(normalize_dot_runs(&long), Some(expected_long));
        // No dots — fast path
        assert_eq!(normalize_dot_runs("no dots here"), None);
        // Single dot (sentence end) — unchanged
        assert_eq!(normalize_dot_runs("end of sentence."), None);
    }

    #[test]
    fn whitespace_runs_tiered_v2_interior_only() {
        // 2 spaces → 1
        assert_eq!(
            normalize_whitespace_runs("a  b"),
            Some("a b".to_string())
        );
        // 3 spaces — unchanged
        assert_eq!(normalize_whitespace_runs("a   b"), None);
        // 4 spaces → 5
        assert_eq!(
            normalize_whitespace_runs("a    b"),
            Some("a     b".to_string())
        );
        // 20 spaces → 5
        let twenty = format!("a{}b", " ".repeat(20));
        assert_eq!(
            normalize_whitespace_runs(&twenty),
            Some("a     b".to_string())
        );
        // 21+ spaces → 20
        let long = format!("a{}b", " ".repeat(42));
        let expected = format!("a{}b", " ".repeat(20));
        assert_eq!(normalize_whitespace_runs(&long), Some(expected));
        // Tabs always fold to spaces
        assert_eq!(
            normalize_whitespace_runs("a\t\tb"),
            Some("a b".to_string())
        );
        assert_eq!(
            normalize_whitespace_runs("a\tb"),
            Some("a b".to_string())
        );
        // No runs
        assert_eq!(normalize_whitespace_runs("a b c"), None);
        assert_eq!(normalize_whitespace_runs(""), None);
    }

    #[test]
    fn whitespace_runs_preserves_leading_indent() {
        // Markdown indented code block: 4-space indent preserved.
        assert_eq!(
            normalize_whitespace_runs("    def add(x, y):"),
            None
        );
        // 8-space indent (nested code) preserved.
        assert_eq!(
            normalize_whitespace_runs("        return x + y"),
            None
        );
        // Leading tab preserved (list-indent convention).
        assert_eq!(
            normalize_whitespace_runs("\titem"),
            None
        );
        // Leading indent + interior run: leading kept, interior bucketed.
        assert_eq!(
            normalize_whitespace_runs("    Chapter 1    title"),
            Some("    Chapter 1     title".to_string())
        );
        // Leading indent + TOC-style long run: leading kept, long run → 20.
        let input = format!("    Chapter 1{}5", " ".repeat(30));
        let expected = format!("    Chapter 1{}5", " ".repeat(20));
        assert_eq!(normalize_whitespace_runs(&input), Some(expected));
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

    // (separator_line_detection + all gfm_table_separator_* tests moved
    // to md_module.rs alongside the relocated functions.)

    #[test]
    fn drop_low_salvage_pages_keeps_all_when_above_threshold() {
        let original = "# Intro\nHello world\nAnother line\n# Second\nMore content\n";
        let cleaned = "# Intro\nHello world\nAnother line\n# Second\nMore content\n";
        let out = drop_low_salvage_pages(original, cleaned, 0.30);
        assert_eq!(out, cleaned);
    }

    #[test]
    fn drop_low_salvage_pages_drops_degraded_page() {
        // Second page lost almost everything.
        let original = "# Intro\nHello world\nNormal prose here\n# Second\nmostly garbage content lost\n";
        let cleaned = "# Intro\nHello world\nNormal prose here\n# Second\n\n";
        let out = drop_low_salvage_pages(original, cleaned, 0.30);
        // Second page dropped entirely; first survives.
        assert!(out.contains("# Intro"));
        assert!(out.contains("Hello world"));
        assert!(!out.contains("# Second"));
    }

    #[test]
    fn drop_low_salvage_pages_returns_input_when_line_counts_differ() {
        // Defensive: if caller passes mismatched line counts we no-op.
        let original = "line a\nline b\nline c\n";
        let cleaned = "line a\nline b\n";
        let out = drop_low_salvage_pages(original, cleaned, 0.50);
        assert_eq!(out, cleaned);
    }

    #[test]
    fn drop_low_salvage_pages_handles_no_headers() {
        // No markdown headers: whole text is one synthetic page.
        let original = "hello world\nplain prose\nno headers here\n";
        let cleaned = "h w\np p\nn h h\n";
        // Retention is low (~10 kept vs ~29 original).
        let retained = count_nonwhitespace_in_range(&cleaned.lines().collect::<Vec<_>>(), 0, 3);
        let orig_count =
            count_nonwhitespace_in_range(&original.lines().collect::<Vec<_>>(), 0, 3);
        let ratio = retained as f64 / orig_count as f64;
        assert!(ratio < 0.30);
        let out = drop_low_salvage_pages(original, cleaned, 0.30);
        assert_eq!(out, "");
    }

    #[test]
    fn drop_low_salvage_pages_preserves_trailing_newline() {
        let original = "# Intro\nbody\n";
        let cleaned = "# Intro\nbody\n";
        let out = drop_low_salvage_pages(original, cleaned, 0.30);
        assert!(out.ends_with('\n'));
    }

    #[test]
    fn is_markdown_header_line_accepts_valid_headers() {
        assert!(is_markdown_header_line("# Header"));
        assert!(is_markdown_header_line("## Subheader"));
        assert!(is_markdown_header_line("#### Deep"));
        assert!(is_markdown_header_line("  ## Indented"));
        // Hash with no following space is NOT a header (could be a hashtag).
        assert!(!is_markdown_header_line("#hashtag"));
        assert!(!is_markdown_header_line("####name"));
        // Plain text, code, non-hash prefixes.
        assert!(!is_markdown_header_line("plain"));
        assert!(!is_markdown_header_line(""));
    }

    #[test]
    fn synthetic_page_line_ranges_splits_on_headers() {
        let lines = vec![
            "intro line",
            "# First",
            "body of first",
            "# Second",
            "body of second",
            "more body",
        ];
        let ranges = synthetic_page_line_ranges(&lines);
        assert_eq!(ranges, vec![(0, 1), (1, 3), (3, 6)]);
    }

    // (code_fence_marker_detection moved to md_module.rs.)
}

// ---------------------------------------------------------------------------
// Wave-2 text-preprocessing passes (Cases 4, 7, 8, 10a, 13 subsets).
// Run at the START of core_clean_text_with_stats, BEFORE the per-line
// filter loop, so recovered chars survive per-char filtering.
// ---------------------------------------------------------------------------

lazy_static! {
    /// HTML named entities → literal chars (Case 4). Conservative: only
    /// the entities we've actually seen in openarchives / web-extracted
    /// Greek corpus docs. Adding more is cheap but should be driven by
    /// corpus evidence, not speculation.
    static ref HTML_NAMED_ENTITY_MAP: HashMap<&'static str, &'static str> = {
        let mut m = HashMap::new();
        m.insert("amp", "&");
        m.insert("lt", "<");
        m.insert("gt", ">");
        m.insert("quot", "\"");
        m.insert("apos", "'");
        m.insert("nbsp", "\u{00A0}");
        m.insert("copy", "©");
        m.insert("reg", "®");
        m.insert("trade", "™");
        m.insert("euro", "€");
        m.insert("pound", "£");
        m.insert("yen", "¥");
        m.insert("sect", "§");
        m.insert("deg", "°");
        m.insert("laquo", "«");
        m.insert("raquo", "»");
        m.insert("hellip", "…");
        m.insert("mdash", "—");
        m.insert("ndash", "–");
        m.insert("lsquo", "\u{2018}");
        m.insert("rsquo", "\u{2019}");
        m.insert("ldquo", "\u{201C}");
        m.insert("rdquo", "\u{201D}");
        m.insert("middot", "·");
        m.insert("bull", "•");
        m
    };

    /// Named-entity pattern: `&name;` where name is a short alphanumeric
    /// token. Doesn't swallow text if the entity name isn't recognised —
    /// the replacer checks the map and falls back to the original.
    static ref HTML_NAMED_ENTITY_REGEX: Regex =
        Regex::new(r"&([A-Za-z][A-Za-z0-9]{1,10});").unwrap();

    /// Numeric HTML entity: `&#1234;` (decimal) or `&#x1F600;` (hex).
    static ref HTML_NUMERIC_ENTITY_REGEX: Regex =
        Regex::new(r"&#(x[0-9A-Fa-f]+|[0-9]+);").unwrap();

    /// GLYPH/uni/gN marker regex (Case 7). Font-local glyph indices
    /// survive text extraction when ToUnicode CMaps are missing; we
    /// delete them rather than substitute (per user-verdict-resolved
    /// Case 7 / Case 9: the N is font-local, no safe global mapping).
    /// `GLYPH<…>` covers both bare numeric forms (`GLYPH<216>`) and
    /// the longer attribute forms emitted by some PDF extractors
    /// (`GLYPH<c=3,font=/SubsetName+FontFamily>`).
    static ref GLYPH_MARKER_REGEX: Regex = Regex::new(
        r"GLYPH<[^>]{1,200}>|/uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+"
    ).unwrap();

    /// Adobe Symbol font PUA → real Unicode (Case 10a). 100% of the
    /// top-30 PUA chars observed on openarchives 01500_pct0033_… were
    /// recovered to real Greek / math chars via this mapping.
    /// Reference: Adobe Symbol Encoding Vector.
    static ref ADOBE_SYMBOL_PUA_MAP: HashMap<char, &'static str> = {
        let mut m = HashMap::new();
        // ASCII-mirrored positions (F020..F07E): shift back to ASCII
        // code for punctuation + digits.
        m.insert('\u{F020}', " "); m.insert('\u{F021}', "!");
        m.insert('\u{F023}', "#"); m.insert('\u{F025}', "%");
        m.insert('\u{F026}', "&"); m.insert('\u{F028}', "(");
        m.insert('\u{F029}', ")"); m.insert('\u{F02A}', "*");
        m.insert('\u{F02B}', "+"); m.insert('\u{F02C}', ",");
        m.insert('\u{F02D}', "-"); m.insert('\u{F02E}', ".");
        m.insert('\u{F02F}', "/"); m.insert('\u{F030}', "0");
        m.insert('\u{F031}', "1"); m.insert('\u{F032}', "2");
        m.insert('\u{F033}', "3"); m.insert('\u{F034}', "4");
        m.insert('\u{F035}', "5"); m.insert('\u{F036}', "6");
        m.insert('\u{F037}', "7"); m.insert('\u{F038}', "8");
        m.insert('\u{F039}', "9"); m.insert('\u{F03A}', ":");
        m.insert('\u{F03B}', ";"); m.insert('\u{F03C}', "<");
        m.insert('\u{F03D}', "="); m.insert('\u{F03E}', ">");
        m.insert('\u{F03F}', "?");
        m.insert('\u{F05B}', "["); m.insert('\u{F05D}', "]");
        // Greek letters (F041..F057 upper, F061..F077 lower — Symbol
        // ordering). These are the positions where Adobe Symbol
        // emits Greek letters when a PDF embeds the font.
        m.insert('\u{F041}', "Α"); m.insert('\u{F042}', "Β");
        m.insert('\u{F043}', "Χ"); m.insert('\u{F044}', "Δ");
        m.insert('\u{F045}', "Ε"); m.insert('\u{F046}', "Φ");
        m.insert('\u{F047}', "Γ"); m.insert('\u{F048}', "Η");
        m.insert('\u{F049}', "Ι"); m.insert('\u{F04A}', "ϑ");
        m.insert('\u{F04B}', "Κ"); m.insert('\u{F04C}', "Λ");
        m.insert('\u{F04D}', "Μ"); m.insert('\u{F04E}', "Ν");
        m.insert('\u{F04F}', "Ο"); m.insert('\u{F050}', "Π");
        m.insert('\u{F051}', "Θ"); m.insert('\u{F052}', "Ρ");
        m.insert('\u{F053}', "Σ"); m.insert('\u{F054}', "Τ");
        m.insert('\u{F055}', "Υ"); m.insert('\u{F057}', "Ω");
        m.insert('\u{F058}', "Ξ"); m.insert('\u{F059}', "Ψ");
        m.insert('\u{F05A}', "Ζ");
        m.insert('\u{F061}', "α"); m.insert('\u{F062}', "β");
        m.insert('\u{F063}', "χ"); m.insert('\u{F064}', "δ");
        m.insert('\u{F065}', "ε"); m.insert('\u{F066}', "φ");
        m.insert('\u{F067}', "γ"); m.insert('\u{F068}', "η");
        m.insert('\u{F069}', "ι"); m.insert('\u{F06A}', "ϕ");
        m.insert('\u{F06B}', "κ"); m.insert('\u{F06C}', "λ");
        m.insert('\u{F06D}', "μ"); m.insert('\u{F06E}', "ν");
        m.insert('\u{F06F}', "ο"); m.insert('\u{F070}', "π");
        m.insert('\u{F071}', "θ"); m.insert('\u{F072}', "ρ");
        m.insert('\u{F073}', "σ"); m.insert('\u{F074}', "τ");
        m.insert('\u{F075}', "υ"); m.insert('\u{F077}', "ω");
        m.insert('\u{F078}', "ξ"); m.insert('\u{F079}', "ψ");
        m.insert('\u{F07A}', "ζ");
        // Math relations / operators commonly used in Greek math docs.
        m.insert('\u{F0A3}', "≤"); m.insert('\u{F0B3}', "≥");
        m.insert('\u{F0B9}', "≠"); m.insert('\u{F0AE}', "→");
        m.insert('\u{F0AC}', "←"); m.insert('\u{F0AD}', "↑");
        m.insert('\u{F0AF}', "↓"); m.insert('\u{F0DE}', "⇒");
        m.insert('\u{F0DC}', "⇐"); m.insert('\u{F0DB}', "⇔");
        m.insert('\u{F0CE}', "∈"); m.insert('\u{F0CF}', "∉");
        m.insert('\u{F0CD}', "⊄"); m.insert('\u{F0C7}', "∩");
        m.insert('\u{F0C8}', "∪"); m.insert('\u{F0C5}', "⊗");
        m.insert('\u{F0C9}', "⊃"); m.insert('\u{F0CB}', "⊂");
        m.insert('\u{F0CC}', "⊆"); m.insert('\u{F0D1}', "∠");
        m.insert('\u{F0D2}', "∇"); m.insert('\u{F0D4}', "∏");
        m.insert('\u{F0D5}', "√"); m.insert('\u{F0D6}', "·");
        m.insert('\u{F0D7}', "¬"); m.insert('\u{F0D8}', "∧");
        m.insert('\u{F0D9}', "∨"); m.insert('\u{F0DA}', "⇔");
        m.insert('\u{F0E5}', "∞"); m.insert('\u{F0E6}', "∫");
        m.insert('\u{F0E8}', "∑"); m.insert('\u{F0B4}', "×");
        m.insert('\u{F0B8}', "÷"); m.insert('\u{F0B1}', "±");
        m
    };
}

/// Decode HTML entities (Case 4): `&amp; &lt; &gt; &quot; &apos; &nbsp;`
/// + named entities in the map above + numeric (decimal `&#38;` and
/// hex `&#x26;`) back to their literal characters. Unknown named
/// entities are left as-is.
pub fn decode_html_entities(text: &str) -> String {
    // First, named: use a regex replace with a closure that falls back
    // to the original match if the name isn't in the map.
    let step1 = HTML_NAMED_ENTITY_REGEX.replace_all(text, |caps: &regex::Captures| {
        let name = &caps[1];
        match HTML_NAMED_ENTITY_MAP.get(name) {
            Some(replacement) => (*replacement).to_string(),
            None => caps[0].to_string(),
        }
    });
    // Then, numeric (decimal and hex). Leave malformed ones alone.
    HTML_NUMERIC_ENTITY_REGEX
        .replace_all(&step1, |caps: &regex::Captures| {
            let body = &caps[1];
            let code = if let Some(hex) = body.strip_prefix('x').or_else(|| body.strip_prefix('X')) {
                u32::from_str_radix(hex, 16).ok()
            } else {
                body.parse::<u32>().ok()
            };
            match code.and_then(char::from_u32) {
                Some(c) => c.to_string(),
                None => caps[0].to_string(),
            }
        })
        .into_owned()
}

/// Delete GLYPH<N> / `/uni...` / `/gN` markers (Case 7). These are
/// font-local glyph indices that survive PDF extraction when a
/// ToUnicode CMap is missing. Per Case 9, there's no safe global
/// substitution (N is per-font, not per-corpus), so we delete.
pub fn strip_glyph_markers(text: &str) -> String {
    GLYPH_MARKER_REGEX.replace_all(text, "").into_owned()
}

/// Silently strip soft hyphens U+00AD (Case 13). Invisible format-only
/// chars; their presence inflates `charset_moji_ratio` despite having
/// no semantic content.
pub fn strip_soft_hyphens(text: &str) -> String {
    if !text.contains('\u{00AD}') {
        return text.to_string();
    }
    text.chars().filter(|&c| c != '\u{00AD}').collect()
}

/// Decode Adobe Symbol font PUA chars (Case 10a). 100% of the top-30
/// observed PUA chars in a sample math thesis were recovered via
/// this mapping. Chars not in the map pass through unchanged; the
/// per-char filter then decides whether to strip them.
pub fn decode_adobe_symbol_pua(text: &str) -> String {
    // Fast check: any PUA chars at all? If not, skip.
    if !text.chars().any(|c| {
        let cp = c as u32;
        (0xF000..=0xF8FF).contains(&cp)
    }) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match ADOBE_SYMBOL_PUA_MAP.get(&c) {
            Some(replacement) => out.push_str(replacement),
            None => out.push(c),
        }
    }
    out
}

/// Paragraph-reflow (Case 8): collapse soft-wrap sequences where a
/// line ends mid-sentence and the next line continues it. PDF column-
/// width line breaks look like `word1\t\n  word2`; we replace with
/// `word1 word2`. Hard breaks (blank lines, headings, tables,
/// separators, list markers, blockquotes, fenced code) are preserved.
///
/// Heuristic — only joins when:
/// - prior line is non-empty AND doesn't end with a sentence terminator
///   (`.?!:;·;·` or closing quote/bracket)
/// - next line is non-empty, doesn't start with `#|>*-` / list marker
///   / fenced-code marker, and starts with a letter or digit
/// - prior line doesn't look like a list item / heading either
///
/// Accounting: the whitespace chars (`\n`, `\t`, leading spaces on the
/// joined line) are replaced by a single space, so output is shorter
/// by `(removed_ws_len) - 1` chars per join.
// (reflow_paragraphs, can_join_lines, line_is_hard_break moved to md_module.rs.)

#[cfg(test)]
mod wave2_tests {
    use super::*;

    #[test]
    fn decode_html_named_core_entities() {
        assert_eq!(decode_html_entities("a &amp; b"), "a & b");
        assert_eq!(decode_html_entities("&lt;div&gt;"), "<div>");
        assert_eq!(decode_html_entities("&quot;hi&quot;"), "\"hi\"");
        assert_eq!(decode_html_entities("&apos;test&apos;"), "'test'");
    }

    #[test]
    fn decode_html_numeric_entities_decimal_and_hex() {
        assert_eq!(decode_html_entities("&#38;"), "&");
        assert_eq!(decode_html_entities("&#x26;"), "&");
        assert_eq!(decode_html_entities("&#8364;"), "€");
        assert_eq!(decode_html_entities("&#x20AC;"), "€");
    }

    #[test]
    fn decode_html_unknown_entity_passes_through() {
        // `&fakename;` isn't in the map — leave as-is.
        assert_eq!(decode_html_entities("a &fakename; b"), "a &fakename; b");
    }

    #[test]
    fn decode_html_handles_mixed_content() {
        assert_eq!(
            decode_html_entities("&lt; item &gt;Αθήνα&lt;/ item &gt;"),
            "< item >Αθήνα</ item >"
        );
    }

    #[test]
    fn strip_glyph_markers_handles_all_variants() {
        assert_eq!(strip_glyph_markers("a GLYPH<216> b"), "a  b");
        assert_eq!(strip_glyph_markers("a /uni03B1 b"), "a  b");
        assert_eq!(strip_glyph_markers("a /g12 b /gid34 c"), "a  b  c");
        assert_eq!(
            strip_glyph_markers("GLYPH<1> GLYPH<216> GLYPH<99>"),
            "  "
        );
    }

    #[test]
    fn strip_glyph_markers_leaves_unrelated_text() {
        assert_eq!(strip_glyph_markers("καλημέρα"), "καλημέρα");
        // Empty `<>` not matched (regex requires 1+ chars inside).
        assert_eq!(strip_glyph_markers("GLYPH<>"), "GLYPH<>");
    }

    #[test]
    fn strip_glyph_markers_handles_long_attribute_form() {
        // PDF extractors sometimes emit `GLYPH<c=N,font=/Subset+Family>`.
        let input = "prefix GLYPH<c=3,font=/QCMXYA+CenturyGothic> suffix";
        assert_eq!(strip_glyph_markers(input), "prefix  suffix");
    }

    #[test]
    fn strip_soft_hyphens_removes_u00ad() {
        let shy = '\u{00AD}';
        let input = format!("co{}operate", shy);
        assert_eq!(strip_soft_hyphens(&input), "cooperate");
    }

    #[test]
    fn strip_soft_hyphens_noop_when_absent() {
        assert_eq!(strip_soft_hyphens("plain text"), "plain text");
    }

    #[test]
    fn decode_pua_recovers_ascii_mirrored_chars() {
        assert_eq!(decode_adobe_symbol_pua("\u{F02D}"), "-");
        assert_eq!(decode_adobe_symbol_pua("\u{F03D}"), "=");
        assert_eq!(decode_adobe_symbol_pua("\u{F02B}"), "+");
    }

    #[test]
    fn decode_pua_recovers_greek_letters() {
        assert_eq!(decode_adobe_symbol_pua("\u{F061}"), "α");
        assert_eq!(decode_adobe_symbol_pua("\u{F06C}"), "λ");
        assert_eq!(decode_adobe_symbol_pua("\u{F06D}"), "μ");
    }

    #[test]
    fn decode_pua_recovers_math_operators() {
        assert_eq!(decode_adobe_symbol_pua("\u{F0A3}"), "≤");
        assert_eq!(decode_adobe_symbol_pua("\u{F0B3}"), "≥");
        assert_eq!(decode_adobe_symbol_pua("\u{F0CE}"), "∈");
    }

    #[test]
    fn decode_pua_skips_unmapped_chars() {
        // Some random PUA codepoint not in our map.
        let unmapped = '\u{F500}';
        let s = unmapped.to_string();
        assert_eq!(decode_adobe_symbol_pua(&s), s);
    }

    #[test]
    fn decode_pua_fast_path_noop_when_no_pua() {
        assert_eq!(decode_adobe_symbol_pua("plain Greek καλημέρα"), "plain Greek καλημέρα");
    }

    // (reflow_* tests moved to md_module.rs.)
}
