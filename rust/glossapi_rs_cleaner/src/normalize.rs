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

/// Tiered bucket for run-length normalization (2026-04-21 policy):
///   1, 2    → 1  (single char / accidental double collapse to one)
///   3       → 3  (unchanged — natural prose triple, e.g. ellipsis)
///   4..=10  → 3  (medium run — treat as an intentional short leader)
///   >10     → 20 (long run — treat as a long TOC-style leader, visible)
/// Uniform across dots and whitespace so the BPE sees a small fixed
/// vocabulary of leader-length tokens regardless of which fill a PDF used.
pub fn bucket_run_length(n: usize) -> usize {
    match n {
        0 | 1 => n,
        2 => 1,
        3 => 3,
        4..=10 => 3,
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
/// runs are treated as one run and emitted as N spaces (tabs are folded
/// to spaces as a side effect — tabs carry no semantic meaning beyond
/// layout, same as the whitespace run itself).
pub fn normalize_whitespace_runs(line: &str) -> Option<String> {
    // Fire if there's a 2+ whitespace run OR any tab at all (single-tab
    // folds to single-space independently of run length).
    if !WHITESPACE_RUN_REGEX.is_match(line) && !line.contains('\t') {
        return None;
    }
    let chars: Vec<char> = line.chars().collect();
    let mut out = String::with_capacity(line.len());
    let mut changed = false;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' || c == '\t' {
            let start = i;
            while i < chars.len() && (chars[i] == ' ' || chars[i] == '\t') {
                i += 1;
            }
            let n = i - start;
            let m = bucket_run_length(n);
            // A single tab collapsing to a single space is still a change.
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
    // Track code-fence state so we don't normalize `|----|`-shaped lines that
    // appear inside fenced code blocks (which must survive intact).
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
    // A GFM table row MUST contain at least one pipe. Without this check a
    // bare `----` (standalone separator) would be (mis)-parsed as a 1-cell
    // table separator and then collapsed to `| --- |` whenever the line
    // above happened to be non-empty.
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
    // A GFM table row MUST contain at least one pipe. A plain prose line
    // without pipes is NOT a 1-cell header — enforcing this keeps the
    // separator pre-pass from pairing up unrelated lines.
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
    fn bucket_run_length_tiered() {
        // {1, 2} → 1
        assert_eq!(bucket_run_length(0), 0);
        assert_eq!(bucket_run_length(1), 1);
        assert_eq!(bucket_run_length(2), 1);
        // {3} → 3 (unchanged)
        assert_eq!(bucket_run_length(3), 3);
        // {4..=10} → 3
        for n in 4..=10 {
            assert_eq!(bucket_run_length(n), 3, "n={n}");
        }
        // > 10 → 20
        assert_eq!(bucket_run_length(11), 20);
        assert_eq!(bucket_run_length(42), 20);
        assert_eq!(bucket_run_length(200), 20);
    }

    #[test]
    fn dot_runs_tiered() {
        // 2 dots → 1
        assert_eq!(normalize_dot_runs("word..here"), Some("word.here".to_string()));
        // 3 dots unchanged — natural prose ellipsis
        assert_eq!(normalize_dot_runs("wait... next"), None);
        // 4–10 dots → 3
        assert_eq!(normalize_dot_runs("Chapter 1 ..... 5"), Some("Chapter 1 ... 5".to_string()));
        assert_eq!(normalize_dot_runs("..........heads"), Some("...heads".to_string())); // 10 dots
        // >10 dots → 20
        let long = "x".to_string() + &".".repeat(42) + "y";
        let expected = "x".to_string() + &".".repeat(20) + "y";
        assert_eq!(normalize_dot_runs(&long), Some(expected));
        // No dots — fast path
        assert_eq!(normalize_dot_runs("no dots here"), None);
        // Single dot (sentence end) — unchanged
        assert_eq!(normalize_dot_runs("end of sentence."), None);
    }

    #[test]
    fn whitespace_runs_tiered() {
        // 2 spaces → 1
        assert_eq!(
            normalize_whitespace_runs("a  b"),
            Some("a b".to_string())
        );
        // 3 spaces — unchanged
        assert_eq!(normalize_whitespace_runs("a   b"), None);
        // 4 spaces → 3
        assert_eq!(
            normalize_whitespace_runs("a    b"),
            Some("a   b".to_string())
        );
        // 10 spaces → 3
        assert_eq!(
            normalize_whitespace_runs("a          b"),
            Some("a   b".to_string())
        );
        // 11+ spaces → 20
        let long = format!("a{}b", " ".repeat(42));
        let expected = format!("a{}b", " ".repeat(20));
        assert_eq!(normalize_whitespace_runs(&long), Some(expected));
        // Tabs always fold to spaces, then bucket
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
    fn gfm_table_separator_single_column() {
        let text = "| Header |\n|--------|\n| cell |\n";
        let reps = scan_gfm_table_separators(text);
        assert_eq!(reps.len(), 1);
        assert_eq!(reps.get(&1).map(String::as_str), Some("| --- |"));
    }

    #[test]
    fn gfm_table_separator_alone_is_not_a_table() {
        // A bare `----` without any pipe in either the separator or the line
        // above is NOT a table separator. Protects standalone thematic
        // breaks from being mis-identified as table separators.
        let text = "Some prose\n----\nMore prose\n";
        let reps = scan_gfm_table_separators(text);
        assert!(reps.is_empty());
    }

    #[test]
    fn gfm_table_separator_inside_fenced_code_not_normalized() {
        // Tables inside a fenced code block are code, not tables.
        let text = "Before\n```\n| H1 | H2 |\n|----|----|\n| a | b |\n```\nAfter\n";
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
