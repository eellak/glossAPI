//! Pre-clean doc-level charset analysis — fast Unicode-block counting
//! exposed to Python via `analyze_charset`.
//!
//! Signal basis (from 69-doc 2026-04-22 Gemini review):
//!
//! - `moji_residue_ratio` — fraction of chars in blocks that act as
//!   mojibake substitutes for real Greek content. Above ~0.30 the
//!   Gemini sample had 0 false positives against "subject_clear=yes"
//!   docs.
//! - `ascii_punct_ratio` — fraction of chars in ASCII punctuation/symbol
//!   range (catches font-substitution mojibake where Greek renders as
//!   `!"#$%&'()…`). Above ~0.30 above baseline indicates the same.
//! - `greek_letter_ratio` — fraction of chars that are actual Greek. A
//!   doc with < 10% Greek is not a Greek-corpus candidate regardless
//!   of the other signals.
//!
//! Performance: single pass over `.chars()`, branchless classification
//! per codepoint, no allocations. ~500 MB/s on a single core.

use pyo3::prelude::*;
use pyo3::types::PyDict;


/// Per-doc charset counts, returned by `analyze_charset`.
///
/// `total` = total chars (including whitespace). All other counts
/// exclude whitespace. `other` = chars not in any named bucket.
#[derive(Default, Debug, Clone)]
pub struct CharsetCounts {
    pub total: usize,
    pub whitespace: usize,
    pub greek: usize,           // U+0370..=U+03FF, U+1F00..=U+1FFF
    pub latin_letters: usize,   // ASCII a-zA-Z
    pub digits: usize,          // ASCII 0-9
    pub ascii_punct: usize,     // ASCII printable non-letter/digit
    pub latin1_supp: usize,     // U+00A1..=U+00FF
    pub latin_ext_a: usize,     // U+0100..=U+017F
    pub latin_ext_b: usize,     // U+0180..=U+024F
    pub ipa_extensions: usize,  // U+0250..=U+02AF
    pub cyrillic: usize,        // U+0400..=U+04FF
    pub pua: usize,             // U+E000..=U+F8FF
    pub specials_fffd: usize,   // U+FFF0..=U+FFFF
    pub other: usize,
}

/// Decide whether a single line should be excluded from the content-
/// ratio denominator. These lines are format scaffolding — their chars
/// are not prose and should not bias the mojibake / language signals.
///
/// Excluded classes (2026-04-23 per user guidance):
/// - MD table rows (contain `|` pipes; parser check minimal)
/// - Standalone separator lines (`---`, `___`, `***`, `===`, `(?:\\_){4,}`,
///   and em-dash / horizontal-bar / box-drawing variants)
/// - Dot-leader lines (runs of `.` only, possibly with whitespace)
/// - Horizontal-rule lines (long runs of `_` or `-`)
/// - Block-HTML-comment-only lines (`<!-- … -->`)
///
/// LaTeX math regions (`$$…$$`) are handled by the caller via state
/// because they span multiple lines.
/// Strip inline HTML-comment spans (`<!-- … -->`) from a single line.
/// Returns a string with the comment regions removed. Unterminated
/// `<!--` at end-of-line is tolerated by dropping from `<!--` to EOL.
pub fn strip_html_comments(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        let after_open = &rest[start + 4..];
        match after_open.find("-->") {
            Some(end_rel) => {
                rest = &after_open[end_rel + 3..];
            }
            None => {
                // Unterminated — drop rest of line.
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

pub fn is_format_scaffolding_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false; // empty lines don't contribute chars anyway
    }
    // MD table row: starts and ends with `|`, and has at least one interior `|`
    if trimmed.starts_with('|') && trimmed.ends_with('|') {
        if trimmed.chars().filter(|&c| c == '|').count() >= 2 {
            return true;
        }
    }
    // Standalone separator / dot-leader line: entire trimmed content is
    // runs of separator chars only.
    let all_sep = trimmed.chars().all(|c| matches!(
        c,
        '-' | '_' | '*' | '=' | '.' | '·' | '\u{2014}' | '\u{2015}'
        | '\u{2500}' | '\u{2550}' | '\\' | ' ' | '\t'
    ));
    if all_sep && trimmed.chars().any(|c| !c.is_whitespace()) {
        return true;
    }
    // HTML-comment-only line.
    if trimmed.starts_with("<!--") && trimmed.ends_with("-->") {
        return true;
    }
    false
}

/// Count chars by Unicode bucket over the string, SKIPPING lines that
/// are pure format scaffolding (MD tables, separator / dot-leader,
/// HTML-comment-only) and block LaTeX math regions. The counts reflect
/// actual content, so the derived ratios measure content-language /
/// content-mojibake / content-punct density — not layout overhead.
///
/// No allocations beyond the line iterator; O(chars).
pub fn count_charsets(text: &str) -> CharsetCounts {
    let mut c = CharsetCounts::default();
    let mut in_latex_block = false;
    for line in text.lines() {
        // LaTeX $$...$$ block tracking. A line containing a single `$$`
        // toggles the state. A line with TWO `$$` opens-and-closes and
        // stays in the same state.
        let dollar_pairs = line.matches("$$").count();
        let starts_or_ends_math = dollar_pairs % 2 == 1;
        if in_latex_block {
            // We're inside a math block — skip this line's content.
            if starts_or_ends_math {
                in_latex_block = false;
            }
            continue;
        }
        if starts_or_ends_math {
            // Entering a math block; skip this line too.
            in_latex_block = true;
            continue;
        }
        if is_format_scaffolding_line(line) {
            // Skip entirely — neither numerator nor denominator affected.
            continue;
        }
        // Strip inline HTML-comment spans (`<!-- image -->`, `<!-- text-missing -->`,
        // etc.) before per-char counting — they're markers from upstream
        // extraction/cleaning, not prose.
        let line_stripped = strip_html_comments(line);
        for ch in line_stripped.chars() {
            c.total += 1;
            if ch.is_whitespace() {
                c.whitespace += 1;
                continue;
            }
            let cp = ch as u32;
            if cp < 0x80 {
                if ch.is_ascii_alphabetic() {
                    c.latin_letters += 1;
                } else if ch.is_ascii_digit() {
                    c.digits += 1;
                } else if cp >= 0x21 && cp <= 0x7E {
                    c.ascii_punct += 1;
                } else {
                    c.other += 1;
                }
                continue;
            }
            match cp {
                0x00A1..=0x00FF => c.latin1_supp += 1,
                0x0100..=0x017F => c.latin_ext_a += 1,
                0x0180..=0x024F => c.latin_ext_b += 1,
                0x0250..=0x02AF => c.ipa_extensions += 1,
                0x0370..=0x03FF => c.greek += 1,
                0x0400..=0x04FF => c.cyrillic += 1,
                0x1F00..=0x1FFF => c.greek += 1,
                0xE000..=0xF8FF => c.pua += 1,
                0xFFF0..=0xFFFF => c.specials_fffd += 1,
                _ => c.other += 1,
            }
        }
    }
    c
}

/// Derived ratios used by the three-rule charset-quality filter.
#[derive(Debug, Clone)]
pub struct CharsetRatios {
    pub greek_letter_ratio: f64,   // greek / non_whitespace
    pub moji_residue_ratio: f64,   // (latin1_supp + ipa + pua + specials_fffd + latin_ext_b) / non_ws
    pub ascii_punct_ratio: f64,    // ascii_punct / non_ws
}

impl CharsetRatios {
    pub fn from_counts(c: &CharsetCounts) -> Self {
        let non_ws = (c.total - c.whitespace).max(1);
        let moji = c.latin1_supp + c.ipa_extensions + c.pua
            + c.specials_fffd + c.latin_ext_b;
        Self {
            greek_letter_ratio: c.greek as f64 / non_ws as f64,
            moji_residue_ratio: moji as f64 / non_ws as f64,
            ascii_punct_ratio: c.ascii_punct as f64 / non_ws as f64,
        }
    }
}

/// Python-exposed `analyze_charset(text) -> dict` with all counts +
/// derived ratios. Caller applies thresholds.
#[pyfunction]
pub fn analyze_charset(py: Python<'_>, text: &str) -> PyResult<PyObject> {
    let c = count_charsets(text);
    let r = CharsetRatios::from_counts(&c);
    let d = PyDict::new(py);
    d.set_item("total", c.total)?;
    d.set_item("whitespace", c.whitespace)?;
    d.set_item("greek", c.greek)?;
    d.set_item("latin_letters", c.latin_letters)?;
    d.set_item("digits", c.digits)?;
    d.set_item("ascii_punct", c.ascii_punct)?;
    d.set_item("latin1_supp", c.latin1_supp)?;
    d.set_item("latin_ext_a", c.latin_ext_a)?;
    d.set_item("latin_ext_b", c.latin_ext_b)?;
    d.set_item("ipa_extensions", c.ipa_extensions)?;
    d.set_item("cyrillic", c.cyrillic)?;
    d.set_item("pua", c.pua)?;
    d.set_item("specials_fffd", c.specials_fffd)?;
    d.set_item("other", c.other)?;
    d.set_item("greek_letter_ratio", r.greek_letter_ratio)?;
    d.set_item("moji_residue_ratio", r.moji_residue_ratio)?;
    d.set_item("ascii_punct_ratio", r.ascii_punct_ratio)?;
    Ok(d.into())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_pure_greek_hits_greek_block() {
        let c = count_charsets("καλημέρα κόσμε");
        assert!(c.greek > 10);
        assert_eq!(c.latin_letters, 0);
        let r = CharsetRatios::from_counts(&c);
        assert!(r.greek_letter_ratio > 0.9);
        assert_eq!(r.moji_residue_ratio, 0.0);
    }

    #[test]
    fn counts_pure_ascii_punct_hits_punct_bucket() {
        let c = count_charsets("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
        assert_eq!(c.greek, 0);
        assert!(c.ascii_punct > 20);
        let r = CharsetRatios::from_counts(&c);
        assert!(r.ascii_punct_ratio > 0.9);
    }

    #[test]
    fn counts_mojibake_micro_sign_hits_latin1() {
        // `µ` = U+00B5 — common Greek-mu mojibake substitute.
        let c = count_charsets("µµµµ text");
        assert_eq!(c.latin1_supp, 4);
        let r = CharsetRatios::from_counts(&c);
        assert!(r.moji_residue_ratio > 0.4);
    }

    #[test]
    fn counts_ipa_extensions_hit_bucket() {
        // IPA phonetic chars — common broken-OCR mojibake. Note `θ` is
        // U+03B8 (Greek theta), not IPA — so this string has 4 IPA chars
        // and 1 Greek char plus " greek" (ASCII).
        let c = count_charsets("ʃθəɐʒ greek");
        assert_eq!(c.ipa_extensions, 4);
        assert_eq!(c.greek, 1);
        let r = CharsetRatios::from_counts(&c);
        assert!(r.moji_residue_ratio >= 0.35);
    }

    #[test]
    fn counts_polytonic_greek_treated_as_greek() {
        // U+1F00..=U+1FFF should count as greek too.
        let c = count_charsets("ὁ λόγος ἀγαθός");
        let r = CharsetRatios::from_counts(&c);
        assert!(r.greek_letter_ratio > 0.7);
    }

    #[test]
    fn excludes_md_table_rows_from_counts() {
        // A table row should not pollute punct ratio with pipes.
        let text = "\
καλημέρα κόσμε
| Column | Value |
| --- | --- |
| alpha | 1 |
";
        let c = count_charsets(text);
        // Only "καλημέρα κόσμε" counted. No `|` should show up in punct.
        let r = CharsetRatios::from_counts(&c);
        assert!(r.ascii_punct_ratio < 0.05,
                "table pipes leaked into punct: got {}", r.ascii_punct_ratio);
        assert!(r.greek_letter_ratio > 0.9);
    }

    #[test]
    fn excludes_separator_and_dot_leader_lines() {
        let text = "\
καλημέρα
---------
..........
Αθήνα
___________
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        // Only the two Greek prose lines should count.
        assert!(r.greek_letter_ratio > 0.95,
                "separator chars leaked into denom: greek_ratio={}",
                r.greek_letter_ratio);
        assert!(r.ascii_punct_ratio < 0.05);
    }

    #[test]
    fn excludes_latex_block_math_region() {
        let text = "\
καλημέρα κόσμε
$$
\\alpha + \\beta = \\gamma
\\int_0^1 x \\, dx
$$
Αθήνα πόλη
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        // The math block's `\alpha \beta \gamma \int` backslash + letters
        // would normally register as latin_letters + ascii_punct. With
        // the block excluded, ratio is dominated by the Greek prose.
        assert!(r.greek_letter_ratio > 0.9,
                "latex block leaked: greek={} punct={} latin={}",
                r.greek_letter_ratio, r.ascii_punct_ratio,
                c.latin_letters);
        assert_eq!(c.latin_letters, 0);
    }

    #[test]
    fn excludes_html_comment_only_lines() {
        let text = "\
καλημέρα
<!-- image -->
κόσμε
<!-- table-removed -->
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        assert!(r.greek_letter_ratio > 0.95);
        assert_eq!(c.latin_letters, 0); // "image" / "table-removed" not counted
    }

    #[test]
    fn excludes_inline_html_comments_from_counts() {
        // <!-- image --> inserted inline on a content line must not
        // inflate ascii_punct (the `<`, `!`, `-`, `-`, `>` chars).
        let text = "καλημέρα <!-- image --> κόσμε";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        // Only Greek letters should count. No "image" latin letters,
        // no comment-syntax punct.
        assert_eq!(c.latin_letters, 0,
                   "inline comment leaked 'image' letters: {:?}", c);
        assert_eq!(c.ascii_punct, 0,
                   "inline comment leaked punct chars: {:?}", c);
        assert!(r.greek_letter_ratio > 0.95);
    }

    #[test]
    fn strip_html_comments_handles_multiple_and_unterminated() {
        assert_eq!(strip_html_comments("a <!-- x --> b <!-- y --> c"),
                   "a  b  c");
        // Unterminated drops rest of line.
        assert_eq!(strip_html_comments("a <!-- x b"), "a ");
        // Non-comment chars kept.
        assert_eq!(strip_html_comments("plain text"), "plain text");
    }

    #[test]
    fn format_line_exclusion_does_not_hide_mojibake_in_prose() {
        // Regression: we still count chars on NON-format lines properly.
        let text = "\
normal Greek καλημέρα
| table | row |
corrupted µµµµµµµ chars
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        // The µ chars are on a content line and must still register
        // in moji_residue_ratio.
        assert!(r.moji_residue_ratio > 0.1,
                "lost mojibake on content line: {:?}",
                r);
    }
}
