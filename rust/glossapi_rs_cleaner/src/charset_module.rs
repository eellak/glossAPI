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
    pub greek: usize,          // U+0370..=U+03FF, U+1F00..=U+1FFF
    pub latin_letters: usize,  // ASCII a-zA-Z
    pub digits: usize,         // ASCII 0-9
    pub ascii_punct: usize,    // ASCII printable non-letter/digit
    pub latin1_supp: usize,    // U+00A1..=U+00FF
    pub latin_ext_a: usize,    // U+0100..=U+017F
    pub latin_ext_b: usize,    // U+0180..=U+024F
    pub ipa_extensions: usize, // U+0250..=U+02AF
    pub cyrillic: usize,       // U+0400..=U+04FF
    pub pua: usize,            // U+E000..=U+F8FF
    pub specials_fffd: usize,  // U+FFF0..=U+FFFF
    pub other: usize,
    /// Subset of `latin1_supp` that is LEGITIMATE non-mojibake content
    /// in a Greek corpus: `«» · § ° ® © ™` + ASCII-currency cousins.
    /// Tracked separately so the moji numerator can subtract them
    /// (they inflate `charset_moji_ratio` on clean thesis/EU docs —
    /// see `reports/user_review_notes.md` Case 13).
    pub latin1_legit_extras: usize,
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
    let all_sep = trimmed.chars().all(|c| {
        matches!(
            c,
            '-' | '_'
                | '*'
                | '='
                | '.'
                | '·'
                | '\u{2014}'
                | '\u{2015}'
                | '\u{2500}'
                | '\u{2550}'
                | '\\'
                | ' '
                | '\t'
        )
    });
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
/// HTML-comment-only) and ALL `$$…$$` LaTeX regions (single-line AND
/// multi-line).
///
/// LaTeX region exclusion delegates to
/// `latex_module::find_dollar_dollar_spans` — the same detector the
/// repetition cropper uses — so the behavior is consistent across
/// passes. This fixes v6-03: the prior state-machine only excluded
/// multi-line `$$\n…\n$$` blocks, which inflated
/// `charset_punct_ratio` on any doc where Docling collapsed math onto
/// single lines (typical output).
///
/// No allocations beyond the line iterator + the span vector; O(chars).
pub fn count_charsets(text: &str) -> CharsetCounts {
    let mut c = CharsetCounts::default();
    // One-shot span scan: byte ranges of every `$$…$$` region. Both
    // inline (same-line) and block (multi-line) spans are returned.
    let latex_spans = crate::latex_module::find_dollar_dollar_spans(text);
    // Walk lines, tracking the byte offset of each line so we can
    // check whether each char falls inside a LaTeX span.
    let mut line_start: usize = 0;
    for line in text.lines() {
        // Recover the line's byte offset inside `text`. `str::lines()`
        // doesn't give it directly; we track it manually.
        let line_len = line.len();
        let line_end = line_start + line_len;

        // If this ENTIRE line is inside a LaTeX span, skip wholesale.
        let whole_line_in_latex = latex_spans
            .iter()
            .any(|span| span.start <= line_start && line_end <= span.end);
        if whole_line_in_latex {
            line_start = line_end + 1; // + newline
            continue;
        }

        if is_format_scaffolding_line(line) {
            line_start = line_end + 1;
            continue;
        }
        // Strip inline HTML-comment spans (`<!-- image -->`, `<!-- text-missing -->`,
        // etc.) before per-char counting — they're markers from upstream
        // extraction/cleaning, not prose.
        let line_stripped = strip_html_comments(line);
        // Walk chars along the ORIGINAL line so we can check each
        // char's byte offset against `latex_spans`. The comment-strip
        // can shift content, but since inline HTML comments never
        // overlap `$$…$$` regions in practice, we can do the LaTeX
        // exclusion against the original-line offsets and then still
        // use `line_stripped` for the per-char counting. Simplest
        // correct implementation: compute a "byte in any $$ span?"
        // predicate closure over the original line offsets, then
        // consume `line_stripped` char-by-char with that predicate.
        //
        // In practice, Docling corpus MD doesn't intermix `$$` inside
        // `<!-- … -->`, so this is safe. Re-walking offsets per char
        // would cost us an O(n*spans) scan; we use a single advancing
        // cursor instead.
        let mut byte_off = line_start;
        // We walk line_stripped. But byte_off tracks the ORIGINAL
        // line's byte positions — an approximation when
        // strip_html_comments removed bytes. For the common case
        // (no inline HTML comments inside math), line_stripped == line
        // and byte_off tracks correctly.
        let stripped_equals_original = line_stripped.len() == line_len;
        for ch in line_stripped.chars() {
            if stripped_equals_original {
                // Precise path: check this char's byte offset against
                // the LaTeX spans.
                let in_latex = latex_spans
                    .iter()
                    .any(|span| span.start <= byte_off && byte_off < span.end);
                byte_off += ch.len_utf8();
                if in_latex {
                    continue;
                }
            }
            // else: HTML-comment-stripped line → fall through to the
            // plain per-char count (LaTeX-inside-HTML-comment isn't a
            // known corpus pattern; accept the approximation).
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
                0x00A1..=0x00FF => {
                    c.latin1_supp += 1;
                    // Track the legit-punct / legit-symbol subset for
                    // moji-FP subtraction. Guillemets, middle-dot, and
                    // common bibliography/currency symbols.
                    match cp {
                        0x00AB | // «
                        0x00BB | // »
                        0x00B7 | // ·
                        0x00A7 | // §
                        0x00B0 | // °
                        0x00AE | // ®
                        0x00A9 | // ©
                        0x00A2 | // ¢
                        0x00A3 | // £
                        0x00A5   // ¥
                          => c.latin1_legit_extras += 1,
                        _ => {}
                    }
                }
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
        // Advance byte-offset tracker past this line + newline.
        line_start = line_end + 1;
    }
    c
}

/// Derived ratios used by the charset-quality filter.
#[derive(Debug, Clone)]
pub struct CharsetRatios {
    pub greek_letter_ratio: f64, // greek / non_whitespace
    /// Fraction of chars that land in mojibake-substitute buckets.
    ///
    /// Numerator = `latin1_supp − latin1_legit_extras + latin_ext_a
    /// + latin_ext_b + ipa + cyrillic + pua + specials_fffd`.
    ///
    /// Expanded 2026-04-24 to add `latin_ext_a` (Polish / Czech /
    /// Turkish — not Greek, signals contamination) and `cyrillic`
    /// (same logic). Subtracted `latin1_legit_extras` (`«» · § °
    /// ® © ¢ £ ¥ ™`) because those are legitimate Greek / bibliography
    /// punctuation and were inflating the ratio on clean EU /
    /// thesis docs (Case 13 of `reports/user_review_notes.md`).
    pub moji_residue_ratio: f64,
    pub ascii_punct_ratio: f64, // ascii_punct / non_ws
}

impl CharsetRatios {
    pub fn from_counts(c: &CharsetCounts) -> Self {
        let non_ws = (c.total - c.whitespace).max(1);
        // 2026-04-24: widened to include latin_ext_a + cyrillic; subtract
        // the legit-extras subset of latin1_supp.
        let moji = c.latin1_supp.saturating_sub(c.latin1_legit_extras)
            + c.latin_ext_a
            + c.latin_ext_b
            + c.ipa_extensions
            + c.cyrillic
            + c.pua
            + c.specials_fffd;
        Self {
            greek_letter_ratio: c.greek as f64 / non_ws as f64,
            moji_residue_ratio: moji as f64 / non_ws as f64,
            ascii_punct_ratio: c.ascii_punct as f64 / non_ws as f64,
        }
    }
}

/// Count non-empty lines + chars on those lines. A line is non-empty
/// if its trimmed form is non-empty AND isn't one of the known marker
/// comments. Char count sums chars on counted lines (newlines excluded).
///
/// Previously this was done in Python (`_non_empty_stats`) for every
/// cleaner-driver doc twice (input + output text). Moving it to Rust
/// eliminates ~10k Python-loop iterations per large doc.
pub fn non_empty_stats(text: &str) -> (usize, usize, usize) {
    const MARKERS: &[&str] = &[
        "<!-- line-removed -->",
        "<!-- text-missing -->",
        "<!-- table-removed -->",
    ];
    let mut total_lines = 0usize;
    let mut non_empty_lines = 0usize;
    let mut non_empty_chars = 0usize;
    for line in text.split('\n') {
        total_lines += 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if MARKERS.contains(&trimmed) {
            continue;
        }
        non_empty_lines += 1;
        non_empty_chars += line.chars().count();
    }
    (total_lines, non_empty_lines, non_empty_chars)
}

/// Python-exposed `non_empty_line_stats(text) -> (total, non_empty, chars)`.
#[pyfunction]
pub fn non_empty_line_stats(text: &str) -> (usize, usize, usize) {
    non_empty_stats(text)
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
    d.set_item("latin1_legit_extras", c.latin1_legit_extras)?;
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
        assert!(
            r.ascii_punct_ratio < 0.05,
            "table pipes leaked into punct: got {}",
            r.ascii_punct_ratio
        );
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
        assert!(
            r.greek_letter_ratio > 0.95,
            "separator chars leaked into denom: greek_ratio={}",
            r.greek_letter_ratio
        );
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
        assert!(
            r.greek_letter_ratio > 0.9,
            "latex block leaked: greek={} punct={} latin={}",
            r.greek_letter_ratio,
            r.ascii_punct_ratio,
            c.latin_letters
        );
        assert_eq!(c.latin_letters, 0);
    }

    // v6-03: single-line `$$…$$` must ALSO be excluded. The old toggle
    // state machine only handled multi-line blocks, so a math-heavy
    // doc where Docling collapsed every equation onto one line had its
    // `charset_punct_ratio` inflated by LaTeX syntax chars.
    #[test]
    fn v6_03_excludes_inline_single_line_double_dollar_math() {
        let text = "\
καλημέρα κόσμε
$$\\frac{a+b}{c} \\cdot \\int_0^1 x^2 \\, dx = \\gamma$$
Αθήνα πόλη
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        assert_eq!(
            c.latin_letters, 0,
            "single-line $$..$$ latin letters leaked: {:?}",
            c
        );
        assert!(
            r.ascii_punct_ratio < 0.05,
            "single-line $$..$$ punct leaked: ratio={}",
            r.ascii_punct_ratio
        );
        assert!(r.greek_letter_ratio > 0.9);
    }

    // Moji bucket expansion 2026-04-24: latin_ext_a + cyrillic in,
    // legit-extras subtracted.

    #[test]
    fn moji_includes_latin_ext_a() {
        // Polish / Czech / Turkish chars (U+0100..=U+017F) — foreign
        // language in a Greek corpus, now counted as moji.
        let c = count_charsets("Łódź ąęłń ğş text");
        assert!(
            c.latin_ext_a >= 8,
            "expected ≥8 latin_ext_a, got {}",
            c.latin_ext_a
        );
        let r = CharsetRatios::from_counts(&c);
        assert!(
            r.moji_residue_ratio > 0.4,
            "expected >0.4 moji ratio, got {}",
            r.moji_residue_ratio
        );
    }

    #[test]
    fn moji_includes_cyrillic() {
        let c = count_charsets("Привет миру здравствуй");
        assert!(c.cyrillic >= 18, "cyrillic count: {}", c.cyrillic);
        let r = CharsetRatios::from_counts(&c);
        assert!(
            r.moji_residue_ratio > 0.9,
            "expected >0.9 moji on pure Cyrillic, got {}",
            r.moji_residue_ratio
        );
    }

    #[test]
    fn moji_excludes_legit_greek_punctuation() {
        // «», middle-dot, §, °, ®, ©, ™, ¢, £, ¥ should NOT inflate
        // moji. A Greek sentence dense with these should have low moji.
        let c = count_charsets("«Καλημέρα»·«κόσμε»·Αθήνα§3°C£10®©");
        let r = CharsetRatios::from_counts(&c);
        assert!(
            c.latin1_legit_extras >= 9,
            "expected ≥9 legit_extras, got {}",
            c.latin1_legit_extras
        );
        assert!(
            r.moji_residue_ratio < 0.05,
            "legit Greek punct leaked into moji: ratio={} (latin1_supp={}, legit_extras={})",
            r.moji_residue_ratio,
            c.latin1_supp,
            c.latin1_legit_extras
        );
    }

    #[test]
    fn moji_still_catches_actual_mojibake_despite_legit_subtraction() {
        // Case 2 sample: CP1253→Latin-1 codepage mojibake. All chars
        // are in latin1_supp, none are in the legit-extras set.
        let c = count_charsets("Ï ñï, üù ï ëÜôùíá, í íáé êôç üëùí");
        let r = CharsetRatios::from_counts(&c);
        assert_eq!(
            c.latin1_legit_extras, 0,
            "no legit-extras should be in codepage-mojibake sample"
        );
        assert!(
            r.moji_residue_ratio > 0.85,
            "codepage mojibake should still trip moji: ratio={}",
            r.moji_residue_ratio
        );
    }

    #[test]
    fn v6_03_excludes_multiple_inline_double_dollars() {
        // Realistic math-paper shape: many $$…$$ spans on one line.
        let text = "\
καλημέρα
$$x^2$$ και $$y^3$$ και $$z = \\frac{1}{2}$$
Αθήνα
";
        let c = count_charsets(text);
        let r = CharsetRatios::from_counts(&c);
        // "και" (Greek) should still count; the math-inside-$$ should not.
        assert_eq!(c.latin_letters, 0, "multi-inline $$..$$ leaked latin");
        assert!(
            r.greek_letter_ratio > 0.9,
            "expected dominant greek, got {}",
            r.greek_letter_ratio
        );
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
        assert_eq!(
            c.latin_letters, 0,
            "inline comment leaked 'image' letters: {:?}",
            c
        );
        assert_eq!(
            c.ascii_punct, 0,
            "inline comment leaked punct chars: {:?}",
            c
        );
        assert!(r.greek_letter_ratio > 0.95);
    }

    #[test]
    fn strip_html_comments_handles_multiple_and_unterminated() {
        assert_eq!(
            strip_html_comments("a <!-- x --> b <!-- y --> c"),
            "a  b  c"
        );
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
        assert!(
            r.moji_residue_ratio > 0.1,
            "lost mojibake on content line: {:?}",
            r
        );
    }

    #[test]
    fn non_empty_stats_counts_lines_and_chars() {
        let text = "alpha\n\nbeta gamma\n";
        let (total, ne, nec) = non_empty_stats(text);
        assert_eq!(total, 4, "split('\\n') over 3 newlines yields 4 segments");
        assert_eq!(ne, 2);
        assert_eq!(nec, 5 + 10);
    }

    #[test]
    fn non_empty_stats_skips_marker_lines() {
        let text = "real line\n<!-- line-removed -->\n<!-- text-missing -->\n<!-- table-removed -->\nother\n";
        let (_, ne, nec) = non_empty_stats(text);
        assert_eq!(ne, 2, "marker lines must not count");
        assert_eq!(nec, "real line".len() + "other".len());
    }

    #[test]
    fn non_empty_stats_skips_whitespace_only_lines() {
        let text = "alpha\n   \n\t\n  \t \nbeta\n";
        let (_, ne, _) = non_empty_stats(text);
        assert_eq!(ne, 2);
    }

    #[test]
    fn non_empty_stats_uses_char_count_not_byte_count() {
        // καλημέρα = 8 codepoints, 16 bytes (each Greek char is 2 bytes UTF-8)
        let text = "καλημέρα\n";
        let (_, ne, nec) = non_empty_stats(text);
        assert_eq!(ne, 1);
        assert_eq!(nec, 8, "must count codepoints, not bytes (got {})", nec);
    }

    #[test]
    fn non_empty_line_stats_pyfunction_matches_internal() {
        // Regression: the PyO3 wrapper must not diverge from non_empty_stats.
        let text = "alpha\n<!-- table-removed -->\nbeta\n";
        assert_eq!(non_empty_line_stats(text), non_empty_stats(text));
    }
}
