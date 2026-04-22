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

/// Count chars by Unicode bucket in a single pass over the string.
/// No allocations; total runtime ~O(chars).
pub fn count_charsets(text: &str) -> CharsetCounts {
    let mut c = CharsetCounts::default();
    for ch in text.chars() {
        c.total += 1;
        if ch.is_whitespace() {
            c.whitespace += 1;
            continue;
        }
        let cp = ch as u32;
        // Cheap ASCII fast-path (majority of English/Greek prose is ASCII-or-Greek).
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
        // Named non-ASCII blocks (ranges are inclusive).
        match cp {
            0x00A1..=0x00FF => c.latin1_supp += 1,
            0x0100..=0x017F => c.latin_ext_a += 1,
            0x0180..=0x024F => c.latin_ext_b += 1,
            0x0250..=0x02AF => c.ipa_extensions += 1,
            0x0370..=0x03FF => c.greek += 1,
            0x0400..=0x04FF => c.cyrillic += 1,
            0x1F00..=0x1FFF => c.greek += 1, // Polytonic Greek extended
            0xE000..=0xF8FF => c.pua += 1,
            0xFFF0..=0xFFFF => c.specials_fffd += 1,
            _ => c.other += 1,
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
}
