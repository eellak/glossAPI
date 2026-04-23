//! LaTeX-segment handling — single source of truth for finding and
//! manipulating LaTeX regions in cleaner inputs.
//!
//! Concentrating per-text-type logic here (per the
//! `feedback_group_cleaner_features_by_text_type` rule) so that
//! consumers — charset ratio counting, repetition cropping, and any
//! future LaTeX-aware passes — share one detector instead of each
//! re-implementing `$$` toggle tracking.
//!
//! Currently handles:
//! - Multi-line `$$ … $$` blocks (matches the existing
//!   `charset_module::count_charsets` state-machine behaviour)
//! - Single-line `$$ … $$` regions on the SAME line (the gap noted in
//!   `user_review_notes.md` Case 5 — observed in math theses where the
//!   PDF extractor collapses each equation onto one line)
//!
//! Deferred (planned but not yet wired):
//! - Inline `$ … $` math
//! - `\begin{env} … \end{env}` environments
//!
//! The repetition-crop helpers below are a Rust port of
//! `_detect_repeated_char_cut` and `_detect_repeated_lines_cut` from
//! `src/glossapi/ocr/utils/cleaning.py`. Same semantics, applied per
//! LaTeX span rather than to whole OCR outputs.

use pyo3::prelude::*;

/// Half-open span `[start, end)` inside a parent string, with metadata
/// on which LaTeX delimiter pattern matched. Byte offsets, not chars
/// — caller is responsible for slicing on char boundaries (we never
/// split inside multibyte chars because `$` is ASCII).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatexSpan {
    pub start: usize,
    pub end: usize,
    pub kind: LatexKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatexKind {
    /// Both `$$` delimiters on the same line.
    InlineDoubleDollar,
    /// `$$` opens on one line, closes on a later line.
    BlockDoubleDollar,
}

/// Locate every `$$ … $$` region in `text`, in source order.
///
/// Two-pass: first all single-line `$$…$$` regions, then a state-
/// machine sweep over remaining text for cross-line blocks. Single-
/// line spans win over the block detector if they overlap a line, so
/// docs with mixed inline + block math don't double-count.
pub fn find_dollar_dollar_spans(text: &str) -> Vec<LatexSpan> {
    let bytes = text.as_bytes();
    let mut spans: Vec<LatexSpan> = Vec::new();

    // Pass 1: walk once. For each `$$` mark we find, look ahead for a
    // closing `$$` on the SAME line; if found → InlineDoubleDollar
    // and skip past it. Otherwise → start of a block; look for the
    // closing `$$` on a later line.
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'$' && bytes[i + 1] == b'$' {
            let start = i;
            // Find end of this line (next \n) and the next `$$` on
            // the same line.
            let mut j = i + 2;
            let mut nl = None;
            while j < bytes.len() {
                if bytes[j] == b'\n' {
                    nl = Some(j);
                    break;
                }
                if j + 1 < bytes.len() && bytes[j] == b'$' && bytes[j + 1] == b'$' {
                    // Inline: closes on same line.
                    spans.push(LatexSpan {
                        start,
                        end: j + 2,
                        kind: LatexKind::InlineDoubleDollar,
                    });
                    i = j + 2;
                    break;
                }
                j += 1;
            }
            if i != j + 2 {
                // No same-line close; either we hit \n or EOF.
                let line_end = nl.unwrap_or(bytes.len());
                // Block: look for next `$$` after line_end.
                let mut k = line_end + 1;
                let mut close = None;
                while k + 1 < bytes.len() {
                    if bytes[k] == b'$' && bytes[k + 1] == b'$' {
                        close = Some(k);
                        break;
                    }
                    k += 1;
                }
                if let Some(c) = close {
                    spans.push(LatexSpan {
                        start,
                        end: c + 2,
                        kind: LatexKind::BlockDoubleDollar,
                    });
                    i = c + 2;
                } else {
                    // Unclosed `$$` → don't claim a span; skip the
                    // opener and continue (matches charset_module
                    // tolerance).
                    i = start + 2;
                }
            }
        } else {
            i += 1;
        }
    }
    spans
}

/// Detect a single-character repetition cut point inside `s`. Returns
/// `Some(idx)` where `idx` is the byte position to truncate at — the
/// repeated run is allowed up to `threshold` chars, anything beyond
/// is cut.
///
/// Direct port of `_detect_repeated_char_cut` from the OCR Python
/// utility. Runs reset across newlines. O(n) time, O(1) space.
pub fn detect_repeated_char_cut(s: &str, threshold: usize) -> Option<usize> {
    if threshold <= 1 {
        return Some(0);
    }
    let mut last_char: Option<char> = None;
    let mut run_len: usize = 0;
    let mut run_start: usize = 0;
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            last_char = None;
            run_len = 0;
            continue;
        }
        if Some(ch) == last_char {
            run_len += 1;
            if run_len >= threshold {
                // Keep up to `threshold` chars; cut after.
                // run_start is the BYTE index of the run's first char;
                // `threshold` chars later in BYTES requires walking.
                let mut byte_offset = run_start;
                let cur = &s[run_start..];
                for (n, (off, _ch)) in cur.char_indices().enumerate() {
                    if n == threshold {
                        byte_offset = run_start + off;
                        return Some(byte_offset);
                    }
                }
                // If we hit EOS before `threshold` chars, no cut needed.
                return Some(s.len());
            }
        } else {
            last_char = Some(ch);
            run_len = 1;
            run_start = i;
        }
    }
    None
}

/// Detect a repeated-line cut point. Returns the byte index where the
/// (`threshold` + 1)-th repetition of a line starts. Lines are compared
/// after `trim()`. O(n) time.
///
/// Direct port of `_detect_repeated_lines_cut` from the OCR Python
/// utility.
pub fn detect_repeated_lines_cut(s: &str, threshold: usize) -> Option<usize> {
    if threshold <= 1 {
        return Some(0);
    }
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut prev_norm: Option<&str> = None;
    let mut run_count: usize = 1;
    let mut i = 0;
    // Track previous line's range so we can do trim comparison.
    let mut prev_line_buf: String = String::new();
    while i <= n {
        let mut j = i;
        while j < n && bytes[j] != b'\n' {
            j += 1;
        }
        let line = &s[i..j];
        let norm = line.trim();
        if let Some(p) = prev_norm {
            if norm == p {
                run_count += 1;
                if run_count > threshold {
                    return Some(i);
                }
            } else {
                prev_line_buf.clear();
                prev_line_buf.push_str(norm);
                prev_norm = Some(unsafe {
                    std::mem::transmute::<&str, &'static str>(prev_line_buf.as_str())
                });
                run_count = 1;
            }
        } else {
            prev_line_buf.clear();
            prev_line_buf.push_str(norm);
            prev_norm = Some(unsafe {
                std::mem::transmute::<&str, &'static str>(prev_line_buf.as_str())
            });
            run_count = 1;
        }
        i = j + 1;
        if i > n {
            break;
        }
    }
    None
}

/// Apply per-LaTeX-segment repetition cropping to `text`. For each
/// detected `$$…$$` span, run the OCR-style repetition detectors
/// against the inner content; if a cut is found, truncate the segment
/// at that point and re-close with `$$`. Returns the rewritten text.
///
/// `char_threshold` and `line_threshold` are passed straight through
/// to the underlying detectors. Pass small values (e.g. 30 char / 3
/// line) for tight LaTeX-segment cropping; the OCR defaults (200 /
/// 10) are tuned for whole-page outputs.
///
/// Pass `enable=false` to short-circuit and return `text` unchanged
/// — the caller's gate on this feature.
pub fn crop_latex_repetitions(
    text: &str,
    enable: bool,
    char_threshold: usize,
    line_threshold: usize,
) -> String {
    if !enable {
        return text.to_string();
    }
    let spans = find_dollar_dollar_spans(text);
    if spans.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for span in &spans {
        // Copy text before this span verbatim.
        out.push_str(&text[cursor..span.start]);
        // Inner content excludes the leading `$$` and trailing `$$`.
        let inner_start = span.start + 2;
        let inner_end = span.end.saturating_sub(2);
        if inner_end <= inner_start {
            // Degenerate; copy the whole span as-is.
            out.push_str(&text[span.start..span.end]);
            cursor = span.end;
            continue;
        }
        let inner = &text[inner_start..inner_end];
        let cut_char = detect_repeated_char_cut(inner, char_threshold);
        let cut_line = detect_repeated_lines_cut(inner, line_threshold);
        let cut = match (cut_char, cut_line) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        out.push_str("$$");
        match cut {
            Some(idx) if idx < inner.len() => {
                out.push_str(&inner[..idx]);
                // Mark the crop visibly so review can spot it.
                out.push_str(" /*…repetition cropped…*/ ");
            }
            _ => {
                out.push_str(inner);
            }
        }
        out.push_str("$$");
        cursor = span.end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Python-exposed: `crop_latex_repetitions(text, enable, char_threshold,
/// line_threshold) -> str`. Defaults match the OCR equivalent.
#[pyfunction]
#[pyo3(signature = (text, enable=false, char_threshold=30, line_threshold=3))]
pub fn crop_latex_repetitions_py(
    text: &str,
    enable: bool,
    char_threshold: usize,
    line_threshold: usize,
) -> String {
    crop_latex_repetitions(text, enable, char_threshold, line_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_inline_double_dollar_span() {
        let t = "before $$x = y$$ after";
        let s = find_dollar_dollar_spans(t);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].kind, LatexKind::InlineDoubleDollar);
        assert_eq!(&t[s[0].start..s[0].end], "$$x = y$$");
    }

    #[test]
    fn finds_block_double_dollar_span() {
        let t = "before $$\nx = y\n$$ after";
        let s = find_dollar_dollar_spans(t);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].kind, LatexKind::BlockDoubleDollar);
        assert!(t[s[0].start..s[0].end].starts_with("$$"));
        assert!(t[s[0].start..s[0].end].ends_with("$$"));
    }

    #[test]
    fn finds_multiple_inline_spans_in_same_doc() {
        let t = "$$a$$ middle $$b$$ end $$c$$";
        let s = find_dollar_dollar_spans(t);
        assert_eq!(s.len(), 3);
        for sp in &s {
            assert_eq!(sp.kind, LatexKind::InlineDoubleDollar);
        }
    }

    #[test]
    fn detect_repeated_char_cut_finds_run() {
        // 250 dots after threshold-200: cut should land 200 chars in.
        let s = "alpha ".to_string() + &".".repeat(250) + " beta";
        let cut = detect_repeated_char_cut(&s, 200).expect("cut");
        // run starts at byte 6 (after "alpha "), keep 200 → cut at 206.
        assert_eq!(cut, 6 + 200);
    }

    #[test]
    fn detect_repeated_char_cut_resets_on_newline() {
        // Newline interrupts the run — so 100 + newline + 100 = no cut.
        let s = ".".repeat(100) + "\n" + &".".repeat(100);
        assert_eq!(detect_repeated_char_cut(&s, 200), None);
    }

    #[test]
    fn detect_repeated_lines_cut_finds_repeats() {
        let s = "alpha\nbeta\nbeta\nbeta\nbeta\ngamma";
        // beta appears 4 times; threshold=3 → 4 > 3 → cut at start of 4th.
        let cut = detect_repeated_lines_cut(s, 3).expect("cut");
        // 4th 'beta' starts at byte: "alpha\n" = 6, "beta\n" = 5 each
        // → 1st beta=6, 2nd=11, 3rd=16, 4th=21
        assert_eq!(cut, 21);
    }

    #[test]
    fn crop_latex_repetitions_disabled_is_noop() {
        let t = "$$".to_string() + &"+".repeat(100) + "$$";
        assert_eq!(crop_latex_repetitions(&t, false, 10, 3), t);
    }

    #[test]
    fn crop_latex_repetitions_crops_char_run_inside_inline_math() {
        // 100 `+` inside $$..$$ with threshold 10 → cropped.
        let t = "before $$a + ".to_string() + &"+".repeat(100) + " b$$ after";
        let out = crop_latex_repetitions(&t, true, 10, 100);
        assert!(out.contains("repetition cropped"), "out = {}", out);
        assert!(out.starts_with("before $$a + "));
        assert!(out.ends_with("$$ after"));
        // Crop marker present means we cut the long + run; full original
        // had 100 plus chars, output should have far fewer.
        let plus_count = out.chars().filter(|&c| c == '+').count();
        assert!(plus_count < 30, "expected cropped + run, got {} +", plus_count);
    }

    #[test]
    fn crop_latex_repetitions_passes_clean_math_through() {
        let t = "before $$x^2 + y^2 = z^2$$ after";
        let out = crop_latex_repetitions(t, true, 30, 3);
        assert_eq!(out, t);
    }

    #[test]
    fn crop_latex_repetitions_handles_block_math() {
        let inner = "x = y\n".to_string() + &"+".repeat(60);
        let t = format!("$$\n{}\n$$", inner);
        let out = crop_latex_repetitions(&t, true, 10, 100);
        assert!(out.contains("repetition cropped"));
    }
}
