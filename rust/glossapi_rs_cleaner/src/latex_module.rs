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

// ---------------------------------------------------------------------------
// LaTeX-syntax-aware element detection (2026-04-24, replaces the earlier
// generic token detector — user feedback: "a repetition that respects latex
// syntax, ie repetitions of latex elements, not just any repetition").
// ---------------------------------------------------------------------------

/// A parsed LaTeX element — what a reader would call a single math
/// atom: a bare command, a command with brace arguments, a letter-
/// or-command with subscript/superscript, or a balanced brace group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatexElement {
    /// Byte offset in the source LaTeX segment.
    pub start: usize,
    pub end: usize,
    /// Normalized canonical string (whitespace collapsed inside).
    pub canonical: String,
    /// Base portion (before `_` / `^`). For `x_{n+1}^m` this is `x`.
    /// For `\Omega_{...}` this is `\Omega`. For `\frac{1}{2}` with no
    /// sub/sup this is `\frac{1}{2}` (no separable base).
    pub base: String,
    /// Subscript value as an integer, IF the subscript is purely
    /// numeric. Used by monotonic-progression detector.
    pub sub_numeric: Option<i64>,
    /// Same for superscript.
    pub sup_numeric: Option<i64>,
}

/// Parse the next LaTeX element starting at or after `pos`. Skips
/// element-separator chars (whitespace, LaTeX thin spaces `\,` `\;` `\!`,
/// commas, plus/minus/equals as binary-op separators). Returns None at
/// EOS.
///
/// Element types parsed:
/// - Command: `\` + `[A-Za-z]+` (e.g. `\Omega`)
/// - Command with arg groups: `\frac{a}{b}`, `\mathbb{R}`, `\sum_{i=1}^{n}`
/// - Letter atom: single `[A-Za-z]` (e.g. `x`) + optional sub/sup
/// - Digit atom: run of digits (standalone numbers)
/// - Braced group: `{...}` as a single element
///
/// Brace balancing uses a stack; max depth tracked to avoid pathological
/// inputs (cap at 32 for safety).
fn next_latex_element(s: &str, pos: usize) -> Option<LatexElement> {
    let bytes = s.as_bytes();
    let n = bytes.len();
    let mut i = pos;

    // Skip inter-element separators.
    while i < n {
        let b = bytes[i];
        if b == b' ' || b == b'\t' || b == b'\n' || b == b',' || b == b';'
            || b == b'+' || b == b'-' || b == b'='
        {
            i += 1;
            continue;
        }
        // LaTeX thin-space markers: `\,` `\;` `\!` `\ ` (backslash-space).
        if b == b'\\' && i + 1 < n {
            let next = bytes[i + 1];
            if next == b',' || next == b';' || next == b'!' || next == b' ' {
                i += 2;
                continue;
            }
        }
        break;
    }
    if i >= n {
        return None;
    }

    let start = i;
    let first = bytes[i];

    // Base + suffix (sub/sup) accumulated here.
    let mut base = String::new();
    let mut canonical = String::new();

    // Parse the base.
    if first == b'\\' && i + 1 < n && (bytes[i + 1] as char).is_ascii_alphabetic() {
        // LaTeX command: `\name` + optional brace args.
        let name_start = i;
        i += 1;
        while i < n && (bytes[i] as char).is_ascii_alphabetic() {
            i += 1;
        }
        let name = &s[name_start..i];
        base.push_str(name);
        canonical.push_str(name);
        // Absorb immediate brace argument groups (greedy).
        while i < n && bytes[i] == b'{' {
            let arg_end = match find_balanced_close(s, i) {
                Some(e) => e,
                None => break,
            };
            canonical.push_str(&collapse_ws(&s[i..=arg_end]));
            base.push_str(&collapse_ws(&s[i..=arg_end]));
            i = arg_end + 1;
        }
    } else if first == b'{' {
        // Braced group as a standalone element.
        let end_idx = find_balanced_close(s, i);
        match end_idx {
            Some(e) => {
                let span = collapse_ws(&s[i..=e]);
                base.push_str(&span);
                canonical.push_str(&span);
                i = e + 1;
            }
            None => {
                // Unbalanced — treat as single char.
                base.push(first as char);
                canonical.push(first as char);
                i += 1;
            }
        }
    } else if (first as char).is_ascii_alphabetic() {
        // Single letter atom.
        base.push(first as char);
        canonical.push(first as char);
        i += 1;
    } else if (first as char).is_ascii_digit() {
        // Standalone digit run.
        while i < n && (bytes[i] as char).is_ascii_digit() {
            base.push(bytes[i] as char);
            canonical.push(bytes[i] as char);
            i += 1;
        }
    } else {
        // Other single char — not something we track for repetition.
        i += 1;
        return Some(LatexElement {
            start,
            end: i,
            canonical: (first as char).to_string(),
            base: (first as char).to_string(),
            sub_numeric: None,
            sup_numeric: None,
        });
    }

    // Parse optional sub/sup in any order.
    let mut sub_numeric: Option<i64> = None;
    let mut sup_numeric: Option<i64> = None;
    loop {
        if i >= n { break; }
        let b = bytes[i];
        if b != b'_' && b != b'^' { break; }
        let marker = b as char;
        i += 1;
        if i >= n { break; }
        let (arg_str, arg_end) = if bytes[i] == b'{' {
            // Braced sub/sup.
            let end_idx = match find_balanced_close(s, i) {
                Some(e) => e,
                None => break,
            };
            let inner = &s[i + 1..end_idx]; // contents without braces
            let collapsed = collapse_ws(inner);
            (collapsed, end_idx + 1)
        } else if bytes[i] == b'\\' && i + 1 < n && (bytes[i + 1] as char).is_ascii_alphabetic() {
            // `\command` as sub/sup arg.
            let cmd_start = i;
            i += 1;
            while i < n && (bytes[i] as char).is_ascii_alphabetic() {
                i += 1;
            }
            (s[cmd_start..i].to_string(), i)
        } else {
            // Single char sub/sup (e.g. `x_1`, `x_n`).
            let c = bytes[i];
            (std::str::from_utf8(&[c]).unwrap_or("?").to_string(), i + 1)
        };
        // Record numeric value if the sub/sup is pure digits.
        let numeric_value: Option<i64> = arg_str.parse::<i64>().ok();
        canonical.push(marker);
        canonical.push('{');
        canonical.push_str(&arg_str);
        canonical.push('}');
        match marker {
            '_' => sub_numeric = numeric_value,
            '^' => sup_numeric = numeric_value,
            _ => {}
        }
        i = arg_end;
    }

    Some(LatexElement {
        start,
        end: i,
        canonical,
        base,
        sub_numeric,
        sup_numeric,
    })
}

fn find_balanced_close(s: &str, open_idx: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let n = bytes.len();
    if open_idx >= n || bytes[open_idx] != b'{' {
        return None;
    }
    let mut depth: i32 = 1;
    let mut i = open_idx + 1;
    let mut max_depth: i32 = 1;
    while i < n {
        match bytes[i] {
            b'{' => {
                depth += 1;
                if depth > max_depth { max_depth = depth; }
                if max_depth > 32 {
                    // Pathological input — bail.
                    return None;
                }
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            b'\\' if i + 1 < n => {
                // Skip `\{` and `\}` as escaped braces.
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn collapse_ws(s: &str) -> String {
    // Inside math-mode LaTeX, whitespace is display-insignificant
    // (spaces are produced by `\,` `\;` `\ ` etc., not literal spaces).
    // For element-canonicalization we drop all whitespace so that
    // `\frac{ 1 }{ 2 }`, `\frac{1}{2}`, and `\frac{  1+2  }{3}` vs
    // `\frac{1+2}{3}` all collapse to comparable canonical forms.
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Detect an EXACT repeat of the same LaTeX element `threshold`+1 times
/// in a row (with element-separators between — whitespace, thin-space,
/// comma, plus, etc.). Returns byte offset to cut at.
///
/// Canonical form comparison: whitespace-collapsed. So `\frac{1}{2}` and
/// `\frac{ 1 }{ 2 }` are the same.
pub fn detect_repeated_element_cut(s: &str, threshold: usize) -> Option<usize> {
    if threshold == 0 {
        return Some(0);
    }
    let mut pos = 0;
    let mut last_canonical: Option<String> = None;
    let mut run_count: usize = 0;
    // We record the END offsets of the last `threshold` matching elements
    // so we can cut right after the `threshold`-th one.
    let mut recent_ends: Vec<usize> = Vec::with_capacity(threshold + 1);
    while let Some(elem) = next_latex_element(s, pos) {
        pos = elem.end;
        match last_canonical {
            Some(ref prev) if prev == &elem.canonical => {
                run_count += 1;
                recent_ends.push(elem.end);
                if run_count > threshold {
                    let cut = recent_ends[threshold - 1];
                    return Some(cut);
                }
            }
            _ => {
                last_canonical = Some(elem.canonical.clone());
                run_count = 1;
                recent_ends.clear();
                recent_ends.push(elem.end);
            }
        }
        if pos == elem.start {
            pos = elem.start + 1; // safety
        }
    }
    None
}

/// Detect a MONOTONIC-progression cut: same base with numeric sub- OR
/// super-script incrementing by exactly 1 between consecutive elements
/// (like `x_1, x_2, x_3, ...`). Threshold is the minimum progression
/// length to trigger.
///
/// Direction of monotonic increase: +1 per step. Strictly monotonic,
/// strict step 1. Matches the semantics of OCR's
/// `_detect_numeric_list_garbage_cut` transplanted to LaTeX subscripts.
pub fn detect_monotonic_element_cut(s: &str, threshold: usize) -> Option<usize> {
    if threshold <= 1 {
        return Some(0);
    }
    let mut pos = 0;
    let mut run_base: Option<String> = None;
    let mut run_count: usize = 0;
    let mut run_is_sub: bool = true;
    let mut next_expected: Option<i64> = None;
    let mut recent_ends: Vec<usize> = Vec::with_capacity(threshold + 1);
    while let Some(elem) = next_latex_element(s, pos) {
        pos = elem.end;
        let sub = elem.sub_numeric;
        let sup = elem.sup_numeric;
        // Only one of sub/sup should drive the progression; prefer sub
        // if both present. Need identical base across the run.
        let (value, is_sub) = match (sub, sup) {
            (Some(v), _) => (Some(v), true),
            (None, Some(v)) => (Some(v), false),
            _ => (None, true),
        };
        match (value, &run_base) {
            (Some(v), Some(prev_base)) if *prev_base == elem.base
                && is_sub == run_is_sub
                && next_expected == Some(v) =>
            {
                run_count += 1;
                recent_ends.push(elem.end);
                next_expected = Some(v + 1);
                if run_count >= threshold {
                    // Cut at the end of the threshold-th element.
                    let cut = recent_ends[threshold - 1];
                    return Some(cut);
                }
            }
            (Some(v), _) => {
                // Start a new progression (or reset).
                run_base = Some(elem.base.clone());
                run_is_sub = is_sub;
                run_count = 1;
                next_expected = Some(v + 1);
                recent_ends.clear();
                recent_ends.push(elem.end);
            }
            _ => {
                // No numeric sub/sup — break any progression.
                run_base = None;
                run_count = 0;
                next_expected = None;
                recent_ends.clear();
            }
        }
        if pos == elem.start {
            pos = elem.start + 1;
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
        // Element-level detector (wave-2.1, 2026-04-24): LaTeX-syntax-
        // aware repetition. Catches `\Omega \, \Omega \, \Omega` AND
        // compound elements like `\frac{1}{2} \frac{1}{2} \frac{1}{2}`.
        // Threshold 4 for exact repeat (tight — rare in real math).
        let cut_elem = detect_repeated_element_cut(inner, 4);
        // Monotonic progression detector — catches `x_1 x_2 x_3 …`
        // style looping. Threshold 8 (see false-positive analysis in
        // the test dataset: real math rarely enumerates 8+ without
        // `\ldots`). Analog of OCR's numeric-list garbage detector.
        let cut_mono = detect_monotonic_element_cut(inner, 8);
        let cut = [cut_char, cut_line, cut_elem, cut_mono]
            .into_iter()
            .flatten()
            .min();
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

    // -----------------------------------------------------------------
    // LaTeX-syntax-aware detector tests + false-positive dataset
    // (2026-04-24). Positive cases assert the detector fires.
    // Negative / legit cases assert it does NOT fire — these form the
    // false-positive guardrail and should grow whenever we discover a
    // legit pattern that was getting caught.
    // -----------------------------------------------------------------

    // --- element parser sanity ---

    #[test]
    fn element_parser_bare_command() {
        let e = next_latex_element("\\Omega rest", 0).expect("elem");
        assert_eq!(e.canonical, "\\Omega");
        assert_eq!(e.base, "\\Omega");
        assert!(e.sub_numeric.is_none() && e.sup_numeric.is_none());
    }

    #[test]
    fn element_parser_command_with_arg_groups() {
        let e = next_latex_element("\\frac{1}{2} rest", 0).expect("elem");
        assert_eq!(e.canonical, "\\frac{1}{2}");
    }

    #[test]
    fn element_parser_subscript_numeric() {
        let e = next_latex_element("x_5 rest", 0).expect("elem");
        assert_eq!(e.base, "x");
        assert_eq!(e.sub_numeric, Some(5));
    }

    #[test]
    fn element_parser_superscript_numeric_braced() {
        let e = next_latex_element("A^{12} rest", 0).expect("elem");
        assert_eq!(e.base, "A");
        assert_eq!(e.sup_numeric, Some(12));
    }

    #[test]
    fn element_parser_braced_group_as_element() {
        let e = next_latex_element("{\\alpha+\\beta} rest", 0).expect("elem");
        assert_eq!(e.canonical, "{\\alpha+\\beta}");
    }

    #[test]
    fn element_parser_skips_separators() {
        // Thin-space `\,` between command and arg — should not break
        // the element's identity.
        let e1 = next_latex_element("\\Omega \\, \\Omega", 0).expect("first");
        let e2 = next_latex_element("\\Omega \\, \\Omega", e1.end).expect("second");
        assert_eq!(e1.canonical, e2.canonical);
    }

    // --- detect_repeated_element_cut positive cases ---

    #[test]
    fn elem_repeat_catches_bare_command_run() {
        let s = "\\Omega \\Omega \\Omega \\Omega \\Omega \\Omega";
        let cut = detect_repeated_element_cut(s, 4).expect("cut");
        let head = &s[..cut];
        assert_eq!(head.matches("\\Omega").count(), 4);
    }

    #[test]
    fn elem_repeat_catches_thin_space_separator() {
        // Actual pattern from openarchives 997003_…_e2cbfdac.
        let s = "a = ".to_string() + &"\\Omega \\, ".repeat(20);
        let cut = detect_repeated_element_cut(&s, 4).expect("cut");
        assert_eq!(s[..cut].matches("\\Omega").count(), 4);
    }

    #[test]
    fn elem_repeat_catches_compound_frac() {
        let s = "a \\frac{1}{2} \\frac{1}{2} \\frac{1}{2} \\frac{1}{2} \\frac{1}{2} b";
        let cut = detect_repeated_element_cut(s, 4).expect("cut");
        assert_eq!(s[..cut].matches("\\frac{1}{2}").count(), 4);
    }

    #[test]
    fn elem_repeat_catches_mathbb() {
        let s = "\\mathbb{R} \\mathbb{R} \\mathbb{R} \\mathbb{R} \\mathbb{R} \\mathbb{R}";
        let cut = detect_repeated_element_cut(s, 4).expect("cut");
        assert_eq!(s[..cut].matches("\\mathbb{R}").count(), 4);
    }

    #[test]
    fn elem_repeat_catches_subscripted_atom() {
        let s = "a_n a_n a_n a_n a_n a_n";
        let cut = detect_repeated_element_cut(s, 4).expect("cut");
        assert_eq!(s[..cut].matches("a_{n}").count(), 0); // canonical
        // count "a_n" substrings in raw head (simpler)
        assert!(s[..cut].matches("a_n").count() >= 4);
    }

    #[test]
    fn elem_repeat_normalizes_whitespace_inside_args() {
        // `\frac{1}{2}` and `\frac{ 1 }{ 2 }` are considered same element.
        let s = "\\frac{1}{2} \\frac{ 1 }{ 2 } \\frac{1}{2} \\frac{  1  }{2} \\frac{1}{2}";
        let cut = detect_repeated_element_cut(s, 4).expect("cut");
        assert!(s[..cut].contains("\\frac"));
    }

    // --- detect_repeated_element_cut NEGATIVE cases (legit math) ---

    #[test]
    fn elem_repeat_does_not_fire_on_distinct_compounds() {
        // Normal math: lots of \frac but with different arguments.
        let s = "\\frac{1}{2} + \\frac{1}{3} + \\frac{1}{4} + \\frac{1}{5} + \\frac{1}{6}";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    #[test]
    fn elem_repeat_does_not_fire_on_derivatives() {
        // `f, f', f'', f'''` — different-canonical-form elements.
        let s = "f, f', f'', f''', f''''";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    #[test]
    fn elem_repeat_does_not_fire_on_greek_alphabet_run() {
        let s = "\\alpha + \\beta + \\gamma + \\delta + \\epsilon + \\zeta";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    #[test]
    fn elem_repeat_does_not_fire_on_polynomial() {
        let s = "x^2 + 2xy + y^2 + 3x^3 - 4y^4";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    #[test]
    fn elem_repeat_does_not_fire_on_common_math_identities() {
        let s = "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    #[test]
    fn elem_repeat_does_not_fire_at_exactly_threshold() {
        // Exactly 4 copies — should NOT trigger at threshold 4 (4 is
        // allowed, 5th is the cut trigger).
        let s = "\\Omega \\Omega \\Omega \\Omega";
        assert_eq!(detect_repeated_element_cut(s, 4), None);
    }

    // --- detect_monotonic_element_cut positive cases ---

    #[test]
    fn mono_catches_simple_x_n_progression() {
        // 10× `x_1, x_2, …, x_{10}` → trigger at threshold 8.
        let s = "x_1 x_2 x_3 x_4 x_5 x_6 x_7 x_8 x_9 x_{10}";
        let cut = detect_monotonic_element_cut(s, 8).expect("cut");
        assert!(cut > 0);
    }

    #[test]
    fn mono_catches_superscript_progression() {
        let s = "A^1 A^2 A^3 A^4 A^5 A^6 A^7 A^8 A^9";
        let cut = detect_monotonic_element_cut(s, 8).expect("cut");
        assert!(cut > 0);
    }

    #[test]
    fn mono_catches_progression_of_latex_command_base() {
        // `\phi_1, \phi_2, …, \phi_10`.
        let s = "\\phi_1 \\phi_2 \\phi_3 \\phi_4 \\phi_5 \\phi_6 \\phi_7 \\phi_8 \\phi_9";
        let cut = detect_monotonic_element_cut(s, 8).expect("cut");
        assert!(cut > 0);
    }

    // --- detect_monotonic_element_cut NEGATIVE cases (legit math) ---
    // These form the false-positive guardrail per user request.

    #[test]
    fn mono_does_not_fire_below_threshold() {
        // 7 terms — below threshold 8.
        let s = "x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_ldots_enumeration() {
        // Standard math: `x_1, x_2, ..., x_n` — `\ldots` breaks the run.
        let s = "x_1, x_2, x_3, \\ldots, x_n";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_non_monotonic_subscripts() {
        // Matrix indices: `a_{11}, a_{12}, a_{13}, a_{21}, a_{22}` —
        // not strictly +1 (jumps 13→21).
        let s = "a_{11} a_{12} a_{13} a_{21} a_{22} a_{23} a_{31} a_{32} a_{33}";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_increment_by_two() {
        // Odd indices: `x_1, x_3, x_5, x_7, x_9, …` — strict-+1 rule
        // excludes this.
        let s = "x_1 x_3 x_5 x_7 x_9 x_{11} x_{13} x_{15} x_{17}";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_different_bases() {
        // `a_1 b_2 c_3 d_4` — numeric progression BUT different bases.
        let s = "a_1 b_2 c_3 d_4 e_5 f_6 g_7 h_8 i_9";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_independent_summations() {
        // Two separate `\sum` with different subscript patterns.
        let s = "\\sum_{i=1}^{n} x_i + \\sum_{j=1}^{m} y_j";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    #[test]
    fn mono_does_not_fire_on_polynomial_with_mixed_exponents() {
        let s = "x^2 + 3x^4 - x^6 + 5x^8";
        assert_eq!(detect_monotonic_element_cut(s, 8), None);
    }

    // --- end-to-end crop tests ---

    #[test]
    fn crop_catches_omega_element_run_end_to_end() {
        // 50× `\Omega \, \Omega \,` inside `$$…$$` — the exact pattern
        // from openarchives 997003_…_e2cbfdac.
        let inner = "a = ".to_string() + &"\\Omega \\, ".repeat(50);
        let doc = format!("$$ {} b $$", inner);
        let out = crop_latex_repetitions(&doc, true, 100, 100);
        assert!(out.contains("repetition cropped"), "expected crop marker, got {:?}", out);
        let kept = out.matches("\\Omega").count();
        assert!(kept <= 5, "expected <=5 \\Omega in cropped output, got {}", kept);
    }

    #[test]
    fn crop_catches_monotonic_x_n_progression_end_to_end() {
        let inner: String = (1..=20).map(|i| format!("x_{} ", i)).collect();
        let doc = format!("$$ {} $$", inner);
        let out = crop_latex_repetitions(&doc, true, 100, 100);
        assert!(out.contains("repetition cropped"), "expected crop marker, got {:?}", out);
    }

    #[test]
    fn crop_preserves_clean_math_with_repeated_structure() {
        // Legitimate-looking math with `\sum` twice and integrals —
        // should NOT trigger.
        let doc = "$$\\sum_{i=1}^{n} f(x_i) = \\int_a^b f(x) dx + O(h^2)$$";
        let out = crop_latex_repetitions(doc, true, 100, 100);
        assert_eq!(out, doc, "clean math should pass through unchanged");
    }

    #[test]
    fn crop_preserves_derivative_sequence() {
        let doc = "$$f, f', f'', f''', f'''' \\text{ are derivatives}$$";
        let out = crop_latex_repetitions(doc, true, 100, 100);
        assert_eq!(out, doc);
    }

    #[test]
    fn crop_preserves_short_x_n_enumeration_with_ldots() {
        let doc = "$$x_1 + x_2 + x_3 + \\ldots + x_n$$";
        let out = crop_latex_repetitions(doc, true, 100, 100);
        assert_eq!(out, doc);
    }
}
