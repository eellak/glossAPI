//! Core noise metric computation logic extracted from standalone binary.
//! Provides library-friendly helpers for Python bindings.
//!
//! Changes and rationale (normalization and robustness):
//! - Removed early exits: previously, table-only/no-Greek returned 0.0 and non-table/no-Greek returned 100.0.
//!   We now always filter out table-like lines first and compute metrics on the remainder.
//!   If there is no Greek after filtering, all per-1000 rates are 0 ⇒ overall score 0. This avoids
//!   hard-coded outcomes and ensures uniform handling across inputs.
//! - Full per-1000 normalization: All contributors are normalized by 1000 Greek base codepoints (combining marks
//!   excluded) after table filtering. This brings components onto a comparable scale and makes scores more stable
//!   with document length.
//! - Short words normalization: replaced the previous ratio-based `short_pen` with
//!   `short_excess_per_1000 = max(0, short_words_per_1000 − BASELINE)`, where BASELINE≈26 was measured on
//!   clean texts. This yields a per-1000 signal and reduces sensitivity to layout.
//! - Sigma weight reduction: misplaced final sigma weight reduced from 5.0× to 2.5× to limit over-dominance
//!   from extraction artifacts (e.g., letter-spacing yielding σ at word end).
//! - Long words weighting: count only Greek words and weight long words by length with a smooth slope.
//!   Weight is `(len − 20)` with cap at 380, summed and normalized per 1000 Greek chars. This captures both
//!   prevalence and extremity of long tokens but avoids unbounded growth.
//! - Tables: table-like Markdown rows are filtered out before analysis, so tables are not scored.
/*
TODO (next iteration): Align core module with analysis changes – new features, removals, and efficiency

What to add/change (spec):
1) Add diacritics_btheta_per_1000 (combined metric)
   - Definition: per-1000 Greek chars count of (a) Unicode combining marks and (b) Greek special letters ϐ (U+03D0) and ϑ (U+03D1).
   - Purpose: detect extraction failure mode with excessive diacritics/variant glyphs (e.g., katharevoussa-like or glyph-substitution artifacts).
   - Exposure: include as a rate in the detailed API tuple (after existing rates), and optionally in flags when very high.

2) Refine slash-adjacent Greek detection (slash_adjacent_greek_rate)
   - Requirement: count a Greek word if it touches a slash-encoded Latin/digit segment with at least one char, e.g. Greek + "/pi1" or "/pi1" + Greek. Accept both prefix and suffix adjacency.
   - Normalization: per-1000 Greek chars.
   - Exposure: include rate in detailed API; keep as separate from score (diagnostic).

3) Keep prior scoring changes
   - longest_word term removed from score (already done here).
   - long words weighted by (len − 20) with cap at 380 (already implemented here).
   - sigma weight halved to 2.5 (already implemented here).
   - short words normalized as short_excess_per_1000 over baseline 26 (already implemented here).

How to implement efficiently (plan):
• Single-pass augmentations in analyse_bytes(buf: &[u8])
  - Maintain counters:
    - diacritics_btheta_count: increment on combining marks; increment on codepoints 0x03D0/0x03D1.
    - slash_adjacent_greek_words: detect adjacency without regex using byte-level checks and minimal lookback/lookahead.
  - State needed:
    - prev_non_greek_cp1/prev_non_greek_cp2 (u32) to detect Greek word start preceded by "/" then [A-Za-z0-9/].
    - At Greek word end, if current delimiter is "/" and next byte is [A-Za-z0-9/], also count suffix adjacency.
    - A boolean current_word_attached set at word start or end as above; add to counter when flushing the word.
  - For combining marks: on is_combining_mark(cp), do not reset the word; only increment diacritics_btheta_count and continue.
  - For special letters: on cp 0x03D0 or 0x03D1, increment diacritics_btheta_count.

• Rates in compute_score_and_details
  - Compute:
    - diacritics_btheta_per_1000 = 1000.0 * diacritics_btheta_count / len_greek
    - slash_adjacent_greek_rate = 1000.0 * slash_adjacent_greek_words / len_greek
  - Append both to the detailed tuple after existing rates (maintain stable order; document positions).
  - Do not include them in the score yet; they are diagnostics.

• Keep hyper-efficiency / turbo-speed
  - No regex; reuse existing fast table detection.
  - O(1) invalid bigram checks (already present here via fast path).
  - #[inline(always)] hot helpers; avoid allocations; reuse the current single-pass UTF-8 decode loop.
  - For directory scoring, prefer par_bridge + fs::read to avoid intermediate Vec<PathBuf>.

Positions in detailed tuple (suggested append):
  (..., v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_excess_per_1000,
   diacritics_btheta_per_1000, slash_adjacent_greek_rate, flags)

Note: after adding these fields, bump the Python bindings accordingly and propagate polytonic_ratio (already computed here) into downstream parquet (already wired in Corpus.clean()).
*/
use glossapi_rs_common::{is_combining_mark, is_greek, scan_script_metrics, ScriptScanner};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use unicode_normalization::UnicodeNormalization;
use walkdir::WalkDir;
// Avoid heavy regex for table detection; use lightweight checks instead

#[inline(always)]
fn is_vowel(cp: u32) -> bool {
    matches!(
        cp,
        0x0391 | 0x03B1 | 0x0386 | 0x03AC | // Αα Άά
        0x0395 | 0x03B5 | 0x0388 | 0x03AD | // Εε Έέ
        0x0397 | 0x03B7 | 0x0389 | 0x03AE | // Ηη Ήή
        0x0399 | 0x03B9 | 0x038A | 0x03AF | 0x03CA | 0x03CB | 0x039F | 0x03BF |
        0x038C | 0x03CC | 0x03C5 | 0x03B0 | 0x03CD | 0x03A5 | 0x038E |
        0x03A9 | 0x03C9 | 0x038F | 0x03CE
    )
}

const LONG_WORD_LIMIT: u64 = 21;
const SHORT_WORD_LIMIT: u64 = 3;
const PAGE_SPLIT_MARKER: &str = "<--- Page Split --->";
const NUMERIC_PAGE_COLLAPSE_MIN_TOKENS: u64 = 64;
const NUMERIC_PAGE_COLLAPSE_MIN_ATOMS: u64 = 64;
const NUMERIC_BLOCK_SEED_MIN_ATOMS: usize = 8;
// Baseline for short words per 1000 Greek characters (empirically ~26 on clean texts)
const SHORT_BASELINE_PER_1000: f64 = 26.0;

#[inline]
fn to_lower_fast(cp: u32) -> u32 {
    // Fast path for basic Greek capitals: add 0x20; otherwise return as-is
    if (0x0391..=0x03A9).contains(&cp) {
        cp + 0x20
    } else {
        cp
    }
}

#[inline]
fn is_invalid_bigram_pair(prev_low: u32, curr_low: u32) -> bool {
    match (prev_low, curr_low) {
        // κ/γ/χ + ξ
        (0x03BA, 0x03BE) | (0x03B3, 0x03BE) | (0x03C7, 0x03BE)
        // π/β/φ + ψ
        | (0x03C0, 0x03C8) | (0x03B2, 0x03C8) | (0x03C6, 0x03C8)
        // ρλ, μρ, γβ, δτ, τδ, βπ, πβ
        | (0x03C1, 0x03BB) | (0x03BC, 0x03C1) | (0x03B3, 0x03B2)
        | (0x03B4, 0x03C4) | (0x03C4, 0x03B4) | (0x03B2, 0x03C0) | (0x03C0, 0x03B2) => true,
        _ => false,
    }
}

static ALLOWED_DOUBLE: [u32; 9] = [
    0x03BB, 0x03BC, 0x03BD, 0x03C1, 0x03C3, 0x03C4, 0x03BA, 0x03C0, 0x03B3,
];

fn allowed_double(cp: u32) -> bool {
    ALLOWED_DOUBLE.contains(&cp)
}

#[inline]
fn is_table_line_trimmed(trimmed: &str) -> bool {
    // A simple check equivalent to /^\s*\|.*\|\s*$/ after trimming
    // i.e., line begins and ends with a '|' ignoring outer whitespace
    !trimmed.is_empty()
        && trimmed.as_bytes()[0] == b'|'
        && trimmed.as_bytes()[trimmed.len() - 1] == b'|'
}

fn table_line_ratio_and_filtered(text: &str) -> (f64, Option<String>, usize, usize) {
    let mut non_empty = 0usize;
    let mut table_like = 0usize;
    // First pass: count table-like rows without allocating filtered buffer unless needed
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            non_empty += 1;
            if is_table_line_trimmed(trimmed) {
                table_like += 1;
            }
        }
    }
    let ratio = if non_empty > 0 {
        table_like as f64 / non_empty as f64
    } else {
        0.0
    };
    if table_like == 0 {
        return (ratio, None, non_empty, table_like);
    }
    // Second pass only if we actually need a filtered buffer (preserve original newlines)
    let mut filtered = String::with_capacity(text.len());
    for seg in text.split_inclusive('\n') {
        let trimmed = seg.trim();
        if trimmed.is_empty() || !is_table_line_trimmed(trimmed) {
            filtered.push_str(seg);
        }
    }
    (ratio, Some(filtered), non_empty, table_like)
}

fn compute_latin_pct(buf: &[u8]) -> f64 {
    let latin_chars = buf
        .iter()
        .filter(|&&b| (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A))
        .count();
    latin_chars as f64 / (buf.len() as f64)
}

#[derive(Debug, Clone)]
pub struct OcrProfileRow {
    pub path: String,
    pub percentage_greek: f64,
    pub latin_percentage: f64,
    pub polytonic_ratio: f64,
    pub non_whitespace_chars: u64,
    pub greek_char_count: u64,
    pub latin_char_count: u64,
    pub ocr_repeat_phrase_run_max: u64,
    pub ocr_repeat_line_run_max: u64,
    pub ocr_repeat_suspicious_line_count: u64,
    pub ocr_repeat_suspicious_line_ratio: f64,
    pub ocr_noise_suspect: bool,
    pub ocr_noise_flags: String,
}

#[derive(Debug, Clone)]
pub struct OcrDebugPageRow {
    pub source_path: String,
    pub output_path: String,
    pub source_stem: String,
    pub base_stem: String,
    pub page_number: u64,
    pub page_index_in_file: u64,
    pub match_types: String,
    pub match_count: u64,
}

#[derive(Debug, Clone)]
struct OcrDebugPageCandidate {
    source_path: String,
    source_stem: String,
    base_stem: String,
    page_number: u64,
    page_index_in_file: u64,
}

#[derive(Debug, Clone)]
struct DebugMatchSpan {
    start: usize,
    end: usize,
    match_type: &'static str,
}

#[derive(Debug, Clone)]
pub struct NumericDebugSpan {
    pub start: usize,
    pub end: usize,
    pub match_type: String,
}

#[derive(Debug, Clone)]
pub struct WordRepeatSpan {
    pub start: usize,
    pub end: usize,
    pub period: usize,
    pub repetitions: usize,
    pub tail_chars: usize,
}

#[derive(Debug, Clone)]
pub struct HybridRepeatSpan {
    pub start: usize,
    pub end: usize,
    pub kind: &'static str,
    pub item_count: usize,
    pub cycle_len: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct LabeledSharedRepeatSpan {
    pub start: usize,
    pub end: usize,
    pub period: usize,
    pub repetitions: usize,
    pub tail_chars: usize,
    pub match_type: &'static str,
}

#[derive(Debug, Clone, Default)]
pub struct PageCharacterNoise {
    pub total_chars: u64,
    pub bad_char_count: u64,
    pub bad_char_ratio: f64,
    pub control_count: u64,
    pub private_use_count: u64,
    pub cjk_count: u64,
    pub replacement_count: u64,
}

const MERGE_SAME_CATEGORY_MAX_NONWHITESPACE_GAP: usize = 10;
const HYBRID_REPEAT_MIN_ITEMS: usize = 4;
const HYBRID_REPEAT_MIN_BODY_ALNUM: usize = 6;
const HYBRID_REPEAT_MAX_CYCLE: usize = 6;
const HYBRID_REPEAT_MIN_CYCLE_ITEMS: usize = 8;
const HYBRID_INLINE_CONTEXT_WORDS: usize = 2;
const HYBRID_INLINE_CONTEXT_MIN_ALPHA_WORDS: usize = 2;
const HYBRID_INLINE_CONTEXT_MIN_CHARS: usize = 8;
const HYBRID_INLINE_REPEAT_MIN_ITEMS: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridFieldKind {
    HeaderCounter,
    NumericValue,
}

#[derive(Debug, Clone)]
struct HybridNumberedItem {
    start: usize,
    end: usize,
    field_kind: HybridFieldKind,
    numbers: Vec<u32>,
    shape: String,
    body_key: String,
    body_is_full: bool,
}

#[derive(Debug, Clone)]
struct HybridInlineItem {
    start: usize,
    end: usize,
    clause_index: usize,
    inline_context_key: String,
    numeric_value: f64,
}

#[derive(Debug, Clone)]
struct HybridCandidate {
    prefix_start_byte: usize,
    prefix_end_byte: usize,
    field_kind: HybridFieldKind,
    numbers: Vec<u32>,
    shape: String,
}

#[derive(Debug, Clone)]
struct HybridToken {
    kind: HybridTokenKind,
    start: usize,
    end: usize,
    token_key: Option<String>,
    numeric_value: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridTokenKind {
    Numeric,
    Alpha,
}

#[derive(Debug, Clone, Copy)]
struct TokenSpan {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct NumericLineSummary {
    has_alpha: bool,
    rejected_non_numeric: bool,
    numeric_token_count: usize,
    numeric_atom_count: usize,
    is_blank: bool,
}

#[inline]
fn is_trim_numeric_edge_char(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '«' | '»' | '“' | '”' | '„' | '‟' | '‘' | '’' | '‚' | '‛'
        )
}

#[inline]
fn is_numeric_page_ignored_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|ch| !ch.is_whitespace() && !ch.is_alphanumeric())
}

fn trim_numeric_token_bounds(token: &str) -> Option<(usize, usize)> {
    if token.is_empty() {
        return None;
    }

    let mut start = 0usize;
    let mut end = token.len();

    while start < end {
        let ch = token[start..].chars().next()?;
        if ch.is_ascii_digit() {
            break;
        }
        if is_trim_numeric_edge_char(ch) {
            start += ch.len_utf8();
        } else {
            return None;
        }
    }

    while start < end {
        let ch = token[..end].chars().next_back()?;
        if ch.is_ascii_digit() {
            break;
        }
        if is_trim_numeric_edge_char(ch) {
            end -= ch.len_utf8();
        } else {
            return None;
        }
    }

    if start >= end {
        None
    } else {
        Some((start, end))
    }
}

#[inline]
fn is_numeric_page_token_body(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }

    if text.chars().all(|ch| ch.is_ascii_digit()) {
        return (1..=4).contains(&text.len());
    }

    let mut saw_digit = false;
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            saw_digit = true;
            continue;
        }
        if matches!(ch, '.' | ',' | ':' | ';' | '/' | '-') {
            continue;
        }
        return false;
    }

    saw_digit
}

fn summarize_numeric_line(line: &str) -> NumericLineSummary {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return NumericLineSummary {
            is_blank: true,
            ..NumericLineSummary::default()
        };
    }

    let tokens = extract_non_whitespace_tokens_with_spans(line);
    let mut summary = NumericLineSummary::default();
    for token in tokens {
        let raw = &line[token.start..token.end];
        if raw.chars().any(char::is_alphabetic) {
            summary.has_alpha = true;
        }
        if is_numeric_page_ignored_token(raw) {
            continue;
        }
        let Some((trim_start, trim_end)) = trim_numeric_token_bounds(raw) else {
            summary.rejected_non_numeric = true;
            continue;
        };
        let trimmed = &raw[trim_start..trim_end];
        if !is_numeric_page_token_body(trimmed) {
            summary.rejected_non_numeric = true;
            continue;
        }
        summary.numeric_token_count += 1;
        summary.numeric_atom_count += extract_digit_group_spans(trimmed).len();
    }
    summary
}

fn parse_simple_number(text: &str) -> Option<f64> {
    if text.is_empty() {
        return None;
    }

    let mut normalized = String::with_capacity(text.len());
    let mut saw_digit = false;
    let mut saw_separator = false;

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            normalized.push(ch);
            saw_digit = true;
        } else if ch == '.' || ch == ',' {
            if saw_separator {
                return None;
            }
            saw_separator = true;
            normalized.push('.');
        } else {
            return None;
        }
    }

    if !saw_digit || normalized.starts_with('.') || normalized.ends_with('.') {
        return None;
    }

    normalized.parse::<f64>().ok()
}

fn repeated_digit_token(text: &str) -> Option<char> {
    let mut digit: Option<char> = None;
    for ch in text.chars() {
        if !ch.is_ascii_digit() {
            return None;
        }
        match digit {
            Some(existing) if existing != ch => return None,
            Some(_) => {}
            None => digit = Some(ch),
        }
    }
    digit
}

#[inline]
fn is_private_use_codepoint(cp: u32) -> bool {
    matches!(
        cp,
        0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD
    )
}

#[inline]
fn is_cjk_codepoint(cp: u32) -> bool {
    matches!(
        cp,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0x2F800..=0x2FA1F
    )
}

pub fn evaluate_page_character_noise_internal(page: &str) -> PageCharacterNoise {
    let mut metrics = PageCharacterNoise::default();
    for ch in page.chars() {
        metrics.total_chars += 1;
        let cp = ch as u32;
        let mut is_bad = false;
        if ch == '\u{FFFD}' {
            metrics.replacement_count += 1;
            is_bad = true;
        } else if ch.is_control() && !matches!(ch, '\n' | '\r' | '\t') {
            metrics.control_count += 1;
            is_bad = true;
        } else if is_private_use_codepoint(cp) {
            metrics.private_use_count += 1;
            is_bad = true;
        } else if is_cjk_codepoint(cp) {
            metrics.cjk_count += 1;
            is_bad = true;
        }
        if is_bad {
            metrics.bad_char_count += 1;
        }
    }

    metrics.bad_char_ratio = if metrics.total_chars > 0 {
        metrics.bad_char_count as f64 / metrics.total_chars as f64
    } else {
        0.0
    };
    metrics
}

fn extract_digit_group_spans(text: &str) -> Vec<TokenSpan> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;

    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            if current_start.is_none() {
                current_start = Some(idx);
            }
        } else if let Some(start) = current_start.take() {
            spans.push(TokenSpan { start, end: idx });
        }
    }

    if let Some(start) = current_start {
        spans.push(TokenSpan {
            start,
            end: text.len(),
        });
    }

    spans
}

#[inline]
fn numeric_step_approx_eq(lhs: f64, rhs: f64) -> bool {
    let scale = lhs.abs().max(rhs.abs()).max(1.0);
    (lhs - rhs).abs() <= 1e-9 * scale
}

#[derive(Debug, Clone, Copy, Default)]
struct OcrRepeatNoiseMetrics {
    phrase_run_max: u64,
    line_run_max: u64,
    suspicious_line_count: u64,
    suspicious_line_ratio: f64,
    suspect: bool,
}

fn extract_non_whitespace_tokens_with_spans(line: &str) -> Vec<TokenSpan> {
    let mut tokens = Vec::new();
    let mut current_start: Option<usize> = None;

    for (idx, ch) in line.char_indices() {
        if !ch.is_whitespace() {
            if current_start.is_none() {
                current_start = Some(idx);
            }
        } else if let Some(start) = current_start.take() {
            tokens.push(TokenSpan { start, end: idx });
        }
    }

    if let Some(start) = current_start {
        tokens.push(TokenSpan {
            start,
            end: line.len(),
        });
    }

    tokens
}

fn normalize_line_for_repetition(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    let mut normalized = String::with_capacity(trimmed.len());
    let mut iter = trimmed.split_whitespace();
    if let Some(first) = iter.next() {
        normalized.push_str(first);
        for token in iter {
            normalized.push(' ');
            normalized.push_str(token);
        }
    }
    Some(normalized)
}

fn phrase_tokens_equal(
    line: &str,
    tokens: &[TokenSpan],
    lhs: usize,
    rhs: usize,
    len: usize,
) -> bool {
    (0..len).all(|offset| {
        let lhs_token = &line[tokens[lhs + offset].start..tokens[lhs + offset].end];
        let rhs_token = &line[tokens[rhs + offset].start..tokens[rhs + offset].end];
        lhs_token == rhs_token
    })
}

fn collect_repeat_phrase_debug_matches(
    line: &str,
    tokens: &[TokenSpan],
    min_repeat_run: u64,
) -> Vec<DebugMatchSpan> {
    let mut spans = Vec::new();
    let min_run = min_repeat_run as usize;
    if min_run < 2 || tokens.len() < min_run {
        return spans;
    }

    let max_phrase_len = 4usize.min(tokens.len() / min_run);
    for phrase_len in 1..=max_phrase_len {
        let mut i = 0usize;
        while i + phrase_len * min_run <= tokens.len() {
            let mut repeats = 1usize;
            while i + phrase_len * (repeats + 1) <= tokens.len()
                && phrase_tokens_equal(line, tokens, i, i + repeats * phrase_len, phrase_len)
            {
                repeats += 1;
            }
            if repeats >= min_run {
                spans.push(DebugMatchSpan {
                    start: tokens[i].start,
                    end: tokens[i + phrase_len * repeats - 1].end,
                    match_type: "repeat_phrase_run",
                });
                i += phrase_len * repeats;
            } else {
                i += 1;
            }
        }
    }

    spans
}

fn debug_match_merge_category(match_type: &'static str) -> Option<&'static str> {
    match match_type {
        "ascending_numeric_sequence"
        | "repeat_numeric_run"
        | "same_digit_numeric_run"
        | "numeric_page_collapse"
        | "numeric_block_collapse" => Some("numeric"),
        "word_repeat" => Some("word"),
        _ => None,
    }
}

fn gap_has_fewer_than_n_nonwhitespace_chars(
    text: &str,
    start: usize,
    end: usize,
    max_nonwhitespace: usize,
) -> bool {
    if start >= end {
        return true;
    }

    let mut count = 0usize;
    for ch in text[start..end].chars() {
        if !ch.is_whitespace() {
            count += 1;
            if count >= max_nonwhitespace {
                return false;
            }
        }
    }
    true
}

fn merge_debug_spans(
    text: &str,
    spans: Vec<DebugMatchSpan>,
) -> Vec<(usize, usize, Vec<&'static str>)> {
    if spans.is_empty() {
        return Vec::new();
    }

    let mut spans = spans;
    spans.sort_by_key(|span| (span.start, span.end));

    let mut merged: Vec<(usize, usize, Vec<&'static str>)> = Vec::new();
    for span in spans {
        if let Some((start, end, types)) = merged.last_mut() {
            let overlaps = span.start <= *end;
            let same_category_gap_merge = !overlaps
                && debug_match_merge_category(span.match_type).is_some()
                && types.iter().any(|kind| {
                    debug_match_merge_category(*kind) == debug_match_merge_category(span.match_type)
                })
                && gap_has_fewer_than_n_nonwhitespace_chars(
                    text,
                    *end,
                    span.start,
                    MERGE_SAME_CATEGORY_MAX_NONWHITESPACE_GAP,
                );
            if overlaps || same_category_gap_merge {
                *end = (*end).max(span.end);
                if !types.contains(&span.match_type) {
                    types.push(span.match_type);
                }
                *start = (*start).min(span.start);
                continue;
            }
        }
        merged.push((span.start, span.end, vec![span.match_type]));
    }

    for (_, _, types) in &mut merged {
        types.sort_unstable();
        types.dedup();
    }

    merged
}

fn annotate_text_with_debug_spans(
    text: &str,
    spans: Vec<DebugMatchSpan>,
) -> Option<(String, Vec<&'static str>, u64)> {
    let merged = merge_debug_spans(text, spans);
    if merged.is_empty() {
        return None;
    }

    let mut annotated = String::with_capacity(text.len() + merged.len() * 48);
    let mut pos = 0usize;
    let mut match_types: Vec<&'static str> = Vec::new();
    for (start, end, types) in &merged {
        if *start > pos {
            annotated.push_str(&text[pos..*start]);
        }
        let type_attr = types.join(",");
        annotated.push_str("<match of type ");
        annotated.push_str(&type_attr);
        annotated.push('>');
        annotated.push_str(&text[*start..*end]);
        annotated.push_str("</match>");
        pos = *end;
        for kind in types {
            if !match_types.contains(kind) {
                match_types.push(*kind);
            }
        }
    }
    if pos < text.len() {
        annotated.push_str(&text[pos..]);
    }

    Some((annotated, match_types, merged.len() as u64))
}

fn collect_numeric_page_collapse_span(page: &str, min_page_tokens: u64) -> Option<DebugMatchSpan> {
    let tokens = extract_non_whitespace_tokens_with_spans(page);
    let mut page_start: Option<usize> = None;
    let mut page_end: Option<usize> = None;
    let mut first_start: Option<usize> = None;
    let mut last_end: Option<usize> = None;
    let mut numeric_token_count = 0usize;
    let mut numeric_atom_count = 0usize;
    for token in tokens {
        let raw = &page[token.start..token.end];
        if page_start.is_none() {
            page_start = Some(token.start);
        }
        page_end = Some(token.end);
        if is_numeric_page_ignored_token(raw) {
            continue;
        }
        let (trim_start, trim_end) = trim_numeric_token_bounds(raw)?;
        let trimmed = &raw[trim_start..trim_end];
        if !is_numeric_page_token_body(trimmed) {
            return None;
        }
        let abs_start = token.start + trim_start;
        let abs_end = token.start + trim_end;
        if first_start.is_none() {
            first_start = Some(abs_start);
        }
        last_end = Some(abs_end);
        numeric_token_count += 1;
        numeric_atom_count += extract_digit_group_spans(trimmed).len();
    }

    if numeric_token_count < min_page_tokens as usize
        && numeric_atom_count < NUMERIC_PAGE_COLLAPSE_MIN_ATOMS as usize
    {
        return None;
    }

    Some(DebugMatchSpan {
        start: page_start.or(first_start)?,
        end: page_end.or(last_end)?,
        match_type: "numeric_page_collapse",
    })
}

fn collect_numeric_block_collapse_spans(page: &str) -> Vec<DebugMatchSpan> {
    let mut lines: Vec<(usize, usize, NumericLineSummary)> = Vec::new();
    let mut offset = 0usize;
    for segment in page.split_inclusive('\n') {
        let line = segment.strip_suffix('\n').unwrap_or(segment);
        let summary = summarize_numeric_line(line);
        lines.push((offset, offset + segment.len(), summary));
        offset += segment.len();
    }
    if offset < page.len() {
        let line = &page[offset..];
        lines.push((offset, page.len(), summarize_numeric_line(line)));
    }

    let mut spans = Vec::new();
    let mut idx = 0usize;
    while idx < lines.len() {
        let (_, _, summary) = lines[idx];
        let is_seed = !summary.has_alpha
            && !summary.rejected_non_numeric
            && summary.numeric_atom_count >= NUMERIC_BLOCK_SEED_MIN_ATOMS;
        if !is_seed {
            idx += 1;
            continue;
        }

        let mut start_idx = idx;
        let mut end_idx = idx;
        let mut total_atoms = summary.numeric_atom_count;

        while start_idx > 0 {
            let prev = lines[start_idx - 1].2;
            let prev_ok = prev.is_blank
                || (!prev.has_alpha && !prev.rejected_non_numeric && prev.numeric_token_count > 0);
            if !prev_ok {
                break;
            }
            start_idx -= 1;
            total_atoms += prev.numeric_atom_count;
        }

        while end_idx + 1 < lines.len() {
            let next = lines[end_idx + 1].2;
            let next_ok = next.is_blank
                || (!next.has_alpha && !next.rejected_non_numeric && next.numeric_token_count > 0);
            if !next_ok {
                break;
            }
            end_idx += 1;
            total_atoms += next.numeric_atom_count;
        }

        if total_atoms >= NUMERIC_PAGE_COLLAPSE_MIN_ATOMS as usize {
            let first_nonblank = (start_idx..=end_idx).find(|i| !lines[*i].2.is_blank);
            let last_nonblank = (start_idx..=end_idx).rfind(|i| !lines[*i].2.is_blank);
            if let (Some(first), Some(last)) = (first_nonblank, last_nonblank) {
                spans.push(DebugMatchSpan {
                    start: lines[first].0,
                    end: lines[last].1,
                    match_type: "numeric_block_collapse",
                });
            }
        }

        idx = end_idx + 1;
    }

    spans
}

fn collect_numeric_progression_matches(
    line: &str,
    tokens: &[TokenSpan],
    min_progress_steps: u64,
) -> Vec<DebugMatchSpan> {
    let min_steps = min_progress_steps as usize;
    if min_steps < 2 || tokens.len() < min_steps {
        return Vec::new();
    }

    let numeric_tokens: Vec<Option<(usize, usize, f64)>> = tokens
        .iter()
        .map(|token| {
            let raw = &line[token.start..token.end];
            let (trim_start, trim_end) = trim_numeric_token_bounds(raw)?;
            let trimmed = &raw[trim_start..trim_end];
            let value = parse_simple_number(trimmed)?;
            Some((token.start + trim_start, token.start + trim_end, value))
        })
        .collect();

    let mut spans = Vec::new();
    let mut i = 0usize;
    while i + min_steps <= numeric_tokens.len() {
        let Some((start, _, first)) = numeric_tokens[i] else {
            i += 1;
            continue;
        };
        let Some((_, _, second)) = numeric_tokens[i + 1] else {
            i += 1;
            continue;
        };

        let step = second - first;
        if !step.is_finite() || step <= 0.0 {
            i += 1;
            continue;
        }

        let mut j = i + 1;
        while j + 1 < numeric_tokens.len() {
            let Some((_, _, current)) = numeric_tokens[j] else {
                break;
            };
            let Some((_, _, next)) = numeric_tokens[j + 1] else {
                break;
            };
            if numeric_step_approx_eq(next - current, step) {
                j += 1;
            } else {
                break;
            }
        }

        let run_len = j - i + 1;
        if run_len >= min_steps {
            let (_, end, _) = numeric_tokens[j].expect("numeric run end");
            spans.push(DebugMatchSpan {
                start,
                end,
                match_type: "ascending_numeric_sequence",
            });
            i = j + 1;
        } else {
            i += 1;
        }
    }

    spans
}

fn collect_compact_repeat_numeric_matches(
    line: &str,
    tokens: &[TokenSpan],
    min_repeat_steps: u64,
) -> Vec<DebugMatchSpan> {
    let min_steps = min_repeat_steps as usize;
    if min_steps < 2 {
        return Vec::new();
    }

    let mut spans = Vec::new();
    for token in tokens {
        let raw = &line[token.start..token.end];
        let Some((trim_start, trim_end)) = trim_numeric_token_bounds(raw) else {
            continue;
        };
        let trimmed = &raw[trim_start..trim_end];
        let digit_groups = extract_digit_group_spans(trimmed);
        if digit_groups.len() < min_steps {
            continue;
        }

        let first_group = &trimmed[digit_groups[0].start..digit_groups[0].end];
        if digit_groups
            .iter()
            .any(|group| &trimmed[group.start..group.end] != first_group)
        {
            continue;
        }

        let mut separators_ok = true;
        for pair in digit_groups.windows(2) {
            let separator = &trimmed[pair[0].end..pair[1].start];
            if separator.is_empty()
                || separator
                    .chars()
                    .any(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
            {
                separators_ok = false;
                break;
            }
        }
        if !separators_ok {
            continue;
        }

        let trailing = &trimmed[digit_groups.last().expect("digit group").end..];
        if trailing
            .chars()
            .any(|ch| ch.is_ascii_alphanumeric() || ch.is_whitespace())
        {
            continue;
        }

        spans.push(DebugMatchSpan {
            start: token.start + trim_start,
            end: token.start + trim_end,
            match_type: "repeat_numeric_run",
        });
    }

    spans
}

fn collect_same_digit_numeric_matches(
    line: &str,
    tokens: &[TokenSpan],
    min_same_digit_steps: u64,
) -> Vec<DebugMatchSpan> {
    let min_steps = min_same_digit_steps as usize;
    if min_steps < 2 || tokens.len() < min_steps {
        return Vec::new();
    }

    let signatures: Vec<Option<(usize, usize, char)>> = tokens
        .iter()
        .map(|token| {
            let raw = &line[token.start..token.end];
            let (trim_start, trim_end) = trim_numeric_token_bounds(raw)?;
            let trimmed = &raw[trim_start..trim_end];
            let digit = repeated_digit_token(trimmed)?;
            Some((token.start + trim_start, token.start + trim_end, digit))
        })
        .collect();

    let mut spans = Vec::new();
    let mut i = 0usize;
    while i + min_steps <= signatures.len() {
        let Some((start, _, digit)) = signatures[i] else {
            i += 1;
            continue;
        };

        let mut j = i + 1;
        while j < signatures.len() && signatures[j].map(|(_, _, current)| current) == Some(digit) {
            j += 1;
        }

        let run_len = j - i;
        if run_len >= min_steps {
            let (_, end, _) = signatures[j - 1].expect("same-digit run end");
            spans.push(DebugMatchSpan {
                start,
                end,
                match_type: "same_digit_numeric_run",
            });
            i = j;
        } else {
            i += 1;
        }
    }

    spans
}

fn annotate_line_with_numeric_debug_matches(
    line: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> Option<(String, Vec<&'static str>, u64)> {
    let tokens = extract_non_whitespace_tokens_with_spans(line);
    if tokens.is_empty() {
        return None;
    }

    let mut spans = Vec::new();
    spans.extend(collect_numeric_progression_matches(
        line,
        &tokens,
        min_progress_steps,
    ));
    spans.extend(collect_compact_repeat_numeric_matches(
        line,
        &tokens,
        min_repeat_steps,
    ));
    spans.extend(collect_same_digit_numeric_matches(
        line,
        &tokens,
        min_same_digit_steps,
    ));
    annotate_text_with_debug_spans(line, spans)
}

fn annotate_line_with_debug_matches(
    line: &str,
    min_repeat_run: u64,
) -> Option<(String, Vec<&'static str>, u64)> {
    let tokens = extract_non_whitespace_tokens_with_spans(line);
    if tokens.is_empty() {
        return None;
    }

    let spans = collect_repeat_phrase_debug_matches(line, &tokens, min_repeat_run);
    let merged = merge_debug_spans(line, spans);
    if merged.is_empty() {
        return None;
    }

    let mut annotated = String::with_capacity(line.len() + merged.len() * 48);
    let mut pos = 0usize;
    let mut line_types: Vec<&'static str> = Vec::new();
    for (start, end, types) in &merged {
        if *start > pos {
            annotated.push_str(&line[pos..*start]);
        }
        let type_attr = types.join(",");
        annotated.push_str("<match of type ");
        annotated.push_str(&type_attr);
        annotated.push('>');
        annotated.push_str(&line[*start..*end]);
        annotated.push_str("</match>");
        pos = *end;
        for kind in types {
            if !line_types.contains(kind) {
                line_types.push(*kind);
            }
        }
    }
    if pos < line.len() {
        annotated.push_str(&line[pos..]);
    }

    Some((annotated, line_types, merged.len() as u64))
}

fn compute_repeat_phrase_run_max(trimmed: &str, min_repeat_run: u64) -> u64 {
    let tokens = extract_non_whitespace_tokens_with_spans(trimmed);
    let min_run = min_repeat_run as usize;
    if min_run < 2 || tokens.len() < min_run {
        return 0;
    }

    let max_phrase_len = 4usize.min(tokens.len() / min_run);
    let mut phrase_run_max = 0u64;
    for phrase_len in 1..=max_phrase_len {
        let mut i = 0usize;
        while i + phrase_len * min_run <= tokens.len() {
            let mut repeats = 1usize;
            while i + phrase_len * (repeats + 1) <= tokens.len()
                && phrase_tokens_equal(trimmed, &tokens, i, i + repeats * phrase_len, phrase_len)
            {
                repeats += 1;
            }
            if repeats >= min_run {
                phrase_run_max = phrase_run_max.max(repeats as u64);
                i += phrase_len * repeats;
            } else {
                i += 1;
            }
        }
    }

    phrase_run_max
}

fn collect_repeat_line_flags(lines: &[Option<String>], min_repeat_run: u64) -> (Vec<bool>, u64) {
    let min_run = min_repeat_run as usize;
    let mut flags = vec![false; lines.len()];
    if min_run < 2 || lines.len() < min_run {
        return (flags, 0);
    }

    let mut run_max = 0u64;
    let mut i = 0usize;
    while i < lines.len() {
        let Some(current) = lines[i].as_ref() else {
            i += 1;
            continue;
        };

        let mut j = i + 1;
        while j < lines.len() && lines[j].as_ref() == Some(current) {
            j += 1;
        }
        let run_len = j - i;
        if run_len >= min_run {
            run_max = run_max.max(run_len as u64);
            for flag in &mut flags[i..j] {
                *flag = true;
            }
        }
        i = j;
    }

    (flags, run_max)
}

fn finalize_ocr_repeat_noise(
    phrase_run_max: u64,
    line_run_max: u64,
    suspicious_line_count: u64,
    non_empty_lines: usize,
) -> OcrRepeatNoiseMetrics {
    let suspicious_line_ratio = if non_empty_lines > 0 {
        suspicious_line_count as f64 / non_empty_lines as f64
    } else {
        0.0
    };
    let suspect = suspicious_line_count > 0;

    OcrRepeatNoiseMetrics {
        phrase_run_max,
        line_run_max,
        suspicious_line_count,
        suspicious_line_ratio,
        suspect,
    }
}

fn compute_ocr_profile(
    text: &str,
    min_repeat_run: u64,
) -> (glossapi_rs_common::ScriptMetrics, OcrRepeatNoiseMetrics) {
    let mut scanner = ScriptScanner::new();
    let mut non_empty_lines = 0usize;
    let mut phrase_run_max = 0u64;
    let mut line_repeat_inputs: Vec<Option<String>> = Vec::new();
    let mut phrase_suspicious_lines: Vec<bool> = Vec::new();

    for segment in text.split_inclusive('\n') {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == PAGE_SPLIT_MARKER || is_table_line_trimmed(trimmed) {
            continue;
        }

        non_empty_lines += 1;
        scanner.observe_str(segment);
        let line_phrase_run_max = compute_repeat_phrase_run_max(trimmed, min_repeat_run);
        phrase_run_max = phrase_run_max.max(line_phrase_run_max);
        phrase_suspicious_lines.push(line_phrase_run_max >= min_repeat_run);
        line_repeat_inputs.push(normalize_line_for_repetition(trimmed));
    }

    let (repeat_line_flags, line_run_max) =
        collect_repeat_line_flags(&line_repeat_inputs, min_repeat_run);
    let suspicious_line_count = phrase_suspicious_lines
        .iter()
        .zip(repeat_line_flags.iter())
        .filter(|(phrase_flag, line_flag)| **phrase_flag || **line_flag)
        .count() as u64;

    (
        scanner.finish(),
        finalize_ocr_repeat_noise(
            phrase_run_max,
            line_run_max,
            suspicious_line_count,
            non_empty_lines,
        ),
    )
}

fn split_pages(text: &str) -> Vec<String> {
    let mut pages = Vec::new();
    let mut current = String::new();

    for segment in text.split_inclusive('\n') {
        if segment.trim() == PAGE_SPLIT_MARKER {
            pages.push(current);
            current = String::new();
            continue;
        }
        current.push_str(segment);
    }
    pages.push(current);
    pages
}

fn parse_source_stem(stem: &str) -> (String, u64) {
    if let Some((base, suffix)) = stem.rsplit_once("__p") {
        if let Some((start, _end)) = suffix.split_once('-') {
            if let Ok(start_page) = start.parse::<u64>() {
                return (base.to_string(), start_page);
            }
        }
    }
    (stem.to_string(), 1)
}

fn annotate_page_for_debug(
    page: &str,
    min_repeat_run: u64,
) -> Option<(String, Vec<&'static str>, u64)> {
    let mut segments: Vec<(&str, &str)> = Vec::new();
    let mut normalized_lines: Vec<Option<String>> = Vec::new();
    for segment in page.split_inclusive('\n') {
        let (line, newline) = if let Some(body) = segment.strip_suffix('\n') {
            (body, "\n")
        } else {
            (segment, "")
        };
        segments.push((line, newline));
        let trimmed = line.trim();
        if trimmed.is_empty() || is_table_line_trimmed(trimmed) {
            normalized_lines.push(None);
        } else {
            normalized_lines.push(normalize_line_for_repetition(trimmed));
        }
    }

    let (repeat_line_flags, _line_run_max) =
        collect_repeat_line_flags(&normalized_lines, min_repeat_run);

    let mut annotated = String::with_capacity(page.len());
    let mut page_types: Vec<&'static str> = Vec::new();
    let mut match_count = 0u64;

    for (idx, (line, newline)) in segments.iter().enumerate() {
        let line_debug = annotate_line_with_debug_matches(line, min_repeat_run);
        let line_repeat_flag = repeat_line_flags.get(idx).copied().unwrap_or(false);

        let mut line_content =
            if let Some((annotated_line, line_types, line_match_count)) = line_debug {
                match_count += line_match_count;
                for kind in line_types {
                    if !page_types.contains(&kind) {
                        page_types.push(kind);
                    }
                }
                annotated_line
            } else {
                (*line).to_string()
            };

        if line_repeat_flag {
            if !page_types.contains(&"repeat_line_run") {
                page_types.push("repeat_line_run");
            }
            match_count += 1;
            line_content = format!("<match of type repeat_line_run>{}</match>", line_content);
        }

        annotated.push_str(&line_content);
        annotated.push_str(newline);
    }

    if match_count == 0 {
        return None;
    }

    page_types.sort_unstable();
    page_types.dedup();
    Some((annotated, page_types, match_count))
}

pub fn annotate_numeric_debug_page_internal(
    page: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> Option<(String, Vec<String>, u64)> {
    let spans = collect_numeric_debug_spans_for_page(
        page,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
    );
    let (annotated_page, match_types, match_count) = annotate_text_with_debug_spans(page, spans)?;
    Some((
        annotated_page,
        match_types.into_iter().map(str::to_string).collect(),
        match_count,
    ))
}

pub fn find_numeric_debug_page_spans_internal(
    page: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> Vec<NumericDebugSpan> {
    collect_numeric_debug_spans_for_page(
        page,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
    )
    .into_iter()
    .map(|span| NumericDebugSpan {
        start: span.start,
        end: span.end,
        match_type: span.match_type.to_string(),
    })
    .collect()
}

const WORD_REPEAT_HASH_MASK: u64 = (1u64 << 63).wrapping_mul(2).wrapping_sub(1);
const WORD_REPEAT_HASH_BASE: u64 = 1469598103934665603u64;

#[inline]
fn hybrid_text_char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = Vec::with_capacity(text.chars().count() + 1);
    for (byte_idx, _) in text.char_indices() {
        boundaries.push(byte_idx);
    }
    boundaries.push(text.len());
    boundaries
}

fn hybrid_byte_to_char_idx(boundaries: &[usize], byte_idx: usize) -> usize {
    match boundaries.binary_search(&byte_idx) {
        Ok(idx) => idx,
        Err(idx) => idx,
    }
}

fn hybrid_normalize_body(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        for lower in ch.to_lowercase() {
            let lower = if lower == 'ς' { 'σ' } else { lower };
            for sub in lower.to_string().nfd() {
                if sub.is_alphanumeric() {
                    let mapped = match sub {
                        'ο' => 'o',
                        'κ' => 'k',
                        _ => sub,
                    };
                    out.push(mapped);
                }
            }
        }
    }
    out
}

fn hybrid_has_markup_body(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    if lower.contains("src=")
        || lower.contains("alt=")
        || lower.contains("image_")
        || lower.contains(".png")
        || lower.contains(".jpg")
        || lower.contains(".jpeg")
        || lower.contains(".gif")
    {
        return true;
    }

    let bytes = text.as_bytes();
    for (idx, byte) in bytes.iter().enumerate() {
        if *byte == b'<' && idx + 2 <= bytes.len() && bytes[idx + 1..].contains(&b'>') {
            return true;
        }
    }
    false
}

fn hybrid_classify_numeric_field(token: &str) -> Option<(HybridFieldKind, Vec<u32>, String)> {
    let token = token.trim();
    if token.is_empty() {
        return None;
    }

    let trailing_paren = token.ends_with(')');
    let trailing_dot = token.ends_with('.');
    let stripped = if trailing_paren || trailing_dot {
        &token[..token.len() - 1]
    } else {
        token
    };
    if stripped.is_empty() {
        return None;
    }

    if stripped.contains('/') {
        return Some((HybridFieldKind::NumericValue, Vec::new(), String::new()));
    }

    let parts: Vec<&str> = stripped.split('.').collect();
    if parts.is_empty() || parts.iter().any(|part| part.is_empty() || !part.chars().all(|ch| ch.is_ascii_digit())) {
        return None;
    }

    let mut numbers = Vec::with_capacity(parts.len());
    for part in &parts {
        numbers.push(part.parse::<u32>().ok()?);
    }

    let mut shape = std::iter::repeat("#")
        .take(numbers.len())
        .collect::<Vec<_>>()
        .join(".");
    if trailing_paren {
        shape.push(')');
    } else if trailing_dot {
        shape.push('.');
    }

    let field_kind = if trailing_paren || trailing_dot {
        HybridFieldKind::HeaderCounter
    } else if numbers.len() >= 3 {
        HybridFieldKind::HeaderCounter
    } else if numbers.len() == 2 && parts.last().map(|part| part.len()).unwrap_or(0) <= 2 {
        HybridFieldKind::HeaderCounter
    } else {
        HybridFieldKind::NumericValue
    };

    Some((field_kind, numbers, shape))
}

fn hybrid_classify_inline_numeric_field(token: &str) -> bool {
    let stripped = token.trim();
    if stripped.is_empty() {
        return false;
    }

    if stripped.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }

    if stripped.matches('/').count() == 1 {
        let mut parts = stripped.split('/');
        let lhs = parts.next().unwrap_or("");
        let rhs = parts.next().unwrap_or("");
        return !lhs.is_empty()
            && !rhs.is_empty()
            && lhs.chars().all(|ch| ch.is_ascii_digit())
            && rhs.chars().all(|ch| ch.is_ascii_digit())
            && rhs != "0";
    }

    let decimal_candidate = stripped.replacen(',', ".", 1);
    if decimal_candidate.matches('.').count() == 1 {
        let mut parts = decimal_candidate.split('.');
        let lhs = parts.next().unwrap_or("");
        let rhs = parts.next().unwrap_or("");
        return !lhs.is_empty()
            && !rhs.is_empty()
            && lhs.chars().all(|ch| ch.is_ascii_digit())
            && rhs.chars().all(|ch| ch.is_ascii_digit());
    }

    false
}

fn hybrid_parse_numeric_value(token: &str) -> Option<f64> {
    let stripped = token.trim();
    if stripped.is_empty() {
        return None;
    }

    if stripped.chars().all(|ch| ch.is_ascii_digit()) {
        return stripped.parse::<u64>().ok().map(|value| value as f64);
    }

    if stripped.matches('/').count() == 1 {
        let mut parts = stripped.split('/');
        let lhs = parts.next().unwrap_or("");
        let rhs = parts.next().unwrap_or("");
        if !lhs.is_empty()
            && !rhs.is_empty()
            && lhs.chars().all(|ch| ch.is_ascii_digit())
            && rhs.chars().all(|ch| ch.is_ascii_digit())
        {
            let lhs_value = lhs.parse::<f64>().ok()?;
            let rhs_value = rhs.parse::<f64>().ok()?;
            if rhs_value != 0.0 {
                return Some(lhs_value / rhs_value);
            }
        }
        return None;
    }

    let decimal_candidate = stripped.replacen(',', ".", 1);
    if decimal_candidate.matches('.').count() == 1 {
        let mut parts = decimal_candidate.split('.');
        let lhs = parts.next().unwrap_or("");
        let rhs = parts.next().unwrap_or("");
        if !lhs.is_empty()
            && !rhs.is_empty()
            && lhs.chars().all(|ch| ch.is_ascii_digit())
            && rhs.chars().all(|ch| ch.is_ascii_digit())
        {
            return decimal_candidate.parse::<f64>().ok();
        }
    }

    None
}

fn hybrid_next_char(text: &str, byte_idx: usize) -> Option<(char, usize)> {
    let ch = text[byte_idx..].chars().next()?;
    Some((ch, byte_idx + ch.len_utf8()))
}

fn hybrid_previous_char(text: &str, byte_idx: usize) -> Option<char> {
    text[..byte_idx].chars().next_back()
}

fn hybrid_parse_prefix_at(text: &str, start: usize) -> Option<usize> {
    if start >= text.len() {
        return None;
    }
    if let Some(prev) = hybrid_previous_char(text, start) {
        if prev.is_ascii_digit() {
            return None;
        }
    }

    let (first, mut idx) = hybrid_next_char(text, start)?;
    if !first.is_ascii_digit() {
        return None;
    }
    while idx < text.len() {
        let (ch, next_idx) = hybrid_next_char(text, idx)?;
        if !ch.is_ascii_digit() {
            break;
        }
        idx = next_idx;
    }

    if idx >= text.len() {
        return None;
    }
    let (delimiter, mut end_idx) = hybrid_next_char(text, idx)?;
    match delimiter {
        ')' => {}
        '.' => {
            loop {
                let mut cursor = end_idx;
                let mut saw_digit = false;
                while cursor < text.len() {
                    let (ch, next_cursor) = hybrid_next_char(text, cursor)?;
                    if !ch.is_ascii_digit() {
                        break;
                    }
                    saw_digit = true;
                    cursor = next_cursor;
                }
                if saw_digit {
                    if cursor < text.len() {
                        let (ch, next_cursor) = hybrid_next_char(text, cursor)?;
                        if ch == '.' {
                            end_idx = next_cursor;
                            continue;
                        }
                    }
                    end_idx = cursor;
                }
                break;
            }
        }
        _ => return None,
    }

    let mut lookahead = end_idx;
    while lookahead < text.len() {
        let (ch, next_idx) = hybrid_next_char(text, lookahead)?;
        if !ch.is_whitespace() {
            return ch.is_alphabetic().then_some(end_idx);
        }
        lookahead = next_idx;
    }
    None
}

fn hybrid_extract_numbered_items(analysis_text: &str) -> Vec<HybridNumberedItem> {
    let boundaries = hybrid_text_char_boundaries(analysis_text);
    let mut candidates: Vec<HybridCandidate> = Vec::new();
    let mut byte_idx = 0usize;
    while byte_idx < analysis_text.len() {
        let (ch, next_idx) = match hybrid_next_char(analysis_text, byte_idx) {
            Some(value) => value,
            None => break,
        };
        if ch.is_ascii_digit() {
            if let Some(prefix_end_byte) = hybrid_parse_prefix_at(analysis_text, byte_idx) {
                let prefix = &analysis_text[byte_idx..prefix_end_byte];
                if let Some((field_kind, numbers, shape)) = hybrid_classify_numeric_field(prefix) {
                    candidates.push(HybridCandidate {
                        prefix_start_byte: byte_idx,
                        prefix_end_byte,
                        field_kind,
                        numbers,
                        shape,
                    });
                }
                byte_idx = prefix_end_byte;
                continue;
            }
        }
        byte_idx = next_idx;
    }

    let mut items: Vec<HybridNumberedItem> = Vec::new();
    for (idx, candidate) in candidates.iter().enumerate() {
        let next_start = candidates
            .get(idx + 1)
            .map(|item| item.prefix_start_byte)
            .unwrap_or_else(|| analysis_text.len());
        let body_raw = analysis_text[candidate.prefix_end_byte..next_start].trim();
        if hybrid_has_markup_body(body_raw) {
            continue;
        }
        let body_key = hybrid_normalize_body(body_raw);
        let has_alpha = body_key.chars().any(|ch| ch.is_alphabetic());
        if !has_alpha {
            continue;
        }
        let body_is_full = body_key.chars().count() >= HYBRID_REPEAT_MIN_BODY_ALNUM;
        items.push(HybridNumberedItem {
            start: hybrid_byte_to_char_idx(&boundaries, candidate.prefix_start_byte),
            end: hybrid_byte_to_char_idx(&boundaries, next_start),
            field_kind: candidate.field_kind,
            numbers: candidate.numbers.clone(),
            shape: candidate.shape.clone(),
            body_key,
            body_is_full,
        });
    }

    items
}

fn hybrid_clause_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut clause_start = 0usize;
    let mut iter = text.char_indices().peekable();
    while let Some((idx, ch)) = iter.next() {
        let is_delimiter = match ch {
            ';' | '\n' => true,
            ',' => match iter.peek() {
                Some((_, next_ch)) => !next_ch.is_ascii_digit(),
                None => true,
            },
            _ => false,
        };
        if is_delimiter {
            ranges.push((clause_start, idx));
            clause_start = idx + ch.len_utf8();
        }
    }
    ranges.push((clause_start, text.len()));
    ranges
}

fn hybrid_extract_inline_items(analysis_text: &str) -> Vec<HybridInlineItem> {
    let boundaries = hybrid_text_char_boundaries(analysis_text);
    let clause_ranges = hybrid_clause_ranges(analysis_text);
    let mut items: Vec<HybridInlineItem> = Vec::new();

    for (clause_index, (raw_start, raw_end)) in clause_ranges.iter().enumerate() {
        let clause = &analysis_text[*raw_start..*raw_end];
        if clause.trim().is_empty() {
            continue;
        }

        let leading_ws = clause.len() - clause.trim_start().len();
        let trailing_ws = clause.len() - clause.trim_end().len();
        let clause_start_abs = raw_start + leading_ws;
        let clause_end_abs = raw_end - trailing_ws;
        if clause_start_abs >= clause_end_abs {
            continue;
        }

        let clause_text = &analysis_text[clause_start_abs..clause_end_abs];
        if clause_text.is_empty() || hybrid_has_markup_body(clause_text) {
            continue;
        }

        let mut working_offset = clause_start_abs;
        let mut working_text = clause_text;
        if let Some(prefix_end) = hybrid_parse_prefix_at(working_text, 0) {
            let trimmed = working_text[prefix_end..].trim_start();
            let trimmed_leading = working_text[prefix_end..].len() - trimmed.len();
            working_offset += prefix_end + trimmed_leading;
            working_text = trimmed;
        }
        if working_text.is_empty() {
            continue;
        }

        let mut tokens: Vec<HybridToken> = Vec::new();
        let mut numeric_positions: Vec<usize> = Vec::new();
        let mut token_byte = 0usize;
        while token_byte < working_text.len() {
            let (ch, next_idx) = match hybrid_next_char(working_text, token_byte) {
                Some(value) => value,
                None => break,
            };
            if ch.is_ascii_digit() {
                let mut end = next_idx;
                loop {
                    let mut cursor = end;
                    while cursor < working_text.len() {
                        let (digit_ch, digit_next) = match hybrid_next_char(working_text, cursor) {
                            Some(value) => value,
                            None => break,
                        };
                        if !digit_ch.is_ascii_digit() {
                            break;
                        }
                        cursor = digit_next;
                    }
                    end = cursor;
                    if end >= working_text.len() {
                        break;
                    }
                    let (sep, sep_next) = match hybrid_next_char(working_text, end) {
                        Some(value) => value,
                        None => break,
                    };
                    if !matches!(sep, '.' | ',' | '/') {
                        break;
                    }
                    if sep_next >= working_text.len() {
                        break;
                    }
                    let (after_sep, _) = match hybrid_next_char(working_text, sep_next) {
                        Some(value) => value,
                        None => break,
                    };
                    if !after_sep.is_ascii_digit() {
                        break;
                    }
                    end = sep_next;
                }
                let token = &working_text[token_byte..end];
                if hybrid_classify_inline_numeric_field(token) {
                    if let Some(parsed_value) = hybrid_parse_numeric_value(token) {
                        numeric_positions.push(tokens.len());
                        tokens.push(HybridToken {
                            kind: HybridTokenKind::Numeric,
                            start: hybrid_byte_to_char_idx(&boundaries, working_offset + token_byte),
                            end: hybrid_byte_to_char_idx(&boundaries, working_offset + end),
                            token_key: None,
                            numeric_value: Some(parsed_value),
                        });
                    }
                }
                token_byte = end;
                continue;
            }
            if ch.is_alphabetic() {
                let mut end = next_idx;
                while end < working_text.len() {
                    let (next_ch, next_end) = match hybrid_next_char(working_text, end) {
                        Some(value) => value,
                        None => break,
                    };
                    if !next_ch.is_alphabetic() {
                        break;
                    }
                    end = next_end;
                }
                let token = &working_text[token_byte..end];
                let token_key = hybrid_normalize_body(token);
                if !token_key.is_empty() {
                    tokens.push(HybridToken {
                        kind: HybridTokenKind::Alpha,
                        start: hybrid_byte_to_char_idx(&boundaries, working_offset + token_byte),
                        end: hybrid_byte_to_char_idx(&boundaries, working_offset + end),
                        token_key: Some(token_key),
                        numeric_value: None,
                    });
                }
                token_byte = end;
                continue;
            }
            token_byte = next_idx;
        }

        if numeric_positions.len() != 1 {
            continue;
        }
        let numeric_pos = numeric_positions[0];
        let numeric_token = &tokens[numeric_pos];
        let left_alpha: Vec<&HybridToken> = tokens[..numeric_pos]
            .iter()
            .filter(|token| token.kind == HybridTokenKind::Alpha)
            .collect();
        let right_alpha: Vec<&HybridToken> = tokens[numeric_pos + 1..]
            .iter()
            .filter(|token| token.kind == HybridTokenKind::Alpha)
            .collect();

        let left_start = left_alpha.len().saturating_sub(HYBRID_INLINE_CONTEXT_WORDS);
        let left_context = &left_alpha[left_start..];
        let right_limit = std::cmp::min(HYBRID_INLINE_CONTEXT_WORDS, right_alpha.len());
        let right_context = &right_alpha[..right_limit];
        let alpha_word_count = left_context.len() + right_context.len();
        if alpha_word_count < HYBRID_INLINE_CONTEXT_MIN_ALPHA_WORDS {
            continue;
        }

        let mut context_parts: Vec<String> =
            Vec::with_capacity(left_context.len() + 1 + right_context.len());
        for token in left_context {
            if let Some(token_key) = &token.token_key {
                context_parts.push(token_key.clone());
            }
        }
        context_parts.push("num".to_string());
        for token in right_context {
            if let Some(token_key) = &token.token_key {
                context_parts.push(token_key.clone());
            }
        }
        let context_key = hybrid_normalize_body(&context_parts.join(" "));
        if context_key.chars().count() < HYBRID_INLINE_CONTEXT_MIN_CHARS {
            continue;
        }

        let item_start = left_context
            .first()
            .map(|token| token.start)
            .unwrap_or(numeric_token.start);
        let item_end = right_context
            .last()
            .map(|token| token.end)
            .unwrap_or(numeric_token.end);
        items.push(HybridInlineItem {
            start: item_start,
            end: item_end,
            clause_index,
            inline_context_key: context_key,
            numeric_value: numeric_token.numeric_value.unwrap_or(0.0),
        });
    }

    items
}

fn hybrid_partial_body_matches(candidate_body_key: &str, target_body_key: &str) -> bool {
    if candidate_body_key.is_empty() || target_body_key.is_empty() || candidate_body_key == target_body_key {
        return false;
    }
    if !target_body_key.starts_with(candidate_body_key) {
        return false;
    }
    let target_len = target_body_key.chars().count();
    let candidate_len = candidate_body_key.chars().count();
    let min_chars = std::cmp::min(4usize, target_len);
    let min_ratio_chars = std::cmp::max(1usize, (target_len + 1) / 2);
    candidate_len >= std::cmp::min(min_chars, min_ratio_chars)
}

fn hybrid_header_progresses(previous: &HybridNumberedItem, current: &HybridNumberedItem) -> bool {
    previous.field_kind == HybridFieldKind::HeaderCounter
        && current.field_kind == HybridFieldKind::HeaderCounter
        && !previous.numbers.is_empty()
        && previous.numbers.len() == current.numbers.len()
        && previous.numbers[..previous.numbers.len() - 1] == current.numbers[..current.numbers.len() - 1]
        && current.numbers.last().copied()
            == previous
                .numbers
                .last()
                .copied()
                .and_then(|value| value.checked_add(1))
}

fn hybrid_header_is_parent(previous: &HybridNumberedItem, current: &HybridNumberedItem) -> bool {
    previous.field_kind == HybridFieldKind::HeaderCounter
        && current.field_kind == HybridFieldKind::HeaderCounter
        && !previous.numbers.is_empty()
        && previous.numbers.len() + 1 == current.numbers.len()
        && current.numbers[..current.numbers.len() - 1] == previous.numbers[..]
}

fn hybrid_extend_tail_span_end(
    items: &[HybridNumberedItem],
    run_start: usize,
    run_end: usize,
    expected_body_key: &str,
) -> usize {
    let span_end = items[run_end - 1].end;
    if run_end >= items.len() {
        return span_end;
    }
    let tail = &items[run_end];
    if tail.field_kind != HybridFieldKind::HeaderCounter
        || tail.shape != items[run_start].shape
        || !hybrid_header_progresses(&items[run_end - 1], tail)
        || !hybrid_partial_body_matches(&tail.body_key, expected_body_key)
    {
        return span_end;
    }
    tail.end
}

fn hybrid_inline_step(previous: &HybridInlineItem, current: &HybridInlineItem) -> Option<f64> {
    if current.clause_index != previous.clause_index + 1
        || current.inline_context_key != previous.inline_context_key
    {
        return None;
    }
    let step = current.numeric_value - previous.numeric_value;
    (step > 0.0).then_some(step)
}

fn hybrid_inline_step_matches(expected_step: f64, actual_step: f64) -> bool {
    let tolerance = f64::max(1e-9, expected_step.abs() * 1e-6);
    (expected_step - actual_step).abs() <= tolerance
}

fn hybrid_find_same_body_progression_spans(items: &[HybridNumberedItem]) -> Vec<HybridRepeatSpan> {
    let mut spans: Vec<HybridRepeatSpan> = Vec::new();
    let mut idx = 0usize;
    while idx < items.len() {
        let item = &items[idx];
        if item.field_kind != HybridFieldKind::HeaderCounter || !item.body_is_full {
            idx += 1;
            continue;
        }

        let mut end_idx = idx + 1;
        while end_idx < items.len()
            && items[end_idx].field_kind == HybridFieldKind::HeaderCounter
            && items[end_idx].body_is_full
            && items[end_idx].body_key == item.body_key
            && items[end_idx].shape == item.shape
            && hybrid_header_progresses(&items[end_idx - 1], &items[end_idx])
        {
            end_idx += 1;
        }

        let run_length = end_idx - idx;
        if run_length >= HYBRID_REPEAT_MIN_ITEMS {
            let mut start_idx = idx;
            if idx > 0 {
                let previous = &items[idx - 1];
                if previous.body_is_full
                    && previous.body_key == item.body_key
                    && hybrid_header_is_parent(previous, item)
                {
                    start_idx = idx - 1;
                }
            }
            let span_end = hybrid_extend_tail_span_end(items, idx, end_idx, &item.body_key);
            spans.push(HybridRepeatSpan {
                start: items[start_idx].start,
                end: span_end,
                kind: "same_body_progression",
                item_count: end_idx - start_idx,
                cycle_len: None,
            });
            idx = end_idx;
            continue;
        }

        idx += 1;
    }
    spans
}

fn hybrid_find_cycle_progression_spans(items: &[HybridNumberedItem]) -> Vec<HybridRepeatSpan> {
    let mut spans: Vec<HybridRepeatSpan> = Vec::new();
    let n_items = items.len();
    for cycle_len in 2..=HYBRID_REPEAT_MAX_CYCLE {
        let mut idx = 0usize;
        while idx + 2 * cycle_len <= n_items {
            let run = &items[idx..idx + 2 * cycle_len];
            if run
                .iter()
                .any(|item| item.field_kind != HybridFieldKind::HeaderCounter || !item.body_is_full)
            {
                idx += 1;
                continue;
            }
            let first_shape = &run[0].shape;
            if run.iter().any(|item| item.shape != *first_shape) {
                idx += 1;
                continue;
            }
            if !(1..run.len()).all(|pos| hybrid_header_progresses(&run[pos - 1], &run[pos])) {
                idx += 1;
                continue;
            }

            let template: Vec<&str> = run[..cycle_len]
                .iter()
                .map(|item| item.body_key.as_str())
                .collect();
            let unique_template_count = template
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<&str>>()
                .len();
            if unique_template_count < 2 {
                idx += 1;
                continue;
            }

            if (cycle_len..run.len()).any(|pos| run[pos].body_key != template[pos % cycle_len]) {
                idx += 1;
                continue;
            }

            let mut end_idx = idx + 2 * cycle_len;
            while end_idx < n_items
                && items[end_idx].field_kind == HybridFieldKind::HeaderCounter
                && items[end_idx].body_is_full
                && items[end_idx].shape == items[idx].shape
                && hybrid_header_progresses(&items[end_idx - 1], &items[end_idx])
                && items[end_idx].body_key == template[(end_idx - idx) % cycle_len]
            {
                end_idx += 1;
            }

            let item_count = end_idx - idx;
            if item_count >= HYBRID_REPEAT_MIN_CYCLE_ITEMS {
                let span_end = hybrid_extend_tail_span_end(
                    items,
                    idx,
                    end_idx,
                    template[(end_idx - idx) % cycle_len],
                );
                spans.push(HybridRepeatSpan {
                    start: items[idx].start,
                    end: span_end,
                    kind: "body_cycle_progression",
                    item_count,
                    cycle_len: Some(cycle_len),
                });
                idx = end_idx;
                continue;
            }
            idx += 1;
        }
    }
    spans
}

fn hybrid_find_inline_progression_spans(items: &[HybridInlineItem]) -> Vec<HybridRepeatSpan> {
    let mut spans: Vec<HybridRepeatSpan> = Vec::new();
    let mut idx = 0usize;
    while idx + HYBRID_INLINE_REPEAT_MIN_ITEMS <= items.len() {
        let first = &items[idx];
        let second = &items[idx + 1];
        let expected_step = match hybrid_inline_step(first, second) {
            Some(step) => step,
            None => {
                idx += 1;
                continue;
            }
        };

        let mut end_idx = idx + 2;
        while end_idx < items.len() {
            let actual_step = match hybrid_inline_step(&items[end_idx - 1], &items[end_idx]) {
                Some(step) => step,
                None => break,
            };
            if !hybrid_inline_step_matches(expected_step, actual_step) {
                break;
            }
            end_idx += 1;
        }

        let item_count = end_idx - idx;
        if item_count >= HYBRID_INLINE_REPEAT_MIN_ITEMS {
            spans.push(HybridRepeatSpan {
                start: items[idx].start,
                end: items[end_idx - 1].end,
                kind: "inline_numeric_progression",
                item_count,
                cycle_len: None,
            });
            idx = end_idx;
            continue;
        }
        idx += 1;
    }
    spans
}

pub fn find_hybrid_repeat_spans_internal(analysis_text: &str) -> Vec<HybridRepeatSpan> {
    let items = hybrid_extract_numbered_items(analysis_text);
    let mut spans = hybrid_find_same_body_progression_spans(&items);
    spans.extend(hybrid_find_cycle_progression_spans(&items));
    let inline_items = hybrid_extract_inline_items(analysis_text);
    spans.extend(hybrid_find_inline_progression_spans(&inline_items));
    spans.sort_by(|lhs, rhs| {
        lhs.start
            .cmp(&rhs.start)
            .then_with(|| (rhs.end - rhs.start).cmp(&(lhs.end - lhs.start)))
    });

    let mut deduped: Vec<HybridRepeatSpan> = Vec::new();
    for span in spans {
        if let Some(previous) = deduped.last() {
            if span.start >= previous.start && span.end <= previous.end {
                continue;
            }
        }
        deduped.push(span);
    }
    deduped
}

fn normalize_alnum_with_map_skip_tags_internal(text: &str) -> (String, Vec<usize>) {
    let mut normalized = String::with_capacity(text.len());
    let mut raw_char_indices: Vec<usize> = Vec::with_capacity(text.len());
    let mut in_tag = false;

    for (raw_idx, ch) in text.chars().enumerate() {
        if in_tag {
            if ch == '>' {
                in_tag = false;
            }
            continue;
        }
        if ch == '<' {
            in_tag = true;
            continue;
        }
        let mut casefolded = String::new();
        for lower in ch.to_lowercase() {
            match lower {
                'ς' => casefolded.push('σ'),
                'ß' => {
                    casefolded.push('s');
                    casefolded.push('s');
                }
                'ſ' => casefolded.push('s'),
                _ => casefolded.push(lower),
            }
        }
        for sub in casefolded.nfd() {
            if sub.is_alphanumeric() {
                let mapped = match sub {
                    'ο' => 'o',
                    'κ' => 'k',
                    _ => sub,
                };
                normalized.push(mapped);
                raw_char_indices.push(raw_idx);
            }
        }
    }

    (normalized, raw_char_indices)
}

pub fn find_labeled_shared_repeat_spans_internal(
    text: &str,
    rep_threshold: usize,
    min_period: usize,
    window: usize,
) -> Vec<LabeledSharedRepeatSpan> {
    let (normalized_text, raw_map) = normalize_alnum_with_map_skip_tags_internal(text);
    let normalized_chars: Vec<char> = normalized_text.chars().collect();
    let spans = find_word_repeat_spans_internal(&normalized_text, rep_threshold, min_period, window);
    let mut labeled: Vec<LabeledSharedRepeatSpan> = Vec::new();

    for span in spans {
        if span.end <= span.start || span.start >= raw_map.len() {
            continue;
        }
        let mut has_letter = false;
        let mut has_digit = false;
        for ch in &normalized_chars[span.start..span.end] {
            if ch.is_alphabetic() {
                has_letter = true;
            }
            if ch.is_ascii_digit() {
                has_digit = true;
            }
        }
        let match_type = if has_letter {
            "word_repeat"
        } else if has_digit {
            "numeric_repeat"
        } else {
            continue;
        };
        labeled.push(LabeledSharedRepeatSpan {
            start: raw_map[span.start],
            end: raw_map[span.end - 1] + 1,
            period: span.period,
            repetitions: span.repetitions,
            tail_chars: span.tail_chars,
            match_type,
        });
    }

    labeled
}

fn word_repeat_hash_slice(pref: &[u64], pw: &[u64], start: usize, end: usize) -> u64 {
    pref[end].wrapping_sub(pref[start].wrapping_mul(pw[end - start])) & WORD_REPEAT_HASH_MASK
}

#[inline]
fn word_repeat_blocks_equal(
    codes: &[u32],
    pref: &[u64],
    pw: &[u64],
    lhs: usize,
    rhs: usize,
    period: usize,
) -> bool {
    word_repeat_hash_slice(pref, pw, lhs, lhs + period)
        == word_repeat_hash_slice(pref, pw, rhs, rhs + period)
        && codes[lhs..lhs + period] == codes[rhs..rhs + period]
}

pub fn find_word_repeat_spans_internal(
    normalized_text: &str,
    rep_threshold: usize,
    min_period: usize,
    window: usize,
) -> Vec<WordRepeatSpan> {
    let codes: Vec<u32> = normalized_text.chars().map(|ch| ch as u32).collect();
    let n_chars = codes.len();
    if rep_threshold == 0 || min_period == 0 || n_chars < rep_threshold.saturating_mul(min_period) {
        return Vec::new();
    }

    let mut pref = vec![0u64; n_chars + 1];
    let mut pw = vec![1u64; n_chars + 1];
    for (idx, code) in codes.iter().enumerate() {
        pref[idx + 1] =
            (pref[idx].wrapping_mul(WORD_REPEAT_HASH_BASE).wrapping_add(*code as u64))
                & WORD_REPEAT_HASH_MASK;
        pw[idx + 1] = pw[idx].wrapping_mul(WORD_REPEAT_HASH_BASE) & WORD_REPEAT_HASH_MASK;
    }

    let max_period = std::cmp::min(
        std::cmp::max(min_period, window / rep_threshold),
        n_chars / rep_threshold,
    );
    let mut spans: Vec<WordRepeatSpan> = Vec::new();

    for period in min_period..=max_period {
        let mut idx = 0usize;
        while idx + rep_threshold * period <= n_chars {
            let mut is_repeat = true;
            for multiple in 1..rep_threshold {
                if !word_repeat_blocks_equal(&codes, &pref, &pw, idx, idx + multiple * period, period)
                {
                    is_repeat = false;
                    break;
                }
            }
            if !is_repeat {
                idx += 1;
                continue;
            }

            let mut left = idx;
            while left >= period
                && word_repeat_blocks_equal(&codes, &pref, &pw, left - period, left, period)
            {
                left -= period;
            }

            let mut right = idx + rep_threshold * period;
            while right + period <= n_chars
                && word_repeat_blocks_equal(&codes, &pref, &pw, right - period, right, period)
            {
                right += period;
            }

            let pattern = &codes[left..left + period];
            let mut tail_chars = 0usize;
            while right + tail_chars < n_chars
                && tail_chars < period
                && codes[right + tail_chars] == pattern[tail_chars]
            {
                tail_chars += 1;
            }

            spans.push(WordRepeatSpan {
                start: left,
                end: right + tail_chars,
                period,
                repetitions: (right - left) / period,
                tail_chars,
            });
            idx = right;
        }
    }

    spans.sort_by(|lhs, rhs| {
        lhs.start
            .cmp(&rhs.start)
            .then((rhs.end - rhs.start).cmp(&(lhs.end - lhs.start)))
            .then(lhs.period.cmp(&rhs.period))
    });

    let mut deduped: Vec<WordRepeatSpan> = Vec::new();
    for span in spans {
        if let Some(previous) = deduped.last() {
            if span.start >= previous.start && span.end <= previous.end {
                continue;
            }
        }
        deduped.push(span);
    }
    deduped
}

fn collect_numeric_debug_spans_for_page(
    page: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> Vec<DebugMatchSpan> {
    if let Some(page_span) =
        collect_numeric_page_collapse_span(page, NUMERIC_PAGE_COLLAPSE_MIN_TOKENS)
    {
        return vec![page_span];
    }

    let block_spans = collect_numeric_block_collapse_spans(page);
    if !block_spans.is_empty() {
        return block_spans;
    }

    let page_tokens = extract_non_whitespace_tokens_with_spans(page);
    let mut spans = collect_numeric_progression_matches(page, &page_tokens, min_progress_steps);
    let mut line_offset = 0usize;

    for segment in page.split_inclusive('\n') {
        let (line, newline) = if let Some(body) = segment.strip_suffix('\n') {
            (body, "\n")
        } else {
            (segment, "")
        };

        let line_tokens = extract_non_whitespace_tokens_with_spans(line);
        spans.extend(
            collect_compact_repeat_numeric_matches(line, &line_tokens, min_repeat_steps)
                .into_iter()
                .map(|span| DebugMatchSpan {
                    start: span.start + line_offset,
                    end: span.end + line_offset,
                    match_type: span.match_type,
                }),
        );
        spans.extend(
            collect_same_digit_numeric_matches(line, &line_tokens, min_same_digit_steps)
                .into_iter()
                .map(|span| DebugMatchSpan {
                    start: span.start + line_offset,
                    end: span.end + line_offset,
                    match_type: span.match_type,
                }),
        );
        line_offset += line.len() + newline.len();
    }

    spans
}

fn collect_ocr_debug_candidates_for_text(
    source_path: &Path,
    source_stem: &str,
    base_stem: &str,
    start_page: u64,
    text: &str,
    min_repeat_run: u64,
) -> Vec<OcrDebugPageCandidate> {
    let mut candidates = Vec::new();
    let pages = split_pages(text);

    for (idx, page) in pages.iter().enumerate() {
        let page_index_in_file = idx as u64 + 1;
        let page_number = start_page + idx as u64;
        if let Some((_annotated_page, _match_types, _match_count)) =
            annotate_page_for_debug(page, min_repeat_run)
        {
            candidates.push(OcrDebugPageCandidate {
                source_path: source_path.to_string_lossy().into_owned(),
                source_stem: source_stem.to_string(),
                base_stem: base_stem.to_string(),
                page_number,
                page_index_in_file,
            });
        }
    }

    candidates
}

fn collect_numeric_debug_candidates_for_text(
    source_path: &Path,
    source_stem: &str,
    base_stem: &str,
    start_page: u64,
    text: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> Vec<OcrDebugPageCandidate> {
    let mut candidates = Vec::new();
    let pages = split_pages(text);

    for (idx, page) in pages.iter().enumerate() {
        let page_index_in_file = idx as u64 + 1;
        let page_number = start_page + idx as u64;
        if !collect_numeric_debug_spans_for_page(
            page,
            min_progress_steps,
            min_repeat_steps,
            min_same_digit_steps,
        )
        .is_empty()
        {
            candidates.push(OcrDebugPageCandidate {
                source_path: source_path.to_string_lossy().into_owned(),
                source_stem: source_stem.to_string(),
                base_stem: base_stem.to_string(),
                page_number,
                page_index_in_file,
            });
        }
    }

    candidates
}

fn render_ocr_debug_candidate(
    candidate: &OcrDebugPageCandidate,
    output_dir: &Path,
    min_repeat_run: u64,
) -> anyhow::Result<OcrDebugPageRow> {
    let source_path = PathBuf::from(&candidate.source_path);
    let buf = fs::read(&source_path)?;
    let text = String::from_utf8_lossy(&buf);
    let pages = split_pages(&text);
    let page_idx = candidate
        .page_index_in_file
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("invalid page index"))? as usize;
    let page = pages
        .get(page_idx)
        .ok_or_else(|| anyhow::anyhow!("page index out of range for {}", candidate.source_path))?;
    let (annotated_page, match_types, match_count) = annotate_page_for_debug(page, min_repeat_run)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "candidate page no longer matches: {}",
                candidate.source_path
            )
        })?;
    let match_types_joined = match_types.join(",");
    let output_name = format!(
        "{}__debug_page_{:05}.md",
        candidate.source_stem, candidate.page_number
    );
    let output_path = output_dir.join(output_name);

    let mut content = String::new();
    content.push_str("<!-- source_path=");
    content.push_str(&candidate.source_path);
    content.push_str(" -->\n");
    content.push_str("<!-- base_stem=");
    content.push_str(&candidate.base_stem);
    content.push_str(" source_stem=");
    content.push_str(&candidate.source_stem);
    content.push_str(" page_number=");
    content.push_str(&candidate.page_number.to_string());
    content.push_str(" page_index_in_file=");
    content.push_str(&candidate.page_index_in_file.to_string());
    content.push_str(" match_types=");
    content.push_str(&match_types_joined);
    content.push_str(" match_count=");
    content.push_str(&match_count.to_string());
    content.push_str(" -->\n\n");
    content.push_str(&annotated_page);
    fs::write(&output_path, content)?;

    Ok(OcrDebugPageRow {
        source_path: candidate.source_path.clone(),
        output_path: output_path.to_string_lossy().into_owned(),
        source_stem: candidate.source_stem.clone(),
        base_stem: candidate.base_stem.clone(),
        page_number: candidate.page_number,
        page_index_in_file: candidate.page_index_in_file,
        match_types: match_types_joined,
        match_count,
    })
}

fn render_numeric_debug_candidate(
    candidate: &OcrDebugPageCandidate,
    output_dir: &Path,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> anyhow::Result<OcrDebugPageRow> {
    let source_path = PathBuf::from(&candidate.source_path);
    let buf = fs::read(&source_path)?;
    let text = String::from_utf8_lossy(&buf);
    let pages = split_pages(&text);
    let page_idx = candidate
        .page_index_in_file
        .checked_sub(1)
        .ok_or_else(|| anyhow::anyhow!("invalid page index"))? as usize;
    let page = pages
        .get(page_idx)
        .ok_or_else(|| anyhow::anyhow!("page index out of range for {}", candidate.source_path))?;
    let spans = collect_numeric_debug_spans_for_page(
        page,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
    );
    let (annotated_page, match_types, match_count) = annotate_text_with_debug_spans(page, spans)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "candidate page no longer matches numeric detector: {}",
                candidate.source_path
            )
        })?;
    let match_types_joined = match_types.join(",");
    let output_name = format!(
        "{}__debug_page_{:05}.md",
        candidate.source_stem, candidate.page_number
    );
    let output_path = output_dir.join(output_name);

    let mut content = String::new();
    content.push_str("<!-- source_path=");
    content.push_str(&candidate.source_path);
    content.push_str(" -->\n");
    content.push_str("<!-- base_stem=");
    content.push_str(&candidate.base_stem);
    content.push_str(" source_stem=");
    content.push_str(&candidate.source_stem);
    content.push_str(" page_number=");
    content.push_str(&candidate.page_number.to_string());
    content.push_str(" page_index_in_file=");
    content.push_str(&candidate.page_index_in_file.to_string());
    content.push_str(" match_types=");
    content.push_str(&match_types_joined);
    content.push_str(" match_count=");
    content.push_str(&match_count.to_string());
    content.push_str(" -->\n\n");
    content.push_str(&annotated_page);
    fs::write(&output_path, content)?;

    Ok(OcrDebugPageRow {
        source_path: candidate.source_path.clone(),
        output_path: output_path.to_string_lossy().into_owned(),
        source_stem: candidate.source_stem.clone(),
        base_stem: candidate.base_stem.clone(),
        page_number: candidate.page_number,
        page_index_in_file: candidate.page_index_in_file,
        match_types: match_types_joined,
        match_count,
    })
}

/// Compute metrics for UTF-8 bytes; ported from original CLI.
fn analyse_bytes(buf: &[u8]) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let mut len_greek = 0u64;
    let mut v_pen = 0u64;
    let mut c_pen = 0u64;
    let mut bad_double = 0u64;
    let mut max_run = 0u64;
    // Long word tracking:
    // - long_word_count: number of Greek words with length >= LONG_WORD_LIMIT
    // - long_word_weight_sum: sum of per-word weights with weight = min(len - 20, 380)
    let mut long_word_count = 0u64;
    let mut long_word_weight_sum = 0u64;
    let mut longest_word = 0u64;
    let mut misplaced_sigma = 0u64;
    let mut invalid_bigram = 0u64;
    let mut short_word_count = 0u64;
    let mut total_word_count = 0u64;

    let mut idx = 0;
    let mut prev_cp = 0u32;
    let mut run_len = 0u64;
    let mut run_is_vowel = false;
    let mut word_len = 0u64;

    while idx < buf.len() {
        let (cp, consumed) = decode_utf8(&buf[idx..]);
        idx += consumed;
        if is_combining_mark(cp) {
            continue;
        }
        if cp == 0 || !is_greek(cp) {
            if run_len > max_run {
                max_run = run_len;
            }
            run_len = 0;
            if word_len > 0 {
                total_word_count += 1;
                if word_len < SHORT_WORD_LIMIT {
                    short_word_count += 1;
                }
                if word_len >= LONG_WORD_LIMIT {
                    long_word_count += 1;
                    let extra = (word_len - LONG_WORD_LIMIT) as u64; // >= 0
                    let mut weight = 1 + extra; // equals (len - 20)
                    if weight > 380 {
                        weight = 380;
                    }
                    long_word_weight_sum += weight;
                }
                if word_len > longest_word {
                    longest_word = word_len;
                }
                if prev_cp == 0x03C3 {
                    misplaced_sigma += 1;
                }
            }
            word_len = 0;
            prev_cp = 0;
            continue;
        }
        len_greek += 1;
        word_len += 1;
        let vowel = is_vowel(cp);
        if run_len == 0 {
            run_is_vowel = vowel;
            run_len = 1;
        } else if vowel == run_is_vowel {
            run_len += 1;
        } else {
            if run_len >= 4 {
                let pen = run_len - 3;
                if run_is_vowel {
                    v_pen += pen;
                } else {
                    c_pen += pen;
                }
            }
            if run_len > max_run {
                max_run = run_len;
            }
            run_is_vowel = vowel;
            run_len = 1;
        }
        if prev_cp != 0 {
            let pc_low = to_lower_fast(prev_cp);
            let cc_low = to_lower_fast(cp);
            if is_invalid_bigram_pair(pc_low, cc_low) {
                invalid_bigram += 1;
            }
        }
        if prev_cp == cp && !allowed_double(cp) {
            bad_double += 1;
        }
        prev_cp = cp;
    }
    if run_len >= 4 {
        let pen = run_len - 3;
        if run_is_vowel {
            v_pen += pen;
        } else {
            c_pen += pen;
        }
    }
    if run_len > max_run {
        max_run = run_len;
    }
    if word_len > 0 {
        total_word_count += 1;
        if word_len < SHORT_WORD_LIMIT {
            short_word_count += 1;
        }
        if word_len >= LONG_WORD_LIMIT {
            long_word_count += 1;
            let extra = (word_len - LONG_WORD_LIMIT) as u64;
            let mut weight = 1 + extra; // equals (len - 20)
            if weight > 380 {
                weight = 380;
            }
            long_word_weight_sum += weight;
        }
        if word_len > longest_word {
            longest_word = word_len;
        }
        if prev_cp == 0x03C3 {
            misplaced_sigma += 1;
        }
    }

    (
        len_greek,
        v_pen,
        c_pen,
        bad_double,
        max_run,
        long_word_count,
        long_word_weight_sum,
        longest_word,
        misplaced_sigma,
        invalid_bigram,
        short_word_count,
        total_word_count,
    )
}

fn decode_utf8(slice: &[u8]) -> (u32, usize) {
    if slice.is_empty() {
        return (0, 0);
    }
    let c0 = slice[0];
    if c0 < 0x80 {
        return (c0 as u32, 1);
    } else if c0 & 0xE0 == 0xC0 && slice.len() >= 2 {
        let cp = ((c0 & 0x1F) as u32) << 6 | (slice[1] & 0x3F) as u32;
        return (cp, 2);
    } else if c0 & 0xF0 == 0xE0 && slice.len() >= 3 {
        let cp =
            ((c0 & 0x0F) as u32) << 12 | ((slice[1] & 0x3F) as u32) << 6 | (slice[2] & 0x3F) as u32;
        return (cp, 3);
    } else if c0 & 0xF8 == 0xF0 && slice.len() >= 4 {
        let cp = ((c0 & 0x07) as u32) << 18
            | ((slice[1] & 0x3F) as u32) << 12
            | ((slice[2] & 0x3F) as u32) << 6
            | (slice[3] & 0x3F) as u32;
        return (cp, 4);
    }
    (0, 1)
}

/// Core computation with details: returns a wide tuple with all components used by scoring
/// (score, latin_pct, table_line_ratio, polytonic_word_ratio,
///  len_greek, total_word_count,
///  v_pen, c_pen, bad_double, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
///  v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_pen,
///  flags)
fn compute_score_and_details(
    buf: &[u8],
) -> (
    f64,
    f64,
    f64,
    f64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    String,
) {
    let latin_pct = compute_latin_pct(buf);

    // Build text and filter out table-like lines
    let text = String::from_utf8_lossy(buf);
    let (table_ratio, filtered_opt, _non_empty, table_like) = table_line_ratio_and_filtered(&text);
    let had_tables = table_like > 0;
    let target: &[u8] = if let Some(ref s) = filtered_opt {
        s.as_bytes()
    } else {
        buf
    };

    let (
        len_greek,
        v_pen,
        c_pen,
        bad_dbl,
        max_run,
        long_word_count,
        long_word_weight_sum,
        longest_word,
        misplaced_sigma,
        invalid_bigram,
        short_word_count,
        total_word_count,
    ) = analyse_bytes(target);

    let mut flags: Vec<&str> = Vec::with_capacity(2);

    let len = len_greek as f64;
    let (v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate) = if len > 0.0 {
        (
            1000.0 * v_pen as f64 / len,
            1000.0 * c_pen as f64 / len,
            1000.0 * bad_dbl as f64 / len,
            1000.0 * misplaced_sigma as f64 / len,
            1000.0 * invalid_bigram as f64 / len,
            1000.0 * (long_word_weight_sum as f64) / len,
        )
    } else {
        // No Greek characters after filtering (e.g., table-only or non-Greek content)
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    };
    let short_ratio = if total_word_count > 0 {
        short_word_count as f64 / total_word_count as f64
    } else {
        0.0
    };
    // Normalized short words: per 1000 Greek chars, then excess over baseline
    let short_per_1000 = if len > 0.0 {
        1000.0 * (short_word_count as f64) / len
    } else {
        0.0
    };
    let short_excess_per_1000 = if short_per_1000 > SHORT_BASELINE_PER_1000 {
        short_per_1000 - SHORT_BASELINE_PER_1000
    } else {
        0.0
    };
    // Halved sigma coefficient from 5.0 to 2.5; removed longest_word term
    let score = v_rate
        + 1.5 * c_rate
        + 2.0 * d_rate
        + 2.5 * sigma_end_rate
        + 2.0 * bigram_rate
        + short_excess_per_1000
        + long_word_rate;

    let poly_ratio = if len_greek == 0 {
        0.0
    } else {
        let target_text: &str = if let Some(ref s) = filtered_opt {
            s.as_str()
        } else {
            text.as_ref()
        };
        scan_script_metrics(target_text).polytonic_ratio()
    };
    if poly_ratio > 0.0 {
        flags.push("polytonic");
    }
    if had_tables {
        flags.push("had_tables");
    }

    (
        score,
        latin_pct,
        table_ratio,
        poly_ratio,
        len_greek,
        total_word_count,
        v_pen,
        c_pen,
        bad_dbl,
        misplaced_sigma,
        invalid_bigram,
        long_word_count,
        longest_word,
        short_word_count,
        max_run,
        v_rate,
        c_rate,
        d_rate,
        sigma_end_rate,
        bigram_rate,
        long_word_rate,
        short_ratio,
        short_excess_per_1000,
        flags.join(","),
    )
}

/// Compute noise score and latin percentage for a UTF-8 buffer. Backward-compatible API.
fn compute_score(buf: &[u8]) -> (f64, f64) {
    let (
        score,
        latin_pct,
        _t,
        _p,
        _lg,
        _tw,
        _v,
        _c,
        _bd,
        _ms,
        _ib,
        _lwc,
        _lw,
        _swc,
        _mr,
        _vr,
        _cr,
        _dr,
        _sr,
        _br,
        _lwr,
        _sr2,
        _sp,
        _f,
    ) = compute_score_and_details(buf);
    (score, latin_pct)
}

fn run_in_thread_pool<T, F>(n_threads: Option<usize>, work: F) -> anyhow::Result<T>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    let threads = n_threads
        .filter(|count| *count > 0)
        .unwrap_or_else(rayon::current_num_threads);
    let pool = ThreadPoolBuilder::new().num_threads(threads).build()?;
    Ok(pool.install(work))
}

pub fn score_markdown_file_internal(path: &Path) -> anyhow::Result<f64> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let (score, _latin_pct) = compute_score(&buf);
    Ok(score)
}

pub fn score_markdown_directory_internal(
    root: &Path,
    n_threads: Option<usize>,
) -> anyhow::Result<Vec<(String, f64, f64)>> {
    run_in_thread_pool(n_threads, || {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| {
                let path = e.path();
                let buf = fs::read(path).expect("read");
                let (score, latin_pct) = compute_score(&buf);
                (path.to_string_lossy().into_owned(), score, latin_pct)
            })
            .collect()
    })
}

// Detailed variants for analysis layer
pub fn score_markdown_file_detailed_internal(
    path: &Path,
) -> anyhow::Result<(
    f64,
    f64,
    f64,
    f64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    u64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    String,
)> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(compute_score_and_details(&buf))
}

pub fn score_markdown_directory_detailed_internal(
    root: &Path,
    n_threads: Option<usize>,
) -> anyhow::Result<
    Vec<(
        String,
        f64,
        f64,
        f64,
        f64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        u64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        String,
    )>,
> {
    run_in_thread_pool(n_threads, || {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| {
                let path = e.path();
                let buf = fs::read(path).expect("read");
                let (
                    score,
                    latin_pct,
                    table_ratio,
                    poly_ratio,
                    len_greek,
                    total_words,
                    v_pen,
                    c_pen,
                    bad_dbl,
                    misplaced_sigma,
                    invalid_bigram,
                    long_word_count,
                    longest_word,
                    short_word_count,
                    max_run,
                    v_rate,
                    c_rate,
                    d_rate,
                    sigma_end_rate,
                    bigram_rate,
                    long_word_rate,
                    short_ratio,
                    short_pen,
                    flags,
                ) = compute_score_and_details(&buf);
                (
                    path.to_string_lossy().into_owned(),
                    score,
                    latin_pct,
                    table_ratio,
                    poly_ratio,
                    len_greek,
                    total_words,
                    v_pen,
                    c_pen,
                    bad_dbl,
                    misplaced_sigma,
                    invalid_bigram,
                    long_word_count,
                    longest_word,
                    short_word_count,
                    max_run,
                    v_rate,
                    c_rate,
                    d_rate,
                    sigma_end_rate,
                    bigram_rate,
                    long_word_rate,
                    short_ratio,
                    short_pen,
                    flags,
                )
            })
            .collect()
    })
}

pub fn score_markdown_directory_ocr_profile_internal(
    root: &Path,
    n_threads: Option<usize>,
    min_repeat_run: u64,
) -> anyhow::Result<Vec<OcrProfileRow>> {
    run_in_thread_pool(n_threads, || {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| {
                let path = e.path();
                let buf = fs::read(path).expect("read");
                let text = String::from_utf8_lossy(&buf);
                let (script, noise) = compute_ocr_profile(&text, min_repeat_run);
                let mut flags = Vec::new();
                if noise.phrase_run_max >= min_repeat_run {
                    flags.push("repeat_phrase_run");
                }
                if noise.line_run_max >= min_repeat_run {
                    flags.push("repeat_line_run");
                }

                OcrProfileRow {
                    path: path.to_string_lossy().into_owned(),
                    percentage_greek: script.percentage_greek(),
                    latin_percentage: script.latin_percentage(),
                    polytonic_ratio: script.polytonic_ratio(),
                    non_whitespace_chars: script.non_whitespace_chars,
                    greek_char_count: script.greek_char_count,
                    latin_char_count: script.latin_char_count,
                    ocr_repeat_phrase_run_max: noise.phrase_run_max,
                    ocr_repeat_line_run_max: noise.line_run_max,
                    ocr_repeat_suspicious_line_count: noise.suspicious_line_count,
                    ocr_repeat_suspicious_line_ratio: noise.suspicious_line_ratio,
                    ocr_noise_suspect: noise.suspect,
                    ocr_noise_flags: flags.join(","),
                }
            })
            .collect()
    })
}

pub fn export_ocr_match_debug_pages_internal(
    root: &Path,
    output_dir: &Path,
    n_threads: Option<usize>,
    min_repeat_run: u64,
    max_pages: Option<usize>,
    sample_seed: u64,
) -> anyhow::Result<Vec<OcrDebugPageRow>> {
    fs::create_dir_all(output_dir)?;
    if let Some(limit) = max_pages {
        let mut candidates: Vec<OcrDebugPageCandidate> = run_in_thread_pool(n_threads, || {
            WalkDir::new(root)
                .into_iter()
                .par_bridge()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
                .map(|e| {
                    let path = e.path();
                    let source_stem = path
                        .file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "unknown".to_string());
                    let (base_stem, start_page) = parse_source_stem(&source_stem);
                    let buf = fs::read(path).expect("read");
                    let text = String::from_utf8_lossy(&buf);
                    collect_ocr_debug_candidates_for_text(
                        path,
                        &source_stem,
                        &base_stem,
                        start_page,
                        &text,
                        min_repeat_run,
                    )
                })
                .reduce(Vec::new, |mut acc, mut item| {
                    acc.append(&mut item);
                    acc
                })
        })?;

        if candidates.len() > limit {
            let mut rng = StdRng::seed_from_u64(sample_seed);
            candidates.shuffle(&mut rng);
            candidates.truncate(limit);
        }
        candidates.sort_by(|a, b| {
            a.source_path
                .cmp(&b.source_path)
                .then(a.page_number.cmp(&b.page_number))
        });

        let output_dir = output_dir.to_path_buf();
        let mut rows: Vec<OcrDebugPageRow> = run_in_thread_pool(n_threads, move || {
            candidates
                .into_par_iter()
                .map(|candidate| {
                    render_ocr_debug_candidate(&candidate, &output_dir, min_repeat_run)
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })??;
        rows.sort_by(|a, b| {
            a.output_path
                .cmp(&b.output_path)
                .then(a.page_number.cmp(&b.page_number))
        });
        return Ok(rows);
    }

    let rows: Vec<Vec<OcrDebugPageRow>> = run_in_thread_pool(n_threads, || {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| {
                let path = e.path();
                let source_stem = path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".to_string());
                let (base_stem, start_page) = parse_source_stem(&source_stem);
                let buf = fs::read(path).expect("read");
                let text = String::from_utf8_lossy(&buf);
                let pages = split_pages(&text);
                let mut page_rows = Vec::new();

                for (idx, page) in pages.iter().enumerate() {
                    let page_index_in_file = idx as u64 + 1;
                    let page_number = start_page + idx as u64;
                    if let Some((annotated_page, match_types, match_count)) =
                        annotate_page_for_debug(page, min_repeat_run)
                    {
                        let match_types_joined = match_types.join(",");
                        let output_name =
                            format!("{}__debug_page_{:05}.md", source_stem, page_number);
                        let output_path = output_dir.join(output_name);
                        let mut content = String::new();
                        content.push_str("<!-- source_path=");
                        content.push_str(&path.to_string_lossy());
                        content.push_str(" -->\n");
                        content.push_str("<!-- base_stem=");
                        content.push_str(&base_stem);
                        content.push_str(" source_stem=");
                        content.push_str(&source_stem);
                        content.push_str(" page_number=");
                        content.push_str(&page_number.to_string());
                        content.push_str(" page_index_in_file=");
                        content.push_str(&page_index_in_file.to_string());
                        content.push_str(" match_types=");
                        content.push_str(&match_types_joined);
                        content.push_str(" match_count=");
                        content.push_str(&match_count.to_string());
                        content.push_str(" -->\n\n");
                        content.push_str(&annotated_page);
                        fs::write(&output_path, content).expect("write debug page");

                        page_rows.push(OcrDebugPageRow {
                            source_path: path.to_string_lossy().into_owned(),
                            output_path: output_path.to_string_lossy().into_owned(),
                            source_stem: source_stem.clone(),
                            base_stem: base_stem.clone(),
                            page_number,
                            page_index_in_file,
                            match_types: match_types_joined,
                            match_count,
                        });
                    }
                }

                page_rows
            })
            .collect()
    })?;

    let mut flat = Vec::new();
    for mut group in rows {
        flat.append(&mut group);
    }
    flat.sort_by(|a, b| {
        a.output_path
            .cmp(&b.output_path)
            .then(a.page_number.cmp(&b.page_number))
    });
    Ok(flat)
}

pub fn export_numeric_match_debug_pages_internal(
    root: &Path,
    output_dir: &Path,
    n_threads: Option<usize>,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
    max_pages: Option<usize>,
    sample_seed: u64,
) -> anyhow::Result<Vec<OcrDebugPageRow>> {
    fs::create_dir_all(output_dir)?;
    if let Some(limit) = max_pages {
        let mut candidates: Vec<OcrDebugPageCandidate> = run_in_thread_pool(n_threads, || {
            WalkDir::new(root)
                .into_iter()
                .par_bridge()
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
                .map(|e| {
                    let path = e.path();
                    let source_stem = path
                        .file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "unknown".to_string());
                    let (base_stem, start_page) = parse_source_stem(&source_stem);
                    let buf = fs::read(path).expect("read");
                    let text = String::from_utf8_lossy(&buf);
                    collect_numeric_debug_candidates_for_text(
                        path,
                        &source_stem,
                        &base_stem,
                        start_page,
                        &text,
                        min_progress_steps,
                        min_repeat_steps,
                        min_same_digit_steps,
                    )
                })
                .reduce(Vec::new, |mut acc, mut item| {
                    acc.append(&mut item);
                    acc
                })
        })?;

        if candidates.len() > limit {
            let mut rng = StdRng::seed_from_u64(sample_seed);
            candidates.shuffle(&mut rng);
            candidates.truncate(limit);
        }
        candidates.sort_by(|a, b| {
            a.source_path
                .cmp(&b.source_path)
                .then(a.page_number.cmp(&b.page_number))
        });

        let output_dir = output_dir.to_path_buf();
        let mut rows: Vec<OcrDebugPageRow> = run_in_thread_pool(n_threads, move || {
            candidates
                .into_par_iter()
                .map(|candidate| {
                    render_numeric_debug_candidate(
                        &candidate,
                        &output_dir,
                        min_progress_steps,
                        min_repeat_steps,
                        min_same_digit_steps,
                    )
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })??;
        rows.sort_by(|a, b| {
            a.output_path
                .cmp(&b.output_path)
                .then(a.page_number.cmp(&b.page_number))
        });
        return Ok(rows);
    }

    let rows: Vec<Vec<OcrDebugPageRow>> = run_in_thread_pool(n_threads, || {
        WalkDir::new(root)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| {
                let path = e.path();
                let source_stem = path
                    .file_stem()
                    .map(|stem| stem.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "unknown".to_string());
                let (base_stem, start_page) = parse_source_stem(&source_stem);
                let buf = fs::read(path).expect("read");
                let text = String::from_utf8_lossy(&buf);
                let pages = split_pages(&text);
                let mut page_rows = Vec::new();

                for (idx, page) in pages.iter().enumerate() {
                    let page_index_in_file = idx as u64 + 1;
                    let page_number = start_page + idx as u64;
                    let spans = collect_numeric_debug_spans_for_page(
                        page,
                        min_progress_steps,
                        min_repeat_steps,
                        min_same_digit_steps,
                    );
                    if let Some((annotated_page, match_types, match_count)) =
                        annotate_text_with_debug_spans(page, spans)
                    {
                        let match_types_joined = match_types.join(",");
                        let output_name =
                            format!("{}__debug_page_{:05}.md", source_stem, page_number);
                        let output_path = output_dir.join(output_name);
                        let mut content = String::new();
                        content.push_str("<!-- source_path=");
                        content.push_str(&path.to_string_lossy());
                        content.push_str(" -->\n");
                        content.push_str("<!-- base_stem=");
                        content.push_str(&base_stem);
                        content.push_str(" source_stem=");
                        content.push_str(&source_stem);
                        content.push_str(" page_number=");
                        content.push_str(&page_number.to_string());
                        content.push_str(" page_index_in_file=");
                        content.push_str(&page_index_in_file.to_string());
                        content.push_str(" match_types=");
                        content.push_str(&match_types_joined);
                        content.push_str(" match_count=");
                        content.push_str(&match_count.to_string());
                        content.push_str(" -->\n\n");
                        content.push_str(&annotated_page);
                        fs::write(&output_path, content).expect("write numeric debug page");

                        page_rows.push(OcrDebugPageRow {
                            source_path: path.to_string_lossy().into_owned(),
                            output_path: output_path.to_string_lossy().into_owned(),
                            source_stem: source_stem.clone(),
                            base_stem: base_stem.clone(),
                            page_number,
                            page_index_in_file,
                            match_types: match_types_joined,
                            match_count,
                        });
                    }
                }

                page_rows
            })
            .collect()
    })?;

    let mut flat = Vec::new();
    for mut group in rows {
        flat.append(&mut group);
    }
    flat.sort_by(|a, b| {
        a.output_path
            .cmp(&b.output_path)
            .then(a.page_number.cmp(&b.page_number))
    });
    Ok(flat)
}
