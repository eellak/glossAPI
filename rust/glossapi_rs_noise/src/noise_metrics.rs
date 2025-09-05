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

use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
// Avoid heavy regex for table detection; use lightweight checks instead

const GREEK_BLOCK_1: std::ops::RangeInclusive<u32> = 0x0370..=0x03FF; // Greek & Coptic
const GREEK_BLOCK_2: std::ops::RangeInclusive<u32> = 0x1F00..=0x1FFF; // Greek Extended

#[inline(always)]
fn is_greek(cp: u32) -> bool {
    GREEK_BLOCK_1.contains(&cp) || GREEK_BLOCK_2.contains(&cp)
}

#[inline(always)]
fn is_combining_mark(cp: u32) -> bool {
    (0x0300..=0x036F).contains(&cp) || (0x1DC0..=0x1DFF).contains(&cp) || (0x20D0..=0x20FF).contains(&cp)
}

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
// Baseline for short words per 1000 Greek characters (empirically ~26 on clean texts)
const SHORT_BASELINE_PER_1000: f64 = 26.0;

#[inline]
fn to_lower_fast(cp: u32) -> u32 {
    // Fast path for basic Greek capitals: add 0x20; otherwise return as-is
    if (0x0391..=0x03A9).contains(&cp) { cp + 0x20 } else { cp }
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
    !trimmed.is_empty() && trimmed.as_bytes()[0] == b'|' && trimmed.as_bytes()[trimmed.len()-1] == b'|'
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
    let ratio = if non_empty > 0 { table_like as f64 / non_empty as f64 } else { 0.0 };
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

fn compute_polytonic_word_ratio(text: &str) -> (u64, u64, f64) {
    let mut greek_words = 0u64;
    let mut polytonic_words = 0u64;
    for w in text.split_whitespace() {
        let mut has_greek = false;
        let mut has_poly = false;
        for ch in w.chars() {
            let cp = ch as u32;
            if is_greek(cp) { has_greek = true; }
            if (0x1F00..=0x1FFF).contains(&cp) || is_combining_mark(cp) { has_poly = true; }
        }
        if has_greek {
            greek_words += 1;
            if has_poly { polytonic_words += 1; }
        }
    }
    let ratio = if greek_words > 0 { polytonic_words as f64 / greek_words as f64 } else { 0.0 };
    (polytonic_words, greek_words, ratio)
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
                    if weight > 380 { weight = 380; }
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
                if run_is_vowel { v_pen += pen; } else { c_pen += pen; }
            }
            if run_len > max_run { max_run = run_len; }
            run_is_vowel = vowel;
            run_len = 1;
        }
        if prev_cp != 0 {
            let pc_low = to_lower_fast(prev_cp);
            let cc_low = to_lower_fast(cp);
            if is_invalid_bigram_pair(pc_low, cc_low) { invalid_bigram += 1; }
        }
        if prev_cp == cp && !allowed_double(cp) { bad_double += 1; }
        prev_cp = cp;
    }
    if run_len >= 4 {
        let pen = run_len - 3;
        if run_is_vowel { v_pen += pen; } else { c_pen += pen; }
    }
    if run_len > max_run { max_run = run_len; }
    if word_len > 0 {
        total_word_count += 1;
        if word_len < SHORT_WORD_LIMIT { short_word_count += 1; }
        if word_len >= LONG_WORD_LIMIT {
            long_word_count += 1;
            let extra = (word_len - LONG_WORD_LIMIT) as u64;
            let mut weight = 1 + extra; // equals (len - 20)
            if weight > 380 { weight = 380; }
            long_word_weight_sum += weight;
        }
        if word_len > longest_word { longest_word = word_len; }
        if prev_cp == 0x03C3 { misplaced_sigma += 1; }
    }

    (len_greek, v_pen, c_pen, bad_double, max_run, long_word_count, long_word_weight_sum, longest_word, misplaced_sigma, invalid_bigram, short_word_count, total_word_count)
}

fn decode_utf8(slice: &[u8]) -> (u32, usize) {
    if slice.is_empty() { return (0, 0); }
    let c0 = slice[0];
    if c0 < 0x80 {
        return (c0 as u32, 1);
    } else if c0 & 0xE0 == 0xC0 && slice.len() >= 2 {
        let cp = ((c0 & 0x1F) as u32) << 6 | (slice[1] & 0x3F) as u32;
        return (cp, 2);
    } else if c0 & 0xF0 == 0xE0 && slice.len() >= 3 {
        let cp = ((c0 & 0x0F) as u32) << 12 | ((slice[1] & 0x3F) as u32) << 6 | (slice[2] & 0x3F) as u32;
        return (cp, 3);
    } else if c0 & 0xF8 == 0xF0 && slice.len() >= 4 {
        let cp = ((c0 & 0x07) as u32) << 18 | ((slice[1] & 0x3F) as u32) << 12 | ((slice[2] & 0x3F) as u32) << 6 | (slice[3] & 0x3F) as u32;
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
    buf: &[u8]
) -> (
    f64, f64, f64, f64,
    u64, u64,
    u64, u64, u64, u64, u64, u64, u64, u64, u64,
    f64, f64, f64, f64, f64, f64, f64, f64,
    String
) {
    let latin_pct = compute_latin_pct(buf);

    // Build text and filter out table-like lines
    let text = String::from_utf8_lossy(buf);
    let (table_ratio, filtered_opt, _non_empty, table_like) = table_line_ratio_and_filtered(&text);
    let had_tables = table_like > 0;
    let target: &[u8] = if let Some(ref s) = filtered_opt { s.as_bytes() } else { buf };

    let (len_greek, v_pen, c_pen, bad_dbl, max_run, long_word_count, long_word_weight_sum, longest_word, misplaced_sigma, invalid_bigram, short_word_count, total_word_count) = analyse_bytes(target);

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
    let short_per_1000 = if len > 0.0 { 1000.0 * (short_word_count as f64) / len } else { 0.0 };
    let short_excess_per_1000 = if short_per_1000 > SHORT_BASELINE_PER_1000 { short_per_1000 - SHORT_BASELINE_PER_1000 } else { 0.0 };
    // Halved sigma coefficient from 5.0 to 2.5; removed longest_word term
    let score = v_rate + 1.5*c_rate + 2.0*d_rate + 2.5*sigma_end_rate + 2.0*bigram_rate + short_excess_per_1000 + long_word_rate;

    let (_poly_words, _greek_words, poly_ratio) = if len_greek == 0 {
        (0, 0, 0.0)
    } else {
        compute_polytonic_word_ratio(if let Some(ref s) = filtered_opt { s } else { &text })
    };
    if poly_ratio > 0.0 { flags.push("polytonic"); }
    if had_tables { flags.push("had_tables"); }

    (
        score, latin_pct, table_ratio, poly_ratio,
        len_greek, total_word_count,
        v_pen, c_pen, bad_dbl, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
        v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_excess_per_1000,
        flags.join(",")
    )
}

/// Compute noise score and latin percentage for a UTF-8 buffer. Backward-compatible API.
fn compute_score(buf: &[u8]) -> (f64, f64) {
    let (score, latin_pct, _t, _p, _lg, _tw, _v,_c,_bd,_ms,_ib,_lwc,_lw,_swc,_mr,_vr,_cr,_dr,_sr,_br,_lwr,_sr2,_sp,_f) = compute_score_and_details(buf);
    (score, latin_pct)
}

pub fn score_markdown_file_internal(path: &Path) -> anyhow::Result<f64> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let (score, _latin_pct) = compute_score(&buf);
    Ok(score)
}

pub fn score_markdown_directory_internal(root: &Path, n_threads: Option<usize>) -> anyhow::Result<Vec<(String, f64, f64)>> {
    if let Some(t) = n_threads { rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok(); }
    let results: Vec<(String, f64, f64)> = WalkDir::new(root)
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
        .collect();
    Ok(results)
}

// Detailed variants for analysis layer
pub fn score_markdown_file_detailed_internal(path: &Path) -> anyhow::Result<(f64, f64, f64, f64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64, String)> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(compute_score_and_details(&buf))
}

pub fn score_markdown_directory_detailed_internal(root: &Path, n_threads: Option<usize>) -> anyhow::Result<Vec<(String, f64, f64, f64, f64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64, String)>> {
    if let Some(t) = n_threads { rayon::ThreadPoolBuilder::new().num_threads(t).build_global().ok(); }
    let results: Vec<(String, f64, f64, f64, f64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, f64, f64, f64, f64, f64, f64, f64, f64, String)> = WalkDir::new(root)
        .into_iter()
        .par_bridge()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        .map(|e| {
            let path = e.path();
            let buf = fs::read(path).expect("read");
            let (
                score, latin_pct, table_ratio, poly_ratio,
                len_greek, total_words,
                v_pen, c_pen, bad_dbl, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
                v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_pen,
                flags
            ) = compute_score_and_details(&buf);
            (
                path.to_string_lossy().into_owned(),
                score, latin_pct, table_ratio, poly_ratio,
                len_greek, total_words,
                v_pen, c_pen, bad_dbl, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
                v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_pen,
                flags
            )
        })
        .collect();
    Ok(results)
}
