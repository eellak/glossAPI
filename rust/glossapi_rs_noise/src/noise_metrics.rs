//! Core noise metric computation logic extracted from standalone binary.
//! Provides library-friendly helpers for Python bindings.

use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const GREEK_BLOCK_1: std::ops::RangeInclusive<u32> = 0x0370..=0x03FF; // Greek & Coptic
const GREEK_BLOCK_2: std::ops::RangeInclusive<u32> = 0x1F00..=0x1FFF; // Greek Extended

fn is_greek(cp: u32) -> bool {
    GREEK_BLOCK_1.contains(&cp) || GREEK_BLOCK_2.contains(&cp)
}

fn is_combining_mark(cp: u32) -> bool {
    (0x0300..=0x036F).contains(&cp) || (0x1DC0..=0x1DFF).contains(&cp) || (0x20D0..=0x20FF).contains(&cp)
}

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

static INVALID_BIGRAMS: Lazy<HashSet<(char, char)>> = Lazy::new(|| {
    let mut set = HashSet::new();
    for &c in &['κ', 'γ', 'χ'] {
        set.insert((c, 'ξ'));
    }
    for &c in &['π', 'β', 'φ'] {
        set.insert((c, 'ψ'));
    }
    set.insert(('ρ', 'λ'));
    set.insert(('μ', 'ρ'));
    set.insert(('γ', 'β'));
    set.insert(('δ', 'τ'));
    set.insert(('τ', 'δ'));
    set.insert(('β', 'π'));
    set.insert(('π', 'β'));
    set
});

static ALLOWED_DOUBLE: [u32; 9] = [
    0x03BB, 0x03BC, 0x03BD, 0x03C1, 0x03C3, 0x03C4, 0x03BA, 0x03C0, 0x03B3,
];

fn allowed_double(cp: u32) -> bool {
    ALLOWED_DOUBLE.contains(&cp)
}

/// Compute metrics for UTF-8 bytes; ported from original CLI.
fn analyse_bytes(buf: &[u8]) -> (u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) {
    let mut len_greek = 0u64;
    let mut v_pen = 0u64;
    let mut c_pen = 0u64;
    let mut bad_double = 0u64;
    let mut max_run = 0u64;
    let mut long_word_count = 0u64;
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
            if let (Some(pc), Some(cc)) = (std::char::from_u32(prev_cp), std::char::from_u32(cp)) {
                let pc_low = pc.to_lowercase().next().unwrap_or(pc);
                let cc_low = cc.to_lowercase().next().unwrap_or(cc);
                if INVALID_BIGRAMS.contains(&(pc_low, cc_low)) { invalid_bigram += 1; }
            }
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
        if word_len >= LONG_WORD_LIMIT { long_word_count += 1; }
        if word_len > longest_word { longest_word = word_len; }
        if prev_cp == 0x03C3 { misplaced_sigma += 1; }
    }

    (len_greek, v_pen, c_pen, bad_double, max_run, long_word_count, longest_word, misplaced_sigma, invalid_bigram, short_word_count, total_word_count)
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

/// Compute noise score and latin percentage for a UTF-8 buffer.
fn compute_score(buf: &[u8]) -> (f64, f64) {
    let (len_greek, v_pen, c_pen, bad_dbl, _max_run, long_word_count, longest_word, misplaced_sigma, invalid_bigram, short_word_count, total_word_count) = analyse_bytes(buf);
    if len_greek == 0 {
        return (100.0, 1.0); // treat as worst possible
    }
    let len = len_greek as f64;
    let v_rate = 1000.0 * v_pen as f64 / len;
    let c_rate = 1000.0 * c_pen as f64 / len;
    let d_rate = 1000.0 * bad_dbl as f64 / len;
    let sigma_end_rate = 1000.0 * misplaced_sigma as f64 / len;
    let bigram_rate = 1000.0 * invalid_bigram as f64 / len;
    let long_word_rate = 1000.0 * long_word_count as f64 / len;
    let short_ratio = if total_word_count > 0 {
        short_word_count as f64 / total_word_count as f64
    } else {
        0.0
    };
    let short_pen = if short_ratio > 0.4 { (short_ratio - 0.4) * 10.0 } else { 0.0 };
    let score = v_rate + 1.5*c_rate + 2.0*d_rate + 5.0*sigma_end_rate + 2.0*bigram_rate + short_pen + long_word_rate + (longest_word as f64 / 10.0);

    // latin percentage (quick heuristic): count non-greek ascii letters
    let latin_chars = buf.iter().filter(|&&b| (b >= 0x41 && b <= 0x5A) || (b >= 0x61 && b <= 0x7A)).count();
    let latin_pct = latin_chars as f64 / (buf.len() as f64);

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
    let paths: Vec<PathBuf> = WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        .map(|e| e.path().to_path_buf())
        .collect();

    let results: Vec<(String, f64, f64)> = paths.par_iter().map(|path| {
        let mut file = File::open(path).expect("open");
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).expect("read");
        let (score, latin_pct) = compute_score(&buf);
        (path.to_string_lossy().into_owned(), score, latin_pct)
    }).collect();

    Ok(results)
}
