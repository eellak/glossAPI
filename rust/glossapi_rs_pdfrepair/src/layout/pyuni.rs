//! Python 3.11 string semantics used by layout v2: `str.isspace`/`isalpha`/`isdecimal`/`isdigit`/`isnumeric`/
//! `isupper` for one char, `re` IGNORECASE folding, `str.casefold`, `unicodedata.normalize("NFC")`, `str.strip`, and
//! Python float rounding. Tables come from the reference interpreter (`tables.rs`, generated), so the sets are
//! Python's (Unicode 14.0.0), not Rust's `char` methods.
use super::tables as t;

fn in_ranges(r: &[(u32, u32)], c: u32) -> bool {
    r.binary_search_by(|&(a, b)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

fn lookup<V>(m: &[(u32, V)], c: u32) -> Option<&V> {
    m.binary_search_by_key(&c, |&(k, _)| k).ok().map(|i| &m[i].1)
}

/// `str.isspace` of one char.
#[inline]
pub fn is_space(c: char) -> bool {
    let c = c as u32;
    if c < 0x80 {
        return (0x09..=0x0D).contains(&c) || (0x1C..=0x20).contains(&c);
    }
    in_ranges(t::SPACE, c)
}

#[inline]
pub fn is_alpha(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_alphabetic();
    }
    in_ranges(t::ALPHA, u)
}

/// `str.isdecimal` (also what `re`'s `\d` matches for str patterns).
#[inline]
pub fn is_decimal(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_digit();
    }
    in_ranges(t::DECIMAL, u)
}

#[inline]
pub fn is_digit(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_digit();
    }
    in_ranges(t::DIGIT, u)
}

#[inline]
pub fn is_numeric(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_digit();
    }
    in_ranges(t::NUMERIC, u)
}

/// `str.isalnum` of one char: isalpha or isdecimal or isdigit or isnumeric.
#[inline]
pub fn is_alnum(c: char) -> bool {
    is_alpha(c) || is_decimal(c) || is_digit(c) || is_numeric(c)
}

/// `str.isupper` of a one-char string.
#[inline]
pub fn is_upper(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_uppercase();
    }
    in_ranges(t::UPPER, u)
}

/// `_sre.unicode_iscased`.
pub fn re_iscased(c: u32) -> bool {
    in_ranges(t::CASED, c)
}

/// `_sre.unicode_tolower` (simple lowercase, one code point).
#[inline]
pub fn re_lower(c: u32) -> u32 {
    if c < 0x80 {
        return if (0x41..=0x5A).contains(&c) { c + 32 } else { c };
    }
    lookup(t::LOWER, c).copied().unwrap_or(c)
}

/// `re._casefix._EXTRA_CASES[lo]`.
pub fn re_extra_cases(lo: u32) -> &'static [u32] {
    lookup(t::EXTRA_CASES, lo).copied().unwrap_or(&[])
}

/// `str.isspace()` of a whole string (non-empty and every char is space).
pub fn str_isspace(s: &[char]) -> bool {
    !s.is_empty() && s.iter().all(|&c| is_space(c))
}

/// `str.strip()` bounds of a char slice.
pub fn strip_bounds(s: &[char]) -> (usize, usize) {
    let mut a = 0;
    let mut b = s.len();
    while a < b && is_space(s[a]) {
        a += 1;
    }
    while b > a && is_space(s[b - 1]) {
        b -= 1;
    }
    (a, b)
}

pub fn strip(s: &[char]) -> &[char] {
    let (a, b) = strip_bounds(s);
    &s[a..b]
}

fn ccc(c: u32) -> u8 {
    if c < 0x300 {
        return 0;
    }
    let r = t::CCC;
    match r.binary_search_by(|&(a, b, _)| {
        if b < c {
            std::cmp::Ordering::Less
        } else if a > c {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => r[i].2,
        Err(_) => 0,
    }
}

const SBASE: u32 = 0xAC00;
const LBASE: u32 = 0x1100;
const VBASE: u32 = 0x1161;
const TBASE: u32 = 0x11A7;
const LCOUNT: u32 = 19;
const VCOUNT: u32 = 21;
const TCOUNT: u32 = 28;
const NCOUNT: u32 = VCOUNT * TCOUNT;
const SCOUNT: u32 = LCOUNT * NCOUNT;

/// `unicodedata` nfd_nfkd (canonical): recursive decomposition, Hangul, then the stable insertion sort by class.
fn nfd(input: &[u32]) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(input.len() + 8);
    let mut stack: Vec<u32> = Vec::new();
    for &c in input {
        stack.push(c);
        while let Some(code) = stack.pop() {
            if (SBASE..SBASE + SCOUNT).contains(&code) {
                let s = code - SBASE;
                let l = LBASE + s / NCOUNT;
                let v = VBASE + (s % NCOUNT) / TCOUNT;
                let tt = TBASE + s % TCOUNT;
                out.push(l);
                out.push(v);
                if tt != TBASE {
                    out.push(tt);
                }
                continue;
            }
            match lookup(t::DECOMP, code) {
                Some(d) => {
                    for &x in d.iter().rev() {
                        stack.push(x);
                    }
                }
                None => out.push(code),
            }
        }
    }
    if out.len() > 1 {
        let mut prev = ccc(out[0]);
        let mut i = 1;
        while i < out.len() {
            let cur = ccc(out[i]);
            if prev == 0 || cur == 0 || prev <= cur {
                prev = cur;
                i += 1;
                continue;
            }
            let mut o = i as isize - 1;
            loop {
                out.swap(o as usize, o as usize + 1);
                o -= 1;
                if o < 0 {
                    break;
                }
                prev = ccc(out[o as usize]);
                if prev == 0 || prev <= cur {
                    break;
                }
            }
            prev = ccc(out[i]);
            i += 1;
        }
    }
    out
}

fn compose(a: u32, b: u32) -> Option<u32> {
    t::COMPOSE.binary_search_by_key(&(a, b), |&(k, _)| k).ok().map(|i| t::COMPOSE[i].1)
}

/// `unicodedata.normalize("NFC", s)`: a port of CPython 3.11 `nfc_nfkc` over the canonical decomposition.
pub fn nfc(s: &[char]) -> Vec<char> {
    let src: Vec<u32> = s.iter().map(|&c| c as u32).collect();
    let data = nfd(&src);
    let len = data.len();
    let mut output: Vec<u32> = Vec::with_capacity(len);
    let mut skipped: Vec<usize> = Vec::new();
    let mut i = 0usize;
    'again: while i < len {
        if let Some(k) = skipped.iter().position(|&x| x == i) {
            skipped.swap_remove(k);
            i += 1;
            continue 'again;
        }
        let mut code = data[i];
        if (LBASE..LBASE + LCOUNT).contains(&code) && i + 1 < len && (VBASE..VBASE + VCOUNT).contains(&data[i + 1]) {
            let li = code - LBASE;
            let vi = data[i + 1] - VBASE;
            code = SBASE + (li * VCOUNT + vi) * TCOUNT;
            i += 2;
            if i < len && TBASE < data[i] && data[i] < TBASE + TCOUNT {
                code += data[i] - TBASE;
                i += 1;
            }
            output.push(code);
            continue;
        }
        if !in_ranges(t::NFC_FIRST, code) {
            output.push(code);
            i += 1;
            continue;
        }
        let mut i1 = i + 1;
        let mut comb: u8 = 0;
        let o = output.len();
        output.push(data[i]);
        while i1 < len {
            let code1 = data[i1];
            let comb1 = ccc(code1);
            if comb != 0 {
                if comb1 == 0 {
                    break;
                }
                if comb >= comb1 {
                    i1 += 1;
                    continue;
                }
            }
            let composed = if in_ranges(t::NFC_LAST, code1) { compose(output[o], code1) } else { None };
            match composed {
                None => {
                    if comb1 == 0 {
                        break;
                    }
                    comb = comb1;
                    i1 += 1;
                    continue;
                }
                Some(c) => {
                    output[o] = c;
                    skipped.push(i1);
                    i1 += 1;
                    if !in_ranges(t::NFC_FIRST, output[o]) {
                        break;
                    }
                }
            }
        }
        i += 1;
    }
    output.into_iter().map(|c| char::from_u32(c).unwrap_or('\u{FFFD}')).collect()
}

/// `str.casefold()`.
pub fn casefold(s: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    for &c in s {
        let u = c as u32;
        if u < 0x80 {
            out.push(c.to_ascii_lowercase());
            continue;
        }
        match lookup(t::CASEFOLD, u) {
            Some(f) => out.extend(f.iter().map(|&x| char::from_u32(x).unwrap_or('\u{FFFD}'))),
            None => out.push(c),
        }
    }
    out
}

/// Python `round(x, 2)`: the correctly rounded decimal (ties to even on the exact binary value), read back.
/// Rust's `{:.2}` formatting is exact with ties to even, as CPython's `_Py_dg_dtoa` mode 3.
pub fn round2(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.2}").parse::<f64>().unwrap_or(x)
}

/// Python `round(x)` for a float (ties to even); `None` where Python raises (NaN, infinity).
pub fn round_int(x: f64) -> Option<f64> {
    if x.is_finite() {
        Some(x.round_ties_even())
    } else {
        None
    }
}

/// Python float `%` (result has the divisor's sign): CPython `float_rem`.
pub fn py_fmod(vx: f64, wx: f64) -> f64 {
    let mut m = vx % wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    m
}

/// Python `min(a, b)` (the first of equal values).
#[inline]
pub fn pmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python `max(a, b)` (the first of equal values).
#[inline]
pub fn pmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}
