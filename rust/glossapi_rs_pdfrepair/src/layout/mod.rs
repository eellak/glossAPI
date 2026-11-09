//! Layout v2 (K5g, ported in K5-A1): labels on the plain PDFium reading order.
//!
//! A line-for-line port of the frozen Python reference `tests/layout_v2_ref/labels_v2.py` (sha256 23d5bae3...). The
//! plain text is never moved: it is cut into blocks only at its own newlines (or at whitespace where PDFium joined two
//! visual rows) and each block gets a label (title, section_header + level, paragraph, list_item, caption, footnote,
//! page_header, page_footer, table, picture, formula). Char geometry comes from the page's `.k5chars` table.
//!
//! Exactness rules (`RUST_PORT.md`): offsets are code points; Python's `isspace`/`isalpha`/`isdigit`/... sets and
//! `re` semantics (`pyuni`, `pyre`); `round` half to even; `round(x, 2)` correctly rounded; float `%` with the
//! divisor's sign; `median_low`; first maximum/minimum on ties; stable sorts; the reference's pass order. The
//! `text_edge` cache quirk is copied (cache key `(angle, round(x0))`, value computed with the first caller's `x0`).
//! Where the reference raises on a degenerate page (ZeroDivisionError, NaN/inf angle), the port returns an error.
pub mod difflib;
pub mod k5chars;
pub mod pyre;
pub mod pyuni;
pub mod reader;
#[rustfmt::skip]
pub mod tables;
#[cfg(test)]
mod tests;

use anyhow::{anyhow, bail, Result};
use k5chars::CharTable;
use pyre::Regex;
use pyuni::{is_alnum, is_alpha, is_decimal, is_digit, is_space, is_upper, pmax, pmin, round2, strip, str_isspace};
use std::collections::HashMap;
use std::sync::OnceLock;

// ---- constants -----------------------------------------------------------------------------------------------------
pub const GENERATED: u8 = 1;
pub const ITALIC: u8 = 8;
const CELL_GAP: f64 = 1.5;
const LEADING: f64 = 1.5;
const SIZE_STEP: f64 = 0.10;
const HEADING: f64 = 1.15;
const FURN_BAND: f64 = 0.13;
const FURN_SHARE: f64 = 0.30;
const FURN_MIN: usize = 3;
const FURN_TOL: f64 = 6.0;
const FOOT_SIZE: f64 = 0.9;

struct Pats {
    bold_name: Regex,
    italic_name: Regex,
    math_font: Regex,
    bullet: Regex,
    enumer: Regex,
    numbered: Regex,
    secnum: Regex,
    caption: Regex,
    footmark: Regex,
    page_num: Regex,
    line_start: Regex,
    figword: Regex,
    table_rule: Regex,
    mark_alone: Regex,
}

fn pats() -> &'static Pats {
    static P: OnceLock<Pats> = OnceLock::new();
    P.get_or_init(|| {
        let r = |p: (&str, bool)| Regex::new(p.0, p.1);
        Pats {
            bold_name: r(tables::BOLD_NAME),
            italic_name: r(tables::ITALIC_NAME),
            math_font: r(tables::MATH_FONT),
            bullet: r(tables::BULLET),
            enumer: r(tables::ENUM),
            numbered: r(tables::NUMBERED),
            secnum: r(tables::SECNUM),
            caption: r(tables::CAPTION),
            footmark: r(tables::FOOTMARK),
            page_num: r(tables::PAGE_NUM),
            line_start: r(tables::LINE_START),
            figword: r(tables::FIGWORD),
            table_rule: r(tables::TABLE_RULE),
            mark_alone: r(tables::MARK_ALONE),
        }
    })
}

/// Test hook: the boolean the reference computes with pattern `name` (search for font names, fullmatch for
/// MARK_ALONE, match otherwise). Unknown names are an error.
pub fn pattern_probe(name: &str, text: &str) -> Result<bool> {
    let p = pats();
    let s: Vec<char> = text.chars().collect();
    Ok(match name {
        "BOLD_NAME" => p.bold_name.is_search(&s),
        "ITALIC_NAME" => p.italic_name.is_search(&s),
        "MATH_FONT" => p.math_font.is_search(&s),
        "BULLET" => p.bullet.is_match(&s),
        "ENUM" => p.enumer.is_match(&s),
        "NUMBERED" => p.numbered.is_match(&s),
        "SECNUM" => p.secnum.is_match(&s),
        "CAPTION" => p.caption.is_match(&s),
        "FOOTMARK" => p.footmark.is_match(&s),
        "PAGE_NUM" => p.page_num.is_match(&s),
        "LINE_START" => p.line_start.is_match(&s),
        "FIGWORD" => p.figword.is_match(&s),
        "TABLE_RULE" => p.table_rule.is_match(&s),
        "MARK_ALONE" => p.mark_alone.is_fullmatch(&s),
        _ => bail!("unknown pattern {name}"),
    })
}

fn median_low(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[(xs.len() - 1) / 2]
}

/// Inputs rounded to 1/64 pt, half away from zero.
fn q(v: f64) -> f64 {
    if v >= 0.0 {
        (v * 64.0 + 0.5).floor() / 64.0
    } else {
        -(-v * 64.0 + 0.5).floor() / 64.0
    }
}

/// `round(s * 2) / 2` (Python round: half to even; NaN/inf raise there, so they are an error here).
fn size_key(s: f64) -> Result<f64> {
    match pyuni::round_int(s * 2.0) {
        Some(r) => Ok(r / 2.0),
        None => bail!("ValueError: size_key of a non-finite size"),
    }
}

/// An insertion-ordered float -> count map (Python dict with float keys; few distinct keys per page).
#[derive(Default)]
struct FMap(Vec<(f64, usize)>);

impl FMap {
    fn add(&mut self, k: f64, v: usize) {
        match self.0.iter_mut().find(|e| e.0 == k) {
            Some(e) => e.1 += v,
            None => self.0.push((k, v)),
        }
    }
    fn get(&self, k: f64) -> usize {
        self.0.iter().find(|e| e.0 == k).map(|e| e.1).unwrap_or(0)
    }
    /// `max(d, key=lambda k: (d[k], -k))` (keys are distinct, so no tie).
    fn mode(&self) -> Option<f64> {
        let mut best: Option<(f64, usize)> = None;
        for &(k, c) in &self.0 {
            best = match best {
                None => Some((k, c)),
                Some((bk, bc)) => {
                    if (c, -k).partial_cmp(&(bc, -bk)) == Some(std::cmp::Ordering::Greater) {
                        Some((k, c))
                    } else {
                        Some((bk, bc))
                    }
                }
            }
        }
        best.map(|b| b.0)
    }
    fn total(&self) -> usize {
        self.0.iter().map(|e| e.1).sum()
    }
}

// ---- 1. units ------------------------------------------------------------------------------------------------------
#[derive(Clone, Debug)]
struct Unit {
    ch: char,
    geo: Option<[f64; 4]>,
    size: f64,
    bold: bool,
    math: bool,
    angle: i32,
    italic: bool,
    weight: i32,
}

/// PDFium UTF-16 code units -> (index of the first unit, code point); a lone surrogate becomes U+FFFD.
fn utf16_units(chs: &[u32]) -> Vec<(usize, char)> {
    let mut out = Vec::with_capacity(chs.len());
    let (mut i, n) = (0usize, chs.len());
    while i < n {
        let c = chs[i];
        if (0xD800..=0xDBFF).contains(&c) && i + 1 < n && (0xDC00..=0xDFFF).contains(&chs[i + 1]) {
            let cp = 0x10000 + ((c - 0xD800) << 10) + (chs[i + 1] - 0xDC00);
            out.push((i, char::from_u32(cp).unwrap_or('\u{FFFD}')));
            i += 2;
            continue;
        }
        out.push((i, char::from_u32(c).unwrap_or('\u{FFFD}')));
        i += 1;
    }
    out
}

/// Radians -> 0/90/180/270 (nearest quarter turn); NaN/inf raise in the reference (math.floor of NaN).
fn quant_angle(a: f64) -> Result<i32> {
    let d = pyuni::py_fmod(a * (180.0f64 / std::f64::consts::PI), 360.0);
    let f = (d / 90.0 + 0.5).floor();
    if !f.is_finite() {
        bail!("ValueError: char angle is not finite");
    }
    Ok(90 * (f as i64).rem_euclid(4) as i32)
}

fn frame(angle: i32) -> ((f64, f64), (f64, f64)) {
    match angle {
        0 => ((1.0, 0.0), (0.0, 1.0)),
        270 => ((0.0, 1.0), (-1.0, 0.0)),
        180 => ((-1.0, 0.0), (0.0, -1.0)),
        _ => ((0.0, -1.0), (1.0, 0.0)),
    }
}

fn min_first(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |m, &x| pmin(m, x))
}

fn max_first(v: &[f64]) -> f64 {
    v[1..].iter().fold(v[0], |m, &x| pmax(m, x))
}

fn to_frame(b: [f64; 4], angle: i32, origin: (f64, f64)) -> [f64; 4] {
    let ((dx, dy), (ux, uy)) = frame(angle);
    let [l, bo, r, t] = b;
    let xs = [l * dx + bo * dy, r * dx + t * dy];
    let ys = [l * ux + bo * uy, r * ux + t * uy];
    let (ox, oy) = origin;
    [min_first(&xs) - ox, oy - max_first(&ys), max_first(&xs) - ox, oy - min_first(&ys)]
}

/// Frame of each quarter turn: (origin, W, H), indexed by angle / 90.
type Frames = [((f64, f64), f64, f64); 4];

fn frames_of(t: &CharTable) -> Frames {
    let corners = [(0.0, 0.0), (t.page_w, 0.0), (0.0, t.page_h), (t.page_w, t.page_h)];
    let mut out = [((0.0, 0.0), 0.0, 0.0); 4];
    for a in [0, 90, 180, 270] {
        let ((dx, dy), (ux, uy)) = frame(a);
        let fx: Vec<f64> = corners.iter().map(|&(px, py)| px * dx + py * dy).collect();
        let fy: Vec<f64> = corners.iter().map(|&(px, py)| px * ux + py * uy).collect();
        out[(a / 90) as usize] =
            ((min_first(&fx), max_first(&fy)), max_first(&fx) - min_first(&fx), max_first(&fy) - min_first(&fy));
    }
    out
}

struct FontFlags {
    bold: bool,
    math: bool,
    ital_name: bool,
}

fn font_flags(name: &str) -> FontFlags {
    let p = pats();
    let tail: Vec<char> = name.rsplit('+').next().unwrap_or("").chars().collect();
    let full: Vec<char> = name.chars().collect();
    FontFlags {
        bold: p.bold_name.is_search(&tail),
        math: p.math_font.is_search(&full),
        ital_name: p.italic_name.is_search(&tail),
    }
}

/// -> (units, frames, display angle, majority angle).
fn page_units(t: &CharTable) -> Result<(Vec<Unit>, Frames, i64, i32)> {
    let raw = utf16_units(&t.ch);
    let frames = frames_of(t);
    let disp = (360 - t.rotation as i64).rem_euclid(360);
    let mut counts = [0usize; 4];
    let mut units = Vec::with_capacity(raw.len());
    let mut cache: Vec<Option<FontFlags>> = (0..t.fonts.len()).map(|_| None).collect();
    let empty = font_flags("");
    let mut last_angle: Option<(u32, i32)> = None;     // chars mostly share one angle: quantise each value once
    for (i, c) in raw {
        let bits = t.angle[i].to_bits();
        let a = match last_angle {
            Some((b, a)) if b == bits => a,
            _ => {
                let a = quant_angle(t.angle[i] as f64)?;
                last_angle = Some((bits, a));
                a
            }
        };
        let fi = t.font[i] as usize;
        let ff = if fi < t.fonts.len() { &*cache[fi].get_or_insert_with(|| font_flags(&t.fonts[fi])) } else { &empty };
        let (bold, mth) = (ff.bold, ff.math);
        let ital = (t.flags[i] & ITALIC) != 0 || ff.ital_name;
        let wt = t.weight[i] as i32;
        let (bb, lo) = (t.bbox[i], t.loose[i]);
        if is_space(c) || bb.iter().chain(lo.iter()).any(|&v| v.is_nan() || v.abs() > 1e6) {
            units.push(Unit { ch: c, geo: None, size: 0.0, bold, math: mth, angle: a, italic: ital, weight: wt });
            continue;
        }
        let (origin, w, h) = frames[(a / 90) as usize];
        let bb = bb.map(q);
        let lo = lo.map(q);
        let [mut x0, mut y0, mut x1, mut y1] = to_frame(bb, a, origin);
        let [lx0, ly0, lx1, ly1] = to_frame(lo, a, origin);
        let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
        if !(0.0 <= cx && cx <= w && 0.0 <= cy && cy <= h) {
            units.push(Unit { ch: c, geo: None, size: 0.0, bold, math: mth, angle: a, italic: ital, weight: wt });
            continue;
        }
        let size = if ly1 > ly0 {
            ly1 - ly0
        } else if y1 > y0 {
            y1 - y0
        } else {
            (t.size[i] as f64).abs()
        };
        if lx1 > lx0 {
            x0 = lx0;
            x1 = lx1;
        }
        if ly1 > ly0 {
            y0 = ly0;
            y1 = ly1;
        }
        counts[(a / 90) as usize] += 1;
        units.push(Unit { ch: c, geo: Some([x0, y0, x1, y1]), size, bold, math: mth, angle: a, italic: ital, weight: wt });
    }
    // max(counts, key=(count, k == disp, -k)) over 0, 90, 180, 270
    let key = |k: i32| (counts[(k / 90) as usize], (k as i64 == disp) as u8, -k);
    let mut major = 0;
    for k in [90, 180, 270] {
        if key(k) > key(major) {
            major = k;
        }
    }
    Ok((units, frames, disp, major))
}

// ---- 2. align ------------------------------------------------------------------------------------------------------
fn align(plain: &[char], units: &[Unit]) -> (Vec<isize>, usize) {
    let pi: Vec<usize> = (0..plain.len()).filter(|&k| !is_space(plain[k])).collect();
    let ui: Vec<usize> = (0..units.len()).filter(|&k| !is_space(units[k].ch)).collect();
    let ps: Vec<char> = pi.iter().map(|&k| plain[k]).collect();
    let us: Vec<char> = ui.iter().map(|&k| units[k].ch).collect();
    let mut m = vec![-1isize; plain.len()];
    if ps == us {
        for (&a, &b) in pi.iter().zip(ui.iter()) {
            m[a] = b as isize;
        }
        return (m, 0);
    }
    let sm = difflib::Matcher::new(&ps, &us);
    let mut matched = 0;
    for (i1, j1, size) in sm.matching_blocks() {
        for o in 0..size {
            m[pi[i1 + o]] = ui[j1 + o] as isize;
        }
        matched += size;
    }
    (m, ps.len() - matched)
}

// ---- 3. lines ------------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    Top,
    Bottom,
}

impl Band {
    pub fn name(self) -> &'static str {
        match self {
            Band::Top => "top",
            Band::Bottom => "bottom",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PLine {
    pub k: usize,
    pub start: usize,
    pub end: usize,
    pub n: usize,
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub size: f64,
    pub bold: f64,
    pub math: f64,
    pub angle: i32,
    pub gaps: Vec<(f64, f64)>,
    pub emph: f64,
    pub wts: Vec<(i32, usize)>,
    pub rows: usize,
    pub fx0: f64,
    pub fy1: f64,
    pub lx1: f64,
    pub ly0: f64,
    pub ly1: f64,
    pub mate_prev: bool,
    pub furniture: Option<Band>,
    pub furn_reason: Option<&'static str>,
}

impl PLine {
    fn new(k: usize, start: usize, end: usize) -> Self {
        PLine { k, start, end, rows: 1, ..Default::default() }
    }
    #[inline]
    pub fn geo(&self) -> bool {
        self.n > 0
    }
    #[inline]
    fn yc(&self) -> f64 {
        (self.y0 + self.y1) / 2.0
    }
}

fn split_lines(plain: &[char]) -> Vec<PLine> {
    let mut out = Vec::new();
    let mut pos = 0;
    for (k, seg) in plain.split(|&c| c == '\n').enumerate() {
        out.push(PLine::new(k, pos, pos + seg.len()));
        pos += seg.len() + 1;
    }
    out
}

fn ugeo<'a>(units: &'a [Unit], amap: &[isize], i: usize) -> Option<&'a Unit> {
    let m = amap[i];
    if m >= 0 {
        let u = &units[m as usize];
        if u.geo.is_some() {
            return Some(u);
        }
    }
    None
}

fn split_rows(lines: Vec<PLine>, plain: &[char], units: &[Unit], amap: &[isize]) -> Vec<PLine> {
    let p = pats();
    let mut out: Vec<PLine> = Vec::new();
    for ln in lines {
        let pos: Vec<usize> = (ln.start..ln.end).filter(|&i| ugeo(units, amap, i).is_some()).collect();
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        if pos.len() >= 2 && !p.table_rule.is_match(&plain[ln.start..ln.end]) {
            let sz = median_low(pos.iter().map(|&i| ugeo(units, amap, i).unwrap().size).collect());
            let mut cur: Option<(f64, i32)> = None;
            let mut prev: Option<usize> = None;
            for &i in &pos {
                let u = ugeo(units, amap, i).unwrap();
                if u.size < 0.8 * sz {
                    prev = Some(i);
                    continue;
                }
                let g = u.geo.unwrap();
                let yc = (g[1] + g[3]) / 2.0;
                if let (Some(c), Some(pv)) = (cur, prev) {
                    if (yc - c.0).abs() > 0.5 * sz || u.angle != c.1 {
                        let gap = &plain[pv + 1..i];
                        if str_isspace(gap) && !p.line_start.is_match(&plain[i..ln.end]) {
                            cuts.push((pv + 1, i));
                        }
                    }
                }
                if cur.is_none_or(|c| (yc - c.0).abs() > 0.5 * sz || u.angle != c.1) {
                    cur = Some((yc, u.angle));
                }
                prev = Some(i);
            }
        }
        let mut a = ln.start;
        for (e, b) in cuts {
            out.push(PLine::new(out.len(), a, e));
            a = b;
        }
        out.push(PLine::new(out.len(), a, ln.end));
    }
    out
}

fn line_geometry(lines: &mut [PLine], units: &[Unit], amap: &[isize], disp: i64) {
    for ln in lines.iter_mut() {
        let us: Vec<&Unit> = (ln.start..ln.end).filter_map(|i| ugeo(units, amap, i)).collect();
        if us.is_empty() {
            continue;
        }
        let mut acount: Vec<(i32, usize)> = Vec::new();
        for u in &us {
            match acount.iter_mut().find(|e| e.0 == u.angle) {
                Some(e) => e.1 += 1,
                None => acount.push((u.angle, 1)),
            }
        }
        let key = |&(a, c): &(i32, usize)| (c, (a as i64 == disp) as u8, -a);
        let mut best = acount[0];
        for e in &acount[1..] {
            if key(e) > key(&best) {
                best = *e;
            }
        }
        ln.angle = best.0;
        let main: Vec<&Unit> = us.iter().copied().filter(|u| u.angle == ln.angle).collect();
        let gm = |u: &Unit| u.geo.unwrap();
        ln.n = main.len();
        ln.x0 = main[1..].iter().fold(gm(main[0])[0], |m, u| pmin(m, gm(u)[0]));
        ln.x1 = main[1..].iter().fold(gm(main[0])[2], |m, u| pmax(m, gm(u)[2]));
        let sz = median_low(main.iter().map(|u| u.size).collect());
        let mut big: Vec<&Unit> = main.iter().copied().filter(|u| u.size >= 0.8 * sz).collect();
        if big.is_empty() {
            big = main.clone();
        }
        let mut rows: Vec<(f64, Vec<&Unit>)> = Vec::new();
        for u in big {
            let g = gm(u);
            let yc = (g[1] + g[3]) / 2.0;
            if let Some(last) = rows.last_mut() {
                if (yc - last.0).abs() <= 0.5 * sz {
                    last.1.push(u);
                    continue;
                }
            }
            if let Some(r) = rows.iter_mut().find(|r| (yc - r.0).abs() <= 0.5 * sz) {
                r.1.push(u);
            } else {
                rows.push((yc, vec![u]));
            }
        }
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let r0 = &rows[0].1;
        let r1 = &rows[rows.len() - 1].1;
        ln.rows = rows.len();
        ln.y0 = median_low(r0.iter().map(|u| gm(u)[1]).collect());
        ln.fy1 = median_low(r0.iter().map(|u| gm(u)[3]).collect());
        ln.ly0 = median_low(r1.iter().map(|u| gm(u)[1]).collect());
        ln.y1 = median_low(r1.iter().map(|u| gm(u)[3]).collect());
        ln.ly1 = ln.y1;
        ln.fx0 = r0[1..].iter().fold(gm(r0[0])[0], |m, u| pmin(m, gm(u)[0]));
        ln.lx1 = r1[1..].iter().fold(gm(r1[0])[2], |m, u| pmax(m, gm(u)[2]));
        ln.size = sz;
        let nm = main.len() as f64;
        ln.bold = main.iter().filter(|u| u.bold).count() as f64 / nm;
        ln.emph = main.iter().filter(|u| u.bold || u.italic).count() as f64 / nm;
        for u in &main {
            match ln.wts.iter_mut().find(|e| e.0 == u.weight) {
                Some(e) => e.1 += 1,
                None => ln.wts.push((u.weight, 1)),
            }
        }
        ln.math = main.iter().filter(|u| u.math).count() as f64 / nm;
        if ln.rows == 1 {
            let mut seq = main.clone();
            seq.sort_by(|a, b| gm(a)[0].partial_cmp(&gm(b)[0]).unwrap_or(std::cmp::Ordering::Equal));
            for w in seq.windows(2) {
                let (a, b) = (gm(w[0]), gm(w[1]));
                if b[0] - a[2] > CELL_GAP * pmax(sz, 1.0) {
                    ln.gaps.push((a[2], b[0]));
                }
            }
        }
    }
    let mut prev: Option<usize> = None;
    for k in 0..lines.len() {
        if !lines[k].geo() {
            continue;
        }
        if let Some(p) = prev {
            let (pv, ln) = (&lines[p], &lines[k]);
            if pv.angle == ln.angle {
                let h = pmin(pv.ly1 - pv.ly0, ln.fy1 - ln.y0);
                let ov = pmin(pv.ly1, ln.fy1) - pmax(pv.ly0, ln.y0);
                if h > 0.0 && ov >= 0.5 * h && ln.fx0 >= pv.lx1 - 0.5 * ln.size {
                    lines[k].mate_prev = true;
                }
            }
        }
        prev = Some(k);
    }
}

// ---- document pass -------------------------------------------------------------------------------------------------
/// NFC, casefold, digits (`\d`) -> '#', whitespace runs -> ' ', strip.
pub fn norm_sig(text: &[char]) -> Vec<char> {
    let t = pyuni::casefold(&pyuni::nfc(text));
    let mut out: Vec<char> = Vec::with_capacity(t.len());
    let mut in_space = false;
    for c in t {
        if is_space(c) {
            if !in_space {
                out.push(' ');
            }
            in_space = true;
            continue;
        }
        in_space = false;
        out.push(if is_decimal(c) { '#' } else { c });
    }
    strip(&out).to_vec()
}

pub struct PageGeo {
    plain: Vec<char>,
    pub lines: Vec<PLine>,
    frames: Frames,
    pub disp: i64,
    pub major: i32,
    pub unmatched: usize,
    pub units: usize,
}

impl PageGeo {
    #[inline]
    fn w(&self, ln: &PLine) -> f64 {
        self.frames[(ln.angle / 90) as usize].1
    }
    #[inline]
    fn h(&self, ln: &PLine) -> f64 {
        self.frames[(ln.angle / 90) as usize].2
    }
    #[inline]
    pub fn text(&self, ln: &PLine) -> &[char] {
        &self.plain[ln.start..ln.end]
    }
}

/// Bounds the reference never meets on real pages; outside them Python's behaviour is undefined (NaN geometry,
/// OverflowError deep inside), so the port refuses the page.
fn check_table(t: &CharTable) -> Result<()> {
    for v in [t.page_w, t.page_h] {
        if !v.is_finite() || v.abs() > 1e7 {
            bail!("ValueError: page box {v} is not a finite PDF size");
        }
    }
    Ok(())
}

pub fn page_geo(plain: &[char], t: &CharTable) -> Result<PageGeo> {
    check_table(t)?;
    let (units, frames, disp, major) = page_units(t)?;
    let (amap, unmatched) = align(plain, &units);
    let mut lines = split_rows(split_lines(plain), plain, &units, &amap);
    line_geometry(&mut lines, &units, &amap, disp);
    Ok(PageGeo { plain: plain.to_vec(), lines, frames, disp, major, unmatched, units: units.len() })
}

/// The plain arm's text rebuilt from the char table (get_text_range + newline normalisation + str.strip()).
pub fn fixture_text(t: &CharTable) -> Vec<char> {
    let s: Vec<char> = utf16_units(&t.ch).into_iter().map(|(_, c)| c).collect();
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == '\r' {
            out.push('\n');
            if i + 1 < s.len() && s[i + 1] == '\n' {
                i += 1;
            }
        } else {
            out.push(s[i]);
        }
        i += 1;
    }
    strip(&out).to_vec()
}

fn band_of(ln: &PLine, g: &PageGeo) -> Option<Band> {
    if ln.angle as i64 != g.disp {
        return None;
    }
    let h = g.h(ln);
    if ln.y1 <= FURN_BAND * h {
        return Some(Band::Top);
    }
    if ln.y0 >= (1.0 - FURN_BAND) * h {
        return Some(Band::Bottom);
    }
    None
}

#[derive(Clone, Debug)]
pub struct Furniture {
    pub band: Band,
    pub signature: String,
    pub y: f64,
    pub pages: usize,
}

#[derive(Clone, Debug)]
pub struct DocContext {
    pub body: f64,
    pub heading_sizes: Vec<f64>,
    pub furniture: Vec<Furniture>,
    pub pages: usize,
}

/// One (band, signature) key and its (page, centre y) occurrences, in insertion order.
type Occurrences = ((Band, Vec<char>), Vec<(i64, f64)>);

/// geos: the document's sampled pages in the reference's dict order (the arm sorts by page number).
pub fn document_context(geos: &[(i64, PageGeo)]) -> Result<DocContext> {
    let mut b = ContextBuilder::default();
    for (pg, g) in geos {
        b.add(*pg, g)?;
    }
    b.finish()
}

/// The reference's document pass, fed one page at a time (K5-A2: production adds the document's context pages in
/// page-number order and keeps no page geometry). Every quantity of `document_context` is a sum or an ordered list
/// over the pages, except the heading-size filter, which needs the body size: each page keeps the sizes of its
/// heading-shaped lines (width and letter conditions) and the size filter runs in `finish`. Adding the pages of
/// `geos` in order and calling `finish` equals the reference's pass over `geos`, including its errors (`size_key` of
/// a non-finite size on a geometric line).
#[derive(Default)]
pub struct ContextBuilder {
    sizes: FMap,
    heading_candidates: Vec<f64>,
    texted: Vec<i64>,
    occ: Vec<Occurrences>,
    occ_ix: HashMap<(u8, Vec<char>), usize>,
    added: usize,
}

impl ContextBuilder {
    pub fn add(&mut self, pg: i64, g: &PageGeo) -> Result<()> {
        for ln in g.lines.iter().filter(|l| l.geo()) {
            self.sizes.add(size_key(ln.size)?, ln.n);
        }
        for ln in &g.lines {
            if ln.geo() && ln.x1 - ln.x0 <= 0.7 * g.w(ln) && g.text(ln).iter().filter(|&&c| is_alpha(c)).count() >= 3 {
                self.heading_candidates.push(ln.size);
            }
        }
        if g.lines.iter().any(|l| l.geo()) {
            self.texted.push(pg);
        }
        for ln in &g.lines {
            if ln.geo() && !strip(g.text(ln)).is_empty() {
                if let Some(band) = band_of(ln, g) {
                    let sig = norm_sig(g.text(ln));
                    let key = (band as u8, sig.clone());
                    let occ = &mut self.occ;
                    let ix = *self.occ_ix.entry(key).or_insert_with(|| {
                        occ.push(((band, sig), Vec::new()));
                        occ.len() - 1
                    });
                    self.occ[ix].1.push((pg, ln.yc()));
                }
            }
        }
        self.added += 1;
        Ok(())
    }

    /// Pages added so far.
    pub fn added(&self) -> usize {
        self.added
    }

    pub fn finish(&self) -> Result<DocContext> {
        let body = self.sizes.mode().unwrap_or(10.0);
        let mut hs = FMap::default();
        for &size in &self.heading_candidates {
            if HEADING * body <= size && size <= 4.0 * body {
                hs.add(size_key(size)?, 1);
            }
        }
        let mut heading_sizes: Vec<f64> = hs.0.iter().map(|e| e.0).collect();
        heading_sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        heading_sizes.truncate(3);
        let texted = &self.texted;
        let need = FURN_MIN.max((FURN_SHARE * texted.len() as f64).ceil() as usize);
        let need_par: Vec<usize> = (0..2)
            .map(|k| {
                let c = texted.iter().filter(|&&pg| pg.rem_euclid(2) == k).count();
                FURN_MIN.max((FURN_SHARE * c as f64).ceil() as usize)
            })
            .collect();
        // sorted(occ.items()): by (band name, signature) in code point order
        let mut occ: Vec<&Occurrences> = self.occ.iter().collect();
        occ.sort_by(|a, b| (a.0 .0.name(), &a.0 .1).cmp(&(b.0 .0.name(), &b.0 .1)));
        let mut furniture = Vec::new();
        for ((band, sig), lst) in occ {
            let mut seen: Vec<f64> = Vec::new();
            for &(pg, yc) in lst {
                let mut pages: Vec<i64> = lst.iter().filter(|(_, y)| (y - yc).abs() <= FURN_TOL).map(|(p, _)| *p).collect();
                pages.sort_unstable();
                pages.dedup();
                let same = pages.iter().filter(|&&p| p.rem_euclid(2) == pg.rem_euclid(2)).count();
                if (pages.len() >= need || same >= need_par[pg.rem_euclid(2) as usize])
                    && !seen.iter().any(|&y| (yc - y).abs() <= 0.01)
                {
                    seen.push(yc);
                    furniture.push(Furniture {
                        band: *band,
                        signature: sig.iter().collect(),
                        y: round2(yc),
                        pages: pages.len(),
                    });
                }
            }
        }
        Ok(DocContext { body, heading_sizes, furniture, pages: texted.len() })
    }
}

/// K5A1-G3 input contract (K5-A2): the reason a table is outside the exact-parity contract, or `None`. The page box
/// must be finite and at most 1e7 pt (as `page_geo` requires), and every char's font size and angle must be finite
/// (the reference raises on a non-finite angle and sorts non-finite sizes differently). Production emits such a page
/// unlabelled with a `layout_input_rejected` warning and keeps it out of the document pass; its text is unchanged.
pub fn input_problem(t: &CharTable) -> Option<String> {
    if let Err(e) = check_table(t) {
        return Some(e.to_string().trim_start_matches("ValueError: ").to_string());
    }
    if let Some(i) = t.size.iter().position(|s| !s.is_finite()) {
        return Some(format!("char {i} has a non-finite font size {}", t.size[i]));
    }
    if let Some(i) = t.angle.iter().position(|a| !a.is_finite()) {
        return Some(format!("char {i} has a non-finite angle {}", t.angle[i]));
    }
    None
}

// ---- per page: furniture, tables, blocks, labels -------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct Block {
    pub first: usize,
    pub last: usize,
    pub label: &'static str,
    pub level: Option<i64>,
    pub start: usize,
    pub end: usize,
    pub bbox: Option<[f64; 4]>,
    pub why: &'static str,
    pub rotated: i64,
    pub cut: Option<&'static str>,
}

impl Block {
    fn new(first: usize, last: usize, label: &'static str, why: &'static str) -> Self {
        Block { first, last, label, level: None, start: 0, end: 0, bbox: None, why, rotated: 0, cut: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Warning {
    pub kind: &'static str,
    pub key: &'static str,
    pub value: i64,
}

pub struct PageLabels {
    pub blocks: Vec<Block>,
    pub warnings: Vec<Warning>,
    pub rotation: i64,
    /// Line pieces (start, end) in code points: the differential test compares the segmentation too.
    pub lines: Vec<(usize, usize)>,
}

fn mark_furniture(g: &mut PageGeo, ctx: &DocContext) {
    let p = pats();
    let geo: Vec<usize> = (0..g.lines.len())
        .filter(|&i| {
            let ln = &g.lines[i];
            ln.geo() && !strip(g.text(ln)).is_empty() && ln.angle as i64 == g.disp
        })
        .collect();
    if geo.is_empty() {
        return;
    }
    let mut top = geo[0];
    let mut bot = geo[0];
    for &i in &geo[1..] {
        let (l, t, b) = (&g.lines[i], &g.lines[top], &g.lines[bot]);
        if (l.y0, l.k).partial_cmp(&(t.y0, t.k)) == Some(std::cmp::Ordering::Less) {
            top = i;
        }
        if (l.y1, -(l.k as i64)).partial_cmp(&(b.y1, -(b.k as i64))) == Some(std::cmp::Ordering::Greater) {
            bot = i;
        }
    }
    for &i in &geo {
        let Some(band) = band_of(&g.lines[i], g) else { continue };
        let txt = strip(g.text(&g.lines[i])).to_vec();
        if (i == top || i == bot) && p.page_num.is_match(&txt) {
            g.lines[i].furniture = Some(band);
            g.lines[i].furn_reason = Some("page_number");
            continue;
        }
        let sig: String = norm_sig(&txt).into_iter().collect();
        let yc = g.lines[i].yc();
        for f in &ctx.furniture {
            if f.band == band && f.signature == sig && (f.y - yc).abs() <= FURN_TOL {
                g.lines[i].furniture = Some(band);
                g.lines[i].furn_reason = Some("repeated");
                break;
            }
        }
    }
    for &i in &geo {
        if g.lines[i].furn_reason != Some("repeated") {
            continue;
        }
        for &j in &geo {
            let (o, ln) = (&g.lines[j], &g.lines[i]);
            if j == i || o.furniture.is_some() || pmin(o.x1, ln.x1) <= pmax(o.x0, ln.x0) {
                continue;
            }
            let gap = pmax(o.y0 - ln.y1, ln.y0 - o.y1);
            if gap < 0.5 * pmin(o.size, ln.size) {
                g.lines[i].furniture = None;
                g.lines[i].furn_reason = None;
                break;
            }
        }
    }
}

fn numeric_share(text: &[char]) -> f64 {
    let t: Vec<char> = text.iter().copied().filter(|&c| !is_space(c)).collect();
    t.iter().filter(|&&c| is_digit(c)).count() as f64 / (t.len().max(1)) as f64
}

fn table_runs(g: &PageGeo) -> Vec<(usize, usize)> {
    let l = &g.lines;
    let n = l.len();
    let ok: Vec<bool> = l.iter().map(|ln| ln.geo() && ln.furniture.is_none()).collect();
    let mut strong = vec![false; n];
    for k in 0..n {
        if !ok[k] {
            continue;
        }
        let ln = &l[k];
        let mate_gap = ln.mate_prev && l[k - 1].geo() && ln.fx0 - l[k - 1].lx1 > CELL_GAP * ln.size;
        let nxt = if k + 1 < n && ok[k + 1] { Some(&l[k + 1]) } else { None };
        let next_gap = nxt.is_some_and(|x| x.mate_prev && x.fx0 - ln.lx1 > CELL_GAP * ln.size);
        strong[k] = ln.math < 0.5
            && (ln.gaps.len() >= 2
                || (!ln.gaps.is_empty() && numeric_share(g.text(ln)) >= 0.3)
                || mate_gap
                || next_gap);
    }
    let overlaps =
        |a: &PLine, b: &PLine| a.gaps.iter().any(|ga| b.gaps.iter().any(|hb| pmin(ga.1, hb.1) > pmax(ga.0, hb.0)));
    for k in 0..n {
        let ln = &l[k];
        if ok[k] && !strong[k] && ln.gaps.len() == 1 && ln.math < 0.3 {
            let mut cnt = 0;
            for j in [k as isize - 2, k as isize - 1, k as isize + 1, k as isize + 2] {
                if j >= 0 && (j as usize) < n {
                    let o = &l[j as usize];
                    if ok[j as usize] && !o.gaps.is_empty() && o.math < 0.3 && o.angle == ln.angle && overlaps(ln, o) {
                        cnt += 1;
                    }
                }
            }
            if cnt >= 2 {
                strong[k] = true;
            }
        }
    }
    let mut weak = vec![false; n];
    for k in 0..n {
        if !ok[k] || strong[k] {
            continue;
        }
        let ln = &l[k];
        if ln.gaps.len() == 1 {
            weak[k] = true;
            continue;
        }
        for j in k.saturating_sub(3)..n.min(k + 4) {
            if strong[j] && l[j].angle == ln.angle && l[j].gaps.iter().any(|&(_, gx1)| (ln.fx0 - gx1).abs() <= 0.5 * ln.size)
            {
                weak[k] = true;
                break;
            }
        }
    }
    let cell: Vec<bool> = strong.iter().zip(weak.iter()).map(|(a, b)| *a || *b).collect();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut k = 0;
    while k < n {
        if !strong[k] {
            k += 1;
            continue;
        }
        let (mut j, ang) = (k, l[k].angle);
        let c = |i: usize| cell[i] && l[i].angle == ang;
        let filler = |i: usize| ok[i] && l[i].angle == ang && l[i].x1 - l[i].x0 <= 0.5 * g.w(&l[i]);
        while j + 1 < n && (c(j + 1) || (j + 2 < n && c(j + 2) && filler(j + 1))) {
            j += if c(j + 1) { 1 } else { 2 };
        }
        while j > k && !strong[j] {
            j -= 1;
        }
        let ls: Vec<&PLine> = (k..=j).map(|i| &l[i]).filter(|x| x.geo()).collect();
        let rows = 1 + ls.iter().skip(1).filter(|x| !x.mate_prev).count();
        let wide = (k..=j).filter(|&i| l[i].gaps.len() >= 3).count();
        let ns = (k..=j).filter(|&i| strong[i]).count();
        let nc = (k..=j).filter(|&i| cell[i]).count();
        if ns >= 2 && (nc >= 3 || wide >= 2) && rows >= 2 {
            runs.push((k, j));
        }
        k = j + 1;
    }
    let col = column_runs(g, &ok, &runs);
    runs.extend(col);
    runs.sort();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (mut a, mut b) in runs {
        let ang = l[a].angle;
        while a >= 1 && cell[a - 1] && l[a - 1].angle == ang {
            a -= 1;
        }
        while b + 1 < n && cell[b + 1] && l[b + 1].angle == ang {
            b += 1;
        }
        if let Some(last) = out.last_mut() {
            if a <= last.1 + 1 && l[last.0].angle == ang {
                last.1 = b.max(last.1);
                continue;
            }
        }
        out.push((a, b));
    }
    out
}

fn column_runs(g: &PageGeo, ok: &[bool], taken: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let l = &g.lines;
    let n = l.len();
    let mut used = vec![false; n];
    for &(a, b) in taken {
        for u in used.iter_mut().take(b + 1).skip(a) {
            *u = true;
        }
    }
    let narrow = |i: usize| {
        let ln = &l[i];
        ok[i] && !used[i] && ln.x1 - ln.x0 <= 0.25 * g.w(ln) && strip(g.text(ln)).len() <= 40
    };
    let mut out = Vec::new();
    let mut k = 0;
    while k < n {
        if !narrow(k) {
            k += 1;
            continue;
        }
        let mut j = k;
        while j + 1 < n && narrow(j + 1) && l[j + 1].angle == l[k].angle {
            j += 1;
        }
        let (mut up, mut down) = (0, 0);
        for i in k + 1..=j {
            let (a, b) = (&l[i - 1], &l[i]);
            if b.y0 < a.y0 - 1.5 * b.size {
                up += 1;
            } else if b.y0 > a.y1 - 0.5 * b.size {
                down += 1;
            }
        }
        if j - k + 1 >= 6 && up >= 2 && down >= 4 {
            out.push((k, j));
        }
        k = j + 1;
    }
    out
}

fn starts_list(text: &[char]) -> bool {
    let p = pats();
    p.bullet.is_match(text) || p.enumer.is_match(text)
}

fn lookalike(text: &[char]) -> Vec<char> {
    text.iter()
        .map(|&c| match tables::LOOKALIKE.iter().find(|e| e.0 == c as u32) {
            Some(e) => char::from_u32(e.1).unwrap_or(c),
            None => c,
        })
        .collect()
}

fn is_caption(text: &[char]) -> bool {
    pats().caption.is_match(&lookalike(text))
}

fn is_op(c: char) -> bool {
    tables::OPS.contains(&(c as u32))
}

fn is_formula(lines: &[&PLine], g: &PageGeo) -> bool {
    let t: Vec<char> = lines.iter().flat_map(|ln| g.text(ln).iter().copied()).filter(|&c| !is_space(c)).collect();
    let mut distinct = t.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if t.len() < 4
        || distinct.len() < 3
        || t.iter().filter(|&&c| is_digit(c) || c == ',' || c == '.').count() as f64 >= 0.6 * t.len() as f64
    {
        return false;
    }
    let mathc: f64 = lines.iter().fold(0.0, |s, ln| s + ln.math * ln.n as f64);
    let sym = t.iter().filter(|&&c| !is_alnum(c)).count();
    let ops = t.iter().any(|&c| is_op(c));
    let total: usize = lines.iter().map(|ln| ln.n).sum();
    mathc >= 0.5 * total.max(1) as f64 || (ops && sym as f64 >= 0.4 * t.len() as f64 && !is_alpha(t[0]))
}

struct EdgeCache(HashMap<(i32, u64), Option<f64>>);

/// Right edge of the running text sharing `a`'s left margin. Cached by (angle, round(x0)) with the first caller's
/// x0 (reference quirk, copied).
fn text_edge(g: &PageGeo, a: &PLine, cache: &mut EdgeCache) -> Result<Option<f64>> {
    let r = pyuni::round_int(a.x0).ok_or_else(|| anyhow!("OverflowError: round of a non-finite x0"))?;
    let key = (a.angle, (r + 0.0).to_bits());
    if let Some(v) = cache.0.get(&key) {
        return Ok(*v);
    }
    let mut ends: Vec<f64> = g
        .lines
        .iter()
        .filter(|l| l.geo() && l.angle == a.angle && strip(g.text(l)).len() >= 20 && (l.x0 - a.x0).abs() <= a.size)
        .map(|l| l.lx1)
        .collect();
    ends.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let v = if ends.len() >= 3 { Some(ends[(0.9 * (ends.len() - 1) as f64) as usize]) } else { None };
    cache.0.insert(key, v);
    Ok(v)
}

#[allow(clippy::too_many_arguments)]
fn cut_reason(
    gls: &[&PLine],
    a: &PLine,
    ln: &PLine,
    body: f64,
    leading: f64,
    g: &PageGeo,
    cache: &mut EdgeCache,
) -> Result<Option<&'static str>> {
    let p = pats();
    let s = pmax(pmax(ln.size, a.size), 1.0);
    let first = gls[0];
    if ln.angle != a.angle {
        return Ok(Some("angle"));
    }
    if ln.mate_prev && a.k + 1 == ln.k {
        return Ok(None);
    }
    if ln.y0 < a.y1 - 0.5 * pmin(ln.size, a.size) {
        return Ok(Some("not_below"));
    }
    let lead = leading * (if body != 0.0 { ln.size / body } else { 1.0 });
    if ln.fy1 - a.ly1 > LEADING * pmax(lead, 0.9 * ln.size) {
        return Ok(Some("leading"));
    }
    if (ln.size - a.size).abs() > SIZE_STEP * s {
        return Ok(Some("size"));
    }
    let in_caption = is_caption(g.text(first));
    if (a.bold >= 0.6) != (ln.bold >= 0.6) && !in_caption {
        return Ok(Some("bold"));
    }
    let right = pmax(gls[1..].iter().fold(gls[0].x1, |m, l| pmax(m, l.x1)), ln.x1);
    let lt = g.text(ln);
    if p.bullet.is_match(lt) {
        return Ok(Some("list"));
    }
    if p.enumer.is_match(lt) && a.lx1 < right - 1.5 * s {
        return Ok(Some("list"));
    }
    if is_caption(lt) && a.lx1 < right - 1.5 * s {
        return Ok(Some("caption"));
    }
    if ln.x0 > a.x1 || ln.x1 < a.x0 {
        return Ok(Some("no_overlap"));
    }
    let head_like = ln.size >= HEADING * body || (ln.bold >= 0.6 && a.bold >= 0.6);
    if in_caption {
        return Ok(None);
    }
    if !head_like && a.lx1 < right - 3.0 * s {
        return Ok(Some("short_line"));
    }
    if head_like {
        if p.secnum.is_match(lt) || (p.numbered.is_match(lt) && !p.numbered.is_match(g.text(first))) {
            return Ok(Some("heading_number"));
        }
        let edge = text_edge(g, a, cache)?;
        if let Some(e) = edge {
            if a.lx1 < e - 6.0 * s {
                return Ok(Some("heading_short"));
            }
        }
        return Ok(None);
    }
    let is_list = starts_list(g.text(gls[0]));
    if gls.len() == 1 {
        let full = first.lx1 >= right - 1.5 * s;
        if ln.fx0 > first.fx0 + 0.8 * s {
            if (full || is_list) && ln.fx0 - first.fx0 <= 6.0 * s {
                return Ok(None);
            }
            return Ok(Some("indent"));
        }
        if ln.fx0 < first.fx0 - 0.8 * s {
            if full && first.fx0 - ln.fx0 <= 6.0 * s {
                return Ok(None);
            }
            return Ok(Some("outdent"));
        }
        return Ok(None);
    }
    let left = gls[2..].iter().fold(gls[1].fx0, |m, l| pmin(m, l.fx0));
    if ln.fx0 > left + 0.8 * s && !is_list {
        return Ok(Some("indent"));
    }
    if ln.fx0 < left - 0.8 * s {
        return Ok(Some("outdent"));
    }
    Ok(None)
}

fn isolated(iso: (Option<f64>, Option<f64>), k: f64) -> bool {
    let sides: Vec<f64> = [iso.0, iso.1].into_iter().flatten().collect();
    if sides.is_empty() {
        return false;
    }
    max_first(&sides) > k && min_first(&sides) > 0.8
}

fn label_block(
    b: &mut Block,
    gls: &[&PLine],
    body: f64,
    ctx: &DocContext,
    g: &PageGeo,
    iso: (Option<f64>, Option<f64>),
) -> Result<()> {
    let p = pats();
    let text = strip(&g.plain[b.start..b.end]);
    if gls.is_empty() || text.is_empty() {
        return Ok(());
    }
    let size = median_low(gls.iter().map(|l| l.size).collect());
    let letters = text.iter().filter(|&&c| is_alpha(c)).count();
    let width = gls[1..].iter().fold(gls[0].x1, |m, l| pmax(m, l.x1)) - gls[1..].iter().fold(gls[0].x0, |m, l| pmin(m, l.x0));
    let nsum: usize = gls.iter().map(|l| l.n).sum();
    let bold = gls.iter().fold(0.0, |s, l| s + l.bold * l.n as f64) / nsum.max(1) as f64;
    if is_caption(text) {
        b.label = "caption";
        b.why = "caption_pattern";
        return Ok(());
    }
    let nonspace = text.iter().filter(|&&c| c != ' ').count();
    if is_formula(gls, g) && (letters as f64) < 0.5 * nonspace as f64 {
        b.label = "formula";
        b.why = "formula";
        return Ok(());
    }
    let nrows: usize = gls.iter().map(|l| l.rows).sum();
    let big = size >= HEADING * body;
    let strong_style = big || bold >= 0.9;
    let short = nrows <= (if strong_style { 4 } else { 3 })
        && text.len() <= 160
        && letters >= 2
        && width <= 0.85 * g.w(gls[0]);
    let last = text[text.len() - 1];
    let ends = last == ',' || last == ';' || (last == '.' && !text.ends_with(&['.', '.']) && !(bold >= 0.9 && nrows <= 4));
    let sec = p.secnum.is_match(text);
    let num_m = p.numbered.is_match(text);
    let numbered = (sec || num_m) && nrows == 1 && text.len() <= 120 && !ends && last != ':';
    let emph_min = gls[1..].iter().fold(gls[0].emph, |m, l| pmin(m, l.emph));
    if short && !ends && (big || (bold >= 0.6 && (size - body).abs() <= 0.15 * body) || (num_m && bold >= 0.6)) {
        b.label = "section_header";
        b.why = if big { "big" } else { "bold" };
    } else if numbered && size >= 0.95 * body && isolated(iso, 1.4) && (sec || text.len() <= 80) {
        b.label = "section_header";
        b.why = "numbered_isolated";
    } else if nrows == 1
        && text.len() <= 80
        && letters >= 3
        && !ends
        && last != ':'
        && emph_min >= 0.8
        && size >= 0.95 * body
        && (is_upper(text[0]) || is_digit(text[0]))
        && isolated(iso, 1.2)
    {
        b.label = "section_header";
        b.why = "emphasis_isolated";
    }
    if b.label == "section_header" {
        let k = size_key(size)?;
        b.level = Some(match ctx.heading_sizes.iter().position(|&h| h == k) {
            Some(i) => i as i64 + 1,
            None => 3.min(ctx.heading_sizes.len() as i64 + 1),
        });
        return Ok(());
    }
    if starts_list(text) {
        b.label = "list_item";
        b.why = "list_marker";
        return Ok(());
    }
    if gls.len() >= 2 && gls[0].fx0 < gls[2..].iter().fold(gls[1].fx0, |m, l| pmin(m, l.fx0)) - 0.8 * size {
        b.label = "list_item";
        b.why = "hanging_indent";
    }
    Ok(())
}

fn picture_runs(blocks: &mut [Block], glines: &[Vec<&PLine>], body: f64, g: &PageGeo) {
    let p = pats();
    let figure_like = ["paragraph", "section_header", "list_item", "formula", "title"];
    for i in 0..blocks.len() {
        if blocks[i].label != "caption" || !p.figword.is_match(&lookalike(&g.plain[blocks[i].start..blocks[i].end])) {
            continue;
        }
        for step in [-1isize, 1] {
            let mut run: Vec<usize> = Vec::new();
            let mut j = i as isize + step;
            while j >= 0 && (j as usize) < blocks.len() {
                let ju = j as usize;
                let (o, gl) = (&blocks[ju], &glines[ju]);
                if !figure_like.contains(&o.label) || gl.is_empty() {
                    break;
                }
                let rows: usize = gl.iter().map(|l| l.rows).sum();
                let txt = strip(&g.plain[o.start..o.end]);
                if !(rows <= 2 && txt.len() <= 100) && numeric_share(txt) < 0.5 {
                    break;
                }
                if p.secnum.is_match(txt) {
                    break;
                }
                run.push(ju);
                j += step;
            }
            let lines: Vec<&PLine> = run.iter().flat_map(|&j| glines[j].iter().copied()).collect();
            let figure_ish = |j: usize| {
                let gl = &glines[j];
                median_low(gl.iter().map(|l| l.size).collect()) <= 0.95 * body
                    || numeric_share(&g.plain[blocks[j].start..blocks[j].end]) >= 0.5
                    || matches!(blocks[j].cut, Some("not_below") | Some("no_overlap") | Some("angle"))
                    || blocks[j].rotated != 0
            };
            if lines.len() >= 2
                && (run.len() >= 3 || lines.iter().any(|l| numeric_share(g.text(l)) >= 0.5))
                && 2 * run.iter().filter(|&&j| figure_ish(j)).count() >= run.len()
            {
                for &j in &run {
                    blocks[j].label = "picture";
                    blocks[j].level = None;
                    blocks[j].why = "figure_text";
                }
            }
        }
    }
}

fn list_neighbours(blocks: &mut [Block], glines: &[Vec<&PLine>]) {
    for i in 0..blocks.len() {
        let gl = &glines[i];
        if blocks[i].label != "paragraph" || gl.is_empty() || gl.iter().map(|l| l.rows).sum::<usize>() != 1 {
            continue;
        }
        for j in [i as isize - 1, i as isize + 1] {
            if j >= 0 && (j as usize) < blocks.len() && blocks[j as usize].why == "hanging_indent" && !glines[j as usize].is_empty() {
                let o = glines[j as usize][0];
                if (o.fx0 - gl[0].fx0).abs() <= 0.5 * o.size && (o.size - gl[0].size).abs() <= SIZE_STEP * o.size {
                    blocks[i].label = "list_item";
                    blocks[i].why = "list_neighbour";
                    break;
                }
            }
        }
    }
}

fn footnote_area(blocks: &mut [Block], body: f64, g: &PageGeo) {
    let p = pats();
    let l = &g.lines;
    let body_idx: Vec<usize> = (0..blocks.len()).filter(|&i| !matches!(blocks[i].label, "page_header" | "page_footer")).collect();
    let mut rf = body;
    for b in blocks.iter() {
        let gl: Vec<&PLine> = (b.first..=b.last).map(|j| &l[j]).filter(|x| x.geo()).collect();
        if b.label == "paragraph" && !gl.is_empty() && gl.iter().map(|x| x.rows).sum::<usize>() >= 3 {
            rf = pmax(rf, median_low(gl.iter().map(|x| x.size).collect()));
        }
    }
    let body = rf;
    let mut run: Vec<usize> = Vec::new();
    for &i in body_idx.iter().rev() {
        let b = &blocks[i];
        let gl: Vec<&PLine> = (b.first..=b.last).map(|j| &l[j]).filter(|x| x.geo()).collect();
        if gl.is_empty() || matches!(b.label, "table" | "section_header") {
            break;
        }
        let size = median_low(gl.iter().map(|x| x.size).collect());
        if size > FOOT_SIZE * body || b.bbox.is_none_or(|bb| bb[1] < 0.35 * g.h(gl[0])) {
            break;
        }
        run.push(i);
    }
    run.reverse();
    let text_of = |i: usize| &g.plain[blocks[i].start..blocks[i].end];
    let marked = |k: usize| {
        let mut t: Vec<char> = text_of(run[k]).to_vec();
        if p.mark_alone.is_fullmatch(&t) && k + 1 < run.len() {
            t.push(' ');
            t.extend_from_slice(text_of(run[k + 1]));
        }
        p.footmark.is_match(&t) && numeric_share(&t) < 0.5
    };
    let low = match run.last() {
        Some(&last) => blocks[last].bbox.unwrap()[3] >= 0.6 * g.h(&l[blocks[last].first]),
        None => false,
    };
    if !run.is_empty() && low && (0..run.len()).any(marked) {
        for &i in &run {
            if matches!(blocks[i].label, "paragraph" | "list_item") {
                blocks[i].label = "footnote";
                blocks[i].why = "footnote_area";
            }
        }
    }
}

fn title_rule(blocks: &mut [Block], body: f64, page_no: i64, g: &PageGeo) {
    if page_no > 2 {
        return;
    }
    let l = &g.lines;
    let bsize = |b: &Block| median_low((b.first..=b.last).filter(|&j| l[j].geo()).map(|j| l[j].size).collect());
    let mut best: Option<(usize, f64)> = None;
    for (i, b) in blocks.iter().enumerate() {
        if !matches!(b.label, "section_header" | "paragraph") || b.bbox.is_none() {
            continue;
        }
        let s = bsize(b);
        best = match best {
            None => Some((i, s)),
            Some((bi, bs)) => {
                let (ka, kb) = ((s, -(b.first as i64)), (bs, -(blocks[bi].first as i64)));
                if ka.partial_cmp(&kb) == Some(std::cmp::Ordering::Greater) {
                    Some((i, s))
                } else {
                    Some((bi, bs))
                }
            }
        };
    }
    if let Some((i, s)) = best {
        if s >= 1.5 * body && blocks[i].label == "section_header" {
            blocks[i].label = "title";
            blocks[i].level = None;
            blocks[i].why = "title_first_pages";
        }
    }
}

/// Blocks of one page over the plain text (the order is the plain order; nothing is moved or dropped).
pub fn label_page(plain: &str, t: &CharTable, ctx: &DocContext, page_no: i64) -> Result<PageLabels> {
    let pc: Vec<char> = plain.chars().collect();
    let mut g = page_geo(&pc, t)?;
    let mut warnings = Vec::new();
    if g.unmatched > 0 {
        warnings.push(Warning { kind: "unaligned_chars", key: "count", value: g.unmatched as i64 });
    }
    let rel = (g.major as i64 - g.disp).rem_euclid(360);
    if rel != 0 {
        warnings.push(Warning { kind: "rotated_content", key: "degrees_cw", value: rel });
    }
    let rot_lines = g.lines.iter().filter(|ln| ln.geo() && ln.angle as i64 != g.disp).count();
    if rot_lines > 0 {
        warnings.push(Warning { kind: "rotated_lines", key: "count", value: rot_lines as i64 });
    }
    if t.rotation != 0 {
        warnings.push(Warning { kind: "page_rotation", key: "degrees", value: t.rotation as i64 });
    }
    let lines_out: Vec<(usize, usize)> = g.lines.iter().map(|l| (l.start, l.end)).collect();
    if pc.is_empty() {
        return Ok(PageLabels { blocks: vec![], warnings, rotation: rel, lines: lines_out });
    }
    mark_furniture(&mut g, ctx);
    let mut body = ctx.body;
    let mut sizes = FMap::default();
    for ln in g.lines.iter().filter(|l| l.geo() && l.furniture.is_none()) {
        sizes.add(size_key(ln.size)?, ln.n);
    }
    if let Some(own) = sizes.mode() {
        let total = sizes.total() as f64;
        let at_body: usize = sizes.0.iter().filter(|e| (e.0 - body).abs() <= 0.15 * body).map(|e| e.1).sum();
        if (own - body).abs() > 0.15 * body && sizes.get(own) as f64 >= 0.4 * total && (at_body as f64) < 0.15 * total {
            body = own;
        }
    }
    let mut steps = Vec::new();
    {
        let gl: Vec<&PLine> = g.lines.iter().filter(|l| l.geo() && l.furniture.is_none()).collect();
        for w in gl.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a.angle != b.angle {
                continue;
            }
            let d = b.fy1 - a.ly1;
            if 0.5 * a.size < d && d < 3.0 * a.size && (a.size - body).abs() <= 0.15 * body {
                steps.push(d);
            }
        }
    }
    let leading = if steps.is_empty() { 1.2 * body } else { median_low(steps) };
    let mut wc: Vec<(i32, usize)> = Vec::new();
    for ln in g.lines.iter().filter(|l| l.geo() && l.furniture.is_none()) {
        for &(w, c) in &ln.wts {
            if w > 0 {
                match wc.iter_mut().find(|e| e.0 == w) {
                    Some(e) => e.1 += c,
                    None => wc.push((w, c)),
                }
            }
        }
    }
    let mut wmode = 0i32;
    if let Some(first) = wc.first() {
        let mut best = *first;
        for e in &wc[1..] {
            if (e.1, -e.0) > (best.1, -best.0) {
                best = *e;
            }
        }
        wmode = best.0;
    }
    if wmode != 0 {
        for ln in g.lines.iter_mut().filter(|l| l.geo()) {
            let heavy = ln.wts.iter().filter(|e| e.0 >= wmode + 40).map(|e| e.1).sum::<usize>() as f64 / ln.n as f64;
            ln.emph = pmax(ln.emph, heavy);
        }
    }
    let runs = table_runs(&g);
    let nl = g.lines.len();
    let mut in_table: Vec<Option<(usize, usize)>> = vec![None; nl];
    for &(a, b) in &runs {
        for slot in in_table.iter_mut().take(b + 1).skip(a) {
            *slot = Some((a, b));
        }
    }
    // -- cut blocks ------------------------------------------------------------------------------------------------
    let g = g;
    let l = &g.lines;
    let mut cache = EdgeCache(HashMap::new());
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur: Option<Block> = None;
    let geo_lines = |b: &Block| -> Vec<&PLine> { (b.first..=b.last).map(|i| &l[i]).filter(|x| x.geo()).collect() };
    let mut k = 0;
    while k < nl {
        let ln = &l[k];
        if let Some((a, b)) = in_table[k] {
            if let Some(c) = cur.take() {
                blocks.push(c);
            }
            let tl: Vec<&PLine> = (a..=b).map(|i| &l[i]).filter(|x| x.geo()).collect();
            let msum = tl.iter().fold(0.0, |s, x| s + x.math * x.n as f64);
            let nsum: usize = tl.iter().map(|x| x.n).sum();
            let mshare = msum / nsum.max(1) as f64;
            if mshare >= 0.3 {
                blocks.push(Block::new(a, b, "formula", "math_run"));
            } else {
                blocks.push(Block::new(a, b, "table", "table_run"));
            }
            k = b + 1;
            continue;
        }
        if let Some(band) = ln.furniture {
            if let Some(c) = cur.take() {
                blocks.push(c);
            }
            let lab = if band == Band::Top { "page_header" } else { "page_footer" };
            match blocks.last_mut() {
                Some(prev) if prev.label == lab && prev.last + 1 == k && ln.mate_prev => prev.last = k,
                _ => blocks.push(Block::new(k, k, lab, ln.furn_reason.unwrap_or(""))),
            }
            k += 1;
            continue;
        }
        let Some(c) = cur.as_mut() else {
            cur = Some(Block::new(k, k, "paragraph", ""));
            k += 1;
            continue;
        };
        let gls = geo_lines(c);
        let a = gls.last().copied();
        let Some(a) = a.filter(|_| ln.geo()) else {
            c.last = k;
            k += 1;
            continue;
        };
        match cut_reason(&gls, a, ln, body, leading, &g, &mut cache)? {
            Some(why) => {
                blocks.push(cur.take().unwrap());
                let mut nb = Block::new(k, k, "paragraph", "");
                nb.cut = Some(why);
                cur = Some(nb);
            }
            None => c.last = k,
        }
        k += 1;
    }
    if let Some(c) = cur.take() {
        blocks.push(c);
    }
    // -- label -----------------------------------------------------------------------------------------------------
    let glines: Vec<Vec<&PLine>> = blocks.iter().map(geo_lines).collect();
    for i in 0..blocks.len() {
        blocks[i].start = l[blocks[i].first].start;
        blocks[i].end = l[blocks[i].last].end;
        let gls = &glines[i];
        if !gls.is_empty() {
            let x0 = gls[1..].iter().fold(gls[0].x0, |m, x| pmin(m, x.x0));
            let y0 = gls[1..].iter().fold(gls[0].y0, |m, x| pmin(m, x.y0));
            let x1 = gls[1..].iter().fold(gls[0].x1, |m, x| pmax(m, x.x1));
            let y1 = gls[1..].iter().fold(gls[0].y1, |m, x| pmax(m, x.y1));
            blocks[i].bbox = Some([round2(x0), round2(y0), round2(x1), round2(y1)]);
            blocks[i].rotated = (gls[0].angle as i64 - g.disp).rem_euclid(360);
        }
        if blocks[i].label != "paragraph" || gls.is_empty() {
            continue;
        }
        let mut iso: [Option<f64>; 2] = [None, None];
        for (side, j) in [i as isize - 1, i as isize + 1].into_iter().enumerate() {
            let o: &[&PLine] = if j >= 0
                && (j as usize) < blocks.len()
                && !matches!(blocks[j as usize].label, "page_header" | "page_footer")
            {
                &glines[j as usize]
            } else {
                &[]
            };
            if o.is_empty() || o[0].angle != gls[0].angle {
                continue;
            }
            let d = if (j as usize) < i { gls[0].fy1 - o[o.len() - 1].ly1 } else { o[0].fy1 - gls[gls.len() - 1].ly1 };
            if d > 0.0 {
                if leading == 0.0 {
                    bail!("ZeroDivisionError: page leading is 0");
                }
                iso[side] = Some(d / leading);
            }
        }
        let mut b = blocks[i].clone();
        label_block(&mut b, gls, body, ctx, &g, (iso[0], iso[1]))?;
        blocks[i] = b;
    }
    list_neighbours(&mut blocks, &glines);
    picture_runs(&mut blocks, &glines, body, &g);
    footnote_area(&mut blocks, body, &g);
    title_rule(&mut blocks, body, page_no, &g);
    // -- the invariant ------------------------------------------------------------------------------------------
    if blocks.is_empty() || blocks[0].first != 0 || blocks[blocks.len() - 1].last != nl - 1 {
        bail!("AssertionError: blocks do not cover the page");
    }
    let mut rebuilt: Vec<char> = Vec::with_capacity(pc.len());
    for i in 0..blocks.len() {
        if i + 1 < blocks.len() && blocks[i].last + 1 != blocks[i + 1].first {
            bail!("AssertionError: blocks are not contiguous");
        }
        rebuilt.extend_from_slice(&pc[blocks[i].start..blocks[i].end]);
        if i + 1 < blocks.len() {
            let sep = &pc[l[blocks[i].last].end..l[blocks[i + 1].first].start];
            if !str_isspace(sep) {
                bail!("AssertionError: a separator is not whitespace");
            }
            rebuilt.extend_from_slice(sep);
        }
    }
    if rebuilt != pc {
        bail!("AssertionError: v2 must not move, add or drop text");
    }
    Ok(PageLabels { blocks, warnings, rotation: rel, lines: lines_out })
}

/// The reference's `page_markdown`: title `# `, section_header `#` x min(6, level + 1), blocks joined by a blank line.
pub fn page_markdown(plain: &str, blocks: &[Block]) -> String {
    let pc: Vec<char> = plain.chars().collect();
    let mut parts: Vec<String> = Vec::with_capacity(blocks.len());
    for b in blocks {
        let text: String = pc[b.start..b.end].iter().collect();
        match b.label {
            "title" => parts.push(format!("# {text}")),
            "section_header" => parts.push(format!("{} {text}", "#".repeat(6.min(b.level.unwrap_or(1) + 1) as usize))),
            _ => parts.push(text),
        }
    }
    parts.join("\n\n")
}
