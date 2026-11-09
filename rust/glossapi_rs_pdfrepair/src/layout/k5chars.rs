//! The `.k5chars` v1 char table: one compact little-endian buffer per page (struct of arrays), the input of layout v2.
//! Layout as in the K5c reference reader (`tests/layout_v2_ref/k5chars.py`):
//!   header  magic b"K5CHARS\0" | u16 version (=1) | u16 rotation | u32 n | u32 nfonts | u32 page | f64 page_w
//!           | f64 page_h | 32 bytes source sha256
//!   fonts   nfonts x (u16 byte length, UTF-8 name)
//!   arrays  ch u32[n] | bbox f64[4n] | loose f64[4n] | size f32[n] | weight i16[n] | angle f32[n] | font u32[n]
//!           | flags u8[n]
//! Boxes are (left, bottom, right, top) in PDF user space as PDFium returns them; `ch` holds PDFium's UTF-16 code
//! units (surrogates are joined by the layout).
use anyhow::{bail, Result};

pub const MAGIC: &[u8; 8] = b"K5CHARS\0";
pub const HEAD: usize = 8 + 2 + 2 + 4 + 4 + 4 + 8 + 8 + 32;
/// Hard ceiling on chars per page (PDFium pages beyond this are refused, as the K5c reader does).
pub const MAX_CHARS: usize = 2_000_000;

#[derive(Clone, Debug, Default)]
pub struct CharTable {
    pub page_w: f64,
    pub page_h: f64,
    pub rotation: u16,
    pub page: u32,
    pub fonts: Vec<String>,
    pub ch: Vec<u32>,
    pub bbox: Vec<[f64; 4]>,
    pub loose: Vec<[f64; 4]>,
    pub size: Vec<f32>,
    pub weight: Vec<i16>,
    pub angle: Vec<f32>,
    pub font: Vec<u32>,
    pub flags: Vec<u8>,
}

struct Rd<'a> {
    d: &'a [u8],
    o: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.o.checked_add(n).filter(|&e| e <= self.d.len());
        match end {
            Some(e) => {
                let s = &self.d[self.o..e];
                self.o = e;
                Ok(s)
            }
            None => bail!("k5chars: truncated table"),
        }
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

pub fn parse(data: &[u8]) -> Result<CharTable> {
    let mut r = Rd { d: data, o: 0 };
    if r.take(8)? != MAGIC {
        bail!("k5chars: bad magic");
    }
    if r.u16()? != 1 {
        bail!("k5chars: unsupported version");
    }
    let rotation = r.u16()?;
    let n = r.u32()? as usize;
    let nf = r.u32()? as usize;
    let page = r.u32()?;
    let page_w = r.f64()?;
    let page_h = r.f64()?;
    r.take(32)?;
    if n > MAX_CHARS {
        bail!("k5chars: {n} chars is over the {MAX_CHARS} limit");
    }
    // every font needs >= 2 bytes and every char 83 bytes: refuse impossible counts before allocating
    if nf.saturating_mul(2) > data.len() || n.saturating_mul(83) > data.len() {
        bail!("k5chars: truncated table");
    }
    let mut fonts = Vec::with_capacity(nf);
    for _ in 0..nf {
        let ln = r.u16()? as usize;
        let b = r.take(ln)?;
        fonts.push(String::from_utf8(b.to_vec()).map_err(|_| anyhow::anyhow!("k5chars: font name is not UTF-8"))?);
    }
    let chb = r.take(4 * n)?;
    let bb = r.take(32 * n)?;
    let lo = r.take(32 * n)?;
    let sz = r.take(4 * n)?;
    let wt = r.take(2 * n)?;
    let an = r.take(4 * n)?;
    let fo = r.take(4 * n)?;
    let fl = r.take(n)?;
    if r.o != data.len() {
        bail!("k5chars: {} trailing bytes", data.len() - r.o);
    }
    let f64s = |b: &[u8], i: usize| f64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap());
    let mut t = CharTable {
        page_w,
        page_h,
        rotation,
        page,
        fonts,
        ch: Vec::with_capacity(n),
        bbox: Vec::with_capacity(n),
        loose: Vec::with_capacity(n),
        size: Vec::with_capacity(n),
        weight: Vec::with_capacity(n),
        angle: Vec::with_capacity(n),
        font: Vec::with_capacity(n),
        flags: fl.to_vec(),
    };
    for i in 0..n {
        let c = u32::from_le_bytes(chb[4 * i..4 * i + 4].try_into().unwrap());
        if c > 0x10FFFF {
            // Python chr() refuses it, so the reference cannot load the table either
            bail!("k5chars: code point 0x{c:X} out of range");
        }
        t.ch.push(c);
        t.bbox.push([f64s(bb, 4 * i), f64s(bb, 4 * i + 1), f64s(bb, 4 * i + 2), f64s(bb, 4 * i + 3)]);
        t.loose.push([f64s(lo, 4 * i), f64s(lo, 4 * i + 1), f64s(lo, 4 * i + 2), f64s(lo, 4 * i + 3)]);
        t.size.push(f32::from_le_bytes(sz[4 * i..4 * i + 4].try_into().unwrap()));
        t.weight.push(i16::from_le_bytes(wt[2 * i..2 * i + 2].try_into().unwrap()));
        t.angle.push(f32::from_le_bytes(an[4 * i..4 * i + 4].try_into().unwrap()));
        t.font.push(u32::from_le_bytes(fo[4 * i..4 * i + 4].try_into().unwrap()));
    }
    Ok(t)
}

/// Serialise a table (the inverse of `parse`; used by the native PDFium reader and tests).
pub fn dump(t: &CharTable, sha: &[u8; 32]) -> Vec<u8> {
    let n = t.ch.len();
    let mut out = Vec::with_capacity(HEAD + 83 * n + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&t.rotation.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());
    out.extend_from_slice(&(t.fonts.len() as u32).to_le_bytes());
    out.extend_from_slice(&t.page.to_le_bytes());
    out.extend_from_slice(&t.page_w.to_le_bytes());
    out.extend_from_slice(&t.page_h.to_le_bytes());
    out.extend_from_slice(sha);
    for f in &t.fonts {
        out.extend_from_slice(&(f.len() as u16).to_le_bytes());
        out.extend_from_slice(f.as_bytes());
    }
    for &c in &t.ch {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for b in &t.bbox {
        for v in b {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for b in &t.loose {
        for v in b {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for v in &t.size {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &t.weight {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &t.angle {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &t.font {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&t.flags);
    out
}
