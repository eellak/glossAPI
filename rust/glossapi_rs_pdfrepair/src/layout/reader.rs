//! Native char reader: the `.k5chars` table of a PDFium text page, read from Rust instead of per-char Python ctypes
//! calls (K5-A1; the Python crossing cost 3.1 us/char, 5.2 ms/page median, against 0.28 ms/page for the Rust labels).
//!
//! No new dependency and no second PDFium: the 11 `FPDFText_*` functions are those of the PDFium library pypdfium2
//! has already loaded. They are resolved and self-tested by the Python binding (`layout_reader_py.rs`), which is the
//! only caller; nothing public takes addresses or raw handles (K5-A1r). The fields copy the K5c reference reader (`freeze_chars.read_chars`) exactly, including its
//! state carried across chars: the box, loose box, flags and the 512-byte font-name buffer are reused, so a failed
//! PDFium call keeps the previous char's values as the reference does; a font name longer than the buffer reads the
//! stale buffer; names are decoded as UTF-8 with replacement; `ch` > U+10FFFF becomes U+FFFD.
//!
//! Safety: the addresses must be those functions of a loaded PDFium and `textpage` a live `FPDF_TEXTPAGE` of it,
//! used under the GIL (PDFium is not thread-safe); `layout_reader_py.rs` establishes both.
// Only the Python binding (feature `python`, not test builds) reads through PDFium.
#![cfg_attr(not(all(feature = "python", not(test))), allow(dead_code))]
use super::k5chars::{CharTable, MAX_CHARS};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::ffi::{c_int, c_uint, c_ulong, c_void};

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct FsRectF {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

type FCount = unsafe extern "C" fn(*mut c_void) -> c_int;
type FUnicode = unsafe extern "C" fn(*mut c_void, c_int) -> c_uint;
type FCharBox = unsafe extern "C" fn(*mut c_void, c_int, *mut f64, *mut f64, *mut f64, *mut f64) -> c_int;
type FLooseBox = unsafe extern "C" fn(*mut c_void, c_int, *mut FsRectF) -> c_int;
type FDouble = unsafe extern "C" fn(*mut c_void, c_int) -> f64;
type FInt = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type FFontInfo = unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, c_ulong, *mut c_int) -> c_ulong;
type FFloat = unsafe extern "C" fn(*mut c_void, c_int) -> f32;
type FMediaBox = unsafe extern "C" fn(*mut c_void, *mut f32, *mut f32, *mut f32, *mut f32) -> c_int;
type FRotation = unsafe extern "C" fn(*mut c_void) -> c_int;

/// The function order of `PDFIUM_FUNCTIONS`.
pub const PDFIUM_FUNCTIONS: [&str; 13] = [
    "FPDFText_CountChars",
    "FPDFText_GetUnicode",
    "FPDFText_GetCharBox",
    "FPDFText_GetLooseCharBox",
    "FPDFText_GetFontSize",
    "FPDFText_GetFontWeight",
    "FPDFText_GetFontInfo",
    "FPDFText_GetCharAngle",
    "FPDFText_IsGenerated",
    "FPDFText_HasUnicodeMapError",
    "FPDFText_GetTextRenderMode",
    "FPDFPage_GetMediaBox",
    "FPDFPage_GetRotation",
];

pub struct PdfiumText {
    count: FCount,
    unicode: FUnicode,
    char_box: FCharBox,
    loose_box: FLooseBox,
    font_size: FDouble,
    font_weight: FInt,
    font_info: FFontInfo,
    angle: FFloat,
    generated: FInt,
    map_error: FInt,
    render_mode: FInt,
    media_box: FMediaBox,
    page_rotation: FRotation,
}

const GENERATED: u8 = 1;
const SOFT_HYPHEN: u8 = 2;
const MAP_ERROR: u8 = 4;
const ITALIC: u8 = 8;
const SYMBOLIC: u8 = 16;
const INVISIBLE: u8 = 32;
const FLAG_SYMBOLIC: c_int = 4;
const FLAG_ITALIC: c_int = 64;
const NAME_BUF: usize = 512;

impl PdfiumText {
    /// # Safety
    /// `addrs` must be the addresses of the `PDFIUM_FUNCTIONS` of one loaded PDFium library, in that order.
    pub(crate) unsafe fn from_addresses(addrs: &[usize]) -> Result<Self> {
        if addrs.len() != PDFIUM_FUNCTIONS.len() {
            bail!("need {} PDFium function addresses, got {}", PDFIUM_FUNCTIONS.len(), addrs.len());
        }
        if let Some(i) = addrs.iter().position(|&a| a == 0) {
            bail!("{} address is null", PDFIUM_FUNCTIONS[i]);
        }
        unsafe {
            Ok(PdfiumText {
                count: std::mem::transmute::<usize, FCount>(addrs[0]),
                unicode: std::mem::transmute::<usize, FUnicode>(addrs[1]),
                char_box: std::mem::transmute::<usize, FCharBox>(addrs[2]),
                loose_box: std::mem::transmute::<usize, FLooseBox>(addrs[3]),
                font_size: std::mem::transmute::<usize, FDouble>(addrs[4]),
                font_weight: std::mem::transmute::<usize, FInt>(addrs[5]),
                font_info: std::mem::transmute::<usize, FFontInfo>(addrs[6]),
                angle: std::mem::transmute::<usize, FFloat>(addrs[7]),
                generated: std::mem::transmute::<usize, FInt>(addrs[8]),
                map_error: std::mem::transmute::<usize, FInt>(addrs[9]),
                render_mode: std::mem::transmute::<usize, FInt>(addrs[10]),
                media_box: std::mem::transmute::<usize, FMediaBox>(addrs[11]),
                page_rotation: std::mem::transmute::<usize, FRotation>(addrs[12]),
            })
        }
    }

    /// Page width, height and rotation as the K5c freeze computed them from pypdfium2: `get_mediabox()` (MediaBox,
    /// or ANSI A (0, 0, 612, 792) when PDFium reports none) -> (x1 - x0, y1 - y0); `get_rotation()` = quarter turns
    /// x 90 (anything else is an error, as pypdfium2's lookup raises).
    ///
    /// # Safety
    /// `page` must be a live `FPDF_PAGE` of the library the functions belong to.
    pub(crate) unsafe fn page_geometry(&self, page: *mut c_void) -> Result<(f64, f64, u16)> {
        if page.is_null() {
            bail!("page handle is null");
        }
        let (mut l, mut b, mut r, mut t) = (0f32, 0f32, 0f32, 0f32);
        let ok = unsafe { (self.media_box)(page, &mut l, &mut b, &mut r, &mut t) };
        let (x0, y0, x1, y1) = if ok != 0 { (l as f64, b as f64, r as f64, t as f64) } else { (0.0, 0.0, 612.0, 792.0) };
        let q = unsafe { (self.page_rotation)(page) };
        if !(0..=3).contains(&q) {
            bail!("page rotation code {q}");
        }
        Ok((x1 - x0, y1 - y0, (q * 90) as u16))
    }

    /// The char table of one text page, as `freeze_chars.read_chars` builds it.
    ///
    /// # Safety
    /// `textpage` must be a live `FPDF_TEXTPAGE` of the library the functions belong to.
    pub(crate) unsafe fn read(&self, textpage: *mut c_void, page_w: f64, page_h: f64, rotation: u16, page: u32) -> Result<CharTable> {
        if textpage.is_null() {
            bail!("text page handle is null");
        }
        let n = unsafe { (self.count)(textpage) };
        if n < 0 || n as usize > MAX_CHARS {
            bail!("CountChars {n} out of range");
        }
        let n = n as usize;
        let mut t = CharTable { page_w, page_h, rotation, page, ..Default::default() };
        t.ch.reserve(n);
        t.bbox.reserve(n);
        t.loose.reserve(n);
        let mut fonts: HashMap<String, u32> = HashMap::new();
        let (mut l, mut r, mut b, mut tt) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut rect = FsRectF::default();
        let mut fl: c_int = 0;
        let mut buf = [0u8; NAME_BUF];
        for i in 0..n as c_int {
            let (u, size, weight, ln, angle, gen, maperr, mode) = unsafe {
                let u = (self.unicode)(textpage, i);
                (self.char_box)(textpage, i, &mut l, &mut r, &mut b, &mut tt);
                (self.loose_box)(textpage, i, &mut rect);
                let size = (self.font_size)(textpage, i);
                let weight = (self.font_weight)(textpage, i);
                let ln = (self.font_info)(textpage, i, buf.as_mut_ptr().cast(), NAME_BUF as c_ulong, &mut fl);
                let angle = (self.angle)(textpage, i);
                let gen = (self.generated)(textpage, i);
                let maperr = (self.map_error)(textpage, i);
                let mode = (self.render_mode)(textpage, i);
                (u, size, weight, ln, angle, gen, maperr, mode)
            };
            let name = if ln > 0 {
                let k = ((ln - 1) as usize).min(NAME_BUF);
                String::from_utf8_lossy(&buf[..k]).into_owned()
            } else {
                String::new()
            };
            let mut f = 0u8;
            if gen == 1 {
                f |= GENERATED;
            }
            if u == 0x02 {
                f |= SOFT_HYPHEN;
            }
            if maperr == 1 {
                f |= MAP_ERROR;
            }
            if ln > 0 && fl & FLAG_ITALIC != 0 {
                f |= ITALIC;
            }
            if ln > 0 && fl & FLAG_SYMBOLIC != 0 {
                f |= SYMBOLIC;
            }
            if mode == 3 {
                f |= INVISIBLE;
            }
            t.ch.push(if u > 0x10FFFF { 0xFFFD } else { u });
            t.bbox.push([l, b, r, tt]);
            t.loose.push([rect.left as f64, rect.bottom as f64, rect.right as f64, rect.top as f64]);
            t.size.push(size as f32);
            t.weight.push(weight.clamp(-32768, 32767) as i16);
            t.angle.push(angle);
            let next = fonts.len() as u32;
            let fid = *fonts.entry(name.clone()).or_insert_with(|| {
                t.fonts.push(name);
                next
            });
            t.font.push(fid);
            t.flags.push(f);
        }
        Ok(t)
    }
}
