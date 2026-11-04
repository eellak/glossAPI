//! Native char reader with Rust-owned PDFium handles (K5-A1 reader; K5-A1r2 ownership, CP-K5a cycle 2 N1/N2).
//!
//! Rust opens the document (`FPDF_LoadDocument` for a path, `FPDF_LoadMemDocument64` for bytes), each page
//! (`FPDF_LoadPage`) and its text page (`FPDFText_LoadPage`) itself. It reads and closes them through `Drop` guards,
//! which also run on errors. No handle ever comes from Python or from a Python object. The functions are those of
//! the PDFium library pypdfium2 has loaded; the Python binding (`layout_reader_py.rs`) resolves them, checks their
//! identity and self-tests them, and serialises every read behind one process-wide lock.
//!
//! Per page it returns:
//! - the `.k5chars` table. Its fields copy the K5c reader (`freeze_chars.read_chars`) exactly, including the state it
//!   carries across chars (box, loose box, flags and the 512-byte font-name buffer are reused after a failed call;
//!   names are UTF-8 decoded with replacement; `ch` > U+10FFFF becomes U+FFFD). Page size and rotation are as the K5c
//!   freeze took them from pypdfium2.
//! - the page text exactly as production gets it. pypdfium2 4.30's `get_text_range()` with default arguments is
//!   `get_text_bounded()`: `FPDF_GetPageBoundingBox`, then `FPDFText_GetBoundedText(left, top, right, bottom)`,
//!   decoded from UTF-16LE with `errors="ignore"` (lone surrogates dropped).
// Only the Python binding (feature `python`, not test builds) reads through PDFium.
#![cfg_attr(not(all(feature = "python", not(test))), allow(dead_code))]
use super::k5chars::{CharTable, MAX_CHARS};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_ulong, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};

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
type FLoadDocument = unsafe extern "C" fn(*const c_char, *const c_char) -> *mut c_void;
type FLoadMemDocument64 = unsafe extern "C" fn(*const c_void, usize, *const c_char) -> *mut c_void;
type FClose = unsafe extern "C" fn(*mut c_void);
type FPageCount = unsafe extern "C" fn(*mut c_void) -> c_int;
type FLoadPage = unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_void;
type FLoadTextPage = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type FBoundingBox = unsafe extern "C" fn(*mut c_void, *mut FsRectF) -> c_int;
type FBoundedText = unsafe extern "C" fn(*mut c_void, f64, f64, f64, f64, *mut u16, c_int) -> c_int;
type FLastError = unsafe extern "C" fn() -> c_ulong;

/// The functions resolved from pypdfium2's PDFium, in this order.
pub const PDFIUM_FUNCTIONS: [&str; 24] = [
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
    "FPDF_LoadDocument",
    "FPDF_LoadMemDocument64",
    "FPDF_CloseDocument",
    "FPDF_GetPageCount",
    "FPDF_LoadPage",
    "FPDF_ClosePage",
    "FPDFText_LoadPage",
    "FPDFText_ClosePage",
    "FPDF_GetPageBoundingBox",
    "FPDFText_GetBoundedText",
    "FPDF_GetLastError",
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
    load_document: FLoadDocument,
    load_mem_document: FLoadMemDocument64,
    close_document: FClose,
    page_count: FPageCount,
    load_page: FLoadPage,
    close_page: FClose,
    load_text_page: FLoadTextPage,
    close_text_page: FClose,
    bounding_box: FBoundingBox,
    bounded_text: FBoundedText,
    last_error: FLastError,
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

/// Handle accounting (test hook): opened and closed documents, pages and text pages, and the most documents open at
/// once (1 when every read is serialised).
pub static OPENED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub static CLOSED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
pub static OPEN_DOCS: AtomicU64 = AtomicU64::new(0);
pub static MAX_OPEN_DOCS: AtomicU64 = AtomicU64::new(0);

/// pypdfium2's `ErrorToStr` names of `FPDF_GetLastError` codes.
pub fn error_name(code: c_ulong) -> &'static str {
    match code {
        0 => "Success",
        1 => "Unknown error",
        2 => "File access error",
        3 => "Data format error",
        4 => "Incorrect password error",
        5 => "Unsupported security scheme error",
        6 => "Page not found or content error",
        _ => "Unknown error code",
    }
}

/// Where a document comes from.
pub enum Source<'a> {
    Path(&'a CStr),
    Bytes(&'a [u8]),
}

/// A document opened and owned by Rust; closed on drop.
pub struct Document<'f, 'a> {
    f: &'f PdfiumText,
    raw: *mut c_void,
    _src: std::marker::PhantomData<&'a [u8]>,
}

impl Drop for Document<'_, '_> {
    fn drop(&mut self) {
        // SAFETY: `raw` was returned non-null by FPDF_Load*Document and is closed exactly once, here, after every page
        // guard (which borrows the document) has been dropped.
        unsafe { (self.f.close_document)(self.raw) };
        CLOSED[0].fetch_add(1, Ordering::SeqCst);
        OPEN_DOCS.fetch_sub(1, Ordering::SeqCst);
    }
}

struct Page<'d> {
    f: &'d PdfiumText,
    raw: *mut c_void,
}

impl Drop for Page<'_> {
    fn drop(&mut self) {
        // SAFETY: opened by FPDF_LoadPage on a document that outlives this guard; closed once.
        unsafe { (self.f.close_page)(self.raw) };
        CLOSED[1].fetch_add(1, Ordering::SeqCst);
    }
}

struct TextPage<'p> {
    f: &'p PdfiumText,
    raw: *mut c_void,
}

impl Drop for TextPage<'_> {
    fn drop(&mut self) {
        // SAFETY: opened by FPDFText_LoadPage on a page that outlives this guard; closed once.
        unsafe { (self.f.close_text_page)(self.raw) };
        CLOSED[2].fetch_add(1, Ordering::SeqCst);
    }
}

/// One page read: the char table and the page text (production's `get_text_range()`).
pub struct PageRead {
    pub chars: CharTable,
    pub text: String,
}

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
        use std::mem::transmute as t;
        unsafe {
            Ok(PdfiumText {
                count: t::<usize, FCount>(addrs[0]),
                unicode: t::<usize, FUnicode>(addrs[1]),
                char_box: t::<usize, FCharBox>(addrs[2]),
                loose_box: t::<usize, FLooseBox>(addrs[3]),
                font_size: t::<usize, FDouble>(addrs[4]),
                font_weight: t::<usize, FInt>(addrs[5]),
                font_info: t::<usize, FFontInfo>(addrs[6]),
                angle: t::<usize, FFloat>(addrs[7]),
                generated: t::<usize, FInt>(addrs[8]),
                map_error: t::<usize, FInt>(addrs[9]),
                render_mode: t::<usize, FInt>(addrs[10]),
                media_box: t::<usize, FMediaBox>(addrs[11]),
                page_rotation: t::<usize, FRotation>(addrs[12]),
                load_document: t::<usize, FLoadDocument>(addrs[13]),
                load_mem_document: t::<usize, FLoadMemDocument64>(addrs[14]),
                close_document: t::<usize, FClose>(addrs[15]),
                page_count: t::<usize, FPageCount>(addrs[16]),
                load_page: t::<usize, FLoadPage>(addrs[17]),
                close_page: t::<usize, FClose>(addrs[18]),
                load_text_page: t::<usize, FLoadTextPage>(addrs[19]),
                close_text_page: t::<usize, FClose>(addrs[20]),
                bounding_box: t::<usize, FBoundingBox>(addrs[21]),
                bounded_text: t::<usize, FBoundedText>(addrs[22]),
                last_error: t::<usize, FLastError>(addrs[23]),
            })
        }
    }

    /// Open a document as pypdfium2's `PdfDocument` does (a NULL handle or fewer than 1 page is an error, named by
    /// `FPDF_GetLastError`). The caller holds the process-wide read lock. For `Source::Bytes` the buffer must outlive
    /// the document, which the borrow guarantees.
    pub fn open<'f, 'a>(&'f self, src: Source<'a>, password: Option<&CStr>) -> std::result::Result<Document<'f, 'a>, String> {
        let pw = password.map(|p| p.as_ptr()).unwrap_or(std::ptr::null());
        // SAFETY: the path is NUL-terminated, the bytes live for 'a, and the password is NUL-terminated or null.
        let raw = unsafe {
            match src {
                Source::Path(p) => (self.load_document)(p.as_ptr(), pw),
                Source::Bytes(b) => (self.load_mem_document)(b.as_ptr().cast(), b.len(), pw),
            }
        };
        if raw.is_null() {
            let code = unsafe { (self.last_error)() };
            return Err(format!("Failed to load document (PDFium: {}).", error_name(code)));
        }
        OPENED[0].fetch_add(1, Ordering::SeqCst);
        let now = OPEN_DOCS.fetch_add(1, Ordering::SeqCst) + 1;
        MAX_OPEN_DOCS.fetch_max(now, Ordering::SeqCst);
        let doc = Document { f: self, raw, _src: std::marker::PhantomData };
        if doc.page_count() < 1 {
            let code = unsafe { (self.last_error)() };
            return Err(format!("Failed to load document (PDFium: {}).", error_name(code)));
        }
        Ok(doc)
    }
}

impl Document<'_, '_> {
    pub fn page_count(&self) -> i32 {
        // SAFETY: `raw` is this guard's open document.
        unsafe { (self.f.page_count)(self.raw) }
    }

    /// Read page `index` (0-based): load page and text page, read chars, page geometry and the production text,
    /// close both (guards, also on error).
    pub fn read_page(&self, index: i32, page_no: u32) -> std::result::Result<PageRead, String> {
        let f = self.f;
        // SAFETY: the document is open for this guard's lifetime; page and text page are closed by their guards in
        // reverse order before this function returns.
        let page_raw = unsafe { (f.load_page)(self.raw, index) };
        if page_raw.is_null() {
            return Err(format!("Failed to load page {page_no}."));
        }
        OPENED[1].fetch_add(1, Ordering::SeqCst);
        let page = Page { f, raw: page_raw };
        let tp_raw = unsafe { (f.load_text_page)(page.raw) };
        if tp_raw.is_null() {
            return Err("Failed to load text page.".to_string());
        }
        OPENED[2].fetch_add(1, Ordering::SeqCst);
        let tp = TextPage { f, raw: tp_raw };
        let (w, h, rotation) = unsafe { f.page_geometry(page.raw) }.map_err(|e| e.to_string())?;
        let chars = unsafe { f.read_chars(tp.raw, w, h, rotation, page_no) }.map_err(|e| e.to_string())?;
        let text = unsafe { f.bounded_text(page.raw, tp.raw) }?;
        drop(tp);
        drop(page);
        Ok(PageRead { chars, text })
    }
}

impl PdfiumText {
    /// pypdfium2 4.30 `PdfTextPage.get_text_bounded()` with default bounds (what `get_text_range()` redirects to).
    ///
    /// # Safety
    /// `page` and `textpage` must be open handles of this module, `textpage` loaded from `page`.
    unsafe fn bounded_text(&self, page: *mut c_void, textpage: *mut c_void) -> std::result::Result<String, String> {
        let mut r = FsRectF::default();
        if unsafe { (self.bounding_box)(page, &mut r) } == 0 {
            return Err("Failed to get page bounding box.".to_string());
        }
        let (l, b, rr, t) = (r.left as f64, r.bottom as f64, r.right as f64, r.top as f64);
        let n = unsafe { (self.bounded_text)(textpage, l, t, rr, b, std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0u16; n as usize];
        unsafe { (self.bounded_text)(textpage, l, t, rr, b, buf.as_mut_ptr(), n) };
        Ok(char::decode_utf16(buf.iter().copied()).filter_map(|c| c.ok()).collect())
    }

    /// Page width, height and rotation as the K5c freeze computed them from pypdfium2: `get_mediabox()` (MediaBox,
    /// or ANSI A (0, 0, 612, 792) when PDFium reports none) -> (x1 - x0, y1 - y0); `get_rotation()` = quarter turns
    /// x 90 (anything else is an error, as pypdfium2's lookup raises).
    ///
    /// # Safety
    /// `page` must be an open page handle of this module.
    unsafe fn page_geometry(&self, page: *mut c_void) -> Result<(f64, f64, u16)> {
        let (mut l, mut b, mut r, mut t) = (0f32, 0f32, 0f32, 0f32);
        let ok = unsafe { (self.media_box)(page, &mut l, &mut b, &mut r, &mut t) };
        let (x0, y0, x1, y1) = if ok != 0 { (l as f64, b as f64, r as f64, t as f64) } else { (0.0, 0.0, 612.0, 792.0) };
        let q = unsafe { (self.page_rotation)(page) };
        if !(0..=3).contains(&q) {
            bail!("page rotation code {q}");
        }
        Ok((x1 - x0, y1 - y0, (q * 90) as u16))
    }

    /// The K5c char loop over one text page.
    ///
    /// # Safety
    /// `textpage` must be a text page this module opened and has not closed.
    unsafe fn read_chars(&self, textpage: *mut c_void, page_w: f64, page_h: f64, rotation: u16, page: u32) -> Result<CharTable> {
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
