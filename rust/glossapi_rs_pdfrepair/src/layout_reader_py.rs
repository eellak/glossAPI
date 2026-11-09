//! Python entry point of the native layout char reader, with Rust-owned PDFium handles (K5-A1r2, CP-K5a cycle 2).
//!
//! `layout_v2_read_document(source, pages, source_sha256, password=None)` takes a document as a path (`str`) or as
//! `bytes`, plus 1-based page numbers. Rust opens the document, each page and its text page itself, reads them and
//! closes them through `Drop` guards (`layout/reader.rs`). It never accepts a pypdfium2 object, a proxy, a subclass or
//! an integer handle, and it never reads a handle out of a Python object. Only exact `str` or `bytes` are accepted,
//! so no user code runs while a handle is open.
//!
//! - **Library identity.** The 24 PDFium functions are resolved here, once per process, from the library pypdfium2
//!   loaded (`pypdfium2_raw.bindings._libs["pdfium"]`). Each must equal pypdfium2's own bound function. A self-test
//!   then opens the built-in fixture natively and compares every char field, the page geometry and the text bit for
//!   bit with pypdfium2's own read of the same bytes. Any failure raises `LayoutV2ReaderUnavailable`, now and on every
//!   later call, so the caller falls back to the Python reader. Each call re-checks pypdfium2's library handle.
//! - **Threading contract.** Every native read (open, read, close) runs under one process-wide lock with the GIL held
//!   throughout. Nothing is released to Python while a handle is open. PDFium is not thread-safe across documents
//!   within one process, so callers must not drive PDFium through pypdfium2 from other threads during a native read.
//!   That is pypdfium2's own rule; any such call already in flight with the GIL released is outside this lock.
//!   glossAPI's own PDFium users and this reader share one process-wide lock (`glossapi.pdfium_guard.PDFIUM_LOCK`,
//!   taken by `glossapi.markdown_layout` around every call), which closes that gap inside the package (K5-A2).
//!
//! **Threat model (K5 quality policy P7, decision DECIDE-reader-threat-model, 4 Nov 2026).** Documents and public
//! argument values are untrusted; the Python runtime, installed dependencies and executable caller code are trusted.
//! The wrapper validates arguments and preserves native handle ownership and lifetime. Dependency replacement, forged
//! ctypes state and uncoordinated library teardown are unsupported. All PDFium users, initialization and destruction
//! require process-wide coordination; the reader's mutex and GIL do not serialize external ctypes calls. Identity
//! checks and self-tests detect accidental incompatibility, not malicious substitution. This boundary does not sandbox
//! PDFium or guarantee unlimited resource availability.
use crate::layout::k5chars;
use crate::layout::reader::{self, PdfiumText, Source, PDFIUM_FUNCTIONS};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use pyo3::{create_exception, intern};
use std::ffi::CString;
use std::sync::atomic::Ordering;
use std::sync::Mutex;

create_exception!(
    glossapi_rs_pdfrepair,
    LayoutV2ReaderUnavailable,
    PyRuntimeError,
    "The native layout char reader cannot be used with the loaded PDFium; use the Python reader."
);
create_exception!(
    glossapi_rs_pdfrepair,
    LayoutV2DocumentError,
    PyValueError,
    "The native layout char reader could not open the document or was given an invalid page number."
);

static SELFTEST_PDF: &[u8] = include_bytes!("../fixtures/layout_selftest.pdf");
const SELFTEST_TEXT: &str = "Ab 1.\r\nCd";
/// Every native read holds this lock from opening the document to closing it.
static READ_LOCK: Mutex<()> = Mutex::new(());

/// pypdfium2's own read of the self-test fixture: the K5c `read_chars` loop, `get_mediabox`, `get_rotation` and
/// `get_text_range()`, on pypdfium2's objects only (nothing crosses into Rust except plain values).
const SELFTEST_PY: &std::ffi::CStr = cr#"
import ctypes, warnings
import pypdfium2 as pdfium
import pypdfium2.raw as raw

def reference(data):
    doc = pdfium.PdfDocument(data)
    try:
        page = doc[0]
        textpage = page.get_textpage()
        tp = textpage.raw
        n = raw.FPDFText_CountChars(tp)
        l, r, b, t = (ctypes.c_double() for _ in range(4))
        rect = raw.FS_RECTF()
        fl = ctypes.c_int()
        buf = ctypes.create_string_buffer(512)
        out = []
        for i in range(n):
            u = raw.FPDFText_GetUnicode(tp, i)
            raw.FPDFText_GetCharBox(tp, i, ctypes.byref(l), ctypes.byref(r), ctypes.byref(b), ctypes.byref(t))
            raw.FPDFText_GetLooseCharBox(tp, i, ctypes.byref(rect))
            size = raw.FPDFText_GetFontSize(tp, i)
            weight = raw.FPDFText_GetFontWeight(tp, i)
            ln = raw.FPDFText_GetFontInfo(tp, i, buf, len(buf), ctypes.byref(fl))
            name = buf.raw[:max(0, ln - 1)].decode("utf-8", "replace") if ln > 0 else ""
            out.append(((int(u), l.value, b.value, r.value, t.value), (rect.left, rect.bottom, rect.right, rect.top),
                        (float(size), int(weight), name, int(fl.value) if ln > 0 else 0),
                        (float(raw.FPDFText_GetCharAngle(tp, i)), int(raw.FPDFText_IsGenerated(tp, i)),
                         int(raw.FPDFText_HasUnicodeMapError(tp, i)), int(raw.FPDFText_GetTextRenderMode(tp, i)))))
        x0, y0, x1, y1 = page.get_mediabox()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = textpage.get_text_range()
        geometry = (float(x1 - x0), float(y1 - y0), int(page.get_rotation()), len(doc))
        textpage.close()
        page.close()
        return out, geometry, text
    finally:
        doc.close()
"#;

/// (unicode, box), loose box, (size, weight, font name, font flags), (angle, generated, map error, render mode).
type SelftestChar = ((u32, f64, f64, f64, f64), (f64, f64, f64, f64), (f64, i64, String, i64), (f64, i64, i64, i64));

struct Resolved {
    text: PdfiumText,
    handle: usize,
    path: String,
    pypdfium2: String,
    libpdfium: String,
    selftest_chars: usize,
}

static PDFIUM: PyOnceLock<Result<Resolved, String>> = PyOnceLock::new();

fn address(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<usize> {
    let ctypes = py.import(intern!(py, "ctypes"))?;
    let v = ctypes.getattr(intern!(py, "cast"))?.call1((func, ctypes.getattr(intern!(py, "c_void_p"))?))?;
    Ok(v.getattr(intern!(py, "value"))?.extract::<Option<usize>>()?.unwrap_or(0))
}

fn library(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("pypdfium2_raw.bindings")?.getattr("_libs")?.get_item("pdfium")
}

fn resolve(py: Python<'_>) -> Result<Resolved, String> {
    let r = (|| -> PyResult<Result<Resolved, String>> {
        let pdfium = py.import("pypdfium2")?;
        let pypdfium2: String = pdfium.getattr("V_PYPDFIUM2")?.str()?.extract()?;
        let libpdfium: String = pdfium.getattr("V_LIBPDFIUM")?.str()?.extract()?;
        if !pypdfium2.starts_with("4.") {
            return Ok(Err(format!("pypdfium2 {pypdfium2}: the reader is qualified with pypdfium2 4.x")));
        }
        let lib = library(py)?;
        let handle: usize = lib.getattr("_handle")?.extract()?;
        let path: String = lib.getattr("_name")?.str()?.extract()?;
        let raw = py.import("pypdfium2.raw")?;
        let mut addrs = Vec::with_capacity(PDFIUM_FUNCTIONS.len());
        for name in PDFIUM_FUNCTIONS {
            let bound = address(py, &raw.getattr(name)?)?;
            let direct = address(py, &lib.get_item(name)?)?;
            if bound == 0 || bound != direct {
                return Ok(Err(format!(
                    "{name}: pypdfium2.raw points to 0x{bound:x}, the loaded library {path} resolves 0x{direct:x}"
                )));
            }
            addrs.push(bound);
        }
        // SAFETY: every address was just resolved by dlsym from pypdfium2's loaded PDFium and equals pypdfium2's own
        // bound function, so the signatures are PDFium's documented ones.
        let text = match unsafe { PdfiumText::from_addresses(&addrs) } {
            Ok(t) => t,
            Err(e) => return Ok(Err(e.to_string())),
        };
        let selftest_chars = match selftest(py, &text)? {
            Ok(n) => n,
            Err(e) => return Ok(Err(format!("self-test: {e}"))),
        };
        Ok(Ok(Resolved { text, handle, path, pypdfium2, libpdfium, selftest_chars }))
    })();
    match r {
        Ok(x) => x,
        Err(e) => Err(format!("cannot resolve PDFium through pypdfium2: {e}")),
    }
}

/// Opens the built-in fixture natively (Rust-owned handles) and compares it with pypdfium2's own read of it.
fn selftest(py: Python<'_>, text: &PdfiumText) -> PyResult<Result<usize, String>> {
    let module = PyModule::from_code(py, SELFTEST_PY, c"glossapi_rs_pdfrepair_layout_selftest.py", c"_layout_selftest")?;
    let (want, geometry, want_text): (Vec<SelftestChar>, (f64, f64, i64, i64), String) =
        module.getattr("reference")?.call1((PyBytes::new(py, SELFTEST_PDF),))?.extract()?;
    let got = {
        let _guard = READ_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let doc = match text.open(Source::Bytes(SELFTEST_PDF), None) {
            Ok(d) => d,
            Err(e) => return Ok(Err(e)),
        };
        let pages = doc.page_count();
        (doc.read_page(0, 1, true), pages)
    };
    let (page, pages) = got;
    let page = match page {
        Ok(p) => p,
        Err(e) => return Ok(Err(e)),
    };
    let Some(t) = &page.chars else { return Ok(Err("the self-test read returned no chars".into())) };
    if pages as i64 != geometry.3 || page.text != want_text || page.text != SELFTEST_TEXT {
        return Ok(Err(format!("pages {pages}/{} text {:?}/{want_text:?}", geometry.3, page.text)));
    }
    if t.page_w.to_bits() != geometry.0.to_bits() || t.page_h.to_bits() != geometry.1.to_bits() || t.rotation as i64 != geometry.2 {
        return Ok(Err("page geometry differs between the native and the pypdfium2 read".into()));
    }
    if t.ch.len() != want.len() {
        return Ok(Err(format!("{} chars natively, {} through pypdfium2", t.ch.len(), want.len())));
    }
    for (i, w) in want.iter().enumerate() {
        let ((u, l, b, r, tt), (ll, lb, lr, lt), (size, weight, ref name, fl), (angle, gen, maperr, mode)) = *w;
        let mut flags = 0u8;
        flags |= if gen == 1 { 1 } else { 0 };
        flags |= if u == 2 { 2 } else { 0 };
        flags |= if maperr == 1 { 4 } else { 0 };
        flags |= if fl & 64 != 0 { 8 } else { 0 };
        flags |= if fl & 4 != 0 { 16 } else { 0 };
        flags |= if mode == 3 { 32 } else { 0 };
        let bits = |a: [f64; 4]| a.map(f64::to_bits);
        let same = t.ch[i] == (if u > 0x10FFFF { 0xFFFD } else { u })
            && bits(t.bbox[i]) == bits([l, b, r, tt])
            && bits(t.loose[i]) == bits([ll, lb, lr, lt])
            && t.size[i].to_bits() == (size as f32).to_bits()
            && t.weight[i] as i64 == weight.clamp(-32768, 32767)
            && t.fonts.get(t.font[i] as usize) == Some(name)
            && t.flags[i] == flags
            && (t.angle[i] as f64).to_bits() == angle.to_bits();
        if !same {
            return Ok(Err(format!("char {i} differs between the native and the pypdfium2 read")));
        }
    }
    Ok(Ok(want.len()))
}

fn resolved(py: Python<'_>) -> PyResult<&'static Resolved> {
    let r = PDFIUM.get_or_init(py, || resolve(py)).as_ref().map_err(|e| LayoutV2ReaderUnavailable::new_err(e.clone()))?;
    let handle: usize = library(py)
        .and_then(|lib| lib.getattr("_handle")?.extract())
        .map_err(|e| LayoutV2ReaderUnavailable::new_err(format!("cannot read pypdfium2's PDFium handle: {e}")))?;
    if handle != r.handle {
        return Err(LayoutV2ReaderUnavailable::new_err("pypdfium2's PDFium library changed after the reader resolved it"));
    }
    let available: bool = py
        .import("pypdfium2.internal.bases")
        .and_then(|m| m.getattr("LIBRARY_AVAILABLE")?.getattr("value")?.extract())
        .map_err(|e| LayoutV2ReaderUnavailable::new_err(format!("cannot read pypdfium2's library state: {e}")))?;
    if !available {
        return Err(LayoutV2ReaderUnavailable::new_err("pypdfium2 has destroyed the PDFium library"));
    }
    Ok(r)
}

/// `layout_v2_read_document(source, pages, source_sha256, password=None, chars=True) -> (page_count, [(page_no, chars,
/// text, error)])`. `chars=False` (K5-A2, plain Markdown mode) returns `None` for every char table and reads the text
/// only.
///
/// `source` is a path (exact `str`) or the document (exact `bytes`). `pages` are 1-based page numbers. Per page:
/// - `chars` is the `.k5chars` table (the K5c reader's fields, byte for byte);
/// - `text` is the page text exactly as production's `get_text_range()` returns it (unnormalised);
/// - or `error` names a page that PDFium could not load, as production's per-page `except` would see it.
///
/// Raises `TypeError` for any other source type, `LayoutV2DocumentError` if the document does not open (PDFium's
/// error name) or a page number is out of range, and `LayoutV2ReaderUnavailable` if the loaded PDFium fails the
/// identity or self-test checks.
#[pyfunction]
#[pyo3(signature = (source, pages, source_sha256, password=None, chars=true))]
pub fn layout_v2_read_document<'py>(
    py: Python<'py>,
    source: Bound<'py, PyAny>,
    pages: Vec<i64>,
    source_sha256: &[u8],
    password: Option<Bound<'py, PyAny>>,
    chars: bool,
) -> PyResult<(i32, Bound<'py, PyList>)> {
    let sha: [u8; 32] = source_sha256.try_into().map_err(|_| PyValueError::new_err("source_sha256 must be 32 bytes"))?;
    let password = match password {
        None => None,
        Some(p) if p.is_exact_instance_of::<PyString>() => Some(
            CString::new(p.extract::<String>()?).map_err(|_| PyValueError::new_err("password contains NUL"))?,
        ),
        Some(_) => return Err(PyTypeError::new_err("password must be a str")),
    };
    // Exact types only: a str or bytes subclass could run user code (__del__, properties) while a handle is open.
    let path: Option<CString>;
    let bytes: Option<Bound<'py, PyBytes>>;
    if source.is_exact_instance_of::<PyString>() {
        let s: String = source.extract()?;
        path = Some(CString::new(s).map_err(|_| PyValueError::new_err("path contains NUL"))?);
        bytes = None;
    } else if source.is_exact_instance_of::<PyBytes>() {
        path = None;
        bytes = Some(source.cast_into::<PyBytes>()?);
    } else {
        return Err(PyTypeError::new_err(
            "layout_v2_read_document needs a path (str) or the document (bytes); pypdfium2 objects, handles and \
             subclasses are not accepted",
        ));
    }
    let r = resolved(py)?;
    // From here on only Rust runs: plain values in, owned handles, plain values out. The GIL is never released.
    let (count, reads) = {
        let _guard = READ_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let src = match (&path, &bytes) {
            (Some(p), _) => Source::Path(p.as_c_str()),
            (None, Some(b)) => Source::Bytes(b.as_bytes()),
            _ => unreachable!(),
        };
        let doc = r.text.open(src, password.as_deref()).map_err(LayoutV2DocumentError::new_err)?;
        let count = doc.page_count();
        if let Some(bad) = pages.iter().find(|&&p| p < 1 || p > count as i64) {
            return Err(LayoutV2DocumentError::new_err(format!("page {bad} is out of range 1..{count}")));
        }
        let reads: Vec<(i64, Result<reader::PageRead, String>)> =
            pages.iter().map(|&p| (p, doc.read_page((p - 1) as i32, p as u32, chars))).collect();
        drop(doc);
        (count, reads)
    };
    let out = PyList::empty(py);
    for (p, read) in reads {
        match read {
            Ok(pr) => out.append((p, pr.chars.map(|t| PyBytes::new(py, &k5chars::dump(&t, &sha))), pr.text, None::<String>))?,
            Err(e) => out.append((p, None::<Bound<'_, PyBytes>>, None::<String>, e))?,
        }
    }
    Ok((count, out))
}

/// `layout_v2_reader_info() -> dict`: the resolved library and self-test result; raises `LayoutV2ReaderUnavailable`.
#[pyfunction]
pub fn layout_v2_reader_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let r = resolved(py)?;
    let d = PyDict::new(py);
    d.set_item("pypdfium2", &r.pypdfium2)?;
    d.set_item("libpdfium", &r.libpdfium)?;
    d.set_item("library", &r.path)?;
    d.set_item("functions", PDFIUM_FUNCTIONS.to_vec())?;
    d.set_item("selftest_chars", r.selftest_chars)?;
    Ok(d)
}

/// Test hook (read-only counters, no handles): documents, pages and text pages opened and closed by the reader, and
/// the most documents it had open at once.
#[pyfunction]
pub fn _layout_v2_handle_counts(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    for (i, k) in ["documents", "pages", "text_pages"].iter().enumerate() {
        d.set_item(*k, (reader::OPENED[i].load(Ordering::SeqCst), reader::CLOSED[i].load(Ordering::SeqCst)))?;
    }
    d.set_item("open_documents", reader::OPEN_DOCS.load(Ordering::SeqCst))?;
    d.set_item("max_open_documents", reader::MAX_OPEN_DOCS.load(Ordering::SeqCst))?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("LayoutV2ReaderUnavailable", py.get_type::<LayoutV2ReaderUnavailable>())?;
    m.add("LayoutV2DocumentError", py.get_type::<LayoutV2DocumentError>())?;
    m.add_function(wrap_pyfunction!(layout_v2_read_document, m)?)?;
    m.add_function(wrap_pyfunction!(layout_v2_reader_info, m)?)?;
    m.add_function(wrap_pyfunction!(_layout_v2_handle_counts, m)?)?;
    Ok(())
}
