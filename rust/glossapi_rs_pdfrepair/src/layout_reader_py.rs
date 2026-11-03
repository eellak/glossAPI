//! Checked Python entry point of the native layout char reader (K5-A1r, CP-K5a H1).
//!
//! `layout_v2_read_textpage(textpage, page_no, source_sha256)` takes a live pypdfium2 `PdfTextPage`, never integers:
//! - **Library identity.** The 13 PDFium addresses (11 `FPDFText_*`, `FPDFPage_GetMediaBox`, `FPDFPage_GetRotation`) are resolved here, once per process, from the PDFium library
//!   pypdfium2 loaded (`pypdfium2_raw.bindings._libs["pdfium"]`). Each must equal the address of pypdfium2's own bound
//!   function (`pypdfium2.raw.<name>`). `FPDF_GetLastError` must resolve too. Then a self-test reads the built-in
//!   fixture `fixtures/layout_selftest.pdf` natively and through pypdfium2's ctypes signatures, and every field must
//!   agree. Any failure raises `LayoutV2ReaderUnavailable` (now and on every later call), so the caller can fall back
//!   to the Python reader. Callers cannot supply addresses.
//! - **Lifetime.** The text page, its page, the page's document and form env (if any) are held for the whole call.
//!   Each must be open: `raw` set, finalizer attached, and pypdfium2's close state still "auto" (a close in progress
//!   in another thread has already set it). The library must still be initialised. Otherwise the call raises
//!   `LayoutV2ClosedObject` before any native call.
//! - **Threading contract.** The GIL is held from the final liveness check to the end of the read, and nothing
//!   between them runs Python code, so a pypdfium2 `close()` from another thread either has already flagged the object
//!   (typed error) or runs after the read. Native reads are serialised by the same GIL hold. pypdfium2 4.30 has no
//!   lock of its own. PDFium calls that another thread already has in flight through ctypes (with the GIL released)
//!   stay under pypdfium2's own rule: one thread per document.
use crate::layout::k5chars;
use crate::layout::reader::{PdfiumText, PDFIUM_FUNCTIONS};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict};
use pyo3::{create_exception, intern};

create_exception!(
    glossapi_rs_pdfrepair,
    LayoutV2ReaderUnavailable,
    PyRuntimeError,
    "The native layout char reader cannot be used with the loaded PDFium; use the Python reader."
);
create_exception!(
    glossapi_rs_pdfrepair,
    LayoutV2ClosedObject,
    PyValueError,
    "A pypdfium2 object given to the native layout char reader is closed or closing."
);

static SELFTEST_PDF: &[u8] = include_bytes!("../fixtures/layout_selftest.pdf");
const SELFTEST_TEXT: &str = "Ab 1.\r\nCd";

/// The reference reader of the self-test: the K5c `read_chars` loop over pypdfium2's own ctypes signatures.
const SELFTEST_PY: &std::ffi::CStr = cr#"
import ctypes
import pypdfium2.raw as raw

def reference(textpage):
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
    return out
"#;

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
            return Ok(Err(format!("pypdfium2 {pypdfium2}: the lifecycle checks are written for pypdfium2 4.x")));
        }
        let lib = library(py)?;
        let handle: usize = lib.getattr("_handle")?.extract()?;
        let path: String = lib.getattr("_name")?.str()?.extract()?;
        let raw = py.import("pypdfium2.raw")?;
        let mut addrs = Vec::with_capacity(PDFIUM_FUNCTIONS.len());
        for name in PDFIUM_FUNCTIONS.iter().chain(std::iter::once(&"FPDF_GetLastError")) {
            let bound = address(py, &raw.getattr(*name)?)?;
            let direct = address(py, &lib.get_item(*name)?)?;
            if bound == 0 || bound != direct {
                return Ok(Err(format!(
                    "{name}: pypdfium2.raw points to 0x{bound:x}, the loaded library {path} resolves 0x{direct:x}"
                )));
            }
            if *name != "FPDF_GetLastError" {
                addrs.push(bound);
            }
        }
        // SAFETY: every address was just resolved by dlsym from pypdfium2's loaded PDFium and equals pypdfium2's own
        // bound function, so the signatures are the documented FPDFText_* ones.
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

/// One char of the self-test reference: (unicode, box), loose box, (size, weight, font name, font flags),
/// (angle, generated, map error, render mode).
type SelftestChar = ((u32, f64, f64, f64, f64), (f64, f64, f64, f64), (f64, i64, String, i64), (f64, i64, i64, i64));

/// Reads the built-in fixture natively and through pypdfium2's ctypes calls; every field must agree.
fn selftest(py: Python<'_>, text: &PdfiumText) -> PyResult<Result<usize, String>> {
    let module = PyModule::from_code(py, SELFTEST_PY, c"glossapi_rs_pdfrepair_layout_selftest.py", c"_layout_selftest")?;
    let pdfium = py.import("pypdfium2")?;
    let doc = pdfium.getattr("PdfDocument")?.call1((PyBytes::new(py, SELFTEST_PDF),))?;
    let out = (|| -> PyResult<Result<usize, String>> {
        let page = doc.get_item(0)?;
        let tp = page.call_method0("get_textpage")?;
        let want: Vec<SelftestChar> = module.getattr("reference")?.call1((&tp,))?.extract()?;
        let (x0, y0, x1, y1): (f64, f64, f64, f64) = page.call_method0("get_mediabox")?.extract()?;
        let rot: i64 = page.call_method0("get_rotation")?.extract()?;
        let got = read_live(py, text, &tp, 1);
        tp.call_method0("close")?;
        page.call_method0("close")?;
        let t = got?;
        if t.ch.len() != want.len() {
            return Ok(Err(format!("{} chars natively, {} through ctypes", t.ch.len(), want.len())));
        }
        if t.page_w.to_bits() != (x1 - x0).to_bits() || t.page_h.to_bits() != (y1 - y0).to_bits() || t.rotation as i64 != rot {
            return Ok(Err(format!("page geometry {}x{} rot {} natively, {}x{} rot {rot} through pypdfium2",
                                  t.page_w, t.page_h, t.rotation, x1 - x0, y1 - y0)));
        }
        let s: String = t.ch.iter().map(|&c| char::from_u32(c).unwrap_or('\u{fffd}')).collect();
        if s != SELFTEST_TEXT {
            return Ok(Err(format!("fixture text {s:?}, expected {SELFTEST_TEXT:?}")));
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
                return Ok(Err(format!("char {i} differs between the native and the ctypes read")));
            }
        }
        Ok(Ok(want.len()))
    })();
    doc.call_method0("close")?;
    out
}

/// The pypdfium2 object is open: raw handle set, finalizer attached, close state "auto" (0).
fn check_open(py: Python<'_>, what: &str, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let raw = obj.getattr(intern!(py, "raw"))?;
    let open = !raw.is_none()
        && raw.is_truthy()?
        && !obj.getattr(intern!(py, "_finalizer"))?.is_none()
        && obj.getattr(intern!(py, "_autoclose_state"))?.getattr(intern!(py, "value"))?.extract::<i64>()? == 0;
    if open {
        Ok(())
    } else {
        Err(LayoutV2ClosedObject::new_err(format!("the {what} is closed or being closed")))
    }
}

struct Owners<'py> {
    textpage: Bound<'py, PyAny>,
    page: Bound<'py, PyAny>,
    pdf: Bound<'py, PyAny>,
    formenv: Option<Bound<'py, PyAny>>,
    library_available: Bound<'py, PyAny>,
}

impl Owners<'_> {
    /// Only attribute reads and C-level ctypes accessors: no Python code runs, so the GIL is not given up.
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        if !self.library_available.getattr(intern!(py, "value"))?.is_truthy()? {
            return Err(LayoutV2ClosedObject::new_err("pypdfium2 has destroyed the PDFium library"));
        }
        check_open(py, "document", &self.pdf)?;
        if let Some(f) = &self.formenv {
            check_open(py, "form env", f)?;
        }
        check_open(py, "page", &self.page)?;
        check_open(py, "text page", &self.textpage)
    }
}

fn read_live(py: Python<'_>, text: &PdfiumText, textpage: &Bound<'_, PyAny>, page_no: u32) -> PyResult<k5chars::CharTable> {
    let pdfium = py.import(intern!(py, "pypdfium2"))?;
    if !textpage.is_instance(&pdfium.getattr(intern!(py, "PdfTextPage"))?)? {
        return Err(PyTypeError::new_err("layout_v2_read_textpage needs a pypdfium2.PdfTextPage"));
    }
    let page = textpage.getattr(intern!(py, "page"))?;
    let pdf = page.getattr(intern!(py, "pdf"))?;
    let formenv = page.getattr(intern!(py, "formenv"))?;
    let owners = Owners {
        textpage: textpage.clone(),
        page,
        pdf,
        formenv: if formenv.is_none() { None } else { Some(formenv) },
        library_available: py.import("pypdfium2.internal.bases")?.getattr("LIBRARY_AVAILABLE")?,
    };
    owners.check(py)?;
    // From here to the end of the read nothing runs Python code or releases the GIL: the handle slots are read with
    // the C-level `ctypes.addressof`, and page size and rotation come from PDFium directly (pypdfium2's own
    // get_mediabox/get_rotation would release the GIL inside ctypes).
    let addressof = py.import(intern!(py, "ctypes"))?.getattr(intern!(py, "addressof"))?;
    let tp_raw = owners.textpage.getattr(intern!(py, "raw"))?;
    let page_raw = owners.page.getattr(intern!(py, "raw"))?;
    let tp_slot: usize = addressof.call1((&tp_raw,))?.extract()?;
    let page_slot: usize = addressof.call1((&page_raw,))?.extract()?;
    owners.check(py)?;
    // SAFETY: the slots are the buffers of the live ctypes pointer objects held above (FPDF_TEXTPAGE, FPDF_PAGE).
    let (tp_handle, page_handle) = unsafe { (*(tp_slot as *const usize), *(page_slot as *const usize)) };
    if tp_handle == 0 || page_handle == 0 {
        return Err(LayoutV2ClosedObject::new_err("a pypdfium2 handle is null"));
    }
    // SAFETY: functions resolved and self-tested against this PDFium; both handles are open objects of it and cannot
    // be closed before the read returns (GIL held, no Python code runs).
    unsafe {
        let (w, h, rotation) = text
            .page_geometry(page_handle as *mut std::ffi::c_void)
            .map_err(|e| PyValueError::new_err(format!("layout v2 reader: {e}")))?;
        text.read(tp_handle as *mut std::ffi::c_void, w, h, rotation, page_no)
    }
    .map_err(|e| PyValueError::new_err(format!("layout v2 reader: {e}")))
}

fn resolved(py: Python<'_>) -> PyResult<&'static Resolved> {
    let r = PDFIUM.get_or_init(py, || resolve(py)).as_ref().map_err(|e| LayoutV2ReaderUnavailable::new_err(e.clone()))?;
    // the same library must still be the one pypdfium2 uses
    let handle: usize = library(py)
        .and_then(|lib| lib.getattr("_handle")?.extract())
        .map_err(|e| LayoutV2ReaderUnavailable::new_err(format!("cannot read pypdfium2's PDFium handle: {e}")))?;
    if handle != r.handle {
        return Err(LayoutV2ReaderUnavailable::new_err("pypdfium2's PDFium library changed after the reader resolved it"));
    }
    Ok(r)
}

/// `layout_v2_read_textpage(textpage, page_no, source_sha256) -> bytes`: the page's `.k5chars` table, read natively
/// from PDFium (the K5c reader's fields, byte for byte). `textpage` must be an open `pypdfium2.PdfTextPage`; page size
/// and rotation come from its page. Raises `LayoutV2ReaderUnavailable` if the loaded PDFium fails the identity or
/// self-test checks, `LayoutV2ClosedObject` for a closed or closing text page, page or document, and `TypeError` for
/// anything that is not a `PdfTextPage`.
#[pyfunction]
pub fn layout_v2_read_textpage<'py>(
    py: Python<'py>,
    textpage: Bound<'py, PyAny>,
    page_no: u32,
    source_sha256: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let sha: [u8; 32] = source_sha256.try_into().map_err(|_| PyValueError::new_err("source_sha256 must be 32 bytes"))?;
    let r = resolved(py)?;
    let t = read_live(py, &r.text, &textpage, page_no)?;
    Ok(PyBytes::new(py, &k5chars::dump(&t, &sha)))
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("LayoutV2ReaderUnavailable", py.get_type::<LayoutV2ReaderUnavailable>())?;
    m.add("LayoutV2ClosedObject", py.get_type::<LayoutV2ClosedObject>())?;
    m.add_function(wrap_pyfunction!(layout_v2_read_textpage, m)?)?;
    m.add_function(wrap_pyfunction!(layout_v2_reader_info, m)?)?;
    Ok(())
}
