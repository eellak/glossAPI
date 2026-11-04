mod cmap;
mod font;
pub mod layout;
#[cfg(all(feature = "python", not(test)))]
mod layout_reader_py;
mod pdf;
mod text;
use anyhow::{anyhow, bail, Result};
use pdf::{decode, serialize, Obj, Pdf};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FontReport {
    pub font_name: String,
    pub subtype: String,
    pub embedded: bool,
    pub font_program_kind: Option<String>,
    pub has_tounicode: bool,
    pub codes: usize,
    pub disagreeing_codes: usize,
    pub greek_codes: usize,
    pub affected_greek_codes: usize,
    pub unmapped_codes: usize,
    pub category: String,
    pub repair_possible: bool,
    pub reason: Option<String>,
    pub error: Option<String>,
    pub object_id: Option<u32>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PdfFontReport {
    pub path: String,
    pub category: String,
    pub pages: usize,
    pub sampled_pages: Vec<usize>,
    pub fonts: Vec<FontReport>,
    pub error: Option<String>,
    /// The OS-level I/O failure behind `error` (errno, message), if any: not serialized; the Python binding raises
    /// it as `OSError` so a full disk is never read as a document error (K4g fix 3).
    #[serde(skip)]
    pub os_error: Option<(i32, String)>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct RepairResult {
    pub src: String,
    pub dst: String,
    pub category: String,
    pub repaired_fonts: usize,
    pub fonts: Vec<FontReport>,
    pub written: bool,
    pub error: Option<String>,
    /// As `PdfFontReport::os_error`.
    #[serde(skip)]
    pub os_error: Option<(i32, String)>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sample {
    pub pages: usize,
    pub sampled_pages: Vec<usize>,
    pub has_text: bool,
}
fn error(e: anyhow::Error) -> String {
    let s = e.to_string();
    if [
        "ValueError:",
        "TTLibError:",
        "KeyError:",
        "UnicodeDecodeError:",
        "FileNotFoundError:",
        "PdfStreamError:",
    ]
    .iter()
    .any(|p| s.starts_with(p))
    {
        s
    } else {
        format!("ValueError: {s}")
    }
}
/// The OS-level I/O failure in an error's cause chain: an `io::Error` that carries an errno (ENOSPC, EDQUOT, EIO,
/// EACCES ...), with the error's message. A decoder `io::Error` (flate2 `InvalidData`) carries no errno and stays a
/// document error.
fn os_error(e: &anyhow::Error) -> Option<(i32, String)> {
    e.chain()
        .filter_map(|c| c.downcast_ref::<std::io::Error>())
        .find_map(std::io::Error::raw_os_error)
        .map(|code| (code, e.to_string()))
}
fn descendant(pdf: &mut Pdf, f: &Obj) -> Result<Obj> {
    let a = pdf.get(f, "DescendantFonts")?.array();
    pdf.resolve(
        a.first()
            .ok_or_else(|| anyhow!("Missing DescendantFonts"))?,
    )
}
/// (code -> Unicode from the embedded program, program kind, embedded, reason it cannot be used)
type Embedded = (BTreeMap<u32, String>, Option<String>, bool, Option<String>);
fn embedded(pdf: &mut Pdf, f: &Obj) -> Result<Embedded> {
    let subtype = f.get("Subtype").name();
    let d = if subtype == "Type0" {
        descendant(pdf, f)?
    } else {
        f.clone()
    };
    let descriptor = pdf.get(&d, "FontDescriptor")?;
    let key = ["FontFile2", "FontFile3", "FontFile"]
        .iter()
        .find(|k| descriptor.has(k))
        .copied();
    let Some(key) = key else {
        return Ok((
            BTreeMap::new(),
            None,
            false,
            Some("Font has no embedded program".into()),
        ));
    };
    let program = pdf.get(&descriptor, key)?;
    let kind = format!(
        "{key}{}",
        if key == "FontFile3" {
            format!("/{}", program.get("Subtype").name())
        } else {
            String::new()
        }
    );
    let unsupported = |reason: &str| {
        Ok((
            BTreeMap::new(),
            Some(kind.clone()),
            true,
            Some(reason.into()),
        ))
    };
    if !["Type0", "TrueType"].contains(&subtype.as_str()) || key == "FontFile" {
        return unsupported("Unsupported font subtype/program");
    }
    if key == "FontFile3"
        && !["OpenType", "CIDFontType0C"].contains(&program.get("Subtype").name().as_str())
    {
        return unsupported("Unsupported FontFile3 subtype");
    }
    let g = font::glyphs(&decode(&program)?, &kind)?;
    let codes: BTreeMap<u32, usize>;
    if subtype == "TrueType" {
        let mut m = g.symbol;
        let enc = pdf.get(f, "Encoding")?;
        if matches!(enc, Obj::Dict(_)) {
            let diff = pdf.get(&enc, "Differences")?.array();
            let mut code = None;
            for item in diff {
                if let Obj::Num(n) = &item {
                    code = Some(n.parse::<i64>()?)
                } else {
                    let c = code.ok_or_else(|| anyhow!("Invalid Differences mapping"))?;
                    if !(0..=255).contains(&c) {
                        bail!("Invalid Differences mapping")
                    }
                    m.remove(&(c as u32));
                    if let Some(gid) = g.names.get(item.name_bytes()) {
                        m.insert(c as u32, *gid);
                    }
                    code = Some(c + 1)
                }
            }
        }
        if m.is_empty() {
            return unsupported(
                "Simple font needs a symbol (3,0) cmap or Differences; no encoding guesses",
            );
        }
        codes = m;
    } else {
        let cid = if d.get("Subtype").name() == "CIDFontType2" {
            let r = if d.has("CIDToGIDMap") {
                d.get("CIDToGIDMap")
            } else {
                Obj::Name("Identity".into())
            };
            if r.name() == "Identity" {
                g.mapping
                    .keys()
                    .map(|n| (*n as u32, *n))
                    .collect::<BTreeMap<_, _>>()
            } else {
                let data = decode(&pdf.resolve(&r)?)?;
                if data.len() % 2 == 1 {
                    bail!("Odd-length CIDToGIDMap")
                }
                data.chunks(2)
                    .enumerate()
                    .map(|(i, b)| (i as u32, u16::from_be_bytes([b[0], b[1]]) as usize))
                    .collect()
            }
        } else if d.get("Subtype").name() == "CIDFontType0" && !g.cids.is_empty() {
            g.cids
        } else {
            return unsupported("CFF/OpenType lacks a usable CID mapping");
        };
        let enc = pdf.get(f, "Encoding")?;
        if ["Identity-H", "Identity-V"].contains(&enc.name().as_str()) {
            codes = cid
        } else if matches!(enc, Obj::Stream(..)) {
            let data = decode(&enc)?;
            let m = cmap::entries(&data, false)?;
            codes = m
                .into_iter()
                .filter_map(|(code, v)| {
                    if let cmap::Value::Cid(n) = v {
                        cid.get(&n).map(|g| (code, *g))
                    } else {
                        None
                    }
                })
                .collect();
            let tokens = cmap::tokens(&data);
            let mut found = false;
            for (i, t) in tokens.iter().enumerate() {
                if t == "begincodespacerange" {
                    found = true;
                    let count = tokens
                        .get(i.wrapping_sub(1))
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(usize::MAX);
                    let end = tokens[i + 1..]
                        .iter()
                        .position(|s| s == "endcodespacerange")
                        .map(|p| p + i + 1)
                        .unwrap_or(i + 1);
                    let block = &tokens[i + 1..end];
                    if block.len() != count.saturating_mul(2)
                        || block.iter().any(|t| {
                            !t.starts_with('<')
                                || !t.ends_with('>')
                                || t[1..t.len() - 1]
                                    .chars()
                                    .filter(|c| !c.is_whitespace())
                                    .count()
                                    != 4
                        })
                    {
                        return unsupported("Composite encoding must have a two-byte codespace");
                    }
                }
            }
            if !found {
                return unsupported("Composite encoding must have a two-byte codespace");
            }
        } else {
            return unsupported("Unsupported predefined composite encoding");
        }
    }
    let mapping: BTreeMap<_, _> = codes
        .into_iter()
        .filter_map(|(c, gid)| {
            if c <= 65535 {
                g.mapping.get(&gid).map(|s| (c, s.clone()))
            } else {
                None
            }
        })
        .collect();
    let reason = if mapping.is_empty() {
        Some("Embedded font has no usable Unicode mapping".into())
    } else {
        None
    };
    Ok((mapping, Some(kind), true, reason))
}
#[cfg(test)]
thread_local! {
    /// Font inspections on this thread (tests prove a repair from an analysis inspects nothing again).
    static INSPECTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
fn inspect(pdf: &mut Pdf, r: &Obj, f: &Obj) -> (FontReport, BTreeMap<u32, String>) {
    #[cfg(test)]
    INSPECTIONS.with(|c| c.set(c.get() + 1));
    let mut report = FontReport {
        font_name: if f.has("BaseFont") {
            pdf.get(f, "BaseFont").unwrap_or(Obj::Null).name()
        } else if f.has("Name") {
            pdf.get(f, "Name").unwrap_or(Obj::Null).name()
        } else {
            "unknown".into()
        },
        subtype: if f.has("Subtype") {
            f.get("Subtype").name()
        } else {
            "unknown".into()
        },
        embedded: false,
        font_program_kind: None,
        has_tounicode: f.has("ToUnicode"),
        codes: 0,
        disagreeing_codes: 0,
        greek_codes: 0,
        affected_greek_codes: 0,
        unmapped_codes: 0,
        category: "unsupported".into(),
        repair_possible: false,
        reason: None,
        error: None,
        object_id: r.id(),
    };
    let res = (|| -> Result<BTreeMap<u32, String>> {
        let descendant = if report.subtype == "Type0" {
            descendant(pdf, f)?
        } else {
            f.clone()
        };
        let d = pdf.get(&descendant, "FontDescriptor")?;
        for key in ["FontFile2", "FontFile3", "FontFile"] {
            if d.has(key) {
                report.embedded = true;
                let p = pdf.get(&d, key)?;
                report.font_program_kind = Some(format!(
                    "{key}{}",
                    if key == "FontFile3" {
                        format!("/{}", p.get("Subtype").name())
                    } else {
                        String::new()
                    }
                ));
                break;
            }
        }
        let old = if report.has_tounicode {
            cmap::text_entries(&decode(&pdf.get(f, "ToUnicode")?)?)?
        } else {
            BTreeMap::new()
        };
        report.codes = old.len();
        let max = if report.subtype == "Type0" {
            65535
        } else {
            255
        };
        if old.keys().any(|n| *n > max) {
            bail!("ToUnicode source codes exceed the supported codespace")
        }
        let (expected, kind, embedded, reason) = embedded(pdf, f)?;
        report.font_program_kind = kind;
        report.embedded = embedded;
        report.reason = reason;
        report.greek_codes = expected.values().filter(|s| font::greek(s)).count();
        report.unmapped_codes = old.keys().filter(|n| !expected.contains_key(n)).count();
        let bad = old
            .iter()
            .filter(|(n, s)| expected.get(n).map(|v| v != *s).unwrap_or(false))
            .collect::<Vec<_>>();
        report.disagreeing_codes = bad.len();
        report.affected_greek_codes = bad
            .iter()
            .filter(|(n, s)| font::greek(s) || font::greek(&expected[n]))
            .count();
        if report.subtype == "Type3" {
            report.reason = Some("Type3 glyph programs are unsupported".into())
        } else if !report.embedded {
            report.category = "not_embedded".into()
        } else if expected.is_empty() {
            report.category = if report.has_tounicode {
                "unsupported"
            } else {
                "no_tounicode"
            }
            .into()
        } else if !report.has_tounicode {
            report.category = "no_tounicode".into();
            report.codes = expected.len();
            report.repair_possible = true
        } else if !bad.is_empty() || old.is_empty() {
            report.category = "repairable".into();
            report.repair_possible = true
        } else if report.unmapped_codes == old.len() {
            report.reason = Some("No existing ToUnicode code has a usable embedded mapping".into())
        } else {
            report.category = "consistent".into()
        }
        if report.repair_possible && report.object_id.is_none() {
            report.category = "unsupported_direct".into();
            report.repair_possible = false;
            report.reason = Some("Direct font dictionaries cannot be updated in isolation".into())
        }
        let mut merged = old;
        merged.extend(expected);
        Ok(merged)
    })();
    match res {
        Ok(m) => (report, m),
        Err(e) => {
            report.error = Some(error(e));
            report.reason = Some("Broken or unsupported font/CMap".into());
            (report, BTreeMap::new())
        }
    }
}
pub fn document_category(fonts: &[FontReport], has_text: bool) -> String {
    if !has_text {
        return "no_text_layer".into();
    }
    if fonts.iter().any(|f| f.repair_possible && f.greek_codes > 0) {
        return "repairable".into();
    }
    let c = fonts
        .iter()
        .map(|f| {
            if ["no_tounicode", "repairable", "unsupported_direct"].contains(&f.category.as_str()) {
                "unsupported"
            } else {
                f.category.as_str()
            }
        })
        .collect::<BTreeSet<_>>();
    if c.len() == 1 {
        c.iter().next().unwrap().to_string()
    } else if c.is_empty() {
        "unsupported".into()
    } else {
        "mixed".into()
    }
}
/// One document's font analysis, kept so `repair_with_analysis` neither re-parses fonts nor re-samples text.
/// It holds the parsed source bytes; a repair refuses a source whose bytes no longer match them.
pub struct Analysis {
    path: String,
    pdf: Pdf,
    fonts: Vec<(Obj, FontReport, BTreeMap<u32, String>)>,
    has_text: bool,
}
impl Analysis {
    pub fn path(&self) -> &str {
        &self.path
    }
}
/// The text-presence sample decided natively from content-stream text operators (no PDFium callback).
fn native_sample(pdf: &mut Pdf, _path: &str) -> Result<Sample> {
    text::native_sample(pdf)
}
/// Analyse a document once with the native text sample. The `Analysis` is `None` exactly when the report
/// carries an error.
pub fn analyze_for_repair(path: &str) -> (PdfFontReport, Option<Analysis>) {
    analyze_inner(path, native_sample)
}
fn analyze_inner(
    path: &str,
    sampler: impl FnOnce(&mut Pdf, &str) -> Result<Sample>,
) -> (PdfFontReport, Option<Analysis>) {
    let mut r = PdfFontReport {
        path: path.into(),
        category: "unsupported".into(),
        pages: 0,
        sampled_pages: vec![],
        fonts: vec![],
        error: None,
        os_error: None,
    };
    let res = (|| -> Result<Analysis> {
        let mut pdf = Pdf::open(path)?;
        let sample = sampler(&mut pdf, path)?;
        r.pages = sample.pages;
        r.sampled_pages = sample.sampled_pages;
        let mut fonts = vec![];
        for (reference, f) in pdf.fonts()? {
            let (report, mut mapping) = inspect(&mut pdf, &reference, &f);
            if !report.repair_possible {
                mapping.clear() // only repairable fonts need their mapping again
            }
            r.fonts.push(report.clone());
            fonts.push((reference, report, mapping));
        }
        r.category = document_category(&r.fonts, sample.has_text);
        Ok(Analysis {
            path: path.into(),
            pdf,
            fonts,
            has_text: sample.has_text,
        })
    })();
    match res {
        Ok(a) => (r, Some(a)),
        Err(e) => {
            r.os_error = os_error(&e);
            r.error = Some(error(e));
            (r, None)
        }
    }
}
/// Analysis with a caller-supplied text sample (an external renderer, or fixed samples in tests).
pub fn analyze_with_sample(
    path: &str,
    sampler: impl FnOnce(&str) -> Result<Sample>,
) -> PdfFontReport {
    analyze_inner(path, |_, p| sampler(p)).0
}
pub fn analyze(path: &str) -> PdfFontReport {
    analyze_for_repair(path).0
}
fn resolved(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        Ok(std::fs::canonicalize(path)?)
    } else {
        let parent = path.parent().unwrap_or(Path::new("."));
        let parent = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        Ok(resolved(parent)?.join(
            path.file_name()
                .ok_or_else(|| anyhow!("Invalid output pathname"))?,
        ))
    }
}
pub fn protect(target: &Path, sources: &[PathBuf]) -> Result<()> {
    let target_resolved = resolved(target)?;
    for source in sources {
        if target_resolved == resolved(source)? {
            bail!(
                "Output would overwrite source PDF: {} aliases {}",
                target.display(),
                source.display()
            )
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if let (Ok(a), Ok(b)) = (std::fs::metadata(target), std::fs::metadata(source)) {
                if a.dev() == b.dev() && a.ino() == b.ino() {
                    bail!(
                        "Output would overwrite source PDF (same inode/hard link): {} aliases {}",
                        target.display(),
                        source.display()
                    )
                }
            }
        }
    }
    Ok(())
}
/// Repair with a caller-supplied text sample (an external renderer, or fixed samples in tests).
pub fn repair_with_sample(
    src: &str,
    dst: &str,
    sources: &[PathBuf],
    sampler: impl FnOnce(&str) -> Result<Sample>,
) -> RepairResult {
    repair_inner(src, dst, sources, |_, p| sampler(p))
}
fn repair_inner(
    src: &str,
    dst: &str,
    sources: &[PathBuf],
    sampler: impl FnOnce(&mut Pdf, &str) -> Result<Sample>,
) -> RepairResult {
    let mut owned = None;
    repair_core(src, dst, sources, |src| {
        let mut pdf = Pdf::open(src)?;
        let sample = sampler(&mut pdf, src)?;
        let mut fonts = vec![];
        for (reference, f) in pdf.fonts()? {
            let (report, mapping) = inspect(&mut pdf, &reference, &f);
            fonts.push((reference, report, mapping));
        }
        Ok(owned.insert(Analysis {
            path: src.into(),
            pdf,
            fonts,
            has_text: sample.has_text,
        }))
    })
}
/// Repair from a prior `analyze_for_repair` of the same source: no second font inspection or text sample.
/// Refused (as a result error) when `src` is not the analysed path or its bytes changed since the analysis.
pub fn repair_with_analysis(
    src: &str,
    dst: &str,
    sources: &[PathBuf],
    analysis: &mut Analysis,
) -> RepairResult {
    repair_core(src, dst, sources, |src| {
        if src != analysis.path {
            bail!("Repair source {src} is not the analysed PDF {}", analysis.path)
        }
        if std::fs::read(src)? != analysis.pdf.bytes {
            bail!("Source PDF changed since its analysis: {src}")
        }
        Ok(analysis)
    })
}
fn repair_core<'a>(
    src: &str,
    dst: &str,
    sources: &[PathBuf],
    load: impl FnOnce(&str) -> Result<&'a mut Analysis>,
) -> RepairResult {
    let mut r = RepairResult {
        src: src.into(),
        dst: dst.into(),
        category: "unsupported".into(),
        repaired_fonts: 0,
        fonts: vec![],
        written: false,
        error: None,
        os_error: None,
    };
    let mut temp = None;
    let res = (|| -> Result<()> {
        let mut sources = sources.to_vec();
        sources.push(src.into());
        let target = Path::new(dst);
        protect(target, &sources)?;
        let analysis = load(src)?;
        let pdf = &mut analysis.pdf;
        let mut changes = BTreeMap::new();
        let mut next = pdf.size();
        for (reference, report, m) in &analysis.fonts {
            let report = report.clone();
            if report.repair_possible {
                let id = report
                    .object_id
                    .ok_or_else(|| anyhow!("Direct font dictionary"))?;
                let stream = cmap::make(m, if report.subtype == "Type0" { 2 } else { 1 })?;
                changes.insert(
                    (next, 0),
                    serialize(&Obj::Stream(
                        vec![("Length".into(), Obj::Num(stream.len().to_string()))],
                        std::sync::Arc::new(stream),
                    )),
                );
                let original = pdf.object_bytes(id)?;
                let updated = pdf::splice_tounicode(&original, next)?;
                let generation = if let Obj::Ref(_, g) = reference { *g } else { 0 };
                changes.insert((id, generation), updated);
                next += 1;
                r.repaired_fonts += 1;
            }
            r.fonts.push(report);
        }
        r.category = document_category(&r.fonts, analysis.has_text);
        let mut bytes = pdf.bytes.clone();
        if !changes.is_empty() {
            bytes.push(10);
            let mut offsets = vec![];
            for ((id, g), obj) in &changes {
                offsets.push((*id, *g, bytes.len()));
                bytes.extend(format!("{id} {g} obj\n").bytes());
                bytes.extend(obj);
                bytes.extend(b"\nendobj\n");
            }
            let xref = bytes.len();
            // Start at object zero to avoid pypdf's nonzero-index recovery
            // scanning malformed older entries and reviving stale ObjStm copies.
            bytes.extend(b"xref\n0 1\n0000000000 65535 f \n");
            for (id, g, off) in offsets {
                bytes.extend(format!("{id} 1\n{off:010} {g:05} n \n").bytes());
            }
            let mut tr = vec![
                ("Size".into(), Obj::Num(next.to_string())),
                ("Prev".into(), Obj::Num(pdf.prev.to_string())),
            ];
            for key in ["Root", "Info", "ID"] {
                if pdf.trailer.has(key) {
                    tr.push((key.into(), pdf.trailer.get(key)))
                }
            }
            bytes.extend(b"trailer\n");
            bytes.extend(serialize(&Obj::Dict(tr)));
            bytes.extend(format!("\nstartxref\n{xref}\n%%EOF\n").bytes());
        }
        let parent = target
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let mut file = None;
        for nonce in 0..1000 {
            let name = parent.join(format!(".font-repair-{}-{nonce}.pdf", std::process::id()));
            protect(&name, &sources)?;
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&name)
            {
                Ok(f) => {
                    temp = Some(name);
                    file = Some(f);
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let mut file = file.ok_or_else(|| anyhow!("Unable to reserve staging file"))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        protect(target, &sources)?;
        std::fs::rename(temp.as_ref().unwrap(), target)?;
        temp = None;
        r.written = true;
        Ok(())
    })();
    if let Some(temp) = temp {
        let _ = std::fs::remove_file(temp);
    }
    if let Err(e) = res {
        r.os_error = os_error(&e);
        r.error = Some(error(e))
    }
    r
}
/// Standalone repair (own analysis, native text sample).
pub fn repair_protected(src: &str, dst: &str, sources: &[PathBuf]) -> RepairResult {
    repair_inner(src, dst, sources, native_sample)
}
pub fn repair(src: &str, dst: &str) -> RepairResult {
    repair_protected(src, dst, &[])
}
/// Native API marker (INTEG-G1): `GLOSSAPI_NATIVE_API:<module>:<level>;`. Raise the level whenever the Python package
/// starts to rely on a new or changed export; `glossapi.capabilities.NATIVE_API_LEVELS` names the level it needs, and
/// reads these bytes from the built binary without importing it. Level 1 (K4g fix 3): an OS-level I/O failure raises
/// `OSError` with its errno instead of returning a `ValueError` text; the repair selector does not use an older build.
/// Level 2 (K5-A2): the package's Markdown uses layout v2 (`layout_v2_read_document`, `LayoutV2ContextBuilder`,
/// `layout_v2_input_problem`, `layout_v2_label_page`); `glossapi.markdown_layout` refuses an older build and emits
/// unlabelled Markdown, and the repair selector uses the Python reference.
/// Exported to Python as `__glossapi_native_api__`, which also keeps the bytes in the binary.
pub const NATIVE_API_MARKER: &str = "GLOSSAPI_NATIVE_API:glossapi_rs_pdfrepair:2;";
#[cfg(all(feature = "python", not(test)))]
mod python {
    use super::*;
    use pyo3::exceptions::{PyOSError, PyValueError};
    use pyo3::prelude::*;
    fn dict(py: Python<'_>, r: &impl Serialize) -> PyResult<Py<PyAny>> {
        let s = serde_json::to_string(r).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(py.import("json")?.getattr("loads")?.call1((s,))?.unbind())
    }
    /// An OS-level I/O failure raises `OSError(errno, message)` (Python picks the errno's subclass, e.g.
    /// FileNotFoundError); document and font problems stay in the returned report's `error` (K4g fix 3).
    fn raise_os(os_error: &Option<(i32, String)>) -> PyResult<()> {
        match os_error {
            Some((code, message)) => Err(PyOSError::new_err((*code, message.clone()))),
            None => Ok(()),
        }
    }
    /// A diagnostics error: `OSError` with its errno for an OS-level I/O failure, otherwise `ValueError`.
    fn py_err(e: anyhow::Error) -> PyErr {
        match os_error(&e) {
            Some((code, message)) => PyOSError::new_err((code, message)),
            None => PyValueError::new_err(e.to_string()),
        }
    }
    /// Opaque result of `analyze_for_repair`, accepted by `repair_pdf(..., analysis=)`.
    #[pyclass(module = "glossapi_rs_pdfrepair", frozen)]
    struct PdfAnalysis(std::sync::Mutex<Analysis>);
    #[pymethods]
    impl PdfAnalysis {
        #[getter]
        fn path(&self) -> String {
            self.0.lock().map(|a| a.path().to_string()).unwrap_or_default()
        }
    }
    #[pyfunction]
    fn analyze_pdf(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
        let report = analyze(path);
        raise_os(&report.os_error)?;
        dict(py, &report)
    }
    /// `(report dict, PdfAnalysis | None)`: the analysis is None exactly when the report has an error.
    #[pyfunction(name = "analyze_for_repair")]
    fn py_analyze_for_repair(py: Python<'_>, path: &str) -> PyResult<(Py<PyAny>, Option<Py<PdfAnalysis>>)> {
        let (report, analysis) = super::analyze_for_repair(path);
        raise_os(&report.os_error)?;
        let handle = match analysis {
            Some(a) => Some(Py::new(py, PdfAnalysis(std::sync::Mutex::new(a)))?),
            None => None,
        };
        Ok((dict(py, &report)?, handle))
    }
    #[pyfunction]
    #[pyo3(signature=(src,dst,protected_sources=Vec::new(),analysis=None))]
    fn repair_pdf(
        py: Python<'_>,
        src: &str,
        dst: &str,
        protected_sources: Vec<String>,
        analysis: Option<Bound<'_, PdfAnalysis>>,
    ) -> PyResult<Py<PyAny>> {
        let sources = protected_sources
            .into_iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        let result = match analysis {
            Some(a) => {
                let mut guard = a
                    .get()
                    .0
                    .lock()
                    .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("PdfAnalysis lock poisoned"))?;
                repair_with_analysis(src, dst, &sources, &mut guard)
            }
            None => repair_protected(src, dst, &sources),
        };
        raise_os(&result.os_error)?;
        dict(py, &result)
    }
    /// Diagnostics for the equality gates: per-page native text presence for every page.
    #[pyfunction]
    fn _native_page_flags_json(path: &str) -> PyResult<String> {
        let mut pdf = Pdf::open(path).map_err(py_err)?;
        let s = text::page_flags(&mut pdf).map_err(py_err)?;
        serde_json::to_string(&s).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    /// Diagnostics for the equality gates: the native text sample, shaped like `_sample_json`'s JSON.
    #[pyfunction]
    fn _native_sample_json(path: &str) -> PyResult<String> {
        let mut pdf = Pdf::open(path).map_err(py_err)?;
        let s = text::native_sample(&mut pdf).map_err(py_err)?;
        serde_json::to_string(&s).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    // ---- layout v2 (K5-A1, wired into the safe Markdown in K5-A2 through glossapi.markdown_layout) -------------------
    /// Document context of layout v2 (body size, heading sizes, repeated furniture), from `layout_v2_context`.
    #[pyclass(module = "glossapi_rs_pdfrepair", frozen)]
    struct LayoutV2Context(layout::DocContext);
    #[pymethods]
    impl LayoutV2Context {
        #[getter]
        fn body(&self) -> f64 {
            self.0.body
        }
        #[getter]
        fn heading_sizes(&self) -> Vec<f64> {
            self.0.heading_sizes.clone()
        }
        #[getter]
        fn pages(&self) -> usize {
            self.0.pages
        }
        /// `[{band, signature, y, pages}]` as the reference's `DocContext.furniture`.
        #[getter]
        fn furniture<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyList>> {
            let out = pyo3::types::PyList::empty(py);
            for f in &self.0.furniture {
                let d = pyo3::types::PyDict::new(py);
                d.set_item("band", f.band.name())?;
                d.set_item("signature", &f.signature)?;
                d.set_item("y", f.y)?;
                d.set_item("pages", f.pages)?;
                out.append(d)?;
            }
            Ok(out)
        }
    }
    fn layout_err(e: anyhow::Error) -> PyErr {
        PyValueError::new_err(format!("layout v2: {e}"))
    }
    /// `layout_v2_context([(page_no, k5chars bytes), ...])`: the reference's document pass (each page's own fixture
    /// text, pages in the given order; the reference's arm sorts them by page number). The GIL is released.
    #[pyfunction]
    fn layout_v2_context(py: Python<'_>, tables: Vec<(i64, Bound<'_, pyo3::types::PyBytes>)>) -> PyResult<LayoutV2Context> {
        let raw: Vec<(i64, &[u8])> = tables.iter().map(|(p, b)| (*p, b.as_bytes())).collect();
        let ctx = py
            .detach(|| -> anyhow::Result<layout::DocContext> {
                let mut geos = Vec::with_capacity(raw.len());
                for (p, data) in &raw {
                    let t = layout::k5chars::parse(data)?;
                    geos.push((*p, layout::page_geo(&layout::fixture_text(&t), &t)?));
                }
                layout::document_context(&geos)
            })
            .map_err(layout_err)?;
        Ok(LayoutV2Context(ctx))
    }
    /// `LayoutV2ContextBuilder()`: the document pass fed one page at a time (`add(page_no, k5chars bytes)` in
    /// page-number order, then `finish()`); equal to `layout_v2_context` over the same pages in the same order, and
    /// it keeps no page geometry (K5-A2 bounded memory). The GIL is released while a page is added.
    #[pyclass(module = "glossapi_rs_pdfrepair")]
    struct LayoutV2ContextBuilder(layout::ContextBuilder);
    #[pymethods]
    impl LayoutV2ContextBuilder {
        #[new]
        fn new() -> Self {
            LayoutV2ContextBuilder(layout::ContextBuilder::default())
        }
        fn add(&mut self, py: Python<'_>, page_no: i64, table: Bound<'_, pyo3::types::PyBytes>) -> PyResult<()> {
            let data = table.as_bytes();
            let b = &mut self.0;
            py.detach(|| -> anyhow::Result<()> {
                let t = layout::k5chars::parse(data)?;
                let g = layout::page_geo(&layout::fixture_text(&t), &t)?;
                b.add(page_no, &g)
            })
            .map_err(layout_err)
        }
        #[getter]
        fn added(&self) -> usize {
            self.0.added()
        }
        fn finish(&self) -> PyResult<LayoutV2Context> {
            Ok(LayoutV2Context(self.0.finish().map_err(layout_err)?))
        }
    }
    /// `layout_v2_input_problem(k5chars bytes) -> str | None`: why the table is outside the exact-parity contract
    /// (K5A1-G3: a non-finite or oversized page box, a non-finite font size or char angle), or `None`.
    #[pyfunction]
    fn layout_v2_input_problem(table: Bound<'_, pyo3::types::PyBytes>) -> PyResult<Option<String>> {
        let t = layout::k5chars::parse(table.as_bytes()).map_err(layout_err)?;
        Ok(layout::input_problem(&t))
    }
    /// `layout_v2_label_page(plain, k5chars bytes, ctx, page_no) -> (rotation, warnings, lines, blocks, markdown)`.
    /// lines: `[(start, end)]` code-point offsets of the line pieces; blocks: `[(start, end, label, level, rotated,
    /// why, cut, first, last, bbox)]` with code-point offsets into `plain` (bbox `None` or `(x0, top, x1, bottom)`);
    /// warnings: `[{"kind": ..., <key>: <value>}]` as the reference. Labels only: the text is the caller's `plain`.
    /// The GIL is released while labelling.
    #[pyfunction]
    #[allow(clippy::type_complexity)]
    fn layout_v2_label_page<'py>(
        py: Python<'py>,
        plain: &str,
        table: Bound<'py, pyo3::types::PyBytes>,
        ctx: Bound<'py, LayoutV2Context>,
        page_no: i64,
    ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let data = table.as_bytes();
        let c = &ctx.get().0;
        let (res, md) = py
            .detach(|| -> anyhow::Result<(layout::PageLabels, String)> {
                let t = layout::k5chars::parse(data)?;
                let res = layout::label_page(plain, &t, c, page_no)?;
                let md = layout::page_markdown(plain, &res.blocks);
                Ok((res, md))
            })
            .map_err(layout_err)?;
        let warnings = pyo3::types::PyList::empty(py);
        for w in &res.warnings {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("kind", w.kind)?;
            d.set_item(w.key, w.value)?;
            warnings.append(d)?;
        }
        let lines = pyo3::types::PyList::new(py, res.lines.iter().map(|&(a, b)| (a, b)))?;
        let blocks = pyo3::types::PyList::empty(py);
        for b in &res.blocks {
            let bbox = b.bbox.map(|v| (v[0], v[1], v[2], v[3]));
            blocks.append((b.start, b.end, b.label, b.level, b.rotated, b.why, b.cut, b.first, b.last, bbox))?;
        }
        (res.rotation, warnings, lines, blocks, md).into_pyobject(py)
    }
    /// Test hook for the differential tests: Python-semantics primitives of the port. kind = a pattern name (the
    /// boolean the reference computes), "norm_sig", "nfc", "casefold" (strings), or "isspace", "isalpha", "isdecimal",
    /// "isdigit", "isnumeric", "isalnum", "isupper", "re_lower" (one entry per char).
    #[pyfunction]
    fn _layout_v2_probe(py: Python<'_>, kind: &str, text: &str) -> PyResult<Py<PyAny>> {
        use layout::pyuni as u;
        let chars: Vec<char> = text.chars().collect();
        let s = |v: Vec<char>| -> PyResult<Py<PyAny>> { Ok(v.into_iter().collect::<String>().into_pyobject(py)?.into_any().unbind()) };
        let per = |f: fn(char) -> bool| -> PyResult<Py<PyAny>> {
            Ok(chars.iter().map(|&c| f(c)).collect::<Vec<bool>>().into_pyobject(py)?.into_any().unbind())
        };
        match kind {
            "norm_sig" => s(layout::norm_sig(&chars)),
            "nfc" => s(u::nfc(&chars)),
            "casefold" => s(u::casefold(&chars)),
            "isspace" => per(u::is_space),
            "isalpha" => per(u::is_alpha),
            "isdecimal" => per(u::is_decimal),
            "isdigit" => per(u::is_digit),
            "isnumeric" => per(u::is_numeric),
            "isalnum" => per(u::is_alnum),
            "isupper" => per(u::is_upper),
            "re_lower" => Ok(chars.iter().map(|&c| u::re_lower(c as u32)).collect::<Vec<u32>>().into_pyobject(py)?.into_any().unbind()),
            name => Ok(pyo3::types::PyBool::new(py, layout::pattern_probe(name, text).map_err(layout_err)?).to_owned().into_any().unbind()),
        }
    }
    /// Test hook: Python `round(x, 2)` as the port computes it.
    #[pyfunction]
    fn _layout_v2_round2(x: f64) -> f64 {
        layout::pyuni::round2(x)
    }
    // gil_used = true keeps the 0.19 contract (the module requires the GIL); PyO3 0.28+ would otherwise declare
    // free-threaded support, which has not been audited (abi3 builds cannot load on free-threaded Python anyway).
    #[pymodule(gil_used = true)]
    fn glossapi_rs_pdfrepair(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__glossapi_native_api__", NATIVE_API_MARKER)?;
        m.add_function(wrap_pyfunction!(analyze_pdf, m)?)?;
        m.add_function(wrap_pyfunction!(repair_pdf, m)?)?;
        m.add_function(wrap_pyfunction!(py_analyze_for_repair, m)?)?;
        m.add_function(wrap_pyfunction!(_native_sample_json, m)?)?;
        m.add_function(wrap_pyfunction!(_native_page_flags_json, m)?)?;
        m.add_class::<PdfAnalysis>()?;
        m.add_class::<LayoutV2Context>()?;
        m.add_function(wrap_pyfunction!(layout_v2_context, m)?)?;
        m.add_function(wrap_pyfunction!(layout_v2_label_page, m)?)?;
        m.add_class::<LayoutV2ContextBuilder>()?;
        m.add_function(wrap_pyfunction!(layout_v2_input_problem, m)?)?;
        crate::layout_reader_py::register(m)?;
        m.add_function(wrap_pyfunction!(_layout_v2_probe, m)?)?;
        m.add_function(wrap_pyfunction!(_layout_v2_round2, m)?)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn review_repair(name: &str) -> (Pdf, Pdf, RepairResult, PathBuf) {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/review/{name}.pdf"));
        // Isolate staging-file namespaces between parallel crate regressions.
        let dst = std::env::temp_dir()
            .join(format!("t11-review-{}-{name}", std::process::id()))
            .join("repaired.pdf");
        let r = repair_with_sample(src.to_str().unwrap(), dst.to_str().unwrap(), &[], |_| {
            Ok(Sample {
                pages: 1,
                sampled_pages: vec![1],
                has_text: true,
            })
        });
        assert!(r.written, "{:?}", r.error);
        (
            Pdf::open(src.to_str().unwrap()).unwrap(),
            Pdf::open(dst.to_str().unwrap()).unwrap(),
            r,
            dst,
        )
    }
    #[test]
    fn review_name_decoding_different_raw_bytes_do_not_repair() {
        let (before, after, r, dst) = review_repair("name_decoding_different_raw_byte");
        assert_eq!(r.repaired_fonts, 0);
        assert!(!r.fonts[0].repair_possible);
        assert!(r.error.is_none());
        assert!(r.fonts[0].error.is_none());
        assert_eq!(before.bytes, after.bytes);
        std::fs::remove_file(dst).unwrap();
    }
    #[test]
    fn review_name_decoding_valid_utf8_do_not_repair() {
        let (before, mut after, r, dst) = review_repair("name_decoding_valid_utf8");
        assert_eq!(r.repaired_fonts, 0);
        assert!(r.error.is_none());
        assert_eq!(before.bytes, after.bytes);
        let names = font::glyphs(
            include_bytes!("../fixtures/review/name_decoding_valid_utf8.ttf"),
            "FontFile2",
        )
        .unwrap()
        .names;
        assert_eq!(names.get(b"X\xC3\xA9Y".as_slice()), Some(&1));
        assert!(!names.contains_key(b"X\xE9Y".as_slice()));
        let parsed = pdf::Parser::new(b"/X#C3#A9Y", 0).obj().unwrap();
        assert_eq!(parsed.name_bytes(), b"X\xC3\xA9Y");
        assert_eq!(
            pdf::Parser::new(&serialize(&parsed), 0).obj().unwrap(),
            parsed
        );
        let font = after.object(r.fonts[0].object_id.unwrap()).unwrap();
        let stream = after.get(&font, "ToUnicode").unwrap();
        assert_eq!(
            cmap::text_entries(&decode(&stream).unwrap()).unwrap()[&65],
            "B"
        );
        std::fs::remove_file(dst).unwrap();
    }
    #[test]
    fn review_lowercase_agl_names_do_not_repair() {
        for name in ["lowercase_agl", "lowercase_u_agl"] {
            let (before, after, r, dst) = review_repair(name);
            assert_eq!(r.repaired_fonts, 0, "{name}");
            assert_eq!(r.fonts[0].category, "unsupported");
            assert_eq!(before.bytes, after.bytes);
            std::fs::remove_file(dst).unwrap();
        }
    }
    #[test]
    fn review_duplicate_names_repair_python_selected_gid() {
        for (name, value) in [
            ("duplicate_post_names", "β"),
            ("format3_duplicate_synth_names", "σ"),
            ("format3_collision_wrong_gid", "α"),
        ] {
            let (_, mut after, r, dst) = review_repair(name);
            assert_eq!(r.repaired_fonts, 1, "{name}");
            let font = after.object(r.fonts[0].object_id.unwrap()).unwrap();
            let stream = after.get(&font, "ToUnicode").unwrap();
            assert_eq!(
                cmap::text_entries(&decode(&stream).unwrap()).unwrap()[&65],
                value,
                "{name}"
            );
            std::fs::remove_file(dst).unwrap();
        }
    }
    // Independent scanner uses parser positions, never normalized serialization.
    fn raw_dict_without_tounicode(bytes: &[u8]) -> Vec<u8> {
        let mut p = pdf::Parser::new(bytes, 2);
        let mut out = bytes[..2].to_vec();
        loop {
            let start = p.p;
            p.skip();
            if bytes.get(p.p..p.p + 2) == Some(b">>") {
                out.extend(&bytes[start..]);
                break;
            }
            let key = p.obj().unwrap().name();
            p.obj().unwrap();
            if key != "ToUnicode" {
                out.extend(&bytes[start..p.p]);
            }
        }
        out
    }
    fn fixture_raw_dict(pdf: &Pdf, id: u32) -> Vec<u8> {
        let marker = format!("{id} 0 obj");
        let start = pdf
            .bytes
            .windows(marker.len())
            .rposition(|b| b == marker.as_bytes())
            .unwrap()
            + marker.len();
        let mut p = pdf::Parser::new(&pdf.bytes, start);
        p.skip();
        let start = p.p;
        p.obj().unwrap();
        pdf.bytes[start..p.p].to_vec()
    }
    #[test]
    fn review_escaped_pdf_name_preserves_every_font_dictionary_byte() {
        for name in [
            "encoded_pdf_name",
            "encoded_pdf_name_compressed",
            "encoded_pdf_name_missing_tounicode",
        ] {
            let (before, after, r, dst) = review_repair(name);
            assert_eq!(r.repaired_fonts, 1, "{name}");
            for font in r.fonts.iter().filter(|f| f.repair_possible) {
                let a = fixture_raw_dict(&before, font.object_id.unwrap());
                let b = fixture_raw_dict(&after, font.object_id.unwrap());
                assert_eq!(
                    raw_dict_without_tounicode(&a),
                    raw_dict_without_tounicode(&b)
                );
                assert!(b.windows(6).any(|w| w == b"/X#E9Y"));
            }
            std::fs::remove_file(dst).unwrap();
        }
    }
    fn fixture(name: &str) -> String {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("fixtures/review/{name}.pdf"))
            .to_str()
            .unwrap()
            .to_string()
    }
    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("k5a-once-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn json<T: Serialize>(v: &T) -> String {
        serde_json::to_string(v).unwrap()
    }
    #[test]
    fn repair_from_analysis_equals_two_pass_repair_and_inspects_nothing_again() {
        for name in [
            "duplicate_post_names",
            "encoded_pdf_name",
            "encoded_pdf_name_compressed",
            "encoded_pdf_name_missing_tounicode",
            "format3_collision_wrong_gid",
            "lowercase_agl",
            "name_decoding_valid_utf8",
        ] {
            let src = fixture(name);
            let d = scratch(name);
            let (a, b) = (d.join("two-pass.pdf"), d.join("once.pdf"));
            let two_pass = repair_inner(&src, a.to_str().unwrap(), &[], native_sample);
            let (report, analysis) = analyze_for_repair(&src);
            let mut analysis = analysis.expect("analysis");
            assert_eq!(json(&report), json(&analyze_with_sample(&src, |p| {
                let mut pdf = Pdf::open(p)?;
                text::native_sample(&mut pdf)
            })));
            let before = INSPECTIONS.with(|c| c.get());
            let once = repair_with_analysis(&src, b.to_str().unwrap(), &[], &mut analysis);
            assert_eq!(INSPECTIONS.with(|c| c.get()), before, "{name}: repair re-inspected fonts");
            assert_eq!(json(&once.fonts), json(&two_pass.fonts), "{name}");
            assert_eq!((once.category.as_str(), once.repaired_fonts, once.written, &once.error),
                (two_pass.category.as_str(), two_pass.repaired_fonts, two_pass.written, &two_pass.error), "{name}");
            assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap(), "{name}");
            assert_eq!(report.category, once.category, "{name}");
            std::fs::remove_dir_all(d).unwrap();
        }
    }
    #[test]
    fn repair_from_analysis_refuses_other_or_changed_sources() {
        let d = scratch("refuse");
        let src = d.join("src.pdf");
        std::fs::copy(fixture("encoded_pdf_name"), &src).unwrap();
        let src = src.to_str().unwrap().to_string();
        let (_, analysis) = analyze_for_repair(&src);
        let mut analysis = analysis.unwrap();
        let other = fixture("duplicate_post_names");
        let r = repair_with_analysis(&other, d.join("o.pdf").to_str().unwrap(), &[], &mut analysis);
        assert!(!r.written && r.error.as_deref().unwrap().contains("not the analysed PDF"), "{:?}", r.error);
        let mut bytes = std::fs::read(&src).unwrap();
        bytes.extend(b"% appended\n");
        std::fs::write(&src, bytes).unwrap();
        let r = repair_with_analysis(&src, d.join("c.pdf").to_str().unwrap(), &[], &mut analysis);
        assert!(!r.written && r.error.as_deref().unwrap().contains("changed since its analysis"), "{:?}", r.error);
        assert!(!d.join("c.pdf").exists());
        // Protection still applies before anything else.
        let r = repair_with_analysis(&src, &src, &[], &mut analysis);
        assert!(r.error.as_deref().unwrap().contains("overwrite source"), "{:?}", r.error);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn analysis_error_has_no_handle() {
        let d = scratch("bad");
        let p = d.join("bad.pdf");
        std::fs::write(&p, b"<html>503</html>").unwrap();
        let (report, analysis) = analyze_for_repair(p.to_str().unwrap());
        assert!(report.error.is_some() && analysis.is_none());
        assert!(report.os_error.is_none(), "a malformed PDF is a document error, not an OS error");
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn os_errors_keep_their_errno_and_decoder_errors_have_none() {
        // K4g fix 3: a full disk (errno 28) under context is found in the cause chain, with the error's message.
        let full = anyhow::Error::from(std::io::Error::from_raw_os_error(28)).context("writing the staged PDF");
        let (code, message) = os_error(&full).unwrap();
        assert_eq!(code, 28);
        assert_eq!(message, "writing the staged PDF");
        let bare = anyhow::Error::from(std::io::Error::from_raw_os_error(28));
        assert_eq!(os_error(&bare).unwrap().0, 28);
        assert!(bare.to_string().contains("os error 28"));
        // A decoder io::Error (no errno) and a plain document error are not OS errors.
        let inflate = anyhow::Error::from(std::io::Error::new(std::io::ErrorKind::InvalidData, "corrupt deflate stream"));
        assert!(os_error(&inflate).is_none());
        assert!(os_error(&anyhow!("Missing DescendantFonts")).is_none());
        // The serialized report never carries the field (the Python binding raises it instead).
        let r = RepairResult {
            src: "a".into(),
            dst: "b".into(),
            category: "unsupported".into(),
            repaired_fonts: 0,
            fonts: vec![],
            written: false,
            error: Some(error(full)),
            os_error: Some((28, "x".into())),
        };
        assert!(!json(&r).contains("os_error"));
    }
    #[test]
    fn real_io_failures_are_typed_in_analysis_and_repair() {
        let d = scratch("oserr");
        let missing = d.join("missing.pdf");
        let (report, analysis) = analyze_for_repair(missing.to_str().unwrap());
        assert!(analysis.is_none());
        assert_eq!(report.os_error.as_ref().map(|e| e.0), Some(2)); // ENOENT
        // The output's parent is a regular file: creating the staging directory fails with ENOTDIR (20).
        let blocker = d.join("blocker");
        std::fs::write(&blocker, b"file").unwrap();
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/review/name_decoding_valid_utf8.pdf");
        let dst = blocker.join("sub").join("repaired.pdf");
        let r = repair_with_sample(src.to_str().unwrap(), dst.to_str().unwrap(), &[], |_| {
            Ok(Sample {
                pages: 1,
                sampled_pages: vec![1],
                has_text: true,
            })
        });
        assert!(!r.written && r.error.is_some());
        assert_eq!(r.os_error.as_ref().map(|e| e.0), Some(20), "{:?}", r.error);
        std::fs::remove_dir_all(d).unwrap();
    }
    #[test]
    fn cmap_ranges() {
        let m = cmap::text_entries(
            b"2 beginbfrange <01> <02> <03B1> <03> <04> [<D83DDE00> <00660069>] endbfrange",
        )
        .unwrap();
        assert_eq!(m[&1], "α");
        assert_eq!(m[&2], "β");
        assert_eq!(m[&3], "😀");
        assert_eq!(m[&4], "fi");
        assert_eq!(cmap::text_entries(&cmap::make(&m, 1).unwrap()).unwrap(), m);
    }
    #[test]
    fn conflicts_and_inheritance() {
        assert!(cmap::text_entries(b"2 beginbfchar <01> <0061> <01> <0062> endbfchar").is_err());
        assert!(cmap::text_entries(b"/Foo usecmap").is_err())
    }
    #[test]
    fn greek_letters() {
        assert!(font::greek("ἀα"));
        assert!(!font::greek(";·"));
    }
    #[test]
    fn pdf_tokens() {
        let b = b"<< /A#20B [ 5 0 R (a\\n(b)) <123> ] >>";
        let o = pdf::Parser::new(b, 0).obj().unwrap();
        assert_eq!(pdf::Parser::new(&serialize(&o), 0).obj().unwrap(), o);
    }
    #[test]
    fn pdf_name_bytes_remain_distinct_after_unescaping() {
        let a = pdf::Parser::new(b"/X#E9Y", 0).obj().unwrap();
        let b = pdf::Parser::new(b"/X#F1Y", 0).obj().unwrap();
        assert_ne!(a, b);
        assert_eq!(a.name_bytes(), b"X\xE9Y");
        assert_eq!(b.name_bytes(), b"X\xF1Y");
        let dict = pdf::Parser::new(b"<< /X#E9Y /X#F1Y /X#F1Y /X#C3#A9Y >>", 0)
            .obj()
            .unwrap();
        assert_eq!(dict.dict().len(), 2);
        assert_eq!(dict.dict()[0].0, b"X\xE9Y");
        assert_eq!(dict.dict()[1].0, b"X\xF1Y");
        assert_eq!(pdf::Parser::new(&serialize(&dict), 0).obj().unwrap(), dict);
    }
}
