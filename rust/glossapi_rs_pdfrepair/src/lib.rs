mod cmap;
mod font;
mod pdf;
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
}
#[derive(Deserialize)]
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
fn descendant(pdf: &mut Pdf, f: &Obj) -> Result<Obj> {
    let a = pdf.get(f, "DescendantFonts")?.array();
    pdf.resolve(
        a.first()
            .ok_or_else(|| anyhow!("Missing DescendantFonts"))?,
    )
}
fn embedded(
    pdf: &mut Pdf,
    f: &Obj,
) -> Result<(BTreeMap<u32, String>, Option<String>, bool, Option<String>)> {
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
fn inspect(pdf: &mut Pdf, r: &Obj, f: &Obj) -> (FontReport, BTreeMap<u32, String>) {
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
pub fn analyze_with_sample(
    path: &str,
    sampler: impl FnOnce(&str) -> Result<Sample>,
) -> PdfFontReport {
    let mut r = PdfFontReport {
        path: path.into(),
        category: "unsupported".into(),
        pages: 0,
        sampled_pages: vec![],
        fonts: vec![],
        error: None,
    };
    let res = (|| -> Result<()> {
        let mut pdf = Pdf::open(path)?;
        let sample = sampler(path)?;
        r.pages = sample.pages;
        r.sampled_pages = sample.sampled_pages;
        for (reference, f) in pdf.fonts()? {
            r.fonts.push(inspect(&mut pdf, &reference, &f).0)
        }
        r.category = document_category(&r.fonts, sample.has_text);
        Ok(())
    })();
    if let Err(e) = res {
        r.error = Some(error(e))
    }
    r
}
fn subprocess_sample(path: &str) -> Result<Sample> {
    let python = std::env::var("GLOSSAPI_PDFREPAIR_PYTHON").unwrap_or_else(|_| "python3".into());
    let out = std::process::Command::new(python)
        .args(["-m", "glossapi.pdf_repair.rust_backend", "--sample", path])
        .output()?;
    if !out.status.success() {
        bail!(
            "PDFium sample failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}
pub fn analyze(path: &str) -> PdfFontReport {
    analyze_with_sample(path, subprocess_sample)
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
pub fn repair_with_sample(
    src: &str,
    dst: &str,
    sources: &[PathBuf],
    sampler: impl FnOnce(&str) -> Result<Sample>,
) -> RepairResult {
    let mut r = RepairResult {
        src: src.into(),
        dst: dst.into(),
        category: "unsupported".into(),
        repaired_fonts: 0,
        fonts: vec![],
        written: false,
        error: None,
    };
    let mut temp = None;
    let res = (|| -> Result<()> {
        let mut sources = sources.to_vec();
        sources.push(src.into());
        let target = Path::new(dst);
        protect(target, &sources)?;
        let mut pdf = Pdf::open(src)?;
        let sample = sampler(src)?;
        let mut changes = BTreeMap::new();
        let mut next = pdf.size();
        for (reference, f) in pdf.fonts()? {
            let (report, m) = inspect(&mut pdf, &reference, &f);
            if report.repair_possible {
                let id = report
                    .object_id
                    .ok_or_else(|| anyhow!("Direct font dictionary"))?;
                let stream = cmap::make(&m, if report.subtype == "Type0" { 2 } else { 1 })?;
                changes.insert(
                    (next, 0),
                    serialize(&Obj::Stream(
                        vec![("Length".into(), Obj::Num(stream.len().to_string()))],
                        std::sync::Arc::new(stream),
                    )),
                );
                let original = pdf.object_bytes(id)?;
                let updated = pdf::splice_tounicode(&original, next)?;
                let generation = if let Obj::Ref(_, g) = reference { g } else { 0 };
                changes.insert((id, generation), updated);
                next += 1;
                r.repaired_fonts += 1;
            }
            r.fonts.push(report);
        }
        r.category = document_category(&r.fonts, sample.has_text);
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
        r.error = Some(error(e))
    }
    r
}
pub fn repair(src: &str, dst: &str) -> RepairResult {
    repair_with_sample(src, dst, &[], subprocess_sample)
}
#[cfg(all(feature = "python", not(test)))]
mod python {
    use super::*;
    use pyo3::prelude::*;
    fn sample(py: Python<'_>, path: &str) -> Result<Sample> {
        let m = py
            .import("glossapi.pdf_repair.rust_backend")
            .map_err(|e| anyhow!("{e}"))?;
        let s = m
            .getattr("_sample_json")
            .and_then(|f| f.call1((path,)))
            .and_then(|o| o.extract::<String>())
            .map_err(|e| anyhow!("{e}"))?;
        Ok(serde_json::from_str(&s)?)
    }
    fn dict(py: Python<'_>, r: &impl Serialize) -> PyResult<PyObject> {
        let s = serde_json::to_string(r)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(py.import("json")?.getattr("loads")?.call1((s,))?.into())
    }
    #[pyfunction]
    fn analyze_pdf(py: Python<'_>, path: &str) -> PyResult<PyObject> {
        dict(py, &analyze_with_sample(path, |p| sample(py, p)))
    }
    #[pyfunction]
    #[pyo3(signature=(src,dst,protected_sources=Vec::new()))]
    fn repair_pdf(
        py: Python<'_>,
        src: &str,
        dst: &str,
        protected_sources: Vec<String>,
    ) -> PyResult<PyObject> {
        dict(
            py,
            &repair_with_sample(
                src,
                dst,
                &protected_sources
                    .into_iter()
                    .map(PathBuf::from)
                    .collect::<Vec<_>>(),
                |p| sample(py, p),
            ),
        )
    }
    #[pymodule]
    fn glossapi_rs_pdfrepair(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(analyze_pdf, m)?)?;
        m.add_function(wrap_pyfunction!(repair_pdf, m)?)?;
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
