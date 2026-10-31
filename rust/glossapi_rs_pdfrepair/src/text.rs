//! Native text-presence sample: the same pages PDFium's `_sample_json` callback samples, decided from the
//! content-stream text-showing operators (Tj, TJ, ', ") instead of a PDFium text page.
//!
//! Page order and count follow PDFium (`CountPages` trusts a positive /Count; leaves are dictionaries without
//! /Kids). A string shows text when one of its codes maps (ToUnicode, else the code itself, as PDFium falls back)
//! to a character that Python's `str.strip()` would keep. Form XObjects are followed through `Do`, inline images
//! are skipped, and text render mode is ignored (PDFium's text page includes invisible text).
use crate::cmap;
use crate::pdf::{decode, Obj, Parser, Pdf};
use crate::Sample;
use anyhow::{bail, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Content-stream decode as PDFium does it: a damaged FlateDecode stream yields the bytes inflated before the
/// damage instead of nothing. Other filter chains use the strict decoder.
fn decode_content(o: &Obj) -> Result<Vec<u8>> {
    let f = o.get("Filter");
    let flate = match &f {
        Obj::Name(_) => matches!(f.name().as_str(), "FlateDecode" | "Fl"),
        Obj::Array(a) => a.len() == 1 && matches!(a[0].name().as_str(), "FlateDecode" | "Fl"),
        _ => false,
    };
    let params = o.get("DecodeParms");
    let predictor = match &params {
        Obj::Array(a) => a.first().map(|p| p.get("Predictor").int()).unwrap_or(0),
        p => p.get("Predictor").int(),
    };
    match o {
        Obj::Stream(_, b) if flate && predictor <= 1 => {
            // Small input steps, so an error loses at most the output of the failing step.
            let mut z = flate2::Decompress::new(true);
            let mut out = Vec::with_capacity(b.len() * 4);
            let mut pos = 0;
            loop {
                if out.capacity() - out.len() < 4096 {
                    out.reserve(4096)
                }
                let (i, o) = (z.total_in(), z.total_out());
                let end = (pos + 256).min(b.len());
                let r = z.decompress_vec(&b[pos..end], &mut out, flate2::FlushDecompress::None);
                pos += (z.total_in() - i) as usize;
                match r {
                    Ok(flate2::Status::StreamEnd) | Err(_) => break,
                    Ok(_) if z.total_in() == i && z.total_out() == o => break,
                    Ok(_) => {}
                }
            }
            Ok(out)
        }
        _ => decode(o),
    }
}

const MAX_FORM_DEPTH: usize = 32;
const PDFIUM_MAX_PAGES: usize = 1_048_576;

/// Python `str.isspace()` for one code point (the characters `str.strip()` removes).
pub fn py_space(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0D
            | 0x1C..=0x20
            | 0x85
            | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

/// Indexes `_sample_json` samples: up to 8 evenly spaced pages, Python `round` (ties to even).
pub fn sample_indexes(count: usize) -> Vec<usize> {
    if count <= 1 {
        return (0..count).collect();
    }
    let n = count.min(8);
    let mut out: Vec<usize> = (0..n)
        .map(|i| ((i * (count - 1)) as f64 / (n - 1) as f64).round_ties_even() as usize)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

enum Leaf {
    Page(Obj, Obj),
    Missing,
}

struct Pages<'a> {
    pdf: &'a mut Pdf,
}

impl Pages<'_> {
    fn count(&mut self, node: &Obj, visited: &mut HashSet<u32>) -> Result<usize> {
        let count = node.get("Count").int();
        if count > 0 && count < PDFIUM_MAX_PAGES {
            return Ok(count);
        }
        let kids = self.pdf.resolve(&node.get("Kids"))?;
        let mut total = 0;
        for kid in kids.array() {
            let id = kid.id();
            let k = match self.pdf.resolve(&kid) {
                Ok(o @ Obj::Dict(_)) => o,
                _ => continue,
            };
            if id.map(|n| visited.contains(&n)).unwrap_or(false) {
                continue;
            }
            if k.has("Kids") {
                if let Some(n) = id {
                    visited.insert(n);
                }
                total += self.count(&k, visited)?;
                if let Some(n) = id {
                    visited.remove(&n);
                }
            } else {
                total += 1
            }
        }
        Ok(total)
    }
    fn leaves(
        &mut self,
        node: &Obj,
        node_id: Option<u32>,
        inherited: Obj,
        out: &mut Vec<Leaf>,
        path: &mut HashSet<u32>,
        limit: usize,
    ) -> Result<()> {
        let kids = self.pdf.resolve(&node.get("Kids"))?;
        for kid in kids.array() {
            if out.len() >= limit {
                return Ok(());
            }
            let id = kid.id();
            let k = match self.pdf.resolve(&kid) {
                Ok(o @ Obj::Dict(_)) => o,
                _ => {
                    out.push(Leaf::Missing);
                    continue;
                }
            };
            if id.is_some() && id == node_id {
                continue;
            }
            let resources = if k.has("Resources") {
                k.get("Resources")
            } else {
                inherited.clone()
            };
            if k.has("Kids") {
                if let Some(n) = id {
                    if !path.insert(n) {
                        continue;
                    }
                }
                self.leaves(&k, id, resources, out, path, limit)?;
                if let Some(n) = id {
                    path.remove(&n);
                }
            } else {
                out.push(Leaf::Page(k, resources))
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct FontMap {
    two_byte: bool,
    unicode: Option<BTreeMap<u32, String>>,
}

struct Scanner<'a> {
    pdf: &'a mut Pdf,
    fonts: HashMap<String, std::rc::Rc<FontMap>>,
}

fn is_number_start(c: u8) -> bool {
    c.is_ascii_digit() || matches!(c, b'+' | b'-' | b'.')
}

impl Scanner<'_> {
    fn font(&mut self, resources: &Obj, name: &[u8]) -> std::rc::Rc<FontMap> {
        let fonts = self.pdf.get(resources, "Font").unwrap_or(Obj::Null);
        let entry = fonts
            .dict()
            .into_iter()
            .find(|(k, _)| k.as_slice() == name)
            .map(|(_, v)| v);
        let Some(r) = entry else {
            return std::rc::Rc::new(FontMap::default());
        };
        let key = format!("{r:?}");
        if let Some(f) = self.fonts.get(&key) {
            return f.clone();
        }
        let f = self.pdf.resolve(&r).unwrap_or(Obj::Null);
        let map = FontMap {
            two_byte: f.get("Subtype").name() == "Type0",
            unicode: self
                .pdf
                .get(&f, "ToUnicode")
                .ok()
                .and_then(|s| decode(&s).ok())
                .and_then(|d| cmap::text_entries(&d).ok()),
        };
        let map = std::rc::Rc::new(map);
        self.fonts.insert(key, map.clone());
        map
    }
    fn shows_text(font: &FontMap, s: &[u8]) -> bool {
        let codes: Vec<u32> = if font.two_byte {
            s.chunks(2)
                .map(|c| {
                    if c.len() == 2 {
                        u16::from_be_bytes([c[0], c[1]]) as u32
                    } else {
                        c[0] as u32
                    }
                })
                .collect()
        } else {
            s.iter().map(|b| *b as u32).collect()
        };
        codes.into_iter().any(|code| {
            let mapped = font
                .unicode
                .as_ref()
                .and_then(|m| m.get(&code))
                .filter(|v| !v.is_empty());
            match mapped {
                Some(v) => v.chars().any(|c| !py_space(c)),
                // PDFium falls back to the character code when a code has no Unicode value; code 0 adds nothing.
                None => code != 0 && char::from_u32(code).map(|c| !py_space(c)).unwrap_or(true),
            }
        })
    }
    fn content(&mut self, contents: &Obj) -> Vec<u8> {
        let c = self.pdf.resolve(contents).unwrap_or(Obj::Null);
        let streams = if let Obj::Array(a) = c {
            a
        } else {
            vec![contents.clone()]
        };
        let mut out = vec![];
        for s in streams {
            if let Ok(o @ Obj::Stream(..)) = self.pdf.resolve(&s) {
                if let Ok(d) = decode_content(&o) {
                    out.extend(d);
                    out.push(b'\n');
                }
            }
        }
        out
    }
    /// True when the stream (or a Form XObject it draws) shows at least one non-whitespace character.
    fn scan(&mut self, data: &[u8], resources: &Obj, forms: &mut Vec<u32>) -> bool {
        let resources = self.pdf.resolve(resources).unwrap_or(Obj::Null);
        let mut p = Parser::new(data, 0);
        let mut operands: Vec<Obj> = vec![];
        let mut font = std::rc::Rc::new(FontMap::default());
        let mut saved: Vec<std::rc::Rc<FontMap>> = vec![];
        loop {
            p.skip();
            let Some(&c) = data.get(p.p) else {
                return false;
            };
            if matches!(c, b'/' | b'(' | b'<' | b'[') {
                match p.obj() {
                    Ok(o) => operands.push(o),
                    Err(_) => return false,
                }
                continue;
            }
            if matches!(c, b']' | b'>' | b')' | b'{' | b'}') {
                p.p += 1;
                continue;
            }
            let Ok(word) = p.word() else {
                p.p += 1;
                continue;
            };
            if is_number_start(c) {
                operands.push(Obj::Num(word));
                continue;
            }
            match word.as_str() {
                "true" | "false" | "null" => {
                    operands.push(Obj::Null);
                    continue;
                }
                "q" => saved.push(font.clone()),
                "Q" => {
                    if let Some(f) = saved.pop() {
                        font = f
                    }
                }
                "Tf" => {
                    if operands.len() >= 2 {
                        if let Obj::Name(n) = &operands[operands.len() - 2] {
                            font = self.font(&resources, n)
                        }
                    }
                }
                "Tj" | "'" | "\"" => {
                    if let Some(Obj::Str(s)) = operands.last() {
                        if Self::shows_text(&font, s) {
                            return true;
                        }
                    }
                }
                "TJ" => {
                    if let Some(Obj::Array(a)) = operands.last() {
                        for item in a {
                            if let Obj::Str(s) = item {
                                if Self::shows_text(&font, s) {
                                    return true;
                                }
                            }
                        }
                    }
                }
                "Do" => {
                    if let Some(Obj::Name(n)) = operands.last() {
                        if self.form(&resources, n, forms) {
                            return true;
                        }
                    }
                }
                "BI" if !skip_inline_image(&mut p) => return false,
                _ => {}
            }
            operands.clear();
        }
    }
    fn form(&mut self, resources: &Obj, name: &[u8], forms: &mut Vec<u32>) -> bool {
        let xs = self.pdf.get(resources, "XObject").unwrap_or(Obj::Null);
        let Some(r) = xs
            .dict()
            .into_iter()
            .find(|(k, _)| k.as_slice() == name)
            .map(|(_, v)| v)
        else {
            return false;
        };
        let Some(id) = r.id() else {
            return false;
        };
        if forms.contains(&id) || forms.len() >= MAX_FORM_DEPTH {
            return false;
        }
        let x = match self.pdf.resolve(&r) {
            Ok(o @ Obj::Stream(..)) => o,
            _ => return false,
        };
        if x.get("Subtype").name() != "Form" {
            return false;
        }
        let Ok(data) = decode_content(&x) else {
            return false;
        };
        // PDFium falls back to the parent resources when a form has none.
        let res = if x.has("Resources") {
            x.get("Resources")
        } else {
            resources.clone()
        };
        forms.push(id);
        let found = self.scan(&data, &res, forms);
        forms.pop();
        found
    }
}

/// Skip `BI <dict> ID <data> EI`; the parser stands after `BI`. Returns false when no end marker exists.
fn skip_inline_image(p: &mut Parser) -> bool {
    loop {
        p.skip();
        if p.p >= p.b.len() {
            return false;
        }
        let c = p.b[p.p];
        if matches!(c, b'/' | b'(' | b'<' | b'[') {
            if p.obj().is_err() {
                return false;
            }
            continue;
        }
        match p.word() {
            Ok(w) if w == "ID" => break,
            Ok(_) => continue,
            Err(_) => p.p += 1,
        }
    }
    // One whitespace byte follows ID; the data ends at whitespace + EI + (whitespace | end).
    p.p += 1;
    let b = p.b;
    let mut i = p.p;
    while i + 1 < b.len() {
        if b[i] == b'E'
            && b[i + 1] == b'I'
            && i > 0
            && matches!(b[i - 1], 0 | 9 | 10 | 12 | 13 | 32)
            && b.get(i + 2)
                .map(|c| matches!(c, 0 | 9 | 10 | 12 | 13 | 32))
                .unwrap_or(true)
        {
            p.p = i + 2;
            return true;
        }
        i += 1;
    }
    false
}

/// The `Sample` `_sample_json` returns, computed without PDFium. PDFium refuses to load a document without
/// pages (the callback then fails), so a document without pages is an error here too.
pub fn native_sample(pdf: &mut Pdf) -> Result<Sample> {
    let root = pdf.get(&pdf.trailer.clone(), "Root")?;
    let pages_ref = root.get("Pages");
    let pages = pdf.resolve(&pages_ref)?;
    if !matches!(pages, Obj::Dict(_)) {
        bail!("Text sample failed: the document has no page tree")
    }
    let (count, leaves) = if !pages.has("Kids") {
        let res = pages.get("Resources");
        (1, vec![Leaf::Page(pages.clone(), res)])
    } else {
        let mut visited: HashSet<u32> = pages_ref.id().into_iter().collect();
        let count = Pages { pdf }.count(&pages, &mut visited)?;
        let mut out = vec![];
        let mut path: HashSet<u32> = pages_ref.id().into_iter().collect();
        Pages { pdf }.leaves(
            &pages,
            pages_ref.id(),
            pages.get("Resources"),
            &mut out,
            &mut path,
            count,
        )?;
        (count, out)
    };
    if count == 0 {
        bail!("Text sample failed: the document has no pages")
    }
    let indexes = sample_indexes(count);
    let mut scanner = Scanner {
        pdf,
        fonts: HashMap::new(),
    };
    let mut has_text = false;
    for &i in &indexes {
        let (page, resources) = match leaves.get(i) {
            Some(Leaf::Page(p, r)) => (p, r),
            _ => bail!("Text sample failed: page {} is unavailable", i + 1),
        };
        if has_text {
            continue;
        }
        let data = scanner.content(&page.get("Contents"));
        has_text = scanner.scan(&data, resources, &mut vec![]);
    }
    Ok(Sample {
        pages: count,
        sampled_pages: indexes.iter().map(|i| i + 1).collect(),
        has_text,
    })
}

/// Per-page text presence for every page in PDFium order (`None` for an unavailable page); for the equality
/// gates and tests, not used by analysis.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub fn page_flags(pdf: &mut Pdf) -> Result<Vec<Option<bool>>> {
    let root = pdf.get(&pdf.trailer.clone(), "Root")?;
    let pages_ref = root.get("Pages");
    let pages = pdf.resolve(&pages_ref)?;
    if !matches!(pages, Obj::Dict(_)) {
        return Ok(vec![]);
    }
    let (count, leaves) = if !pages.has("Kids") {
        let res = pages.get("Resources");
        (1, vec![Leaf::Page(pages.clone(), res)])
    } else {
        let mut visited: HashSet<u32> = pages_ref.id().into_iter().collect();
        let count = Pages { pdf }.count(&pages, &mut visited)?;
        let mut out = vec![];
        let mut path: HashSet<u32> = pages_ref.id().into_iter().collect();
        Pages { pdf }.leaves(
            &pages,
            pages_ref.id(),
            pages.get("Resources"),
            &mut out,
            &mut path,
            count,
        )?;
        (count, out)
    };
    let mut scanner = Scanner {
        pdf,
        fonts: HashMap::new(),
    };
    Ok((0..count)
        .map(|i| match leaves.get(i) {
            Some(Leaf::Page(p, r)) => {
                let data = scanner.content(&p.get("Contents"));
                Some(scanner.scan(&data, r, &mut vec![]))
            }
            _ => None,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// A classic-xref PDF from object bodies 1..=n (object 1 is the catalog).
    fn build(objects: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"%PDF-1.7\n".to_vec();
        let mut offsets = vec![];
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend(format!("{} 0 obj\n", i + 1).bytes());
            out.extend(body);
            out.extend(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
        for off in offsets {
            out.extend(format!("{off:010} 00000 n \n").bytes());
        }
        out.extend(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .bytes(),
        );
        out
    }
    fn stream(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut v = format!("<< {dict} /Length {} >>\nstream\n", data.len()).into_bytes();
        v.extend(data);
        v.extend(b"\nendstream");
        v
    }
    fn flate(data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut e = flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }
    /// One page drawing `content`; object 5 is a Helvetica font F1, object 6 an optional extra object.
    fn one_page(content: &[u8], xobjects: &str, extra: Option<Vec<u8>>) -> Vec<Vec<u8>> {
        let mut objs = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> {xobjects} >> >>").into_bytes(),
            stream("", content),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ];
        if let Some(e) = extra {
            objs.push(e)
        }
        objs
    }
    fn sample_of(objects: &[Vec<u8>], name: &str) -> Sample {
        let dir = std::env::temp_dir().join(format!("k5a-text-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.pdf");
        std::fs::write(&path, build(objects)).unwrap();
        // Set to keep the fixtures for a PDFium cross-check (K5a: `_sample_json` on the same files).
        if let Ok(keep) = std::env::var("PDFREPAIR_KEEP_TEXT_FIXTURES") {
            std::fs::create_dir_all(&keep).unwrap();
            std::fs::copy(&path, Path::new(&keep).join(format!("{name}.pdf"))).unwrap();
            let s = native_sample(&mut Pdf::open(path.to_str().unwrap()).unwrap()).unwrap();
            std::fs::write(
                Path::new(&keep).join(format!("{name}.json")),
                serde_json::to_string(&s).unwrap(),
            )
            .unwrap();
        }
        let mut pdf = Pdf::open(path.to_str().unwrap()).unwrap();
        let s = native_sample(&mut pdf).unwrap();
        let flags = page_flags(&mut pdf).unwrap();
        assert_eq!(
            s.has_text,
            s.sampled_pages.iter().any(|p| flags[p - 1] == Some(true))
        );
        std::fs::remove_dir_all(dir).unwrap();
        s
    }
    fn has_text(objects: &[Vec<u8>], name: &str) -> bool {
        sample_of(objects, name).has_text
    }

    #[test]
    fn plain_text_operators() {
        for (i, c) in [
            &b"BT /F1 12 Tf 10 10 Td (Hello) Tj ET"[..],
            b"BT /F1 12 Tf [(H) -20 (i)] TJ ET",
            b"BT /F1 12 Tf 10 100 Td 14 TL (Hi) ' ET",
            b"BT /F1 12 Tf 1 2 (Hi) \" ET",
            b"BT /F1 12 Tf <48656c6c6f> Tj ET",
        ]
        .iter()
        .enumerate()
        {
            assert!(
                has_text(&one_page(c, "", None), &format!("plain{i}")),
                "{}",
                String::from_utf8_lossy(c)
            );
        }
    }
    #[test]
    fn text_drawn_only_in_a_form_xobject() {
        let form = stream("/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Resources << /Font << /F1 5 0 R >> >>",
            b"BT /F1 12 Tf 10 10 Td (Inside a form) Tj ET");
        assert!(has_text(
            &one_page(
                b"q 1 0 0 1 0 0 cm /X0 Do Q",
                "/XObject << /X0 6 0 R >>",
                Some(form)
            ),
            "form"
        ));
        // The same form, never drawn, shows nothing.
        let form = stream(
            "/Type /XObject /Subtype /Form /BBox [0 0 200 200]",
            b"BT /F1 12 Tf (x) Tj ET",
        );
        assert!(!has_text(
            &one_page(b"q Q", "/XObject << /X0 6 0 R >>", Some(form)),
            "form-unused"
        ));
    }
    #[test]
    fn form_that_draws_itself_terminates() {
        let form = stream("/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << /XObject << /X0 6 0 R >> >>", b"/X0 Do");
        assert!(!has_text(
            &one_page(b"/X0 Do", "/XObject << /X0 6 0 R >>", Some(form)),
            "form-cycle"
        ));
    }
    #[test]
    fn invisible_text_counts_as_text() {
        // Render mode 3 (OCR text layers): PDFium's text page includes it, so the native decision does too.
        assert!(has_text(
            &one_page(b"BT 3 Tr /F1 12 Tf 10 10 Td (Hidden layer) Tj ET", "", None),
            "invisible"
        ));
    }
    #[test]
    fn empty_and_whitespace_text_objects_show_nothing() {
        for (i, c) in [
            &b"BT ET"[..],
            b"BT /F1 12 Tf ET",
            b"BT /F1 12 Tf () Tj ET",
            b"BT /F1 12 Tf (   ) Tj [( ) 120 (\\t)] TJ ET",
            b"",
        ]
        .iter()
        .enumerate()
        {
            assert!(
                !has_text(&one_page(c, "", None), &format!("empty{i}")),
                "{}",
                String::from_utf8_lossy(c)
            );
        }
    }
    #[test]
    fn inline_image_without_text() {
        // Image data containing operator-looking bytes must be skipped, not parsed.
        let c = b"q 10 0 0 10 0 0 cm BI /W 4 /H 1 /BPC 8 /CS /G ID (Tj) Tj\nEI Q";
        assert!(!has_text(&one_page(c, "", None), "inline-image"));
        let c = b"q BI /W 2 /H 1 /BPC 8 /CS /G ID \x00\xff EI Q BT /F1 9 Tf (after) Tj ET";
        assert!(has_text(&one_page(c, "", None), "inline-image-then-text"));
    }
    #[test]
    fn image_xobject_is_not_text() {
        let img = stream("/Type /XObject /Subtype /Image /Width 1 /Height 1 /BitsPerComponent 8 /ColorSpace /DeviceGray", b"\x00");
        assert!(!has_text(
            &one_page(b"q /Im0 Do Q", "/XObject << /Im0 6 0 R >>", Some(img)),
            "image-xobject"
        ));
    }
    #[test]
    fn tounicode_whitespace_and_code_fallback() {
        let cmap = b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap 1 begincodespacerange <00> <FF> endcodespacerange 2 beginbfchar <01> <0020> <02> <0041> endbfchar endcmap end end";
        let objs = |content: &[u8]| {
            vec![
                b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
                b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
                b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_vec(),
                stream("", content),
                b"<< /Type /Font /Subtype /TrueType /BaseFont /X /ToUnicode 6 0 R >>".to_vec(),
                stream("", cmap),
            ]
        };
        assert!(!has_text(&objs(b"BT /F1 9 Tf <0101> Tj ET"), "tu-space"));
        assert!(has_text(&objs(b"BT /F1 9 Tf <0102> Tj ET"), "tu-letter"));
        // An unmapped code falls back to the code itself (PDFium); code 0 adds nothing.
        assert!(has_text(&objs(b"BT /F1 9 Tf <41> Tj ET"), "tu-unmapped"));
        assert!(!has_text(&objs(b"BT /F1 9 Tf <00> Tj ET"), "tu-zero"));
    }
    #[test]
    fn damaged_flate_content_keeps_the_inflated_prefix() {
        let mut text = b"BT /F1 12 Tf (Readable before the damage) Tj ET\n".to_vec();
        for i in 0..400 {
            text.extend(format!("{} {} m {} {} l S\n", i, i * 3, i + 7, i * 5).bytes());
        }
        let mut z = flate(&text);
        let n = z.len();
        for b in &mut z[n / 2..n - 4] {
            *b = 0
        }
        let mut objs = one_page(b"", "", None);
        objs[3] = stream("/Filter /FlateDecode", &z);
        assert!(has_text(&objs, "damaged-flate"));
    }
    #[test]
    fn sampled_pages_follow_the_python_rule() {
        // Python: sorted({round(i*(n-1)/(min(n,8)-1)) for i in range(min(n,8))}) if n > 1 else list(range(n)).
        assert_eq!(sample_indexes(0), Vec::<usize>::new());
        assert_eq!(sample_indexes(1), vec![0]);
        assert_eq!(sample_indexes(5), vec![0, 1, 2, 3, 4]);
        assert_eq!(sample_indexes(10), vec![0, 1, 3, 4, 5, 6, 8, 9]);
        assert_eq!(sample_indexes(101), vec![0, 14, 29, 43, 57, 71, 86, 100]);
        assert_eq!(sample_indexes(206), vec![0, 29, 59, 88, 117, 146, 176, 205]);
    }
    #[test]
    fn page_tree_count_and_order_follow_pdfium() {
        // Nested tree, page 2 of 3 has the only text; /Count on the root is trusted.
        let objs = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 3 /Resources << /Font << /F1 8 0 R >> >> >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /Contents 6 0 R >>".to_vec(),
            b"<< /Type /Pages /Parent 2 0 R /Kids [5 0 R 9 0 R] /Count 2 >>".to_vec(),
            b"<< /Type /Page /Parent 4 0 R /Contents 7 0 R >>".to_vec(),
            stream("", b""),
            stream("", b"BT /F1 9 Tf (inherited resources) Tj ET"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
            b"<< /Type /Page /Parent 4 0 R /Contents 6 0 R >>".to_vec(),
        ];
        let s = sample_of(&objs, "tree");
        assert_eq!(
            (s.pages, s.sampled_pages.clone(), s.has_text),
            (3, vec![1, 2, 3], true)
        );
        // No /Count: leaves are counted.
        let mut objs2 = objs.clone();
        objs2[1] = b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Resources << /Font << /F1 8 0 R >> >> >>"
            .to_vec();
        objs2[3] = b"<< /Type /Pages /Parent 2 0 R /Kids [5 0 R 9 0 R] >>".to_vec();
        assert_eq!(sample_of(&objs2, "tree-nocount").pages, 3);
    }
    #[test]
    fn no_pages_is_an_error_like_pdfium() {
        let objs = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [] /Count 0 >>".to_vec(),
        ];
        let dir = std::env::temp_dir().join(format!("k5a-text-{}-nopages", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.pdf");
        std::fs::write(&path, build(&objs)).unwrap();
        let err = native_sample(&mut Pdf::open(path.to_str().unwrap()).unwrap()).unwrap_err();
        assert!(err.to_string().contains("no pages"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn off_page_text_is_counted_known_difference_from_pdfium() {
        // DECLARED DIFFERENCE (K5a gaps): PDFium's sample reads text inside the page box only
        // (get_text_range -> get_text_bounded(page bbox)); the native decision does not clip, so text drawn
        // wholly outside the page counts. No such page exists in the 5,095 compared pages.
        assert!(has_text(
            &one_page(b"BT /F1 12 Tf 10 -500 Td (Below the page) Tj ET", "", None),
            "off-page"
        ));
    }
    #[test]
    fn python_isspace_table() {
        for c in [
            '\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1f}', ' ', '\u{85}', '\u{a0}',
            '\u{2003}', '\u{3000}',
        ] {
            assert!(py_space(c), "{c:?}");
        }
        for c in ['\0', 'a', 'α', '\u{200b}', '\u{feff}', '\u{8}'] {
            assert!(!py_space(c), "{c:?}");
        }
    }
}
