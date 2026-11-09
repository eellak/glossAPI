//! Native text-presence sample: the same pages PDFium's `_sample_json` callback samples, decided from the
//! content-stream text-showing operators (Tj, TJ, ', ") instead of a PDFium text page.
//!
//! Page order and count follow PDFium (`CountPages` trusts a positive /Count; leaves are dictionaries without
//! /Kids). A string shows text when one of its codes maps (ToUnicode, else the code itself, as PDFium falls back)
//! to a character that Python's `str.strip()` would keep. Form XObjects are followed through `Do`, inline images
//! are skipped, and text render mode is ignored (PDFium's text page includes invisible text).
use crate::cmap;
use crate::pdf::{decode, lzw_decode, run_length_decode, Obj, Parser, Pdf};
use crate::Sample;
use anyhow::{bail, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Content-stream decode as PDFium does it: a damaged FlateDecode stream yields the bytes inflated before the
/// damage instead of nothing. Other filter chains use the strict decoder.
fn decode_content(o: &Obj) -> Result<Vec<u8>> {
    decode_content_checked(o).map(|(data, _)| data)
}

/// [`decode_content`] plus what the decoder observed (K5 quality policy P4b): `Complete` when every filter
/// decoded to its end, `Partial` when a FlateDecode stream failed or ended before its end marker (the returned bytes
/// are the prefix PDFium also keeps), `Unknown` for a filter chain the strict decoder does not implement (its error
/// says nothing about the stream). A strict-decoder failure of an implemented chain is an error (`Partial` for the
/// caller).
fn decode_content_checked(o: &Obj) -> Result<(Vec<u8>, Decoded)> {
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
    // K5-D (K5A-M2): a single LZWDecode or RunLengthDecode filter keeps the prefix decoded before damage, as
    // FlateDecode does (Partial); a clean decode is Complete.
    let single = match &f {
        Obj::Name(_) => Some(f.name()),
        Obj::Array(a) if a.len() == 1 => Some(a[0].name()),
        _ => None,
    };
    let early = match &params {
        Obj::Array(a) => a.first().map(|p| p.get("EarlyChange")).unwrap_or(Obj::Null),
        p => p.get("EarlyChange"),
    };
    match o {
        Obj::Stream(_, b) if predictor <= 1 && matches!(single.as_deref(), Some("LZWDecode" | "LZW" | "RunLengthDecode" | "RL")) => {
            let (data, complete) = if matches!(single.as_deref(), Some("LZWDecode" | "LZW")) {
                lzw_decode(b, early == Obj::Null || early.int() != 0)
            } else {
                run_length_decode(b)
            };
            Ok((data, if complete { Decoded::Complete } else { Decoded::Partial }))
        }
        Obj::Stream(_, b) if flate && predictor <= 1 => {
            // K5-D (cost): an intact stream is inflated with all of its input at once; it reaches its end marker with
            // exactly the bytes the stepwise loop below produces. Anything else is decoded again by that loop, so a
            // damaged stream keeps exactly the prefix it kept before.
            if let Some(out) = inflate_whole(b) {
                return Ok((out, Decoded::Complete));
            }
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
                    Ok(flate2::Status::StreamEnd) => return Ok((out, Decoded::Complete)),
                    Err(_) => break,
                    Ok(_) if z.total_in() == i && z.total_out() == o => break,
                    Ok(_) => {}
                }
            }
            Ok((out, Decoded::Partial))
        }
        _ => {
            let filters = match &f {
                Obj::Array(a) => a.iter().map(|x| x.name()).collect::<Vec<_>>(),
                Obj::Null => vec![],
                other => vec![other.name()],
            };
            let implemented = filters
                .iter()
                .all(|n| {
                    matches!(
                        n.as_str(),
                        "FlateDecode" | "Fl" | "ASCIIHexDecode" | "AHx" | "ASCII85Decode" | "A85" | "LZWDecode" | "LZW" | "RunLengthDecode" | "RL"
                    )
                });
            match decode(o) {
                Ok(data) => Ok((data, Decoded::Complete)),
                Err(_) if !implemented => Ok((vec![], Decoded::Unknown)),
                Err(e) => Err(e),
            }
        }
    }
}

/// A zlib stream inflated in one pass, only when it reaches its end marker (`None` otherwise: error, truncation, stall).
fn inflate_whole(b: &[u8]) -> Option<Vec<u8>> {
    let mut z = flate2::Decompress::new(true);
    let mut out = Vec::with_capacity(b.len() * 4 + 64);
    loop {
        if out.capacity() - out.len() < 4096 {
            out.reserve(out.capacity().max(4096))
        }
        let (i, o) = (z.total_in(), z.total_out());
        let r = z.decompress_vec(&b[z.total_in() as usize..], &mut out, flate2::FlushDecompress::None);
        match r {
            Ok(flate2::Status::StreamEnd) => return Some(out),
            Err(_) => return None,
            Ok(_) if z.total_in() == i && z.total_out() == o => return None,
            Ok(_) => {}
        }
    }
}

/// What a content-stream decode observed; see [`decode_content_checked`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decoded {
    Complete,
    Partial,
    Unknown,
}

const MAX_FORM_DEPTH: usize = 32;
/// Resource dictionaries up to this size are checked whole before a content stream is tokenized (K5-D).
const SHORTCUT_ENTRIES: usize = 64;
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

/// Per-page decode status of the content streams (K5 quality policy P4b, K5-A2b), in PDFium page order:
/// `("complete" | "partial" | "unknown", reason)`. A page is `partial` when one of its content streams, or of the Form
/// XObjects it draws (`Do`, followed as the text sample does), or the ToUnicode CMap of a font it selects (`Tf`;
/// K5-D), failed to decode or ended before its end marker (the readable prefix is what PDFium extracts); `unknown` when the check cannot decide (an unavailable page, an
/// unreadable stream object, a filter this decoder does not implement); `complete` otherwise. A document this parser
/// cannot open (encrypted, damaged beyond its xref recovery) is an error here, and every page `unknown` for the caller.
/// PDFium's own error state is not consulted (it reports nothing for such pages).
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub fn page_decode_status(pdf: &mut Pdf) -> Result<Vec<(&'static str, Option<String>)>> {
    let root = pdf.get(&pdf.trailer.clone(), "Root")?;
    let pages_ref = root.get("Pages");
    let pages = pdf.resolve(&pages_ref)?;
    if !matches!(pages, Obj::Dict(_)) {
        bail!("Decode check failed: the document has no page tree")
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
    let mut checker = DecodeCheck { pdf, forms: vec![], checked: HashMap::new(), fonts: HashMap::new() };
    Ok((0..count)
        .map(|i| match leaves.get(i) {
            Some(Leaf::Page(p, r)) => checker.page(&p.get("Contents"), r),
            _ => ("unknown", Some("page unavailable".into())),
        })
        .collect())
}

struct DecodeCheck<'a> {
    pdf: &'a mut Pdf,
    forms: Vec<u32>,
    /// Status per XObject id, so an image or form drawn on many pages is read once (a document whose 169 pages draw
    /// the same images took 14 s without it).
    checked: HashMap<u32, (&'static str, Option<String>)>,
    /// ToUnicode status per font object id (K5-D).
    fonts: HashMap<u32, (&'static str, Option<String>)>,
}

/// The worse of two statuses: partial > unknown > complete; the first reason of the worst status is kept.
fn worse(a: (&'static str, Option<String>), b: (&'static str, Option<String>)) -> (&'static str, Option<String>) {
    let rank = |s: &str| match s {
        "partial" => 2,
        "unknown" => 1,
        _ => 0,
    };
    if rank(b.0) > rank(a.0) {
        b
    } else {
        a
    }
}

impl DecodeCheck<'_> {
    fn page(&mut self, contents: &Obj, resources: &Obj) -> (&'static str, Option<String>) {
        let c = match self.pdf.resolve(contents) {
            Ok(c) => c,
            Err(e) => return ("unknown", Some(format!("content object unreadable: {e}"))),
        };
        let streams = match c {
            Obj::Null => return ("complete", None),
            Obj::Array(a) => a,
            _ => vec![contents.clone()],
        };
        let mut status = ("complete", None);
        for s in streams {
            status = worse(status, self.stream(&s, resources, "content stream"));
            if status.0 == "partial" {
                break;
            }
        }
        status
    }
    fn stream(&mut self, s: &Obj, resources: &Obj, what: &str) -> (&'static str, Option<String>) {
        let o = match self.pdf.resolve(s) {
            Ok(o @ Obj::Stream(..)) => o,
            Ok(Obj::Null) => return ("complete", None),
            Ok(_) => return ("unknown", Some(format!("{what} is not a stream"))),
            Err(e) => return ("unknown", Some(format!("{what} unreadable: {e}"))),
        };
        let data = match decode_content_checked(&o) {
            Ok((_, Decoded::Partial)) => {
                let filter = match o.get("Filter") {
                    Obj::Array(a) if a.len() == 1 => a[0].name(),
                    f @ Obj::Name(_) => f.name(),
                    _ => String::new(),
                };
                let filter = if matches!(filter.as_str(), "LZWDecode" | "LZW" | "RunLengthDecode" | "RL") { filter } else { "FlateDecode".into() };
                return ("partial", Some(format!("{what}: {filter} failed or ended early")));
            }
            Ok((_, Decoded::Unknown)) => return ("unknown", Some(format!("{what}: filter not implemented by the check"))),
            Ok((d, Decoded::Complete)) => d,
            Err(e) => return ("partial", Some(format!("{what}: decode failed: {e}"))),
        };
        // K5-D (cost): which fonts and forms the stream selects matters only if one of the candidates in its resources
        // is not complete. When every font's ToUnicode and every XObject is complete, any selection is complete, so
        // the stream is complete without tokenizing it (the tokenizer was most of this check's time). Each candidate's
        // status is the one a selection would compute (same cache, cycle guard and depth); large dictionaries take the
        // tokenizing path, as before.
        if self.candidates_complete(resources) {
            return ("complete", None);
        }
        let (mut names, mut fonts) = drawn_resources(&data);
        if names.is_empty() && fonts.is_empty() {
            return ("complete", None);
        }
        names.sort_unstable();
        names.dedup();
        fonts.sort_unstable();
        fonts.dedup();
        let resources = self.pdf.resolve(resources).unwrap_or(Obj::Null);
        // K5-D (K5A2B-G4): the ToUnicode CMap of every font the stream selects is read by PDFium to turn codes into
        // text; a damaged one loses mappings exactly as a damaged content stream loses operators.
        if !fonts.is_empty() {
            let fs: HashMap<Vec<u8>, Obj> = self.pdf.get(&resources, "Font").unwrap_or(Obj::Null).dict().into_iter().collect();
            let mut status = ("complete", None);
            for name in fonts {
                if let Some(r) = fs.get(&name) {
                    status = worse(status, self.tounicode_status(r));
                    if status.0 == "partial" {
                        return status;
                    }
                }
            }
            if names.is_empty() {
                return status;
            }
            let rest = self.drawn(&resources, names);
            return worse(status, rest);
        }
        self.drawn(&resources, names)
    }
    /// True when every font (its ToUnicode) and every XObject named in `resources` is complete (at most
    /// `SHORTCUT_ENTRIES` of each; more, or an unreadable dictionary, is `false`: the caller tokenizes the stream).
    fn candidates_complete(&mut self, resources: &Obj) -> bool {
        let resources = self.pdf.resolve(resources).unwrap_or(Obj::Null);
        let fonts = match self.pdf.get(&resources, "Font") {
            Ok(Obj::Null) => vec![],
            Ok(d @ Obj::Dict(_)) => d.dict(),
            _ => return false,
        };
        let xobjects = match self.pdf.get(&resources, "XObject") {
            Ok(Obj::Null) => vec![],
            Ok(d @ Obj::Dict(_)) => d.dict(),
            _ => return false,
        };
        if fonts.len() > SHORTCUT_ENTRIES || xobjects.len() > SHORTCUT_ENTRIES {
            return false;
        }
        if !fonts.iter().all(|(_, r)| self.tounicode_status(r).0 == "complete") {
            return false;
        }
        for (_, r) in xobjects {
            // A form without its own resources draws with the caller's and is never cached: checking it here could
            // recurse through every sibling; the caller tokenizes instead.
            let cached = r.id().map(|n| self.checked.contains_key(&n)).unwrap_or(false);
            if !cached {
                match self.pdf.resolve(&r) {
                    Ok(x @ Obj::Stream(..)) if x.get("Subtype").name() == "Form" && !x.has("Resources") => return false,
                    _ => {}
                }
            }
            if self.form(&resources, &r).0 != "complete" {
                return false;
            }
        }
        true
    }
    /// The status of a font's ToUnicode stream (cached per font object): `complete` without one.
    fn tounicode_status(&mut self, r: &Obj) -> (&'static str, Option<String>) {
        let id = r.id();
        if let Some(status) = id.and_then(|n| self.fonts.get(&n)) {
            return status.clone();
        }
        let font = self.pdf.resolve(r).unwrap_or(Obj::Null);
        let status = if font.has("ToUnicode") {
            let tu = font.get("ToUnicode");
            match self.pdf.resolve(&tu) {
                Ok(o @ Obj::Stream(..)) => match decode_content_checked(&o) {
                    Ok((_, Decoded::Complete)) => ("complete", None),
                    Ok((_, Decoded::Partial)) => ("partial", Some("font ToUnicode: FlateDecode failed or ended early".into())),
                    Ok((_, Decoded::Unknown)) => ("unknown", Some("font ToUnicode: filter not implemented by the check".into())),
                    Err(e) => ("partial", Some(format!("font ToUnicode: decode failed: {e}"))),
                },
                // A name (/Identity-H) or nothing usable: PDFium ignores it; not a decode failure.
                Ok(_) => ("complete", None),
                Err(e) => ("unknown", Some(format!("font ToUnicode unreadable: {e}"))),
            }
        } else {
            ("complete", None)
        };
        if let Some(n) = id {
            self.fonts.insert(n, status.clone());
        }
        status
    }
    fn drawn(&mut self, resources: &Obj, names: Vec<Vec<u8>>) -> (&'static str, Option<String>) {
        // One name -> entry map per stream (a shared XObject dictionary can hold thousands of entries).
        let xs: HashMap<Vec<u8>, Obj> = self.pdf.get(resources, "XObject").unwrap_or(Obj::Null).dict().into_iter().collect();
        let mut status = ("complete", None);
        for name in names {
            if let Some(r) = xs.get(&name) {
                status = worse(status, self.form(resources, r));
                if status.0 == "partial" {
                    break;
                }
            }
        }
        status
    }
    fn form(&mut self, resources: &Obj, r: &Obj) -> (&'static str, Option<String>) {
        let r = r.clone();
        let Some(id) = r.id() else {
            return ("complete", None);
        };
        if self.forms.contains(&id) || self.forms.len() >= MAX_FORM_DEPTH {
            return ("complete", None);
        }
        if let Some(status) = self.checked.get(&id) {
            return status.clone();
        }
        let x = match self.pdf.resolve(&r) {
            Ok(o @ Obj::Stream(..)) => o,
            _ => return ("complete", None),
        };
        if x.get("Subtype").name() != "Form" {
            self.checked.insert(id, ("complete", None));
            return ("complete", None);
        }
        // A form without resources draws with its parent's: its status is cached only when it has its own.
        let own = x.has("Resources");
        let res = if own { x.get("Resources") } else { resources.clone() };
        self.forms.push(id);
        let status = self.stream(&r, &res, "form XObject");
        self.forms.pop();
        if own {
            self.checked.insert(id, status.clone());
        }
        status
    }
}

/// Names drawn with `Do` and fonts selected with `Tf` in a content stream (inline images skipped), in order.
fn drawn_resources(data: &[u8]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut p = Parser::new(data, 0);
    let mut operands: Vec<Obj> = vec![];
    let mut out = vec![];
    let mut fonts = vec![];
    loop {
        p.skip();
        let Some(&c) = data.get(p.p) else {
            return (out, fonts);
        };
        if matches!(c, b'/' | b'(' | b'<' | b'[') {
            match p.obj() {
                Ok(o) => operands.push(o),
                Err(_) => return (out, fonts),
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
            "Do" => {
                if let Some(Obj::Name(n)) = operands.last() {
                    out.push(n.clone())
                }
            }
            "Tf" => {
                if let Some(Obj::Name(n)) = operands.len().checked_sub(2).and_then(|i| operands.get(i)) {
                    fonts.push(n.clone())
                }
            }
            "BI" if !skip_inline_image(&mut p) => return (out, fonts),
            _ => {}
        }
        operands.clear();
    }
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

    fn decode_status_of(objects: &[Vec<u8>], name: &str) -> Vec<(&'static str, Option<String>)> {
        let dir = std::env::temp_dir().join(format!("k5a2b-decode-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.pdf");
        std::fs::write(&path, build(objects)).unwrap();
        let out = page_decode_status(&mut Pdf::open(path.to_str().unwrap()).unwrap()).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        out
    }
    /// One page whose content stream (object 4) has the dictionary entries `dict` and the bytes `data`.
    fn page_with_stream(dict: &str, data: &[u8], xobjects: &str, extra: Option<Vec<u8>>) -> Vec<Vec<u8>> {
        let mut objs = one_page(b"", xobjects, extra);
        objs[3] = stream(dict, data);
        objs
    }
    /// The K5P-093 derivation (`k5_prep/pageset/make_derived.py`): the middle third of a FlateDecode stream zeroed.
    fn zero_middle_third(data: &[u8]) -> Vec<u8> {
        let mut out = data.to_vec();
        let (a, b) = (data.len() / 3, 2 * data.len() / 3);
        out[a..b].iter_mut().for_each(|x| *x = 0);
        out
    }

    #[test]
    fn decode_status_complete_partial_and_unknown() {
        let text = b"BT /F1 12 Tf 10 10 Td (Hello, a page with enough text to compress) Tj ET\n".repeat(20);
        let good = flate(&text);
        assert_eq!(decode_status_of(&page_with_stream("/Filter /FlateDecode", &good, "", None), "good"), vec![("complete", None)]);
        assert_eq!(decode_status_of(&one_page(b"BT /F1 12 Tf (x) Tj ET", "", None), "raw")[0].0, "complete");
        let corrupt = zero_middle_third(&good);
        let status = decode_status_of(&page_with_stream("/Filter /FlateDecode", &corrupt, "", None), "corrupt");
        assert_eq!(status[0].0, "partial", "{status:?}");
        let truncated = good[..good.len() / 2].to_vec();
        assert_eq!(decode_status_of(&page_with_stream("/Filter /FlateDecode", &truncated, "", None), "truncated")[0].0, "partial");
        // A filter the check does not implement cannot be decided (K5-D: LZW and RunLength are implemented now).
        assert_eq!(decode_status_of(&page_with_stream("/Filter /CCITTFaxDecode", b"\x80\x0b", "", None), "ccitt")[0].0, "unknown");
        assert_eq!(decode_status_of(&page_with_stream("/Filter /LZWDecode", b"\x80\x0b", "", None), "lzw")[0].0, "complete");
        // A strict-decoder failure of an implemented chain is partial (Flate with a predictor goes through it).
        let strict = page_with_stream("/Filter /FlateDecode /DecodeParms << /Predictor 12 /Columns 4 >>", &corrupt, "", None);
        assert_eq!(decode_status_of(&strict, "strict")[0].0, "partial");
        // A page without /Contents has nothing to decode.
        let mut empty = one_page(b"", "", None);
        empty[2] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>".to_vec();
        assert_eq!(decode_status_of(&empty, "empty"), vec![("complete", None)]);
        // A dangling content reference is not an observed decode failure, and the check cannot decide: unknown.
        let mut dangling = one_page(b"", "", None);
        dangling[2] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 9 0 R >>".to_vec();
        assert_eq!(decode_status_of(&dangling, "dangling")[0].0, "unknown");
    }

    /// PDF LZW encoder for the tests (Clear first, EOD last; `early` = EarlyChange). The decoder adds the table entry
    /// of a code only when it reads the next code, so the k-th code after Clear (k >= 2) is written with the width
    /// of a table of 256 + k entries (capped at 4096), as the decoder will read it.
    fn lzw_encode(input: &[u8], early: bool) -> Vec<u8> {
        let mut dict: HashMap<Vec<u8>, usize> = (0..256).map(|i| (vec![i as u8], i)).collect();
        let mut next = 258usize;
        let (mut out, mut acc, mut nbits, mut k) = (vec![], 0u64, 0u32, 0usize);
        let width_for = |k: usize| -> u32 {
            if k < 2 {
                return 9;
            }
            match (256 + k).min(4096) + usize::from(early) {
                0..=511 => 9,
                512..=1023 => 10,
                1024..=2047 => 11,
                _ => 12,
            }
        };
        let mut emit = |code: usize, width: u32, out: &mut Vec<u8>| {
            acc = (acc << width) | code as u64;
            nbits += width;
            while nbits >= 8 {
                out.push((acc >> (nbits - 8)) as u8);
                nbits -= 8;
                acc &= (1 << nbits) - 1;
            }
        };
        emit(256, 9, &mut out);
        let mut w: Vec<u8> = vec![];
        for &c in input {
            let mut wc = w.clone();
            wc.push(c);
            if dict.contains_key(&wc) {
                w = wc;
                continue;
            }
            k += 1;
            emit(dict[&w], width_for(k), &mut out);
            if next < 4096 {
                dict.insert(wc, next);
                next += 1;
            }
            w = vec![c];
        }
        if !w.is_empty() {
            k += 1;
            emit(dict[&w], width_for(k), &mut out);
        }
        k += 1;
        emit(257, width_for(k), &mut out);
        if nbits > 0 {
            out.push((acc << (8 - nbits)) as u8);
        }
        out
    }

    #[test]
    fn lzw_and_run_length_decode_like_the_specification() {
        // PDF 32000-1 7.4.4.2 example: "-----A---B" with EarlyChange 1.
        let (data, ok) = crate::pdf::lzw_decode(&[0x80, 0x0B, 0x60, 0x50, 0x22, 0x0C, 0x0C, 0x85, 0x01], true);
        assert!(ok);
        assert_eq!(data, b"-----A---B");
        // Round trips across the 9 -> 12 bit widths, both EarlyChange values, and a table reset.
        let long: Vec<u8> = (0..20000u32).map(|i| b"BT /F1 12 Tf (abcdefghij) Tj ET\n"[(i as usize * 7 + i as usize / 13) % 32]).collect();
        for early in [true, false] {
            let (data, ok) = crate::pdf::lzw_decode(&lzw_encode(&long, early), early);
            assert!(ok);
            assert_eq!(data, long);
        }
        // An invalid code stops with the prefix decoded before it.
        // Codes Clear, 'A', 'B', 300 (the table holds 259 entries then): "AB" and not clean.
        let (prefix, ok) = crate::pdf::lzw_decode(&[0x80, 0x10, 0x48, 0x52, 0xC0], true);
        assert_eq!((prefix, ok), (b"AB".to_vec(), false));
        // RunLength: literal and repeated runs, EOD; a cut run keeps the prefix.
        assert_eq!(crate::pdf::run_length_decode(&[2, b'a', b'b', b'c', 254, b'x', 128, b'z']), (b"abcxxx".to_vec(), true));
        assert_eq!(crate::pdf::run_length_decode(&[4, b'a', b'b']), (b"ab".to_vec(), false));
        assert_eq!(crate::pdf::run_length_decode(&[255]), (vec![], false));
    }

    #[test]
    fn lzw_and_run_length_content_show_text_and_report_damage() {
        // K5-D (K5A-M2): text in an LZW or RunLength content stream is text (it was "no text" natively).
        let text = b"BT /F1 12 Tf 10 10 Td (Hello, text behind an old filter) Tj ET\n".repeat(30);
        let lzw = lzw_encode(&text, true);
        let mut rl = vec![];
        for chunk in text.chunks(128) {
            rl.push((chunk.len() - 1) as u8);
            rl.extend_from_slice(chunk);
        }
        rl.push(128);
        // Names unique across the tests: `decode_status_of` uses a per-name scratch folder, and tests run in parallel.
        for (name, dict, data) in [("k5d-lzw", "/Filter /LZWDecode", lzw.clone()), ("k5d-rl", "/Filter [/RunLengthDecode]", rl.clone())] {
            let objects = page_with_stream(dict, &data, "", None);
            assert_eq!(decode_status_of(&objects, name), vec![("complete", None)], "{name}");
            let dir = std::env::temp_dir().join(format!("k5d-filter-{}-{name}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("t.pdf");
            std::fs::write(&path, build(&objects)).unwrap();
            let sample = native_sample(&mut Pdf::open(path.to_str().unwrap()).unwrap()).unwrap();
            std::fs::remove_dir_all(dir).unwrap();
            assert!(sample.has_text, "{name}");
        }
        // Damage keeps the readable prefix and is reported partial.
        let mut cut = rl[..rl.len() / 2].to_vec();
        cut.push(127);
        assert_eq!(decode_status_of(&page_with_stream("/Filter /RunLengthDecode", &cut, "", None), "rlcut")[0].0, "partial");
        let mut damaged = lzw[..lzw.len() / 2].to_vec();
        damaged.extend([0xFF; 4]);
        assert_eq!(decode_status_of(&page_with_stream("/Filter /LZWDecode", &damaged, "", None), "lzwbad")[0].0, "partial");
        // EarlyChange 0 is honoured.
        let late = lzw_encode(&text, false);
        assert_eq!(decode_status_of(&page_with_stream("/Filter /LZWDecode /DecodeParms << /EarlyChange 0 >>", &late, "", None), "late")[0].0, "complete");
    }

    #[test]
    fn decode_status_covers_the_tounicode_of_selected_fonts() {
        // K5-D (K5A2B-G4): a damaged ToUnicode CMap of a font the page selects (Tf) is partial; of a font it never
        // selects, nothing; a name value or no ToUnicode, complete.
        let mut cmap = b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n".to_vec();
        for i in 0..200 {
            cmap.extend(format!("1 beginbfchar <{i:02X}> <{:04X}> endbfchar\n", 0x391 + i).bytes());
        }
        cmap.extend(b"endcmap CMapName currentdict /CMap defineresource pop end end\n");
        let damaged = zero_middle_third(&flate(&cmap));
        let with_cmap = |content: &[u8], cmap_stream: Vec<u8>| {
            let mut objs = one_page(content, "", Some(cmap_stream));
            objs[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>".to_vec();
            objs
        };
        let bad = stream("/Filter /FlateDecode", &damaged);
        let status = decode_status_of(&with_cmap(b"BT /F1 12 Tf (x) Tj ET", bad.clone()), "tu-bad");
        assert_eq!(status[0].0, "partial", "{status:?}");
        assert!(status[0].1.as_deref().unwrap_or("").starts_with("font ToUnicode"), "{status:?}");
        assert_eq!(decode_status_of(&with_cmap(b"0 0 m 10 10 l S", bad), "tu-unused")[0], ("complete", None));
        let good = stream("/Filter /FlateDecode", &flate(&cmap));
        assert_eq!(decode_status_of(&with_cmap(b"BT /F1 12 Tf (x) Tj ET", good), "tu-good")[0], ("complete", None));
        let mut named = one_page(b"BT /F1 12 Tf (x) Tj ET", "", None);
        named[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode /Identity-H >>".to_vec();
        assert_eq!(decode_status_of(&named, "tu-name")[0], ("complete", None));
        // A font selected inside a drawn form is followed too.
        let form = stream("/Type /XObject /Subtype /Form /BBox [0 0 200 200]", b"BT /F1 12 Tf (y) Tj ET");
        let mut drawn = one_page(b"q /X0 Do Q", "/XObject << /X0 7 0 R >>", Some(stream("/Filter /FlateDecode", &damaged)));
        drawn[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>".to_vec();
        drawn.push(form);
        assert_eq!(decode_status_of(&drawn, "tu-form")[0].0, "partial");
    }

    #[test]
    fn open_for_check_reads_plain_documents_and_names_what_it_refuses() {
        // K5-D: the check's opener equals `open` on an unencrypted document; an encrypted one it cannot decrypt (here
        // a V 4 / AESV2 dictionary) is refused with the handler's reason, never read as plain.
        let dir = std::env::temp_dir().join(format!("k5d-open-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("plain.pdf");
        std::fs::write(&plain, build(&one_page(b"BT /F1 12 Tf (x) Tj ET", "", None))).unwrap();
        let mut a = Pdf::open_for_check(plain.to_str().unwrap()).unwrap();
        let mut b = Pdf::open(plain.to_str().unwrap()).unwrap();
        assert_eq!(page_decode_status(&mut a).unwrap(), page_decode_status(&mut b).unwrap());
        let mut bytes = build(&one_page(b"BT /F1 12 Tf (x) Tj ET", "", None));
        let at = bytes.windows(9).position(|w| w == b"/Root 1 0").unwrap();
        bytes.splice(at..at, b"/Encrypt << /Filter /Standard /V 4 /R 4 /Length 128 /O <00> /U <00> /P -4 >> ".iter().copied());
        let aes = dir.join("aes.pdf");
        std::fs::write(&aes, &bytes).unwrap();
        let refused = Pdf::open_for_check(aes.to_str().unwrap()).map(|_| ()).unwrap_err().to_string();
        std::fs::remove_dir_all(dir).unwrap();
        assert!(refused.contains("Encrypted PDFs are unsupported") && refused.contains("V 4 R 4"), "{refused}");
    }

    #[test]
    fn decode_status_follows_drawn_forms_and_cannot_open_encryption() {
        let form_text = flate(&b"BT /F1 12 Tf 10 10 Td (Inside a form with text) Tj ET\n".repeat(20));
        let form = stream("/Type /XObject /Subtype /Form /BBox [0 0 200 200] /Filter /FlateDecode", &zero_middle_third(&form_text));
        let drawn = one_page(b"q /X0 Do Q", "/XObject << /X0 6 0 R >>", Some(form.clone()));
        assert_eq!(decode_status_of(&drawn, "form")[0].0, "partial");
        // The same broken form, never drawn, is not part of the page's content.
        let undrawn = one_page(b"BT /F1 12 Tf (x) Tj ET", "/XObject << /X0 6 0 R >>", Some(form));
        assert_eq!(decode_status_of(&undrawn, "undrawn")[0].0, "complete");
        // An encrypted document is undecidable: the parser refuses it (the Python caller reports every page unknown).
        let mut bytes = build(&one_page(b"BT /F1 12 Tf (x) Tj ET", "", None));
        let at = bytes.windows(9).position(|w| w == b"/Root 1 0").unwrap();
        bytes.splice(at..at, b"/Encrypt << /Filter /Standard >> ".iter().copied());
        let dir = std::env::temp_dir().join(format!("k5a2b-decode-{}-enc", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.pdf");
        std::fs::write(&path, &bytes).unwrap();
        let opened = Pdf::open(path.to_str().unwrap()).map(|_| ());
        std::fs::remove_dir_all(dir).unwrap();
        assert!(opened.unwrap_err().to_string().contains("Encrypted"));
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
