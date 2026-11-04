//! Minimal lazy PDF parser: classic/xref-stream revisions, object streams and filters.
use anyhow::{anyhow, bail, Result};
use flate2::read::ZlibDecoder;
use std::{
    collections::{BTreeMap, HashSet},
    io::Read,
};
#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Num(String),
    Name(Vec<u8>),
    Str(Vec<u8>),
    Array(Vec<Obj>),
    Dict(Vec<(Vec<u8>, Obj)>),
    Ref(u32, u16),
    Stream(Vec<(Vec<u8>, Obj)>, std::sync::Arc<Vec<u8>>),
}
impl Obj {
    pub fn get(&self, k: &str) -> Obj {
        match self {
            Self::Dict(d) | Self::Stream(d, _) => d
                .iter()
                .find(|(n, _)| n.as_slice() == k.as_bytes())
                .map(|(_, v)| v.clone())
                .unwrap_or(Self::Null),
            _ => Self::Null,
        }
    }
    pub fn name(&self) -> String {
        if let Self::Name(s) = self {
            String::from_utf8(s.clone()).unwrap_or_default()
        } else {
            String::new()
        }
    }
    /// Unescaped PDF name bytes; glyph matching must never use decoded text.
    pub fn name_bytes(&self) -> &[u8] {
        if let Self::Name(s) = self {
            s
        } else {
            b""
        }
    }
    pub fn int(&self) -> usize {
        if let Self::Num(s) = self {
            s.parse().unwrap_or(0)
        } else {
            0
        }
    }
    pub fn array(&self) -> Vec<Obj> {
        if let Self::Array(a) = self {
            a.clone()
        } else {
            vec![]
        }
    }
    pub fn dict(&self) -> Vec<(Vec<u8>, Obj)> {
        match self {
            Self::Dict(d) | Self::Stream(d, _) => d.clone(),
            _ => vec![],
        }
    }
    pub fn id(&self) -> Option<u32> {
        if let Self::Ref(n, _) = self {
            Some(*n)
        } else {
            None
        }
    }
    pub fn has(&self, k: &str) -> bool {
        self.dict()
            .iter()
            .any(|(n, _)| n.as_slice() == k.as_bytes())
    }
}
pub struct Parser<'a> {
    pub b: &'a [u8],
    pub p: usize,
}
fn ws(c: u8) -> bool {
    matches!(c, 0 | 9 | 10 | 12 | 13 | 32)
}
fn delim(c: u8) -> bool {
    ws(c) || b"()<>[]{}/%".contains(&c)
}
impl<'a> Parser<'a> {
    pub fn new(b: &'a [u8], p: usize) -> Self {
        Self { b, p }
    }
    pub fn skip(&mut self) {
        loop {
            while self.p < self.b.len() && ws(self.b[self.p]) {
                self.p += 1
            }
            if self.b.get(self.p) == Some(&b'%') {
                while self.p < self.b.len() && !b"\r\n".contains(&self.b[self.p]) {
                    self.p += 1
                }
            } else {
                break;
            }
        }
    }
    pub fn word(&mut self) -> Result<String> {
        self.skip();
        let s = self.p;
        while self.p < self.b.len() && !delim(self.b[self.p]) {
            self.p += 1
        }
        if s == self.p {
            bail!("Missing PDF token at {}", s)
        }
        Ok(String::from_utf8_lossy(&self.b[s..self.p]).into())
    }
    pub fn obj(&mut self) -> Result<Obj> {
        self.skip();
        let c = *self
            .b
            .get(self.p)
            .ok_or_else(|| anyhow!("Unexpected end of PDF"))?;
        match c {
            b'/' => {
                self.p += 1;
                let s = self.p;
                while self.p < self.b.len() && !delim(self.b[self.p]) {
                    self.p += 1
                }
                let mut out = vec![];
                let mut i = s;
                while i < self.p {
                    if self.b[i] == b'#' && i + 2 < self.p {
                        out.push(u8::from_str_radix(
                            std::str::from_utf8(&self.b[i + 1..i + 3])?,
                            16,
                        )?);
                        i += 3
                    } else {
                        out.push(self.b[i]);
                        i += 1
                    }
                }
                Ok(Obj::Name(out))
            }
            b'[' => {
                self.p += 1;
                let mut a = vec![];
                loop {
                    self.skip();
                    if self.b.get(self.p) == Some(&b']') {
                        self.p += 1;
                        break;
                    }
                    a.push(self.obj()?)
                }
                Ok(Obj::Array(a))
            }
            b'<' if self.b.get(self.p + 1) == Some(&b'<') => {
                self.p += 2;
                let mut d = vec![];
                loop {
                    self.skip();
                    if self.b.get(self.p..self.p + 2) == Some(b">>") {
                        self.p += 2;
                        break;
                    }
                    let k = match self.obj()? {
                        Obj::Name(n) => n,
                        _ => bail!("Invalid dictionary key"),
                    };
                    if k.is_empty() {
                        bail!("Invalid dictionary key")
                    }
                    let v = self.obj()?;
                    d.push((k, v))
                }
                Ok(Obj::Dict(d))
            }
            b'<' => {
                self.p += 1;
                let mut h = String::new();
                while let Some(&c) = self.b.get(self.p) {
                    self.p += 1;
                    if c == b'>' {
                        break;
                    }
                    if !ws(c) {
                        h.push(c as char)
                    }
                }
                if h.len() % 2 == 1 {
                    h.push('0')
                }
                let mut v = vec![];
                for i in (0..h.len()).step_by(2) {
                    v.push(u8::from_str_radix(&h[i..i + 2], 16)?)
                }
                Ok(Obj::Str(v))
            }
            b'(' => {
                self.p += 1;
                let mut depth = 1;
                let mut v = vec![];
                while depth > 0 {
                    let c = *self
                        .b
                        .get(self.p)
                        .ok_or_else(|| anyhow!("Unterminated string"))?;
                    self.p += 1;
                    match c {
                        b'\\' => {
                            let c = *self.b.get(self.p).ok_or_else(|| anyhow!("Bad escape"))?;
                            self.p += 1;
                            match c {
                                b'n' => v.push(10),
                                b'r' => v.push(13),
                                b't' => v.push(9),
                                b'b' => v.push(8),
                                b'f' => v.push(12),
                                b'\n' => (),
                                b'\r' => {
                                    if self.b.get(self.p) == Some(&10) {
                                        self.p += 1
                                    }
                                }
                                b'0'..=b'7' => {
                                    let mut n = (c - b'0') as u16;
                                    for _ in 0..2 {
                                        if let Some(c @ b'0'..=b'7') = self.b.get(self.p) {
                                            n = n * 8 + (c - b'0') as u16;
                                            self.p += 1
                                        } else {
                                            break;
                                        }
                                    }
                                    v.push(n as u8)
                                }
                                _ => v.push(c),
                            }
                        }
                        b'(' => {
                            depth += 1;
                            v.push(c)
                        }
                        b')' => {
                            depth -= 1;
                            if depth > 0 {
                                v.push(c)
                            }
                        }
                        _ => v.push(c),
                    }
                }
                Ok(Obj::Str(v))
            }
            _ => {
                let w = self.word()?;
                match w.as_str() {
                    "null" => Ok(Obj::Null),
                    "true" => Ok(Obj::Bool(true)),
                    "false" => Ok(Obj::Bool(false)),
                    _ => {
                        if w.parse::<f64>().is_err() {
                            bail!("Unexpected PDF token {w}")
                        }
                        let end = self.p;
                        if let Ok(n) = w.parse::<u32>() {
                            if let Ok(g) = self.word() {
                                if let Ok(g) = g.parse::<u16>() {
                                    if self.word().ok().as_deref() == Some("R") {
                                        return Ok(Obj::Ref(n, g));
                                    }
                                }
                            }
                        }
                        self.p = end;
                        Ok(Obj::Num(w))
                    }
                }
            }
        }
    }
}
#[derive(Clone)]
enum Entry {
    Offset(usize),
    Compressed(u32, usize),
}
pub struct Pdf {
    pub bytes: Vec<u8>,
    entries: BTreeMap<u32, Entry>,
    cache: BTreeMap<u32, Obj>,
    pub trailer: Obj,
    pub prev: usize,
    loading: HashSet<u32>,
    /// The RC4 file key when the document was opened for the decode check (`open_for_check`); never set by `open`.
    crypt: Option<Vec<u8>>,
}
impl Pdf {
    pub fn open(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let s = bytes
            .windows(9)
            .rposition(|w| w == b"startxref")
            .ok_or_else(|| anyhow!("Missing startxref"))?;
        let prev = Parser::new(&bytes, s + 9).word()?.parse()?;
        let mut pdf = Self {
            bytes,
            entries: BTreeMap::new(),
            cache: BTreeMap::new(),
            trailer: Obj::Null,
            prev,
            loading: HashSet::new(),
            crypt: None,
        };
        pdf.xref(prev, &mut HashSet::new())?;
        if pdf.trailer.has("Encrypt") {
            bail!("Encrypted PDFs are unsupported")
        }
        Ok(pdf)
    }
    /// `open` for the read-only content-stream decode check (K5-D, K5A2B-G4): an encrypted document whose security
    /// handler is the standard RC4 one (V 1/2, R 2/3) and whose user password is empty (an owner-password document,
    /// which PDFium opens) is decrypted object by object as streams are read; any other encrypted document is still
    /// refused, with the reason. Never used by analysis or repair.
    #[cfg_attr(not(feature = "python"), allow(dead_code))]
    pub fn open_for_check(path: &str) -> Result<Self> {
        match Self::open(path) {
            Err(e) if e.to_string() == "Encrypted PDFs are unsupported" => {}
            other => return other,
        }
        let bytes = std::fs::read(path)?;
        let s = bytes
            .windows(9)
            .rposition(|w| w == b"startxref")
            .ok_or_else(|| anyhow!("Missing startxref"))?;
        let prev = Parser::new(&bytes, s + 9).word()?.parse()?;
        let mut pdf = Self {
            bytes,
            entries: BTreeMap::new(),
            cache: BTreeMap::new(),
            trailer: Obj::Null,
            prev,
            loading: HashSet::new(),
            crypt: None,
        };
        pdf.xref(prev, &mut HashSet::new())?;
        let enc = pdf.resolve(&pdf.trailer.get("Encrypt"))?;
        pdf.cache.clear(); // nothing read before the key is known stays cached
        let num = |o: &Obj| match o {
            Obj::Num(s) => s.parse::<f64>().map(|v| v as i64).unwrap_or(0),
            _ => 0,
        };
        let text = |o: &Obj| match o {
            Obj::Str(b) => b.clone(),
            _ => vec![],
        };
        let id0 = match pdf.trailer.get("ID") {
            Obj::Array(a) => a.first().map(text).unwrap_or_default(),
            _ => vec![],
        };
        let v = num(&enc.get("V"));
        let length = if enc.has("Length") { num(&enc.get("Length")) } else { 40 };
        let key = crate::crypt::empty_password_key(
            &enc.get("Filter").name(),
            v,
            num(&enc.get("R")),
            length,
            &text(&enc.get("O")),
            &text(&enc.get("U")),
            num(&enc.get("P")),
            &id0,
        )?;
        pdf.crypt = Some(key);
        Ok(pdf)
    }
    fn raw(&self, off: usize) -> Result<Obj> {
        let mut p = Parser::new(&self.bytes, off);
        let number = p.word()?;
        let generation = p.word()?;
        if p.word()? != "obj" {
            bail!("Expected obj")
        }
        let o = p.obj()?;
        if let Obj::Dict(d) = o.clone() {
            let end = p.p;
            if p.word().ok().as_deref() == Some("stream") {
                if self.bytes.get(p.p) == Some(&13) {
                    p.p += 1
                }
                if self.bytes.get(p.p) == Some(&10) {
                    p.p += 1
                }
                let len = o.get("Length").int();
                let stop = if len > 0 && p.p + len <= self.bytes.len() {
                    p.p + len
                } else {
                    self.bytes[p.p..]
                        .windows(9)
                        .position(|w| w == b"endstream")
                        .map(|i| p.p + i)
                        .ok_or_else(|| anyhow!("Unterminated stream"))?
                };
                let mut data = self.bytes[p.p..stop].to_vec();
                if let Some(key) = &self.crypt {
                    // Streams of an RC4 document are encrypted with their object's key; cross-reference streams are not.
                    if o.get("Type").name() != "XRef" {
                        let (n, g) = (number.parse::<u32>().unwrap_or(0), generation.parse::<u16>().unwrap_or(0));
                        data = crate::crypt::rc4(&crate::crypt::object_key(key, n, g), &data);
                    }
                }
                return Ok(Obj::Stream(d, std::sync::Arc::new(data)));
            }
            let _ = end;
        }
        Ok(o)
    }
    fn xref(&mut self, off: usize, seen: &mut HashSet<usize>) -> Result<()> {
        if !seen.insert(off) {
            return Ok(());
        }
        let mut p = Parser::new(&self.bytes, off);
        let mut add = vec![];
        let tr;
        if p.word().ok().as_deref() == Some("xref") {
            loop {
                let w = p.word()?;
                if w == "trailer" {
                    tr = p.obj()?;
                    break;
                }
                let start: u32 = w.parse()?;
                let count: usize = p.word()?.parse()?;
                for i in 0..count {
                    let offset = p.word()?.parse()?;
                    p.word()?;
                    let flag = p.word()?;
                    if flag == "n" {
                        add.push((start + i as u32, Entry::Offset(offset)))
                    } else {
                        // reserve deleted IDs against previous revisions
                        add.push((start + i as u32, Entry::Offset(0)));
                    }
                }
            }
        } else {
            tr = self.raw(off)?;
            let data = decode(&tr)?;
            let widths = tr.get("W").array();
            if widths.len() != 3 {
                bail!("Invalid xref W")
            }
            let widths: Vec<_> = widths.iter().map(Obj::int).collect();
            let idx = if tr.has("Index") {
                tr.get("Index").array()
            } else {
                vec![Obj::Num("0".into()), tr.get("Size")]
            };
            let mut pos = 0;
            for pair in idx.chunks(2) {
                if pair.len() != 2 {
                    bail!("Invalid xref Index")
                }
                for id in pair[0].int()..pair[0].int() + pair[1].int() {
                    let mut fields = [0u64; 3];
                    for j in 0..3 {
                        if widths[j] == 0 {
                            fields[j] = if j == 0 { 1 } else { 0 }
                        } else {
                            for _ in 0..widths[j] {
                                fields[j] = (fields[j] << 8)
                                    | *data.get(pos).ok_or_else(|| anyhow!("Truncated xref"))?
                                        as u64;
                                pos += 1
                            }
                        }
                    }
                    let entry = match fields[0] {
                        0 => Entry::Offset(0),
                        1 => Entry::Offset(fields[1] as usize),
                        2 => Entry::Compressed(fields[1] as u32, fields[2] as usize),
                        _ => continue,
                    };
                    add.push((id as u32, entry));
                }
            }
        }
        for (n, e) in add {
            self.entries.entry(n).or_insert(e);
        }
        if self.trailer == Obj::Null {
            self.trailer = tr.clone()
        } else {
            let mut d = self.trailer.dict();
            for (k, v) in tr.dict() {
                if !d.iter().any(|(key, _)| key == &k) {
                    d.push((k, v))
                }
            }
            self.trailer = Obj::Dict(d)
        }
        if tr.has("XRefStm") {
            self.xref(tr.get("XRefStm").int(), seen)?
        }
        if tr.has("Prev") {
            self.xref(tr.get("Prev").int(), seen)?
        }
        Ok(())
    }
    pub fn resolve(&mut self, o: &Obj) -> Result<Obj> {
        if let Obj::Ref(n, _) = o {
            self.object(*n)
        } else {
            Ok(o.clone())
        }
    }
    pub fn object(&mut self, n: u32) -> Result<Obj> {
        if let Some(o) = self.cache.get(&n) {
            return Ok(o.clone());
        }
        if !self.loading.insert(n) {
            bail!("Cyclic object reference")
        }
        let res = (|| match self
            .entries
            .get(&n)
            .cloned()
            .ok_or_else(|| anyhow!("Missing object {n}"))?
        {
            Entry::Offset(off) => {
                if off == 0 {
                    bail!("Deleted object {n}")
                }
                self.raw(off)
            }
            Entry::Compressed(stream, index) => {
                let o = self.object(stream)?;
                let bytes = decode(&o)?;
                let count = o.get("N").int();
                let first = o.get("First").int();
                let mut p = Parser::new(&bytes, 0);
                let mut pairs = vec![];
                for _ in 0..count {
                    pairs.push((p.word()?.parse::<u32>()?, p.word()?.parse::<usize>()?))
                }
                for (i, (id, off)) in pairs.iter().enumerate() {
                    // A later revision can replace an object from this stream.
                    // Never cache the obsolete compressed copy over its update.
                    if matches!(self.entries.get(id), Some(Entry::Compressed(s, j)) if *s == stream && *j == i)
                    {
                        let val = Parser::new(&bytes, first + off).obj()?;
                        self.cache.insert(*id, val);
                    }
                }
                let id = pairs
                    .get(index)
                    .ok_or_else(|| anyhow!("Bad object-stream index"))?
                    .0;
                if id != n {
                    bail!("Object-stream index mismatch")
                }
                Ok(self.cache[&n].clone())
            }
        })();
        self.loading.remove(&n);
        if let Ok(o) = &res {
            self.cache.insert(n, o.clone());
        }
        res
    }
    /// Original object body, including encodings and whitespace. Compressed
    /// objects use the original decoded ObjStm bytes, never serialization.
    pub fn object_bytes(&mut self, n: u32) -> Result<Vec<u8>> {
        let entry = self
            .entries
            .get(&n)
            .cloned()
            .ok_or_else(|| anyhow!("Missing object {n}"))?;
        let (bytes, start) = match entry {
            Entry::Offset(off) => {
                if off == 0 {
                    bail!("Deleted object {n}")
                }
                let mut p = Parser::new(&self.bytes, off);
                if p.word()?.parse::<u32>()? != n {
                    bail!("Object ID mismatch")
                }
                p.word()?;
                if p.word()? != "obj" {
                    bail!("Expected obj")
                }
                p.skip();
                let start = p.p;
                if !matches!(p.obj()?, Obj::Dict(_)) {
                    bail!("Font object is not a dictionary")
                }
                return Ok(self.bytes[start..p.p].to_vec());
            }
            Entry::Compressed(stream, index) => {
                let o = self.object(stream)?;
                let bytes = decode(&o)?;
                let mut p = Parser::new(&bytes, 0);
                let mut start = None;
                for i in 0..o.get("N").int() {
                    let id = p.word()?.parse::<u32>()?;
                    let off = p.word()?.parse::<usize>()?;
                    if i == index {
                        if id != n {
                            bail!("Object-stream index mismatch")
                        }
                        start = Some(o.get("First").int() + off);
                    }
                }
                (
                    bytes,
                    start.ok_or_else(|| anyhow!("Bad object-stream index"))?,
                )
            }
        };
        let mut p = Parser::new(&bytes, start);
        p.skip();
        let start = p.p;
        if !matches!(p.obj()?, Obj::Dict(_)) {
            bail!("Font object is not a dictionary")
        }
        Ok(bytes[start..p.p].to_vec())
    }
    pub fn get(&mut self, o: &Obj, k: &str) -> Result<Obj> {
        let o = self.resolve(o)?;
        self.resolve(&o.get(k))
    }
    pub fn fonts(&mut self) -> Result<Vec<(Obj, Obj)>> {
        let root = self.get(&self.trailer.clone(), "Root")?;
        let pages = root.get("Pages");
        let mut out = vec![];
        let mut resources = HashSet::new();
        let mut fonts = HashSet::new();
        self.pages(
            &pages,
            Obj::Null,
            &mut out,
            &mut resources,
            &mut fonts,
            &mut HashSet::new(),
        )?;
        Ok(out)
    }
    fn pages(
        &mut self,
        r: &Obj,
        inherited: Obj,
        out: &mut Vec<(Obj, Obj)>,
        rs: &mut HashSet<String>,
        fs: &mut HashSet<String>,
        seen: &mut HashSet<u32>,
    ) -> Result<()> {
        if let Some(n) = r.id() {
            if !seen.insert(n) {
                return Ok(());
            }
        }
        let o = self.resolve(r)?;
        let resources = if o.has("Resources") {
            o.get("Resources")
        } else {
            inherited
        };
        if o.get("Type").name() == "Page" {
            self.resources(&resources, out, rs, fs)?
        } else {
            for kid in o.get("Kids").array() {
                self.pages(&kid, resources.clone(), out, rs, fs, seen)?
            }
        }
        Ok(())
    }
    fn resources(
        &mut self,
        r: &Obj,
        out: &mut Vec<(Obj, Obj)>,
        rs: &mut HashSet<String>,
        fs: &mut HashSet<String>,
    ) -> Result<()> {
        if *r == Obj::Null {
            return Ok(());
        }
        let key = format!("{r:?}");
        if !rs.insert(key) {
            return Ok(());
        }
        let o = self.resolve(r)?;
        let fonts = self.get(&o, "Font")?;
        for (_, r) in fonts.dict() {
            let key = format!("{r:?}");
            if fs.insert(key) {
                out.push((r.clone(), self.resolve(&r)?))
            }
        }
        let xs = self.get(&o, "XObject")?;
        for (_, r) in xs.dict() {
            let x = self.resolve(&r)?;
            if x.get("Subtype").name() == "Form" {
                self.resources(&x.get("Resources"), out, rs, fs)?
            }
        }
        Ok(())
    }
    pub fn size(&self) -> u32 {
        (self.entries.keys().next_back().copied().unwrap_or(0) + 1)
            .max(self.trailer.get("Size").int() as u32)
    }
}
pub fn decode(o: &Obj) -> Result<Vec<u8>> {
    let mut data = if let Obj::Stream(_, b) = o {
        b.as_ref().clone()
    } else {
        bail!("Expected stream")
    };
    let f = o.get("Filter");
    let filters = if matches!(f, Obj::Array(_)) {
        f.array()
    } else if f == Obj::Null {
        vec![]
    } else {
        vec![f]
    };
    let params = o.get("DecodeParms");
    let params = if matches!(params, Obj::Array(_)) {
        params.array()
    } else {
        vec![params]
    };
    for (i, f) in filters.iter().enumerate() {
        match f.name().as_str() {
            "FlateDecode" | "Fl" => {
                let mut out = vec![];
                ZlibDecoder::new(&data[..]).read_to_end(&mut out)?;
                data = out
            }
            "ASCIIHexDecode" | "AHx" => {
                let mut h: Vec<u8> = data
                    .iter()
                    .copied()
                    .filter(|c| !ws(*c) && *c != b'>')
                    .collect();
                if h.len() % 2 == 1 {
                    h.push(b'0')
                }
                data = h
                    .chunks(2)
                    .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap_or("?"), 16))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
            "ASCII85Decode" | "A85" => {
                let mut out = vec![];
                let mut block = vec![];
                for c in data.iter().copied().filter(|c| !ws(*c)) {
                    if c == b'~' {
                        break;
                    }
                    if c == b'z' && block.is_empty() {
                        out.extend([0; 4]);
                        continue;
                    }
                    if !(b'!'..=b'u').contains(&c) {
                        bail!("Invalid ASCII85")
                    }
                    block.push((c - b'!') as u64);
                    if block.len() == 5 {
                        let n = block.iter().fold(0, |n, c| n * 85 + c);
                        out.extend((n as u32).to_be_bytes());
                        block.clear()
                    }
                }
                if !block.is_empty() {
                    let len = block.len();
                    while block.len() < 5 {
                        block.push(84)
                    }
                    let n = block.iter().fold(0, |n, c| n * 85 + c);
                    out.extend(&(n as u32).to_be_bytes()[..len - 1])
                }
                data = out
            }
            "LZWDecode" | "LZW" => {
                let early = params.get(i).map(|p| p.get("EarlyChange")).unwrap_or(Obj::Null);
                let (out, complete) = lzw_decode(&data, early == Obj::Null || early.int() != 0);
                if !complete {
                    bail!("Invalid LZW code")
                }
                data = out
            }
            "RunLengthDecode" | "RL" => {
                let (out, complete) = run_length_decode(&data);
                if !complete {
                    bail!("Truncated RunLength run")
                }
                data = out
            }
            other => bail!("Unsupported stream filter {other}"),
        }
        let p = params.get(i).cloned().unwrap_or(Obj::Null);
        let pred = p.get("Predictor").int();
        if pred > 1 {
            let cols = p.get("Columns").int().max(1);
            let colors = p.get("Colors").int().max(1);
            let bits = p.get("BitsPerComponent").int().max(8);
            let row = (cols * colors * bits).div_ceil(8);
            let bpp = (colors * bits).div_ceil(8);
            if pred >= 10 {
                let mut out = vec![];
                let mut prev = vec![0u8; row];
                for block in data.chunks(row + 1) {
                    if block.len() != row + 1 {
                        bail!("Invalid predictor row")
                    }
                    let mut current = block[1..].to_vec();
                    for j in 0..row {
                        let a = if j >= bpp { current[j - bpp] } else { 0 };
                        let b = prev[j];
                        let c = if j >= bpp { prev[j - bpp] } else { 0 };
                        let val = match block[0] {
                            0 => 0,
                            1 => a,
                            2 => b,
                            3 => ((a as u16 + b as u16) / 2) as u8,
                            4 => {
                                let p = a as i32 + b as i32 - c as i32;
                                let (pa, pb, pc) = (
                                    (p - a as i32).abs(),
                                    (p - b as i32).abs(),
                                    (p - c as i32).abs(),
                                );
                                if pa <= pb && pa <= pc {
                                    a
                                } else if pb <= pc {
                                    b
                                } else {
                                    c
                                }
                            }
                            _ => bail!("Invalid predictor"),
                        };
                        current[j] = current[j].wrapping_add(val)
                    }
                    out.extend(&current);
                    prev = current
                }
                data = out
            } else if pred == 2 && bits == 8 {
                for block in data.chunks_mut(row) {
                    for j in bpp..block.len() {
                        block[j] = block[j].wrapping_add(block[j - bpp])
                    }
                }
            } else {
                bail!("Unsupported predictor")
            }
        }
    }
    Ok(data)
}
/// LZWDecode (PDF 32000-1 7.4.4): 9- to 12-bit codes, Clear-Table 256, EOD 257, `early` = EarlyChange (default 1:
/// the code width grows one code early). Returns the decoded bytes and whether the stream decoded cleanly: an
/// invalid code (beyond the table) stops with the prefix decoded so far and `false`; the end of the input without an
/// EOD code is a clean end (PDFium's decoder also stops there). K5-D (K5A-M2): content streams with this filter
/// were undecodable natively, so text-presence said "no text" where PDFium reads text.
pub(crate) fn lzw_decode(data: &[u8], early: bool) -> (Vec<u8>, bool) {
    // Each table entry is a (start, len) range of `out`: a new entry is the previous string followed by the first
    // byte of the current one, and the previous string always ends where the current one starts (output is
    // append-only), so the entry is the contiguous range `previous.start .. previous.end + 1`.
    let mut out = Vec::with_capacity(data.len() * 3);
    let mut next = 258usize;
    let mut table: Vec<(usize, usize)> = vec![(0, 0); 4096];
    let (mut width, mut buffer, mut bits, mut pos) = (9u32, 0u32, 0u32, 0usize);
    let mut previous: Option<(usize, usize)> = None;
    loop {
        while bits < width {
            let Some(&b) = data.get(pos) else {
                return (out, true);
            };
            buffer = (buffer << 8) | b as u32;
            bits += 8;
            pos += 1;
        }
        let code = ((buffer >> (bits - width)) & ((1 << width) - 1)) as usize;
        bits -= width;
        buffer &= (1u32 << bits) - 1;
        if code == 256 {
            next = 258;
            width = 9;
            previous = None;
            continue;
        }
        if code == 257 {
            return (out, true);
        }
        let start = out.len();
        let len = match (code, previous) {
            (0..=255, _) => {
                out.push(code as u8);
                1
            }
            (258.., _) if code < next => {
                let (s, l) = table[code];
                out.extend_from_within(s..s + l);
                l
            }
            (_, Some((s, l))) if code == next => {
                // KwKwK: the previous string plus its own first byte.
                out.extend_from_within(s..s + l);
                out.push(out[s]);
                l + 1
            }
            _ => return (out, false),
        };
        if let Some((s, l)) = previous {
            if next < 4096 {
                table[next] = (s, l + 1);
                next += 1;
            }
        }
        previous = Some((start, len));
        let reach = next + usize::from(early);
        width = match reach {
            0..=511 => 9,
            512..=1023 => 10,
            1024..=2047 => 11,
            _ => 12,
        };
    }
}

/// RunLengthDecode (PDF 32000-1 7.4.5): length byte 0-127 copies the next n+1 bytes, 129-255 repeats the next byte
/// 257-n times, 128 is EOD. A run cut short by the end of the input stops with the prefix and `false`.
pub(crate) fn run_length_decode(data: &[u8]) -> (Vec<u8>, bool) {
    let mut out = Vec::with_capacity(data.len() * 2);
    let mut pos = 0;
    while let Some(&n) = data.get(pos) {
        pos += 1;
        match n {
            128 => return (out, true),
            0..=127 => {
                let end = pos + n as usize + 1;
                if end > data.len() {
                    out.extend_from_slice(&data[pos..]);
                    return (out, false);
                }
                out.extend_from_slice(&data[pos..end]);
                pos = end;
            }
            _ => {
                let Some(&b) = data.get(pos) else {
                    return (out, false);
                };
                pos += 1;
                out.extend(std::iter::repeat_n(b, 257 - n as usize));
            }
        }
    }
    (out, true)
}
// Byte ranges are found with the same parser used for decisions, but copied
// verbatim. Only top-level /ToUnicode values are replaced; nested keys survive.
struct DictEntry {
    key: Vec<u8>,
    start: usize,
    value: usize,
    end: usize,
}
fn dictionary_layout(bytes: &[u8]) -> Result<(Vec<DictEntry>, usize)> {
    if !bytes.starts_with(b"<<") {
        bail!("Expected font dictionary bytes")
    }
    let mut p = Parser::new(bytes, 2);
    let mut entries = vec![];
    loop {
        let start = p.p;
        p.skip();
        if bytes.get(p.p..p.p + 2) == Some(b">>") {
            if p.p + 2 != bytes.len() {
                bail!("Trailing font dictionary bytes")
            }
            return Ok((entries, start));
        }
        let key = match p.obj()? {
            Obj::Name(n) => n,
            _ => bail!("Invalid dictionary key"),
        };
        p.skip();
        let value = p.p;
        p.obj()?;
        entries.push(DictEntry {
            key,
            start,
            value,
            end: p.p,
        });
    }
}
pub fn without_tounicode(bytes: &[u8]) -> Result<Vec<u8>> {
    let (entries, _) = dictionary_layout(bytes)?;
    let mut out = vec![];
    let mut cursor = 0;
    for entry in entries.iter().filter(|e| e.key == b"ToUnicode") {
        out.extend(&bytes[cursor..entry.start]);
        cursor = entry.end;
    }
    out.extend(&bytes[cursor..]);
    Ok(out)
}
pub fn splice_tounicode(bytes: &[u8], id: u32) -> Result<Vec<u8>> {
    let (entries, closing) = dictionary_layout(bytes)?;
    let mut out = vec![];
    let mut cursor = 0;
    let mut found = false;
    for entry in entries.iter().filter(|e| e.key == b"ToUnicode") {
        out.extend(&bytes[cursor..entry.value]);
        out.extend(format!("{id} 0 R").bytes());
        cursor = entry.end;
        found = true;
    }
    if !found {
        out.extend(&bytes[..closing]);
        out.extend(format!("\n/ToUnicode {id} 0 R").bytes());
        cursor = closing;
    }
    out.extend(&bytes[cursor..]);
    // Fail before publication if any byte outside the top-level entry changed.
    if without_tounicode(bytes)? != without_tounicode(&out)? {
        bail!("Refusing unrelated font dictionary byte changes")
    }
    Ok(out)
}
pub fn serialize(o: &Obj) -> Vec<u8> {
    match o {
        Obj::Null => b"null".to_vec(),
        Obj::Bool(b) => b.to_string().into_bytes(),
        Obj::Num(s) => s.as_bytes().to_vec(),
        Obj::Name(s) => {
            let mut out = b"/".to_vec();
            for &c in s {
                if delim(c) || !(33..127).contains(&c) || c == b'#' {
                    out.extend(format!("#{c:02X}").bytes())
                } else {
                    out.push(c)
                }
            }
            out
        }
        Obj::Str(b) => format!(
            "<{}>",
            b.iter().map(|c| format!("{c:02X}")).collect::<String>()
        )
        .into_bytes(),
        Obj::Ref(n, g) => format!("{n} {g} R").into_bytes(),
        Obj::Array(a) => {
            let mut out = b"[ ".to_vec();
            for o in a {
                out.extend(serialize(o));
                out.push(b' ')
            }
            out.push(b']');
            out
        }
        Obj::Dict(d) => {
            let mut out = b"<<\n".to_vec();
            for (k, v) in d {
                out.extend(serialize(&Obj::Name(k.clone())));
                out.push(b' ');
                out.extend(serialize(v));
                out.push(10)
            }
            out.extend(b">>");
            out
        }
        Obj::Stream(d, b) => {
            let mut out = serialize(&Obj::Dict(d.clone()));
            out.extend(b"\nstream\n");
            out.extend(b.iter());
            out.extend(b"\nendstream");
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_object_stream_cannot_shadow_later_update() {
        let header = b"10 0 11 5 ";
        let mut stream = header.to_vec();
        stream.extend(b"null /Old");
        let mut bytes = format!(
            "1 0 obj\n<< /Type /ObjStm /N 2 /First {} /Length {} >>\nstream\n",
            header.len(),
            stream.len()
        )
        .into_bytes();
        bytes.extend(stream);
        bytes.extend(b"\nendstream\nendobj\n");
        let updated_offset = bytes.len();
        bytes.extend(b"11 0 obj /New endobj\n");
        let mut pdf = Pdf {
            bytes,
            entries: BTreeMap::from([
                (1, Entry::Offset(0)),
                (10, Entry::Compressed(1, 0)),
                (11, Entry::Offset(updated_offset)),
            ]),
            cache: BTreeMap::new(),
            trailer: Obj::Null,
            prev: 0,
            loading: HashSet::new(),
            crypt: None,
        };
        // Offset zero is reserved for deleted entries; add a prefix byte.
        pdf.bytes.insert(0, b'\n');
        pdf.entries.insert(1, Entry::Offset(1));
        pdf.entries.insert(11, Entry::Offset(updated_offset + 1));
        assert_eq!(pdf.object(10).unwrap(), Obj::Null);
        assert_eq!(pdf.object(11).unwrap(), Obj::Name("New".into()));
    }
}
