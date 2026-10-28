use anyhow::{anyhow, bail, Result};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Deserialize)]
struct EncodedTables {
    uv2agl: BTreeMap<u32, String>,
    agl: BTreeMap<String, String>,
    post: Vec<String>,
    cff: Vec<String>,
    charsets: Vec<Vec<String>>,
    greek: BTreeSet<u32>,
}
struct Tables {
    uv2agl: BTreeMap<u32, String>,
    agl: BTreeMap<Vec<u8>, String>,
    post: Vec<Vec<u8>>,
    cff: Vec<Vec<u8>>,
    charsets: Vec<Vec<Vec<u8>>>,
    greek: BTreeSet<u32>,
}
static TABLES: Lazy<Tables> = Lazy::new(|| {
    let t: EncodedTables =
        serde_json::from_str(include_str!("../fixtures/font_tables.json")).unwrap();
    Tables {
        uv2agl: t.uv2agl,
        agl: t
            .agl
            .into_iter()
            .map(|(k, v)| (k.into_bytes(), v))
            .collect(),
        post: t.post.into_iter().map(String::into_bytes).collect(),
        cff: t.cff.into_iter().map(String::into_bytes).collect(),
        charsets: t
            .charsets
            .into_iter()
            .map(|cs| cs.into_iter().map(String::into_bytes).collect())
            .collect(),
        greek: t.greek,
    }
});
pub fn greek(s: &str) -> bool {
    s.chars().any(|c| TABLES.greek.contains(&(c as u32)))
}
// fontTools.agl.toUnicode(name), with its default isZapfDingbats=False.
// LEGACY_AGL2UV is case-sensitive. Python's regex `$` permits one final LF.
// Match its ASCII name vocabulary and lexical rules on exact bytes. Arbitrary
// post/CFF bytes must never become replacement characters or normalized text.
fn agl(name: &[u8]) -> String {
    name.split(|c| *c == b'.')
        .next()
        .unwrap_or(b"")
        .split(|c| *c == b'_')
        .map(agl_component)
        .collect()
}
fn agl_component(name: &[u8]) -> String {
    if let Some(s) = TABLES.agl.get(name) {
        return s.clone();
    }
    let name = name.strip_suffix(b"\n").unwrap_or(name);
    let (digits, bmp) = if let Some(s) = name.strip_prefix(b"uni") {
        (s, true)
    } else if let Some(s) = name.strip_prefix(b"u") {
        (s, false)
    } else {
        return String::new();
    };
    if digits.is_empty()
        || !digits
            .iter()
            .all(|c| c.is_ascii_digit() || (b'A'..=b'F').contains(c))
    {
        return String::new();
    }
    if bmp {
        if digits.len() % 4 != 0 {
            return String::new();
        }
        let mut out = String::new();
        for chunk in digits.chunks(4) {
            let n = u32::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16).unwrap();
            if let Some(c) = char::from_u32(n) {
                out.push(c);
            } else {
                return String::new();
            }
        }
        out
    } else if (4..=6).contains(&digits.len()) {
        u32::from_str_radix(std::str::from_utf8(digits).unwrap(), 16)
            .ok()
            .and_then(char::from_u32)
            .map(|c| c.to_string())
            .unwrap_or_default()
    } else {
        String::new()
    }
}
// post.build_psNameMapping: retain the first name in glyph order; suffix later
// occurrences, skipping suffixes already present anywhere in the original order.
fn unique_post_names(names: &mut [Vec<u8>]) {
    let original: BTreeSet<_> = names.iter().cloned().collect();
    let mut used: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    for (gid, name) in names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("glyph{gid:05}").into_bytes();
        }
        if let Some(mut n) = used.get(name).copied() {
            let suffixed = |n| [name.as_slice(), format!(".{n}").as_bytes()].concat();
            while original.contains(&suffixed(n)) {
                n += 1;
            }
            used.insert(name.clone(), n + 1);
            *name = suffixed(n);
        }
        used.insert(name.clone(), 1);
    }
}
fn slice(b: &[u8], p: usize, n: usize) -> Result<&[u8]> {
    b.get(
        p..p.checked_add(n)
            .ok_or_else(|| anyhow!("Font offset overflow"))?,
    )
    .ok_or_else(|| anyhow!("Truncated font table"))
}
fn u16b(b: &[u8], p: usize) -> Result<usize> {
    let a = slice(b, p, 2)?;
    Ok(u16::from_be_bytes([a[0], a[1]]) as usize)
}
fn u32b(b: &[u8], p: usize) -> Result<usize> {
    let a = slice(b, p, 4)?;
    Ok(u32::from_be_bytes(a.try_into().unwrap()) as usize)
}
#[derive(Default)]
pub struct Glyphs {
    pub mapping: BTreeMap<usize, String>,
    pub names: BTreeMap<Vec<u8>, usize>,
    pub symbol: BTreeMap<u32, usize>,
    pub cids: BTreeMap<u32, usize>,
}
fn cmap(b: &[u8]) -> Result<BTreeMap<u32, usize>> {
    let fmt = u16b(b, 0)?;
    let mut out = BTreeMap::new();
    match fmt {
        0 => {
            slice(b, 6, 256)?;
            for i in 0..256 {
                if b[6 + i] != 0 {
                    out.insert(i as u32, b[6 + i] as usize);
                }
            }
        }
        4 => {
            let len = u16b(b, 2)?;
            if len > b.len() {
                bail!(
                    "TTLibError: cmap subtable is truncated: needs {} bytes, only {} available",
                    len,
                    b.len()
                )
            }
            let seg = u16b(b, 6)? / 2;
            let min = 16 + seg * 8;
            if len < min {
                bail!(
                    "TTLibError: cmap subtable is truncated: needs {} bytes, only {} available",
                    min - 6,
                    len.saturating_sub(6)
                )
            }
            let b = slice(b, 0, len)?;
            let ends = 14;
            let starts = 16 + seg * 2;
            let deltas = 16 + seg * 4;
            let offsets = 16 + seg * 6;
            for i in 0..seg {
                let start = u16b(b, starts + i * 2)?;
                let stop = u16b(b, ends + i * 2)?;
                let delta = u16b(b, deltas + i * 2)?;
                let ro = u16b(b, offsets + i * 2)?;
                for c in start..=stop {
                    if c == 65535 {
                        continue;
                    }
                    let mut gid = if ro == 0 {
                        (c + delta) & 65535
                    } else {
                        u16b(b, offsets + i * 2 + ro + (c - start) * 2)?
                    };
                    if ro != 0 && gid != 0 {
                        gid = (gid + delta) & 65535
                    }
                    if gid != 0 {
                        out.insert(c as u32, gid);
                    }
                }
            }
        }
        6 => {
            let first = u16b(b, 6)?;
            let count = u16b(b, 8)?;
            for i in 0..count {
                let g = u16b(b, 10 + i * 2)?;
                if g != 0 {
                    out.insert((first + i) as u32, g);
                }
            }
        }
        10 => {
            let first = u32b(b, 12)?;
            let count = u32b(b, 16)?;
            for i in 0..count {
                let g = u16b(b, 20 + i * 2)?;
                if g != 0 {
                    out.insert((first + i) as u32, g);
                }
            }
        }
        12 | 13 => {
            let count = u32b(b, 12)?;
            for i in 0..count {
                let p = 16 + i * 12;
                let start = u32b(b, p)?;
                let stop = u32b(b, p + 4)?;
                let g = u32b(b, p + 8)?;
                if stop < start || stop - start > 0x110000 {
                    bail!("Excessive cmap range")
                }
                for c in start..=stop {
                    let g = if fmt == 12 { g + c - start } else { g };
                    if g != 0 {
                        out.insert(c as u32, g);
                    }
                }
            }
        }
        2 => {
            let keys = (0..256)
                .map(|i| u16b(b, 6 + i * 2))
                .collect::<Result<Vec<_>>>()?;
            for hi in 0..256 {
                let key = keys[hi];
                let p = 518 + key;
                let first = u16b(b, p)?;
                let count = u16b(b, p + 2)?;
                let delta = u16b(b, p + 4)?;
                let off = u16b(b, p + 6)?;
                if key == 0 && hi != 0 {
                    continue;
                }
                for lo in first..first + count {
                    let mut gid = u16b(b, p + 6 + off + (lo - first) * 2)?;
                    if gid != 0 {
                        gid = (gid + delta) & 65535;
                        out.insert(
                            if key == 0 {
                                lo as u32
                            } else {
                                (hi * 256 + lo) as u32
                            },
                            gid,
                        );
                    }
                }
            }
        }
        14 => (),
        _ => bail!("Unsupported cmap format {fmt}"),
    };
    Ok(out)
}
fn index(b: &[u8], p: &mut usize) -> Result<Vec<Vec<u8>>> {
    let count = u16b(b, *p)?;
    *p += 2;
    if count == 0 {
        return Ok(vec![]);
    }
    let width = *slice(b, *p, 1)?.first().unwrap() as usize;
    *p += 1;
    if !(1..=4).contains(&width) {
        bail!("Invalid CFF INDEX")
    }
    let mut offsets = vec![];
    for _ in 0..=count {
        let mut n = 0usize;
        for c in slice(b, *p, width)? {
            n = n * 256 + *c as usize
        }
        offsets.push(n);
        *p += width
    }
    let base = *p;
    let mut out = vec![];
    for i in 0..count {
        if offsets[i] == 0 || offsets[i + 1] < offsets[i] {
            bail!("Invalid CFF offset")
        }
        out.push(slice(b, base + offsets[i] - 1, offsets[i + 1] - offsets[i])?.to_vec())
    }
    *p = base + offsets[count] - 1;
    Ok(out)
}
fn cff_dict(b: &[u8]) -> Result<BTreeMap<u16, Vec<i64>>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![];
    let mut p = 0;
    while p < b.len() {
        let v = b[p];
        p += 1;
        let n = match v {
            32..=246 => Some(v as i64 - 139),
            247..=250 => {
                let c = slice(b, p, 1)?[0];
                p += 1;
                Some((v as i64 - 247) * 256 + c as i64 + 108)
            }
            251..=254 => {
                let c = slice(b, p, 1)?[0];
                p += 1;
                Some(-(v as i64 - 251) * 256 - c as i64 - 108)
            }
            28 => {
                let n = u16b(b, p)? as u16 as i16 as i64;
                p += 2;
                Some(n)
            }
            29 => {
                let n = u32b(b, p)? as u32 as i32 as i64;
                p += 4;
                Some(n)
            }
            30 => {
                loop {
                    let c = slice(b, p, 1)?[0];
                    p += 1;
                    if c & 15 == 15 || c >> 4 == 15 {
                        break;
                    }
                }
                Some(0)
            }
            _ => None,
        };
        if let Some(n) = n {
            stack.push(n)
        } else {
            let op = if v == 12 {
                let c = slice(b, p, 1)?[0];
                p += 1;
                0x0c00 + c as u16
            } else {
                v as u16
            };
            out.insert(op, std::mem::take(&mut stack));
        }
    }
    Ok(out)
}
fn cff_names(b: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut p = slice(b, 2, 1)?[0] as usize;
    index(b, &mut p)?;
    let tops = index(b, &mut p)?;
    let strings = index(b, &mut p)?;
    let top = cff_dict(
        tops.first()
            .ok_or_else(|| anyhow!("Missing CFF top dictionary"))?,
    )?;
    let charset = top.get(&15).and_then(|v| v.first()).copied().unwrap_or(0) as usize;
    let cs = top
        .get(&17)
        .and_then(|v| v.first())
        .copied()
        .ok_or_else(|| anyhow!("Missing CFF CharStrings"))? as usize;
    let mut q = cs;
    let count = index(b, &mut q)?.len();
    if charset <= 2 {
        return Ok(TABLES.charsets[charset]
            .iter()
            .take(count)
            .cloned()
            .collect());
    }
    let cid = top.contains_key(&0x0c1e);
    let name = |sid: usize| -> Result<Vec<u8>> {
        if cid {
            Ok(format!("cid{sid:05}").into_bytes())
        } else if sid < TABLES.cff.len() {
            Ok(TABLES.cff[sid].clone())
        } else {
            Ok(slice(
                strings
                    .get(sid - TABLES.cff.len())
                    .ok_or_else(|| anyhow!("Invalid CFF SID"))?,
                0,
                strings[sid - TABLES.cff.len()].len(),
            )?
            .to_vec())
        }
    };
    let mut out = vec![".notdef".into()];
    p = charset;
    let fmt = slice(b, p, 1)?[0];
    p += 1;
    while out.len() < count {
        let first = u16b(b, p)?;
        p += 2;
        let left = match fmt {
            0 => 0,
            1 => {
                let n = slice(b, p, 1)?[0] as usize;
                p += 1;
                n
            }
            2 => {
                let n = u16b(b, p)?;
                p += 2;
                n
            }
            _ => bail!("Invalid CFF charset"),
        };
        for sid in first..=first + left {
            out.push(name(sid)?);
            if out.len() == count {
                break;
            }
        }
    }
    Ok(out)
}
pub fn glyphs(b: &[u8], kind: &str) -> Result<Glyphs> {
    let sfnt = b
        .get(..4)
        .map(|s| s == [0, 1, 0, 0] || s == b"OTTO" || s == b"true")
        .unwrap_or(false);
    if !sfnt {
        if kind != "FontFile3/CIDFontType0C" {
            bail!("TTLibError: Not a TrueType or OpenType font (bad sfntVersion)")
        }
        let names = cff_names(b)?;
        return finish(names, BTreeMap::new(), BTreeMap::new(), true, true);
    }
    let count = u16b(b, 4)?;
    let mut tables = BTreeMap::new();
    for i in 0..count {
        let p = 12 + i * 16;
        let tag = slice(b, p, 4)?.to_vec();
        let off = u32b(b, p + 8)?;
        let len = u32b(b, p + 12)?;
        tables.insert(tag, slice(b, off, len)?);
    }
    let ng = tables
        .get(b"maxp".as_slice())
        .map(|t| u16b(t, 4))
        .transpose()?
        .unwrap_or(0);
    let mut candidates: BTreeMap<usize, BTreeSet<char>> = BTreeMap::new();
    let mut symbol = BTreeMap::new();
    let mut reverse: BTreeMap<usize, u32> = BTreeMap::new();
    if let Some(t) = tables.get(b"cmap".as_slice()) {
        let count = u16b(t, 2)?;
        for i in 0..count {
            let p = 4 + i * 8;
            let platform = u16b(t, p)?;
            let enc = u16b(t, p + 2)?;
            let off = u32b(t, p + 4)?;
            let unicode = platform == 0 || (platform == 3 && [0, 1, 10].contains(&enc));
            if !unicode {
                continue;
            }
            let m = cmap(t.get(off..).ok_or_else(|| anyhow!("Bad cmap offset"))?)?;
            for (c, g) in m {
                reverse
                    .entry(g)
                    .and_modify(|n| *n = (*n).min(c))
                    .or_insert(c);
                if g >= ng {
                    bail!("KeyError: 'glyph{g:05}'")
                }
                if (platform, enc) == (3, 0) {
                    if c < 256 || (0xf000..=0xf0ff).contains(&c) {
                        let c = c & 255;
                        if symbol.insert(c, g).map(|old| old != g).unwrap_or(false) {
                            bail!("Ambiguous symbol cmap")
                        }
                    }
                } else if let Some(c) = char::from_u32(c) {
                    candidates.entry(g).or_default().insert(c);
                }
            }
        }
    }
    let mut has_names = false;
    let is_cff = tables.contains_key(b"CFF ".as_slice());
    let mut names = (0..ng)
        .map(|g| {
            if g == 0 {
                ".notdef".into()
            } else {
                format!("glyph{g:05}").into_bytes()
            }
        })
        .collect::<Vec<_>>();
    if let Some(t) = tables.get(b"CFF ".as_slice()) {
        names = cff_names(t)?;
        has_names = true
    } else if let Some(t) = tables.get(b"post".as_slice()) {
        let format = u32b(t, 0)?;
        match format {
            0x10000 => {
                has_names = true;
                for i in 0..ng.min(TABLES.post.len()) {
                    names[i] = TABLES.post[i].clone()
                }
            }
            0x20000 => {
                has_names = true;
                let count = u16b(t, 32)?.min(ng);
                names.fill(Vec::new());
                let ids = (0..count)
                    .map(|i| u16b(t, 34 + i * 2))
                    .collect::<Result<Vec<_>>>()?;
                let mut p = 34 + count * 2;
                let max = ids.iter().copied().max().unwrap_or(0);
                let mut custom = vec![];
                for _ in 258..=max {
                    let len = slice(t, p, 1)?[0] as usize;
                    p += 1;
                    custom.push(slice(t, p, len)?.to_vec());
                    p += len
                }
                for (i, id) in ids.into_iter().enumerate().take(ng) {
                    names[i] = if id < 258 {
                        TABLES.post[id].clone()
                    } else {
                        custom[id - 258].clone()
                    }
                }
                unique_post_names(&mut names);
            }
            0x25000 => {
                has_names = true;
                for i in 0..ng {
                    let delta = slice(t, 32 + i, 1)?[0] as i8;
                    let idx = (i as i32 + delta as i32) as usize;
                    names[i] = TABLES
                        .post
                        .get(idx)
                        .ok_or_else(|| anyhow!("Invalid post name"))?
                        .clone()
                }
            }
            0x30000 => (),
            _ => bail!(
                "TTLibError: 'post' table format {:.6} not supported",
                format as f64 / 65536.0
            ),
        }
    }
    if !has_names {
        // TTFont._getGlyphNamesFromCmap uses minimum Unicode position per GID,
        // then .alt1, .alt2, ... for repeated generated names in GID order.
        let mut used = BTreeMap::<String, usize>::new();
        for (g, c) in reverse {
            let n = TABLES.uv2agl.get(&c).cloned().unwrap_or_else(|| {
                if c <= 0xffff {
                    format!("uni{c:04X}")
                } else {
                    format!("u{c:X}")
                }
            });
            let count = used.entry(n.clone()).or_default();
            names[g] = if *count == 0 {
                n.into_bytes()
            } else {
                format!("{n}.alt{count}").into_bytes()
            };
            *count += 1;
        }
    }
    finish(names, candidates, symbol, has_names, is_cff)
}
fn finish(
    names: Vec<Vec<u8>>,
    candidates: BTreeMap<usize, BTreeSet<char>>,
    symbol: BTreeMap<u32, usize>,
    has_names: bool,
    is_cff: bool,
) -> Result<Glyphs> {
    let mut out = Glyphs {
        symbol,
        ..Default::default()
    };
    for (g, n) in names.into_iter().enumerate() {
        let named = if has_names { agl(&n) } else { String::new() };
        let vs = candidates.get(&g).cloned().unwrap_or_default();
        let value = if vs.len() == 1 {
            vs.iter().next().unwrap().to_string()
        } else if !named.is_empty() && (vs.is_empty() || vs.iter().any(|c| c.to_string() == named))
        {
            named
        } else if !vs.is_empty() {
            vs.iter()
                .min_by_key(|c| {
                    (
                        !((0x370..=0x3ff).contains(&(**c as u32)) && c.is_alphabetic()),
                        **c as u32,
                    )
                })
                .unwrap()
                .to_string()
        } else {
            String::new()
        };
        if !value.is_empty() {
            out.mapping.insert(g, value);
        }
        if is_cff {
            if let Some(cid) = n
                .strip_prefix(b"cid")
                .and_then(|s| std::str::from_utf8(s).ok())
                .and_then(|s| s.parse::<u32>().ok())
            {
                out.cids.insert(cid, g);
            }
        }
        out.names.insert(n, g);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glyph_name_identity_and_agl_use_exact_bytes() {
        let g = glyphs(
            include_bytes!("../fixtures/review/name_decoding_different_raw_byte.ttf"),
            "FontFile2",
        )
        .unwrap();
        assert_eq!(g.names.get(b"X\xE9Y".as_slice()), Some(&1));
        assert!(!g.names.contains_key(b"X\xF1Y".as_slice()));
        assert!(!g.names.contains_key("X�Y".as_bytes()));
        assert_eq!(g.mapping[&1], "α");
        let mut names = vec![
            b"X\xE9Y".to_vec(),
            b"X\xF1Y".to_vec(),
            b"X\xE9Y".to_vec(),
            b"X\xE9Y.1".to_vec(),
        ];
        unique_post_names(&mut names);
        assert_eq!(
            names,
            vec![
                b"X\xE9Y".to_vec(),
                b"X\xF1Y".to_vec(),
                b"X\xE9Y.2".to_vec(),
                b"X\xE9Y.1".to_vec()
            ]
        );
        for name in [b"X\xE9Y".as_slice(), b"X\xF1Y", "XéY".as_bytes()] {
            assert_eq!(agl(name), "");
        }
        assert_eq!(agl(b"X\xE9Y_alpha"), "α");
        assert_eq!(agl(b"alpha.\xE9"), "α");
    }
    #[test]
    fn agl_matches_fonttools_name_set_and_lexical_rules() {
        let expected: BTreeMap<String, String> =
            serde_json::from_str(include_str!("../fixtures/review/agl_expected.json")).unwrap();
        for (name, value) in expected {
            assert_eq!(agl(name.as_bytes()), value, "AGL name {name:?}");
        }
    }
    #[test]
    fn review_real_and_synthesized_duplicate_glyph_names() {
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/review/expected.json")).unwrap();
        for name in [
            "duplicate_post_names",
            "format3_duplicate_synth_names",
            "format3_collision_wrong_gid",
        ] {
            let bytes = std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("fixtures/review/{name}.ttf")),
            )
            .unwrap();
            let g = glyphs(&bytes, "FontFile2").unwrap();
            assert_eq!(
                serde_json::to_value(
                    g.names
                        .iter()
                        .map(|(n, gid)| (std::str::from_utf8(n).unwrap(), gid))
                        .collect::<BTreeMap<_, _>>()
                )
                .unwrap(),
                expected[name]["names"],
                "{name} names"
            );
            assert_eq!(
                serde_json::to_value(&g.mapping).unwrap(),
                expected[name]["glyphs"],
                "{name} mapping"
            );
        }
    }
    #[test]
    fn nameless_aliases_prefer_greek_block_letters() {
        let g = glyphs(
            include_bytes!("../fixtures/alias_nameless.ttf"),
            "FontFile2",
        )
        .unwrap();
        assert_eq!(g.mapping[&1], "α");
    }
    #[test]
    fn many_glyphs_can_have_one_unicode_value() {
        let g = glyphs(include_bytes!("../fixtures/many_to_one.ttf"), "FontFile2").unwrap();
        assert_eq!(g.mapping[&1], "σ");
        assert_eq!(g.mapping[&2], "σ");
    }
    #[test]
    fn post_names_and_multi_unicode_fallback() {
        let g = glyphs(include_bytes!("../fixtures/post_only.ttf"), "FontFile2").unwrap();
        assert_eq!(g.mapping[&1], "α");
        assert_eq!(g.mapping[&2], "β");
        assert_eq!(g.mapping[&3], "fi");
    }
}
