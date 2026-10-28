use anyhow::{anyhow, bail, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::BTreeMap;
static COMMENTS: Lazy<Regex> = Lazy::new(|| Regex::new(r"%[^\r\n]*").unwrap());
static TOKENS: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^<>]*>|\[|\]|[^\s<>\[\]]+").unwrap());
pub fn tokens(b: &[u8]) -> Vec<String> {
    let s = String::from_utf8_lossy(b);
    let s = COMMENTS.replace_all(&s, "");
    TOKENS
        .find_iter(&s)
        .map(|m| m.as_str().to_string())
        .collect()
}
fn hex(s: &str) -> Result<Vec<u8>> {
    if !s.starts_with('<') || !s.ends_with('>') {
        bail!("Invalid CMap hex string")
    }
    let mut s = s[1..s.len() - 1]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    if s.len() % 2 == 1 {
        s.push('0')
    }
    Ok((0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect::<std::result::Result<Vec<_>, _>>()?)
}
fn num(b: &[u8]) -> Result<u32> {
    if b.len() > 4 {
        bail!("Excessive CMap code")
    }
    Ok(b.iter().fold(0, |n, c| n * 256 + *c as u32))
}
fn utf16(b: &[u8]) -> Result<String> {
    let units = b
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect::<Vec<_>>();
    let mut out = String::new();
    for (i, c) in char::decode_utf16(units.iter().copied()).enumerate() {
        match c {
            Ok(c) => out.push(c),
            Err(_) => {
                let pos = i * 2;
                let reason = if units[i] >= 0xd800 && units[i] <= 0xdbff {
                    if i + 1 == units.len() {
                        "unexpected end of data"
                    } else {
                        "illegal UTF-16 surrogate"
                    }
                } else {
                    "illegal encoding"
                };
                bail!("UnicodeDecodeError: 'utf-16-be' codec can't decode bytes in position {}-{}: {}",pos,pos+1,reason)
            }
        }
    }
    if b.len() % 2 == 1 {
        bail!("UnicodeDecodeError: 'utf-16-be' codec can't decode byte 0x{:02x} in position {}: truncated data",b[b.len()-1],b.len()-1)
    }
    Ok(out)
}
#[derive(Clone, PartialEq)]
pub enum Value {
    Text(String),
    Cid(u32),
}
pub fn entries(b: &[u8], unicode: bool) -> Result<BTreeMap<u32, Value>> {
    let t = tokens(b);
    if t.iter().any(|s| s.contains("usecmap")) {
        bail!("Inherited CMaps (usecmap) are unsupported")
    }
    let mut out = BTreeMap::new();
    let mut i = 0;
    while i < t.len() {
        let kind = &t[i];
        let kinds = if unicode {
            ["beginbfchar", "beginbfrange"]
        } else {
            ["begincidchar", "begincidrange"]
        };
        if !kinds.contains(&kind.as_str()) {
            i += 1;
            continue;
        }
        let count = t
            .get(i.wrapping_sub(1))
            .ok_or_else(|| anyhow!("Invalid CMap count"))?
            .parse::<usize>()?;
        if count > 65536 {
            bail!("Excessive CMap block")
        }
        i += 1;
        for _ in 0..count {
            let start = num(&hex(t.get(i).ok_or_else(|| anyhow!("Truncated CMap"))?)?)?;
            i += 1;
            let stop = if kind.ends_with("char") {
                start
            } else {
                let n = num(&hex(t.get(i).ok_or_else(|| anyhow!("Truncated CMap"))?)?)?;
                i += 1;
                n
            };
            if stop < start || stop - start >= 65536 {
                bail!("Invalid/excessive CMap range")
            }
            let token = t.get(i).ok_or_else(|| anyhow!("Truncated CMap"))?;
            let array = token == "[";
            let mut bytes = vec![];
            let mut cid = 0;
            if array {
                i += 1
            } else {
                if unicode {
                    bytes = hex(token)?
                } else {
                    cid = token.parse()?
                }
                i += 1
            }
            for code in start..=stop {
                let value = if array {
                    let v = utf16(&hex(t.get(i).ok_or_else(|| anyhow!("Truncated CMap"))?)?)?;
                    i += 1;
                    Value::Text(v)
                } else if unicode {
                    Value::Text(utf16(&bytes)?)
                } else {
                    Value::Cid(cid)
                };
                if out
                    .insert(code, value.clone())
                    .map(|v| v != value)
                    .unwrap_or(false)
                {
                    bail!("Conflicting CMap definitions")
                }
                if code != stop {
                    if unicode && !array {
                        let mut carry = true;
                        for c in bytes.iter_mut().rev() {
                            if carry {
                                let (v, overflow) = c.overflowing_add(1);
                                *c = v;
                                carry = overflow
                            }
                        }
                        if carry {
                            bail!("CMap range overflow")
                        }
                    } else {
                        cid += 1
                    }
                }
            }
            if array {
                if t.get(i).map(String::as_str) != Some("]") {
                    bail!("Invalid CMap range array")
                }
                i += 1
            }
        }
        let end = kind.replacen("begin", "end", 1);
        if t.get(i) != Some(&end) {
            bail!("Invalid CMap block terminator")
        }
        i += 1;
    }
    Ok(out)
}
pub fn text_entries(b: &[u8]) -> Result<BTreeMap<u32, String>> {
    Ok(entries(b, true)?
        .into_iter()
        .filter_map(|(n, v)| {
            if let Value::Text(s) = v {
                Some((n, s))
            } else {
                None
            }
        })
        .collect())
}
pub fn make(mapping: &BTreeMap<u32, String>, width: usize) -> Result<Vec<u8>> {
    let maximum = if width == 1 { 255 } else { 65535 };
    if mapping.keys().any(|n| *n > maximum) {
        bail!("CMap source code does not fit the output codespace")
    }
    let mut lines = vec![
        "/CIDInit /ProcSet findresource begin".into(),
        "12 dict begin".into(),
        "begincmap".into(),
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def".into(),
        "/CMapName /GlossAPI-UCS def".into(),
        "/CMapType 2 def".into(),
        "1 begincodespacerange".into(),
        format!("<{}> <{}>", "00".repeat(width), "FF".repeat(width)),
        "endcodespacerange".into(),
    ];
    let entries = mapping.iter().collect::<Vec<_>>();
    for block in entries.chunks(100) {
        lines.push(format!("{} beginbfchar", block.len()));
        for (n, s) in block {
            lines.push(format!(
                "<{n:0w$X}> <{}>",
                s.encode_utf16()
                    .map(|n| format!("{n:04X}"))
                    .collect::<String>(),
                w = width * 2
            ));
        }
        lines.push("endbfchar".into())
    }
    lines.extend(
        [
            "endcmap",
            "CMapName currentdict /CMap defineresource pop",
            "end",
            "end",
        ]
        .map(String::from),
    );
    Ok((lines.join("\n") + "\n").into_bytes())
}
