//! K5-D (K5A2B-G4): the PDF standard security handler for RC4 documents (V 1/2, R 2/3) opened with the empty user
//! password, for the content-stream decode check only (`Pdf::open_for_check`). The repair path never decrypts:
//! `Pdf::open` still refuses every encrypted document. MD5 (RFC 1321) and RC4 are implemented here because the
//! offline dependency lock has no crate for either; both are tested against their published vectors.
use anyhow::{bail, Result};

/// MD5 digest of `data` (RFC 1321).
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15,
        21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend(bits.to_le_bytes());
    for chunk in msg.chunks(64) {
        let m: Vec<u32> = chunk.chunks(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, v) in [a0, b0, c0, d0].iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// RC4 of `data` with `key` (encryption and decryption are the same).
pub fn rc4(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut s: Vec<u8> = (0..=255).collect();
    let mut j = 0u8;
    for i in 0..256 {
        j = j.wrapping_add(s[i]).wrapping_add(key[i % key.len()]);
        s.swap(i, j as usize);
    }
    let (mut i, mut j) = (0u8, 0u8);
    data.iter()
        .map(|&b| {
            i = i.wrapping_add(1);
            j = j.wrapping_add(s[i as usize]);
            s.swap(i as usize, j as usize);
            b ^ s[(s[i as usize].wrapping_add(s[j as usize])) as usize]
        })
        .collect()
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
const PAD: [u8; 32] = [
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41, 0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08, 0x2E, 0x2E, 0x00, 0xB6,
    0xD0, 0x68, 0x3E, 0x80, 0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
];

/// The file key of an RC4 standard-security document for the empty user password (PDF 32000-1 algorithm 2), verified
/// against `/U` (algorithms 4/5). `Err` when the handler is not one this check implements (AES, V 4/5, another
/// filter) or the empty password is not the user password.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
pub fn empty_password_key(filter: &str, v: i64, r: i64, length_bits: i64, o: &[u8], u: &[u8], p: i64, id0: &[u8]) -> Result<Vec<u8>> {
    if filter != "Standard" {
        bail!("Encrypted PDFs are unsupported: security handler {filter}")
    }
    if !matches!(v, 1 | 2) || !matches!(r, 2 | 3) {
        bail!("Encrypted PDFs are unsupported: V {v} R {r} (only RC4 V 1/2, R 2/3 are decrypted by the check)")
    }
    let n = if r == 2 { 5 } else { (length_bits.clamp(40, 128) / 8) as usize };
    if o.len() < 32 || u.len() < 16 {
        bail!("Encrypted PDFs are unsupported: malformed /O or /U")
    }
    let mut input = PAD.to_vec();
    input.extend_from_slice(&o[..32]);
    input.extend_from_slice(&(p as i32 as u32).to_le_bytes());
    input.extend_from_slice(id0);
    let mut digest = md5(&input);
    if r >= 3 {
        for _ in 0..50 {
            digest = md5(&digest[..n]);
        }
    }
    let key = digest[..n].to_vec();
    let ok = if r == 2 {
        rc4(&key, &PAD)[..] == u[..32.min(u.len())]
    } else {
        let mut x = md5(&[&PAD[..], id0].concat()).to_vec();
        x = rc4(&key, &x);
        for i in 1..=19u8 {
            let k: Vec<u8> = key.iter().map(|b| b ^ i).collect();
            x = rc4(&k, &x);
        }
        x[..16] == u[..16]
    };
    if !ok {
        bail!("Encrypted PDFs are unsupported: a user password is required")
    }
    Ok(key)
}

/// The RC4 key of object `num` generation `gen` (algorithm 1).
pub fn object_key(key: &[u8], num: u32, gen: u16) -> Vec<u8> {
    let mut input = key.to_vec();
    input.extend_from_slice(&num.to_le_bytes()[..3]);
    input.extend_from_slice(&gen.to_le_bytes());
    md5(&input)[..(key.len() + 5).min(16)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
    #[test]
    fn empty_password_key_refuses_what_it_does_not_implement_and_round_trips_rc4() {
        // A document whose /U was made with the empty user password (algorithm 5, R 3) opens; another /U does not.
        let (o, p, id0) = ([7u8; 32], -1068i64, b"0123456789abcdef".to_vec());
        let mut input = PAD.to_vec();
        input.extend_from_slice(&o);
        input.extend_from_slice(&(p as i32 as u32).to_le_bytes());
        input.extend_from_slice(&id0);
        let mut digest = md5(&input);
        for _ in 0..50 {
            digest = md5(&digest[..16]);
        }
        let key = digest[..16].to_vec();
        let mut x = rc4(&key, &md5(&[&PAD[..], &id0[..]].concat()));
        for i in 1..=19u8 {
            x = rc4(&key.iter().map(|b| b ^ i).collect::<Vec<u8>>(), &x);
        }
        let mut u = x.clone();
        u.extend([0u8; 16]);
        assert_eq!(empty_password_key("Standard", 2, 3, 128, &o, &u, p, &id0).unwrap(), key);
        let mut wrong = u.clone();
        wrong[0] ^= 1;
        assert!(empty_password_key("Standard", 2, 3, 128, &o, &wrong, p, &id0).unwrap_err().to_string().contains("user password"));
        assert!(empty_password_key("Standard", 4, 4, 128, &o, &u, p, &id0).unwrap_err().to_string().contains("V 4 R 4"));
        assert!(empty_password_key("Adobe.PubSec", 2, 3, 128, &o, &u, p, &id0).unwrap_err().to_string().contains("handler"));
        // Object keys: n + 5 bytes, at most 16; RC4 is its own inverse.
        assert_eq!(object_key(&key[..5], 12, 0).len(), 10);
        assert_eq!(object_key(&key, 12, 0).len(), 16);
        let k = object_key(&key, 12, 0);
        assert_eq!(rc4(&k, &rc4(&k, b"BT (x) Tj ET")), b"BT (x) Tj ET");
    }

    #[test]
    fn md5_and_rc4_published_vectors() {
        assert_eq!(hex(&md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(&md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(&md5(&[b'a'; 1000])), "cabe45dcc9ae5b66ba86600cca6b8ba8");
        assert_eq!(hex(&md5(b"The quick brown fox jumps over the lazy dog")), "9e107d9d372bb6826bd81d3542a419d6");
        assert_eq!(hex(&rc4(b"Key", b"Plaintext")), "bbf316e8d940af0ad3");
        assert_eq!(hex(&rc4(b"Wiki", b"pedia")), "1021bf0420");
        assert_eq!(hex(&rc4(b"Secret", b"Attack at dawn")), "45a01f645fc35b383552544b9bf5");
    }
}
