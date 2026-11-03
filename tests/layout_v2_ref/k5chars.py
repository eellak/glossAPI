"""`.k5chars` char-table fixtures (K5c; design k5_prep/design/rust_layout.md §2-§3). Standard library only.

One file = one page, little-endian, struct of arrays:
  header  magic b"K5CHARS\\0" | u16 version (=1) | u16 rotation (deg, /Rotate) | u32 n | u32 nfonts | u32 page (1-based)
          | f64 page_w | f64 page_h (unrotated media box, PDF points) | 32 bytes source sha256
  fonts   nfonts x (u16 byte length, UTF-8 font name)               font = index into this table (interned)
  arrays  ch u32[n] | bbox f64[4n] | loose f64[4n] | size f32[n] | weight i16[n] | angle f32[n] | font u32[n]
          | flags u8[n]
bbox = FPDFText_GetCharBox (tight) and loose = FPDFText_GetLooseCharBox, both stored as (left, bottom, right, top) in
PDF user space exactly as PDFium returns them (raw f64; rounding to 1/64 pt happens on ingestion in the layout code).
Context bundles (`.k5ctx`, one per document): magic b"K5CTX\\0\\0\\1" | u32 count | count x (u32 page | u64 length |
k5chars bytes); the files on disk are gzip-compressed with mtime 0 (deterministic).
"""
from __future__ import annotations

import struct
from array import array
from dataclasses import dataclass, field

MAGIC = b"K5CHARS\0"
CTX_MAGIC = b"K5CTX\0\0\1"
VERSION = 1
GENERATED, SOFT_HYPHEN, MAP_ERROR, ITALIC, SYMBOLIC, INVISIBLE = 1, 2, 4, 8, 16, 32
_HEAD = struct.Struct("<8sHHIII dd 32s")


@dataclass
class CharTable:
    page_w: float
    page_h: float
    rotation: int = 0
    page: int = 0
    source_sha256: bytes = b"\0" * 32
    fonts: list = field(default_factory=list)
    ch: list = field(default_factory=list)       # str (one code point each)
    bbox: list = field(default_factory=list)     # (l, b, r, t)
    loose: list = field(default_factory=list)    # (l, b, r, t)
    size: list = field(default_factory=list)
    weight: list = field(default_factory=list)
    angle: list = field(default_factory=list)
    font: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def __len__(self):
        return len(self.ch)


def _arr(code, values):
    a = array(code, values)
    if a.itemsize and struct.pack("=H", 1) != struct.pack("<H", 1):   # big-endian host
        a.byteswap()
    return a.tobytes()


def dumps(t: CharTable) -> bytes:
    n = len(t.ch)
    out = [_HEAD.pack(MAGIC, VERSION, t.rotation, n, len(t.fonts), t.page, t.page_w, t.page_h, t.source_sha256)]
    for name in t.fonts:
        b = name.encode("utf-8")
        out.append(struct.pack("<H", len(b)) + b)
    out.append(_arr("I", [ord(c) for c in t.ch]))
    out.append(_arr("d", [v for box in t.bbox for v in box]))
    out.append(_arr("d", [v for box in t.loose for v in box]))
    out.append(_arr("f", t.size))
    out.append(_arr("h", t.weight))
    out.append(_arr("f", t.angle))
    out.append(_arr("I", t.font))
    out.append(bytes(t.flags))
    return b"".join(out)


def loads(data: bytes) -> CharTable:
    magic, ver, rot, n, nf, page, w, h, sha = _HEAD.unpack_from(data, 0)
    if magic != MAGIC or ver != VERSION:
        raise ValueError(f"not a k5chars v{VERSION} table")
    off = _HEAD.size
    fonts = []
    for _ in range(nf):
        (ln,) = struct.unpack_from("<H", data, off)
        fonts.append(data[off + 2:off + 2 + ln].decode("utf-8"))
        off += 2 + ln

    def take(code, count):
        nonlocal off
        a = array(code)
        size = a.itemsize * count
        a.frombytes(data[off:off + size])
        if struct.pack("=H", 1) != struct.pack("<H", 1):
            a.byteswap()
        off += size
        return a.tolist()

    ch = [chr(c) for c in take("I", n)]
    bb, lo = take("d", 4 * n), take("d", 4 * n)
    size, weight, angle, font = take("f", n), take("h", n), take("f", n), take("I", n)
    flags = list(data[off:off + n])
    off += n
    if off != len(data):
        raise ValueError(f"k5chars: {len(data) - off} trailing bytes")
    return CharTable(w, h, rot, page, sha, fonts, ch, [tuple(bb[4 * i:4 * i + 4]) for i in range(n)],
                     [tuple(lo[4 * i:4 * i + 4]) for i in range(n)], size, weight, angle, font, flags)


def dumps_ctx(tables) -> bytes:
    out = [CTX_MAGIC, struct.pack("<I", len(tables))]
    for t in tables:
        b = dumps(t)
        out.append(struct.pack("<IQ", t.page, len(b)) + b)
    return b"".join(out)


def loads_ctx(data: bytes) -> list:
    if data[:2] == b"\x1f\x8b":                  # bundles are stored gzip-compressed (freeze_chars.py)
        import gzip
        data = gzip.decompress(data)
    if data[:8] != CTX_MAGIC:
        raise ValueError("not a k5ctx bundle")
    (count,) = struct.unpack_from("<I", data, 8)
    off, out = 12, []
    for _ in range(count):
        page, ln = struct.unpack_from("<IQ", data, off)
        t = loads(data[off + 12:off + 12 + ln])
        if t.page != page:
            raise ValueError("k5ctx index/page mismatch")
        out.append(t)
        off += 12 + ln
    return out
