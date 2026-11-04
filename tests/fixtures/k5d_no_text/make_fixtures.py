"""Generate the K5-D "no text" fixtures (K5A-M2): small hand-written PDFs for the paths where the native text sample
and the content-stream decode check meet structure PDFium handles on its own.

Run ``python make_fixtures.py`` to (re)write the PDFs beside this script; ``python make_fixtures.py --check`` exits 1
if a checked-in file differs from what this script builds (the tests run the same check). Standard library only;
the bytes are deterministic (no timestamps, no ids).

Fixtures (one Helvetica text line per page unless stated):
- ``lzw_content.pdf``: the page's content stream is LZWDecode (EarlyChange 1), long enough to reach 10-bit codes.
- ``runlength_content.pdf``: the content stream is RunLengthDecode.
- ``rebuilt_xref.pdf``: every xref offset points 7 bytes too early; PDFium rebuilds the cross-reference table.
- ``broken_startxref.pdf``: ``startxref`` points past the end of the file; PDFium rebuilds.
- ``count_more.pdf``: the page tree says ``/Count 3`` over 2 leaves (PDFium trusts a positive /Count).
- ``count_less.pdf``: ``/Count 1`` over 2 leaves.
- ``no_text_layer.pdf``: two pages that draw only paths and an inline image (no text operator): has_text false.
- ``whitespace_text.pdf``: a page whose only text operator shows spaces: has_text false (``str.strip`` semantics).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"


def lzw_encode(data: bytes, early: bool = True) -> bytes:
    """PDF LZWDecode encoder: Clear first, EOD last; the k-th code after Clear (k >= 2) is written with the width the
    decoder uses then (a table of 256 + k entries, EarlyChange counted)."""
    table = {bytes([i]): i for i in range(256)}
    next_code, codes, word = 258, [], b""
    for value in data:
        candidate = word + bytes([value])
        if candidate in table:
            word = candidate
            continue
        codes.append(table[word])
        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
        word = bytes([value])
    if word:
        codes.append(table[word])
    codes.append(257)

    def width(k):
        if k < 2:
            return 9
        reach = min(256 + k, 4096) + int(early)
        return 9 if reach < 512 else 10 if reach < 1024 else 11 if reach < 2048 else 12
    bits = format(256, "09b") + "".join(format(code, f"0{width(k)}b") for k, code in enumerate(codes, 1))
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def run_length_encode(data: bytes) -> bytes:
    out = bytearray()
    for start in range(0, len(data), 128):
        chunk = data[start:start + 128]
        out.append(len(chunk) - 1)
        out += chunk
    out.append(128)
    return bytes(out)


def text_content(lines: list[str]) -> bytes:
    rows = [b"BT /F1 11 Tf 14 TL 56 780 Td"]
    for line in lines:
        rows.append(b"(" + line.encode("latin-1") + b") Tj T*")
    rows.append(b"ET")
    return b"\n".join(rows) + b"\n"


def stream(dictionary: bytes, data: bytes) -> bytes:
    return b"<< " + dictionary + b" /Length " + str(len(data)).encode() + b" >>\nstream\n" + data + b"\nendstream"


def write_pdf(objects: list[bytes], *, xref_shift: int = 0, startxref: int | None = None) -> bytes:
    """Objects numbered 1..n; object 1 is the catalog. ``xref_shift`` moves every recorded offset; ``startxref``
    overrides the pointer to the cross-reference table."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    table = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % (offset - xref_shift)
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\nstartxref\n"
    out += str(table if startxref is None else startxref).encode() + b"\n%%EOF\n"
    return bytes(out)


def pages(contents: list[bytes], *, count: int | None = None) -> list[bytes]:
    """Catalog, page tree, one page per content stream (objects 3.., then their streams), the font last."""
    n = len(contents)
    page_ids = [3 + i for i in range(n)]
    content_ids = [3 + n + i for i in range(n)]
    font_id = 3 + 2 * n
    kids = b" ".join(b"%d 0 R" % i for i in page_ids)
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n if count is None else count).encode() + b" >>"]
    for content_id in content_ids:
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 "
                       + str(font_id).encode() + b" 0 R >> >> /Contents " + str(content_id).encode() + b" 0 R >>")
    objects += contents
    objects.append(FONT)
    return objects


LINES = [f"Line {i:02d}: a text layer kept behind an old content filter, page one." for i in range(40)]
SHORT = ["A short page of plain text for the cross-reference checks."]


def build() -> dict[str, bytes]:
    plain = text_content(LINES)
    short = text_content(SHORT)
    drawing = (b"0.2 0.4 0.6 rg 72 600 300 120 re f\n1 w 72 500 m 400 500 l S\n"
               b"q 40 0 0 40 72 400 cm BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\xff\xff\x00 EI Q\n")
    return {
        "lzw_content.pdf": write_pdf(pages([stream(b"/Filter /LZWDecode", lzw_encode(plain))])),
        "runlength_content.pdf": write_pdf(pages([stream(b"/Filter /RunLengthDecode", run_length_encode(plain))])),
        "rebuilt_xref.pdf": write_pdf(pages([stream(b"", short)]), xref_shift=7),
        "broken_startxref.pdf": write_pdf(pages([stream(b"", short)]), startxref=999999),
        "count_more.pdf": write_pdf(pages([stream(b"", short), stream(b"", text_content(["The second leaf."]))], count=3)),
        "count_less.pdf": write_pdf(pages([stream(b"", short), stream(b"", text_content(["The second leaf."]))], count=1)),
        "no_text_layer.pdf": write_pdf(pages([stream(b"", drawing), stream(b"", drawing)])),
        "whitespace_text.pdf": write_pdf(pages([stream(b"", b"BT /F1 11 Tf 56 780 Td (    ) Tj ET\n")])),
    }


def main(argv: list[str]) -> int:
    built = build()
    if "--check" in argv:
        stale = [name for name, data in built.items() if not (HERE / name).is_file() or (HERE / name).read_bytes() != data]
        print("stale:", stale) if stale else print("fixtures match")
        return 1 if stale else 0
    for name, data in built.items():
        (HERE / name).write_bytes(data)
    print("wrote", len(built), "fixtures,", sum(map(len, built.values())), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
