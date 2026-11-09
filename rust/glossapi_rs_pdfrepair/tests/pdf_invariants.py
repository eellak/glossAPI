"""PDF regression helpers ported from 53d63c8's experiment equivalence harness.

Only test assertions live here; campaign ledgers remain on the source ref.
"""

from contextlib import closing
import pypdfium2 as pdfium
from pypdf import PdfReader
from pypdf.generic import DictionaryObject, IndirectObject


def extracted(path):
    texts = []
    with closing(pdfium.PdfDocument(str(path))) as pdf:
        for i in range(min(20, len(pdf))):
            page = pdf[i]
            try:
                text = page.get_textpage()
                try:
                    texts.append(text.get_text_range())
                finally:
                    text.close()
            finally:
                page.close()
    return texts


def _skip_pdf_space(stream):
    """Skip PDF whitespace/comments without normalizing the original bytes."""
    while True:
        char = stream.read(1)
        if not char:
            raise ValueError('Unexpected end of raw font dictionary')
        if char in b'\x00\t\n\f\r ':
            continue
        if char == b'%':
            while char and char not in (b'\r', b'\n'):
                char = stream.read(1)
            continue
        stream.seek(-1, 1)
        return


def raw_font_dictionary(reader, n, g=0):
    """Read exact dictionary tokens from an indirect object or decoded ObjStm."""
    from io import BytesIO
    from pypdf.generic import read_object
    if n in reader.xref.get(g, {}):
        stream = reader.stream
        stream.seek(reader.xref[g][n])
        object_id, generation = reader.read_object_header(stream)
        if (object_id, generation) != (n, g):
            raise ValueError('Raw font object header mismatch')
    else:
        stream_id, index = reader.xref_objStm[n]
        obj = reader.get_object(stream_id)
        data = obj.get_data()
        first = int(obj['/First'])
        header = list(map(int, data[:first].split()))
        if header[2 * index] != n:
            raise ValueError('Raw font ObjStm index mismatch')
        stream = BytesIO(data)
        stream.seek(first + header[2 * index + 1])
    _skip_pdf_space(stream)
    start = stream.tell()
    obj = read_object(stream, reader)
    end = stream.tell()
    if not isinstance(obj, DictionaryObject) or hasattr(obj, 'get_data'):
        raise ValueError('Updated font object is not a dictionary')
    stream.seek(start)
    return stream.read(end - start)


def raw_without_tounicode(raw, reader):
    """Remove only top-level entries, retaining all other original byte spans."""
    from io import BytesIO
    from pypdf.generic import read_object
    if not raw.startswith(b'<<'):
        raise ValueError('Expected raw font dictionary')
    stream = BytesIO(raw)
    stream.seek(2)
    out = bytearray(raw[:2])
    while True:
        start = stream.tell()
        _skip_pdf_space(stream)
        if stream.read(2) == b'>>':
            out.extend(raw[start:])
            return bytes(out)
        stream.seek(-2, 1)
        key = read_object(stream, reader)
        _skip_pdf_space(stream)
        read_object(stream, reader)
        if key != '/ToUnicode':
            out.extend(raw[start:stream.tell()])


def object_invariants(src, dst, repaired_ids):
    """Only the eligible font dictionaries may differ, only at ToUnicode.

    Compare all original indirect objects, including compressed ones, and ensure
    the only new objects are ToUnicode streams (classic xref adds no object).
    """
    from io import BytesIO
    before, after = PdfReader(src), PdfReader(dst)
    ids = {(n, g) for g, entries in before.xref.items() for n in entries if n}
    ids.update((n, 0) for n in before.xref_objStm)
    def serialized(o):
        b = BytesIO()
        o.write_to_stream(b)
        return b.getvalue()
    for n, g in sorted(ids):
        a = before.get_object(IndirectObject(n, g, before))
        b = after.get_object(IndirectObject(n, g, after))
        if n in repaired_ids:
            a = DictionaryObject({k: v for k, v in a.items() if k != '/ToUnicode'})
            b = DictionaryObject({k: v for k, v in b.items() if k != '/ToUnicode'})
        if serialized(a) != serialized(b):
            return f'original object {n} {g} changed'
        if n in repaired_ids:
            raw_a = raw_without_tounicode(raw_font_dictionary(before, n, g), before)
            raw_b = raw_without_tounicode(raw_font_dictionary(after, n, g), after)
            if raw_a != raw_b:
                return f'original font dictionary {n} {g} bytes changed outside ToUnicode'
    new = {(n, g) for g, entries in after.xref.items() for n in entries if n} - ids
    pointers = {after.get_object(n).raw_get('/ToUnicode').idnum for n in repaired_ids}
    if {n for n, _ in new} != pointers:
        return f'new object IDs {sorted(new)} differ from ToUnicode IDs {sorted(pointers)}'
    return None
