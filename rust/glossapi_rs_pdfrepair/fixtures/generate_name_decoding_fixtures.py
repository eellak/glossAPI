"""Adapt CP-R1c2/name_decoding_probe.py into self-contained crate fixtures.

The self-contained builders below preserve the historical reproducer's synthetic cases.
Only crate fixture files are written; expectations use the unchanged reference.
"""
from pathlib import Path
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pypdf import PdfReader
from pypdf.generic import read_object
from glossapi.pdf_repair.tounicode import analyze_pdf, repair_pdf
import pypdfium2 as pdfium

# Reuse the copied cycle-1 review's synthetic font/PDF builders without running
# its fixture-generation body.
ns = {'__file__': str(HERE / 'generate_review_fixtures.py')}
exec((HERE / 'generate_review_fixtures.py').read_text().split('cases = [')[0], ns)
expected = {}
for case, post_name, escaped in [
    ('name_decoding_different_raw_byte', 'XéY', b'/X#F1Y'),
    # Both names are valid UTF-8, but their bytes differ. fontTools reads post
    # custom names as Latin-1; the PDF name is UTF-8. Neither may be normalized.
    ('name_decoding_valid_utf8', 'XÃ©Y', b'/X#C3#B1Y'),
]:
    font = ns['font_bytes'](['.notdef', 'first'], [(0, 3, {945: 'first'})],
                            postmap={'first': post_name})
    (ns['OUT'] / (case + '.ttf')).write_bytes(font)
    source = ns['pdf'](case, font, post_name)
    reader = PdfReader(source)
    obj = reader.pages[0]['/Resources']['/Font']['/F1']
    ident = obj.indirect_reference.idnum
    stream = reader.stream
    stream.seek(reader.xref[0][ident])
    reader.read_object_header(stream)
    while stream.read(1) in b'\x00\t\n\f\r ':
        pass
    stream.seek(-1, 1)
    start = stream.tell()
    read_object(stream, reader)
    end = stream.tell()
    stream.seek(start)
    raw = stream.read(end - start)
    token = b'/' + b''.join(bytes([c]) if 33 <= c < 127 else f'#{c:02X}'.encode()
                           for c in post_name.encode('utf-8'))
    assert token in raw
    raw = raw.replace(token, escaped)
    original = source.read_bytes()
    prev = int(re.findall(rb'startxref\s+(\d+)', original)[-1])
    root = reader.trailer.raw_get('/Root')
    size = int(reader.trailer['/Size'])
    data = bytearray(original + b'\n')
    offset = len(data)
    data.extend(f'{ident} 0 obj\n'.encode() + raw + b'\nendobj\n')
    xref = len(data)
    data.extend(f'xref\n0 1\n0000000000 65535 f \n{ident} 1\n{offset:010d} 00000 n \ntrailer\n<< /Size {size} /Root {root.idnum} {root.generation} R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    source.write_bytes(data)
    target = Path('/tmp/t11-cycle3') / (case + '-reference.pdf')
    target.parent.mkdir(exist_ok=True)
    result = repair_pdf(source, target).to_dict()
    assert result['written'] and result['error'] is None
    assert result['repaired_fonts'] == 0
    assert target.read_bytes() == source.read_bytes()
    with pdfium.PdfDocument(str(target)) as doc:
        page = doc[0]
        text = page.get_textpage()
        value = text.get_text_range()
        text.close()
        page.close()
    assert value == 'B'
    expected[case] = dict(post_hex=post_name.encode('latin1').hex(),
                          differences_token=escaped.decode(), text=value,
                          analysis=analyze_pdf(source).to_dict(), repair=result)
(ns['OUT'] / 'name_decoding_expected.json').write_text(
    json.dumps(expected, ensure_ascii=False, indent=2) + '\n')
print('Generated 2 CP-R1c2 name-decoding cases: reference 0 repairs, text B')
