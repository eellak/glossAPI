from pathlib import Path
import json, sys
root = Path.cwd()
sys.path[:0] = [str(root/'rust/glossapi_rs_pdfrepair/target/python'), str(root/'src')]
from glossapi.pdf_repair import rust_backend
from glossapi.pdf_repair.tounicode import repair_pdf, _page_text
import pypdfium2 as pdfium
rows = []
for name, old, new in [('same_raw_byte', b'/X#F1Y', b'/X#E9Y'), ('same_utf8_bytes', b'/X#C3#B1Y', b'/X#C3#A9Y')]:
    fixture = 'name_decoding_different_raw_byte' if name == 'same_raw_byte' else 'name_decoding_valid_utf8'
    data = (root/'rust/glossapi_rs_pdfrepair/fixtures/review'/f'{fixture}.pdf').read_bytes()
    assert data.count(old) == 1 and len(old) == len(new)
    src = Path('/tmp/t11-cycle3')/f'{name}.pdf'
    src.write_bytes(data.replace(old, new))
    row = {'case': name}
    for label, fn in [('python', repair_pdf), ('rust', rust_backend.repair_pdf)]:
        dst = src.with_name(f'{name}-{label}.pdf')
        result = fn(src, dst)
        if hasattr(result, 'to_dict'): result = result.to_dict()
        row[label] = {'repaired_fonts': result['repaired_fonts'], 'error': result['error']}
        with pdfium.PdfDocument(str(dst)) as pdf:
            row[label]['text'] = _page_text(pdf, 0)
    rows.append(row)
Path('/tmp/t11-cycle3/codec-boundary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(rows, ensure_ascii=False, indent=2))
