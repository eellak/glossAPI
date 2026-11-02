"""HISTORICAL PROBE - provenance copy only; not runnable from the repository and run by no test (T1-G10).

Verbatim copy (below this docstring) of the CP-R1c2 review reproducer
`glossapi_improvement_20261006/review_scratch/CP-R1c2/name_decoding_probe.py`, original sha256
1a2a6cd5c9a8c085257b0c3fa19016daba388baeda3cefb79f312f39b3a9a4da. It needs that review scratch: `adversarial.py`
(sha256 fd5bca4155b455a1e51e5bea2b9e69213bfeca91823b507f6bde8821cb1663ed) and `independent_invariants.py`
(sha256 66d414af79f5c5ae7d4cb40f1066da5e5f054d60446e3f7dabbea84f657cffd0, a review-only helper, not a package), a
`/tmp/CP-R1c2-native` build and the `/tmp/CP-R1c2-glyph-probe` binary. The self-contained replacement is
`generate_name_decoding_fixtures.py`; its outputs are committed under `fixtures/review/name_decoding_*`.
"""
from pathlib import Path
import sys,json,re,subprocess
SCRATCH=Path('/Users/foivoskarounos-zamparloukos/Projects/glossapi-ocr-local/experiments/glossapi_improvement_20261006/review_scratch/CP-R1c2')
ROOT=Path('/Users/foivoskarounos-zamparloukos/Projects/glossapi-development')
sys.path[:0]=['/tmp/CP-R1c2-native',str(ROOT/'src')]
ns={};exec((SCRATCH/'adversarial.py').read_text().split('cases=[]')[0],ns)
out=Path('/tmp/CP-R1c2-name-decoding');out.mkdir(exist_ok=True);ns['OUT']=out
from pypdf import PdfReader
from glossapi.pdf_repair.tounicode import _glyph_unicode
from independent_invariants import raw_dict,check
b=ns['font_bytes'](['.notdef','first'],[(0,3,{945:'first'})],postmap={'first':'XéY'})
(out/'nonascii_post_name.ttf').write_bytes(b)
rsnames=json.loads(subprocess.check_output(['/tmp/CP-R1c2-glyph-probe',str(out/'nonascii_post_name.ttf')]))['names']
records=[]
for case,escaped in [('same_raw_byte',b'/X#E9Y'),('different_raw_byte',b'/X#F1Y')]:
 src=ns['pdf'](case,b,'XéY');r=PdfReader(src);font=r.pages[0]['/Resources']['/Font']['/F1'];id=font.indirect_reference.idnum
 raw=raw_dict(r,id,0);assert b'/X#C3#A9Y' in raw,raw
 raw=raw.replace(b'/X#C3#A9Y',escaped)
 original=src.read_bytes();prev=int(re.findall(rb'startxref\s+(\d+)',original)[-1]);root=r.trailer.raw_get('/Root');size=int(r.trailer['/Size']);data=bytearray(original+b'\n');off=len(data)
 data.extend(f'{id} 0 obj\n'.encode()+raw+b'\nendobj\n');xref=len(data)
 data.extend(f'xref\n0 1\n0000000000 65535 f \n{id} 1\n{off:010d} 00000 n \ntrailer\n<< /Size {size} /Root {root.idnum} {root.generation} R /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n'.encode());src.write_bytes(data)
 original=src.read_bytes()
 rows={'case':case,'source':str(src),'post_name_bytes_hex':'58e959','differences_token':escaped.decode(),'python_post_names':_glyph_unicode(b,'FontFile2')[1],'rust_post_names':rsnames,'python_differences':str(PdfReader(src).pages[0]['/Resources']['/Font']['/F1']['/Encoding']['/Differences'][1])}
 rows['python_analysis']=ns['analyze_pdf'](src).to_dict();rows['rust_analysis']=ns['rust_backend'].analyze_pdf(src)
 for label,fun in [('python',ns['repair_pdf']),('rust',ns['rust_backend'].repair_pdf)]:
  dst=out/f'{case}-{label}.pdf';res=fun(src,dst);res=res.to_dict() if hasattr(res,'to_dict') else res;rows[label+'_repair']=res
  assert res['written'] and not res['error']
  rows[label+'_text']=ns['text'](dst)
  if label=='rust':rows['rust_invariants']=check(src,dst,{f['object_id'] for f in res['fonts'] if f['repair_possible']})
 assert rows['python_repair']['repaired_fonts']==0 and rows['python_text']=='B'
 assert rows['rust_repair']['repaired_fonts']==1 and rows['rust_text']=='α'
 assert src.read_bytes()==original
 records.append(rows)
(SCRATCH/'name-decoding-probe.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
for row in records:
 print(json.dumps({k:v for k,v in row.items() if k not in ['python_analysis','rust_analysis','python_repair','rust_repair']},ensure_ascii=False,indent=2))
print('Both false-repair cases independently reproduced: Python 0 repairs / B; Rust 1 repair / α, null errors, original dictionary bytes retained.')
