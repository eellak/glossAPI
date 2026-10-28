"""CP-R1 HIGH reproductions copied from the read-only review's adversarial.py.

Run with .venv-exp; expectations come from the unchanged Python reference.
No external font binaries or native-backend expectations are used.
"""
from io import BytesIO
from pathlib import Path
import json, struct, sys
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools import agl
from pypdf.generic import ArrayObject as A, DictionaryObject as D, NameObject as N, NumberObject as I, DecodedStreamObject as S
from pypdf import PdfWriter
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from glossapi.pdf_repair.tounicode import _glyph_unicode, _inspect_font, _iter_fonts
from pypdf import PdfReader
OUT = Path(__file__).parent / 'review'
OUT.mkdir(exist_ok=True)
def font_bytes(order, maps, named=True, cff=False, postmap=None, postformat=None):
    fb = FontBuilder(1000, isTTF=not cff)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(maps[0][2])
    if cff:
        charstrings = {}
        for n in order:
            pen = T2CharStringPen(500, None)
            pen.moveTo((0,0)); pen.lineTo((400,0)); pen.lineTo((400,600)); pen.closePath()
            charstrings[n] = pen.getCharString()
        fb.setupCFF('Synthetic', {'FullName':'Synthetic','FamilyName':'Synthetic','Weight':'Regular'}, charstrings, {})
    else:
        glyphs = {}
        for n in order:
            pen = TTGlyphPen(None)
            pen.moveTo((0,0)); pen.lineTo((400,0)); pen.lineTo((400,600)); pen.closePath()
            glyphs[n] = pen.glyph()
        fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({n:(500,0) for n in order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable(dict(familyName='Synthetic',styleName='Regular'))
    fb.setupOS2(sTypoAscender=800,sTypoDescender=-200,usWinAscent=800,usWinDescent=200)
    fb.setupPost(keepGlyphNames=named)
    tables=[]
    for platform, encoding, mapping in maps:
        t=CmapSubtable.newSubtable(4)
        t.platformID,t.platEncID,t.language=platform,encoding,0
        t.cmap=mapping
        tables.append(t)
    fb.font['cmap'].tables=tables
    fb.font['maxp'].numGlyphs=len(order)
    if postmap:
        fb.font['post'].mapping=postmap
    if postformat:
        table=DefaultTable('post')
        table.data=struct.pack('>I',postformat)+fb.font['post'].compile(fb.font)[4:32]
        # Historical 2.5 is unsupported by the installed fontTools reference.
        table.data += (struct.pack('>'+'H'*len(order), *([0xFFFF]+[945]*(len(order)-1)))
                       if postformat == 0x40000 else bytes([0]*len(order)))
        fb.font['post']=table
    b=BytesIO(); fb.save(b)
    return b.getvalue()

def cmap(mapping, width=1):
    rows='\n'.join(f'<{c:0{width*2}X}> <{s.encode("utf-16-be").hex()}>' for c,s in mapping.items())
    return ('/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n'
            '/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n'
            '/CMapName /Fixture def\n/CMapType 2 def\n'
            f'1 begincodespacerange <{"00"*width}> <{"FF"*width}> endcodespacerange\n'
            f'{len(mapping)} beginbfchar\n{rows}\nendbfchar\n'
            'endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n').encode()

def stream(w, data, extra=None):
    o=S(); o.set_data(data)
    for k,v in (extra or {}).items(): o[N(k)]=v
    return w._add_object(o)

def pdf(name, data, differences=None, composite=False, opentype=False, custom=False):
    w=PdfWriter(); page=w.add_blank_page(200,200)
    program=stream(w,data,{'/Subtype':N('/OpenType')} if opentype else {})
    desc=D({N('/Type'):N('/FontDescriptor'),N('/FontName'):N('/Synthetic'),N('/Flags'):I(32 if differences is not None else 4),
            N('/FontBBox'):A([I(x) for x in (0,0,1000,1000)]),N('/Ascent'):I(800),N('/Descent'):I(-200),
            N('/CapHeight'):I(800),N('/StemV'):I(80),N('/ItalicAngle'):I(0),
            N('/FontFile3' if opentype else '/FontFile2'):program})
    if composite:
        child=D({N('/Type'):N('/Font'),N('/Subtype'):N('/CIDFontType2'),N('/BaseFont'):N('/Synthetic'),
                 N('/FontDescriptor'):w._add_object(desc),N('/CIDToGIDMap'):N('/Identity'),
                 N('/CIDSystemInfo'):D({N('/Registry'):N('/Adobe'),N('/Ordering'):N('/Identity'),N('/Supplement'):I(0)})})
        f=D({N('/Type'):N('/Font'),N('/Subtype'):N('/Type0'),N('/BaseFont'):N('/Synthetic'),
             N('/DescendantFonts'):A([w._add_object(child)]),N('/Encoding'):N('/Identity-H')})
        f[N('/ToUnicode')]=stream(w,cmap({1:'B',2:'C'},2))
        content=b'BT /F1 20 Tf 20 100 Td <00010002> Tj ET'
    else:
        f=D({N('/Type'):N('/Font'),N('/Subtype'):N('/TrueType'),N('/BaseFont'):N('/Synthetic'),
             N('/FontDescriptor'):w._add_object(desc),N('/FirstChar'):I(65),N('/LastChar'):I(65),N('/Widths'):A([I(500)])})
        if differences is not None:
            f[N('/Encoding')]=D({N('/Differences'):A([I(65),N('/'+differences)])})
        f[N('/ToUnicode')]=stream(w,cmap({65:'B'}))
        content=b'BT /F1 20 Tf 20 100 Td (A) Tj ET'
    if custom: f[N('/CustomName')]=N('/XaY')
    page[N('/Resources')]=D({N('/Font'):D({N('/F1'):w._add_object(f)})})
    page[N('/Contents')]=stream(w,content)
    src=OUT/(name+'.pdf'); w.write(src)
    if custom:
        # A valid escaped non-UTF8 name, same length to preserve all xref offsets.
        raw=src.read_bytes()
        assert raw.count(b'/XaY')==1
        src.write_bytes(raw.replace(b'/XaY', b'/X#E9Y'))
        # Rebuild via raw append xref offsets below because replacement adds 2 bytes.
        raw=src.read_bytes()
        start=raw.rfind(b'\nxref\n')+1
        oldstart=int(raw[raw.rfind(b'startxref')+9:].split()[0])
        pivot=raw.find(b'/X#E9Y')
        section=raw[start:raw.index(b'trailer',start)]
        lines=section.splitlines(keepends=True)
        for i,line in enumerate(lines):
            if line.endswith(b'n \n'):
                off=int(line[:10])
                if off>pivot: lines[i]=f'{off+2:010d}'.encode()+line[10:]
        raw=raw[:start]+b''.join(lines)+raw[raw.index(b'trailer',start):]
        pos=raw.rfind(b'startxref')+9
        tail=raw[pos:]; off=tail.split()[0]
        raw=raw[:pos]+tail.replace(off,str(oldstart+2).encode(),1)
        src.write_bytes(raw)
    return src
cases = [
    ('lowercase_agl', font_bytes(['.notdef','uni03b1'],[(0,3,{})]), 'uni03b1', False),
    ('lowercase_u_agl', font_bytes(['.notdef','u03b1'],[(0,3,{})]), 'u03b1', False),
    ('duplicate_post_names', font_bytes(['.notdef','first','second'],[(0,3,{946:'first',947:'second'})],postmap={'first':'alpha','second':'alpha'}), 'alpha', False),
    ('format3_duplicate_synth_names', font_bytes(['.notdef','one','two'],[(0,3,{963:'one'}),(3,1,{963:'two'})],named=False), 'sigma.alt1', False),
    ('format3_collision_wrong_gid', font_bytes(['.notdef','one','two'],[(0,3,{65:'one',945:'one'}),(3,1,{65:'two',946:'two'})],named=False), 'A', False),
    ('encoded_pdf_name', font_bytes(['.notdef','alpha'],[(0,3,{945:'alpha'})]), 'alpha', True),
]
expected = {}
for name, data, differences, custom in cases:
    (OUT / (name + '.ttf')).write_bytes(data)
    src = pdf(name, data, differences, custom=custom)
    report, mapping = _inspect_font(next(_iter_fonts(PdfReader(src))))
    glyphs, names, _, _ = _glyph_unicode(data, 'FontFile2')
    expected[name] = dict(glyphs=glyphs, names=names, mapping=mapping, repair_possible=report.repair_possible, category=report.category)
(OUT / 'expected.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2) + '\n')
# Full LEGACY_AGL2UV name set, case variants and lexical/component/suffix edges.
names = set(agl.LEGACY_AGL2UV)
names.update(n.swapcase() for n in agl.LEGACY_AGL2UV)
names.update(['uni03b1', 'u03b1', 'uni03B1', 'u03B1', 'uni03B103B2', 'uniD800', 'uniD800DC00', 'uDFFF', 'u10FFFF', 'u110000', 'u000000', 'u0000', 'u000', 'u0000000', 'uni', 'u', 'Uni03B1', 'U03B1', 'uni03B1\n', 'u03B1\n', 'uni03B1\n\n', 'alpha_unknown_beta', 'uni03b1_beta', '_alpha__beta_', 'alpha.suffix_beta', '.alpha', 'f_i', 'a1'])
(OUT / 'agl_expected.json').write_text(json.dumps({n: agl.toUnicode(n) for n in sorted(names)}, ensure_ascii=False) + '\n')
print(f'Generated {len(cases)} CP-R1 PDF/font cases and {len(names)} AGL expectations')

# Exercise the same escaped-name reproduction in an ObjStm. Retain the source
# revision and append a compressed authoritative copy plus an xref stream.
source = OUT / 'encoded_pdf_name.pdf'
reader = PdfReader(source)
font = next(_iter_fonts(reader))
id = font.indirect_reference.idnum
stream = reader.stream
stream.seek(reader.xref[0][id])
reader.read_object_header(stream)
from pypdf.generic import read_object
while stream.read(1) in b'\x00\t\n\f\r ':
    pass
stream.seek(-1, 1)
start = stream.tell()
read_object(stream, reader)
end = stream.tell()
stream.seek(start)
raw_font = stream.read(end - start)
original = source.read_bytes()
prev = int(original[original.rfind(b'startxref') + 9:].split()[0])
root_ref = reader.trailer.raw_get('/Root')
root_entry = f'{root_ref.idnum} {root_ref.generation} R'
stream_id = int(reader.trailer['/Size'])
header = f'{id} 0 '.encode()
content = header + raw_font
raw = bytearray(original + b'\n')
stream_offset = len(raw)
raw.extend(f'{stream_id} 0 obj\n<< /Type /ObjStm /N 1 /First {len(header)} /Length {len(content)} >>\nstream\n'.encode() + content + b'\nendstream\nendobj\n')
xref_offset = len(raw)
xref_id = stream_id + 1
entries = struct.pack('>BIH', 2, stream_id, 0) + struct.pack('>BIH', 1, stream_offset, 0) + struct.pack('>BIH', 1, xref_offset, 0)
raw.extend(f'{xref_id} 0 obj\n<< /Type /XRef /Size {xref_id + 1} /Root {root_entry} /Prev {prev} /W [1 4 2] /Index [{id} 1 {stream_id} 2] /Length {len(entries)} >>\nstream\n'.encode() + entries + b'\nendstream\nendobj\n')
raw.extend(f'startxref\n{xref_offset}\n%%EOF\n'.encode())
(OUT / 'encoded_pdf_name_compressed.pdf').write_bytes(raw)

# Missing ToUnicode and varied byte encodings/comments in unrelated values.
import re
raw_font = re.sub(rb'/ToUnicode \d+ 0 R\s*', b'', raw_font)
raw_font = raw_font[:-2] + b'/N#65sted << /ToUnicode (leave\\040this) /N /X#E9Y >>\n/Values [+005.00 -000 (a\\n(b)) <Ab Cd> [null true /X#e9Y]]\n% keep this comment\n>>'
raw = bytearray(original + b'\n')
offset = len(raw)
raw.extend(f'{id} 0 obj\n'.encode() + raw_font + b'\nendobj\n')
xref = len(raw)
raw.extend(f'xref\n0 1\n0000000000 65535 f \n{id} 1\n{offset:010d} 00000 n \ntrailer\n<< /Size {stream_id} /Root {root_entry} /Prev {prev} >>\nstartxref\n{xref}\n%%EOF\n'.encode())
(OUT / 'encoded_pdf_name_missing_tounicode.pdf').write_bytes(raw)
print('Generated compressed and missing-ToUnicode byte-preservation variants')
