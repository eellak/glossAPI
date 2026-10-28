"""Tiny synthetic fonts: no system-font files or external resources are used."""
from pathlib import Path
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._c_m_a_p import CmapSubtable

root = Path(__file__).parent
for name, mappings, named in (
    ('alias_nameless', [{65: 'alpha', 945: 'alpha', 0x1f00: 'alpha'}], False),
    ('many_to_one', [{963: 'alpha'}, {963: 'beta'}], True),
    ('post_only', [{}], True),
):
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(['.notdef', 'alpha', 'beta', 'f_i'])
    fb.setupCharacterMap(mappings[0])
    glyphs = {}
    for n in fb.font.getGlyphOrder():
        pen = TTGlyphPen(None)
        pen.moveTo((0, 0)); pen.lineTo((400, 0)); pen.lineTo((400, 600))
        pen.lineTo((0, 600)); pen.closePath()
        glyphs[n] = pen.glyph()
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({n: (500, 0) for n in glyphs})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable(dict(familyName='Synthetic', styleName='Regular'))
    fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
    fb.setupPost(keepGlyphNames=named)
    tables = []
    for i, mapping in enumerate(mappings):
        table = CmapSubtable.newSubtable(4)
        table.platformID, table.platEncID, table.language = ((0, 3, 0) if i == 0 else (3, 1, 0))
        table.cmap = mapping
        tables.append(table)
    fb.font['cmap'].tables = tables
    fb.save(root / (name + '.ttf'))
