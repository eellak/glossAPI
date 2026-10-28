"""Reproduce static oracle tables from the installed fontTools/Unicode version.

These are standard lookup data, not corpus decisions or precomputed reports.
"""
import json
from pathlib import Path
import unicodedata
from fontTools import agl
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.cffLib import (cffStandardStrings, cffISOAdobeStrings,
                              cffIExpertStrings, cffExpertSubsetStrings)

Path(__file__).with_name('font_tables.json').write_text(json.dumps(dict(
    agl={k: ''.join(map(chr, v)) for k, v in agl.LEGACY_AGL2UV.items()},
    uv2agl=agl.UV2AGL, post=standardGlyphOrder, cff=cffStandardStrings,
    charsets=[cffISOAdobeStrings, cffIExpertStrings, cffExpertSubsetStrings],
    greek=[i for i in range(0x110000) if chr(i).isalpha()
           and 'GREEK' in unicodedata.name(chr(i), '')]), ensure_ascii=False))
