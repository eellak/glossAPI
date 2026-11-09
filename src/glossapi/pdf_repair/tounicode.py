"""Repair Unicode maps without changing glyphs, content, or font programs.

Document categories give no-text sampling precedence, then Greek repairability.
Otherwise a homogeneous set is consistent/not_embedded/unsupported; ``mixed``
means multiple remaining font categories (no_tounicode and unsupported_direct
count as unsupported). Sampling is diagnostic, not proof that every page is a
scan. Unknown mappings are preserved; nameless cmap aliases prefer a Greek-block
letter, then the lowest code point. Encrypted PDFs and non-incremental rewrites are
not supported.
"""

from __future__ import annotations

from contextlib import closing

from dataclasses import asdict, dataclass, field
from functools import lru_cache
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from typing import Any, Iterable, Iterator

from .protection import protect_sources

from ..optional_dependencies import optional_dependencies

with optional_dependencies("font-repair"):
    from fontTools import agl
    from fontTools.cffLib import CFFFontSet
    from fontTools.ttLib import TTFont
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
import pypdfium2 as pdfium
from ..pdfium_guard import pdfium_guard


from .reports import FontReport, PdfFontReport, RepairResult


def _is_greek(char: str) -> bool:
    """Count Greek letters, excluding punctuation in the Greek Unicode blocks."""
    return char.isalpha() and "GREEK" in unicodedata.name(char, "")


def _hex_bytes(token: bytes) -> bytes:
    """Decode a PDF hex string, including its specified odd-nibble padding."""
    value = re.sub(rb"\s", b"", token[1:-1])
    if len(value) % 2:
        value += b"0"
    return bytes.fromhex(value.decode("ascii"))


def _cmap_entries(data: bytes, unicode: bool = True) -> dict[int, str | int]:
    """Read bfchar/bfrange or cidchar/cidrange, rejecting ambiguous definitions."""
    data = re.sub(rb"%[^\r\n]*", b"", data)
    if b"usecmap" in data:
        raise ValueError("Inherited CMaps (usecmap) are unsupported")
    tokens = re.findall(rb"<[^<>]*>|\[|\]|[^\s<>\[\]]+", data)
    result: dict[int, str | int] = {}
    kinds = {b"beginbfchar", b"beginbfrange"} if unicode else {b"begincidchar", b"begincidrange"}

    def store(code: int, value: bytes | int) -> None:
        decoded = value.decode("utf-16-be") if isinstance(value, bytes) else value
        if code in result and result[code] != decoded:
            raise ValueError("Conflicting CMap definitions")
        result[code] = decoded

    i = 0
    while i < len(tokens):
        kind = tokens[i]
        if kind not in kinds:
            i += 1
            continue
        count = int(tokens[i - 1])
        if not 0 <= count <= 65536:
            raise ValueError("Excessive CMap block")
        i += 1
        for _ in range(count):
            start = int.from_bytes(_hex_bytes(tokens[i]), "big")
            i += 1
            if kind.endswith(b"char"):
                value = _hex_bytes(tokens[i]) if unicode else int(tokens[i])
                i += 1
                store(start, value)
            else:
                stop = int.from_bytes(_hex_bytes(tokens[i]), "big")
                i += 1
                if not 0 <= stop - start < 65536:
                    raise ValueError("Invalid/excessive CMap range")
                if tokens[i] == b"[":
                    i += 1
                    for code in range(start, stop + 1):
                        store(code, _hex_bytes(tokens[i]))
                        i += 1
                    if tokens[i] != b"]":
                        raise ValueError("Invalid CMap range array")
                    i += 1
                else:
                    base = _hex_bytes(tokens[i]) if unicode else int(tokens[i])
                    i += 1
                    for offset, code in enumerate(range(start, stop + 1)):
                        value = ((int.from_bytes(base, "big") + offset).to_bytes(len(base), "big")
                                 if isinstance(base, bytes) else base + offset)
                        store(code, value)
        if tokens[i] != kind.replace(b"begin", b"end", 1):
            raise ValueError("Invalid CMap block terminator")
        i += 1
    return result


def _iter_fonts(pdf: Any) -> Iterator[DictionaryObject]:
    """Visit each page/form resource font once, including shared resources."""
    seen_resources: set[int] = set()
    seen_fonts: set[int] = set()

    def visit(resources: Any) -> Iterator[DictionaryObject]:
        if resources is None:
            return
        resources = resources.get_object()
        if id(resources) in seen_resources:
            return
        seen_resources.add(id(resources))
        fonts = resources.get("/Font")
        for ref in fonts.get_object().values() if fonts else ():
            font = ref.get_object()
            if id(font) not in seen_fonts:
                seen_fonts.add(id(font))
                yield font
        xobjects = resources.get("/XObject")
        if xobjects:
            for ref in xobjects.get_object().values():
                obj = ref.get_object()
                if obj.get("/Subtype") == "/Form":
                    yield from visit(obj.get("/Resources"))

    for page in pdf.pages:
        yield from visit(page.get("/Resources"))


def _glyph_unicode(data: bytes, kind: str) -> tuple[dict[int, str], dict[str, int], dict[int, int], dict[int, int] | None]:
    """Invert cmap by GID; resolve nameless aliases by Greek block, then code point."""
    try:
        font = TTFont(BytesIO(data), lazy=False)
    except Exception:
        if kind != "FontFile3/CIDFontType0C":
            raise
        cff = CFFFontSet()
        cff.decompile(BytesIO(data), None)
        top = cff[cff.fontNames[0]]
        order = top.charset
        mapping = {gid: agl.toUnicode(name) for gid, name in enumerate(order) if agl.toUnicode(name)}
        cids = {int(name[3:]): gid for gid, name in enumerate(order) if name.startswith("cid") and name[3:].isdigit()}
        return mapping, {name: gid for gid, name in enumerate(order)}, {}, cids or None
    try:
        order = font.getGlyphOrder()
        names = {name: gid for gid, name in enumerate(order)}
        candidates: dict[int, set[str]] = {}
        symbol: dict[int, int] = {}
        if "cmap" in font:
            for table in font["cmap"].tables:
                # fontTools.isUnicode() also includes (3,0), whose PUA
                # positions are byte encodings, not a Unicode oracle.
                if (table.platformID, table.platEncID) == (3, 0):
                    for code, name in table.cmap.items():
                        if code < 256 or 0xF000 <= code <= 0xF0FF:
                            byte = code & 255
                            gid = names[name]
                            if byte in symbol and symbol[byte] != gid:
                                raise ValueError("Ambiguous symbol cmap")
                            symbol[byte] = gid
                elif table.isUnicode():
                    for code, name in table.cmap.items():
                        if code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF:
                            candidates.setdefault(names[name], set()).add(chr(code))
        # A format-3 post table has no names: fontTools may synthesize them
        # from cmap positions, which must not become an independent oracle.
        has_names = "CFF " in font or (
            "post" in font and font["post"].formatType in (1.0, 2.0, 2.5)
        )
        mapping = {}
        for gid, name in enumerate(order):
            values = candidates.get(gid, set())
            named = agl.toUnicode(name) if has_names else ""
            if len(values) == 1:
                mapping[gid] = next(iter(values))
            elif named and (not values or named in values):
                mapping[gid] = named
            elif values:
                # CP0 F4: names synthesized from post format 3 are not usable;
                # choose deterministically among the font's actual cmap values.
                mapping[gid] = min(values, key=lambda char: (
                    not (0x0370 <= ord(char) <= 0x03FF and char.isalpha()), ord(char)
                ))
        cids = None
        if "CFF " in font:
            top = font["CFF "].cff.topDictIndex[0]
            cids = {int(name[3:]): gid for gid, name in enumerate(top.charset)
                    if name.startswith("cid") and name[3:].isdigit()} or None
        return mapping, names, symbol, cids
    finally:
        font.close()


def _embedded_mapping(font: DictionaryObject) -> tuple[dict[int, str], str | None, bool, str | None]:
    """Translate source codes through encoding/CID/GID before assigning Unicode."""
    subtype = str(font.get("/Subtype", ""))
    descendant = font["/DescendantFonts"][0].get_object() if subtype == "/Type0" else font
    descriptor = descendant.get("/FontDescriptor")
    descriptor = descriptor.get_object() if descriptor else {}
    key = next((key for key in ("/FontFile2", "/FontFile3", "/FontFile") if key in descriptor), None)
    if key is None:
        return {}, None, False, "Font has no embedded program"
    program = descriptor[key]
    kind = key[1:] + ("/" + str(program.get("/Subtype", "")).lstrip("/") if key == "/FontFile3" else "")
    if subtype not in ("/Type0", "/TrueType") or key == "/FontFile":
        return {}, kind, True, "Unsupported font subtype/program"
    if key == "/FontFile3" and program.get("/Subtype") not in ("/OpenType", "/CIDFontType0C"):
        return {}, kind, True, "Unsupported FontFile3 subtype"
    glyphs, names, symbol, cff_cids = _glyph_unicode(program.get_data(), kind)
    if subtype == "/TrueType":
        codes_to_gid = dict(symbol)
        encoding = font.get("/Encoding")
        if encoding is not None and isinstance(encoding.get_object(), DictionaryObject):
            differences = encoding.get_object().get("/Differences", [])
            code = None
            for item in differences:
                if isinstance(item, int):
                    code = int(item)
                else:
                    if code is None or not 0 <= code <= 255:
                        raise ValueError("Invalid Differences mapping")
                    # Explicit Differences override the base/symbol mapping.
                    codes_to_gid.pop(code, None)
                    if str(item)[1:] in names:
                        codes_to_gid[code] = names[str(item)[1:]]
                    code += 1
        if not codes_to_gid:
            return {}, kind, True, "Simple font needs a symbol (3,0) cmap or Differences; no encoding guesses"
    else:
        if descendant.get("/Subtype") == "/CIDFontType2":
            cidmap = descendant.get("/CIDToGIDMap", "/Identity")
            if cidmap == "/Identity":
                cid_to_gid = {gid: gid for gid in glyphs}
            elif hasattr(cidmap.get_object(), "get_data"):
                data = cidmap.get_object().get_data()
                if len(data) % 2:
                    raise ValueError("Odd-length CIDToGIDMap")
                cid_to_gid = {cid: int.from_bytes(data[2 * cid:2 * cid + 2], "big")
                              for cid in range(len(data) // 2)}
            else:
                raise ValueError("Unsupported CIDToGIDMap")
        elif descendant.get("/Subtype") == "/CIDFontType0" and cff_cids:
            cid_to_gid = cff_cids
        else:
            return {}, kind, True, "CFF/OpenType lacks a usable CID mapping"
        encoding = font.get("/Encoding")
        if encoding in ("/Identity-H", "/Identity-V"):
            codes_to_gid = cid_to_gid
        elif encoding is not None and hasattr(encoding.get_object(), "get_data"):
            code_to_cid = _cmap_entries(encoding.get_object().get_data(), unicode=False)
            codes_to_gid = {code: cid_to_gid[cid] for code, cid in code_to_cid.items() if cid in cid_to_gid}
            # Only two-byte composite encodings are safely emitted by this pass.
            data = re.sub(rb"%[^\r\n]*", b"", encoding.get_object().get_data())
            blocks = re.findall(rb"(\d+)\s+begincodespacerange(.*?)endcodespacerange", data, re.DOTALL)
            if not blocks:
                return {}, kind, True, "Composite encoding must have a two-byte codespace"
            for count, block in blocks:
                spaces = re.findall(rb"<([0-9A-Fa-f\s]+)>\s*<([0-9A-Fa-f\s]+)>", block)
                if len(spaces) != int(count) or any(
                    len(re.sub(rb"\s", b"", a)) != 4 or len(re.sub(rb"\s", b"", b)) != 4
                    for a, b in spaces
                ):
                    return {}, kind, True, "Composite encoding must have a two-byte codespace"
        else:
            return {}, kind, True, "Unsupported predefined composite encoding"
    mapping = {code: glyphs[gid] for code, gid in codes_to_gid.items() if gid in glyphs and 0 <= code <= 65535}
    return mapping, kind, True, None if mapping else "Embedded font has no usable Unicode mapping"


def _inspect_font(font: DictionaryObject) -> tuple[FontReport, dict[int, str]]:
    """Record uncertain fonts and preserve their original map entries."""
    report = FontReport(
        font_name=str(font.get("/BaseFont", font.get("/Name", "unknown"))).lstrip("/"),
        subtype=str(font.get("/Subtype", "unknown")).lstrip("/"),
        has_tounicode="/ToUnicode" in font,
        object_id=getattr(getattr(font, "indirect_reference", None), "idnum", None),
    )
    try:
        descendant = font["/DescendantFonts"][0].get_object() if report.subtype == "Type0" else font
        descriptor = descendant.get("/FontDescriptor")
        if descriptor:
            descriptor = descriptor.get_object()
            for key in ("/FontFile2", "/FontFile3", "/FontFile"):
                if key in descriptor:
                    report.embedded = True
                    report.font_program_kind = key[1:]
                    if key == "/FontFile3":
                        report.font_program_kind += "/" + str(descriptor[key].get("/Subtype", "")).lstrip("/")
                    break
        old = _cmap_entries(font["/ToUnicode"].get_data()) if report.has_tounicode else {}
        report.codes = len(old)
        maximum = 65535 if report.subtype == "Type0" else 255
        if any(code > maximum for code in old):
            raise ValueError("ToUnicode source codes exceed the supported codespace")
        expected, report.font_program_kind, report.embedded, report.reason = _embedded_mapping(font)
        report.greek_codes = sum(any(_is_greek(c) for c in value) for value in expected.values())
        report.unmapped_codes = sum(code not in expected for code in old)
        bad = {code for code in old if code in expected and old[code] != expected[code]}
        report.disagreeing_codes = len(bad)
        report.affected_greek_codes = sum(any(_is_greek(c) for c in expected[code] + str(old[code])) for code in bad)
        if report.subtype == "Type3":
            report.category = "unsupported"
            report.reason = "Type3 glyph programs are unsupported"
        elif not report.embedded:
            report.category = "not_embedded"
        elif not expected:
            report.category = "unsupported" if report.has_tounicode else "no_tounicode"
        elif not report.has_tounicode:
            report.category = "no_tounicode"
            report.codes = len(expected)
            report.repair_possible = True
        elif bad or not old:
            report.category = "repairable"
            report.repair_possible = True
        elif old and report.unmapped_codes == len(old):
            report.category = "unsupported"
            report.reason = "No existing ToUnicode code has a usable embedded mapping"
        else:
            report.category = "consistent"
        if report.repair_possible and report.object_id is None:
            report.category = "unsupported_direct"
            report.repair_possible = False
            report.reason = "Direct font dictionaries cannot be updated in isolation"
        # Retain unknown codes, but replace every code for which the oracle exists.
        merged = dict(old)
        merged.update(expected)
        return report, merged
    except Exception as exc:
        report.category = "unsupported"
        report.error = f"{type(exc).__name__}: {exc}"
        report.reason = "Broken or unsupported font/CMap"
        return report, {}


def _page_text(pdf: Any, index: int) -> str:
    """Close page/text handles explicitly, including on extraction failure."""
    page = pdf[index]
    try:
        text = page.get_textpage()
        try:
            return text.get_text_range()
        finally:
            text.close()
    finally:
        page.close()


def _sample_text(path: str | Path) -> tuple[int, list[int], bool]:
    """Sample up to eight evenly spread pages and close all PDFium handles."""
    with pdfium_guard(), closing(pdfium.PdfDocument(str(path))) as pdf:
        count = len(pdf)
        indexes = sorted({round(i * (count - 1) / (min(count, 8) - 1)) for i in range(min(count, 8))}) if count > 1 else list(range(count))
        has_text = False
        for index in indexes:
            has_text |= bool(_page_text(pdf, index).strip())
        return count, [index + 1 for index in indexes], has_text


def _document_category(fonts: list[FontReport], has_text: bool) -> str:
    """Prioritize sampled scans and Greek repair; define mixed by remaining categories."""
    if not has_text:
        return "no_text_layer"
    if any(font.repair_possible and font.greek_codes for font in fonts):
        return "repairable"
    categories = {"unsupported" if font.category in ("no_tounicode", "repairable", "unsupported_direct") else font.category for font in fonts}
    return next(iter(categories)) if len(categories) == 1 else "mixed" if categories else "unsupported"


def _raise_os_error(exc: Exception) -> None:
    """An OS-level I/O failure (an OSError with an errno: ENOSPC, EDQUOT, EIO, ENOENT ...) is raised, as the native
    backend raises it (K4g fix 3): only document and font problems become an error-bearing report, so a full disk is
    never read as a broken PDF."""
    if isinstance(exc, OSError) and exc.errno is not None:
        raise exc


def analyze_pdf(path: str | Path) -> PdfFontReport:
    """Analyze without writing; bad PDFs return an error-bearing report."""
    report = PdfFontReport(str(path))
    try:
        reader = PdfReader(path, strict=False)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        report.pages, report.sampled_pages, has_text = _sample_text(path)
        report.fonts = [_inspect_font(font)[0] for font in _iter_fonts(reader)]
        report.category = _document_category(report.fonts, has_text)
    except Exception as exc:
        _raise_os_error(exc)
        report.error = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).debug("PDF analysis failed for %s", path, exc_info=True)
    return report


def _make_cmap(mapping: dict[int, str], width: int = 2) -> bytes:
    """Emit at most 100 bfchar entries per block with UTF-16 surrogate pairs."""
    maximum = (1 << (8 * width)) - 1
    lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap",
             "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def",
             "/CMapName /GlossAPI-UCS def", "/CMapType 2 def", "1 begincodespacerange",
             f"<{'00' * width}> <{'FF' * width}>", "endcodespacerange"]
    if any(not 0 <= code <= maximum for code in mapping):
        raise ValueError("CMap source code does not fit the output codespace")
    entries = sorted(mapping.items())
    for offset in range(0, len(entries), 100):
        block = entries[offset:offset + 100]
        lines.append(f"{len(block)} beginbfchar")
        lines.extend(f"<{code:0{width * 2}X}> <{value.encode('utf-16-be').hex().upper()}>" for code, value in block)
        lines.append("endbfchar")
    lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
    return ("\n".join(lines) + "\n").encode("ascii")


def repair_pdf(src: str | Path, dst: str | Path, *,
               protected_sources: Iterable[str | Path] = ()) -> RepairResult:
    """Append only ToUnicode streams/font dictionary updates; never overwrite src.

    Reject any unrelated mutation detected by the incremental writer. Publish
    atomically, leaving an existing destination intact on failure.
    """
    result = RepairResult(str(src), str(dst))
    temporary: str | None = None
    try:
        source, target = Path(src), Path(dst)
        sources = (source, *protected_sources)
        protect_sources(target, sources)
        writer = PdfWriter(source, incremental=True)
        if writer._reader.is_encrypted:
            raise ValueError("Encrypted PDFs are unsupported")
        _, _, has_text = _sample_text(source)
        candidates = [(font, *_inspect_font(font)) for font in _iter_fonts(writer)]
        result.fonts = [report for _, report, _ in candidates]
        result.category = _document_category(result.fonts, has_text)
        allowed: set[int] = set()
        for font, report, mapping in candidates:
            if not report.repair_possible:
                continue
            stream = DecodedStreamObject()
            stream.set_data(_make_cmap(mapping, 2 if report.subtype == "Type0" else 1))
            ref = writer._add_object(stream)
            font[NameObject("/ToUnicode")] = ref
            allowed.update((font.indirect_reference.idnum, ref.idnum))
            result.repaired_fonts += 1
        changed = {ref.idnum for ref in writer.list_objects_in_increment()}
        if changed - allowed:
            raise ValueError(f"Refusing unrelated incremental changes: {sorted(changed - allowed)}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".font-repair-", suffix=".pdf", dir=target.parent)
        os.close(fd)
        protect_sources(temporary, sources)
        if result.repaired_fonts:
            writer.write(temporary)
        else:
            shutil.copyfile(source, temporary)
        protect_sources(target, sources)
        os.replace(temporary, target)
        temporary = None
        result.written = True
    except Exception as exc:
        _raise_os_error(exc)
        result.error = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).debug("PDF repair failed for %s", src, exc_info=True)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return result


from .text_stats import extract_text_stats, _load_words, _text_stats
