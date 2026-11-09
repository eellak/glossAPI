"""Test PDF fixtures built with PDFium (pypdfium2), replacing the former PyMuPDF (fitz) builders.

Only what the fixtures need: pages, text (standard Helvetica or an embedded TrueType font, written as
PDFium's Type0/Identity-H font whose codes are glyph IDs, as PyMuPDF's ``insert_font`` wrote), solid or
PIL raster images, page import, rotation and crop boxes. Coordinates follow the former PyMuPDF calls:
points from the top-left corner, text at its baseline, image rectangles as ``(x0, y0, x1, y1)``.
Multi-line text uses PyMuPDF's line advance: font size times (ascender - descender), i.e. 1.374 for
Helvetica and the hhea metrics for TrueType files. Equivalence with the PyMuPDF-built fixtures (text,
page geometry, image bounds, pixels and page renders) is recorded in the K2 batch 2a notes.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import struct

import pypdfium2 as pdfium
import pypdfium2.raw as raw

HELVETICA_LINE = 1.374  # PyMuPDF base-14 "helv": ascender 1.075, descender -0.299


def _truetype_line_factor(data: bytes) -> float:
    """(hhea ascender - descender) / unitsPerEm, read from the sfnt table directory."""
    tables = {}
    count = struct.unpack('>H', data[4:6])[0]
    for index in range(count):
        tag, _checksum, offset, length = struct.unpack('>4sIII', data[12 + 16 * index:28 + 16 * index])
        tables[tag] = data[offset:offset + length]
    units = struct.unpack('>H', tables[b'head'][18:20])[0]
    ascender, descender = struct.unpack('>hh', tables[b'hhea'][4:8])
    return (ascender - descender) / units


class FixturePdf:
    """A new PDF document; ``save`` writes it and releases every native handle."""

    def __init__(self):
        self.pdf = pdfium.PdfDocument.new()
        self._fonts = []
        self._buffers = []
        self._line = {}
        self._pages = []

    def new_page(self, width: float = 595, height: float = 842):
        page = self.pdf.new_page(width, height)
        self._pages.append(page)
        return page

    def load_truetype(self, path: str | Path):
        """Embed a TrueType file as a Type0/Identity-H (CID = glyph ID) font with a ToUnicode map."""
        data = Path(path).read_bytes()
        buffer = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        font = raw.FPDFText_LoadFont(self.pdf.raw, buffer, len(data), raw.FPDF_FONT_TRUETYPE, True)
        if not font:
            raise ValueError(f'PDFium could not load the TrueType font {path}')
        self._fonts.append(font)
        self._buffers.append(buffer)
        self._line[ctypes.addressof(font.contents)] = _truetype_line_factor(data)
        return font

    def insert_text(self, page, point, text: str, *, fontsize: float = 11, font=None) -> None:
        """Insert text with its first baseline at ``point``; '\\n' starts a new line."""
        factor = HELVETICA_LINE if font is None else self._line[ctypes.addressof(font.contents)]
        x, y = point
        height = page.get_height()
        for number, line in enumerate(text.split('\n')):
            if font is None:
                obj = raw.FPDFPageObj_NewTextObj(self.pdf.raw, b'Helvetica', ctypes.c_float(fontsize))
            else:
                obj = raw.FPDFPageObj_CreateTextObj(self.pdf.raw, font, ctypes.c_float(fontsize))
            encoded = ctypes.create_string_buffer((line + '\0').encode('utf-16-le'))
            if not raw.FPDFText_SetText(obj, ctypes.cast(encoded, ctypes.POINTER(raw.FPDF_WCHAR))):
                raw.FPDFPageObj_Destroy(obj)
                raise ValueError(f'PDFium could not set text {line!r}')
            raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, height - y - number * fontsize * factor)
            raw.FPDFPage_InsertObject(page.raw, obj)

    def insert_image(self, page, rect, image, *, fit: bool = False) -> None:
        """Place a raster in ``rect``: ``image`` is a PIL image or ``(width, height, (r, g, b))``.

        ``fit=True`` keeps the aspect ratio and centres the image (PyMuPDF ``keep_proportion`` as it
        applied to PIL PNGs); the default fills the rectangle (as PyMuPDF placed its Pixmap PNGs).
        Objects are drawn in insertion order, so insert an underlay before the text.
        """
        if isinstance(image, tuple):
            width, height, (red, green, blue) = image
            bitmap = pdfium.PdfBitmap.new_native(width, height, raw.FPDFBitmap_BGR)
            bitmap.fill_rect(0, 0, width, height, (red, green, blue, 255))
        else:
            width, height = image.size
            bitmap = pdfium.PdfBitmap.from_pil(image)
        x0, y0, x1, y1 = rect
        w, h = x1 - x0, y1 - y0
        if fit:
            scale = min(w / width, h / height)
            x0, y0, w, h = x0 + (w - width * scale) / 2, y0 + (h - height * scale) / 2, width * scale, height * scale
        obj = pdfium.PdfImage.new(self.pdf)
        obj.set_bitmap(bitmap)
        bitmap.close()
        obj.set_matrix(pdfium.PdfMatrix(w, 0, 0, h, x0, page.get_height() - y0 - h))
        page.insert_obj(obj)

    def import_pdf(self, path: str | Path) -> None:
        """Append every page of another PDF (PyMuPDF ``insert_pdf``)."""
        source = pdfium.PdfDocument(str(path))
        try:
            self.pdf.import_pages(source)
        finally:
            source.close()

    def save(self, path: str | Path) -> Path:
        """Generate the content streams of the pages written to, save, and close every handle."""
        try:
            for page in self._pages:
                page.gen_content()
            self.pdf.save(str(path))
        finally:
            for font in self._fonts:
                raw.FPDFFont_Close(font)
            self._fonts.clear()
            for page in self._pages:
                page.close()
            self._pages.clear()
            self.pdf.close()
        return Path(path)
