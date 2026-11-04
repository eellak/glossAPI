"""Page text and char tables for the safe extractor: one PDFium load per page (K5-A2, policy P2/P7).

Reader threat model (K5 quality policy P7, decision DECIDE-reader-threat-model, 4 Nov 2026): Documents and public
argument values are untrusted; the Python runtime, installed dependencies and executable caller code are trusted. The
wrapper validates arguments and preserves native handle ownership and lifetime. Dependency replacement, forged ctypes
state and uncoordinated library teardown are unsupported. All PDFium users, initialization and destruction require
process-wide coordination; the reader's mutex and GIL do not serialize external ctypes calls. Identity checks and
self-tests detect accidental incompatibility, not malicious substitution. This boundary does not sandbox PDFium or
guarantee unlimited resource availability.

``open_page_source(path, chars=...)`` returns the reader the safe extractor uses for every page of a document:

- **Native** (default): ``glossapi_rs_pdfrepair.layout_v2_read_document(str(path), pages, sha)`` opens the document,
  each page and each text page with Rust-owned handles and returns, per page, the text exactly as production's
  ``PdfTextPage.get_text_range()`` returns it (pypdfium2 4.30 redirects it to ``get_text_bounded()``; byte-equal on
  141/141 Mac and 24,551/24,551 cohort242 pages) together with the page's ``.k5chars`` table, from the same text
  page. Pages are read in chunks of ``CHUNK_PAGES`` on demand, in page order. With ``chars=False`` (the plain Markdown
  mode) the reader reads the text only (``layout_v2_read_document(..., chars=False)``).
- **Exactness of failures.** A document the native reader cannot open (``LayoutV2DocumentError``) is opened with
  pypdfium2 instead, so the caller sees exactly pypdfium2's exception (or, if pypdfium2 can open it, reads it the
  pypdfium2 way). A page the native reader reports as failed is read the pypdfium2 way, so the caller sees exactly
  pypdfium2's per-page exception or text. Such pages have no char table.
- **Python reader** (fallback): ``LayoutV2ReaderUnavailable`` (pypdfium2 other than 4.x, an identity or self-test
  failure, a destroyed library) selects pypdfium2 for the text, as before K5-A2, and the K5c Python char reader on
  the same text page (exact, about 10 ms/page). A missing or stale native extension does the same without char
  tables (no labels).

Every call into PDFium here runs under ``glossapi.pdfium_guard`` (process-wide lock); the caller holds it for the
document's lifetime too.
"""
from __future__ import annotations

import ctypes
import struct
from array import array
from pathlib import Path
from typing import Any, Optional

from ..pdfium_guard import pdfium_guard

CHUNK_PAGES = 32
NO_SHA = b"\0" * 32
_FLAG_SYMBOLIC, _FLAG_ITALIC = 4, 64
GENERATED, SOFT_HYPHEN, MAP_ERROR, ITALIC, SYMBOLIC, INVISIBLE = 1, 2, 4, 8, 16, 32
_HEAD = struct.Struct("<8sHHIII dd 32s")
_MAGIC = b"K5CHARS\0"
_REQUIRED = ("layout_v2_read_document", "LayoutV2ContextBuilder", "layout_v2_input_problem", "layout_v2_label_page",
             "LayoutV2ReaderUnavailable", "LayoutV2DocumentError", "page_decode_status_json")
_native_state: Optional[tuple[Any, Optional[str]]] = None


def native_module() -> tuple[Any, Optional[str]]:
    """``(module, None)`` when ``glossapi_rs_pdfrepair`` declares this package's native API level and the layout
    exports, else ``(None, reason)``. A stale build (older level) is refused, as the repair selector refuses it."""
    global _native_state
    if _native_state is None:
        from ..capabilities import NATIVE_API_LEVELS
        name = "glossapi_rs_pdfrepair"
        wanted = f"GLOSSAPI_NATIVE_API:{name}:{NATIVE_API_LEVELS[name]};"
        try:
            import importlib
            module = importlib.import_module(name)
            module = getattr(module, name, module)
        except ImportError as error:
            _native_state = (None, f"native module {name} is not installed: {error}")
        else:
            found = getattr(module, "__glossapi_native_api__", None)
            missing = [attr for attr in _REQUIRED if not hasattr(module, attr)]
            if found != wanted:
                _native_state = (None, f"stale native module {name}: declares {found!r}, this glossapi needs "
                                       f"{wanted!r}; rebuild core")
            elif missing:
                _native_state = (None, f"native module {name} lacks {', '.join(missing)}")
            else:
                _native_state = (module, None)
    return _native_state


def _reset_native_state() -> None:
    """Test hook: forget the cached module check."""
    global _native_state
    _native_state = None


# ---- the K5c Python char reader (exact reference of the native reader; the fallback) --------------------------------
def _arr(code: str, values) -> bytes:
    data = array(code, values)
    if struct.pack("=H", 1) != struct.pack("<H", 1):   # big-endian host
        data.byteswap()
    return data.tobytes()


def read_chars_python(raw, textpage_raw, page_w: float, page_h: float, rotation: int, page_no: int,
                      sha: bytes = NO_SHA) -> bytes:
    """The K5c fixture reader (``k5_prep/layout/freeze_chars.read_chars`` + ``k5chars.dumps``) on a pypdfium2 text
    page: the ``.k5chars`` v1 bytes, equal to the native reader's on every tested page."""
    n = raw.FPDFText_CountChars(textpage_raw)
    fonts: dict[str, int] = {}
    ch, bbox, loose, size, weight, angle, font, flags = [], [], [], [], [], [], [], []
    left, right, bottom, top = (ctypes.c_double() for _ in range(4))
    rect = raw.FS_RECTF()
    font_flags = ctypes.c_int()
    buf = ctypes.create_string_buffer(512)
    for i in range(n):
        u = raw.FPDFText_GetUnicode(textpage_raw, i)
        raw.FPDFText_GetCharBox(textpage_raw, i, ctypes.byref(left), ctypes.byref(right), ctypes.byref(bottom),
                                ctypes.byref(top))
        raw.FPDFText_GetLooseCharBox(textpage_raw, i, ctypes.byref(rect))
        s = raw.FPDFText_GetFontSize(textpage_raw, i)
        w = raw.FPDFText_GetFontWeight(textpage_raw, i)
        ln = raw.FPDFText_GetFontInfo(textpage_raw, i, buf, len(buf), ctypes.byref(font_flags))
        name = buf.raw[:max(0, ln - 1)].decode("utf-8", "replace") if ln > 0 else ""
        a = raw.FPDFText_GetCharAngle(textpage_raw, i)
        f = 0
        f |= GENERATED if raw.FPDFText_IsGenerated(textpage_raw, i) == 1 else 0
        f |= SOFT_HYPHEN if u == 0x02 else 0
        f |= MAP_ERROR if raw.FPDFText_HasUnicodeMapError(textpage_raw, i) == 1 else 0
        f |= ITALIC if ln > 0 and font_flags.value & _FLAG_ITALIC else 0
        f |= SYMBOLIC if ln > 0 and font_flags.value & _FLAG_SYMBOLIC else 0
        f |= INVISIBLE if raw.FPDFText_GetTextRenderMode(textpage_raw, i) == 3 else 0
        ch.append(u if 0 <= u <= 0x10FFFF else 0xFFFD)
        bbox.extend((left.value, bottom.value, right.value, top.value))
        loose.extend((rect.left, rect.bottom, rect.right, rect.top))
        size.append(float(s))
        weight.append(max(-32768, min(32767, int(w))))
        angle.append(float(a))
        font.append(fonts.setdefault(name, len(fonts)))
        flags.append(f)
    out = [_HEAD.pack(_MAGIC, 1, rotation, n, len(fonts), page_no, page_w, page_h, sha)]
    for name in fonts:
        encoded = name.encode("utf-8")
        out.append(struct.pack("<H", len(encoded)) + encoded)
    out += [_arr("I", ch), _arr("d", bbox), _arr("d", loose), _arr("f", size), _arr("h", weight), _arr("f", angle),
            _arr("I", font), bytes(flags)]
    return b"".join(out)


def real_chars(table: bytes) -> int:
    """Chars of a ``.k5chars`` table that PDFium did not generate (the K5c fixture rule: a page with none has no
    text layer). The flags array is the table's last ``n`` bytes."""
    n = _HEAD.unpack_from(table, 0)[3]
    if not n:
        return 0
    flags = table[-n:]
    return n - flags.translate(_GENERATED_BIT).count(1)


_GENERATED_BIT = bytes(b & GENERATED for b in range(256))


# ---- page sources ---------------------------------------------------------------------------------------------------
def _legacy_text(page) -> str:
    """Production's per-page text sequence before K5-A2 (text page, ``get_text_range()``, close)."""
    text_page = page.get_textpage()
    try:
        return text_page.get_text_range()
    finally:
        text_page.close()


class LegacySource:
    """pypdfium2 for the text (the pre-K5-A2 sequence, unchanged), plus the Python char reader on the same text page
    when ``chars`` is requested. ``reason`` says why the native reader is not used (None: chars not requested)."""

    kind = "pypdfium2"

    def __init__(self, path: Path, *, chars: bool, page_no_offset: int = 1, sha: bytes = NO_SHA,
                 reason: Optional[str] = None):
        import pypdfium2 as pdfium
        self.path = Path(path)
        self.reason = reason
        self._chars_wanted = chars
        self._sha = sha
        self._tables: dict[int, Optional[bytes]] = {}
        self.reader = "python" if chars else None
        with pdfium_guard():
            self.doc = pdfium.PdfDocument(str(self.path))
            try:
                self.page_count = len(self.doc)
            except BaseException:
                self.doc.close()
                raise

    def text(self, index: int) -> str:
        with pdfium_guard():
            page = self.doc[index]
            try:
                text_page = page.get_textpage()
                try:
                    content = text_page.get_text_range()
                    if self._chars_wanted:
                        self._tables[index] = self._read_chars(page, text_page, index)
                finally:
                    text_page.close()
            finally:
                page.close()
        return content

    def _read_chars(self, page, text_page, index: int) -> Optional[bytes]:
        try:
            import pypdfium2.raw as raw
            x0, y0, x1, y1 = page.get_mediabox()
            return read_chars_python(raw, text_page.raw, x1 - x0, y1 - y0, page.get_rotation(), index + 1, self._sha)
        except Exception:      # a char-table failure never changes the text; the page is emitted unlabelled
            return None

    def chars(self, index: int) -> Optional[bytes]:
        return self._tables.pop(index, None)

    def reread_chars(self, index: int) -> Optional[bytes]:
        with pdfium_guard():
            try:
                page = self.doc[index]
            except Exception:
                return None
            try:
                text_page = page.get_textpage()
                try:
                    return self._read_chars(page, text_page, index)
                finally:
                    text_page.close()
            except Exception:
                return None
            finally:
                page.close()

    def close(self) -> None:
        with pdfium_guard():
            self.doc.close()


class NativeSource:
    """Rust-owned handles: text and char table of each page from one load (``layout_v2_read_document``)."""

    kind = "native"
    reader = "native"
    reason = None

    def __init__(self, native, path: Path, *, sha: bytes = NO_SHA, chars: bool = True):
        self.native = native
        self._chars = chars        # False: text only (the plain Markdown mode reads no char tables)
        self.path = Path(path)
        self._source = str(self.path)
        self._sha = sha
        self._entries: dict[int, tuple[Optional[bytes], Optional[str], Optional[str]]] = {}
        self._legacy = None
        self.reads = 0
        with pdfium_guard():
            count, _ = native.layout_v2_read_document(self._source, [], sha)
        self.page_count = int(count)

    def _load(self, index: int) -> None:
        pages = list(range(index + 1, min(self.page_count, index + CHUNK_PAGES) + 1))
        try:
            with pdfium_guard():
                _, rows = self.native.layout_v2_read_document(self._source, pages, self._sha, chars=self._chars)
        except Exception as error:      # the document changed or the library went away: pypdfium2 for these pages
            rows = [(page_no, None, None, f"native read failed: {type(error).__name__}: {error}") for page_no in pages]
        self.reads += 1
        for page_no, table, text, error in rows:
            self._entries[page_no - 1] = (table, text, error)

    def _legacy_doc(self):
        if self._legacy is None:
            import pypdfium2 as pdfium
            with pdfium_guard():
                self._legacy = pdfium.PdfDocument(str(self.path))
        return self._legacy

    def text(self, index: int) -> str:
        if index not in self._entries:
            self._load(index)
        table, text, error = self._entries[index]
        if error is not None or text is None:
            # The native reader could not load this page: read it the pypdfium2 way so the caller sees exactly
            # pypdfium2's exception or text. The page has no char table.
            self._entries[index] = (None, None, error or "no text")
            with pdfium_guard():
                page = self._legacy_doc()[index]
                try:
                    return _legacy_text(page)
                finally:
                    page.close()
        return text

    def chars(self, index: int) -> Optional[bytes]:
        entry = self._entries.pop(index, None)
        return entry[0] if entry else None

    def reread_chars(self, index: int) -> Optional[bytes]:
        try:
            with pdfium_guard():
                _, rows = self.native.layout_v2_read_document(self._source, [index + 1], self._sha)
        except Exception:
            return None
        self.reads += 1
        return rows[0][1] if rows else None

    def close(self) -> None:
        self._entries.clear()
        if self._legacy is not None:
            with pdfium_guard():
                self._legacy.close()
            self._legacy = None


def open_page_source(path: Path, *, chars: bool, sha: bytes = NO_SHA):
    """The page reader for one document: ``NativeSource`` when the native reader is usable, else ``LegacySource``
    (pypdfium2 text; Python char reader when ``chars``). Raises exactly what ``pypdfium2.PdfDocument(str(path))``
    raises for a document that does not open."""
    native, why = native_module()
    if native is not None:
        try:
            return NativeSource(native, path, sha=sha, chars=chars)
        except native.LayoutV2ReaderUnavailable as error:
            return LegacySource(path, chars=chars, sha=sha, reason=f"native reader unavailable: {error}")
        except native.LayoutV2DocumentError:
            # pypdfium2 opens it (or raises its own exception, which the caller records as before K5-A2).
            return LegacySource(path, chars=chars, sha=sha, reason="native reader could not open the document")
    return LegacySource(path, chars=False, sha=sha, reason=why)
