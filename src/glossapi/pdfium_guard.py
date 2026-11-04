"""Process-wide coordination of PDFium use (K5 quality policy P7, K5-A2).

PDFium is not thread-safe, and glossAPI drives it two ways in one process: through pypdfium2 (ctypes calls that
release the GIL while PDFium runs) and through the Rust-owned layout reader of ``glossapi_rs_pdfrepair``
(``layout_v2_read_document``), which holds the GIL and its own mutex for a whole read. The GIL and that mutex do not
serialize a pypdfium2 call that another thread already has in flight. ``PDFIUM_LOCK`` does: every glossAPI module that
opens a PDF with pypdfium2 holds it for the document's lifetime, and ``glossapi.markdown_layout`` holds it around
every native read, so within this package no two PDFium users ever overlap in one process, whatever the threads.

Library initialization and destruction are never done by glossAPI or by the native reader. pypdfium2 initializes
PDFium when it is first imported (under Python's import lock) and destroys it at interpreter exit (its ``atexit``
handler, in the main thread after non-daemon threads have been joined); glossAPI runs no PDFium work on daemon
threads. The native reader only checks, on each call, that pypdfium2's library is still loaded and initialized.

It is a re-entrant lock: a guarded function may call another guarded function. Worker processes each have their own
PDFium and their own lock; a forked child starts with an unlocked copy (``os.register_at_fork``), because the lock
holder's thread does not exist in the child.

Docling's own pypdfium2 backend serializes its calls with Docling's lock; it runs only on the Docling route, whose
conversions return before the safe route's next document starts in the same process.
"""
from __future__ import annotations

import os
import threading

PDFIUM_LOCK = threading.RLock()


def _reset_after_fork() -> None:
    global PDFIUM_LOCK
    PDFIUM_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def pdfium_guard() -> threading.RLock:
    """The process-wide PDFium lock, for ``with pdfium_guard():`` (looked up at call time, so a forked child's
    fresh lock is used)."""
    return PDFIUM_LOCK
