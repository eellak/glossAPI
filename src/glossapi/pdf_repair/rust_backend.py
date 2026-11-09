"""Optional native repair backend helpers.

The Rust extension decides text presence natively from content-stream text
operators; it no longer calls back into Python. ``_sample_json`` is kept as the
PDFium reference sample that the native decision is gated against (the K5a
decision-equality gate and ``--sample``). The dict-returning adapters use the
same optional-extension selector as the typed public API.
"""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import pypdfium2 as pdfium
from ..pdfium_guard import pdfium_guard


def _sample_json(path: str) -> str:
    """PDFium reference text-presence sample (not on the default path; see module docstring)."""
    with pdfium_guard(), closing(pdfium.PdfDocument(path)) as pdf:
        count = len(pdf)
        indexes = (sorted({round(i * (count - 1) / (min(count, 8) - 1))
                           for i in range(min(count, 8))})
                   if count > 1 else list(range(count)))
        has_text = False
        for index in indexes:
            page = pdf[index]
            try:
                text = page.get_textpage()
                try:
                    has_text |= bool(text.get_text_range().strip())
                finally:
                    text.close()
            finally:
                page.close()
    return json.dumps(dict(pages=count, sampled_pages=[i + 1 for i in indexes],
                           has_text=has_text))


def analyze_pdf(path: str | Path) -> dict:
    from .selector import get_backend
    return get_backend().analyze_pdf(path).to_dict()


def repair_pdf(src: str | Path, dst: str | Path, *, protected_sources=()) -> dict:
    from .selector import get_backend
    return get_backend().repair_pdf(src, dst, protected_sources=protected_sources).to_dict()




def _cli_worker(*args):
    # Reuse the proven bounded worker/report/metrics infrastructure. Only the
    # analyze/repair decisions and PDF/font parsing are replaced by Rust.
    from glossapi.pdf_repair.reports import PdfFontReport, FontReport, RepairResult
    from glossapi.scripts.pdf_font_repair import _worker as _reference_worker
    import glossapi.scripts.pdf_font_repair as cli

    def analyze(path):
        result = analyze_pdf(path)
        result['fonts'] = [FontReport(**f) for f in result['fonts']]
        return PdfFontReport(**result)

    def repair(src, dst, *, protected_sources=()):
        result = repair_pdf(src, dst, protected_sources=protected_sources)
        result['fonts'] = [FontReport(**f) for f in result['fonts']]
        return RepairResult(**result)

    cli.analyze_pdf = analyze
    cli.repair_pdf = repair
    _reference_worker(*args)


def cli_main(argv):
    import glossapi.scripts.pdf_font_repair as cli
    cli._worker = _cli_worker
    return cli.main(argv)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--sample':
        print(_sample_json(sys.argv[2]))
    elif len(sys.argv) > 2 and sys.argv[1] == '--cli':
        raise SystemExit(cli_main(sys.argv[2:]))
    else:
        raise SystemExit('usage: rust_backend --sample PDF | --cli classify|repair ...')
