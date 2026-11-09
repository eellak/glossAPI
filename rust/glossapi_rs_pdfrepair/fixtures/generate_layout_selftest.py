"""Write fixtures/layout_selftest.pdf: the tiny text page the native layout reader checks itself against at first use
(K5-A1r). Two Helvetica lines (12 pt, 9 pt), so PDFium reports real and generated chars, boxes, sizes and a font name.
Run from the repo root: python rust/glossapi_rs_pdfrepair/fixtures/generate_layout_selftest.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
from pdf_fixtures import FixturePdf  # noqa: E402

fx = FixturePdf()
page = fx.new_page(200, 100)
fx.insert_text(page, (20, 30), "Ab 1.", fontsize=12)
fx.insert_text(page, (20, 60), "Cd", fontsize=9)
fx.save(HERE / "layout_selftest.pdf")
print((HERE / "layout_selftest.pdf").stat().st_size)
