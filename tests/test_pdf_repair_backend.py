"""Public repair selection and optional-extension failure contracts."""
import importlib.abc
import sys
from types import SimpleNamespace

import pytest

from glossapi import pdf_repair
from glossapi.pdf_repair import rust_backend, tounicode


class MissingExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'glossapi_rs_pdfrepair':
            raise ImportError('injected unavailable extension')


@pytest.mark.parametrize('operation', ['analyze_pdf', 'repair_pdf'])
def test_rust_adapter_falls_back_on_importerror(tmp_path, monkeypatch, operation):
    monkeypatch.delitem(sys.modules, 'glossapi_rs_pdfrepair', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [MissingExtension(), *sys.meta_path])
    calls = []
    report = (tounicode.PdfFontReport('source') if operation == 'analyze_pdf'
              else tounicode.RepairResult('source', 'destination'))
    def reference(*args, **kwargs):
        calls.append((args, kwargs))
        return report
    monkeypatch.setattr(tounicode, operation, reference)
    arguments = (tmp_path / 'source.pdf',) if operation == 'analyze_pdf' else (
        tmp_path / 'source.pdf', tmp_path / 'destination.pdf')
    assert getattr(rust_backend, operation)(*arguments) == report.to_dict()
    assert len(calls) == 1


@pytest.mark.parametrize('operation', ['analyze_pdf', 'repair_pdf'])
def test_public_api_selects_available_rust_and_keeps_typed_result(tmp_path, monkeypatch, operation):
    calls = []
    report = (tounicode.PdfFontReport('source') if operation == 'analyze_pdf'
              else tounicode.RepairResult('source', 'destination'))
    def native(*args):
        calls.append(args)
        return report.to_dict()
    extension = SimpleNamespace(**{operation: native})
    monkeypatch.setitem(sys.modules, 'glossapi_rs_pdfrepair', extension)
    monkeypatch.setattr(tounicode, operation, lambda *a, **kw: pytest.fail('Rust must be selected'))
    arguments = (tmp_path / 'source.pdf',) if operation == 'analyze_pdf' else (
        tmp_path / 'source.pdf', tmp_path / 'destination.pdf')
    result = getattr(pdf_repair, operation)(*arguments)
    assert isinstance(result, type(report))
    assert result.backend == 'rust'
    assert len(calls) == 1


def test_public_fallback_receipt_and_native_runtime_error(monkeypatch):
    monkeypatch.delitem(sys.modules, 'glossapi_rs_pdfrepair', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [MissingExtension(), *sys.meta_path])
    report = tounicode.PdfFontReport('source')
    monkeypatch.setattr(tounicode, 'analyze_pdf', lambda path: report)
    assert pdf_repair.analyze_pdf('source') is report
    assert report.to_dict()['backend'] == 'python'
    def broken_native(path):
        raise RuntimeError('injected native execution error')
    monkeypatch.setitem(sys.modules, 'glossapi_rs_pdfrepair', SimpleNamespace(analyze_pdf=broken_native))
    with pytest.raises(RuntimeError, match='native execution error'):
        pdf_repair.analyze_pdf('source')
