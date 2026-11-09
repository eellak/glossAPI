import json

import pandas as pd
import pytest
import sys
from types import SimpleNamespace


@pytest.fixture
def plugin_factory(monkeypatch):
    calls = []
    def create_options(kind, **kwargs):
        calls.append((kind, kwargs))
        if kind != "stub-plugin":
            raise ValueError("unknown OCR engine")
        return SimpleNamespace(kind=kind, **kwargs)
    def get_ocr_factory(*, allow_external_plugins):
        assert allow_external_plugins is True
        return SimpleNamespace(create_options=create_options)
    import types
    factory_module = types.ModuleType("docling.models.factories")
    factory_module.get_ocr_factory = get_ocr_factory
    monkeypatch.setitem(sys.modules, "docling.models.factories", factory_module)
    return calls


from glossapi.scripts import full_pipeline_checkpoint as checkpoint


def test_read_metadata_counts_handles_missing_and_populated_parquet(tmp_path):
    missing = checkpoint._read_metadata_counts(tmp_path / "missing.parquet")
    assert missing["rows_total"] == 0

    parquet_path = tmp_path / "download_results.parquet"
    pd.DataFrame(
        [
            {"filename": "a.pdf", "needs_ocr": True, "ocr_success": False, "text": ""},
            {"filename": "b.pdf", "needs_ocr": False, "ocr_success": True, "text": "hello"},
        ]
    ).to_parquet(parquet_path, index=False)

    counts = checkpoint._read_metadata_counts(parquet_path)
    assert counts == {
        "rows_total": 2,
        "needs_ocr_true": 1,
        "ocr_success_true": 1,
        "text_nonempty": 1,
    }


def test_full_pipeline_checkpoint_main_writes_summary(tmp_path, monkeypatch, plugin_factory):
    class DummyCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir

        def _metadata_path(self):
            path = self.output_dir / "download_results" / "download_results.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def extract(self, **kwargs):
            md = self.output_dir / "markdown"
            md.mkdir(parents=True, exist_ok=True)
            (md / "doc.md").write_text("raw text", encoding="utf-8")
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": False, "text": ""}]
            ).to_parquet(self._metadata_path(), index=False)

        def clean(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": True, "ocr_success": False, "text": ""}]
            ).to_parquet(self._metadata_path(), index=False)

        def ocr(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": True, "text": "fixed text"}]
            ).to_parquet(self._metadata_path(), index=False)

        def jsonl(self, output_path, **kwargs):
            output_path.write_text(json.dumps({"text": "fixed text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint, "Corpus", DummyCorpus)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "doc.pdf").write_bytes(b"%PDF-1.4\n")

    output_dir = tmp_path / "out"
    export_path = tmp_path / "export.jsonl"
    report_path = tmp_path / "report.json"

    rc = checkpoint.main(
        [
            "--ocr-engine", "stub-plugin",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-path",
            str(export_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["post_clean_counts"]["needs_ocr_true"] == 1
    assert report["post_ocr_counts"]["ocr_success_true"] == 1
    assert report["export_records"] == 1


def test_full_pipeline_checkpoint_can_resume_from_ocr_phase(tmp_path, monkeypatch, plugin_factory):
    class DummyCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir

        def _metadata_path(self):
            path = self.output_dir / "download_results" / "download_results.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def extract(self, **kwargs):
            raise AssertionError("extract should have been skipped")

        def clean(self, **kwargs):
            raise AssertionError("clean should have been skipped")

        def ocr(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": True, "text": "fixed text"}]
            ).to_parquet(self._metadata_path(), index=False)

        def jsonl(self, output_path, **kwargs):
            output_path.write_text(json.dumps({"text": "fixed text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint, "Corpus", DummyCorpus)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    metadata_path = output_dir / "download_results" / "download_results.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"filename": "doc.pdf", "needs_ocr": True, "ocr_success": False, "text": ""}]
    ).to_parquet(metadata_path, index=False)

    export_path = tmp_path / "export.jsonl"
    report_path = tmp_path / "report.json"

    rc = checkpoint.main(
        [
            "--ocr-engine", "stub-plugin",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-path",
            str(export_path),
            "--report-path",
            str(report_path),
            "--skip-extract",
            "--skip-clean",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["skipped_phases"] == ["extract", "clean"]
    assert report["post_extract_counts"]["needs_ocr_true"] == 1
    assert report["post_ocr_counts"]["ocr_success_true"] == 1
    assert report["export_records"] == 1




def test_full_pipeline_checkpoint_retries_empty_export_when_ocr_text_exists(tmp_path, monkeypatch, plugin_factory):
    calls = {"jsonl": 0}

    class DummyCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir

        def _metadata_path(self):
            path = self.output_dir / "download_results" / "download_results.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        def extract(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": True, "ocr_success": False, "text": ""}]
            ).to_parquet(self._metadata_path(), index=False)

        def clean(self, **kwargs):
            return None

        def ocr(self, **kwargs):
            pd.DataFrame(
                [{"filename": "doc.pdf", "needs_ocr": False, "ocr_success": True, "text": "fixed text"}]
            ).to_parquet(self._metadata_path(), index=False)

        def jsonl(self, output_path, **kwargs):
            calls["jsonl"] += 1
            if calls["jsonl"] == 1:
                output_path.write_text("", encoding="utf-8")
                return
            output_path.write_text(json.dumps({"text": "fixed text"}) + "\n", encoding="utf-8")

    monkeypatch.setattr(checkpoint, "Corpus", DummyCorpus)

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    export_path = tmp_path / "export.jsonl"
    report_path = tmp_path / "report.json"

    rc = checkpoint.main(
        [
            "--ocr-engine", "stub-plugin",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--export-path",
            str(export_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert rc == 0
    assert calls["jsonl"] == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["post_ocr_counts"]["text_nonempty"] == 1
    assert report["export_records"] == 1


def test_checkpoint_forwards_plugin_options_through_real_ocr(tmp_path, monkeypatch, plugin_factory):
    from pypdf import PdfWriter
    from glossapi.corpus.ocr import dispatch
    source = tmp_path / 'input'
    source.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(source / 'source.pdf')
    out = tmp_path / 'output'
    metadata = out / 'download_results/download_results.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame([dict(filename='source.pdf', needs_ocr=True, ocr_pages='')]).to_parquet(metadata)
    seen = []
    def build(request):
        seen.append(request.ocr_options)
        return SimpleNamespace(convert=lambda *a, **kw: SimpleNamespace(status='success',
            document=SimpleNamespace(export_to_markdown=lambda **kw: 'STUB_PLUGIN_OUTPUT\n\nCorrect body.')))
    monkeypatch.setattr(dispatch, '_build_converter', build)
    export = tmp_path / 'export.jsonl'
    report = tmp_path / 'report.json'
    assert checkpoint.main(['--input-dir', str(source), '--output-dir', str(out),
        '--export-path', str(export), '--report-path', str(report), '--skip-extract', '--skip-clean',
        '--ocr-engine', 'stub-plugin', '--ocr-options', '{"lang":["el"],"force_full_page_ocr":true}']) == 0
    assert plugin_factory == [('stub-plugin', {'lang': ['el'], 'force_full_page_ocr': True})]
    assert seen[0].kind == 'stub-plugin' and seen[0].lang == ['el']
    assert seen[0].force_full_page_ocr is True
    assert json.loads(report.read_text())['post_ocr_counts']['ocr_success_true'] == 1
    assert 'STUB_PLUGIN_OUTPUT' in json.loads(export.read_text())['text']


@pytest.mark.parametrize('flags', [[], ['--ocr-engine', 'stub-plugin', '--ocr-options', '[]'],
    ['--ocr-engine', 'stub-plugin', '--ocr-options', '{'],
    ['--ocr-engine', 'unknown'], ['--ocr-engine', 'stub-plugin', '--ocr-options', '{"kind":"other"}']])
def test_bad_plugin_config_fails_before_output_mutation(tmp_path, monkeypatch, plugin_factory, flags):
    out = tmp_path / 'output'
    out.mkdir()
    marker = out / 'keep.txt'
    marker.write_text('keep')
    monkeypatch.setattr(checkpoint, 'Corpus', lambda **kw: pytest.fail('constructed Corpus'))
    with pytest.raises(ValueError):
        checkpoint.main(['--input-dir', str(tmp_path), '--output-dir', str(out),
            '--export-path', str(tmp_path / 'export'), '--report-path', str(tmp_path / 'report'),
            '--clean-output-dir', *flags])
    assert marker.read_text() == 'keep'


def test_checkpoint_loads_stub_external_docling_plugin(tmp_path, monkeypatch):
    """Real Docling factory discovery/options validation, with a stub entry point."""
    from typing import ClassVar
    factories = pytest.importorskip('docling.models.factories')
    from docling.datamodel.pipeline_options import OcrOptions
    from docling.models.factories import base_factory
    from glossapi.corpus.ocr import dispatch
    from pypdf import PdfWriter

    class StubOptions(OcrOptions):
        kind: ClassVar[str] = 'stub-external'
        token: str

    class StubEngine:
        @classmethod
        def get_options_type(cls):
            return StubOptions

    plugin = SimpleNamespace(__name__='external_stub_plugin',
                             ocr_engines=lambda: {'ocr_engines': [StubEngine]})
    class Plugins:
        def __init__(self, name):
            assert name == 'docling'
        def load_setuptools_entrypoints(self, name):
            assert name == 'docling'
        def list_name_plugin(self):
            return [('stub-external', plugin)]

    monkeypatch.setattr(base_factory, 'PluginManager', Plugins)
    factories.get_ocr_factory.cache_clear()
    source = tmp_path / 'input'
    source.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(source / 'source.pdf')
    out = tmp_path / 'output'
    metadata = out / 'download_results/download_results.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame([dict(filename='source.pdf', needs_ocr=True, ocr_pages='')]).to_parquet(metadata)
    def build(request):
        assert isinstance(request.ocr_options, StubOptions)
        assert request.ocr_options.token == 'sentinel'
        return SimpleNamespace(convert=lambda *a, **kw: SimpleNamespace(status='success',
            document=SimpleNamespace(export_to_markdown=lambda **kw: 'EXTERNAL_PLUGIN_SENTINEL')))
    monkeypatch.setattr(dispatch, '_build_converter', build)
    try:
        assert checkpoint.main(['--input-dir', str(source), '--output-dir', str(out),
            '--export-path', str(tmp_path / 'export.jsonl'), '--report-path', str(tmp_path / 'report.json'),
            '--skip-extract', '--skip-clean', '--ocr-engine', 'stub-external',
            '--ocr-options', '{"token":"sentinel","lang":["el"]}']) == 0
        assert 'EXTERNAL_PLUGIN_SENTINEL' in (tmp_path / 'export.jsonl').read_text()
    finally:
        factories.get_ocr_factory.cache_clear()
