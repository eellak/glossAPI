import importlib

import pytest

from glossapi.scripts.docling_cost_benchmark import fake_docling


@pytest.fixture
def docling_pipeline():
    # These tests exercise option wiring; use the existing offline fake rather
    # than requiring the optional model stack just to collect the suite.
    with fake_docling("baseline", {}):
        yield importlib.import_module("glossapi.ocr.docling.pipeline")


def test_apply_common_pdf_options_prefers_threaded_pipeline_options_when_available(docling_pipeline):
    acc, _ = docling_pipeline._resolve_accelerator("cuda:0")
    opts = docling_pipeline._apply_common_pdf_options(
        acc=acc,
        images_scale=1.25,
        formula_enrichment=False,
        code_enrichment=False,
    )

    expected_cls = docling_pipeline.ThreadedPdfPipelineOptions or docling_pipeline.PdfPipelineOptions
    assert isinstance(opts, expected_cls)


def test_apply_runtime_overrides_updates_docling_page_batch_size(monkeypatch, docling_pipeline):
    class Perf:
        page_batch_size = 4

    class Settings:
        perf = Perf()

    monkeypatch.setenv("GLOSSAPI_DOCLING_PAGE_BATCH_SIZE", "8")
    monkeypatch.setattr(docling_pipeline, "docling_settings", Settings(), raising=False)

    acc, _ = docling_pipeline._resolve_accelerator("cuda:0")
    docling_pipeline._apply_common_pdf_options(
        acc=acc,
        images_scale=1.25,
        formula_enrichment=False,
        code_enrichment=False,
    )

    assert Settings.perf.page_batch_size == 8
