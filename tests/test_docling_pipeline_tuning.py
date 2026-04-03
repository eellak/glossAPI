from glossapi.ocr.docling import pipeline as docling_pipeline


def test_apply_common_pdf_options_prefers_threaded_pipeline_options_when_available():
    acc, _ = docling_pipeline._resolve_accelerator("cuda:0")
    opts = docling_pipeline._apply_common_pdf_options(
        acc=acc,
        images_scale=1.25,
        formula_enrichment=False,
        code_enrichment=False,
    )

    expected_cls = docling_pipeline.ThreadedPdfPipelineOptions or docling_pipeline.PdfPipelineOptions
    assert isinstance(opts, expected_cls)
