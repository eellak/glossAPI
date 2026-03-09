from __future__ import annotations

from typing import Tuple

from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    LayoutOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    TableFormerMode,
    TableStructureOptions,
)


def _resolve_accelerator(device: str | None) -> Tuple[AcceleratorOptions, bool]:
    """Return accelerator options and whether CUDA was requested."""
    dev = device or "cuda:0"
    if isinstance(dev, str) and dev.lower().startswith(("cuda", "mps", "cpu")):
        acc = AcceleratorOptions(device=dev)
        want_cuda = dev.lower().startswith("cuda")
    else:
        want_cuda = str(dev).lower().startswith("cuda")
        acc = AcceleratorOptions(
            device=AcceleratorDevice.CUDA if want_cuda else AcceleratorDevice.CPU
        )
    return acc, want_cuda


def _apply_common_pdf_options(
    *,
    acc: AcceleratorOptions,
    images_scale: float,
    formula_enrichment: bool,
    code_enrichment: bool,
) -> PdfPipelineOptions:
    table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    try:
        if hasattr(table_opts, "do_cell_matching"):
            table_opts.do_cell_matching = True
    except Exception:
        pass

    opts = PdfPipelineOptions(
        accelerator_options=acc,
        layout_options=LayoutOptions(),
        do_ocr=False,
        do_table_structure=True,
        do_formula_enrichment=bool(formula_enrichment),
        do_code_enrichment=bool(code_enrichment),
        force_backend_text=False,
        generate_parsed_pages=False,
        table_structure_options=table_opts,
        allow_external_plugins=True,
    )
    try:
        if hasattr(opts, "do_picture_description"):
            opts.do_picture_description = False
        if getattr(opts, "picture_description_options", None) is None:
            opts.picture_description_options = PictureDescriptionApiOptions()
        if hasattr(opts, "enable_remote_services"):
            opts.enable_remote_services = False
    except Exception:
        pass
    try:
        setattr(opts, "images_scale", images_scale)
    except Exception:
        pass
    return opts


def build_layout_pipeline(
    *,
    device: str = "cuda:0",
    images_scale: float = 1.25,
    formula_enrichment: bool = False,
    code_enrichment: bool = False,
) -> Tuple[object, PdfPipelineOptions]:
    """Create a Docling layout-only PDF pipeline."""

    acc, _ = _resolve_accelerator(device)
    opts = _apply_common_pdf_options(
        acc=acc,
        images_scale=float(images_scale),
        formula_enrichment=formula_enrichment,
        code_enrichment=code_enrichment,
    )

    try:
        from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
    except Exception:  # pragma: no cover
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore

    pipeline = StandardPdfPipeline(opts)  # type: ignore[arg-type]
    return pipeline, opts
