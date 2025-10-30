from __future__ import annotations

import logging
from typing import Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    LayoutOptions,
    PictureDescriptionApiOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from ._paths import resolve_packaged_onnx_and_keys
from .pool import GLOBAL_RAPID_OCR_POOL
from .safe import SafeRapidOcrModel, patch_docling_rapidocr

_logger = logging.getLogger(__name__)

patch_docling_rapidocr()


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
    # Prefer lightweight placeholder picture descriptions to avoid heavy VLM backends.
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
    """Builder for a Docling PDF pipeline without RapidOCR.

    Returns ``(converter, PdfPipelineOptions)`` where ``converter`` is a
    ``StandardPdfPipeline`` configured for layout extraction only.
    """

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


def build_rapidocr_pipeline(
    *,
    device: str = "cuda:0",
    text_score: float = 0.45,
    images_scale: float = 1.25,
    formula_enrichment: bool = False,
    code_enrichment: bool = False,
) -> Tuple[object, PdfPipelineOptions]:
    """Canonical builder for Docling + RapidOCR pipeline.

    Returns a tuple (engine, PdfPipelineOptions). Prefers explicit RapidOCR injection
    when supported; otherwise returns a DocumentConverter using the factory path.
    """

    def _fallback_layout(reason: str) -> Tuple[object, PdfPipelineOptions]:
        _logger.warning(
            "RapidOCR pipeline fallback: %s. Using Docling layout-only configuration.",
            reason,
        )
        pipeline, opts = build_layout_pipeline(
            device=device,
            images_scale=images_scale,
            formula_enrichment=formula_enrichment,
            code_enrichment=code_enrichment,
        )
        return pipeline, opts

    acc, want_cuda = _resolve_accelerator(device)

    # Optional provider preflight only when CUDA requested
    if want_cuda:
        try:
            import onnxruntime as ort  # type: ignore

            prov = ort.get_available_providers()
            if "CUDAExecutionProvider" not in prov:
                raise RuntimeError(f"CUDAExecutionProvider not available: {prov}")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"onnxruntime-gpu not available or misconfigured: {e}")

    r = resolve_packaged_onnx_and_keys()
    if not (r.det and r.rec and r.cls and r.keys):
        return _fallback_layout("packaged RapidOCR ONNX assets missing")

    ocr_opts = RapidOcrOptions(
        backend="onnxruntime",
        lang=["el", "en"],
        force_full_page_ocr=False,
        use_det=True,
        use_cls=False,
        use_rec=True,
        text_score=text_score,
        det_model_path=r.det,
        rec_model_path=r.rec,
        cls_model_path=r.cls,
        print_verbose=False,
    )
    ocr_opts.rec_keys_path = r.keys

    opts = _apply_common_pdf_options(
        acc=acc,
        images_scale=float(images_scale),
        formula_enrichment=formula_enrichment,
        code_enrichment=code_enrichment,
    )
    opts.do_ocr = True
    opts.ocr_options = ocr_opts

    # Prefer explicit injection of RapidOCR model when available
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel  # type: ignore

        try:
            from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        except Exception:  # pragma: no cover
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore

        import inspect

        sig = inspect.signature(StandardPdfPipeline.__init__)
        if "ocr_model" not in sig.parameters:
            raise RuntimeError("Docling build does not support RapidOCR injection")

        def _factory():
            try:
                return SafeRapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover
                # Fall back to the stock implementation if our wrapper misbehaves.
                return RapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]

        pooled_model = GLOBAL_RAPID_OCR_POOL.get(
            str(acc.device),
            ocr_opts,
            _factory,
            expected_type=SafeRapidOcrModel,
        )
        pipeline = StandardPdfPipeline(opts, ocr_model=pooled_model)  # type: ignore
        return pipeline, opts
    except Exception as exc:
        _logger.warning(
            "RapidOCR injection unavailable (%s); using DocumentConverter factory path.",
            exc,
        )

    # Fallback: use DocumentConverter factory
    try:
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        return converter, opts
    except Exception as exc:
        return _fallback_layout(f"DocumentConverter failed: {exc}")


__all__ = ["build_layout_pipeline", "build_rapidocr_pipeline"]
