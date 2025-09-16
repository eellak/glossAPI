from __future__ import annotations

from typing import Tuple

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    LayoutOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from ._rapidocr_paths import resolve_packaged_onnx_and_keys
from .rapidocr_safe import SafeRapidOcrModel


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
    # Respect literal device strings like "cuda:0" / "mps" / "cpu"
    dev = device or "cuda:0"
    if isinstance(dev, str) and dev.lower().startswith(("cuda", "mps", "cpu")):
        acc = AcceleratorOptions(device=dev)
        want_cuda = dev.lower().startswith("cuda")
    else:
        want_cuda = str(dev).lower().startswith("cuda")
        acc = AcceleratorOptions(device=AcceleratorDevice.CUDA if want_cuda else AcceleratorDevice.CPU)

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
        raise FileNotFoundError("Packaged RapidOCR ONNX models/keys not found under glossapi.models.")

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

    table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    opts = PdfPipelineOptions(
        accelerator_options=acc,
        ocr_options=ocr_opts,
        layout_options=LayoutOptions(),
        do_ocr=True,
        do_table_structure=True,
        do_formula_enrichment=bool(formula_enrichment),
        do_code_enrichment=bool(code_enrichment),
        force_backend_text=False,
        generate_parsed_pages=False,
        table_structure_options=table_opts,
        allow_external_plugins=True,
    )
    try:
        setattr(opts, "images_scale", images_scale)
    except Exception:
        pass

    # Prefer explicit injection of RapidOCR model when available
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel  # type: ignore
        try:
            from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        except Exception:  # pragma: no cover
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        try:
            ocr_model = SafeRapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover
            # Fall back to the stock implementation if our wrapper misbehaves.
            ocr_model = RapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]
        pipeline = StandardPdfPipeline(opts, ocr_model=ocr_model)  # type: ignore
        return pipeline, opts
    except Exception:
        pass

    # Fallback: use DocumentConverter factory
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return converter, opts


__all__ = ["build_rapidocr_pipeline"]
