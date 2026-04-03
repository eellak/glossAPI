from __future__ import annotations

import os
from typing import Tuple

from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)

try:  # pragma: no cover - depends on installed Docling version
    from docling.datamodel.pipeline_options import LayoutOptions
except ImportError:  # pragma: no cover - older Docling versions
    LayoutOptions = None

try:  # pragma: no cover - depends on installed Docling version
    from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
except ImportError:  # pragma: no cover - older Docling versions
    PictureDescriptionApiOptions = None

try:  # pragma: no cover - depends on installed Docling version
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
except ImportError:  # pragma: no cover - older Docling versions
    ThreadedPdfPipelineOptions = None

try:  # pragma: no cover - depends on installed Docling version
    from docling.datamodel.settings import settings as docling_settings
except ImportError:  # pragma: no cover - older Docling versions
    docling_settings = None


def _resolve_accelerator(device: str | None) -> Tuple[AcceleratorOptions, bool]:
    """Return accelerator options and whether CUDA was requested."""
    dev = device or "cuda:0"
    if isinstance(dev, str) and dev.lower().startswith(("cuda", "mps", "cpu")):
        try:
            acc = AcceleratorOptions(device=dev)
        except Exception:
            acc = AcceleratorOptions(device=dev.split(":", 1)[0])
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
    def _supports_kwarg(model_cls, field_name: str) -> bool:
        fields = getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", None)
        if fields is None:
            return True
        return field_name in fields

    options_cls = ThreadedPdfPipelineOptions or PdfPipelineOptions
    table_opts = TableStructureOptions(mode=TableFormerMode.ACCURATE)
    try:
        if hasattr(table_opts, "do_cell_matching"):
            table_opts.do_cell_matching = True
    except Exception:
        pass

    option_kwargs = {
        "accelerator_options": acc,
        "do_ocr": False,
        "do_table_structure": True,
        "do_formula_enrichment": bool(formula_enrichment),
        "do_code_enrichment": bool(code_enrichment),
        "force_backend_text": False,
        "generate_parsed_pages": False,
        "allow_external_plugins": True,
    }
    if LayoutOptions is not None and _supports_kwarg(options_cls, "layout_options"):
        option_kwargs["layout_options"] = LayoutOptions()
    if _supports_kwarg(options_cls, "table_structure_options"):
        option_kwargs["table_structure_options"] = table_opts
    opts = options_cls(**{key: value for key, value in option_kwargs.items() if _supports_kwarg(options_cls, key)})
    try:
        if hasattr(opts, "do_picture_description"):
            opts.do_picture_description = False
        if PictureDescriptionApiOptions is not None and getattr(opts, "picture_description_options", None) is None:
            opts.picture_description_options = PictureDescriptionApiOptions()
        if hasattr(opts, "enable_remote_services"):
            opts.enable_remote_services = False
    except Exception:
        pass
    try:
        setattr(opts, "images_scale", images_scale)
    except Exception:
        pass
    _apply_runtime_overrides(opts)
    return opts


def _apply_runtime_overrides(opts: PdfPipelineOptions) -> None:
    """Apply optional runtime tuning knobs exposed by newer Docling releases."""

    int_env_map = {
        "GLOSSAPI_DOCLING_LAYOUT_BATCH_SIZE": "layout_batch_size",
        "GLOSSAPI_DOCLING_TABLE_BATCH_SIZE": "table_batch_size",
        "GLOSSAPI_DOCLING_OCR_BATCH_SIZE": "ocr_batch_size",
        "GLOSSAPI_DOCLING_QUEUE_MAX_SIZE": "queue_max_size",
        "GLOSSAPI_DOCLING_DOCUMENT_TIMEOUT": "document_timeout",
    }
    float_env_map = {
        "GLOSSAPI_DOCLING_BATCH_POLL_INTERVAL": "batch_polling_interval_seconds",
    }

    for env_name, attr_name in int_env_map.items():
        raw = os.getenv(env_name)
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or not hasattr(opts, attr_name):
            continue
        try:
            setattr(opts, attr_name, value)
        except Exception:
            pass

    for env_name, attr_name in float_env_map.items():
        raw = os.getenv(env_name)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value <= 0 or not hasattr(opts, attr_name):
            continue
        try:
            setattr(opts, attr_name, value)
        except Exception:
            pass

    raw_page_batch_size = os.getenv("GLOSSAPI_DOCLING_PAGE_BATCH_SIZE")
    if raw_page_batch_size and docling_settings is not None:
        try:
            page_batch_size = int(raw_page_batch_size)
        except ValueError:
            page_batch_size = 0
        if page_batch_size > 0:
            try:
                perf_settings = getattr(docling_settings, "perf", None)
                if perf_settings is not None and hasattr(perf_settings, "page_batch_size"):
                    setattr(perf_settings, "page_batch_size", page_batch_size)
            except Exception:
                pass


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
