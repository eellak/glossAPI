"""Docling + RapidOCR (ONNX) pipeline for batch PDF OCR.

Provides build_pipeline() and convert_dir() mirroring the behavior of the
repro script greek_pdf_ocr.py, but self-contained inside glossapi and with
packaged ONNX models/keys. Includes robust logging and native Docling timeout.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import inspect
from pathlib import Path
from typing import Iterable, Optional, Tuple

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
from docling.document_converter import (
    ConversionResult,
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.settings import settings

from ._rapidocr_paths import resolve_packaged_onnx_and_keys
# Ensure RapidOCR factory is registered (avoids masked errors in older paths)
import docling.models.rapid_ocr_model  # noqa: F401


log = logging.getLogger(__name__)


def _available_ort_providers() -> str:
    try:
        import onnxruntime as ort  # type: ignore
        return ",".join(ort.get_available_providers())
    except Exception as e:
        return f"unavailable: {e}"


def _supports_native_timeout(converter: DocumentConverter) -> Optional[str]:
    try:
        sig = inspect.signature(converter.convert)
        for name in ("timeout", "timeout_s"):
            if name in sig.parameters:
                return name
    except Exception:
        pass
    return None


def _convert_with_timeout(converter: DocumentConverter, *, source: str, raises_on_error: bool, timeout_s: Optional[int] = None, **kwargs):
    kw = dict(raises_on_error=raises_on_error)
    kw.update(kwargs)
    if timeout_s is not None:
        tkw = _supports_native_timeout(converter)
        if tkw:
            kw[tkw] = int(timeout_s)
    return converter.convert(source=source, **kw)


def _convert_all_with_timeout(converter: DocumentConverter, *, sources: Iterable[str], raises_on_error: bool, timeout_s: Optional[int] = None, **kwargs):
    kw = dict(raises_on_error=raises_on_error)
    kw.update(kwargs)
    if timeout_s is not None:
        tkw = _supports_native_timeout(converter)
        if tkw:
            kw[tkw] = int(timeout_s)
    return list(converter.convert_all(sources, **kw))


def build_pipeline(
    *,
    device: str = "cuda:0",
    text_score: float = 0.45,
    images_scale: float = 1.25,
    formula_enrichment: bool = False,
    code_enrichment: bool = False,
) -> Tuple[object, PdfPipelineOptions]:
    acc = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if device.lower().startswith("cuda") else AcceleratorDevice.CPU
    )

    r = resolve_packaged_onnx_and_keys()
    if not (r.det and r.rec and r.cls and r.keys):
        raise FileNotFoundError(
            "Packaged RapidOCR ONNX models/keys not found under glossapi.models."
        )

    ocr_opts = RapidOcrOptions(
        backend="onnxruntime",
        lang=["el", "en"],
        force_full_page_ocr=False,
        use_det=True,
        use_cls=True,
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

    # images_scale attribute (best-effort attachment)
    try:
        setattr(opts, "images_scale", images_scale)
    except Exception:
        pass

    # Prefer explicit injection of RapidOCR to avoid Rec.keys_path mapping issues
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel  # type: ignore
        try:
            from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        except Exception:
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        # Docling 2.48.0 RapidOcrModel signature: (enabled, artifacts_path, options, accelerator_options)
        ocr_model = RapidOcrModel(True, None, ocr_opts, acc)  # type: ignore[arg-type]
        pipeline = StandardPdfPipeline(opts, ocr_model=ocr_model)  # type: ignore
        return pipeline, opts
    except Exception as e:
        # Fallback: use DocumentConverter factory (keys mapping may require patched docling)
        log.warning("Explicit RapidOCR injection failed; using factory path: %s", e)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        return converter, opts


def convert_dir(
    input_dir: Path,
    output_dir: Path,
    *,
    device: str = "cuda:0",
    text_score: float = 0.45,
    images_scale: float = 1.25,
    formula_enrichment: bool = False,
    code_enrichment: bool = False,
    normalize_output: bool = True,
    timeout_s: Optional[int] = 600,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: tune CodeFormula batch size and math precision when enrichment is requested
    if formula_enrichment:
        try:
            import torch  # type: ignore
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        except Exception:
            pass

    engine, opts = build_pipeline(
        device=device,
        text_score=text_score,
        images_scale=images_scale,
        formula_enrichment=formula_enrichment,
        code_enrichment=code_enrichment,
    )

    # Logging block
    log.info("Docling+RapidOCR pipeline ready")
    log.info("device=%s text_score=%.2f images_scale=%.2f formula=%s code=%s", device, text_score, images_scale, formula_enrichment, code_enrichment)
    log.info("ORT providers: %s", _available_ort_providers())
    log.info("Caches: HF_HOME=%s XDG_CACHE_HOME=%s DOCLING_CACHE_DIR=%s", os.getenv("HF_HOME"), os.getenv("XDG_CACHE_HOME"), os.getenv("DOCLING_CACHE_DIR"))
    try:
        r = resolve_packaged_onnx_and_keys()
        import os as _os
        log.info(
            "Models: det=%s rec=%s cls=%s keys=%s",
            _os.path.basename(r.det) if r.det else None,
            _os.path.basename(r.rec) if r.rec else None,
            _os.path.basename(r.cls) if r.cls else None,
            _os.path.basename(r.keys) if r.keys else None,
        )
    except Exception:
        pass

    # Collect PDFs
    pdfs = sorted(str(p) for p in input_dir.rglob("*.pdf") if p.is_file())
    if not pdfs:
        log.warning("No PDFs under %s", input_dir)
        return

    # Enable timing profile
    try:
        settings.debug.profile_pipeline_timings = True
    except Exception:
        pass

    total_start = time.time()
    # If we got a StandardPdfPipeline, it has a .convert method similar in spirit
    # to DocumentConverter.convert; detect native timeout support by signature.
    def _native_timeout_kw(obj) -> Optional[str]:
        try:
            sig = inspect.signature(obj.convert)
            for name in ("timeout", "timeout_s"):
                if name in sig.parameters:
                    return name
        except Exception:
            return None
        return None

    tkw = _native_timeout_kw(engine)
    for src in pdfs:
        try:
            kwargs = {}
            if tkw and timeout_s is not None:
                kwargs[tkw] = int(timeout_s)
            conv = engine.convert(source=src, **kwargs)  # type: ignore
            _export(conv, output_dir, normalize_output=normalize_output)
            log.info("[OK] %s", src)
        except Exception as e:
            log.error("[FAIL] %s: %s", src, e)
    log.info("Done in %.2fs", time.time() - total_start)


def _normalize_text(s: str) -> str:
    import unicodedata, re
    zw = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
    s = unicodedata.normalize("NFC", s)
    return zw.sub("", s)


def _normalize_obj(o):
    if isinstance(o, str):
        return _normalize_text(o)
    if isinstance(o, list):
        return [_normalize_obj(x) for x in o]
    if isinstance(o, dict):
        return {k: _normalize_obj(v) for k, v in o.items()}
    return o


def _export(conv: ConversionResult, out_dir: Path, *, normalize_output: bool) -> None:
    doc = conv.document
    p = Path(conv.input.file)
    md_path = out_dir / f"{p.stem}.md"
    json_path = out_dir / f"{p.stem}.json"
    metrics_path = out_dir / f"{p.stem}.metrics.json"

    md = doc.export_to_markdown()
    dd = doc.export_to_dict()
    if normalize_output:
        md = _normalize_text(md)
        dd = _normalize_obj(dd)
    md_path.write_text(md, encoding="utf-8")
    import json as _json
    json_path.write_text(_json.dumps(dd, ensure_ascii=False, indent=2), encoding="utf-8")

    # Timings if present
    try:
        from typing import Any, Dict, List
        def _q(vals: list[float], q: float) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            i = int(round((len(s) - 1) * q))
            return float(s[i])
        metrics: Dict[str, Any] = {"file": str(p), "timings": {}}
        for key, item in conv.timings.items():
            times = list(item.times)
            cnt = int(item.count)
            tot = float(sum(times)) if times else 0.0
            avg = float(tot / cnt) if cnt else 0.0
            metrics["timings"][key] = {
                "scope": str(item.scope.value) if hasattr(item, "scope") else "unknown",
                "count": cnt,
                "total_sec": tot,
                "avg_sec": avg,
                "p50_sec": _q(times, 0.50),
                "p90_sec": _q(times, 0.90),
            }
        import json as _json
        metrics_path.write_text(_json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    _setup_logging()
    ap = argparse.ArgumentParser(description="Batch OCR with Docling + RapidOCR (ONNX)")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--device", default=os.getenv("GLOSSAPI_DOCLING_DEVICE", "cuda:0"))
    ap.add_argument("--text-score", type=float, default=float(os.getenv("GLOSSAPI_TEXT_SCORE", "0.45")))
    ap.add_argument("--images-scale", type=float, default=float(os.getenv("GLOSSAPI_IMAGES_SCALE", "1.25")))
    ap.add_argument("--docling-formula", dest="docling_formula", action="store_true", help="Enable formula enrichment (CodeFormula)")
    ap.add_argument("--no-docling-formula", dest="docling_formula", action="store_false")
    ap.set_defaults(docling_formula=False)
    ap.add_argument("--formula-batch", type=int, default=int(os.getenv("GLOSSAPI_FORMULA_BATCH", "8")), help="CodeFormula batch size (default 8)")
    ap.add_argument("--docling-code", dest="docling_code", action="store_true", help="Enable code enrichment")
    ap.add_argument("--no-docling-code", dest="docling_code", action="store_false")
    ap.set_defaults(docling_code=False)
    ap.add_argument("--normalize-output", action="store_true")
    ap.add_argument("--no-normalize-output", dest="normalize_output", action="store_false")
    ap.set_defaults(normalize_output=True)
    ap.add_argument("--timeout-s", type=int, default=int(os.getenv("GLOSSAPI_DOCLING_TIMEOUT", "600")))
    args = ap.parse_args()
    # Apply formula batch size if requested
    try:
        if getattr(args, "docling_formula", False):
            from docling.models.code_formula_model import CodeFormulaModel  # type: ignore
            if isinstance(args.formula_batch, int) and args.formula_batch > 0:
                CodeFormulaModel.elements_batch_size = int(args.formula_batch)  # type: ignore[attr-defined]
    except Exception:
        pass
    convert_dir(
        args.input_dir,
        args.output_dir,
        device=args.device,
        text_score=args["text_score"] if isinstance(args, dict) else args.text_score,
        images_scale=args.images_scale,
        formula_enrichment=args.docling_formula,
        code_enrichment=args.docling_code,
        normalize_output=args.normalize_output,
        timeout_s=args.timeout_s,
    )
