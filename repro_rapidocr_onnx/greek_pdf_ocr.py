#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch OCR for Greek academic PDFs with Docling + RapidOCR.

Intent:
- Force OCR on every page (ignore embedded PDF text)
- Use Docling layout model (GPU if available) for better structure
- Default OCR backend: RapidOCR with Paddle (PP-OCRv5 Greek)
- Optional ONNX backend (will attempt to resolve ONNX models)

Outputs: per-PDF Markdown (.md) and structured JSON (.json)
"""
from __future__ import annotations

import argparse
import json
import tempfile
import unicodedata
import re
from pathlib import Path
from typing import Iterable

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
from typing import Dict, Any, List

# Ensure RapidOCR model registers with Docling's factory before pipeline init
import docling.models.rapid_ocr_model  # force-register 'rapidocr' class


# --- Normalization helpers (helps math-heavy Unicode stability) ---
_ZW_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")  # zero-width space/joiners + BOM


def _normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = _ZW_RE.sub("", s)
    return s


def _normalize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _normalize_str(obj)
    if isinstance(obj, list):
        return [_normalize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_obj(v) for k, v in obj.items()}
    return obj


# (removed heuristic math wrapping; rely on explicit math OCR injection instead)


"""
External math OCR hook removed. Docling's CodeFormula enrichment is the only math path.
"""


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_pipeline_options(
    backend: str,
    device: str,
    text_score: float,
    hf_cache_dir: str | None,
    onnx_det: str | None = None,
    onnx_rec: str | None = None,
    onnx_cls: str | None = None,
    images_scale: float = 1.25,
    rec_keys: str | None = None,
    formula_enrichment: bool = False,
    code_enrichment: bool = False,
    force_ocr: bool = True,
) -> PdfPipelineOptions:
    acc = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if device.lower().startswith("cuda") else AcceleratorDevice.CPU
    )

    ocr_opts = RapidOcrOptions(
        backend=backend,                 # 'paddle' or 'onnxruntime'
        lang=["el", "en"],               # Greek + English
        force_full_page_ocr=bool(force_ocr),
        use_det=True,
        use_cls=True,
        use_rec=True,
        text_score=text_score,
        print_verbose=False,
    )

    if backend == "onnxruntime":
        if onnx_det and onnx_rec and onnx_cls:
            ocr_opts.det_model_path = onnx_det
            ocr_opts.rec_model_path = onnx_rec
            ocr_opts.cls_model_path = onnx_cls
            if rec_keys:
                ocr_opts.rec_keys_path = rec_keys
        else:
            try:
                from huggingface_hub import snapshot_download  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Provide --onnx-det/--onnx-rec/--onnx-cls or install huggingface_hub to auto-download."
                ) from e
            base = Path(
                snapshot_download(
                    repo_id="RapidAI/RapidOCR",
                    cache_dir=hf_cache_dir,
                    local_files_only=False,
                    allow_patterns=[
                        "onnx/PP-OCRv5/det/*",
                        "onnx/PP-OCRv5/rec/*",
                        "onnx/PP-OCRv4/cls/*",
                    ],
                )
            )
            det = base / "onnx" / "PP-OCRv5" / "det" / "ch_PP-OCRv5_server_det.onnx"
            cand = [
                base / "onnx" / "PP-OCRv5" / "rec" / "el_PP-OCRv5_rec_server_infer.onnx",
                base / "onnx" / "PP-OCRv5" / "rec" / "latin_PP-OCRv5_rec_server_infer.onnx",
                base / "onnx" / "PP-OCRv5" / "rec" / "en_PP-OCRv5_rec_server_infer.onnx",
            ]
            rec = next((c for c in cand if c.exists()), None)
            cls = base / "onnx" / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
            if not det.exists() or rec is None or not cls.exists():
                raise FileNotFoundError(
                    "Provide explicit Greek ONNX paths via --onnx-det/--onnx-rec/--onnx-cls; HF bundle may lack Greek."
                )
            ocr_opts.det_model_path = str(det)
            ocr_opts.rec_model_path = str(rec)
            ocr_opts.cls_model_path = str(cls)
            if rec_keys:
                ocr_opts.rec_keys_path = rec_keys

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

    # Nudge thin glyphs for better det/rec on math-heavy Greek
    try:
        setattr(opts, "images_scale", images_scale)
    except Exception:
        pass

    # Belt-and-suspenders in case older Docling ignores ctor flag
    try:
        setattr(opts, "allow_external_plugins", True)
    except Exception:
        pass

    return opts


def convert_pdf(converter: DocumentConverter, pdf_path: Path) -> ConversionResult:
    return converter.convert(source=str(pdf_path))


def export_results(
    conv: ConversionResult,
    out_dir: Path,
    pdf_path: Path,
    normalize_output: bool = False,
) -> None:
    doc = conv.document
    md_path = out_dir / f"{pdf_path.stem}.md"
    json_path = out_dir / f"{pdf_path.stem}.json"
    metrics_path = out_dir / f"{pdf_path.stem}.metrics.json"
    ensure_parent(md_path)
    ensure_parent(json_path)
    md = doc.export_to_markdown()
    as_dict = doc.export_to_dict()
    if normalize_output:
        md = _normalize_str(md)
    md_path.write_text(md, encoding="utf-8")
    if normalize_output:
        as_dict = _normalize_obj(as_dict)
    json_path.write_text(json.dumps(as_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # Export timings if profiling is enabled
    try:
        metrics: Dict[str, Any] = {"file": str(pdf_path), "timings": {}}
        def _quantiles(vals: List[float], q: float) -> float:
            if not vals:
                return 0.0
            s = sorted(vals)
            idx = int(round((len(s)-1) * q))
            return float(s[idx])
        for key, item in conv.timings.items():
            times = list(item.times)
            count = int(item.count)
            total = float(sum(times)) if times else 0.0
            avg = float(total / count) if count else 0.0
            metrics["timings"][key] = {
                "scope": str(item.scope.value) if hasattr(item, "scope") else "unknown",
                "count": count,
                "total_sec": total,
                "avg_sec": avg,
                "p50_sec": _quantiles(times, 0.50),
                "p90_sec": _quantiles(times, 0.90),
                "times_sec": times,
            }
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Do not fail conversion because of metrics
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch Greek OCR with Docling + RapidOCR")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--backend", default="paddle", choices=["paddle", "onnxruntime"])
    ap.add_argument("--device", default="cuda:0", help="Docling accelerator device, e.g., cuda:0 or cpu")
    ap.add_argument("--text-score", type=float, default=0.50)
    ap.add_argument("--hf-cache-dir", type=str, default=None)
    ap.add_argument("--onnx-det", type=str, default=None, help="Path to detection ONNX (inference.onnx)")
    ap.add_argument("--onnx-rec", type=str, default=None, help="Path to recognition ONNX (inference.onnx)")
    ap.add_argument("--onnx-cls", type=str, default=None, help="Path to classifier ONNX (inference.onnx)")
    ap.add_argument("--images-scale", type=float, default=1.25, help="Raster scale factor before OCR (e.g., 1.25–1.5)")
    ap.add_argument("--rec-keys", type=str, default=None, help="Path to recognition keys dict (ppocr keys)")
    ap.add_argument("--force-ocr", dest="force_ocr", action="store_true", help="Force full-page OCR (ignore embedded text)")
    ap.add_argument("--no-force-ocr", dest="force_ocr", action="store_false", help="Let Docling decide when to OCR (use embedded text when available)")
    ap.set_defaults(force_ocr=True)
    ap.add_argument("--docling-formula", dest="docling_formula", action="store_true", help="Enable Docling formula enrichment (CodeFormula)")
    ap.add_argument("--no-docling-formula", dest="docling_formula", action="store_false")
    ap.set_defaults(docling_formula=False)
    ap.add_argument("--formula-batch", type=int, default=5, help="Docling CodeFormula batch size (default 5)")
    ap.add_argument("--docling-code", dest="docling_code", action="store_true", help="Enable Docling code enrichment (shares model with formula)")
    ap.add_argument("--no-docling-code", dest="docling_code", action="store_false")
    ap.set_defaults(docling_code=False)
    # Output normalization (helps math-heavy pages by stabilizing Unicode forms)
    ap.add_argument("--normalize-output", dest="normalize_output", action="store_true")
    ap.add_argument("--no-normalize-output", dest="normalize_output", action="store_false")
    ap.set_defaults(normalize_output=True)
    # (external math OCR removed; rely on Docling formula enrichment only)
    args = ap.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Enable per-stage timing collection so we can export metrics
    try:
        settings.debug.profile_pipeline_timings = True
    except Exception:
        pass

    # Optional: tune CodeFormula batch size and torch matmul precision
    if args.docling_formula:
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                try:
                    torch.set_float32_matmul_precision('high')
                except Exception:
                    pass
        except Exception:
            pass
        try:
            from docling.models.code_formula_model import CodeFormulaModel  # type: ignore
            if isinstance(args.formula_batch, int) and args.formula_batch > 0:
                CodeFormulaModel.elements_batch_size = int(args.formula_batch)  # type: ignore[attr-defined]
        except Exception:
            pass

    opts = make_pipeline_options(
        backend=args.backend,
        device=args.device,
        text_score=args.text_score,
        hf_cache_dir=args.hf_cache_dir,
        onnx_det=args.onnx_det,
        onnx_rec=args.onnx_rec,
        onnx_cls=args.onnx_cls,
        images_scale=args.images_scale,
        rec_keys=args.rec_keys,
        formula_enrichment=args.docling_formula,
        code_enrichment=args.docling_code,
        force_ocr=args.force_ocr,
    )

    # If ONNX backend with explicit ONNX paths is provided and PdfPipeline is available,
    # build RapidOCR explicitly and inject into the pipeline to avoid factory name quirks.
    try:
        from docling.models.rapid_ocr_model import RapidOcrModel  # type: ignore
        # Prefer the standard pipeline class that supports OCR injection
        try:
            from docling.pipelines.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
        except Exception:
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline  # type: ignore
    except Exception:
        RapidOcrModel = None  # type: ignore
        StandardPdfPipeline = None  # type: ignore

    if (
        args.backend == "onnxruntime"
        and args.onnx_det and args.onnx_rec and args.onnx_cls
        and RapidOcrModel is not None and 'StandardPdfPipeline' in globals() and StandardPdfPipeline is not None
    ):
        try:
            ocr_opts = RapidOcrOptions(
                backend="onnxruntime",
                lang=["el", "en"],
                force_full_page_ocr=True,
                use_det=True,
                use_cls=True,
                use_rec=True,
                text_score=args.text_score,
                det_model_path=args.onnx_det,
                rec_model_path=args.onnx_rec,
                cls_model_path=args.onnx_cls,
                print_verbose=False,
            )
            if args.rec_keys:
                # Ensure Greek keys are respected in the explicit injection path
                ocr_opts.rec_keys_path = args.rec_keys
            acc = opts.accelerator_options
            ocr_model = RapidOcrModel(ocr_opts, acc)  # type: ignore[arg-type]
            pipeline = StandardPdfPipeline(opts, ocr_model=ocr_model)  # type: ignore

            found = False
            for pdf in iter_pdfs(args.input_dir):
                found = True
                try:
                    conv = pipeline.convert(source=str(pdf))
                    export_results(
                        conv,
                        args.output_dir,
                        pdf,
                        normalize_output=args.normalize_output,
                    )
                    print(f"[OK] {pdf}")
                except Exception as e:
                    print(f"[FAIL] {pdf}: {e}")
            if not found:
                raise SystemExit(f"No PDFs found under {args.input_dir}")
            return
        except Exception as e:
            print(f"[WARN] Explicit RapidOCR injection failed, falling back to factory: {e}")

    # Factory path (works for Paddle and for ONNX when registration is OK)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    found = False
    for pdf in iter_pdfs(args.input_dir):
        found = True
        try:
            conv = convert_pdf(converter, pdf)
            export_results(
                conv,
                args.output_dir,
                pdf,
                normalize_output=args.normalize_output,
            )
            print(f"[OK] {pdf}")
        except Exception as e:
            print(f"[FAIL] {pdf}: {e}")
    if not found:
        raise SystemExit(f"No PDFs found under {args.input_dir}")


if __name__ == "__main__":
    main()
