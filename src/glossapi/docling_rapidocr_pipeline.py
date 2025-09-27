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
import importlib
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
from .metrics import compute_per_page_metrics
# Ensure RapidOCR factory is registered (avoids masked errors in older paths)
import docling.models.rapid_ocr_model  # noqa: F401


log = logging.getLogger(__name__)


def _maybe_import_torch(*, force: bool = False):
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        return torch_mod
    flag = str(os.environ.get("GLOSSAPI_IMPORT_TORCH", "0")).strip().lower()
    if force or flag in {"1", "true", "yes"}:
        try:
            return importlib.import_module("torch")  # type: ignore
        except Exception:
            return None
    return None


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
    # Delegate to canonical pipeline builder to avoid duplication
    try:
        from ._pipeline import build_rapidocr_pipeline  # type: ignore
    except Exception as _e:  # pragma: no cover
        # Backward-compat fallback: inline builder (kept minimal to satisfy tests)
        from docling.datamodel.pipeline_options import AcceleratorOptions, TableStructureOptions, TableFormerMode, LayoutOptions, PdfPipelineOptions, RapidOcrOptions  # type: ignore
        dev = device or "cuda:0"
        acc = AcceleratorOptions(device=dev)
        r = resolve_packaged_onnx_and_keys()
        if not (r.det and r.rec and r.cls and r.keys):
            raise FileNotFoundError("Packaged RapidOCR ONNX models/keys not found under glossapi.models.")
        ocr_opts = RapidOcrOptions(
            backend="onnxruntime", lang=["el", "en"], force_full_page_ocr=False,
            use_det=True, use_cls=False, use_rec=True, text_score=text_score,
            det_model_path=r.det, rec_model_path=r.rec, cls_model_path=r.cls, print_verbose=False,
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
        from docling.document_converter import DocumentConverter, PdfFormatOption  # type: ignore
        from docling.datamodel.base_models import InputFormat  # type: ignore
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}), opts
    return build_rapidocr_pipeline(
        device=device,
        text_score=text_score,
        images_scale=images_scale,
        formula_enrichment=formula_enrichment,
        code_enrichment=code_enrichment,
    )


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

    # Device-aware preflight: only enforce CUDA provider when device requests CUDA
    want_cuda = isinstance(device, str) and device.lower().startswith("cuda")
    if want_cuda:
        try:
            import onnxruntime as _ort  # type: ignore
            _providers = _ort.get_available_providers()
            if "CUDAExecutionProvider" not in _providers:
                raise RuntimeError(f"CUDAExecutionProvider not available in onnxruntime providers={_providers}")
        except Exception as e:
            raise RuntimeError(f"onnxruntime-gpu not available or misconfigured: {e}")
    if formula_enrichment and want_cuda:
        try:
            torch_mod = _maybe_import_torch(force=True)
            if torch_mod is None or not torch_mod.cuda.is_available():
                raise RuntimeError("Torch CUDA not available but formula enrichment requested.")
        except Exception as e:
            raise RuntimeError(f"Torch CUDA preflight failed: {e}")

    # Optional: tune CodeFormula batch size and math precision when enrichment is requested
    if formula_enrichment:
        try:
            torch_mod = _maybe_import_torch()
            if torch_mod is not None and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available():
                try:
                    torch_mod.set_float32_matmul_precision("high")
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
            # Per-page metrics and per-page console logs
            try:
                per_page = compute_per_page_metrics(conv)
                # Harmonize with GlossExtract: write to sibling json/metrics/
                metrics_dir = output_dir.parent / "json" / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                pp = metrics_dir / f"{Path(src).stem}.per_page.metrics.json"
                import json as _json
                pp.write_text(_json.dumps(per_page, ensure_ascii=False, indent=2), encoding="utf-8")
                for row in per_page.get("pages", []):
                    log.info("[PAGE] %s p%d: parse=%.3fs ocr=%.3fs formulas=%d code=%d",
                             Path(src).name,
                             int(row.get("page_no", 0)),
                             float(row.get("parse_sec", 0.0)),
                             float(row.get("ocr_sec", 0.0)),
                             int(row.get("formula_count", 0)),
                             int(row.get("code_count", 0)))
            except Exception as _e:
                log.warning("Failed to compute per-page metrics for %s: %s", src, _e)
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
    # Write Docling JSON under sibling json/ directory (no JSON in markdown dir)
    json_dir = out_dir.parent / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{p.stem}.docling.json"
    # Harmonize metrics location with GlossExtract: sibling json/metrics/
    metrics_dir = out_dir.parent / "json" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{p.stem}.metrics.json"

    md = doc.export_to_markdown()
    if normalize_output:
        md = _normalize_text(md)
    md_path.write_text(md, encoding="utf-8")
    # Export DoclingDocument JSON via helper (compressed by default)
    try:
        from .json_io import export_docling_json  # type: ignore
        # Attach minimal meta for provenance
        meta = {"source_pdf_relpath": str(p)}
        export_docling_json(doc, json_path, compress="zstd", meta=meta)  # type: ignore[arg-type]
    except Exception:
        # Fallback: write plain JSON under json/ without compression
        try:
            import json as _json
            dd = doc.export_to_dict()
            if normalize_output:
                dd = _normalize_obj(dd)
            json_path.write_text(_json.dumps(dd, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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


def _compute_per_page_metrics(conv: ConversionResult):
    try:
        doc = conv.document
    except Exception:
        return {"pages": []}
    try:
        page_count = len(doc.pages)  # type: ignore[attr-defined]
    except Exception:
        page_count = 0
    timings = {}
    try:
        for key, item in conv.timings.items():
            times = list(item.times)
            timings[key] = {
                "scope": str(getattr(getattr(item, 'scope', None), 'value', 'unknown')),
                "times": times,
                "total": float(sum(times)) if times else float(getattr(item, 'total', 0.0)),
            }
    except Exception:
        pass
    def _pt(k):
        arr = timings.get(k, {}).get("times", []) or []
        if page_count and len(arr) == page_count:
            return [float(x) for x in arr]
        return [float(x) for x in (arr + [0.0] * page_count)[:page_count]]
    ocr = _pt("ocr")
    parse = _pt("page_parse")
    layout = _pt("layout")
    table = _pt("table_structure")
    # counts with sanitization and capping
    fcnt = [0] * max(1, page_count)
    fch = [0] * max(1, page_count)
    ftr = [0] * max(1, page_count)
    ftrc = [0] * max(1, page_count)
    ccnt = [0] * max(1, page_count)
    try:
        as_dict = doc.export_to_dict()
        import re as _re
        _run_pat = _re.compile(r"\\\\\s*&(?P<ws>(?:\\quad|\\;|\\:|\\,|\\\\s|\s){200,})")
        _ws_collapse = _re.compile(r"(?:(?:\\quad|\\;|\\:|\\,|\\\\s)|\s){2,}")
        _CAP = 3000
        def _sanitize(s: str):
            dropped=0
            m=_run_pat.search(s)
            if m:
                s_new=s[:m.start('ws')]; dropped+=len(s)-len(s_new); s=s_new
            if len(s)>_CAP:
                cut=s.rfind('\\\\',0,_CAP); cut = cut if cut>=0 else _CAP; dropped+=len(s)-cut; s=s[:cut]
            s2=_ws_collapse.sub(' ', s)
            return s2, dropped
        def _walk(label, cnt, chars=False):
            for node in as_dict.get("texts", []):
                if str(node.get("label")) != label:
                    continue
                raw = str(node.get("text") or node.get("orig") or "")
                txt, dropped = _sanitize(raw) if label=='formula' else (raw,0)
                ch = len(txt)
                for prov in node.get("prov", []) or []:
                    pno = int(prov.get("page_no") or 0)
                    if 1 <= pno <= len(cnt):
                        cnt[pno - 1] += 1
                        if chars:
                            fch[pno - 1] += ch
                        if label=='formula' and dropped:
                            ftr[pno - 1] += 1
                            ftrc[pno - 1] += int(dropped)
        _walk("formula", fcnt, True)
        _walk("code", ccnt, False)
    except Exception:
        pass
    try:
        den_total = float(timings.get("doc_enrich", {}).get("total", 0.0))
    except Exception:
        den_total = 0.0
    shares = [0.0] * max(1, page_count)
    if den_total and page_count:
        s = float(sum(fch)) or float(sum(fcnt)) or 0.0
        if s > 0:
            base = fch if sum(fch) > 0 else fcnt
            shares = [den_total * (float(x) / s) for x in base]
    rows = []
    n = max(page_count, len(ocr), len(parse))
    for i in range(n):
        rows.append({
            "page_no": i + 1,
            "ocr_sec": float(ocr[i]) if i < len(ocr) else 0.0,
            "parse_sec": float(parse[i]) if i < len(parse) else 0.0,
            "layout_sec": float(layout[i]) if i < len(layout) else 0.0,
            "table_sec": float(table[i]) if i < len(table) else 0.0,
            "formula_count": int(fcnt[i]) if i < len(fcnt) else 0,
            "formula_chars": int(fch[i]) if i < len(fch) else 0,
            "formula_truncated": int(ftr[i]) if i < len(ftr) else 0,
            "formula_truncated_chars": int(ftrc[i]) if i < len(ftrc) else 0,
            "code_count": int(ccnt[i]) if i < len(ccnt) else 0,
            "doc_enrich_share_sec": float(shares[i]) if i < len(shares) else 0.0,
        })
    return {"file": str(getattr(conv.input.file, 'name', 'unknown')), "page_count": int(page_count), "totals": {"doc_enrich_total_sec": den_total}, "pages": rows}


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
