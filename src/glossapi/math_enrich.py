from __future__ import annotations

import json
import time
from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pypdfium2 as pdfium  # type: ignore
except Exception as e:  # pragma: no cover
    pdfium = None  # type: ignore

from docling_core.types.doc import DocItemLabel  # type: ignore


@dataclass
class RoiEntry:
    page_no: int  # 1-based
    bbox: Any     # Docling BoundingBox
    label: str
    per_page_ix: int


class PageRasterCache:
    def __init__(self, pdf_path: Path, max_cached_pages: int = 4):
        if pdfium is None:
            raise RuntimeError("pypdfium2 not available; cannot rasterize PDF")
        self.doc = pdfium.PdfDocument(str(pdf_path))
        self.cache: dict[Tuple[int, int], Any] = {}
        self.max_cached = int(max_cached_pages)

    def get_pil(self, page_no_1b: int, dpi: int):
        key = (int(page_no_1b), int(dpi))
        if key in self.cache:
            return self.cache[key]
        page = self.doc[int(page_no_1b) - 1]
        scale = float(dpi) / 72.0
        bm = page.render(scale=scale)
        im = bm.to_pil()
        # naive LRU: cap cache size
        self.cache[key] = im
        if len(self.cache) > self.max_cached:
            # pop arbitrary oldest key
            self.cache.pop(next(iter(self.cache)))
        return im


def _dpi_for_bbox(px_h: float, *, base: int = 220, lo: int = 180, hi: int = 320) -> int:
    # Simple heuristic: small crops get higher dpi, big crops get lower
    if px_h < 40:
        return int(min(hi, max(base + 60, lo)))
    if px_h < 120:
        return int(min(hi, max(base + 20, lo)))
    if px_h > 800:
        return int(max(lo, base - 40))
    return int(base)


def _crop_box_pixels(bbox, pil_h: int, dpi: int) -> Tuple[int, int, int, int]:
    # Convert Docling bbox to top-left pixel coords using the page pixel height
    try:
        b2 = bbox.scaled(scale=float(dpi) / 72.0).to_top_left_origin(page_height=float(pil_h))
        l, t, r, b = map(int, (b2.l, b2.t, b2.r, b2.b))
        return max(0, l), max(0, t), max(0, r), max(0, b)
    except Exception:
        # fallback: assume bbox already pixel with top-left origin
        l, t, r, b = map(int, (getattr(bbox, 'l', 0), getattr(bbox, 't', 0), getattr(bbox, 'r', 0), getattr(bbox, 'b', 0)))
        return max(0, l), max(0, t), max(0, r), max(0, b)


def enrich_from_docling_json(
    json_path: Path,
    pdf_path: Path,
    out_md_path: Path,
    out_map_jsonl: Path,
    *,
    device: str = "cuda",
    batch_size: int = 8,
    pad_px: int = 3,
    dpi_base: int = 220,
    dpi_lo: int = 180,
    dpi_hi: int = 320,
    targets: Optional[List[Tuple[int, int]]] = None,
) -> dict:
    """Load a DoclingDocument JSON and enrich FORMULA/CODE items using Docling CodeFormula.

    Writes final Markdown and a latex_map.jsonl sidecar; returns a small metrics dict.
    """
    from .json_io import load_docling_json  # type: ignore
    from docling.models.code_formula_model import CodeFormulaModel, CodeFormulaModelOptions  # type: ignore
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice  # type: ignore
    from docling.datamodel.base_models import ItemAndImageEnrichmentElement  # type: ignore

    doc = load_docling_json(json_path)
    rc = PageRasterCache(pdf_path)

    # Collect ROIs (optionally filter by explicit (page_no, per_page_ix) targets)
    rois: list[RoiEntry] = []
    per_page_ix: dict[int, int] = {}
    it = getattr(doc, 'iterate_items', None)
    if not callable(it):
        raise RuntimeError("DoclingDocument missing iterate_items()")
    targets_set = set((int(p), int(ix)) for (p, ix) in (targets or []))
    for element, _level in it():
        lab = str(getattr(element, 'label', '')).lower()
        if lab not in {"formula", "code"}:
            continue
        prov = getattr(element, 'prov', []) or []
        if not prov:
            continue
        p = prov[0]
        page_no = int(getattr(p, 'page_no', 0))
        bbox = getattr(p, 'bbox', None)
        if not page_no or bbox is None:
            continue
        per_page_ix[page_no] = per_page_ix.get(page_no, 0) + 1
        cur_ix = per_page_ix[page_no]
        if targets_set and (page_no, cur_ix) not in targets_set:
            continue
        rois.append(RoiEntry(page_no=page_no, bbox=bbox, label=lab, per_page_ix=cur_ix))

    if not rois:
        # Nothing to enrich; still emit the base markdown (and fall back to raw text if empty)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_markdown(out_md_path)
        try:
            rendered = out_md_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            rendered = ""
        if not rendered.strip():
            # Some Docling docs store text only in the lightweight texts list; surface it so the
            # markdown is not empty (important for tests and downstream auditability).
            fallback: list[str] = []
            text_items = getattr(doc, "texts", None)
            if text_items:
                for item in text_items:
                    value = getattr(item, "text", None) if hasattr(item, "text") else None
                    if value:
                        fallback.append(str(value))
            if fallback:
                out_md_path.write_text("\n\n".join(fallback), encoding="utf-8")
        return {"items": 0, "accepted": 0, "time_sec": 0.0}

    # Build recognizer
    acc = AcceleratorOptions(device=device if device else AcceleratorDevice.AUTO)
    opts = CodeFormulaModelOptions(do_code_enrichment=True, do_formula_enrichment=True)
    model = CodeFormulaModel(enabled=True, artifacts_path=None, options=opts, accelerator_options=acc)
    # Optional: attach efficient early-stop guards to generation if supported
    try:
        from ._formula_earlystop import attach_early_stop  # type: ignore
        attach_early_stop(model)
    except Exception:
        pass
    # Optional batch size tuning
    try:
        setattr(model, 'elements_batch_size', int(batch_size))
    except Exception:
        pass

    # Process in batches
    t0 = time.time()
    accepted = 0
    written = 0
    # Truncate/create map file early so we can append per batch and observe progress
    out_map_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_map_jsonl.open("w", encoding="utf-8") as fp:
        fp.write("")

    batch: list[ItemAndImageEnrichmentElement] = []
    binfo: list[Tuple[int, int, int]] = []  # (page_no, per_page_ix, dpi)

    # Centralized accept heuristic
    from .text_sanitize import accept_latex, load_latex_policy, sanitize_latex, tail_run  # type: ignore

    def _sanitize_latex(text: str, *, max_len: int = 3000, max_tail_repeats: int = 50) -> tuple[str, dict]:
        """Apply lightweight, efficient guards to latex text.

        - Tail repetition gate: if the last whitespace-delimited token repeats more than
          `max_tail_repeats` times at the end, crop the run to exactly `max_tail_repeats`.
        - Length cap: prefer cutting at whitespace before `max_len`; otherwise hard cut.

        Returns the sanitized text and a small info dict with flags.
        """
        s = text or ""
        info = {
            "orig_len": len(s),
            "truncated_by_repeat": False,
            "truncated_by_len": False,
            "tail_token": "",
            "tail_run": 0,
        }

        # Tail repetition detection (O(n) split; cheap for our sizes)
        toks = s.split()
        if toks:
            last = toks[-1]
            run = 1
            i = len(toks) - 2
            while i >= 0 and toks[i] == last:
                run += 1
                i -= 1
            info["tail_token"] = last
            info["tail_run"] = run
            if run > int(max_tail_repeats):
                keep = int(max_tail_repeats)
                # i currently points to the token just before the repeated run
                new_toks = toks[: (i + 1 + keep)]
                s = " ".join(new_toks)
                info["truncated_by_repeat"] = True

        # Length cap (prefer whitespace boundary)
        if len(s) > int(max_len):
            cut_ws = max(s.rfind(" ", 0, int(max_len)), s.rfind("\n", 0, int(max_len)), s.rfind("\t", 0, int(max_len)))
            cut = cut_ws if cut_ws != -1 else int(max_len)
            s = s[:cut].rstrip()
            info["truncated_by_len"] = True

        return s, info

    def _env_int(name: str, default: int) -> int:
        try:
            v = os.getenv(name)
            return int(v) if v is not None and str(v).strip() != "" else default
        except Exception:
            return default

    # Post-processing policy: apply only on failed cases by default
    policy = load_latex_policy()

    def _tail_run(s: str) -> int:
        toks = s.split()
        if not toks:
            return 0
        last = toks[-1]
        run = 1
        i = len(toks) - 2
        while i >= 0 and toks[i] == last:
            run += 1
            i -= 1
        return run

    def flush_batch():
        nonlocal accepted, batch, binfo, written
        if not batch:
            return
        t_call = time.time()
        for out_item in model(doc, batch):  # yields NodeItem with .text set
            pass  # items are mutated in-place
        dt_ms = int((time.time() - t_call) * 1000.0)
        per_ms = int(dt_ms / max(1, len(batch)))
        # Write map entries with naive acceptance (present & non-empty)
        out_lines: list[str] = []
        # engine info (best-effort)
        _engine = "code_formula"
        try:
            import docling as _dl  # type: ignore
            _engine_ver = getattr(_dl, "__version__", "") or ""
        except Exception:
            _engine_ver = ""
        for (page_no, ix, dpi_used), el in zip(binfo, batch):
            latex = getattr(el.item, 'text', '') or ''
            # Centralized post-processing policy
            do_post = True
            if policy.post_only_failed:
                tr = 0
                try:
                    tr = tail_run(latex)
                except Exception:
                    tr = 0
                do_post = (tr > policy.post_repeat_gate) or (len(latex) > policy.post_max_chars)
            sinfo = {"orig_len": len(latex), "truncated_by_repeat": False, "truncated_by_len": False, "tail_token": "", "tail_run": 0}
            if do_post:
                sanitized, sinfo = sanitize_latex(latex, policy)
                if sanitized != latex:
                    try:
                        setattr(el.item, 'text', sanitized)
                    except Exception:
                        pass
                latex = sanitized
            ok = (accept_latex(latex) >= 1.0)
            if ok:
                accepted += 1
            row = {
                "page_no": int(page_no),
                "item_index": int(ix),
                "latex": latex,
                "accept_score": 1.0 if ok else 0.0,
                "compile_ok": False,
                "dpi": int(dpi_used),
                # Minimal metrics for observability (non-breaking additions)
                "orig_len": int(sinfo.get("orig_len", len(latex))),
                "truncated_by_repeat": bool(sinfo.get("truncated_by_repeat", False)),
                "truncated_by_len": bool(sinfo.get("truncated_by_len", False)),
                "tail_token": str(sinfo.get("tail_token", "")),
                "tail_run": int(sinfo.get("tail_run", 0)),
                "post_applied": bool(do_post),
                "engine": _engine,
                "engine_version": _engine_ver,
                "time_ms": int(per_ms),
            }
            out_lines.append(json.dumps(row, ensure_ascii=False))
        # Append to file to expose progress
        if out_lines:
            with out_map_jsonl.open("a", encoding="utf-8") as fp:
                fp.write("\n".join(out_lines) + "\n")
        # Progress log
        nonlocal written
        written += len(out_lines)
        if written % max(1, batch_size) == 0:
            print(f"[Phase-2] Wrote {written}/{len(rois)} items for current doc…")
        batch = []
        binfo = []

    # Prepare items
    print(f"[Phase-2] {json_path.stem}: {len(rois)} items to enrich …")
    for entry in rois:
        # Raster & crop
        # First render to estimate crop height for DPI choice
        # Use base DPI for initial render then compute precise crop box
        dpi = int(dpi_base)
        im = rc.get_pil(entry.page_no, dpi)
        l, t, r, b = _crop_box_pixels(entry.bbox, pil_h=im.height, dpi=dpi)
        # Adjust DPI based on crop height
        dpi = _dpi_for_bbox(b - t, base=dpi_base, lo=dpi_lo, hi=dpi_hi)
        im = rc.get_pil(entry.page_no, dpi)
        l, t, r, b = _crop_box_pixels(entry.bbox, pil_h=im.height, dpi=dpi)
        # Pad
        l = max(0, l - int(pad_px)); t = max(0, t - int(pad_px))
        r = min(im.width, r + int(pad_px)); b = min(im.height, b + int(pad_px))
        crop = im.crop((l, t, r, b))
        batch.append(ItemAndImageEnrichmentElement(item=getattr(entry, 'item', None) or _find_item(doc, entry), image=crop))
        binfo.append((entry.page_no, entry.per_page_ix, int(dpi)))
        if len(batch) >= int(batch_size):
            flush_batch()
    flush_batch()

    # Export final Markdown
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save_as_markdown(out_md_path)

    # latex_map has been appended per batch already

    return {"items": len(rois), "accepted": accepted, "time_sec": time.time() - t0}


def _find_item(doc, entry: RoiEntry):
    # Find the actual NodeItem instance matching page_no + per_page_ix for FORMULA/CODE
    c = 0
    for element, _level in doc.iterate_items():  # type: ignore[attr-defined]
        lab = str(getattr(element, 'label', '')).lower()
        if lab != entry.label:
            continue
        prov = getattr(element, 'prov', []) or []
        if not prov:
            continue
        p = prov[0]
        pn = int(getattr(p, 'page_no', 0))
        if pn != entry.page_no:
            continue
        c += 1
        if c == entry.per_page_ix:
            return element
    return element  # type: ignore  # last seen element of same type as fallback
