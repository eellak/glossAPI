from __future__ import annotations

from typing import Any, Dict
try:
    from .text_sanitize import load_latex_policy, sanitize_latex  # type: ignore
except Exception:  # pragma: no cover
    load_latex_policy = None
    sanitize_latex = None


def compute_per_page_metrics(conv) -> Dict[str, Any]:
    """Compute per-page OCR/parse timings and formula/code counts from a ConversionResult.

    Returns a dict with keys: file, page_count, totals{doc_enrich_total_sec}, pages[...].
    Mirrors the previous implementation used in gloss_extract and CLI, centralized here
    to avoid duplication.
    """
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
    fcnt = [0] * max(1, page_count)
    fch = [0] * max(1, page_count)
    ftr = [0] * max(1, page_count)
    ftrc = [0] * max(1, page_count)
    ccnt = [0] * max(1, page_count)
    try:
        as_dict = doc.export_to_dict()
        # Use centralized sanitizer for formula text to keep metrics in sync with post-processing
        policy = load_latex_policy() if callable(load_latex_policy) else None
        def _walk(label, cnt, chars=False):
            for node in as_dict.get("texts", []):
                if str(node.get("label")) != label:
                    continue
                raw = str(node.get("text") or node.get("orig") or "")
                txt = raw
                dropped = 0
                if label == "formula" and callable(sanitize_latex):
                    sanitized, sinfo = sanitize_latex(raw, policy)
                    dropped = max(0, len(raw) - len(sanitized))
                    txt = sanitized
                ch = len(txt)
                for prov in node.get("prov", []) or []:
                    pno = int(prov.get("page_no") or 0)
                    if 1 <= pno <= len(cnt):
                        cnt[pno - 1] += 1
                        if chars:
                            fch[pno - 1] += ch
                        if label == 'formula' and dropped:
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
        rows.append(
            {
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
            }
        )
    file_name = str(getattr(conv.input.file, 'name', 'unknown'))
    return {
        "file": file_name,
        "page_count": int(page_count),
        "totals": {"doc_enrich_total_sec": den_total},
        "pages": rows,
    }


__all__ = ["compute_per_page_metrics"]
