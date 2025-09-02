from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import importlib
import os


@dataclass
class ResolvedOnnx:
    det: Optional[str]
    rec: Optional[str]
    cls: Optional[str]
    keys: Optional[str]


def _find_first(base: Path, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        for p in base.rglob(pat):
            if p.is_file():
                return str(p)
    return None


def _resolve_packaged_cls_fallback() -> Optional[str]:
    try:
        rapidocr = importlib.import_module("rapidocr")
        base = Path(rapidocr.__file__).resolve().parent / "models"
        pref = base / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        if pref.exists():
            return str(pref)
        return _find_first(base, ["*cls*infer*.onnx", "*cls*.onnx"])
    except Exception:
        return None


def resolve_packaged_onnx_and_keys() -> ResolvedOnnx:
    """Locate ONNX det/rec/cls and Greek keys packaged with the glossapi package.

    Search order:
    1) GLOSSAPI_RAPIDOCR_ONNX_DIR (env var) with heuristic file names
    2) Under the installed glossapi package folder `models/` and common subfolders
    3) CLS only: fallback to RapidOCR’s bundled cls model if missing
    """
    # 1) Explicit override directory
    override = os.getenv("GLOSSAPI_RAPIDOCR_ONNX_DIR")
    if override:
        base = Path(override)
        det = _find_first(base, [
            "**/det/**/inference.onnx",
            "*det*server*onnx",
            "*PP*det*.onnx",
            "det*.onnx",
        ])
        rec = _find_first(base, [
            "**/rec/**/inference.onnx",
            "*el*rec*onnx",
            "*greek*rec*onnx",
            "*PP*rec*.onnx",
            "rec*.onnx",
        ])
        cls = _find_first(base, ["*cls*infer*.onnx", "*cls*.onnx"])
        keys = _find_first(base, ["*greek*keys*.txt", "*ppocr*keys*.txt", "*keys*.txt"])
        if det or rec or cls or keys:
            return ResolvedOnnx(det, rec, cls, keys)

    # 2) Search inside installed glossapi package
    try:
        glossapi = importlib.import_module("glossapi")
        pkg_root = Path(glossapi.__file__).resolve().parent
        # Candidate asset directories inside the package
        candidates = [
            pkg_root / "models",
            pkg_root / "models" / "rapidocr",
            pkg_root / "models" / "rapidocr" / "onnx",
            pkg_root / "models" / "rapidocr" / "keys",
            pkg_root / "resources",
            pkg_root / "assets",
            pkg_root / "data",
        ]
        det = rec = cls = keys = None
        for base in candidates:
            if not base.exists():
                continue
            det = det or _find_first(base, [
                "**/det/**/inference.onnx",
                "*det*server*onnx",
                "*PP*det*.onnx",
                "det*.onnx",
            ])
            rec = rec or _find_first(base, [
                "**/rec/**/inference.onnx",
                "*el*rec*onnx",
                "*greek*rec*onnx",
                "*PP*rec*.onnx",
                "rec*.onnx",
            ])
            cls = cls or _find_first(base, ["*cls*infer*.onnx", "*cls*.onnx"])
            keys = keys or _find_first(base, ["*greek*keys*.txt", "*ppocr*keys*.txt", "*keys*.txt"])

        if cls is None:
            cls = _resolve_packaged_cls_fallback()
        return ResolvedOnnx(det, rec, cls, keys)
    except Exception:
        return ResolvedOnnx(None, None, _resolve_packaged_cls_fallback(), None)


def summarize_resolution() -> Tuple[bool, str]:
    r = resolve_packaged_onnx_and_keys()
    ok = bool(r.det and r.rec and r.cls and r.keys)
    msg = f"det={bool(r.det)} rec={bool(r.rec)} cls={bool(r.cls)} keys={bool(r.keys)}"
    return ok, msg

