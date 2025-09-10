from __future__ import annotations

import io
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

try:
    import zstandard as zstd  # type: ignore
except Exception:  # pragma: no cover
    zstd = None  # type: ignore

try:
    from docling_core.types.doc.document import DoclingDocument  # type: ignore
except Exception:
    DoclingDocument = None  # type: ignore


def _sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export_docling_json(
    doc: Any,
    out_path: Path,
    *,
    coord_precision: int = 2,
    confid_precision: int = 2,
    compress: str | None = "zstd",
    image_mode: str = "referenced",
    meta: Optional[dict[str, Any]] = None,
) -> Path:
    """Export a DoclingDocument to JSON with optional zstd compression and metadata.

    Parameters
    - doc: DoclingDocument (from docling_core)
    - out_path: path ending with .docling.json (we add .zst if compress)
    - coord_precision/confid_precision: trim numeric fields for smaller payloads
    - compress: 'zstd' to compress, None to write plain JSON
    - image_mode: recorded in meta; caller is responsible for not embedding images
    - meta: optional extra metadata dict to merge under 'meta' key
    """
    out_path = Path(out_path)
    if out_path.suffix not in {".json", ".docling.json"}:
        out_path = out_path.with_suffix(".docling.json")

    # Export to dict with precision context (docling_core honors these via pydantic context)
    try:
        payload = doc.export_to_dict(coord_precision=coord_precision, confid_precision=confid_precision)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"Docling export_to_dict failed: {e}")

    # Attach/merge metadata
    md = dict(meta or {})
    md.setdefault("coord_precision", coord_precision)
    md.setdefault("confid_precision", confid_precision)
    md.setdefault("image_mode", image_mode)
    # Allow caller to pass docling/pipeline versions if known
    if isinstance(payload, dict):
        payload.setdefault("meta", {})
        try:
            payload["meta"].update(md)
        except Exception:
            # If meta is not a dict, replace it
            payload["meta"] = md

    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    if compress and compress.lower().startswith("zst"):
        if zstd is None:
            # Fallback to plain JSON if zstd not available
            out_path.write_text(text, encoding="utf-8")
            return out_path
        out_path = out_path.with_suffix(out_path.suffix + ".zst")
        c = zstd.ZstdCompressor()
        with out_path.open("wb") as fp:
            fp.write(c.compress(text.encode("utf-8")))
        return out_path

    out_path.write_text(text, encoding="utf-8")
    return out_path


def load_docling_json(path: Path) -> Any:
    """Load DoclingDocument from JSON (optionally zstd compressed)."""
    path = Path(path)
    data: bytes
    if path.suffix.endswith(".zst") and zstd is not None:
        d = zstd.ZstdDecompressor()
        with path.open("rb") as fp:
            data = d.decompress(fp.read())
    else:
        data = path.read_bytes()

    if DoclingDocument is None:
        raise RuntimeError("docling_core is not available to load DoclingDocument")
    return DoclingDocument.model_validate_json(data.decode("utf-8"))


__all__ = [
    "export_docling_json",
    "load_docling_json",
    "_sha256_file",
]

