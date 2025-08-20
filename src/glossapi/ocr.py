"""OCR fallback helpers for GlossAPI.

Supports the Nanonets OCR model via Hugging Face. By default, we use the
model id `nanonets/Nanonets-OCR-s` and let `transformers` handle caching and
downloads automatically. Users may also pass a local model directory path to
reuse pre-downloaded weights.

The API purposefully keeps the public surface extremely small and lightweight
so that the heavy OCR stack is only imported when required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


class NanonetsHFOCR:
    """Embedded minimal implementation of the Nanonets OCR helper.
    Only the pieces needed by GlossAPI are retained; CLI helpers and verbose
    printing were removed for brevity.
    """

    def __init__(self, model_id_or_path: str, *, device: str | None = None):
        # Lazy heavy imports
        import importlib
        torch = importlib.import_module("torch")
        self._torch = torch
        self._Image = importlib.import_module("PIL.Image")
        self._fitz = importlib.import_module("fitz")
        transformers = importlib.import_module("transformers")
        self._QwenModel = transformers.Qwen2_5_VLForConditionalGeneration
        self._AutoProcessor = transformers.AutoProcessor

        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model and processor
        self._load_model_and_processor(model_id_or_path)

    def _load_model_and_processor(self, model_id_or_path: str):
        """Load model and processor via Transformers. Let HF handle caching/downloads."""
        # Safer dtype selection: float16 on CUDA, float32 on CPU
        dtype = self._torch.float16 if self.device.startswith("cuda") else self._torch.float32

        self.model = self._QwenModel.from_pretrained(
            model_id_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = self._AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)

    def _pdf_to_images(self, pdf_path: str, dpi: int = 300):
        """Render PDF pages to PIL images in parallel."""
        doc = self._fitz.open(pdf_path)
        import concurrent.futures as _cf
        import io as _io

        def _render(page):
            pix = page.get_pixmap(matrix=self._fitz.Matrix(dpi / 72, dpi / 72))
            return self._Image.open(_io.BytesIO(pix.tobytes("png")))

        with _cf.ThreadPoolExecutor() as ex:
            imgs = list(ex.map(_render, doc))
        doc.close()
        return imgs

    def _process_image(self, image):
        """Process a single image and return OCR text."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "This is a page from a scanned document. Please perform OCR and return the full text content. Preserve the original formatting, including paragraphs and line breaks, as much as possible."}
            ]
        }]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        
        with self._torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def process_pdf(self, pdf_path: str, *, max_pages: Optional[int] = None, dpi: int = 300) -> dict:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(pdf_path)
        images = self._pdf_to_images(pdf_path, dpi=dpi)
        if max_pages is not None:
            images = images[:max_pages]

        # Page-wise OCR to reduce memory pressure
        pages_md = [self._process_image(img) for img in images]
        return {
            "markdown_text": "\n\n---\n\n".join(pages_md),
            "pages": len(images),
        }

__all__ = [
    "run_nanonets_ocr",
]


# (No default local directory decision. We rely on HF model id by default.)


# Resolved once at import time—not ideal for mutability but fine for defaults.
# Lightweight cache to reuse an already-loaded OCR engine across multiple PDFs
_ENGINE_CACHE: dict[tuple[str, str], "NanonetsHFOCR"] = {}

DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR-s"



def _ensure_model_dir(model_dir: Path) -> Path:
    """Deprecated: no-op placeholder kept for backward compatibility.
    Model downloads are handled by `transformers.from_pretrained` automatically.
    """
    return model_dir


def _lazy_import_nanonets():
    """Return the embedded NanonetsHFOCR class without side effects."""
    return NanonetsHFOCR


def run_nanonets_ocr(
    pdf_path: Path | str | None = None,
    *,
    images: Optional[List[Any]] = None,
    device: Optional[str] = None,
    model_dir: Optional[Union[str, Path]] = None,
    max_pages: int | None = None,
) -> Dict[str, Any]:
    """Run Nanonets OCR on either a PDF *or* a pre-rendered list of page images.

        Path to the PDF file. **Mutually exclusive** with *images*.
    images : list[PIL.Image] | None
        Pre-rendered page images to OCR. When given, *pdf_path* must be *None*.
    device : str | None
        Device spec understood by PyTorch – e.g. "cuda", "cuda:1". If *None*,
{{ ... }}
    model_dir : Path | None
        Device spec understood by PyTorch – e.g. "cuda", "cuda:1", "cpu". If
        *None*, the helper will auto-select GPU if available.
    model_dir: Path | None
        Directory containing the Nanonets weights / helper script.
    max_pages: int | None
        Optional safety cut-off. If provided, only the first *max_pages* will be
        OCR-processed.

    Returns
    -------
    dict
        A dictionary with at minimum keys::

            {
              "markdown_text": str,  # OCR'd md
              "duration_s": float,   # wall-clock seconds
              "pages": int,          # number of processed pages
            }

    Parameters
    ----------
    pdf_path: Path | str
        Absolute or relative path to the PDF file.
    device: str | None, optional
        Device spec understood by PyTorch – e.g. "cuda", "cuda:1", "cpu". If
        *None*, the helper will auto-select GPU if available.
    model_dir: Union[str, Path], optional
        Hugging Face model id (e.g., 'nanonets/Nanonets-OCR-s') or a local path.
        If None, defaults to 'nanonets/Nanonets-OCR-s'. The Transformers library
        handles downloading and caching.

    Returns
    -------
    dict
        A dictionary with at minimum keys::

            {
              "markdown_text": str,  # OCR'd md
              "duration_s": float,   # wall-clock seconds
              "pages": int,          # number of processed pages
            }

        The exact structure is delegated to `NanonetsHFOCR.process_pdf`.
    """
    from time import perf_counter
    from pathlib import Path as _Path

    # ---------------- argument validation ----------------
    if (pdf_path is None) == (images is None):
        raise ValueError("Provide exactly one of pdf_path or images")
    # Resolve model identifier or path
    model_id_or_path = str(model_dir) if model_dir is not None else DEFAULT_MODEL_ID

    if pdf_path is not None:
        pdf_path = _Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import torch
    # Device resolution with graceful CPU fallback
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[GlossAPI] Requested CUDA but no CUDA device available; falling back to CPU.")
        device = "cpu"
    # basic validation
    if not (device.startswith("cuda") or device == "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")

    # Reuse engine if cached
    cache_key = (str(model_id_or_path), device)
    if cache_key in _ENGINE_CACHE:
        ocr_engine = _ENGINE_CACHE[cache_key]
    else:
        NanonetsHFOCRClass = _lazy_import_nanonets()
        ocr_engine = NanonetsHFOCRClass(str(model_id_or_path), device=device)
        _ENGINE_CACHE[cache_key] = ocr_engine

    t0 = perf_counter()
    if images is not None:
        if max_pages is not None:
            images = images[:max_pages]
        pages_md = [ocr_engine._process_image(img) for img in images]
        result = {
            "markdown_text": "\n\n---\n\n".join(pages_md),
            "pages": len(images),
        }
    else:
        # Render pages and process one-by-one to limit memory
        page_images = ocr_engine._pdf_to_images(str(pdf_path))
        if max_pages is not None:
            page_images = page_images[:max_pages]
        pages_md = [ocr_engine._process_image(img) for img in page_images]
        result = {
            "markdown_text": "\n\n---\n\n".join(pages_md),
            "pages": len(page_images),
        }
    duration = perf_counter() - t0

    # Normalise return payload
    markdown_text: str
    pages: int
    if isinstance(result, dict):
        markdown_text = result.get("markdown_text") or result.get("text") or ""
        pages = int(result.get("pages") or result.get("page_count") or 0)
    else:
        # Fallback: treat as raw text
        markdown_text = str(result)
        pages = 0

    return {
        "markdown_text": markdown_text,
        "duration_s": duration,
        "pages": pages,
        "model_dir": str(model_id_or_path),
    }
