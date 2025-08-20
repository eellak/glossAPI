"""OCR fallback helpers for GlossAPI.

Currently supports Nanonets OCR model downloaded locally under
`/mnt/data/test_ocr/ocr_models/nanonets`.

The API purposefully keeps the public surface extremely small and lightweight
so that the heavy OCR stack is only imported when required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from huggingface_hub import snapshot_download

class NanonetsHFOCR:
    """Embedded minimal implementation of the Nanonets OCR helper.
    Only the pieces needed by GlossAPI are retained; CLI helpers and verbose
    printing were removed for brevity.
    """

    def __init__(self, model_path: str, *, device: str | None = None):
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
        self._load_model_and_processor(model_path)

    def _load_model_and_processor(self, model_path: str):
        """Load model, converting to safetensors on first run for speed."""
        dtype = self._torch.float16 if self.device.startswith("cuda") else self._torch.bfloat16
        safetensors_file = Path(model_path) / "model.safetensors"

        if not safetensors_file.is_file():
            print(f"[GlossAPI] Converting model to safetensors for faster loading... (one-time operation)")
            model = self._QwenModel.from_pretrained(
                model_path, torch_dtype=dtype, trust_remote_code=True
            )
            model.save_pretrained(model_path, safe_serialization=True)
            del model # free memory

        self.model = self._QwenModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device.startswith("cuda") else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = self._AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

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

    def process_pdf(self, pdf_path: str) -> dict:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(pdf_path)
        images = self._pdf_to_images(pdf_path)

        # --- Run OCR on all images in a single batch ---
        messages = []
        for pil_image in images:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "This is a page from a scanned document. Please perform OCR and return the full text content. Preserve the original formatting, including paragraphs and line breaks, as much as possible."}
                    ]
                }
            )

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        
        responses = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return {
            "markdown_text": "\n\n---\n\n".join(responses),
            "pages": len(images),
        }

__all__ = [
    "run_nanonets_ocr",
]


def _get_default_model_dir() -> Path:
    """Return a best-guess location of the Nanonets OCR directory.

    Preference order:
    1. `glossapi/models/nanonets` inside the *installed* package (data files).
    2. Current working directory `./nanonets` (if present).
    """
    pkg_dir = Path(__file__).resolve().parent / "models" / "nanonets"
    if pkg_dir.exists():
        return pkg_dir

    cwd_dir = Path.cwd() / "nanonets"
    return cwd_dir


# Resolved once at import time—not ideal for mutability but fine for defaults.
# Lightweight cache to reuse an already-loaded OCR engine across multiple PDFs
_ENGINE_CACHE: dict[tuple[str, str], "NanonetsHFOCR"] = {}

_DEFAULT_MODEL_DIR = _get_default_model_dir()



def _ensure_model_dir(model_dir: Path) -> Path:
    """Ensure the Nanonets model code is present locally; download if missing.

    Uses `huggingface_hub.snapshot_download` which stores files under the given
    directory. The full 7.5 GB model will be fetched on first use.
    """
    if model_dir.exists():
        # Ensure python implementation present; if missing, pull it from the full
        # open-source repo on Hugging Face (includes code + weights).
        impl_path = model_dir / "nanonets_ocr.py"
        if not impl_path.exists():
            snapshot_download(
                repo_id="nanonets/Nanonets-OCR-s",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[GlossAPI] Downloading Nanonets OCR model to {model_dir} … (first time only, 7.5 GB)")
    snapshot_download(
        repo_id="nanonets/Nanonets-OCR-s",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return model_dir


def _lazy_import_nanonets(model_dir: Optional[Path] = None):
    """Dynamically import the NanonetsHFOCR class from the local checkout.

    We avoid a hard import at module load-time to keep dependency overhead
    minimal for pipelines that never trigger OCR fallback.
    """
    """Ensure model weights present and return embedded NanonetsHFOCR class."""
    _ensure_model_dir(Path(model_dir or _DEFAULT_MODEL_DIR))
    return NanonetsHFOCR


def run_nanonets_ocr(
    pdf_path: Path | str | None = None,
    *,
    images: Optional[List[Any]] = None,
    device: Optional[str] = None,
    model_dir: Optional[Path] = None,
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
    model_dir: Path, optional
        Directory containing `nanonets_ocr.py` implementation. Defaults to the
        canonical shared test location used in the project.

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

    model_dir = model_dir or _DEFAULT_MODEL_DIR
    model_dir = _ensure_model_dir(Path(model_dir))

    if pdf_path is not None:
        pdf_path = _Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import torch
    # ----- GPU requirement -----
    if not torch.cuda.is_available():
        print("[GlossAPI] CUDA device not found – skipping OCR for", pdf_path or "images input")
        return {
            "markdown_text": "",
            "duration_s": 0.0,
            "pages": 0,
            "skipped": "no_gpu",
            "model_dir": None,
        }

    if device is None:
        device = "cuda"
    # basic validation
    if not (device.startswith("cuda") or device == "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")

    # Reuse engine if cached
    cache_key = (str(Path(model_dir).resolve()), device)
    if cache_key in _ENGINE_CACHE:
        ocr_engine = _ENGINE_CACHE[cache_key]
    else:
        NanonetsHFOCR = _lazy_import_nanonets(model_dir)
        ocr_engine = NanonetsHFOCR(str(model_dir), device=device)
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
        result = ocr_engine.process_pdf(str(pdf_path))
        if max_pages is not None and isinstance(result, dict) and result.get("pages", 0) > max_pages:
            # Truncate markdown_text to first max_pages delimiters
            md_split = result.get("markdown_text", "").split("\n\n---\n\n")[:max_pages]
            result["markdown_text"] = "\n\n---\n\n".join(md_split)
            result["pages"] = max_pages
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
        "model_dir": str(model_dir),
    }
