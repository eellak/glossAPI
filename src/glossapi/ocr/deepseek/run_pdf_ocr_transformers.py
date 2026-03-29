"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files."""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Iterable, List

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)
PROMPT_GROUNDED_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "
PROMPT_PLAIN_OCR = "<image>\nExtract the text from the document page in reading order."
PAGE_SPLIT = "\n<--- Page Split --->\n"


def _profile_defaults(profile: str) -> dict:
    profile_norm = str(profile or "markdown_grounded").strip().lower()
    if profile_norm == "plain_ocr":
        return {
            "prompt": PROMPT_PLAIN_OCR,
            "base_size": 768,
            "image_size": 512,
            "crop_mode": True,
        }
    return {
        "prompt": PROMPT_GROUNDED_MARKDOWN,
        "base_size": 1024,
        "image_size": 768,
        "crop_mode": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    parser.add_argument("--attn-backend", default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--base-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--crop-mode", dest="crop_mode", action="store_true")
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    parser.set_defaults(crop_mode=None)
    parser.add_argument("--content-debug", action="store_true")
    return parser.parse_args()


def _iter_pdfs(input_dir: Path, files: List[str]) -> List[Path]:
    if files:
        return [(input_dir / name).resolve() for name in files]
    return sorted(input_dir.glob("*.pdf"))


def _render_pages(pdf_path: Path, max_pages: int | None, render_dpi: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        zoom = float(render_dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for idx in range(page_count):
            page = doc[idx]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def _clean_markdown(text: str) -> str:
    text = (text or "").replace("<｜end▁of▁sentence｜>", "").strip()
    pattern = re.compile(r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL)
    matches = pattern.findall(text)
    for full_match, label, _coords in matches:
        if label == "image":
            text = text.replace(full_match, "")
        else:
            text = text.replace(full_match, "")
    return text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:").strip()


def _resolve_attn_backend(attn_backend: str) -> str:
    requested = str(attn_backend or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _supports_retry_with_eager(exc: Exception, attn_impl: str) -> bool:
    if str(attn_impl) == "eager":
        return False
    message = str(exc)
    markers = (
        "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention",
        'load your model with the argument `attn_implementation="eager"` meanwhile',
    )
    return any(marker in message for marker in markers)


def _load_model(model_dir: Path, device: str, attn_backend: str):
    attn_impl = _resolve_attn_backend(attn_backend)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            model_dir,
            _attn_implementation=attn_impl,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except ValueError as exc:
        if not _supports_retry_with_eager(exc, attn_impl):
            raise
        LOGGER.warning(
            "DeepSeek model rejected attention backend `%s`; retrying with eager attention: %s",
            attn_impl,
            exc,
        )
        attn_impl = "eager"
        model = AutoModel.from_pretrained(
            model_dir,
            _attn_implementation=attn_impl,
            trust_remote_code=True,
            use_safetensors=True,
        )
    if device.startswith("cuda"):
        model = model.eval().to(device).to(torch.bfloat16)
    else:
        model = model.eval().to(device)
    return tokenizer, model, attn_impl


def _infer_page(
    model,
    tokenizer,
    image_path: Path,
    output_dir: Path,
    *,
    prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
) -> str:
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
        eval_mode=True,
    )
    return _clean_markdown(str(result))


def _write_outputs(output_dir: Path, stem: str, markdown: str, page_count: int) -> None:
    md_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    progress_dir = output_dir / "sidecars" / "ocr_progress"
    md_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    (md_dir / f"{stem}.md").write_text(markdown.strip() + "\n", encoding="utf-8")
    metrics = {
        "page_count": page_count,
        "model": "deepseek-ai/DeepSeek-OCR-2",
    }
    (metrics_dir / f"{stem}.metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    partial_path = progress_dir / f"{stem}.partial.md"
    if partial_path.exists():
        partial_path.unlink()


def _write_progress(
    output_dir: Path,
    stem: str,
    page_outputs: List[str],
    total_pages: int,
    completed_pages: int,
) -> None:
    """Emit lightweight progress artifacts during long OCR runs."""
    md_dir = output_dir / "markdown"
    metrics_dir = output_dir / "json" / "metrics"
    progress_dir = output_dir / "sidecars" / "ocr_progress"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    partial_markdown = PAGE_SPLIT.join(page_outputs).strip()
    if partial_markdown:
        (progress_dir / f"{stem}.partial.md").write_text(partial_markdown + "\n", encoding="utf-8")
    progress = {
        "completed_pages": completed_pages,
        "total_pages": total_pages,
        "status": "running" if completed_pages < total_pages else "complete",
        "model": "deepseek-ai/DeepSeek-OCR-2",
    }
    (metrics_dir / f"{stem}.progress.json").write_text(
        json.dumps(progress, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    pdfs = _iter_pdfs(input_dir, args.files)
    if not pdfs:
        return 0

    profile_defaults = _profile_defaults(args.ocr_profile)
    prompt = profile_defaults["prompt"]
    base_size = int(args.base_size) if args.base_size is not None else int(profile_defaults["base_size"])
    image_size = int(args.image_size) if args.image_size is not None else int(profile_defaults["image_size"])
    crop_mode = bool(args.crop_mode) if args.crop_mode is not None else bool(profile_defaults["crop_mode"])

    tokenizer, model, attn_impl = _load_model(model_dir, args.device, args.attn_backend)

    for pdf_path in pdfs:
        images = _render_pages(pdf_path, args.max_pages, args.render_dpi)
        page_outputs: List[str] = []
        total_pages = len(images)
        _write_progress(output_dir, pdf_path.stem, page_outputs, total_pages, 0)
        with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_deepseek_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for idx, image in enumerate(images):
                page_png = tmp_dir / f"page_{idx + 1:04d}.png"
                image.save(page_png, format="PNG")
                page_text = _infer_page(
                    model,
                    tokenizer,
                    page_png,
                    tmp_dir / f"page_{idx + 1:04d}",
                    prompt=prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                )
                if args.content_debug:
                    page_text = f"<!-- page:{idx + 1} -->\n{page_text}".strip()
                page_outputs.append(page_text)
                _write_progress(
                    output_dir,
                    pdf_path.stem,
                    page_outputs,
                    total_pages,
                    idx + 1,
                )
        markdown = PAGE_SPLIT.join(page_outputs) if page_outputs else "[[Blank page]]"
        _write_outputs(output_dir, pdf_path.stem, markdown, len(images))
        metrics_path = output_dir / "json" / "metrics" / f"{pdf_path.stem}.metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics.update(
                    {
                        "ocr_profile": args.ocr_profile,
                        "attn_backend": attn_impl,
                        "base_size": base_size,
                        "image_size": image_size,
                        "crop_mode": crop_mode,
                        "render_dpi": int(args.render_dpi),
                    }
                )
                metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            except Exception:
                pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
