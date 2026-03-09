"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Iterable, List

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
PAGE_SPLIT = "\n<--- Page Split --->\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--content-debug", action="store_true")
    return parser.parse_args()


def _iter_pdfs(input_dir: Path, files: List[str]) -> List[Path]:
    if files:
        return [(input_dir / name).resolve() for name in files]
    return sorted(input_dir.glob("*.pdf"))


def _render_pages(pdf_path: Path, max_pages: int | None) -> List[Image.Image]:
    images: List[Image.Image] = []
    doc = fitz.open(pdf_path)
    try:
        page_count = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        zoom = 144 / 72.0
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


def _load_model(model_dir: Path, device: str):
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception:
        attn_impl = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
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
    return tokenizer, model


def _infer_page(model, tokenizer, image_path: Path, output_dir: Path) -> str:
    result = model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=str(image_path),
        output_path=str(output_dir),
        base_size=1024,
        image_size=768,
        crop_mode=True,
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

    tokenizer, model = _load_model(model_dir, args.device)

    for pdf_path in pdfs:
        images = _render_pages(pdf_path, args.max_pages)
        page_outputs: List[str] = []
        total_pages = len(images)
        _write_progress(output_dir, pdf_path.stem, page_outputs, total_pages, 0)
        with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_deepseek_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for idx, image in enumerate(images):
                page_png = tmp_dir / f"page_{idx + 1:04d}.png"
                image.save(page_png, format="PNG")
                page_text = _infer_page(model, tokenizer, page_png, tmp_dir / f"page_{idx + 1:04d}")
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

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
