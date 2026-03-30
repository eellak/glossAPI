"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List

import fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glossapi.ocr.utils.cleaning import (  # noqa: E402
    apply_early_stop,
    canonicalize_markdown,
    clean_output,
    strip_prompt_echo,
)

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
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
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


def _postprocess_page_text(
    text: str,
    *,
    prompt: str,
    content_debug: bool,
) -> tuple[str, dict]:
    metrics: dict = {}
    cleaned = _clean_markdown(text)
    cleaned = strip_prompt_echo(cleaned, prompt)
    cleaned = clean_output(cleaned, keep_refdet=False, metrics=metrics)
    cleaned = canonicalize_markdown(cleaned)
    cleaned = apply_early_stop(cleaned, content_debug=content_debug, metrics=metrics)
    return cleaned.strip(), metrics


def _resolve_attn_backend(attn_backend: str) -> str:
    requested = str(attn_backend or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        # DeepSeek-OCR-2's custom decoder path has not behaved reliably with SDPA
        # on the stacks we have exercised; if FA2 is unavailable, prefer the known
        # fallback instead of silently selecting a backend that then downgrades.
        return "eager"


def _supports_retry_with_eager(exc: Exception, attn_impl: str) -> bool:
    if str(attn_impl) == "eager":
        return False
    message = str(exc)
    markers = (
        "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention",
        'load your model with the argument `attn_implementation="eager"` meanwhile',
    )
    return any(marker in message for marker in markers)


def _configure_generate(
    model,
    *,
    max_new_tokens: int | None,
    repetition_penalty: float | None,
    no_repeat_ngram_size: int | None,
):
    if (
        max_new_tokens is None
        and repetition_penalty is None
        and no_repeat_ngram_size is None
    ):
        return
    capped = None
    if max_new_tokens is not None:
        capped = int(max_new_tokens)
        if capped <= 0:
            raise ValueError("max_new_tokens must be > 0")
    repetition_penalty_value = None
    if repetition_penalty is not None:
        repetition_penalty_value = float(repetition_penalty)
        if repetition_penalty_value <= 0:
            raise ValueError("repetition_penalty must be > 0")
    no_repeat_ngram_value = None
    if no_repeat_ngram_size is not None:
        no_repeat_ngram_value = int(no_repeat_ngram_size)
        if no_repeat_ngram_value <= 0:
            raise ValueError("no_repeat_ngram_size must be > 0")
    original_generate = model.generate

    def _wrapped_generate(*args, **kwargs):
        if capped is not None:
            current = kwargs.get("max_new_tokens")
            if current is None:
                kwargs["max_new_tokens"] = capped
            else:
                kwargs["max_new_tokens"] = min(int(current), capped)
        if repetition_penalty_value is not None and kwargs.get("repetition_penalty") is None:
            kwargs["repetition_penalty"] = repetition_penalty_value
        if no_repeat_ngram_value is not None and kwargs.get("no_repeat_ngram_size") is None:
            kwargs["no_repeat_ngram_size"] = no_repeat_ngram_value
        return original_generate(*args, **kwargs)

    model.generate = _wrapped_generate


def _load_model(
    model_dir: Path,
    device: str,
    attn_backend: str,
    max_new_tokens: int | None,
    repetition_penalty: float | None,
    no_repeat_ngram_size: int | None,
):
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
    _configure_generate(
        model,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
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


def _write_outputs(
    output_dir: Path,
    stem: str,
    markdown: str,
    page_count: int,
    extra_metrics: dict | None = None,
) -> None:
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
    if extra_metrics:
        metrics.update(extra_metrics)
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

    tokenizer, model, attn_impl = _load_model(
        model_dir,
        args.device,
        args.attn_backend,
        args.max_new_tokens,
        args.repetition_penalty,
        args.no_repeat_ngram_size,
    )

    for pdf_path in pdfs:
        doc_start = time.perf_counter()
        render_start = time.perf_counter()
        images = _render_pages(pdf_path, args.max_pages, args.render_dpi)
        render_sec = time.perf_counter() - render_start
        page_outputs: List[str] = []
        page_metrics: List[dict] = []
        total_pages = len(images)
        _write_progress(output_dir, pdf_path.stem, page_outputs, total_pages, 0)
        with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_deepseek_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            for idx, image in enumerate(images):
                page_png = tmp_dir / f"page_{idx + 1:04d}.png"
                image.save(page_png, format="PNG")
                infer_start = time.perf_counter()
                raw_page_text = _infer_page(
                    model,
                    tokenizer,
                    page_png,
                    tmp_dir / f"page_{idx + 1:04d}",
                    prompt=prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                )
                infer_sec = time.perf_counter() - infer_start
                page_text, postprocess_metrics = _postprocess_page_text(
                    raw_page_text,
                    prompt=prompt,
                    content_debug=bool(args.content_debug),
                )
                if args.content_debug:
                    page_text = f"<!-- page:{idx + 1} -->\n{page_text}".strip()
                page_outputs.append(page_text)
                page_metrics.append(
                    {
                        "page_number": int(idx + 1),
                        "infer_sec": float(infer_sec),
                        "raw_chars": int(len(str(raw_page_text or "").strip())),
                        "final_chars": int(len(page_text.strip())),
                        **postprocess_metrics,
                    }
                )
                _write_progress(
                    output_dir,
                    pdf_path.stem,
                    page_outputs,
                    total_pages,
                    idx + 1,
                )
        markdown = PAGE_SPLIT.join(page_outputs) if page_outputs else "[[Blank page]]"
        _write_outputs(
            output_dir,
            pdf_path.stem,
            markdown,
            len(images),
            extra_metrics={
                "ocr_profile": args.ocr_profile,
                "attn_backend": attn_impl,
                "base_size": base_size,
                "image_size": image_size,
                "crop_mode": crop_mode,
                "render_dpi": int(args.render_dpi),
                "max_new_tokens": args.max_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "render_sec": float(render_sec),
                "infer_sec_total": float(sum(item["infer_sec"] for item in page_metrics)),
                "wall_time_sec": float(time.perf_counter() - doc_start),
                "page_metrics": page_metrics,
            },
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
