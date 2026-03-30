"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files using vLLM."""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image

from glossapi.ocr.deepseek.run_pdf_ocr_transformers import (
    PAGE_SPLIT,
    _iter_pdfs,
    _postprocess_page_text,
    _profile_defaults,
    _render_pages,
    _write_outputs,
    _write_progress,
)

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    parser.add_argument("--attn-backend", default="vllm")
    parser.add_argument("--base-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--crop-mode", dest="crop_mode", action="store_true")
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    parser.set_defaults(crop_mode=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--disable-fp8-kv", action="store_true")
    parser.add_argument("--content-debug", action="store_true")
    return parser.parse_args()


def _load_vllm(model_dir: Path, gpu_memory_utilization: float, disable_fp8_kv: bool):
    from vllm import LLM

    logits_processors = None
    try:
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        logits_processors = [NGramPerReqLogitsProcessor]
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("DeepSeek OCR logits processor unavailable in vLLM; continuing without it: %s", exc)

    engine_kwargs = {
        "model": str(model_dir),
        "tokenizer": str(model_dir),
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "tensor_parallel_size": 1,
    }
    if disable_fp8_kv:
        engine_kwargs["kv_cache_dtype"] = "auto"
    if logits_processors:
        engine_kwargs["logits_processors"] = logits_processors
    return LLM(**engine_kwargs)


def _sampling_params(max_new_tokens: int | None):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=int(max_new_tokens or 8192),
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
        },
    )


def _batched(items: List[dict], batch_size: int) -> List[List[dict]]:
    size = max(1, int(batch_size))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


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

    llm = _load_vllm(
        model_dir,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        disable_fp8_kv=bool(args.disable_fp8_kv),
    )
    sampling_params = _sampling_params(args.max_new_tokens)

    with tempfile.TemporaryDirectory(prefix="deepseek_vllm_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        doc_states: Dict[str, dict] = {}
        jobs: List[dict] = []

        for pdf_path in pdfs:
            doc_start = time.perf_counter()
            render_start = time.perf_counter()
            images = _render_pages(pdf_path, args.max_pages, args.render_dpi)
            render_sec = time.perf_counter() - render_start
            total_pages = len(images)
            state = {
                "stem": pdf_path.stem,
                "page_outputs": [""] * total_pages,
                "page_metrics": [],
                "render_sec": float(render_sec),
                "doc_start": float(doc_start),
                "completed_pages": 0,
                "total_pages": total_pages,
            }
            doc_states[pdf_path.stem] = state
            _write_progress(output_dir, pdf_path.stem, [], total_pages, 0)
            for idx, image in enumerate(images):
                page_path = tmp_dir / f"{pdf_path.stem}_page_{idx + 1:04d}.png"
                image.save(page_path, format="PNG")
                image.close()
                jobs.append(
                    {
                        "stem": pdf_path.stem,
                        "page_number": int(idx + 1),
                        "image_path": page_path,
                    }
                )

        for batch in _batched(jobs, args.batch_size):
            prompt_batch = []
            images: List[Image.Image] = []
            for item in batch:
                image = Image.open(item["image_path"]).convert("RGB")
                images.append(image)
                prompt_batch.append(
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                )
            infer_start = time.perf_counter()
            outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
            infer_sec = time.perf_counter() - infer_start
            per_item_sec = infer_sec / max(1, len(batch))
            for image in images:
                image.close()

            for item, output in zip(batch, outputs):
                state = doc_states[item["stem"]]
                raw_text = ""
                if getattr(output, "outputs", None):
                    raw_text = str(output.outputs[0].text)
                page_text, postprocess_metrics = _postprocess_page_text(
                    raw_text,
                    prompt=prompt,
                    content_debug=bool(args.content_debug),
                )
                if args.content_debug:
                    page_text = f"<!-- page:{item['page_number']} -->\n{page_text}".strip()
                state["page_outputs"][item["page_number"] - 1] = page_text
                state["page_metrics"].append(
                    {
                        "page_number": int(item["page_number"]),
                        "infer_sec": float(per_item_sec),
                        "raw_chars": int(len(raw_text.strip())),
                        "final_chars": int(len(page_text.strip())),
                        **postprocess_metrics,
                    }
                )
                state["completed_pages"] = int(state["completed_pages"]) + 1
                progress_pages = [page for page in state["page_outputs"] if page]
                _write_progress(
                    output_dir,
                    item["stem"],
                    progress_pages,
                    int(state["total_pages"]),
                    int(state["completed_pages"]),
                )

        for stem, state in doc_states.items():
            markdown = PAGE_SPLIT.join(state["page_outputs"]) if state["page_outputs"] else "[[Blank page]]"
            page_metrics = sorted(state["page_metrics"], key=lambda item: int(item["page_number"]))
            _write_outputs(
                output_dir,
                stem,
                markdown,
                int(state["total_pages"]),
                extra_metrics={
                    "ocr_profile": args.ocr_profile,
                    "attn_backend": "vllm",
                    "runtime_backend": "vllm",
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "render_dpi": int(args.render_dpi),
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": int(args.batch_size),
                    "gpu_memory_utilization": float(args.gpu_memory_utilization),
                    "disable_fp8_kv": bool(args.disable_fp8_kv),
                    "render_sec": float(state["render_sec"]),
                    "infer_sec_total": float(sum(item["infer_sec"] for item in page_metrics)),
                    "wall_time_sec": float(time.perf_counter() - float(state["doc_start"])),
                    "page_metrics": page_metrics,
                },
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
