"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files using vLLM."""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
REPAIR_TILE_SPECS: Tuple[Tuple[str, float, float], ...] = (
    ("top", 0.0, 0.5),
    ("mid", 0.35, 0.8),
    ("bottom", 0.65, 1.0),
)
REPAIR_DARK_THRESHOLD = 235
REPAIR_SHORT_CHARS = 700
REPAIR_EXTREME_SHORT_CHARS = 120
REPAIR_PUA_THRESHOLD = 64
REPAIR_MIN_HALF_DARK = 0.08
REPAIR_MAX_OVERALL_DARK = 0.25
REPAIR_MIN_OVERALL_DARK = 0.04


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    parser.add_argument("--prompt-override", default=None)
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
    parser.add_argument("--repair-mode", default="auto", choices=["auto", "off"])
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


def _image_content_stats(image: Image.Image) -> dict:
    sample = image.convert("L")
    sample.thumbnail((256, 256))
    width, height = sample.size
    pixels = list(sample.getdata())

    def _dark_ratio(y0: int, y1: int) -> float:
        values = []
        for row in range(y0, y1):
            start = row * width
            values.extend(pixels[start : start + width])
        total = len(values)
        if total <= 0:
            return 0.0
        dark = sum(1 for value in values if value < REPAIR_DARK_THRESHOLD)
        return float(dark) / float(total)

    half = max(1, height // 2)
    dark_total = sum(1 for value in pixels if value < REPAIR_DARK_THRESHOLD)
    return {
        "top_dark_ratio": _dark_ratio(0, half),
        "bottom_dark_ratio": _dark_ratio(half, height),
        "overall_dark_ratio": float(dark_total) / float(max(1, len(pixels))),
    }


def _count_private_use_chars(text: str) -> int:
    return sum(
        1
        for ch in str(text or "")
        if 0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
    )


def _text_quality_metrics(text: str) -> dict:
    stripped = str(text or "").strip()
    letters = sum(1 for ch in stripped if ch.isalpha())
    digits = sum(1 for ch in stripped if ch.isdigit())
    pua_chars = _count_private_use_chars(stripped)
    score = float(letters) + (0.10 * float(len(stripped))) + (0.05 * float(digits)) - (20.0 * float(pua_chars))
    return {
        "chars": int(len(stripped)),
        "letters": int(letters),
        "digits": int(digits),
        "pua_chars": int(pua_chars),
        "quality_score": float(score),
    }


def _classify_repair(text: str, image_stats: dict, repair_mode: str) -> tuple[str, str | None]:
    if str(repair_mode or "off").strip().lower() != "auto":
        return "none", None
    quality = _text_quality_metrics(text)
    chars = int(quality["chars"])
    pua_chars = int(quality["pua_chars"])
    pua_ratio = float(pua_chars) / float(max(1, chars))
    if pua_chars >= REPAIR_PUA_THRESHOLD or pua_ratio >= 0.10:
        return "plain", "markdown_garbage"
    if chars <= REPAIR_EXTREME_SHORT_CHARS:
        return "plain", "extreme_short"
    top_dark = float(image_stats.get("top_dark_ratio", 0.0))
    bottom_dark = float(image_stats.get("bottom_dark_ratio", 0.0))
    overall_dark = float(image_stats.get("overall_dark_ratio", 0.0))
    if (
        chars <= REPAIR_SHORT_CHARS
        and top_dark >= REPAIR_MIN_HALF_DARK
        and bottom_dark >= REPAIR_MIN_HALF_DARK
        and REPAIR_MIN_OVERALL_DARK <= overall_dark <= REPAIR_MAX_OVERALL_DARK
    ):
        return "tile", "short_coverage"
    return "none", None


def _load_job_image(item: dict) -> Image.Image:
    image = Image.open(item["image_path"]).convert("RGB")
    crop_box = item.get("crop_box")
    if not crop_box:
        return image
    width, height = image.size
    x0_norm, y0_norm, x1_norm, y1_norm = crop_box
    crop_pixels = (
        int(round(float(x0_norm) * width)),
        int(round(float(y0_norm) * height)),
        int(round(float(x1_norm) * width)),
        int(round(float(y1_norm) * height)),
    )
    cropped = image.crop(crop_pixels)
    image.close()
    return cropped


def _generate_batch_outputs(
    llm,
    *,
    jobs: List[dict],
    prompt: str,
    batch_size: int,
    sampling_params,
) -> List[dict]:
    outputs_by_key: Dict[tuple[str, int, str], dict] = {}
    for batch in _batched(jobs, batch_size):
        prompt_batch = []
        opened_images: List[Image.Image] = []
        keys: List[tuple[str, int, str]] = []
        for item in batch:
            image = _load_job_image(item)
            opened_images.append(image)
            keys.append((str(item["stem"]), int(item["page_number"]), str(item.get("variant", "page"))))
            prompt_batch.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            )
        infer_start = time.perf_counter()
        batch_outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
        infer_sec = time.perf_counter() - infer_start
        per_item_sec = infer_sec / max(1, len(batch))
        for image in opened_images:
            image.close()
        for item, key, output in zip(batch, keys, batch_outputs):
            raw_text = ""
            if getattr(output, "outputs", None):
                raw_text = str(output.outputs[0].text)
            outputs_by_key[key] = {
                "item": item,
                "raw_text": raw_text,
                "infer_sec": float(per_item_sec),
            }
    ordered = []
    for item in jobs:
        ordered.append(outputs_by_key[(str(item["stem"]), int(item["page_number"]), str(item.get("variant", "page")))])
    return ordered


def _stitch_tiled_markdown(parts: List[str]) -> str:
    stitched: List[str] = []
    previous_lines: List[str] = []
    for part in parts:
        lines = [line.rstrip() for line in str(part or "").splitlines() if line.strip()]
        if not lines:
            continue
        overlap = 0
        max_overlap = min(len(previous_lines), len(lines), 12)
        for size in range(max_overlap, 0, -1):
            if previous_lines[-size:] == lines[:size]:
                overlap = size
                break
        stitched.extend(lines[overlap:])
        previous_lines = lines
    return "\n".join(stitched).strip()


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    pdfs = _iter_pdfs(input_dir, args.files)
    if not pdfs:
        return 0

    profile_defaults = _profile_defaults(args.ocr_profile)
    prompt = str(args.prompt_override) if args.prompt_override else profile_defaults["prompt"]
    plain_prompt = _profile_defaults("plain_ocr")["prompt"]
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
                "page_metrics": [None] * total_pages,
                "render_sec": float(render_sec),
                "doc_start": float(doc_start),
                "completed_pages": 0,
                "total_pages": total_pages,
            }
            doc_states[pdf_path.stem] = state
            _write_progress(output_dir, pdf_path.stem, [], total_pages, 0)
            for idx, image in enumerate(images):
                page_path = tmp_dir / f"{pdf_path.stem}_page_{idx + 1:04d}.png"
                image_stats = _image_content_stats(image)
                image.save(page_path, format="PNG")
                image.close()
                jobs.append(
                    {
                        "stem": pdf_path.stem,
                        "page_number": int(idx + 1),
                        "image_path": page_path,
                        "image_stats": image_stats,
                        "variant": "page",
                    }
                )

        plain_repair_jobs: List[dict] = []
        tile_repair_requests: List[dict] = []
        first_pass_outputs = _generate_batch_outputs(
            llm,
            jobs=jobs,
            prompt=prompt,
            batch_size=int(args.batch_size),
            sampling_params=sampling_params,
        )
        for result in first_pass_outputs:
            item = result["item"]
            state = doc_states[item["stem"]]
            raw_text = str(result["raw_text"])
            image_stats = dict(item.get("image_stats", {}))
            page_text, postprocess_metrics = _postprocess_page_text(
                raw_text,
                prompt=prompt,
                content_debug=bool(args.content_debug),
            )
            if args.content_debug:
                page_text = f"<!-- page:{item['page_number']} -->\n{page_text}".strip()
            state["page_outputs"][item["page_number"] - 1] = page_text
            quality = _text_quality_metrics(page_text)
            repair_strategy, repair_reason = _classify_repair(
                page_text,
                image_stats=image_stats,
                repair_mode=args.repair_mode,
            )
            metric = {
                "page_number": int(item["page_number"]),
                "infer_sec": float(result["infer_sec"]),
                "raw_chars": int(len(raw_text.strip())),
                "final_chars": int(len(page_text.strip())),
                "first_pass_quality_score": float(quality["quality_score"]),
                "first_pass_letters": int(quality["letters"]),
                "first_pass_digits": int(quality["digits"]),
                "first_pass_pua_chars": int(quality["pua_chars"]),
                "repair_strategy": repair_strategy,
                "repair_reason": repair_reason,
                "repair_attempted": False,
                "repair_applied": False,
                **image_stats,
                **postprocess_metrics,
            }
            state["page_metrics"][item["page_number"] - 1] = metric
            if repair_strategy == "plain":
                plain_repair_jobs.append(item)
            elif repair_strategy == "tile":
                tile_repair_requests.append(item)
            state["completed_pages"] = int(state["completed_pages"]) + 1
            progress_pages = [page for page in state["page_outputs"] if page]
            _write_progress(
                output_dir,
                item["stem"],
                progress_pages,
                int(state["total_pages"]),
                int(state["completed_pages"]),
            )

        if plain_repair_jobs:
            plain_repair_outputs = _generate_batch_outputs(
                llm,
                jobs=plain_repair_jobs,
                prompt=plain_prompt,
                batch_size=int(args.batch_size),
                sampling_params=sampling_params,
            )
            for result in plain_repair_outputs:
                item = result["item"]
                state = doc_states[item["stem"]]
                metric = state["page_metrics"][item["page_number"] - 1]
                original_text = state["page_outputs"][item["page_number"] - 1]
                repair_text, repair_postprocess = _postprocess_page_text(
                    str(result["raw_text"]),
                    prompt=plain_prompt,
                    content_debug=bool(args.content_debug),
                )
                if args.content_debug:
                    repair_text = f"<!-- page:{item['page_number']} -->\n{repair_text}".strip()
                original_quality = _text_quality_metrics(original_text)
                repair_quality = _text_quality_metrics(repair_text)
                apply_repair = bool(repair_text.strip()) and (
                    float(repair_quality["quality_score"]) >= float(original_quality["quality_score"])
                    or str(metric.get("repair_reason")) in {"markdown_garbage", "extreme_short"}
                )
                metric["repair_attempted"] = True
                metric["repair_infer_sec"] = float(result["infer_sec"])
                metric["repair_raw_chars"] = int(len(str(result["raw_text"]).strip()))
                metric["repair_final_chars"] = int(len(repair_text.strip()))
                metric["repair_quality_score"] = float(repair_quality["quality_score"])
                metric["repair_profile"] = "plain_ocr"
                metric.update({f"repair_{key}": value for key, value in repair_postprocess.items()})
                metric["infer_sec"] = float(metric["infer_sec"]) + float(result["infer_sec"])
                if apply_repair:
                    state["page_outputs"][item["page_number"] - 1] = repair_text
                    metric["repair_applied"] = True
                    metric["final_chars"] = int(len(repair_text.strip()))
                    _write_progress(
                        output_dir,
                        item["stem"],
                        [page for page in state["page_outputs"] if page],
                        int(state["total_pages"]),
                        int(state["completed_pages"]),
                    )

        if tile_repair_requests:
            tile_jobs: List[dict] = []
            for item in tile_repair_requests:
                for tile_name, y0, y1 in REPAIR_TILE_SPECS:
                    tile_jobs.append(
                        {
                            "stem": item["stem"],
                            "page_number": int(item["page_number"]),
                            "image_path": item["image_path"],
                            "variant": tile_name,
                            "crop_box": (0.0, y0, 1.0, y1),
                        }
                    )
            tile_outputs = _generate_batch_outputs(
                llm,
                jobs=tile_jobs,
                prompt=prompt,
                batch_size=int(args.batch_size),
                sampling_params=sampling_params,
            )
            grouped_tile_outputs: Dict[tuple[str, int], List[dict]] = {}
            for result in tile_outputs:
                key = (str(result["item"]["stem"]), int(result["item"]["page_number"]))
                grouped_tile_outputs.setdefault(key, []).append(result)
            for item in tile_repair_requests:
                key = (str(item["stem"]), int(item["page_number"]))
                state = doc_states[item["stem"]]
                metric = state["page_metrics"][item["page_number"] - 1]
                original_text = state["page_outputs"][item["page_number"] - 1]
                grouped = sorted(
                    grouped_tile_outputs.get(key, []),
                    key=lambda value: {"top": 0, "mid": 1, "bottom": 2}.get(str(value["item"].get("variant")), 99),
                )
                tile_parts: List[str] = []
                repair_infer_sec = 0.0
                for result in grouped:
                    repair_infer_sec += float(result["infer_sec"])
                    tile_text, _ = _postprocess_page_text(
                        str(result["raw_text"]),
                        prompt=prompt,
                        content_debug=bool(args.content_debug),
                    )
                    tile_parts.append(tile_text)
                stitched = _stitch_tiled_markdown(tile_parts)
                if args.content_debug:
                    stitched = f"<!-- page:{item['page_number']} -->\n{stitched}".strip()
                original_quality = _text_quality_metrics(original_text)
                stitched_quality = _text_quality_metrics(stitched)
                apply_repair = bool(stitched.strip()) and (
                    float(stitched_quality["quality_score"]) > float(original_quality["quality_score"])
                    and int(stitched_quality["chars"]) >= int(original_quality["chars"])
                )
                metric["repair_attempted"] = True
                metric["repair_infer_sec"] = float(metric.get("repair_infer_sec", 0.0)) + float(repair_infer_sec)
                metric["repair_final_chars"] = int(len(stitched.strip()))
                metric["repair_quality_score"] = float(stitched_quality["quality_score"])
                metric["repair_tile_count"] = int(len(grouped))
                metric["repair_profile"] = "markdown_grounded_tiled"
                metric["infer_sec"] = float(metric["infer_sec"]) + float(repair_infer_sec)
                if apply_repair:
                    state["page_outputs"][item["page_number"] - 1] = stitched
                    metric["repair_applied"] = True
                    metric["final_chars"] = int(len(stitched.strip()))
                    _write_progress(
                        output_dir,
                        item["stem"],
                        [page for page in state["page_outputs"] if page],
                        int(state["total_pages"]),
                        int(state["completed_pages"]),
                    )

        for stem, state in doc_states.items():
            markdown = PAGE_SPLIT.join(state["page_outputs"]) if state["page_outputs"] else "[[Blank page]]"
            page_metrics = sorted(
                [item for item in state["page_metrics"] if item],
                key=lambda item: int(item["page_number"]),
            )
            repair_summary = {
                "repair_mode": str(args.repair_mode),
                "pages_flagged": int(sum(1 for item in page_metrics if str(item.get("repair_strategy")) != "none")),
                "pages_repaired": int(sum(1 for item in page_metrics if bool(item.get("repair_applied")))),
                "plain_repairs": int(sum(1 for item in page_metrics if str(item.get("repair_profile")) == "plain_ocr" and bool(item.get("repair_applied")))),
                "tiled_repairs": int(sum(1 for item in page_metrics if str(item.get("repair_profile")) == "markdown_grounded_tiled" and bool(item.get("repair_applied")))),
            }
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
                    "repair_mode": str(args.repair_mode),
                    "render_sec": float(state["render_sec"]),
                    "infer_sec_total": float(sum(item["infer_sec"] for item in page_metrics)),
                    "wall_time_sec": float(time.perf_counter() - float(state["doc_start"])),
                    "repair_summary": repair_summary,
                    "page_metrics": page_metrics,
                },
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
