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
    DEFAULT_MAX_NEW_TOKENS,
    PAGE_SPLIT,
    _iter_pdf_jobs,
    _postprocess_page_text,
    _profile_defaults,
    _render_pages,
    _write_outputs,
    _write_progress,
)
from glossapi.ocr.utils.cleaning import StreamingGarbageDetector

LOGGER = logging.getLogger(__name__)
REPAIR_DARK_THRESHOLD = 235
EMPTY_PAGE_OVERALL_DARK_MAX = 0.0015
EMPTY_PAGE_BAND_DARK_MAX = 0.0025
GARBAGE_EARLY_STOP_MIN_OUTPUT_TOKENS = 48
GARBAGE_EARLY_STOP_WINDOW_TOKENS = 160


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--page-ranges", nargs="*", default=[])
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ocr-profile", default="markdown_grounded", choices=["markdown_grounded", "plain_ocr"])
    parser.add_argument("--prompt-override", default=None)
    parser.add_argument("--attn-backend", default="vllm")
    parser.add_argument("--base-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--render-dpi", type=int, default=144)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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

    logits_processors = []
    try:
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        logits_processors.append(NGramPerReqLogitsProcessor)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("DeepSeek OCR logits processor unavailable in vLLM; continuing without it: %s", exc)

    try:
        from transformers import AutoTokenizer
        from vllm.sampling_params import SamplingParams
        from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

        class _GarbageStopPerReqLogitsProcessor:
            def __init__(
                self,
                tokenizer,
                eos_token_id: int | None,
                *,
                min_output_tokens: int,
                window_tokens: int,
            ) -> None:
                self.tokenizer = tokenizer
                self.eos_token_id = eos_token_id
                self.min_output_tokens = int(min_output_tokens)
                self.window_tokens = int(window_tokens)
                self.detector = StreamingGarbageDetector()
                self.seen_output_tokens = 0

            def __call__(self, prompt_ids: list[int], output_ids: list[int], logits):
                del prompt_ids
                if self.eos_token_id is None:
                    return logits
                current_len = len(output_ids)
                if current_len <= self.seen_output_tokens:
                    return logits
                new_ids = output_ids[self.seen_output_tokens :]
                self.seen_output_tokens = current_len
                if not new_ids:
                    return logits
                new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                if new_text:
                    self.detector.feed(new_text)
                if current_len < self.min_output_tokens or self.detector.triggered_reason is None:
                    return logits
                eos_token_id = int(self.eos_token_id)
                eos_value = logits[eos_token_id].clone()
                logits[:] = float("-inf")
                logits[eos_token_id] = eos_value
                return logits

        class GarbageEarlyStopLogitsProcessor(AdapterLogitsProcessor):
            @classmethod
            def validate_params(cls, params: SamplingParams):
                extra = params.extra_args or {}
                enabled = extra.get("garbage_early_stop")
                if enabled is None:
                    return
                if not isinstance(enabled, bool):
                    raise ValueError("garbage_early_stop must be a bool when provided")
                min_output_tokens = extra.get("garbage_min_output_tokens")
                if min_output_tokens is not None and int(min_output_tokens) <= 0:
                    raise ValueError("garbage_min_output_tokens must be > 0")
                window_tokens = extra.get("garbage_window_tokens")
                if window_tokens is not None and int(window_tokens) <= 0:
                    raise ValueError("garbage_window_tokens must be > 0")

            def __init__(self, vllm_config, device, is_pin_memory):
                super().__init__(vllm_config, device, is_pin_memory)
                self._tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
                self._eos_token_id = self._tokenizer.eos_token_id

            def is_argmax_invariant(self) -> bool:
                return False

            def new_req_logits_processor(self, params: SamplingParams):
                extra = params.extra_args or {}
                if not bool(extra.get("garbage_early_stop", False)):
                    return None
                return _GarbageStopPerReqLogitsProcessor(
                    self._tokenizer,
                    self._eos_token_id,
                    min_output_tokens=int(
                        extra.get("garbage_min_output_tokens", GARBAGE_EARLY_STOP_MIN_OUTPUT_TOKENS)
                    ),
                    window_tokens=int(
                        extra.get("garbage_window_tokens", GARBAGE_EARLY_STOP_WINDOW_TOKENS)
                    ),
                )

        logits_processors.append(GarbageEarlyStopLogitsProcessor)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Garbage-stop logits processor unavailable in vLLM; continuing without it: %s", exc)

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


def _sampling_params(max_new_tokens: int | None, *, enable_garbage_early_stop: bool):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        max_tokens=int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS),
        skip_special_tokens=False,
        extra_args={
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": {128821, 128822},
            "garbage_early_stop": bool(enable_garbage_early_stop),
            "garbage_min_output_tokens": int(GARBAGE_EARLY_STOP_MIN_OUTPUT_TOKENS),
            "garbage_window_tokens": int(GARBAGE_EARLY_STOP_WINDOW_TOKENS),
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
    third = max(1, height // 3)
    top_third_end = min(height, third)
    middle_third_end = min(height, third * 2)
    dark_total = sum(1 for value in pixels if value < REPAIR_DARK_THRESHOLD)
    return {
        "top_dark_ratio": _dark_ratio(0, half),
        "bottom_dark_ratio": _dark_ratio(half, height),
        "top_third_dark_ratio": _dark_ratio(0, top_third_end),
        "middle_third_dark_ratio": _dark_ratio(top_third_end, middle_third_end),
        "bottom_third_dark_ratio": _dark_ratio(middle_third_end, height),
        "overall_dark_ratio": float(dark_total) / float(max(1, len(pixels))),
    }


def _text_quality_metrics(text: str) -> dict:
    stripped = str(text or "").strip()
    letters = sum(1 for ch in stripped if ch.isalpha())
    digits = sum(1 for ch in stripped if ch.isdigit())
    pua_chars = sum(
        1
        for ch in stripped
        if 0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
    )
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    avg_line_length = (sum(len(line) for line in lines) / float(len(lines))) if lines else 0.0
    score = float(letters) + (0.10 * float(len(stripped))) + (0.05 * float(digits)) - (20.0 * float(pua_chars))
    return {
        "chars": int(len(stripped)),
        "letters": int(letters),
        "digits": int(digits),
        "pua_chars": int(pua_chars),
        "line_count": int(len(lines)),
        "avg_line_length": float(avg_line_length),
        "quality_score": float(score),
    }


def _is_effectively_empty_page(image_stats: dict, repair_mode: str) -> bool:
    if str(repair_mode or "off").strip().lower() != "auto":
        return False
    overall_dark = float(image_stats.get("overall_dark_ratio", 0.0))
    if overall_dark > EMPTY_PAGE_OVERALL_DARK_MAX:
        return False
    return all(
        float(image_stats.get(key, 0.0)) <= EMPTY_PAGE_BAND_DARK_MAX
        for key in (
            "top_dark_ratio",
            "bottom_dark_ratio",
            "top_third_dark_ratio",
            "middle_third_dark_ratio",
            "bottom_third_dark_ratio",
        )
    )


def _load_job_image(item: dict) -> Image.Image:
    return Image.open(item["image_path"]).convert("RGB")


def _generate_batch_outputs(
    llm,
    *,
    jobs: List[dict],
    prompt: str,
    batch_size: int,
    sampling_params,
) -> List[dict]:
    outputs_by_key: Dict[tuple[str, int], dict] = {}
    for batch in _batched(jobs, batch_size):
        prompt_batch = []
        opened_images: List[Image.Image] = []
        keys: List[tuple[str, int]] = []
        for item in batch:
            image = _load_job_image(item)
            opened_images.append(image)
            keys.append((str(item["stem"]), int(item["page_number"])))
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
    return [outputs_by_key[(str(item["stem"]), int(item["page_number"]))] for item in jobs]


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    jobs_to_run = _iter_pdf_jobs(input_dir, args.files, args.page_ranges)
    if not jobs_to_run:
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
    sampling_params = _sampling_params(
        args.max_new_tokens,
        enable_garbage_early_stop=str(args.repair_mode or "off").strip().lower() == "auto",
    )

    with tempfile.TemporaryDirectory(prefix="deepseek_vllm_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        doc_states: Dict[str, dict] = {}
        jobs: List[dict] = []
        plain_retry_jobs: List[dict] = []

        for job in jobs_to_run:
            pdf_path = Path(job["pdf_path"])
            stem = str(job["stem"])
            doc_start = time.perf_counter()
            render_start = time.perf_counter()
            images = _render_pages(
                pdf_path,
                args.max_pages,
                args.render_dpi,
                start_page=int(job["start_page"]),
                end_page=job["end_page"],
            )
            render_sec = time.perf_counter() - render_start
            total_pages = len(images)
            state = {
                "stem": stem,
                "source_name": str(job["source_name"]),
                "source_stem": str(job["source_stem"]),
                "source_start_page": int(job["start_page"]),
                "page_outputs": [""] * total_pages,
                "page_metrics": [None] * total_pages,
                "render_sec": float(render_sec),
                "doc_start": float(doc_start),
                "completed_pages": 0,
                "total_pages": total_pages,
            }
            doc_states[stem] = state
            _write_progress(output_dir, stem, [], total_pages, 0)
            for idx, image in enumerate(images):
                page_path = tmp_dir / f"{stem}_page_{idx + 1:04d}.png"
                image_stats = _image_content_stats(image)
                if _is_effectively_empty_page(image_stats, args.repair_mode):
                    state["page_metrics"][idx] = {
                        "page_number": int(idx + 1),
                        "infer_sec": 0.0,
                        "raw_chars": 0,
                        "final_chars": 0,
                        "first_pass_quality_score": 0.0,
                        "first_pass_letters": 0,
                        "first_pass_digits": 0,
                        "first_pass_pua_chars": 0,
                        "repair_strategy": "skip_empty",
                        "repair_reason": "empty_page",
                        "repair_attempted": False,
                        "repair_applied": False,
                        "empty_page_skipped": True,
                        "garbage_early_stop_applied": False,
                        **image_stats,
                    }
                    state["completed_pages"] = int(state["completed_pages"]) + 1
                    _write_progress(
                        output_dir,
                        stem,
                        [page for page in state["page_outputs"] if page],
                        int(state["total_pages"]),
                        int(state["completed_pages"]),
                    )
                    image.close()
                    continue
                image.save(page_path, format="PNG")
                image.close()
                jobs.append(
                    {
                        "stem": stem,
                        "page_number": int(idx + 1),
                        "image_path": page_path,
                        "image_stats": image_stats,
                    }
                )

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
            metric = {
                "page_number": int(item["page_number"]),
                "infer_sec": float(result["infer_sec"]),
                "raw_chars": int(len(raw_text.strip())),
                "final_chars": int(len(page_text.strip())),
                "first_pass_quality_score": float(quality["quality_score"]),
                "first_pass_letters": int(quality["letters"]),
                "first_pass_digits": int(quality["digits"]),
                "first_pass_pua_chars": int(quality["pua_chars"]),
                "repair_strategy": "plain" if bool(postprocess_metrics.get("early_stops", 0)) else "none",
                "repair_reason": "early_stop_markdown_garbage" if bool(postprocess_metrics.get("early_stops", 0)) else None,
                "repair_attempted": False,
                "repair_applied": False,
                "empty_page_skipped": False,
                "garbage_early_stop_applied": bool(postprocess_metrics.get("early_stops", 0)),
                **image_stats,
                **postprocess_metrics,
            }
            state["page_metrics"][item["page_number"] - 1] = metric
            if bool(postprocess_metrics.get("early_stops", 0)) and str(args.repair_mode or "off").strip().lower() == "auto":
                plain_retry_jobs.append(item)
            state["completed_pages"] = int(state["completed_pages"]) + 1
            _write_progress(
                output_dir,
                item["stem"],
                [page for page in state["page_outputs"] if page],
                int(state["total_pages"]),
                int(state["completed_pages"]),
            )

        if plain_retry_jobs:
            plain_repair_outputs = _generate_batch_outputs(
                llm,
                jobs=plain_retry_jobs,
                prompt=plain_prompt,
                batch_size=int(args.batch_size),
                sampling_params=sampling_params,
            )
            for result in plain_repair_outputs:
                item = result["item"]
                state = doc_states[item["stem"]]
                metric = state["page_metrics"][item["page_number"] - 1]
                repair_text, repair_postprocess = _postprocess_page_text(
                    str(result["raw_text"]),
                    prompt=plain_prompt,
                    content_debug=bool(args.content_debug),
                )
                if args.content_debug:
                    repair_text = f"<!-- page:{item['page_number']} -->\n{repair_text}".strip()
                metric["repair_attempted"] = True
                metric["repair_infer_sec"] = float(result["infer_sec"])
                metric["repair_raw_chars"] = int(len(str(result["raw_text"]).strip()))
                metric["repair_final_chars"] = int(len(repair_text.strip()))
                metric["repair_profile"] = "plain_ocr"
                metric["repair_quality_score"] = float(_text_quality_metrics(repair_text)["quality_score"])
                metric["repair_garbage_early_stop_applied"] = bool(repair_postprocess.get("early_stops", 0))
                metric.update({f"repair_{key}": value for key, value in repair_postprocess.items()})
                metric["infer_sec"] = float(metric["infer_sec"]) + float(result["infer_sec"])
                if repair_text.strip():
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
                "tiled_repairs": 0,
                "empty_pages_skipped": int(sum(1 for item in page_metrics if bool(item.get("empty_page_skipped")))),
                "pages_with_early_stop": int(sum(1 for item in page_metrics if bool(item.get("garbage_early_stop_applied")))),
            }
            _write_outputs(
                output_dir,
                stem,
                markdown,
                int(state["total_pages"]),
                extra_metrics={
                    "source_file": str(state["source_name"]),
                    "source_stem": str(state["source_stem"]),
                    "source_start_page": int(state["source_start_page"]),
                    "source_end_page": int(state["source_start_page"]) + max(0, len(page_metrics) - 1),
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
