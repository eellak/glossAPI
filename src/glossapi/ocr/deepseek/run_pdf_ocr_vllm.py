"""CLI wrapper for DeepSeek-OCR-2 inference over PDF files using vLLM."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glossapi.ocr.deepseek.run_pdf_ocr_transformers import (
    DEFAULT_MAX_NEW_TOKENS,
    _join_page_outputs,
    _count_rendered_pages,
    _iter_pdf_jobs,
    _iter_rendered_pages,
    _postprocess_page_text,
    _profile_defaults,
    _split_page_outputs,
    _write_outputs,
    _write_progress,
)
from glossapi.ocr.deepseek.work_queue import (
    QUEUE_MAIN,
    QUEUE_REPAIR,
    STATUS_PENDING,
    STATUS_RUNNING,
    claim_next_batch,
    enqueue_batches,
    heartbeat_batch,
    mark_batch_done,
    mark_batch_failed,
    work_queue_counts,
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
    parser.add_argument("--work-db", default=None)
    parser.add_argument("--worker-id", default=None)
    parser.add_argument("--worker-runtime-file", default=None)
    parser.add_argument("--work-stale-after-sec", type=float, default=900.0)
    parser.add_argument("--work-heartbeat-sec", type=float, default=10.0)
    parser.add_argument("--work-max-attempts", type=int, default=2)
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


def _resolve_job_image(item: dict) -> Tuple[Image.Image, bool]:
    image = item.get("image")
    if isinstance(image, Image.Image):
        return image, False
    return Image.open(item["image_path"]).convert("RGB"), True


def _close_job_image(item: dict) -> None:
    image = item.pop("image", None)
    if isinstance(image, Image.Image):
        image.close()


def _empty_page_metric(*, page_number: int, image_stats: dict) -> dict:
    return {
        "page_number": int(page_number),
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
        "page_dropped_after_repair": False,
        "empty_page_skipped": True,
        "garbage_early_stop_applied": False,
        **image_stats,
    }


def _utc_now_iso(now_ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(now_ts) if now_ts is not None else time.time()))


def _write_worker_runtime(runtime_file: Optional[Path], state: dict) -> None:
    if runtime_file is None:
        return
    runtime_path = Path(runtime_file).expanduser().resolve()
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = _utc_now_iso()
    runtime_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_jobs_from_batch(input_dir: Path, batch: dict) -> List[dict]:
    files = list(batch.get("files") or [])
    page_ranges = list(batch.get("page_ranges") or [])
    return _iter_pdf_jobs(input_dir, files, page_ranges)


def _iter_selected_rendered_pages(
    pdf_path: Path,
    *,
    render_dpi: int,
    source_page_numbers: List[int],
):
    import fitz

    doc = fitz.open(pdf_path)
    try:
        zoom = float(render_dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for source_page_number in source_page_numbers:
            idx = int(source_page_number) - 1
            if idx < 0 or idx >= int(doc.page_count):
                raise ValueError(f"Requested page {source_page_number} outside document bounds for {pdf_path}")
            page = doc[idx]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            yield int(source_page_number), Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    finally:
        doc.close()


def _emit_progress(output_dir: Path, stem: str, state: dict) -> None:
    _write_progress(
        output_dir,
        stem,
        state["page_outputs"],
        int(state["total_pages"]),
        int(state["completed_pages"]),
    )


def _resolve_repair_disposition(*, repair_text: str, repair_postprocess: dict) -> dict:
    if bool(repair_postprocess.get("early_stops", 0)):
        return {
            "final_text": "",
            "repair_applied": False,
            "page_dropped_after_repair": True,
            "drop_reason": "repeat_garbage_cutoff",
        }
    if repair_text.strip():
        return {
            "final_text": repair_text,
            "repair_applied": True,
            "page_dropped_after_repair": False,
            "drop_reason": None,
        }
    return {
        "final_text": None,
        "repair_applied": False,
        "page_dropped_after_repair": False,
        "drop_reason": None,
    }


def _repair_summary_from_page_metrics(page_metrics: List[dict], repair_mode: str) -> dict:
    return {
        "repair_mode": str(repair_mode),
        "pages_flagged": int(sum(1 for item in page_metrics if str(item.get("repair_strategy")) != "none")),
        "pages_repaired": int(sum(1 for item in page_metrics if bool(item.get("repair_applied")))),
        "plain_repairs": int(
            sum(1 for item in page_metrics if str(item.get("repair_profile")) == "plain_ocr" and bool(item.get("repair_applied")))
        ),
        "tiled_repairs": 0,
        "pages_dropped_after_repeat_cutoff": int(sum(1 for item in page_metrics if bool(item.get("page_dropped_after_repair")))),
        "empty_pages_skipped": int(sum(1 for item in page_metrics if bool(item.get("empty_page_skipped")))),
        "pages_with_early_stop": int(sum(1 for item in page_metrics if bool(item.get("garbage_early_stop_applied")))),
    }


def _load_persisted_doc_state(output_dir: Path, stem: str) -> dict:
    markdown_path = output_dir / "markdown" / f"{stem}.md"
    metrics_path = output_dir / "json" / "metrics" / f"{stem}.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    page_count = int(metrics.get("page_count", 0))
    page_outputs = _split_page_outputs(markdown_path.read_text(encoding="utf-8")) if markdown_path.exists() else []
    if len(page_outputs) < page_count:
        page_outputs.extend([""] * (page_count - len(page_outputs)))
    elif len(page_outputs) > page_count:
        page_outputs = page_outputs[:page_count]
    metrics_by_page = {
        int(item["page_number"]): dict(item)
        for item in list(metrics.get("page_metrics") or [])
        if item is not None and "page_number" in item
    }
    page_metrics = [metrics_by_page.get(page_number) for page_number in range(1, page_count + 1)]
    extra_metrics = dict(metrics)
    extra_metrics.pop("page_count", None)
    extra_metrics.pop("model", None)
    return {
        "stem": stem,
        "page_outputs": page_outputs,
        "page_metrics": page_metrics,
        "total_pages": page_count,
        "extra_metrics": extra_metrics,
    }


def _build_repair_batches(*, doc_states: Dict[str, dict], retry_pages_by_stem: Dict[str, List[int]], origin_batch_id: int) -> List[dict]:
    batches: List[dict] = []
    for stem, retry_pages in sorted(retry_pages_by_stem.items()):
        unique_retry_pages = sorted({int(page_number) for page_number in retry_pages})
        if not unique_retry_pages:
            continue
        state = doc_states[stem]
        batches.append(
            {
                "queue_key": f"repair:{int(origin_batch_id)}:{stem}",
                "origin_batch_id": int(origin_batch_id),
                "stem": stem,
                "pdf_path": str(state["pdf_path"]),
                "source_name": str(state["source_name"]),
                "source_stem": str(state["source_stem"]),
                "source_start_page": int(state["source_start_page"]),
                "source_end_page": int(state["source_start_page"]) + max(0, int(state["total_pages"]) - 1),
                "repair_page_numbers": unique_retry_pages,
                "pages": int(len(unique_retry_pages)),
            }
        )
    return batches


def _run_vllm_batch(
    llm,
    *,
    batch: List[dict],
    prompt: str,
    sampling_params,
) -> List[dict]:
    if not batch:
        return []

    prompt_batch = []
    opened_images: List[Image.Image] = []
    keys: List[tuple[str, int]] = []
    for item in batch:
        image, should_close = _resolve_job_image(item)
        if should_close:
            opened_images.append(image)
        keys.append((str(item["stem"]), int(item["page_number"])))
        prompt_batch.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        )

    try:
        infer_start = time.perf_counter()
        batch_outputs = llm.generate(prompt_batch, sampling_params=sampling_params)
        infer_sec = time.perf_counter() - infer_start
    finally:
        for image in opened_images:
            image.close()

    per_item_sec = infer_sec / max(1, len(batch))
    results: List[dict] = []
    for item, key, output in zip(batch, keys, batch_outputs):
        raw_text = ""
        if getattr(output, "outputs", None):
            raw_text = str(output.outputs[0].text)
        results.append(
            {
                "key": key,
                "item": item,
                "raw_text": raw_text,
                "infer_sec": float(per_item_sec),
            }
        )
    return results


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
        for result in _run_vllm_batch(
            llm,
            batch=batch,
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            outputs_by_key[result["key"]] = {
                "item": result["item"],
                "raw_text": result["raw_text"],
                "infer_sec": result["infer_sec"],
            }
    return [outputs_by_key[(str(item["stem"]), int(item["page_number"]))] for item in jobs]


def _run_jobs_to_outputs(
    args: argparse.Namespace,
    *,
    jobs_to_run: List[dict],
    output_dir: Path,
    work_db: Optional[Path],
    origin_batch_id: Optional[int],
    llm,
    prompt: str,
    plain_prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    sampling_params,
) -> dict:
    batch_wall_start = time.perf_counter()
    batch_size = max(1, int(args.batch_size))
    doc_states: Dict[str, dict] = {}
    plain_retry_jobs: List[dict] = []
    retry_pages_by_stem: Dict[str, List[int]] = {}
    state_lock = threading.Lock()
    render_queue: "queue.Queue[dict | None]" = queue.Queue(maxsize=max(2, batch_size * 2))
    producer_errors: List[BaseException] = []
    first_infer_started_at: Optional[float] = None
    last_infer_completed_at: Optional[float] = None
    shared_repair_queue = (
        work_db is not None
        and origin_batch_id is not None
        and str(args.repair_mode or "off").strip().lower() == "auto"
    )

    def _render_producer() -> None:
        try:
            for job in jobs_to_run:
                pdf_path = Path(job["pdf_path"])
                stem = str(job["stem"])
                doc_start = time.perf_counter()
                total_pages = _count_rendered_pages(
                    pdf_path,
                    args.max_pages,
                    start_page=int(job["start_page"]),
                    end_page=job["end_page"],
                )
                state = {
                    "stem": stem,
                    "pdf_path": str(pdf_path),
                    "source_name": str(job["source_name"]),
                    "source_stem": str(job["source_stem"]),
                    "source_start_page": int(job["start_page"]),
                    "page_outputs": [""] * total_pages,
                    "page_metrics": [None] * total_pages,
                    "render_sec": 0.0,
                    "doc_start": float(doc_start),
                    "completed_pages": 0,
                    "total_pages": total_pages,
                }
                with state_lock:
                    doc_states[stem] = state
                    _emit_progress(output_dir, stem, state)

                render_start = time.perf_counter()
                for page_number, image in enumerate(
                    _iter_rendered_pages(
                        pdf_path,
                        args.max_pages,
                        args.render_dpi,
                        start_page=int(job["start_page"]),
                        end_page=job["end_page"],
                    ),
                    start=1,
                ):
                    image_stats = _image_content_stats(image)
                    if _is_effectively_empty_page(image_stats, args.repair_mode):
                        with state_lock:
                            state["page_metrics"][page_number - 1] = _empty_page_metric(
                                page_number=page_number,
                                image_stats=image_stats,
                            )
                            state["completed_pages"] = int(state["completed_pages"]) + 1
                            _emit_progress(output_dir, stem, state)
                        image.close()
                        continue
                    render_queue.put(
                        {
                            "stem": stem,
                            "page_number": int(page_number),
                            "image": image,
                            "image_stats": image_stats,
                        }
                    )

                with state_lock:
                    state["render_sec"] = float(time.perf_counter() - render_start)
        except BaseException as exc:  # pragma: no cover - exercised in integration flows
            producer_errors.append(exc)
        finally:
            render_queue.put(None)

    producer = threading.Thread(target=_render_producer, name="deepseek-vllm-render", daemon=True)
    producer.start()

    in_flight_batch: List[dict] = []
    producer_done = False
    queue_wait_timeout = 0.05
    queue_flush_marker = "__flush__"
    try:
        while not producer_done or in_flight_batch:
            if not producer_done and len(in_flight_batch) < batch_size:
                try:
                    item = render_queue.get(timeout=queue_wait_timeout)
                except queue.Empty:
                    item = queue_flush_marker if in_flight_batch else None
                if item is None:
                    if producer.is_alive():
                        continue
                    producer_done = True
                elif item == queue_flush_marker:
                    pass
                else:
                    in_flight_batch.append(item)
                    if len(in_flight_batch) < batch_size:
                        continue

            if not in_flight_batch:
                continue

            batch_infer_started_at = time.time()
            if first_infer_started_at is None:
                first_infer_started_at = batch_infer_started_at
            batch_results = _run_vllm_batch(
                llm,
                batch=in_flight_batch,
                prompt=prompt,
                sampling_params=sampling_params,
            )
            last_infer_completed_at = time.time()
            for result in batch_results:
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
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": bool(postprocess_metrics.get("early_stops", 0)),
                    **image_stats,
                    **postprocess_metrics,
                }
                with state_lock:
                    state["page_outputs"][item["page_number"] - 1] = page_text
                    state["page_metrics"][item["page_number"] - 1] = metric
                    state["completed_pages"] = int(state["completed_pages"]) + 1
                    _emit_progress(output_dir, item["stem"], state)

                if bool(postprocess_metrics.get("early_stops", 0)) and str(args.repair_mode or "off").strip().lower() == "auto":
                    if shared_repair_queue:
                        retry_pages_by_stem.setdefault(str(item["stem"]), []).append(int(item["page_number"]))
                        _close_job_image(item)
                    else:
                        plain_retry_jobs.append(item)
                else:
                    _close_job_image(item)

            in_flight_batch = []

        producer.join()
        if producer_errors:
            raise producer_errors[0]

        if plain_retry_jobs:
            repair_started_at = time.time()
            if first_infer_started_at is None:
                first_infer_started_at = repair_started_at
            plain_repair_outputs = _generate_batch_outputs(
                llm,
                jobs=plain_retry_jobs,
                prompt=plain_prompt,
                batch_size=batch_size,
                sampling_params=sampling_params,
            )
            last_infer_completed_at = time.time()
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
                metric["repair_profile"] = "plain_ocr"
                disposition = _resolve_repair_disposition(
                    repair_text=repair_text,
                    repair_postprocess=repair_postprocess,
                )
                repair_effective_text = disposition["final_text"] or ""
                metric["repair_final_chars"] = int(len(repair_effective_text.strip()))
                metric["repair_quality_score"] = float(_text_quality_metrics(repair_effective_text)["quality_score"])
                metric["repair_garbage_early_stop_applied"] = bool(repair_postprocess.get("early_stops", 0))
                metric["repair_applied"] = bool(disposition["repair_applied"])
                metric["page_dropped_after_repair"] = bool(disposition["page_dropped_after_repair"])
                if disposition["drop_reason"] is not None:
                    metric["drop_reason"] = str(disposition["drop_reason"])
                metric.update({f"repair_{key}": value for key, value in repair_postprocess.items()})
                metric["infer_sec"] = float(metric["infer_sec"]) + float(result["infer_sec"])
                with state_lock:
                    if disposition["final_text"] is not None:
                        state["page_outputs"][item["page_number"] - 1] = repair_effective_text
                        metric["final_chars"] = int(len(repair_effective_text.strip()))
                        _emit_progress(output_dir, item["stem"], state)
                _close_job_image(item)
    finally:
        for item in in_flight_batch:
            _close_job_image(item)
        for item in plain_retry_jobs:
            _close_job_image(item)

        for stem, state in doc_states.items():
            markdown = _join_page_outputs(state["page_outputs"]) if state["page_outputs"] else "[[Blank page]]"
            page_metrics = sorted(
                [item for item in state["page_metrics"] if item],
                key=lambda item: int(item["page_number"]),
            )
            repair_summary = _repair_summary_from_page_metrics(page_metrics, str(args.repair_mode))
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
        if shared_repair_queue and retry_pages_by_stem:
            enqueue_batches(
                work_db,
                queue_name=QUEUE_REPAIR,
                batches=_build_repair_batches(
                    doc_states=doc_states,
                    retry_pages_by_stem=retry_pages_by_stem,
                    origin_batch_id=int(origin_batch_id),
                ),
            )

    return {
        "docs": int(len(doc_states)),
        "pages": int(sum(int(state["total_pages"]) for state in doc_states.values())),
        "render_sec_total": float(sum(float(state["render_sec"]) for state in doc_states.values())),
        "infer_sec_total": float(
            sum(
                sum(float(item["infer_sec"]) for item in state["page_metrics"] if item is not None)
                for state in doc_states.values()
            )
        ),
        "first_infer_started_at": _utc_now_iso(first_infer_started_at) if first_infer_started_at is not None else None,
        "last_infer_completed_at": _utc_now_iso(last_infer_completed_at) if last_infer_completed_at is not None else None,
        "repair_batches_enqueued": int(sum(1 for pages in retry_pages_by_stem.values() if pages)),
        "batch_wall_time_sec": float(time.perf_counter() - batch_wall_start),
    }


def _run_repair_batch_to_outputs(
    args: argparse.Namespace,
    *,
    batch: dict,
    output_dir: Path,
    llm,
    plain_prompt: str,
    sampling_params,
) -> dict:
    batch_wall_start = time.perf_counter()
    stem = str(batch["stem"])
    state = _load_persisted_doc_state(output_dir, stem)
    source_start_page = int(batch["source_start_page"])
    repair_page_numbers = sorted({int(page_number) for page_number in list(batch.get("repair_page_numbers") or [])})
    if not repair_page_numbers:
        return {
            "docs": 1,
            "pages": 0,
            "render_sec_total": 0.0,
            "infer_sec_total": 0.0,
            "first_infer_started_at": None,
            "last_infer_completed_at": None,
            "batch_wall_time_sec": float(time.perf_counter() - batch_wall_start),
        }

    render_start = time.perf_counter()
    source_page_numbers = [source_start_page + page_number - 1 for page_number in repair_page_numbers]
    repair_jobs: List[dict] = []
    for source_page_number, image in _iter_selected_rendered_pages(
        Path(str(batch["pdf_path"])),
        render_dpi=int(args.render_dpi),
        source_page_numbers=source_page_numbers,
    ):
        repair_jobs.append(
            {
                "stem": stem,
                "page_number": int(source_page_number) - source_start_page + 1,
                "image": image,
            }
        )
    render_sec = float(time.perf_counter() - render_start)
    if not repair_jobs:
        return {
            "docs": 1,
            "pages": 0,
            "render_sec_total": render_sec,
            "infer_sec_total": 0.0,
            "first_infer_started_at": None,
            "last_infer_completed_at": None,
            "batch_wall_time_sec": float(time.perf_counter() - batch_wall_start),
        }

    first_infer_started_at = time.time()
    repair_outputs = _generate_batch_outputs(
        llm,
        jobs=repair_jobs,
        prompt=plain_prompt,
        batch_size=max(1, int(args.batch_size)),
        sampling_params=sampling_params,
    )
    last_infer_completed_at = time.time()
    try:
        for result in repair_outputs:
            item = result["item"]
            page_number = int(item["page_number"])
            metric = state["page_metrics"][page_number - 1]
            if metric is None:
                metric = {
                    "page_number": page_number,
                    "infer_sec": 0.0,
                    "raw_chars": 0,
                    "final_chars": 0,
                    "first_pass_quality_score": 0.0,
                    "first_pass_letters": 0,
                    "first_pass_digits": 0,
                    "first_pass_pua_chars": 0,
                    "repair_strategy": "plain",
                    "repair_reason": "early_stop_markdown_garbage",
                    "repair_attempted": False,
                    "repair_applied": False,
                    "page_dropped_after_repair": False,
                    "empty_page_skipped": False,
                    "garbage_early_stop_applied": False,
                }
                state["page_metrics"][page_number - 1] = metric
            repair_text, repair_postprocess = _postprocess_page_text(
                str(result["raw_text"]),
                prompt=plain_prompt,
                content_debug=bool(args.content_debug),
            )
            if args.content_debug:
                repair_text = f"<!-- page:{page_number} -->\n{repair_text}".strip()
            metric["repair_attempted"] = True
            metric["repair_infer_sec"] = float(result["infer_sec"])
            metric["repair_raw_chars"] = int(len(str(result["raw_text"]).strip()))
            metric["repair_profile"] = "plain_ocr"
            disposition = _resolve_repair_disposition(
                repair_text=repair_text,
                repair_postprocess=repair_postprocess,
            )
            repair_effective_text = disposition["final_text"] or ""
            metric["repair_final_chars"] = int(len(repair_effective_text.strip()))
            metric["repair_quality_score"] = float(_text_quality_metrics(repair_effective_text)["quality_score"])
            metric["repair_garbage_early_stop_applied"] = bool(repair_postprocess.get("early_stops", 0))
            metric["repair_applied"] = bool(disposition["repair_applied"])
            metric["page_dropped_after_repair"] = bool(disposition["page_dropped_after_repair"])
            if disposition["drop_reason"] is not None:
                metric["drop_reason"] = str(disposition["drop_reason"])
            metric.update({f"repair_{key}": value for key, value in repair_postprocess.items()})
            metric["infer_sec"] = float(metric.get("infer_sec", 0.0)) + float(result["infer_sec"])
            if disposition["final_text"] is not None:
                state["page_outputs"][page_number - 1] = repair_effective_text
                metric["final_chars"] = int(len(repair_effective_text.strip()))
            _close_job_image(item)
    finally:
        for item in repair_jobs:
            _close_job_image(item)

    page_metrics = sorted([item for item in state["page_metrics"] if item], key=lambda item: int(item["page_number"]))
    extra_metrics = dict(state["extra_metrics"])
    extra_metrics["repair_summary"] = _repair_summary_from_page_metrics(page_metrics, extra_metrics.get("repair_mode", args.repair_mode))
    extra_metrics["page_metrics"] = page_metrics
    extra_metrics["infer_sec_total"] = float(sum(float(item["infer_sec"]) for item in page_metrics))
    _write_outputs(
        output_dir,
        stem,
        _join_page_outputs(state["page_outputs"]) if state["page_outputs"] else "[[Blank page]]",
        int(state["total_pages"]),
        extra_metrics=extra_metrics,
    )
    return {
        "docs": 1,
        "pages": int(len(repair_page_numbers)),
        "render_sec_total": render_sec,
        "infer_sec_total": float(sum(float(result["infer_sec"]) for result in repair_outputs)),
        "first_infer_started_at": _utc_now_iso(first_infer_started_at),
        "last_infer_completed_at": _utc_now_iso(last_infer_completed_at),
        "batch_wall_time_sec": float(time.perf_counter() - batch_wall_start),
    }


def _queue_has_pending_or_running(counts: Dict[str, object], queue_name: str) -> bool:
    queue_counts = counts.get("by_queue", {}).get(queue_name, {})
    return int(queue_counts.get(STATUS_PENDING, 0)) > 0 or int(queue_counts.get(STATUS_RUNNING, 0)) > 0


def _claim_next_phase_batch(
    work_db: Path,
    *,
    worker_id: str,
    stale_after_sec: float,
) -> Tuple[Optional[str], Optional[Dict[str, object]], bool]:
    batch = claim_next_batch(
        work_db,
        worker_id=worker_id,
        stale_after_sec=stale_after_sec,
        queue_name=QUEUE_MAIN,
    )
    if batch is not None:
        return QUEUE_MAIN, batch, False

    counts = work_queue_counts(work_db)
    # Repairs are a distinct global phase: no worker should start repair work
    # while any first-pass batch is still pending or running elsewhere.
    if _queue_has_pending_or_running(counts, QUEUE_MAIN):
        return None, None, True

    batch = claim_next_batch(
        work_db,
        worker_id=worker_id,
        stale_after_sec=stale_after_sec,
        queue_name=QUEUE_REPAIR,
    )
    if batch is not None:
        return QUEUE_REPAIR, batch, False

    counts = work_queue_counts(work_db)
    if _queue_has_pending_or_running(counts, QUEUE_REPAIR):
        return None, None, True
    return None, None, False


def _run_work_queue(
    args: argparse.Namespace,
    *,
    input_dir: Path,
    output_dir: Path,
    llm,
    prompt: str,
    plain_prompt: str,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    sampling_params,
) -> int:
    work_db = Path(str(args.work_db)).expanduser().resolve()
    worker_id = str(args.worker_id or f"worker-{int(time.time())}")
    runtime_file = Path(str(args.worker_runtime_file)).expanduser().resolve() if args.worker_runtime_file else None
    heartbeat_interval = float(max(1.0, args.work_heartbeat_sec))
    stale_after_sec = float(max(30.0, args.work_stale_after_sec))
    max_attempts = int(max(1, args.work_max_attempts))
    runtime_state = {
        "worker_id": worker_id,
        "status": "starting",
        "started_at": _utc_now_iso(),
        "engine_ready_at": _utc_now_iso(),
        "current_batch_id": None,
        "current_queue_name": None,
        "completed_batches": [],
        "first_batch_started_at": None,
        "last_batch_finished_at": None,
    }
    _write_worker_runtime(runtime_file, runtime_state)

    while True:
        queue_name, batch, should_wait = _claim_next_phase_batch(
            work_db,
            worker_id=worker_id,
            stale_after_sec=stale_after_sec,
        )
        if batch is None:
            if should_wait:
                time.sleep(min(heartbeat_interval, 1.0))
                continue
            runtime_state["status"] = "complete"
            runtime_state["current_batch_id"] = None
            runtime_state["current_queue_name"] = None
            _write_worker_runtime(runtime_file, runtime_state)
            return 0

        batch_id = int(batch["batch_id"])
        heartbeat_stop = threading.Event()

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.wait(heartbeat_interval):
                heartbeat_batch(work_db, batch_id=batch_id, worker_id=worker_id)
                runtime_state["heartbeat_at"] = _utc_now_iso()
                _write_worker_runtime(runtime_file, runtime_state)

        heartbeat_thread = threading.Thread(target=_heartbeat_loop, name=f"{worker_id}-heartbeat", daemon=True)
        heartbeat_thread.start()
        try:
            runtime_state["status"] = f"running_{queue_name}"
            runtime_state["current_batch_id"] = batch_id
            runtime_state["current_queue_name"] = queue_name
            runtime_state["current_batch_pages"] = int(batch.get("pages", 0))
            runtime_state["heartbeat_at"] = _utc_now_iso()
            _write_worker_runtime(runtime_file, runtime_state)
            if queue_name == QUEUE_MAIN:
                result = _run_jobs_to_outputs(
                    args,
                    jobs_to_run=_build_jobs_from_batch(input_dir, batch),
                    output_dir=output_dir,
                    work_db=work_db,
                    origin_batch_id=batch_id,
                    llm=llm,
                    prompt=prompt,
                    plain_prompt=plain_prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    sampling_params=sampling_params,
                )
            else:
                result = _run_repair_batch_to_outputs(
                    args,
                    batch=batch,
                    output_dir=output_dir,
                    llm=llm,
                    plain_prompt=plain_prompt,
                    sampling_params=sampling_params,
                )
            if runtime_state["first_batch_started_at"] is None:
                runtime_state["first_batch_started_at"] = result.get("first_infer_started_at")
            runtime_state["last_batch_finished_at"] = result.get("last_infer_completed_at")
            runtime_state["completed_batches"].append(
                {
                    "batch_id": batch_id,
                    "queue_name": queue_name,
                }
            )
            mark_batch_done(work_db, batch_id=batch_id, worker_id=worker_id, result=result)
        except Exception as exc:
            runtime_state["status"] = "failed"
            runtime_state["current_batch_id"] = batch_id
            runtime_state["current_queue_name"] = queue_name
            runtime_state["last_error"] = str(exc)
            _write_worker_runtime(runtime_file, runtime_state)
            mark_batch_failed(
                work_db,
                batch_id=batch_id,
                worker_id=worker_id,
                error=str(exc),
                max_attempts=max_attempts,
            )
            raise
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=max(1.0, heartbeat_interval))
            runtime_state["current_batch_id"] = None
            runtime_state["current_queue_name"] = None
            _write_worker_runtime(runtime_file, runtime_state)


def main() -> int:
    args = _parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

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

    if args.work_db:
        return _run_work_queue(
            args,
            input_dir=input_dir,
            output_dir=output_dir,
            llm=llm,
            prompt=prompt,
            plain_prompt=plain_prompt,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            sampling_params=sampling_params,
        )

    jobs_to_run = _iter_pdf_jobs(input_dir, args.files, args.page_ranges)
    if not jobs_to_run:
        return 0
    _run_jobs_to_outputs(
        args,
        jobs_to_run=jobs_to_run,
        output_dir=output_dir,
        work_db=None,
        origin_batch_id=None,
        llm=llm,
        prompt=prompt,
        plain_prompt=plain_prompt,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        sampling_params=sampling_params,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
