"""Backend dispatch helpers for corpus OCR orchestration."""

from __future__ import annotations

from ...ocr.deepseek import runner as _deepseek_runner
from .config import OcrRequest
from .context import CorpusOcrContext


def run_deepseek_ocr(
    context: CorpusOcrContext,
    *,
    request: OcrRequest,
    filenames: list[str],
) -> None:
    _deepseek_runner.run_for_files(
        context,
        filenames,
        model_dir=request.model_dir,
        max_pages=request.max_pages,
        persist_engine=request.persist_engine,
        precision=request.precision,
        device=request.device,
        use_gpus=request.use_gpus,
        devices=request.devices,
        workers_per_gpu=request.workers_per_gpu,
        runtime_backend=request.runtime_backend,
        ocr_profile=request.ocr_profile,
        prompt_override=request.prompt_override,
        attn_backend=request.attn_backend,
        base_size=request.base_size,
        image_size=request.image_size,
        crop_mode=request.crop_mode,
        render_dpi=request.render_dpi,
        max_new_tokens=request.max_new_tokens,
        repetition_penalty=request.repetition_penalty,
        no_repeat_ngram_size=request.no_repeat_ngram_size,
        vllm_batch_size=request.vllm_batch_size,
        gpu_memory_utilization=request.gpu_memory_utilization,
        disable_fp8_kv=request.disable_fp8_kv,
        repair_mode=request.repair_mode,
        repair_exec_batch_target_pages=request.repair_exec_batch_target_pages,
        repair_exec_batch_target_items=request.repair_exec_batch_target_items,
        scheduler=request.scheduler,
        target_batch_pages=request.target_batch_pages,
        shard_pages=request.shard_pages,
        shard_threshold_pages=request.shard_threshold_pages,
        content_debug=request.content_debug,
    )
