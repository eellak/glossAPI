"""Compatibility wrappers for the reorganized corpus OCR package."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..ocr.deepseek.defaults import (
    DEFAULT_ATTN_BACKEND,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OCR_PROFILE,
    DEFAULT_RENDER_DPI,
    DEFAULT_REPAIR_MODE,
    DEFAULT_RUNTIME_BACKEND,
    DEFAULT_TARGET_BATCH_PAGES,
    DEFAULT_WORKERS_PER_GPU,
)
from .ocr.artifacts import (
    apply_ocr_success_updates as _apply_ocr_success_updates,
    build_ocr_stage_artifact_update as _build_ocr_stage_artifact_update,
)
from .ocr.config import normalize_ocr_request
from .ocr.math_pipeline import formula_enrich_from_json as _formula_enrich_from_json
from .ocr.math_pipeline import triage_math as _triage_math
from .ocr.math_worker import gpu_math_worker as _gpu_math_worker
from .ocr.pipeline import run_ocr_phase
from .ocr.targets import normalize_ocr_target_filenames as _normalize_ocr_target_filenames


class OcrMathPhaseMixin:
    def ocr(
        self,
        *,
        fix_bad: bool = True,
        mode: Optional[str] = None,
        backend: str = "deepseek",
        device: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        max_pages: Optional[int] = None,
        persist_engine: bool = True,
        limit: Optional[int] = None,
        dpi: Optional[int] = None,
        precision: Optional[str] = None,
        workers_per_gpu: int = DEFAULT_WORKERS_PER_GPU,
        runtime_backend: str = DEFAULT_RUNTIME_BACKEND,
        ocr_profile: str = DEFAULT_OCR_PROFILE,
        prompt_override: Optional[str] = None,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        base_size: Optional[int] = None,
        image_size: Optional[int] = None,
        crop_mode: Optional[bool] = None,
        render_dpi: Optional[int] = DEFAULT_RENDER_DPI,
        max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        vllm_batch_size: Optional[int] = None,
        gpu_memory_utilization: Optional[float] = DEFAULT_GPU_MEMORY_UTILIZATION,
        disable_fp8_kv: bool = False,
        repair_mode: str = DEFAULT_REPAIR_MODE,
        repair_exec_batch_target_pages: Optional[int] = None,
        repair_exec_batch_target_items: Optional[int] = None,
        scheduler: str = "auto",
        target_batch_pages: int = DEFAULT_TARGET_BATCH_PAGES,
        shard_pages: int = 0,
        shard_threshold_pages: int = 0,
        math_enhance: bool = True,
        math_targets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        math_batch_size: int = 8,
        math_dpi_base: int = 220,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
        force: Optional[bool] = None,
        reprocess_completed: Optional[bool] = None,
        skip_existing: Optional[bool] = None,
        content_debug: bool = False,
        CONTENT_DEBUG: Optional[bool] = None,
        internal_debug: bool = False,
        INTERNAL_DEBUG: Optional[bool] = None,
    ) -> None:
        """OCR and/or math enrichment with explicit mode control."""

        del limit, dpi
        request = normalize_ocr_request(
            logger=self.logger,
            fix_bad=fix_bad,
            mode=mode,
            backend=backend,
            device=device,
            model_dir=model_dir,
            max_pages=max_pages,
            persist_engine=persist_engine,
            precision=precision,
            workers_per_gpu=workers_per_gpu,
            runtime_backend=runtime_backend,
            ocr_profile=ocr_profile,
            prompt_override=prompt_override,
            attn_backend=attn_backend,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            render_dpi=render_dpi,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            vllm_batch_size=vllm_batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_fp8_kv=disable_fp8_kv,
            repair_mode=repair_mode,
            repair_exec_batch_target_pages=repair_exec_batch_target_pages,
            repair_exec_batch_target_items=repair_exec_batch_target_items,
            scheduler=scheduler,
            target_batch_pages=target_batch_pages,
            shard_pages=shard_pages,
            shard_threshold_pages=shard_threshold_pages,
            math_enhance=math_enhance,
            math_targets=math_targets,
            math_batch_size=math_batch_size,
            math_dpi_base=math_dpi_base,
            use_gpus=use_gpus,
            devices=devices,
            force=force,
            reprocess_completed=reprocess_completed,
            skip_existing=skip_existing,
            content_debug=content_debug,
            CONTENT_DEBUG=CONTENT_DEBUG,
            internal_debug=internal_debug,
            INTERNAL_DEBUG=INTERNAL_DEBUG,
        )
        if request is None:
            return
        run_ocr_phase(self, request)

    def formula_enrich_from_json(
        self,
        files: Optional[List[str]] = None,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        dpi_base: int = 220,
        targets_by_stem: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> None:
        return _formula_enrich_from_json(
            self,
            files=files,
            device=device,
            batch_size=batch_size,
            dpi_base=dpi_base,
            targets_by_stem=targets_by_stem,
        )

    def triage_math(self) -> None:
        return _triage_math(self)
