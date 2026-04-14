"""OCR and math enrichment helpers split from Corpus."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._naming import canonical_stem
from ..gloss_downloader import GlossDownloader
from ..gloss_section import GlossSection
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
# Avoid importing classifier here; OCR/math phase does not require it at import time.
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch
from .ocr.config import OcrRequest, normalize_ocr_request
from .ocr.math_targets import discover_docling_json_stems, filter_math_only_stems
from .ocr.pipeline import run_ocr_phase
from .ocr.targets import build_ocr_selection


def _build_ocr_stage_artifact_update(
    *,
    markdown_dir: Path,
    metrics_dir: Path,
    stem: str,
) -> Optional[Dict[str, object]]:
    """Return direct OCR-owned artifact fields for one canonical OCR document.

    The OCR stage should hand off the same row identity that upstream stages
    produced, with corrected text embedded back into parquet. Markdown and
    metrics remain sidecars, but detached markdown alone is not the full stage
    contract.
    """

    markdown_path = Path(markdown_dir) / f"{stem}.md"
    if not markdown_path.exists():
        return None
    text_payload = markdown_path.read_text(encoding="utf-8")
    metrics_path = Path(metrics_dir) / f"{stem}.metrics.json"
    return {
        "text": text_payload,
        "ocr_markdown_relpath": str(Path("markdown") / markdown_path.name),
        "ocr_metrics_relpath": (
            str(Path("json") / "metrics" / metrics_path.name) if metrics_path.exists() else None
        ),
        "ocr_text_sha256": hashlib.sha256(text_payload.encode("utf-8")).hexdigest(),
    }


def _apply_ocr_success_updates(
    df_meta: pd.DataFrame,
    *,
    filenames: List[str],
    markdown_dir: Path,
    metrics_dir: Path,
    backend_norm: str,
) -> pd.DataFrame:
    """Apply only direct, obvious OCR-owned metadata updates to the parquet rows."""

    if "filename" not in df_meta.columns:
        return df_meta

    if "filter" not in df_meta.columns:
        df_meta["filter"] = "ok"
    if "needs_ocr" not in df_meta.columns:
        df_meta["needs_ocr"] = False
    if "ocr_success" not in df_meta.columns:
        df_meta["ocr_success"] = False
    if "extraction_mode" not in df_meta.columns:
        df_meta["extraction_mode"] = None

    direct_columns = ("text", "ocr_markdown_relpath", "ocr_metrics_relpath", "ocr_text_sha256")
    for column in direct_columns:
        if column not in df_meta.columns:
            df_meta[column] = None

    filename_series = df_meta["filename"].astype(str)
    stem_series = filename_series.map(canonical_stem)

    for fname in filenames:
        stem = canonical_stem(fname)
        mask = stem_series == stem
        if not bool(mask.any()):
            continue
        artifact_update = _build_ocr_stage_artifact_update(
            markdown_dir=markdown_dir,
            metrics_dir=metrics_dir,
            stem=stem,
        )
        df_meta.loc[mask, "filter"] = "ok"
        df_meta.loc[mask, "needs_ocr"] = False
        df_meta.loc[mask, "ocr_success"] = True
        if backend_norm == "deepseek":
            df_meta.loc[mask, "extraction_mode"] = "deepseek"
        if artifact_update is None:
            continue
        for column, value in artifact_update.items():
            df_meta.loc[mask, column] = value

    return df_meta


def _normalize_ocr_target_filenames(*, filenames: List[str], input_dir: Path) -> List[str]:
    """Collapse chunk-like metadata rows back to real OCR source files when possible."""

    source_by_stem: Dict[str, str] = {}
    try:
        for path in sorted(Path(input_dir).glob("*.pdf")):
            source_by_stem.setdefault(canonical_stem(path.name), path.name)
    except Exception:
        source_by_stem = {}

    normalized: List[str] = []
    seen: Set[str] = set()
    for fname in filenames:
        resolved = source_by_stem.get(canonical_stem(fname), str(fname))
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)
    return normalized


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
        if request.mode == "math_only":
            self._run_math_only_request(request)
            return
        run_ocr_phase(self, request)

    def _run_math_only_request(self, request: OcrRequest) -> None:
        selection = build_ocr_selection(
            self,
            mode=request.mode,
            reprocess_completed=request.reprocess_completed,
        )
        stems = discover_docling_json_stems(self.output_dir)
        stems = filter_math_only_stems(
            stems=stems,
            bad_files=selection.bad_files,
            math_done_stems=selection.math_done_stems,
            reprocess_completed=request.reprocess_completed,
            logger=self.logger,
        )
        self._run_math_targets(
            stems=stems,
            request=request,
            skip_mgr=selection.skip_mgr,
            skiplist_path=selection.skiplist_path,
        )

    def _run_math_targets(
        self,
        *,
        stems: List[str],
        request: OcrRequest,
        skip_mgr: Optional[_SkiplistManager],
        skiplist_path: Path,
    ) -> None:
        if not stems:
            self.logger.info("No Docling JSON found for math enrichment.")
            return

        initial_math_targets = len(stems)
        current_skips = skip_mgr.reload() if skip_mgr else set()
        if current_skips:
            before = len(stems)
            stems = [stem for stem in stems if stem not in current_skips]
            removed = before - len(stems)
            if removed:
                self.logger.warning(
                    "Skip-list %s filtered %d document(s) from Phase-2 math.",
                    skiplist_path,
                    removed,
                )
            if not stems:
                self.logger.info("All math targets filtered by skip-list; nothing to do.")
                return

        self.logger.info(
            "Math targets: total=%d kept=%d filtered_skiplist=%d",
            initial_math_targets,
            len(stems),
            initial_math_targets - len(stems),
        )

        local_targets = None
        if request.math_targets:
            local_targets = {stem: request.math_targets.get(stem) for stem in stems if stem in request.math_targets}

        if str(request.use_gpus).lower() != "multi":
            self.formula_enrich_from_json(
                files=stems,
                device=(request.device or "cuda"),
                batch_size=int(request.math_batch_size),
                dpi_base=int(request.math_dpi_base),
                targets_by_stem=local_targets,
            )
            return

        devs = list(request.devices or [])
        if not devs:
            try:
                proc = subprocess.run(
                    ["nvidia-smi", "-L"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0 and proc.stdout:
                    for line in proc.stdout.splitlines():
                        if line.startswith("GPU "):
                            try:
                                devs.append(int(line.split(":", 1)[0].split()[1]))
                            except Exception:
                                pass
            except Exception:
                pass
            if not devs:
                torch_mod = _maybe_import_torch()
                try:
                    if torch_mod is not None and getattr(torch_mod, "cuda", None) and torch_mod.cuda.is_available():
                        devs = list(range(torch_mod.cuda.device_count()))
                except Exception:
                    pass

        if not devs:
            msg = "Multi-GPU math requested but no GPUs detected; aborting math enhancement"
            self.logger.error(msg)
            raise RuntimeError(msg)

        from multiprocessing import get_context

        ctx = get_context("spawn")
        work_q = ctx.Queue()
        result_q = ctx.Queue()
        manager = ctx.Manager()
        status_map = manager.dict()
        for stem in stems:
            work_q.put(stem)

        worker_log_dir_env = os.environ.get("GLOSSAPI_WORKER_LOG_DIR")
        worker_log_dir_to_use = worker_log_dir_env
        if not worker_log_dir_to_use:
            default_worker_log_dir = self.logs_dir / "math_workers"
            try:
                default_worker_log_dir.mkdir(parents=True, exist_ok=True)
                worker_log_dir_to_use = str(default_worker_log_dir)
            except Exception as exc:
                self.logger.warning(
                    "Unable to prepare worker log directory %s: %s",
                    default_worker_log_dir,
                    exc,
                )
                worker_log_dir_to_use = None
        if worker_log_dir_to_use:
            os.environ["GLOSSAPI_WORKER_LOG_DIR"] = worker_log_dir_to_use
        marker_base = Path(worker_log_dir_to_use) if worker_log_dir_to_use else (self.logs_dir / "math_workers")
        try:
            marker_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        marker_files: Dict[int, Path] = {dev_id: marker_base / f"gpu{dev_id}.current" for dev_id in devs}

        procs: List[Any] = []
        active: List[Any] = []
        proc_gpu: Dict[int, int] = {}
        try:
            respawn_cap = int(os.environ.get("GLOSSAPI_MATH_RESPAWN_CAP", "5"))
        except Exception:
            respawn_cap = 5
        respawn_cap = max(0, respawn_cap)
        respawn_counts: Dict[int, int] = {dev_id: 0 for dev_id in devs}

        for dev_id in devs:
            proc = ctx.Process(
                target=_gpu_math_worker,
                args=(
                    dev_id,
                    str(self.input_dir),
                    str(self.output_dir),
                    work_q,
                    int(request.math_batch_size),
                    int(request.math_dpi_base),
                    request.device or "cuda",
                    local_targets or {},
                    result_q,
                    status_map,
                    str(marker_base),
                ),
            )
            proc.start()
            procs.append(proc)
            active.append(proc)
            if proc.pid is not None:
                proc_gpu[proc.pid] = dev_id

        try:
            last_summary = time.time()
            while active:
                for proc in list(active):
                    proc.join(timeout=0.05)
                    if proc.is_alive():
                        continue
                    active.remove(proc)
                    if proc in procs:
                        procs.remove(proc)
                    pid = proc.pid or -1
                    gpu_id = proc_gpu.pop(pid, None)
                    exitcode = proc.exitcode
                    stems_for_skip: List[str] = []
                    if gpu_id is not None:
                        current_entry = status_map.pop(gpu_id, None)
                        if current_entry:
                            if isinstance(current_entry, (list, tuple, set)):
                                entries = list(current_entry)
                            else:
                                entries = [current_entry]
                            stems_for_skip = [str(item) for item in entries if item]
                        marker_path = marker_files.get(gpu_id)
                        if marker_path:
                            try:
                                marker_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                    if exitcode not in (0, None) and gpu_id is not None:
                        if stems_for_skip and skip_mgr is not None:
                            skip_mgr.add(canonical_stem(stem) for stem in stems_for_skip)
                        self.logger.warning("Math worker GPU%s exited with %s", gpu_id, exitcode)
                        respawn_counts[gpu_id] = respawn_counts.get(gpu_id, 0) + 1
                        attempts = respawn_counts[gpu_id]
                        if respawn_cap and attempts > respawn_cap:
                            self.logger.error(
                                "Math worker GPU%s exceeded respawn cap (%s); not respawning",
                                gpu_id,
                                respawn_cap,
                            )
                            continue
                        replacement = ctx.Process(
                            target=_gpu_math_worker,
                            args=(
                                gpu_id,
                                str(self.input_dir),
                                str(self.output_dir),
                                work_q,
                                int(request.math_batch_size),
                                int(request.math_dpi_base),
                                request.device or "cuda",
                                local_targets or {},
                                result_q,
                                status_map,
                                str(marker_base),
                            ),
                        )
                        replacement.start()
                        procs.append(replacement)
                        active.append(replacement)
                        if replacement.pid is not None:
                            proc_gpu[replacement.pid] = gpu_id
                        continue

                while True:
                    try:
                        event = result_q.get_nowait()
                    except queue.Empty:
                        break
                    if not event:
                        continue
                    if event.get("event") == "math_batch":
                        stems_bad = event.get("problematic", [])
                        if stems_bad and skip_mgr is not None:
                            skip_mgr.add(canonical_stem(stem) for stem in stems_bad)
                        worker = event.get("worker")
                        try:
                            worker_gpu = int(worker)
                        except Exception:
                            worker_gpu = None
                        if worker_gpu is not None:
                            status_map.pop(worker_gpu, None)
                            marker_path = marker_files.get(worker_gpu)
                            if marker_path:
                                try:
                                    marker_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                    elif event.get("event") == "exit" and event.get("exitcode", 0) not in (0, None):
                        self.logger.warning(
                            "Math worker GPU%s reported exit code %s",
                            event.get("worker"),
                            event.get("exitcode"),
                        )

                now = time.time()
                if now - last_summary > 30:
                    try:
                        qsize = work_q.qsize()
                    except NotImplementedError:
                        qsize = -1
                    self.logger.info(
                        "Math progress: queue=%d active_workers=%d",
                        qsize,
                        len(active),
                    )
                    last_summary = now

                if not active:
                    break

            remaining_after_cap: List[str] = []
            try:
                while True:
                    item = work_q.get_nowait()
                    if isinstance(item, str) and item.strip():
                        remaining_after_cap.append(item)
            except queue.Empty:
                pass
            if remaining_after_cap:
                if skip_mgr is not None:
                    skip_mgr.add(canonical_stem(stem) for stem in remaining_after_cap)
                self.logger.error(
                    "No active math workers remain; skipped %d pending item(s)",
                    len(remaining_after_cap),
                )
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.join()
            try:
                manager.shutdown()
            except Exception:
                pass
            if worker_log_dir_env is not None:
                os.environ["GLOSSAPI_WORKER_LOG_DIR"] = worker_log_dir_env
            else:
                os.environ.pop("GLOSSAPI_WORKER_LOG_DIR", None)

    def formula_enrich_from_json(
        self,
        files: Optional[List[str]] = None,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        dpi_base: int = 220,
        targets_by_stem: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    ) -> None:
        """Phase‑2: Enrich math/code from Docling JSON without re‑running layout.

        Args:
            files: list of stems (without extension) to process; if None, auto‑discover.
            device: 'cuda'|'cpu'
            batch_size: batch size for recognizer
            dpi_base: base DPI for crops; actual DPI adapts per ROI size
        """
        from ..ocr import math as _math_pkg  # type: ignore

        try:
            enrich_from_docling_json = getattr(_math_pkg, "enrich_from_docling_json")
        except AttributeError as exc:
            raise RuntimeError("Math enrichment backend unavailable") from exc
        if not callable(enrich_from_docling_json):
            raise RuntimeError("Math enrichment backend missing 'enrich_from_docling_json'")
        json_dir = self.output_dir / "json"
        md_dir = self.markdown_dir
        dl_dir = self.output_dir / "downloads"
        stems: List[str] = []
        if files:
            stems = list(files)
        else:
            # Discover stems exclusively from json/
            candidates = []
            if json_dir.exists():
                candidates += list(json_dir.glob("*.docling.json")) + list(json_dir.glob("*.docling.json.zst"))
            stems = [p.name.replace(".docling.json.zst", "").replace(".docling.json", "") for p in candidates]
        if not stems:
            self.logger.info("No Docling JSON files found for Phase‑2 enrichment")
            return
        self.logger.info("Phase‑2: enriching %d document(s) from JSON", len(stems))
        # Pre-create sidecar entries for visibility even if enrichment fails later.
        try:
            self.logger.info("Phase‑2: placeholder sidecars for stems: %s", ",".join(stems))
            sc_dir = self.output_dir / 'sidecars' / 'math'
            sc_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            for _stem in stems:
                p = sc_dir / f"{_stem}.json"
                if not p.exists():
                    p.write_text(_json.dumps({"items": 0, "accepted": 0, "time_sec": 0.0}, ensure_ascii=False), encoding='utf-8')
                    try:
                        self.logger.info("Phase‑2: wrote placeholder sidecar: %s", p)
                    except Exception:
                        pass
        except Exception as _e:
            try:
                self.logger.warning("Phase‑2: failed to write placeholder sidecars: %s", _e)
            except Exception:
                pass
        # Parquet route: prefer stems marked for math in parquet if available
        try:
            from glossapi.parquet_schema import ParquetSchema
            ps = ParquetSchema({'url_column': self.url_column})
            pq = self._resolve_metadata_parquet(ps, ensure=True, search_input=True)
        except Exception:
            pq = None
        if pq and pq.exists():
            try:
                import pandas as _pd
                _df = _pd.read_parquet(pq)
                # derive stems from filename without extension
                _df['stem'] = _df['filename'].astype(str).str.replace(r"\.pdf$", "", regex=True)
                # prefer explicit phase or any formula signal (formula_total or math_equations_detected)
                _phase = _df['phase_recommended'].astype(str) == '2A' if 'phase_recommended' in _df.columns else ((_df['filename'] == _df['filename']) & False)
                _ft = (_df['formula_total'].fillna(0).astype('float') > 0) if 'formula_total' in _df.columns else ((_df['filename'] == _df['filename']) & False)
                _med = (_df['math_equations_detected'].fillna(0).astype('float') > 0) if 'math_equations_detected' in _df.columns else ((_df['filename'] == _df['filename']) & False)
                mask = _phase | _ft | _med
                parq_stems = _df.loc[mask, 'stem'].dropna().astype(str).tolist()
                if parq_stems:
                    try:
                        self.logger.info("Phase-2: parquet-selected stems: %s", ",".join(sorted(set(parq_stems))))
                    except Exception:
                        pass
                    stems = [s for s in stems if s in set(parq_stems)]
            except Exception:
                pass
        # Ensure placeholders exist for final target stems (post-parquet filter)
        try:
            sc_dir = self.output_dir / 'sidecars' / 'math'
            sc_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            for _stem in stems:
                p = sc_dir / f"{_stem}.json"
                if not p.exists():
                    p.write_text(_json.dumps({"items": 0, "accepted": 0, "time_sec": 0.0}, ensure_ascii=False), encoding='utf-8')
                    try:
                        self.logger.info("Phase‑2: ensured placeholder sidecar: %s", p)
                    except Exception:
                        pass
        except Exception:
            pass
        for stem in stems:
            try:
                try:
                    self.logger.info("Phase-2: processing stem=%s", stem)
                except Exception:
                    pass
                # Resolve JSON path under json/
                jp = None
                if (json_dir / f"{stem}.docling.json.zst").exists():
                    jp = json_dir / f"{stem}.docling.json.zst"
                elif (json_dir / f"{stem}.docling.json").exists():
                    jp = json_dir / f"{stem}.docling.json"
                if jp is None:
                    self.logger.warning("JSON not found for %s", stem)
                    continue
                # Resolve PDF path
                pdfp = None
                if (dl_dir / f"{stem}.pdf").exists():
                    pdfp = dl_dir / f"{stem}.pdf"
                else:
                    # Attempt from alongside JSON meta if present
                    try:
                        from ..ocr.utils.json_io import load_docling_json  # type: ignore
                        doc = load_docling_json(jp)
                        meta = getattr(doc, 'meta', {}) or {}
                        rp = meta.get('source_pdf_relpath') or ''
                        if rp:
                            pp = Path(rp)
                            if not pp.is_absolute():
                                pp = (self.output_dir / rp)
                            if pp.exists():
                                pdfp = pp
                    except Exception:
                        pass
                if pdfp is None:
                    self.logger.warning("PDF not found for %s; skipping", stem)
                    continue
                else:
                    try:
                        self.logger.info("Phase-2: PDF resolved for %s -> %s", stem, pdfp)
                    except Exception:
                        pass
                # Output paths: write enriched Markdown into the canonical markdown directory
                out_md = self.markdown_dir / f"{stem}.md"
                out_map = json_dir / f"{stem}.latex_map.jsonl"
                out_md.parent.mkdir(parents=True, exist_ok=True)
                json_dir.mkdir(parents=True, exist_ok=True)
                # Optional targeted picks for this stem
                picks = None
                try:
                    if targets_by_stem and stem in targets_by_stem:
                        picks = [(int(p), int(ix)) for (p, ix) in targets_by_stem.get(stem, [])]
                except Exception:
                    picks = None
                # Emit a placeholder sidecar early to ensure metrics artifact exists
                try:
                    # Write sidecar early (best-effort)
                    sc_dir = self.output_dir / 'sidecars' / 'math'
                    sc_dir.mkdir(parents=True, exist_ok=True)
                    import json as _json
                    (sc_dir / f"{stem}.json").write_text(
                        _json.dumps({"items": 0, "accepted": 0, "time_sec": 0.0}, ensure_ascii=False),
                        encoding='utf-8',
                    )
                except Exception as _e:
                    try:
                        self.logger.warning("Phase-2: failed to write placeholder for %s: %s", stem, _e)
                    except Exception:
                        pass
                try:
                    from ..ocr.utils.triage import update_math_enrich_results  # type: ignore
                    pq_path = self._get_cached_metadata_parquet()
                    if pq_path is None:
                        from ..parquet_schema import ParquetSchema as _ParquetSchema
                        pq_schema = _ParquetSchema({"url_column": self.url_column})
                        pq_path = self._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                    if pq_path is None:
                        pq_path = self.output_dir / 'download_results' / 'download_results.parquet'
                    self._cache_metadata_parquet(pq_path)
                    update_math_enrich_results(pq_path, stem, items=0, accepted=0, time_sec=0.0)
                except Exception:
                    pass

                stats = enrich_from_docling_json(
                    jp, pdfp, out_md, out_map, device=device, batch_size=int(batch_size), dpi_base=int(dpi_base), targets=picks
                )
                self.logger.info("Phase‑2: %s -> items=%s accepted=%s time=%.2fs", stem, stats.get('items'), stats.get('accepted'), stats.get('time_sec'))
                # Update parquet with enrichment results
                try:
                    from ..ocr.utils.triage import update_math_enrich_results  # type: ignore
                    pq_path = self._get_cached_metadata_parquet()
                    if pq_path is None:
                        from ..parquet_schema import ParquetSchema as _ParquetSchema

                        pq_schema = _ParquetSchema({"url_column": self.url_column})
                        pq_path = self._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                    if pq_path is None:
                        pq_path = self.output_dir / 'download_results' / 'download_results.parquet'
                    self._cache_metadata_parquet(pq_path)
                    update_math_enrich_results(pq_path, stem, items=int(stats.get('items', 0)), accepted=int(stats.get('accepted', 0)), time_sec=float(stats.get('time_sec', 0.0)))
                except Exception as _e:
                    self.logger.warning("Parquet math-enrich update failed for %s: %s", stem, _e)
            except Exception as e:
                self.logger.warning("Phase‑2 failed for %s: %s", stem, e)
                # Emit a minimal sidecar to aid downstream bookkeeping, even on failure.
                try:
                    from ..ocr.utils.triage import update_math_enrich_results  # type: ignore
                    pq_path = self._get_cached_metadata_parquet()
                    if pq_path is None:
                        from ..parquet_schema import ParquetSchema as _ParquetSchema
                        pq_schema = _ParquetSchema({"url_column": self.url_column})
                        pq_path = self._resolve_metadata_parquet(pq_schema, ensure=True, search_input=True)
                    if pq_path is None:
                        pq_path = self.output_dir / 'download_results' / 'download_results.parquet'
                    self._cache_metadata_parquet(pq_path)
                    update_math_enrich_results(pq_path, stem, items=0, accepted=0, time_sec=0.0)
                except Exception:
                    pass

    def triage_math(self) -> None:
        """Summarize per-page formula density and update routing recommendation in parquet.

        Scans `markdown_dir` for `{stem}.per_page.metrics.json`, computes summary metrics, and
        writes `formula_total`, `formula_avg_pp`, `formula_p90_pp`, `pages_with_formula`,
        `pages_total`, and `phase_recommended` into the consolidated download_results parquet
        if present.
        """
        try:
            from ..ocr.utils.triage import summarize_math_density_from_metrics, recommend_phase, update_download_results_parquet
        except Exception as e:
            self.logger.warning(f"Triage utilities unavailable: {e}")
            return
        md = Path(self.markdown_dir)
        if not md.exists():
            self.logger.warning("markdown_dir %s not found for triage", md)
            return
        # Support metrics stored under json/metrics (preferred) or markdown tree (legacy)
        metrics_files_set = set()
        json_metrics = self.output_dir / 'json' / 'metrics'
        if json_metrics.exists():
            metrics_files_set |= set(json_metrics.glob("*.per_page.metrics.json"))
        # Also scan markdown recursively for backward compatibility
        metrics_files_set |= set(md.rglob("*.per_page.metrics.json"))
        metrics_files = sorted(metrics_files_set)
        if not metrics_files:
            self.logger.info("No per-page metrics JSON found under %s", md)
            return
        for mpath in metrics_files:
            stem = mpath.name.replace(".per_page.metrics.json", "")
            try:
                summary = summarize_math_density_from_metrics(mpath)
                # Add max as helper
                summary["formula_max_pp"] = float(summary.get("formula_p90_pp", 0.0))
                rec = recommend_phase(summary)
                update_download_results_parquet(self.output_dir, stem, summary, rec, url_column=self.url_column)
                self.logger.info("Triage: %s -> %s", stem, rec)
            except Exception as e:
                self.logger.warning("Triage failed for %s: %s", stem, e)


def _gpu_math_worker(
    device_id: int,
    in_dir: str,
    out_dir: str,
    work_q,
    batch_size: int,
    dpi_base: int,
    device: str,
    targets_map: Dict[str, List[Tuple[int, int]]],
    result_q=None,
    status_map=None,
    marker_dir: str | None = None,
) -> None:
    import os as _os
    from pathlib import Path as _Path
    import sys as _sys

    def _ensure_thread_caps():
        caps = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
        for k, v in caps.items():
            _os.environ.setdefault(k, v)
        try:
            import sys as _sys

            _torch = _sys.modules.get("torch")
            if _torch is not None and hasattr(_torch, "set_num_threads"):
                _torch.set_num_threads(1)
        except Exception:
            pass

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _ensure_thread_caps()
    _status_proxy = status_map
    _marker_path = None
    if marker_dir:
        try:
            _marker_path = _Path(marker_dir).expanduser() / f"gpu{device_id}.current"
        except Exception:
            _marker_path = None
    # Worker GPU binding banner (prints by default; disable with GLOSSAPI_WORKER_LOG_VERBOSE=0)
    try:
        _verbose = str(_os.environ.get("GLOSSAPI_WORKER_LOG_VERBOSE", "1")).strip().lower()
        if _verbose not in ("0", "false", "no", "off", ""):  # default on
            try:
                import sys as _sys, importlib

                _torch = _sys.modules.get("torch")
                if _torch is None:
                    try:
                        _torch = importlib.import_module("torch")  # type: ignore
                    except Exception:
                        _torch = None
                if _torch is not None:
                    _torch_name = (
                        _torch.cuda.get_device_name(0)
                        if getattr(_torch, "cuda", None) and _torch.cuda.is_available()
                        else "no-cuda"
                    )
                else:
                    _torch_name = "unloaded"
            except Exception:
                _torch_name = "unknown"
            try:
                import onnxruntime as _ort  # type: ignore

                _ort_prov = _ort.get_available_providers()
            except Exception:
                _ort_prov = []
            try:
                import subprocess as _sp

                _nvsmi = _sp.run(
                    ["nvidia-smi", "-L"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                _phys = (
                    _nvsmi.stdout.splitlines()[0].strip()
                    if _nvsmi.returncode == 0 and _nvsmi.stdout
                    else ""
                )
            except Exception:
                _phys = ""
            try:
                print(
                    f"[MATH GPU{device_id}] bound: CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES','')} "
                    f"pid={_os.getpid()} torch={_torch_name} ORT={_ort_prov}"
                )
                if _phys:
                    print(f"[MATH GPU{device_id}] physical: {_phys}")
            except Exception:
                pass
    except Exception:
        pass
    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            import sys as _sys, pathlib as _pl

            _sys.path.insert(
                0, str((_pl.Path(out_dir).resolve().parents[1] / "src").resolve())
            )
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as _e:
            try:
                print(f"[MATH GPU{device_id}] Cannot import glossapi in worker: {_e}")
            except Exception:
                pass
            if result_q is not None:
                try:
                    result_q.put(
                        {
                            "event": "exit",
                            "worker": device_id,
                            "exitcode": 1,
                            "pid": _os.getpid(),
                        }
                    )
                except Exception:
                    pass
            _sys.exit(1)
    c = _Corpus(input_dir=in_dir, output_dir=out_dir)
    B = max(1, int(batch_size))
    exit_code = 0
    import queue as _queue

    def _report_failure(err: Exception, items: List[str], *, fatal: bool = False) -> None:
        nonlocal exit_code
        try:
            print(f"[MATH GPU{device_id}] Batch failed ({len(items)}): {err}")
        except Exception:
            pass
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "math_batch",
                        "worker": device_id,
                        "problematic": list(items),
                        "pid": _os.getpid(),
                        "error": str(err),
                    }
                )
            except Exception:
                pass
        if fatal:
            exit_code = 1

    def _quarantine_items(items: List[str]) -> None:
        if not items:
            return
        try:
            downloads_root = _Path(out_dir) / "downloads"
            if not downloads_root.exists():
                return
            quarantine_root = downloads_root / "problematic_math"
            quarantine_root.mkdir(parents=True, exist_ok=True)
            json_root = _Path(out_dir) / "json"
            json_quarantine = json_root / "problematic_math"
            try:
                json_quarantine.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            import shutil as _shutil

            for stem in items:
                try:
                    name = str(stem).strip()
                    if not name:
                        continue
                    pdf_src = downloads_root / f"{name}.pdf"
                    if pdf_src.exists():
                        dst = quarantine_root / pdf_src.name
                        if not dst.exists():
                            _shutil.copy2(pdf_src, dst)
                    if json_root.exists():
                        for suffix in (
                            ".docling.json.zst",
                            ".docling.json",
                            ".latex_map.jsonl",
                        ):
                            candidate = json_root / f"{name}{suffix}"
                            if candidate.exists():
                                dst = json_quarantine / candidate.name
                                if not dst.exists():
                                    _shutil.copy2(candidate, dst)
                except Exception:
                    pass
        except Exception:
            pass

    def _update_current(batch_items: List[str]) -> None:
        if not batch_items:
            return
        if _status_proxy is not None:
            try:
                _status_proxy[device_id] = list(batch_items)
            except Exception:
                pass
        if _marker_path is not None:
            try:
                _marker_path.write_text(
                    "\n".join(batch_items) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

    def _clear_current() -> None:
        if _status_proxy is not None:
            try:
                _status_proxy.pop(device_id, None)
            except Exception:
                pass
        if _marker_path is not None:
            try:
                _marker_path.unlink(missing_ok=True)
            except Exception:
                pass

    try:
        while True:
            try:
                nm = work_q.get(timeout=1.0)
            except _queue.Empty:
                break
            if not isinstance(nm, str):
                continue
            stem = nm.strip()
            if not stem:
                continue
            pending = [stem]
            _update_current(pending)
            _targets = (
                {stem: targets_map.get(stem)} if targets_map and stem in targets_map else None
            )
            try:
                c.formula_enrich_from_json(
                    files=pending,
                    device=(device or "cuda"),
                    batch_size=B,
                    dpi_base=int(dpi_base),
                    targets_by_stem=_targets,
                )
            except Exception as _e:
                _report_failure(_e, pending, fatal=False)
                _quarantine_items(pending)
            finally:
                _clear_current()
    except Exception as _unexpected:
        if exit_code == 0:
            exit_code = 1
        try:
            print(f"[MATH GPU{device_id}] Unexpected error: {_unexpected}")
        except Exception:
            pass
    finally:
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "exit",
                        "worker": device_id,
                        "exitcode": exit_code,
                        "pid": _os.getpid(),
                    }
                )
            except Exception:
                pass
        _sys.exit(exit_code)
