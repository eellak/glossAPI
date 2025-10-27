"""OCR and math enrichment helpers split from Corpus."""
from __future__ import annotations

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
from ..gloss_section_classifier import GlossSectionClassifier
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch


class OcrMathPhaseMixin:
    def ocr(
        self,
        *,
        fix_bad: bool = True,
        mode: Optional[str] = None,
        backend: str = "rapidocr",
        device: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        max_pages: Optional[int] = None,
        persist_engine: bool = True,
        limit: Optional[int] = None,
        dpi: Optional[int] = None,        # reserved for future use
        precision: Optional[str] = None,  # reserved for future use ("fp16","bf16")
        # Integrated math enrichment controls
        math_enhance: bool = True,
        math_targets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        math_batch_size: int = 8,
        math_dpi_base: int = 220,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
        force: Optional[bool] = None,
        reprocess_completed: Optional[bool] = None,
        skip_existing: Optional[bool] = None,
        # Content debug: keep page separators and truncation markers when True
        content_debug: bool = False,
        CONTENT_DEBUG: Optional[bool] = None,
        # Back-compat aliases (deprecated):
        internal_debug: bool = False,
        INTERNAL_DEBUG: Optional[bool] = None,
    ) -> None:
        """OCR and/or math enrichment with explicit mode control.

        Parameters
        - mode: one of
          - 'ocr_bad': re‑OCR only documents flagged as bad by Rust cleaner (parquet 'filter' != 'ok').
          - 'math_only': run math enrichment from Docling JSON (generate JSON without OCR when missing).
          - 'ocr_bad_then_math': re‑OCR bad documents, then run math enrichment on those.
          If not provided, falls back to legacy flags (fix_bad, math_enhance):
            fix_bad and math_enhance -> 'ocr_bad_then_math';
            fix_bad only -> 'ocr_bad';
            math_enhance only -> 'math_only';
            neither -> no‑op.
        - backend: 'rapidocr' (default) uses the Docling + RapidOCR path via Phase‑1 extract().
                   'deepseek' uses the DeepSeek‑OCR path (no Docling JSON, math unsupported).
        - fix_bad: re-run OCR on documents marked bad by the cleaner (default True).
        - math_enhance: run math/code enrichment after OCR (default True).
        - force: [DEPRECATED] alias for fix_bad retained for backward compatibility.
        - reprocess_completed: when False, skip documents already flagged as successfully
          OCRed or math-enriched in metadata. Set True to force reprocessing. Defaults to False
          unless ``skip_existing`` overrides it.
        - skip_existing: legacy alias for ``reprocess_completed`` (``skip_existing=True`` equals
          ``reprocess_completed=False``). Prefer the explicit ``reprocess_completed`` toggle.
        """
        # Normalize backend
        backend_norm = str(backend or "rapidocr").strip().lower()
        if backend_norm not in {"rapidocr", "deepseek"}:
            raise ValueError("backend must be 'rapidocr' or 'deepseek'")

        # CONTENT_DEBUG override (preferred uppercase alias)
        # Priority: CONTENT_DEBUG > INTERNAL_DEBUG > content_debug/internal_debug flags
        if CONTENT_DEBUG is not None:
            content_debug = bool(CONTENT_DEBUG)
        elif INTERNAL_DEBUG is not None:
            content_debug = bool(INTERNAL_DEBUG)
        elif internal_debug:
            content_debug = True

        # Normalize mode from explicit value or legacy flags
        mode_norm = None
        fix_bad_effective = bool(fix_bad)
        if force is not None:
            try:
                self.logger.warning("Corpus.ocr(force=...) is deprecated; use fix_bad=... instead")
            except Exception:
                pass
            fix_bad_effective = bool(force)
        if mode:
            m = str(mode).strip().lower()
            if m in {"ocr_bad", "math_only", "ocr_bad_then_math"}:
                mode_norm = m
            else:
                self.logger.warning("Unknown mode '%s'; falling back to legacy flags", mode)
        if mode_norm is None:
            if fix_bad_effective and math_enhance:
                mode_norm = "ocr_bad_then_math"
            elif fix_bad_effective:
                mode_norm = "ocr_bad"
            elif math_enhance:
                mode_norm = "math_only"
            else:
                self.logger.info(
                    "OCR: no operation requested (enable fix_bad and/or math_enhance or set mode='ocr_bad'|'math_only'|'ocr_bad_then_math')"
                )
                return
        reprocess_explicit = reprocess_completed is not None
        reprocess_flag = bool(reprocess_completed) if reprocess_explicit else False
        if skip_existing is not None:
            skip_flag = bool(skip_existing)
            try:
                self.logger.warning(
                    "Corpus.ocr(skip_existing=...) is deprecated; use reprocess_completed=... instead."
                )
            except Exception:
                pass
            desired = not skip_flag
            if reprocess_explicit and desired != reprocess_flag:
                try:
                    self.logger.info(
                        "Corpus.ocr(): skip_existing=%s overrides reprocess_completed=%s (effective reprocess_completed=%s).",
                        skip_flag,
                        reprocess_flag,
                        desired,
                    )
                except Exception:
                    pass
            reprocess_flag = desired
        reprocess_completed = reprocess_flag

        # DeepSeek semantics note
        if backend_norm == "deepseek":
            try:
                self.logger.info(
                    "DeepSeek backend: Phase-2 math is not required; equations are included inline via OCR."
                )
            except Exception:
                pass
        # Identify bad documents from parquet (Rust cleaner output)
        bad_files: List[str] = []
        skipped_completed = 0
        skipped_skiplist = 0
        parquet_meta: Optional["pd.DataFrame"] = None
        ocr_done_files: List[str] = []
        ocr_done_stems: Set[str] = set()
        math_done_files: List[str] = []
        math_done_stems: Set[str] = set()
        try:
            from glossapi.parquet_schema import ParquetSchema
            parquet_schema = ParquetSchema({"url_column": self.url_column})
            parquet_path = self._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
            if parquet_path and parquet_path.exists():
                import pandas as _pd
                df = _pd.read_parquet(parquet_path)
                if "filename" in df.columns and "needs_ocr" in df.columns:
                    bad_files = df.loc[df["needs_ocr"] == True, "filename"].dropna().astype(str).tolist()
                else:
                    # No fallback: selection relies strictly on the 'needs_ocr' flag
                    # populated by the cleaner. If missing, we skip OCR selection.
                    bad_files = []
                ocr_done: Set[str] = set()
                if "ocr_success" in df.columns:
                    ocr_done_files = df.loc[df["ocr_success"].fillna(False), "filename"].dropna().astype(str).tolist()
                    ocr_done = {canonical_stem(str(name)) for name in ocr_done_files}
                    ocr_done_stems = set(ocr_done)
                if "math_enriched" in df.columns:
                    math_done_files = df.loc[df["math_enriched"].fillna(False), "filename"].dropna().astype(str).tolist()
                elif "enriched_math" in df.columns:
                    math_done_files = df.loc[df["enriched_math"].fillna(False), "filename"].dropna().astype(str).tolist()
                if math_done_files:
                    math_done_stems = {canonical_stem(str(name)) for name in math_done_files}
                if not reprocess_completed and ocr_done:
                    before = len(bad_files)
                    bad_files = [name for name in bad_files if canonical_stem(name) not in ocr_done]
                    removed = before - len(bad_files)
                    if removed:
                        skipped_completed = removed
                        self.logger.info(
                            "OCR: skipping %d already completed document(s) (reprocess_completed=False).",
                            removed,
                        )
                if reprocess_completed and mode_norm in {"ocr_bad", "ocr_bad_then_math"} and ocr_done_files:
                    pending = {str(f) for f in bad_files}
                    for fname in ocr_done_files:
                        if fname not in pending:
                            bad_files.append(fname)
                            pending.add(fname)
                parquet_meta = df
            else:
                parquet_meta = None
        except Exception:
            pass

        ocr_candidates_initial = len(bad_files)
        skiplist_path = _resolve_skiplist_path(self.output_dir, self.logger)
        skip_mgr = _SkiplistManager(skiplist_path, self.logger)
        skip_stems = skip_mgr.load()
        if skip_stems:
            before = len(bad_files)
            bad_files = [name for name in bad_files if canonical_stem(name) not in skip_stems]
            removed = before - len(bad_files)
            if removed:
                skipped_skiplist = removed
                self.logger.warning(
                    "Skip-list %s filtered %d document(s) from Phase-3 OCR.",
                    skiplist_path,
                    removed,
                )
        try:
            self.logger.info(
                "OCR targets: total=%d kept=%d skipped_completed=%d skipped_skiplist=%d",
                ocr_candidates_initial,
                len(bad_files),
                skipped_completed,
                skipped_skiplist,
            )
        except Exception:
            pass

        # Helper to run Phase‑2 enrichment over stems
        def _run_math(stems: List[str]) -> None:
            if not stems:
                self.logger.info("No Docling JSON found for math enrichment.")
                return
            initial_math_targets = len(stems)
            current_skips = skip_mgr.reload() if skip_mgr else set()
            if current_skips:
                before = len(stems)
                stems = [s for s in stems if s not in current_skips]
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
            try:
                self.logger.info(
                    "Math targets: total=%d kept=%d filtered_skiplist=%d",
                    initial_math_targets,
                    len(stems),
                    initial_math_targets - len(stems),
                )
            except Exception:
                pass
            local_targets = None
            if math_targets:
                local_targets = {s: math_targets.get(s) for s in stems if s in math_targets}
            if str(use_gpus).lower() == "multi":
                # Detect GPU devices
                devs = devices or []
                if not devs:
                    try:
                        import subprocess
                        p = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                        if p.returncode == 0 and p.stdout:
                            for line in p.stdout.splitlines():
                                if line.startswith("GPU "):
                                    try:
                                        idx = int(line.split(":", 1)[0].split()[1])
                                        devs.append(idx)
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
                else:
                    from multiprocessing import get_context

                    ctx = get_context("spawn")
                    work_q = ctx.Queue()
                    result_q = ctx.Queue()
                    manager = ctx.Manager()
                    status_map = manager.dict()
                    for s in stems:
                        work_q.put(s)

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
                        p = ctx.Process(
                            target=_gpu_math_worker,
                            args=(
                                dev_id,
                                str(self.input_dir),
                                str(self.output_dir),
                                work_q,
                                int(math_batch_size),
                                int(math_dpi_base),
                                device or "cuda",
                                local_targets or {},
                                result_q,
                                status_map,
                                str(marker_base),
                            ),
                        )
                        p.start()
                        procs.append(p)
                        active.append(p)
                        if p.pid is not None:
                            proc_gpu[p.pid] = dev_id

                    try:
                        last_summary = time.time()
                        while active:
                            for p in list(active):
                                p.join(timeout=0.05)
                                if p.is_alive():
                                    continue
                                active.remove(p)
                                if p in procs:
                                    procs.remove(p)
                                pid = p.pid or -1
                                gpu_id = proc_gpu.pop(pid, None)
                                exitcode = p.exitcode
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
                                    if stems_for_skip:
                                        skip_mgr.add(canonical_stem(s) for s in stems_for_skip)
                                    self.logger.warning(
                                        "Math worker GPU%s exited with %s",
                                        gpu_id,
                                        exitcode,
                                    )
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
                                            int(math_batch_size),
                                            int(math_dpi_base),
                                            device or "cuda",
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
                                    if stems_bad:
                                        skip_mgr.add(canonical_stem(s) for s in stems_bad)
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
                            skip_mgr.add(canonical_stem(s) for s in remaining_after_cap)
                            self.logger.error(
                                "No active math workers remain; skipped %d pending item(s)",
                                len(remaining_after_cap),
                            )
                    finally:
                        for p in procs:
                            if p.is_alive():
                                p.join()
                        try:
                            manager.shutdown()
                        except Exception:
                            pass
                        if worker_log_dir_env is not None:
                            os.environ["GLOSSAPI_WORKER_LOG_DIR"] = worker_log_dir_env
                        else:
                            os.environ.pop("GLOSSAPI_WORKER_LOG_DIR", None)
                    return
            # Single-GPU path
            self.formula_enrich_from_json(
                files=stems,
                device=(device or "cuda"),
                batch_size=int(math_batch_size),
                dpi_base=int(math_dpi_base),
                targets_by_stem=local_targets,
            )

        # Branches
        if mode_norm == "math_only":
            if not math_enhance:
                self.logger.info("OCR: fix_bad=False and math_enhance=False → nothing to do")
                return
            # Math-only: ensure JSON exists; if not, generate without OCR
            json_dir = self.output_dir / "json"
            stems: List[str] = []
            if json_dir.exists():
                stems = sorted({canonical_stem(p) for p in json_dir.glob("*.docling.json*")})
            # Do not generate layout JSON here; Phase‑1 is responsible for JSON artifacts.
            # Never run math on files that need OCR
            if bad_files:
                before = len(stems)
                bad_set = {canonical_stem(s) for s in bad_files}
                stems = [s for s in stems if s not in bad_set]
                removed = before - len(stems)
                if removed:
                    try:
                        self.logger.info(
                            "Math-only: skipping %d document(s) flagged for OCR",
                            removed,
                        )
                    except Exception:
                        pass
            if not reprocess_completed and stems and parquet_meta is not None:
                if math_done_stems:
                    before = len(stems)
                    stems = [s for s in stems if s not in math_done_stems]
                    removed = before - len(stems)
                    if removed:
                        self.logger.info(
                            "Math enrichment: skipping %d already enriched document(s) (reprocess_completed=False).",
                            removed,
                        )
            _run_math(stems)
            return

        # 'ocr_bad' and 'ocr_bad_then_math' paths: OCR bad files first
        if mode_norm in {"ocr_bad", "ocr_bad_then_math"} and not bad_files:
            self.logger.info("OCR: no bad documents flagged by cleaner; skipping OCR fix")
            if mode_norm == "ocr_bad_then_math":
                json_dir = self.output_dir / "json"
                stems = []
                if json_dir.exists():
                    stems = sorted({canonical_stem(p) for p in json_dir.glob("*.docling.json*")})
                _run_math(stems)
            return

        reran_ocr = False

        if mode_norm in {"ocr_bad", "ocr_bad_then_math"}:
            if backend_norm == "deepseek":
                # DeepSeek path: run OCR via dedicated runner (no Docling JSON)
                try:
                    from glossapi.ocr import deepseek_runner as _deepseek_runner  # type: ignore

                    _deepseek_runner.run_for_files(
                        self,
                        bad_files,
                        model_dir=Path(model_dir) if model_dir else None,
                        content_debug=bool(content_debug),
                    )
                except Exception as _e:
                    self.logger.error("DeepSeek OCR runner failed: %s", _e)
                    raise
            else:
                # RapidOCR/Docling path via Phase-1 extract
                self.extract(
                    input_format="pdf",
                    num_threads=os.cpu_count() or 4,
                    accel_type="CUDA",
                    force_ocr=True,
                    formula_enrichment=False,
                    code_enrichment=False,
                    filenames=bad_files,
                    skip_existing=False,
                    use_gpus=use_gpus,
                    devices=devices,
                    # Do not generate Docling JSON for OCR targets; math will skip them
                    export_doc_json=False,
                    emit_formula_index=False,
                    phase1_backend="docling",
                )
            reran_ocr = True
            # Update metadata to reflect successful OCR reruns
            try:
                from glossapi.parquet_schema import ParquetSchema as _ParquetSchema

                success_files: List[str] = []
                for _fname in bad_files:
                    stem = canonical_stem(_fname)
                    if (self.markdown_dir / f"{stem}.md").exists():
                        success_files.append(_fname)

                if success_files:
                    parquet_schema = _ParquetSchema({"url_column": self.url_column})
                    parquet_path = self._resolve_metadata_parquet(parquet_schema, ensure=True, search_input=True)
                    if parquet_path and parquet_path.exists():
                        import pandas as _pd

                        df_meta = _pd.read_parquet(parquet_path)
                        if "filename" in df_meta.columns:
                            if "filter" not in df_meta.columns:
                                df_meta["filter"] = "ok"
                            if "needs_ocr" not in df_meta.columns:
                                df_meta["needs_ocr"] = False
                            if "ocr_success" not in df_meta.columns:
                                df_meta["ocr_success"] = False
                            for _fname in success_files:
                                mask = df_meta["filename"].astype(str) == str(_fname)
                                if mask.any():
                                    df_meta.loc[mask, "filter"] = "ok"
                                    df_meta.loc[mask, "needs_ocr"] = False
                                    df_meta.loc[mask, "ocr_success"] = True
                            self._cache_metadata_parquet(parquet_path)
                            parquet_schema.write_metadata_parquet(df_meta, parquet_path)
                    # Keep sectioner in sync with newly recovered files
                    try:
                        stems = [canonical_stem(_f) for _f in success_files]
                        if hasattr(self, "good_files"):
                            for _stem in stems:
                                if _stem not in getattr(self, "good_files", []):
                                    self.good_files.append(_stem)
                    except Exception:
                        pass
            except Exception as _e:
                self.logger.warning("Failed to update OCR success metadata: %s", _e)

        if reran_ocr:
            try:
                self.logger.info("Re-running Rust cleaner after OCR rerun to refresh metrics")
                self.clean(
                    input_dir=self.markdown_dir,
                    drop_bad=False,
                )
            except Exception as _e:
                self.logger.warning("Cleaner refresh after OCR failed: %s", _e)

        if mode_norm == "ocr_bad_then_math":
            try:
                # Run math only on documents that do NOT require OCR
                json_dir = self.output_dir / "json"
                stems: List[str] = []
                if json_dir.exists():
                    stems = sorted({canonical_stem(p) for p in json_dir.glob("*.docling.json*")})
                bad_set = {canonical_stem(f) for f in bad_files}
                if stems:
                    stems = [s for s in stems if s not in bad_set]
                if not reprocess_completed:
                    if math_done_stems:
                        before = len(stems)
                        stems = [s for s in stems if s not in math_done_stems]
                        removed = before - len(stems)
                        if removed:
                            self.logger.info(
                                "Math enrichment: skipping %d already enriched document(s) (reprocess_completed=False).",
                                removed,
                            )
                if not stems:
                    self.logger.info("Math enrichment: no pending documents after filtering.")
                    return
                _run_math(stems)
            except Exception as _e:
                self.logger.warning("Phase‑2 enrichment after OCR failed: %s", _e)

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
        from ..math_enrich import enrich_from_docling_json  # type: ignore
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
                    stems = [s for s in stems if s in set(parq_stems)]
            except Exception:
                pass
        for stem in stems:
            try:
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
                        from ..json_io import load_docling_json  # type: ignore
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
                stats = enrich_from_docling_json(
                    jp, pdfp, out_md, out_map, device=device, batch_size=int(batch_size), dpi_base=int(dpi_base), targets=picks
                )
                self.logger.info("Phase‑2: %s -> items=%s accepted=%s time=%.2fs", stem, stats.get('items'), stats.get('accepted'), stats.get('time_sec'))
                # Update parquet with enrichment results
                try:
                    from ..triage import update_math_enrich_results  # type: ignore
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

    def triage_math(self) -> None:
        """Summarize per-page formula density and update routing recommendation in parquet.

        Scans `markdown_dir` for `{stem}.per_page.metrics.json`, computes summary metrics, and
        writes `formula_total`, `formula_avg_pp`, `formula_p90_pp`, `pages_with_formula`,
        `pages_total`, and `phase_recommended` into the consolidated download_results parquet
        if present.
        """
        try:
            from ..triage import summarize_math_density_from_metrics, recommend_phase, update_download_results_parquet
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
