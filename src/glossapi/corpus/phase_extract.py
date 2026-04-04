"""Phase-1 extraction helpers split from Corpus."""
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .._naming import canonical_stem
from ..gloss_downloader import GlossDownloader
# Avoid importing section/classifier here; extract phase does not use them.
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch as _maybe_import_torch_fallback

# Wrapper to allow tests to monkeypatch glossapi.corpus._maybe_import_torch.
def _maybe_import_torch(force: bool = False):
    try:
        import glossapi.corpus as _corpus_pkg  # defer import to avoid cycles
        pkg_fn = getattr(_corpus_pkg, "_maybe_import_torch", None)
        if callable(pkg_fn):
            return pkg_fn(force=force)
    except Exception:
        pass
    return _maybe_import_torch_fallback(force=force)


def _resolve_docling_max_batch_files(default: int = 1) -> int:
    """Resolve the per-worker Docling document batch size for Phase-1 extraction.

    GlossAPI keeps the default conservative because fresh GPU nodes have been
    more sensitive to bootstrap/runtime drift than to raw scheduler limits.
    Strong GPUs can still be benchmarked explicitly by raising this knob.
    """

    fallback = max(1, int(default))
    raw = os.getenv("GLOSSAPI_DOCLING_MAX_BATCH_FILES")
    if not raw:
        return fallback
    try:
        return max(1, int(raw))
    except Exception:
        return fallback


def _resolve_docling_batch_target_pages(default: int = 256) -> int:
    """Resolve the target page budget per queued Docling extraction work item."""

    fallback = max(1, int(default))
    raw = os.getenv("GLOSSAPI_DOCLING_BATCH_TARGET_PAGES")
    if not raw:
        return fallback
    try:
        return max(1, int(raw))
    except Exception:
        return fallback


def _estimate_extract_work_pages(path: Path) -> int:
    """Best-effort PDF page estimate used for Phase-1 queue packing."""

    suffix = path.suffix.lower()
    if suffix != ".pdf":
        return 1

    try:
        import pypdfium2 as pdfium  # type: ignore

        pdf = pdfium.PdfDocument(str(path))
        try:
            return max(1, int(len(pdf)))
        finally:
            close = getattr(pdf, "close", None)
            if callable(close):
                close()
    except Exception:
        pass

    for module_name, attr_name in (
        ("pypdf", "PdfReader"),
        ("PyPDF2", "PdfReader"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr_name])
            reader_cls = getattr(module, attr_name)
            reader = reader_cls(str(path))
            return max(1, int(len(reader.pages)))
        except Exception:
            continue

    return 1


def _build_extract_work_items(
    paths: Iterable[Path],
    *,
    max_batch_files: int,
    target_batch_pages: int,
    long_pdf_page_threshold: int = 600,
    page_counter: Optional[Callable[[Path], int]] = None,
) -> List[List[Path]]:
    """Pack extraction work into steadier page-budget batches for multi-GPU runs."""

    files = [Path(path) for path in paths]
    if not files:
        return []

    max_files = max(1, int(max_batch_files))
    target_pages = max(1, int(target_batch_pages))
    long_threshold = max(1, int(long_pdf_page_threshold))
    counter = page_counter or _estimate_extract_work_pages

    packed: List[Tuple[List[Path], int]] = []
    standalone: List[Tuple[List[Path], int]] = []

    for path in files:
        try:
            est_pages = max(1, int(counter(path)))
        except Exception:
            est_pages = 1

        if path.suffix.lower() == ".pdf" and est_pages > long_threshold:
            standalone.append(([path], est_pages))
            continue

        best_idx: Optional[int] = None
        best_leftover: Optional[int] = None
        for idx, (bundle_paths, bundle_pages) in enumerate(packed):
            if len(bundle_paths) >= max_files:
                continue
            new_pages = bundle_pages + est_pages
            if bundle_paths and new_pages > target_pages:
                continue
            leftover = max(0, target_pages - new_pages)
            if best_leftover is None or leftover < best_leftover:
                best_idx = idx
                best_leftover = leftover
        if best_idx is None:
            packed.append(([path], est_pages))
        else:
            packed[best_idx][0].append(path)
            packed[best_idx] = (packed[best_idx][0], packed[best_idx][1] + est_pages)

    work_items = standalone + packed
    work_items.sort(key=lambda item: item[1], reverse=True)
    return [bundle_paths for bundle_paths, _ in work_items]


def _resolve_docling_queue_policy(extractor: Any | None = None) -> Tuple[int, int]:
    """Return the Docling queue packing knobs the multi-GPU planner should use."""

    max_batch_files = _resolve_docling_max_batch_files()
    long_pdf_page_threshold = 600
    if extractor is None:
        return max_batch_files, long_pdf_page_threshold
    try:
        max_batch_files = max(1, int(getattr(extractor, "max_batch_files", max_batch_files)))
    except Exception:
        max_batch_files = _resolve_docling_max_batch_files()
    try:
        long_pdf_page_threshold = max(
            1, int(getattr(extractor, "long_pdf_page_threshold", long_pdf_page_threshold))
        )
    except Exception:
        long_pdf_page_threshold = 600
    return max_batch_files, long_pdf_page_threshold


class ExtractPhaseMixin:
    def prime_extractor(
        self,
        input_format: str = "all",
        num_threads: Optional[int] = None,
        accel_type: str = "CUDA",
        *,
        force_ocr: bool = False,
        formula_enrichment: bool = False,
        code_enrichment: bool = False,
        use_cls: bool = False,
        benchmark_mode: bool = False,
        export_doc_json: bool = True,
        emit_formula_index: bool = False,
        phase1_backend: str = "auto",
    ) -> None:
        """Prepare and initialize the underlying extractor once per configuration.

        Builds the Docling converter only if the effective configuration changed.
        """
        # Lazy import + instantiate GlossExtract
        if self.extractor is None:
            from ..gloss_extract import GlossExtract
            self.extractor = GlossExtract(url_column=self.url_column)

        # Propagate toggles used by Phase‑1 helpers
        try:
            setattr(self.extractor, "export_doc_json", bool(export_doc_json))
            setattr(self.extractor, "emit_formula_index", bool(emit_formula_index))
        except Exception as e:
            self.logger.debug(f"[Extract Phase] Failed to propagate extractor toggles: {e}")
            pass
        # Resolve backend preference (safe vs docling)
        backend_choice = self._resolve_phase1_backend(
            phase1_backend,
            force_ocr=bool(force_ocr),
            formula_enrichment=bool(formula_enrichment),
            code_enrichment=bool(code_enrichment),
        )
        self._phase1_backend = backend_choice

        # Threads
        try:
            if num_threads is not None:
                threads_effective = int(num_threads)
            else:
                cpu_total = max(1, os.cpu_count() or 0)
                threads_effective = min(cpu_total, 2)
                threads_effective = max(2, threads_effective)
        except Exception:
            threads_effective = int(num_threads) if isinstance(num_threads, int) else 2
        self.extractor.enable_accel(threads=threads_effective, type=accel_type)

        # Images scale env default
        try:
            import os as _os
            images_scale_env = _os.getenv("GLOSSAPI_IMAGES_SCALE", "1.25")
        except Exception:
            images_scale_env = "1.25"

        if force_ocr:
            self.logger.warning(
                "Corpus.extract(force_ocr=True) is deprecated and no longer executes OCR. "
                "Use Corpus.ocr(backend='deepseek') for OCR remediation."
            )

        # Hard GPU preflight before we attempt to build OCR/enrichment pipelines
        self._gpu_preflight(
            accel_type=accel_type,
            require_ocr=False,
            require_math=bool(formula_enrichment or code_enrichment),
            require_backend_gpu=(backend_choice == "docling"),
        )

        # Configure batch/backend policy based on resolved choice
        if backend_choice == "docling":
            # Keep docling runs conservative by default, but expose an explicit
            # Phase-1 tuning hook for benchmark nodes and strong GPUs.
            self.extractor.configure_batch_policy(
                "docling",
                max_batch_files=_resolve_docling_max_batch_files(),
                prefer_safe_backend=False,
            )
        else:
            self.extractor.configure_batch_policy("safe", max_batch_files=1, prefer_safe_backend=True)

        # Ensure converter exists (reuse when unchanged)
        self.extractor.ensure_extractor(
            enable_ocr=False,
            force_full_page_ocr=False,
            formula_enrichment=bool(formula_enrichment),
            code_enrichment=bool(code_enrichment),
            images_scale=float(images_scale_env),
            use_cls=bool(use_cls),
            profile_timings=not bool(benchmark_mode),
        )

    def _resolve_phase1_backend(
        self,
        requested: Optional[str],
        *,
        force_ocr: bool,
        formula_enrichment: bool,
        code_enrichment: bool,
    ) -> str:
        valid = {"auto", "safe", "docling"}
        choice = (requested or "auto").strip().lower()
        if choice not in valid:
            raise ValueError(
                f"Invalid phase1_backend='{requested}'. Expected one of: 'auto', 'safe', 'docling'."
            )
        needs_gpu = bool(formula_enrichment or code_enrichment)
        if choice == "auto":
            choice = "docling" if needs_gpu else "safe"
        if choice == "safe" and needs_gpu:
            self.logger.info(
                "Phase-1 backend 'safe' overridden to 'docling' because math/code enrichment was requested."
            )
            choice = "docling"
        return choice

    def _gpu_preflight(
        self,
        *,
        accel_type: str,
        require_ocr: bool,
        require_math: bool,
        require_backend_gpu: bool = False,
    ) -> None:
        """Abort early when GPU-backed Docling work is requested but CUDA is unavailable."""
        if not (require_ocr or require_math or require_backend_gpu):
            return

        instructions = (
            "GPU-backed Docling extraction and math enrichment require CUDA-enabled torch. "
            "Install the CUDA wheels and ensure NVIDIA drivers expose the desired devices."
        )

        # Enforce non-CPU accelerator selection when OCR/math is forced
        accel_lower = str(accel_type or "").strip().lower()
        if accel_lower.startswith("cpu"):
            raise RuntimeError(
                "GPU-backed Docling extraction was requested but accel_type='CPU'. "
                f"{instructions}"
            )

        torch_mod = _maybe_import_torch(force=True)
        if torch_mod is None or not getattr(torch_mod, "cuda", None) or not torch_mod.cuda.is_available():
            raise RuntimeError(
                "Torch CUDA is not available but GPU-backed Docling extraction/math was requested. "
                "Install the CUDA wheel and ensure CUDA drivers/devices are visible."
            )

        device_count = torch_mod.cuda.device_count()
        if device_count < 1:
            raise RuntimeError(
                "Torch CUDA initialised but reports zero devices visible. "
                "Set CUDA_VISIBLE_DEVICES appropriately before running GPU OCR."
            )
        device_names = []
        for idx in range(device_count):
            try:
                device_names.append(torch_mod.cuda.get_device_name(idx))
            except Exception:
                device_names.append(f"cuda:{idx}")

        if not self._gpu_banner_logged:
            self.logger.info(
                "GPU preflight: using torch-backed Docling extraction; ensure CUDA drivers are available."
            )
            self._gpu_banner_logged = True

        self.logger.info(
            "GPU preflight OK: torch_devices=%s",
            ", ".join(device_names) or "<none>",
        )

    def extract(
        self,
        input_format: str = "all",
        num_threads: Optional[int] = None,
        accel_type: str = "CUDA",
        *,
        force_ocr: bool = False,
        formula_enrichment: bool = False,
        code_enrichment: bool = False,
        filenames: Optional[List[str]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        skip_existing: bool = True,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
        use_cls: bool = False,
        benchmark_mode: bool = False,
        export_doc_json: bool = True,
        emit_formula_index: bool = False,
        phase1_backend: str = "auto",
        workers_per_device: int = 1,
        _prepared: bool = False,
    ) -> None:
        """
        Extract input files to markdown format.

        Args:
            input_format: Input format ("pdf", "docx", "xml_jats", "html", "pptx", "csv", "md", "all") (default: "all")
                          Note: Old .doc format (pre-2007) is not supported
            num_threads: Number of threads for processing (default: 4)
            accel_type: Acceleration type ("Auto", "CPU", "CUDA", "MPS") (default: "Auto")
            export_doc_json: When True (default), writes Docling layout JSON to `json/<stem>.docling.json(.zst)`
            emit_formula_index: Also emit `json/<stem>.formula_index.jsonl` (default: False)
            phase1_backend: Selects the Phase-1 backend. ``"auto"`` (default) keeps the safe backend unless
                math/code enrichment is requested, ``"safe"`` forces the PyPDFium backend, and ``"docling"``
                forces the Docling backend.
            workers_per_device: Number of extraction workers to bind to each visible GPU when ``use_gpus='multi'``.

        """
        if not file_paths:
            self.logger.info(f"Extracting {input_format} files to markdown...")

        # We will prepare the extractor later (single-GPU branch). For multi-GPU,
        # we avoid importing heavy OCR deps in the parent.
        import os as _os
        images_scale_env = _os.getenv("GLOSSAPI_IMAGES_SCALE", "1.25")
        formula_batch_env = _os.getenv("GLOSSAPI_FORMULA_BATCH", "16")
        # Create output directory for downstream stages
        os.makedirs(self.markdown_dir, exist_ok=True)

        backend_choice = self._resolve_phase1_backend(
            phase1_backend,
            force_ocr=bool(force_ocr),
            formula_enrichment=bool(formula_enrichment),
            code_enrichment=bool(code_enrichment),
        )

        # Define supported formats
        supported_formats = ["pdf", "docx", "xml", "html", "pptx", "csv", "md"]

        # Look for the downloads directory first
        downloads_dir = self.output_dir / "downloads"

        # If downloads directory doesn't exist or is empty, check input directory and move files
        if not downloads_dir.exists() or not any(downloads_dir.iterdir()):
            self.logger.info(f"Downloads directory not found or empty at {downloads_dir}, checking input directory...")

            # Create downloads directory if it doesn't exist
            os.makedirs(downloads_dir, exist_ok=True)

            # Check input directory for supported files and move them
            input_files_to_move = []
            for ext in supported_formats:
                found_files = list(self.input_dir.glob(f"*.{ext}"))
                if found_files:
                    self.logger.info(f"Found {len(found_files)} .{ext} files in input directory, moving to downloads...")
                    input_files_to_move.extend(found_files)

            # Move files to downloads directory
            for file_path in input_files_to_move:
                target_path = downloads_dir / file_path.name
                if not target_path.exists():
                    shutil.copy2(file_path, target_path)
                    self.logger.debug(f"Copied {file_path.name} to downloads directory")

            self.logger.info(f"Moved {len(input_files_to_move)} files to downloads directory")

        # Get input files from downloads directory unless explicit paths were provided
        input_files: List[Path] = []
        if file_paths:
            try:
                input_files = [Path(p) for p in file_paths]
            except Exception as exc:
                raise ValueError(f"[Extract Phase] Invalid file path supplied to extract(): {exc}")
            self.logger.info(f"[Worker Batch] Processing {len(input_files)} direct file paths")
        elif input_format.lower() == "all":
            input_files = []
            for ext in supported_formats:
                found_files = list(downloads_dir.glob(f"*.{ext}"))
                input_files.extend(found_files)
                if found_files:
                    self.logger.info(f"Found {len(found_files)} .{ext} files in downloads directory")

            doc_files = list(downloads_dir.glob("*.doc"))
            if doc_files:
                self.logger.warning(
                    f"Found {len(doc_files)} .doc files which are not supported by Docling (pre-2007 Word format)"
                )
        elif "," in input_format.lower():
            input_files = []
            formats = [fmt.strip().lower() for fmt in input_format.split(",")]
            for ext in formats:
                if ext == "xml_jats":
                    ext = "xml"

                if ext == "doc":
                    self.logger.warning(
                        "The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first."
                    )
                    continue

                current_files = list(downloads_dir.glob(f"*.{ext}"))
                self.logger.info(f"Found {len(current_files)} files with extension .{ext}")
                input_files.extend(current_files)
        else:
            ext = "xml" if input_format.lower() == "xml_jats" else input_format.lower()

            if ext == "doc":
                self.logger.error(
                    "The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first."
                )
                return

            input_files = list(downloads_dir.glob(f"*.{ext}"))

        if filenames and not file_paths:
            names = {str(n) for n in filenames}
            input_files = [p for p in input_files if p.name in names]
            self.logger.info(f"Filtered to {len(input_files)} files from provided filename list")

        if not input_files:
            self.logger.warning(f"No {input_format} files found in {downloads_dir}")
            return

        skiplist_path = _resolve_skiplist_path(self.output_dir, self.logger)
        skip_mgr = _SkiplistManager(skiplist_path, self.logger)
        skipped_stems = skip_mgr.load()
        if skipped_stems:
            before = len(input_files)
            input_files = [p for p in input_files if canonical_stem(p) not in skipped_stems]
            removed = before - len(input_files)
            if removed:
                self.logger.warning(
                    "Skip-list %s filtered %d file(s) from Phase-1 dispatch.",
                    skiplist_path,
                    removed,
                )
        else:
            skipped_stems = set()

        self.logger.info(f"Found {len(input_files)} files to extract")

        # Process all files
        self.logger.info(f"Processing {len(input_files)} files...")

        # Multi-GPU orchestrator
        if str(use_gpus).lower() == "multi":
            # Detect devices if not provided
            devs = devices
            if not devs:
                # Prefer nvidia-smi, fallback to torch
                devs = []
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
                self.logger.error("Multi-GPU requested but no GPUs detected. Falling back to single GPU.")
            else:
                try:
                    self.logger.info("Multi-GPU: using %d device(s): %s", len(devs), ",".join(str(d) for d in devs))
                except Exception:
                    pass
                # Compute effective threads if auto
                try:
                    if num_threads is not None:
                        threads_effective = int(num_threads)
                    else:
                        cpu_total = max(1, os.cpu_count() or 0)
                        desired = max(2, 2 * max(1, len(devs)))
                        threads_effective = min(cpu_total, desired)
                except Exception:
                    threads_effective = int(num_threads) if isinstance(num_threads, int) else max(2, 2 * max(1, len(devs)))

                workers_per_device = max(1, int(workers_per_device or 1))
                configured_batch_hint = 1
                if backend_choice == "docling":
                    extractor = getattr(self, "extractor", None)
                    configured_batch_hint, _ = _resolve_docling_queue_policy(extractor)
                self.logger.info(
                    "Phase-1 config: backend=%s max_batch_files=%s threads=%s workers_per_device=%s skip_existing=%s benchmark=%s",
                    backend_choice,
                    configured_batch_hint,
                    threads_effective,
                    workers_per_device,
                    bool(skip_existing),
                    bool(benchmark_mode),
                )

                state_mgr = _ProcessingStateManager(
                    self.markdown_dir / ".processing_state.pkl", url_column=self.url_column
                )
                processed_files, problematic_files = state_mgr.load()
                if skip_existing and processed_files:
                    self.logger.info(
                        "State resume: %d processed, %d problematic", len(processed_files), len(problematic_files)
                    )

                pending_files = input_files
                if skip_existing and processed_files:
                    processed_stems = {canonical_stem(name) for name in processed_files}
                    pending_files = [p for p in pending_files if canonical_stem(p) not in processed_stems]
                if problematic_files:
                    problematic_stems = {canonical_stem(name) for name in problematic_files}
                    before_prob = len(pending_files)
                    pending_files = [p for p in pending_files if canonical_stem(p) not in problematic_stems]
                    removed_prob = before_prob - len(pending_files)
                    if removed_prob:
                        self.logger.warning(
                            "State resume: filtered %d pending file(s) already marked problematic.",
                            removed_prob,
                        )

                if not pending_files:
                    self.logger.info("No pending files left after state filtering; skipping extraction.")
                    return

                # Dynamic work queue across GPUs
                from .corpus_orchestrator import gpu_extract_worker_queue
                from multiprocessing import get_context
                ctx = get_context("spawn")
                manager = ctx.Manager()
                task_q = ctx.Queue()
                result_q = ctx.Queue()
                status_map = manager.dict()
                batch_target_pages = 1
                configured_max_batch_files = 1
                long_pdf_page_threshold = 600
                work_items: List[List[Path]] = [[Path(p)] for p in pending_files]
                extractor = getattr(self, "extractor", None)
                configured_max_batch_files, long_pdf_page_threshold = _resolve_docling_queue_policy(extractor)
                if backend_choice == "docling":
                    batch_target_pages = _resolve_docling_batch_target_pages()
                    work_items = _build_extract_work_items(
                        pending_files,
                        max_batch_files=configured_max_batch_files,
                        target_batch_pages=batch_target_pages,
                        long_pdf_page_threshold=long_pdf_page_threshold,
                    )
                queue_items = [[str(path.resolve()) for path in item] for item in work_items]
                for queue_item in queue_items:
                    task_q.put(queue_item)
                total_estimated_pages = 0
                try:
                    total_estimated_pages = sum(_estimate_extract_work_pages(path) for path in pending_files)
                except Exception:
                    total_estimated_pages = 0
                self.logger.info(
                    "Phase-1 dispatch: %d file(s) -> %d work item(s) (backend=%s max_batch_files=%d target_pages=%d est_pages=%d)",
                    len(pending_files),
                    len(queue_items),
                    backend_choice,
                    configured_max_batch_files,
                    batch_target_pages,
                    total_estimated_pages,
                )
                worker_log_dir_env = os.environ.get("GLOSSAPI_WORKER_LOG_DIR")
                worker_log_dir_to_use = worker_log_dir_env
                if not worker_log_dir_to_use:
                    try:
                        default_worker_log_dir = self.logs_dir / "ocr_workers"
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
                marker_base = Path(worker_log_dir_to_use) if worker_log_dir_to_use else (self.logs_dir / "ocr_workers")
                try:
                    marker_base.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    self.logger.debug("Unable to prepare marker directory %s: %s", marker_base, exc)
                worker_specs: List[Dict[str, Any]] = []
                for dev_id in devs:
                    for worker_slot in range(workers_per_device):
                        worker_specs.append(
                            {
                                "device_id": int(dev_id),
                                "worker_slot": int(worker_slot),
                                "worker_key": f"gpu{dev_id}-w{worker_slot}",
                            }
                        )
                procs: List[Any] = []
                proc_specs: Dict[int, Dict[str, Any]] = {}
                marker_files: Dict[str, Path] = {
                    spec["worker_key"]: marker_base / f"{spec['worker_key']}.current"
                    for spec in worker_specs
                }
                for spec in worker_specs:
                    p = ctx.Process(
                        target=gpu_extract_worker_queue,
                        args=(
                            spec["device_id"],
                            spec["worker_slot"],
                            spec["worker_key"],
                            str(self.input_dir),
                            str(self.output_dir),
                            task_q,
                            bool(force_ocr),
                            bool(formula_enrichment),
                            bool(code_enrichment),
                            bool(use_cls),
                            bool(skip_existing),
                            str(input_format),
                            int(threads_effective),
                            bool(benchmark_mode),
                            bool(export_doc_json),
                            bool(emit_formula_index),
                            backend_choice,
                            result_q,
                            status_map,
                            str(marker_base),
                        ),
                    )
                    p.start()
                    procs.append(p)
                    if p.pid is not None:
                        proc_specs[p.pid] = dict(spec)
                active = list(procs)
                any_fail = False
                last_summary = time.time()
                last_activity = time.time()
                heartbeat: Dict[int, float] = {p.pid or -1: time.time() for p in procs}
                try:
                    while active:
                        for p in list(active):
                            p.join(timeout=0.05)
                            if p.is_alive():
                                continue
                            active.remove(p)
                            if p in procs:
                                procs.remove(p)
                            pid = p.pid or -1
                            heartbeat[pid] = time.time()
                            worker_spec = proc_specs.pop(pid, None)
                            worker_key = worker_spec["worker_key"] if worker_spec else None
                            if p.exitcode not in (0, None):
                                any_fail = True
                                self.logger.warning("GPU worker pid=%s exited with code %s", p.pid, p.exitcode)
                                current_paths: List[str] = []
                                stems_for_skip: List[str] = []
                                if worker_key is not None:
                                    current_entry = status_map.pop(worker_key, None)
                                    if current_entry:
                                        if not isinstance(current_entry, (list, tuple, set)):
                                            current_entry = [current_entry]
                                        current_paths = [str(x) for x in current_entry]
                                        stems_for_skip = [canonical_stem(path) for path in current_paths]
                                    marker_path = marker_files.get(worker_key)
                                    if marker_path:
                                        try:
                                            marker_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass
                                if current_paths:
                                    problematic_files.update(current_paths)
                                    state_mgr.save(processed_files, problematic_files)
                                if stems_for_skip:
                                    skip_mgr.add(stems_for_skip)
                                if worker_spec is not None:
                                    self.logger.info(
                                        "Respawning %s after crash.",
                                        worker_spec["worker_key"],
                                    )
                                    replacement = ctx.Process(
                                        target=gpu_extract_worker_queue,
                                        args=(
                                            worker_spec["device_id"],
                                            worker_spec["worker_slot"],
                                            worker_spec["worker_key"],
                                            str(self.input_dir),
                                            str(self.output_dir),
                                            task_q,
                                            bool(force_ocr),
                                            bool(formula_enrichment),
                                            bool(code_enrichment),
                                            bool(use_cls),
                                            bool(skip_existing),
                                            str(input_format),
                                            int(threads_effective),
                                            bool(benchmark_mode),
                                            bool(export_doc_json),
                                            bool(emit_formula_index),
                                            backend_choice,
                                            result_q,
                                            status_map,
                                            str(marker_base),
                                        ),
                                    )
                                    replacement.start()
                                    procs.append(replacement)
                                    active.append(replacement)
                                    if replacement.pid is not None:
                                        proc_specs[replacement.pid] = dict(worker_spec)
                                        heartbeat[replacement.pid] = time.time()
                                continue
                            else:
                                if worker_key is not None:
                                    status_map.pop(worker_key, None)
                                    marker_path = marker_files.get(worker_key)
                                    if marker_path:
                                        try:
                                            marker_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass
                        drained = False
                        while True:
                            try:
                                result = result_q.get_nowait()
                            except queue.Empty:
                                break
                            drained = True
                            last_activity = time.time()
                            event_type = result.get("event")
                            if event_type == "batch":
                                ok_raw = [str(x) for x in (result.get("processed", []) or [])]
                                bad_raw = [str(x) for x in (result.get("problematic", []) or [])]
                                ok_stems = [canonical_stem(x) for x in ok_raw]
                                bad_stems = [canonical_stem(x) for x in bad_raw]
                                if ok_stems:
                                    processed_files.update(ok_stems)
                                    problematic_files.difference_update(ok_stems)
                                if bad_stems:
                                    problematic_files.update(bad_stems)
                                    skip_mgr.add(bad_stems)
                                state_mgr.save(processed_files, problematic_files)
                                self.logger.info(
                                    "%s batch complete: +%d processed, +%d problematic (totals: %d processed, %d problematic)",
                                    result.get("worker"),
                                    len(ok_stems),
                                    len(bad_stems),
                                    len(processed_files),
                                    len(problematic_files),
                                )
                                worker_pid = result.get("pid")
                                if worker_pid is not None:
                                    heartbeat[worker_pid] = time.time()
                            elif event_type == "exit":
                                if result.get("exitcode", 0) not in (0, None):
                                    any_fail = True
                                    self.logger.warning(
                                        "%s reported non-zero exit: %s", result.get("worker"), result.get("exitcode")
                                    )
                                worker_pid = result.get("pid")
                                if worker_pid is not None:
                                    heartbeat[worker_pid] = time.time()
                                worker_key = result.get("worker")
                                if worker_key is not None:
                                    status_map.pop(worker_key, None)
                                    marker_path = marker_files.get(str(worker_key))
                                    if marker_path:
                                        try:
                                            marker_path.unlink(missing_ok=True)
                                        except Exception:
                                            pass

                        now = time.time()
                        if now - last_summary > 30:
                            try:
                                pending = task_q.qsize()
                            except NotImplementedError:
                                pending = -1
                            self.logger.info(
                                "Progress summary: processed=%d problematic=%d queue=%d active_workers=%d",
                                len(processed_files),
                                len(problematic_files),
                                pending,
                                len(active),
                            )
                            last_summary = now

                        if not drained:
                            time.sleep(0.05)

                        if now - last_activity > 120:
                            self.logger.warning(
                                "No batch completions reported for %.0fs (active workers: %d). Still waiting.",
                                now - last_activity,
                                len(active),
                            )
                            last_activity = now
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

                remaining_after_failure: List[str] = []
                try:
                    while True:
                        pending_item = task_q.get_nowait()
                        if isinstance(pending_item, str) and pending_item.strip():
                            remaining_after_failure.append(pending_item)
                            continue
                        if isinstance(pending_item, (list, tuple, set)):
                            remaining_after_failure.extend(
                                str(item).strip()
                                for item in pending_item
                                if str(item).strip()
                            )
                except queue.Empty:
                    pass
                if remaining_after_failure:
                    skip_mgr.add(canonical_stem(x) for x in remaining_after_failure)
                    self.logger.error(
                        "No active extraction workers remain; skipped %d pending item(s)",
                        len(remaining_after_failure),
                    )

                if any_fail:
                    self.logger.warning("One or more GPU workers exited with non-zero status.")
                else:
                    self.logger.info(
                        "Multi-GPU extraction complete. Processed %d files (%d problematic)",
                        len(processed_files),
                        len(problematic_files),
                    )
                return

        # Single GPU path
        # Prepare extractor (lazy import + instantiate)
        if self.extractor is None:
            try:
                from ..gloss_extract import GlossExtract  # local import to avoid import-time heavy deps
                self.extractor = GlossExtract(url_column=self.url_column)
            except Exception as e:
                self.logger.error(f"[Extract Phase] Failed to initialize GlossExtract: {e}")
                raise
        # Configure Phase-1 helpers on extractor
        try:
            setattr(self.extractor, "export_doc_json", bool(export_doc_json))
            setattr(self.extractor, "emit_formula_index", bool(emit_formula_index))
        except Exception:
            pass
        # Determine effective thread count (auto when None)
        try:
            threads_effective = int(num_threads) if num_threads is not None else (os.cpu_count() or 4)
        except Exception:
            threads_effective = (os.cpu_count() or 4)
        self.extractor.enable_accel(threads=threads_effective, type=accel_type)
        # Harmonize GPU math throughput settings and images scale across runs
        try:
            # Torch matmul precision for CodeFormula
            if formula_enrichment:
                torch_mod = _maybe_import_torch(force=True)
                try:
                    if torch_mod is not None and hasattr(torch_mod, "set_float32_matmul_precision"):
                        torch_mod.set_float32_matmul_precision("high")
                except Exception:
                    pass
                try:
                    from docling.models.code_formula_model import CodeFormulaModel  # type: ignore
                    fb = int(formula_batch_env) if str(formula_batch_env).isdigit() else 16
                    CodeFormulaModel.elements_batch_size = int(fb)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass
        # Log cache policy and settings
        try:
            self.logger.info(
                "Caches: HF_HOME=%s XDG_CACHE_HOME=%s DOCLING_CACHE_DIR=%s",
                _os.getenv("HF_HOME"), _os.getenv("XDG_CACHE_HOME"), _os.getenv("DOCLING_CACHE_DIR"),
            )
            self.logger.info(
                "GPU math settings: formula_enrichment=%s batch=%s matmul_precision=high images_scale=%s",
                bool(formula_enrichment), formula_batch_env, images_scale_env,
            )
        except Exception:
            pass
        # Prepare converter when not already primed by caller (internal fast path)
        if not bool(_prepared):
            self.prime_extractor(
                input_format=input_format,
                num_threads=num_threads,
                accel_type=accel_type,
                force_ocr=bool(force_ocr),
                formula_enrichment=bool(formula_enrichment),
                code_enrichment=bool(code_enrichment),
                use_cls=bool(use_cls),
                benchmark_mode=bool(benchmark_mode),
                export_doc_json=bool(export_doc_json),
                emit_formula_index=bool(emit_formula_index),
                phase1_backend=backend_choice,
            )
        # Propagate benchmark mode to extractor to trim auxiliary I/O
        try:
            setattr(self.extractor, "benchmark_mode", bool(benchmark_mode))
        except Exception:
            pass
        # Extract files to markdown
        os.makedirs(self.markdown_dir, exist_ok=True)
        self.extractor.extract_path(input_files, self.markdown_dir, skip_existing=skip_existing)
        self.logger.info(f"Extraction complete. Markdown files saved to {self.markdown_dir}")
        try:
            release = getattr(self.extractor, "release_resources", None)
            if callable(release):
                release()
        except Exception as exc:
            self.logger.debug("Extractor resource release failed: %s", exc)
