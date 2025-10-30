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
try:
    from ..gloss_section_classifier import GlossSectionClassifier  # type: ignore
except Exception:
    GlossSectionClassifier = None  # type: ignore[assignment]
from .corpus_skiplist import _SkiplistManager, _resolve_skiplist_path
from .corpus_state import _ProcessingStateManager
from .corpus_utils import _maybe_import_torch
from .phase_download import DownloadPhaseMixin
from .phase_extract import ExtractPhaseMixin
from .phase_clean import CleanPhaseMixin
from .phase_ocr_math import OcrMathPhaseMixin
from .phase_sections import SectionPhaseMixin
from .phase_annotate import AnnotatePhaseMixin
from .phase_export import ExportPhaseMixin

class Corpus(
    DownloadPhaseMixin,
    ExtractPhaseMixin,
    CleanPhaseMixin,
    OcrMathPhaseMixin,
    SectionPhaseMixin,
    AnnotatePhaseMixin,
    ExportPhaseMixin,
):
    """
    A high-level wrapper for the GlossAPI academic document processing pipeline.
    
    This class provides a unified interface to extract PDFs to markdown,
    extract sections, and classify them using machine learning.
    
    Example:
        corpus = Corpus(input_dir="path/to/pdfs", output_dir="path/to/output")
        corpus.extract()  # Extract PDFs to markdown
        corpus.section()  # Extract sections from markdown files
        corpus.annotate()  # Classify sections using ML
    """

    
    def __init__(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path],
        section_classifier_model_path: Optional[Union[str, Path]] = None,
        extraction_model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        annotation_mapping: Optional[Dict[str, str]] = None,
        downloader_config: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
        verbose: bool = False
    ):
        """
        Initialize the Corpus processor.
        
        Args:
            input_dir: Directory containing input files (PDF or markdown)
            output_dir: Base directory for all outputs
            section_classifier_model_path: Path to the pre-trained section classifier model
            extraction_model_path: Path to the pre-trained kmeans clustering model for extraction
            metadata_path: Path to metadata file with document types (optional)
            annotation_mapping: Dictionary mapping document types to annotation methods (optional)
                               e.g. {'Κεφάλαιο': 'chapter'} means documents with type 'Κεφάλαιο' use chapter annotation
            downloader_config: Configuration parameters for the GlossDownloader (optional)
            log_level: Logging level (default: logging.INFO)
            verbose: Whether to enable verbose logging for debugging (default: False)
        """
        # Setup module logger without forcing global configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        try:
            if not self.logger.handlers:
                _handler = logging.StreamHandler()
                _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s", "%H:%M:%S"))
                self.logger.addHandler(_handler)
            self.logger.propagate = False
        except Exception:
            pass
        
        # Verbose flag for detailed logging
        self.verbose = verbose
        
        # Store paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self._metadata_parquet_path: Optional[Path] = None
        
        # Package directory for default models
        package_dir = Path(__file__).parent
        
        # Handle section classifier model path
        if section_classifier_model_path:
            self.section_classifier_model_path = Path(section_classifier_model_path)
        else:
            # Use default model path in the package
            self.section_classifier_model_path = package_dir / "models" / "section_classifier.joblib"
        
        # Handle extraction model path
        if extraction_model_path:
            self.extraction_model_path = Path(extraction_model_path)
        else:
            # Use default model path in the package
            self.extraction_model_path = package_dir / "models" / "kmeans_weights.joblib"
            
        self.metadata_path = Path(metadata_path) if metadata_path else None
        if self.metadata_path is not None:
            try:
                if self.metadata_path.exists():
                    self._metadata_parquet_path = self.metadata_path
                else:
                    self.logger.warning("Provided metadata_path does not exist: %s", self.metadata_path)
                    self.metadata_path = None
            except Exception:
                self.metadata_path = None
        
        # Store annotation mapping - default is to treat 'Κεφάλαιο' as chapter
        self.annotation_mapping = annotation_mapping or {'Κεφάλαιο': 'chapter'}
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize downloader config first
        self.downloader_config = downloader_config or {}
        
        # Initialize component classes
        # Get the URL column from downloader config or use default 'url'
        self.url_column = self.downloader_config.get('url_column', 'url')
        # Lazy-create extractor to avoid heavy imports unless needed
        self.extractor = None
        self.sectioner = GlossSection()
        try:
            self.classifier = GlossSectionClassifier() if GlossSectionClassifier is not None else None  # type: ignore[call-arg, assignment]
        except Exception:
            self.classifier = None
        
        self.output_dir = Path(output_dir)
        self.downloads_dir = self.output_dir / "downloads"
        self.markdown_dir = self.output_dir / "markdown"
        self.ocr_model_dir = None  # Will use default discovery or user-specified path
        self.sections_dir = self.output_dir / "sections"
        # Directory that will hold cleaned markdown after Rust-powered cleaning
        self.cleaned_markdown_dir = self.output_dir / "clean_markdown"
        # Define models_dir path but don't create the directory yet - only create it when needed
        self.models_dir = self.output_dir / "models"
        self.logs_dir = self.output_dir / "logs"

        # Track whether we've already printed the GPU setup banner in this process
        self._gpu_banner_logged = False
        self._phase1_backend = "safe"

        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        os.makedirs(self.cleaned_markdown_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Setup output files
        self.sections_parquet = self.sections_dir / "sections_for_annotation.parquet"
        self.classified_parquet = self.output_dir / "classified_sections.parquet"
        self.fully_annotated_parquet = self.output_dir / "fully_annotated_sections.parquet"
        
        # Initialize document type mapping
        self.filename_to_doctype = {}
        
        self._load_metadata()

    def _get_cached_metadata_parquet(self) -> Optional[Path]:
        """Return cached metadata parquet path if it still exists."""

        if self._metadata_parquet_path is not None:
            if self._metadata_parquet_path.exists():
                return self._metadata_parquet_path
            self._metadata_parquet_path = None
        return None

    def _cache_metadata_parquet(self, candidate: Optional[Union[str, Path]]) -> Optional[Path]:
        """Remember the provided parquet path for subsequent lookups."""

        if candidate is None:
            return None
        path = Path(candidate)
        self._metadata_parquet_path = path
        return path

    def _resolve_metadata_parquet(
        self,
        parquet_schema: "ParquetSchema",
        *,
        ensure: bool = True,
        search_input: bool = True,
    ) -> Optional[Path]:
        """Return a best-effort metadata parquet path, caching the result."""

        cached = self._get_cached_metadata_parquet()
        if cached is not None:
            return cached
        if ensure:
            ensured = parquet_schema.ensure_metadata_parquet(self.output_dir)
            if ensured is not None:
                return self._cache_metadata_parquet(ensured)
        found = parquet_schema.find_metadata_parquet(self.output_dir)
        if found is not None:
            return self._cache_metadata_parquet(found)
        if search_input:
            input_dirs = [self.input_dir]
            dl_dir = self.input_dir / "download_results"
            if dl_dir.exists():
                input_dirs.append(dl_dir)
            for directory in input_dirs:
                if directory.exists():
                    located = parquet_schema.find_metadata_parquet(directory)
                    if located is not None:
                        return self._cache_metadata_parquet(located)
        return None

    def _load_metadata(self) -> None:
        """Load metadata file if provided and extract document type mapping."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                self.logger.info(f"Loading metadata from {self.metadata_path}")
                metadata_df = pd.read_parquet(self.metadata_path)
                
                # Debug information
                self.logger.info(f"Metadata file has {len(metadata_df)} rows and columns: {metadata_df.columns.tolist()}")
                try:
                    self.logger.info(f"Sample filenames: {metadata_df['filename'].head(3).tolist()}")
                except Exception:
                    pass
                if 'document_type' not in metadata_df.columns:
                    import pandas as _pd
                    # Create a blank document_type column for downstream compatibility
                    metadata_df['document_type'] = _pd.Series([_pd.NA] * len(metadata_df))
                    self.logger.info("Added missing 'document_type' column to metadata (blank values)")
                else:
                    self.logger.info(f"Sample document types: {metadata_df['document_type'].head(3).tolist()}")
                
                # Create a mapping from filename to document_type
                if 'filename' in metadata_df.columns and 'document_type' in metadata_df.columns:
                    self.logger.info("Both 'filename' and 'document_type' columns found in metadata")
                    
                    # Check if filenames have extensions
                    sample_filenames = metadata_df['filename'].head(100).tolist()
                    if any('.' in str(f) for f in sample_filenames):
                        self.logger.warning("Some filenames in metadata contain extensions. This may cause matching issues.")
                        self.logger.warning("Will attempt to match filenames both with and without extensions.")
                        
                        # Create a mapping that works with or without extensions
                        self.filename_to_doctype = {}
                        
                        for idx, row in metadata_df.iterrows():
                            filename = row['filename']
                            doctype = row['document_type']
                            # Skip empty/NA document types
                            try:
                                if doctype is None:
                                    continue
                                import pandas as _pd
                                if doctype is _pd.NA or _pd.isna(doctype):
                                    continue
                                if not str(doctype).strip():
                                    continue
                            except Exception:
                                pass
                            
                            # Add the original filename
                            self.filename_to_doctype[filename] = doctype
                            
                            # Add filename without extension
                            if '.' in filename:
                                base_filename = filename.rsplit('.', 1)[0]
                                self.filename_to_doctype[base_filename] = doctype
                            
                            # Add filename with .md extension
                            if not filename.endswith('.md'):
                                md_filename = f"{filename}.md"
                                self.filename_to_doctype[md_filename] = doctype
                    else:
                        # Simple dictionary mapping without extension handling (skip empty types)
                        try:
                            import pandas as _pd
                            df_nt = metadata_df.copy()
                            mask = (~df_nt['document_type'].isna()) & (df_nt['document_type'].astype(str).str.strip() != '')
                            df_nt = df_nt[mask]
                            self.filename_to_doctype = dict(zip(
                                df_nt['filename'], 
                                df_nt['document_type']
                            ))
                        except Exception:
                            self.filename_to_doctype = {}
                    
                    self.logger.info(f"Loaded {len(self.filename_to_doctype)} filename-to-doctype mappings")
                else:
                    self.logger.warning("Metadata file does not contain 'filename' or 'document_type' columns")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        else:
            if self.metadata_path:
                self.logger.warning(f"Metadata file not found: {self.metadata_path}")

    # Download phase                                                     #
    # Extraction phase                                                   #



    

    






    # Cleaning phase                                                     #

    # Backwards-compatibility shim – filter() now delegates to clean()
    
    # OCR & math enrichment                                             #


    # Sectioning & annotation                                           #
    

    
    
    


    

# Top-level worker function for multi-GPU extraction (picklable by multiprocessing)
def gpu_extract_worker_queue(
    device_id: int,
    in_dir: str,
    out_dir: str,
    work_q,  # multiprocessing Queue of filename strings
    force: bool,
    fe: bool,
    ce: bool,
    use_cls_w: bool,
    skip: bool,
    input_fmt: str,
    threads: int,
    benchmark: bool,
    export_json: bool,
    emit_index: bool,
    backend: str,
    result_q=None,
    status_map=None,
    marker_dir: Optional[str] = None,
) -> None:
    import os as _os
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

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
            _torch = _maybe_import_torch()
            if _torch is not None and hasattr(_torch, "set_num_threads"):
                _torch.set_num_threads(1)
        except Exception:
            pass

    _ensure_thread_caps()
    _status_proxy = status_map
    _marker_path = _Path(marker_dir).expanduser() / f"gpu{device_id}.current" if marker_dir else None

    def _update_current(batch_items: List[str]) -> None:
        if _status_proxy is not None:
            try:
                _status_proxy[device_id] = list(batch_items)
            except Exception:
                pass
        if _marker_path is not None:
            try:
                _marker_path.write_text("\n".join(batch_items) + "\n", encoding="utf-8")
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
    _worker_log_handle = None
    try:
        _log_dir = _os.environ.get("GLOSSAPI_WORKER_LOG_DIR")
        if _log_dir:
            _log_path = _Path(_log_dir).expanduser()
            _log_path.mkdir(parents=True, exist_ok=True)
            _worker_log_file = _log_path / f"gpu{device_id}_{_os.getpid()}.log"
            _worker_log_handle = open(_worker_log_file, "a", encoding="utf-8", buffering=1)
            _sys.stdout = _worker_log_handle
            _sys.stderr = _worker_log_handle
    except Exception:
        _worker_log_handle = None
    # Bind this worker to a single GPU id
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _os.environ["GLOSSAPI_DOCLING_DEVICE"] = "cuda:0"
    # Worker GPU binding banner (prints by default; disable with GLOSSAPI_WORKER_LOG_VERBOSE=0)
    try:
        _verbose = str(_os.environ.get("GLOSSAPI_WORKER_LOG_VERBOSE", "1")).strip().lower()
        if _verbose not in ("0", "false", "no", "off", ""):  # default on
            try:
                _torch = _maybe_import_torch()
                if _torch is not None and getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                    _torch_name = _torch.cuda.get_device_name(0)
                elif _torch is not None:
                    _torch_name = "no-cuda"
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
                _nvsmi = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=2)
                _phys = _nvsmi.stdout.splitlines()[0].strip() if _nvsmi.returncode == 0 and _nvsmi.stdout else ""
            except Exception:
                _phys = ""
            try:
                print(f"[GPU{device_id}] bound: CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES','')} pid={_os.getpid()} torch={_torch_name} ORT={_ort_prov}")
                if _phys:
                    print(f"[GPU{device_id}] physical: {_phys}")
            except Exception:
                pass
    except Exception:
        pass
    # Light import of Corpus (prefer installed package; fallback to repo src)
    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            import sys as _sys, pathlib as _pl
            _sys.path.insert(0, str((_pl.Path(out_dir).resolve().parents[1] / 'src').resolve()))
            _ensure_thread_caps()
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as _e:
            print(f"[GPU{device_id}] Cannot import glossapi in worker: {_e}")
            if result_q is not None:
                try:
                    result_q.put(
                        {
                            "event": "exit",
                            "worker": device_id,
                            "exitcode": 1,
                            "pid": _os.getpid(),
                            "error": str(_e),
                        }
                    )
                except Exception:
                    pass
            _sys.exit(1)
    c = _Corpus(input_dir=in_dir, output_dir=out_dir)
    # Prime once per worker (persistent converter)
    try:
        c.prime_extractor(
            input_format=input_fmt,
            num_threads=threads,
            accel_type="cuda:0",
            force_ocr=force,
            formula_enrichment=fe,
            code_enrichment=ce,
            use_cls=use_cls_w,
            benchmark_mode=benchmark,
            export_doc_json=bool(export_json),
            emit_formula_index=bool(emit_index),
            phase1_backend=backend,
        )
    except Exception as _e:
        msg = f"[GPU{device_id}] Prime failed: {_e}"
        print(msg)
        if result_q is not None:
            try:
                result_q.put(
                    {
                        "event": "exit",
                        "worker": device_id,
                        "exitcode": 1,
                        "pid": _os.getpid(),
                        "error": str(_e),
                    }
                )
            except Exception:
                pass
        raise
    try:
        if c.extractor is not None:
            c.extractor.external_state_updates = result_q is not None

            if result_q is not None:

                def _report_batch(ok_list, bad_list):
                    try:
                        result_q.put(
                            {
                                "event": "batch",
                                "worker": device_id,
                                "processed": [str(x) for x in ok_list],
                                "problematic": [str(x) for x in bad_list],
                                "pid": _os.getpid(),
                            }
                        )
                    except Exception as exc:
                        print(f"[GPU{device_id}] Failed to report batch: {exc}")

                c.extractor.batch_result_callback = _report_batch
    except Exception as _e:
        print(f"[GPU{device_id}] Unable to set batch callback: {_e}")
    # Prepare persistent extractor in this worker on first call
    # Process queue items in small batches to reduce function-call overhead
    batch: list[str] = []
    try:
        _batch_env = int(str(_os.environ.get("GLOSSAPI_GPU_BATCH_SIZE", "")).strip() or 0)
    except Exception:
        _batch_env = 0
    default_batch = 5 if not force else 1
    try:
        extractor = getattr(c, "extractor", None)
        if extractor is not None:
            configured = int(getattr(extractor, "max_batch_files", default_batch))
            if force:
                default_batch = 1
            else:
                default_batch = max(1, configured)
    except Exception:
        pass
    BATCH_SIZE = max(1, _batch_env) if _batch_env else max(1, default_batch)
    import queue as _queue
    last_progress = _time.time()
    processed = 0
    exit_code = 0
    try:
        while True:
            try:
                nm = work_q.get_nowait()
            except _queue.Empty:
                # queue.Empty or other -> flush any pending batch then exit
                if batch:
                    try:
                        _update_current(list(batch))
                        c.extract(
                            input_format=input_fmt,
                            num_threads=threads,
                            accel_type="cuda:0",
                            force_ocr=force,
                            formula_enrichment=fe,
                            code_enrichment=ce,
                            file_paths=list(batch),
                            skip_existing=skip,
                            use_gpus="single",
                            use_cls=use_cls_w,
                            benchmark_mode=benchmark,
                            export_doc_json=bool(export_json),
                            emit_formula_index=bool(emit_index),
                            phase1_backend=backend,
                            _prepared=True,
                        )
                        processed += len(batch)
                        _clear_current()
                    except Exception as _e:
                        exit_code = 1
                        print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
                        if result_q is not None:
                            try:
                                result_q.put(
                                    {
                                        "event": "batch",
                                        "worker": device_id,
                                        "processed": [],
                                        "problematic": list(batch),
                                        "pid": _os.getpid(),
                                        "error": str(_e),
                                    }
                                )
                            except Exception:
                                pass
                        _clear_current()
                    batch.clear()
                break
            except Exception as exc:
                exit_code = 1
                print(f"[GPU{device_id}] Queue receive error: {exc}")
                break
            if isinstance(nm, str) and nm.strip():
                batch.append(nm)
            if len(batch) >= BATCH_SIZE:
                try:
                    _update_current(list(batch))
                    c.extract(
                        input_format=input_fmt,
                        num_threads=threads,
                        accel_type="cuda:0",
                        force_ocr=force,
                        formula_enrichment=fe,
                        code_enrichment=ce,
                        file_paths=list(batch),
                        skip_existing=skip,
                        use_gpus="single",
                        use_cls=use_cls_w,
                        benchmark_mode=benchmark,
                        export_doc_json=bool(export_json),
                        emit_formula_index=bool(emit_index),
                        phase1_backend=backend,
                        _prepared=True,
                    )
                    processed += len(batch)
                    _clear_current()
                except Exception as _e:
                    exit_code = 1
                    print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
                    if result_q is not None:
                        try:
                            result_q.put(
                                {
                                    "event": "batch",
                                    "worker": device_id,
                                    "processed": [],
                                    "problematic": list(batch),
                                    "pid": _os.getpid(),
                                    "error": str(_e),
                                }
                            )
                        except Exception:
                            pass
                    _clear_current()
                batch.clear()
            # Occasional heartbeat
            if _time.time() - last_progress > 30:
                try:
                    print(f"[GPU{device_id}] processed ~{processed} files…")
                except Exception:
                    pass
                last_progress = _time.time()
    except Exception as exc:
        exit_code = 1
        print(f"[GPU{device_id}] Fatal worker error: {exc}")

    _clear_current()

    try:
        extractor = getattr(c, "extractor", None)
        release = getattr(extractor, "release_resources", None)
        if callable(release):
            release()
    except Exception as exc:
        try:
            print(f"[GPU{device_id}] Failed to release extractor resources: {exc}")
        except Exception:
            pass

    if result_q is not None:
        try:
            result_q.put({
                "event": "exit",
                "worker": device_id,
                "exitcode": exit_code,
                "pid": _os.getpid(),
            })
        except Exception as exc:
            print(f"[GPU{device_id}] Failed to report exit: {exc}")

    if _worker_log_handle is not None:
        try:
            _worker_log_handle.flush()
            _worker_log_handle.close()
        except Exception:
            pass
    _sys.exit(exit_code)
