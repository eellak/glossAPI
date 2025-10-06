from typing import Dict, Set, List, Optional, Iterable, Tuple, Any, Union, Callable
import importlib
import sys
import warnings

from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    LayoutOptions,
    TableStructureOptions,
    TableFormerMode,
)

try:
    from docling.datamodel.pipeline_options import PictureDescriptionApiOptions
except ImportError:  # pragma: no cover - older docling versions
    class PictureDescriptionApiOptions:  # type: ignore
        """Compatibility shim when PictureDescriptionApiOptions is missing."""

        def __init__(self, *args, **kwargs):
            pass
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings


def _maybe_import_torch(*, force: bool = False):
    """Return the torch module if already loaded or explicitly requested."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        return torch_mod
    try:
        return importlib.import_module("torch")  # type: ignore
    except Exception:
        return None
    return None
DocumentConverter = None
PdfFormatOption = None
WordFormatOption = None
HTMLFormatOption = None
XMLJatsFormatOption = None
PowerpointFormatOption = None
MarkdownFormatOption = None
CsvFormatOption = None
StandardPdfPipeline = None
DoclingParseV2DocumentBackend = None
DoclingParseDocumentBackend = None
PyPdfiumDocumentBackend = None


class _NoOpOption:  # minimal stand-ins for optional helpers
    def __init__(self, *args, **kwargs):
        pass


_DOC_CONVERTER_LOADED = False
_DOC_PIPELINE_LOADED = False


def _ensure_docling_converter_loaded() -> None:
    global _DOC_CONVERTER_LOADED, DocumentConverter, PdfFormatOption
    global WordFormatOption, HTMLFormatOption, XMLJatsFormatOption
    global PowerpointFormatOption, MarkdownFormatOption, CsvFormatOption
    if _DOC_CONVERTER_LOADED:
        return
    try:
        module = importlib.import_module("docling.document_converter")
        DocumentConverter = getattr(module, "DocumentConverter")
        PdfFormatOption = getattr(module, "PdfFormatOption")
        WordFormatOption = getattr(module, "WordFormatOption", _NoOpOption)
        HTMLFormatOption = getattr(module, "HTMLFormatOption", _NoOpOption)
        XMLJatsFormatOption = getattr(module, "XMLJatsFormatOption", _NoOpOption)
        PowerpointFormatOption = getattr(module, "PowerpointFormatOption", _NoOpOption)
        MarkdownFormatOption = getattr(module, "MarkdownFormatOption", _NoOpOption)
        CsvFormatOption = getattr(module, "CsvFormatOption", _NoOpOption)
        _DOC_CONVERTER_LOADED = True
    except Exception as exc:
        raise RuntimeError(f"Failed to load docling document converter components: {exc}")


def _ensure_docling_pipeline_loaded() -> None:
    global _DOC_PIPELINE_LOADED, StandardPdfPipeline
    global DoclingParseV2DocumentBackend, DoclingParseDocumentBackend, PyPdfiumDocumentBackend
    if _DOC_PIPELINE_LOADED:
        return
    try:
        StandardPdfPipeline = importlib.import_module(
            "docling.pipeline.standard_pdf_pipeline"
        ).StandardPdfPipeline
        DoclingParseV2DocumentBackend = importlib.import_module(
            "docling.backend.docling_parse_v2_backend"
        ).DoclingParseV2DocumentBackend
        DoclingParseDocumentBackend = importlib.import_module(
            "docling.backend.docling_parse_backend"
        ).DoclingParseDocumentBackend
        PyPdfiumDocumentBackend = importlib.import_module(
            "docling.backend.pypdfium2_backend"
        ).PyPdfiumDocumentBackend
        _DOC_PIPELINE_LOADED = True
    except Exception as exc:
        raise RuntimeError(f"Failed to load docling PDF pipeline components: {exc}")


from docling.pipeline.simple_pipeline import SimplePipeline
# Ensure RapidOCR plugin is registered for factory-based OCR construction
import docling.models.rapid_ocr_model  # noqa: F401
from ._rapidocr_paths import resolve_packaged_onnx_and_keys
import inspect

import ftfy
import logging
from contextlib import redirect_stdout
import os
import pickle
import signal
import time
import re
from pathlib import Path
from typing import Iterable, List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
from shutil import copy2
from collections import defaultdict
import json
from contextlib import contextmanager
import pandas as pd

LOGGING_INITIALIZED = False

class GlossExtract:
    """
    A class for extracting content from PDF documents to Markdown using Docling, and for
    clustering documents based on their quality (good vs. bad extractions).
    """
    
    def __init__(self, url_column='url'):
        """Initialize the GlossExtract class with default settings.
        
        Args:
            url_column: The URL column name to use in the parquet schema
        """
        # Default timeout for processing files (10 minutes in seconds)
        self.processing_timeout = 600
        # Default to layout-only extraction; OCR is enabled explicitly by callers
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True
        # Prefer lightweight placeholder picture descriptions while keeping the
        # table enhancer on its most accurate setting when available.
        try:
            if hasattr(self.pipeline_options, "do_picture_description"):
                self.pipeline_options.do_picture_description = False
            if getattr(self.pipeline_options, "picture_description_options", None) is None:
                self.pipeline_options.picture_description_options = PictureDescriptionApiOptions()
            if hasattr(self.pipeline_options, "enable_remote_services"):
                self.pipeline_options.enable_remote_services = False
        except Exception:
            pass
        try:
            if self.pipeline_options.table_structure_options is not None:
                tso = self.pipeline_options.table_structure_options
                if hasattr(tso, "mode"):
                    tso.mode = TableFormerMode.ACCURATE
                if hasattr(tso, "do_cell_matching"):
                    tso.do_cell_matching = True
        except Exception:
            pass
        self.converter = None
        # Enable accurate table structure by default
        try:
            self.pipeline_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
            self.pipeline_options.table_structure_options.do_cell_matching = True
        except Exception:
            pass
        # Default enrichment off (enable only when requested)
        try:
            self.pipeline_options.do_formula_enrichment = False
            self.pipeline_options.do_code_enrichment = False
            self.pipeline_options.layout_options = LayoutOptions()
        except Exception:
            pass
        self.USE_V2 = True
        self.log_file = Path('.') / 'conversion.log'
        self.url_column = url_column  # Store the URL column name for later use
        self._metadata_parquet_path = None  # Store metadata parquet path once found
        # Chunking defaults for long PDFs
        self.long_pdf_page_threshold = 600
        self.chunk_size = 200
        self.chunk_timeout_s = 600
        self.max_chunk_timeouts = 2
        global LOGGING_INITIALIZED
        if not LOGGING_INITIALIZED:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(str(self.log_file), mode='a'),
                    logging.StreamHandler()
                ]
            )
            LOGGING_INITIALIZED = True
        # Per-instance logger
        self._log = logging.getLogger(__name__)
        try:
            if not self._log.handlers:
                _handler = logging.StreamHandler()
                _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s", "%H:%M:%S"))
                self._log.addHandler(_handler)
            self._log.propagate = False
        except Exception:
            pass
        try:
            logging.getLogger("docling").setLevel(logging.WARNING)
            logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
            logging.getLogger("docling.utils").setLevel(logging.WARNING)
            class _DoclingFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    msg = record.getMessage()
                    if record.name == "root" and (
                        "Using engine_name" in msg or
                        "Accelerator device" in msg or
                        msg.startswith("Using /")
                    ):
                        return False
                    return True
            root_logger = logging.getLogger()
            if not any(isinstance(f, _DoclingFilter) for f in root_logger.filters):
                root_logger.addFilter(_DoclingFilter())
        except Exception:
            pass
        # Trim auxiliary I/O and profiling when running benchmarks
        self.benchmark_mode: bool = False
        # Phase-1 helpers: toggle JSON export and formula index emission
        self.export_doc_json: bool = False
        self.emit_formula_index: bool = False
        # Track last extractor configuration for reuse decisions
        self._last_extractor_cfg = None
        self._active_pdf_options: Optional[PdfPipelineOptions] = None
        self._current_ocr_enabled: bool = False
        self.batch_result_callback: Optional[Callable[[List[str], List[str]], None]] = None
        self.external_state_updates: bool = False
        # Phase-1 extraction safety controls
        self.batch_policy: str = "safe"
        self.max_batch_files: int = 1
        self.use_pypdfium_backend: bool = True
        policy_env = os.getenv("GLOSSAPI_BATCH_POLICY")
        max_env = os.getenv("GLOSSAPI_BATCH_MAX")
        max_override: Optional[int] = None
        if policy_env or max_env:
            warnings.warn(
                "GLOSSAPI_BATCH_POLICY and GLOSSAPI_BATCH_MAX are deprecated. "
                "Use Corpus.extract(... phase1_backend='docling') to select the Docling backend.",
                DeprecationWarning,
                stacklevel=2,
            )
        if max_env:
            try:
                max_override = int(max_env)
            except Exception:
                try:
                    self._log.warning("Ignoring invalid GLOSSAPI_BATCH_MAX=%s", max_env)
                except Exception:
                    pass
                max_override = None
        self.configure_batch_policy(policy_env or "safe", max_batch_files=max_override, prefer_safe_backend=None if (policy_env or max_env) else True)
        self._thread_caps_applied: bool = False

    def configure_batch_policy(
        self,
        policy: str,
        *,
        max_batch_files: Optional[int] = None,
        prefer_safe_backend: Optional[bool] = None,
    ) -> None:
        policy_norm = (policy or "safe").strip().lower()
        self.batch_policy = policy_norm
        if max_batch_files is not None:
            self.max_batch_files = max(1, int(max_batch_files))
        else:
            if policy_norm in {"safe", "pypdfium"}:
                self.max_batch_files = 1
            elif policy_norm in {"docling", "throughput", "docling_batched"}:
                self.max_batch_files = 5
            else:
                self.max_batch_files = max(1, self.max_batch_files)
        if prefer_safe_backend is None:
            prefer_safe_backend = policy_norm in {"safe", "pypdfium"}
        self.use_pypdfium_backend = bool(prefer_safe_backend)
        try:
            backend = "pypdfium" if self.use_pypdfium_backend else "docling"
            self._log.info(
                "Configured batch policy=%s max_batch_files=%d backend=%s",
                self.batch_policy,
                self.max_batch_files,
                backend,
            )
        except Exception:
            pass

    def _apply_thread_caps(self) -> None:
        if self._thread_caps_applied:
            return
        import os as _os

        defaults = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        for key, value in defaults.items():
            _os.environ.setdefault(key, value)
        torch_mod = _maybe_import_torch()
        try:
            if torch_mod is not None and hasattr(torch_mod, "set_num_threads"):
                torch_mod.set_num_threads(1)
        except Exception:
            pass
        self._thread_caps_applied = True

    def _supports_native_timeout(self) -> str | None:
        """Return the timeout kwarg name if supported by Docling, else None."""
        try:
            sig = inspect.signature(self.converter.convert)  # type: ignore[attr-defined]
            for name in ("timeout", "timeout_s"):
                if name in sig.parameters:
                    return name
        except Exception:
            pass
        return None

    def _convert_all_with_timeout(self, files: Iterable[Path], timeout_s: int, **kwargs):
        """Use Docling native timeout if available; otherwise call directly (no fallback wrapper)."""
        try:
            sig_all = inspect.signature(self.converter.convert_all)  # type: ignore[attr-defined]
            timeout_kw = next((n for n in ("timeout", "timeout_s") if n in sig_all.parameters), None)
        except Exception:
            timeout_kw = None

        kw = dict(raises_on_error=False)
        kw.update(kwargs)

        backend_cls = getattr(self, "_active_pdf_backend", None)
        is_native_backend = backend_cls is DoclingParseV2DocumentBackend if backend_cls else False

        if timeout_kw and not is_native_backend:
            kw[timeout_kw] = int(timeout_s)
            return list(self.converter.convert_all(files, **kw))  # type: ignore

        results = []
        for f in files:
            results.append(self._convert_with_timeout(Path(f), timeout_s, **kwargs))
        return results

    def _convert_with_timeout(self, file: Path, timeout_s: int, **kwargs):
        timeout_kw = self._supports_native_timeout()
        kw = dict(raises_on_error=False)
        kw.update(kwargs)
        if timeout_kw:
            kw[timeout_kw] = int(timeout_s)
        return self.converter.convert(file, **kw)  # type: ignore
                    
    def set_log_file(self, logfile):
        """Set the log file path."""
        self.log_file = logfile

    def get_log_file(self):
        """Get the current log file path."""
        return self.log_file

    def enable_accel(self, threads, type='Auto'):
        """
        Enable acceleration for document processing.
        
        Args:
            threads (int): Number of threads to use
            type (str): Type of acceleration ('CUDA', 'MPS', 'Auto', or 'CPU')
        """
        # Accept common values and keep string devices like "cuda:1" when provided
        t = str(type or 'Auto').strip()
        tl = t.lower()
        if tl.startswith(('cuda', 'mps', 'cpu')):
            # Always pass a lowercase device string to AcceleratorOptions
            if ':' in tl:
                dev = tl  # e.g., 'cuda:0'
            elif tl in ('cpu', 'mps'):
                dev = tl
            else:
                dev = 'cuda'
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=dev
            )
        elif t == 'Auto':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.AUTO
            )
        elif t == 'CUDA':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.CUDA
            )
        elif t == 'MPS':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.MPS
            )
        elif t == 'CPU':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.CPU
            )
        else:
            print('Error: Wrong Acceleration type. Defaulting to Auto')
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.AUTO
            )

    def _current_device_str(self) -> str:
        try:
            dev = getattr(self.pipeline_options, "accelerator_options", None)
            dv = getattr(dev, "device", None)
            return str(dv)
        except Exception:
            return ""

    def _cfg_signature(
        self,
        *,
        enable_ocr: bool,
        force_full_page_ocr: bool,
        text_score: float,
        images_scale: float,
        formula_enrichment: bool,
        code_enrichment: bool,
        use_cls: bool,
        ocr_langs: list[str] | None,
        profile_timings: bool,
    ) -> tuple:
        langs = tuple((ocr_langs or ["el", "en"]))
        return (
            self._current_device_str(),
            bool(enable_ocr), bool(force_full_page_ocr), float(text_score), float(images_scale),
            bool(formula_enrichment), bool(code_enrichment), bool(use_cls), bool(profile_timings), langs,
            bool(getattr(self, "use_pypdfium_backend", False)),
        )

    def ensure_extractor(
        self,
        *,
        enable_ocr: bool = False,
        force_full_page_ocr: bool = False,
        text_score: float = 0.45,
        images_scale: float = 1.25,
        formula_enrichment: bool = False,
        code_enrichment: bool = False,
        use_cls: bool = False,
        ocr_langs: list[str] | None = None,
        profile_timings: bool = True,
    ):
        """Ensure a converter exists; rebuild only when configuration changes."""
        # Mark extractor as supporting external state updates when pooling is active
        if enable_ocr:
            self.external_state_updates = True

        sig = self._cfg_signature(
            enable_ocr=enable_ocr,
            force_full_page_ocr=force_full_page_ocr,
            text_score=text_score,
            images_scale=images_scale,
            formula_enrichment=formula_enrichment,
            code_enrichment=code_enrichment,
            use_cls=use_cls,
            ocr_langs=ocr_langs,
            profile_timings=profile_timings,
        )
        if getattr(self, "converter", None) is not None and self._last_extractor_cfg == sig:
            try:
                self._log.info("Reusing existing Docling converter (config unchanged)")
            except Exception:
                pass
            return
        return self.create_extractor(
            enable_ocr=enable_ocr,
            force_full_page_ocr=force_full_page_ocr,
            text_score=text_score,
            images_scale=images_scale,
            formula_enrichment=formula_enrichment,
            code_enrichment=code_enrichment,
            use_cls=use_cls,
            ocr_langs=ocr_langs,
            profile_timings=profile_timings,
        )

    def create_extractor(
        self,
        *,
        enable_ocr: bool = False,
        force_full_page_ocr: bool = False,
        text_score: float = 0.45,
        images_scale: float = 1.25,
        formula_enrichment: bool = False,
        code_enrichment: bool = False,
        use_cls: bool = False,
        ocr_langs: list[str] | None = None,
        profile_timings: bool = True,
    ):
        """Create a document converter with configured options using the canonical builder.

        Delegates PDF pipeline construction to `glossapi._pipeline.build_rapidocr_pipeline`
        to avoid duplicated provider checks and option wiring. Falls back to the legacy
        inline path if the canonical builder is unavailable.
        """
        _ensure_docling_converter_loaded()
        _ensure_docling_pipeline_loaded()
        # Enable/disable Docling pipeline timings collection (for benchmarks)
        try:
            from docling.datamodel.settings import settings as _settings  # type: ignore
            _settings.debug.profile_pipeline_timings = bool(profile_timings)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Record the PDF backend name for provenance (default to native backend)
        self.pdf_backend_name = "docling_parse_v2"
        self._active_pdf_backend = DoclingParseV2DocumentBackend

        # Best-effort Torch preflight only if Phase‑1 is asked to do enrichment
        try:
            if formula_enrichment:
                torch_mod = _maybe_import_torch(force=True)
                if torch_mod is None:
                    raise RuntimeError("Torch not available but formula enrichment requested.")
                if hasattr(torch_mod, "cuda") and isinstance(getattr(self, "pipeline_options", None), PdfPipelineOptions):
                    dev = getattr(self.pipeline_options, "accelerator_options", None)
                    dv = getattr(dev, "device", None)
                    if (isinstance(dv, str) and dv.lower().startswith("cuda")) and not torch_mod.cuda.is_available():
                        raise RuntimeError("Torch CUDA not available but formula enrichment requested.")
        except Exception as e:
            raise RuntimeError(f"Torch CUDA preflight failed: {e}")

        # Build PDF pipeline via the canonical builder (preferred)
        opts = None
        active_backend = DoclingParseV2DocumentBackend
        try:
            from ._pipeline import build_layout_pipeline, build_rapidocr_pipeline  # type: ignore
            device_str = self._current_device_str() or "cuda:0"
            builder = build_rapidocr_pipeline if enable_ocr else build_layout_pipeline
            engine, opts = builder(
                device=device_str,
                images_scale=float(images_scale),
                formula_enrichment=bool(formula_enrichment),
                code_enrichment=bool(code_enrichment),
                **({"text_score": float(text_score)} if enable_ocr else {}),
            )

            if enable_ocr and hasattr(opts, "ocr_options") and getattr(opts, "ocr_options", None) is not None:
                if use_cls is not None:
                    setattr(opts.ocr_options, "use_cls", bool(use_cls))  # type: ignore[attr-defined]
                if ocr_langs:
                    setattr(opts.ocr_options, "lang", list(ocr_langs))  # type: ignore[attr-defined]
                if force_full_page_ocr is not None:
                    setattr(opts.ocr_options, "force_full_page_ocr", bool(force_full_page_ocr))  # type: ignore[attr-defined]

            try:
                setattr(opts, "images_scale", float(images_scale))
            except Exception:
                pass

            self._active_pdf_options = opts
            self._current_ocr_enabled = bool(enable_ocr)

            # Create a multi-format DocumentConverter using the built PDF options
            pdf_backend = DoclingParseV2DocumentBackend
            if not enable_ocr:
                try:
                    if getattr(self, "use_pypdfium_backend", False):
                        pdf_backend = PyPdfiumDocumentBackend
                        self.pdf_backend_name = "pypdfium"
                except Exception:
                    pdf_backend = DoclingParseV2DocumentBackend
            if opts is None:
                opts = self.pipeline_options
            active_backend = pdf_backend

            self.converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.XML_JATS,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.CSV,
                    InputFormat.MD,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=opts,
                        pipeline_cls=StandardPdfPipeline,
                        backend=active_backend,
                    ),
                    InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                    InputFormat.XML_JATS: XMLJatsFormatOption(),
                    InputFormat.HTML: HTMLFormatOption(),
                    InputFormat.PPTX: PowerpointFormatOption(),
                    InputFormat.CSV: CsvFormatOption(),
                    InputFormat.MD: MarkdownFormatOption(),
                },
            )
            self._active_pdf_backend = active_backend
        except Exception:
            # Fallback to legacy inline configuration path
            if enable_ocr:
                r = resolve_packaged_onnx_and_keys()
                if not (r.det and r.rec and r.cls and r.keys):
                    raise FileNotFoundError(
                        "RapidOCR ONNX models/keys not found. Ensure models exist under glossapi.models/rapidocr or set GLOSSAPI_RAPIDOCR_ONNX_DIR."
                    )
                langs = ocr_langs or ["el", "en"]
                ocr_opts = RapidOcrOptions(
                    backend="onnxruntime",
                    lang=langs,
                    force_full_page_ocr=bool(force_full_page_ocr),
                    use_det=True,
                    use_cls=bool(use_cls),
                    use_rec=True,
                    text_score=float(text_score),
                    det_model_path=r.det,
                    rec_model_path=r.rec,
                    cls_model_path=r.cls,
                    print_verbose=False,
                )
                ocr_opts.rec_keys_path = r.keys
                self.pipeline_options.ocr_options = ocr_opts
            # Attach core toggles to existing pipeline_options
            try:
                self.pipeline_options.do_ocr = bool(enable_ocr)
                self.pipeline_options.do_formula_enrichment = bool(formula_enrichment)
                self.pipeline_options.do_code_enrichment = bool(code_enrichment)
                try:
                    setattr(self.pipeline_options, "images_scale", float(images_scale))
                except Exception:
                    pass
            except Exception:
                pass
            if not enable_ocr:
                try:
                    setattr(self.pipeline_options, "ocr_options", None)
                except Exception:
                    pass

            pdf_backend = DoclingParseV2DocumentBackend
            if not enable_ocr:
                try:
                    if getattr(self, "use_pypdfium_backend", False):
                        pdf_backend = PyPdfiumDocumentBackend
                        self.pdf_backend_name = "pypdfium"
                except Exception:
                    pdf_backend = DoclingParseV2DocumentBackend

            active_backend = pdf_backend

            self.converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.DOCX,
                    InputFormat.XML_JATS,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.CSV,
                    InputFormat.MD,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=self.pipeline_options,
                        pipeline_cls=StandardPdfPipeline,
                        backend=active_backend,
                    ),
                },
            )

            self._active_pdf_options = self.pipeline_options
            self._current_ocr_enabled = bool(enable_ocr)
            self._active_pdf_backend = active_backend

        # Record last configuration for reuse
        try:
            self._last_extractor_cfg = self._cfg_signature(
                enable_ocr=enable_ocr,
                force_full_page_ocr=force_full_page_ocr,
                text_score=text_score,
                images_scale=images_scale,
                formula_enrichment=formula_enrichment,
                code_enrichment=code_enrichment,
                use_cls=use_cls,
                ocr_langs=ocr_langs,
                profile_timings=profile_timings,
            )
        except Exception:
            self._last_extractor_cfg = None
    
    def _load_processing_state(self, state_file: Path) -> Dict[str, Set[str]]:
        """
        Load the processing state from a pickle file.
        
        Args:
            state_file: Path to the pickle file
            
        Returns:
            Dictionary with 'processed' and 'problematic' sets of filenames
        """
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self._log.warning(f"Failed to load processing state: {e}. Starting fresh.")
        
        # If no state file or loading failed, check if output directory has existing files
        output_dir = state_file.parent
        if output_dir.exists():
            self._log.info(f"No state file found, checking for existing output files in {output_dir}")
            # Get all markdown files in the output directory
            processed_files = set()
            try:
                for md_file in output_dir.glob("*.md"):
                    # Extract the base filename without extension
                    base_name = md_file.stem
                    # For each likely input format, add a possible filename to the set
                    for ext in ['pdf', 'docx', 'xml', 'html', 'pptx', 'csv', 'md']:
                        processed_files.add(f"{base_name}.{ext}")
                if processed_files:
                    self._log.info(f"Found {len(processed_files) // 7} existing markdown files in output directory")
            except Exception as e:
                self._log.error(f"Error while scanning existing files: {e}")
                
            return {'processed': processed_files, 'problematic': set()}
                
        # Default state structure if file doesn't exist or can't be loaded
        return {'processed': set(), 'problematic': set()}
    
    def _save_processing_state(self, state: Dict[str, Set[str]], state_file: Path) -> None:
        """
        Save the processing state to a pickle file.
        
        Args:
            state: Dictionary with 'processed' and 'problematic' sets of filenames
            state_file: Path to the pickle file
        """
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            self._log.error(f"Failed to save processing state: {e}")

    def _get_pdf_page_count(self, file_path: Path) -> Optional[int]:
        """
        Lightweight preflight to get total page count for a PDF.
        Uses pypdfium2 which is already available via Docling backend.
        Returns None if not a PDF or on failure.
        """
        try:
            if file_path.suffix.lower() != ".pdf":
                return None
            import pypdfium2 as pdfium  # local import to avoid hard dep at import time
            pdf = pdfium.PdfDocument(str(file_path))
            n = len(pdf)
            # Explicit close to release resources
            try:
                pdf.close()
            except Exception:
                pass
            return int(n)
        except Exception as e:
            self._log.warning(f"Failed to get page count for {file_path}: {e}")
            return None

    def _process_file_chunked(self, file_path: Path, output_dir: Path, timeout_dir: Optional[Path] = None) -> bool:
        """
        Process a single long PDF in chunks using Docling's native page_range.
        Writes per-chunk markdown under chunks/{stem}/ and assembles final {stem}.md.
        Also writes a manifest {stem}.chunks.json with per-chunk status and timings.

        Returns True if all chunks succeeded (or partial-success) and final MD assembled; False otherwise.
        """
        stem = file_path.stem
        page_count = self._get_pdf_page_count(file_path)
        if page_count is None:
            # Fallback: treat as normal file (not chunked)
            try:
                conv_results = self._convert_all_with_timeout([file_path], timeout_s=self.processing_timeout)
                # Export and determine success
                success_count, partial_success_count, failure_count = self._export_documents(conv_results, output_dir=output_dir)
                return (success_count + partial_success_count) > 0 and failure_count == 0
            except Exception as e:
                self._log.error(f"Fallback non-chunked processing failed for {file_path.name}: {e}")
                return False

        # Mark as chunked in existing metadata parquet (minimal provenance)
        try:
            self._mark_is_chunked(output_dir=output_dir, src_file=file_path)
        except Exception:
            pass

        chunk_dir = output_dir / "chunks" / stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / f"{stem}.chunks.json"
        manifest: Dict[str, Any] = {
            "file": str(file_path),
            "page_count": page_count,
            "chunk_size": self.chunk_size,
            "entries": [],
        }

        all_segments: List[str] = []
        strikes = 0
        completed = True

        idx = 0
        for start in range(1, page_count + 1, self.chunk_size):
            end = min(start + self.chunk_size - 1, page_count)
            idx += 1
            tries = 0
            status = None
            t0 = time.time()
            last_error: Optional[str] = None

            while True:
                tries += 1
                try:
                    conv_res = self._convert_with_timeout(
                        file_path,
                        timeout_s=self.chunk_timeout_s,
                        page_range=(start, end),
                        max_num_pages=(end - start + 1),
                    )
                    if conv_res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                        markdown_content = conv_res.document.export_to_markdown()
                        fixed_content = self._fix_greek_text(markdown_content)
                        chunk_name = f"{stem}__p{start:04d}-{end:04d}.md"
                        with (chunk_dir / chunk_name).open("w", encoding="utf-8") as fp:
                            fp.write(fixed_content)
                        all_segments.append(fixed_content)
                        status = "ok" if conv_res.status == ConversionStatus.SUCCESS else "partial"
                        break
                    else:
                        status = "error"
                        last_error = "; ".join([getattr(e, "error_message", str(e)) for e in getattr(conv_res, "errors", [])]) or "unknown"
                        break
                except TimeoutError as te:
                    status = "timeout"
                    last_error = str(te)
                except Exception as e:
                    status = "error"
                    last_error = str(e)

                if status == "timeout" and tries < self.max_chunk_timeouts:
                    # retry this chunk once
                    continue
                else:
                    break

            duration_s = time.time() - t0
            manifest["entries"].append({
                "index": idx,
                "page_start": start,
                "page_end": end,
                "status": status,
                "duration_s": round(duration_s, 3),
                "retries": max(0, tries - 1),
            })

            if status == "timeout":
                strikes += 1
                if strikes >= self.max_chunk_timeouts:
                    completed = False
                    break
            if status == "error":
                completed = False
                break

        # Persist manifest
        try:
            with manifest_path.open("w", encoding="utf-8") as fp:
                json.dump(manifest, fp, ensure_ascii=False, indent=2)
        except Exception as e:
            self._log.error(f"Failed to write chunk manifest for {file_path.name}: {e}")

        if not completed:
            # Record failure/timeout provenance in parquet
            try:
                last_status = manifest["entries"][-1]["status"] if manifest.get("entries") else "error"
                self._update_extraction_metadata(
                    output_dir=output_dir,
                    src_file=file_path,
                    status=last_status,
                    extraction_mode="chunked",
                    page_count=page_count,
                    chunk_threshold=self.long_pdf_page_threshold,
                    chunk_size=self.chunk_size,
                    chunk_count=len(manifest.get("entries", [])),
                    chunk_manifest_path=manifest_path,
                )
            except Exception as e:
                self._log.warning(f"Failed to record chunked extraction metadata for {file_path.name}: {e}")
            # Copy source file to timeout_dir for inspection if provided
            if timeout_dir is not None:
                try:
                    copy2(file_path, timeout_dir / file_path.name)
                except Exception as e:
                    self._log.error(f"Failed to copy timeout/failed file {file_path.name}: {e}")
            return False

        # Assemble final markdown
        try:
            final_md = "\n\n".join(all_segments)
            out_md_path = output_dir / f"{stem}.md"
            with out_md_path.open("w", encoding="utf-8") as fp:
                fp.write(final_md)
        except Exception as e:
            self._log.error(f"Failed to assemble final markdown for {file_path.name}: {e}")
            return False
        # Record success provenance in parquet
        try:
            self._update_extraction_metadata(
                output_dir=output_dir,
                src_file=file_path,
                status="ok",
                extraction_mode="chunked",
                page_count=page_count,
                chunk_threshold=self.long_pdf_page_threshold,
                chunk_size=self.chunk_size,
                chunk_count=len(manifest.get("entries", [])),
                chunk_manifest_path=manifest_path,
            )
        except Exception as e:
            self._log.warning(f"Failed to record chunked extraction metadata for {file_path.name}: {e}")

        return True
    
    def _get_unprocessed_files(self, input_doc_paths: List[Path], 
                              processed_files: Set[str], 
                              problematic_files: Set[str]) -> List[Path]:
        """
        Get the list of files that haven't been processed yet.
        
        Args:
            input_doc_paths: List of input file paths
            processed_files: Set of filenames that have been processed
            problematic_files: Set of filenames that were problematic
            
        Returns:
            List of unprocessed file paths
        """
        # Create a list of unprocessed files
        unprocessed_files = []
        for file_path in input_doc_paths:
            filename = Path(file_path).name
            if filename not in processed_files and filename not in problematic_files:
                unprocessed_files.append(file_path)
                
        return unprocessed_files

    def _find_metadata_parquet(self, input_dir: Union[str, Path]) -> Optional[Path]:
        """
        Locate the metadata parquet (e.g. *download_results.parquet*) starting in
        ``input_dir``. The search order is:
        1. ``input_dir``
        2. ``input_dir/download_results``
        3. ``input_dir.parent``
        4. ``input_dir.parent/download_results``
        The first match is cached in ``self._metadata_parquet_path`` so later
        look-ups are O(1).
        """
        if self._metadata_parquet_path is not None:
            return self._metadata_parquet_path

        from glossapi.parquet_schema import ParquetSchema  # local import to avoid circular deps
        import logging

        logger = logging.getLogger(__name__)
        input_dir = Path(input_dir)

        parquet_schema = ParquetSchema({'url_column': getattr(self, 'url_column', 'url')})
        logger.info(f"Using URL column: {parquet_schema.url_column}")

        input_parquet_path: Optional[Path] = parquet_schema.find_metadata_parquet(input_dir, require_url_column=False)

        # Fallback: look inside download_results sub-directory
        if input_parquet_path is None:
            download_results_dir = input_dir / "download_results"
            if download_results_dir.exists():
                input_parquet_path = parquet_schema.find_metadata_parquet(download_results_dir, require_url_column=False)

        # Additional fallbacks: parent directory and its download_results
        if input_parquet_path is None:
            parent_dir = input_dir.parent
            if parent_dir.exists():
                input_parquet_path = parquet_schema.find_metadata_parquet(parent_dir, require_url_column=False)
                if input_parquet_path is None:
                    parent_download_dir = parent_dir / "download_results"
                    if parent_download_dir.exists():
                        input_parquet_path = parquet_schema.find_metadata_parquet(parent_download_dir, require_url_column=False)

        if input_parquet_path is not None:
            self._metadata_parquet_path = input_parquet_path
            logger.info(f"Found metadata parquet file: {input_parquet_path}")

        return input_parquet_path

    def _ensure_metadata_parquet(self, markdown_dir: Path) -> Optional[Path]:
        """Ensure a metadata parquet exists and return its path.

        We search via _find_metadata_parquet starting from the markdown_dir. If none is
        found, create a basic one under the pipeline root (parent of markdown_dir)
        using currently available markdown files.
        """
        try:
            from glossapi.parquet_schema import ParquetSchema
        except Exception as e:
            self._log.error(f"ParquetSchema import failed: {e}")
            return None

        existing = self._find_metadata_parquet(markdown_dir)
        if existing is not None:
            return existing

        # Create a basic parquet from markdown files
        pipeline_root = markdown_dir.parent
        schema = ParquetSchema({'url_column': getattr(self, 'url_column', 'url')})
        created = schema.create_basic_metadata_parquet(markdown_dir, pipeline_root)
        if created is None:
            self._log.warning("Could not create a basic metadata parquet; extraction metadata will not be recorded.")
        else:
            self._metadata_parquet_path = created
        return created

    def _mark_is_chunked(self, output_dir: Path, src_file: Path) -> None:
        """Record chunked extraction via sidecar (no direct parquet writes)."""
        try:
            root = Path(output_dir).parent
            sc_dir = root / "sidecars" / "extract"
            sc_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(src_file).stem
            p = sc_dir / f"{stem}.json"
            data = {}
            if p.exists():
                try:
                    import json as _json
                    data = _json.loads(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            data["extraction_mode"] = "chunked"
            import json as _json
            p.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self._log.warning(f"Failed to write extract sidecar for {src_file}: {e}")


    def _process_single_document(
        self,
        file_path: Path,
        *,
        output_dir: Path,
        timeout_dir: Optional[Path] = None,
    ) -> bool:
        """Process one document safely, updating metadata on failure."""
        if getattr(self, "use_pypdfium_backend", False):
            try:
                self._log.info("Using PyPDFium safe extractor for %s", Path(file_path).name)
            except Exception:
                pass
            return self._process_single_document_pypdfium(
                file_path,
                output_dir=output_dir,
                timeout_dir=timeout_dir,
            )
        filename = Path(file_path).name
        try:
            conv_results = self._convert_all_with_timeout([file_path], timeout_s=self.processing_timeout)
            success_count, partial_success_count, failure_count = self._export_documents(
                conv_results, output_dir=output_dir
            )
            if success_count > 0 or partial_success_count > 0:
                return True
            self._log.error(f"Failed to process file: {filename}")
        except TimeoutError as timeout_error:
            self._log.error(f"Timeout processing file {filename}: {timeout_error}")
            if timeout_dir:
                try:
                    copy2(file_path, timeout_dir / filename)
                    self._log.info(f"Copied timeout file to {timeout_dir / filename}")
                except Exception as copy_exc:
                    self._log.error(f"Failed to copy timeout file {filename}: {copy_exc}")
            try:
                self._update_extraction_metadata(
                    output_dir=output_dir,
                    src_file=Path(file_path),
                    status="timeout",
                    extraction_mode="standard",
                    page_count=self._get_pdf_page_count(Path(file_path)),
                )
            except Exception as meta_exc:
                self._log.warning(f"Failed to record timeout metadata for {filename}: {meta_exc}")
            return False
        except Exception as individual_error:
            self._log.error("Failed to process file %s: %s", filename, individual_error, exc_info=True)
        # Record general failure metadata
        try:
            self._update_extraction_metadata(
                output_dir=output_dir,
                src_file=Path(file_path),
                status="failure",
                extraction_mode="standard",
                page_count=self._get_pdf_page_count(Path(file_path)),
            )
        except Exception as meta_exc:
            self._log.warning(f"Failed to record failure metadata for {filename}: {meta_exc}")
        return False

    def _process_single_document_pypdfium(
        self,
        file_path: Path,
        *,
        output_dir: Path,
        timeout_dir: Optional[Path] = None,
    ) -> bool:
        """Minimal PyPDFium-based extraction path for safe mode."""
        try:
            import pypdfium2 as pdfium  # type: ignore
        except ImportError as exc:
            self._log.error("PyPDFium backend requested but pypdfium2 is not installed: %s", exc)
            return False

        stem = Path(file_path).stem
        out_md_path = output_dir / f"{stem}.md"

        try:
            doc = pdfium.PdfDocument(str(file_path))
        except Exception as exc:
            self._log.error("PyPDFium failed to open %s: %s", Path(file_path).name, exc)
            return False

        texts: List[str] = []
        page_count = 0
        try:
            page_count = len(doc)
            for index in range(page_count):
                try:
                    page = doc[index]
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text = content.replace("\r\n", "\n").replace("\r", "\n").strip()
                    if text:
                        texts.append(text)
                except Exception as page_exc:
                    self._log.warning(
                        "PyPDFium failed on page %d of %s: %s",
                        index + 1,
                        Path(file_path).name,
                        page_exc,
                    )
            markdown = "\n\n".join(texts)
            out_md_path.write_text(markdown, encoding="utf-8")
            self._update_extraction_metadata(
                output_dir=output_dir,
                src_file=Path(file_path),
                status="ok",
                extraction_mode="pypdfium",
                page_count=page_count,
            )
            return True
        except Exception as exc:
            self._log.error("PyPDFium extraction failed for %s: %s", Path(file_path).name, exc)
            try:
                self._update_extraction_metadata(
                    output_dir=output_dir,
                    src_file=Path(file_path),
                    status="failure",
                    extraction_mode="pypdfium",
                    page_count=page_count or None,
                )
            except Exception:
                pass
            if timeout_dir is not None:
                try:
                    copy2(file_path, timeout_dir / Path(file_path).name)
                except Exception:
                    pass
            return False
        finally:
            try:
                doc.close()
            except Exception:
                pass

    def _update_extraction_metadata(
        self,
        output_dir: Path,
        src_file: Path,
        status: str,
        extraction_mode: str,
        page_count: Optional[int] = None,
        chunk_threshold: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_count: Optional[int] = None,
        chunk_manifest_path: Optional[Path] = None,
        *,
        no_partial_output: bool = False,
    ) -> None:
        """Write extraction metadata to sidecar (eliminate in-worker parquet writes)."""
        try:
            root = Path(output_dir).parent
            sc_dir = root / "sidecars" / "extract"
            sc_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(src_file).stem
            p = sc_dir / f"{stem}.json"
            data = {}
            if p.exists():
                try:
                    import json as _json
                    data = _json.loads(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    data = {}
            # Update fields
            data["extraction_mode"] = str(extraction_mode)
            if page_count is not None:
                data["page_count"] = int(page_count)
            if extraction_mode == "chunked":
                if chunk_threshold is not None:
                    data["chunk_threshold"] = int(chunk_threshold)
                if chunk_size is not None:
                    data["chunk_size"] = int(chunk_size)
                if chunk_count is not None:
                    data["chunk_count"] = int(chunk_count)
                if chunk_manifest_path is not None:
                    data["chunk_manifest_path"] = str(chunk_manifest_path)
            # Backend and failure
            backend_name = getattr(self, "pdf_backend_name", None) or ("docling_parse_v2" if getattr(self, "USE_V2", True) else "docling_parse")
            data["extraction_backend"] = backend_name
            if status in ("timeout", "error", "failure"):
                data["failure_mode"] = status
            if no_partial_output:
                data["no_partial_output"] = True
            import json as _json
            p.write_text(_json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self._log.warning(f"Failed to write extraction sidecar for {src_file}: {e}")

    
    # Removed legacy SIGALRM timeout; rely on Docling native timeout when available
    
    def _process_batch(self, batch: List[Path], output_dir: Path, timeout_dir: Path = None) -> Tuple[List[str], List[str]]:
        """
        Process a batch of files and return the successful and problematic filenames.
        
        Args:
            batch: List of file paths to process
            output_dir: Output directory
            timeout_dir: Directory to store timeout files (optional)
            
        Returns:
            Tuple of (successful_filenames, problematic_filenames)
        """
        self._apply_thread_caps()
        policy = getattr(self, "batch_policy", "safe") or "safe"
        max_batch_files = max(1, getattr(self, "max_batch_files", 1))

        successful: List[str] = []
        problematic: List[str] = []

        # Preflight to split long PDFs (chunked) vs normal files (batch-capable)
        normal_files: List[Path] = []
        long_files: List[Path] = []
        for file_path in batch:
            try:
                if file_path.suffix.lower() == ".pdf":
                    n_pages = self._get_pdf_page_count(file_path)
                    if n_pages is not None and n_pages > self.long_pdf_page_threshold:
                        long_files.append(file_path)
                    else:
                        normal_files.append(file_path)
                else:
                    normal_files.append(file_path)
            except Exception as e:
                self._log.warning(f"Preflight failed for {file_path.name} (treating as normal): {e}")
                normal_files.append(file_path)

        if normal_files:
            if max_batch_files == 1:
                if policy != "safe" and not self.use_pypdfium_backend:
                    self._log.info("Batch policy capped to 1; processing documents sequentially.")
                for file_path in normal_files:
                    try:
                        self._log.info(
                            "Processing single document via %s backend",
                            "pypdfium" if self.use_pypdfium_backend else "docling",
                        )
                    except Exception:
                        pass
                    if self._process_single_document(file_path, output_dir=output_dir, timeout_dir=timeout_dir):
                        successful.append(Path(file_path).name)
                    else:
                        problematic.append(Path(file_path).name)
            else:
                for start in range(0, len(normal_files), max_batch_files):
                    chunk = normal_files[start : start + max_batch_files]
                    try:
                        conv_results = self._convert_all_with_timeout(
                            chunk, timeout_s=self.processing_timeout
                        )
                        self._export_documents(conv_results, output_dir=output_dir)
                        for res in conv_results:
                            fname = Path(res.input.file).name
                            if res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                                successful.append(fname)
                            else:
                                problematic.append(fname)
                    except Exception as batch_error:
                        self._log.warning(
                            "Batch processing (%d files) failed: %s. Falling back to per-file mode.",
                            len(chunk),
                            batch_error,
                        )
                        for file_path in chunk:
                            if self._process_single_document(
                                file_path, output_dir=output_dir, timeout_dir=timeout_dir
                            ):
                                successful.append(Path(file_path).name)
                            else:
                                problematic.append(Path(file_path).name)

        # Process long PDFs individually with chunking
        for file_path in long_files:
            try:
                ok = self._process_file_chunked(file_path, output_dir=output_dir, timeout_dir=timeout_dir)
                if ok:
                    successful.append(Path(file_path).name)
                else:
                    problematic.append(Path(file_path).name)
            except Exception as e:
                problematic.append(Path(file_path).name)
                self._log.error(f"Failed during chunked processing for {Path(file_path).name}: {e}")

        return successful, problematic
        
    def extract_path(self, input_doc_paths, output_dir, batch_size: int = 5, *, skip_existing: bool = True):
        """
        Extract all documents in the input paths to Markdown with robust batch processing and resumption.
        
        Args:
            input_doc_paths (List[Path]): List of paths to documents (PDF, DOCX, XML, etc.)
            output_dir (Path): Directory to save the extracted Markdown files
            batch_size (int): Number of files to process in each batch
        """
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure converter is created only when using Docling pipelines
        if self.converter is None and not getattr(self, "use_pypdfium_backend", False):
            self.create_extractor()
        
        # Create directories for problematic files and timeout files
        problematic_dir = output_dir / "problematic_files"
        problematic_dir.mkdir(exist_ok=True)
        
        # Create a separate directory specifically for timeout files
        timeout_dir = output_dir / "timeout_files"
        timeout_dir.mkdir(exist_ok=True)
        
        # State file for tracking progress
        state_file = output_dir / ".processing_state.pkl"

        if self.external_state_updates:
            processed_files = set()
            problematic_files = set()
        else:
            state = self._load_processing_state(state_file)
            if not skip_existing:
                state = {'processed': set(), 'problematic': set()}
            processed_files = state.get('processed', set())
            problematic_files = state.get('problematic', set())
        
        self._log.info(f"Found {len(processed_files)} already processed files")
        self._log.info(f"Found {len(problematic_files)} problematic files")
        
        # Convert all paths to Path objects for consistency
        input_doc_paths = [Path(p) if not isinstance(p, Path) else p for p in input_doc_paths]
        
        # Get files that haven't been processed yet
        unprocessed_files = self._get_unprocessed_files(
            input_doc_paths, processed_files, problematic_files
        )
        
        total_files = len(input_doc_paths)
        remaining_files = len(unprocessed_files)
        
        self._log.info(f"Processing {remaining_files} out of {total_files} files")
        
        # Check if all files have already been processed
        if remaining_files == 0:
            self._log.info("All files have already been processed. Nothing to do.")
            end_time = time.time() - start_time
            self._log.info(f"Document extraction verification complete in {end_time:.2f} seconds.")
            return
        
        # Process files in batches
        batch_count = (remaining_files + batch_size - 1) // batch_size  # Ceiling division
        success_count = 0
        partial_success_count = 0
        failure_count = 0
        
        backend_name = getattr(self, "pdf_backend_name", "unknown")

        for i in range(0, len(unprocessed_files), batch_size):
            batch = unprocessed_files[i:i + batch_size]
            batch_start_time = time.time()
            
            self._log.info(f"Processing batch {i//batch_size + 1}/{batch_count} ({len(batch)} files)")
            try:
                # Surface intended OCR mode for this batch
                batch_mode = "disabled"
                try:
                    if getattr(self, "_current_ocr_enabled", False):
                        ocr_opts = None
                        opts = getattr(self, "_active_pdf_options", None)
                        if opts is not None:
                            ocr_opts = getattr(opts, "ocr_options", None)
                        if ocr_opts is None:
                            ocr_opts = getattr(self.pipeline_options, "ocr_options", None)
                        forced = False
                        if ocr_opts is not None:
                            try:
                                forced = bool(getattr(ocr_opts, "force_full_page_ocr", False))
                            except Exception:
                                forced = False
                        batch_mode = "forced" if forced else "auto"
                except Exception:
                    pass
                self._log.info("Batch OCR mode: %s", batch_mode)
                self._log.info("Batch backend: %s", backend_name)
                for idx, _p in enumerate(batch, 1):
                    self._log.debug("Queueing [%d/%d]: %s", idx, len(batch), Path(_p).name)
            except Exception:
                pass
            
            # Process the batch
            successful, problematic = self._process_batch(batch, output_dir, timeout_dir)
            
            # Update counts
            success_count += len(successful)
            failure_count += len(problematic)
            
            # Update processed and problematic files
            processed_files.update(successful)
            problematic_files.update(problematic)

            if self.external_state_updates and self.batch_result_callback:
                try:
                    self.batch_result_callback(successful, problematic)
                except Exception as exc:
                    self._log.warning("Batch result callback failed: %s", exc)
            
            # Move problematic files to the problematic directory
            for filename in problematic:
                for input_path in input_doc_paths:
                    if Path(input_path).name == filename:
                        try:
                            # Create a copy of the problematic file
                            copy2(input_path, problematic_dir / filename)
                            self._log.info(f"Copied problematic file to {problematic_dir / filename}")
                            break
                        except Exception as e:
                            self._log.error(f"Failed to copy problematic file {filename}: {e}")
            
            if not self.external_state_updates:
                # Save the current state after each batch
                self._save_processing_state({
                    'processed': processed_files,
                    'problematic': problematic_files
                }, state_file)
            
            batch_duration = time.time() - batch_start_time
            self._log.info(f"Batch processed in {batch_duration:.2f} seconds")
            self._log.info(f"Progress: {len(processed_files)}/{total_files} files ({len(problematic_files)} problematic)")
        
        # Check if all files have been processed
        if len(processed_files) + len(problematic_files) >= total_files:
            self._log.info("All files have been processed")
            if not self.external_state_updates:
                # Keep the state file for resumption capabilities
                self._log.info("Preserving processing state file for resumption functionality")
        
        end_time = time.time() - start_time
        self._log.info(f"Document extraction complete in {end_time:.2f} seconds.")
        self._log.info(f"Successfully extracted: {success_count}")
        self._log.info(f"Partially extracted: {partial_success_count}")

        if failure_count > 0:
            self._log.warning(f"Failed to extract {failure_count} out of {total_files} documents.")
            
    def _fix_greek_text(self, text):
        """Fix Unicode issues in text, particularly for Greek characters."""
        return ftfy.fix_text(text)

    def _export_documents(self, conv_results: Iterable[ConversionResult], output_dir: Path):
        """
        Export extracted documents to Markdown files.
        
        Args:
            conv_results: Iterable of extraction results
            output_dir: Directory to save the Markdown files
            
        Returns:
            Tuple of (success_count, partial_success_count, failure_count)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0
        partial_success_count = 0
        
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                doc_filename = conv_res.input.file.stem

                # Export Docling document format to markdown
                markdown_content = conv_res.document.export_to_markdown()
                
                # Fix any Unicode issues in the markdown content
                fixed_content = self._fix_greek_text(markdown_content)
                
                # Write the fixed content to file
                with (output_dir / f"{doc_filename}.md").open("w", encoding='utf-8') as fp:
                    fp.write(fixed_content)
                try:
                    self._log.info("[OK] %s", Path(conv_res.input.file).name)
                except Exception:
                    pass
                # Update parquet metadata for standard extraction
                try:
                    self._update_extraction_metadata(
                        output_dir=output_dir,
                        src_file=Path(conv_res.input.file),
                        status="success",
                        extraction_mode="standard",
                        page_count=self._get_pdf_page_count(Path(conv_res.input.file)),
                    )
                except Exception as e:
                    self._log.warning(f"Failed to update extraction metadata for {doc_filename}: {e}")
                # Optional Phase-1 artifacts for later enrichment/scheduling
                try:
                    if getattr(self, "export_doc_json", False):
                        self._export_docling_json_with_meta(conv_res, output_dir, doc_filename)
                    if getattr(self, "emit_formula_index", False):
                        self._emit_formula_index(conv_res, output_dir, doc_filename)
                except Exception as e:
                    self._log.warning(f"Optional Phase-1 artifacts failed for {doc_filename}: {e}")

                # Optionally can also export to other formats like JSON
                # with (output_dir / f"{doc_filename}.json").open("w", encoding='utf-8') as fp:
                #     json.dump(conv_res.document.export_to_dict(), fp, ensure_ascii=False, indent=2)

            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                self._log.info(
                    f"Document {conv_res.input.file} was partially extracted with the following errors:"
                )
                for item in conv_res.errors:
                    self._log.info(f"\t{item.error_message}")
                
                # Still try to export the partial content
                doc_filename = conv_res.input.file.stem
                markdown_content = conv_res.document.export_to_markdown()
                fixed_content = self._fix_greek_text(markdown_content)

                note = "markdown suppressed" if fixed_content.strip() else "empty markdown"
                try:
                    self._log.info("[PARTIAL] %s (%s)", Path(conv_res.input.file).name, note)
                except Exception:
                    pass

                partial_success_count += 1
                # Update parquet metadata for partial standard extraction
                try:
                    self._update_extraction_metadata(
                        output_dir=output_dir,
                        src_file=Path(conv_res.input.file),
                        status="partial",
                        extraction_mode="standard",
                        page_count=self._get_pdf_page_count(Path(conv_res.input.file)),
                        no_partial_output=True,
                    )
                except Exception as e:
                    self._log.warning(f"Failed to update extraction metadata for {doc_filename}: {e}")
                # Optional Phase-1 artifacts for later enrichment/scheduling
                try:
                    if getattr(self, "export_doc_json", False):
                        self._export_docling_json_with_meta(conv_res, output_dir, doc_filename)
                    if getattr(self, "emit_formula_index", False):
                        self._emit_formula_index(conv_res, output_dir, doc_filename)
                except Exception as e:
                    self._log.warning(f"Optional Phase-1 artifacts failed for {doc_filename}: {e}")
            else:
                # Attempt best-effort export even on failure if a document exists
                self._log.info(f"Document {conv_res.input.file} failed to extract.")
                try:
                    if getattr(conv_res, "document", None) is not None:
                        doc_filename = getattr(conv_res.input.file, "stem", None)
                        if not doc_filename:
                            try:
                                doc_filename = Path(conv_res.input.file).stem
                            except Exception:
                                doc_filename = None
                        if doc_filename:
                            markdown_content = conv_res.document.export_to_markdown()
                            fixed_content = self._fix_greek_text(markdown_content)
                            note = "markdown suppressed" if fixed_content.strip() else "empty markdown"
                            partial_success_count += 1
                            try:
                                self._log.info("[FAIL->PARTIAL] %s (%s)", Path(conv_res.input.file).name, note)
                            except Exception:
                                pass
                        else:
                            failure_count += 1
                    else:
                        failure_count += 1
                except Exception:
                    failure_count += 1
                # Update parquet metadata for failed standard extraction
                try:
                    self._update_extraction_metadata(
                        output_dir=output_dir,
                        src_file=Path(conv_res.input.file),
                        status="failure",
                        extraction_mode="standard",
                        page_count=self._get_pdf_page_count(Path(conv_res.input.file)),
                        no_partial_output=True,
                    )
                except Exception as e:
                    self._log.warning(f"Failed to update extraction metadata for {Path(conv_res.input.file).name}: {e}")
                # Optional Phase-1 artifacts even on fail->partial
                try:
                    if doc_filename and getattr(self, "export_doc_json", False):
                        self._export_docling_json_with_meta(conv_res, output_dir, doc_filename)
                    if doc_filename and getattr(self, "emit_formula_index", False):
                        self._emit_formula_index(conv_res, output_dir, doc_filename)
                except Exception as e:
                    self._log.warning(f"Optional Phase-1 artifacts failed for {doc_filename}: {e}")
            # Write per-document metrics JSON (Docling timings) and per-page metrics (skipped in benchmark mode)
            if not getattr(self, "benchmark_mode", False):
                try:
                    import json as _json
                    # Per-document timings (same structure as module runner)
                    metrics = {"file": str(getattr(conv_res.input.file, 'name', 'unknown')), "timings": {}}
                    for key, item in conv_res.timings.items():
                        times = list(item.times)
                        cnt = int(item.count)
                        tot = float(sum(times)) if times else 0.0
                        avg = float(tot / cnt) if cnt else 0.0
                        metrics["timings"][key] = {
                            "scope": str(item.scope.value) if hasattr(item, "scope") else "unknown",
                            "count": cnt,
                            "total_sec": tot,
                            "avg_sec": avg,
                            "p50_sec": float(times[int(round((len(times)-1)*0.50))]) if times else 0.0,
                            "p90_sec": float(times[int(round((len(times)-1)*0.90))]) if times else 0.0,
                            "times_sec": times,
                        }
                    # Write metrics under the JSON artifacts tree for better separation
                    json_dir = output_dir.parent / "json"
                    metrics_dir = json_dir / "metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    mpath = metrics_dir / f"{doc_filename}.metrics.json"
                    with mpath.open("w", encoding="utf-8") as fp:
                        fp.write(_json.dumps(metrics, ensure_ascii=False, indent=2))
                    # Per-page metrics
                    try:
                        per_page = self._compute_per_page_metrics(conv_res)
                        pppath = metrics_dir / f"{doc_filename}.per_page.metrics.json"
                        with pppath.open("w", encoding="utf-8") as fp:
                            fp.write(_json.dumps(per_page, ensure_ascii=False, indent=2))
                        # Emit concise per-page log line (debug only)
                        for row in per_page.get("pages", []):
                            if self._log.isEnabledFor(logging.DEBUG):
                                self._log.debug(
                                    "[PAGE] %s p%d: parse=%.3fs ocr=%.3fs formulas=%d code=%d",
                                    getattr(conv_res.input.file, 'name', doc_filename),
                                    int(row.get("page_no", 0)),
                                    float(row.get("parse_sec", 0.0)),
                                    float(row.get("ocr_sec", 0.0)),
                                    int(row.get("formula_count", 0)),
                                    int(row.get("code_count", 0)),
                                )
                    except Exception as _e:
                        self._log.warning("Failed to compute per-page metrics for %s: %s", doc_filename, _e)
                except Exception as _e:
                    self._log.debug("Metrics export failed for %s: %s", doc_filename, _e)

        return success_count, partial_success_count, failure_count

    def _export_docling_json_with_meta(self, conv_res: ConversionResult, output_dir: Path, stem: str) -> None:
        """Export DoclingDocument JSON (compressed) with metadata stamping for Phase-1."""
        try:
            from .json_io import export_docling_json, _sha256_file  # type: ignore
        except Exception:
            self._log.warning("glossapi.json_io not available; skipping JSON export")
            return
        try:
            # Attempt to resolve the original file path for hashing
            pdf_path = None
            try:
                pdf_path = Path(getattr(conv_res.input.file, 'path', None) or conv_res.input.file)  # type: ignore
            except Exception:
                pass
            doc_id = _sha256_file(pdf_path) if (pdf_path and pdf_path.exists()) else ""
            meta = {
                "schema_version": 1,
                "sha256_pdf": doc_id,
                "page_count": len(getattr(conv_res.document, 'pages', []) or []),
                "source_pdf_relpath": str(pdf_path) if pdf_path else "",
            }
            json_dir = output_dir.parent / "json"
            json_dir.mkdir(parents=True, exist_ok=True)
            out_json = json_dir / f"{stem}.docling.json"
            export_docling_json(conv_res.document, out_json, compress="zstd", meta=meta)  # type: ignore
        except Exception as e:
            self._log.warning(f"JSON export failed for {stem}: {e}")

    def _emit_formula_index(self, conv_res: ConversionResult, output_dir: Path, stem: str) -> int:
        """Emit formula/code index JSONL from typed DoclingDocument for Phase-1 scheduling.

        Returns the number of items written.
        """
        try:
            doc = conv_res.document
            # Protect against missing iterate_items
            it = getattr(doc, 'iterate_items', None)
            if not callable(it):
                return 0
            # Best-effort page dimensions for normalized bbox
            pages = []
            try:
                pages = list(getattr(doc, 'pages', []) or [])
            except Exception:
                pages = []
            # Compute sha256 of source PDF if available
            try:
                from .json_io import _sha256_file  # type: ignore
                from pathlib import Path as _Path
                pdf_path = _Path(getattr(getattr(conv_res, 'input', None), 'file', ''))
                sha256_pdf = _sha256_file(pdf_path) if (pdf_path and pdf_path.exists()) else ""
            except Exception:
                sha256_pdf = ""
            per_page_ix: dict[int, int] = {}
            out_lines: list[str] = []
            for element, _level in it():
                try:
                    label = getattr(element, 'label', '')
                except Exception:
                    continue
                lab = str(label).lower()
                if lab not in {"formula", "code"}:
                    continue
                prov = getattr(element, 'prov', []) or []
                for p in prov:
                    try:
                        page_no = int(getattr(p, 'page_no', 0))
                        bbox = getattr(p, 'bbox', None)
                    except Exception:
                        continue
                    if not page_no or bbox is None:
                        continue
                    per_page_ix[page_no] = per_page_ix.get(page_no, 0) + 1
                    row = {
                        "page_no": page_no,
                        "label": lab,
                        "item_index": per_page_ix[page_no],
                        "bbox_pdf_pt": {
                            "l": float(getattr(bbox, 'l', 0.0)),
                            "t": float(getattr(bbox, 't', 0.0)),
                            "r": float(getattr(bbox, 'r', 0.0)),
                            "b": float(getattr(bbox, 'b', 0.0)),
                            "origin": str(getattr(getattr(bbox, 'coord_origin', ''), 'value', getattr(bbox, 'coord_origin', ''))),
                        },
                        "doc_sha256": sha256_pdf,
                        "placeholder_id": f"p:{page_no}-i:{per_page_ix[page_no]}",
                    }
                    # Add normalized bbox if page dimensions available
                    try:
                        W = H = 0.0
                        if pages and 1 <= page_no <= len(pages):
                            pg = pages[page_no - 1]
                            W = float(getattr(pg, 'width', 0.0) or getattr(getattr(pg, 'bbox', None), 'r', 0.0))
                            H = float(getattr(pg, 'height', 0.0) or getattr(getattr(pg, 'bbox', None), 'b', 0.0))
                        if W and H:
                            l = float(getattr(bbox, 'l', 0.0)); t = float(getattr(bbox, 't', 0.0)); r = float(getattr(bbox, 'r', 0.0)); b = float(getattr(bbox, 'b', 0.0))
                            row["bbox_norm"] = {"l": l / W, "t": t / H, "r": r / W, "b": b / H}
                    except Exception:
                        pass
                    # Rotation, display guess, optional context
                    try:
                        row["rotation_deg"] = int(getattr(p, 'rotation', 0) or 0)
                    except Exception:
                        row["rotation_deg"] = 0
                    try:
                        disp = "inline" if bool(getattr(element, 'is_inline', False)) else "block"
                    except Exception:
                        disp = "block"
                    row["display"] = disp
                    try:
                        ctx = str(getattr(element, 'text', '') or getattr(element, 'orig', '') or '')
                        if ctx:
                            ctx = ctx.strip()
                            if len(ctx) > 80:
                                ctx = ctx[:77] + "…"
                        row["context"] = ctx
                    except Exception:
                        pass
                    out_lines.append(json.dumps(row, ensure_ascii=False))
            if out_lines:
                json_dir = output_dir.parent / "json"
                json_dir.mkdir(parents=True, exist_ok=True)
                idx_path = json_dir / f"{stem}.formula_index.jsonl"
                with idx_path.open("w", encoding="utf-8") as fp:
                    fp.write("\n".join(out_lines) + "\n")
                return len(out_lines)
        except Exception as e:
            self._log.warning(f"Formula index emit failed for {stem}: {e}")
            return 0

    def _compute_per_page_metrics(self, conv_res: ConversionResult):
        # Delegate to the shared helper
        try:
            from .metrics import compute_per_page_metrics  # type: ignore
            return compute_per_page_metrics(conv_res)
        except Exception:
            return {"pages": []}
