from typing import Dict, Set, List, Optional, Iterable, Tuple, Any, Union

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
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
from docling.datamodel.document import ConversionResult
from docling.datamodel.settings import settings
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    HTMLFormatOption,
    XMLJatsFormatOption,
    PowerpointFormatOption,
    MarkdownFormatOption,
    CsvFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from ._rapidocr_paths import resolve_packaged_onnx_and_keys
import inspect

import ftfy
import logging
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
        # Default to GPU-first OCR with auto/full-page control passed in create_extractor
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        # Enable accurate table structure by default
        try:
            self.pipeline_options.table_structure_options = TableStructureOptions(mode=TableFormerMode.ACCURATE)
            self.pipeline_options.table_structure_options.do_cell_matching = True
        except Exception:
            pass
        # Default enrichment on
        try:
            self.pipeline_options.do_formula_enrichment = True
            self.pipeline_options.do_code_enrichment = True
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
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(self.log_file), mode='w'),  
                logging.StreamHandler()
            ]
        )
        # Per-instance logger
        self._log = logging.getLogger(__name__)

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
        timeout_kw = self._supports_native_timeout()
        kw = dict(raises_on_error=False)
        kw.update(kwargs)
        if timeout_kw:
            kw[timeout_kw] = int(timeout_s)
        return list(self.converter.convert_all(files, **kw))  # type: ignore

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
        if type == 'CUDA':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.CUDA
            )
        elif type == 'MPS':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.MPS
            )
        elif type == 'Auto':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.AUTO
            )
        elif type == 'CPU':
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.CPU
            )
            print('Error : Wrong Acceleration type. Defaulting to Auto')
            self.pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=threads, device=AcceleratorDevice.AUTO
            )

    def create_extractor(
        self,
        *,
        enable_ocr: bool = True,
        force_full_page_ocr: bool = False,
        text_score: float = 0.45,
        images_scale: float = 1.25,
        formula_enrichment: bool = True,
        code_enrichment: bool = True,
        ocr_langs: list[str] | None = None,
    ):
        """Create a document converter with configured options and RapidOCR (ONNX).

        Parameters control OCR and enrichment. Models and keys are resolved from
        packaged assets (or via env override) using resolve_packaged_onnx_and_keys().
        """
        # GPU-only preflight (enforce ORT CUDA provider; Torch CUDA when formula enrichment is enabled)
        try:
            import onnxruntime as _ort  # type: ignore
            _providers = _ort.get_available_providers()
            if "CUDAExecutionProvider" not in _providers:
                raise RuntimeError(f"GPU-only policy: CUDAExecutionProvider not available in onnxruntime providers={_providers}")
        except Exception as e:
            raise RuntimeError(f"GPU-only policy: onnxruntime-gpu not available or misconfigured: {e}")
        if formula_enrichment:
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU-only policy: Torch CUDA not available but formula enrichment requested.")
            except Exception as e:
                raise RuntimeError(f"GPU-only policy: Torch CUDA preflight failed: {e}")

        # Record the PDF backend that will be used so we can write it to parquet metadata
        # Currently we use Docling v2 backend which corresponds to the "vl_parse_2" engine.
        self.pdf_backend_name = "vl_parse_2"
        # Attach OCR and enrichment settings
        try:
            self.pipeline_options.do_ocr = bool(enable_ocr)
            self.pipeline_options.do_formula_enrichment = bool(formula_enrichment)
            self.pipeline_options.do_code_enrichment = bool(code_enrichment)
            # Best-effort image scaling for better detection on thin glyphs
            try:
                setattr(self.pipeline_options, "images_scale", images_scale)
            except Exception:
                pass
        except Exception:
            pass

        # Resolve ONNX and keys and configure RapidOCR options
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
                use_cls=True,
                use_rec=True,
                text_score=float(text_score),
                det_model_path=r.det,
                rec_model_path=r.rec,
                cls_model_path=r.cls,
                print_verbose=False,
            )
            ocr_opts.rec_keys_path = r.keys
            self.pipeline_options.ocr_options = ocr_opts

            # Enable pipeline timing profile (Docling) for richer metrics
            try:
                settings.debug.profile_pipeline_timings = True  # type: ignore[attr-defined]
            except Exception:
                pass

            # Log OCR configuration (fine-grained visibility)
            try:
                import os as _os
                self._log.info(
                    "OCR enabled: backend=%s forced=%s langs=%s text_score=%.2f det=%s rec=%s cls=%s keys=%s",
                    ocr_opts.backend,
                    ocr_opts.force_full_page_ocr,
                    ",".join(ocr_opts.lang),
                    float(ocr_opts.text_score or 0.0),
                    _os.path.basename(r.det) if r.det else None,
                    _os.path.basename(r.rec) if r.rec else None,
                    _os.path.basename(r.cls) if r.cls else None,
                    _os.path.basename(r.keys) if r.keys else None,
                )
            except Exception:
                pass
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,  # .docx (Office Open XML)
                # Note: Old .doc format (pre-2007) is not supported by Docling
                InputFormat.XML_JATS,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.CSV,
                InputFormat.MD,
            ],  # whitelist formats, non-matching files are ignored
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseV2DocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline
                ),
                InputFormat.XML_JATS: XMLJatsFormatOption(),
                InputFormat.HTML: HTMLFormatOption(),
                InputFormat.PPTX: PowerpointFormatOption(),
                InputFormat.CSV: CsvFormatOption(),
                InputFormat.MD: MarkdownFormatOption(),
            }
        )
    
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
        """Mark a file as chunked in the existing metadata parquet by setting column 'extraction_mode' to 'chunked'.

        - Only updates if a metadata parquet already exists.
        - Does not create a parquet or add a new row if the filename does not exist in the parquet.
        """
        try:
            parquet_path = self._find_metadata_parquet(Path(output_dir))
            if parquet_path is None or not Path(parquet_path).exists():
                return

            df = pd.read_parquet(parquet_path)
            if 'filename' not in df.columns:
                return

            filename = Path(src_file).name
            mask = df['filename'] == filename
            if not mask.any():
                # Do not add new rows; minimal footprint
                return

            # Ensure column exists and set to 'chunked' for this file
            if 'extraction_mode' not in df.columns:
                df['extraction_mode'] = 'standard'
            df.loc[mask, 'extraction_mode'] = 'chunked'

            try:
                df.to_parquet(parquet_path, index=False, compression='zstd')
            except Exception:
                df.to_parquet(parquet_path, index=False)
        except Exception as e:
            self._log.warning(f"Failed to mark is_chunked for {src_file}: {e}")

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
    ) -> None:
        """Update or create metadata parquet row with minimal extraction details for a source file.

        Writes only:
        - extraction_backend: records Docling engine used (e.g., 'vl_parse_2'); OCR is performed inside extract
        - extraction_mode: 'standard' or 'chunked'
        - failure_mode: '', 'timeout', 'error', etc.
        - ocr_success: boolean flag for OCR outcome (left unset here; updated in Corpus.ocr())
        Also appends 'extract' to processing_stage.
        """
        try:
            parquet_path = self._ensure_metadata_parquet(Path(output_dir))
            filename = Path(src_file).name

            if parquet_path is not None and Path(parquet_path).exists():
                df = pd.read_parquet(parquet_path)
            else:
                # Create a minimal DataFrame with this single filename
                pipeline_root = Path(output_dir).parent
                download_results_dir = pipeline_root / "download_results"
                download_results_dir.mkdir(parents=True, exist_ok=True)
                parquet_path = download_results_dir / "download_results.parquet"
                df = pd.DataFrame([{ 'filename': filename, getattr(self, 'url_column', 'url'): '' }])

            # Ensure required columns exist
            for col in [
                'filename',
                'processing_stage',
                'extraction_backend',
                'extraction_mode',
                'failure_mode',
                'ocr_success',
                'page_count',
                'is_chunked',
                'chunk_threshold',
                'chunk_size',
                'chunk_count',
                'chunk_manifest_path',
            ]:
                if col not in df.columns:
                    # Initialize appropriate defaults
                    if col == 'extraction_mode':
                        df[col] = 'standard'
                    elif col == 'ocr_success':
                        df[col] = pd.NA
                    else:
                        df[col] = pd.NA

            mask = df['filename'] == filename
            if not mask.any():
                df = pd.concat([df, pd.DataFrame([{'filename': filename}] )], ignore_index=True)
                mask = df['filename'] == filename

            # Minimal extraction fields
            df.loc[mask, 'extraction_mode'] = extraction_mode
            df.loc[mask, 'page_count'] = page_count if page_count is not None else pd.NA
            df.loc[mask, 'is_chunked'] = (extraction_mode == 'chunked')
            if extraction_mode == 'chunked':
                df.loc[mask, 'chunk_threshold'] = chunk_threshold if chunk_threshold is not None else pd.NA
                df.loc[mask, 'chunk_size'] = chunk_size if chunk_size is not None else pd.NA
                df.loc[mask, 'chunk_count'] = chunk_count if chunk_count is not None else pd.NA
                df.loc[mask, 'chunk_manifest_path'] = str(chunk_manifest_path) if chunk_manifest_path else pd.NA

            # Record the Docling extraction backend used (OCR updates later)
            backend_name = getattr(self, 'pdf_backend_name', None)
            if not backend_name:
                # Fallback mapping if attribute is missing
                backend_name = 'vl_parse_2' if getattr(self, 'USE_V2', True) else 'docling_parse'
            df.loc[mask, 'extraction_backend'] = backend_name

            # Derive failure_mode from status
            failure = ''
            if status in ('timeout', 'error', 'failure'):
                failure = status
            df.loc[mask, 'failure_mode'] = (failure if failure else pd.NA)

            # processing_stage logic: append 'extract'
            def _append_stage(val: Any) -> str:
                s = '' if pd.isna(val) else str(val)
                return 'extract' if not s else (s if 'extract' in s.split(',') else f"{s},extract")

            df.loc[mask, 'processing_stage'] = df.loc[mask, 'processing_stage'].apply(_append_stage)

            # Persist with compression (zstd)
            try:
                df.to_parquet(parquet_path, index=False, compression='zstd')
            except Exception:
                # Fallback without explicit compression
                df.to_parquet(parquet_path, index=False)
        except Exception as e:
            self._log.warning(f"Failed to update extraction metadata for {src_file}: {e}")

    
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

        # First try to process normal files as a batch
        if normal_files:
            try:
                conv_results = self._convert_all_with_timeout(
                    normal_files, timeout_s=self.processing_timeout
                )
                # Export results
                success_count, partial_success_count, failure_count = self._export_documents(
                    conv_results, output_dir=output_dir
                )
                # Derive per-file outcomes from statuses
                for res in conv_results:
                    fname = Path(res.input.file).name
                    if res.status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                        successful.append(fname)
                    else:
                        problematic.append(fname)
            except Exception as batch_error:
                self._log.warning(f"Batch processing for normal files failed: {batch_error}. Falling back to per-file.")
                for file_path in normal_files:
                    try:
                        conv_results = self._convert_all_with_timeout([file_path], timeout_s=self.processing_timeout)
                        success_count, partial_success_count, failure_count = self._export_documents(
                            conv_results, output_dir=output_dir
                        )
                        if success_count > 0 or partial_success_count > 0:
                            successful.append(Path(file_path).name)
                        else:
                            problematic.append(Path(file_path).name)
                            self._log.error(f"Failed to process file: {Path(file_path).name}")
                    except TimeoutError as timeout_error:
                        filename = Path(file_path).name
                        problematic.append(filename)
                        self._log.error(f"Timeout processing file {filename}: {timeout_error}")
                        if timeout_dir:
                            try:
                                copy2(file_path, timeout_dir / filename)
                                self._log.info(f"Copied timeout file to {timeout_dir / filename}")
                            except Exception as e:
                                self._log.error(f"Failed to copy timeout file {filename}: {e}")
                        # Update parquet metadata for timeout in standard mode
                        try:
                            self._update_extraction_metadata(
                                output_dir=output_dir,
                                src_file=Path(file_path),
                                status="timeout",
                                extraction_mode="standard",
                                page_count=self._get_pdf_page_count(Path(file_path)),
                            )
                        except Exception as e:
                            self._log.warning(f"Failed to record timeout metadata for {filename}: {e}")
                    except Exception as individual_error:
                        problematic.append(Path(file_path).name)
                        self._log.error(f"Failed to process file {Path(file_path).name}: {individual_error}")
                        # Update parquet metadata for failure in standard mode
                        try:
                            self._update_extraction_metadata(
                                output_dir=output_dir,
                                src_file=Path(file_path),
                                status="failure",
                                extraction_mode="standard",
                                page_count=self._get_pdf_page_count(Path(file_path)),
                            )
                        except Exception as e:
                            self._log.warning(f"Failed to record failure metadata for {Path(file_path).name}: {e}")

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
        # Ensure converter is created
        if self.converter is None:
            self.create_extractor()
        
        # Create directories for problematic files and timeout files
        problematic_dir = output_dir / "problematic_files"
        problematic_dir.mkdir(exist_ok=True)
        
        # Create a separate directory specifically for timeout files
        timeout_dir = output_dir / "timeout_files"
        timeout_dir.mkdir(exist_ok=True)
        
        # State file for tracking progress
        state_file = output_dir / ".processing_state.pkl"
        
        # Load the current processing state
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
        
        for i in range(0, len(unprocessed_files), batch_size):
            batch = unprocessed_files[i:i + batch_size]
            batch_start_time = time.time()
            
            self._log.info(f"Processing batch {i//batch_size + 1}/{batch_count} ({len(batch)} files)")
            try:
                # Surface intended OCR mode for this batch
                forced = False
                try:
                    forced = bool(getattr(self.pipeline_options.ocr_options, "force_full_page_ocr", False))  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._log.info("Batch OCR mode: %s", "forced" if forced else "auto")
                for idx, _p in enumerate(batch, 1):
                    self._log.info("Queueing [%d/%d]: %s", idx, len(batch), Path(_p).name)
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
                
                with (output_dir / f"{doc_filename}_partial.md").open("w", encoding='utf-8') as fp:
                    fp.write(fixed_content)
                try:
                    self._log.info("[PARTIAL] %s", Path(conv_res.input.file).name)
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
                    )
                except Exception as e:
                    self._log.warning(f"Failed to update extraction metadata for {doc_filename}: {e}")
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
                            with (output_dir / f"{doc_filename}_partial.md").open("w", encoding='utf-8') as fp:
                                fp.write(fixed_content)
                            partial_success_count += 1
                            try:
                                self._log.info("[FAIL->PARTIAL] %s", Path(conv_res.input.file).name)
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
                    )
                except Exception as e:
                    self._log.warning(f"Failed to update extraction metadata for {Path(conv_res.input.file).name}: {e}")
            # Write per-document metrics JSON (Docling timings) and per-page metrics
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
                mpath = output_dir / f"{doc_filename}.metrics.json"
                with mpath.open("w", encoding="utf-8") as fp:
                    fp.write(_json.dumps(metrics, ensure_ascii=False, indent=2))
                # Per-page metrics
                try:
                    per_page = self._compute_per_page_metrics(conv_res)
                    pppath = output_dir / f"{doc_filename}.per_page.metrics.json"
                    with pppath.open("w", encoding="utf-8") as fp:
                        fp.write(_json.dumps(per_page, ensure_ascii=False, indent=2))
                    # Emit concise per-page log line
                    for row in per_page.get("pages", []):
                        self._log.info("[PAGE] %s p%d: parse=%.3fs ocr=%.3fs formulas=%d code=%d",
                                       getattr(conv_res.input.file, 'name', doc_filename),
                                       int(row.get("page_no", 0)),
                                       float(row.get("parse_sec", 0.0)),
                                       float(row.get("ocr_sec", 0.0)),
                                       int(row.get("formula_count", 0)),
                                       int(row.get("code_count", 0)))
                except Exception as _e:
                    self._log.warning("Failed to compute per-page metrics for %s: %s", doc_filename, _e)
            except Exception as _e:
                self._log.debug("Metrics export failed for %s: %s", doc_filename, _e)

        return success_count, partial_success_count, failure_count

    def _compute_per_page_metrics(self, conv_res: ConversionResult):
        try:
            doc = conv_res.document
        except Exception:
            return {"pages": []}
        try:
            page_count = len(doc.pages)  # type: ignore[attr-defined]
        except Exception:
            page_count = 0
        timings = {}
        try:
            for key, item in conv_res.timings.items():
                times = list(item.times)
                timings[key] = {
                    "scope": str(getattr(getattr(item, 'scope', None), 'value', 'unknown')),
                    "times": times,
                    "total": float(sum(times)) if times else float(getattr(item, 'total', 0.0)),
                }
        except Exception:
            pass
        def _pt(k):
            arr = timings.get(k, {}).get("times", []) or []
            if page_count and len(arr) == page_count:
                return [float(x) for x in arr]
            return [float(x) for x in (arr + [0.0] * page_count)[:page_count]]
        ocr = _pt("ocr")
        parse = _pt("page_parse")
        layout = _pt("layout")
        table = _pt("table_structure")
        fcnt = [0] * max(1, page_count)
        fch = [0] * max(1, page_count)
        ftr = [0] * max(1, page_count)
        ftrc = [0] * max(1, page_count)
        ccnt = [0] * max(1, page_count)
        try:
            as_dict = doc.export_to_dict()
            import re as _re
            _run_pat = _re.compile(r"\\\\\s*&(?P<ws>(?:\\quad|\\;|\\:|\\,|\\\\s|\s){200,})")
            _ws_collapse = _re.compile(r"(?:(?:\\quad|\\;|\\:|\\,|\\\\s)|\s){2,}")
            _CAP = 3000
            def _sanitize(s: str):
                dropped=0
                m=_run_pat.search(s)
                if m:
                    s_new=s[:m.start('ws')]; dropped+=len(s)-len(s_new); s=s_new
                if len(s)>_CAP:
                    cut=s.rfind('\\\\',0,_CAP); cut = cut if cut>=0 else _CAP; dropped+=len(s)-cut; s=s[:cut]
                s2=_ws_collapse.sub(' ', s)
                return s2, dropped
            def _walk(label, cnt, chars=False):
                for node in as_dict.get("texts", []):
                    if str(node.get("label")) != label:
                        continue
                    raw = str(node.get("text") or node.get("orig") or "")
                    txt, dropped = _sanitize(raw) if label=='formula' else (raw,0)
                    ch = len(txt)
                    for prov in node.get("prov", []) or []:
                        pno = int(prov.get("page_no") or 0)
                        if 1 <= pno <= len(cnt):
                            cnt[pno - 1] += 1
                            if chars:
                                fch[pno - 1] += ch
                            if label=='formula' and dropped:
                                ftr[pno - 1] += 1
                                ftrc[pno - 1] += int(dropped)
            _walk("formula", fcnt, True)
            _walk("code", ccnt, False)
        except Exception:
            pass
        try:
            den_total = float(timings.get("doc_enrich", {}).get("total", 0.0))
        except Exception:
            den_total = 0.0
        shares = [0.0] * max(1, page_count)
        if den_total and page_count:
            s = float(sum(fch)) or float(sum(fcnt)) or 0.0
            if s > 0:
                base = fch if sum(fch) > 0 else fcnt
                shares = [den_total * (float(x) / s) for x in base]
        rows = []
        n = max(page_count, len(ocr), len(parse))
        for i in range(n):
            rows.append({
                "page_no": i + 1,
                "ocr_sec": float(ocr[i]) if i < len(ocr) else 0.0,
                "parse_sec": float(parse[i]) if i < len(parse) else 0.0,
                "layout_sec": float(layout[i]) if i < len(layout) else 0.0,
                "table_sec": float(table[i]) if i < len(table) else 0.0,
                "formula_count": int(fcnt[i]) if i < len(fcnt) else 0,
                "formula_chars": int(fch[i]) if i < len(fch) else 0,
                "formula_truncated": int(ftr[i]) if i < len(ftr) else 0,
                "formula_truncated_chars": int(ftrc[i]) if i < len(ftrc) else 0,
                "code_count": int(ccnt[i]) if i < len(ccnt) else 0,
                "doc_enrich_share_sec": float(shares[i]) if i < len(shares) else 0.0,
            })
        return {"file": str(getattr(conv_res.input.file, 'name', 'unknown')), "page_count": int(page_count), "totals": {"doc_enrich_total_sec": den_total}, "pages": rows}
