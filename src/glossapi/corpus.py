import logging
from pathlib import Path
import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Optional, Union, List, Any, Tuple
import shutil

from .gloss_section import GlossSection
from .gloss_section_classifier import GlossSectionClassifier
from .gloss_downloader import GlossDownloader

class Corpus:
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
        
        # Verbose flag for detailed logging
        self.verbose = verbose
        
        # Store paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
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
        self.classifier = GlossSectionClassifier()
        
        self.output_dir = Path(output_dir)
        self.downloads_dir = self.output_dir / "downloads"
        self.markdown_dir = self.output_dir / "markdown"
        self.ocr_model_dir = None  # Will use default discovery or user-specified path
        self.sections_dir = self.output_dir / "sections"
        # Directory that will hold cleaned markdown after Rust-powered cleaning
        self.cleaned_markdown_dir = self.output_dir / "clean_markdown"
        # Define models_dir path but don't create the directory yet - only create it when needed
        self.models_dir = self.output_dir / "models"
        
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        os.makedirs(self.cleaned_markdown_dir, exist_ok=True)
        
        # Setup output files
        self.sections_parquet = self.sections_dir / "sections_for_annotation.parquet"
        self.classified_parquet = self.output_dir / "classified_sections.parquet"
        self.fully_annotated_parquet = self.output_dir / "fully_annotated_sections.parquet"
        
        # Initialize document type mapping
        self.filename_to_doctype = {}
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load metadata file if provided and extract document type mapping."""
        if self.metadata_path and self.metadata_path.exists():
            try:
                self.logger.info(f"Loading metadata from {self.metadata_path}")
                metadata_df = pd.read_parquet(self.metadata_path)
                
                # Debug information
                self.logger.info(f"Metadata file has {len(metadata_df)} rows and columns: {metadata_df.columns.tolist()}")
                self.logger.info(f"Sample filenames: {metadata_df['filename'].head(3).tolist()}")
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
                        # Simple dictionary mapping without extension handling
                        self.filename_to_doctype = dict(zip(
                            metadata_df['filename'], 
                            metadata_df['document_type']
                        ))
                    
                    self.logger.info(f"Loaded {len(self.filename_to_doctype)} filename-to-doctype mappings")
                else:
                    self.logger.warning("Metadata file does not contain 'filename' or 'document_type' columns")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
        else:
            if self.metadata_path:
                self.logger.warning(f"Metadata file not found: {self.metadata_path}")
    
    def clean(
        self,
        input_dir: Union[str, Path] = None,
        threshold: float = 0.10,
        num_threads: int = None,
        drop_bad: bool = True,
        *,
        ocr_model_dir: Union[str, Path, None] = None,
        force_ocr_fallback: bool = False,
    ) -> None:
        """Clean markdown files and evaluate badness using the Rust extension.

        Args:
            input_dir: Folder with `.md` files to process (defaults to `self.markdown_dir`).
            threshold: Badness threshold for optional dropping.
            num_threads: Rayon thread-count to pass to Rust.
            drop_bad: If True, files with badness_score > threshold are removed from downstream processing. Set to False to keep all files and only record the score.
            ocr_model_dir: [DEPRECATED – no effect] Use Corpus.ocr(model_dir=...) instead.
            force_ocr_fallback: [DEPRECATED – no effect] Use Corpus.ocr(force=True) instead.
        """
        import importlib
        from pathlib import Path
        import shutil
        import pandas as pd
        from glossapi.parquet_schema import ParquetSchema

        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)
        
        # Handle OCR model directory override
        if ocr_model_dir is not None:
            self.ocr_model_dir = Path(ocr_model_dir)

        # Try to import the compiled extension; build it on-the-fly if absent
        try:
            cleaner_mod = importlib.import_module("glossapi_rs_cleaner")
            self.logger.info("Using compiled glossapi_rs_cleaner extension for fast cleaning")
        except ModuleNotFoundError:
            self.logger.warning("Rust extension glossapi_rs_cleaner missing; attempting in-place build via maturin …")
            import subprocess, sys
            build_success = False
            try:
                root_dir = Path(__file__).resolve().parents[3]  # project root containing `rust/`
                manifest = root_dir / "rust" / "glossapi_rs_cleaner" / "Cargo.toml"
                # Ensure maturin present
                subprocess.run([sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"], check=True)
                subprocess.run([sys.executable, "-m", "maturin", "develop", "--release", "--manifest-path", str(manifest)], check=True)
                cleaner_mod = importlib.import_module("glossapi_rs_cleaner")
                self.logger.info("Successfully built and loaded glossapi_rs_cleaner via maturin")
                build_success = True
            except Exception as build_err:
                self.logger.error(f"Automatic build of glossapi_rs_cleaner failed: {build_err}")
            if not build_success:
                raise RuntimeError(
                    "The Rust extension 'glossapi_rs_cleaner' is required but could not be built automatically. "
                    "Ensure Rust toolchain and maturin are installed, or install the pre-built wheel."
                )

        # Ensure cleaned directory exists and is empty (idempotent runs)
        if self.cleaned_markdown_dir.exists():
            shutil.rmtree(self.cleaned_markdown_dir)
        self.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)

        # Prepare parquet helper
        parquet_schema = ParquetSchema({"url_column": self.url_column})
        parquet_path = parquet_schema.find_metadata_parquet(self.input_dir) or (
            self.input_dir / "download_results" / "download_results.parquet"
        )

        import os
        records: list = []  # will hold metrics for parquet merge

        # ----- Call Rust high-level pipeline once -----
        scripts_to_keep = ["greek", "latin"]  # keep common alphabetic scripts; numbers/punctuation are added internally
        report_parquet_path = self.cleaned_markdown_dir.parent / "cleaning_report.parquet"

        self.logger.info(
            "Invoking glossapi_rs_cleaner.run_complete_pipeline on %d markdown files…",
            len(list(input_dir.glob("*.md"))),
        )
        try:
            cleaner_mod.run_complete_pipeline(
                str(input_dir),
                str(self.cleaned_markdown_dir),
                str(report_parquet_path),
                scripts_to_keep,
                int(num_threads or os.cpu_count() or 4),
            )
        except Exception as e:
            self.logger.error("Rust cleaning pipeline failed: %s", e)
            raise

        # ----- Parse metrics Parquet produced by Rust -----
        if report_parquet_path.exists():
            try:
                df_metrics_parquet = pd.read_parquet(report_parquet_path)
                for _, row in df_metrics_parquet.iterrows():
                    records.append(
                        {
                            "filename": f"{Path(row['file_name']).stem}.pdf",  # match original PDF filename
                            "badness_score": row.get("badness_score_all_chars", 0.0),
                            "percentage_greek": row.get("percentage_greek_cleaned"),
                            "percentage_latin": row.get("percentage_latin_cleaned"),
                        }
                    )
            except Exception as e:
                self.logger.warning("Failed to parse cleaning report %s: %s", report_parquet_path, e)
        else:
            self.logger.warning("Cleaning report Parquet not found: %s", report_parquet_path)


        # ---- Delete cleaning report to avoid retaining it ----
        try:
            if report_parquet_path.exists():
                report_parquet_path.unlink(missing_ok=True)
                self.logger.debug("Deleted temporary cleaning report %s", report_parquet_path)
        except Exception as e:
            self.logger.warning("Could not delete cleaning report %s: %s", report_parquet_path, e)

        self.logger.info(f"Cleaned {len(records)} markdown files → {self.cleaned_markdown_dir}")

        # ------------------------------------------------------------------
        # Update parquet with Mojibake metrics (single authoritative schema)
        # ------------------------------------------------------------------
        if records:
            df_metrics = pd.DataFrame(records)
            df_metrics = df_metrics.rename(columns={
                "badness_score": "mojibake_badness_score",
                "percentage_latin": "mojibake_latin_percentage",
            })

            if parquet_path and parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            else:
                parquet_path = self.output_dir / "download_results" / "download_results.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame({"filename": df_metrics["filename"]})

            # ensure full schema exists
            for col in [
                "mojibake_badness_score",
                "mojibake_latin_percentage",
                "greek_badness_score",
                "greek_latin_percentage",
                "rejection_reason",
            ]:
                if col not in df.columns:
                    df[col] = pd.NA

            # fill mojibake values
            df = df.set_index("filename")
            df.update(df_metrics.set_index("filename"))
            df.reset_index(inplace=True)
            df.to_parquet(parquet_path, index=False)
            self.logger.info(f"Mojibake metrics updated in {parquet_path}")

        # ----- Noise-metrics scoring (Rust) -----
        try:
            import importlib
            self.logger.info("Scoring cleaned markdown files with glossapi_rs_noise …")
            try:
                noise_mod = importlib.import_module("glossapi_rs_noise")
            except ModuleNotFoundError:
                # Attempt in-place build like with cleaner
                self.logger.warning("Rust extension glossapi_rs_noise missing; attempting in-place build via maturin …")
                import subprocess, sys
                try:
                    root_dir = Path(__file__).resolve().parents[3]
                    manifest = root_dir / "rust" / "glossapi_rs_noise" / "Cargo.toml"
                    subprocess.run([sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"], check=True)
                    subprocess.run([sys.executable, "-m", "maturin", "develop", "--release", "--manifest-path", str(manifest)], check=True)
                    noise_mod = importlib.import_module("glossapi_rs_noise")
                    self.logger.info("Successfully built and loaded glossapi_rs_noise via maturin")
                except Exception as build_err:
                    self.logger.error(f"Automatic build of glossapi_rs_noise failed: {build_err}")
                    raise

            # Use detailed API to retrieve polytonic ratio as well
            results = noise_mod.score_markdown_directory_detailed(str(self.cleaned_markdown_dir), os.cpu_count())
            if results:
                # Unpack only the fields we need: path, score, latin_pct, table_ratio, poly_ratio
                rows = []
                for row in results:
                    # row layout: (path, score, latin_pct, table_ratio, poly_ratio, ...)
                    try:
                        path, score, latin_pct, _table_ratio, poly_ratio = row[:5]
                    except Exception:
                        # Fallback: skip malformed rows
                        continue
                    rows.append((path, float(score), float(latin_pct), float(poly_ratio)))

                df_scores = pd.DataFrame(rows, columns=["filepath", "greek_badness_score", "greek_latin_percentage", "polytonic_ratio"])
                # round polytonic ratio to 2 decimals here
                df_scores["polytonic_ratio"] = df_scores["polytonic_ratio"].round(2)
                df_scores["md_filename"] = df_scores["filepath"].apply(lambda p: Path(p).name)
                df_scores["stem"] = df_scores["md_filename"].str.replace(r"\.md$", "", regex=True)
                # Only compute a provisional reason based on noise metrics available here.
                # IMPORTANT: latin percentage must come from rs_cleaner (mojibake_latin_percentage),
                # not rs_noise. Therefore, do NOT use greek_latin_percentage here. The authoritative
                # filter is recomputed later after merging cleaner metrics.
                conditions = [
                    df_scores["greek_badness_score"] > 60,
                ]
                choices = ["greek>60"]
                df_scores["rejection_reason"] = np.select(conditions, choices, default="ok")
                # Load authoritative parquet (must exist from earlier step)
                if not parquet_path or not parquet_path.exists():
                    self.logger.error("Expected parquet %s not found when adding noise metrics", parquet_path)
                else:
                    df = pd.read_parquet(parquet_path)
                    df["stem"] = df["filename"].str.replace(r"\.pdf$", "", regex=True)

                    for _, row in df_scores.iterrows():
                        idx = df["stem"] == row["stem"]
                        # Ensure destination columns exist
                        for col in ["greek_badness_score", "greek_latin_percentage", "polytonic_ratio", "rejection_reason"]:
                            if col not in df.columns:
                                df[col] = pd.NA
                        df.loc[idx, [
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "polytonic_ratio",
                            "rejection_reason",
                        ]] = row[[
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "polytonic_ratio",
                            "rejection_reason",
                        ]].values
                    df.drop(columns=["stem"], inplace=True)
                    df.to_parquet(parquet_path, index=False)
                    self.logger.info(f"Noise metrics filled in {parquet_path}")
        except Exception as e:
            self.logger.warning("Noise-metrics scoring failed: %s", e)


        # Determine good / bad list based on rejection_reason
        if parquet_path and parquet_path.exists():
            df_final = pd.read_parquet(parquet_path)
            # --- tidy schema ---
            df_final.rename(columns={
                "badness_score": "mojibake_badness_score",
                "percentage_latin": "mojibake_latin_percentage",
                "mojibake_latin_percentage": "latin_percentage",  # ADD THIS
                "rejection_reason": "filter"                      # ADD THIS
            }, inplace=True, errors="ignore")

            # drop duplicate pandas merge suffixes and keep clean names
            df_final = df_final.loc[:, ~df_final.columns.str.endswith('_x')]
            df_final.columns = df_final.columns.str.replace('_y$','', regex=True)

            # round Greek scores for readability
            for _col in ("greek_badness_score", "greek_latin_percentage"):
                if _col in df_final.columns:
                    df_final[_col] = df_final[_col].round(3)
            if "polytonic_ratio" in df_final.columns:
                df_final["polytonic_ratio"] = df_final["polytonic_ratio"].round(2)

            # drop any leftover placeholder columns to avoid duplicates
            df_final.drop(columns=["badness_score", "percentage_latin"], errors="ignore", inplace=True)
            # ADD: Drop unwanted columns
            df_final.drop(columns=["greek_latin_percentage", "badness_before", "badness_after"], errors="ignore", inplace=True)

            # ensure no duplicate column names
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]

            # recompute rejection_reason with correct thresholds (ignore latin percentage)
            if {"greek_badness_score", "mojibake_badness_score"}.issubset(df_final.columns):
                _conds = [
                    df_final["greek_badness_score"] > 60,
                    df_final["mojibake_badness_score"] > 0.1,
                ]
                _choices = ["greek>60", "mojibake>0.1"]
                df_final["filter"] = np.select(_conds, _choices, default="ok")

            # OCR moved to Corpus.ocr(); no OCR is performed in clean()

            # persist cleaned parquet
            df_final.to_parquet(parquet_path, index=False)
# ... (rest of the code remains the same)
            if drop_bad:
                good_df = df_final[df_final["filter"] == "ok"]
                self.good_files = [Path(f).stem for f in good_df["filename"].tolist()]
                self.logger.info(f"After filtering, {len(self.good_files)} good files remain")
            else:
                self.good_files = [Path(f).stem for f in df_final["filename"].tolist()]
        else:
            self.good_files = []

        # After cleaning, point markdown_dir to cleaned files for downstream stages
        self.markdown_dir = self.cleaned_markdown_dir

    # ------------------------------------------------------------------
    # Backwards-compatibility shim – filter() now delegates to clean()
    # ------------------------------------------------------------------
    def filter(self, *args, **kwargs):  # type: ignore[override]
        """Deprecated: use :py:meth:`clean` instead.  Retained for one release."""
        self.logger.warning("Corpus.filter() is deprecated – calling clean() instead")
        self.clean(*args, **kwargs)
    
    def ocr(
        self,
        *,
        force: bool = True,
        device: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        max_pages: Optional[int] = None,
        persist_engine: bool = True,
        limit: Optional[int] = None,
        dpi: Optional[int] = None,        # reserved for future use
        precision: Optional[str] = None,  # reserved for future use ("fp16","bf16")
        # Integrated math enrichment controls
        math_enhance: bool = False,
        math_targets: Optional[Dict[str, List[Tuple[int, int]]]] = None,
        math_batch_size: int = 8,
        math_dpi_base: int = 220,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
    ) -> None:
        """Compatibility shim: rerun extract with forced OCR.

        This method now delegates to extract() with force_ocr=True and skips
        existing outputs (reprocess) to produce OCR-enforced markdown.
        """
        # Try to narrow to bad files if parquet is present; otherwise process PDFs
        filenames: Optional[List[str]] = None
        try:
            from glossapi.parquet_schema import ParquetSchema
            parquet_schema = ParquetSchema({"url_column": self.url_column})
            parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)
            if parquet_path and parquet_path.exists():
                import pandas as _pd
                df = _pd.read_parquet(parquet_path)
                if "filename" in df.columns and "filter" in df.columns:
                    filenames = df.loc[df["filter"] != "ok", "filename"].dropna().astype(str).tolist() or None
        except Exception:
            pass

        self.extract(
            input_format="pdf",
            num_threads=os.cpu_count() or 4,
            accel_type="CUDA",
            force_ocr=True,
            formula_enrichment=False,
            code_enrichment=False,
            filenames=filenames,
            skip_existing=False,
            # When math enrichment is requested, ensure Phase‑1 JSON and index are emitted
            export_doc_json=bool(math_enhance),
            emit_formula_index=bool(math_enhance),
        )
        # Optional Phase‑2 enrichment from JSON immediately after OCR
        if math_enhance:
            try:
                stems: List[str] = []
                if filenames:
                    stems = [Path(f).stem for f in filenames]
                else:
                    json_dir = self.output_dir / "json"
                    if json_dir.exists():
                        stems = [p.name.replace(".docling.json.zst", "").replace(".docling.json", "") for p in json_dir.glob("*.docling.json*")]
                if not stems:
                    self.logger.info("No Docling JSON found for math enrichment.")
                    return
                if str(use_gpus).lower() == "multi":
                    # Detect GPU devices
                    devs = devices
                    if not devs:
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
                            try:
                                import torch  # type: ignore
                                if torch.cuda.is_available():
                                    devs = list(range(torch.cuda.device_count()))
                            except Exception:
                                pass
                    if not devs:
                        self.logger.warning("Multi-GPU math requested but no GPUs detected; falling back to single GPU")
                    else:
                        from multiprocessing import get_context
                        ctx = get_context("spawn")
                        work_q = ctx.Queue()
                        for s in stems:
                            work_q.put(s)
                        procs = []
                        for dev_id in devs:
                            p = ctx.Process(target=_gpu_math_worker, args=(
                                dev_id,
                                str(self.input_dir),
                                str(self.output_dir),
                                work_q,
                                int(math_batch_size),
                                int(math_dpi_base),
                                device or "cuda",
                                math_targets or {},
                            ))
                            p.start()
                            procs.append(p)
                        for p in procs:
                            p.join()
                        return
                # Single-GPU
                self.formula_enrich_from_json(
                    files=stems,
                    device=(device or "cuda"),
                    batch_size=int(math_batch_size),
                    dpi_base=int(math_dpi_base),
                    targets_by_stem=math_targets,
                )
            except Exception as _e:
                self.logger.warning("Phase‑2 enrichment after OCR failed: %s", _e)

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
        export_doc_json: bool = False,
        emit_formula_index: bool = False,
    ) -> None:
        """Prepare and initialize the underlying extractor once per configuration.

        Builds the Docling converter only if the effective configuration changed.
        """
        # Lazy import + instantiate GlossExtract
        if self.extractor is None:
            from .gloss_extract import GlossExtract
            self.extractor = GlossExtract(url_column=self.url_column)

        # Propagate toggles used by Phase‑1 helpers
        try:
            setattr(self.extractor, "export_doc_json", bool(export_doc_json))
            setattr(self.extractor, "emit_formula_index", bool(emit_formula_index))
        except Exception:
            pass
        # Threads
        try:
            threads_effective = int(num_threads) if num_threads is not None else (os.cpu_count() or 4)
        except Exception:
            threads_effective = (os.cpu_count() or 4)
        self.extractor.enable_accel(threads=threads_effective, type=accel_type)

        # Images scale env default
        try:
            import os as _os
            images_scale_env = _os.getenv("GLOSSAPI_IMAGES_SCALE", "1.25")
        except Exception:
            images_scale_env = "1.25"

        # Ensure converter exists (reuse when unchanged)
        self.extractor.ensure_extractor(
            enable_ocr=bool(force_ocr),
            force_full_page_ocr=bool(force_ocr),
            formula_enrichment=bool(formula_enrichment),
            code_enrichment=bool(code_enrichment),
            images_scale=float(images_scale_env),
            use_cls=bool(use_cls),
            profile_timings=not bool(benchmark_mode),
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
        skip_existing: bool = True,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
        use_cls: bool = False,
        benchmark_mode: bool = False,
        export_doc_json: bool = False,
        emit_formula_index: bool = False,
        _prepared: bool = False,
    ) -> None:
        """
        Extract input files to markdown format.
        
        Args:
            input_format: Input format ("pdf", "docx", "xml_jats", "html", "pptx", "csv", "md", "all") (default: "all")
                          Note: Old .doc format (pre-2007) is not supported
            num_threads: Number of threads for processing (default: 4)
            accel_type: Acceleration type ("Auto", "CPU", "CUDA", "MPS") (default: "Auto")

        """
        self.logger.info(f"Extracting {input_format} files to markdown...")
        
        # We will prepare the extractor later (single-GPU branch). For multi-GPU,
        # we avoid importing heavy OCR deps in the parent.
        import os as _os
        images_scale_env = _os.getenv("GLOSSAPI_IMAGES_SCALE", "1.25")
        formula_batch_env = _os.getenv("GLOSSAPI_FORMULA_BATCH", "16")
        # Create output directory for downstream stages
        os.makedirs(self.markdown_dir, exist_ok=True)
        
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
        
        # Get input files from downloads directory
        if input_format.lower() == "all":
            # Include all supported formats
            input_files = []
            for ext in supported_formats:
                found_files = list(downloads_dir.glob(f"*.{ext}"))
                input_files.extend(found_files)
                if found_files:
                    self.logger.info(f"Found {len(found_files)} .{ext} files in downloads directory")
            
            # Log a warning about doc files
            doc_files = list(downloads_dir.glob("*.doc"))
            if doc_files:
                self.logger.warning(f"Found {len(doc_files)} .doc files which are not supported by Docling (pre-2007 Word format)")
        elif "," in input_format.lower():
            # Handle comma-separated format list
            input_files = []
            formats = [fmt.strip().lower() for fmt in input_format.split(",")]
            for ext in formats:
                # Handle special case for XML formats
                if ext == "xml_jats":
                    ext = "xml"  # Use the file extension .xml
                    
                if ext == "doc":
                    self.logger.warning(f"The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first.")
                    continue
                    
                current_files = list(downloads_dir.glob(f"*.{ext}"))
                self.logger.info(f"Found {len(current_files)} files with extension .{ext}")
                input_files.extend(current_files)
        else:
            # Handle special case for XML formats
            if input_format.lower() == "xml":
                ext = "xml"  # Still use the file extension .xml
            else:
                ext = input_format.lower()
                
            if ext == "doc":
                self.logger.error(f"The .doc format (pre-2007 Word) is not supported by Docling. Please convert to .docx first.")
                return
                
            input_files = list(downloads_dir.glob(f"*.{ext}"))
        
        # Optional whitelist filtering by filenames list
        if filenames:
            names = set(str(n) for n in filenames)
            input_files = [p for p in input_files if p.name in names]

        if not input_files:
            self.logger.warning(f"No {input_format} files found in {downloads_dir}")
            return
        
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
                    try:
                        import torch  # type: ignore
                        if torch.cuda.is_available():
                            devs = list(range(torch.cuda.device_count()))
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
                    threads_effective = int(num_threads) if num_threads is not None else max(1, (os.cpu_count() or 4) // max(1, len(devs)))
                except Exception:
                    threads_effective = int(num_threads) if isinstance(num_threads, int) else (os.cpu_count() or 4)
                # Dynamic work queue across GPUs
                from multiprocessing import get_context
                ctx = get_context("spawn")
                task_q = ctx.Queue()
                name_list = [p.name for p in input_files]
                for nm in name_list:
                    task_q.put(nm)
                procs = []
                for dev_id in devs:
                    p = ctx.Process(
                        target=gpu_extract_worker_queue,
                        args=(
                            dev_id,
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
                        ),
                    )
                    p.start()
                    procs.append(p)
                any_fail = False
                for p in procs:
                    p.join()
                    if p.exitcode not in (0, None):
                        any_fail = True
                if any_fail:
                    self.logger.warning("One or more GPU workers exited with non-zero status.")
                self.logger.info("Multi-GPU extraction complete. Markdown files saved to %s", self.markdown_dir)
                return

        # Single GPU path
        # Prepare extractor (lazy import + instantiate)
        if self.extractor is None:
            try:
                from .gloss_extract import GlossExtract  # local import to avoid import-time heavy deps
                self.extractor = GlossExtract(url_column=self.url_column)
            except Exception as e:
                self.logger.error(f"Failed to initialize GlossExtract: {e}")
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
                try:
                    import torch  # type: ignore
                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")
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
        

    

    
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.
        
        Uses files marked with 'good' extraction quality (if available) or all markdown files.
        """
        self.logger.info("Extracting sections from markdown files...")
        
        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Determine which markdown files to section
        # Priority 1: self.good_files collected from clean()
        # Priority 2: legacy parquet 'extraction' column logic (for backward compatibility)
        # ------------------------------------------------------------------
        
        good_filenames: List[str] = []

        if getattr(self, "good_files", None):
            good_filenames = self.good_files
            self.logger.info(f"Using {len(good_filenames)} good filenames from clean()")
        else:
            # Fallback path: derive good filenames from parquet metadata
            self.logger.info("No good_files from clean(); using parquet filter/ocr_success if available")
            from glossapi.parquet_schema import ParquetSchema
            parquet_schema = ParquetSchema({'url_column': self.downloader_config.get('url_column', 'url')})
            # Prefer the output_dir parquet which consolidates current run metadata
            parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)
            if parquet_path is None:
                # Try legacy input_dir locations
                parquet_path = parquet_schema.find_metadata_parquet(self.input_dir)
                if parquet_path is None:
                    dl_dir = self.input_dir / 'download_results'
                    if dl_dir.exists():
                        parquet_path = parquet_schema.find_metadata_parquet(dl_dir)

            if parquet_path is not None and Path(parquet_path).exists():
                try:
                    df_meta = pd.read_parquet(parquet_path)
                    mask = pd.Series(False, index=df_meta.index)
                    if 'filter' in df_meta.columns:
                        mask = mask | (df_meta['filter'] == 'ok')
                    if 'ocr_success' in df_meta.columns:
                        mask = mask | (df_meta['ocr_success'].fillna(False))
                    good_rows = df_meta[mask]
                    # Legacy fallback: if nothing selected yet, try 'extraction' == 'good'
                    if good_rows.empty and 'extraction' in df_meta.columns:
                        legacy_rows = df_meta[df_meta['extraction'] == 'good']
                        if not legacy_rows.empty:
                            good_rows = legacy_rows
                    if not good_rows.empty and 'filename' in good_rows.columns:
                        good_filenames = [os.path.splitext(fn)[0] for fn in good_rows['filename'].astype(str).tolist() if fn]
                        self.logger.info(f"Selected {len(good_filenames)} files via metadata from {parquet_path}")
                        # Update processing_stage for selected rows
                        try:
                            if 'processing_stage' not in df_meta.columns:
                                df_meta['processing_stage'] = pd.NA
                            sel_idx = good_rows.index
                            df_meta.loc[sel_idx, 'processing_stage'] = df_meta.loc[sel_idx, 'processing_stage'].apply(
                                lambda x: (str(x) + ',section') if (pd.notna(x) and 'section' not in str(x)) else ('download,extract,section' if pd.isna(x) else x)
                            )
                            # Write back in-place
                            df_meta.to_parquet(parquet_path, index=False)
                        except Exception as e:
                            self.logger.warning(f"Failed to update processing_stage in {parquet_path}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error reading parquet file {parquet_path}: {e}")
            else:
                self.logger.info("No metadata parquet found for section selection; will fall back to all markdown files")
        
        self.logger.info(f"Found {len(good_filenames)} good quality files for sectioning")
        if good_filenames:
            self.logger.info(f"Good filenames: {good_filenames}")
            
        if not good_filenames:
            self.logger.warning("No files marked as 'good' – falling back to processing all extracted markdown files.")
            good_filenames = [
                os.path.splitext(p.name)[0]
                for p in Path(self.markdown_dir).glob("*.md")
            ]
            if not good_filenames:
                error_msg = "No markdown files found to section. Extraction might have failed."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Extract sections - pass list of good filenames to the sectioner
        # We will pass the original markdown directory and the list of good filenames 
        # rather than creating a separate directory
        self.sectioner.to_parquet(
            input_dir=str(self.markdown_dir),  # Use the markdown directory directly
            output_dir=str(self.sections_dir),
            filenames_to_process=good_filenames  # Pass the list of good filenames
        )
        
        self.logger.info(f"Finished sectioning {len(good_filenames)} good quality files")
        self.logger.info(f"Section extraction complete. Parquet file saved to {self.sections_parquet}")
    

    def annotate(self, annotation_type: str = "text", fully_annotate: bool = True) -> None:
        """
        Annotate extracted sections with classification information.
        
        Args:
            annotation_type: Type of annotation to use: 'text' or 'chapter'
                           - 'text': Use text-based annotation with section titles (default)
                           - 'chapter': Use chapter-based annotation with chapter numbers
            fully_annotate: Whether to perform full annotation of sections (default: True)
        """
        self.logger.info("Running section classification...")
        
        # Check if input parquet file exists
        if not self.sections_parquet.exists():
            self.logger.error(f"Sections file not found: {self.sections_parquet}. Please run section() first.")
            return
        
        # Check if section classifier model exists
        model_exists = self.section_classifier_model_path.exists()
        if not model_exists:
            self.logger.warning(f"Model file not found at {self.section_classifier_model_path}. To train a new model, run GlossSectionClassifier.train_from_csv()")
        
        # If no trained model, skip annotation with a clear message
        if not model_exists:
            self.logger.warning(
                "No section-classifier model found at %s. "
                "If you are running from a git checkout (not the pip package), make sure the "
                "'models/section_classifier.joblib' file is present or pass "
                "section_classifier_model_path explicitly. Skipping annotation.",
                self.section_classifier_model_path
            )
            return

        model_path = str(self.section_classifier_model_path)
        # Classify sections and save output to 'classified_sections.parquet'
        self.classifier.classify_sections(
            input_parquet=str(self.sections_parquet),
            output_parquet=str(self.classified_parquet),
            model_path=model_path,
            n_cpus=4,
            column_name='title'
        )

        
        # Perform full annotation if requested
        if fully_annotate:
            self.logger.info("Performing full annotation...")
            
            # If we're using auto annotation and have document types and annotation mappings available
            if annotation_type == "auto" and self.filename_to_doctype and self.annotation_mapping:
                # Create a mapping from filename to annotation type based on document types
                filename_to_annotation = {}
                for filename, doc_type in self.filename_to_doctype.items():
                    # Look up the annotation method for this document type in our mapping
                    # Default to 'text' if no mapping exists
                    filename_to_annotation[filename] = self.annotation_mapping.get(doc_type, 'text')
                
                self.logger.info(f"Using document-type specific annotation based on metadata")
                
                # Read the classified parquet file
                df = pd.read_parquet(str(self.classified_parquet))
                
                # Group by filename and process each document according to its annotation type
                updated_groups = []
                
                for filename, group in df.groupby('filename'):
                    # Determine annotation type for this file
                    doc_annotation = filename_to_annotation.get(filename, 'text')
                    
                    # Process according to annotation type
                    if doc_annotation == 'chapter':
                        self.logger.debug(f"Processing {filename} as chapter")
                        updated_group = self.classifier.fully_annotate_chapter_group(group)
                    else:
                        self.logger.debug(f"Processing {filename} as text")
                        updated_group = self.classifier.fully_annotate_text_group(group)
                    
                    if updated_group is not None:
                        updated_groups.append(updated_group)
                
                # Concatenate and save results
                if updated_groups:
                    df_updated = pd.concat(updated_groups)
                    df_updated.to_parquet(str(self.fully_annotated_parquet), index=False)
                else:
                    self.logger.warning("No valid document groups to process. Output file not created.")
            else:
                # Use the standard fully_annotate method with the specified annotation type
                self.classifier.fully_annotate(
                    input_parquet=str(self.classified_parquet),
                    output_parquet=str(self.fully_annotated_parquet),
                    document_types=self.filename_to_doctype if self.filename_to_doctype else None,
                    annotation_type=annotation_type
                )
            
            # Use the fully annotated output for adding document types
            self._add_document_types(self.fully_annotated_parquet)
            
            # Update processing_stage in the fully annotated parquet
            try:
                # Read the fully annotated parquet
                df = pd.read_parquet(self.fully_annotated_parquet)
                
                # Add annotate to processing stage
                if 'processing_stage' in df.columns:
                    df['processing_stage'] = df['processing_stage'].apply(lambda x: x + ',annotate' if 'annotate' not in str(x) else x)
                else:
                    df['processing_stage'] = 'section,annotate'
                    
                # Write back
                df.to_parquet(self.fully_annotated_parquet, index=False)
                self.logger.info("Updated processing_stage to include 'annotate' stage")
            except Exception as e:
                self.logger.warning(f"Failed to update processing_stage in fully annotated parquet: {e}")
        else:
            # Add document types to the classified output
            self._add_document_types(self.classified_parquet)
            
            # Update processing_stage in the classified parquet when not doing full annotation
            try:
                # Read the classified parquet
                df = pd.read_parquet(self.classified_parquet)
                
                # Add annotate to processing stage
                if 'processing_stage' in df.columns:
                    df['processing_stage'] = df['processing_stage'].apply(lambda x: x + ',annotate' if 'annotate' not in str(x) else x)
                else:
                    df['processing_stage'] = 'section,annotate'
                    
                # Write back
                df.to_parquet(self.classified_parquet, index=False)
                self.logger.info("Updated processing_stage to include 'annotate' stage")
            except Exception as e:
                self.logger.warning(f"Failed to update processing_stage in classified parquet: {e}")
    
    def _add_document_types(self, parquet_file: Path) -> None:
        """
        Add document_type information to the classified sections.
        
        Args:
            parquet_file: Path to the Parquet file to update
        """
        if not self.filename_to_doctype:
            self.logger.warning("No document type information available. Skipping document type addition.")
            return
        
        if parquet_file.exists():
            try:
                # Read the parquet file
                df = pd.read_parquet(parquet_file)
                
                # Add document_type based on filename
                df['document_type'] = df['filename'].map(self.filename_to_doctype)
                
                # Check for missing document types
                missing_count = df['document_type'].isna().sum()
                if missing_count > 0:
                    self.logger.warning(f"{missing_count} sections ({missing_count/len(df):.2%}) have no document type!")
                    missing_filenames = df[df['document_type'].isna()]['filename'].unique()[:5]
                    self.logger.warning(f"Sample filenames with missing document types: {missing_filenames}")
                    
                    # Check if the issue might be due to .md extension
                    if any('.md' in str(f) for f in self.filename_to_doctype.keys()):
                        self.logger.warning("Possible cause: Metadata filenames contain .md extension but sections filenames don't")
                    elif any('.md' in str(f) for f in df['filename'].unique()[:100]):
                        self.logger.warning("Possible cause: Sections filenames contain .md extension but metadata filenames don't")
                
                # Save the updated file
                df.to_parquet(parquet_file, index=False)
                self.logger.info(f"Added document types to {parquet_file}")
            except Exception as e:
                self.logger.error(f"Error adding document types: {e}")
        else:
            self.logger.warning(f"File not found: {parquet_file}")
    
    def download(
        self,
        input_parquet: Optional[Union[str, Path]] = None,
        url_column: str = 'url',
        verbose: Optional[bool] = None,
        *,
        parallelize_by: Optional[str] = None,
        links_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Download files from URLs in a parquet file.
        
        If input_parquet is not specified, it will automatically look for any .parquet file
        in the input_dir and use the first one found.
        
        Args:
            input_parquet: Path to input parquet file with URLs (optional)
                           If not provided, will search input_dir for parquet files
            url_column: Name of column containing URLs (defaults to 'url')
            verbose: Whether to enable verbose logging (overrides instance setting if provided)
            **kwargs: Additional parameters to override default downloader config
        
        Returns:
            pd.DataFrame: DataFrame with download results
        """
        # If input_parquet not specified, find parquet files in input_dir
        if input_parquet is None:
            parquet_files = list(self.input_dir.glob('*.parquet'))
            if not parquet_files:
                raise ValueError(f"No parquet files found in {self.input_dir}")
            input_parquet = parquet_files[0]
            self.logger.info(f"Using parquet file: {input_parquet}")
        else:
            input_parquet = Path(input_parquet)
            
        # Load the input file with URLs to download
        original_input_filename = Path(input_parquet).name
        input_df = pd.read_parquet(input_parquet)
        total_urls = len(input_df)
        self.logger.info(f"Total URLs in input file: {total_urls}")
        
        # Respect links_column override early so resume filter uses correct column name
        if links_column:
            url_column = links_column

        # Look for existing download results file by the specific input filename first
        input_filename = Path(input_parquet).name
        download_results_dir = Path(self.output_dir) / "download_results"
        specific_results_path = download_results_dir / f"download_results_{input_filename}"
        partial_results_path = download_results_dir / f"download_results_{input_filename}.partial.parquet"
        
        existing_results = None
        existing_results_path = None
        found_existing = False
        
        # Check for specific download results file
        if os.path.exists(specific_results_path):
            self.logger.info(f"Found existing download results: {specific_results_path}")
            try:
                existing_results = pd.read_parquet(specific_results_path)
                existing_results_path = specific_results_path
                found_existing = True
            except Exception as e:
                self.logger.warning(f"Failed to read specific download results: {e}")
        elif os.path.exists(partial_results_path):
            self.logger.info(f"Found partial download checkpoint: {partial_results_path}")
            try:
                existing_results = pd.read_parquet(partial_results_path)
                existing_results_path = partial_results_path
                found_existing = True
            except Exception as e:
                self.logger.warning(f"Failed to read partial results: {e}")
                
        # If specific results not found, look in the directory for any download results
        if not found_existing and os.path.exists(download_results_dir):
            result_files = list(download_results_dir.glob('*.parquet'))
            for file in result_files:
                try:
                    test_df = pd.read_parquet(file)
                    if url_column in test_df.columns and 'download_success' in test_df.columns:
                        self.logger.info(f"Found alternative download results: {file}")
                        existing_results = test_df
                        existing_results_path = file
                        found_existing = True
                        break
                except Exception:
                    continue
                    
        # Filter out already downloaded URLs and prepare to download only remaining ones
        if found_existing and url_column in existing_results.columns:
            # Find filenames that have already been assigned (whether download succeeded or not)
            # to ensure we don't reuse the same filenames and overwrite files
            existing_filenames = []
            if 'filename' in existing_results.columns:
                existing_filenames = existing_results['filename'].dropna().tolist()
                self.logger.info(f"Found {len(existing_filenames)} existing filenames to avoid")

            # Build the set of successful URLs from checkpoint/results
            successful_urls = []
            if 'download_success' in existing_results.columns:
                successful_urls = existing_results[
                    existing_results['download_success'] == True
                ][url_column].dropna().astype(str).tolist()

            if successful_urls:
                self.logger.info(f"Found {len(successful_urls)} previously successful downloads")

                # If input uses list/JSON URLs, expand to one-URL-per-row before filtering
                def _looks_like_list(s: str) -> bool:
                    try:
                        t = str(s).strip()
                        return t.startswith('[') or t.startswith('{')
                    except Exception:
                        return False

                need_expand = False
                try:
                    sample = input_df[url_column].dropna().astype(str).head(50).tolist()
                    need_expand = any(_looks_like_list(x) for x in sample)
                except Exception:
                    need_expand = False

                if need_expand:
                    try:
                        # Reuse downloader's expansion to mirror runtime behavior
                        dl_tmp = GlossDownloader(url_column=url_column, output_dir=str(self.output_dir))
                        expanded_df = dl_tmp._expand_and_mark_duplicates(input_df.copy())  # type: ignore[attr-defined]
                        # Keep URL and provenance columns if present
                        keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug") if c in expanded_df.columns]
                        remaining_df = expanded_df[~expanded_df[url_column].isin(successful_urls)][keep_cols]
                        self.logger.info(
                            f"Expanded list/JSON URLs to {len(expanded_df)} rows; pending {len(remaining_df)}"
                        )
                    except Exception:
                        # Fallback: basic JSON parse expansion
                        import json as _json
                        rows = []
                        for _, row in input_df.iterrows():
                            val = row.get(url_column)
                            if isinstance(val, str) and _looks_like_list(val):
                                try:
                                    arr = _json.loads(val)
                                    if isinstance(arr, list):
                                        rows.extend([str(u) for u in arr if isinstance(u, (str,))])
                                    elif isinstance(arr, dict):
                                        u = arr.get('url') or arr.get('href') or arr.get('link')
                                        if u:
                                            rows.append(str(u))
                                except Exception:
                                    pass
                            elif isinstance(val, str) and val.strip():
                                rows.append(val.strip())
                        import pandas as _pd
                        expanded_df = _pd.DataFrame({url_column: rows})
                        keep_cols = [url_column]
                        remaining_df = expanded_df[~expanded_df[url_column].isin(successful_urls)][keep_cols]
                        self.logger.info(
                            f"Expanded (fallback) to {len(expanded_df)} rows; pending {len(remaining_df)}"
                        )
                else:
                    # Simple string URLs: filter directly and keep provenance if present
                    keep_cols = [url_column] + [c for c in ("source_row", "url_index", "collection_slug") if c in input_df.columns]
                    remaining_df = input_df[~input_df[url_column].isin(successful_urls)][keep_cols]

                # If all URLs already downloaded, return existing results
                if len(remaining_df) == 0:
                    self.logger.info("All files already successfully downloaded")
                    return existing_results

                self.logger.info(
                    f"Processing {len(remaining_df)} remaining URLs after skipping successes"
                )

                # Save filtered per-URL input to a temporary file for the downloader
                temp_input = self.output_dir / "temp_download_input.parquet"
                remaining_df.to_parquet(temp_input, index=False)
                input_parquet = temp_input
        else:
            self.logger.info("No existing download results found or usable")
            existing_results = pd.DataFrame()
            
        # Initialize downloader configuration (kwargs take precedence)
        dl_cfg = dict(self.downloader_config)
        dl_cfg.update(kwargs)
        # Allow caller to override which column holds links
        if links_column:
            url_column = links_column
        # Allow caller to choose grouping for scheduler (e.g., 'collection_slug' or 'base_domain')
        if parallelize_by:
            dl_cfg['scheduler_group_by'] = parallelize_by
        # Build used filename bases set to avoid collisions on resume
        used_bases = set()
        try:
            used_bases |= {str(Path(fn).stem) for fn in existing_filenames if isinstance(fn, str)}
        except Exception:
            pass
        try:
            # Also include on-disk stems
            downloads_dir = Path(self.output_dir) / 'downloads'
            if downloads_dir.exists():
                used_bases |= {p.stem for p in downloads_dir.glob('*') if p.is_file()}
        except Exception:
            pass

        downloader = GlossDownloader(
            url_column=url_column,
            output_dir=str(self.output_dir),
            log_level=self.logger.level,
            verbose=verbose if verbose is not None else self.verbose,
            **{k: v for k, v in dl_cfg.items() if k not in {'input_parquet'}},
            _used_filename_bases=used_bases
        )
        
        # Download files
        self.logger.info(f"Downloading files from URLs in {input_parquet}...")
        new_results = downloader.download_files(input_parquet=str(input_parquet))
        
        # Merge with existing results
        if not existing_results.empty:
            # Filter out rows from existing_results that are in new_results (based on URL)
            if url_column in new_results.columns and url_column in existing_results.columns:
                processed_urls = new_results[url_column].tolist()
                existing_filtered = existing_results[~existing_results[url_column].isin(processed_urls)]
                
                # Combine existing and new results
                final_results = pd.concat([existing_filtered, new_results], ignore_index=True)
                self.logger.info(f"Merged {len(existing_filtered)} existing results with {len(new_results)} new results")
        else:
            final_results = new_results
            
        # Ensure we have a download_results directory
        os.makedirs(download_results_dir, exist_ok=True)
        
        # Save results using the input filename pattern
        output_parquet = download_results_dir / f"download_results_{original_input_filename}"
        final_results.to_parquet(output_parquet, index=False)
        self.logger.info(f"Saved download results to {output_parquet}")
        
        # Clean up temporary files if created
        temp_path = self.output_dir / "temp_download_input.parquet"
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Report download completion
        success_count = len(final_results[final_results['download_success'] == True]) if 'download_success' in final_results.columns else 0
        self.logger.info(f"Download complete. {success_count} files downloaded to {self.output_dir / 'downloads'}")
        
        return final_results
        
    def process_all(self, input_format: str = "pdf", fully_annotate: bool = True, annotation_type: str = "auto", download_first: bool = False) -> None:
        """
        Run the complete processing pipeline: extract, section, and annotate.
        
        Args:
            input_format: Input format (default: "pdf")
            fully_annotate: Whether to perform full annotation after classification (default: True)
            annotation_type: Annotation method to use (default: "auto")
            download_first: Whether to run the downloader before extraction (default: False)
        """
        if download_first:
            try:
                self.download()
                self.logger.info("Download step completed, proceeding with extraction...")
            except Exception as e:
                self.logger.error(f"Error during download step: {e}")
                self.logger.warning("Continuing with extraction of already downloaded files...")
                
        self.extract(input_format=input_format)
        self.section()
        self.annotate(fully_annotate=fully_annotate, annotation_type=annotation_type)
        
        self.logger.info("Complete processing pipeline finished successfully.")

    def triage_math(self) -> None:
        """Summarize per-page formula density and update routing recommendation in parquet.

        Scans `markdown_dir` for `{stem}.per_page.metrics.json`, computes summary metrics, and
        writes `formula_total`, `formula_avg_pp`, `formula_p90_pp`, `pages_with_formula`,
        `pages_total`, and `phase_recommended` into the consolidated download_results parquet
        if present.
        """
        try:
            from .triage import summarize_math_density_from_metrics, recommend_phase, update_download_results_parquet
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
        from .math_enrich import enrich_from_docling_json  # type: ignore
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
            pq = ps.find_metadata_parquet(self.output_dir)
        except Exception:
            pq = None
        if pq and Path(pq).exists():
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
                        from .json_io import load_docling_json  # type: ignore
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
                    from .triage import update_math_enrich_results  # type: ignore
                    pq_path = self.output_dir / 'download_results' / 'download_results.parquet'
                    update_math_enrich_results(pq_path, stem, items=int(stats.get('items', 0)), accepted=int(stats.get('accepted', 0)), time_sec=float(stats.get('time_sec', 0.0)))
                except Exception as _e:
                    self.logger.warning("Parquet math-enrich update failed for %s: %s", stem, _e)
            except Exception as e:
                self.logger.warning("Phase‑2 failed for %s: %s", stem, e)

def _gpu_math_worker(device_id: int, in_dir: str, out_dir: str, work_q, batch_size: int, dpi_base: int, device: str, targets_map: Dict[str, List[Tuple[int, int]]]) -> None:
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            import sys as _sys, pathlib as _pl
            _sys.path.insert(0, str((_pl.Path(out_dir).resolve().parents[1] / 'src').resolve()))
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as _e:
            print(f"[MATH GPU{device_id}] Cannot import glossapi in worker: {_e}")
            return
    c = _Corpus(input_dir=in_dir, output_dir=out_dir)
    batch: list[str] = []
    B = max(1, int(batch_size))
    while True:
        try:
            nm = work_q.get_nowait()
        except Exception:
            if batch:
                _targets = {s: targets_map.get(s) for s in batch if s in targets_map} if targets_map else None
                try:
                    c.formula_enrich_from_json(files=list(batch), device=(device or "cuda"), batch_size=B, dpi_base=int(dpi_base), targets_by_stem=_targets)
                except Exception as _e:
                    print(f"[MATH GPU{device_id}] Batch failed ({len(batch)}): {_e}")
            break
        if isinstance(nm, str) and nm.strip():
            batch.append(nm)
        if len(batch) >= B:
            _targets = {s: targets_map.get(s) for s in batch if s in targets_map} if targets_map else None
            try:
                c.formula_enrich_from_json(files=list(batch), device=(device or "cuda"), batch_size=B, dpi_base=int(dpi_base), targets_by_stem=_targets)
            except Exception as _e:
                print(f"[MATH GPU{device_id}] Batch failed ({len(batch)}): {_e}")
            batch.clear()

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
) -> None:
    import os as _os
    import time as _time
    from pathlib import Path as _Path
    # Bind this worker to a single GPU id
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    _os.environ["GLOSSAPI_DOCLING_DEVICE"] = "cuda:0"
    # Light import of Corpus (prefer installed package; fallback to repo src)
    try:
        from glossapi import Corpus as _Corpus  # type: ignore
    except Exception:
        try:
            import sys as _sys, pathlib as _pl
            _sys.path.insert(0, str((_pl.Path(out_dir).resolve().parents[1] / 'src').resolve()))
            from glossapi import Corpus as _Corpus  # type: ignore
        except Exception as _e:
            print(f"[GPU{device_id}] Cannot import glossapi in worker: {_e}")
            return
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
        )
    except Exception as _e:
        print(f"[GPU{device_id}] Prime failed: {_e}")
    # Prepare persistent extractor in this worker on first call
    # Process queue items in small batches to reduce function-call overhead
    batch: list[str] = []
    BATCH_SIZE = 8
    import queue as _queue
    last_progress = _time.time()
    processed = 0
    while True:
        try:
            nm = work_q.get_nowait()
        except Exception as e:
            # queue.Empty or other -> flush any pending batch then exit
            if batch:
                try:
                    c.extract(
                        input_format=input_fmt,
                        num_threads=threads,
                        accel_type="cuda:0",
                        force_ocr=force,
                        formula_enrichment=fe,
                        code_enrichment=ce,
                        filenames=list(batch),
                        skip_existing=skip,
                        use_gpus="single",
                        use_cls=use_cls_w,
                        benchmark_mode=benchmark,
                        export_doc_json=bool(export_json),
                        emit_formula_index=bool(emit_index),
                        _prepared=True,
                    )
                    processed += len(batch)
                except Exception as _e:
                    print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
                batch.clear()
            break
        if isinstance(nm, str) and nm.strip():
            batch.append(nm)
        if len(batch) >= BATCH_SIZE:
            try:
                c.extract(
                    input_format=input_fmt,
                    num_threads=threads,
                    accel_type="cuda:0",
                    force_ocr=force,
                    formula_enrichment=fe,
                    code_enrichment=ce,
                    filenames=list(batch),
                    skip_existing=skip,
                    use_gpus="single",
                    use_cls=use_cls_w,
                    benchmark_mode=benchmark,
                    export_doc_json=bool(export_json),
                    emit_formula_index=bool(emit_index),
                    _prepared=True,
                )
                processed += len(batch)
            except Exception as _e:
                print(f"[GPU{device_id}] Batch failed ({len(batch)}): {_e}")
            batch.clear()
        # Occasional heartbeat
        if _time.time() - last_progress > 30:
            try:
                print(f"[GPU{device_id}] processed ~{processed} files…")
            except Exception:
                pass
            last_progress = _time.time()
