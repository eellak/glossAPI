import logging
from pathlib import Path
import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Optional, Union, List, Any
import shutil

from .gloss_extract import GlossExtract
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
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
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
        self.extractor = GlossExtract(url_column=self.url_column)
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

            results = noise_mod.score_markdown_directory(str(self.cleaned_markdown_dir), os.cpu_count())
            if results:
                df_scores = pd.DataFrame(results, columns=["filepath", "greek_badness_score", "greek_latin_percentage"])
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
                        df.loc[idx, [
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "rejection_reason",
                        ]] = row[[
                            "greek_badness_score",
                            "greek_latin_percentage",
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

            # drop any leftover placeholder columns to avoid duplicates
            df_final.drop(columns=["badness_score", "percentage_latin"], errors="ignore", inplace=True)
            # ADD: Drop unwanted columns
            df_final.drop(columns=["greek_latin_percentage", "badness_before", "badness_after"], errors="ignore", inplace=True)

            # ensure no duplicate column names
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]

            # recompute rejection_reason with correct thresholds
            if {"greek_badness_score", "mojibake_badness_score", "latin_percentage"}.issubset(df_final.columns):
                _conds = [
                    df_final["greek_badness_score"] > 60,
                    df_final["mojibake_badness_score"] > 0.1,
                    df_final["latin_percentage"] > 0.6,
                ]
                _choices = ["greek>60", "mojibake>0.1", "latin>0.6"]
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
        force: bool = False,
        device: Optional[str] = None,
        model_dir: Optional[Union[str, Path]] = None,
        max_pages: Optional[int] = None,
        persist_engine: bool = True,
        limit: Optional[int] = None,
        dpi: Optional[int] = None,        # reserved for future use
        precision: Optional[str] = None,  # reserved for future use ("fp16","bf16")
    ) -> None:
        """Run OCR on PDFs and update outputs/Parquet.

        This is the sole OCR entrypoint. It does not recompute Rust metrics.

        Args:
            force: If True, run OCR on all PDFs (except already OCR'd) without requiring Rust filter columns.
            device: Device string (e.g., "cuda", "cuda:0", "cpu"). If None, autodetect handled in OCR helper.
            model_dir: Hugging Face model id (e.g., "nanonets/Nanonets-OCR-s") or a local model directory path.
                       If None, uses self.ocr_model_dir or the OCR helper's default model id.
            max_pages: Optional cap on number of pages to OCR per document.
            persist_engine: Keep OCR model resident on GPU for the entire run.
            limit: Optional max number of files to process in this call.
            dpi: Reserved for future use.
            precision: Reserved for future use.
        """
        from glossapi.parquet_schema import ParquetSchema
        import pandas as pd
        import traceback

        # Lazy import heavy OCR helper
        try:
            from glossapi.ocr import run_nanonets_ocr as _run_nanonets_ocr
        except ImportError:
            self.logger.error("glossapi.ocr module missing; cannot run OCR")
            return

        # Ensure output dirs exist
        os.makedirs(self.cleaned_markdown_dir, exist_ok=True)

        parquet_schema = ParquetSchema({"url_column": self.url_column})
        parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)

        df = None
        candidates: List[str] = []

        if parquet_path is not None:
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                self.logger.error("Failed to read parquet %s: %s", parquet_path, e)
                df = None

        if df is not None:
            if "filename" not in df.columns:
                self.logger.error("Parquet %s missing required 'filename' column", parquet_path)
                return

            # ensure extraction_backend column present
            if "extraction_backend" not in df.columns:
                df["extraction_backend"] = None
            # ensure ocr_success column present
            if "ocr_success" not in df.columns:
                df["ocr_success"] = pd.NA

            if not force:
                if "filter" not in df.columns:
                    self.logger.error("Parquet missing 'filter' column; run clean() first or use force=True")
                    return
                need_idx = (df["filter"] != "ok") & (df["extraction_backend"].fillna("") != "nanonets_ocr")
            else:
                need_idx = df["extraction_backend"].fillna("") != "nanonets_ocr"

            candidates = df.loc[need_idx, "filename"].astype(str).tolist()
        else:
            if force:
                # Fall back to scanning downloads dir for PDFs
                candidates = [p.name for p in sorted(self.downloads_dir.glob("*.pdf"))]
                if not candidates:
                    self.logger.warning("No PDFs found in %s", self.downloads_dir)
            else:
                self.logger.error("No metadata parquet found; cannot select OCR candidates without force=True")
                return

        if not candidates:
            self.logger.info("No OCR candidates to process (force=%s)", force)
            return

        if limit is not None:
            candidates = candidates[: int(limit)]

        self.logger.info("Running OCR on %d file(s) (force=%s)…", len(candidates), force)

        ocr_success_files: List[str] = []
        # Accept either an HF model id string or a local path; don't coerce to Path
        chosen_model_dir = model_dir if model_dir is not None else self.ocr_model_dir

        for filename in candidates:
            pdf_path = self.downloads_dir / filename
            if not pdf_path.exists():
                self.logger.warning("PDF not found, skipping: %s", pdf_path)
                continue
            try:
                result = _run_nanonets_ocr(
                    pdf_path=pdf_path,
                    device=device,
                    model_dir=chosen_model_dir,
                    max_pages=max_pages,
                )
                md_text = result.get("markdown_text", "")
                pages = int(result.get("pages", 0) or 0)
                if md_text and pages > 0:
                    md_out = self.cleaned_markdown_dir / f"{pdf_path.stem}.md"
                    md_out.write_text(md_text, encoding="utf-8")
                    ocr_success_files.append(filename)
                else:
                    self.logger.warning("OCR produced no text for %s (result: %s)", filename, result.get("skipped", "no_text"))
            except Exception:
                self.logger.error("OCR failed for %s: %s", filename, traceback.format_exc())

        # Update parquet if available
        if df is not None:
            try:
                attempted = set(candidates)
                succ = set(ocr_success_files)
                fail = list(attempted - succ)

                if succ:
                    sel = df["filename"].isin(list(succ))
                    df.loc[sel, "extraction_backend"] = "nanonets_ocr"
                    df.loc[sel, "ocr_success"] = True
                if fail:
                    sel_fail = df["filename"].isin(fail)
                    df.loc[sel_fail, "ocr_success"] = False

                df.to_parquet(parquet_path, index=False)
                self.logger.info(
                    "Parquet updated: %d success OCR, %d failed attempts (path=%s)",
                    len(succ), len(fail), parquet_path,
                )
            except Exception as e:
                self.logger.error("Failed to write parquet %s: %s", parquet_path, e)

        # Optionally clear engine cache to release GPU memory
        if not persist_engine:
            try:
                import glossapi.ocr as _ocr_mod
                if hasattr(_ocr_mod, "_ENGINE_CACHE"):
                    _ocr_mod._ENGINE_CACHE.clear()
            except Exception:
                pass

    def extract(
        self, 
        input_format: str = "all", 
        num_threads: int = 4, 
        accel_type: str = "Auto"
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
        
        # Prepare extractor
        self.extractor.enable_accel(threads=num_threads, type=accel_type)
        self.extractor.create_extractor()
        
        # Create output directory
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
        
        if not input_files:
            self.logger.warning(f"No {input_format} files found in {downloads_dir}")
            return
        
        self.logger.info(f"Found {len(input_files)} files to extract")
        
        # Process all files
        self.logger.info(f"Processing {len(input_files)} files...")
        
        # Extract files to markdown
        os.makedirs(self.markdown_dir, exist_ok=True)
        
        # Use multiple threads for extraction
        self.extractor.extract_path(input_files, self.markdown_dir)
        
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
        input_df = pd.read_parquet(input_parquet)
        total_urls = len(input_df)
        self.logger.info(f"Total URLs in input file: {total_urls}")
        
        # Look for existing download results file by the specific input filename first
        input_filename = Path(input_parquet).name
        download_results_dir = Path(self.output_dir) / "download_results"
        specific_results_path = download_results_dir / f"download_results_{input_filename}"
        
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
                
            # Filter out successfully downloaded URLs
            successful_urls = []
            if 'download_success' in existing_results.columns:
                successful_urls = existing_results[
                    existing_results['download_success'] == True
                ][url_column].tolist()
                
                if successful_urls:
                    self.logger.info(f"Found {len(successful_urls)} previously successful downloads")
                    # Filter out URLs that were successfully downloaded
                    remaining_df = input_df[~input_df[url_column].isin(successful_urls)]
                    
                    # If all URLs already downloaded, return existing results
                    if len(remaining_df) == 0:
                        self.logger.info("All files already successfully downloaded")
                        return existing_results
                    
                    self.logger.info(f"Processing {len(remaining_df)} remaining URLs out of {total_urls} total")
                    
                    # Save filtered input to a temporary file for the downloader
                    temp_input = self.output_dir / "temp_download_input.parquet"
                    remaining_df.to_parquet(temp_input, index=False)
                    input_parquet = temp_input
        else:
            self.logger.info("No existing download results found or usable")
            existing_results = pd.DataFrame()
            
        # Initialize downloader with the existing filenames to avoid
        downloader = GlossDownloader(
            url_column=url_column,
            output_dir=str(self.output_dir),
            log_level=self.logger.level,
            verbose=verbose if verbose is not None else self.verbose
        )
        
        # Download files
        self.logger.info(f"Downloading files from URLs in {input_parquet}...")
        new_results = downloader.download_files(input_parquet=str(input_parquet), **kwargs)
        
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
        output_parquet = download_results_dir / f"download_results_{Path(input_parquet).name}"
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