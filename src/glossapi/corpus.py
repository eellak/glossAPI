import logging
from pathlib import Path
import os
import pandas as pd
import random
import numpy as np
from typing import Dict, Optional, Union, List, Any, Tuple, Set, Iterable
import shutil
import math
import re
import subprocess
import sys
import pickle
import queue
import time
import json
from dataclasses import dataclass

from ._naming import canonical_stem
from .gloss_section import GlossSection
from .gloss_section_classifier import GlossSectionClassifier
from .gloss_downloader import GlossDownloader


def _maybe_import_torch(*, force: bool = False):
    """Return torch module if already loaded or explicitly requested via env."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        return torch_mod
    try:
        import importlib

        return importlib.import_module("torch")  # type: ignore
    except Exception:
        return None


@dataclass
class _SkiplistManager:
    """Single-writer helper around the on-disk fatal skip-list."""

    path: Path
    logger: logging.Logger
    _cache: Optional[Set[str]] = None

    @staticmethod
    def _normalize(entry: Optional[str]) -> Optional[str]:
        if not entry:
            return None
        stem = canonical_stem(entry.strip())
        return stem or None

    def load(self) -> Set[str]:
        if self._cache is not None:
            return set(self._cache)
        stems: Set[str] = set()
        try:
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    norm = self._normalize(line)
                    if norm:
                        stems.add(norm)
        except Exception as exc:
            self.logger.warning("Failed to read skip-list %s: %s", self.path, exc)
        self._cache = stems
        return set(stems)

    def add(self, new_entries: Iterable[str]) -> Set[str]:
        current = self.load()
        to_add = {stem for stem in (self._normalize(val) for val in new_entries) if stem}
        if not to_add or to_add.issubset(current):
            return current
        merged = current | to_add
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text("\n".join(sorted(merged)) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
            self._cache = merged
            self.logger.warning(
                "Skip-list updated (%d new stem%s): %s",
                len(to_add),
                "s" if len(to_add) != 1 else "",
                ", ".join(sorted(to_add)),
            )
        except Exception as exc:
            self.logger.error("Failed to update skip-list %s: %s", self.path, exc)
        return self.load()

    def reload(self) -> Set[str]:
        self._cache = None
        return self.load()


def _resolve_skiplist_path(output_dir: Path, logger: logging.Logger) -> Path:
    env_override = os.environ.get("GLOSSAPI_SKIPLIST_PATH")
    if env_override:
        return Path(env_override)

    candidate = output_dir / "skiplists" / "fatal_skip.txt"
    legacy = output_dir.parent / "aws_bundle" / "skiplists" / "fatal_skip.txt"

    for option in (candidate, legacy):
        try:
            option.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("Skip-list path %s could not be prepared: %s", option, exc)
        if option.exists():
            return option

    return candidate


class _ProcessingStateManager:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.logger = logging.getLogger(__name__)

    def load(self) -> Tuple[Set[str], Set[str]]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "rb") as handle:
                    state = pickle.load(handle)
                processed = set(state.get("processed", set()))
                problematic = set(state.get("problematic", set()))
                return processed, problematic
            except Exception as exc:
                self.logger.warning("Failed to load processing state %s: %s", self.state_file, exc)
        return set(), set()

    def save(self, processed: Set[str], problematic: Set[str]) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "wb") as handle:
                pickle.dump({"processed": set(processed), "problematic": set(problematic)}, handle)
        except Exception as exc:
            self.logger.warning("Failed to persist processing state %s: %s", self.state_file, exc)


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

    def jsonl(self, output_path: Union[str, Path]) -> Path:
        """Export cleaned markdown and metadata into a JSONL corpus."""

        output_path = Path(output_path)
        download_dir = self.output_dir / "download_results"
        metadata_path = download_dir / "download_results.parquet"
        if not metadata_path.exists():
            candidates = sorted(download_dir.glob("*.parquet")) if download_dir.exists() else []
            if not candidates:
                raise FileNotFoundError(f"Metadata parquet not found in {download_dir}")
            preferred = [p for p in candidates if p.name.startswith("download_results_")]
            metadata_path = preferred[0] if preferred else candidates[0]

        df = pd.read_parquet(metadata_path)
        if df.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")
            return output_path

        def _stem_for(value: str) -> str:
            if not value:
                return ""
            return canonical_stem(value)

        df["__stem__"] = df["filename"].astype(str).map(_stem_for)
        metadata_by_stem: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            data = row.to_dict()
            stem = str(data.pop("__stem__", ""))
            if stem:
                metadata_by_stem[stem] = data

        markdown_root = self.cleaned_markdown_dir if any(self.cleaned_markdown_dir.glob("*.md")) else self.markdown_dir

        def _load_metrics(stem: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
            metrics_dir = self.output_dir / "json" / "metrics"
            candidates = [
                metrics_dir / f"{stem}.metrics.json",
                metrics_dir / f"{stem}.per_page.metrics.json",
            ]
            data = None
            for candidate in candidates:
                if candidate.exists():
                    try:
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        continue
            if not data:
                return None, None, None
            page_count = data.get("page_count")
            pages = data.get("pages") or []
            try:
                formula_total = sum(int(p.get("formula_count", 0) or 0) for p in pages)
            except Exception:
                formula_total = None
            try:
                code_total = sum(int(p.get("code_count", 0) or 0) for p in pages)
            except Exception:
                code_total = None
            try:
                if page_count is not None:
                    page_count = int(page_count)
            except Exception:
                page_count = None
            return page_count, formula_total, code_total

        def _load_math_accepts(stem: str) -> tuple[bool, int]:
            latex_map = self.output_dir / "json" / f"{stem}.latex_map.jsonl"
            if not latex_map.exists():
                return False, 0
            accepted = 0
            try:
                with latex_map.open("r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        score = payload.get("accept_score")
                        try:
                            if score is not None and float(score) >= 1.0:
                                accepted += 1
                        except Exception:
                            continue
            except Exception:
                return True, 0
            return True, accepted

        def _normalize_value(value: Any) -> Any:
            if value is None:
                return None
            try:
                if isinstance(value, float) and math.isnan(value):
                    return None
            except Exception:
                pass
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            return value

        output_path.parent.mkdir(parents=True, exist_ok=True)
        records_written = 0
        with output_path.open("w", encoding="utf-8") as fp:
            for md_path in sorted(markdown_root.glob("*.md")):
                stem = canonical_stem(md_path)
                metadata = metadata_by_stem.get(stem)
                if metadata is None:
                    continue
                record = {k: _normalize_value(v) for k, v in metadata.items()}
                record["document"] = md_path.read_text(encoding="utf-8")

                filetype = record.get("filetype") or record.get("file_ext")
                if not filetype:
                    filename_value = record.get("filename")
                    if isinstance(filename_value, str):
                        filetype = Path(filename_value).suffix.lstrip(".")
                record["filetype"] = filetype or None

                metrics_page_count, metrics_formula, metrics_code = _load_metrics(stem)
                if metrics_page_count is not None:
                    record["page_count"] = metrics_page_count

                existing_formula = record.get("formula_total")
                if metrics_formula is not None:
                    record["formula_total"] = metrics_formula
                else:
                    try:
                        record["formula_total"] = int(existing_formula)
                    except Exception:
                        record["formula_total"] = 0

                existing_code = record.get("code_total")
                if metrics_code is not None:
                    record["code_total"] = metrics_code
                else:
                    try:
                        record["code_total"] = int(existing_code)
                    except Exception:
                        record["code_total"] = 0

                math_enriched, math_accepted = _load_math_accepts(stem)
                record["math_enriched"] = bool(math_enriched)
                record["math_accepted"] = int(math_accepted)

                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1

        if records_written == 0:
            output_path.write_text("", encoding="utf-8")
        return output_path
    
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

    @staticmethod
    def _project_root() -> Path:
        """Locate the repository root that houses the Rust crates."""
        here = Path(__file__).resolve()
        for candidate in here.parents:
            rust_dir = candidate / "rust"
            if rust_dir.exists() and rust_dir.is_dir():
                return candidate
        return here.parents[2]

    def _load_rust_extension(self, module_name: str, manifest_relative: str):
        """Import a Rust extension, building it with maturin if necessary."""
        import importlib

        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            self.logger.warning(
                "Rust extension %s missing; attempting in-place build via maturin …",
                module_name,
            )
            root_dir = self._project_root()
            manifest = root_dir / manifest_relative
            if not manifest.exists():
                raise RuntimeError(
                    f"Cannot locate Cargo manifest for {module_name} at {manifest}"
                )
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0"],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "maturin",
                        "develop",
                        "--release",
                        "--manifest-path",
                        str(manifest),
                    ],
                    check=True,
                )
                return importlib.import_module(module_name)
            except Exception as build_err:
                raise RuntimeError(
                    f"Automatic build of {module_name} failed: {build_err}"
                )

    def _load_metrics_dataframe(
        self, parquet_path: Path, filenames: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        """Load an analytics parquet or seed an empty frame keyed by filename."""
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        names: List[str] = []
        if filenames is not None:
            seen: Set[str] = set()
            for item in filenames:
                if item is None:
                    continue
                name = str(item)
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return pd.DataFrame({"filename": names})

    @staticmethod
    def _ensure_metric_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> None:
        """Ensure metric columns exist with provided defaults."""
        for column, default in defaults.items():
            if column not in df.columns:
                df[column] = default

    @staticmethod
    def _merge_metric_dataframe(
        base: pd.DataFrame, updates: pd.DataFrame, *, key: str = "filename"
    ) -> pd.DataFrame:
        """Overlay scorer output onto the authoritative metrics dataframe."""
        if updates.empty:
            return base
        base_idx = base.set_index(key, drop=False)
        update_idx = updates.set_index(key, drop=False)
        base_idx = base_idx.combine_first(update_idx)
        base_idx.update(update_idx)
        return base_idx.reset_index(drop=True)

    def clean(
        self,
        input_dir: Union[str, Path] = None,
        threshold: float = 0.10,
        num_threads: int = None,
        drop_bad: bool = True,
        *,
        ocr_model_dir: Union[str, Path, None] = None,
        force_ocr_fallback: bool = False,
        empty_char_threshold: int = 0,
        empty_min_pages: int = 0,
    ) -> None:
        """Clean markdown files and evaluate badness using the Rust extension.

        Args:
            input_dir: Folder with `.md` files to process (defaults to `self.markdown_dir`).
            threshold: Badness threshold for optional dropping.
            num_threads: Rayon thread-count to pass to Rust.
            drop_bad: If True, files with badness_score > threshold are removed from downstream processing. Set to False to keep all files and only record the score.
            ocr_model_dir: [DEPRECATED – no effect] Use Corpus.ocr(model_dir=...) instead.
            force_ocr_fallback: [DEPRECATED – no effect] Use Corpus.ocr(fix_bad=True) instead.
            empty_char_threshold: Character threshold (after stripping comments and whitespace) that flags markdown as nearly empty. Default 0 only enforces the zero-character safeguard.
            empty_min_pages: Minimum page count for a low-character document to trigger an OCR rerun recommendation.
        """
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

        self._load_rust_extension(
            "glossapi_rs_cleaner", "rust/glossapi_rs_cleaner/Cargo.toml"
        )
        self.logger.info("Using compiled glossapi_rs_cleaner extension for fast cleaning")

        # Ensure cleaned directory exists and is empty (idempotent runs)
        if self.cleaned_markdown_dir.exists():
            shutil.rmtree(self.cleaned_markdown_dir)
        self.cleaned_markdown_dir.mkdir(parents=True, exist_ok=True)

        # Prepare parquet helper
        parquet_schema = ParquetSchema({"url_column": self.url_column})
        existing_metadata = parquet_schema.find_metadata_parquet(self.input_dir)
        parquet_path: Optional[Path] = Path(existing_metadata) if existing_metadata else None
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.output_dir)
            if ensured is not None:
                parquet_path = Path(ensured)
        if parquet_path is None:
            ensured = parquet_schema.ensure_metadata_parquet(self.input_dir)
            if ensured is not None:
                parquet_path = Path(ensured)
        if parquet_path is None:
            parquet_path = None
            metadata_target = self.output_dir / "download_results" / "download_results.parquet"
            self.logger.info(
                "Cleaner: no metadata parquet found; will bootstrap %s when metrics become available.",
                metadata_target,
            )
        else:
            metadata_target = parquet_path
        parquet_path = metadata_target

        import os
        records: list = []  # will hold metrics for parquet merge
        metrics_dir = self.output_dir / "json" / "metrics"

        def _page_count_for(stem: str) -> Optional[int]:
            candidates = [
                metrics_dir / f"{stem}.metrics.json",
                metrics_dir / f"{stem}.per_page.metrics.json",
            ]
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, dict):
                    pc = data.get("page_count")
                    if pc is not None:
                        try:
                            return int(pc)
                        except Exception:
                            pass
                    pages = data.get("pages")
                    if isinstance(pages, list):
                        return len(pages)
            return None

        # ----- Call Rust high-level pipeline once -----
        scripts_to_keep = ["greek", "latin"]  # keep common alphabetic scripts; numbers/punctuation are added internally
        report_parquet_path = self.cleaned_markdown_dir.parent / "cleaning_report.parquet"

        md_files = sorted(input_dir.glob("*.md"))
        total_files = len(md_files)

        self.logger.info(
            "Invoking glossapi_rs_cleaner.run_complete_pipeline on %d markdown files…",
            total_files,
        )

        class _CleanerProgress:
            def __init__(self, logger: logging.Logger, total: int) -> None:
                self.logger = logger
                self.total = total
                self.processed: set[str] = set()
                self.buffer = ""
                if total > 0:
                    step = max(1, math.ceil(total * 0.02))
                else:
                    step = 1
                self.step = step
                self.next_target = step
                self.logged_full = False
                self.last_message: Optional[str] = None

            def write(self, text: str) -> int:
                if not text:
                    return 0
                self.buffer += text
                while "\n" in self.buffer:
                    line, self.buffer = self.buffer.split("\n", 1)
                    self._handle_line(line.strip())
                return len(text)

            def flush(self) -> None:  # pragma: no cover - required by IO interface
                return

            def handle_line(self, line: str) -> None:
                self._handle_line(line.strip())

            def _handle_line(self, line: str) -> None:
                if not line:
                    return
                match = re.search(r"Processing file:\s*(.+)", line)
                if match:
                    path = match.group(1).strip()
                    stem = Path(path).stem if path else None
                    if stem and stem not in self.processed:
                        self.processed.add(stem)
                        self._log_progress()
                    return
                if "complete pipeline finished successfully" in line or "Parquet report written successfully" in line:
                    self.last_message = line

            def _log_progress(self) -> None:
                if self.total <= 0:
                    return
                processed = len(self.processed)
                while self.next_target <= self.total and processed >= self.next_target:
                    percent = min(100, int(round(self.next_target * 100 / self.total)))
                    self.logger.info(
                        "Rust cleaning progress: %d%% (%d/%d)", percent, processed, self.total
                    )
                    if percent >= 100:
                        self.logged_full = True
                    self.next_target += self.step

            def finalize(self) -> None:
                if self.total == 0:
                    self.logger.info("Rust cleaning progress: 100%% (0/0)")
                elif not self.logged_full:
                    processed = len(self.processed)
                    self.logger.info(
                        "Rust cleaning progress: 100%% (%d/%d)", processed, self.total
                    )
                if self.last_message:
                    self.logger.debug(self.last_message)

        progress = _CleanerProgress(self.logger, total_files)
        cmd = (
            "import glossapi_rs_cleaner\n"
            f"glossapi_rs_cleaner.run_complete_pipeline({repr(str(input_dir))}, "
            f"{repr(str(self.cleaned_markdown_dir))}, {repr(str(report_parquet_path))}, "
            f"{repr(scripts_to_keep)}, {int(num_threads or os.cpu_count() or 4)})\n"
        )

        process = subprocess.Popen(
            [sys.executable, "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                progress.handle_line(line)
            return_code = process.wait()
        except Exception:
            process.kill()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            progress.finalize()

        if return_code != 0:
            raise RuntimeError("Rust cleaning pipeline failed")

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
                            "char_count_no_comments": row.get("char_count_no_comments"),
                            "is_empty": row.get("is_empty", False),
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
            df_metrics = pd.DataFrame(records).rename(
                columns={
                    "badness_score": "mojibake_badness_score",
                    "percentage_latin": "mojibake_latin_percentage",
                }
            )

            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            df = self._load_metrics_dataframe(parquet_path, df_metrics.get("filename"))
            self._ensure_metric_columns(
                df,
                {
                    "mojibake_badness_score": pd.NA,
                    "mojibake_latin_percentage": pd.NA,
                    "percentage_greek": pd.NA,
                    "greek_badness_score": pd.NA,
                    "greek_latin_percentage": pd.NA,
                    "rejection_reason": pd.NA,
                    "char_count_no_comments": pd.NA,
                    "is_empty": pd.NA,
                },
            )

            df = self._merge_metric_dataframe(
                df,
                df_metrics[
                    [
                        "filename",
                        "mojibake_badness_score",
                        "mojibake_latin_percentage",
                        "percentage_greek",
                        "char_count_no_comments",
                        "is_empty",
                    ]
                ],
            )
            parquet_schema.write_metadata_parquet(df, parquet_path)
            self.logger.info("Mojibake metrics updated in %s", parquet_path)

        # ----- Noise-metrics scoring (Rust) -----
        try:
            self.logger.info("Scoring cleaned markdown files with glossapi_rs_noise …")
            noise_mod = self._load_rust_extension(
                "glossapi_rs_noise", "rust/glossapi_rs_noise/Cargo.toml"
            )
            results = noise_mod.score_markdown_directory_detailed(
                str(self.cleaned_markdown_dir), os.cpu_count()
            )
            if results:
                rows = []
                for row in results:
                    try:
                        path, score, latin_pct, _table_ratio, poly_ratio = row[:5]
                    except Exception:
                        continue
                    rows.append((path, float(score), float(latin_pct), float(poly_ratio)))

                df_scores = pd.DataFrame(
                    rows,
                    columns=[
                        "filepath",
                        "greek_badness_score",
                        "greek_latin_percentage",
                        "polytonic_ratio",
                    ],
                )
                df_scores["polytonic_ratio"] = df_scores["polytonic_ratio"].round(2)
                df_scores["stem"] = df_scores["filepath"].apply(lambda p: Path(p).name)
                df_scores["stem"] = df_scores["stem"].str.replace(r"\.md$", "", regex=True)
                df_scores["filename"] = df_scores["stem"] + ".pdf"
                df_scores["rejection_reason"] = np.select(
                    [df_scores["greek_badness_score"] > 60],
                    ["greek>60"],
                    default="ok",
                )
                if not parquet_path.exists():
                    self.logger.error(
                        "Expected parquet %s not found when adding noise metrics",
                        parquet_path,
                    )
                else:
                    df = self._load_metrics_dataframe(parquet_path)
                    self._ensure_metric_columns(
                        df,
                        {
                            "greek_badness_score": pd.NA,
                            "greek_latin_percentage": pd.NA,
                            "polytonic_ratio": pd.NA,
                            "rejection_reason": pd.NA,
                        },
                    )
                    updates = df_scores[
                        [
                            "filename",
                            "greek_badness_score",
                            "greek_latin_percentage",
                            "polytonic_ratio",
                            "rejection_reason",
                        ]
                    ]
                    df = self._merge_metric_dataframe(df, updates)
                    parquet_schema.write_metadata_parquet(df, parquet_path)
                    self.logger.info("Noise metrics filled in %s", parquet_path)
        except Exception as e:
            self.logger.warning("Noise-metrics scoring failed: %s", e)


        # Determine good / bad list based on enriched metrics
        if parquet_path.exists():
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

            def _collapse_measure(df: pd.DataFrame, base: str) -> None:
                cols = [col for col in df.columns if col == base or col.startswith(f"{base}_")]
                if not cols:
                    return
                collapsed = None
                for col in cols:
                    values = pd.to_numeric(df[col], errors="coerce")
                    collapsed = values if collapsed is None else collapsed.combine_first(values)
                df[base] = collapsed
                for col in cols:
                    if col != base:
                        df.drop(columns=col, inplace=True, errors="ignore")

            _collapse_measure(df_final, "char_count_no_comments")
            _collapse_measure(df_final, "page_count")

            if "char_count_no_comments" in df_final.columns:
                df_final["char_count_no_comments"] = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce")
            if "page_count" in df_final.columns:
                df_final["page_count"] = pd.to_numeric(df_final["page_count"], errors="coerce")

            df_final["filter"] = "ok"
            df_final["needs_ocr"] = False
            if "is_empty" in df_final.columns:
                df_final["is_empty"] = df_final["is_empty"].fillna(False).astype(bool)
            else:
                df_final["is_empty"] = False

            filename_series = df_final.get("filename")
            if filename_series is None:
                pdf_mask = pd.Series(False, index=df_final.index)
            else:
                pdf_mask = filename_series.astype(str).str.lower().str.endswith(".pdf")
                pdf_mask = pdf_mask.fillna(False)

            def _append_reason(mask: pd.Series, reason: str, *, requires_ocr: bool) -> None:
                if df_final.empty:
                    return
                if not isinstance(mask, pd.Series):
                    mask = pd.Series(mask, index=df_final.index)
                mask = mask.fillna(False)
                applicable = (mask & pdf_mask).fillna(False)
                if not bool(applicable.any()):
                    return
                current = df_final.loc[applicable, "filter"].astype(str)

                def _merge_reason(value: str) -> str:
                    if value == "ok" or not value:
                        return reason
                    parts = [part for part in value.split(";") if part]
                    if reason not in parts:
                        parts.append(reason)
                    return ";".join(parts)

                df_final.loc[applicable, "filter"] = current.apply(_merge_reason)
                if requires_ocr:
                    needs_targets = applicable
                    if "ocr_success" in df_final.columns:
                        success_mask = df_final["ocr_success"].fillna(False)
                        needs_targets = needs_targets & ~success_mask
                    df_final.loc[needs_targets, "needs_ocr"] = True

            try:
                empty_threshold_int = int(empty_char_threshold) if empty_char_threshold is not None else 0
            except Exception:
                empty_threshold_int = 0
            if empty_threshold_int < 0:
                empty_threshold_int = 0
            try:
                min_pages = int(empty_min_pages) if empty_min_pages is not None else 0
            except Exception:
                min_pages = 0
            if min_pages < 0:
                min_pages = 0

            mojibake_series = pd.to_numeric(df_final.get("mojibake_badness_score"), errors="coerce")
            if mojibake_series.notna().any():
                _append_reason(mojibake_series > 0.1, "mojibake>0.1", requires_ocr=True)

            greek_series = pd.to_numeric(df_final.get("greek_badness_score"), errors="coerce")
            if greek_series.notna().any():
                _append_reason(greek_series > 60, "non_greek_text", requires_ocr=True)

            if "char_count_no_comments" in df_final.columns:
                char_series = pd.to_numeric(df_final["char_count_no_comments"], errors="coerce").fillna(0)
                page_series_raw = df_final.get("page_count")
                if page_series_raw is not None:
                    page_series = pd.to_numeric(page_series_raw, errors="coerce")
                else:
                    page_series = pd.Series(np.nan, index=df_final.index, dtype="float64")
                page_series = page_series.fillna(min_pages if min_pages else 0)

                zero_mask = char_series <= 0
                zero_pdf = (zero_mask & pdf_mask).fillna(False)
                if bool(zero_pdf.any()):
                    df_final.loc[zero_pdf.index, "is_empty"] = df_final.loc[zero_pdf.index, "is_empty"] | zero_pdf
                if empty_threshold_int == 0:
                    zeros = int(zero_pdf.sum())
                    if zeros:
                        self.logger.info("Empty text check: %d files have zero characters", zeros)
                    _append_reason(zero_pdf, "empty_text==0", requires_ocr=True)
                elif empty_threshold_int > 0:
                    low_mask = char_series < empty_threshold_int
                    long_mask = page_series >= max(1, min_pages)
                    _append_reason(low_mask & long_mask, f"empty_text<{empty_threshold_int}", requires_ocr=True)
                    if min_pages > 0:
                        _append_reason(low_mask & ~long_mask, f"empty_text<{empty_threshold_int}_short", requires_ocr=False)
                    total_low = int(low_mask.fillna(False).sum())
                    long_low = int((low_mask & long_mask).fillna(False).sum())
                    self.logger.info(
                        "Empty text check: %d files below %d chars; %d have >= %d pages",
                        total_low,
                        empty_threshold_int,
                        long_low,
                        min_pages,
                    )

            df_final["needs_ocr"] = df_final["needs_ocr"].fillna(False).astype(bool)

            # persist cleaned parquet
            parquet_schema.write_metadata_parquet(df_final, parquet_path)
            if drop_bad:
                good_df = df_final[df_final["needs_ocr"] == False]
                filenames = good_df.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
                self.logger.info(f"After filtering, {len(self.good_files)} good files remain")
            else:
                filenames = df_final.get("filename", pd.Series(dtype=str))
                self.good_files = [canonical_stem(f) for f in filenames.dropna().astype(str).tolist()]
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
        fix_bad: bool = True,
        mode: Optional[str] = None,
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
        - fix_bad: re-run OCR on documents marked bad by the cleaner (default True).
        - math_enhance: run math/code enrichment after OCR (default True).
        - force: [DEPRECATED] alias for fix_bad retained for backward compatibility.
        - reprocess_completed: when False, skip documents already flagged as successfully
          OCRed or math-enriched in metadata. Set True to force reprocessing. Defaults to False
          unless ``skip_existing`` overrides it.
        - skip_existing: legacy alias for ``reprocess_completed`` (``skip_existing=True`` equals
          ``reprocess_completed=False``). Prefer the explicit ``reprocess_completed`` toggle.
        """
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
            parquet_schema.ensure_metadata_parquet(self.output_dir)
            parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)
            if parquet_path and parquet_path.exists():
                import pandas as _pd
                df = _pd.read_parquet(parquet_path)
                if "filename" in df.columns and "needs_ocr" in df.columns:
                    bad_files = df.loc[df["needs_ocr"] == True, "filename"].dropna().astype(str).tolist()
                elif "filename" in df.columns and "filter" in df.columns:
                    bad_files = df.loc[df["filter"] != "ok", "filename"].dropna().astype(str).tolist()
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
            if not stems:
                self.extract(
                    input_format="pdf",
                    num_threads=os.cpu_count() or 4,
                    accel_type="CUDA",
                    force_ocr=False,
                    formula_enrichment=False,
                    code_enrichment=False,
                    filenames=None,
                    skip_existing=False,
                    export_doc_json=True,
                    emit_formula_index=True,
                    phase1_backend="docling",
                )
                if json_dir.exists():
                    stems = sorted({canonical_stem(p) for p in json_dir.glob("*.docling.json*")})
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
                # When math follows we need JSON; otherwise it's optional
                export_doc_json=bool(mode_norm == "ocr_bad_then_math"),
                emit_formula_index=bool(mode_norm == "ocr_bad_then_math"),
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
                    parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)
                    if parquet_path and Path(parquet_path).exists():
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
                stems = [canonical_stem(f) for f in bad_files]
                if not stems:
                    json_dir = self.output_dir / "json"
                    if json_dir.exists():
                        stems = sorted({canonical_stem(p) for p in json_dir.glob("*.docling.json*")})
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
            from .gloss_extract import GlossExtract
            self.extractor = GlossExtract(url_column=self.url_column)

        # Propagate toggles used by Phase‑1 helpers
        try:
            setattr(self.extractor, "export_doc_json", bool(export_doc_json))
            setattr(self.extractor, "emit_formula_index", bool(emit_formula_index))
        except Exception:
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

        # Hard GPU preflight before we attempt to build OCR/enrichment pipelines
        self._gpu_preflight(
            accel_type=accel_type,
            require_ocr=bool(force_ocr),
            require_math=bool(formula_enrichment or code_enrichment),
            require_backend_gpu=(backend_choice == "docling"),
        )

        # Configure batch/backend policy based on resolved choice
        if backend_choice == "docling":
            # Keep docling runs conservative: process one document per batch for stability
            self.extractor.configure_batch_policy("docling", max_batch_files=1, prefer_safe_backend=False)
        else:
            self.extractor.configure_batch_policy("safe", max_batch_files=1, prefer_safe_backend=True)

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
        file_paths: Optional[List[Union[str, Path]]] = None,
        skip_existing: bool = True,
        use_gpus: str = "single",
        devices: Optional[List[int]] = None,
        use_cls: bool = False,
        benchmark_mode: bool = False,
        export_doc_json: bool = True,
        emit_formula_index: bool = False,
        phase1_backend: str = "auto",
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
                OCR/math is requested, ``"safe"`` forces the PyPDFium backend, and ``"docling"`` forces the
                Docling backend.

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
                raise ValueError(f"Invalid file path supplied to extract(): {exc}")
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

                batch_hint = 5 if backend_choice == "docling" and not force_ocr else 1
                self.logger.info(
                    "Phase-1 config: backend=%s batch_size=%s threads=%s skip_existing=%s benchmark=%s",
                    backend_choice,
                    batch_hint,
                    threads_effective,
                    bool(skip_existing),
                    bool(benchmark_mode),
                )

                state_mgr = _ProcessingStateManager(self.markdown_dir / ".processing_state.pkl")
                processed_files, problematic_files = state_mgr.load()
                if skip_existing and processed_files:
                    self.logger.info(
                        "State resume: %d processed, %d problematic", len(processed_files), len(problematic_files)
                    )

                pending_files = input_files
                if skip_existing and processed_files:
                    processed_names = {Path(name).name for name in processed_files}
                    pending_files = [p for p in pending_files if p.name not in processed_names]
                if problematic_files:
                    problematic_names = {Path(name).name for name in problematic_files}
                    before_prob = len(pending_files)
                    pending_files = [p for p in pending_files if p.name not in problematic_names]
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
                from multiprocessing import get_context
                ctx = get_context("spawn")
                manager = ctx.Manager()
                task_q = ctx.Queue()
                result_q = ctx.Queue()
                status_map = manager.dict()
                path_list = [str(p.resolve()) for p in pending_files]
                for full_path in path_list:
                    task_q.put(full_path)
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
                procs: List[Any] = []
                proc_gpu: Dict[int, int] = {}
                marker_files: Dict[int, Path] = {dev_id: marker_base / f"gpu{dev_id}.current" for dev_id in devs}
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
                            backend_choice,
                            result_q,
                            status_map,
                            str(marker_base),
                        ),
                    )
                    p.start()
                    procs.append(p)
                    if p.pid is not None:
                        proc_gpu[p.pid] = dev_id
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
                            gpu_id = proc_gpu.pop(pid, None)
                            if p.exitcode not in (0, None):
                                any_fail = True
                                self.logger.warning("GPU worker pid=%s exited with code %s", p.pid, p.exitcode)
                                current_paths: List[str] = []
                                stems_for_skip: List[str] = []
                                if gpu_id is not None:
                                    current_entry = status_map.pop(gpu_id, None)
                                    if current_entry:
                                        if not isinstance(current_entry, (list, tuple, set)):
                                            current_entry = [current_entry]
                                        current_paths = [str(x) for x in current_entry]
                                        stems_for_skip = [canonical_stem(path) for path in current_paths]
                                    marker_path = marker_files.get(gpu_id)
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
                                if gpu_id is not None:
                                    self.logger.info("Respawning GPU%s worker after crash.", gpu_id)
                                    replacement = ctx.Process(
                                        target=gpu_extract_worker_queue,
                                        args=(
                                            gpu_id,
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
                                        proc_gpu[replacement.pid] = gpu_id
                                        heartbeat[replacement.pid] = time.time()
                                continue
                            else:
                                if gpu_id is not None:
                                    status_map.pop(gpu_id, None)
                                    marker_path = marker_files.get(gpu_id)
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
                                    "GPU%s batch complete: +%d processed, +%d problematic (totals: %d processed, %d problematic)",
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
                                        "GPU%s reported non-zero exit: %s", result.get("worker"), result.get("exitcode")
                                    )
                                worker_pid = result.get("pid")
                                if worker_pid is not None:
                                    heartbeat[worker_pid] = time.time()
                                worker_gpu = result.get("worker")
                                if worker_gpu is not None:
                                    try:
                                        worker_gpu_int = int(worker_gpu)
                                    except Exception:
                                        worker_gpu_int = None
                                    else:
                                        status_map.pop(worker_gpu_int, None)
                                        marker_path = marker_files.get(worker_gpu_int)
                                        if marker_path:
                                            try:
                                                marker_path.unlink(missing_ok=True)
                                            except Exception:
                                                pass

                        now = time.time()
                        if now - last_summary > 30:
                            try:
                                pending = result_q.qsize()
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
        needs_gpu = bool(force_ocr or formula_enrichment or code_enrichment)
        if choice == "auto":
            choice = "docling" if needs_gpu else "safe"
        if choice == "safe" and needs_gpu:
            self.logger.info(
                "Phase-1 backend 'safe' overridden to 'docling' because OCR/math enrichment was requested."
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
        """Abort early when GPU OCR/math is requested but CUDA is unavailable."""
        if not (require_ocr or require_math or require_backend_gpu):
            return

        instructions = (
            "GPU OCR and math enrichment require CUDA-enabled torch and onnxruntime-gpu. "
            "Install the CUDA wheels and ensure NVIDIA drivers expose the desired devices."
        )

        # Enforce non-CPU accelerator selection when OCR/math is forced
        accel_lower = str(accel_type or "").strip().lower()
        if accel_lower.startswith("cpu"):
            raise RuntimeError(
                "GPU OCR was requested (force_ocr/math) but accel_type='CPU'. "
                f"{instructions}"
            )

        try:
            import onnxruntime as _ort  # type: ignore
            providers = _ort.get_available_providers()
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime not available while attempting GPU OCR. "
                "Install onnxruntime-gpu and rerun."
            ) from exc

        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError(
                "CUDAExecutionProvider missing from onnxruntime providers. "
                f"Detected providers={providers}. {instructions}"
            )

        torch_mod = _maybe_import_torch(force=True)
        if torch_mod is None or not getattr(torch_mod, "cuda", None) or not torch_mod.cuda.is_available():
            raise RuntimeError(
                "Torch CUDA is not available but GPU OCR/math was requested. "
                "Install the CUDA wheel (e.g. torch==2.5.1+cu121) and ensure CUDA drivers/devices are visible."
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
                "GPU preflight: using torch + onnxruntime GPU backends; ensure CUDA drivers are available."
            )
            self._gpu_banner_logged = True

        self.logger.info(
            "GPU preflight OK: providers=%s torch_devices=%s",
            ",".join(providers),
            ", ".join(device_names) or "<none>",
        )


    

    
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
            parquet_schema.ensure_metadata_parquet(self.output_dir)
            parquet_path = parquet_schema.find_metadata_parquet(self.output_dir)
            if parquet_path is None:
                # Try legacy input_dir locations
                parquet_schema.ensure_metadata_parquet(self.input_dir)
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
                            parquet_schema.write_metadata_parquet(df_meta, Path(parquet_path))
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
            used_bases |= {canonical_stem(str(fn)) for fn in existing_filenames if isinstance(fn, str)}
        except Exception:
            pass
        try:
            # Also include on-disk stems
            downloads_dir = Path(self.output_dir) / 'downloads'
            if downloads_dir.exists():
                used_bases |= {canonical_stem(p) for p in downloads_dir.glob('*') if p.is_file()}
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
                    _torch_name = _torch.cuda.get_device_name(0) if getattr(_torch, "cuda", None) and _torch.cuda.is_available() else "no-cuda"
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
                print(f"[MATH GPU{device_id}] bound: CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES','')} pid={_os.getpid()} torch={_torch_name} ORT={_ort_prov}")
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
            _sys.path.insert(0, str((_pl.Path(out_dir).resolve().parents[1] / 'src').resolve()))
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
    batch: list[str] = []
    B = max(1, int(batch_size))
    exit_code = 0
    import queue as _queue

    def _report_failure(err: Exception, items: List[str]) -> None:
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
        exit_code = 1

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

    try:
        while True:
            try:
                nm = work_q.get_nowait()
            except _queue.Empty:
                if batch:
                    pending = list(batch)
                    _targets = {s: targets_map.get(s) for s in pending if s in targets_map} if targets_map else None
                    try:
                        _update_current(pending)
                        c.formula_enrich_from_json(
                            files=pending,
                            device=(device or "cuda"),
                            batch_size=B,
                            dpi_base=int(dpi_base),
                            targets_by_stem=_targets,
                        )
                    except Exception as _e:
                        _report_failure(_e, pending)
                        batch.clear()
                        break
                    else:
                        _clear_current()
                        batch.clear()
                break
            if isinstance(nm, str) and nm.strip():
                batch.append(nm)
                _update_current(list(batch))
            if len(batch) >= B:
                pending = list(batch)
                _targets = {s: targets_map.get(s) for s in pending if s in targets_map} if targets_map else None
                try:
                    _update_current(pending)
                    c.formula_enrich_from_json(
                        files=pending,
                        device=(device or "cuda"),
                        batch_size=B,
                        dpi_base=int(dpi_base),
                        targets_by_stem=_targets,
                    )
                except Exception as _e:
                    _report_failure(_e, pending)
                    batch.clear()
                    break
                else:
                    batch.clear()
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
