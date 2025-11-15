from typing import Dict, Set, List, Optional, Iterable, Tuple, Any, Union

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
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
import json


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
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.USE_V2 = True
        self.log_file = Path('.') / 'conversion.log'
        self.url_column = url_column  # Store the URL column name for later use
        self._metadata_parquet_path = None  # Store metadata parquet path once found
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(self.log_file), mode='w'),  
                logging.StreamHandler()
            ]
        )
        self._log = logging.getLogger(__name__)
        self.converter = None
                    
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

    def create_extractor(self):
        """Create a document converter with the configured options for multiple formats."""
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

        if input_parquet_path is not None:
            self._metadata_parquet_path = input_parquet_path
            logger.info(f"Found metadata parquet file: {input_parquet_path}")

        return input_parquet_path

    
    @contextmanager
    def _timeout(self, seconds):
        """Context manager for setting a timeout for a block of code.
        
        Args:
            seconds: Timeout in seconds
            
        Raises:
            TimeoutError: If the timeout is reached
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Processing timed out after {seconds} seconds")
            
        # Set the timeout handler
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Cancel the alarm and restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    
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
        successful = []
        problematic = []
        
        # Try processing as a batch first
        try:
            # Apply timeout to batch processing
            with self._timeout(self.processing_timeout):
                # Convert all input documents
                conv_results = self.converter.convert_all(
                    batch,
                    raises_on_error=False,
                )
                
                # Export results to markdown files
                success_count, partial_success_count, failure_count = self._export_documents(
                    conv_results, output_dir=output_dir
                )
                
                # All files in batch were processed successfully
                successful = [Path(file_path).name for file_path in batch]
                return successful, problematic
            
        except Exception as batch_error:
            self._log.warning(f"Batch processing failed with error: {batch_error}. Processing files individually.")
            
            # Process files individually to identify problematic ones
            for file_path in batch:
                try:
                    # Apply timeout to individual file processing
                    with self._timeout(self.processing_timeout):
                        # Try to process this file individually
                        conv_results = self.converter.convert_all(
                            [file_path],
                            raises_on_error=False,
                        )
                        
                        # Export results to markdown files
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
                    
                    # Copy timeout file to timeout directory if provided
                    if timeout_dir:
                        try:
                            copy2(file_path, timeout_dir / filename)
                            self._log.info(f"Copied timeout file to {timeout_dir / filename}")
                        except Exception as e:
                            self._log.error(f"Failed to copy timeout file {filename}: {e}")
                        
                except Exception as individual_error:
                    problematic.append(Path(file_path).name)
                    self._log.error(f"Failed to process file {Path(file_path).name}: {individual_error}")
        
        return successful, problematic
        
    def extract_path(self, input_doc_paths, output_dir, batch_size: int = 5):
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
        
        # Create directories for problematic files and timeout files
        problematic_dir = output_dir / "problematic_files"
        problematic_dir.mkdir(exist_ok=True)
        
        # Create a separate directory specifically for timeout files
        timeout_dir = output_dir / "timeout_files"
        timeout_dir.mkdir(exist_ok=True)
        
        # State file for tracking progress
        state_file = output_dir / ".processing_state.parquet"

        
        # Load the current processing state
        state = self._load_processing_state(state_file)
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
                    
                partial_success_count += 1
            else:
                self._log.info(f"Document {conv_res.input.file} failed to extract.")
                failure_count += 1
                
        return success_count, partial_success_count, failure_count