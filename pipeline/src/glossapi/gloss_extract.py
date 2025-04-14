from typing import Dict, Set, List, Optional, Iterable, Tuple, Any

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    TesseractOcrOptions,
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
import time
import re
from pathlib import Path
from typing import Iterable, List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import shutil
from collections import defaultdict
import json
import joblib

class GlossExtract:
    """
    A class for extracting content from PDF documents to Markdown using Docling, and for
    clustering documents based on their quality (good vs. bad extractions).
    """
    
    def __init__(self):
        """Initialize the GlossExtract class with default settings."""
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        self.USE_V2 = True
        self.log_file = Path('.') / 'conversion.log'  
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
        
        # Define good and bad trigrams for quality assessment
        self.good_trigrams = set([
            "και", "του", "τικ", "ται", "της", "την", "ική", "των", "ετα", "ματ", 
            "στο", "που", "δια", "στη", "ικό", "προ", "ίνα", "σης", "από", "παρ", 
            "ναι", "ντα", "είν", "οπο", "ους", "ηση", "περ", "ιστ", "μέν", "ερι", 
            "οντ", "ατα", "σει", "κατ", "για", "συν", "μετ", "τον", "τητ", "ικά", 
            "επι", "ικο", "τερ", "ουν", "αντ", "αρα", "εις", "ότη", "αυτ", "απο"
        ])
        
        # Define bad trigrams with pattern information
        # For trigrams with numbers, we'll use regex patterns when matching
        self.bad_trigrams_raw = [
            "mag", "ima", "age", "/un", "PH<", "YPH", "LYP", "GLY", "θαη", "ni0", 
            "i03", "ηνπ", "εηα", "ηαη", "ηεο", "36>", "uni", "H<1", "3BC", "03B", 
            "ηελ", "H<2", "ηηθ", "���", "ζηε", "………", "/g5", "i1F", "ni1", "1F7", 
            "ηζη", "the", "ζην", "ηαζ", "πνπ", "ησλ", "καη", "ζεη", "δηα", "<23", 
            "πξν", "<13", "και", "λαη", "ζεο", "νπν", "ηεξ", "/g3", "αηα", "του"
        ]
        
        # Process bad trigrams to create regex patterns for those with numbers
        self.bad_trigrams = set()
        self.bad_trigram_patterns = []
        
        for trigram in self.bad_trigrams_raw:
            if any(char.isdigit() for char in trigram):
                # Replace digits with \d in regex pattern
                pattern = ''.join([r'\d' if char.isdigit() else re.escape(char) for char in trigram])
                self.bad_trigram_patterns.append(re.compile(f'^{pattern}$'))
            else:
                self.bad_trigrams.add(trigram)
    
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
        else:
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
    
    def _process_batch(self, batch: List[Path], output_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Process a batch of files and return the successful and problematic filenames.
        
        Args:
            batch: List of file paths to process
            output_dir: Output directory
            
        Returns:
            Tuple of (successful_filenames, problematic_filenames)
        """
        successful = []
        problematic = []
        
        # Try processing as a batch first
        try:
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
        
        # Create a directory for problematic files
        problematic_dir = output_dir / "problematic_files"
        problematic_dir.mkdir(exist_ok=True)
        
        # State file for tracking progress
        state_file = output_dir / ".processing_state.pkl"
        
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
            successful, problematic = self._process_batch(batch, output_dir)
            
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
    
    def _clean_text(self, text):
        """Remove sequences of dots, dashes, pipes, /gX patterns, and underscores"""
        # Remove sequences of dots, dashes, and pipes with or without spaces
        text = re.sub(r'[\s]*\.{2,}[\s]*', ' ', text)  # Remove ...
        text = re.sub(r'[\s]*\|[\s]*', ' ', text)      # Remove |
        text = re.sub(r'[\s]*-{2,}[\s]*', ' ', text)   # Remove --
        
        # Remove sequences of underscores (with optional backslashes)
        text = re.sub(r'[\\]*_+[\\]*', ' ', text)
        
        # Clean up extra spaces that might be left
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _preprocess_text(self, text):
        """Preprocess text by cleaning special characters and markdown formatting."""
        # First clean special characters
        text = self._clean_text(text)
        # Remove image tags
        text = re.sub(r'<!--\s*image\s*-->', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    def _process_file(self, args):
        """Process a single file for parallel execution"""
        filepath, folder = args
        try:
            with open(filepath, 'r', encoding='utf-8') as infile:
                text = infile.read()
            text = self._preprocess_text(text)
            return (filepath, text, folder)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def _get_all_files(self, input_folders):
        """Get list of all files to process"""
        all_files = []
        for folder in input_folders:
            for filename in os.listdir(folder):
                if filename.endswith('.md'):
                    filepath = os.path.join(folder, filename)
                    all_files.append((filepath, folder))
        return all_files
    
    def _custom_tokenizer(self, text):
        """Custom tokenizer to exclude trigrams with only dots, dashes, pipes, or any spaces"""
        trigrams = []
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            # Skip trigrams that contain spaces or contain only dots, dashes, pipes
            if ' ' not in trigram and not re.match(r'^[\.\-\|]+$', trigram):
                trigrams.append(trigram)
        return trigrams
    
    def _is_bad_trigram(self, trigram):
        """Check if a trigram matches any bad trigram or pattern."""
        # Check if it's in the exact match set
        if trigram in self.bad_trigrams:
            return True
        
        # Check if it matches any of the regex patterns
        for pattern in self.bad_trigram_patterns:
            if pattern.match(trigram):
                return True
        
        return False
    
    def _determine_cluster_quality(self, clusters_top_trigrams, good_trigrams, bad_trigrams):
        """Determine which cluster is good and which is bad based on trigram frequency."""
        cluster_scores = {}
        
        for cluster_idx, top_trigrams in clusters_top_trigrams.items():
            good_matches = 0
            bad_matches = 0
            
            # Count matches, handling bad trigram patterns
            for trigram in top_trigrams:
                if trigram in good_trigrams:
                    good_matches += 1
                
                if self._is_bad_trigram(trigram):
                    bad_matches += 1
            
            # Calculate weighted score: weight matches higher in the list more strongly
            weighted_good = 0
            weighted_bad = 0
            
            for i, trigram in enumerate(top_trigrams):
                weight = 1.0 - (i / len(top_trigrams))  # Higher weight for higher ranked trigrams
                if trigram in good_trigrams:
                    weighted_good += weight
                if self._is_bad_trigram(trigram):
                    weighted_bad += weight
            
            # Calculate combined score: positive means more good trigrams, negative means more bad trigrams
            weighted_score = weighted_good - weighted_bad
            raw_score = good_matches - bad_matches
            
            cluster_scores[cluster_idx] = {
                'raw_score': raw_score,
                'weighted_score': weighted_score,
                'good_matches': good_matches,
                'bad_matches': bad_matches,
                'total_matches': good_matches + bad_matches
            }
        
        # Determine which cluster is good and which is bad based on weighted score
        sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        
        # The cluster with the highest score is considered 'good'
        good_cluster = int(sorted_clusters[0][0].split('_')[1])
        bad_cluster = int(sorted_clusters[1][0].split('_')[1])
        
        print(f"\nCluster quality assessment:")
        for cluster_idx, stats in sorted_clusters:
            cluster_type = "GOOD" if int(cluster_idx.split('_')[1]) == good_cluster else "BAD"
            print(f"Cluster {cluster_idx}: {cluster_type}")
            print(f"  Good trigram matches: {stats['good_matches']}")
            print(f"  Bad trigram matches: {stats['bad_matches']}")
            print(f"  Raw score: {stats['raw_score']}")
            print(f"  Weighted score: {stats['weighted_score']:.2f}")
        
        return good_cluster, bad_cluster

    def training(self, input_folder,model_path='kmeans_weights.joblib'):
        """
        Processes all Markdown files in input_folder:
          - Computes a trigram representation and clusters them.
          - Creates 'good' and 'bad' subdirectories in output_folder.
          - Copies files to these folders according to cluster labels.
        """
        print("Starting document analysis...")

        # Get all files from the input folder
        all_files = self._get_all_files([input_folder])
        print(f"Found {len(all_files)} files to process")

        # Process files sequentially to avoid pickling issues
        print("Processing files sequentially...")
        documents = []
        filenames = []
        folder_sources = []
        
        for file_info in tqdm(all_files, desc="Processing files"):
            result = self._process_file(file_info)
            if result is not None:
                filepath, text, folder = result
                documents.append(text)
                filenames.append(os.path.basename(filepath))
                folder_sources.append(folder)

        print(f"\nSuccessfully processed {len(documents)} documents")

        print("\nCreating trigram representation...")
        vectorizer = TfidfVectorizer(
            analyzer=self._custom_tokenizer,
            lowercase=False,
            max_features=10000
        )
        X = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing documents", unit="doc"))
        print(f"Extracted trigram feature matrix of shape: {X.shape}")

        print("\nPerforming clustering...")
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = np.array(list(tqdm(
            kmeans.fit_predict(X),
            desc="Clustering documents",
            total=len(documents),
            unit="doc"
        )))
        
        # Analyze top trigrams per cluster for quality assessment
        print("\nAnalyzing top trigrams per cluster...")
        feature_names = vectorizer.get_feature_names_out()
        clusters_top_trigrams = {}
        for cluster_idx in range(n_clusters):
            centroid = kmeans.cluster_centers_[cluster_idx]
            top_indices = np.argsort(centroid)[::-1][:50]
            top_trigrams = [feature_names[i] for i in top_indices]
            clusters_top_trigrams[f"cluster_{cluster_idx}"] = top_trigrams
            print(f"\nCluster {cluster_idx} top 15 trigrams:")
            print(", ".join(top_trigrams[:15]))
        
        # Determine which cluster is good and which is bad using cluster-level analysis
        good_cluster, bad_cluster = self._determine_cluster_quality(
            clusters_top_trigrams, self.good_trigrams, self.bad_trigrams
        )
        
        # If the bad cluster is labeled as 0 and good as 1, we can keep the model as is
        # Otherwise we need to invert the labels to maintain the convention that 0=bad, 1=good
        if bad_cluster == 0 and good_cluster == 1:
            print("\nModel cluster labels align with convention (0=bad, 1=good)")
        else:
            print("\nInverting cluster labels to maintain convention (0=bad, 1=good)")
            # Invert the labels
            labels = 1 - labels
            # Also need to swap the cluster centers
            kmeans.cluster_centers_ = kmeans.cluster_centers_[::-1]
        
        # Save the model with standardized labels (0=bad, 1=good)
        joblib.dump(kmeans, model_path)
        print(f"\nSaved model to {model_path} with standardized cluster labels (0=bad, 1=good)")
    
    def split_bad(self, input_folder, output_folder, model_file='kmeans_weights.joblib'):
        """
        Processes all Markdown files in input_folder:
          - Computes a trigram representation and clusters them.
          - Creates 'good' and 'bad' subdirectories in output_folder.
          - Copies files to these folders according to cluster labels.
        """
        print("Starting document analysis...")

        # Get all files from the input folder
        all_files = self._get_all_files([input_folder])
        print(f"Found {len(all_files)} files to process")

        # Process files sequentially to avoid pickling issues
        print("Processing files sequentially...")
        documents = []
        filenames = []
        folder_sources = []
        
        for file_info in tqdm(all_files, desc="Processing files"):
            result = self._process_file(file_info)
            if result is not None:
                filepath, text, folder = result
                documents.append(text)
                filenames.append(os.path.basename(filepath))
                folder_sources.append(folder)

        print(f"\nSuccessfully processed {len(documents)} documents")

        print("\nCreating trigram representation...")
        vectorizer = TfidfVectorizer(
            analyzer=self._custom_tokenizer,
            lowercase=False,
            max_features=10000
        )
        X = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing documents", unit="doc"))
        print(f"Extracted trigram feature matrix of shape: {X.shape}")

        print("\nPerforming clustering using pre-trained model...")
        n_clusters = 2
        kmeans = joblib.load(model_file)
        
        # Handle feature dimension mismatch
        n_features = X.shape[1]
        expected_features = kmeans.cluster_centers_.shape[1]
        
        # Check if we have fewer features than the model expects (typically 10,000)
        if n_features < expected_features:
            print(f"\nWARNING: Insufficient trigram features detected. The clustering model expects {expected_features} features but only {n_features} were found.")
            print(f"This usually happens with very small datasets or short documents.")
            print(f"Skipping clustering and treating all documents as 'good' quality for testing purposes.")
            
            # Skip clustering entirely - create good and bad folders
            good_dir = os.path.join(output_folder, 'good')
            os.makedirs(good_dir, exist_ok=True)
            
            # Copy all files to good folder
            copied_count = 0
            # Reconstruct source file paths from folder_sources and filenames
            for i, filename in enumerate(filenames):
                source_file = os.path.join(input_folder, filename)
                dest_path = os.path.join(good_dir, filename)
                try:
                    shutil.copy2(source_file, dest_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
            
            print(f"\nCopied all {copied_count} files to 'good' folder: {good_dir}")
            return
            
        elif n_features != expected_features:
            print(f"Warning: Feature dimension mismatch. Model expects {expected_features} features but data has {n_features} features.")
            print("Using simple heuristic-based quality detection instead of KMeans clustering.")
            
            # Simple heuristic: use document length as a proxy for quality
            # Short documents are more likely to be bad quality
            doc_lengths = [len(doc) for doc in documents]
            median_length = np.median(doc_lengths)
            # Mark documents shorter than 20% of median length as 'bad quality'
            threshold = median_length * 0.2
            labels = np.array([1 if length >= threshold else 0 for length in doc_lengths])
        else:
            # Original code path if dimensions match
            labels = np.array(list(tqdm(
                kmeans.predict(X),
                desc="Clustering documents",
                total=len(documents),
                unit="doc"
            )))

        print("\nCalculating Silhouette Score...")
        try:
            score = silhouette_score(X, labels)
            print(f"Silhouette Score for {n_clusters} clusters: {score:.3f}")
        except ValueError as e:
            print(f"Warning: Could not calculate Silhouette Score: {e}")
            print("Continuing with classification...")
        
        print("\nAnalyzing top trigrams per cluster...")
        feature_names = vectorizer.get_feature_names_out()
        clusters_top_trigrams = {}
        for cluster_idx in range(n_clusters):
            centroid = kmeans.cluster_centers_[cluster_idx]
            top_indices = np.argsort(centroid)[::-1][:50]
            top_trigrams = [feature_names[i] for i in top_indices]
            clusters_top_trigrams[f"cluster_{cluster_idx}"] = top_trigrams
            print(f"\nCluster {cluster_idx} top 15 trigrams:")
            print(", ".join(top_trigrams[:15]))
        
        print("\nUsing pre-trained model conventions: Cluster 0 = Bad, Cluster 1 = Good")
        # The model is already trained to use 0 as bad and 1 as good
        bad_cluster = 0
        good_cluster = 1
        
        # Save top trigrams JSON in the output_folder
        os.makedirs(output_folder, exist_ok=True)
        json_path = os.path.join(output_folder, 'top_trigrams.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters_top_trigrams, f, ensure_ascii=False, indent=2)
        print(f"\nSaved top 50 trigrams for each cluster to {json_path}")

        # Create output subdirectories for good and bad files
        good_dir = os.path.join(output_folder, 'good')
        bad_dir = os.path.join(output_folder, 'bad')
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)

        # Copy files to appropriate directories based on cluster label and trigram voting
        print("\nCopying files to good/bad directories using cluster labels and trigram voting...")
        copied_count = {'good': 0, 'bad': 0}
        
        # Create vectorizer for extracting trigrams from individual documents
        doc_vectorizer = TfidfVectorizer(
            analyzer=self._custom_tokenizer,
            lowercase=False,
            max_features=10000
        )
        
        # Get cluster centroids for voting
        good_centroid = kmeans.cluster_centers_[good_cluster]
        bad_centroid = kmeans.cluster_centers_[bad_cluster]
        
        # Process each document individually for voting
        for idx, (filename, label, source_folder, document) in enumerate(zip(filenames, labels, folder_sources, documents)):
            source_path = os.path.join(source_folder, filename)
            
            # Extract top 50 trigrams from this document
            try:
                # Fit vectorizer only on this document
                doc_vectors = doc_vectorizer.fit_transform([document])
                feature_names = doc_vectorizer.get_feature_names_out()
                
                # Get top trigrams for this document
                doc_vector = doc_vectors.toarray()[0]
                top_indices = np.argsort(doc_vector)[::-1][:50]  # Get top 50
                top_trigrams = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
                
                # Count matches with good and bad trigram lists
                good_count = sum(1 for trigram in top_trigrams if trigram in self.good_trigrams)
                bad_count = sum(1 for trigram in top_trigrams if self._is_bad_trigram(trigram))
                
                # Apply voting: if the document contains more bad trigrams than good ones,
                # regardless of cluster, label it as bad
                if bad_count > good_count:
                    dest_dir = bad_dir
                    copied_count['bad'] += 1
                # If it has more good trigrams, or equal counts but was assigned to good cluster
                elif good_count > bad_count or (good_count == bad_count and label == good_cluster):
                    dest_dir = good_dir
                    copied_count['good'] += 1
                # Default to the cluster assignment
                else:
                    if label == bad_cluster:
                        dest_dir = bad_dir
                        copied_count['bad'] += 1
                    else:
                        dest_dir = good_dir
                        copied_count['good'] += 1
                
            except Exception as e:
                # If there's an error in trigram extraction, fall back to cluster labeling
                print(f"Error in trigram voting for {filename}: {e}")
                if label == bad_cluster:
                    dest_dir = bad_dir
                    copied_count['bad'] += 1
                else:
                    dest_dir = good_dir
                    copied_count['good'] += 1
            
            # Copy file to appropriate directory
            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"Error copying {filename}: {e}")
        
        print(f"\nFiles copied:")
        print(f"Good files: {copied_count['good']}")
        print(f"Bad files: {copied_count['bad']}")
        print("\nAnalysis complete! Check the classified files in the output folder.")
        
    def annotate_parquet_with_extraction_quality(self, markdown_folder, input_dir, model_file='kmeans_weights.joblib'):
        """
        Processes all Markdown files in markdown_folder and adds an 'extraction' column to the
        input parquet file with values 'good' or 'bad' based on quality assessment.
        
        Unlike split_bad(), this function doesn't copy files but updates the input parquet directly.
        It looks for a parquet file in input_dir that has a 'filename' column and adds or updates
        the 'extraction' column for rows where filenames match markdown files.
        
        Args:
            markdown_folder: Path to directory containing extracted markdown files
            input_dir: Directory containing the input parquet file (used by downloader)
            model_file: Path to the pre-trained model for clustering
        
        Returns:
            bool: True if successful, False otherwise
        """
        from pathlib import Path
        import pandas as pd
        from glossapi.parquet_schema import ParquetSchema
        
        print("Starting extraction quality assessment for parquet annotation...")
        
        # Convert to Path objects for consistency
        markdown_folder = Path(markdown_folder)
        input_dir = Path(input_dir)
        
        # Step 1: Find input parquet file
        print("Looking for input parquet file...")
        # Initialize with proper URL column configuration
        parquet_schema = ParquetSchema({
            'url_column': getattr(self, 'url_column', 'preferred_url')  # Use the class url_column if exists or default to preferred_url
        })
        print(f"Using URL column: {parquet_schema.url_column}")
        input_parquet_path = parquet_schema.find_metadata_parquet(input_dir)
        
        # If not found in input_dir, check download_results folder
        if input_parquet_path is None:
            download_results_dir = input_dir / "download_results"
            if download_results_dir.exists():
                input_parquet_path = parquet_schema.find_metadata_parquet(download_results_dir)
        
        if input_parquet_path is None:
            print("Error: Could not find a valid input parquet file with filename column")
            return False
        
        print(f"Found input parquet file: {input_parquet_path}")
        
        # Step 2: Read the input parquet file
        try:
            df = pd.read_parquet(input_parquet_path)
            if 'filename' not in df.columns:
                print("Error: Input parquet does not have 'filename' column")
                return False
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            return False
        
        # Step 3: Process markdown files for quality assessment
        # Use the same quality determination logic as in split_bad
        all_files = self._get_all_files([markdown_folder])
        print(f"Found {len(all_files)} markdown files to analyze")
        
        documents = []
        md_filenames = []
        folder_sources = []
        
        for file_info in tqdm(all_files, desc="Processing markdown files"):
            result = self._process_file(file_info)
            if result is not None:
                filepath, text, folder = result
                documents.append(text)
                md_filenames.append(os.path.basename(filepath))
                folder_sources.append(folder)
        
        print(f"\nSuccessfully processed {len(documents)} documents")
        
        # If no documents found, return early
        if not documents:
            print("No markdown files found. Cannot determine extraction quality.")
            return False
            
        # Step 4: Run quality assessment using the same technique as split_bad
        # Create trigram representation
        print("\nCreating trigram representation...")
        vectorizer = TfidfVectorizer(
            analyzer=self._custom_tokenizer,
            lowercase=False,
            max_features=10000
        )
        X = vectorizer.fit_transform(tqdm(documents, desc="Vectorizing documents", unit="doc"))
        
        # Perform clustering
        print("\nAssessing document quality using pre-trained model...")
        n_clusters = 2
        kmeans = joblib.load(model_file)
        
        # Handle feature dimension mismatch
        n_features = X.shape[1]
        expected_features = kmeans.cluster_centers_.shape[1]
        
        # Determine good and bad documents
        if n_features < expected_features:
            print(f"\nWARNING: Insufficient trigram features detected. Treating all documents as 'good' quality.")
            # Default to all good
            extraction_quality = {filename: "good" for filename in md_filenames}
        elif n_features != expected_features:
            print(f"Warning: Feature dimension mismatch. Using heuristic-based quality detection.")
            # Simple heuristic: use document length as a proxy for quality
            doc_lengths = [len(doc) for doc in documents]
            median_length = np.median(doc_lengths)
            threshold = median_length * 0.2
            extraction_quality = {}
            for i, filename in enumerate(md_filenames):
                if doc_lengths[i] >= threshold:
                    extraction_quality[filename] = "good"
                else:
                    extraction_quality[filename] = "bad"
        else:
            # Original code path if dimensions match
            labels = kmeans.predict(X)
            bad_cluster = 0  # Based on model convention
            good_cluster = 1
            
            # Create a vectorizer for individual document trigram extraction
            doc_vectorizer = TfidfVectorizer(
                analyzer=self._custom_tokenizer,
                lowercase=False,
                max_features=10000
            )
            
            # Process each document individually
            extraction_quality = {}
            for idx, (filename, label, document) in enumerate(zip(md_filenames, labels, documents)):
                try:
                    # Extract top trigrams from this document
                    doc_vectors = doc_vectorizer.fit_transform([document])
                    feature_names = doc_vectorizer.get_feature_names_out()
                    
                    doc_vector = doc_vectors.toarray()[0]
                    top_indices = np.argsort(doc_vector)[::-1][:50]
                    top_trigrams = [feature_names[i] for i in top_indices if doc_vector[i] > 0]
                    
                    # Count matches with good and bad trigram lists
                    good_count = sum(1 for trigram in top_trigrams if trigram in self.good_trigrams)
                    bad_count = sum(1 for trigram in top_trigrams if self._is_bad_trigram(trigram))
                    
                    # Determine quality based on trigram voting and cluster label
                    if bad_count > good_count:
                        extraction_quality[filename] = "bad"
                    elif good_count > bad_count or (good_count == bad_count and label == good_cluster):
                        extraction_quality[filename] = "good"
                    else:
                        extraction_quality[filename] = "bad" if label == bad_cluster else "good"
                except Exception as e:
                    print(f"Error in trigram voting for {filename}: {e}")
                    # Fall back to cluster label if error
                    extraction_quality[filename] = "bad" if label == bad_cluster else "good"
        
        # Count good and bad files
        good_count = sum(1 for quality in extraction_quality.values() if quality == "good")
        bad_count = sum(1 for quality in extraction_quality.values() if quality == "bad")
        print(f"\nQuality assessment complete:")
        print(f"Good files: {good_count}")
        print(f"Bad files: {bad_count}")
        
        # Step 5: Add extraction column to parquet
        print("\nUpdating parquet file with extraction quality...")
        
        # Add extraction column if it doesn't exist
        if 'extraction' not in df.columns:
            df['extraction'] = "unknown"
        
        # Map markdown filenames to parquet rows
        match_count = 0
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Matching files"):
            if 'filename' in row and row['filename']:
                # Remove file extension for comparison
                parquet_filename = row['filename']
                # Strip extension if present
                parquet_basename = os.path.splitext(parquet_filename)[0]
                
                # Try to find a matching markdown file
                for md_filename in md_filenames:
                    md_basename = os.path.splitext(md_filename)[0]
                    
                    # Check if filenames match after removing extensions
                    if md_basename == parquet_basename:
                        df.at[i, 'extraction'] = extraction_quality[md_filename]
                        match_count += 1
                        break
        
        print(f"Found and updated {match_count} matching files in parquet")
        
        # Step 6: Save updated parquet
        if match_count > 0:
            try:
                # Add processing_stage metadata
                df['processing_stage'] = df.get('processing_stage', 'download') + ",extract"
                
                # Save the updated parquet
                df.to_parquet(input_parquet_path, index=False)
                print(f"Successfully updated parquet file at {input_parquet_path}")
                return True
            except Exception as e:
                print(f"Error saving updated parquet file: {e}")
                return False
        else:
            print("No matches found between markdown files and parquet entries. Parquet not updated.")
            return False
