from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
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
from docling.document_converter import DocumentConverter, PdfFormatOption

import ftfy
import logging
import os
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
from sklearn.decomposition import PCA
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
        """Create a document converter with the configured options."""
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )
    
    def extract_path(self, input_doc_paths, output_dir):
        """
        Convert all PDF documents in the input paths to Markdown.
        
        Args:
            input_doc_paths (List[Path]): List of paths to PDF documents
            output_dir (Path): Directory to save the converted Markdown files
        """
        start_time = time.time()

        conv_results = self.converter.convert_all(
            input_doc_paths,
            raises_on_error=False,  # to let conversion run through all and examine results at the end
        )
        success_count, partial_success_count, failure_count = self._export_documents(
            conv_results, output_dir=output_dir
        )
        end_time = time.time() - start_time

        self._log.info(f"Document conversion complete in {end_time:.2f} seconds.")

        if failure_count > 0:
            self._log.warning(f"Failed to convert {failure_count} out of {len(input_doc_paths)} documents.")
            # Don't raise an exception, just continue with the successfully converted files
            
    def _fix_greek_text(self, text):
        """Fix Unicode issues in text, particularly for Greek characters."""
        return ftfy.fix_text(text)

    def _export_documents(self, conv_results: Iterable[ConversionResult], output_dir: Path):
        """
        Export converted documents to Markdown files.
        
        Args:
            conv_results: Iterable of conversion results
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

            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                self._log.info(
                    f"Document {conv_res.input.file} was partially converted with the following errors:"
                )
                for item in conv_res.errors:
                    self._log.info(f"\t{item.error_message}")
                partial_success_count += 1
            else:
                self._log.info(f"Document {conv_res.input.file} failed to convert.")
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
        joblib.dump(kmeans,model_path)
    
    def split_bad(self, input_folder, output_folder,model_file='kmeans_weights.joblib'):
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
        kmeans = joblib.load(model_file)
        labels = np.array(list(tqdm(
            kmeans.fit_predict(X),
            desc="Clustering documents",
            total=len(documents),
            unit="doc"
        )))

        print("\nCalculating Silhouette Score...")
        score = silhouette_score(X, labels)
        print(f"Silhouette Score for {n_clusters} clusters: {score:.3f}")

        print("\nCreating visualization...")
        sample_size = min(5000, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        pca = PCA(n_components=2)
        X_dense = X.toarray()
        X_sampled = X_dense[sample_indices]
        labels_sampled = labels[sample_indices]
        X_2d = pca.fit_transform(X_sampled)
        # (Visualization plotting code can be added here if needed)

        print("\nAnalyzing top trigrams per cluster...")
        feature_names = vectorizer.get_feature_names_out()
        clusters_top_trigrams = {}
        for cluster_idx in range(n_clusters):
            centroid = kmeans.cluster_centers_[cluster_idx]
            top_indices = np.argsort(centroid)[::-1][:50]
            top_trigrams = [feature_names[i] for i in top_indices]
            # Explicitly name cluster 0 as bad and cluster 1 as good
            if cluster_idx == 0:
                cluster_name = "bad"
            else:
                cluster_name = "good"
            clusters_top_trigrams[cluster_name] = top_trigrams
            print(f"\nCluster {cluster_idx} ({cluster_name}) top 15 trigrams:")
            print(", ".join(top_trigrams[:15]))

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

        # Copy files to appropriate directories based on cluster label:
        # Always treat cluster 0 as "bad" and cluster 1 as "good"
        print("\nCopying files to good/bad directories...")
        copied_count = {'good': 0, 'bad': 0}
        for filename, label, source_folder in zip(filenames, labels, folder_sources):
            source_path = os.path.join(source_folder, filename)
            # Explicitly define cluster 0 as bad and cluster 1 as good
            if label == 0:
                dest_dir = bad_dir
                copied_count['bad'] += 1
            else:
                dest_dir = good_dir
                copied_count['good'] += 1

            dest_path = os.path.join(dest_dir, filename)
            try:
                shutil.copy2(source_path, dest_path)
            except Exception as e:
                print(f"Error copying {filename}: {e}")

        print(f"\nFiles copied:")
        print(f"Good files: {copied_count['good']}")
        print(f"Bad files: {copied_count['bad']}")
        print("\nAnalysis complete! Check the visualization and classified files in the output folder.")
