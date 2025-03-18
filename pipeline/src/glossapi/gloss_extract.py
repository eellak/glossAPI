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
