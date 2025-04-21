import logging
from pathlib import Path
import os
import pandas as pd
import random
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
        
        # Create necessary directories
        self.markdown_dir = self.output_dir / "markdown"
        self.sections_dir = self.output_dir / "sections"
        # Define models_dir path but don't create the directory yet - only create it when needed
        self.models_dir = self.output_dir / "models"
        
        os.makedirs(self.markdown_dir, exist_ok=True)
        os.makedirs(self.sections_dir, exist_ok=True)
        
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
    
    def filter(self, input_dir: Union[str, Path] = None, split_bad: bool = True, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Filter markdown files based on quality to separate good from bad quality.
        
        Args:
            input_dir: Directory containing markdown files to filter (defaults to self.markdown_dir)
            split_bad: Whether to separate files into good/bad quality or keep all as good
            model_path: Path to the pre-trained model for clustering (defaults to self.extraction_model_path)
        """
        # Handle default parameters
        if input_dir is None:
            input_dir = self.markdown_dir
        else:
            input_dir = Path(input_dir)
            
        # Skip quality assessment if not splitting bad files
        if not split_bad:
            self.logger.info("Skipping quality clustering as split_bad=False")
            # Just use the original markdown directory for good files
            self.good_markdown_dir = input_dir
            self.markdown_dir = input_dir
            self.logger.info(f"Using all files from {input_dir} as good quality")
            return
        
        # First run the parquet-based quality assessment
        if model_path is None:
            model_path = str(self.extraction_model_path)
            
        # We need the parquet file first before we can update it with extraction quality
        import pandas as pd
        from glossapi.parquet_schema import ParquetSchema
        
        # First, find or create the parquet file
        download_results_dir = self.input_dir / "download_results"
        parquet_schema = ParquetSchema({
            'url_column': self.downloader_config.get('url_column', 'url')
        })
        
        # Find the parquet file
        input_parquet_path = None
        for dir_to_check in [self.input_dir, download_results_dir]:
            if dir_to_check.exists():
                # In filter stage, we don't need URL column, just filename
                found_path = parquet_schema.find_metadata_parquet(dir_to_check, require_url_column=False)
                if found_path:
                    input_parquet_path = found_path
                    break
        
        # If no parquet file was found, create a new one based on markdown files
        if input_parquet_path is None:
            self.logger.info(f"No metadata parquet file found, creating one from markdown files in {input_dir}")
            input_parquet_path = parquet_schema.create_basic_metadata_parquet(input_dir, self.output_dir)
            
            if input_parquet_path is None:
                self.logger.warning("Failed to create metadata parquet file")
            else:
                self.logger.info(f"Created new metadata parquet file at {input_parquet_path}")
        
        # Now use the modern approach - update parquet file with extraction quality
        self.logger.info("Analyzing markdown files and updating parquet file with extraction quality...")
        # Call extractor.split_bad which updates the parquet file with extraction quality
        if input_parquet_path:
            self.extractor.split_bad(
                input_folder=str(input_dir),
                # No need to pass output_folder as it's no longer used
                model_file=model_path
            )
            self.logger.info("Extraction quality assessment complete and saved to parquet.")
        else:
            self.logger.warning("Skipping extraction quality assessment as no parquet file is available.")
        
        # Read extraction quality from parquet
        good_count = 0
        bad_count = 0
        good_files = []
        bad_files = []
        
        if input_parquet_path:
            try:
                df = pd.read_parquet(input_parquet_path)
                if 'filename' in df.columns and 'extraction' in df.columns:
                    self.logger.info(f"Found extraction quality information in {input_parquet_path}")
                    for _, row in df.iterrows():
                        if 'filename' in row and 'extraction' in row:
                            if row['extraction'] == 'good':
                                good_files.append(row['filename'])
                                good_count += 1
                            elif row['extraction'] == 'bad':
                                bad_files.append(row['filename'])
                                bad_count += 1
            except Exception as e:
                self.logger.warning(f"Error reading extraction quality from parquet: {e}")
        
        self.logger.info(f"Found {good_count} good quality files and {bad_count} bad quality files in parquet")
        
        # Set the good_markdown_dir to be the same as the input_dir
        # This ensures that section() will read from the original directory
        # but only process files marked as 'good' in the parquet
        self.good_markdown_dir = input_dir
        self.markdown_dir = input_dir
        
        # Store the good filenames for later use when needed
        self.good_files = [f for f in good_files]

    def extract(
        self, 
        input_format: str = "all", 
        num_threads: int = 4, 
        accel_type: str = "Auto",
        split_bad: bool = True,
        model_path: str = None
    ) -> None:
        """
        Extract input files to markdown format.
        
        Args:
            input_format: Input format ("pdf", "docx", "xml_jats", "html", "pptx", "csv", "md", "all") (default: "all")
                          Note: Old .doc format (pre-2007) is not supported
            num_threads: Number of threads for processing (default: 4)
            accel_type: Acceleration type ("Auto", "CPU", "CUDA", "MPS") (default: "Auto")
            split_bad: Whether to perform clustering to separate good and bad files (default: True)
            model_path: Path to the KMeans clustering model (default: None, will use default path)
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
        
        # Run filtering on extracted markdown files
        self.filter(input_dir=self.markdown_dir, split_bad=split_bad, model_path=model_path)
    
    def split_bad(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """
        Analyze markdown files for extraction quality and update the input parquet file.
        This adds an 'extraction' column to the parquet with values 'good' or 'bad'.
        
        Args:
            model_path: Path to the pre-trained model for clustering (defaults to self.extraction_model_path)
        """
        self.logger.info("Analyzing extraction quality and updating parquet file...")
        
        # Set extraction model path
        if model_path is None:
            model_path = str(self.extraction_model_path)
            
        # Simply delegate to the GlossExtract implementation
        self.extractor.split_bad(
            input_folder=str(self.markdown_dir),
            model_file=model_path
        )
        
        self.logger.info("Parquet file successfully updated with extraction quality information.")
    
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.
        
        Uses files marked with 'good' extraction quality (if available) or all markdown files.
        """
        self.logger.info("Extracting sections from markdown files...")
        
        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Filter markdown files based on extraction quality in parquet files
        # Initialize the good_filenames list that will be used with the sectioner
        good_filenames = []
        
        # Try to find files marked as 'good' in the parquet
        from glossapi.parquet_schema import ParquetSchema
        
        # Initialize with proper URL column configuration
        parquet_schema = ParquetSchema({
            'url_column': self.downloader_config.get('url_column', 'url')  # Use the configured URL column or default to 'url'
        })
        self.logger.info(f"Using URL column for parquet search: {parquet_schema.url_column}")
        
        # Look for input parquet with extraction column
        input_parquet_path = parquet_schema.find_metadata_parquet(self.input_dir)
        
        # If not in input_dir, check download_results folder
        if input_parquet_path is None:
            download_results_dir = self.input_dir / "download_results"
            if download_results_dir.exists():
                input_parquet_path = parquet_schema.find_metadata_parquet(download_results_dir)
            
        if input_parquet_path is not None:
            try:
                # Load parquet and filter by 'good' extraction
                df = pd.read_parquet(input_parquet_path)
                if 'filename' in df.columns and 'extraction' in df.columns:
                    good_rows = df[df['extraction'] == 'good']
                    if not good_rows.empty:
                        # Get filenames (without extension) of good extractions
                        good_filenames = [
                            os.path.splitext(filename)[0] 
                            for filename in good_rows['filename'].tolist() 
                            if filename
                        ]
                        self.logger.info(f"Found {len(good_filenames)} files marked as 'good' in parquet")
                        
                        # Update the processing_stage in the download results parquet
                        try:
                            # Update processing_stage for all good rows
                            if 'processing_stage' in df.columns:
                                # Only update rows where extraction is 'good'
                                for idx in good_rows.index:
                                    current_stage = df.loc[idx, 'processing_stage']
                                    # Append section to stages if not already there
                                    if current_stage is not None and 'section' not in str(current_stage):
                                        df.loc[idx, 'processing_stage'] = current_stage + ',section'
                            else:
                                # Create processing_stage column if it doesn't exist
                                df['processing_stage'] = None
                                for idx in good_rows.index:
                                    df.loc[idx, 'processing_stage'] = 'download,extract,section'
                            
                            standard_path = Path(os.path.dirname(input_parquet_path)) / "download_results.parquet"
                            
                            # If the file already has the standardized name, just update it
                            # Otherwise, save with standardized name and log the change
                            df.to_parquet(standard_path, index=False)
                            self.logger.info(f"Updated processing_stage column in {standard_path} for good quality files")
                            
                            # If we renamed the file, log this and remove the original
                            if standard_path != input_parquet_path:
                                self.logger.info(f"Standardized parquet name from {os.path.basename(input_parquet_path)} to download_results.parquet")
                                # Remove the original file to avoid duplication
                                try:
                                    os.remove(input_parquet_path)
                                    self.logger.info(f"Removed original parquet file: {input_parquet_path}")
                                except Exception as e:
                                    self.logger.warning(f"Failed to remove original parquet file: {e}")
                        except Exception as e:
                            self.logger.warning(f"Error updating processing_stage in download results parquet: {e}")
            except Exception as e:
                self.logger.warning(f"Error reading parquet for extraction quality: {e}")
        
        self.logger.info(f"Found {len(good_filenames)} good quality files for sectioning")
        if good_filenames:
            self.logger.info(f"Good filenames: {good_filenames}")
            
        if not good_filenames:
            error_msg = "No good quality files found for sectioning. Check extraction quality or run split_bad() first."
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
        
        # Use section classifier model path
        model_path = str(self.section_classifier_model_path) if model_exists else None
        
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