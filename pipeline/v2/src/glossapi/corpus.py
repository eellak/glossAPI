import logging
from pathlib import Path
import os
import pandas as pd
import random
from typing import Dict, Optional, Union, List
import shutil

from .gloss_extract import GlossExtract
from .gloss_section import GlossSection
from .gloss_section_classifier import GlossSectionClassifier

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
        log_level: int = logging.INFO
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
            log_level: Logging level (default: logging.INFO)
        """
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
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
        
        # Initialize component classes
        self.extractor = GlossExtract()
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
            
        # Create quality clustering directory structure
        quality_dir = self.output_dir / 'quality_clustering'
        os.makedirs(quality_dir, exist_ok=True)
        
        good_dir = quality_dir / 'good'
        bad_dir = quality_dir / 'bad'
        os.makedirs(good_dir, exist_ok=True)
        
        if split_bad:
            # Only create bad dir if we're going to use it
            os.makedirs(bad_dir, exist_ok=True)
            
            # Set extraction model path
            if model_path is None:
                model_path = str(self.extraction_model_path)
            
            # Check if model exists
            if not os.path.exists(model_path):
                self.logger.warning(f"Clustering model not found at {model_path}. Training a new model...")
                # Create models directory only when needed for training a new model
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Train model
                self.extractor.training(str(input_dir), model_path=model_path)
                self.logger.info(f"Model trained and saved to {model_path}")
            
            # Run split_bad to separate good and bad files
            self.logger.info("Running clustering to separate good and bad quality documents...")
            self.extractor.split_bad(
                input_folder=str(input_dir),
                output_folder=str(quality_dir),
                model_file=model_path
            )
            self.logger.info(f"Clustering complete. Files sorted into good/bad folders.")
        else:
            # If split_bad is disabled, just copy all files to the good directory
            self.logger.info("Clustering disabled. Copying all files to 'good' folder...")
            
            # Get all markdown files
            markdown_files = list(Path(input_dir).glob("*.md"))
            
            # Copy all files to good folder
            copied_count = 0
            for source_path in markdown_files:
                filename = source_path.name
                dest_path = good_dir / filename
                try:
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                except Exception as e:
                    self.logger.error(f"Error copying {filename}: {e}")
            
            self.logger.info(f"Copied {copied_count} files to good folder")
        
        # Update markdown_dir to use good files for further processing
        self.good_markdown_dir = good_dir
        self.logger.info(f"Files processed. Good files saved to {self.good_markdown_dir}")
        
        # For subsequent operations, use the good files
        self.markdown_dir = self.good_markdown_dir

    def extract(
        self, 
        input_format: str = "pdf", 
        num_threads: int = 4, 
        accel_type: str = "Auto",
        split_bad: bool = True,
        model_path: str = None
    ) -> None:
        """
        Extract input files to markdown format.
        
        Args:
            input_format: Input format (default: "pdf")
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
        
        # Get input files
        if input_format.lower() == "pdf":
            input_files = list(self.input_dir.glob("*.pdf"))
        else:
            input_files = list(self.input_dir.glob(f"*.{input_format}"))
        
        if not input_files:
            self.logger.warning(f"No {input_format} files found in {self.input_dir}")
            return
        
        self.logger.info(f"Found {len(input_files)} {input_format} files to extract")
        
        # Process all PDF files
        self.logger.info(f"Processing {len(input_files)} {input_format.upper()} files...")
        
        # Extract PDFs to markdown
        os.makedirs(self.markdown_dir, exist_ok=True)
        
        # Use multiple threads for extraction
        self.extractor.extract_path(input_files, self.markdown_dir)
        
        self.logger.info(f"Extraction complete. Markdown files saved to {self.markdown_dir}")
        
        # Run filtering on extracted markdown files
        self.filter(input_dir=self.markdown_dir, split_bad=split_bad, model_path=model_path)
    
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.
        """
        self.logger.info("Extracting sections from markdown files...")
        
        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Use the good markdown directory if available, otherwise fall back to other options
        if hasattr(self, 'good_markdown_dir') and self.good_markdown_dir.exists():
            input_dir = self.good_markdown_dir
            self.logger.info(f"Using good quality markdown files from {input_dir}")
        else:
            # Check if markdown directory exists
            input_dir = self.markdown_dir if self.markdown_dir.exists() else self.input_dir
            self.logger.info(f"Using markdown files from {input_dir}")
        
        # Extract sections
        self.sectioner.to_parquet(
            input_dir=str(input_dir),
            output_dir=str(self.sections_dir)
        )
        
        self.logger.info(f"Section extraction complete. Parquet file saved to {self.sections_parquet}")
    
    # Keep extract method for backward compatibility
    def convert(self) -> None:
        """
        Extract sections from markdown files (alias for section()).
        
        This method is kept for backward compatibility and will be deprecated in future versions.
        Please use section() instead.
        """
        self.logger.warning("convert() is deprecated and will be removed in a future version. Use section() instead.")
        self.section()
    
    def annotate(self, annotation_type: str = "auto", fully_annotate: bool = True) -> None:
        """
        Annotate extracted sections with classification information.
        
        Args:
            annotation_type: Type of annotation to use: 'auto', 'text', or 'chapter'
                           - 'auto': Use document type from metadata to determine appropriate annotation
                           - 'text': Use text-based annotation with section titles
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
        else:
            # Add document types to the classified output
            self._add_document_types(self.classified_parquet)
    
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
    
    def process_all(self, input_format: str = "pdf", fully_annotate: bool = True, annotation_type: str = "auto") -> None:
        """
        Run the complete processing pipeline: extract, section, and annotate.
        
        Args:
            input_format: Input format (default: "pdf")
            fully_annotate: Whether to perform full annotation after classification (default: True)
            annotation_type: Annotation method to use (default: "auto")
        """
        self.extract(input_format=input_format)
        self.section()
        self.annotate(fully_annotate=fully_annotate, annotation_type=annotation_type)
        
        self.logger.info("Complete processing pipeline finished successfully.")
    