import logging
from pathlib import Path
import os
import pandas as pd
import random
from typing import Dict, Optional, Union, List

from .gloss_convert import GlossConvert
from .gloss_extraction import GlossExtraction
from .gloss_academic_classifier import GlossAcademicClassifier

class Corpus:
    """
    A high-level wrapper for the GlossAPI academic document processing pipeline.
    
    This class provides a unified interface to convert PDFs to markdown,
    extract sections, and classify them using machine learning.
    
    Example:
        corpus = Corpus(input_dir="path/to/pdfs", output_dir="path/to/output")
        corpus.convert()  # Convert PDFs to markdown
        corpus.section()  # Extract sections from markdown files
        corpus.annotate()  # Classify sections using ML
    """
    
    def __init__(
        self, 
        input_dir: Union[str, Path], 
        output_dir: Union[str, Path],
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Corpus processor.
        
        Args:
            input_dir: Directory containing input files (PDF or markdown)
            output_dir: Base directory for all outputs
            model_path: Path to the pre-trained classifier model (if None, will train a new model)
            metadata_path: Path to metadata file with document types (optional)
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
        
        # Handle model path - if not provided, use default path in package
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Use default model path in the package
            package_dir = Path(__file__).parent
            self.model_path = package_dir / "models" / "academic_classifier.joblib"
            
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize component classes
        self.converter = GlossConvert()
        self.extractor = GlossExtraction()
        self.classifier = GlossAcademicClassifier()
        
        # Create necessary directories
        self.markdown_dir = self.output_dir / "markdown"
        self.sections_dir = self.output_dir / "sections"
        
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
                    self.filename_to_doctype = dict(zip(
                        metadata_df['filename'], 
                        metadata_df['document_type']
                    ))
                    self.logger.info(f"Loaded document type information for {len(self.filename_to_doctype)} files")
                else:
                    self.logger.warning(f"Missing required columns in metadata. Available columns: {metadata_df.columns.tolist()}")
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
    
    def convert(
        self, 
        input_format: str = "pdf", 
        num_threads: int = 4, 
        accel_type: str = "Auto"
    ) -> None:
        """
        Convert input files to markdown format.
        
        Args:
            input_format: Input format (default: "pdf")
            num_threads: Number of threads for processing (default: 4)
            accel_type: Acceleration type ("Auto", "CPU", "CUDA", "MPS") (default: "Auto")
        """
        self.logger.info(f"Converting {input_format} files to markdown...")
        
        # Prepare converter
        self.converter.enable_accel(threads=num_threads, type=accel_type)
        self.converter.create_converter()
        
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
        
        self.logger.info(f"Found {len(input_files)} {input_format} files to convert")
        
        # Convert files
        self.converter.convert_path(input_files, self.markdown_dir)
        self.logger.info(f"Conversion complete. Markdown files saved to {self.markdown_dir}")
    
    def section(self) -> None:
        """
        Extract sections from markdown files and save to Parquet format.
        """
        self.logger.info("Extracting sections from markdown files...")
        
        # Create output directory
        os.makedirs(self.sections_dir, exist_ok=True)
        
        # Check if markdown directory exists
        input_dir = self.markdown_dir if self.markdown_dir.exists() else self.input_dir
        
        # Extract sections
        self.extractor.to_parquet(
            input_dir=str(input_dir),
            output_dir=str(self.sections_dir)
        )
        
        self.logger.info(f"Section extraction complete. Parquet file saved to {self.sections_parquet}")
    
    # Keep extract method for backward compatibility
    def extract(self) -> None:
        """
        Extract sections from markdown files (alias for section()).
        
        This method is kept for backward compatibility and will be deprecated in future versions.
        Please use section() instead.
        """
        self.logger.warning("extract() is deprecated and will be removed in a future version. Use section() instead.")
        self.section()
    
    def annotate(self, fully_annotate: bool = True, save_model: bool = True) -> None:
        """
        Classify sections using machine learning.
        
        Args:
            fully_annotate: Whether to perform full annotation after classification (default: True)
            save_model: Whether to save the trained model for future use (default: True)
        """
        self.logger.info("Classifying sections...")
        
        # Check if sections parquet exists
        if not self.sections_parquet.exists():
            self.logger.error(f"Sections parquet file not found: {self.sections_parquet}")
            return
        
        # Initialize the classifier
        self.classifier.build_pipeline()
        
        # Load pre-trained model if available
        if self.model_path.exists():
            self.logger.info(f"Loading pre-trained model from {self.model_path}")
            self.classifier.load_model(str(self.model_path))
            
            # Process with pre-trained model
            self.classifier._process_predictions_with_dask(
                input_parquet=str(self.sections_parquet),
                output_parquet=str(self.classified_parquet)
            )
        else:
            # Train a new model
            self.logger.info("Training a new classifier model")
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Train and save the model
            self.classifier.train_model(
                input_parquet=str(self.sections_parquet),
                output_parquet=str(self.classified_parquet),
                model_save_path=str(self.model_path) if save_model else None
            )
            
            if save_model:
                self.logger.info(f"Model saved to {self.model_path}")
        
        # Perform full annotation if requested
        if fully_annotate:
            self.logger.info("Performing full annotation...")
            self.classifier.fully_annotate(
                input_parquet=str(self.classified_parquet),
                output_parquet=str(self.fully_annotated_parquet)
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
            self.logger.info("No document type information available. Skipping document type annotation.")
            return
        
        self.logger.info(f"Adding document_type to classified sections in {parquet_file}")
        
        try:
            # Load the classified data
            classified_df = pd.read_parquet(str(parquet_file))
            
            # Create a new column for document_type
            classified_df['document_type'] = None
            
            # Update document_type based on filename
            for filename, group_df in classified_df.groupby('filename'):
                doc_type = self.filename_to_doctype.get(filename, None)
                if doc_type:
                    # Update document_type for all rows with this filename
                    classified_df.loc[classified_df['filename'] == filename, 'document_type'] = doc_type
            
            # Save the updated classified data
            classified_df.to_parquet(str(parquet_file))
            
            # Count document types for verification
            doc_type_counts = classified_df['document_type'].value_counts(dropna=False)
            self.logger.info("Document type distribution in classified sections:")
            for doc_type, count in doc_type_counts.items():
                if pd.notna(doc_type):
                    self.logger.info(f"  {doc_type}: {count} sections")
                else:
                    self.logger.info(f"  None/Unknown: {count} sections")
                    
        except Exception as e:
            self.logger.error(f"Error adding document types: {e}")
    
    def process_all(self, input_format: str = "pdf", fully_annotate: bool = True) -> None:
        """
        Run the complete processing pipeline: convert, extract, and annotate.
        
        Args:
            input_format: Input format (default: "pdf")
            fully_annotate: Whether to perform full annotation after classification (default: True)
        """
        self.convert(input_format=input_format)
        self.section()
        self.annotate(fully_annotate=fully_annotate)
        
        self.logger.info("Complete processing pipeline finished successfully.")
    
    def sample(self, size: int = 100, output_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Sample a specified number of unique filenames from the fully annotated parquet file.
        
        Args:
            size: Number of unique filenames to sample
            output_csv: Optional path to save the sampled data as CSV
            
        Returns:
            DataFrame containing all rows for the sampled filenames
        """
        if not self.fully_annotated_parquet.exists():
            self.logger.error(f"Fully annotated parquet file not found: {self.fully_annotated_parquet}")
            self.logger.info("Please run the annotate() method with fully_annotate=True first.")
            return pd.DataFrame()
        
        sampler = Sampler(self, self.fully_annotated_parquet)
        return sampler.sample(size=size, output_csv=output_csv)
    
    def to_text(self, input_data: Union[str, Path, pd.DataFrame], output_dir: Union[str, Path]) -> None:
        """
        Convert parquet or CSV data to formatted text files.
        
        Args:
            input_data: Path to parquet/CSV file or DataFrame containing the data
            output_dir: Directory to save the output text files
        """
        self.logger.info("Converting data to formatted text files...")
        
        # Create output directory
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data if input is a file path
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)
            if input_path.suffix.lower() == '.csv':
                df = pd.read_csv(input_path)
            elif input_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(input_path)
            else:
                self.logger.error(f"Unsupported file format: {input_path.suffix}")
                return
        else:
            # Assume input_data is a DataFrame
            df = input_data
        
        # Group by filename and sort by id
        self.logger.info("Grouping data by filename...")
        grouped = df.groupby('filename')
        
        # Process each unique filename
        for filename, group in grouped:
            # Sort by id to maintain the correct order of sections
            group = group.sort_values('id')
            
            # Create output file path
            output_file_path = output_dir / f"{filename}.txt"
            
            # Write formatted content
            with open(output_file_path, 'w', encoding='utf-8') as f:
                # Write filename at the top
                f.write(f"# Document: {filename}\n\n")
                
                for _, row in group.iterrows():
                    # Write row with formatting
                    section_type = row.get('predicted_section', '')
                    row_id = row.get('row_id', '')
                    header = row.get('header', '')
                    section = row.get('section', '')
                    
                    f.write(f"{{{row_id}, {section_type}}} {header}\n\n")
                    f.write(f"{section}\n\n")
            
            self.logger.info(f"Processed file: {output_file_path}")
        
        self.logger.info(f"Conversion complete. Text files saved to {output_dir}")

class Sampler:
    """
    A class for sampling unique filenames from a parquet file and extracting their rows.
    
    This class is used to create a representative sample of documents from the corpus
    for analysis or testing purposes.
    """
    
    def __init__(self, corpus, parquet_file: Union[str, Path]):
        """
        Initialize the Sampler.
        
        Args:
            corpus: The parent Corpus instance
            parquet_file: Path to the parquet file to sample from
        """
        self.corpus = corpus
        self.parquet_file = Path(parquet_file)
        self.logger = corpus.logger
    
    def sample(self, size: int = 100, output_csv: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Sample a specified number of unique filenames and extract all their rows.
        
        Args:
            size: Number of unique filenames to sample
            output_csv: Optional path to save the sampled data as CSV
            
        Returns:
            DataFrame containing all rows for the sampled filenames
        """
        if not self.parquet_file.exists():
            self.logger.error(f"Parquet file not found: {self.parquet_file}")
            return pd.DataFrame()
        
        self.logger.info(f"Sampling {size} unique filenames from {self.parquet_file}...")
        
        # Read the parquet file
        df = pd.read_parquet(self.parquet_file)
        
        # Get unique filenames
        unique_filenames = df['filename'].unique()
        
        if len(unique_filenames) <= size:
            self.logger.warning(f"Requested sample size ({size}) is greater than or equal to the number of unique filenames ({len(unique_filenames)}). Using all available filenames.")
            sampled_filenames = unique_filenames
        else:
            # Randomly sample unique filenames
            sampled_filenames = random.sample(list(unique_filenames), size)
        
        # Extract all rows for the sampled filenames
        sampled_df = df[df['filename'].isin(sampled_filenames)]
        
        self.logger.info(f"Sampled {len(sampled_filenames)} unique filenames with {len(sampled_df)} total rows")
        
        # Save to CSV if output path is provided
        if output_csv:
            output_path = Path(output_csv)
            os.makedirs(output_path.parent, exist_ok=True)
            sampled_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved sampled data to {output_path}")
        
        return sampled_df