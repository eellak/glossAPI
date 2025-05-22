"""
Standardized Parquet Schema definitions for GlossAPI pipeline.

This module defines standard schemas for parquet files used throughout the GlossAPI
pipeline, ensuring consistency between different pipeline stages.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple


class ParquetSchema:
    """
    Defines standardized schema for parquet files in the GlossAPI pipeline.
    
    This class provides methods to validate, read, and write parquet files
    with consistent schemas for different pipeline stages.
    
    The pipeline uses two distinct types of parquet files:
    
    1. Metadata Parquet:
       - Each row represents a file (one-to-one relationship with files)
       - Essential columns: filename, URL column (configurable), extraction quality
       - Used by: downloader, extractor, and filter stages
       - Example: download_results.parquet
       - Typical location: {output_dir}/download_results/
       - Schema: METADATA_SCHEMA, DOWNLOAD_SCHEMA
    
    2. Sections Parquet:
       - Each row represents a section from a file (many-to-one relationship with files)
       - Essential columns: filename, title, content, section, predicted_section
       - Used by: section and annotation stages
       - Examples: sections_for_annotation.parquet, classified_sections.parquet
       - Typical location: {output_dir}/sections/
       - Schema: SECTION_SCHEMA, CLASSIFIED_SCHEMA
    
    When the pipeline runs, it first creates and populates a metadata parquet,
    then uses it to filter files, and finally creates section parquets from the
    filtered files.
    """
    
    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ParquetSchema with optional pipeline configuration.
        
        Args:
            pipeline_config: Configuration dictionary with settings such as
                url_column, which will be used throughout the pipeline
        """
        # TODO: Add more robust configuration options for each parquet type from input metadata and downloder, to section, and two phases of annotaiton.
        # TODO: Add support for consolidated sections parquet handling
        # TODO: Add methods to find the latest sections parquet in a pipeline
        self.config = pipeline_config or {}
        self.url_column = self.config.get('url_column', 'url')
    
    # Basic schema with common fields used across all parquet files
    COMMON_SCHEMA = pa.schema([
        ('id', pa.string()),
        ('row_id', pa.int64()),
        ('filename', pa.string()),
    ])
    
    # Metadata schema for files used by downloader and quality assessment
    METADATA_SCHEMA = pa.schema([
        ('filename', pa.string()),
        ('url', pa.string()),  # Can be customized with url_column parameter
        ('download_success', pa.bool_()),
        ('download_error', pa.string()),
        ('extraction_quality', pa.string()),  # Values: "good", "bad", "unknown"
        ('processing_stage', pa.string()),  # Tracks progress through pipeline
    ])
    
    # Additional schemas for specific pipeline stages
    DOWNLOAD_SCHEMA = pa.schema([
        ('url', pa.string()),  # Will be replaced with the actual url_column
        ('download_success', pa.bool_()),
        ('download_error', pa.string()),
        ('download_retry_count', pa.int32()),
        ('filename', pa.string()),
    ])
    
    SECTION_SCHEMA = pa.schema([
        ('id', pa.string()),
        ('row_id', pa.int64()),
        ('filename', pa.string()),
        ('title', pa.string()),
        ('content', pa.string()),
        ('section', pa.string()),
    ])
    
    CLASSIFIED_SCHEMA = pa.schema([
        ('id', pa.string()),
        ('row_id', pa.int64()),
        ('filename', pa.string()),
        ('title', pa.string()),
        ('content', pa.string()),
        ('section', pa.string()),
        ('predicted_section', pa.string()),
        ('probability', pa.float64()),
    ])
    
    def get_required_metadata(self) -> Dict[str, str]:
        """
        Get required metadata fields for GlossAPI parquet files.
        
        Returns:
            Dict[str, str]: Dictionary of required metadata fields and their descriptions
        """
        return {
            'pipeline_version': 'GlossAPI pipeline version',
            'created_at': 'ISO format timestamp when the file was created',
            'source_file': 'Original source file that generated this parquet',
            'processing_stage': 'Pipeline processing stage (download, extract, section, etc)'
        }
    
    def validate_schema(self, df: pd.DataFrame, schema_type: str = 'common') -> Tuple[bool, List[str]]:
        """
        Validate that a DataFrame conforms to the specified schema.
        
        Args:
            df: DataFrame to validate
            schema_type: Type of schema to validate against ('common', 'download', 'section', 'classified', 'metadata')
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, missing_columns)
        """
        if schema_type.lower() == 'download':
            required_columns = [field.name for field in self.DOWNLOAD_SCHEMA]
            # Make sure to use the configured url_column
            if self.url_column != 'url' and 'url' in required_columns:
                required_columns.remove('url')
                required_columns.append(self.url_column)
        elif schema_type.lower() == 'section':
            required_columns = [field.name for field in self.SECTION_SCHEMA]
        elif schema_type.lower() == 'classified':
            required_columns = [field.name for field in self.CLASSIFIED_SCHEMA]
        elif schema_type.lower() == 'metadata':
            required_columns = ['filename']
            # Make sure to use the configured url_column
            required_columns.append(self.url_column)  
        else:  # Default to common schema
            required_columns = [field.name for field in self.COMMON_SCHEMA]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        return len(missing_columns) == 0, missing_columns
    
    def add_metadata(self, table: pa.Table, metadata: Dict[str, str]) -> pa.Table:
        """
        Add metadata to a PyArrow Table.
        
        Args:
            table: PyArrow Table to add metadata to
            metadata: Dictionary of metadata to add
            
        Returns:
            pa.Table: Table with added metadata
        """
        # Add pipeline configuration to metadata
        if self.config:
            for key, value in self.config.items():
                if key not in metadata:
                    metadata[f'config_{key}'] = str(value)
        # Convert all metadata values to strings
        metadata_bytes = {k.encode(): str(v).encode() for k, v in metadata.items()}
        
        # Add required metadata if missing
        required_metadata = self.get_required_metadata()
        for key in required_metadata:
            if key not in metadata:
                metadata_bytes[key.encode()] = f"MISSING: {required_metadata[key]}".encode()
        
        return table.replace_schema_metadata(metadata_bytes)
    
    def read_parquet(self, file_path: Union[str, Path], validate: bool = True, schema_type: str = 'common') -> pd.DataFrame:
        """
        Read a parquet file with validation.
        
        Args:
            file_path: Path to parquet file
            validate: Whether to validate the schema
            schema_type: Type of schema to validate against
            
        Returns:
            pd.DataFrame: DataFrame from parquet file
        """
        df = pd.read_parquet(file_path)
        
        if validate:
            is_valid, missing_columns = self.validate_schema(df, schema_type)
            if not is_valid:
                print(f"Warning: Parquet file {file_path} is missing required columns: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error']:
                        df[col] = ''
                    elif col in ['row_id', 'download_retry_count']:
                        df[col] = 0
                    elif col == 'download_success':
                        df[col] = False
                    elif col == 'probability':
                        df[col] = 0.0
        
        return df
    
    def find_metadata_parquet(self, directory: Union[str, Path], require_url_column: bool = False) -> Optional[Path]:
        """
        Find the first valid metadata parquet file in a directory.
        
        Looks for parquet files that don't have section-specific columns
        like 'title' and 'header', and prioritizes files with the url_column.
        
        Args:
            directory: Directory to search for parquet files
            require_url_column: If True, require the URL column to be present; if False, only require filename column
            
        Returns:
            Optional[Path]: Path to the first valid metadata parquet, or None if not found
        """
        import logging
        logger = logging.getLogger(__name__)
        
        directory = Path(directory)
        if not directory.exists():
            logger.debug(f"Directory {directory} does not exist")
            return None
            
        # Get all parquet files in the directory
        parquet_files = list(directory.glob('**/*.parquet'))
        if not parquet_files:
            logger.debug(f"No parquet files found in {directory}")
            return None
            
        # Check for download_results files first
        download_files = [f for f in parquet_files if 'download_results' in str(f)]
        if download_files:
            logger.debug(f"Found {len(download_files)} download_results files")
        
        # Examine all files
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                columns = df.columns.tolist()
                
                # Skip section parquets - they have title/header columns
                if 'title' in columns or 'header' in columns or 'section' in columns:
                    logger.debug(f"Skipping sections parquet: {file_path}")
                    continue
                    
                # For metadata parquets - they don't have title/header but have filename
                if 'filename' in columns:
                    if require_url_column:
                        # Check if required URL column exists
                        if self.url_column in columns:
                            logger.info(f"Found metadata parquet with filename and {self.url_column}: {file_path}")
                            return file_path
                        else:
                            # Missing URL column
                            logger.warning(f"Found parquet with filename column but no {self.url_column} column: {file_path}")
                            logger.debug(f"Available columns: {columns}")
                    else:
                        # URL not required, filename is enough
                        logger.info(f"Found metadata parquet with filename (URL not required): {file_path}")
                        return file_path
                else:
                    logger.debug(f"Found parquet without filename column: {file_path}")
            except Exception as e:
                logger.debug(f"Error reading parquet {file_path}: {e}")
                continue
                
        logger.warning(f"No suitable metadata parquet found in {directory}")
        return None
    
    def is_valid_metadata_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file conforms to the metadata schema used by downloader.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has the required metadata fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for url_column (which might be custom) and filename
            required_fields = [self.url_column, 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def create_basic_metadata_parquet(self, markdown_dir: Union[str, Path], output_dir: Union[str, Path]) -> Union[Path, None]:
        """
        Create a simple metadata parquet file from a directory of markdown files.
        This is used when there is no existing parquet file to update.
        
        Args:
            markdown_dir: Directory containing markdown files
            output_dir: Directory where to create the parquet file
            
        Returns:
            Path: Path to the created parquet file, or None if creation failed
        """
        try:
            markdown_dir = Path(markdown_dir)
            output_dir = Path(output_dir)
            
            # Create output directory if it doesn't exist
            download_results_dir = output_dir / "download_results"
            os.makedirs(download_results_dir, exist_ok=True)
            
            # Get all markdown files in the input directory
            markdown_files = list(markdown_dir.glob("*.md"))
            if not markdown_files:
                print(f"No markdown files found in {markdown_dir}")
                return None
                
            # Create a DataFrame with just filenames
            data = []
            for md_file in markdown_files:
                entry = {
                    'filename': md_file.name,
                    self.url_column: ""  # Minimal URL placeholder
                }
                data.append(entry)
                
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Set output path for the parquet file
            output_path = download_results_dir / "download_results.parquet"
            
            # Write to parquet without adding complex metadata
            pq.write_table(pa.Table.from_pandas(df), output_path)
            
            print(f"Created new metadata parquet file at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating metadata parquet file: {e}")
            return None
            
    def is_download_result_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains download results with success/error information.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has download result fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for download result fields
            required_fields = ['download_success', 'filename']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
            
    def is_sections_parquet(self, filepath: Union[str, Path]) -> bool:
        """
        Check if a parquet file contains section data from extracted files.
        This identifies the second type of parquet in the pipeline - the sections parquet.
        
        Args:
            filepath: Path to the parquet file to check
            
        Returns:
            bool: True if the file has section data fields
        """
        try:
            schema = pq.read_schema(filepath)
            # Check for required section fields
            required_fields = ['filename', 'title', 'content', 'section']
            return all(field in schema.names for field in required_fields)
        except Exception:
            return False
        
    def add_processing_stage(self, df: pd.DataFrame, stage: str) -> pd.DataFrame:
        """
        Add or update processing stage column in a DataFrame.
        
        Args:
            df: Input DataFrame to update
            stage: Processing stage value to set (e.g., 'downloaded', 'extracted', 'classified')
            
        Returns:
            pd.DataFrame: Updated DataFrame with processing_stage column
        """
        df['processing_stage'] = stage
        return df
        
    def verify_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if a DataFrame contains all required columns and return missing ones.
        
        Args:
            df: DataFrame to check
            required_columns: List of column names that should be present
            
        Returns:
            Tuple containing:
            - bool: True if all required columns are present
            - List[str]: List of missing columns (empty if all present)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        return (len(missing_columns) == 0, missing_columns)
    
    def write_parquet(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        schema_type: str = 'common',
        validate: bool = True
    ) -> None:
        """
        Write a DataFrame to parquet with standard schema and metadata.
        
        Args:
            df: DataFrame to write
            file_path: Path to write parquet file
            metadata: Dictionary of metadata to include
            schema_type: Type of schema to use
            validate: Whether to validate the schema before writing
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Validate and fix schema if needed
        if validate:
            is_valid, missing_columns = self.validate_schema(df_copy, schema_type)
            if not is_valid:
                print(f"Adding missing columns to DataFrame: {missing_columns}")
                
                # Add missing columns with default values
                for col in missing_columns:
                    if col in ['id', 'filename', 'title', 'section', 'predicted_section', 'download_error']:
                        df_copy[col] = ''
                    elif col in ['row_id', 'download_retry_count']:
                        df_copy[col] = 0
                    elif col == 'download_success':
                        df_copy[col] = False
                    elif col == 'probability':
                        df_copy[col] = 0.0
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df_copy)
        
        # Add metadata if provided
        if metadata:
            table = self.add_metadata(table, metadata)
        
        # Write to parquet
        pq.write_table(table, file_path)
        print(f"Parquet file written to {file_path} with schema type '{schema_type}'")
