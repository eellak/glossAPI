"""
Sampler module for extracting samples from processed corpus data.

This module provides functionality for sampling documents from processed
parquet files, with options for filtering by column values and splitting
into parts for cross-validation.
"""

import logging
import os
import pandas as pd
import random
from pathlib import Path
from typing import Dict, Optional, Union, List, Any, Tuple

class Sampler:
    """
    A class for sampling documents from parquet files with flexible filtering options.
    
    This class allows sampling unique filenames based on specific criteria and
    extracting all their rows for analysis or further processing.
    
    Example:
        sampler = Sampler("/path/to/processed_data")
        
        # Sample 200 files where 'document_type' is 'Κεφάλαιο'
        sample_df = sampler.sample(sample_from={'document_type': 'Κεφάλαιο'}, n=200)
        
        # Sample 200 files from everything except where 'document_type' is 'Κεφάλαιο'
        sample_df = sampler.sample(sample_from_all_except={'document_type': 'Κεφάλαιο'}, n=200)
        
        # Sample and split into 2 equal parts for cross-validation
        sample_df = sampler.sample(n=200, parts=2)
    """
    
    def __init__(
        self, 
        base_dir: Union[str, Path],
        parquet_file: Optional[Union[str, Path]] = None,
        project_dir: Optional[Union[str, Path]] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize the Sampler.
        
        Args:
            base_dir: Base directory where processed data is stored
            parquet_file: Optional specific parquet file to sample from
                          (default: fully_annotated_sections.parquet in base_dir)
            project_dir: Optional project directory for text outputs
                         (default: v2 directory in parent of base_dir)
            log_level: Logging level (default: INFO)
        """
        self.base_dir = Path(base_dir)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set the default parquet file if not specified
        if parquet_file is None:
            self.parquet_file = self.base_dir / "fully_annotated_sections.parquet"
        else:
            self.parquet_file = Path(parquet_file)
        
        # Set up datasets directory in the base directory
        self.datasets_dir = self.base_dir / "datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)
        
        # Set up project directory for text outputs
        if project_dir is None:
            try:
                # Try to find 'v2' directory in parent of base_dir
                parent_dir = self.base_dir.parent
                if (parent_dir / "v2").exists():
                    self.project_dir = parent_dir / "v2"
                else:
                    # Fall back to base_dir if v2 not found
                    self.project_dir = self.base_dir
            except Exception:
                self.project_dir = self.base_dir
        else:
            self.project_dir = Path(project_dir)
        
        # Set up text samples directory in the project directory
        self.text_dir = self.project_dir / "text_samples"
        os.makedirs(self.text_dir, exist_ok=True)
    
    def sample(
        self, 
        n: int = 100, 
        parts: int = 1, 
        output_csv: Optional[Union[str, Path]] = None,
        sample_from: Optional[Dict[str, Any]] = None,
        sample_from_all_except: Optional[Dict[str, Any]] = None,
        output_name: Optional[str] = None
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Sample a specified number of unique filenames and extract all their rows.
        
        Args:
            n: Number of unique filenames to sample
            parts: Number of even parts to split the sample into (default: 1)
            output_csv: Optional path to save the sampled data as CSV
                       If not specified, will use output_name with default location
            sample_from: Optional dictionary {column: value} to sample only from rows
                        where column has the specified value
            sample_from_all_except: Optional dictionary {column: value} to sample only
                                   from rows where column does NOT have the specified value
            output_name: Base name for output files (without extension)
                        If not specified, will generate based on sampling criteria
            
        Returns:
            If parts=1: DataFrame containing all rows for the sampled filenames
            If parts>1: List of DataFrames, each containing rows for a part of the sampled filenames
            
        Raises:
            ValueError: If the specified column or label doesn't exist in the data
        """
        if not self.parquet_file.exists():
            self.logger.error(f"Parquet file not found: {self.parquet_file}")
            return pd.DataFrame()
        
        self.logger.info(f"Reading data from {self.parquet_file}...")
        
        # Read the parquet file
        df = pd.read_parquet(self.parquet_file)
        
        # Check if filtering criteria are valid
        if sample_from:
            for column, value in sample_from.items():
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found in the parquet file")
                if value not in df[column].values:
                    raise ValueError(f"Value '{value}' not found in column '{column}'")
        
        if sample_from_all_except:
            for column, value in sample_from_all_except.items():
                if column not in df.columns:
                    raise ValueError(f"Column '{column}' not found in the parquet file")
                if value not in df[column].values:
                    raise ValueError(f"Value '{value}' not found in column '{column}'")
        
        # Apply filters to the DataFrame
        filtered_df = df.copy()
        
        # Generate default output name if not provided
        if output_name is None:
            if sample_from and len(sample_from) == 1:
                col, val = next(iter(sample_from.items()))
                output_name = f"{val.lower().replace(' ', '_')}_samples"
            elif sample_from_all_except and len(sample_from_all_except) == 1:
                col, val = next(iter(sample_from_all_except.items()))
                output_name = f"non_{val.lower().replace(' ', '_')}_samples"
            else:
                output_name = "samples"
        
        # Apply filters to the DataFrame
        if sample_from:
            for column, value in sample_from.items():
                filtered_df = filtered_df[filtered_df[column] == value]
                self.logger.info(f"Filtered to rows where {column} = '{value}' ({len(filtered_df)} rows)")
        
        if sample_from_all_except:
            for column, value in sample_from_all_except.items():
                filtered_df = filtered_df[filtered_df[column] != value]
                self.logger.info(f"Filtered to rows where {column} != '{value}' ({len(filtered_df)} rows)")
        
        # Get unique filenames from the filtered data
        unique_filenames = filtered_df['filename'].unique()
        total_unique = len(unique_filenames)
        
        if total_unique == 0:
            self.logger.error("No matching filenames found after applying filters")
            return pd.DataFrame()
        
        self.logger.info(f"Found {total_unique} unique filenames after filtering")
        
        if total_unique <= n:
            self.logger.warning(f"Requested sample size ({n}) is greater than or equal to the number of unique filenames ({total_unique}). Using all available filenames.")
            sampled_filenames = unique_filenames
        else:
            # Randomly sample unique filenames
            sampled_filenames = random.sample(list(unique_filenames), n)
        
        # Extract all rows for the sampled filenames
        sampled_df = df[df['filename'].isin(sampled_filenames)]
        
        self.logger.info(f"Sampled {len(sampled_filenames)} unique filenames with {len(sampled_df)} total rows")
        
        # Set up default output CSV path if not provided
        if output_csv is None and parts == 1:
            output_csv = self.datasets_dir / f"{output_name}.csv"
        
        # Save to CSV if output path is provided
        if output_csv and parts == 1:
            output_path = Path(output_csv)
            os.makedirs(output_path.parent, exist_ok=True)
            sampled_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved sampled data to {output_path}")
        
        # Split into parts if requested
        if parts > 1:
            self.logger.info(f"Splitting sample into {parts} equal parts")
            
            # Split the sampled filenames into equal parts
            random.shuffle(sampled_filenames)
            filename_parts = [sampled_filenames[i::parts] for i in range(parts)]
            
            # Create a DataFrame for each part
            result_parts = []
            for i, filenames in enumerate(filename_parts):
                part_df = df[df['filename'].isin(filenames)]
                result_parts.append(part_df)
                self.logger.info(f"Part {i+1}: {len(filenames)} filenames, {len(part_df)} rows")
                
                # Set up default output CSV path for each part if not provided
                if output_csv is None:
                    part_output = self.datasets_dir / f"{output_name}_{i+1}.csv"
                else:
                    # If output_csv is provided, create part-specific paths
                    output_stem = Path(output_csv).stem
                    output_suffix = Path(output_csv).suffix
                    output_dir = Path(output_csv).parent
                    part_output = output_dir / f"{output_stem}_{i+1}{output_suffix}"
                
                # Save each part
                os.makedirs(Path(part_output).parent, exist_ok=True)
                part_df.to_csv(part_output, index=False)
                self.logger.info(f"Saved part {i+1} to {part_output}")
            
            return result_parts
        
        return sampled_df
    
    def to_text(
        self, 
        input_data: Union[str, Path, pd.DataFrame], 
        output_dir: Optional[Union[str, Path]] = None,
        folder_name: Optional[str] = None
    ) -> None:
        """
        Convert parquet or CSV data to formatted text files.
        
        Args:
            input_data: Path to parquet/CSV file or DataFrame containing the data
            output_dir: Directory to save the output text files
                      If None, creates a directory in text_samples based on folder_name
            folder_name: Name for the output directory if output_dir is None
                       If None, uses a default name based on timestamp
        """
        # Set up output directory
        if output_dir is None:
            if folder_name is None:
                # Generate a timestamp-based name if no folder name provided
                if isinstance(input_data, pd.DataFrame):
                    # Try to infer a good name from the DataFrame if available
                    if 'document_type' in input_data.columns and len(input_data['document_type'].unique()) == 1:
                        folder_name = f"{input_data['document_type'].iloc[0].lower().replace(' ', '_')}_samples"
                    else:
                        folder_name = f"samples_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    # Use the input filename if it's a file
                    if isinstance(input_data, (str, Path)):
                        folder_name = Path(input_data).stem
                    else:
                        folder_name = f"samples_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_dir = self.text_dir / folder_name
        
        self.logger.info(f"Converting data to formatted text files in {output_dir}...")
        
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
            if 'id' in group.columns:
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
