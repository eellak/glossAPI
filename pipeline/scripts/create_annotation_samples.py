#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_annotation_samples(
    input_parquet_path: str,
    output_dir: str,
    section_type_1: str = "π",
    section_type_2: str = "β",
    samples_per_type: int = 100,
    output_file_1: str = "Ioanna_sample.csv",
    output_file_2: str = "Katerina_sample.csv"
):
    """
    Sample rows from a parquet file containing annotated sections and split them into two CSV files.
    
    Args:
        input_parquet_path: Path to the input parquet file
        output_dir: Directory to save the output CSV files
        section_type_1: First section type to sample (default: "π")
        section_type_2: Second section type to sample (default: "β")
        samples_per_type: Number of samples to pick from each section type (default: 100)
        output_file_1: Name of the first output CSV file (default: "Ioanna_sample.csv")
        output_file_2: Name of the second output CSV file (default: "Katerina_sample.csv")
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the parquet file
    print(f"Reading parquet file from {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    print(f"Loaded {len(df)} rows from parquet file")
    
    # Filter rows with the specified section types
    df_type_1 = df[df["predicted_section"] == section_type_1]
    df_type_2 = df[df["predicted_section"] == section_type_2]
    
    print(f"Found {len(df_type_1)} rows with section type '{section_type_1}'")
    print(f"Found {len(df_type_2)} rows with section type '{section_type_2}'")
    
    # Sample rows randomly
    if len(df_type_1) < samples_per_type:
        print(f"Warning: Found only {len(df_type_1)} rows for section type '{section_type_1}', using all available")
        sampled_type_1 = df_type_1
    else:
        sampled_type_1 = df_type_1.sample(n=samples_per_type, random_state=42)
    
    if len(df_type_2) < samples_per_type:
        print(f"Warning: Found only {len(df_type_2)} rows for section type '{section_type_2}', using all available")
        sampled_type_2 = df_type_2
    else:
        sampled_type_2 = df_type_2.sample(n=samples_per_type, random_state=42)
    
    # Combine and shuffle
    combined_df = pd.concat([sampled_type_1, sampled_type_2])
    shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Combined and shuffled {len(shuffled_df)} rows")
    
    # Split into two equal parts
    half_size = len(shuffled_df) // 2
    first_half = shuffled_df[:half_size]
    second_half = shuffled_df[half_size:]
    
    # Save to CSV
    output_path_1 = os.path.join(output_dir, output_file_1)
    output_path_2 = os.path.join(output_dir, output_file_2)
    
    first_half.to_csv(output_path_1, index=False)
    second_half.to_csv(output_path_2, index=False)
    
    print(f"Saved {len(first_half)} rows to {output_path_1}")
    print(f"Saved {len(second_half)} rows to {output_path_2}")
    
    return output_path_1, output_path_2

if __name__ == "__main__":
    # Define paths
    input_path = "/mnt/data/kallipos_processed/fully_annotated_sections.parquet"
    output_dir = "/mnt/data/glossAPI/pipeline/v2/data/annotation_samples"
    
    # Create the samples
    output_files = create_annotation_samples(
        input_parquet_path=input_path,
        output_dir=output_dir
    )
    
    print(f"Annotation samples created successfully!")
    print(f"Sample files: {output_files}")
