#!/usr/bin/env python3
"""
Script to sample data from a parquet file and convert it to text files.
"""

import os
import logging
import argparse
from pathlib import Path
from pipeline.v2.src.glossapi_academic import Corpus
from pipeline.v2.src.glossapi_academic.corpus import Sampler

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Sample data from a parquet file and convert it to text files.')
    parser.add_argument('--parquet-file', required=True, help='Path to the parquet file to sample from')
    parser.add_argument('--output-dir', required=True, help='Directory to save the text files')
    parser.add_argument('--sample-size', type=int, default=2, help='Number of unique filenames to sample')
    parser.add_argument('--temp-csv', help='Optional path to save the sampled data as CSV')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a Corpus instance (with dummy input/output dirs)
        logger.info("Initializing Corpus...")
        corpus = Corpus(
            input_dir="/tmp",  # Dummy input dir
            output_dir="/tmp",  # Dummy output dir
            log_level=logging.INFO
        )
        
        # Sample from the parquet file
        logger.info(f"Sampling {args.sample_size} unique filenames from {args.parquet_file}...")
        sampler = Sampler(corpus, args.parquet_file)
        sampled_df = sampler.sample(size=args.sample_size, output_csv=args.temp_csv)
        
        if sampled_df.empty:
            logger.error("No data was sampled. Exiting.")
            return 1
        
        # Convert sampled data to text files
        logger.info(f"Converting sampled data to text files in {args.output_dir}...")
        corpus.to_text(sampled_df, args.output_dir)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
