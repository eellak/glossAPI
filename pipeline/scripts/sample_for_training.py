#!/usr/bin/env python3
"""
Script to sample data from the kallipos processed data.

This script performs the following:
1. Creates 200 samples from 'Κεφάλαιο' document type, split into 2 parts
2. Creates 200 samples from all document types except 'Κεφάλαιο', split into 2 parts
3. Converts all samples to text format for analysis
"""

import os
import logging
from pathlib import Path
from sampler import Sampler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory with processed data
WORKING_DIR = "downloads/"

def main():
    logger.info("Creating sampler instance...")
    sampler = Sampler(WORKING_DIR)
    
    # Sample from Κεφάλαιο (200 samples in 2 parts)
    logger.info("Sampling from Κεφάλαιο document type...")
    kefalaia_parts = sampler.sample(
        sample_from={'document_type': 'Κεφάλαιο','header' : 'regex(Βλάχοι)'},
        n=5,
        parts=2,
        output_name="kefalaia_samples"
    )
    
    # Sample from everything except Κεφάλαιο (200 samples in 2 parts)
    logger.info("Sampling from all document types except Κεφάλαιο...")
    non_kefalaia_parts = sampler.sample(
        sample_from_all_except={'document_type': 'Κεφάλαιο','header' : 'regex(Ανάλυση)'},
        n=2,
        parts=2,
        output_name="non_kefalaia_samples"
    )
    
    # Convert each part to text with custom folder names
    logger.info("Converting kefalaia part 1 to text...")
    sampler.to_text(kefalaia_parts[0], folder_name="kefalaia_chapter_1")
    
    logger.info("Converting kefalaia part 2 to text...")
    sampler.to_text(kefalaia_parts[1], folder_name="kefalaia_chapter_2")
    
    logger.info("Converting non-kefalaia part 1 to text...")
    sampler.to_text(non_kefalaia_parts[0], folder_name="non_kefalaia_1")
    
    logger.info("Converting non-kefalaia part 2 to text...")
    sampler.to_text(non_kefalaia_parts[1], folder_name="non_kefalaia_2")
    
    # Print summary of samples
    logger.info("\nSampling summary:")
    logger.info(f"Kefalaia part 1: {len(kefalaia_parts[0])} rows from {len(kefalaia_parts[0]['filename'].unique())} unique files")
    logger.info(f"Kefalaia part 2: {len(kefalaia_parts[1])} rows from {len(kefalaia_parts[1]['filename'].unique())} unique files")
    logger.info(f"Non-kefalaia part 1: {len(non_kefalaia_parts[0])} rows from {len(non_kefalaia_parts[0]['filename'].unique())} unique files")
    logger.info(f"Non-kefalaia part 2: {len(non_kefalaia_parts[1])} rows from {len(non_kefalaia_parts[1]['filename'].unique())} unique files")
    
    # Print output information
    logger.info("\nOutput locations:")
    logger.info(f"CSV files: {sampler.datasets_dir}")
    logger.info(f"Text files: {sampler.text_dir}")

if __name__ == "__main__":
    main()
