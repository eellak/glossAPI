#!/usr/bin/env python3
"""
Example script showing how to process markdown files using the GlossAPI pipeline.
This script skips the PDF extraction step and starts directly with markdown files.
"""

import os
from pathlib import Path
import logging
import argparse
from glossapi import Corpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("markdown_processing.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def process_markdown_repository(
    input_dir: str, 
    output_dir: str, 
    metadata_path: str = None,
    annotation_type: str = "auto"
):
    """
    Process a repository of markdown files through the GlossAPI pipeline.
    
    Args:
        input_dir (str): Path to directory containing markdown files
        output_dir (str): Path to directory for output files
        metadata_path (str, optional): Path to metadata file (parquet format)
        annotation_type (str, optional): Type of annotation to use ('auto', 'text', 'chapter')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define annotation mapping (customize as needed)
    annotation_mapping = {
        'Κεφάλαιο': 'chapter',  # Chapters need special handling
    }
    
    logger.info(f"Processing markdown files from {input_dir}")
    logger.info(f"Output will be saved to {output_dir}")
    
    # Initialize Corpus
    corpus = Corpus(
        input_dir=input_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        annotation_mapping=annotation_mapping
    )
    
    # Step 1: Filter markdown files (quality control)
    logger.info("Step 1: Filtering markdown files...")
    corpus.filter(input_dir=input_dir)
    
    # Step 2: Extract sections from filtered markdown
    logger.info("Step 2: Extracting sections from markdown...")
    corpus.section()
    
    # Step 3: Classify and annotate sections
    logger.info("Step 3: Classifying and annotating sections...")
    corpus.annotate(annotation_type=annotation_type)
    
    logger.info("Processing complete!")
    logger.info(f"Results saved to: {output_dir}")
    
    # Return paths to output files
    return {
        "good_quality_files": Path(output_dir) / "quality_clustering" / "good",
        "bad_quality_files": Path(output_dir) / "quality_clustering" / "bad",
        "sections": Path(output_dir) / "sections" / "sections_for_annotation.parquet",
        "annotated_sections": Path(output_dir) / "annotated_sections" / "annotated_sections.parquet",
        "fully_annotated_sections": Path(output_dir) / "fully_annotated_sections.parquet"
    }

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process markdown files using the GlossAPI pipeline")
    parser.add_argument("--input-dir", required=True, help="Path to directory containing markdown files")
    parser.add_argument("--output-dir", required=True, help="Path to directory for output files")
    parser.add_argument("--metadata-path", help="Path to metadata file (parquet format)")
    parser.add_argument("--annotation-type", default="auto", choices=["auto", "text", "chapter"], 
                        help="Type of annotation to use")
    
    args = parser.parse_args()
    
    # Process the repository
    process_markdown_repository(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_path=args.metadata_path,
        annotation_type=args.annotation_type
    )
