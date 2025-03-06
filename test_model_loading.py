#!/usr/bin/env python3
"""
Test script to verify that the section classifier model can be loaded and used in the pipeline.
"""

import os
import logging
from pathlib import Path
from pipeline.v2.src.glossapi_academic import Corpus, GlossSectionClassifier

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Define paths
    markdown_dir = Path("/mnt/data/test_output/markdown")
    output_dir = Path("/mnt/data/test_output")
    model_path = Path(__file__).parent / "pipeline" / "v2" / "models" / "section_classifier.joblib"
    
    # Verify that the model file exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Using model from {model_path}")
    
    # Initialize the Corpus with the model
    logger.info(f"Initializing Corpus with markdown dir: {markdown_dir} and output: {output_dir}")
    corpus = Corpus(
        input_dir=markdown_dir,
        output_dir=output_dir,
        model_path=model_path,
        log_level=logging.INFO
    )
    
    # Section the markdown files
    logger.info("Starting content sectioning from markdown...")
    corpus.extract()
    
    # Classify the sections
    logger.info("Starting section classification...")
    corpus.annotate(fully_annotate=True)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
