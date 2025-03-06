#!/usr/bin/env python3
"""
Test script for the GlossAPI corpus processing pipeline.
This script demonstrates the use of the refactored GlossAPI classes.
"""

import os
import sys
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
    
    # Define input and output directories
    input_dir = Path("/mnt/data/test_pdfs")
    output_dir = Path("/mnt/data/test_output")
    
    # Define model paths - one in the local repo and one in the mounted data directory
    local_model_dir = Path(__file__).parent / "pipeline" / "v2" / "models"
    local_model_path = local_model_dir / "section_classifier.joblib"
    
    # Create output and model directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(local_model_dir, exist_ok=True)
    
    # Define training CSV path
    training_csv = Path("/mnt/data/section_classifier/training_dataset_updated_06_02.csv")
    
    # Check if we need to train a model
    if not training_csv.exists():
        logger.error(f"Training CSV file not found: {training_csv}")
        return
    
    # Train a model from the CSV file and save it to the local repository
    logger.info(f"Training a model from {training_csv}...")
    classifier = GlossSectionClassifier()
    classifier.train_from_csv(str(training_csv), str(local_model_path))
    logger.info(f"Model trained and saved to {local_model_path}")
    
    # Verify that the model can be loaded
    logger.info("Verifying that the model can be loaded...")
    verification_classifier = GlossSectionClassifier()
    try:
        verification_classifier.load_model(str(local_model_path))
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Initialize the Corpus with the trained model
    logger.info(f"Initializing Corpus with input: {input_dir} and output: {output_dir}")
    corpus = Corpus(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=local_model_path,
        log_level=logging.INFO
    )
    
    # Skip extraction step and work with existing files
    logger.info("Skipping content extraction, working with existing files...")
    
    # Section the markdown files
    logger.info("Starting content sectioning from markdown...")
    corpus.extract()
    
    # Classify the sections
    logger.info("Starting section classification...")
    corpus.annotate(fully_annotate=True)
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()
