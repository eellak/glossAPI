#!/usr/bin/env python3
"""
Simple test of the GlossAPI Corpus functionality with the refactored pipeline
"""
import logging
import os
import sys
from pathlib import Path
from glossapi.corpus import Corpus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_test")

# Resolved repository root directory
REPO_ROOT = Path(__file__).resolve().parent


def get_test_dirs():
    """Resolve input and output directories portably with optional override support."""
    override_dir = os.getenv("GLOSSAPI_TEST_DIR", sys.argv[1] if len(sys.argv) > 1 else None)
    if override_dir:
        base = Path(override_dir)
        return base, base

    sample_corpus = REPO_ROOT / "samples" / "lightweight_pdf_corpus" / "pdfs"
    output_dir = REPO_ROOT / "artifacts" / "test_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    return sample_corpus, output_dir


def main():
    input_dir, output_dir = get_test_dirs()

    # Create a basic corpus object
    logger.info("Creating Corpus object (input=%s, output=%s)", input_dir, output_dir)
    corpus = Corpus(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    # Skipping download step since we already have the PDF file
    logger.info("Skipping download step (already have the PDF file)")
    
    # 2. Extract
    logger.info("Running extract step")
    # Specify the formats we know are in the downloads directory
    corpus.extract()
        
    # 4. Section - now uses files marked as 'good' quality
    logger.info("Running section step")
    corpus.section()
    
    # 5. Annotate
    logger.info("Running annotate step")
    corpus.annotate(annotation_type="chapter")
    
    # Check results
    logger.info("Pipeline completed")
    

if __name__ == "__main__":
    main()
