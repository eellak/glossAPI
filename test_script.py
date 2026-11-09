#!/usr/bin/env python3
"""
Simple test of the GlossAPI Corpus functionality with the refactored pipeline
"""
import logging
import sys
from pathlib import Path
from glossapi.corpus import Corpus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_test")

def main():
    # Corpus directory (input and output) holding the downloaded paper(s); no machine default.
    if len(sys.argv) != 2:
        sys.exit("usage: python test_script.py CORPUS_DIR")
    TEST_DIR = Path(sys.argv[1])
    # Create a basic corpus object - using same directory for input and output
    logger.info("Creating Corpus object")
    corpus = Corpus(
        input_dir=TEST_DIR,
        output_dir=TEST_DIR
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
