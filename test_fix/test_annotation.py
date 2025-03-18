import os
import logging
import pandas as pd
from glossapi.gloss_section_classifier import GlossSectionClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_parquet():
    """Create a test parquet file with minimal data to simulate PDF processing"""
    input_dir = os.path.join(os.path.dirname(__file__), "input")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the PDF filename
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logger.error("No PDF files found in input directory!")
        return None
    
    pdf_filename = pdf_files[0]
    logger.info(f"Using PDF file: {pdf_filename}")
    
    # Create a simple DataFrame to simulate PDF extraction
    df = pd.DataFrame({
        'filename': [pdf_filename] * 10,
        'document_type': ['law'] * 10,
        'id': list(range(10)),
        'text': [f'Sample text from {pdf_filename} section {i}' for i in range(10)],
        'predicted_section': ['άλλο'] * 10  # All sections are "other"
    })
    
    # Save to a parquet file
    parquet_path = os.path.join(output_dir, "extracted_text.parquet")
    df.to_parquet(parquet_path)
    logger.info(f"Created test parquet: {parquet_path}")
    
    return parquet_path

def main():
    # Initialize paths
    input_dir = os.path.join(os.path.dirname(__file__), "input")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test parquet file (simulating PDF processing)
    input_parquet = create_test_parquet()
    if not input_parquet:
        return
    
    # Test annotation directly with GlossSectionClassifier
    classifier = GlossSectionClassifier()
    output_parquet = os.path.join(output_dir, "annotated_text.parquet")
    
    # Get PDF filename for document_types dictionary
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
    document_types = {pdf_files[0]: 'law'}
    
    logger.info("Testing annotation with fully_annotate=True...")
    classifier.fully_annotate(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        document_types=document_types,
        annotation_type='auto'
    )
    
    # Check if the output file was created
    if os.path.exists(output_parquet):
        logger.info(f"Success! Output file created: {output_parquet}")
        
        # Read the output file
        df_output = pd.read_parquet(output_parquet)
        logger.info(f"Output DataFrame shape: {df_output.shape}")
        logger.info(f"Output columns: {df_output.columns}")
        logger.info(f"Output sections: {df_output['predicted_section'].unique()}")
    else:
        logger.error("Failed to create output file!")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
