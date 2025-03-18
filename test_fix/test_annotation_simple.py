import os
import logging
import pandas as pd
from glossapi.gloss_section_classifier import GlossSectionClassifier

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'filename': ['test.pdf'] * 5,
        'document_type': ['law'] * 5,
        'id': list(range(5)),
        'text': ['Sample text ' + str(i) for i in range(5)],
        'predicted_section': ['άλλο'] * 5  # All sections are "other"
    })
    
    # Save to a parquet file
    input_parquet = os.path.join(output_dir, "test_input.parquet")
    output_parquet = os.path.join(output_dir, "test_output.parquet")
    df.to_parquet(input_parquet)
    
    logger.info(f"Created test input parquet: {input_parquet}")
    
    # Initialize the classifier
    classifier = GlossSectionClassifier()
    
    # Test the fully_annotate method
    logger.info("Testing fully_annotate method...")
    classifier.fully_annotate(
        input_parquet=input_parquet,
        output_parquet=output_parquet,
        document_types={'test.pdf': 'law'},
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
