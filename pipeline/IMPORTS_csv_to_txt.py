import pandas as pd
import os

def process_file(input_csv_path, base_output_dir):
    """
    Process a CSV file and create a formatted text file for each unique filename.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        base_output_dir (str): Base directory to save the output text files
    """
    # Get the base filename of the CSV without extension
    csv_base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    
    # Create output directory for this specific CSV
    output_dir = os.path.join(base_output_dir, csv_base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    
    # Group by filename and sort by id
    grouped = df.groupby('filename').apply(lambda x: x.sort_values('id'))
    
    # Process each unique filename
    for filename, group in grouped.groupby(level=0):
        # Create output file path
        output_file_path = os.path.join(output_dir, f"{filename}.txt")
        
        # Write formatted content
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for _, row in group.iterrows():
                # Write row with formatting
                f.write(f"{{{row['row_id']}, {row['predicted_section']}}} {row['header']}\n\n")
                f.write(f"{row['section']}\n\n")
        
        print(f"Processed file: {output_file_path}")

def main():
    """
    Main function to process CSV files.
    """
    # Base output directory
    base_output_dir = '/mnt/data/section_classifier/datasets_in_txt'
    
    # Input files
    unclassified_csv = '/mnt/data/section_classifier/dataset/random_100_SVM_w_heuristic.csv'
    #classified_csv = '/mnt/data/section_classifier/dataset/random_100_classified_files.csv'
    
    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process unclassified files
    process_file(unclassified_csv, base_output_dir)
    
    # Process classified files
    #process_file(classified_csv, base_output_dir)

if __name__ == "__main__":
    main()
