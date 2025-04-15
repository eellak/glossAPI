import sys
import os
import json
import pandas as pd

# Adjust path to import from the parent directory's src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from glossapi.gloss_section import GlossSection, Section

def reconstruct_section(processed_content: list) -> str:
    """
    Reconstructs the original raw text from the processed section content.
    
    Args:
        processed_content: The list of dictionaries representing the section's
                           categorized content (e.g., [{'text': '...'}, {'list': '...'}]).
                           
    Returns:
        The reconstructed raw text as a single string.
    """
    reconstructed_lines = []
    for item in processed_content:
        # The structure is {type: value}
        content_type, content_value = list(item.items())[0]
        # The value itself contains the original line breaks
        reconstructed_lines.append(content_value)
        
    # Join the content blocks with newlines to form the full raw text
    return "\n".join(reconstructed_lines)



def test_reconstruction_from_parquet(parquet_path: str = "/mnt/data/pipeline_refactor/output/sections/sections_for_annotation.parquet", test_all: bool = True):
    """
    Tests reconstruction by reading data from the pipeline's output Parquet file.
    
    Args:
        parquet_path: Path to the sections Parquet file.
        sample_size: Number of sections to randomly sample and test.
    """
    print(f"\n--- Running Reconstruction Test from Parquet ({parquet_path}) ---")
    
    if not os.path.exists(parquet_path):
        print(f"❌ ERROR: Parquet file not found at {parquet_path}")
        return False
        
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} sections from Parquet.")
    except Exception as e:
        print(f"❌ ERROR: Failed to load Parquet file: {e}")
        return False
        
    if len(df) == 0:
        print("⚠️ WARN: Parquet file is empty. No sections to test.")
        return True # Technically passed as no failures

    # Test all sections
    sample_df = df
    print(f"Testing reconstruction for all {len(sample_df)} sections...")
    
    all_passed = True
    failures = []
    
    for index, row in sample_df.iterrows():
        raw_content = row['section']  # This column contains the raw text
        section_json_str = row['json_section']  # This column contains the JSON representation
        filename = row['filename']
        header = row['header']
        
        try:
            processed_content = json.loads(section_json_str)
        except json.JSONDecodeError as e:
            print(f"❌ FAILED: Section {index} (File: {filename}, Header: '{header}') - Invalid JSON: {e}")
            failures.append(f"Index {index} (File: {filename}, Header: '{header}') - JSON Decode Error")
            all_passed = False
            continue
            
        reconstructed_text = reconstruct_section(processed_content)
        
        if raw_content != reconstructed_text:
            all_passed = False
            failures.append(f"Index {index} (File: {filename}, Header: '{header}') - Content Mismatch")
            print(f"❌ FAILED: Section {index} (File: {filename}, Header: '{header}') - Mismatch detected!")
            # You could add detailed diff printing here if needed for debugging
            # print(f"  Original:\n```\n{raw_content}\n```")
            # print(f"  Reconstructed:\n```\n{reconstructed_text}\n```")
        # else:
            # Optional: Print pass messages for verbosity
            # print(f"✅ PASSED: Section {index} (File: {filename}, Header: '{header}')")

    print("\n--- Parquet Test Summary ---")
    if all_passed:
        print(f"✅ All {len(sample_df)} sampled sections reconstructed successfully from Parquet!")
    else:
        print(f"❌ Reconstruction failed for {len(failures)}/{len(sample_df)} sampled sections:")
        for failure in failures:
            print(f"  - {failure}")
            
    return all_passed

if __name__ == "__main__":
    # Run the test using the real Parquet data
    test_passed = test_reconstruction_from_parquet()
    
    print("\n--- Overall Test Results ---")
    if test_passed:
        print("✅✅ All sections reconstructed successfully! ✅✅")
        sys.exit(0) 
    else:
        print("❌❌ Some sections failed reconstruction. ❌❌")
        sys.exit(1)
