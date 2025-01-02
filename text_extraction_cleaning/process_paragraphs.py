import os
import pandas as pd
from paragraph_cleaning import paragraph_maker, paragraph_merger

# Directory containing MD files
md_dir = "/home/fivos/Desktop/text_sources/sxolika_vivlia/md_output_unique/"

# List to store all paragraphs with their source file
all_paragraphs = []

# Process each MD file
for filename in os.listdir(md_dir):
    if filename.endswith(".md"):
        file_path = os.path.join(md_dir, filename)
        
        # Read the MD file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Process the content with paragraph_maker
        paragraphs = paragraph_maker(content, maxpadding=1)
        paragraphs = paragraph_merger(paragraphs,1000,25)
        
        # Add each paragraph along with its source file
        for paragraph in paragraphs:
            if paragraph.strip():  # Only add non-empty paragraphs
                all_paragraphs.append({
                    'paragraphs': paragraph,
                    'type': '',
                    'source_file': filename
                })

# Create DataFrame
df = pd.DataFrame(all_paragraphs)

# Save the DataFrame to parquet
output_file = 'paragraphs_output.parquet'
df.to_parquet(output_file, index=False)

print(f"Processed {len(all_paragraphs)} total paragraphs from {sum(1 for f in os.listdir(md_dir) if f.endswith('.md'))} MD files")
print(f"DataFrame saved to {output_file}")
