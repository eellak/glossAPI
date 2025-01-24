import re
import os
import datetime
import paragraph_cleaning_tools as pct

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'full_clean_output')
os.makedirs(output_dir, exist_ok=True)

# Source directory containing MD files
source_dir = '/home/fivos/Desktop/zzz_extraction_separation_combined/well_extracted'

# Get list of all MD files
md_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.md')])
total_files = len(md_files)

# Create stats file
stat_outputfile = os.path.join(output_dir, 'stat_file.txt')
with open(stat_outputfile, 'w', encoding='utf-8') as stat_file:
    stat_file.write('Processing started at: ' + str(datetime.datetime.now()) + '\n\n')

# Process each file
for i, file in enumerate(md_files, 1):
    print(f"Processing file {i}/{total_files}: {file}...")
    
    try:
        # Read input file
        with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as infile:
            text = infile.read()

        # Apply cleaning pipeline
        paragraphs = pct.paragraph_maker(text, maxpadding=1)
        paragraphs = pct.paragraph_clean_image(paragraphs)
        paragraphs = pct.paragraph_clean_dotlines(paragraphs)
        paragraphs = pct.paragraph_remove_artifacts(paragraphs)
        paragraphs = pct.paragraph_fix_broken_line(paragraphs)
        paragraphs = pct.paragraph_merger(paragraphs, 500, 10)
        paragraphs = pct.remove_numbered_title(paragraphs, pct.remove_title_number_pattern)
        
        # Remove various sections
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.summary_tags, print=True)
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.catalog_tags, 
                                               ending_tags=pct.CONTENT_and_CATALOG_end_tags, print=True)
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.bibliography_tags, print=True)
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.euritirio_tags, print=True)
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.glossary_tags, print=True)
        
        # Combine ending tags for content removal
        combined_end_tags = pct.CONTENT_and_CATALOG_end_tags + pct.catalog_tags
        paragraphs = pct.remove_taged_paragraphs(paragraphs=paragraphs, tags=pct.content_tags, 
                                               ending_tags=combined_end_tags, print=True, skip_paragraphs=1)
        
        # Final cleaning steps
        paragraphs = pct.all_paragraph_not_char_end(paragraphs, pct.endings, print=True)
        paragraphs = pct.remove_noise(paragraphs, pct.noise_pattern)
        paragraphs = pct.paragraph_remove_artifacts(paragraphs)

        # Write output
        output_file_path = os.path.join(output_dir, f'clean_{file}')
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            pct.test_write_text(paragraphs, output_file)
            
        # Write statistics
        with open(stat_outputfile, 'a+', encoding='utf-8') as stat_file:
            pct.stat_assembly(pct.total_paragraphs(paragraphs), paragraphs)
            stat_file.write(f'{file} : {str(pct.file_stat_list)}\n')
        pct.file_stat_list = pct.file_reset_list()
        
        if i % 10 == 0:  # Progress update every 10 files
            print(f"Completed {i}/{total_files} files ({(i/total_files)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print("\nProcessing complete! Check the full_clean_output directory for results.")
print("Converting statistics to CSV...")

# Run the stats_to_csv conversion
import stats_to_csv
stats_to_csv.convert_stats_to_csv(
    stat_outputfile,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'statistics.csv')
)
print("Done! Statistics have been converted to CSV format.")
