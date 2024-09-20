import os
import argparse
import json

def process_pdfs(folder_path, base_name, json_file):
    metadata_dict = {}
    
    if not json_file.endswith('.json'):
        json_file += '.json'
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    pdf_files.sort()
    
    for index, original_filename in enumerate(pdf_files, start=1):
        new_filename = f"{base_name}_{index}.pdf"
        original_filepath = os.path.join(folder_path, original_filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        os.rename(original_filepath, new_filepath)
        
        metadata_dict[f"{base_name}_{index}"] = original_filename
    
    json_filepath = os.path.join(folder_path, json_file)
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and rename PDF files in a given folder.')
    parser.add_argument('--path', type=str, help='Path to the folder containing PDF files.')
    parser.add_argument('--filetype', type=str, help='Base name for renamed PDF files; paper_1 or book_1, generally {filetype}_1')
    parser.add_argument('--index', type=str, required=True, help='Output JSON file with metadata.')

    args = parser.parse_args()
    
    process_pdfs(args.path, args.filetype, args.index)
