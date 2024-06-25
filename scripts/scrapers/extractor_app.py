import argparse
import os
import json
import logging
import time
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdfminer.psparser import PSEOF

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
        logging.info("Loaded existing metadata dictionary.")
        return metadata_dict
    except FileNotFoundError:
        logging.info("No existing metadata dictionary found. Starting fresh.")
        return {}
    
def get_indexes(papers):
    if papers:
        nums = []
        for p in papers:
            num = p.split("_")[-1]
            nums.append(int(num))
        return sorted(nums)[-1:]
    return []

def process_pdfs(downloaded_files_path, progress_report_path):
    extracted_files_dir = os.path.join(downloaded_files_path, "extracted_pdfs")
    problematic_file_path = os.path.join(extracted_files_dir, "problematic_pdfs.json")
    extracted_files_path = os.path.join(extracted_files_dir, "extracted_files.json")

    os.makedirs(extracted_files_dir, exist_ok=True)
    problematic_files = []
    extracted_files = read_json_file(extracted_files_path)
    progress_report = read_json_file(progress_report_path)
    index = 1
    if extracted_files:
            index = get_indexes(list(extracted_files.keys()))[0]

    pdf_files = [f for f in os.listdir(downloaded_files_path) if f.endswith('.pdf')]
    if not pdf_files:
        logging.info("No PDF files to process in the specified folder.")
        return

    logging.info("Starting the PDF processing script...")
    processed = list(extracted_files.keys())
    for file_name in pdf_files:
        base_name = os.path.splitext(file_name)[0]
        if base_name in processed:
            logging.info(f"{file_name} has already been processed.")
            continue

        file_path = os.path.join(downloaded_files_path, file_name)
        try:
            text = extract_text(file_path)
            if text:
                output_file_path = os.path.join(extracted_files_dir, f"paper_{index}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"Extracted and saved text from {file_name}")
                # Save metadata without the text file information
                extracted_files[f"paper_{index}"] = progress_report.get(file_name[:-4], "No metadata found")
                index += 1
        except (PDFSyntaxError, PSEOF, Exception) as e:
            problematic_files.append({"file_name": file_name, "error": str(e)})
            logging.error(f"Error processing {file_name}: {type(e).__name__}: {e}")

    if problematic_files:
        with open(problematic_file_path, 'w', encoding='utf-8') as f:
            json.dump(problematic_files, f, indent=4, ensure_ascii=False)

    # Save updated metadata dictionary
    with open(extracted_files_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_files, f, indent=4, ensure_ascii=False)

    logging.info("Files processing details have been updated and recorded.")



def run(current_path):
    setup_logging()
    args = type('args', (object,), {'path': current_path, 'json': 'progress_report.json'})
    process_pdfs(args.path, args.json)


if __name__ == "__main__":
    start_time = time.time()  # Capture start time
    setup_logging()
    parser = argparse.ArgumentParser(description="Process PDF files and associate extracted text with metadata.")
    parser.add_argument("--path", type=str, help="Path to the folder containing PDF files to be processed.")
    parser.add_argument("--json", type=str, help="Path to the JSON file containing metadata associations.")
    args = parser.parse_args()

    if not args.path or not args.json:
        parser.print_help()
        parser.exit()
    else:
        process_pdfs(args.path, args.json)

    end_time = time.time()  # Capture end time
    total_time = end_time - start_time  # Calculate runtime
    logging.info(f"Total Runtime: {total_time:.2f} seconds")  # Log the total runtime