from pathlib import Path
from src.glossapi.gloss_extract import GlossExtract

extractor = GlossExtract()
extractor.create_extractor(enable_ocr=False)

input_files = [Path("samples/lightweight_pdf_corpus/pdfs/alpha.pdf")]
output_dir = Path("output")

extractor.extract_path(input_files, output_dir)
print("Extraction finished.")
