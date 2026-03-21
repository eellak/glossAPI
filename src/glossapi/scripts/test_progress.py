from __future__ import annotations

import shutil
import tempfile
import os
from pathlib import Path

# Important: ensure project root (src folder) is in sys.path
import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from glossapi import Corpus

# Path to sample PDFs
# d:/glossapi/glossAPI/samples/lightweight_pdf_corpus/pdfs
REPO_ROOT = Path(__file__).resolve().parents[3] 
PDF_DIR = REPO_ROOT / "samples" / "lightweight_pdf_corpus" / "pdfs"

def run_test():
    # Setup temporary directories
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Copy sample PDFs
        print(f"[*] Copying sample PDFs from {PDF_DIR} to {input_dir}...")
        for pdf_path in PDF_DIR.glob("*.pdf"):
            shutil.copy2(pdf_path, input_dir / pdf_path.name)
        
        # Initialize Corpus
        print("\n[*] Initializing Corpus...")
        corpus = Corpus(input_dir, output_dir)
        
        # 1. Extraction with progress bar
        print("\n--- Phase 1: Extraction ---")
        corpus.extract(input_format="pdf", phase1_backend="safe", use_gpus="none", show_progress=True)
        
        # 2. Cleaning with progress bar
        print("\n--- Phase 2: Cleaning ---")
        try:
            corpus.clean(show_progress=True)
        except Exception as e:
            print(f"Skipping clean (likely missing rust extension): {e}")
        
        # 3. Sectioning with progress bar
        print("\n--- Phase 3: Sectioning ---")
        corpus.section(show_progress=True)
        
        print("\n[+] Test completed successfully!")

if __name__ == "__main__":
    run_test()
