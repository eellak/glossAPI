#!/usr/bin/env python3
"""
Smoke-test script for GlossAPI with a tiny, 10-PDF dataset pulled from arXiv.

It performs the following steps:
1. Build a Parquet file `sample_urls.parquet` containing 10 PDF URLs.
2. Use `Corpus.download()` to fetch those PDFs into `tests/output/downloads/`.
3. Run the full GlossAPI pipeline: extract → filter → section → annotate.
4. Final artefacts land in `tests/output/`.

Run with:
    python tests/run_glossapi_test.py
"""

from pathlib import Path
import sys
import argparse
import shutil
import pandas as pd

# Allow running without installing the package: add pipeline/src to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]  # glossAPI/
local_src = project_root / "pipeline" / "src"
if local_src.exists() and str(local_src) not in sys.path:
    sys.path.insert(0, str(local_src))

from glossapi import Corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="GlossAPI smoke test")
    parser.add_argument("--parquet", type=str, default=None, help="Path to a Parquet file with a column 'url'")
    parser.add_argument("--clean", action="store_true", help="Delete existing workspace before running")
    args = parser.parse_args()

    # Ten small public PDFs from the arXiv archive (first ten IDs of 2021-01-01).
    pdf_urls = [
        "https://arxiv.org/pdf/2101.00001.pdf",
        "https://arxiv.org/pdf/2101.00002.pdf",
        "https://arxiv.org/pdf/2101.00003.pdf",
        "https://arxiv.org/pdf/2101.00004.pdf",
        "https://arxiv.org/pdf/2101.00005.pdf",
        "https://arxiv.org/pdf/2101.00006.pdf",
        "https://arxiv.org/pdf/2101.00007.pdf",
        "https://arxiv.org/pdf/2101.00008.pdf",
        "https://arxiv.org/pdf/2101.00009.pdf",
        "https://arxiv.org/pdf/2101.00010.pdf",
    ]

    # Workspace under /mnt/data so artefacts persist
    base_dir = Path("/mnt/data/glossapi_smoke")
    if args.clean and base_dir.exists():
        print(f"Cleaning workspace {base_dir} ...")
        shutil.rmtree(base_dir, ignore_errors=True)

    output_dir = base_dir / "output"

    # Determine which Parquet file to use
    if args.parquet:
        parquet_path = Path(args.parquet).expanduser().resolve()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Provided parquet not found: {parquet_path}")
    else:
        parquet_path = base_dir / "sample_urls.parquet"
        # Create sample Parquet if it doesn't exist yet
        if not parquet_path.exists():
            base_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"url": pdf_urls}).to_parquet(parquet_path, index=False)
            print(f"Saved default test URL list to {parquet_path}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate high-level pipeline wrapper
    # Determine packaged model path (works both for git checkout and pip install)
    packaged_model_path = local_src / "glossapi" / "models" / "section_classifier.joblib"

    corpus = Corpus(
        input_dir=str(base_dir),  # where the Parquet file lives (and where downloads could be found if re-run)
        output_dir=str(output_dir),
        section_classifier_model_path=str(packaged_model_path),
        verbose=True,
    )

    # 1. Download PDFs
    corpus.download(
        input_parquet=str(parquet_path),
        url_column="url",
        concurrency=5,
        verbose=True,
    )

    # 2. Convert to Markdown
    corpus.extract(num_threads=2, accel_type="CPU")  # Use CPU to stay portable

    # 3. Cluster good/bad quality (optional but quick on small set)
    corpus.filter()

    # 4. Split into logical sections
    corpus.section()

    # 5. Classify sections
    corpus.annotate()

    print("\nGlossAPI smoke test finished successfully.")
    print(f"Results saved under: {output_dir}")


if __name__ == "__main__":
    main()
