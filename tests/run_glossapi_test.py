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
import pandas as pd
from glossapi import Corpus


def main() -> None:
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

    base_dir = Path(__file__).resolve().parent  # tests/
    output_dir = base_dir / "output"
    parquet_path = base_dir / "sample_urls.parquet"

    # Ensure base & output directories exist
    base_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the Parquet file once
    if not parquet_path.exists():
        pd.DataFrame({"url": pdf_urls}).to_parquet(parquet_path, index=False)
        print(f"Saved test URL list to {parquet_path}")

    # Instantiate high-level pipeline wrapper
    corpus = Corpus(
        input_dir=str(base_dir),  # where the Parquet file lives (and where downloads could be found if re-run)
        output_dir=str(output_dir),
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
