#!/usr/bin/env python3
"""GlossAPI smoke test using the installed package (from venv).

This script verifies that the `glossapi` package installed in the current
Python environment works end-to-end on a small Greek corpus.  It performs:
    1. Download of Greek PDFs listed in `/mnt/data/greek_pdf_urls.parquet`
    2. Extraction → Sectioning → Cleaning

Run with:
    python tests/run_glossapi_test.py [--clean]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytest

if not os.environ.get("GLOSSAPI_ENABLE_INTEGRATION"):
    pytest.skip(
        "GlossAPI integration smoke test disabled (set GLOSSAPI_ENABLE_INTEGRATION=1 to run)",
        allow_module_level=True,
    )

# Import from the site-packages installation, not the local repo
try:
    from glossapi import Corpus  # type: ignore
except ImportError as exc:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists():
        sys.stderr.write(
            "Failed to import glossapi from site-packages; falling back to local source path\n"
        )
        sys.path.insert(0, str(repo_src))
        try:
            from glossapi import Corpus  # type: ignore
        except ImportError as exc2:
            print("Failed to import glossapi from local source:", exc2, file=sys.stderr)
            sys.exit(1)
    else:
        print("Failed to import glossapi from site-packages:", exc, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="GlossAPI smoke test (Greek corpus)")
    parser.add_argument("--clean", action="store_true", help="Remove previous workspace before running")
    args = parser.parse_args()

    parquet_path = Path("/mnt/data/greek_pdf_urls.parquet")
    if not parquet_path.exists():
        print(f"Required parquet not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    # Workspace where artefacts are stored
    base_dir = Path("/mnt/data/glossapi_smoke")
    if args.clean and base_dir.exists():
        import shutil

        print(f"Cleaning existing workspace at {base_dir} …")
        shutil.rmtree(base_dir, ignore_errors=True)

    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise wrapper (model path auto-discovered by the package)
    corpus = Corpus(
        input_dir=str(base_dir),
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
    corpus.extract(num_threads=2, accel_type="CPU")

    # 3. Sectioning before cleaning so the Rust cleaner works per-section
    corpus.section()

    # 4. Clean (includes Rust badness score); do not drop anything yet
    corpus.clean(drop_bad=False)

    print("\nGlossAPI smoke test finished successfully.")
    print(f"Results saved under: {output_dir}\n")


if __name__ == "__main__":
    main()
