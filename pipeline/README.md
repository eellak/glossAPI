# GlossAPI

[![Release Version](https://img.shields.io/github/v/release/eellak/glossAPI)](https://github.com/eellak/glossAPI/releases)
[![PyPI Test Status](https://img.shields.io/badge/PyPI%20Test-glossapi-blue?logo=pypi)](https://test.pypi.org/project/glossapi/)

A library for processing academic texts in Greek and other languages, developed by [ΕΕΛΛΑΚ](https://eellak.gr/).

## Features

- **PDF Processing**: Extract text content from academic PDFs with structure preservation
- **Quality Control**: Filter and cluster documents based on extraction quality
- **Section Extraction**: Identify and extract academic sections from documents
- **Section Classification**: Classify sections using machine learning models
- **Greek Language Support**: Specialized processing for Greek academic texts
- **Metadata Handling**: Process academic texts with accompanying metadata
- **Customizable Annotation**: Map section titles to standardized categories

## Installation

```bash
pip install -i https://test.pypi.org/simple/ glossapi==0.0.6
```

## Usage

The recommended way to use GlossAPI is through the `Corpus` class, which provides a complete pipeline for processing academic documents:

```python
from glossapi import Corpus
import logging

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# Initialize Corpus with input and output directories
corpus = Corpus(
    input_dir="/path/to/documents",
    output_dir="/path/to/output",
    metadata_path="/path/to/metadata.parquet",  # Optional
    annotation_mapping={
        'Κεφάλαιο': 'chapter',
        # Add more mappings as needed
    }
)

# Step 1: Filter documents (quality control)
corpus.filter()

# Step 2: Extract sections from filtered documents
corpus.section()

# Step 3: Classify and annotate sections
corpus.annotate()
```

## License

This project is licensed under the European Union Public Licence 1.2 (EUPL 1.2).
