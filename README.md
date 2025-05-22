# GlossAPI

[![PyPI Status](https://img.shields.io/pypi/v/glossapi?logo=pypi)](https://pypi.org/project/glossapi/)

A library for processing texts in Greek and other languages, developed by [Open Technologies Alliance(GFOSS)](https://gfoss.eu/).

## Features

- **Document Processing**: Extract text content from academic PDFs, DOCX, HTML, and other formats with structure preservation
- **Document Downloading**: Download documents from URLs with automatic handling of various formats
- **Quality Control**: Filter and cluster documents based on extraction quality
- **Section Extraction**: Identify and extract academic sections from documents
- **Section Classification**: Classify sections using machine learning models
- **Greek Language Support**: Specialized processing for Greek academic texts
- **Metadata Handling**: Process academic texts with accompanying metadata
- **Customizable Annotation**: Map section titles to standardized categories
- **Flexible Pipeline**: Start the processing from any stage in the pipeline

## Installation

```bash
pip install glossapi
```

## Usage

The recommended way to use GlossAPI is through the `Corpus` class, which provides a complete pipeline for processing academic documents. You can use the same directory for both input and output:

```python
from glossapi import Corpus
import logging

# Configure logging (optional)
logging.basicConfig(level=logging.INFO)

# Set the directory path (use the same for input and output)
folder = "/path/to/corpus"  # Use abstract path names

# Initialize Corpus with input and output directories
corpus = Corpus(
    input_dir=folder,
    output_dir=folder
    # metadata_path="/path/to/metadata.parquet",  # Optional
    # annotation_mapping={
    #     'Κεφάλαιο': 'chapter',
    #     # Add more mappings as needed
    # }
)

# The pipeline can start from any of these steps:

# Step 1: Download documents (if URLs are provided)
corpus.download(url_column='a_column_name')  # Specify column with URLs, default column name is 'url'

# Step 2: Extract documents
corpus.extract()

# Step 3: Extract sections from filtered documents
corpus.section()

# Step 4: Classify and annotate sections
corpus.annotate()  # or corpus.annotate(annotation_type="chapter") For texts without TOC or bibliography
```

## Folder Structure

After running the pipeline, the following folder structure will be created:

```
corpus/  # Your specified folder
├── download_results # stores metadata file with annotation from previous processing steps
├── downloads/  # Downloaded documents (if download() is used)
├── markdown/    # Extracted text files in markdown format 
├── sections/    # Contains the processed sections in parquet format
│   ├── sections_for_annotation.parquet
├── classified_sections.parquet    # Intermediate processing form
├── fully_annotated_sections.parquet  # Final processing form with section predictions
```

The `fully_annotated_sections.parquet` file contains the final processing form. The `predicted_sections` column shows the type of section: 'π' (table of contents), 'β' (bibliography), 'ε.σ.' (introductory note), 'κ' (main text), or 'a' (appendix). For files without table of contents or bibliography, the annotation will be "άλλο" (other).

## Note on Starting Points

**Option 1: Start with Document Download**
Create a corpus folder and add a parquet file with URLs for downloading:
```
corpus/
└── metadata.parquet (with a column containing document URLs)
```
Then use `corpus.download(url_column='column_name')` with the URL column name from your parquet file.

**Option 2: Start with Document Extraction**
Alternatively, place documents directly in the corpus folder and skip download:
```
corpus/
└── document1.pdf, document2.docx, etc.
```
GlossAPI will automatically create a metadata folder in downloads if starting from extract.

## License

This project is licensed under the [European Union Public Licence 1.2 (EUPL 1.2)](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12).