# GlossAPI

[![PyPI Status](https://img.shields.io/pypi/v/glossapi?logo=pypi)](https://pypi.org/project/glossapi/)

A library for processing academic texts in Greek and other languages, developed by [ΕΕΛΛΑΚ](https://eellak.gr/).

## Features

- **PDF Processing**: Extract text content from academic PDFs with structure preservation
- **Quality Control**: Filter and cluster documents based on extraction quality
- **Section Extraction**: Identify and extract academic sections from documents
- **Section Classification**: Classify sections using machine learning models
- **Greek Language Support**: Specialized processing for Greek academic texts
- **Metadata Handling**: Process academic texts with accompanying metadata
- **Customizable Annotation**: Map section titles to standardized categories
- **PDF Downloaders**: Download Greek academic texts from various sources

## Installation

```bash
pip install glossapi
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
        'Κεφάλαιο': 'chapter', # i.e. a label in document_type column : references text type to be annotated chapter or text for now
        # Add more mappings as needed
    }
)

# Step 1: Extract documents (with quality control)
corpus.extract()

# Step 2: Extract sections from filtered documents
corpus.section()

# Step 3: Classify and annotate sections
corpus.annotate()
```

## PDF Downloaders

GlossAPI includes tools for downloading academic PDFs from various Greek sources:

- **Kodiko**: Legal documents and laws (over 26,000 documents)
- **Greek Language**: Greek language learning resources (approximately 50 documents)
- **Cyprus Exams**: Examination papers and educational materials from Cyprus
- **Kallipos**: Academic textbooks and educational materials
- **Pergamos**: University theses and archives

Each downloader is specialized for its target website and includes features for:
- Concurrent downloading with appropriate rate limiting
- Progress tracking and resumable downloads
- Error handling and automatic retries
- File organization and metadata preservation

### Using the Downloaders

The downloaders are located in the `scraping/download_and_extract_scripts` directory:

```bash
# Kodiko downloader (legal documents)
python downloader_kodiko.py --json ../../scraping/json_sitemaps/kodiko_pdf.json --type pdf --req get --output ../../downloads/kodiko --batch 10 --sleep 2

# Greek Language downloader (language resources)
python downloader_greek_language.py --json ../../scraping/json_sitemaps/greek-language_pdf.json --type pdf --req get --output ../../downloads/greek-language --batch 5 --sleep 3

# Cyprus Exams downloader (examination papers)
python downloader_cyprus_exams.py --json ../../scraping/json_sitemaps/cyprus-exams_pdf.json --type pdf --req get --output ../../downloads/cyprus-exams --batch 5 --sleep 3
```

For automated downloading, use the provided shell scripts:

```bash
# Download all Kodiko documents
bash download_all_kodiko.sh

# Download all Greek Language resources
bash download_all_greek_language.sh

# Download all Cyprus Exams documents
bash download_all_cyprus_exams.sh
```

For more details, see the documentation in:
- [Kodiko Downloader](scraping/download_and_extract_scripts/README_KODIKO.md)
- [Greek Language Downloader](scraping/download_and_extract_scripts/README_GREEK_LANGUAGE.md)
- [Cyprus Exams Downloader](scraping/download_and_extract_scripts/README_CYPRUS_EXAMS.md)

## Contributing

Contributions to GlossAPI are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

Current areas for contribution include:
- Improving PDF downloaders for various sources
- Enhancing text extraction quality
- Developing section classifiers for different document types
- Adding support for additional languages

For those specifically interested in contributing to the downloaders, please see our [DOWNLOADER_CONTRIBUTION.md](scraping/download_and_extract_scripts/DOWNLOADER_CONTRIBUTION.md) guide.

## License

This project is licensed under the European Union Public Licence 1.2 (EUPL 1.2).
