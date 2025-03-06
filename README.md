# GlossAPI Academic

[![Release Version](https://img.shields.io/github/v/release/eellak/glossAPI)](https://github.com/eellak/glossAPI/releases)
[![PyPI Test Status](https://img.shields.io/badge/PyPI%20Test-glossapi-blue?logo=pypi)](https://test.pypi.org/project/glossapi/)

A library for processing academic texts in Greek and other languages.

## Features

- Extract content from PDFs with Docling
- Cluster documents based on extraction quality
- Extract and clean academic sections
- Classify sections using machine learning

## Installation

```bash
pip install glossapi
```

## Usage

### Individual Components

```python
from glossapi import GlossExtract, GlossSection, GlossSectionClassifier

# Extract content from PDF
extractor = GlossExtract()
extracted_content = extractor.extract_from_pdf("path/to/document.pdf")

# Process sections
section_processor = GlossSection()
sections = section_processor.process(extracted_content)

# Classify sections
classifier = GlossSectionClassifier()
classifier.load_model("path/to/model.joblib")
classified_sections = classifier.classify(sections)
```

### Using the Corpus Class (Recommended)

```python
from glossapi import Corpus

# Initialize the Corpus with input and output directories
corpus = Corpus(
    input_dir="path/to/pdfs",
    output_dir="path/to/output"
)

# Run the complete pipeline
corpus.extract()  # Extract PDFs to markdown
corpus.section()  # Extract sections from markdown files
corpus.annotate()  # Classify sections using ML
```

## Model Training

```python
from glossapi import GlossSectionClassifier

# Train a new model from CSV data
classifier = GlossSectionClassifier()
classifier.train_from_csv("path/to/training_data.csv", "section_text", "section_label")
classifier.save_model("path/to/output_model.joblib")
```

## Version

Current version: 0.0.3.4

## License

This project is licensed under the EUPL License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
