# GlossAPI Academic

A library for processing academic texts in Greek and other languages.

## Features

- Extract content from PDFs with Docling
- Cluster documents based on extraction quality
- Extract and clean academic sections
- Classify sections using machine learning
- Support for document-specific annotation based on document types

## Installation

```bash
pip install glossapi
```

## Usage

### Individual Components

```python
from glossapi_academic import GlossExtract, GlossSection, GlossSectionClassifier

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
from glossapi_academic import Corpus

# Initialize the Corpus with input and output directories
corpus = Corpus(
    input_dir="path/to/pdfs",
    output_dir="path/to/output",
    metadata_path="path/to/metadata.parquet",  # Optional: Provides document type info
    annotation_mapping={
        'Κεφάλαιο': 'chapter'  # Only specify non-default mappings
    }
)

# Run the complete pipeline
corpus.extract()  # Extract PDFs to markdown
corpus.section()  # Extract sections from markdown files

# Classify sections with document-type specific annotation
corpus.annotate(annotation_type="auto")  # Uses document type to determine annotation method

# Or force a specific annotation method for all documents
corpus.annotate(annotation_type="text")   # Use text annotation for all documents
corpus.annotate(annotation_type="chapter")  # Use chapter annotation for all documents
```

### Processing Existing Markdown Files

If you already have markdown files and want to skip the PDF extraction step, you can use:

```python
from glossapi_academic import Corpus

# Initialize with existing markdown directory
corpus = Corpus(
    input_dir="path/to/markdown_files",
    output_dir="path/to/output",
    metadata_path="path/to/metadata.parquet"  # Optional
)

# Filter markdown files by quality (separates into good/bad)
corpus.filter(input_dir="path/to/markdown_files")

# Process only the good quality files
corpus.section()

# Classify and annotate sections
corpus.annotate()
```

### Annotation Types

The GlossAPI supports different annotation methods based on document types:

1. **Text Annotation** (`'text'`): For full-length academic texts like books and articles
   - Identifies introduction (ε.σ.), prologue (π), main content (κ), bibliography (β), and appendix (α)
   - Requires both prologue (π) and bibliography (β) markers to properly structure the document

2. **Chapter Annotation** (`'chapter'`): For individual chapters or shorter texts
   - Treats all content as main content (κ) except for bibliography sections (β)
   - Doesn't require or expect an index/prologue (π) section

### Running as a Background Process

For processing large repositories, you may want to run the pipeline in the background:

```bash
nohup python process_script.py > processing_output.log 2>&1 &
```

This will:
1. Run the script detached from the terminal
2. Continue processing even if you close your terminal session
3. Redirect all output to a log file

## Model Training

```python
from glossapi_academic import GlossSectionClassifier

# Train a new model from CSV data
classifier = GlossSectionClassifier()
classifier.train_from_csv("path/to/training_data.csv", "section_text", "section_label")
classifier.save_model("path/to/output_model.joblib")
```

## Version

Current version: 0.0.3.5

## License

This project is licensed under the EUPL License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
