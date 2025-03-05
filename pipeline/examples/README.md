# GlossAPI Examples

This directory contains example scripts demonstrating how to use the GlossAPI library.

## Scripts

### `process_markdown_repository.py`

Demonstrates how to process a repository of existing markdown files without going through the PDF extraction step.

Usage:
```bash
python process_markdown_repository.py --input-dir /path/to/markdown_files --output-dir /path/to/output [--metadata-path /path/to/metadata.parquet] [--annotation-type auto|text|chapter]
```

For processing large repositories, you may want to run the script as a background process:

```bash
nohup python process_markdown_repository.py --input-dir /path/to/markdown_files --output-dir /path/to/output > process.log 2>&1 &
```

This will:
1. Run the script detached from the terminal
2. Continue processing even if you close your terminal session
3. Redirect all output to `process.log`

## Output Structure

All example scripts produce the same output structure:

```
output_dir/
├── quality_clustering/
│   ├── good/            # High-quality markdown files
│   └── bad/             # Low-quality markdown files
├── sections/
│   └── sections_for_annotation.parquet   # Extracted sections
├── annotated_sections/
│   └── annotated_sections.parquet   # Classified sections with basic annotation
└── fully_annotated_sections.parquet  # Final output with document structure
```
