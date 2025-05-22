# GlossAPI Pipeline Refactoring TODO List

## Build and Install

**IMPORTANT:** After implementing the changes, you need to build and install the package in the virtual environment for the changes to take effect:

```bash
# Activate the virtual environment first
source /mnt/data/venv/bin/activate

# Go to the pipeline directory
cd /mnt/data/glossAPI/pipeline

# Install the package in development mode
pip install -e .

# Now you can run the simple_test.py script
python /mnt/data/simple_test.py
```

**ALWAYS KEEP IN MIND:** The pipeline must work with the existing interface in simple_test.py using the "corpus.command()" pattern.

## ✅ COMPLETED

### 1) Modified GlossDownloader

- Updated `GlossDownloader` class to use a dedicated "downloads" folder:
  - Modified the `download_files()` method to use `self.output_dir / "downloads"` instead of `self.input_dir`
  - All downloaded files are now saved in this subdirectory
  - Updated the `Corpus.download()` method to create and use this downloads folder
  - Added validation to check if downloaded files are of the supported types (pdf, docx, xml, html, pptx, csv, md)

### 2) Updated GlossExtract

- Modified the `extract()` method in `Corpus` class to:
  - Look for files in the "downloads" directory first
  - If "downloads" directory doesn't exist, check for supported file types in the input folder and move them to a new "downloads" folder
  - Continue processing from the "downloads" folder
  - Updated the file location handling across the pipeline to reflect this change

### 3) Created a Standardized Parquet Class

- Created a new file called `parquet_schema.py` with a standardized schema class:
  - Defined required metadata fields for processing
  - Implemented standard schemas for different pipeline stages
  - Defined standard columns (id, row_id, filename, title, section, predicted_section)
  - Added methods for reading/writing with standard schema validation

### 4) Improved Bad File Filtering in Sectioning

- Made `filenames_to_process` a required parameter in section.py
- Enhanced filtering to ensure only good files (based on extraction quality in parquet) are processed
- Added detailed logging for processed and skipped files
- Verified that section.py correctly handles all sectioning scenarios:
  - Text between two headers
  - Text before the first header
  - Text after the last header
  - Documents with no headers at all
- Fixed indentation issues in corpus.py that were causing execution problems

## TODO

### 1) Finish Removing Redundant Code

- Remove the remaining redundant code related to good/bad folders:
  - The `extract_quality` method in corpus.py still deals with good/bad folders
  - Remove all code related to copying files to good/bad directories
  - Remove references to `good_markdown_dir` since we're using extraction quality markers in parquet
  - Update all methods to use the simplified directory structure

### 2) Complete Two-Parquet Pipeline Implementation

**Progress**: We've successfully implemented the first parquet (downloader parquet with extraction quality) but need to consolidate the section-related parquets.

- Currently we still have 3 section parquet files that need to be consolidated:
  - `sections_for_annotation.parquet`
  - `classified_sections.parquet`
  - `fully_annotated_sections.parquet`

- Implementation tasks:
  - Consolidate the 3 section-related parquet files into a single sections parquet
  - Update all methods to work with the consolidated parquet structure
  - Ensure all metadata columns are preserved during consolidation
  - Add metadata column "processing_stage" to track progress through pipeline
  - Update the verification method to check for required columns rather than specific filenames
  - Throw clear error messages when required columns are missing

### 3) Make Split_Bad an Explicit Pipeline Step

- Extract the split_bad functionality from internal GlossExtract methods
- Create a dedicated method in Corpus class
- Make it explicitly update extraction quality in the downloader parquet
- Update the processing_stage column to include extraction as a completed stage

### 4) Remove All Fallback Options

- **Critical**: Remove any remaining code that silently falls back to processing all files:
  - Some of these fallbacks have been removed, but others may still exist
  - Remove any code that ignores extraction quality filter failures
  - Flag fallbacks as explicit errors rather than silent recovery
  - Ensure section() and other methods require good quality files and don't have hidden fallbacks

### 5) Add More Robust Error Messages

- Add clear error messages when filtering operations fail instead of using defaults
  - For example: "No good quality files found. Pipeline stopped." instead of using all files
- Document all pipeline decision points in code comments
  - Specify where the pipeline can branch and under what conditions
  - Explain the rationale for each decision point

### 6) Testing and Documentation

- Test the refactored pipeline using the examples in /mnt/data/eu_test
- Ensure the extraction_test_bad_file.py script correctly filters bad files
- Add detailed logging for all pipeline stages
- Document the new two-parquet approach in comments and docstrings
- Update the parquet schema documentation to reflect the new approach
