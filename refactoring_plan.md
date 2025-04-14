# GlossAPI Refactoring Plan

## Overview
This document outlines the planned changes to the GlossAPI section classification pipeline, focusing on simplifying the section processing logic and changing the output structure.

## Key Changes

### 1. Simplification of Section Processing
- Rename `_process_academic_text_with_positions` to `_process_sections` in all places in the code
- Replace hierarchical section processing with flat processing:
  - Find text between two headers and define it as a section
  - Use the header above as the section's header
  - Process all markdown headers flatly instead of maintaining a hierarchical structure
- **Important**: Maintain the existing functionality that protects lists and tables from cleaning and reformatting by detecting them and processing them differently

### 2. Changes to Output Schema
- Remove the following columns from to_parquet:
  - `label` (string)
  - `section_propo` (int64)
  - `section_length` (int64)
- Remove all related functionality for calculating `section_propo` and `section_length`

### 3. Section Content Structure Changes
- Modify the logic in both `academic_section.py` and `gloss_section.py` 
- Return sections as JSON objects that contain, in the order they appear in the text, entries with keys:
  - "text" - for regular text content
  - "table" - for table content
  - "list" - for list content
  - "footnote" - for footnote content
- Instead of deleting footnotes, annotate them appropriately
- **Keep** the existing flags (`has_table`, `has_list`) in the output schema
- **Add** new flags `has_footnote` and `has_text` to indicate presence of those content types
- Implement detection logic to identify if a section contains non-empty lines that don't belong to tables, lists, or footnotes (for the `has_text` flag)

### 4. Implementation Plan
1. First, create new versions of the modules with the updated functionality
2. Ensure all dependencies and references are updated
3. Make sure the section processing works with these simplified changes
4. Test the pipeline with sample documents

## Files to be Changed
- `/mnt/data/glossAPI/pipeline/src/glossapi/gloss_section.py`
- `/mnt/data/glossAPI/pipeline/src/glossapi/academic_section.py`
- Any other files that reference the renamed functions or changed outputs
