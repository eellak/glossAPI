import re
import os
import json
from typing import List, Tuple, Dict, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class Section:
    """
    A data structure representing a section in an academic document.
    
    Attributes:
        title (str): The section title
        start_line (int): The starting line number in the original document
        end_line (int): The ending line number in the original document
        content (List[Dict]): List of content elements. Each element is a dict with one of these keys:
            - 'text': Regular text content including empty lines
            - 'table': Table content in markdown format
            - 'list': List items with their continuation lines
            - 'other': Standalone references, image placeholders, etc.
        raw_content (str): Raw text content of the section (unprocessed)
        has_table (bool): Flag indicating if section contains tables
        has_list (bool): Flag indicating if section contains lists
        has_text (bool): Flag indicating if section contains regular text
        has_other (bool): Flag indicating if section contains other content (refs, images, etc)
    """
    def __init__(self, title: str = "", start_line: int = 0):
        self.title = title
        self.start_line = start_line
        self.end_line = start_line
        self.content = []
        self.raw_content = ""
        self.has_table = False
        self.has_list = False
        self.has_other = False
        self.has_text = False
    
    def add_content(self, content_type: str, content_value: str):
        """Add a content element to this section"""
        # Create a dictionary with the content type as the key
        content_dict = {content_type: content_value}
        self.content.append(content_dict)
        
        # Update flags based on content type
        if content_type == "table":
            self.has_table = True
        elif content_type == "list":
            self.has_list = True
        elif content_type == "other":
            self.has_other = True
        elif content_type == "text":
            self.has_text = True


class GlossSection:
    """
    A class for sectioning, processing, and exporting academic document sections to Parquet format.
    Handles parsing markdown documents, identifying structural elements like headers, tables, 
    lists, and footnotes, and processes them for further analysis.
    """
    
    def _is_list_bullet_line(self, line: str) -> bool:
        """
        Check if a line indicates a bullet item.
        Examples:
        - 1. text
        -  text
        -  text
        1. text
        etc.

        We'll unify them with a small regex set.
        """
        test = line.strip()
        if not test:
            return False

        # This pattern matches lines that begin with:
        #  - optional dash, then optional digits, then optional bullet symbols
        #  - final check for '.' or ' ' => bullet indicator
        # e.g. "- 1. ", "-  ", "- ", "1. ", "2."
        bullet_pat = re.compile(r'''
            ^
            (
            -\s*\d*\.?\s*[\u2022\u2023\u25E6\u00BB\u2023]*  # dash + optional digits + period + bullet char
            |\d+\.\s+
            |-\s*
            )
            .*
            ''', re.VERBOSE)
        return bool(bullet_pat.match(test))

    def _looks_like_list_paragraph(self, para: str) -> bool:
        """
        Check if a paragraph is marked as a bullet block by our sentinel.
        """
        return para.startswith("<<__LIST_ITEM__>>")

    ###############################################################################
    # 1) Other Utility Functions
    ###############################################################################
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to a specified width while preserving words."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            # +1 for space if not first in line
            if current_length + len(word) + (1 if current_line else 0) <= width:
                current_line.append(word)
                current_length += len(word) + (1 if current_line else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines

    def _is_standalone_reference(self, para: str, min_text_length: int = 10) -> bool:
        """
        Determine if a paragraph appears to be a standalone reference/footnote marker.
        Only very short paragraphs (fewer than a threshold number of characters) that
        consist solely of reference-style tokens are flagged.

        Parameters:
        - para: The paragraph (as a string) to check.
        - min_text_length: A lower bound (in characters) below which the paragraph
                            is considered too short to be meaningful text.
        
        Returns:
        True if the paragraph appears to be just a reference marker.
        """
        trimmed = para.strip()
        
        # Empty lines should never be considered footnotes
        if len(trimmed) == 0:
            return False

        # Only match if the entire paragraph exactly equals one of the expected reference tokens.
        reference_patterns = [
            #r'^\d+$',                  # Only digits (e.g., "12")
            #r'^\d+[\-–]\d+$',          # A simple digit range (e.g., "12-14")
            r'^(Ibid|op\.cit\.?|loc\.cit\.?|et\.al\.?|cf\.)$',  # Common citation markers
            r'^(βλ\.|πρβλ\.|σσ\.|σελ\.|ό\.π\.)$',             # Greek shorthand markers
        ]
        
        # Try each pattern; if one matches the entire trimmed paragraph, flag it.
        for pattern in reference_patterns:
            if re.match(pattern, trimmed, re.IGNORECASE):
                return True

        # Otherwise, we do not consider it a standalone reference.
        return False

    def _detect_other_lines(self, paragraphs: List[str], max_length: int = 20, min_text_length: int = 20) -> List[Dict]:
        """
        Identify short paragraphs that should be categorized as "other" content rather than regular text.
        
        This function simply categorizes very short lines as "other" and everything else as "text".
        
        Parameters:
        - paragraphs: List of paragraph strings.
        - max_length: Maximum length (in characters) a paragraph can have
                     to be considered for the "other" category.
        - min_text_length: Not currently used.
        
        Returns:
        A list of dictionaries with the content type as key and content as value.
        """
        categorized = []
        for para in paragraphs:
            trimmed = para.strip()
            
            # Simply categorize short lines as "other"
            if len(trimmed) > 0 and len(trimmed) < max_length:
                categorized.append({"other": para})
            else:
                # Regular text content or empty lines
                categorized.append({"text": para})
        return categorized

    def _should_merge_paragraphs(self, para1: str, para2: str) -> bool:
        """
        Decide if para1 and para2 likely form a single continued sentence.
        """
        if not para1 or not para2:
            return False
        
        p1_end = para1.rstrip()
        p2_start = para2.lstrip()
        
        # Hyphen or open parenthesis
        if p1_end.endswith('-') or p1_end.endswith('('):
            return True
        
        end_char_1 = p1_end[-1] if p1_end else ''
        start_char_2 = p2_start[0] if p2_start else ''
        
        # e.g. ends with lower, next starts with lower => likely a single sentence
        if end_char_1.islower() and start_char_2.islower():
            return True
        # ends with punctuation and next starts with lower
        if end_char_1 in ',:·' and start_char_2.islower():
            return True
        # ends with digit, next starts with '°'
        if end_char_1.isdigit() and start_char_2 == '°':
            return True
        
        return False

    def _is_table_line(self, line: str) -> bool:
        """Check if the line (stripped) starts & ends with '|' => table line."""
        ls = line.strip()
        return ls.startswith("|") and ls.endswith("|") if ls else False

    def _looks_like_table_block(self, paragraph: str) -> bool:
        """
        If every non-blank line in paragraph starts & ends with '|', treat as a table block.
        """
        lines = paragraph.splitlines()
        for ln in lines:
            ln_str = ln.strip()
            if ln_str and (not (ln_str.startswith("|") and ln_str.endswith("|"))):
                return False
        return True

    def _is_header(self, line: str) -> bool:
        """Check if line is a markdown header (#...)."""
        return line.strip().startswith('#')

    def _extract_section_level(self, line: str) -> Tuple[int, str]:
        """Extract header level and title from a markdown header line."""
        match = re.match(r'^(#+)\s*(.+)$', line.strip())
        if match:
            level = len(match.group(1))  # Count the number of # symbols
            title = match.group(2)
            return level, title
        return 0, line

    def _process_sections(self, lines: List[str]) -> List[Section]:
        """
        Process text to identify sections based on headers.
        Text between two headers becomes a section with the preceding header as title.
        This only divides the document into sections based on headers - content
        categorization happens in _process_section_content.
        
        Enhanced to handle:
        1. Documents that start with content before the first header
        2. Documents with no headers at all
        3. Content after the last header
        
        Parameters:
        - lines: List of text lines from the document
        
        Returns:
        List of Section objects representing the document structure
        """
        sections = []
        current_section = None
        n = len(lines)
        found_any_headers = False
        
        # Store raw lines between headers
        raw_section_lines = []
        
        # Handle case 1: Document starts with content before any header
        # Create an initial section if the first line is not a header
        if n > 0 and not self._is_header(lines[0].strip()):
            # Use first line as title if it's not empty, otherwise use "Document"
            first_line = lines[0].strip() if lines[0].strip() else "Document"
            current_section = Section(title=first_line, start_line=0)
        
        i = 0
        while i < n:
            raw_line = lines[i].rstrip('\n')
            
            # Markdown heading - start of a new section
            if self._is_header(raw_line.strip()):
                found_any_headers = True
                
                # If we had a previous section, finalize it
                if current_section is not None:
                    current_section.end_line = i - 1
                    
                    # Store raw section content
                    current_section.raw_content = "\n".join(raw_section_lines)
                    raw_section_lines = []
                    
                    sections.append(current_section)
                
                # Create a new section based on the header
                _, title = self._extract_section_level(raw_line)
                current_section = Section(title=title, start_line=i)
                i += 1
                continue
            
            # Just store the raw line - content processing happens later
            if current_section is not None:
                raw_section_lines.append(raw_line)
            else:
                # This should generally not happen since we create an initial section if needed,
                # but in case first_line logic changes, keep this safety check
                raw_section_lines.append(raw_line)
            
            i += 1

        # Handle case 2 & 3: Document has no headers or content after the last header
        # Finalize the last section if there is one
        if current_section:
            current_section.end_line = n - 1
            current_section.raw_content = "\n".join(raw_section_lines)
            sections.append(current_section)
        elif raw_section_lines:  # Handle case where no section was created but we collected content
            first_line = raw_section_lines[0].strip() if raw_section_lines and raw_section_lines[0].strip() else "Document"
            default_section = Section(title=first_line, start_line=0)
            default_section.end_line = n - 1
            default_section.raw_content = "\n".join(raw_section_lines[1:] if len(raw_section_lines) > 1 else raw_section_lines)
            sections.append(default_section)
        
        # Handle case 2: If no headers were found and we have no sections yet, create a default section
        if not found_any_headers and not sections and n > 0:
            title = lines[0].strip() if lines[0].strip() else "Document"
            content = "\n".join(lines[1:] if len(lines) > 1 else lines)
            default_section = Section(title=title, start_line=0)
            default_section.end_line = n - 1
            default_section.raw_content = content
            sections.append(default_section)
        
        return sections

    def _process_section_content(self, sections: List[Section]):
        """
        Process the raw content of each section to categorize it into appropriate content types:
        1. Tables: Identified by markdown table formatting (|)
        2. Lists: Identified by bullet points or numbered items
        3. Other: Standalone references, image placeholders, etc.
        4. Text: All remaining content, including empty lines
        
        The original structure with line breaks is preserved within each content block.
        
        Parameters:
        - sections: List of Section objects to process
        """
        for section in sections:
            # Clear existing content and start fresh from raw content
            section.content = []
            
            # Split raw content into lines for processing
            if not section.raw_content:
                continue
                
            raw_lines = section.raw_content.split('\n')
            i = 0
            n = len(raw_lines)
            
            # Buffer to collect text content including empty lines
            text_buffer = []
            
            while i < n:
                line = raw_lines[i]
                
                # 1. Check for tables (lines with | at start and end)
                if self._is_table_line(line):
                    # Flush any text buffer first
                    if text_buffer:
                        section.add_content("text", "\n".join(text_buffer))
                        text_buffer = []
                        
                    # Collect all table lines
                    table_lines = [line]
                    i += 1
                    while i < n and self._is_table_line(raw_lines[i]):
                        table_lines.append(raw_lines[i])
                        i += 1
                    # Add as table content
                    section.add_content("table", "\n".join(table_lines))
                    continue
                
                # 2. Check for list items
                elif self._is_list_bullet_line(line):
                    # Flush any text buffer first
                    if text_buffer:
                        section.add_content("text", "\n".join(text_buffer))
                        text_buffer = []
                        
                    # Collect the list item and any continuation lines
                    list_item = [line]
                    i += 1
                    while i < n:
                        next_line = raw_lines[i]
                        if (not next_line.strip() or 
                            self._is_list_bullet_line(next_line) or
                            self._is_table_line(next_line) or
                            self._is_header(next_line)):
                            break
                        # Add continuation line preserving its formatting
                        list_item.append(next_line)
                        i += 1
                    # Add as list content, preserving line breaks
                    section.add_content("list", "\n".join(list_item))
                    continue
                
                # 3. Check for 'other' content (standalone refs, image placeholders, etc)
                elif self._detect_other_lines([line])[0].get('other'):
                    # Flush any text buffer first
                    if text_buffer:
                        section.add_content("text", "\n".join(text_buffer))
                        text_buffer = []
                        
                    section.add_content("other", line)
                    i += 1
                    continue
                
                # 4. Regular text content and empty lines - add to buffer
                else:
                    # Add to text buffer (preserves empty lines and formatting)
                    text_buffer.append(line)
                    i += 1
            
            # Don't forget to add any remaining text in buffer
            if text_buffer:
                section.add_content("text", "\n".join(text_buffer))
            
            # Update section flags based on content
            section.has_table = any("table" in item for item in section.content)
            section.has_list = any("list" in item for item in section.content)
            section.has_text = any("text" in item for item in section.content)
            section.has_other = any("other" in item for item in section.content)
            

    
    def _format_academic_document(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Process a document and format it into structured data for output.
        
        Parameters:
        - text: The text content of the document
        - filename: The filename of the document
        
        Returns:
        A list of dictionaries with structured section data for Parquet output
        """
        lines = text.splitlines()
        
        # 1) Identify sections in the document based on markdown headers
        sections = self._process_sections(lines)
        
        # 2) Process section content - categorize each line appropriately
        self._process_section_content(sections)
        
        # 3) Format the data for output
        rows = []
        for section in sections:
            # Calculate section position as fraction of total document
            start_frac = section.start_line / max(1, len(lines))
            end_frac = section.end_line / max(1, len(lines))
            place_str = f"{start_frac:.2f}-{end_frac:.2f}"
            
            # Create a list of dictionaries for JSON serialization
            json_items = []
            for item in section.content:
                # Each item is a dict with a single key (content type) and value
                content_type = list(item.keys())[0]
                content_value = item[content_type]
                
                # Create proper dictionary object for JSON serialization
                json_items.append({content_type: content_value})
            
            # Use Python's json module for proper JSON serialization with all escaping handled
            json_content = json.dumps(json_items, ensure_ascii=False, indent=2)
            
            row = {
                "id": len(rows),
                "filename": filename,
                "has_table": section.has_table,
                "has_list": section.has_list,
                "has_other": section.has_other,
                "has_text": section.has_text,
                "header": section.title.strip(),
                "place": place_str,
                "section": section.raw_content,  # Store the original unprocessed section text
                "json_section": json_content  # Store the formatted JSON content
            }
            rows.append(row)
            
        return rows
    
    def to_parquet(self, input_dir, output_dir, filenames_to_process):
        """
        Process Markdown files from input_dir and write structured data to a Parquet file.
        
        Args:
            input_dir (str): Directory containing Markdown files to process
            output_dir (str): Directory where the output Parquet file will be written
            filenames_to_process (list): List of filenames (without extensions) to process.
                Only files matching these names will be processed.
                This should be a list of base filenames without extensions.
            
        The output Parquet file will contain structured data about sections from all documents,
        including information about tables, lists, footnotes, and regular text.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        parquet_path = os.path.join(output_dir, "sections_for_annotation.parquet")
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("row_id", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("has_table", pa.bool_()),
            pa.field("has_list", pa.bool_()),
            pa.field("has_other", pa.bool_()),
            pa.field("has_text", pa.bool_()),
            pa.field("header", pa.string()),
            pa.field("place", pa.string()),
            pa.field("section", pa.string()),  # Raw section text
            pa.field("json_section", pa.string()),  # Formatted JSON content
            pa.field("section_length", pa.int64()),  # Number of non-empty lines in section
            pa.field("section_propo", pa.int64()),  # Proportion of document (0-1000)
        ])
        
        writer = pq.ParquetWriter(parquet_path, schema=schema)
        row_counter = 1  # global row id counter

        # Process each Markdown file individually to keep memory usage low.
        processed_files_count = 0
        skipped_files = []
        print(f"\n===== SECTIONING PHASE =====")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Good files list (length {len(filenames_to_process)}): {filenames_to_process}")
        print(f"Available files in directory:")
        md_files = [f for f in os.listdir(input_dir) if f.endswith(".md")]
        for i, md_file in enumerate(md_files):
            print(f"  {i+1}. {md_file} (basename: {os.path.splitext(md_file)[0]})")
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".md"):
                # Get the base name without extension for filtering
                base_name = os.path.splitext(filename)[0]
                
                # Only process files that are in our whitelist
                if base_name not in filenames_to_process:
                    skipped_files.append(base_name)
                    print(f"⚠️ SKIPPED: {base_name} - not in the good files list")
                    continue  # Skip this file as it's not in our list of good files
                
                processed_files_count += 1
                print(f"✅ PROCESSING: {base_name} - in good files list")
                input_path = os.path.join(input_dir, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                short_name = os.path.splitext(filename)[0]
                doc_rows = self._format_academic_document(text, short_name)
                
                # Calculate section_length for each row (number of non-empty lines)
                for row in doc_rows:
                    section_lines = row.get("section", "").splitlines()
                    section_length = sum(1 for line in section_lines if line.strip())
                    row['section_length'] = section_length
                
                # Calculate the total article length (sum of all section lengths)
                article_length = sum(row.get("section_length", 0) for row in doc_rows)
                
                # Calculate section_propo for each row (proportion * 1000, rounded)
                for row in doc_rows:
                    if article_length > 0:
                        section_propo = round((row.get("section_length", 0) / article_length) * 1000)
                    else:
                        section_propo = 0
                    row['section_propo'] = section_propo
                
                # Add id and row_id to each row
                for row in doc_rows:
                    row['id'] = row_counter
                    row['row_id'] = f'row_{row_counter}'
                    row_counter += 1
                
                if doc_rows:
                    df = pd.DataFrame(doc_rows)
                    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                    writer.write_table(table)
        
        writer.close()
        
        # More informative logging
        print(f"\nSection processing summary:")
        print(f"  - Good files list contained {len(filenames_to_process)} files: {filenames_to_process}")
        print(f"  - Found {processed_files_count} markdown files matching good files list")
        if skipped_files:
            print(f"  - Skipped {len(skipped_files)} files that weren't in good list: {skipped_files}")
        print(f"  - Saved {row_counter - 1} total sections to {parquet_path}")
