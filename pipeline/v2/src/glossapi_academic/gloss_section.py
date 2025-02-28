import re
import os
from typing import List, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .academic_section import AcademicSection


class GlossSection:
    """
    A class for sectioning, processing, and exporting academic document sections to Parquet format.
    Handles parsing markdown documents, identifying structural elements like headers, tables, 
    and bullet lists, and processes them for further analysis.
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
            -\s*\d*\.?\s*[\u2022\u2023\u25E6\u00BB\u2023]*  # dash + optional digits + period + bullet char
            |\d+\.\s+
            |-\s*[]
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
        
        # If the paragraph is extremely short, assume it is a candidate.
        if len(trimmed) < min_text_length:
            return True

        # Only match if the entire paragraph exactly equals one of the expected reference tokens.
        reference_patterns = [
            r'^\d+$',                  # Only digits (e.g., "12")
            r'^\d+[\-–]\d+$',          # A simple digit range (e.g., "12-14")
            r'^(Ibid|op\.cit\.?|loc\.cit\.?|et\.al\.?|cf\.)$',  # Common citation markers
            r'^(βλ\.|πρβλ\.|σσ\.|σελ\.|ό\.π\.)$',             # Greek shorthand markers
        ]
        
        # Try each pattern; if one matches the entire trimmed paragraph, flag it.
        for pattern in reference_patterns:
            if re.match(pattern, trimmed, re.IGNORECASE):
                return True

        # Otherwise, we do not consider it a standalone reference.
        return False

    def _detect_footnotes(self, paragraphs: List[str], footnote_max_length: int = 20, min_text_length: int = 20) -> List[str]:
        """
        Filter out paragraphs that are likely to be footnotes or reference markers.
        
        The function takes a less aggressive approach by only removing paragraphs that
        are both very short and match known standalone reference patterns.
        
        Parameters:
        - paragraphs: List of paragraph strings.
        - footnote_max_length: Maximum length (in characters) a paragraph can have
                                to be considered a standalone footnote.
        - min_text_length: Minimum number of characters below which we suspect the
                            paragraph is not part of the main text.
        
        Returns:
        A new list of paragraphs with the likely footnotes removed.
        """
        filtered = []
        for para in paragraphs:
            # Only consider removing if the paragraph is short.
            if len(para.strip()) < footnote_max_length and self._is_standalone_reference(para, min_text_length):
                # Skip this paragraph as it appears to be a footnote/reference.
                continue
            filtered.append(para)
        return filtered

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
        if end_char_1 in ',:' and start_char_2.islower():
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
        match = re.match(r'^(#+)\s*(.+)$', line.strip())
        if match:
            level = len(match.group(1)) - 1  # 0-based
            title = match.group(2)
            return level, title
        return 0, line

    ###############################################################################
    # 3) Parsing with line positions (extended to detect bullet lines)
    ###############################################################################
    def _process_academic_text_with_positions(self, lines: List[str]) -> AcademicSection:
        """
        Similar to your original 'process_academic_text', but now we store
        start_line/end_line for each section and we detect bullet lines.
        """
        root = AcademicSection(level=0, title="", start_line=0)
        section_stack = [root]
        
        current_content: List[str] = []
        n = len(lines)
        
        i = 0
        while i < n:
            raw_line = lines[i].rstrip('\n')
            
            # (A) Markdown heading
            if self._is_header(raw_line.strip()):
                # finalize the previous section's end_line
                section_stack[-1].end_line = i - 1
                # flush current content
                if current_content:
                    section_stack[-1].content.append(" ".join(current_content))
                    current_content = []
                
                level, title = self._extract_section_level(raw_line)
                
                # pop up if needed
                while len(section_stack) > 1 and section_stack[-1].level >= level:
                    section_stack.pop()
                
                new_section = AcademicSection(level=level,
                                            title=title,
                                            start_line=i)
                
                # Mark if possibly bibliography
                if re.search(r'\b(ΒΙΒΛΙΟΓΡΑΦΙΑ|BIBLIOGRAPHY|REFERENCES)\b',
                            title, re.IGNORECASE):
                    new_section.is_bibliography = True
                
                section_stack[-1].add_subsection(new_section)
                section_stack.append(new_section)
                
                i += 1
                continue
            
            # (B) Table detection
            if self._is_table_line(raw_line):
                # flush any "normal" text
                if current_content:
                    section_stack[-1].content.append(" ".join(current_content))
                    current_content = []
                
                table_lines = []
                while i < n and self._is_table_line(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                # store the table block
                section_stack[-1].content.append("\n".join(table_lines))
                continue
            
            # (C) List/bullet detection
            if self._is_list_bullet_line(raw_line):
                # flush normal text paragraphs
                if current_content:
                    section_stack[-1].content.append(" ".join(current_content))
                    current_content = []
                
                bullet_lines = [raw_line]
                i += 1
                # gather bullet lines until next bullet, heading, table or blank
                while i < n:
                    nxt = lines[i].rstrip('\n')
                    if (not nxt.strip() or
                        self._is_header(nxt.strip()) or
                        self._is_table_line(nxt) or
                        self._is_list_bullet_line(nxt)):
                        break
                    bullet_lines.append(nxt)
                    i += 1
                
                # store as a single paragraph with sentinel
                single_bullet_para = "<<__LIST_ITEM__>>" + " ".join(bullet_lines)
                section_stack[-1].content.append(single_bullet_para)
                continue
            
            # (D) Normal line
            trimmed = raw_line.strip()
            if trimmed:
                current_content.append(trimmed)
            else:
                # blank => flush
                if current_content:
                    section_stack[-1].content.append(" ".join(current_content))
                    current_content = []
            i += 1
        
        # finalize last section
        section_stack[-1].end_line = n - 1
        if current_content:
            section_stack[-1].content.append(" ".join(current_content))
        
        # root covers from 0..(n-1)
        root.end_line = n - 1
        
        return root

    ###############################################################################
    # 4) Post-Processing: footnotes, merges, skipping merges for table or list blocks
    ###############################################################################
    def _process_section_paragraphs(self, section: AcademicSection):
        """
        - Remove footnotes from normal paragraphs (not tables or lists).
        - Merge paragraphs if `should_merge_paragraphs()` says so.
        - Don't merge table blocks or list blocks.
        - Recursively handle subsections.
        """
        if not section.is_bibliography and section.content:
            filtered = []
            # 4A) Remove footnotes except for table/list paragraphs
            for para in section.content:
                if self._looks_like_table_block(para) or self._looks_like_list_paragraph(para):
                    filtered.append(para)
                else:
                    tmp = self._detect_footnotes([para])
                    if tmp:
                        filtered.extend(tmp)
            
            # 4B) Merge paragraphs
            merged = []
            i = 0
            while i < len(filtered):
                currp = filtered[i]
                
                # skip merges for table or list
                if self._looks_like_table_block(currp) or self._looks_like_list_paragraph(currp):
                    merged.append(currp)
                    i += 1
                    continue
                
                if i < len(filtered) - 1:
                    nxt = filtered[i + 1]
                    # don't merge if next is table or list
                    if (not self._looks_like_table_block(nxt)
                        and not self._looks_like_list_paragraph(nxt)
                        and self._should_merge_paragraphs(currp, nxt)):
                        merged.append(currp + " " + nxt)
                        i += 2
                        continue
                
                merged.append(currp)
                i += 1
            
            section.content = merged
        
        for sub in section.subsections:
            self._process_section_paragraphs(sub)

    ###############################################################################
    # 5) Collect data for CSV (removing the sentinel, plus new has_table, has_list)
    ###############################################################################
    def _collect_section_data(self, section: AcademicSection,
                            all_lines_count: int,
                            filename: str,
                            rows: List[dict]):
        """
        Recursively collect rows from each sub-section.

        For each sub-section, we produce one row with columns:
        {
            "filename": str,
            "has_table": bool,
            "has_list": bool,
            "header": str,
            "place": str,    # e.g. "0.35-0.39"
            "section": str,  # final text (footnotes removed, bullet sentinel removed)
            "label": str
        }

        We detect has_table and has_list by examining the final paragraphs in sub.content:
        - If any paragraph is recognized as a table block, has_table=True
        - If any paragraph is recognized as a list block, has_list=True
        """
        for sub in section.subsections:
            start_frac = sub.start_line / max(1, all_lines_count)
            end_frac   = sub.end_line   / max(1, all_lines_count)
            place_str  = f"{start_frac:.2f}-{end_frac:.2f}"
            
            # Detect presence of table or list in this sub's content
            has_table = any(self._looks_like_table_block(p) for p in sub.content)
            has_list  = any(self._looks_like_list_paragraph(p) for p in sub.content)
            
            # Remove the sentinel in bullet paragraphs before building final section text
            final_paragraphs = []
            for para in sub.content:
                if self._looks_like_list_paragraph(para):
                    # Remove sentinel, e.g. "<<__LIST_ITEM__>>"
                    real_text = para.replace("<<__LIST_ITEM__>>", "", 1).strip()
                    final_paragraphs.append(real_text)
                else:
                    final_paragraphs.append(para)
            
            sub_text = "\n".join(final_paragraphs)
            
            row = {
                "filename": filename,
                "has_table": has_table,
                "has_list": has_list,
                "header": sub.title.strip(),
                "place": place_str,
                "section": sub_text,
                "label": ""  # optional
            }
            rows.append(row)
            
            # recurse deeper
            self._collect_section_data(sub, all_lines_count, filename, rows)

    ###############################################################################
    # 6) Main pipeline: parse -> post-process -> collect -> write Parquet
    ###############################################################################
    def _format_academic_document_with_positions(self, text: str, filename: str) -> List[dict]:
        """
        This wraps up all the steps to produce rows for a single document:
        - parse with bullet detection
        - remove footnotes, merge paragraphs
        - collect heading data with line positions
        """
        lines = text.splitlines()
        
        # 1) Build the structure (headings, bullets, etc.)
        doc_root = self._process_academic_text_with_positions(lines)
        # 2) Post-process merges, footnotes
        self._process_section_paragraphs(doc_root)
        # 3) Collect data for CSV/Parquet
        rows: List[dict] = []
        self._collect_section_data(doc_root, len(lines), filename, rows)
        return rows

    def to_parquet(self, input_dir, output_dir):
        """
        Process all Markdown files in input_dir and write structured data to a Parquet file.
        
        Args:
            input_dir (str): Directory containing Markdown files to process
            output_dir (str): Directory where the output Parquet file will be written
            
        The output Parquet file will contain structured data about sections from all documents,
        including information about tables, lists, section lengths, and proportions.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        parquet_path = os.path.join(output_dir, "sections_for_annotation.parquet")
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("row_id", pa.string()),
            pa.field("filename", pa.string()),
            pa.field("has_table", pa.bool_()),
            pa.field("has_list", pa.bool_()),
            pa.field("header", pa.string()),
            pa.field("place", pa.string()),
            pa.field("section", pa.string()),
            pa.field("label", pa.string()),
            pa.field("section_propo", pa.int64()),
            pa.field("section_length", pa.int64()),
        ])
        
        writer = pq.ParquetWriter(parquet_path, schema=schema)
        row_counter = 1  # global row id counter

        # Process each Markdown file individually to keep memory usage low.
        for filename in os.listdir(input_dir):
            if filename.endswith(".md"):
                input_path = os.path.join(input_dir, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                short_name = os.path.splitext(filename)[0]
                doc_rows = self._format_academic_document_with_positions(text, short_name)
                
                # For each row, compute section_length as the number of non-empty lines in the section.
                for row in doc_rows:
                    section_lines = row.get("section", "").splitlines()
                    section_length = sum(1 for line in section_lines if line.strip())
                    row['section_length'] = section_length
                
                # Compute article_length as the total number of lines across all sections in this file.
                article_length = sum(row.get("section_length", 0) for row in doc_rows)
                
                # Compute section_propo for each row: proportion (0-1) multiplied by 1000 and rounded.
                for row in doc_rows:
                    if article_length > 0:
                        section_propo = round((row.get("section_length", 0) / article_length) * 1000)
                    else:
                        section_propo = 0
                    row['section_propo'] = section_propo
                    row['id'] = row_counter
                    row['row_id'] = f'row_{row_counter}'
                    row_counter += 1
                
                if doc_rows:
                    df = pd.DataFrame(doc_rows)
                    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                    writer.write_table(table)
        
        writer.close()
        print(f"Saved {row_counter - 1} rows to {parquet_path}")
