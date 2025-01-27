#!/usr/bin/env python3
import os
import re
import csv
from typing import List, Tuple

###############################################################################
# 0) List Handling (same logic as first script)
###############################################################################
def is_list_bullet_line(line: str) -> bool:
    """
    Check if a line indicates a bullet item.
    Examples:
      - 1. text
      -  text
      -  text
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
    # e.g. "- 1. ", "-  ", "- ", "1. ", "2."
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

def looks_like_list_paragraph(para: str) -> bool:
    """
    Check if a paragraph is marked as a bullet block by our sentinel.
    """
    return para.startswith("<<__LIST_ITEM__>>")


###############################################################################
# 1) Other Utility Functions (same or similar to first script)
###############################################################################
def wrap_text(text: str, width: int) -> List[str]:
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

def detect_footnotes(paragraphs: List[str]) -> List[str]:
    """
    Remove paragraphs matching known footnote or page-number patterns.
    Similar to your first script.
    """
    footnote_patterns = [
        r'^\s*\d+\s*$',
        r'^\s*\d+[\-–]\d+\s*$',
        r'^\s*(Ibid|op\.cit\.?|loc\.cit\.?|et\.al\.?|cf\.)\s*$',
        r'^\s*(βλ\.|πρβλ\.|σσ\.|σελ\.|ό\.π\.)\s*$',
        r'^\s*[\d\s\.,\-–]+\s*$',
        r'^\s*[\.,\-–]+\s*$',
        r'^\s*,\s*σσ\.\s*\d+.*$',
    ]
    footnote_pattern = re.compile('|'.join(footnote_patterns), re.IGNORECASE)
    
    filtered = []
    for para in paragraphs:
        if footnote_pattern.match(para):
            continue
        # If very short with almost no letters, skip
        if len(para) < 20 and not any(c.isalpha() for c in para):
            continue
        filtered.append(para)
    return filtered

def should_merge_paragraphs(para1: str, para2: str) -> bool:
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

def is_table_line(line: str) -> bool:
    """Check if the line (stripped) starts & ends with '|' => table line."""
    ls = line.strip()
    return ls.startswith("|") and ls.endswith("|") if ls else False

def looks_like_table_block(paragraph: str) -> bool:
    """
    If every non-blank line in paragraph starts & ends with '|', treat as a table block.
    """
    lines = paragraph.splitlines()
    for ln in lines:
        ln_str = ln.strip()
        if ln_str and (not (ln_str.startswith("|") and ln_str.endswith("|"))):
            return False
    return True

def is_header(line: str) -> bool:
    """Check if line is a markdown header (#...)."""
    return line.strip().startswith('#')

def extract_section_level(line: str) -> Tuple[int, str]:
    match = re.match(r'^(#+)\s*(.+)$', line.strip())
    if match:
        level = len(match.group(1)) - 1  # 0-based
        title = match.group(2)
        return level, title
    return 0, line

###############################################################################
# 2) AcademicSection (stores line-range info) - same as second script
###############################################################################
class AcademicSection:
    def __init__(self, 
                 level: int = 0, 
                 title: str = "", 
                 start_line: int = 0):
        self.level = level
        self.title = title
        self.start_line = start_line
        self.end_line = start_line
        self.content: List[str] = []
        self.subsections: List['AcademicSection'] = []
        
        self.is_bibliography = False

    def add_subsection(self, subsection: 'AcademicSection'):
        self.subsections.append(subsection)

###############################################################################
# 3) Parsing with line positions (extended to detect bullet lines, like first script)
###############################################################################
def process_academic_text_with_positions(lines: List[str]) -> AcademicSection:
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
        if is_header(raw_line.strip()):
            # finalize the previous section's end_line
            section_stack[-1].end_line = i - 1
            # flush current content
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            level, title = extract_section_level(raw_line)
            
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
        if is_table_line(raw_line):
            # flush any "normal" text
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            table_lines = []
            while i < n and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            # store the table block
            section_stack[-1].content.append("\n".join(table_lines))
            continue
        
        # (C) List/bullet detection
        if is_list_bullet_line(raw_line):
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
                    is_header(nxt.strip()) or
                    is_table_line(nxt) or
                    is_list_bullet_line(nxt)):
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
def process_section_paragraphs(section: AcademicSection):
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
            if looks_like_table_block(para) or looks_like_list_paragraph(para):
                filtered.append(para)
            else:
                tmp = detect_footnotes([para])
                if tmp:
                    filtered.extend(tmp)
        
        # 4B) Merge paragraphs
        merged = []
        i = 0
        while i < len(filtered):
            currp = filtered[i]
            
            # skip merges for table or list
            if looks_like_table_block(currp) or looks_like_list_paragraph(currp):
                merged.append(currp)
                i += 1
                continue
            
            if i < len(filtered) - 1:
                nxt = filtered[i + 1]
                # don't merge if next is table or list
                if (not looks_like_table_block(nxt)
                    and not looks_like_list_paragraph(nxt)
                    and should_merge_paragraphs(currp, nxt)):
                    merged.append(currp + " " + nxt)
                    i += 2
                    continue
            
            merged.append(currp)
            i += 1
        
        section.content = merged
    
    for sub in section.subsections:
        process_section_paragraphs(sub)

###############################################################################
# 5) Collect data for CSV (removing the sentinel, plus new has_table, has_list)
###############################################################################
def collect_section_data(section: AcademicSection,
                         all_lines_count: int,
                         filename: str,
                         rows: List[dict]):
    """
    Recursively collect rows from each sub-section.

    For each sub-section, we produce one CSV row with columns:
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
        has_table = any(looks_like_table_block(p) for p in sub.content)
        has_list  = any(looks_like_list_paragraph(p) for p in sub.content)
        
        # Remove the sentinel in bullet paragraphs before building final section text
        final_paragraphs = []
        for para in sub.content:
            if looks_like_list_paragraph(para):
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
        collect_section_data(sub, all_lines_count, filename, rows)

###############################################################################
# 6) Main pipeline: parse -> post-process -> collect -> write CSV
###############################################################################
def format_academic_document_with_positions(text: str, filename: str) -> List[dict]:
    """
    This wraps up all the steps to produce CSV rows for a single document:
      - parse with bullet detection
      - remove footnotes, merge paragraphs
      - collect heading data with line positions
    """
    lines = text.splitlines()
    
    # 1) Build the structure (headings, bullets, etc.)
    doc_root = process_academic_text_with_positions(lines)
    # 2) Post-process merges, footnotes
    process_section_paragraphs(doc_root)
    # 3) Collect data for CSV
    rows: List[dict] = []
    collect_section_data(doc_root, len(lines), filename, rows)
    return rows

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'well_extracted_sample')
    output_dir = os.path.join(base_dir, 'well_extracted_sample_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # We'll store a CSV for all processed sections
    csv_path = os.path.join(output_dir, "sections_for_annotation.csv")
    
    all_rows = []
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            short_name = os.path.splitext(filename)[0]
            # produce the structured data
            doc_rows = format_academic_document_with_positions(text, short_name)
            all_rows.extend(doc_rows)
    
    # Add row_id to each row
    for idx, row in enumerate(all_rows, start=1):
        row['row_id'] = f'row_{idx}'
    
    # Note the extra columns "row_id", "has_table" and "has_list" right after "filename"
    fieldnames = [
        "row_id",
        "filename",
        "has_table",
        "has_list",
        "header",
        "place",
        "section",
        "label"
    ]
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    
    print(f"Saved {len(all_rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
