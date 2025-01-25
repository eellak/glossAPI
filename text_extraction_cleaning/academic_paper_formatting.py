#!/usr/bin/env python3
import os
import re
from typing import List, Tuple

###############################################################################
# Text Wrapping
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

###############################################################################
# Footnote Detector
###############################################################################
def detect_footnotes(paragraphs: List[str]) -> List[str]:
    """
    Remove paragraphs matching known footnote or page-number patterns.
    """
    footnote_patterns = [
        r'^\s*\d+\s*$',               # Just a number
        r'^\s*\d+[\-–]\d+\s*$',       # Number range
        r'^\s*(Ibid|op\.cit\.?|loc\.cit\.?|et\.al\.?|cf\.)\s*$',
        r'^\s*(βλ\.|πρβλ\.|σσ\.|σελ\.|ό\.π\.)\s*$',
        r'^\s*[\d\s\.,\-–]+\s*$',     # Just numbers/punctuation
        r'^\s*[\.,\-–]+\s*$',         # Just punctuation
        r'^\s*,\s*σσ\.\s*\d+.*$',     # Greek page refs with commas
    ]
    footnote_pattern = re.compile('|'.join(footnote_patterns), re.IGNORECASE)
    
    filtered = []
    for p in paragraphs:
        # If entire paragraph matches footnote pattern, skip it
        if footnote_pattern.match(p):
            continue
        # If it's very short with almost no letters, skip it
        if len(p) < 20 and not any(c.isalpha() for c in p):
            continue
        filtered.append(p)
    
    return filtered

###############################################################################
# Paragraph Merge Logic
###############################################################################
def should_merge_paragraphs(para1: str, para2: str) -> bool:
    """
    Decide if para1 and para2 likely form a single continued sentence.
    Check how para1 ends and how para2 begins.
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

###############################################################################
# Table Handling
###############################################################################
def is_table_line(line: str) -> bool:
    """
    Strict definition: a line that starts and ends with '|' after stripping
    """
    ls = line.strip()
    return ls.startswith("|") and ls.endswith("|") if ls else False

def looks_like_table_block(paragraph: str) -> bool:
    """
    If each line in the paragraph (split by newlines) is a table line => block is a table
    """
    lines = paragraph.splitlines()
    return all(is_table_line(ln) for ln in lines if ln.strip())

###############################################################################
# List Handling
###############################################################################
def is_list_bullet_line(line: str) -> bool:
    """
    Check if line indicates a bullet item.
    Examples:
      - 1. text
      -  text
      -  text
      - text
      1. text
    We'll unify them with a small regex set.
    """
    # quick checks
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
    """Check if a paragraph is marked as a bullet block (see sentinel below)."""
    return para.startswith("<<__LIST_ITEM__>>")

###############################################################################
# AcademicSection Class
###############################################################################
class AcademicSection:
    def __init__(self, level: int = 0, title: str = "", content: List[str] = None):
        self.level = level
        self.title = title
        self.content = content or []
        self.subsections = []
        
        # If heading matches references, bibliography, etc., mark it
        self.is_bibliography = False

    def add_subsection(self, section):
        self.subsections.append(section)
        
    def __str__(self) -> str:
        indent = "    " * self.level
        result = []
        
        # Heading Format
        if self.title:
            if self.level == 0:
                result.append("\n" + "=" * 80 + "\n")
                result.append(f"{self.title.upper()}\n")
                result.append("=" * 80 + "\n")
            elif self.level == 1:
                result.append("\n" + "-" * 60 + "\n")
                result.append(f"{self.title}\n")
                result.append("-" * 60 + "\n")
            else:
                result.append(f"\n{indent}{self.title}\n")
                result.append(f"{indent}" + "-" * len(self.title) + "\n")
        
        # Content
        for paragraph in self.content:
            # Table block => keep lines intact
            if looks_like_table_block(paragraph):
                for line in paragraph.splitlines():
                    result.append(indent + line + "\n")
                result.append("\n")
            # List block => remove sentinel, keep as single paragraph
            elif looks_like_list_paragraph(paragraph):
                real_text = paragraph.replace("<<__LIST_ITEM__>>", "", 1).strip()
                wrapped = wrap_text(real_text, width=80 - len(indent))
                for w in wrapped:
                    result.append(indent + w + "\n")
                result.append("\n")
            else:
                # Normal paragraph
                wrapped = wrap_text(paragraph, width=80 - len(indent))
                for wline in wrapped:
                    result.append(indent + wline + "\n")
                result.append("\n")
        
        # Subsections
        for s in self.subsections:
            result.append(str(s))
            
        return "".join(result)

###############################################################################
# Detect Markdown Heading
###############################################################################
def is_header(line: str) -> bool:
    return line.strip().startswith('#')

def extract_section_level(line: str) -> Tuple[int, str]:
    match = re.match(r'^(#+)\s*(.+)$', line.strip())
    if match:
        level = len(match.group(1)) - 1
        ttl = match.group(2)
        return level, ttl
    return 0, line

###############################################################################
# Main Parsing: we detect headings, tables, bullet items, normal paragraphs
###############################################################################
def process_academic_text(text: str) -> AcademicSection:
    lines = text.splitlines()
    root = AcademicSection()
    section_stack = [root]
    
    current_content: List[str] = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i].rstrip('\n')
        
        # 1) Header check
        if is_header(line.strip()):
            # flush current content
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            lvl, ttl = extract_section_level(line)
            while len(section_stack) > 1 and section_stack[-1].level >= lvl:
                section_stack.pop()
            
            new_sec = AcademicSection(level=lvl, title=ttl)
            # is bibliography?
            if re.search(r'(ΒΙΒΛΙΟΓΡΑΦΙΑ|BIBLIOGRAPHY|REFERENCES)', ttl, re.IGNORECASE):
                new_sec.is_bibliography = True
            
            section_stack[-1].add_subsection(new_sec)
            section_stack.append(new_sec)
            i += 1
            continue
        
        # 2) Table check
        if is_table_line(line):
            # flush current text
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            # gather consecutive lines of table
            table_lines = []
            while i < n and is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            # store as single paragraph
            section_stack[-1].content.append("\n".join(table_lines))
            continue
        
        # 3) List check: bullet item
        if is_list_bullet_line(line):
            # flush normal text
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            # gather lines until next bullet or blank or table or heading
            bullet_lines = [line]
            i += 1
            while i < n:
                nxt = lines[i].rstrip('\n')
                # stop if new heading, table line, or new bullet, or blank
                if (is_header(nxt.strip())
                    or is_table_line(nxt)
                    or is_list_bullet_line(nxt)
                    or not nxt.strip()):
                    break
                bullet_lines.append(nxt)
                i += 1
            
            # store as single paragraph with sentinel
            # to skip merges later
            single_para = "<<__LIST_ITEM__>>" + " ".join(bullet_lines)
            section_stack[-1].content.append(single_para)
            continue
        
        # 4) Normal line
        trim = line.strip()
        if trim:
            current_content.append(trim)
        else:
            # blank => flush
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
        i += 1
    
    # flush leftover
    if current_content:
        section_stack[-1].content.append(" ".join(current_content))
    
    return root

###############################################################################
# Post-Processing: Merges, Footnotes
###############################################################################
def process_section_paragraphs(section: AcademicSection):
    """
    If not bibliography => remove footnotes, then merge paragraphs
    Skip merging if paragraph is a table block or a list block.
    """
    if not section.is_bibliography:
        # footnote removal except for table or list block
        filtered = []
        for para in section.content:
            if looks_like_table_block(para) or looks_like_list_paragraph(para):
                # keep as-is
                filtered.append(para)
            else:
                outp = detect_footnotes([para])
                filtered.extend(outp)  # might remove it or keep
                
        # now merges
        merged = []
        i = 0
        while i < len(filtered):
            currp = filtered[i]
            # skip merges for tables or list items
            if looks_like_table_block(currp) or looks_like_list_paragraph(currp):
                merged.append(currp)
                i += 1
                continue
            
            if i < len(filtered) - 1:
                nxtp = filtered[i+1]
                # do not merge if next is table or list item
                if (not looks_like_table_block(nxtp)) and (not looks_like_list_paragraph(nxtp)):
                    if should_merge_paragraphs(currp, nxtp):
                        merged.append(currp + " " + nxtp)
                        i += 2
                        continue
            merged.append(currp)
            i += 1
        section.content = merged
    
    # Recurse
    for s in section.subsections:
        process_section_paragraphs(s)

###############################################################################
# Final API
###############################################################################
def format_academic_document(text: str) -> str:
    doc = process_academic_text(text)
    process_section_paragraphs(doc)
    return str(doc)

###############################################################################
# File Processing
###############################################################################
def process_file(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    formatted_text = format_academic_document(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)

###############################################################################
# main()
###############################################################################
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'test_output')
    output_dir = os.path.join(base_dir, 'academic_output_2')
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    main()
