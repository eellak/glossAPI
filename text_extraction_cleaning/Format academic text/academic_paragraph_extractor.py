#!/usr/bin/env python3
import os
import re
from typing import List, Dict, Tuple

class AcademicSection:
    def __init__(self, level: int = 0, title: str = "", content: List[str] = None):
        self.level = level
        self.title = title
        self.content = content or []
        self.subsections = []
        
    def add_subsection(self, section):
        self.subsections.append(section)
        
    def __str__(self) -> str:
        indent = "    " * self.level
        result = []
        
        # Format title based on level
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
        
        # Add content with proper indentation and spacing
        if self.content:
            for paragraph in self.content:
                # Format paragraph with proper indentation
                wrapped_lines = wrap_text(paragraph, width=80 - len(indent))
                formatted_para = "\n".join(indent + line for line in wrapped_lines)
                result.append(formatted_para + "\n\n")
        
        # Add subsections
        for subsection in self.subsections:
            result.append(str(subsection))
            
        return "".join(result)

def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to specified width while preserving words."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return lines

def extract_section_level(line: str) -> Tuple[int, str]:
    """Extract section level and clean title from markdown header."""
    match = re.match(r'^(#+)\s*(.+)$', line.strip())
    if match:
        level = len(match.group(1)) - 1  # -1 because we want 0-based levels
        title = match.group(2)
        return level, title
    return 0, line

def is_header(line: str) -> bool:
    """Check if line is a markdown header."""
    return line.strip().startswith('#')

def is_table_section(lines: List[str]) -> bool:
    """
    Detect if a group of lines forms a table. 
    TO BE IMPLEMENTED
    
    The function should check for:
    1. Vertically aligned '|' characters across multiple lines
    2. Horizontal separator lines (-----, _____, etc.)
    3. Consistent spacing/alignment between elements
    4. Patterns of dots/periods forming table-like structures (........)
    5. Common table headers (e.g., "Contents", "Περιεχόμενα")
    
    Input will be a sequence of lines that might form a table
    Output will be True if it's a table section, False otherwise
    """
    pass

def extract_table_lines(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    """
    Extract all lines that are part of the same table.
    TO BE IMPLEMENTED
    
    The function should:
    1. Start from start_idx and collect all consecutive table lines
    2. Handle multi-line table cells
    3. Include table headers and footers
    4. Preserve separator lines
    5. Stop when table pattern ends
    
    Input will be all document lines and a starting index
    Output will be the table lines and the index after the table
    """
    pass

def preserve_table_formatting(lines: List[str]) -> str:
    """
    Keep table formatting intact.
    TO BE IMPLEMENTED
    
    The function should:
    1. Preserve original whitespace and alignment
    2. Keep vertical bars and horizontal separators
    3. Maintain cell spacing and alignment
    4. Not modify any table formatting characters
    
    Input will be the table lines
    Output will be the formatted table as a single string
    """
    pass

def is_list_section(lines: List[str]) -> bool:
    """
    Detect if lines form a list.
    TO BE IMPLEMENTED
    
    The function should check for:
    1. Lines starting with bullets (-, *, •)
    2. Lines starting with numbers (1., 2., etc.)
    3. Consistent indentation patterns
    4. Common list section headers (Bibliography, References, Βιβλιογραφία)
    5. Handle both ordered and unordered lists
    
    Input will be a sequence of lines that might form a list
    Output will be True if it's a list section, False otherwise
    """
    pass

def extract_list_lines(lines: List[str], start_idx: int) -> Tuple[List[str], int]:
    """
    Extract all lines that are part of the same list.
    TO BE IMPLEMENTED
    
    The function should:
    1. Start from start_idx and collect all consecutive list items
    2. Handle multi-line list items
    3. Include list headers
    4. Preserve empty lines between items
    5. Stop when list pattern ends
    
    Input will be all document lines and a starting index
    Output will be the list lines and the index after the list
    """
    pass

def preserve_list_formatting(lines: List[str]) -> str:
    """
    Keep list formatting intact.
    TO BE IMPLEMENTED
    
    The function should:
    1. Preserve bullets and numbers
    2. Keep indentation and spacing
    3. Maintain empty lines between items
    4. Not modify any list markers or formatting
    
    Input will be the list lines
    Output will be the formatted list as a single string
    """
    pass

def process_academic_text(text: str) -> AcademicSection:
    """Process text into an academic document structure."""
    lines = text.splitlines()
    root = AcademicSection()
    section_stack = [root]
    current_content = []
    
    i = 0
    while i < len(lines):
        if is_table_section(lines[i:]):
            # Process entire table as one unit
            table_lines, i = extract_table_lines(lines, i)
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            section_stack[-1].content.append(preserve_table_formatting(table_lines))
            continue
            
        if is_list_section(lines[i:]):
            # Process entire list as one unit
            list_lines, i = extract_list_lines(lines, i)
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            section_stack[-1].content.append(preserve_list_formatting(list_lines))
            continue
        
        # Regular paragraph processing
        line = lines[i].strip()
        
        if is_header(line):
            # Save any accumulated content
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
            
            # Create new section
            level, title = extract_section_level(line)
            
            # Pop stack until we find appropriate parent
            while len(section_stack) > 1 and section_stack[-1].level >= level:
                section_stack.pop()
            
            new_section = AcademicSection(level=level, title=title)
            section_stack[-1].add_subsection(new_section)
            section_stack.append(new_section)
            
        elif line:
            # Accumulate non-empty lines into current content
            current_content.append(line)
        else:
            # Empty line marks paragraph boundary
            if current_content:
                section_stack[-1].content.append(" ".join(current_content))
                current_content = []
        
        i += 1
    
    # Add any remaining content
    if current_content:
        section_stack[-1].content.append(" ".join(current_content))
    
    return root

def detect_footnotes(paragraphs: List[str]) -> List[str]:
    """Detect and remove paragraphs that are actually footnotes or page numbers."""
    # Patterns that indicate footnotes
    footnote_patterns = [
        r'^\s*\d+\s*$',  # Just a number
        r'^\s*\d+[\-–]\d+\s*$',  # Number range
        r'^\s*(Ibid|op\.cit\.?|loc\.cit\.?|et\.al\.?|cf\.)\s*$',  # Latin references
        r'^\s*(βλ\.|πρβλ\.|σσ\.|σελ\.|ό\.π\.)\s*$',  # Greek references
        r'^\s*[\d\s\.,\-–]+\s*$',  # Just numbers and punctuation
        r'^\s*[\.,\-–]+\s*$',  # Just punctuation
        r'^\s*,\s*σσ\.\s*\d+.*$',  # Page number references
    ]
    
    # Combine patterns
    footnote_pattern = re.compile('|'.join(footnote_patterns), re.IGNORECASE)
    
    # Filter out footnotes
    filtered = []
    for para in paragraphs:
        # Skip if it matches a footnote pattern
        if footnote_pattern.match(para):
            continue
            
        # Skip if it's very short and doesn't look like a sentence
        if len(para) < 20 and not any(c.isalpha() for c in para):
            continue
            
        # Keep the paragraph
        filtered.append(para)
    
    return filtered

def should_merge_paragraphs(para1: str, para2: str) -> bool:
    """Determine if two consecutive paragraphs should be merged."""
    if not para1 or not para2:
        return False
        
    # Case 1: Hyphenated word
    if para1.rstrip().endswith('-'):
        return True
        
    # Case 2: Open parenthesis
    if para1.rstrip().endswith('('):
        return True
        
    # Case 3: Sentence continuation
    para1_end = para1.rstrip()[-1] if para1.strip() else ''
    para2_start = para2.lstrip()[0] if para2.strip() else ''
    
    # Check if it looks like a continuing sentence
    if (para1_end.islower() and para2_start.islower()) or \
       (para1_end in ',:') or \
       (para1_end.isdigit() and para2_start == '°'):
        return True
        
    return False

def process_section_paragraphs(section: AcademicSection):
    """Process all paragraphs in a section and its subsections."""
    # Process current section's paragraphs
    if section.content:
        # First remove footnotes
        section.content = detect_footnotes(section.content)
        
        # Then merge paragraphs that should be combined
        merged_content = []
        i = 0
        while i < len(section.content):
            current = section.content[i]
            if i < len(section.content) - 1:
                next_para = section.content[i + 1]
                if should_merge_paragraphs(current, next_para):
                    # Merge with next paragraph and skip it
                    merged_content.append(current + " " + next_para)
                    i += 2
                    continue
            merged_content.append(current)
            i += 1
        section.content = merged_content
    
    # Process subsections recursively
    for subsection in section.subsections:
        process_section_paragraphs(subsection)

def format_academic_document(text: str) -> str:
    """Format text as an academic document."""
    document = process_academic_text(text)
    process_section_paragraphs(document)
    return str(document)

def process_file(input_path: str, output_path: str):
    """Process a single file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    formatted_text = format_academic_document(text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)

def main():
    # Define input and output directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'test_output')
    output_dir = os.path.join(base_dir, 'academic_output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all .txt files
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_file(input_path, output_path)
            print(f"Processed {filename}")

if __name__ == "__main__":
    main()
