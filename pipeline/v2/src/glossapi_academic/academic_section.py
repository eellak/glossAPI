from typing import List

class AcademicSection:
    """
    A data structure representing a section in an academic document.
    
    Attributes:
        level (int): The heading level (0 for document root)
        title (str): The section title
        start_line (int): The starting line number in the original document
        end_line (int): The ending line number in the original document
        content (List[str]): List of paragraph strings in this section
        subsections (List['AcademicSection']): List of subsections
        is_bibliography (bool): Flag indicating if this section is a bibliography
    """
    def __init__(self, level: int = 0, title: str = "", start_line: int = 0):
        self.level = level
        self.title = title
        self.start_line = start_line
        self.end_line = start_line
        self.content: List[str] = []
        self.subsections: List['AcademicSection'] = []            
        self.is_bibliography = False

    def add_subsection(self, subsection: 'AcademicSection'):
        """Add a subsection to this section"""
        self.subsections.append(subsection)
