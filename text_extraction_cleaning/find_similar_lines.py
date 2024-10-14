# find_similar_lines.py

import unicodedata
import re
from collections import defaultdict

def normalize_text(text):
    """
    Normalize the text by:
    - Converting to lowercase
    - Removing accents
    - Replacing multiple whitespace with a single space
    - Stripping leading and trailing whitespace
    """
    # Convert to lowercase
    text = text.lower()
    # Remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def is_numeric_line(line):
    """
    Determine if a line is purely numeric.

    Parameters:
    - line (str): The normalized line to check.

    Returns:
    - bool: True if the line consists solely of numbers, possibly including spaces, commas, or decimal points; False otherwise.
    """
    return bool(re.fullmatch(r'[\d\s.,]+', line))

def levenshtein_distance(s1, s2, max_distance):
    """
    Compute the Levenshtein distance between two strings.
    Optimization: Stop if distance exceeds max_distance.

    Parameters:
    - s1 (str): The first string.
    - s2 (str): The second string.
    - max_distance (int): The maximum distance to compute.

    Returns:
    - int: The Levenshtein distance if it's <= max_distance; otherwise, max_distance + 1.
    """
    if abs(len(s1) - len(s2)) > max_distance:
        return max_distance + 1

    # Initialize the previous row of distances
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        min_distance_in_row = current_row[0]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1    # Insertion
            deletions = current_row[j] + 1          # Deletion
            substitutions = previous_row[j] + (c1 != c2)  # Substitution
            current_cell = min(insertions, deletions, substitutions)
            current_row.append(current_cell)
            min_distance_in_row = min(min_distance_in_row, current_cell)
        if min_distance_in_row > max_distance:
            return max_distance + 1
        previous_row = current_row

    distance = previous_row[-1]
    return distance

class BKTree:
    """
    Burkhard-Keller Tree (BK-Tree) implementation for efficient similarity search.
    """
    def __init__(self, distance_func):
        """
        Initialize the BK-Tree.

        Parameters:
        - distance_func (callable): A function that computes the distance between two items.
        """
        self.tree = None
        self.distance_func = distance_func

    def add(self, item, max_distance=3):
        """
        Add an item to the BK-Tree.

        Parameters:
        - item (str): The item to add.
        - max_distance (int): The maximum distance threshold.
        """
        if self.tree is None:
            self.tree = ([item], {})
            return

        node_items, children = self.tree
        distance = self.distance_func(item, node_items[0], max_distance)
        if distance == 0:
            node_items.append(item)
        elif distance <= max_distance:
            if distance in children:
                children[distance].add(item, max_distance)
            else:
                children[distance] = BKTree(self.distance_func)
                children[distance].tree = ([item], {})
        else:
            pass  # Do not add lines with distance > max_distance

    def search(self, item, max_distance):
        """
        Search for items within a specified distance from the given item.

        Parameters:
        - item (str): The item to search for.
        - max_distance (int): The maximum distance threshold.

        Returns:
        - list: A list of matching items within the specified distance.
        """
        results = []
        if self.tree is None:
            return results

        node_items, children = self.tree
        distance = self.distance_func(item, node_items[0], max_distance)
        if distance <= max_distance:
            results.extend(node_items)  # Add all matching items

        # Iterate through possible distances
        for dist in range(distance - max_distance, distance + max_distance + 1):
            if dist < 0:
                continue  # Distance cannot be negative
            child = children.get(dist)
            if child is not None:
                results.extend(child.search(item, max_distance))
        return results

def find_similar_lines(lines, max_distance=3, use_regex=True, min_length=25):
    """
    Identify all unique lines that have at least one similar line within the specified Levenshtein distance.
    Lines that consist solely of numbers or are shorter than min_length characters are ignored.

    Parameters:
    - lines (list of str): The list of lines from a text file.
    - max_distance (int): The maximum Levenshtein distance to consider lines as similar.
    - use_regex (bool): If True, use regex to determine numeric lines; otherwise, use str.isdigit().
    - min_length (int): The minimum number of characters a line must have to be considered.

    Returns:
    - list of int: A sorted list of unique line numbers (1-based) that have at least one similar line.
    """
    normalized_lines = [normalize_text(line) for line in lines]
    bk_tree = BKTree(levenshtein_distance)
    similar_line_numbers = set()

    # Map normalized lines to their line indices (0-based)
    line_map = defaultdict(list)

    for idx, line in enumerate(normalized_lines):
        if not line:
            continue  # Skip empty lines

        # Determine if the line is purely numeric
        if use_regex:
            if is_numeric_line(line):
                continue  # Skip lines that are purely numeric
        else:
            if line.isdigit():
                continue  # Skip lines that are purely numeric

        # Check if the line meets the minimum length requirement
        if len(line) <= min_length:
            continue  # Skip lines shorter than or equal to min_length characters

        # Search for similar lines in the BK-Tree
        matches = bk_tree.search(line, max_distance)
        for match in matches:
            # Retrieve all original indices for this matched line
            for original_idx in line_map[match]:
                # Add both current line and matched line numbers to the set
                similar_line_numbers.add(original_idx + 1)  # 1-based indexing
                similar_line_numbers.add(idx + 1)

        # Add the current line to the BK-Tree and the mapping
        bk_tree.add(line, max_distance)
        line_map[line].append(idx)

    # Return a sorted list of unique line numbers
    return sorted(similar_line_numbers)
