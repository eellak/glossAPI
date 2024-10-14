import re
import unicodedata

# Regex patterns
dotted_pattern = re.compile(r"[\.\_]")
protocol_pattern = re.compile(r"https?://")
path_component_pattern = re.compile(r"/[a-zA-Z0-9\-\._~%!$&'()*+,;=]+")
encoded_path_pattern = re.compile(r"%[0-9A-Fa-f]{2}")
eikona_pattern = re.compile(r"^εικονα ?\d{1,2}\.\d{1,2}(\.\d{1,2})?", re.UNICODE)

# Function to remove accents
def remove_accents(line):
    line = line.lower()
    accentless_line = ''.join(
        c for c in unicodedata.normalize('NFD', line)
        if unicodedata.category(c) != 'Mn'
    )
    return accentless_line

# Function to find dotted index pattern
def find_index_line(line):
    return bool(dotted_pattern.search(line))

# Function to find URL patterns
def find_url(line):
    return bool(protocol_pattern.search(line) or path_component_pattern.search(line) or encoded_path_pattern.search(line))

# Function to test the behavior of `find_index_line`, `find_url`, and `eikona_pattern`
def test_find_exclude_line(input_line, line_number):
    print(f"Original Line: {input_line}")
    
    # Remove accents
    accentless_line = remove_accents(input_line)
    # Remove whitespaces
    concat_line = re.sub(r'\s', '', accentless_line)
    
    if find_index_line(input_line) or find_url(accentless_line) or eikona_pattern.search(concat_line):
        # Check conditions
        index_line = find_index_line(input_line)
        url_line = find_url(accentless_line)
        eikona_match = eikona_pattern.search(concat_line)

        # Print intermediate stages
        print(f"Accentless Line: {accentless_line}")
        print(f"Concatenated Line: {concat_line}")
        
        # Debugging output based on the result
        if index_line:
            print(f"Index Line Match (contains multiple dots/underscores): {index_line}")
        if url_line:
            print(f"URL Line Match (contains URL-like elements): {url_line}")
        if eikona_match:
            print(f"Εικόνα Match (matches εικονα x.x.x): {eikona_match.group()}")

        # Decision making
        if index_line or url_line or eikona_match:
            print(f"Line {line_number} should be excluded.")
        else:
            print(f"Line {line_number} should NOT be excluded.")
    
    # Return values for potential assertion in tests
    return index_line, url_line, eikona_match


# Example test cases
test_find_exclude_line("Ε ι κ ό ν α 6 . 1 1 κ α ι 6 . 1 2", 1)
test_find_exclude_line("Ε ι κ ό ν α 6 . 1 3 κ α ι 6 . 1 4", 2)
test_find_exclude_line("Ε ι κ ό ν α 6 . 1 5 κ α ι 6 . 1 6", 3)
test_find_exclude_line("Ε ι κ ό ν α 6 . 1 0", 4)
