from unidecode import unidecode
import re
import unicodedata

def is_short_line(line: str) -> bool:
    """Check if a line is less than 25 characters long."""
    return len(line) < 25

def starts_with_hashtags(line: str) -> bool:
    """Check if a line starts with '##'."""
    return line.startswith("##")

def normalize_case(line: str) -> str:
    """Convert the line to lowercase."""
    return line.lower()

def normalize_spaces(line: str) -> str:
    """Strip leading/trailing spaces and replace multiple spaces with a single space."""
    return re.sub(r'\s+', ' ', line.strip())

def latinify(line: str) -> str:
    """Convert Greek text to Latin characters."""
    return unidecode(line)

def remove_greek_accents(line: str) -> str:
    """Remove accents from Greek text while keeping Greek characters.
    
    Examples:
        'Καλημέρα' -> 'Καλημερα'
        'Ελλάδα' -> 'Ελλαδα'
    """
    # NFD decomposes characters into their base form and combining characters
    # We then keep only non-combining characters (those with combining class 0)
    return ''.join(c for c in unicodedata.normalize('NFD', line)
                  if not unicodedata.combining(c))

# Mapping of Latin characters that look like Greek ones
LATIN_TO_GREEK_MAP = {
    # Uppercase mappings
    'A': 'Α',  # Latin A to Greek Alpha
    'B': 'Β',  # Latin B to Greek Beta
    'E': 'Ε',  # Latin E to Greek Epsilon
    'Z': 'Ζ',  # Latin Z to Greek Zeta
    'H': 'Η',  # Latin H to Greek Eta
    'I': 'Ι',  # Latin I to Greek Iota
    'K': 'Κ',  # Latin K to Greek Kappa
    'M': 'Μ',  # Latin M to Greek Mu
    'N': 'Ν',  # Latin N to Greek Nu
    'O': 'Ο',  # Latin O to Greek Omicron
    'P': 'Ρ',  # Latin P to Greek Rho
    'T': 'Τ',  # Latin T to Greek Tau
    'Y': 'Υ',  # Latin Y to Greek Upsilon
    'X': 'Χ',  # Latin X to Greek Chi
    
    # Lowercase mappings
    'a': 'α',  # Latin a to Greek alpha
    'b': 'β',  # Latin b to Greek beta
    'e': 'ε',  # Latin e to Greek epsilon
    'i': 'ι',  # Latin i to Greek iota
    'k': 'κ',  # Latin k to Greek kappa
    'n': 'η',  # Latin n to Greek eta
    'o': 'ο',  # Latin o to Greek omicron
    'p': 'ρ',  # Latin p to Greek rho
    'u': 'υ',  # Latin u to Greek upsilon
    'v': 'ν',  # Latin v to Greek nu
    'x': 'χ',  # Latin x to Greek chi
    'y': 'γ',  # Latin y to Greek gamma
}

def fix_greek_latin_mix(title: str) -> str:
    """
    Convert any Latin characters that look like Greek ones to their proper Greek equivalents.
    This is useful for fixing PDF-extracted text where Latin characters might have been
    incorrectly substituted for Greek ones.
    
    Examples:
        'ABΓ' -> 'ΑΒΓ' (converts Latin A,B to Greek Α,Β)
        'Olympos' -> 'Ολυμπος'
    
    Args:
        title: A string that may contain a mix of Greek and Latin characters
        
    Returns:
        A string with Latin lookalike characters converted to their Greek equivalents
    """
    # First check if the title starts with markdown heading symbols
    if not title.startswith('#'):
        return title
        
    # Convert each Latin lookalike to its Greek equivalent
    result = ''
    for char in title:
        result += LATIN_TO_GREEK_MAP.get(char, char)
    
    return result

def normalize_title(title: str) -> str:
    """Apply all normalization steps to a title."""
    normalized = title
    normalized = fix_greek_latin_mix(normalized)  # First convert any Latin lookalikes
    normalized = normalize_case(normalized)
    normalized = normalize_spaces(normalized)
    normalized = remove_greek_accents(normalized)
    return normalized

# Example usage
if __name__ == "__main__":
    example_title = "## Kαλημέρa   Eλλάδα"  # Mix of Latin and Greek characters
    print(f"Original title: {example_title}")
    print(f"After fix_greek_latin_mix: {fix_greek_latin_mix(example_title)}")
    print(f"After normalize_case: {normalize_case(fix_greek_latin_mix(example_title))}")
    print(f"After normalize_spaces: {normalize_spaces(normalize_case(fix_greek_latin_mix(example_title)))}")
    print(f"After remove_greek_accents: {remove_greek_accents(normalize_spaces(normalize_case(fix_greek_latin_mix(example_title))))}")
    print(f"Final normalized result: {normalize_title(example_title)}")
    print(f"Is short? {is_short_line(example_title)}")
    print(f"Starts with ##? {starts_with_hashtags(example_title)}")
