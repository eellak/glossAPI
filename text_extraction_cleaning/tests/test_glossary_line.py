import re
import unicodedata

# Regex patterns
glossario_pattern = re.compile(r"γλωσσ[αa][pρ][iι][oο]", re.UNICODE)
evretirio_pattern = re.compile(r"[εe][υy][pρ][εe][τt][ηh][pρ][iι][oο]", re.UNICODE)
evretiria_pattern = re.compile(r"[εe][υy][pρ][εe][τt][ηh][pρ][iι][αa]", re.UNICODE)
alphabitiko_evretirio_pattern = re.compile(r"[αa]λφ[αa][βb][ηh][τt][iι][κk][oο] [εe][υy][pρ][εe][τt][ηh][pρ][iι][oο]", re.UNICODE)
simeiosis_pattern = re.compile(r"σ[ηh]μ[εe][iι]ωσ[εe][iι]σ", re.UNICODE)
glossari_pattern = re.compile(r"γλωσσ[αa][pρ][iι]", re.UNICODE)
evretirio_ennion_pattern = re.compile(r"[εe][υy][pρ][εe][τt][ηh][pρ][iι][oο] [εe]νν[oο][iι]ων", re.UNICODE)
evretirio_onomaton_pattern = re.compile(r"[εe][υy][pρ][εe][τt][ηh][pρ][iι][oο] [oο]ν[oο]μ[αa][τt]ων", re.UNICODE)
lexilogio_oron_pattern = re.compile(r"λ[εe]ξ[iι]λ[oο]γ[iι][oο] [oο][pρ]ων", re.UNICODE)
lexiko_pattern = re.compile(r"λ[εe]ξ[iι][κk][oο]", re.UNICODE)
vasiki_orologia_pattern = re.compile(r"[βb][αa]σ[iι][κk][ηh] [oο][pρ][oο]λ[oο]γ[iι][αa]", re.UNICODE)
akrostixida_pattern = re.compile(r"[αa][κk][pρ][oο]σ[τt][iι]χ[iι]δ[αa]", re.UNICODE)

def remove_accents(line):
    ready_line = line.lower()
    accentless_line = ''.join(
        c for c in unicodedata.normalize('NFD', ready_line)
        if unicodedata.category(c) != 'Mn'
    )
    return accentless_line

def find_glossaries_etc(accentless_line):
    single_spaced_line = re.sub(r'\s+', ' ', accentless_line).strip()
    
    if len(single_spaced_line) < 25:
        if glossario_pattern.search(single_spaced_line):
            return "glossario"
        if evretirio_pattern.search(single_spaced_line):
            return "evretirio"
        if evretiria_pattern.search(single_spaced_line):
            return "evretiria"
        if alphabitiko_evretirio_pattern.search(single_spaced_line):
            return "alphabitiko_evretirio"
        if glossari_pattern.search(single_spaced_line):
            return "glossari"
        if evretirio_ennion_pattern.search(single_spaced_line):
            return "evretirio_ennion"
        if evretirio_onomaton_pattern.search(single_spaced_line):
            return "evretirio_onomaton"
        if lexilogio_oron_pattern.search(single_spaced_line):
            return "lexilogio_oron"
        if lexiko_pattern.search(single_spaced_line):
            return "lexiko"
        if vasiki_orologia_pattern.search(single_spaced_line):
            return "vasiki_orologia"
    return ""

def process_and_match(input_line):
    print(f"Original input: '{input_line}'")
    
    # Step 1: Convert to lowercase
    lowercase = input_line.lower()
    print(f"Lowercase: '{lowercase}'")
    
    # Step 2: Remove accents
    deaccented = remove_accents(lowercase)
    print(f"Deaccented: '{deaccented}'")
    
    # Step 3: Replace multiple spaces with single space and strip
    single_spaced = re.sub(r'\s+', ' ', deaccented).strip()
    print(f"Single-spaced and stripped: '{single_spaced}'")
    
    # Step 4: Check length and apply regex patterns
    if len(single_spaced) < 25:
        patterns = [
            ("glossario", glossario_pattern),
            ("evretirio", evretirio_pattern),
            ("evretiria", evretiria_pattern),
            ("alphabitiko_evretirio", alphabitiko_evretirio_pattern),
            ("glossari", glossari_pattern),
            ("evretirio_ennion", evretirio_ennion_pattern),
            ("evretirio_onomaton", evretirio_onomaton_pattern),
            ("lexilogio_oron", lexilogio_oron_pattern),
            ("lexiko", lexiko_pattern),
            ("vasiki_orologia", vasiki_orologia_pattern),
        ]
        
        for name, pattern in patterns:
            if pattern.search(single_spaced):
                print(f"Matched: {name}")
                return
    
    print("No match")

# Test the function with various inputs
test_inputs = [
    "Γλωσσάριο",
    "EYPETHPIA",
    "ΕΥΡΕΤΗΡΙΟ",
    "Αλφαβητικό   Ευρετήριο  ",
]

for input_line in test_inputs:
    process_and_match(input_line)
    print()  # Add a blank line between inputs for readability