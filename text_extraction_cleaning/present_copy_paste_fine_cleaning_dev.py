import os
import re
import csv
import unicodedata
from find_similar_lines import find_similar_lines

#*********************************************************************************************************
# Define the Presentation variable
Presentation = True  # Set to False to omit annotations and exclude lines without annotation

# Define the directory paths
input_directory = '/home/fivos/Projects/GlossAPI/raw_txt/sxolika/paste_texts/xondrikos_katharismos_papers'

# Dynamically set the output_directory based on the Presentation variable
if Presentation:
    output_directory = os.path.join(input_directory, 'fine_cleaning_presentation_v4')
else:
    output_directory = os.path.join(input_directory, 'fine_cleaning_v4')

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)
#*********************************************************************************************************

# Patterns look weird because with copy/pasting PDF sometimes capital letters of Greek and English
# that look the same such as P,T,K,Z,O,I,N,M,Y,B,A,H
# are pasted as latin characters when in fact they should be Greek

# Patterns to find bibliography lines
bibliography_pattern = re.compile(r".*[βb][iι][βb]λ[iι][oο]γ[pρ][αa]φ[iι][αa]([-–α-ωΑ-Ω\w]{0,10})($|\n)", re.UNICODE)
bibliography_atend = re.compile(r"[βb][iι][βb]λ[iι][oο]γ[pρ][αa]φ[iι][αa] {0,5}($|\n)", re.UNICODE)

# Find index
dotted_pattern = re.compile(r"\.{5,}|…{3,}|(\. ){5,}|_{3,}|(_ ){3,}", re.UNICODE)


# Patterns to find chapter lines
kefalaio_atstart_pattern = re.compile(r"^([κk][εe]φ[αa]λ[αa][iι][oο] .{1,2}:?)", re.UNICODE)
kefalaio_pattern = re.compile(r"([κk][εe]φ[αa]λ[αa][iι][oο]:?)", re.UNICODE)
negative_kefalaio_pattern = re.compile(r"([κk][εe]φ[αa]λ[αa][iι][oο][α-ω.,])", re.UNICODE)
enotita_pattern = re.compile(r".*([εe]ν[oο][τt][ηh][τt][αa] .*\d+.*)$", re.UNICODE)
negative_enotita_pattern = re.compile(r"([εe]ν[oο][τt][ηh][τt][αa][α-ω.,])", re.UNICODE)
kef_pattern = re.compile(r".*[κk][εe]φ\.\s.*", re.UNICODE)
section_number = re.compile(r"(\d\d?)\.(\d\d?)\.(\d\d?)?")

# Patterns to find exercises
askiseis_pattern = re.compile(r"^[αa]σ[κk][ηh]σ[εe][iι]σ", re.UNICODE)
askisi_pattern = re.compile(r"^[αa]σ[κk][ηh]σ[ηh] \d{1,2}", re.UNICODE)
erotiseis_pattern = re.compile(r"^[εe][pρ]ω[τt][ηh]σ[εe][iι]σ", re.UNICODE)
erotiseis_askiseis_pattern = re.compile(r"[εe][pρ]ω[τt][ηh]σ[εe][iι]σ[\s-]*[αa]σ[κk][ηh]σ[εe][iι]σ[\s-]*[πp][pρ][oο][βb]λ[ηh]μ[αa][τt][αa]", re.UNICODE)
fyllo_ergasias_pattern = re.compile(r"φ[υy]λλ[oο] [εe][pρ]γ[αa]σ[iι][αa]σ(\ [αa]ξ[iι][oο]λ[oο]γ[ηh]σ[ηh]σ)?", re.UNICODE)
askiseis_chapter_pattern = re.compile(r"[αa]σ[κk][ηh]σ[εe][iι]σ\s+\w+\s+[κk][εe]φ[αa]λ[αa][iι][oο][υy]", re.UNICODE)
erotimatologio_pattern = re.compile(r"[εe][pρ]ω[τt][ηh]μ[αa][τt][oο]λ[oο]γ[iι][oο]", re.UNICODE)
akrostixida_pattern = re.compile(r"[αa][κk][pρ][oο]σ[τt][iι][xχ][iι]δ[αa]", re.UNICODE)

#detect image
eikona_pattern = re.compile(r"^[εe][iι][κk][oο]ν[αa] ?\d{1,2}\.\d{1,2}(\.\d{1,2})?", re.UNICODE)
#see also "Εικ. 10.13"

# url regex
protocol_pattern = re.compile(r'https?://')
path_component_pattern =  re.compile(r'(\.[a-z]{2,3})(\/[A-Za-z0-9?=]+)|(\/[A-Za-z0-9?=]+){3,}')
encoded_path_pattern = re.compile(r'(%[0-9A-Fa-f]+){2,}')

#last segment (of bibliography-less documents) cleaning
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


def contains_weird_characters(line):
    # Pattern to match any single character that is not in Greek, Ancient Greek, English, punctuation or other common characters, numbers or common Mathematical and scientific notation
    # and any other symbol that might be used pictorially or as list element
    allowed_chars = (
        r'a-zA-Z\d\s'  # English letters, digits, and whitespace
        r'\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F'  # Greek and Ancient Greek
        r'\u0387\u00B7'  # Greek punctuation
        r'.,;:!?()\[\]{}'  # Basic punctuation
        r'\-\–\—\―\‒\_'  # Various dashes and underscore
        r'«»'  # Greek quotation marks
        r"''΄'’´"  # Apostrophes and stress mark
        r'…·'  # Ellipsis and ano teleia
        r'+=*/\\|<>~@#$%^&€£¥§©®™°±²³¹½¼¾'  # Special characters and symbols
        r'×÷πΠ∞∑√∫≈≠≤≥←↑→↓↔⇒⇔∀∃∈∩∪⊂⊃⊄⊅⊆⊇⊕⊗⋅∅∇∂∫∮∴∵∼≅≈≡'  # Mathematical symbols
        r'•○●◦◘◙♦♣♠♥♪♫'  # other common symbols
    )
    weird_char_pattern = re.compile(f'[^{allowed_chars}]')
    
    return bool(weird_char_pattern.search(line))


def remove_accents(line):
    ready_line = line.lower()
    accentless_line = ''.join(
        c for c in unicodedata.normalize('NFD', ready_line)
        if unicodedata.category(c) != 'Mn'
    )
    return accentless_line

def find_url(line):
    return bool( protocol_pattern.search(line) or path_component_pattern.search(line) or encoded_path_pattern.search(line) )

def find_index_line(line):
    return bool(dotted_pattern.search(line))

def find_bibliography_line(line):
    match = False
    accentless_line = remove_accents(line)
    concat_line = re.sub(r'\s', '', accentless_line)
    if len(concat_line) < 40:
        match = bibliography_pattern.search(concat_line)
    if not match:
        match = bibliography_atend.search(concat_line)
    return match

def find_chapter_line(line):
    match = False
    accentless_line = remove_accents(line.strip())
    concat_line = re.sub(r'\s', '', accentless_line)
    match = kefalaio_atstart_pattern.search(accentless_line)
    if not match and len(concat_line) < 40:
        if not negative_kefalaio_pattern.search(accentless_line):
            match = kefalaio_pattern.search(accentless_line)
    if not match and len(concat_line) < 30:
        if not negative_kefalaio_pattern.search(accentless_line): # avoids other grammatical forms such as κεφαλαιου
            match = kefalaio_pattern.search(concat_line) # meant to detect "Κ Ε Φ Α Λ Α Ι Ό 8230"
        if not match and not negative_enotita_pattern.search(accentless_line):# avoids other grammatical forms such as ενοτητας
            match = enotita_pattern.search(accentless_line)
        if not match:
            match = kef_pattern.search(accentless_line)
        if not match:
            match = section_number.search(accentless_line)
    return match

def find_excercise_line(line):
    accentless_line = remove_accents(line)
    single_spaced_line = re.sub(r'\s+', ' ', accentless_line).strip()
    
    if len(single_spaced_line) <= 20:
        if askiseis_pattern.search(single_spaced_line):
            return "askiseis"
        if askisi_pattern.search(single_spaced_line):
            return "askisi"
        #if drastiriotites_pattern.search(single_spaced_line):
         #   return "drastiriotites"
        if erotiseis_askiseis_pattern.search(single_spaced_line):
            return "erotiseis_askiseis"
        if erotiseis_pattern.search(single_spaced_line):
            return "erotiseis"
        if fyllo_ergasias_pattern.search(single_spaced_line):
            return "fyllo_ergasias"
        if askiseis_chapter_pattern.search(single_spaced_line):
            return "askiseis_chapter"
        if erotimatologio_pattern.search(single_spaced_line):
            return "erotimatologio"
        if akrostixida_pattern.search(single_spaced_line):
            return "akrostixida"
    return ""

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
        #if simeiosis_pattern.search(single_spaced_line): probably too much
        #    return "simeiosis"
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


def process_file(lines):
    bib_ranges = []
    excercise_ranges = []
    glossary_range = []
    
    in_bibliography_section = False
    bib_start_line_number = None
    
    in_excercise = False
    excercise_start_line_number = None
    
    txt_length = len(lines)
    lines_to_exclude = set()
    
    glossary_etc_first_line = None
    in_glossary_or_similar = False

    for line_number, line in enumerate(lines, 1):
        # Check for dotted lines
        if in_glossary_or_similar:
            break
        
        if contains_weird_characters(line):
            lines_to_exclude.add(line_number)
            continue
        
        accentless_line = remove_accents(line)
        concat_line = re.sub(r'\s', '', accentless_line)
        
        # Check if a single line contains multiple . or _ ; url elements ; or εικονα x.x.x
        if find_index_line(line) or find_url(accentless_line) or eikona_pattern.search(concat_line):
            lines_to_exclude.add(line_number)

        # Check if we have reached the bibliography section
        if not in_bibliography_section and not in_excercise and line_number / txt_length > 0.02 and find_bibliography_line(line):
            in_bibliography_section = True
            bib_start_line_number = line_number  # Start at "βιβλιογραφια" line

        # Check if we have reached the exercise section
        if not in_excercise and not in_bibliography_section and find_excercise_line(line):
            in_excercise = True
            excercise_start_line_number = line_number  # Start at the exercise line

        # If we are in the bibliography section and find a new chapter, close the range
        if in_bibliography_section and find_chapter_line(line):
            bib_end_line_number = line_number - 1  # End before the "chapter" line
            if bib_start_line_number <= bib_end_line_number:
                bib_ranges.append((bib_start_line_number, bib_end_line_number))
            in_bibliography_section = False  # Reset for the next potential bibliography section
            bib_start_line_number = None

        # If we are in the exercise section and find a new chapter, close the range
        if in_excercise and find_chapter_line(line):
            excercise_end_line = line_number - 1  # End before the "chapter" line
            if excercise_start_line_number <= excercise_end_line:
                excercise_ranges.append((excercise_start_line_number, excercise_end_line))
            in_excercise = False
            excercise_start_line_number = None
            
        if line_number > (txt_length - 500) and find_glossaries_etc(accentless_line):
            glossary_etc_first_line = line_number
            in_glossary_or_similar = True

    # Close any sections that reach the end of the file
    if in_bibliography_section and bib_start_line_number is not None:
        bib_ranges.append((bib_start_line_number, txt_length))

    if in_excercise and excercise_start_line_number is not None:
        excercise_ranges.append((excercise_start_line_number, txt_length))
        
    if in_glossary_or_similar:
        glossary_range.append((glossary_etc_first_line, txt_length))

    return bib_ranges, excercise_ranges, glossary_range, lines_to_exclude


import os
import csv
from find_similar_lines import find_similar_lines

def main():
    # Prepare CSV output file
    csv_output_path = os.path.join(output_directory, 'filtering_presentation.csv')
    
    # Open the CSV file for writing
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the header row in the CSV
        csvwriter.writerow([
            'Filename',
            'Total Excluded Lines',
            'Excluded Percentage',
            'Total Bibliography Lines',
            'Bibliography Percentage',
            'Total Exercise Lines',
            'Exercise Percentage',
            'Total Glossary Lines',
            'Glossary Percentage',
            'Total Similar Lines',
            'Similar Percentage',
            'Total Dotted Lines',
            'Dotted Percentage',
            'Internal Bibliography',
            'Contains Exercises',
            'Contains Glossary',
            'Contains Similar Lines',
            'Contains Dotted Lines'
        ])
    
        # Process each file in the input directory
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
            
            # Skip directories and non-text files
            if os.path.isdir(file_path) or not filename.endswith('.txt'):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
    
            # Process the file to identify ranges and lines to exclude
            bib_ranges, excercise_ranges, glossary_range, lines_to_exclude = process_file(lines)
            
            # Identify similar lines using find_similar_lines
            similar_lines_set = find_similar_lines(lines)
    
            if bib_ranges:
                print(filename, " έχει εσωτερική βιβλιογραφία")
            if excercise_ranges:
                print(filename, " έχει ασκήσεις")
            if glossary_range:
                print(filename, " έχει γλωσσάριο ή παρόμοιο τμήμα στο τέλος")
            if similar_lines_set:
                print(filename, " έχει επαναλαμβανόμενες γραμμές")
    
            # Get sets of line numbers for each exclusion category
            bibliography_lines_set = set()
            for start_line, end_line in bib_ranges:
                bibliography_lines_set.update(range(start_line, end_line + 1))
    
            excercise_lines_set = set()
            for start_line, end_line in excercise_ranges:
                excercise_lines_set.update(range(start_line, end_line + 1))
    
            glossary_lines_set = set()
            for start_line, end_line in glossary_range:
                glossary_lines_set.update(range(start_line, end_line + 1))
    
            # Lines to exclude from find_similar_lines are individual lines, not ranges
            # They are already in similar_lines_set
    
            # Combine all exclusion sets
            excluded_lines_set = bibliography_lines_set.union(
                excercise_lines_set, glossary_lines_set, lines_to_exclude, similar_lines_set
            )
    
            # Prepare output lines with or without annotations based on Presentation
            output_lines = []
            for line_number, line in enumerate(lines, 1):
                if Presentation:
                    if line_number in bibliography_lines_set:
                        output_lines.append("[ΕΚΤ:ΒΙΒΛ]" + line)
                    elif line_number in excercise_lines_set:
                        output_lines.append("[ΕΚΤ:ΑΣΚ]" + line)
                    elif line_number in glossary_lines_set:
                        output_lines.append("[ΕΚΤ:ΓΛΩΣ]" + line)
                    elif line_number in lines_to_exclude:
                        output_lines.append("[ΕΚΤ:ΓΡΑΜ]" + line)
                    elif line_number in similar_lines_set:
                        output_lines.append("[ΕΚΤ:ΕΠΑΝ.]" + line)
                    else:
                        output_lines.append(line)
                else:
                    # When Presentation is False, exclude the lines without annotation
                    if line_number not in excluded_lines_set:
                        output_lines.append(line)
            # End of loop over lines
    
            # Calculate counts and percentages
            total_lines = len(lines)
            total_bibliography_lines = len(bibliography_lines_set)
            total_excercise_lines = len(excercise_lines_set)
            total_glossary_lines = len(glossary_lines_set)
            total_similar_lines = len(similar_lines_set)
            total_dotted_lines = len(lines_to_exclude)
            total_excluded_lines = len(excluded_lines_set)
    
            bibliography_percentage = (total_bibliography_lines / total_lines) * 100 if total_lines > 0 else 0
            excercise_percentage = (total_excercise_lines / total_lines) * 100 if total_lines > 0 else 0
            glossary_percentage = (total_glossary_lines / total_lines) * 100 if total_lines > 0 else 0
            similar_percentage = (total_similar_lines / total_lines) * 100 if total_lines > 0 else 0
            dotted_percentage = (total_dotted_lines / total_lines) * 100 if total_lines > 0 else 0
            excluded_percentage = (total_excluded_lines / total_lines) * 100 if total_lines > 0 else 0
    
            internal_bib = 1 if total_bibliography_lines > 0 else 0
            contains_excercises = 1 if total_excercise_lines > 0 else 0
            contains_glossary = 1 if total_glossary_lines > 0 else 0
            contains_similar_lines = 1 if total_similar_lines > 0 else 0
            contains_dotted_lines = 1 if total_dotted_lines > 0 else 0
    
            # Write data to CSV
            csvwriter.writerow([
                filename,
                total_excluded_lines,
                f'{excluded_percentage:.2f}%',
                total_bibliography_lines,
                f'{bibliography_percentage:.2f}%',
                total_excercise_lines,
                f'{excercise_percentage:.2f}%',
                total_glossary_lines,
                f'{glossary_percentage:.2f}%',
                total_similar_lines,
                f'{similar_percentage:.2f}%',
                total_dotted_lines,
                f'{dotted_percentage:.2f}%',
                internal_bib,
                contains_excercises,
                contains_glossary,
                contains_similar_lines,
                contains_dotted_lines
            ])
            
            # Write the transformed lines to the output file
            output_file_path = os.path.join(output_directory, filename)
            try:
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.writelines(output_lines)
            except Exception as e:
                print(f"Error writing to {output_file_path}: {e}")
                continue
            

if __name__ == '__main__':
    main()
