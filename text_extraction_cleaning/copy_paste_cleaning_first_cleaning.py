import os
import re
import unicodedata
import math

# Define the directory paths
input_directory = '/home/fivos/Projects/GlossAPI/raw_txt/sxolika/paste_texts'
# Outputs here:
output_directory = os.path.join(input_directory, 'xondrikos_katharismos_papers')

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Intended match: Greek_words, optional space, more than 3 dots, optional space and/or "σελ.", finally a number and nothing else concatenated to it.
#   eg 'ΠΡΟΚΑΤΑΡΚΤΙΚΕΣ ΕΡΓΑΣΙΕΣ ΣΥΝΤΗΡΗΣΗΣ ........................................................50'
index_pattern = re.compile(r"([α-ωa-zA-Α-ΩΆ-Ώά-ώ \d:]+)\s{0,4}(\.{4,})\s{0,2}(σελ\.|Σελ\.\:|«)?\s{0,4}\d+(?!\S)")
index_without_dots = re.compile(r"([Α-ΩΆ-ΏΪΫ– ])+ {1,2}\d{1,3}$")
# Pattern for bibliography
bibliography_with_fullstop = re.compile(r".*βιβλιογραφια\.$")
bibliography_pattern = re.compile(r".*βιβλιογραφια.*")
legal_statement_pattern = re.compile(r".*Βάσει του ν\. 3966/2011 τα διδακτικά βιβλία.*", re.IGNORECASE)
pagination_pattern = re.compile(r"^((\d){1,2}|(. (\d) .)|\[(\d)\]|(vi{0,3}))$")

def find_bibliography_line(line):
    # Removes combined characters (accents) from Greek letters
    accentless_line = ''.join(c for c in unicodedata.normalize('NFD', line) if unicodedata.category(c) != 'Mn')
    # Remove any non-printable characters and collapse whitespace
    concat_line = re.sub(r'[^α-ωΑ-Ω]', '', accentless_line)  # Keeps only Greek characters
    if len(concat_line) < 40:
        match = bibliography_pattern.match(concat_line.lower())
        return match
    else:
        return False
    
def not_with_fullstop(line):
    accentless_line = ''.join(c for c in unicodedata.normalize('NFD', line.lower()) if unicodedata.category(c) != 'Mn')
    if bibliography_with_fullstop.match(accentless_line): return False
    else: return True

def find_legal_statement_line(line):
    # Check if the line matches the legal statement pattern
    match = legal_statement_pattern.match(line.lower())
    return match

def find_page_number(line):
    match = pagination_pattern.search(line)
    if match:
        page = match.group(1)
        if page.isdigit():
            return int(page)
        else:
            if page == "v": return 5
            elif page == "vi": return 6
            elif page == "vii": return 7
            elif page == "viii": return 8
            else: return 0
    return 0

def process_file(file_path):
    intro_cutoff_point = 0
    last_index_dottedline_number = 0
    last_index_undottedline_number = 0
    bibliography_line_number = None
    legal_statement_line_number = None
    page_number = 0  # Read pages variable
    page_number_line = 1

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        # Set bibliography_line_number to last line number by default
        txt_length = len(lines)
        bibliography_line_number = txt_length
        
        for line_number, line in enumerate(lines, 1):
            # First, check for index entries (within the first half of the text)
            if line_number < 400:
                if index_pattern.search(line):
                    last_index_dottedline_number = line_number
                if index_without_dots.search(line):
                    last_index_undottedline_number = line_number
                # Simultaneously check for the seventh page patterns and store the line if found
                if page_number not in {7, 8}:
                    new_page_number = find_page_number(line)
                    if new_page_number - page_number > 0 and new_page_number < 9:
                        page_number = new_page_number
                        page_number_line = line_number
            
            # Check for bibliography after 90% of the document
            if (line_number / txt_length) > 0.9 and find_bibliography_line(line) and not_with_fullstop(line):
                bibliography_line_number = line_number - 1
                break  # Stop processing after finding bibliography
            
            # If bibliography not found, check for legal statement
            if bibliography_line_number == txt_length and find_legal_statement_line(line):
                bibliography_line_number = line_number - 1
        
        # If no index pattern was found, fall back to the seventh page line number if it exists
        if last_index_dottedline_number:
            intro_cutoff_point = last_index_dottedline_number
        elif last_index_undottedline_number:
            intro_cutoff_point = last_index_undottedline_number
        elif page_number_line is not None:
            intro_cutoff_point = page_number_line
    if intro_cutoff_point < 2:
        #print(os.path.basename(file_path))
        pass
    return intro_cutoff_point, bibliography_line_number, lines

def print_presentation(file_distances):
    # Calculate the top x% files with the longest distances
    total_files = len(file_distances)
    prcnt = 0.2
    top_n = max(1, math.ceil(total_files * prcnt))
    # Sort the files by distance in descending order
    sorted_files = sorted(file_distances, key=lambda x: x[1], reverse=True)
    top_files = sorted_files[:top_n]
    
    # Print the names and distances of the top 5% files
    print(f"\nTop {str(prcnt*100)}% files with the longest distance from bibliography_line to end of file:")
    for filename, distance in top_files:
        print(f"{filename}: {distance} lines after bibliography_line")

def main():
    # List to store distances from bibliography_line to end of file
    file_distances = []

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)
        
        # Skip directories and non-text files
        if os.path.isdir(file_path) or not filename.endswith('.txt'):
            continue
        
        # Process the file
        last_index_line_number, bibliography_line_number, lines = process_file(file_path)
        
        txt_length = len(lines)
        distance = txt_length - bibliography_line_number  # Distance from bibliography_line to end of file
        
        # Store the filename and distance
        file_distances.append((filename, distance))
        
        # Prepare output lines
        output_lines = []
        inside_content = False
        for line_number, line in enumerate(lines, 1):
            if line_number == last_index_line_number + 1:
                # Start of content
                inside_content = True
            if line_number == bibliography_line_number + 1:
                # End of content
                inside_content = False
            if inside_content:
                output_lines.append(line)
                
        # Write the transformed lines to the output file
        output_file_path = os.path.join(output_directory, filename)
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(output_lines)

if __name__ == '__main__':
    main()
