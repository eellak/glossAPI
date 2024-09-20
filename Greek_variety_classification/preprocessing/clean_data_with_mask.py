import pandas as pd
import re
import os
from transformers import AutoTokenizer

# Set the working directory and filename
working_directory = "/home/fivos/Downloads"
file_name = "dataset_Sep_3.csv"

os.chdir(working_directory)

# Load the data
data = pd.read_csv(file_name, sep=",", engine="python")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

print(data.columns)

# Ensure 'text' column contains strings
if 'text' in data.columns:
    data["text"] = data["text"].astype(str)
else:
    print("Column 'text' not found in the dataset.")

# Function to check if text is mostly Latin characters
def is_mostly_latin(text, threshold=0.5):
    latin_chars = re.findall(r"[a-zA-Z]", text)
    return (len(latin_chars) / len(text)) > threshold if len(text) > 0 else False

# Function to check if text is more than 50% numbers
def is_mostly_numbers(text, threshold=0.5):
    num_chars = re.findall(r"[0-9]", text)
    return (len(num_chars) / len(text)) > threshold if len(text) > 0 else False

# Function to check if text has fewer than 4 words
def too_short(text):
    return len(text.split()) < 4

# Function to check if text has more than 512 tokens
def has_more_than_512_tokens(text):
    # Fragments should be smaller than 512 tokens for GreekBERT
    return len(tokenizer.encode(text)) > 512

# Function to clean text
def clean_text(text):
    # Remove formatting characters
    text = re.sub(r"[\n\t]", " ", text)
    # Remove leading, trailing, and multiple spaces
    text = ' '.join(text.split())
    # Remove numerical ordering elements with ()
    text = re.sub(r"\(?\d+\)|\d+\.", "", text)
    # Remove numerical ordering elements with {}
    text = re.sub(r"\{\d+\}", "", text)
    # Remove ordering elements with Greek numerals
    text = re.sub(
        r"(?<!\S)(?:\(?(στ|[α-ω]|ΣΤ|[Α-Ω])(\ʹ|\'|\´|\)|\.))(?!\S)", "", text
    )
    # Remove annoying characters
    text = re.sub(r"[↩\[\]—―]", "", text)
    # Remove & character that is not seperated by space
    # Merge non-space sequences around '&'
    text = re.sub(r'(\S+)&(\S+)', r'\1\2', text)
    # Remove '&' before non-space sequences
    text = re.sub(r'&(\S+)', r'\1', text)
    # Remove '&' after non-space sequences
    text = re.sub(r'(\S+)&', r'\1', text)

    # Remove numbers of the form /number/
    text = re.sub(r'(/\d+/)', "", text)
    # Remove Latin numbers (up to 20 for this example)
    latin_numbers = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                     'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']
    latin_number_pattern = r'\b(' + '|'.join(latin_numbers) + r')(\.|\))'
    text = re.sub(latin_number_pattern, "", text)
    
    return text

# Create the mask column
data['mask'] = 1

# Apply the cleaning function to all rows
data["text"] = data["text"].apply(clean_text)

# Strip trailing whitespaces
data["text"] = data["text"].apply(str.strip)

# Final step: Remove any instances of "<>", "()", "{}", and other similar characters
data["text"] = data["text"].apply(lambda x: re.sub(r'[<>\[\]\(\)\{\}]', '', x))

# Update mask for empty text cells
data.loc[~data["text"].str.strip().astype(bool), 'mask'] = 0

# Update mask for mostly Latin text
data.loc[data["text"].apply(is_mostly_latin), 'mask'] = 0

# Update mask for mostly numbers
data.loc[data["text"].apply(is_mostly_numbers), 'mask'] = 0

# Update mask for too_short text
data.loc[data["text"].apply(too_short), 'mask'] = 0

# Update mask for text with more than 512 tokens
data.loc[data["text"].apply(has_more_than_512_tokens), 'mask'] = 0

# Save the result to a new CSV file
output_file_path = os.path.join(os.getcwd(), os.path.splitext(file_name)[0] + "_masked.csv")
data.to_csv(output_file_path, index=False, quoting=1)  # quoting=1 ensures all fields are quoted

print("Cleaned data with mask saved")