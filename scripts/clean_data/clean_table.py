import pandas as pd
import re
import os

# Set the working directory
os.chdir("~/glossAPI/data/")

# Load the data
data = pd.read_csv("text_and_annotation.csv", sep=",", engine="python")

print(data.columns)

# Ensure 'text' column contains strings
if 'text' in data.columns:
    data["text"] = data["text"].astype(str)
else:
    print("Column 'text' not found in the dataset.")

# Remove rows with empty text cells
data = data[data["text"].str.strip().astype(bool)]

# Function to check if text is mostly Latin characters
def is_mostly_latin(text, threshold=0.5):
    latin_chars = re.findall(r"[a-zA-Z]", text)
    return (len(latin_chars) / len(text)) > threshold

# Function to check if text is more than 50% numbers
def is_mostly_numbers(text, threshold=0.5):
    num_chars = re.findall(r"[0-9]", text)
    return (len(num_chars) / len(text)) > threshold

# Function to check if text has fewer than 15 words
def has_fewer_than_15_words(text):
    return len(text.split()) < 15

# Function to clean text
def clean_text(text):
    # Remove leading, trailing, and multiple spaces
    text = re.sub(r"\s\s+", " ", text.strip())
    # Remove formatting characters
    text = re.sub(r"[\n\t]", " ", text)
    # Remove numerical ordering elements with ()
    text = re.sub(r"\(?\d+\)|\d+\.", "", text)
    # Remove numerical ordering elements with {}
    text = re.sub(r"\{?\d+\}", "", text)
    # Remove numerical ordering elements with Greek numerals
    text = re.sub(
        r"(?<!\S)(?:\(?(στ|[αβγδεζηθικλμνξοπρσςτυφχψω])(ʹ|\)|\.))(?!\S)", "", text
    )
    # Remove annoying characters
    text = re.sub(r"[↩\[\]—―]", "", text)
    return text

# Remove rows where text is mostly Latin characters
data = data[~data["text"].apply(is_mostly_latin)]

# Remove rows where text is more than 50% numbers
data = data[~data["text"].apply(is_mostly_numbers)]

# Remove rows where text has fewer than 15 words
data = data[~data["text"].apply(has_fewer_than_15_words)]

# Apply the cleaning function
data["text"] = data["text"].apply(clean_text)
# Strip trailing whitespaces
data["text"] = data["text"].apply(str.strip)

# Save the result to a new CSV file
data.to_csv("cleaned_text_and_annotation.csv", index=False, quoting=1)  # quoting=1 ensures all fields are quoted

print("Cleaned data saved")
