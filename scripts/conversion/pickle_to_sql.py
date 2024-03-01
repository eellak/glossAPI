import sqlite3
import pickle
import os
import string

# Function to check if a word contains punctuation
def contains_punctuation(word):
    return any(char in string.punctuation for char in word)

# Connect to SQLite database (or replace with your DB connection)
conn = sqlite3.connect('greek_words2.db')
cursor = conn.cursor()

# Create a table (only run this once)
cursor.execute('''CREATE TABLE IF NOT EXISTS words (id INTEGER PRIMARY KEY, word TEXT, source TEXT)''')

# Function to load words from a pickle file and insert into DB
def insert_words_from_pickle(pickle_file, source):
    with open(pickle_file, 'rb') as file:
        words = pickle.load(file)

    for word in words:
        if not contains_punctuation(word):
            cursor.execute("INSERT INTO words (word, source) VALUES (?, ?)", (word, source))

# Insert words from each corpus
insert_words_from_pickle('bible.p', 'Bible')
insert_words_from_pickle('hnc_words.p', 'HNC')

# Loop through Europarl pickle files
pickle_directory = 'text_pickles'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'Europarl')

# Loop through GlobalVoices pickle files
pickle_directory = 'global_voices/pickles'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'GlobalVoices')

# Loop through Wikipedia pickle files
pickle_directory = 'wikipedia/pickles'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'Wikipedia')

# Commit and close
conn.commit()
conn.close()
