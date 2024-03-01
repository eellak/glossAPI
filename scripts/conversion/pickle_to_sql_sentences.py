import sqlite3
import pickle
import os
import string

# Connect to SQLite database (or replace with your DB connection)
conn = sqlite3.connect('greek_sentences.db')
cursor = conn.cursor()

# Create a table (only run this once)
cursor.execute('''CREATE TABLE IF NOT EXISTS sentences (id INTEGER PRIMARY KEY, sentence TEXT, source TEXT)''')

# Function to load words from a pickle file and insert into DB
def insert_words_from_pickle(pickle_file, source):
    with open(pickle_file, 'rb') as file:
        sentences = pickle.load(file)

    for sentence in sentences:
        cursor.execute("INSERT INTO sentences (sentence, source) VALUES (?, ?)", (sentence, source))

# Insert words from each corpus
insert_words_from_pickle('bible_sentences.p', 'Bible')
insert_words_from_pickle('hnc_sentences.p', 'HNC')

# Loop through Europarl pickle files
pickle_directory = 'pickle_sentences/europarl'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'Europarl')

# Loop through GlobalVoices pickle files
pickle_directory = 'pickle_sentences/global_voices'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'GlobalVoices')

# Loop through Wikipedia pickle files
pickle_directory = 'pickle_sentences/wikipedia'  # Update with the correct path
for filename in os.listdir(pickle_directory):
    if filename.endswith('.p'):
        insert_words_from_pickle(os.path.join(pickle_directory, filename), 'Wikipedia')

# Commit and close
conn.commit()
conn.close()
