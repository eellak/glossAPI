from gensim import corpora, models
from nltk.tokenize import word_tokenize
import gensim
import pickle
import sqlite3
import string

# Load Greek stop words
with open('stpwds.txt') as f:
    greek_stop_words = [x.strip() for x in f.readlines()] 

def preprocess_text(text):
    # Basic preprocessing steps: lowercasing, removing punctuation, and tokenizing
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    word_tokens = word_tokenize(text, language='greek')  # Ensure proper language setting

    # Removing Greek stop words
    return [word for word in word_tokens if word not in greek_stop_words]

# Flexible data generator
def lda_data_generator(sources, percentages, L, use_db=True):
    if use_db:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for source, percentage in zip(sources, percentages):
                sample_size = round(percentage * L)
                cursor.execute("SELECT sentence FROM sentences WHERE source=? ORDER BY RANDOM() LIMIT ?", (source, sample_size))
                for row in cursor.fetchall():
                    yield preprocess_text(row[0])
            cursor.close()
    else:
        # Use the pickled dataset
        with open('result_sentences.pkl', 'rb') as f:
            sample_sentences = pickle.load(f)
            for sentence in sample_sentences:
                yield preprocess_text(sentence)

# Usage
# Define sources, percentages, and L
sources = ['Wikipedia', 'GlobalVoices', 'Europarl', 'HNC']
percentages = [0.2, 0.25, 0.15, 0.4]
L = 500


use_db = False  # Set to True to use data from DB, False to use pickled data
documents = list(lda_data_generator(sources, percentages, L, use_db))

# Create a Gensim Dictionary and Corpus in one go (to avoid generator exhaustion)
dictionary = corpora.Dictionary()
corpus = []

for document in documents:
    dictionary.add_documents([document])
    corpus.append(dictionary.doc2bow(document))

with open("lda_data_objects.p", "wb") as f:
    out = {'dictionary' : dictionary, "corpus" : corpus }
    pickle.dump(out, f)
    

##### ENTER GENSIM #####

run_model = False
# Set up the LDA model

if run_model:
    lda_model = gensim.models.LdaMulticore(corpus, id2word=dictionary, num_topics=16, passes=10, workers=2)
    # Save the model for later use
    lda_model.save("lda_model")
