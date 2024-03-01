from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import sqlite3
import threading

# Thread-local storage
threadLocal = threading.local()

threadLocal = threading.local()

def get_db_connection():
    db = getattr(threadLocal, 'db', None)
    if db is None:
        db = threadLocal.db = sqlite3.connect('greek_words.db')
    return db

def data_generator():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT word FROM words")
    for word_record in cursor:
        yield word_record[0]
    cursor.close()
    conn.close()

# Initialize a tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Initialize the trainer for BPE
trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
                     vocab_size=60_000,
                     min_frequency=5)

# Train the tokenizer
tokenizer.train_from_iterator(data_generator(), trainer=trainer)

tokenizer.save("greek_tokenizer.json")
