import pandas as pd
import numpy as np
from transformers import AutoTokenizer, RobertaModel
from transformers import AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast
import pickle
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import PreTrainedTokenizerFast
from transformers import RobertaModel, RobertaConfig
import argparse
import os
import random
import sqlite3
import threading
import torch
import torch.nn as nn



parser = argparse.ArgumentParser(
        prog="This is the language finetuning script",
        description="Train TinyLlama on a selection of Greek texts from glossapi",
        epilog="This is experimental work in progress, use with caution. No warranty whatsoever."
        )
parser.add_argument('-d', '--device', help='Define the device', default="cuda")
parser.add_argument('-b', '--batch-size', help='Set the batch size', default=8)
parser.add_argument('-m', '--base-model', help='Set the foundation model to finetune', default="TinyLlama/TinyLlama-1.1B-step-50K-105b")
# parser.add_argument('-l', '--num-labels', help='Set number of outcome labels', default=6)
parser.add_argument('-e', '--num-epochs', help='Set number of epochs', default=1)
parser.add_argument('-r', '--learning-rate', help='Set learning rate', default=9e-5)
parser.add_argument('-w', '--num-workers', help='Set number of workers', default=8)
# parser.add_argument('-t', '--decision-threshold', help='Set the decision threshold', default=0.7)
# parser.add_argument('-p', '--tokenizer-parallelism', help='Set the enviroment variable to true or false', default="true")
parser.add_argument('-s', '--save-dir', help='Directory in which to Save the resulting model', default="saved_model")
# parser.add_argument('-c', '--convert', help='Run the data conversion scripts first (When you update data files)', default=False)
args = parser.parse_args()


TOKENIZER_MODEL=''
BASE_MODEL = args.base_model 
BATCH_SIZE = int(args.batch_size) # Set your batch size
ACCUMULATION_STEPS=1
N_WORKERS = int(args.num_workers)  # Set the number of workers
MASK_PROBABILITY = 0.1 # How probable it is for a token to be masked
SAVEDIR = args.save_dir
LEARNING_RATE = float(args.learning_rate)
N_EPOCHS = int(args.num_epochs)
DEVICE = args.device

# This is to resolve the "Too many open files" Runtime Error
torch.multiprocessing.set_sharing_strategy('file_system')

print("Libraries are loaded")

# Thread-local storage
threadLocal = threading.local()

# Set up SQLite connection
def get_db_connection():
    db = getattr(threadLocal, 'db', None)
    if db is None:
        db = threadLocal.db = sqlite3.connect('greek_sentences.db', check_same_thread=False)
    return db

print("Working with Database streaming")

# Define a data generator function

def data_generator(sources, percentages, L):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for source, percentage in zip(sources, percentages):
            sample_size = round(percentage * L)
            cursor.execute("SELECT sentence FROM sentences WHERE source=? ORDER BY RANDOM() LIMIT ?", (source, sample_size))
            for row in cursor.fetchall():
                yield row[0]
        cursor.close()


class GreekDataset(Dataset):
    def __init__(self, sources, percentages, L, tokenizer):
        self.generator = data_generator(sources, percentages, L)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id

    def __getitem__(self, idx):
        sentence = next(self.generator)
        # Ensure token_type_ids are not returned by the tokenizer
        tokens = self.tokenizer(sentence, padding="max_length", truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False)
        
        # Prepare labels for MLM, similar to before
        labels = tokens['input_ids'].clone()
        rand = torch.rand(labels.shape)
        mask_arr = (rand < MASK_PROBABILITY) * (labels != self.pad_token_id)
        tokens['input_ids'][mask_arr] = self.mask_token_id
        labels[~mask_arr] = -100  # Only compute loss for masked tokens

        item = {key: val.squeeze(0) for key, val in tokens.items()}  # Remove batch dimension
        item['labels'] = labels.squeeze(0)

        return item

    def __len__(self):
        # Return an estimated length based on L and the average sentence length
        return int(L * len(sources) // len(percentages))

# Initialize tokenizer and model
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="greek_tokenizer160.json")

# GPT-4 rather has it like 
tokenizer = PreTrainedTokenizerFast(tokenizer_file="greek_tokenizer160.json",
                                     model_max_length=512,
                                     padding_side='right',
                                     truncation_side='right',
                                     pad_token='<pad>',
                                     mask_token='<mask>',
                                     unk_token='<unk>',
                                     bos_token='<s>',
                                     eos_token='</s>')

# Ensure tokenizer's special tokens are set if they're not automatically recognized
# tokenizer.add_special_tokens({'pad_token': '<pad>', 'mask_token': '<mask>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'})

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

tokenizer.add_special_tokens({'pad_token': '<pad>', 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'mask_token': '<mask>'})
model.resize_token_embeddings(len(tokenizer))

# Define sources, percentages, and L
sources = ['Europarl', 'HNC']
percentages = [.5, .5]
L = 2000

# Select a single fold for testing
fold = 0  # You can change this to test different folds
# train_sources = [sources[i] for i in range(len(sources)) if i % 2 != fold]
# val_sources = [sources[i] for i in range(len(sources)) if i % 2 == fold]

# Initialize DataLoader instances for training and validation
train_dataset = GreekDataset(sources, percentages, L, tokenizer)
val_dataset = GreekDataset(sources, percentages, L, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

# Initialize the model, optimizer, and learning rate scheduler for this fold
model = model.to(torch.device(DEVICE))
optim = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-06)
num_training_steps = N_EPOCHS * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)

############################## NEW TRAINING LOOP ################################################


training_logs = []

# Initialize the DataLoader with the new batch size
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS)

# Adjust the training loop to handle gradient accumulation correctly
optimization_steps = 0
model.train()
for epoch in range(N_EPOCHS):
    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss 
        
        loss.backward()

        optim.step()
        lr_scheduler.step()
        optim.zero_grad()
        optimization_steps += 1

        # Log training progress
        if (i + 1) % ACCUMULATION_STEPS == 0:
            print(f"Epoch: {epoch}, Step: {i+1}, Loss: {loss.item()}")

            # Record training information
            training_logs.append({
                'epoch': epoch,
                'batch': batch,
                'loss': loss.item()
                # You can add more metrics here if needed
            })
            print(f"Loss: {loss.item()}")

            # Optional: print more information for monitoring
            input_snippet = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
            print(f"Input snippet: {input_snippet[:50]}...")

    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # No need to track gradients for validation
        for batch in val_dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
    val_loss /= len(val_dataloader)
    print(f"Validation Loss: {val_loss}")
    model.train()  # Set the model back to training mode

df = pd.DataFrame.from_records(training_logs)

df.to_csv('training_logs.csv', index=False)

# Save the model for this fold
model.save_pretrained(os.path.join(SAVEDIR, f"saved_model_fold_{fold}"))
