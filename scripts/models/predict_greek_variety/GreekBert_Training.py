import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

data_for_BERT = pd.read_csv("~/Projects/GlossAPI/glossAPI/data/data_for_bert.csv", sep=None, engine='python')

# Ensure 'text' column contains strings
data_for_BERT['text'] = data_for_BERT['text'].astype(str)

print("columns of csv: ", data_for_BERT.columns)

print( "counts: ", data_for_BERT.label.value_counts() )

# Split the dataset into train and validation sets
train_data, val_data = train_test_split(data_for_BERT, test_size=0.3, random_state=42)

# Load the GreekBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# Tokenize the data
def tokenize_data(data):
    texts = data["text"].tolist()
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512)  # with 512 session -Moshan

train_encodings = tokenize_data(train_data)
val_encodings = tokenize_data(val_data)

train_labels = train_data["label"].tolist()
val_labels = val_data["label"].tolist()

# Convert the encoded inputs and labels to PyTorch tensors
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

num_labels= max(data_for_BERT["label"]) + 1

# Load the pre-trained GreekBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/bert-base-greek-uncased-v1", num_labels=4)

#device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    #no_cuda=True,  # Force CPU usage
    dataloader_num_workers=8,
    num_train_epochs=10,
    per_device_train_batch_size=8, #with 16 also session was crashing
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    gradient_accumulation_steps=8,  # Simulate a larger batch size
    log_level="info",
    logging_steps=100,
)

# Define the function to compute the accuracy
def compute_accuracy(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_accuracy,
)

# Training loop
trainer.train()

# Evaluate the model
trainer.evaluate()

# Access the trained model from the Trainer object
trained_model = trainer.model

# Save the trained model
trained_model.save_pretrained("~/Projects/GlossAPI/glossAPI/scripts/model/saved_models")


