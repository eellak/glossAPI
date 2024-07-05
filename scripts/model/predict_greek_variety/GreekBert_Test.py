import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import os
import numpy as np

# Load the data
data_for_BERT = pd.read_csv("~/Projects/GlossAPI/glossAPI/data/data_for_bert.csv", sep=None, engine='python')
data_for_BERT['text'] = data_for_BERT['text'].astype(str)

# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data_for_BERT, test_size=0.3, random_state=42)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

model_path = os.path.expanduser("~/Projects/GlossAPI/glossAPI/scripts/model/saved_models")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)


# Set the model to evaluation mode
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_text(text):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    return predicted_class, probabilities.squeeze().tolist()

# Predict on test data
true_labels = []
predicted_labels = []
probabilities_list = []

for _, row in test_data.iterrows():
    text = row['text']
    true_label = row['label']
    
    predicted_class, probabilities = predict_text(text)
    
    true_labels.append(true_label)
    predicted_labels.append(predicted_class)
    probabilities_list.append(probabilities)
    
    print(f"Text: {text[:50]}...")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_class}")
    print(f"Probabilities: {probabilities}")
    print("--------------------")

# Convert labels to 0-based index
true_labels = np.array(true_labels) - 1
predicted_labels = np.array(predicted_labels)

# Print classification report
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

# Calculate and print overall accuracy
accuracy = (true_labels == predicted_labels).mean()
print(f"\nOverall Accuracy: {accuracy:.4f}")