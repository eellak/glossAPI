import os
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# Step 1: Import necessary libraries (done above)

# Step 2: Load the CSV file
input_path = "~/Projects/GlossAPI/glossAPI/data/draw2texts.csv"
df = pd.read_csv(input_path)

# Step 3: Load the model and tokenizer
model_path = os.path.expanduser("~/Projects/GlossAPI/glossAPI/scripts/model/saved_models")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-greek-uncased-v1")

# Set the model to evaluation mode
model.eval()

# Ensure we're using CPU
device = torch.device("cpu")
model = model.to(device)

# Step 4: Prepare the data for classification
def prepare_input(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    return {k: v.to(device) for k, v in inputs.items()}

# Step 5: Perform the classification
predictions = []

for text in tqdm(df['text'], desc="Classifying texts"):
    with torch.no_grad():
        inputs = prepare_input(text)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predictions.append(predicted_class + 1)  # Add 1 to match your desired output format

# Step 6: Add predictions to the DataFrame
df['predicted class'] = predictions

# Step 7: Save the results to a new CSV file
output_path = "/home/fivos/Projects/GlossAPI/glossAPI/data/draw2texts_predicted.csv"
df.to_csv(output_path, index=False)

print(f"Classification completed. Results saved to {output_path}")
