import pandas as pd

# Read the CSV file
df = pd.read_csv('random_paragraph_sample.csv')

# Drop the source_file column
df = df.drop('source_file', axis=1)

# Save back to the same file
df.to_csv('random_paragraph_sample.csv', index=False)

print("Removed source_file column from random_paragraph_sample.csv")
