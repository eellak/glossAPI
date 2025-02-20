"""
This script extracts random articles from a parquet dataset for annotation purposes.
It keeps track of previously extracted articles to ensure no duplicates are selected
across multiple runs.
"""

import numpy as np
import pandas as pd
import os 

# 1. Load the source dataset
# Using a smaller dataset for testing/development purposes
input_file = '/mnt/data/Kostis/mini_df.parquet'
# Production dataset path (commented out)
# input_file = '/mnt/data/sections_for_annotation_predictions.parquet'
df = pd.read_parquet(input_file)

# 2. Initialize article pools
# Get all unique article filenames from the source dataset
filename_pool = df['filename'].unique()

# Initialize an empty DataFrame to store previously extracted articles
folder_df = pd.DataFrame()

# 3. Load previously extracted articles
# Check if the output directory exists and load any existing extracted datasets
if os.path.exists("sections_for_annotation_datasets"): 
    for file in os.listdir('sections_for_annotation_datasets'):
        file_df = pd.read_csv(f'sections_for_annotation_datasets/{file}')
        print('articles in file:', len(file_df['filename'].unique()))
        folder_df = pd.concat([folder_df, file_df])
    print('articles in folder:', len(folder_df['filename'].unique()))

# 4. Create pool of unused articles
# Get filenames of previously used articles, empty list if none exist
try:
    used_filename_pool = folder_df['filename'].unique()
except:
    used_filename_pool = []

# Filter out previously used articles to get clean pool of unused articles
clean_filename_pool = [filename for filename in filename_pool if filename not in used_filename_pool]

# Verify the math: clean pool + used pool should equal total pool
print(len(clean_filename_pool), '=', len(filename_pool), '-', len(used_filename_pool))

def get_n_number_of_articles(dataframe, n=100):
    """
    Extract n random articles from the dataframe that haven't been used before.
    
    Args:
        dataframe (pd.DataFrame): Source dataframe containing all articles
        n (int): Number of articles to extract (default: 100)
    
    Returns:
        pd.DataFrame: New dataframe containing the randomly selected articles
    """
    # Randomly select n filenames from the unused pool
    random_filenames = np.random.choice(clean_filename_pool, size=n, replace=False)

    # Create a new dataframe to store the selected articles
    new_dataframe = pd.DataFrame()

    # Extract all rows for each selected article and combine them
    for filename in random_filenames:
        single_article_df = dataframe[dataframe['filename'] == filename].sort_values('row_id')
        new_dataframe = pd.concat([new_dataframe, single_article_df], axis=0)

    return new_dataframe

# 5. Extract new batch of random articles
df_test = get_n_number_of_articles(df)

# 6. Save the extracted articles
# Create output directory if it doesn't exist
if not os.path.exists("sections_for_annotation_datasets"): 
    os.makedirs("sections_for_annotation_datasets") 

# Save the new dataset with an incremental number
dataset_no = len(os.listdir("sections_for_annotation_datasets")) + 1
df_test.to_csv(f'sections_for_annotation_datasets/sections_for_annotation_dataset_{dataset_no}.csv')
