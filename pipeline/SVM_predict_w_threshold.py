#!/usr/bin/env python3
import os
import re
import time
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import FunctionTransformer

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utility: Function to combine header and section text (used only in training)
# -----------------------------------------------------------------------------
def combine_text(df):
    return (df['header'].fillna('') + " " + df['section'].fillna('')).values

# -----------------------------------------------------------------------------
# Build the Training Pipeline
# -----------------------------------------------------------------------------
combined_text_pipeline = Pipeline([
    ('combine', FunctionTransformer(combine_text, validate=False)),
    ('tfidf', TfidfVectorizer(max_features=2000))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('combined_text', combined_text_pipeline, ['header', 'section']),
        ('binary', 'passthrough', ['has_table', 'has_list'])
    ],
    remainder='drop'
)

clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC(class_weight='balanced', max_iter=10000))
])

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_model():
    train_csv_file = "/mnt/data/section_classifier/training_dataset_updated_06_02.csv"
    logger.info(f"Loading training data from {train_csv_file}...")
    df_train = pd.read_csv(train_csv_file)
    logger.info(f"Loaded training data with {len(df_train)} rows.")
    
    # Preprocess training data
    df_train["header"] = df_train["header"].fillna("")
    df_train["section"] = df_train["section"].fillna("")
    df_train['has_table'] = df_train['has_table'].astype(int)
    df_train['has_list'] = df_train['has_list'].astype(int)
    
    # Select features and target
    X = df_train[['header', 'section', 'has_table', 'has_list']]
    y = df_train['label']
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Training set: {len(X_train)} rows; Validation set: {len(X_val)} rows.")
    
    # Train the pipeline (this will combine text and vectorize)
    logger.info("Starting model training...")
    start_time = time.time()
    clf_pipeline.fit(X_train, y_train)
    end_time = time.time()
    logger.info(f"Model training completed in {end_time - start_time:.2f} seconds.")
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    y_val_pred = clf_pipeline.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    report = classification_report(y_val, y_val_pred)
    logger.info("Confusion Matrix:\n" + str(cm))
    logger.info("Classification Report:\n" + report)

# -----------------------------------------------------------------------------
# Your robust non-ML function to detect likely index sections
# -----------------------------------------------------------------------------
import re
import pandas as pd

def compute_likely_index_for_section(text, min_seq_length=7, allowed_consecutive_equals=4):
    """
    Process the section text line by line.
    
    For every table row, the algorithm:
      - Checks for lines starting and ending with '|'
      - Merges two consecutive lines if the first (after stripping) starts with '|'
        but does not end with '|' and the following line (after stripping) does not start with '|'
        but ends with '|'.
      - Removes extra spaces and cleans repetitive noise (e.g. multiple dots, dashes, underscores).
      - Splits the cleaned line into two halves.
      - [New Rule 3] If the first half contains only numbers (and punctuation), reject this row.
      - Searches for numbers (1-3 digits and less than 400) from the second half:
          * First, looks for a range (e.g. "1-17") and captures the first number.
          * Then, if no range is detected, uses a general pattern.
          * [New Rule 1] Rejects a candidate if it contains a comma.
          * [New Rule 2] When scanning from the right, if any English or Greek letter appears after the candidate,
            the candidate is rejected.
          * Appends the first valid number found (starting from the rightmost candidate) to a list.
    
    [New Rule 4] After processing all lines, if the section appears to consist almost entirely of table rows
    (i.e. all lines or all but one are table rows), then even if the list has fewer than min_seq_length numbers,
    if there are at least 4 numbers and they are in ascending order, accept the section as an index.
    
    Then it checks for a contiguous non-decreasing sequence of at least `min_seq_length` numbers,
    allowing up to `allowed_consecutive_equals` consecutive identical numbers.
    
    Returns a tuple: (1 if such a sequence is found (or the table-only rule applies), else 0, list of numbers detected)
    """
    # Handle NaN or non-string values
    if pd.isna(text) or not isinstance(text, str):
        return 0, []
    
    # General number pattern: allow an optional fractional part.
    pattern = re.compile(r'\b(\d{1,3}(?:\.\d+)?)\b')
    # Pattern to capture a range like "1-17" (we want the first number)
    range_pattern = re.compile(r'\b(\d{1,3})\s*-\s*(\d{1,3})\b')
    
    numbers = []
    lines = text.splitlines()
    i = 0
    table_lines_count = 0  # count of lines processed as table rows
    total_lines = len(lines)
    
    while i < len(lines):
        line = lines[i].strip()
        combined_line = line

        # If the current line starts with '|' but doesn't end with it,
        # check if the next line can be merged.
        if line.startswith('|') and not line.endswith('|'):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (not next_line.startswith('|')) and next_line.endswith('|'):
                    combined_line = line + " " + next_line
                    i += 1  # Skip the next line since it is merged

        # Process the combined line if it qualifies as a table row.
        if combined_line.startswith('|') and combined_line.endswith('|'):
            table_lines_count += 1
            # Collapse multiple spaces.
            cleaned_line = " ".join(combined_line.split())
            # --- Additional cleaning ---
            # Replace sequences (of 3 or more) of dots, dashes, or underscores (with optional spaces) with a single space.
            cleaned_line = re.sub(r'(?:(?:[.\-_]\s?){3,})', ' ', cleaned_line)
            
            mid_index = len(cleaned_line) // 2

            # [New Rule 3] Check if the first half has only numbers/punctuation.
            first_half = cleaned_line[:mid_index]
            if re.search(r'[A-Za-zΑ-Ωα-ω]', first_half) is None:
                # Reject this row (skip number extraction for it)
                i += 1
                continue

            second_half = cleaned_line[mid_index:]
            
            # --- First, try to detect a range (e.g. "1-17") ---
            range_match = range_pattern.search(second_half)
            if range_match:
                candidate = range_match.group(1)
                after_candidate = second_half[range_match.end(1):]
                # [New Rule 1] Reject candidate if it contains a comma.
                if ',' in candidate:
                    pass
                # [New Rule 2] Reject if any letter appears after the candidate.
                elif re.search(r'[A-Za-zΑ-Ωα-ω]', after_candidate):
                    pass
                else:
                    try:
                        num = int(candidate)
                        if num < 400:
                            numbers.append(num)
                            i += 1
                            continue  # Found a valid candidate; move to next line.
                    except ValueError:
                        pass

            # --- Fallback: use the general number pattern ---
            matches = list(pattern.finditer(second_half))
            if matches:
                # Iterate in reverse order (rightmost candidate first)
                for m in reversed(matches):
                    candidate = m.group(1)
                    # [New Rule 1] Reject candidate if it contains a comma.
                    if ',' in candidate:
                        continue
                    # [New Rule 2] Reject if any letter appears after the candidate.
                    after_candidate = second_half[m.end():]
                    if re.search(r'[A-Za-zΑ-Ωα-ω]', after_candidate):
                        continue
                    try:
                        num = int(candidate)
                        if num < 400:
                            numbers.append(num)
                            break  # Pick only one number per row.
                    except ValueError:
                        continue
        i += 1

    # [New Rule 4] If nearly all lines are table rows, accept with a lower threshold.
    if len(numbers) < min_seq_length:
        if table_lines_count >= (total_lines - 1) and len(numbers) >= 4:
            ascending = True
            for j in range(1, len(numbers)):
                if numbers[j] < numbers[j - 1]:
                    ascending = False
                    break
            if ascending:
                return 1, numbers

    # Not enough numbers found for normal processing.
    if len(numbers) < min_seq_length:
        return 0, numbers

    # Check for a contiguous non-decreasing sequence of at least min_seq_length numbers,
    # allowing allowed_consecutive_equals consecutive identical numbers.
    current_seq = 1
    equal_count = 1  # count for consecutive equal numbers
    for j in range(1, len(numbers)):
        if numbers[j] > numbers[j - 1]:
            current_seq += 1
            equal_count = 1  # reset count of equal numbers
        elif numbers[j] == numbers[j - 1]:
            equal_count += 1
            if equal_count <= allowed_consecutive_equals:
                current_seq += 1
            else:
                # Exceeded allowed consecutive equals; reset sequence.
                current_seq = 1
                equal_count = 1
        else:
            current_seq = 1
            equal_count = 1

        if current_seq >= min_seq_length:
            return 1, numbers

    return 0, numbers

# -----------------------------------------------------------------------------
# Adjust Predictions with Combined Logic
# -----------------------------------------------------------------------------
def adjust_predictions_with_index_detection(df, length_threshold=300, propo_threshold=300):
    """
    For each article (grouped by filename), sort sections in ascending order of 'id',
    and accumulate 'section_length' and 'section_propo'. For rows before the thresholds
    are exceeded, run the compute_likely_index_for_section function on the section text
    (without the header). If it returns 1, override the SVM prediction with "π".
    
    For rows after the thresholds are exceeded, the SVM prediction remains unchanged.
    
    Returns the modified DataFrame.
    """
    def process_group(group):
        group = group.sort_values('id').copy()
        running_length = 0
        running_propo = 0
        new_preds = []
        passed_threshold = False  # Flag to track if we've passed the thresholds
        # Process each row in order
        for idx, row in group.iterrows():
            running_length += row['section_length']
            running_propo += row['section_propo']
            # Check if both cumulative metrics are below or equal to the thresholds.
            if running_length <= length_threshold and running_propo <= propo_threshold:
                # Run the robust index detection on the section text (without header)
                flag, nums = compute_likely_index_for_section(row['section'])
                if flag == 1:
                    # If detected as index, override with "π" and skip SVM decision.
                    new_preds.append("π")
                    continue
            else:
                passed_threshold = True  # Mark that we've passed the thresholds
            
            # For sections after threshold, convert "π" predictions to "άλλο"
            if passed_threshold and row['predicted_section'] == "π":
                new_preds.append("άλλο")
            else:
                # Otherwise, keep the SVM prediction
                new_preds.append(row['predicted_section'])
        
        group['predicted_section'] = new_preds
        return group

    return df.groupby('filename', group_keys=False).apply(process_group)

# -----------------------------------------------------------------------------
# Process Predictions Using Dask with Progress Bar and Checkpointing
# -----------------------------------------------------------------------------
def process_predictions_with_dask(input_parquet, output_parquet):
    """
    Reads the input Parquet file in parallel using Dask, applies the fitted 
    clf_pipeline to generate predictions, then post-processes the predictions.
    
    For each article (grouped by filename and sorted by id), for rows before the 
    cumulative section_length and section_propo thresholds, the section text (without 
    the header) is passed to compute_likely_index_for_section. If that function returns 1,
    the predicted label is overridden to "π" (skipping the SVM prediction). Otherwise, 
    the SVM prediction is used.
    
    Finally, the results are written to an output Parquet file.
    """
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    # Read required columns from the input Parquet file.
    needed_columns = ['id', 'row_id', 'filename', 'has_table', 'has_list', 'header', 
                      'place', 'section', 'label', 'section_propo', 'section_length']
    logger.info(f"Reading data from Parquet file: {input_parquet} (columns: {needed_columns})")
    ddf = dd.read_parquet(input_parquet, columns=needed_columns)
    
    # Preprocess key columns
    ddf['header'] = ddf['header'].fillna('')
    ddf['section'] = ddf['section'].fillna('')
    ddf['has_table'] = ddf['has_table'].astype(int)
    ddf['has_list'] = ddf['has_list'].astype(int)
    
    # Identify already processed rows via output file checkpoint (read only 'id')
    processed_ids = set()
    if os.path.exists(output_parquet):
        try:
            logger.info(f"Found existing output file {output_parquet}. Loading processed IDs...")
            ddf_existing = dd.read_parquet(output_parquet, columns=['id'])
            processed_ids = set(ddf_existing['id'].unique().compute().tolist())
            logger.info(f"{len(processed_ids)} rows have already been processed.")
        except Exception as e:
            logger.error("Error reading existing output file. Processing all rows.", exc_info=e)
    
    if processed_ids:
        # Filter out already predicted rows using efficient isin on the small set of IDs.
        ddf = ddf[~ddf['id'].isin(list(processed_ids))]
        remaining = int(ddf.shape[0].compute())
        logger.info(f"Processing only {remaining} remaining rows.")
    else:
        logger.info("No prior progress found; processing all rows.")
    
    # (Optional) Repartition to a moderate number of partitions
    ddf = ddf.repartition(npartitions=20)
    ddf = ddf.persist()  # Cache in memory to avoid repeated computations
    
    # Create metadata that includes the new column "predicted_section"
    meta = ddf._meta.copy()
    meta['predicted_section'] = ''
    
    def predict_partition(df):
        # Run SVM prediction on the subset of features.
        X = df[['header', 'section', 'has_table', 'has_list']]
        df = df.copy()  # Avoid SettingWithCopyWarning
        # Initially assign SVM predictions.
        df['predicted_section'] = clf_pipeline.predict(X)
        return df
    
    logger.info("Running predictions in parallel using Dask...")
    start = time.time()
    ddf_pred = ddf.map_partitions(predict_partition, meta=meta)
    
    temp_output = output_parquet + ".tmp"
    logger.info("Writing new predictions to temporary Parquet file with progress bar...")
    with ProgressBar():
        ddf_pred.to_parquet(temp_output, write_index=False)
    end = time.time()
    logger.info(f"New predictions written in {end - start:.2f} seconds.")
    
    # Merge new predictions with existing ones, if any.
    if processed_ids:
        logger.info("Merging new predictions with existing progress...")
        df_existing = dd.read_parquet(output_parquet).compute()
        df_new = dd.read_parquet(temp_output).compute()
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = dd.read_parquet(temp_output).compute()
    
    # Post-process predictions: for each article, before the cumulative thresholds are exceeded,
    # run compute_likely_index_for_section on the section text (without the header) and override
    # the predicted label with "π" if detected.
    logger.info("Adjusting predictions based on index detection and cumulative thresholds...")
    df_final = adjust_predictions_with_index_detection(df_final, length_threshold=300, propo_threshold=300)
    
    logger.info("Writing merged and adjusted predictions to final output Parquet file...")
    df_final.to_parquet(output_parquet, index=False)
    logger.info("Predictions saved and progress maintained.")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    overall_start = time.time()
    
    # 1. Train the model using the training dataset.
    train_model()
    
    # 2. Process predictions on the Parquet file.
    #    The output file will have a "predicted_section" column annotated by both
    #    methods (table detection and SVM), following the threshold logic.
    input_parquet = "/mnt/data/sections_for_annotation_section_sizes.parquet"
    output_parquet = "/mnt/data/sections_for_annotation_predictions.parquet"
    process_predictions_with_dask(input_parquet, output_parquet)
    
    overall_end = time.time()
    logger.info(f"Total elapsed time: {overall_end - overall_start:.2f} seconds.")

# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
