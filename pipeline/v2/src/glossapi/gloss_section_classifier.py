import logging
import os
import re
import time
import numpy as np
import pandas as pd
import joblib
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import FunctionTransformer
from typing import Dict


# Define the combine_text function outside of the class for compatibility
def combine_text(df):
    """
    Combine header and section text into a single text field for analysis.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'header' and 'section' columns
        
    Returns:
        numpy.ndarray: Array of combined text strings
    """
    return (df['header'].fillna('') + " " + df['section'].fillna('')).values


class GlossSectionClassifier:
    """
    A classifier for document sections that uses machine learning to categorize
    sections into different types (e.g., introduction, index, bibliography, etc.)
    
    The class provides functionality to train a classifier model, process predictions
    in parallel using Dask, and annotate document sections based on specific boundary markers.
    """
    
    def __init__(self):
        # Setup Logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.loaded_model = None
        self.clf_pipeline = None
        self.combined_text_pipeline = None
        self.preprocessor = None
        
    # Keep this method for backward compatibility with existing saved models
    def _combine_text(self, df):
        """
        Combine header and section text into a single text field for analysis.
        For backward compatibility with existing models.
        
        Args:
            df (pandas.DataFrame): DataFrame containing 'header' and 'section' columns
            
        Returns:
            numpy.ndarray: Array of combined text strings
        """
        return combine_text(df)

    def load_model(self, model_file=None):
        """
        Load a pretrained classifier model from a file.
        
        Args:
            model_file (str, optional): Path to the joblib file containing the saved model.
                                       If None, fails with an appropriate error.
            
        Returns:
            The loaded model
        """
        if model_file is None:
            raise ValueError("Model file path is required. Please specify a valid model path.")
                
        self.logger.info(f"Loading model from: {model_file}")
        try:
            self.loaded_model = joblib.load(model_file)
            self.clf_pipeline = self.loaded_model
            return self.loaded_model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    # -----------------------------------------------------------------------------
    # Build the Training Pipeline
    # -----------------------------------------------------------------------------
    def build_pipeline(self):
        """
        Build the machine learning pipeline for processing and classifying text sections.
        
        The pipeline combines header and section text, vectorizes it using TF-IDF,
        and includes binary features (has_table, has_list).
        """
        self.combined_text_pipeline = Pipeline([
            ('combine', FunctionTransformer(combine_text, validate=False)),
            ('tfidf', TfidfVectorizer(max_features=2000))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('combined_text', self.combined_text_pipeline, ['header', 'section']),
                ('binary', 'passthrough', ['has_table', 'has_list'])
            ],
            remainder='drop'
        )

        self.clf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', LinearSVC(class_weight='balanced', max_iter=10000))
        ])

    # -----------------------------------------------------------------------------
    # Training Function with CSV input
    # -----------------------------------------------------------------------------
    def train_from_csv(self, csv_file, output_model_file=None):
        """
        Train the classification model using a CSV dataset and optionally save it.
        
        Args:
            csv_file (str): Path to the CSV file containing training data
            output_model_file (str, optional): Path where to save the trained model
                If None, the model is trained but not saved
                
        Returns:
            sklearn.pipeline.Pipeline: The trained classifier pipeline
        """
        self.logger.info(f"Loading training data from {csv_file}...")
        df_train = pd.read_csv(csv_file)
        self.logger.info(f"Loaded training data with {len(df_train)} rows.")
        
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
        self.logger.info(f"Training set: {len(X_train)} rows; Validation set: {len(X_val)} rows.")
        
        # Build the pipeline before training
        self.build_pipeline()
        
        # Train the pipeline (this will combine text and vectorize)
        self.logger.info("Starting model training...")
        start_time = time.time()
        self.clf_pipeline.fit(X_train, y_train)
        end_time = time.time()
        self.logger.info(f"Model training completed in {end_time - start_time:.2f} seconds.")
        
        # Evaluate on validation set
        self.logger.info("Evaluating on validation set...")
        y_val_pred = self.clf_pipeline.predict(X_val)
        cm = confusion_matrix(y_val, y_val_pred)
        report = classification_report(y_val, y_val_pred)
        self.logger.info("Confusion Matrix:\n" + str(cm))
        self.logger.info("Classification Report:\n" + report)
        
        # Save the model if requested
        if output_model_file:
            self.logger.info(f"Saving model to {output_model_file}...")
            os.makedirs(os.path.dirname(os.path.abspath(output_model_file)), exist_ok=True)
            joblib.dump(self.clf_pipeline, output_model_file)
            self.logger.info(f"Model saved successfully to {output_model_file}")
        
        return self.clf_pipeline

    # -----------------------------------------------------------------------------
    # Adjust Predictions with Combined Logic
    # -----------------------------------------------------------------------------
    def _adjust_predictions_with_index_detection(self, df, length_threshold=300, propo_threshold=300):
        """
        Adjust predictions by identifying potential index sections based on cumulative metrics.
        
        For each article (grouped by filename), sections are processed in order and cumulative
        metrics are tracked. Index detection is applied to sections before thresholds are exceeded.
        
        Args:
            df (pandas.DataFrame): DataFrame with predicted sections
            length_threshold (int): Threshold for cumulative section length
            propo_threshold (int): Threshold for cumulative section proportion
            
        Returns:
            pandas.DataFrame: DataFrame with adjusted predictions
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
                    flag, nums = self._compute_likely_index_for_section(row['section'])
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
    def _process_predictions_with_dask(self, input_parquet, output_parquet, model_path=None, pretrained_model=True, n_cpus=4):
        """
        Process predictions on the input Parquet file using Dask for parallel processing.
        
        Args:
            input_parquet (str): Path to the input Parquet file with sections to classify
            output_parquet (str): Path where the classified Parquet file will be saved
            model_path (str, optional): Path to the model file to load
            pretrained_model (bool): If True, use the loaded model; otherwise, use the trained model
            n_cpus (int, optional): Number of CPU cores to use for parallel processing. Default is 4.
        """
        if pretrained_model:
            # Ensure the model is loaded
            if model_path is not None:
                self.logger.info(f"Loading model from specified path: {model_path}")
                self.loaded_model = self.load_model(model_path)
            elif self.loaded_model is None:
                self.logger.info("Loading default model")
                self.load_model()  # Load default model
            self.clf_pipeline = self.loaded_model
            
        # Read all columns from the input Parquet file to preserve them
        self.logger.info(f"Reading data from Parquet file: {input_parquet}")
        ddf = dd.read_parquet(input_parquet)
        
        # Get list of all columns to preserve them
        all_columns = ddf.columns.tolist()
        
        # Verify required columns exist
        required_columns = ['id', 'header', 'section', 'has_table', 'has_list']
        missing_columns = [col for col in required_columns if col not in all_columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Input parquet file missing required columns: {missing_columns}")
        
        # Preprocess key columns
        ddf['header'] = ddf['header'].fillna('')
        ddf['section'] = ddf['section'].fillna('')
        ddf['has_table'] = ddf['has_table'].astype(int)
        ddf['has_list'] = ddf['has_list'].astype(int)
        
        # Identify already processed rows via output file checkpoint (read only 'id')
        processed_ids = set()
        if os.path.exists(output_parquet):
            try:
                self.logger.info(f"Found existing output file {output_parquet}. Loading processed IDs...")
                ddf_existing = dd.read_parquet(output_parquet, columns=['id'])
                processed_ids = set(ddf_existing['id'].unique().compute().tolist())
                self.logger.info(f"{len(processed_ids)} rows have already been processed.")
            except Exception as e:
                self.logger.error("Error reading existing output file. Processing all rows.", exc_info=e)
        
        if processed_ids:
            # Filter out already predicted rows using efficient isin on the small set of IDs.
            ddf = ddf[~ddf['id'].isin(list(processed_ids))]
            remaining = int(ddf.shape[0].compute())
            self.logger.info(f"Processing only {remaining} remaining rows.")
            
            # Skip processing if no remaining rows
            if remaining == 0:
                self.logger.info("No new rows to process. Using existing output file.")
                return
        else:
            self.logger.info("No prior progress found; processing all rows.")
        
        # (Optional) Repartition to a moderate number of partitions
        # Use n_cpus to determine the number of partitions (5 partitions per CPU)
        npartitions = max(5 * n_cpus, 1)  # Ensure at least 1 partition
        self.logger.info(f"Repartitioning data into {npartitions} partitions based on {n_cpus} CPU cores")
        ddf = ddf.repartition(npartitions=npartitions)
        ddf = ddf.persist()  # Cache in memory to avoid repeated computations
        
        # Create metadata that includes the new column "predicted_section"
        meta = ddf._meta.copy()
        meta['predicted_section'] = ''
        
        # Define a closure that captures self
        def predict_partition(df):
            # Run SVM prediction on the subset of features.
            X = df[['header', 'section', 'has_table', 'has_list']]
            df = df.copy()  # Avoid SettingWithCopyWarning
            # Initially assign SVM predictions.
            df['predicted_section'] = self.clf_pipeline.predict(X)
            return df
        
        self.logger.info("Running predictions in parallel using Dask...")
        start = time.time()
        ddf_pred = ddf.map_partitions(predict_partition, meta=meta)
        
        temp_output = output_parquet + ".tmp"
        self.logger.info("Writing new predictions to temporary Parquet file with progress bar...")
        with ProgressBar():
            ddf_pred.to_parquet(temp_output, write_index=False)
        end = time.time()
        self.logger.info(f"New predictions written in {end - start:.2f} seconds.")
        
        # Merge new predictions with existing ones, if any.
        if processed_ids:
            self.logger.info("Merging new predictions with existing progress...")
            df_existing = dd.read_parquet(output_parquet).compute()
            df_new = dd.read_parquet(temp_output).compute()
            
            # Ensure both dataframes have the same columns for clean merging
            missing_cols_new = [c for c in df_existing.columns if c not in df_new.columns]
            for col in missing_cols_new:
                df_new[col] = None if df_existing[col].dtype == 'object' else np.nan
                
            missing_cols_existing = [c for c in df_new.columns if c not in df_existing.columns]
            for col in missing_cols_existing:
                df_existing[col] = None if df_new[col].dtype == 'object' else np.nan
            
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = dd.read_parquet(temp_output).compute()
        
        # Post-process predictions: for each article, before the cumulative thresholds are exceeded,
        # run compute_likely_index_for_section on the section text (without the header) and override
        # the predicted label with "π" if detected.
        self.logger.info("Adjusting predictions based on index detection and cumulative thresholds...")
        df_final = self._adjust_predictions_with_index_detection(df_final, length_threshold=300, propo_threshold=300)
        
        self.logger.info("Writing merged and adjusted predictions to final output Parquet file...")
        df_final.to_parquet(output_parquet, index=False)
        self.logger.info("Predictions saved and progress maintained.")

    # -----------------------------------------------------------------------------
    # Compute Likely Index for Section
    # -----------------------------------------------------------------------------
    def _compute_likely_index_for_section(self, text, min_seq_length=7, allowed_consecutive_equals=4):
        """
        Analyze section text to determine if it's likely to be an index/table of contents.
        
        This function examines patterns in the text line by line, looking for 
        numerical sequences that would indicate an index structure.
        
        Args:
            text (str): The section text to analyze
            min_seq_length (int): Minimum sequence length to consider as an index
            allowed_consecutive_equals (int): Maximum allowed consecutive equal numbers
            
        Returns:
            tuple: (flag, numbers) where flag is 1 if the section is likely an index, 0 otherwise,
                  and numbers is the list of numbers extracted from the text
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

    def fully_annotate(self, input_parquet: str, output_parquet: str, document_types: Dict[str, str] = None, annotation_type: str = "auto") -> None:
        """
        Fully annotate sections in a parquet file based on document type.
        
        This is a dispatcher method that delegates to the appropriate specialized annotation 
        method based on document_type or explicit annotation_type.
        
        Args:
            input_parquet: Path to input parquet file with predicted sections
            output_parquet: Path to save fully annotated parquet file
            document_types: Dict mapping filename to document_type (optional)
            annotation_type: Annotation method to use ('text', 'chapter', or 'auto')
                            - 'text': Use fully_annotate_text for all documents
                            - 'chapter': Use fully_annotate_chapter for all documents
                            - 'auto': Determine method based on document_type
        """
        self.logger.info(f"Reading parquet file from {input_parquet}...")
        # Read all columns to ensure we preserve everything
        df = pd.read_parquet(input_parquet)
        
        # If document_types are provided and document_type column doesn't exist yet, add it
        if document_types and 'document_type' not in df.columns:
            df['document_type'] = df['filename'].map(document_types)
        
        # Determine annotation type for each document
        if annotation_type == "text":
            # Use text annotation for all documents
            df_updated = self.fully_annotate_text(df)
        elif annotation_type == "chapter":
            # Use chapter annotation for all documents
            df_updated = self.fully_annotate_chapter(df)
        else:  # annotation_type == "auto"
            # Group by filename and process each document according to its type
            updated_groups = []
            
            for filename, group in df.groupby('filename'):
                # Sort sections by id (which reflects absolute order)
                group = group.sort_values('id').copy()
                
                # Determine document type if available
                doc_type = None
                if 'document_type' in group.columns and not group['document_type'].isna().all():
                    # Use document_type from the DataFrame if available
                    doc_type = group['document_type'].iloc[0]
                elif document_types and filename in document_types:
                    # Fall back to the provided document_types mapping
                    doc_type = document_types[filename]
                    if 'document_type' not in group.columns:
                        group['document_type'] = doc_type
                
                # Select annotation method based on document type
                if doc_type == 'Κεφάλαιο':
                    self.logger.debug(f"Processing chapter document: {filename}")
                    # Process as chapter
                    updated_group = self.fully_annotate_chapter_group(group)
                else:
                    # Process as text
                    updated_group = self.fully_annotate_text_group(group)
                
                updated_groups.append(updated_group)
            
            # Concatenate all groups
            df_updated = pd.concat(updated_groups) if updated_groups else pd.DataFrame()
        
        # Save to output parquet file
        self.logger.info(f"Saving fully annotated parquet to {output_parquet}...")
        df_updated.to_parquet(output_parquet, index=False)
        
        self.logger.info("Processing complete.")
        
    def fully_annotate_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fully annotate sections in a DataFrame based on π and β boundaries.
        
        Args:
            df: DataFrame with predicted sections
            
        Returns:
            DataFrame with fully annotated sections
        """
        files_missing_boundaries = 0  # Count files without both a π and a β boundary
        updated_groups = []  # To collect processed groups
        
        self.logger.info("Processing each file group for text annotation...")
        for filename, group in df.groupby('filename'):
            updated_group = self.fully_annotate_text_group(group)
            if updated_group is None:
                files_missing_boundaries += 1
            else:
                updated_groups.append(updated_group)
        
        # Concatenate updated groups back into a single DataFrame
        df_updated = pd.concat(updated_groups) if updated_groups else pd.DataFrame()
        
        self.logger.info(f"Files missing one or both boundaries (π and β): {files_missing_boundaries}")
        return df_updated
    
    def fully_annotate_text_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single document group for text annotation.
        
        Args:
            group: DataFrame group for a single document
            
        Returns:
            Processed DataFrame group or None if missing boundaries
        """
        # Sort sections by id (which reflects absolute order)
        group = group.sort_values('id').copy()
        
        # Check if both boundary labels are present
        has_pi = (group['predicted_section'] == 'π').any()
        has_beta = (group['predicted_section'] == 'β').any()
        
        if has_pi and has_beta:
            # The first occurrence of 'π' defines the upper boundary
            first_pi_id = group.loc[group['predicted_section'] == 'π', 'id'].iloc[0]
            # The last occurrence of 'β' defines the lower boundary
            last_beta_id = group.loc[group['predicted_section'] == 'β', 'id'].iloc[-1]
            
            # Check if markers are in the correct order
            if first_pi_id > last_beta_id:
                return None  # Skip this file's annotation
            
            # Create a boolean mask for rows with label "άλλο" (i.e. not yet fully annotated)
            mask = group['predicted_section'] == 'άλλο'
            
            # Define conditions (vectorized) on the 'id' of each row
            cond_intro = group['id'] < first_pi_id    # Before the first 'π'
            cond_appendix = group['id'] > last_beta_id  # After the last 'β'
            
            # For rows with label "άλλο", assign:
            #   'ε.σ.' if the row is before the first π,
            #   'α' if the row is after the last β,
            #   and 'κ' otherwise.
            new_labels = np.select(
                [cond_intro, cond_appendix],
                ['ε.σ.', 'α'],
                default='κ'
            )
            # Update only the rows with label "άλλο"
            group.loc[mask, 'predicted_section'] = new_labels[mask]
            return group
        else:
            return None  # Signal missing boundaries
    
    def fully_annotate_chapter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fully annotate sections in a DataFrame as chapter content.
        
        Args:
            df: DataFrame with predicted sections
            
        Returns:
            DataFrame with fully annotated sections as chapters
        """
        updated_groups = []  # To collect processed groups
        
        self.logger.info("Processing each file group for chapter annotation...")
        for filename, group in df.groupby('filename'):
            updated_groups.append(self.fully_annotate_chapter_group(group))
        
        # Concatenate updated groups back into a single DataFrame
        df_updated = pd.concat(updated_groups) if updated_groups else pd.DataFrame()
        
        return df_updated
    
    def fully_annotate_chapter_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Process a single document group for chapter annotation.
        
        Args:
            group: DataFrame group for a single document
            
        Returns:
            Processed DataFrame group
        """
        # Sort sections by id (which reflects absolute order)
        group = group.sort_values('id').copy()
        
        # For chapters: convert all 'π' and 'άλλο' to 'κ', leave 'β' as is
        mask = group['predicted_section'].isin(['π', 'άλλο'])
        group.loc[mask, 'predicted_section'] = 'κ'
        
        return group

    def classify_sections(self, input_parquet, output_parquet, model_path, n_cpus=4, column_name='title'):
        """
        Classify the sections in a parquet file and save the predictions to an output parquet file.
        
        Args:
            input_parquet (str): Path to input parquet file with sections data
            output_parquet (str): Path to save the output parquet file with predictions
            model_path (str): Path to the model file to use for predictions
            n_cpus (int, optional): Number of CPU cores to use for parallel processing. Default is 4.
            column_name (str, optional): Name of the column containing text to classify. Default is 'title'.
        """
        overall_start = time.time()
        
        # Use a pre-trained model for prediction
        self.logger.info(f"Using model from {model_path} for prediction")
        self.load_model(model_path)
        
        # Process using Dask
        self._process_predictions_with_dask(input_parquet, output_parquet, model_path=model_path, pretrained_model=True, n_cpus=n_cpus)
        
        overall_end = time.time()
        self.logger.info(f"Total elapsed time: {overall_end - overall_start:.2f} seconds.")