"""Frozen reference: the dask prediction path of GlossSectionClassifier at c91590b (K2 oracle).

Copied verbatim (dedented, `self` passed explicitly) from src/glossapi/gloss_section_classifier.py at
c91590b, lines 265-407. Differential tests run it against the own partition path when dask is
installed; it is never imported by glossapi itself.
"""
import os
import time
import traceback

import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import glossapi.gloss_section_classifier as _classifier
from glossapi.ledger.encoding import file_hash, typed_value


def observe(*args, **kwargs):
    return _classifier.observe(*args, **kwargs)


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
    prediction_failures = []
    def predict_partition(df):
        # If the dataframe is empty, return it with the predicted_section column added
        if len(df) == 0:
            df = df.copy()  # Avoid SettingWithCopyWarning
            df['predicted_section'] = pd.Series(dtype='object')
            return df
            
        # Run SVM prediction on the subset of features for non-empty dataframes
        X = df[['header', 'section', 'has_table', 'has_list']]
        df = df.copy()  # Avoid SettingWithCopyWarning
        
        try:
            # Assign SVM predictions
            df['predicted_section'] = self.clf_pipeline.predict(X)
        except Exception as e:
            prediction_failures.append({"error_type": type(e).__module__ + "." + type(e).__name__, "message": str(e), "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__)), "row_ids": df["id"].tolist()})
            # If prediction fails, assign a default value
            print(f"Error during prediction: {str(e)}")
            print(f"Assigning default prediction value to all rows")
            df['predicted_section'] = 'άλλο'  # Default to general section for ML classifier
            
        return df
    
    self.logger.info("Running predictions in parallel using Dask...")
    start = time.time()
    ddf_pred = ddf.map_partitions(predict_partition, meta=meta)
    
    temp_output = output_parquet + ".tmp"
    self.logger.info("Writing new predictions to temporary Parquet file with progress bar...")
    with ProgressBar():
        ddf_pred.to_parquet(temp_output, write_index=False)
    end = time.time()
    for failure in prediction_failures:
        observe("annotation_prediction_fallback", reported_error={key:failure[key] for key in ("error_type", "message", "traceback")}, row_ids=failure["row_ids"], fallback_label="άλλο", degraded=True)
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
    observe("legacy_observed", annotation_model_labels=typed_value(df_final[[column for column in ("id", "row_id", "filename", "predicted_section") if column in df_final.columns]].to_dict("records")), model_path=str(model_path), model_sha256=file_hash(model_path) if model_path and os.path.isfile(model_path) else None)
    df_final = self._adjust_predictions_with_index_detection(df_final, length_threshold=300, propo_threshold=300)
    
    self.logger.info("Writing merged and adjusted predictions to final output Parquet file...")
    df_final.to_parquet(output_parquet, index=False)
    observe("annotate_completed", path=output_parquet, status="ok")
