use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::HashSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex}; // Mutex will be needed for collecting report_entries if done from parallel loop
use walkdir;
use csv;
use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

use chrono;

use crate::cleaning_module;
use crate::directory_processor; // Add this import

// Struct for the final 4-column CSV report
#[derive(Debug, serde::Serialize)] // Added serde::Serialize
struct FinalReportEntry {
    file_name: String,
    badness_score_all_chars: Option<f64>,
    percentage_greek_cleaned: Option<f64>,
    percentage_latin_cleaned: Option<f64>,
}

/// Orchestrates the complete document processing pipeline.
/// 1. Initial cleaning of markdown text.
/// 2. Table detection and preparation for removal.
/// 3. Table removal.
/// 4. Writing processed files to the output directory.
/// 5. Final analysis (badness, script percentages) on processed files, generating a CSV report.
#[pyfunction]
pub fn run_complete_pipeline(
    py: Python,
    input_dir_str: &str,
    output_cleaned_files_dir_str: &str,
    output_report_csv_str: &str,
    scripts_to_keep_input: Vec<String>,
    num_threads: usize,
) -> PyResult<()> {
    println!("Rust: Starting REFACTORED STAGED complete pipeline (calling batch functions)...");
    println!("Rust: Input directory: {}", input_dir_str);
    println!("Rust: Output cleaned files directory (final): {}", output_cleaned_files_dir_str);
    println!("Rust: Output report CSV: {}", output_report_csv_str);
    println!("Rust: Scripts to keep: {:?}", scripts_to_keep_input);
    println!("Rust: Number of threads: {}", num_threads);

    let main_input_path = Path::new(input_dir_str);
    if !main_input_path.is_dir() {
        let err_msg = format!("Input path is not a directory: {}", input_dir_str);
        println!("Rust: ERROR - {}", err_msg);
        return Err(PyValueError::new_err(err_msg));
    }
    println!("Rust: Input path validated as directory.");

    // Ensure final output directory for cleaned files exists (batch_remove_tables_from_files will need it)
    let final_output_cleaned_files_path = Path::new(output_cleaned_files_dir_str);
    if !final_output_cleaned_files_path.exists() {
        println!("Rust: Creating final output directory for cleaned files: {}...", final_output_cleaned_files_path.display());
        fs::create_dir_all(final_output_cleaned_files_path).map_err(|e| PyValueError::new_err(format!("Failed to create final output directory {}: {}", final_output_cleaned_files_path.display(), e)))?;
        println!("Rust: Final output directory for cleaned files created.");
    }

    // --- Setup Main Temporary Directory for this Pipeline Run ---
    // Let's use a subdirectory within the system's temp directory for better isolation and cleanup.
    let temp_pipeline_root_name = format!("text_cleaner_rs_pipeline_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default());
    let mut temp_pipeline_root_dir = std::env::temp_dir();
    temp_pipeline_root_dir.push(temp_pipeline_root_name);

    println!("Rust: Creating main temporary directory for pipeline: {}...", temp_pipeline_root_dir.display());
    fs::create_dir_all(&temp_pipeline_root_dir).map_err(|e| PyValueError::new_err(format!("Failed to create main temporary pipeline directory {}: {}", temp_pipeline_root_dir.display(), e)))?;
    println!("Rust: Main temporary pipeline directory created.");
    
    // --- Define paths for intermediate stages ---
    let temp_stage1_cleaned_dir = temp_pipeline_root_dir.join("stage1_cleaned_files");
    let temp_stage2_detailed_report_csv = temp_pipeline_root_dir.join("stage2_detailed_table_report.csv");

    // Convert paths to &str for the directory_processor functions
    let temp_stage1_cleaned_dir_str = temp_stage1_cleaned_dir.to_string_lossy().into_owned();
    let temp_stage2_detailed_report_csv_str = temp_stage2_detailed_report_csv.to_string_lossy().into_owned();
    
    // --- Setup for character sets (used by Stage 4, generate_analysis_report_for_directory sets its own) ---
    println!("Rust: Setting up allowed characters for final analysis stage...");
    let mut allowed_chars_final_analysis = HashSet::new();
    let base_scripts = vec!["punctuation", "numbers", "common_symbols"];
    for script_key_str in scripts_to_keep_input.iter().map(|s| s.as_str()).chain(base_scripts.iter().copied()) {
         if let Some(script_set) = cleaning_module::SCRIPT_SETS.get(script_key_str) {
            allowed_chars_final_analysis.extend(script_set);
        }
    }
    allowed_chars_final_analysis.insert(' ');
    allowed_chars_final_analysis.insert('\t');
    allowed_chars_final_analysis.insert('\n');
    let allowed_chars_final_analysis_arc = Arc::new(allowed_chars_final_analysis);
    let unusual_chars_final_analysis_arc = Arc::new(cleaning_module::SCRIPT_SETS.get("unusual").cloned().unwrap_or_default());
    let scripts_to_keep_input_arc_final = Arc::new(scripts_to_keep_input.clone());
    println!("Rust: Allowed characters for final analysis setup complete.");


    // Wrap the core logic in a result block to ensure cleanup happens
    let pipeline_result = (|| {
        // === STAGE 1: Initial Cleaning to Temporary Directory ===
        println!("\nRust: === Starting STAGE 1: Initial Cleaning to {} ===", temp_stage1_cleaned_dir.display());
        fs::create_dir_all(&temp_stage1_cleaned_dir).map_err(|e| PyValueError::new_err(format!("Failed to create stage 1 temp output directory {}: {}", temp_stage1_cleaned_dir.display(), e)))?;
        
        directory_processor::generate_analysis_report_for_directory(
            py, 
            input_dir_str,
            None, // output_csv_path_str: We don't want a CSV from this stage
            Some(&temp_stage1_cleaned_dir_str), // output_dir_cleaned_files_str: Output to our temp dir
            scripts_to_keep_input.clone(), // scripts_to_keep
            num_threads,
        ).map_err(|e| {
            let err_msg = format!("Stage 1 (generate_analysis_report_for_directory) failed: {}", e);
            println!("Rust: ERROR - {}", err_msg);
            PyValueError::new_err(err_msg) 
        })?;
        println!("Rust: === STAGE 1: Initial Cleaning COMPLETED ===\n");

        // === STAGE 2: Detailed Table Report Generation ===
        println!("Rust: === Starting STAGE 2: Detailed Table Report Generation to {} ===", temp_stage2_detailed_report_csv.display());
        directory_processor::batch_generate_detailed_table_report_csv(
            py, 
            &temp_stage1_cleaned_dir_str, // input_dir_str: From Stage 1 output
            &temp_stage2_detailed_report_csv_str, // output_csv_path_str: Our temp CSV
            num_threads,
        ).map_err(|e| {
            let err_msg = format!("Stage 2 (batch_generate_detailed_table_report_csv) failed: {}", e);
            println!("Rust: ERROR - {}", err_msg);
            PyValueError::new_err(err_msg)
        })?;
        println!("Rust: === STAGE 2: Detailed Table Report Generation COMPLETED ===\n");

        // === STAGE 3: Table Removal ===
        println!("Rust: === Starting STAGE 3: Table Removal (from {} to {}) ===", temp_stage1_cleaned_dir.display(), final_output_cleaned_files_path.display());
        directory_processor::batch_remove_tables_from_files(
            py, 
            &temp_stage1_cleaned_dir_str, // input_markdown_dir_str: From Stage 1 output
            &temp_stage2_detailed_report_csv_str, // detailed_report_csv_path_str: From Stage 2 output
            output_cleaned_files_dir_str, // output_processed_md_dir_str: Final user-specified dir
            num_threads,
        ).map_err(|e| {
            let err_msg = format!("Stage 3 (batch_remove_tables_from_files) failed: {}", e);
            println!("Rust: ERROR - {}", err_msg);
            PyValueError::new_err(err_msg)
        })?;
        println!("Rust: === STAGE 3: Table Removal COMPLETED ===\n");

        // === STAGE 4: Final Analysis & Report Generation (from Final Output) ===
        println!("Rust: === Starting STAGE 4: Final Analysis & Report Generation ===");
        let report_entries_mutex: Arc<Mutex<Vec<FinalReportEntry>>> = Arc::new(Mutex::new(Vec::new()));
        
        let final_processed_files_for_report: Vec<PathBuf> = walkdir::WalkDir::new(final_output_cleaned_files_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file() && e.path().extension().map_or(false, |ext| ext == "md"))
            .map(|e| e.path().to_path_buf())
            .collect();

        println!("Rust: Found {} files in final_output_dir for Stage 4 analysis.", final_processed_files_for_report.len());

        if final_processed_files_for_report.is_empty() {
            println!("Rust: No files found in the final output directory for Stage 4 analysis. Report will be empty or headers only.");
        } else {
            // Configure Rayon thread pool (can re-use or create new for this specific stage if needed)
            // For simplicity, let's assume the previous pool configuration is fine, or recreate one.
            // Using a new pool specifically for this stage to ensure no interference
             let pool_s4 = ThreadPoolBuilder::new()
                .num_threads(if num_threads > 0 { num_threads } else { rayon::current_num_threads() })
                .build()
                .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool for Stage 4: {}", e)))?;
            
            pool_s4.install(|| {
                final_processed_files_for_report
                    .par_iter()
                    .for_each(|final_file_path| {
                        let thread_id = rayon::current_thread_index().unwrap_or(997);
                        let report_relative_path = final_file_path.strip_prefix(final_output_cleaned_files_path).unwrap_or(final_file_path);
                        // println!("Rust: [S4-Thread {}] Analyzing for report: {}", thread_id, report_relative_path.display()); // Kept for context, can be too verbose

                        match fs::read_to_string(final_file_path) {
                            Ok(final_content) => {
                                // println!("Rust: [S4-Thread {}] Read final_content for report: {}", thread_id, report_relative_path.display()); // Verbose
                                let analysis_result = cleaning_module::perform_text_analysis(
                                    &final_content,
                                    &*allowed_chars_final_analysis_arc, 
                                    &*unusual_chars_final_analysis_arc,
                                    &*scripts_to_keep_input_arc_final, 
                                    true, 
                                    None, 
                                );
                                // println!("Rust: [S4-Thread {}] Final analysis done for report: {}", thread_id, report_relative_path.display()); // Verbose

                                // Debug print for raw analysis values
                                println!(
                                    "Rust: [S4-Thread {}] Raw Analysis for {}: BadnessAllChars: {:?}, GreekCount: {:?}, LatinCount: {:?}, CleanedNonWS: {:?}",
                                    thread_id,
                                    report_relative_path.display(),
                                    analysis_result.badness_score_all_chars,
                                    analysis_result.greek_char_count_after_clean,
                                    analysis_result.latin_char_count_after_clean,
                                    analysis_result.cleaned_non_whitespace_chars_after_clean
                                );

                                // Badness score processing
                                let rounded_badness_score = analysis_result.badness_score_all_chars.map(|b_val| {
                                    let clamped_b_val = b_val.max(0.0); // Clamp if negative
                                    (clamped_b_val * 1000.0).round() / 1000.0
                                });

                                // Greek percentage processing
                                let rounded_percentage_greek = analysis_result.greek_char_count_after_clean.and_then(|count| {
                                    analysis_result.cleaned_non_whitespace_chars_after_clean.map(|total_script_chars| {
                                        if total_script_chars > 0 && count > 0 {
                                            let val = (count as f64 / total_script_chars as f64) * 100.0;
                                            (val * 1000.0).round() / 1000.0
                                        } else {
                                            0.0
                                        }
                                    })
                                });

                                // Latin percentage processing
                                let rounded_percentage_latin = analysis_result.latin_char_count_after_clean.and_then(|count| {
                                    analysis_result.cleaned_non_whitespace_chars_after_clean.map(|total_script_chars| {
                                        if total_script_chars > 0 && count > 0 {
                                            let val = (count as f64 / total_script_chars as f64) * 100.0;
                                            (val * 1000.0).round() / 1000.0
                                        } else {
                                            0.0
                                        }
                                    })
                                });
                                
                                let report_entry = FinalReportEntry {
                                    file_name: report_relative_path.to_string_lossy().into_owned(),
                                    badness_score_all_chars: rounded_badness_score,
                                    percentage_greek_cleaned: rounded_percentage_greek,
                                    percentage_latin_cleaned: rounded_percentage_latin,
                                };
                                
                                report_entries_mutex.lock().unwrap().push(report_entry);
                                // println!("Rust: [S4-Thread {}] Report entry pushed for: {}", thread_id, report_relative_path.display()); // Verbose
                            }
                            Err(e) => {
                                eprintln!("Rust: [S4-Thread {}] Error reading final file {} for analysis: {}. Skipping for report.", thread_id, final_file_path.display(), e);
                            }
                        }
                    });
            });
        }
        println!("Rust: === STAGE 4: Final Analysis & Report Generation COMPLETED ===\n");

        // --- Final CSV Writing (after all stages) ---
        let final_report_entries = Arc::try_unwrap(report_entries_mutex)
            .map_err(|_e| PyValueError::new_err("Mutex for report data was poisoned (arc unwrap failed) for Stage 4"))?
            .into_inner()
            .map_err(|_e| PyValueError::new_err("Mutex for report data was poisoned (into_inner failed) for Stage 4"))?;
        
        println!("Rust: Total files processed for Stage 4 report: {}", final_report_entries.len());

        if !output_report_csv_str.is_empty() {
            println!("Rust: Preparing to write final Parquet report to: {}", output_report_csv_str);
            if let Some(parent_dir) = Path::new(output_report_csv_str).parent() {
                if !parent_dir.as_os_str().is_empty() && !parent_dir.exists() {
                   println!("Rust: Creating parent directory for Parquet: {}", parent_dir.display());
                   fs::create_dir_all(parent_dir).map_err(|e| PyValueError::new_err(format!("Failed to create parent for Parquet {}: {}", parent_dir.display(), e)))?;
                   println!("Rust: Parent directory for Parquet created.");
                }
            }

            // Build Arrow arrays from report entries
            let file_names: Vec<&str> = final_report_entries.iter().map(|e| e.file_name.as_str()).collect();
            let badness_vals: Vec<f64> = final_report_entries
                .iter()
                .map(|e| e.badness_score_all_chars.unwrap_or(0.0))
                .collect();
            let greek_vals: Vec<f64> = final_report_entries
                .iter()
                .map(|e| e.percentage_greek_cleaned.unwrap_or(0.0))
                .collect();
            let latin_vals: Vec<f64> = final_report_entries
                .iter()
                .map(|e| e.percentage_latin_cleaned.unwrap_or(0.0))
                .collect();

            let file_name_array: ArrayRef = Arc::new(StringArray::from(file_names));
            let badness_array: ArrayRef = Arc::new(Float64Array::from(badness_vals));
            let greek_array: ArrayRef = Arc::new(Float64Array::from(greek_vals));
            let latin_array: ArrayRef = Arc::new(Float64Array::from(latin_vals));

            let schema = Schema::new(vec![
                Field::new("file_name", DataType::Utf8, false),
                Field::new("badness_score_all_chars", DataType::Float64, true),
                Field::new("percentage_greek_cleaned", DataType::Float64, true),
                Field::new("percentage_latin_cleaned", DataType::Float64, true),
            ]);

            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![file_name_array, badness_array, greek_array, latin_array],
            ).map_err(|e| PyValueError::new_err(format!("Failed to build RecordBatch: {}", e)))?;

            let file = File::create(output_report_csv_str)
                .map_err(|e| PyValueError::new_err(format!("Failed to create Parquet file: {}", e)))?;
            let props = WriterProperties::builder().build();
            let mut writer = ArrowWriter::try_new(file, Arc::new(schema), Some(props))
                .map_err(|e| PyValueError::new_err(format!("Failed to create Parquet writer: {}", e)))?;
            writer.write(&batch).map_err(|e| PyValueError::new_err(format!("Failed to write batch to Parquet: {}", e)))?;
            writer.close().map_err(|e| PyValueError::new_err(format!("Failed to close Parquet writer: {}", e)))?;

            println!("Rust: Parquet report written successfully at {}.", output_report_csv_str);
        } else {
            println!("Rust: Parquet output path is empty, skipping report generation for Stage 4.");
        }
        Ok(()) // Success for the inner lambda
    })(); // End of the main try block for pipeline stages


    // --- Cleanup Main Temporary Directory ---
    println!("Rust: Attempting to clean up main temporary pipeline directory: {}...", temp_pipeline_root_dir.display());
    if temp_pipeline_root_dir.exists() {
        match fs::remove_dir_all(&temp_pipeline_root_dir) {
            Ok(_) => {
                println!("Rust: Main temporary pipeline directory successfully marked for removal by fs::remove_dir_all.");
                // Verify removal immediately
                if temp_pipeline_root_dir.exists() {
                    eprintln!("Rust: WARNING - Temporary directory {} STILL EXISTS after attempted removal call.", temp_pipeline_root_dir.display());
                } else {
                    println!("Rust: Temporary directory {} successfully confirmed removed after check.", temp_pipeline_root_dir.display());
                }
            }
            Err(e) => {
                eprintln!("Rust: WARNING - Failed to remove main temporary pipeline directory {}: {}. Manual cleanup might be needed.", temp_pipeline_root_dir.display(), e);
            }
        }
    } else {
        println!("Rust: Main temporary pipeline directory not found for cleanup (might have failed creation or been removed already).");
    }
    
    // Return the result of the pipeline execution
    match pipeline_result {
        Ok(_) => {
            println!("Rust: REFACTORED STAGED complete pipeline finished successfully.");
            Ok(())
        }
        Err(e) => {
            println!("Rust: REFACTORED STAGED complete pipeline FAILED: {}", e);
            Err(e) // Propagate the error
        }
    }
} 