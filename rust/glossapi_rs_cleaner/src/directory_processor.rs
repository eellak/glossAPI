use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::collections::{HashSet, HashMap};
use std::fs::{self};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;
use csv::Writer;
use serde::Serialize;

use crate::table_analysis_module;
use crate::cleaning_module;
use crate::table_remover_module;

// Helper to convert Path to String, lossy
fn path_to_str(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

// Struct to hold summary data for each file before writing to CSV
#[derive(Debug, Serialize)]
struct FileTableSummary {
    file_path: String,
    total_tables: usize,
    malformed_tables: usize,
}

// Struct for detailed table issue reporting
#[derive(Debug, Serialize, Clone)]
struct DetailedTableIssueReportEntry {
    file_path: String,
    issue_description: String,
    table_start_line: usize,
    table_end_line: usize,
}

// Define operation output variants
#[derive(Clone)]
pub enum PerFileOperationOutput {
    Content(String),
    TableScanResult(table_analysis_module::TableScan),
    DetailedTableIssues(Vec<DetailedTableIssueReportEntry>),
    OperationSuccess(PathBuf),
    OperationError(PathBuf, String),
    Empty,
}

/// Core directory processing function with concurrency support
fn process_directory_core<OpConfig, OpFn, PerFileOutput>(
    _py: Python,
    input_dir_str: &str,
    output_dir_str: Option<&str>,
    num_threads: usize,
    operation_config: Arc<OpConfig>,
    file_operation: OpFn,
) -> PyResult<Vec<PerFileOutput>>
where
    OpConfig: Send + Sync + 'static,
    OpFn: Fn(Python, PathBuf, &str, &Arc<OpConfig>) -> PyResult<Option<PerFileOutput>> + Send + Sync + 'static,
    PerFileOutput: Send + Sync + Clone + 'static,
{
    println!("DEBUG: Entering process_directory_core");
    let input_path = Path::new(input_dir_str);
    let output_path_opt = output_dir_str.map(Path::new);

    if !input_path.is_dir() {
        println!("ERROR: Input path is not a directory: {}", input_dir_str);
        return Err(PyValueError::new_err(format!("Input path is not a directory: {}", input_dir_str)));
    }

    if let Some(out_p) = output_path_opt {
        if !out_p.exists() {
            println!("INFO: Creating output directory: {}", out_p.display());
            fs::create_dir_all(out_p).map_err(|e| 
                PyValueError::new_err(format!("Failed to create output directory {}: {}", out_p.display(), e)))?;
        } else if !out_p.is_dir() {
            println!("ERROR: Output path exists but is not a directory: {}", out_p.display());
            return Err(PyValueError::new_err(format!("Output path exists but is not a directory: {}", out_p.display())));
        }
    }

    // Collect markdown files
    println!("DEBUG: Collecting markdown files from: {}", input_path.display());
    let md_files: Vec<PathBuf> = WalkDir::new(input_path)
        .into_iter().filter_map(Result::ok)
        .filter(|e| e.path().is_file() && e.path().extension().is_some_and(|ext| ext == "md"))
        .map(|e| e.path().to_path_buf())
        .collect();

    println!("DEBUG: Found {} markdown files", md_files.len());

    if md_files.is_empty() {
        println!("INFO: No markdown files found in input directory");
        return Ok(Vec::new());
    }

    println!("getting warmer");

    // Configure thread pool
    let thread_count = if num_threads > 0 {
        println!("counting threads");
        num_threads 
    } else { 
        // Default: use number of logical cores
        println!("using default threads");
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    };

    println!("INFO: Configuring thread pool with {} threads", thread_count);
    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool: {}", e)))?;

    let results = Arc::new(Mutex::new(Vec::<PerFileOutput>::new()));

    println!("INFO: Starting parallel processing with Rayon");
    println!("INFO: Releasing the GIL before starting Rayon tasks");
    
    // Release the GIL before starting Rayon tasks to avoid deadlock
    _py.allow_threads(|| {
        println!("DEBUG: Inside allow_threads, GIL is released");
        
        pool.install(|| {
            println!("DEBUG: Inside Rayon pool.install");
            
            md_files.par_iter().for_each(|md_file_path| {
                let config_clone = Arc::clone(&operation_config);
                println!("DEBUG: Processing file: {}", md_file_path.display());
                
                // Read the file content outside of the GIL
                match fs::read_to_string(md_file_path) {
                    Ok(content) => {
                        println!("DEBUG: Read {} chars from {}", content.len(), md_file_path.display());
                        
                        // Re-acquire the GIL only when needed for Python operations
                        Python::with_gil(|py_thread| {
                            println!("DEBUG: Re-acquired GIL for processing file");
                            
                            match file_operation(py_thread, md_file_path.clone(), &content, &config_clone) {
                                Ok(Some(output_data)) => {
                                    results.lock().unwrap().push(output_data);
                                }
                                Ok(None) => {
                                    // Operation completed but produced no data to collect (e.g. direct file write)
                                }
                                Err(err) => { 
                                    println!("ERROR: Failed to process file: {}: {:?}", md_file_path.display(), err);
                                }
                            }
                        });
                    }
                    Err(err) => { 
                        println!("ERROR: Failed to read file: {}: {:?}", md_file_path.display(), err);
                    }
                }
            });
            
            println!("DEBUG: Completed Rayon parallel processing");
        });
        
        println!("DEBUG: Exited Rayon pool.install");
    });
    
    println!("DEBUG: GIL re-acquired after Rayon processing");

    // Transfer results from Arc<Mutex<Vec<...>>> to Vec<...>
    let final_results = match Arc::try_unwrap(results) {
        Ok(mutex) => mutex.into_inner().map_err(|_e| PyValueError::new_err("Mutex for results was poisoned"))?,
        Err(arc_still_shared) => arc_still_shared.lock().map_err(|_e| PyValueError::new_err("Mutex for results was poisoned during clone"))?.clone(),
    };
    
    Ok(final_results)
}

// Configuration for batch cleaning operations
struct BatchCleanOpConfig {
    allowed_chars: HashSet<char>,
    unusual_chars: HashSet<char>,
}

// Configuration for table analysis operations
struct TableAnalysisConfig {
    min_rows_for_table_start: usize,
}

#[derive(Debug, Serialize)]
struct FileReportData {
    file_name: String,
    original_chars: usize,
    cleaned_chars: usize,
    removed_chars_total: usize,
    badness_score_non_ws: Option<f64>,
    badness_score_all_chars: Option<f64>,
    greek_chars_cleaned: Option<usize>,
    latin_chars_cleaned: Option<usize>,
    percentage_greek_cleaned: Option<f64>,
    percentage_latin_cleaned: Option<f64>,
    cleaned_non_whitespace_chars: Option<usize>,
    error_message: Option<String>,
}

#[pyfunction]
#[pyo3(signature = (input_dir_str, output_csv_path_str, output_dir_cleaned_files_str, scripts_to_keep, num_threads))]
pub fn generate_analysis_report_for_directory(
    py: Python,
    input_dir_str: &str,
    output_csv_path_str: Option<&str>,
    output_dir_cleaned_files_str: Option<&str>,
    scripts_to_keep: Vec<String>,
    num_threads: usize,
) -> PyResult<PyObject> {
    // Initialize script character sets
    let mut allowed_chars = HashSet::new();
    for key in &scripts_to_keep {
        if let Some(script_set) = cleaning_module::SCRIPT_SETS.get(key) {
            allowed_chars.extend(script_set);
        }
    }
    // Ensure essential whitespace is always allowed for cleaning and analysis coherence
    allowed_chars.insert(' ');
    allowed_chars.insert('\t');
    allowed_chars.insert('\n');

    let final_allowed_chars_arc = Arc::new(allowed_chars);
    let unusual_chars_arc = Arc::new(cleaning_module::SCRIPT_SETS.get("unusual").cloned().unwrap_or_default());

    let input_path = PathBuf::from(input_dir_str);
    let output_cleaned_path_opt = output_dir_cleaned_files_str.map(PathBuf::from);

    // Collect markdown files
    let md_files: Vec<PathBuf> = WalkDir::new(&input_path)
        .into_iter().filter_map(Result::ok)
        .filter(|e| e.path().is_file() && e.path().extension().is_some_and(|ext| ext == "md"))
        .map(|e| e.path().to_path_buf())
        .collect();

    if md_files.is_empty() {
        println!("INFO: No markdown files found in {}. If CSV output was expected, it will be empty or have headers only.", input_dir_str);
        // If CSV path is provided, write headers for an empty report
        if let Some(csv_path) = output_csv_path_str {
            if !csv_path.is_empty() {
                 if let Some(parent_dir) = Path::new(csv_path).parent() {
                    if !parent_dir.as_os_str().is_empty() && !parent_dir.exists() {
                        fs::create_dir_all(parent_dir).map_err(|e| PyValueError::new_err(format!("Failed to create parent for empty CSV {}: {}", parent_dir.display(), e)))?;
                    }
                }
                let mut wtr = Writer::from_path(csv_path).map_err(|e| PyValueError::new_err(format!("Failed to create CSV for empty report {}: {}", csv_path, e)))?;
                wtr.write_record(["File Name", "Badness", "Greek Percentage", "Latin Percentage"]).map_err(|e| PyValueError::new_err(format!("CSV header write error for empty report: {}", e)))?;
                wtr.flush().map_err(|e| PyValueError::new_err(format!("CSV flush error for empty report: {}", e)))?;
            }
        }
        let summary = PyDict::new(py);
        summary.set_item("output_csv_path", output_csv_path_str.unwrap_or("N/A"))?;
        summary.set_item("total_issues_found", 0)?; // Corrected from "total_issues_found" to a more generic term
        summary.set_item("files_with_errors", 0)?;
        return Ok(summary.into());
    }

    // Configure thread pool
    let pool = ThreadPoolBuilder::new()
        .num_threads(if num_threads > 0 { num_threads } else { rayon::current_num_threads() })
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool: {}", e)))?;

    let collected_report_data_arc: Arc<Mutex<Vec<FileReportData>>> = Arc::new(Mutex::new(Vec::new()));
    
    let input_path_arc = Arc::new(input_path); // Used for stripping prefix
    let output_cleaned_path_opt_arc = Arc::new(output_cleaned_path_opt); // Used for writing cleaned files

    let analysis_phase_start = std::time::Instant::now();

    py.allow_threads(|| {
        pool.install(|| {
            md_files.par_iter().for_each(|md_file_path| {
                let file_content = match fs::read_to_string(md_file_path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("Error reading file {}: {}", md_file_path.display(), e);
                        // Optionally add to an error list if detailed error reporting per file is needed
                        return; 
                    }
                };

                // Clone Arcs for use in this thread
                let allowed_chars_for_thread = Arc::clone(&final_allowed_chars_arc);
                let unusual_chars_for_thread = Arc::clone(&unusual_chars_arc);
                
                // Perform the text analysis
                let analysis_result = cleaning_module::perform_text_analysis(
                    &file_content,
                    &allowed_chars_for_thread,
                    &unusual_chars_for_thread,
                    &scripts_to_keep, // scripts_to_keep is Vec<String>, perform_text_analysis expects &[String]
                    true, // calculate_specific_counts
                    None  // min_chars_for_comment_override
                );

                let removed_total_chars = analysis_result.original_total_chars.saturating_sub(analysis_result.cleaned_total_chars);

                let percentage_greek_cleaned: Option<f64> = 
                    analysis_result.greek_char_count_after_clean.and_then(|greek_count| {
                        analysis_result.cleaned_non_whitespace_chars_after_clean.map(|total_script_chars| {
                            if total_script_chars > 0 && greek_count > 0 {
                                (greek_count as f64 / total_script_chars as f64) * 100.0
                            } else {
                                0.0
                            }
                        })
                    });

                let percentage_latin_cleaned: Option<f64> = 
                    analysis_result.latin_char_count_after_clean.and_then(|latin_count| {
                        analysis_result.cleaned_non_whitespace_chars_after_clean.map(|total_script_chars| {
                            if total_script_chars > 0 && latin_count > 0 {
                                (latin_count as f64 / total_script_chars as f64) * 100.0
                            } else {
                                0.0
                            }
                        })
                    });
                
                // Optionally save cleaned file
                if let Some(output_base_path) = &*output_cleaned_path_opt_arc {
                    let relative_path = md_file_path.strip_prefix(&*input_path_arc).unwrap_or(md_file_path);
                    let target_file_path = output_base_path.join(relative_path);
                    
                    if let Some(parent_dir) = target_file_path.parent() {
                        if !parent_dir.exists() {
                            if let Err(e) = fs::create_dir_all(parent_dir) {
                                eprintln!("ERROR: Failed to create directory {}: {}", parent_dir.display(), e);
                                // Decide if this error is critical enough to stop or just log
                            }
                        }
                    }
                    
                    if let Err(e) = fs::write(&target_file_path, &analysis_result.cleaned_text_content) {
                        eprintln!("ERROR: Failed to write cleaned file {}: {}", target_file_path.display(), e);
                    }
                }

                let report_entry = FileReportData {
                    file_name: md_file_path.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                    original_chars: analysis_result.original_total_chars,
                    cleaned_chars: analysis_result.cleaned_total_chars,
                    removed_chars_total: removed_total_chars,
                    badness_score_non_ws: analysis_result.badness_score_non_ws,
                    badness_score_all_chars: analysis_result.badness_score_all_chars,
                    greek_chars_cleaned: analysis_result.greek_char_count_after_clean,
                    latin_chars_cleaned: analysis_result.latin_char_count_after_clean,
                    percentage_greek_cleaned,
                    percentage_latin_cleaned,
                    cleaned_non_whitespace_chars: analysis_result.cleaned_non_whitespace_chars_after_clean,
                    error_message: None, // Can be enhanced to capture specific file processing errors
                };
                collected_report_data_arc.lock().unwrap().push(report_entry);
            });
        })
    });

    let analysis_phase_duration = analysis_phase_start.elapsed();
    println!("Parallel analysis phase completed in: {:.2?}", analysis_phase_duration);

    let mut collected_report_data = Arc::try_unwrap(collected_report_data_arc)
        .map_err(|_e| PyValueError::new_err("Mutex for report data was poisoned (arc unwrap failed)"))?
        .into_inner()
        .map_err(|_e| PyValueError::new_err("Mutex for report data was poisoned (into_inner failed)"))?;
    
    collected_report_data.sort_by(|a, b| a.file_name.cmp(&b.file_name));

    // CSV writing
    if let Some(csv_path) = output_csv_path_str {
        if !csv_path.is_empty() {
            println!("Starting CSV writing phase for {}...", csv_path);
            let csv_writing_start = std::time::Instant::now();
            
            if let Some(parent_dir) = Path::new(csv_path).parent() {
                 if !parent_dir.as_os_str().is_empty() && !parent_dir.exists() { // Check parent is not empty string
                    fs::create_dir_all(parent_dir).map_err(|e| PyValueError::new_err(format!(
                        "Failed to create parent directory '{}' for CSV: {}", parent_dir.display(), e
                    )))?;
                }
            }

            match Writer::from_path(csv_path) {
                Ok(mut wtr) => {
                    wtr.write_record([
                        "File Name", 
                        "Badness",
                        "Greek Percentage", 
                        "Latin Percentage",
                    ]).map_err(|e| PyValueError::new_err(format!("CSV header write error: {}", e)))?; 

                    for report_item in &collected_report_data {
                        wtr.write_record([
                            report_item.file_name.clone(),
                            report_item.badness_score_all_chars.map_or_else(|| "N/A".to_string(), |v| format!("{:.4}", v)),
                            report_item.percentage_greek_cleaned.map_or_else(|| "N/A".to_string(), |v| format!("{:.2}%", v)),
                            report_item.percentage_latin_cleaned.map_or_else(|| "N/A".to_string(), |v| format!("{:.2}%", v)),
                        ]).map_err(|e| PyValueError::new_err(format!("CSV row write error for {}: {}", report_item.file_name, e)))?;
                    }
                    wtr.flush().map_err(|e| PyValueError::new_err(format!("CSV flush error: {}", e)))?;
                    let csv_writing_duration = csv_writing_start.elapsed();
                    println!("CSV writing phase completed in: {:.2?}", csv_writing_duration);
                }
                Err(e) => {
                    return Err(PyValueError::new_err(format!("Failed to create CSV writer for path '{}': {}", csv_path, e)));
                }
            }
        } else {
            println!("CSV output path was an empty string, CSV writing skipped.");
        }
    } else {
        println!("CSV writing skipped as no output path was provided.");
    }

    let files_processed_count = collected_report_data.len();
    // This count is slightly inaccurate as md_files could have read errors not diminishing this count.
    // A more accurate error count would require tracking errors during the par_iter.
    let files_with_potential_read_errors = md_files.len().saturating_sub(files_processed_count); 

    println!(
        "CSV report generated at: {}. Files processed for report: {}, Potential file read issues: {}",
        output_csv_path_str.unwrap_or("N/A (not specified)"),
        files_processed_count,
        files_with_potential_read_errors // Renamed for clarity
    );

    let summary = PyDict::new(py);
    summary.set_item("output_csv_path", output_csv_path_str.unwrap_or("N/A"))?;
    summary.set_item("files_processed_for_report", files_processed_count)?; // Clarified name
    summary.set_item("files_with_potential_read_errors", files_with_potential_read_errors)?; // Clarified name
    Ok(summary.into())
}

/// Python-exposed function for batch cleaning of markdown files
#[pyfunction]
pub fn batch_clean_markdown_files(
    py: Python,
    input_dir: &str,
    output_dir: &str,
    scripts_to_keep: Vec<String>,
    num_threads: usize,
) -> PyResult<PyObject> {
    // Debug prints
    println!("INFO: Starting batch_clean_markdown_files with {} threads", num_threads);
    println!("DEBUG: Input dir: {}", input_dir);
    println!("DEBUG: Output dir: {}", output_dir);
    println!("DEBUG: Scripts to keep: {:?}", scripts_to_keep);

    // Prepare character sets for cleaning
    let mut allowed_chars = HashSet::new();
    
    // Debug print for available CPU cores
    println!("INFO: Available CPU cores: {}, using {} threads", 
             std::thread::available_parallelism().map_or(4, |n| n.get()), num_threads);
    
    // Fix script mapping to match what's in SCRIPT_SETS (lat->lat, not lat->latin)
    println!("DEBUG: Script mapping from user input: {:?}", scripts_to_keep);
    
    // Check if scripts exist and add their characters
    for key in &scripts_to_keep {
        if let Some(script_set) = cleaning_module::SCRIPT_SETS.get(key) {
            println!("DEBUG: Adding {} characters from script: {}", script_set.len(), key);
            allowed_chars.extend(script_set);
        } else {
            println!("WARNING: Script '{}' not found in SCRIPT_SETS", key);
        }
    }
    
    // Include common non-alphabetic sets if not specified - use correct keys that match SCRIPT_SETS
    let keys_to_include = ["punctuation", "numbers", "common_symbols"]; // Corrected keys
    println!("DEBUG: Also adding characters from: {:?}", keys_to_include);
    
    for key_to_always_include in keys_to_include {
        if !scripts_to_keep.contains(&key_to_always_include.to_string()) {
            if let Some(script_set) = cleaning_module::SCRIPT_SETS.get(key_to_always_include) {
                println!("DEBUG: Adding {} characters from always-included script: {}", 
                        script_set.len(), key_to_always_include);
                allowed_chars.extend(script_set);
            } else {
                println!("WARNING: Always-include script '{}' not found in SCRIPT_SETS", key_to_always_include);
            }
        }
    }

    // Add essential whitespace characters
    allowed_chars.insert(' ');
    allowed_chars.insert('\t');
    allowed_chars.insert('\n');
    println!("DEBUG: Added whitespace characters");

    let unusual_chars = cleaning_module::SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
    println!("DEBUG: Using {} unusual characters for detection", unusual_chars.len());
    println!("DEBUG: Total allowed characters: {}", allowed_chars.len());
    
    let config = Arc::new(BatchCleanOpConfig { 
        allowed_chars, 
        unusual_chars 
    });

    // Define base paths for path manipulation within the closure
    let input_base_path = PathBuf::from(input_dir);
    let output_base_path = PathBuf::from(output_dir);

    // Define the cleaning operation
    let clean_file_op = move |_py_thread: Python, md_file_path: PathBuf, content: &str, op_conf: &Arc<BatchCleanOpConfig>| -> PyResult<Option<PerFileOperationOutput>> {
        let cleaned_content_tuple = cleaning_module::core_clean_text(
            content, 
            &op_conf.allowed_chars, 
            &op_conf.unusual_chars,
            None // min_chars_for_comment_override
        );
        
        // Determine output path for the cleaned file
        let relative_path = md_file_path.strip_prefix(&input_base_path) // Use cloned input_base_path
            .unwrap_or(&md_file_path); 
        let target_file_path = output_base_path.join(relative_path); // Use cloned output_base_path

        // Ensure parent directory exists
        if let Some(parent_dir) = target_file_path.parent() {
            if !parent_dir.exists() {
                fs::create_dir_all(parent_dir).map_err(|e| {
                    PyValueError::new_err(format!("Failed to create directory {}: {}", parent_dir.display(), e))
                })?;
            }
        }

        // Write the cleaned file
        fs::write(&target_file_path, cleaned_content_tuple.0).map_err(|e| {
            PyValueError::new_err(format!("Failed to write cleaned file {}: {}", target_file_path.display(), e))
        })?;
        
        Ok(Some(PerFileOperationOutput::OperationSuccess(md_file_path.clone())))
    };

    // Process the directory using the core function
    let results: Vec<PerFileOperationOutput> = process_directory_core::<
        BatchCleanOpConfig, 
        _,                  
        PerFileOperationOutput 
    >(
        py, 
        input_dir,
        Some(output_dir), 
        num_threads,
        config,
        clean_file_op,
    )?;

    let mut files_processed_count = 0;
    let mut files_error_count = 0; 
    
    for res_item in results {
        match res_item {
            PerFileOperationOutput::OperationSuccess(_) => files_processed_count += 1,
            PerFileOperationOutput::OperationError(_, _) => files_error_count += 1, 
            _ => {} 
        }
    }

    let summary = pyo3::types::PyDict::new(py); 
    summary.set_item("status", "completed")?;
    summary.set_item("message", format!("Batch cleaning completed. {} files processed. See output directory.", files_processed_count))?;
    summary.set_item("files_processed", files_processed_count)?;
    summary.set_item("files_with_errors", files_error_count)?;

    Ok(summary.into())
}

/// Python-exposed function for processing directory with original behavior
/// This maintains compatibility with existing Python code
#[pyfunction]
pub fn process_directory_native(
    py: Python,
    input_dir: &str,
    output_dir: &str,
    scripts_to_keep: Vec<String>,
    num_threads: usize,
) -> PyResult<PyObject> {
    // This is a wrapper around batch_clean_markdown_files for backward compatibility
    batch_clean_markdown_files(py, input_dir, output_dir, scripts_to_keep, num_threads)
}

#[pyfunction]
pub fn batch_generate_table_summary_csv(
    py: Python, 
    input_dir_str: &str, 
    output_csv_path_str: &str, 
    num_threads: usize
) -> PyResult<()> {
    let input_path = Path::new(input_dir_str);
    if !input_path.is_dir() {
        return Err(PyValueError::new_err(format!(
            "Input path is not a directory: {}",
            input_dir_str
        )));
    }

    let md_files: Vec<PathBuf> = WalkDir::new(input_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| {
            e.path().is_file() && e.path().extension().map_or(false, |ext| ext == "md")
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if md_files.is_empty() {
        println!("No markdown files found in input directory: {}", input_dir_str);
        let mut wtr = Writer::from_path(output_csv_path_str)
            .map_err(|e| PyValueError::new_err(format!("Failed to create CSV writer: {}", e)))?;
        wtr.write_record(&["file", "total_tables", "malformed_tables"])
            .map_err(|e| PyValueError::new_err(format!("Failed to write CSV header: {}", e)))?;
        wtr.flush().map_err(|e| PyValueError::new_err(format!("Failed to flush CSV: {}", e)))?;
        return Ok(());
    }

    let num_effective_threads = if num_threads == 0 {
        rayon::current_num_threads()
    } else {
        num_threads
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_effective_threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("Failed to build Rayon thread pool: {}", e)))?;

    // Corrected GIL handling: py.allow_threads is outermost for the parallel section.
    // The `pool.install` closure itself should be Send.
    let results: Vec<FileTableSummary> = py.allow_threads(|| {
        pool.install(|| {
            md_files
                .par_iter()
                .filter_map(|md_file_path| {
                    Python::with_gil(|py_thread_token| {
                        match fs::read_to_string(md_file_path) {
                            Ok(content) => {
                                match table_analysis_module::analyze_table_file_op(py_thread_token, &content) {
                                    Ok(scan_result) => {
                                        let relative_path = md_file_path.strip_prefix(input_path)
                                            .unwrap_or(md_file_path)
                                            .to_string_lossy()
                                            .into_owned();
                                        
                                        Some(FileTableSummary {
                                            file_path: relative_path,
                                            total_tables: scan_result.total_tables,
                                            malformed_tables: scan_result.issues.len(),
                                        })
                                    }
                                    Err(e) => {
                                        eprintln!("Error analyzing file {}: {}", md_file_path.display(), e);
                                        None
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Error reading file {}: {}", md_file_path.display(), e);
                                None
                            }
                        }
                    })
                })
                .collect()
        })
    });

    if let Some(parent_dir) = Path::new(output_csv_path_str).parent() {
        if !parent_dir.exists() {
            fs::create_dir_all(parent_dir).map_err(|e| {
                PyValueError::new_err(format!(
                    "Failed to create output directory {}: {}",
                    parent_dir.display(),
                    e
                ))
            })?;
        }
    }
    
    let mut wtr = Writer::from_path(output_csv_path_str)
        .map_err(|e| PyValueError::new_err(format!("Failed to create CSV writer for {}: {}", output_csv_path_str, e)))?;
    
    wtr.write_record(&["file", "total_tables", "malformed_tables"])
        .map_err(|e| PyValueError::new_err(format!("Failed to write CSV header: {}", e)))?;

    for summary in results {
        wtr.write_record(&[
            summary.file_path,
            summary.total_tables.to_string(),
            summary.malformed_tables.to_string(),
        ])
        .map_err(|e| PyValueError::new_err(format!("Failed to write CSV row: {}", e)))?;
    }

    wtr.flush().map_err(|e| PyValueError::new_err(format!("Failed to flush CSV: {}", e)))?;

    println!("Table summary CSV report generated at: {}", output_csv_path_str);
    Ok(())
}

// For detailed table issue reporting by batch_generate_detailed_table_report_csv
impl Default for TableAnalysisConfig {
    fn default() -> Self {
        TableAnalysisConfig {
            min_rows_for_table_start: 2, // Default value
        }
    }
}

// Function to generate a detailed CSV report of table issues (file_path, issue_description, start_line, end_line)
#[pyfunction]
pub fn batch_generate_detailed_table_report_csv(
    py: Python,
    input_dir_str: &str,
    output_csv_path_str: &str,
    num_threads: usize,
) -> PyResult<PyObject> { // Return PyDict summary
    println!("Starting detailed table report generation for CSV: {}", output_csv_path_str);

    // Ensure the output directory for the CSV exists
    if let Some(parent_dir) = Path::new(output_csv_path_str).parent() {
        if !parent_dir.exists() {
            fs::create_dir_all(parent_dir).map_err(|e| PyValueError::new_err(format!("Failed to create parent directory for CSV {}: {}", parent_dir.display(), e)))?;
        }
    }
    
    let mut wtr = csv::WriterBuilder::new()
        .has_headers(false) // Explicitly disable automatic header writing by serialize
        .from_path(output_csv_path_str)
        .map_err(|e| PyValueError::new_err(format!("Failed to create CSV writer for {}: {}", output_csv_path_str, e)))?;

    // Manually write header once.
    wtr.write_record(&["file_path", "issue_description", "table_start_line", "table_end_line"])
        .map_err(|e| PyValueError::new_err(format!("Failed to write CSV header to {}: {}", output_csv_path_str, e)))?;
    
    // Use a dummy config as analyze_table_file_op doesn't require a specific one.
    let dummy_config = Arc::new(()); 

    let operation_fn = move |py_token: Python, file_path: PathBuf, content: &str, _config: &Arc<()>| -> PyResult<Option<PerFileOperationOutput>> {
        let relative_path_str = path_to_str(&file_path); 
        match table_analysis_module::analyze_table_file_op(py_token, content) { // Call analyze_table_file_op
            Ok(scan_result) => {
                let issues_for_file: Vec<DetailedTableIssueReportEntry> = scan_result.issues.into_iter().map(|issue_obj| {
                    // Assuming issue_obj is Py<TableIssue>, need to borrow to access fields
                    let issue = issue_obj.as_ref(py_token).borrow();
                    DetailedTableIssueReportEntry {
                        file_path: relative_path_str.clone(),
                        issue_description: issue.description.clone(), // Clone if String
                        table_start_line: issue.start_line,
                        table_end_line: issue.end_line,
                    }
                }).collect();
                if issues_for_file.is_empty() {
                    Ok(None) 
                } else {
                    Ok(Some(PerFileOperationOutput::DetailedTableIssues(issues_for_file)))
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to analyze tables in {}: {}", relative_path_str, e);
                println!("ERROR: {}", err_msg);
                Err(PyValueError::new_err(err_msg))
            }
        }
    };
    
    let processed_results = process_directory_core(
        py,
        input_dir_str,
        None, 
        num_threads,
        dummy_config, // Pass the dummy config
        operation_fn, 
    )?;
    
    let mut total_issues_found = 0;
    let mut files_with_issues = 0;

    for file_results_enum in processed_results {
        if let PerFileOperationOutput::DetailedTableIssues(entries) = file_results_enum {
            if !entries.is_empty() {
                files_with_issues += 1;
            }
            for entry in entries {
                wtr.serialize(entry).map_err(|e| PyValueError::new_err(format!("Failed to serialize detailed issue to CSV: {}", e)))?;
                total_issues_found += 1;
            }
        } else {
            // Log if a file produced an unexpected PerFileOperationOutput variant or an error handled internally by process_directory_core
            // println!("WARN: Received unexpected output variant or error for a file.");
        }
    }

    wtr.flush().map_err(|e| PyValueError::new_err(format!("Failed to flush CSV writer for {}: {}", output_csv_path_str, e)))?;
    println!("Detailed table report generation complete. Total issues found: {}, Files with issues: {}", total_issues_found, files_with_issues);

    let summary = PyDict::new(py);
    summary.set_item("output_csv_path", output_csv_path_str)?;
    summary.set_item("total_issues_found", total_issues_found)?;
    summary.set_item("files_with_issues", files_with_issues)?;
    Ok(summary.into())
}

// --- New Batch Function for Table Removal ---
#[pyfunction]
#[pyo3(signature = (input_markdown_dir_str, detailed_report_csv_path_str, output_processed_md_dir_str, num_threads))]
pub fn batch_remove_tables_from_files(
    py: Python,
    input_markdown_dir_str: &str,
    detailed_report_csv_path_str: &str,
    output_processed_md_dir_str: &str,
    num_threads: usize,
) -> PyResult<()> {
    // println!("Starting batch table removal...");

    // 1. Load table locations from the CSV report
    let table_locations_map = table_remover_module::load_table_locations_from_csv(detailed_report_csv_path_str)
        .map_err(|e| PyValueError::new_err(format!("Failed to load table locations from CSV {}: {}", detailed_report_csv_path_str, e)))?;
    
    let arc_table_locations_map = Arc::new(table_locations_map);
    let input_base_path = PathBuf::from(input_markdown_dir_str);
    let output_base_path = PathBuf::from(output_processed_md_dir_str);

    // Ensure output directory exists (process_directory_core might also do this, but good to be sure)
    fs::create_dir_all(&output_base_path).map_err(|e|
        PyValueError::new_err(format!("Failed to create output directory {}: {}", output_base_path.display(), e))
    )?;

    // Define the file operation for table removal
    let file_op = move |_py_thread: Python, md_file_path: PathBuf, content: &str, config: &Arc<HashMap<PathBuf, Vec<table_remover_module::LineRange>>>| -> PyResult<Option<()>> {
        // Determine relative path for looking up in map and for output path
        let relative_path = md_file_path.strip_prefix(&input_base_path)
            .unwrap_or(&md_file_path); // Fallback to full path if strip_prefix fails (should not happen if md_file_path is from input_dir)
        
        let target_output_file_path = output_base_path.join(relative_path);

        if let Some(parent_dir) = target_output_file_path.parent() {
            if !parent_dir.exists() {
                fs::create_dir_all(parent_dir).map_err(|e| 
                    PyValueError::new_err(format!("Failed to create parent directory for {}: {}", target_output_file_path.display(), e))
                )?;
            }
        }
        
        // Check if this file has tables to remove
        // The keys in `config` (our `table_locations_map`) might be absolute or relative.
        // We need to ensure consistent lookup. `load_table_locations_from_csv` stores paths as read.
        // For robustness, one might normalize paths before inserting into map and during lookup.
        // Assuming paths in CSV are relative to `input_markdown_dir_str` or absolute and match `md_file_path`.
        // For simplicity, let's assume `md_file_path` (which is absolute) is what we should use for lookup if CSV has absolute.
        // Or, if CSV paths are relative to `input_markdown_dir_str`, we'd use `relative_path`.
        // The current `load_table_locations_from_csv` uses `PathBuf::from(record.file_path)`.
        // Let's try looking up with the absolute path `md_file_path`.

        if let Some(locations_for_this_file) = config.get(&md_file_path) {
            if !locations_for_this_file.is_empty() {
                let modified_content = table_remover_module::remove_tables_from_content(content, locations_for_this_file);
                fs::write(&target_output_file_path, modified_content).map_err(|e| 
                    PyValueError::new_err(format!("Failed to write modified file {}: {}", target_output_file_path.display(), e))
                )?;
            } else {
                // File was in report, but no locations (empty vec), copy verbatim
                fs::write(&target_output_file_path, content).map_err(|e| 
                    PyValueError::new_err(format!("Failed to copy (no-op) file {}: {}", target_output_file_path.display(), e))
                )?;
            }
        } else {
            // File not in report, copy verbatim
            fs::write(&target_output_file_path, content).map_err(|e| 
                PyValueError::new_err(format!("Failed to copy (verbatim) file {}: {}", target_output_file_path.display(), e))
            )?;
        }
        Ok(None) // No data to collect, side effect is file writing
    };

    // Use process_directory_core to iterate and apply the operation.
    // The output_dir_str for process_directory_core is the output_processed_md_dir_str,
    // which it can use to create the base output directory.
    let _: Vec<()> = process_directory_core( // Result type is Vec<() because file_op returns Option<()
        py,
        input_markdown_dir_str,
        Some(output_processed_md_dir_str), // Pass output dir so core can create it if needed
        num_threads,
        arc_table_locations_map, // Pass the map as config
        file_op,
    )?;
    
    // println!("Batch table removal completed. Processed files are in: {}", output_processed_md_dir_str);
    Ok(())
}
