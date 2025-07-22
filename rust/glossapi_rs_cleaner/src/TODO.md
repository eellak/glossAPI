# TODO: Rust Core Logic and Python Wrapper Enhancements

## Goal
Enhance the `text_cleaner_rs` Rust library and its Python wrapper (`table_detector.py`) to:
1.  Provide accurate counts of `total_tables` and `malformed_tables` per Markdown file, with Rust performing all core analysis and CSV generation in batch mode.
2.  Maximize processing efficiency by leveraging Rust for file discovery, parallel processing, analysis, and CSV report generation.
3.  Python wrapper (`table_detector.py`) to act as a thin layer, primarily invoking the Rust batch processing function.
4.  Output CSV: `file,total_tables,malformed_tables`.

## Phase 1: Rust Core Logic Enhancements (`extraction_metrics_rs`)

1.  **Core Table Analysis (`table_analysis_module.rs`):**
    *   **Define `TableScan` Struct**:
        ```rust
        // pub struct TableScan<'py> { // Or without lifetime if Py<TableIssue> handles it well across calls
        pub struct TableScan {
            pub total: usize,
            pub issues: Vec<Py<table_analysis_module::TableIssue>>, // Assuming TableIssue is defined in this module
        }
        ```
    *   **Modify `core_detect_malformed_tables`**:
        *   Return `PyResult<TableScan>`.
        *   Implement logic to count `total_tables` (e.g., each `TABLE_SEPARATOR_REGEX` match) and collect `issues`.
        *   Employ robust loop for table body skipping as suggested:
            ```rust
            // while i < lines.len() {
            //     if TABLE_SEPARATOR_REGEX.is_match(lines[i]) {
            //         total_tables += 1;
            //         // ... existing mismatch checks, push issues ...
            //         // skip body rows
            //         while i + 1 < lines.len() && TABLE_ROW_REGEX.is_match(lines[i + 1]) {
            //             i += 1;
            //         }
            //     }
            //     i += 1;
            // }
            ```
        *   (Status: The user-provided `table_analysis_module.rs` already implements the core logic for returning total count and issues as a tuple, which is a good foundation. Adopting `TableScan` internally is a refinement.)
    *   **Update `analyze_tables_in_string` (PyO3 exposed function):**
        *   Call `core_detect_malformed_tables`.
        *   Convert `TableScan` to `PyResult<(usize, Vec<Py<TableIssue>>)>` for Python.
        ```rust
        // pub fn analyze_tables_in_string<'py>( ... ) -> PyResult<(usize, Vec<Py<'py, TableIssue>>)> {
        //     let scan = core_detect_malformed_tables(py, markdown_text)?;
        //     Ok((scan.total, scan.issues))
        // }
        ```
    *   **Update `analyze_table_file_op` (internal for batch processing):**
        *   This function, used by `directory_processor.rs`, should also call `core_detect_malformed_tables` and return `PyResult<TableScan>`.

2.  **Batch Processing & CSV Generation (`directory_processor.rs`):**
    *   **Extend `PerFileOperationOutput` Enum**:
        ```rust
        // pub enum PerFileOperationOutput<'py> { // Propagate lifetime if needed
        pub enum PerFileOperationOutput {
            Content(String), // For cleaning operations
            TableScanResult(table_analysis_module::TableScan), // Contains total and issues
            Empty,
        }
        ```
    *   **Critical Action: Enhance `batch_analyze_tables_in_files` (or a new dedicated function e.g., `batch_generate_table_summary_csv`):**
        *   This function will be the primary entry point from Python for batch mode.
        *   **Parameters**: `input_dir: &str`, `output_csv_path: &str`, `num_threads: usize`.
        *   **Internal Logic**:
            *   Use `WalkDir` to find all `.md` files in `input_dir`.
            *   Use Rayon for parallel processing of files.
            *   For each file:
                *   Read content.
                *   Call `table_analysis_module::analyze_table_file_op` (which returns `TableScan`).
                *   The result for each file processing step within Rayon should be `(String, TableScan)` (file_path, scan_result).
            *   Collect these `(String, TableScan)` results from all threads.
            *   **Transform Results**: Convert `Vec<(String, TableScan)>` into `Vec<FileTableSummary>`:
                ```rust
                // struct FileTableSummary {
                //     file: String, // Relative path
                //     total: usize,
                //     malformed: usize, // Calculated as scan.issues.len()
                // }
                ```
            *   **Direct CSV Writing**:
                *   Open/create `output_csv_path`.
                *   Use `csv::Writer` to write the header: `file,total_tables,malformed_tables`.
                *   Iterate through `Vec<FileTableSummary>` and write each record.
        *   **Return Value**: `PyResult<()>` or a PyDict with summary stats (e.g., files processed, time taken) if desired, but the main output is the CSV file.

## Phase 2: Python Wrapper Modifications (`table_detector.py`)

1.  **Primary Batch Mode Invocation:**
    *   Modify `main()` in `table_detector.py` to primarily call the Rust batch function (e.g., `text_cleaner_rs.batch_generate_table_summary_csv(input_dir, output_csv_path, num_threads)`).
    *   The Python script will handle argument parsing and then delegate entirely to Rust for processing and CSV generation.
    *   The script will report success/failure based on the Rust function's result.

2.  **Single-File Analysis (Optional/Debug):**
    *   The existing `analyze_file` function (calling `text_cleaner_rs.analyze_tables_in_string` and returning `(file, total, malformed, error)`) can be kept.
    *   This path might be triggered by a command-line flag (e.g., `--mode=single_file_debug`) or used for processing a very small sample if the main batch function doesn't handle sampling.
    *   If kept, its CSV writing part would be separate or removed if this mode is only for stdout.

## Phase 3: Build and Testing

1.  **Build Rust Library:**
    *   Run `cargo build --release` or `maturin develop --release` in `extraction_metrics_rs`.
2.  **Test:**
    *   Unit test Rust functions, especially the new `TableScan` logic and CSV generation.
    *   Test `table_detector.py` in its primary batch mode against the full dataset.
    *   Verify the `table_summary_report.csv` content and structure.
    *   Optionally, test the single-file debug path if retained.

## Phase 4: Post-Implementation Review & Refinements

1.  **Review Character Cleaning Logic (Unexpected Characters):**
    *   **Observation**: Some seemingly "bad" or unexpected characters were observed in cleaned files (e.g., in `cleaned_output_gazette_tag_fix/COH_723.md` around lines 783-856).
    *   **Action**: Re-evaluate the `SCRIPT_SETS` definition in `cleaning_module.rs`, particularly the construction of the "unusual" set and the breadth of "latin", "greek", "punctuation", and "common_symbols".
    *   **Goal**: Understand why these characters are preserved. The current logic removes characters if they are in "unusual" AND NOT in "allowed_chars" (which is built from `scripts_to_keep` + defaults like punctuation/symbols).
    *   **Consider**: Are these characters:
        *   Correctly part of an allowed script (e.g., extended Latin, specific punctuation)?
        *   Not included in the "unusual" set by its current definition?
        *   Intentionally kept due to the specific `scripts_to_keep` used when generating the cleaned output?
    *   This is not necessarily a bug, but a review point to ensure the cleaning behavior aligns with expectations for character sets.

## Rust Implementation Notes & Tips (from User Suggestion):
*   **Lifetimes**:
    *   If `PerFileOperationOutput` holds `Py<TableIssue>`, ensure lifetimes (`'py`) are correctly propagated. E.g., `PerFileOperationOutput<'py>`, `process_directory_core<'py, ...>`.
    *   When building `Vec<Py<TableIssue>>`, maintain the same `'py` lifetime. Propagate generics rather than using `Python::with_gil(|py| { ... })` multiple times if passing Python objects around.
*   **Build/Clenliness**: Run `cargo clippy -- -D warnings` and `cargo fmt`.
*   The core principle: Rust handles computation-intensive tasks, file I/O, parallelism, and CSV generation. Python is a thin orchestrator.

---
**Next Immediate Step**: Implement Rust changes in `table_analysis_module.rs` (adopt `TableScan`) and then the significant modifications in `directory_processor.rs` for batch processing and direct CSV writing. 

---
## Future Enhancements: Two-Stage Cleaning & Global Badness Score

**Overall Goal**: Implement a more robust cleaning pipeline where initial text cleaning is followed by specific cleaning/removal of malformed tables. Then, calculate a global badness score reflecting both stages.

**Phase 5: Implement Malformed Table Removal (Second Cleaning Stage)**

*   **Objective**: Create a Rust function that reads a directory of (stage-1 cleaned) markdown files and a table issues report (CSV detailing malformed tables with line numbers, e.g., `analysis_gazette_tag_fix.csv` or a similar detailed report from `batch_generate_table_summary_csv` if it's adapted to output line-level details).
    It will then produce a new set of markdown files where identified malformed tables are replaced with a `<!-- table-missing -->` comment.

*   **Rust Core (`extraction_metrics_rs`):**
    0.  **Encapsulate Table Removal Logic**: Design the table removal functionality in a well-encapsulated manner. This might involve creating a new Rust module (e.g., `table_remover_module.rs`) if the complexity warrants it, or organizing it clearly within existing modules like `directory_processor.rs`.
    1.  **New Function in `directory_processor.rs` (e.g., `batch_process_table_structures`):**
        *   **Inputs**:
            *   `input_md_dir_stage1`: Path to directory of markdown files already processed by initial cleaning.
            *   `table_issues_csv_path`: Path to the CSV report detailing malformed tables (must include filename and line numbers of issues).
            *   `output_md_dir_stage2`: Directory to save files after table processing.
            *   `mode`: An enum or string, e.g., "remove_malformed_tables", "tag_malformed_tables" (initially focus on removal).
            *   `num_threads`: For parallel processing.
        *   **Logic**:
            *   **Parse CSV**: Read the `table_issues_csv_path` to get a list of files and the line numbers of table issues within them. Group issues by file.
            *   **Process Files**: For each file in `input_md_dir_stage1` that has reported issues:
                *   Read the file content.
                *   For each reported issue line number, identify the full extent (start/end lines) of the malformed table. This is a key challenge.
                    *   *Initial Strategy*: From an issue line (e.g., a separator), scan upwards for a potential header and downwards for table rows (`TABLE_ROW_REGEX`) to define the block. Attempt to merge overlapping blocks if multiple issues point to the same table.
                    *   *Future Improvement*: Augment `TableScan` / `TableIssue` in `table_analysis_module.rs` to store `start_line` and `end_line` for each detected table structure during the analysis phase. This would make identification much more precise.
                *   **Replace Table**: Replace the identified lines of each unique malformed table with a single `<!-- table-missing -->` comment. Ensure a table is replaced only once even if it has multiple reported issues.
                *   Write the modified content to `output_md_dir_stage2`.
            *   Files from `input_md_dir_stage1` with no table issues reported in the CSV should be copied verbatim to `output_md_dir_stage2`.
        *   **PyO3 Exposure**: Expose this new function in `lib.rs`.

*   **Python Wrapper (`table_detector.py` or new script):**
    1.  Add CLI arguments to invoke `batch_process_table_structures`.
    2.  The user will need to provide paths to the stage-1 cleaned files, the table issues CSV, and the new output directory for stage-2 files.

**Phase 6: Update Badness Score Calculation**

*   **Objective**: Refine or introduce new badness scores that account for both cleaning stages.
*   **Considerations**:
    1.  **Expose Character Counts from Stage 1**: Ensure the initial cleaning process (e.g., `generate_analysis_report_for_directory` or `batch_clean_markdown_files`) outputs:
        *   `original_raw_chars_count` (before any processing).
        *   `chars_after_stage1_text_cleaning`.
    2.  **Character Counts from Stage 2 (Table Removal)**:
        *   The `batch_process_table_structures` function should calculate/report the number of characters effectively removed by replacing tables with comments (`chars_removed_from_tables`).
    3.  **New Badness Score Logic**:
        *   Develop a Rust function (e.g., in `cleaning_module.rs` or `directory_processor.rs`) that can compute a global badness score.
        *   Example: `global_badness = (original_raw_chars_count - (chars_after_stage1_text_cleaning - chars_removed_from_tables)) / original_raw_chars_count`.
        *   This function would need access to the character counts from both stages.
    4.  **Reporting**: Update CSV reports or create new ones to include these multi-stage character counts and the new global badness score.

**Phase 7: Integrated Pipeline (Future Direction)**

*   **Objective**: Create a high-level Python script to orchestrate the entire multi-stage pipeline:
    1.  Raw MD input -> Stage 1 Text Cleaning -> Stage 1 Cleaned MD output & Initial Cleaning Report (with char counts).
    2.  Stage 1 Cleaned MD input -> Table Analysis -> Table Issues CSV.
    3.  Stage 1 Cleaned MD input & Table Issues CSV -> Stage 2 Table Processing -> Stage 2 Cleaned MD output & Table Removal Stats (char counts).
    4.  Initial Cleaning Report & Table Removal Stats -> Global Badness Score Calculation -> Final Report.

This phased approach allows for incremental development and testing. 