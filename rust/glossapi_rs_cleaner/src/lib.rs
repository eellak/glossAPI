// Main module for the text_cleaner_rs Python module
// Implements refactored code with better separation of concerns

// Internal modules
mod cleaning_module;
mod directory_processor;
mod pipeline_module;
mod table_analysis_module;
mod table_remover_module;

// Export public items from modules via PyO3
use directory_processor::{
    batch_generate_detailed_table_report_csv, batch_generate_table_summary_csv,
    batch_remove_tables_from_files, generate_analysis_report_for_directory,
};
use pipeline_module::run_complete_pipeline; // Bring the #[pyfunction] into scope
use pyo3::prelude::*;

// Python module definition
#[pymodule]
fn glossapi_rs_cleaner(_py: Python, m: &PyModule) -> PyResult<()> {
    // Registering the new pipeline function (now directly as it's a #[pyfunction])
    m.add_function(wrap_pyfunction!(run_complete_pipeline, m)?)?;

    // Re-exposing older functions for individual script compatibility
    m.add_function(wrap_pyfunction!(generate_analysis_report_for_directory, m)?)?;
    m.add_function(wrap_pyfunction!(
        batch_generate_detailed_table_report_csv,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(batch_remove_tables_from_files, m)?)?;
    m.add_function(wrap_pyfunction!(batch_generate_table_summary_csv, m)?)?;

    // For now, only exposing the main pipeline function and essential classes.
    // Other individual functions from submodules can be re-exposed later if needed,
    // after verifying their exact names and signatures as defined within their modules.

    m.add_class::<table_analysis_module::TableIssue>()?; // TableIssue is a PyClass
                                                         // Other classes like TableScan and SlimTextAnalysisResult are not PyClasses and were removed.

    // Most functions from directory_processor are likely superseded by the new pipeline for typical use.
    // If specific ones like generate_analysis_report_for_directory are still needed independently,
    // they can be added back carefully.

    Ok(())
}
