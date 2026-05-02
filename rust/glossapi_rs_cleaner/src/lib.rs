//! GlossAPI Rust cleaner — production cleaning pipeline (PyO3 module).
//!
//! # Boundary with `glossapi_rs_noise`
//!
//! Per `CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25` Point 7:
//!
//! - **This crate** owns *cleaning* and the *production counters*
//!   that drive corpus-cleaning sample cuts. `clean_text_with_stats`
//!   emits per-rule counts (`rule_a_match_count`, `rule_b_match_count`,
//!   `residue_line_drop_count`) directly inside `CleanStats`, so
//!   production drivers don't need a second pass through `glossapi_rs_noise`.
//! - **`glossapi_rs_noise`** owns *diagnostic / debug exports* —
//!   OCR scoring, word-repeat span extraction, token-category review
//!   bundling. Cleaner production paths never import from it.
//!
//! Phase A (markdown formatting): default is Pilot B
//! (`md_format_surgical::format_surgical_checked`).
//! `cmark-gfm` is OPTIONAL: if installed, it serves as ground-truth
//! oracle (per-doc subprocess overhead); if not, the in-process
//! `dual_verify` (comrak + pulldown-cmark) path is used. Production
//! assumes the dual_verify path.
//!
//! Stage 1 of the cleaner integration: new modules are imported but
//! not yet wired into Corpus.clean()'s call path. Stage 3 flips the
//! production wiring.

// Lint posture (CLEANER_PIPELINE_CLEANUP_PLAN_2026-04-25 Item 5):
// `dead_code` is allowed at crate level because several utility
// functions / variants are kept around as parts of an evolving public
// surface (e.g. `analyze_text`, `list_available_scripts`,
// `drop_low_salvage_pages`, `process_directory_native`,
// `batch_clean_markdown_files`) — they are documented dev/audit tools
// or back-compat exports that are not invoked from the production
// hot path but should remain available. Real bugs (unused variables,
// unread assignments) still warn; only DEAD-CODE noise is silenced.
#![allow(dead_code)]

// Internal modules
mod charset_module;
mod cleaning_module;
mod cmark_gfm_oracle;
mod directory_processor;
mod latex_module;
mod md_format;
mod md_format_surgical;
mod md_module;
mod md_verify;
mod normalize;
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
