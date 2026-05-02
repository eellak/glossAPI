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
//! (`PhaseAMode::ParserSurgicalVerified` → `md_format_surgical::format_surgical_checked`).
//! `LineBased` is regression-test-only — never use for production.
//! `cmark-gfm` is OPTIONAL: if installed, it serves as ground-truth
//! oracle (per-doc subprocess overhead); if not, the in-process
//! `dual_verify` (comrak + pulldown-cmark) path is used. Production
//! assumes the dual_verify path.

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
use charset_module::{analyze_charset, non_empty_line_stats};
use cleaning_module::{clean_text, clean_text_with_stats};
use cmark_gfm_oracle::cmark_gfm_verify_py;
use directory_processor::{
    batch_generate_detailed_table_report_csv, batch_generate_table_summary_csv,
    batch_remove_tables_from_files, generate_analysis_report_for_directory,
};
use latex_module::crop_latex_repetitions_py;
// Dead exports excised in the cleaner-integration-20260430 PR:
// - format_parsed_py (Pilot A — superseded by Pilot B's
//   format_surgical_checked).
// - dual_verify_py (dev-only oracle exposure; dual_verify itself
//   stays as Rust-internal for format_surgical_checked).
// - format_surgical_py (Pilot B without oracle check; dev-only).
// - apply_phase_a / phase_a_alteration_stats / phase_a_stats_jsonl_line
//   (LineBased path instrumentation; LineBased was removed entirely
//   from md_module.rs).
use md_format_surgical::{format_surgical_checked_py, phase_a_policy_py};
use md_verify::{verify_md_preview_equivalent_py, verify_md_structural_py};
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

    // Per-row cleaning entry for scripts that need to clean corpus-parquet
    // `text` columns without round-tripping through markdown files.
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_charset, m)?)?;
    m.add_function(wrap_pyfunction!(non_empty_line_stats, m)?)?;
    m.add_function(wrap_pyfunction!(crop_latex_repetitions_py, m)?)?;
    m.add_function(wrap_pyfunction!(verify_md_preview_equivalent_py, m)?)?;
    m.add_function(wrap_pyfunction!(verify_md_structural_py, m)?)?;
    m.add_function(wrap_pyfunction!(cmark_gfm_verify_py, m)?)?;
    m.add_function(wrap_pyfunction!(format_surgical_checked_py, m)?)?;
    m.add_function(wrap_pyfunction!(phase_a_policy_py, m)?)?;

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
