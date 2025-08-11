//! PyO3 bindings for noise-based markdown quality metrics

mod noise_metrics;

use pyo3::prelude::*;
use noise_metrics::{score_markdown_file_internal, score_markdown_directory_internal};

/// Compute the badness score for a single markdown file.
/// Returns the numeric score as `float`.
#[pyfunction]
fn score_markdown_file(path: &str) -> PyResult<f64> {
    score_markdown_file_internal(std::path::Path::new(path)).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute badness scores for all `.md` files under `input_dir` in parallel.
/// The result is a list of `(file_path, score, latin_percentage)` tuples.
#[pyfunction]
fn score_markdown_directory(input_dir: &str, n_threads: Option<usize>) -> PyResult<Vec<(String, f64, f64)>> {
    score_markdown_directory_internal(std::path::Path::new(input_dir), n_threads).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pymodule]
fn glossapi_rs_noise(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_markdown_file, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory, m)?)?;
    Ok(())
}
