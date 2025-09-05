//! PyO3 bindings for noise-based markdown quality metrics

mod noise_metrics;

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use noise_metrics::{
    score_markdown_file_internal,
    score_markdown_directory_internal,
    score_markdown_file_detailed_internal,
    score_markdown_directory_detailed_internal,
};

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

/// Detailed score for a single file: returns a Python tuple of all raw and derived metrics
#[pyfunction]
fn score_markdown_file_detailed(py: Python<'_>, path: &str) -> PyResult<Py<PyTuple>> {
    let (
        score, latin_pct, table_ratio, poly_ratio,
        len_greek, total_words,
        v_pen, c_pen, bad_dbl, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
        v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_pen,
        flags
    ) = score_markdown_file_detailed_internal(std::path::Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let tup = PyTuple::new(
        py,
        vec![
            score.into_py(py), latin_pct.into_py(py), table_ratio.into_py(py), poly_ratio.into_py(py),
            (len_greek as u128).into_py(py), (total_words as u128).into_py(py),
            (v_pen as u128).into_py(py), (c_pen as u128).into_py(py), (bad_dbl as u128).into_py(py), (misplaced_sigma as u128).into_py(py), (invalid_bigram as u128).into_py(py), (long_word_count as u128).into_py(py), (longest_word as u128).into_py(py), (short_word_count as u128).into_py(py), (max_run as u128).into_py(py),
            v_rate.into_py(py), c_rate.into_py(py), d_rate.into_py(py), sigma_end_rate.into_py(py), bigram_rate.into_py(py), long_word_rate.into_py(py), short_ratio.into_py(py), short_pen.into_py(py),
            flags.into_py(py),
        ],
    );
    Ok(tup.into())
}

/// Detailed scores for directory: returns a list of Python tuples with path followed by all metrics
#[pyfunction]
fn score_markdown_directory_detailed(py: Python<'_>, input_dir: &str, n_threads: Option<usize>) -> PyResult<Vec<Py<PyTuple>>> {
    let rows = score_markdown_directory_detailed_internal(std::path::Path::new(input_dir), n_threads)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut out: Vec<Py<PyTuple>> = Vec::with_capacity(rows.len());
    for (
        path, score, latin_pct, table_ratio, poly_ratio,
        len_greek, total_words,
        v_pen, c_pen, bad_dbl, misplaced_sigma, invalid_bigram, long_word_count, longest_word, short_word_count, max_run,
        v_rate, c_rate, d_rate, sigma_end_rate, bigram_rate, long_word_rate, short_ratio, short_pen,
        flags
    ) in rows.into_iter() {
        let tup = PyTuple::new(
            py,
            vec![
                path.into_py(py),
                score.into_py(py), latin_pct.into_py(py), table_ratio.into_py(py), poly_ratio.into_py(py),
                (len_greek as u128).into_py(py), (total_words as u128).into_py(py),
                (v_pen as u128).into_py(py), (c_pen as u128).into_py(py), (bad_dbl as u128).into_py(py), (misplaced_sigma as u128).into_py(py), (invalid_bigram as u128).into_py(py), (long_word_count as u128).into_py(py), (longest_word as u128).into_py(py), (short_word_count as u128).into_py(py), (max_run as u128).into_py(py),
                v_rate.into_py(py), c_rate.into_py(py), d_rate.into_py(py), sigma_end_rate.into_py(py), bigram_rate.into_py(py), long_word_rate.into_py(py), short_ratio.into_py(py), short_pen.into_py(py),
                flags.into_py(py),
            ],
        );
        out.push(tup.into());
    }
    Ok(out)
}

#[pymodule]
fn glossapi_rs_noise(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_markdown_file, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_file_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory_detailed, m)?)?;
    Ok(())
}
