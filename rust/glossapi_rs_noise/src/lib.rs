//! PyO3 bindings for noise-based markdown quality metrics

mod noise_metrics;

use noise_metrics::{
    annotate_numeric_debug_page_internal, evaluate_page_character_noise_internal,
    export_numeric_match_debug_pages_internal, export_ocr_match_debug_pages_internal,
    export_token_category_debug_pages_internal, find_hybrid_repeat_spans_internal,
    find_labeled_shared_repeat_spans_internal, find_numeric_debug_page_spans_internal,
    find_word_repeat_spans_internal, match_token_category_debug_text_internal,
    score_markdown_directory_detailed_internal, score_markdown_directory_internal,
    score_markdown_directory_ocr_profile_internal, score_markdown_file_detailed_internal,
    score_markdown_file_internal,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;

/// Compute the badness score for a single markdown file.
/// Returns the numeric score as `float`.
#[pyfunction]
fn score_markdown_file(path: &str) -> PyResult<f64> {
    score_markdown_file_internal(std::path::Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute badness scores for all `.md` files under `input_dir` in parallel.
/// The result is a list of `(file_path, score, latin_percentage)` tuples.
#[pyfunction]
fn score_markdown_directory(
    input_dir: &str,
    n_threads: Option<usize>,
) -> PyResult<Vec<(String, f64, f64)>> {
    score_markdown_directory_internal(std::path::Path::new(input_dir), n_threads)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Detailed score for a single file: returns a Python tuple of all raw and derived metrics
#[pyfunction]
fn score_markdown_file_detailed(py: Python<'_>, path: &str) -> PyResult<Py<PyTuple>> {
    let (
        score,
        latin_pct,
        table_ratio,
        poly_ratio,
        len_greek,
        total_words,
        v_pen,
        c_pen,
        bad_dbl,
        misplaced_sigma,
        invalid_bigram,
        long_word_count,
        longest_word,
        short_word_count,
        max_run,
        v_rate,
        c_rate,
        d_rate,
        sigma_end_rate,
        bigram_rate,
        long_word_rate,
        short_ratio,
        short_pen,
        flags,
    ) = score_markdown_file_detailed_internal(std::path::Path::new(path))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let tup = PyTuple::new(
        py,
        vec![
            score.into_py(py),
            latin_pct.into_py(py),
            table_ratio.into_py(py),
            poly_ratio.into_py(py),
            (len_greek as u128).into_py(py),
            (total_words as u128).into_py(py),
            (v_pen as u128).into_py(py),
            (c_pen as u128).into_py(py),
            (bad_dbl as u128).into_py(py),
            (misplaced_sigma as u128).into_py(py),
            (invalid_bigram as u128).into_py(py),
            (long_word_count as u128).into_py(py),
            (longest_word as u128).into_py(py),
            (short_word_count as u128).into_py(py),
            (max_run as u128).into_py(py),
            v_rate.into_py(py),
            c_rate.into_py(py),
            d_rate.into_py(py),
            sigma_end_rate.into_py(py),
            bigram_rate.into_py(py),
            long_word_rate.into_py(py),
            short_ratio.into_py(py),
            short_pen.into_py(py),
            flags.into_py(py),
        ],
    );
    Ok(tup.into())
}

/// Detailed scores for directory: returns a list of Python tuples with path followed by all metrics
#[pyfunction]
fn score_markdown_directory_detailed(
    py: Python<'_>,
    input_dir: &str,
    n_threads: Option<usize>,
) -> PyResult<Vec<Py<PyTuple>>> {
    let rows =
        score_markdown_directory_detailed_internal(std::path::Path::new(input_dir), n_threads)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    let mut out: Vec<Py<PyTuple>> = Vec::with_capacity(rows.len());
    for (
        path,
        score,
        latin_pct,
        table_ratio,
        poly_ratio,
        len_greek,
        total_words,
        v_pen,
        c_pen,
        bad_dbl,
        misplaced_sigma,
        invalid_bigram,
        long_word_count,
        longest_word,
        short_word_count,
        max_run,
        v_rate,
        c_rate,
        d_rate,
        sigma_end_rate,
        bigram_rate,
        long_word_rate,
        short_ratio,
        short_pen,
        flags,
    ) in rows.into_iter()
    {
        let tup = PyTuple::new(
            py,
            vec![
                path.into_py(py),
                score.into_py(py),
                latin_pct.into_py(py),
                table_ratio.into_py(py),
                poly_ratio.into_py(py),
                (len_greek as u128).into_py(py),
                (total_words as u128).into_py(py),
                (v_pen as u128).into_py(py),
                (c_pen as u128).into_py(py),
                (bad_dbl as u128).into_py(py),
                (misplaced_sigma as u128).into_py(py),
                (invalid_bigram as u128).into_py(py),
                (long_word_count as u128).into_py(py),
                (longest_word as u128).into_py(py),
                (short_word_count as u128).into_py(py),
                (max_run as u128).into_py(py),
                v_rate.into_py(py),
                c_rate.into_py(py),
                d_rate.into_py(py),
                sigma_end_rate.into_py(py),
                bigram_rate.into_py(py),
                long_word_rate.into_py(py),
                short_ratio.into_py(py),
                short_pen.into_py(py),
                flags.into_py(py),
            ],
        );
        out.push(tup.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_dir, n_threads=None, min_repeat_run=6))]
fn score_markdown_directory_ocr_profile(
    py: Python<'_>,
    input_dir: &str,
    n_threads: Option<usize>,
    min_repeat_run: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = score_markdown_directory_ocr_profile_internal(
        std::path::Path::new(input_dir),
        n_threads,
        min_repeat_run,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(rows.len());
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("path", row.path)?;
        item.set_item("percentage_greek", row.percentage_greek)?;
        item.set_item("latin_percentage", row.latin_percentage)?;
        item.set_item("polytonic_ratio", row.polytonic_ratio)?;
        item.set_item("non_whitespace_chars", row.non_whitespace_chars)?;
        item.set_item("greek_char_count", row.greek_char_count)?;
        item.set_item("latin_char_count", row.latin_char_count)?;
        item.set_item("ocr_repeat_phrase_run_max", row.ocr_repeat_phrase_run_max)?;
        item.set_item("ocr_repeat_line_run_max", row.ocr_repeat_line_run_max)?;
        item.set_item(
            "ocr_repeat_suspicious_line_count",
            row.ocr_repeat_suspicious_line_count,
        )?;
        item.set_item(
            "ocr_repeat_suspicious_line_ratio",
            row.ocr_repeat_suspicious_line_ratio,
        )?;
        item.set_item("ocr_noise_suspect", row.ocr_noise_suspect)?;
        item.set_item("ocr_noise_flags", row.ocr_noise_flags)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, n_threads=None, min_repeat_run=6, max_pages=None, sample_seed=0))]
fn export_ocr_match_debug_pages(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    n_threads: Option<usize>,
    min_repeat_run: u64,
    max_pages: Option<usize>,
    sample_seed: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = export_ocr_match_debug_pages_internal(
        std::path::Path::new(input_dir),
        std::path::Path::new(output_dir),
        n_threads,
        min_repeat_run,
        max_pages,
        sample_seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(rows.len());
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("source_path", row.source_path)?;
        item.set_item("output_path", row.output_path)?;
        item.set_item("source_stem", row.source_stem)?;
        item.set_item("base_stem", row.base_stem)?;
        item.set_item("page_number", row.page_number)?;
        item.set_item("page_index_in_file", row.page_index_in_file)?;
        item.set_item("match_types", row.match_types)?;
        item.set_item("match_count", row.match_count)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, n_threads=None, min_progress_steps=10, min_repeat_steps=8, min_same_digit_steps=10, max_pages=None, sample_seed=0))]
fn export_numeric_match_debug_pages(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    n_threads: Option<usize>,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
    max_pages: Option<usize>,
    sample_seed: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = export_numeric_match_debug_pages_internal(
        std::path::Path::new(input_dir),
        std::path::Path::new(output_dir),
        n_threads,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
        max_pages,
        sample_seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(rows.len());
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("source_path", row.source_path)?;
        item.set_item("output_path", row.output_path)?;
        item.set_item("source_stem", row.source_stem)?;
        item.set_item("base_stem", row.base_stem)?;
        item.set_item("page_number", row.page_number)?;
        item.set_item("page_index_in_file", row.page_index_in_file)?;
        item.set_item("match_types", row.match_types)?;
        item.set_item("match_count", row.match_count)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, category_specs_path, n_threads=None, max_pages=None, sample_seed=0, synthetic_page_target_chars=4000, synthetic_page_min_header_chars=1200, synthetic_page_hard_max_chars=6000))]
fn export_token_category_debug_pages(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    category_specs_path: &str,
    n_threads: Option<usize>,
    max_pages: Option<usize>,
    sample_seed: u64,
    synthetic_page_target_chars: usize,
    synthetic_page_min_header_chars: usize,
    synthetic_page_hard_max_chars: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = export_token_category_debug_pages_internal(
        std::path::Path::new(input_dir),
        std::path::Path::new(output_dir),
        std::path::Path::new(category_specs_path),
        n_threads,
        max_pages,
        sample_seed,
        synthetic_page_target_chars,
        synthetic_page_min_header_chars,
        synthetic_page_hard_max_chars,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(rows.len());
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("source_path", row.source_path)?;
        item.set_item("output_path", row.output_path)?;
        item.set_item("source_stem", row.source_stem)?;
        item.set_item("base_stem", row.base_stem)?;
        item.set_item("page_kind", row.page_kind)?;
        item.set_item("page_number", row.page_number)?;
        item.set_item("page_index_in_file", row.page_index_in_file)?;
        item.set_item("page_char_count", row.page_char_count)?;
        item.set_item("match_categories", row.match_categories)?;
        item.set_item("match_pattern_families", row.match_pattern_families)?;
        item.set_item("match_count", row.match_count)?;
        item.set_item("page_text", row.page_text)?;
        item.set_item("matches_json", row.matches_json)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, output_dir, category_specs_path, source_path, source_stem, base_stem, start_page=1, synthetic_page_target_chars=4000, synthetic_page_min_header_chars=1200, synthetic_page_hard_max_chars=6000))]
fn match_token_category_debug_text(
    py: Python<'_>,
    text: &str,
    output_dir: &str,
    category_specs_path: &str,
    source_path: &str,
    source_stem: &str,
    base_stem: &str,
    start_page: u64,
    synthetic_page_target_chars: usize,
    synthetic_page_min_header_chars: usize,
    synthetic_page_hard_max_chars: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let rows = match_token_category_debug_text_internal(
        std::path::Path::new(output_dir),
        std::path::Path::new(category_specs_path),
        source_path,
        source_stem,
        base_stem,
        start_page,
        text,
        synthetic_page_target_chars,
        synthetic_page_min_header_chars,
        synthetic_page_hard_max_chars,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(rows.len());
    for row in rows {
        let item = PyDict::new(py);
        item.set_item("source_path", row.source_path)?;
        item.set_item("output_path", row.output_path)?;
        item.set_item("source_stem", row.source_stem)?;
        item.set_item("base_stem", row.base_stem)?;
        item.set_item("page_kind", row.page_kind)?;
        item.set_item("page_number", row.page_number)?;
        item.set_item("page_index_in_file", row.page_index_in_file)?;
        item.set_item("page_char_count", row.page_char_count)?;
        item.set_item("match_categories", row.match_categories)?;
        item.set_item("match_pattern_families", row.match_pattern_families)?;
        item.set_item("match_count", row.match_count)?;
        item.set_item("page_text", row.page_text)?;
        item.set_item("matches_json", row.matches_json)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (page, min_progress_steps=10, min_repeat_steps=8, min_same_digit_steps=10))]
fn annotate_numeric_debug_page(
    py: Python<'_>,
    page: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> PyResult<Option<Py<PyDict>>> {
    let Some((annotated_page, match_types, match_count)) = annotate_numeric_debug_page_internal(
        page,
        min_progress_steps,
        min_repeat_steps,
        min_same_digit_steps,
    ) else {
        return Ok(None);
    };

    let item = PyDict::new(py);
    item.set_item("annotated_page", annotated_page)?;
    item.set_item("match_types", match_types)?;
    item.set_item("match_count", match_count)?;
    Ok(Some(item.into()))
}

#[pyfunction]
#[pyo3(signature = (page, min_progress_steps=10, min_repeat_steps=8, min_same_digit_steps=10))]
fn find_numeric_debug_page_spans(
    py: Python<'_>,
    page: &str,
    min_progress_steps: u64,
    min_repeat_steps: u64,
    min_same_digit_steps: u64,
) -> PyResult<Vec<Py<PyDict>>> {
    let spans = py.allow_threads(|| {
        find_numeric_debug_page_spans_internal(
            page,
            min_progress_steps,
            min_repeat_steps,
            min_same_digit_steps,
        )
    });
    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(spans.len());
    for span in spans {
        let item = PyDict::new(py);
        item.set_item("start", span.start)?;
        item.set_item("end", span.end)?;
        item.set_item("match_type", span.match_type)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (normalized_text, rep_threshold=4, min_period=3, window=96))]
fn find_word_repeat_spans(
    py: Python<'_>,
    normalized_text: &str,
    rep_threshold: usize,
    min_period: usize,
    window: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let spans = py.allow_threads(|| {
        find_word_repeat_spans_internal(normalized_text, rep_threshold, min_period, window)
    });
    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(spans.len());
    for span in spans {
        let item = PyDict::new(py);
        item.set_item("start", span.start)?;
        item.set_item("end", span.end)?;
        item.set_item("period", span.period)?;
        item.set_item("repetitions", span.repetitions)?;
        item.set_item("tail_chars", span.tail_chars)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
fn find_hybrid_repeat_spans(py: Python<'_>, analysis_text: &str) -> PyResult<Vec<Py<PyDict>>> {
    let spans = py.allow_threads(|| find_hybrid_repeat_spans_internal(analysis_text));
    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(spans.len());
    for span in spans {
        let item = PyDict::new(py);
        item.set_item("start", span.start)?;
        item.set_item("end", span.end)?;
        item.set_item("match_types", vec!["hybrid_repeat"])?;
        item.set_item("category", "hybrid")?;
        item.set_item("kind", span.kind)?;
        item.set_item("item_count", span.item_count)?;
        if let Some(cycle_len) = span.cycle_len {
            item.set_item("cycle_len", cycle_len)?;
        }
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (analysis_text, rep_threshold=4, min_period=3, window=96))]
fn find_labeled_shared_repeat_spans(
    py: Python<'_>,
    analysis_text: &str,
    rep_threshold: usize,
    min_period: usize,
    window: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let spans = py.allow_threads(|| {
        find_labeled_shared_repeat_spans_internal(analysis_text, rep_threshold, min_period, window)
    });
    let mut out: Vec<Py<PyDict>> = Vec::with_capacity(spans.len());
    for span in spans {
        let item = PyDict::new(py);
        item.set_item("start", span.start)?;
        item.set_item("end", span.end)?;
        item.set_item("period", span.period)?;
        item.set_item("repetitions", span.repetitions)?;
        item.set_item("tail_chars", span.tail_chars)?;
        item.set_item("match_type", span.match_type)?;
        out.push(item.into());
    }
    Ok(out)
}

#[pyfunction]
fn evaluate_page_character_noise(py: Python<'_>, page: &str) -> PyResult<Py<PyDict>> {
    let metrics = py.allow_threads(|| evaluate_page_character_noise_internal(page));
    let item = PyDict::new(py);
    item.set_item("total_chars", metrics.total_chars)?;
    item.set_item("bad_char_count", metrics.bad_char_count)?;
    item.set_item("bad_char_ratio", metrics.bad_char_ratio)?;
    item.set_item("control_count", metrics.control_count)?;
    item.set_item("private_use_count", metrics.private_use_count)?;
    item.set_item("cjk_count", metrics.cjk_count)?;
    item.set_item("replacement_count", metrics.replacement_count)?;
    Ok(item.into())
}

#[pymodule]
fn glossapi_rs_noise(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_markdown_file, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_file_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory_detailed, m)?)?;
    m.add_function(wrap_pyfunction!(score_markdown_directory_ocr_profile, m)?)?;
    m.add_function(wrap_pyfunction!(export_ocr_match_debug_pages, m)?)?;
    m.add_function(wrap_pyfunction!(export_numeric_match_debug_pages, m)?)?;
    m.add_function(wrap_pyfunction!(export_token_category_debug_pages, m)?)?;
    m.add_function(wrap_pyfunction!(match_token_category_debug_text, m)?)?;
    m.add_function(wrap_pyfunction!(annotate_numeric_debug_page, m)?)?;
    m.add_function(wrap_pyfunction!(find_numeric_debug_page_spans, m)?)?;
    m.add_function(wrap_pyfunction!(find_word_repeat_spans, m)?)?;
    m.add_function(wrap_pyfunction!(find_hybrid_repeat_spans, m)?)?;
    m.add_function(wrap_pyfunction!(find_labeled_shared_repeat_spans, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_page_character_noise, m)?)?;
    Ok(())
}
