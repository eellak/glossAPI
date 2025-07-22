use pyo3::prelude::*;
use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    // Regular expressions for table detection
    static ref TABLE_SEPARATOR_REGEX: Regex = Regex::new(r"^[\s]*\|[\s]*[-:]+[\s]*\|").unwrap();
    static ref TABLE_ROW_REGEX: Regex = Regex::new(r"^[\s]*\|.*\|[\s]*$").unwrap();
    // Regex to identify a potential table header (similar to a row, but used contextually)
    // For simplicity, we assume a header looks like a normal table row.
    // More sophisticated header detection might be needed if headers can be multi-line or have distinct patterns.
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct TableIssue {
    #[pyo3(get, set)]
    pub line_number: usize, // 1-based line number of the primary issue trigger (e.g., separator)
    #[pyo3(get, set)]
    pub description: String,
    #[pyo3(get, set)]
    pub expected_columns: Option<usize>,
    #[pyo3(get, set)]
    pub found_columns: Option<usize>,
    #[pyo3(get, set)]
    pub start_line: usize, // 0-indexed start line of the entire table block
    #[pyo3(get, set)]
    pub end_line: usize,   // 0-indexed end line of the entire table block (inclusive)
}

// New struct for richer return type from core logic
#[derive(Debug, Clone)]
pub struct TableScan {
    pub total_tables: usize,
    pub issues: Vec<Py<TableIssue>>,
}

#[allow(non_local_definitions)]
#[pymethods]
impl TableIssue {
    #[new]
    #[pyo3(signature = (line_number, description, expected_columns, found_columns, start_line, end_line))]
    pub fn new(
        line_number: usize, 
        description: String, 
        expected_columns: Option<usize>, 
        found_columns: Option<usize>,
        start_line: usize,
        end_line: usize
    ) -> Self {
        TableIssue { line_number, description, expected_columns, found_columns, start_line, end_line }
    }

    fn __repr__(&self) -> String {
        format!(
            "TableIssue(line: {}, desc: '{}', expected: {:?}, found: {:?}, start_idx: {}, end_idx: {})",
            self.line_number, self.description, self.expected_columns, self.found_columns, self.start_line, self.end_line
        )
    }
}

/// Core function to detect malformed tables in markdown text
pub fn core_detect_malformed_tables(py: Python, markdown_text: &str) -> PyResult<TableScan> {
    let mut issues: Vec<Py<TableIssue>> = Vec::new();
    let mut total_tables_count: usize = 0;
    let lines: Vec<&str> = markdown_text.lines().collect();
    
    if lines.is_empty() {
        return Ok(TableScan { total_tables: 0, issues });
    }

    let mut i = 0; // Current line index (0-based)
    while i < lines.len() {
        let line_content = lines[i];
        if TABLE_SEPARATOR_REGEX.is_match(line_content) {
            total_tables_count += 1;
            
            let current_separator_line_idx = i; // 0-based index of the separator line

            // Determine table boundaries
            let mut table_start_line_idx = current_separator_line_idx;
            if current_separator_line_idx > 0 && TABLE_ROW_REGEX.is_match(lines[current_separator_line_idx - 1]) {
                // Header exists, so table started with the header
                table_start_line_idx = current_separator_line_idx - 1;
                // Potentially scan further up if multi-line headers were supported
            } else {
                // No conventional header row right above separator, table effectively starts at separator for boundary purposes
                // Or, if a prior line is a table row but not immediately preceding (e.g. blank line between header and separator),
                // this logic would need adjustment. For now, simple adjacency.
                // Consider if a separator *must* have a header to be part of a "malformed" table issue report.
                // If a separator alone is an issue, its start and end is itself.
                // For now, if no header, the table effectively starts at the separator line for error reporting.
            }

            let mut table_end_line_idx = current_separator_line_idx;
            let mut next_line_after_separator_idx = current_separator_line_idx + 1;
            while next_line_after_separator_idx < lines.len() && TABLE_ROW_REGEX.is_match(lines[next_line_after_separator_idx]) {
                table_end_line_idx = next_line_after_separator_idx;
                next_line_after_separator_idx += 1;
            }
            // At this point, `table_end_line_idx` is the 0-based index of the last row of the current table.
            // If there were no rows after separator, `table_end_line_idx` remains `current_separator_line_idx`.

            // Issue detection logic (similar to before, but now we pass table_start_line_idx and table_end_line_idx)
            let separator_columns = count_table_columns(line_content);
            
            // Check for header mismatch
            if current_separator_line_idx > 0 && TABLE_ROW_REGEX.is_match(lines[current_separator_line_idx - 1]) { // Header exists
                let header_content = lines[current_separator_line_idx - 1];
                let header_columns = count_table_columns(header_content);
                if header_columns != separator_columns {
                    let issue = Py::new(py, TableIssue::new(
                        current_separator_line_idx + 1, // 1-based line number for separator
                        "Table header and separator column count mismatch".to_string(),
                        Some(header_columns),
                        Some(separator_columns),
                        table_start_line_idx, // 0-indexed
                        table_end_line_idx    // 0-indexed
                    ))?;
                    issues.push(issue);
                }
            } else { // No header row immediately above
                let issue = Py::new(py, TableIssue::new(
                    current_separator_line_idx + 1, // 1-based line number for separator
                    "Table separator without header row".to_string(),
                    None, 
                    Some(separator_columns),
                    table_start_line_idx, // If no header, this would be current_separator_line_idx
                    table_end_line_idx
                ))?;
                issues.push(issue);
            }
            
            // Check table body rows for column consistency against the separator
            let mut body_row_idx = current_separator_line_idx + 1;
            while body_row_idx <= table_end_line_idx { // Iterate through identified body rows
                if body_row_idx < lines.len() { // Ensure we are within bounds (redundant if table_end_line_idx is correct)
                    let row_content = lines[body_row_idx];
                     if TABLE_ROW_REGEX.is_match(row_content) { // Double check, though loop condition should suffice
                        let row_columns = count_table_columns(row_content);
                        if separator_columns > 0 && row_columns != separator_columns { // Only issue if separator had columns
                            let issue = Py::new(py, TableIssue::new(
                                body_row_idx + 1, // 1-based line number for current row
                                "Table row has inconsistent column count".to_string(),
                                Some(separator_columns),
                                Some(row_columns),
                                table_start_line_idx, // The entire table block this row belongs to
                                table_end_line_idx
                            ))?;
                            issues.push(issue);
                        }
                    }
                }
                body_row_idx += 1;
            }
            
            i = table_end_line_idx; // Advance main loop index `i` to the end of the processed table.
                                    // The outer loop will increment `i` by 1, so it starts checking after this table.
        }
        i += 1;
    }
    
    Ok(TableScan { total_tables: total_tables_count, issues })
}

/// Helper function to count columns in a table row
fn count_table_columns(row: &str) -> usize {
    let trimmed = row.trim();
    if trimmed.starts_with('|') && trimmed.ends_with('|') {
        if trimmed.len() <= 1 { return 0; } // Handle degenerate cases like "|" or "||" -> 0 or 1 col respectively
        let inner = &trimmed[1..trimmed.len()-1]; // Content between the outer pipes
        return inner.matches('|').count() + 1;
    }
    0 // Not a valid table row structure for column counting this way
}

/// Python-exposed function for table analysis on a single string
// Converts TableScan to tuple for Python
#[pyfunction]
pub fn analyze_tables_in_string(py: Python, markdown_text: &str) -> PyResult<(usize, Vec<Py<TableIssue>>)> {
    let scan_result = core_detect_malformed_tables(py, markdown_text)?;
    Ok((scan_result.total_tables, scan_result.issues))
}

/// Process a single file for table analysis - intended for use by directory_processor
// Returns TableScan directly for internal Rust use.
pub fn analyze_table_file_op(py: Python, content: &str) -> PyResult<TableScan> {
    core_detect_malformed_tables(py, content)
}
