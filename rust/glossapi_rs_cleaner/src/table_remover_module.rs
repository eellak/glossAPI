// check_badness/extraction_metrics_rs/src/table_remover_module.rs
use serde::Deserialize;
use std::path::PathBuf;
use std::collections::HashMap;

// Represents a range of lines to be removed (0-indexed, inclusive end)
#[derive(Debug, Clone, Deserialize)]
pub struct LineRange {
    pub start_line: usize,
    pub end_line: usize,
}

// Holds information about tables to be removed, read from the detailed CSV.
// Used when parsing the CSV.
#[derive(Debug, Deserialize)]
pub struct TableLocationInfo {
    #[serde(alias = "file_path")] // Keep original for clarity, alias for robustness
    pub file_path: String,
    #[serde(alias = "table_start_line")]
    pub start_line: usize, // 0-indexed
    #[serde(alias = "table_end_line")]
    pub end_line: usize,   // 0-indexed, inclusive
}

/// Processes the content of a single Markdown file to remove specified table sections.
///
/// Args:
/// * `file_content`: The string content of the Markdown file.
/// * `table_locations_for_file`: A vector of `LineRange` structs indicating which
///   parts of the file (tables) to remove. These should be sorted by start_line
///   in ascending order if applying iteratively, or handled carefully if not.
///   For safety, it's often best to process removals from bottom-to-top
///   if line numbers are based on the original file.
///
/// Returns:
/// * A `String` with the specified tables replaced by a placeholder.
pub fn remove_tables_from_content(
    file_content: &str,
    table_locations_for_file: &[LineRange],
) -> String {
    if table_locations_for_file.is_empty() {
        return file_content.to_string();
    }

    let original_lines: Vec<&str> = file_content.lines().collect();
    if original_lines.is_empty() { // All lines were part of ranges, or original was empty.
        // If original content was just newlines, and all are to be removed.
        let all_original_lines_are_whitespace = original_lines.iter().all(|l| l.trim().is_empty());
        if !table_locations_for_file.is_empty() && all_original_lines_are_whitespace && original_lines.len() <= table_locations_for_file.iter().map(|r| r.end_line - r.start_line + 1).sum::<usize>() {
             return "<!-- table-removed -->".to_string(); // No newline if only whitespace lines were removed
        }
         // If file_content was not empty but original_lines is (e.g. file_content = "\\n\\n" and lines() makes it empty),
         // and we are removing everything, this is tricky.
         // Let's assume if original_lines is empty due to lines() behavior on whitespace-only content,
         // and ranges cover it, then placeholder without newline.
        if file_content.chars().all(char::is_whitespace) && !table_locations_for_file.is_empty() {
            // Heuristic: if original was all whitespace and we are removing parts of it (effectively all of it)
            // then just the placeholder.
             let max_end_line = table_locations_for_file.iter().map(|r| r.end_line).max().unwrap_or(0);
             if max_end_line >= original_lines.len().saturating_sub(1) && !original_lines.is_empty() { // Covers all effective lines
                return "<!-- table-removed -->".to_string();
             }
             if original_lines.is_empty() && !file_content.is_empty() { // e.g. file_content = "\\n" -> original_lines = [""]
                 // This case implies file_content was non-empty whitespace, and lines() might give one empty string.
                 // If ranges cover this (e.g. start_line: 0, end_line: 0 for a single line "\\n"),
                 // then return just the placeholder.
                 if table_locations_for_file.iter().any(|r| r.start_line == 0 && r.end_line == 0 && original_lines.len()==1) {
                    return "<!-- table-removed -->".to_string();
                 }
             }
        }
    }


    let mut lines_to_keep: Vec<String> = Vec::new();
    let mut removed_block_active = false;

    let mut line_flags = vec![false; original_lines.len()]; // true if line is part of a table to remove
    for range in table_locations_for_file {
        for i in range.start_line..=std::cmp::min(range.end_line, original_lines.len().saturating_sub(1)) {
            if i < line_flags.len() {
                line_flags[i] = true;
            }
        }
    }
    
    for i in 0..original_lines.len() {
        if line_flags[i] {
            if !removed_block_active {
                lines_to_keep.push("<!-- table-removed -->".to_string());
                removed_block_active = true;
            }
        } else {
            lines_to_keep.push(original_lines[i].to_string());
            removed_block_active = false;
        }
    }

    if lines_to_keep.is_empty() && !original_lines.is_empty() && !table_locations_for_file.is_empty() {
        // All lines were removed
        return "<!-- table-removed -->".to_string();
    } else if lines_to_keep.is_empty() && original_lines.is_empty() {
        // Original was empty, no ops
        return String::new();
    }


    let mut result = lines_to_keep.join("\n");

    // Final newline adjustment:
    // If the original content ended with a newline, the result should too (unless it's just the placeholder).
    // If the original content did NOT end with a newline, the result should not either.
    if file_content.ends_with('\n') {
        if !result.ends_with('\n') && result != "<!-- table-removed -->" {
            result.push('\n');
        }
    } else { // Original did not end with \n
        if result.ends_with('\n') {
            result.pop();
        }
    }
    
    // Special case: if result is ONLY the placeholder, it should not have a trailing newline.
    if result == "<!-- table-removed -->\n" {
        result.pop();
    }


    result
}

// Helper function to parse the detailed CSV report.
// This would typically be called by the batch processing function in directory_processor.rs
pub fn load_table_locations_from_csv(
    csv_path: &str,
) -> Result<HashMap<PathBuf, Vec<LineRange>>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(csv_path)?;
    let mut map: HashMap<PathBuf, Vec<LineRange>> = HashMap::new();

    for result in rdr.deserialize() {
        let record: TableLocationInfo = result?;
        map.entry(PathBuf::from(record.file_path))
            .or_default()
            .push(LineRange {
                start_line: record.start_line,
                end_line: record.end_line,
            });
    }
    // Sort line ranges for each file to make processing easier if needed (e.g., bottom-up)
    // Although the current `remove_tables_from_content` handles overlaps without sorting.
    for ranges in map.values_mut() {
        ranges.sort_by_key(|r| r.start_line);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_single_table() {
        let content = "line1\nline2 (table start)\nline3 (table mid)\nline4 (table end)\nline5";
        let ranges = vec![LineRange { start_line: 1, end_line: 3 }];
        let expected = "line1\n<!-- table-removed -->\nline5";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_remove_multiple_tables() {
        let content = "line1\nt1s\nt1e\nline2\nt2s\nt2e\nline3";
        let ranges = vec![
            LineRange { start_line: 1, end_line: 2 },
            LineRange { start_line: 4, end_line: 5 },
        ];
        let expected = "line1\n<!-- table-removed -->\nline2\n<!-- table-removed -->\nline3";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_remove_overlapping_tables() {
        let content = "line1\nt1s\nt1m\nt1e_t2s\nt2m\nt2e\nline2";
        let ranges = vec![
            LineRange { start_line: 1, end_line: 3 }, 
            LineRange { start_line: 3, end_line: 5 }, 
        ];
        let expected = "line1\n<!-- table-removed -->\nline2";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_no_tables_to_remove() {
        let content = "line1\nline2\nline3";
        let ranges = vec![];
        assert_eq!(remove_tables_from_content(content, &ranges), content);
    }

    #[test]
    fn test_remove_table_at_start() {
        let content = "t1s\nt1e\nline2\nline3";
        let ranges = vec![LineRange { start_line: 0, end_line: 1 }];
        let expected = "<!-- table-removed -->\nline2\nline3";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_remove_table_at_end() {
        let content = "line1\nline2\nt1s\nt1e"; // Ends with newline
        let ranges = vec![LineRange { start_line: 2, end_line: 3 }];
        let expected = "line1\nline2\n<!-- table-removed -->"; // Placeholder does not get extra newline if it's the last thing
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }
    
    #[test]
    fn test_remove_all_lines() {
        let content = "line1\nline2"; // Ends with newline
        let ranges = vec![LineRange { start_line: 0, end_line: 1 }];
        let expected = "<!-- table-removed -->"; // Just the placeholder, no newline
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_remove_all_lines_no_eof_newline() {
        let content = "line1\nline2NOEOL"; // No EOF newline
        let ranges = vec![LineRange { start_line: 0, end_line: 1 }];
        let expected = "<!-- table-removed -->";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }


     #[test]
    fn test_empty_content() {
        let content = "";
        let ranges = vec![];
        assert_eq!(remove_tables_from_content(content, &ranges), "");
    }
    
    #[test]
    fn test_empty_content_with_remove_op() {
        let content = "";
        let ranges = vec![LineRange { start_line: 0, end_line: 0}]; // Range is technically out of bounds but benign
        assert_eq!(remove_tables_from_content(content, &ranges), ""); // Should be no-op
    }

    #[test]
    fn test_content_with_only_newline() {
        let content = "\n";
        let ranges = vec![];
        assert_eq!(remove_tables_from_content(content, &ranges), "\n");
    }

    #[test]
    fn test_remove_single_actual_newline_char_content() {
        let content = "\n"; // file_content.lines() produces [""]
        let ranges = vec![LineRange { start_line: 0, end_line: 0 }];
        let expected = "<!-- table-removed -->"; 
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }
    
    #[test]
    fn test_remove_multiple_actual_newline_chars_content() {
        let content = "\n\n"; // file_content.lines() produces ["", ""]
        let ranges = vec![LineRange { start_line: 0, end_line: 1 }]; // remove both empty strings
        let expected = "<!-- table-removed -->";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }


    #[test]
    fn test_remove_table_respect_no_final_newline() {
        let content = "line1\nline2 (table start)\nline3 (table mid)\nline4 (table end)\nline5NOEOL";
        let ranges = vec![LineRange { start_line: 1, end_line: 3 }];
        let expected = "line1\n<!-- table-removed -->\nline5NOEOL";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }

    #[test]
    fn test_remove_table_at_end_no_final_newline() {
        let content = "line1\nline2\nt1s\nt1eNOEOL";
        let ranges = vec![LineRange { start_line: 2, end_line: 3 }];
        let expected = "line1\nline2\n<!-- table-removed -->";
        assert_eq!(remove_tables_from_content(content, &ranges), expected);
    }
} 