use lazy_static::lazy_static;
use pyo3::prelude::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use once_cell::sync::Lazy;
use serde::Serialize;
use aho_corasick::AhoCorasick;
use htmlentity::entity::{decode, ICodedDataTrait};
use memchr::{memchr}; // For Step 5.1
use memchr::memmem; // For optimizing comment search in strip_tags_custom

// Constants
const TEXT_MISSING_COMMENT: &str = "<!-- text-missing -->";
const TABLE_REMOVED_COMMENT: &str = "<!-- table-removed -->"; // Added for badness adjustment

lazy_static! {
    // Regular expressions for detection (compiled once) - Most are now unused
    // pub static ref GLYPH_TAG_REGEX_RAW: Regex = Regex::new(r"(?:^|\s)glyph<c=\d+,font=/[^>]+>(?:\s|$)").unwrap();
    // pub static ref GLYPH_TAG_REGEX_HTML: Regex = Regex::new(r"(?:^|\s)glyph&lt;c=\d+,font=/[^>]+&gt;(?:\s|$)").unwrap();
    // pub static ref ANY_TAG_REGEX: Regex = Regex::new(r"(?:^|\s)<[^>]*>(?:\s|$)").unwrap();
    // pub static ref IS_COMMENT_REGEX: Regex = Regex::new(r"^<!--").unwrap(); // Replaced by direct byte check
    // pub static ref HTML_ENTITY_REGEX: Regex = Regex::new(r"&[a-zA-Z]+;|&#\d+;|&lt;|&gt;|&amp;").unwrap(); // Replaced by htmlentity crate
    
    // Regex for HTML comments (captures the whole comment) - STILL USED
    pub static ref COMMENT_REGEX: Regex = Regex::new(r"<!--.*?-->").unwrap();
    
    // Regex for HTML/XML tags (for cleaning, non-comment tags) - Replaced by strip_tags_custom
    // pub static ref ANY_TAG_CLEANING_REGEX: Regex = Regex::new(r"<[^>]*>").unwrap();

    // Central HashMap for character scripts
    pub static ref SCRIPT_SETS: HashMap<String, HashSet<char>> = {
        let mut map = HashMap::new();
        
        map.insert("latin".to_string(), "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect());
        
        let mut greek_chars = HashSet::new();
        for code in 0x0370..0x03E2 { if let Some(c) = std::char::from_u32(code) { greek_chars.insert(c); }}
        for code in 0x03F0..0x0400 { if let Some(c) = std::char::from_u32(code) { greek_chars.insert(c); }}
        let accented_greek = "άέήίόύώΆΈΉΊΌΎΏϊϋΪΫΐΰ";
        for c in accented_greek.chars() { greek_chars.insert(c); }
        greek_chars.insert('\u{00B5}'); // Add MICRO SIGN
        map.insert("greek".to_string(), greek_chars);
        
        let french_specific = "àâçéèêëîïôùûüÿæœÀÂÇÉÈÊËÎÏÔÙÛÜŸÆŒ«»";
        map.insert("french".to_string(), french_specific.chars().collect());
        
        let spanish_specific = "áéíóúüñÁÉÍÓÚÜÑ¿¡";
        map.insert("spanish".to_string(), spanish_specific.chars().collect());
        
        let punctuation = ".,;:!?()[]{}\'\"&@#$%^*_-+=|\\<>/~`";
        map.insert("punctuation".to_string(), punctuation.chars().collect());
        
        let digits = "0123456789";
        map.insert("numbers".to_string(), digits.chars().collect());
        
        let common_symbols = "€£¥©®™°§";
        map.insert("common_symbols".to_string(), common_symbols.chars().collect());
        
        let mut unusual_chars = HashSet::new();
        for code in 0x0080..0x0100 { // Latin-1 Supplement
            if let Some(c) = std::char::from_u32(code) {
                if !french_specific.contains(c) && !spanish_specific.contains(c) && 
                   !accented_greek.contains(c) && !common_symbols.contains(c) && 
                   !punctuation.contains(c) {
                    unusual_chars.insert(c);
                }
            }
        }
        for code in 0x0100..0x0180 { // Latin Extended-A
            if let Some(c) = std::char::from_u32(code) {
                if !french_specific.contains(c) && !spanish_specific.contains(c) {
                    unusual_chars.insert(c);
                }
            }
        }
        for code in 0x0180..0x0250 { unusual_chars.extend(std::char::from_u32(code)); } // Latin Extended-B
        for code in 0x0250..0x02B0 { unusual_chars.extend(std::char::from_u32(code)); } // IPA Extensions
        for code in 0x1E00..0x1F00 { unusual_chars.extend(std::char::from_u32(code)); } // Latin Extended Additional
        for code in 0x03E2..0x03F0 { unusual_chars.extend(std::char::from_u32(code)); } // Coptic from Greek block
        for code in 0x2C80..0x2D00 { unusual_chars.extend(std::char::from_u32(code)); } // Dedicated Coptic block
        for code in 0x0400..0x0500 { unusual_chars.extend(std::char::from_u32(code)); } // Cyrillic block
        for code in 0x0500..0x0530 { unusual_chars.extend(std::char::from_u32(code)); } // Cyrillic Supplement
        map.insert("unusual".to_string(), unusual_chars);
        
        map
    };
}

// Artefact triggers for Aho-Corasick (Step 2.1)
static BAD_LINE_AC: Lazy<AhoCorasick> = Lazy::new(|| {
    AhoCorasick::new([
        "glyph<c=", 
        "glyph&lt;c=", 
        "MS-Bold-", 
        "font=/",       // Common in Docling XML-like font tags e.g. <glyph font=/NUMPTY+ImprintMTnum>1</glyph>
        "FontName="    // Common in some other PDF text extractions for font changes
    ]).unwrap()
});

// Helper function for Step 5.1: Stream-strip tags using memchr
// Takes a mutable buffer for the result, clears it, and appends to it.
// Returns count of removed non-whitespace tag characters.
fn strip_tags_custom(line: &str, result_buf: &mut String) -> usize {
    result_buf.clear();
    result_buf.reserve(line.len()); // Pre-reserve capacity
    let mut removed_non_ws_tag_chars = 0;
    let mut current_pos = 0;
    let bytes = line.as_bytes();
    let comment_closer = memmem::Finder::new(b"-->"); // Create finder for "-->"

    while current_pos < bytes.len() {
        match memchr(b'<', &bytes[current_pos..]) {
            Some(i) => {
                result_buf.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[current_pos..current_pos + i]) });
                current_pos += i;
                if bytes.get(current_pos..current_pos + 4) == Some(b"<!--") {
                    let search_start_for_comment_end = current_pos + 4;
                    // Use memmem::find for faster "-->" search
                    match comment_closer.find(&bytes[search_start_for_comment_end..]) {
                        Some(j) => { // j is the start index of "-->" within the slice
                            let comment_end_in_slice = search_start_for_comment_end + j + 3; // end of "-->"
                            result_buf.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[current_pos..comment_end_in_slice]) });
                            current_pos = comment_end_in_slice;
                        }
                        None => { 
                            result_buf.push('<');
                            current_pos += 1;
                        }
                    }
                } else { 
                    match memchr(b'>', &bytes[current_pos..]) {
                        Some(j) => {
                            let tag_content_bytes = &bytes[current_pos + 1..current_pos + j];
                            let tag_content_str = unsafe { std::str::from_utf8_unchecked(tag_content_bytes) };
                            for _char_in_tag in tag_content_str.chars().filter(|c| !c.is_whitespace()) {
                                removed_non_ws_tag_chars += 1;
                            }
                            current_pos += j + 1;
                        }
                        None => { 
                            result_buf.push('<');
                            current_pos += 1;
                        }
                    }
                }
            }
            None => { 
                result_buf.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[current_pos..]) });
                current_pos = bytes.len();
            }
        }
    }
    removed_non_ws_tag_chars // No longer returns the String, it's modified in place
}

/// Core text cleaning function - removes unwanted characters based on script sets
/// Returns a tuple: (cleaned_text, original_chars_count_for_badness, kept_chars_count_for_badness)
/// original_chars_count_for_badness: count of characters in lines not fully rejected by BAD_LINE_AC.
/// kept_chars_count_for_badness: count of characters remaining in those lines after cleaning.
pub fn core_clean_text(text: &str, allowed_chars: &HashSet<char>, unusual_chars_set: &HashSet<char>, min_chars_for_comment_override: Option<usize>) -> (String, usize, usize) {
    let min_comment_chars = min_chars_for_comment_override.unwrap_or(5);
    let mut cleaned_output_string_builder = String::new(); // Used to build the final string with newlines
    let mut original_chars_for_badness: usize = 0; // Sum of original line content lengths (excluding their newlines)
    
    // New counter for the sum of *content characters* of lines added to the output,
    // before specific placeholder penalties are applied.
    let mut sum_kept_line_content_chars: usize = 0;

    let mut inline_tmc_additions_count: usize = 0;
    let mut standalone_tmc_replacements_on_processed_lines_count: usize = 0;

    // Step 5.3: Build local bitmaps for faster char checking in the 0-1023 range.
    let mut local_allowed_bitmap: [bool; 1024] = [false; 1024];
    for &ch_allowed in allowed_chars {
        let u_val = ch_allowed as u32;
        if u_val < 1024 {
            local_allowed_bitmap[u_val as usize] = true;
        }
    }

    let mut local_unusual_bitmap: [bool; 1024] = [false; 1024];
    for &ch_unusual in unusual_chars_set {
        let u_val = ch_unusual as u32;
        if u_val < 1024 {
            local_unusual_bitmap[u_val as usize] = true;
        }
    }

    // Step 5.4: Define line-level buffers outside the loop to reuse them
    let mut processed_line_segment_buf = String::new();
    let mut current_line_removed_chars_buffer_buf = String::new();
    let mut line_after_tag_handling_buf = String::new(); // Buffer for strip_tags_custom output

    for line in text.lines() {
        // Step 2.2: Early-reject lines based on BAD_LINE_AC
        if BAD_LINE_AC.is_match(line) {
            cleaned_output_string_builder.push_str(TEXT_MISSING_COMMENT);
            cleaned_output_string_builder.push('\n');
            // These lines do not contribute to original_chars_for_badness or sum_kept_line_content_chars
            continue; 
        }

        original_chars_for_badness += line.chars().count(); // Original line length (content only)

        processed_line_segment_buf.clear();
        current_line_removed_chars_buffer_buf.clear();
        // line_after_tag_handling_buf is cleared inside strip_tags_custom

        // Step 4.1: Decode HTML entities FIRST from the original line
        let decoded_entity_data = decode(line.as_bytes());
        let line_after_entity_decoding_str = decoded_entity_data.to_string().unwrap_or_else(|_| line.to_string());

        // Step 5.1 & 5.4: Use strip_tags_custom with a reusable buffer on the DECODED line content
        let removed_from_tags_count = strip_tags_custom(&line_after_entity_decoding_str, &mut line_after_tag_handling_buf);

        // The result of tag stripping (line_after_tag_handling_buf) is now what we iterate for character filtering.
        for ch in line_after_tag_handling_buf.chars() { // Iterate the result of tag stripping
            let ch_u32 = ch as u32;
            let is_char_allowed_by_scripts; 
            let is_char_in_unusual_set;

            if ch_u32 < 1024 {
                is_char_allowed_by_scripts = local_allowed_bitmap[ch_u32 as usize];
                is_char_in_unusual_set = local_unusual_bitmap[ch_u32 as usize];
            } else {
                is_char_allowed_by_scripts = allowed_chars.contains(&ch);
                is_char_in_unusual_set = unusual_chars_set.contains(&ch);
            }

            // Condition for removal: It's in the unusual set AND it's NOT specifically allowed by current scripts_to_keep.
            if is_char_in_unusual_set && !is_char_allowed_by_scripts {
                if !ch.is_whitespace() {
                    current_line_removed_chars_buffer_buf.push(ch);
                }
            } else {
                processed_line_segment_buf.push(ch);
            }
        }
        
        let removed_chars_on_line_for_comment_decision = removed_from_tags_count + current_line_removed_chars_buffer_buf.chars().count();

        let trimmed_segment_for_comment_check = processed_line_segment_buf.trim();
        let mut line_content_to_add = String::new();

        // Check if the line (after processing up to character filtering) is exclusively an HTML comment.
        let is_exclusively_comment = if !trimmed_segment_for_comment_check.is_empty() && COMMENT_REGEX.is_match(trimmed_segment_for_comment_check) {
            if let Some(mat) = COMMENT_REGEX.find(trimmed_segment_for_comment_check) {
                mat.start() == 0 && mat.end() == trimmed_segment_for_comment_check.len()
            } else {
                false
            }
        } else {
            false
        };

        if is_exclusively_comment {
            line_content_to_add.push_str(&processed_line_segment_buf);
        } else {
            if !processed_line_segment_buf.trim().is_empty() { // P_trimmed is not empty
                if removed_chars_on_line_for_comment_decision >= min_comment_chars {
                    line_content_to_add.push_str(processed_line_segment_buf.trim_end());
                    line_content_to_add.push(' ');
                    line_content_to_add.push_str(TEXT_MISSING_COMMENT);
                    inline_tmc_additions_count += 1;
                } else {
                    line_content_to_add.push_str(&processed_line_segment_buf);
                }
            } else { // processed_line_segment_buf is empty or whitespace only
                if removed_chars_on_line_for_comment_decision >= min_comment_chars 
                   && line.chars().any(|c| !c.is_whitespace()) { // original line had content
                    line_content_to_add.push_str(TEXT_MISSING_COMMENT);
                    standalone_tmc_replacements_on_processed_lines_count += 1;
                } else {
                    line_content_to_add.push_str(&processed_line_segment_buf); 
                }
            }
        }
        
        sum_kept_line_content_chars += line_content_to_add.chars().count(); // Add length of the content part of the line
        cleaned_output_string_builder.push_str(&line_content_to_add);
        cleaned_output_string_builder.push('\n'); 
    }

    let mut final_cleaned_text = cleaned_output_string_builder;

    // Adjust final newline if original text didn't have one.
    // This affects the final string, but sum_kept_line_content_chars and original_chars_for_badness 
    // are based on line contents only, so they remain unaffected by this specific string manipulation.
    if !text.is_empty() && !text.ends_with('\n') {
        if final_cleaned_text.ends_with('\n') {
            final_cleaned_text.pop();
        }
    }

    // Calculate the total content penalty from placeholders.
    let mut total_placeholder_content_penalty = 0;

    // Penalty for TEXT_MISSING_COMMENT added inline (P_trimmed + " " + TMC)
    // The content added was P_trimmed + space + TMC. We want to keep only P_trimmed.
    // So, penalty is length of (space + TMC).
    total_placeholder_content_penalty += inline_tmc_additions_count * (TEXT_MISSING_COMMENT.len() + 1); // +1 for the space

    // Penalty for TEXT_MISSING_COMMENT that replaced an entire processed line.
    // The content added was TMC. We want to keep 0 from original for this line part.
    // So, penalty is length of TMC.
    total_placeholder_content_penalty += standalone_tmc_replacements_on_processed_lines_count * TEXT_MISSING_COMMENT.len();

    // Penalty for TABLE_REMOVED_COMMENT instances.
    // If final_cleaned_text contains lines that are exactly TABLE_REMOVED_COMMENT (and these lines were part of processed lines, not BAD_LINE_AC rejects),
    // their contribution to sum_kept_line_content_chars was TABLE_REMOVED_COMMENT.len().
    // These should be fully penalized. This relies on TABLE_REMOVED_COMMENT not being a substring of other kept content.
    // This part is tricky: how to identify which TABLE_REMOVED_COMMENT in final_cleaned_text came from processed lines vs. original content?
    // For now, let's assume that if `core_clean_text` is called, any TABLE_REMOVED_COMMENT it *outputs* that wasn't from a BAD_LINE_AC line
    // implies that the *entire line content* became TABLE_REMOVED_COMMENT.
    // The current logic of `line_content_to_add` doesn't explicitly create TABLE_REMOVED_COMMENT.
    // This comment is typically added by `table_remover_module` *before* `core_clean_text` (in Stage 4 of pipeline).
    // So, if a line *input* to core_clean_text is TABLE_REMOVED_COMMENT, and it passes BAD_LINE_AC, and isn't further modified,
    // its content length (TABLE_REMOVED_COMMENT.len()) is added to sum_kept_line_content_chars.
    // We want to penalize this entirely.
    
    // Count TABLE_REMOVED_COMMENT instances that are *stand-alone* on lines in the final output.
    // This is an approximation. A more robust way would be to flag lines that became TRC *during this function*.
    // However, this function doesn't generate TRC; it processes text that might *contain* TRC from prior steps.
    let mut num_table_removed_comments_as_full_lines_in_output = 0;
    for output_line in final_cleaned_text.lines() {
        if output_line == TABLE_REMOVED_COMMENT {
             // We need to be sure this line wasn't a BAD_LINE_AC reject originally.
             // This check is complex here. A simpler assumption is made in perform_text_analysis
             // by calculating badness based on what this function returns.
             // For the purpose of *this function's* returned `kept_chars`, if a line becomes TRC,
             // its length (TRC.len()) was added to sum_kept_line_content_chars. We penalize it.
            num_table_removed_comments_as_full_lines_in_output +=1;
        }
    }
    total_placeholder_content_penalty += num_table_removed_comments_as_full_lines_in_output * TABLE_REMOVED_COMMENT.len();
    
    let adjusted_kept_chars_for_badness = sum_kept_line_content_chars.saturating_sub(total_placeholder_content_penalty);

    (final_cleaned_text, original_chars_for_badness, adjusted_kept_chars_for_badness)
}

/// Python-exposed function to clean a single string
#[pyfunction]
pub fn clean_text(text: &str, scripts_to_keep: Vec<String>, min_chars_for_comment: Option<usize>) -> PyResult<String> {
    let mut allowed_chars = HashSet::new();
    for key in &scripts_to_keep {
        if let Some(script_set) = SCRIPT_SETS.get(key) {
            allowed_chars.extend(script_set);
        } else {
            // Optionally, log a warning if a script key is not found
            // log::warn!("Script key '{}' not found in SCRIPT_SETS", key);
        }
    }
    
    // Ensure common scripts are included even if not specified
    // Using .to_string() for comparison as keys in SCRIPT_SETS are String
    for key_str in ["punctuation", "numbers", "common_symbols"].iter() {
        let key = key_str.to_string();
        if !scripts_to_keep.contains(&key) { // Check if scripts_to_keep (Vec<String>) contains the current key (String)
            if let Some(script_set) = SCRIPT_SETS.get(&key) {
                allowed_chars.extend(script_set);
            }
        }
    }
    
    // Add essential whitespace that should always be allowed regardless of script choices
    allowed_chars.insert(' ');
    allowed_chars.insert('\t');
    allowed_chars.insert('\n'); // Though lines are processed and newlines re-added, having it in allowed_chars is safe.

    let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
    let (cleaned_string, _, _) = core_clean_text(text, &allowed_chars, &unusual_chars, min_chars_for_comment);
    Ok(cleaned_string)
}

// Helper function for script percentage calculation (moved from analyze_text for clarity)
/*
fn calc_script_percentages(py: Python, text: &str, scripts_to_keep: &[String]) -> PyResult<PyObject> {
    let percentages_dict = PyDict::new(py);
    
    if !scripts_to_keep.is_empty() {
        let non_whitespace_chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        let total_chars_for_percentage = non_whitespace_chars.len(); // Use count of non-whitespace for script percentage
        
        if total_chars_for_percentage > 0 {
            for script_key_str in scripts_to_keep {
                // script_key_str is already a &String, no need to convert further for SCRIPT_SETS.get()
                if let Some(charset) = SCRIPT_SETS.get(script_key_str) {
                    let script_count = non_whitespace_chars.iter()
                        .filter(|c| charset.contains(c))
                        .count();
                    
                    let percentage = (script_count as f64 / total_chars_for_percentage as f64) * 100.0;
                    percentages_dict.set_item(script_key_str, percentage)?;
                }
            }
        }
    }
    
    Ok(percentages_dict.to_object(py))
}
*/

// Define the SLIMMED DOWN struct to hold only essential analysis results for CSV
#[derive(Debug, Serialize)]
pub struct SlimTextAnalysisResult {
    pub original_total_chars: usize,
    pub cleaned_total_chars: usize,
    pub original_non_whitespace_chars: Option<usize>,
    pub greek_char_count_after_clean: Option<usize>,
    pub latin_char_count_after_clean: Option<usize>,
    pub cleaned_non_whitespace_chars_after_clean: Option<usize>,
    pub cleaned_text_content: String,
    pub badness_score_all_chars: Option<f64>,    // New score based on all chars in processed lines
    pub badness_score_non_ws: Option<f64>,       // Existing badness score, now explicitly named
}

// Internal function to perform text analysis and return the SLIMMED DOWN struct
pub fn perform_text_analysis(
    text: &str, 
    allowed_chars_ref: &HashSet<char>,
    unusual_chars_ref: &HashSet<char>,
    scripts_for_percentage_and_specific_counts: &[String], 
    calculate_specific_counts: bool,
    min_chars_for_comment: Option<usize>
) -> SlimTextAnalysisResult {
    let original_total_chars_abs = text.chars().count(); 
    let original_non_whitespace_chars_abs = text.chars().filter(|c| !c.is_whitespace()).count();

    let (cleaned_text, original_chars_processed_lines, kept_chars_processed_lines) = 
        core_clean_text(text, allowed_chars_ref, unusual_chars_ref, min_chars_for_comment);
    
    let cleaned_total_chars_abs = cleaned_text.chars().count();

    // Calculate badness_score_all_chars
    let badness_all_chars = if original_chars_processed_lines > 0 {
        Some(1.0 - (kept_chars_processed_lines as f64 / original_chars_processed_lines as f64))
    } else {
        Some(0.0) // Or None if original_chars_processed_lines is 0 implies no processing happened
    };

    let mut greek_char_count_cleaned: Option<usize> = None;
    let mut latin_char_count_cleaned: Option<usize> = None;
    #[allow(unused_assignments)] // Clippy seems to miss its usage in the struct below
    let mut cleaned_non_whitespace_chars_val: Option<usize> = None;

    // This block already calculates cleaned_non_whitespace_chars_val correctly after cleaning
    if calculate_specific_counts {
        let mut current_greek_count = 0;
        let mut current_latin_count = 0;
        let mut current_cleaned_non_ws_count = 0;

        let greek_set = SCRIPT_SETS.get("greek").cloned().unwrap_or_default();
        let latin_set = SCRIPT_SETS.get("latin").cloned().unwrap_or_default();

        for ch in cleaned_text.chars() {
            if !ch.is_whitespace() {
                current_cleaned_non_ws_count += 1;
            }
            if scripts_for_percentage_and_specific_counts.contains(&"greek".to_string()) && greek_set.contains(&ch) {
                current_greek_count += 1;
            }
            if scripts_for_percentage_and_specific_counts.contains(&"latin".to_string()) && latin_set.contains(&ch) {
                current_latin_count += 1;
            }
        }
        greek_char_count_cleaned = Some(current_greek_count);
        latin_char_count_cleaned = Some(current_latin_count);
        cleaned_non_whitespace_chars_val = Some(current_cleaned_non_ws_count);
    } else {
        cleaned_non_whitespace_chars_val = Some(cleaned_text.chars().filter(|c| !c.is_whitespace()).count());
    }

    // Calculate badness_score_non_ws
    let removed_non_whitespace_chars = original_non_whitespace_chars_abs.saturating_sub(cleaned_non_whitespace_chars_val.unwrap_or(0));
    let badness_non_ws = if original_non_whitespace_chars_abs > 0 {
        Some(removed_non_whitespace_chars as f64 / original_non_whitespace_chars_abs as f64)
    } else {
        Some(0.0)
    };
    
    SlimTextAnalysisResult {
        original_total_chars: original_total_chars_abs,
        cleaned_total_chars: cleaned_total_chars_abs,
        original_non_whitespace_chars: Some(original_non_whitespace_chars_abs),
        greek_char_count_after_clean: greek_char_count_cleaned,
        latin_char_count_after_clean: latin_char_count_cleaned,
        cleaned_non_whitespace_chars_after_clean: cleaned_non_whitespace_chars_val,
        cleaned_text_content: cleaned_text,
        badness_score_all_chars: badness_all_chars,
        badness_score_non_ws: badness_non_ws,
    }
}

/// Python-exposed function to analyze text metrics (still returns full HashMap for compatibility if needed elsewhere)
/// However, its internal call now uses the slimmed-down analysis.
/// If this function is ONLY used by the CSV generation, it could be removed or simplified further.
#[pyfunction]
pub fn analyze_text(py: Python, text: &str, scripts_to_keep: Vec<String>, calculate_specific_counts: bool, min_chars_for_comment: Option<usize>) -> PyResult<HashMap<String, PyObject>> {
    let mut allowed_chars = HashSet::new();
    for key in &scripts_to_keep {
        if let Some(script_set) = SCRIPT_SETS.get(key) {
            allowed_chars.extend(script_set);
        }
    }
    for key_str in ["punctuation", "numbers", "common_symbols"].iter() {
        let key = key_str.to_string();
        if !scripts_to_keep.contains(&key) {
            if let Some(script_set) = SCRIPT_SETS.get(&key) {
                allowed_chars.extend(script_set);
            }
        }
    }
    let unusual_chars_set = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();

    let analysis_result = perform_text_analysis(
        text, 
        &allowed_chars, 
        &unusual_chars_set, 
        &scripts_to_keep,
        calculate_specific_counts,
        min_chars_for_comment
    );

    let mut results = HashMap::new();
    results.insert("original_total_chars".to_string(), analysis_result.original_total_chars.to_object(py));
    results.insert("cleaned_total_chars".to_string(), analysis_result.cleaned_total_chars.to_object(py));
    results.insert("original_non_whitespace_chars".to_string(), analysis_result.original_non_whitespace_chars.unwrap_or(0).to_object(py));
    results.insert("cleaned_non_whitespace_chars".to_string(), analysis_result.cleaned_non_whitespace_chars_after_clean.unwrap_or(0).to_object(py));
    
    // The definition of removed_chars_count for the old badness_score was (original_total_chars - cleaned_total_chars).
    // Let's keep that for a general removed count, and use specific badness scores from analysis_result.
    let removed_chars_count_total = analysis_result.original_total_chars.saturating_sub(analysis_result.cleaned_total_chars);
    results.insert("removed_chars_count_total".to_string(), removed_chars_count_total.to_object(py));

    // Add the two badness scores
    results.insert("badness_score_all_chars".to_string(), analysis_result.badness_score_all_chars.unwrap_or(0.0).to_object(py));
    results.insert("badness_score_non_ws".to_string(), analysis_result.badness_score_non_ws.unwrap_or(0.0).to_object(py));

    // The old "badness_score" key used original_total_chars - cleaned_total_chars / original_non_whitespace_chars.
    // This is different from badness_score_non_ws if removed whitespace is significant.
    // For clarity, I am only exposing the two new specific badness scores.
    // If the old one is critical, it can be re-calculated here.

    if calculate_specific_counts {
        if let Some(greek_count) = analysis_result.greek_char_count_after_clean {
            results.insert("greek_chars_cleaned".to_string(), greek_count.to_object(py));
            if analysis_result.cleaned_non_whitespace_chars_after_clean.unwrap_or(0) > 0 {
                let percentage_greek = greek_count as f64 / analysis_result.cleaned_non_whitespace_chars_after_clean.unwrap_or(1) as f64 * 100.0;
                results.insert("percentage_greek_cleaned".to_string(), percentage_greek.to_object(py));
            } else {
                results.insert("percentage_greek_cleaned".to_string(), 0.0.to_object(py));
            }
        }
        if let Some(latin_count) = analysis_result.latin_char_count_after_clean {
            results.insert("latin_chars_cleaned".to_string(), latin_count.to_object(py));
            if analysis_result.cleaned_non_whitespace_chars_after_clean.unwrap_or(0) > 0 {
                let percentage_latin = latin_count as f64 / analysis_result.cleaned_non_whitespace_chars_after_clean.unwrap_or(1) as f64 * 100.0;
                results.insert("percentage_latin_cleaned".to_string(), percentage_latin.to_object(py));
            } else {
                results.insert("percentage_latin_cleaned".to_string(), 0.0.to_object(py));
            }
        }
    }
    
    results.insert("cleaned_text".to_string(), analysis_result.cleaned_text_content.to_object(py));

    Ok(results)
}

/// Python-exposed function to list available script keys
#[pyfunction]
pub fn list_available_scripts() -> PyResult<Vec<String>> {
    Ok(SCRIPT_SETS.keys()
        .filter(|&k| **k != *"unusual")
        .cloned()
        .collect())
}
