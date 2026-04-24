use aho_corasick::AhoCorasick;
use glossapi_rs_common::scan_script_metrics;
use htmlentity::entity::{decode, ICodedDataTrait};
use lazy_static::lazy_static;
use memchr::memchr; // For Step 5.1
use memchr::memmem;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::collections::{HashMap, HashSet}; // For optimizing comment search in strip_tags_custom

use crate::md_module;
use crate::normalize;

// Constants
const TEXT_MISSING_COMMENT: &str = "<!-- text-missing -->";
const TABLE_REMOVED_COMMENT: &str = "<!-- table-removed -->"; // Added for badness adjustment
// Emitted when an individual LINE is dropped (BAD_LINE_AC / glyph
// regex / rule-B coverage predicate). Preserves the fact that a line
// was here for downstream stats + line-alignment invariants.
const LINE_REMOVED_COMMENT: &str = "<!-- line-removed -->";

/// Per-doc char/line accounting returned by `core_clean_text_with_stats`.
///
/// Invariants (approximate, modulo saturating_sub clamps on rare entity
/// expansions):
///
/// Over the INPUT chars:
///   input_chars ≈ content_chars_kept
///                 + chars_dropped_by_line_drop
///                 + chars_dropped_by_normalization
///                 + chars_dropped_by_per_char_filter
///                 + marker_chars_passthrough
///
/// Over the OUTPUT chars:
///   output_chars = content_chars_kept
///                 + marker_chars_passthrough
///                 + marker_chars_added
///
/// Where `marker_chars_passthrough` counts input chars whose LINE was
/// itself a marker (pre-existing `<!-- text-missing -->` / `<!--
/// table-removed -->`), and `marker_chars_added` counts marker chars we
/// emitted during cleaning (`<!-- line-removed -->`, inline TMC
/// additions, standalone TMC replacements).
///
/// `content_chars_kept` EXCLUDES all comment markers. Callers that want
/// "chars_after" without markers should use this field directly.
#[derive(Debug, Clone, Default)]
pub struct CleanStats {
    pub content_chars_kept: usize,
    pub chars_dropped_by_line_drop: usize,
    pub chars_dropped_by_normalization: usize,
    pub chars_dropped_by_per_char_filter: usize,
    pub lines_dropped_count: usize,
    /// Input lines that were themselves marker comments, passed through.
    /// Sums to input-side invariant.
    pub marker_chars_passthrough: usize,
    /// Markers we emitted (LINE_REMOVED_COMMENT, inline/standalone TMC).
    /// Sums to output-side invariant; NOT accounted against input.
    pub marker_chars_added: usize,
    // Back-compat fields used by `perform_text_analysis` for badness scoring.
    pub original_chars_for_badness: usize,
    pub sum_kept_line_content_chars: usize,
}

lazy_static! {
    // Regular expressions for detection (compiled once) - Most are now unused
    // pub static ref GLYPH_TAG_REGEX_RAW: Regex = Regex::new(r"(?:^|\s)glyph<c=\d+,font=/[^>]+>(?:\s|$)").unwrap();
    // pub static ref GLYPH_TAG_REGEX_HTML: Regex = Regex::new(r"(?:^|\s)glyph&lt;c=\d+,font=/[^>]+&gt;(?:\s|$)").unwrap();
    // pub static ref ANY_TAG_REGEX: Regex = Regex::new(r"(?:^|\s)<[^>]*>(?:\s|$)").unwrap();
    // pub static ref IS_COMMENT_REGEX: Regex = Regex::new(r"^<!--").unwrap(); // Replaced by direct byte check
    // pub static ref HTML_ENTITY_REGEX: Regex = Regex::new(r"&[a-zA-Z]+;|&#\d+;|&lt;|&gt;|&amp;").unwrap(); // Replaced by htmlentity crate

    // Regex for HTML comments (captures the whole comment) - STILL USED
    pub static ref COMMENT_REGEX: Regex = Regex::new(r"<!--.*?-->").unwrap();
    pub static ref GLYPH_FONT_TAG_REGEX: Regex =
        Regex::new(r"(?i)glyph<c=\d+,font=/[^>]+>").unwrap();
    pub static ref FONT_GLYPH_TAG_REGEX: Regex =
        Regex::new(r"(?i)<c=\d+,font=/[^>]+>glyph").unwrap();
    pub static ref DOT_LEADER_RUN_REGEX: Regex = Regex::new(r"\.{4,}").unwrap();
    // PostScript-glyph-name residue from PDF extraction. `/uni<hex>` is the
    // Unicode-codepoint glyph-name form (e.g. /uni03B1 for α); `/gid<N>` is
    // the glyph-ID form. Both appear when a PDF extractor dumps raw Adobe
    // glyph names instead of decoding to Unicode. Probe on openarchives.gr
    // first 5k rows (2026-04-21) found 45 docs / 68,510 total hits for
    // /uni<hex> and 24 docs / 19,085 hits for /gid<N> — catastrophic
    // corruption never caught by the current BAD_LINE_AC triggers.
    pub static ref PDF_GLYPH_NAME_REGEX: Regex =
        Regex::new(r"/uni[0-9A-Fa-f]{4,6}|/g(?:id)?\d+").unwrap();
    // PDF font-subset naming convention /<6-uppercase>+<FontName>
    // (e.g. /XQDMQS+CenturyGothic, /NUMPTY+ImprintMTnum).
    pub static ref PDF_FONT_SUBSET_REGEX: Regex =
        Regex::new(r"/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+").unwrap();

    // Regex for HTML/XML tags (for cleaning, non-comment tags) - Replaced by strip_tags_custom
    // pub static ref ANY_TAG_CLEANING_REGEX: Regex = Regex::new(r"<[^>]*>").unwrap();

    // Central HashMap for character scripts
    pub static ref SCRIPT_SETS: HashMap<String, HashSet<char>> = {
        let mut map = HashMap::new();

        map.insert("latin".to_string(), "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect());

        let mut greek_chars = HashSet::new();
        for code in 0x0370..0x03E2 { if let Some(c) = std::char::from_u32(code) { greek_chars.insert(c); }}
        for code in 0x03F0..0x0400 { if let Some(c) = std::char::from_u32(code) { greek_chars.insert(c); }}
        // Polytonic Greek range U+1F00..U+2000 (Greek Extended block).
        // Explicit so a future edit cannot silently move it to `unusual`.
        for code in 0x1F00..0x2000 { if let Some(c) = std::char::from_u32(code) { greek_chars.insert(c); }}
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
        let mut common_symbols_set: HashSet<char> = common_symbols.chars().collect();
        // Wave-2 (Case 12): widen common_symbols with math / arrows /
        // geometric-shapes / super-subscripts / letterlike. CS + math +
        // bilingual theses carry these as legitimate content; stripping
        // them to strip mojibake is a false economy.
        // - U+2070..U+209F super/subscripts
        // - U+2100..U+214F letterlike (ℓ ™ ℵ etc.)
        // - U+2190..U+21FF arrows
        // - U+2200..U+22FF math operators
        // - U+2500..U+257F box drawing (table-border chars)
        // - U+25A0..U+25FF geometric shapes (bullets, markers)
        for range in &[
            (0x2070u32, 0x209Fu32),
            (0x2100u32, 0x214Fu32),
            (0x2190u32, 0x21FFu32),
            (0x2200u32, 0x22FFu32),
            (0x2500u32, 0x257Fu32),
            (0x25A0u32, 0x25FFu32),
        ] {
            for cp in range.0..=range.1 {
                if let Some(c) = std::char::from_u32(cp) {
                    common_symbols_set.insert(c);
                }
            }
        }
        map.insert("common_symbols".to_string(), common_symbols_set);

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
        // Armenian, Hebrew, Arabic, Georgian, Math Alphanumeric Greek etc.
        // are INTENTIONALLY NOT stripped here. Policy (2026-04-21): we only
        // add a range to `unusual` (strip) when the codepoints carry no
        // semantic meaning — i.e., noise. For meaningful scripts not in
        // Apertus's vocab we should FOLD (e.g., Math Alphanumeric Greek →
        // regular Greek, handled in `normalize::fold_codepoint`), not
        // strip. For Armenian/Hebrew/Arabic/Georgian, Apertus's multilingual
        // training covers them; they should be preserved as-is.
        map.insert("unusual".to_string(), unusual_chars);

        map
    };
}

// Artefact triggers for Aho-Corasick (Step 2.1)
//
// Expanded 2026-04-21 to cover PostScript glyph-name residue from PDF
// extraction that was previously missed. The five-char `GLYPH` literal
// (without `<` after) matched 172 docs / 84,798 hits in an openarchives
// probe of 5,000 rows — catastrophic noise that passed through
// unrejected because the original set only matched `GLYPH<`. PS glyph
// names like /hyphenminus, /space, /period are appended too (see
// `glyph_marker_extended_spec.md`).
// Structural-PDF-residue triggers: line gets dropped (emits
// LINE_REMOVED_COMMENT). These are whole-line PDF-extractor artefact
// markers with zero legitimate-prose use.
static BAD_LINE_AC: Lazy<AhoCorasick> = Lazy::new(|| {
    AhoCorasick::new([
        "glyph<c=",
        "glyph&lt;c=",
        "GLYPH<",
        "GLYPH&lt;",
        "MS-Bold-",
        "font=/",
        "FontName=",
        "GLYPH",        // bare word — 84k hits in 172 docs per 5k-row probe
        "hyphenminus",  // bare PostScript glyph name (not a real word)
    ])
    .unwrap()
});

// Rule A literals — PostScript glyph-name + CID prefix. Gemini wave on
// 1000 sampled lines (2026-04-22) showed 86.5% span-drop preferred.
// Stripped inline as spans; never triggers line-removal on their own.
static RULE_A_LITERALS_AC: Lazy<AhoCorasick> = Lazy::new(|| {
    // LeftmostLongest prevents `/hyphen` from eating the `/hyphen` prefix
    // of `/hyphenminus` (leaving "minus" residue). Same concern for
    // `/plus` vs `/plusminus`, `/dagger` vs `/daggerdbl`, `/registered`
    // vs the shorter variants — always prefer the longer glyph name.
    aho_corasick::AhoCorasickBuilder::new()
        .match_kind(aho_corasick::MatchKind::LeftmostLongest)
        .build([
            "/hyphenminus", "/space", "/period", "/comma", "/colon", "/semicolon",
            "/slash", "/backslash", "/parenleft", "/parenright",
            "/bracketleft", "/bracketright", "/braceleft", "/braceright",
            "/quotesingle", "/quotedbl", "/exclam", "/question",
            "/asterisk", "/plus", "/minus", "/equal", "/less", "/greater",
            "/ampersand", "/percent", "/at", "/dollar", "/numbersign",
            "/underscore", "/asciitilde", "/asciicircum",
            "/endash", "/emdash", "/hyphen", "/bullet",
            "/copyright", "/registered", "/trademark", "/degree",
            "/plusminus", "/multiply", "/divide", "/section",
            "/paragraph", "/dagger", "/daggerdbl", "/ellipsis",
            "/glyph", "CID+",
        ])
        .unwrap()
});

/// Strip rule A + rule B matches from a line. Rule A is always span-
/// stripped. For rule B (PDF_GLYPH_NAME_REGEX = /uni<hex> | /g<N>),
/// also check the coverage predicate: if rule-B match count ≥ 10
/// AND rule-B-matches / non-whitespace-length ≥ 0.09, the line is
/// flagged for removal (caller emits LINE_REMOVED_COMMENT).
///
/// Gemini wave (2026-04-22, 1000 rule-B cases): coverage ≥ 0.10
/// alone gave precision 95.7% / recall 82%. User chose the more
/// conservative `count ≥ 10 AND coverage ≥ 0.09` (precision 96.3%,
/// recall 60.4% — trades recall for precision).
///
/// Returns (stripped_line, rule_b_line_drop_flag).
fn apply_glyph_span_strip_and_rule_b(line: &str) -> (String, bool) {
    // Count rule B matches before we strip
    let rule_b_count = PDF_GLYPH_NAME_REGEX.find_iter(line).count();
    let non_ws_len = line.chars().filter(|c| !c.is_whitespace()).count();
    let rule_b_coverage = if non_ws_len > 0 {
        rule_b_count as f64 / non_ws_len as f64
    } else {
        0.0
    };
    let rule_b_line_drop = rule_b_count >= 10 && rule_b_coverage >= 0.09;

    // Strip rule A literal spans
    let mut out = String::with_capacity(line.len());
    let mut last_end = 0;
    for m in RULE_A_LITERALS_AC.find_iter(line) {
        out.push_str(&line[last_end..m.start()]);
        last_end = m.end();
    }
    out.push_str(&line[last_end..]);
    // Strip rule B regex spans (subsequent pass — RULE_A shouldn't overlap)
    let stripped = PDF_GLYPH_NAME_REGEX.replace_all(&out, "").into_owned();
    (stripped, rule_b_line_drop)
}

fn has_decoded_glyph_font_artefact(line: &str) -> bool {
    // GLYPH_FONT_TAG_REGEX / FONT_GLYPH_TAG_REGEX are structural PDF
    // font-tag residue; PDF_FONT_SUBSET_REGEX (/XQDMQS+CenturyGothic)
    // is the Adobe font-subset convention. All stay as whole-line
    // drop triggers (user-review classification: ✅).
    //
    // PDF_GLYPH_NAME_REGEX is now handled by apply_glyph_span_strip_
    // and_rule_b per the 2026-04-22 Gemini wave.
    GLYPH_FONT_TAG_REGEX.is_match(line)
        || FONT_GLYPH_TAG_REGEX.is_match(line)
        || PDF_FONT_SUBSET_REGEX.is_match(line)
}

fn is_unicode_noise_char(ch: char) -> bool {
    match ch {
        '\t' | '\n' => false,
        // Invisible formatting / directional / control / replacement codepoints
        // with no semantic purpose in Greek text. U+200E (LRM) and U+200F (RLM)
        // were added 2026-04-21 after wave11 surfaced them as untouched
        // bidi-mark residue in Greek-Wikipedia translation patterns.
        '\u{00AD}' | '\u{03A2}' | '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{200E}'
        | '\u{200F}' | '\u{2060}' | '\u{FEFF}' | '\u{FFFD}' => true,
        _ => {
            let code = ch as u32;
            code < 0x20
                || code == 0x7F
                || (0x80..=0x9F).contains(&code)
                || (0xE000..=0xF8FF).contains(&code)
                || (0xF0000..=0xFFFFD).contains(&code)
                || (0x100000..=0x10FFFD).contains(&code)
        }
    }
}

fn normalize_layout_leader_runs(line: &str) -> Option<String> {
    if line.is_empty()
        || line == TEXT_MISSING_COMMENT
        || line == TABLE_REMOVED_COMMENT
    {
        return None;
    }
    // Tiered bucket per normalize.rs: {2}→1, {3}→3, {4..10}→3, {>10}→20.
    // Uniform with the whitespace-run rule.
    normalize::normalize_dot_runs(line)
}

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
                result_buf.push_str(unsafe {
                    std::str::from_utf8_unchecked(&bytes[current_pos..current_pos + i])
                });
                current_pos += i;
                if bytes.get(current_pos..current_pos + 4) == Some(b"<!--") {
                    let search_start_for_comment_end = current_pos + 4;
                    // Use memmem::find for faster "-->" search
                    match comment_closer.find(&bytes[search_start_for_comment_end..]) {
                        Some(j) => {
                            // j is the start index of "-->" within the slice
                            let comment_end_in_slice = search_start_for_comment_end + j + 3; // end of "-->"
                            result_buf.push_str(unsafe {
                                std::str::from_utf8_unchecked(
                                    &bytes[current_pos..comment_end_in_slice],
                                )
                            });
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
                            let tag_content_str =
                                unsafe { std::str::from_utf8_unchecked(tag_content_bytes) };
                            for _char_in_tag in
                                tag_content_str.chars().filter(|c| !c.is_whitespace())
                            {
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
                result_buf
                    .push_str(unsafe { std::str::from_utf8_unchecked(&bytes[current_pos..]) });
                current_pos = bytes.len();
            }
        }
    }
    removed_non_ws_tag_chars // No longer returns the String, it's modified in place
}

/// Thin wrapper over `core_clean_text_with_stats` that returns just the
/// legacy `(cleaned_text, original_chars_count_for_badness,
/// kept_chars_count_for_badness)` tuple used by `perform_text_analysis`
/// and existing tests. New call sites that need the four-way char split
/// or lines_dropped_count should call `core_clean_text_with_stats`
/// directly.
pub fn core_clean_text(
    text: &str,
    allowed_chars: &HashSet<char>,
    unusual_chars_set: &HashSet<char>,
    min_chars_for_comment_override: Option<usize>,
) -> (String, usize, usize) {
    let (cleaned, stats) = core_clean_text_with_stats(
        text,
        allowed_chars,
        unusual_chars_set,
        min_chars_for_comment_override,
    );
    (
        cleaned,
        stats.original_chars_for_badness,
        stats.sum_kept_line_content_chars,
    )
}

/// Core text cleaning function with full char accounting.
///
/// See `CleanStats` for the invariant and field meanings.
pub fn core_clean_text_with_stats(
    text: &str,
    allowed_chars: &HashSet<char>,
    unusual_chars_set: &HashSet<char>,
    min_chars_for_comment_override: Option<usize>,
) -> (String, CleanStats) {
    // -----------------------------------------------------------------
    // Wave-2 preprocessing (Cases 4, 7, 10a, 8 — 2026-04-23).
    // Applied BEFORE the per-line filter loop so recovered chars (from
    // entity decode and Adobe Symbol PUA decode) survive per-char
    // filtering. Char-count delta attributed to `chars_dropped_by_
    // normalization`. The final step is the full Phase A orchestrator
    // (`md_module::normalize_md_syntax`) — a single entry point that
    // canonicalizes GFM table separators, HR rules, and reflows
    // paragraphs in the correct order. Routing through the orchestrator
    // (rather than calling `reflow_paragraphs` alone) is required so
    // optional-pipe GFM tables like `a | b\n--- | ---\n1 | 2` are
    // identified as tables BEFORE reflow decides whether to fuse rows.
    // -----------------------------------------------------------------
    let wave2_in_len = text.chars().count();
    let step1 = normalize::decode_html_entities(text);
    let step2 = normalize::decode_adobe_symbol_pua(&step1);
    let step3 = normalize::strip_glyph_markers(&step2);
    let step4 = normalize::strip_soft_hyphens(&step3);
    let step5 = md_module::normalize_md_syntax(&step4);
    let wave2_out_len = step5.chars().count();
    let wave2_preprocessing_delta = wave2_in_len.saturating_sub(wave2_out_len);
    // Re-alias `text` so the rest of the function sees the preprocessed
    // string. `text` is a reference to the original; we need to bind a
    // new owned string and reborrow.
    let text_owned = step5;
    let text = text_owned.as_str();
    // ---- end wave-2 preprocessing ----

    let min_comment_chars = min_chars_for_comment_override.unwrap_or(5);
    let mut cleaned_output_string_builder = String::new(); // Used to build the final string with newlines
    let mut original_chars_for_badness: usize = 0; // Sum of original line content lengths (excluding their newlines)

    // New counter for the sum of *content characters* of lines added to the output,
    // before specific placeholder penalties are applied.
    let mut sum_kept_line_content_chars: usize = 0;

    let mut inline_tmc_additions_count: usize = 0;
    let mut standalone_tmc_replacements_on_processed_lines_count: usize = 0;

    // Four-way char accounting (see `CleanStats` doc-comment).
    let mut content_chars_kept: usize = 0;
    let mut chars_dropped_by_line_drop: usize = 0;
    // Seed with the wave-2 preprocessing delta (entity decode, PUA
    // recovery net char change, GLYPH marker deletion, soft-hyphen
    // strip, paragraph reflow whitespace collapse).
    let mut chars_dropped_by_normalization: usize = wave2_preprocessing_delta;
    let mut chars_dropped_by_per_char_filter: usize = 0;
    let mut lines_dropped_count: usize = 0;
    let mut marker_chars_passthrough: usize = 0;
    let mut marker_chars_added: usize = 0;

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

    let mut carry_math_state = false;

    // Pre-pass: GFM table separator row canonicalization (parser-validated).
    let table_replacements = md_module::scan_gfm_table_separators(text);
    // Code-fence state carried across lines — inside a fenced block we skip
    // all normalizations so code indentation and punctuation survive intact.
    let mut in_code_fence = false;

    for (line_index, line) in text.lines().enumerate() {
        let trimmed_line = line.trim();
        if trimmed_line == TEXT_MISSING_COMMENT {
            let line_chars = line.chars().count();
            original_chars_for_badness += line_chars;
            // Input line IS the TMC marker — it's a marker pass-through, not
            // content. Attribute all its chars to marker_chars_passthrough so
            // they don't pollute content_chars_kept or any drop bucket.
            marker_chars_passthrough += line_chars;
            cleaned_output_string_builder.push_str(TEXT_MISSING_COMMENT);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        // GFM table separator: if the pre-pass identified this line as a
        // parser-validated separator row under a header, swap in the canonical
        // minimum-width form. This is a semantics-preserving normalization
        // (alignment + column count are preserved per GFM spec), so it must
        // stay badness-neutral: count the original line chars as kept.
        if let Some(canonical) = table_replacements.get(&line_index) {
            let line_chars = line.chars().count();
            let canonical_chars = canonical.chars().count();
            original_chars_for_badness += line_chars;
            sum_kept_line_content_chars += line_chars;
            content_chars_kept += canonical_chars;
            chars_dropped_by_normalization += line_chars.saturating_sub(canonical_chars);
            cleaned_output_string_builder.push_str(canonical);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        // Code-fence state: toggle on ``` / ~~~ markers. Pass the marker and
        // everything inside through unchanged so normalizations don't collapse
        // meaningful code indentation or punctuation. Pass the RAW line so
        // the detector can apply CommonMark's ≥4-column indented-code rule
        // (a ``` at that indentation is literal content, not a fence).
        if md_module::is_code_fence_marker(line) {
            in_code_fence = !in_code_fence;
            let line_chars = line.chars().count();
            original_chars_for_badness += line_chars;
            sum_kept_line_content_chars += line_chars;
            content_chars_kept += line_chars;
            cleaned_output_string_builder.push_str(line);
            cleaned_output_string_builder.push('\n');
            continue;
        }
        if in_code_fence {
            let line_chars = line.chars().count();
            original_chars_for_badness += line_chars;
            sum_kept_line_content_chars += line_chars;
            content_chars_kept += line_chars;
            cleaned_output_string_builder.push_str(line);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        // Decode entities before artefact checks so html-escaped GLYPH/font tags
        // are caught by the same canonical matcher family as raw XML-like forms.
        let decoded_entity_data = decode(line.as_bytes());
        let line_after_entity_decoding_str = decoded_entity_data
            .to_string()
            .unwrap_or_else(|_| line.to_string());

        let mut skip_bad_line_check = carry_math_state || trimmed_line == "$$";
        if !skip_bad_line_check && trimmed_line.contains("$$") {
            // Handle inline math by skipping BAD_LINE_AC so \text in math isn't penalised.
            let dollar_pairs = trimmed_line.match_indices("$$").count();
            if dollar_pairs > 0 && dollar_pairs % 2 == 0 {
                skip_bad_line_check = true;
            }
        }

        // Rule A (PS-glyph literal set) + Rule B (PS-glyph regex).
        // Both span-stripped inline; rule B additionally triggers
        // whole-line removal if coverage predicate met (mc ≥ 10 AND
        // rule-B matches / non-whitespace chars ≥ 0.09). Per 2026-04-22
        // Gemini wave: P=96.3%, R=60.4% on rule-B predicate.
        // Applied BEFORE BAD_LINE_AC so `/hyphenminus`-style spans get
        // stripped and don't trigger the `hyphenminus` substring trigger
        // in BAD_LINE_AC.
        // Skip in math context.
        let rule_b_line_drop;
        let post_rule_strip = if skip_bad_line_check {
            rule_b_line_drop = false;
            line_after_entity_decoding_str.clone()
        } else {
            let (stripped, drop) =
                apply_glyph_span_strip_and_rule_b(&line_after_entity_decoding_str);
            rule_b_line_drop = drop;
            stripped
        };
        if rule_b_line_drop {
            let line_chars = line.chars().count();
            original_chars_for_badness += line_chars;
            chars_dropped_by_line_drop += line_chars;
            lines_dropped_count += 1;
            marker_chars_added += LINE_REMOVED_COMMENT.chars().count();
            cleaned_output_string_builder.push_str(LINE_REMOVED_COMMENT);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        if !skip_bad_line_check
            && (BAD_LINE_AC.is_match(&post_rule_strip)
                || has_decoded_glyph_font_artefact(&post_rule_strip))
        {
            let line_chars = line.chars().count();
            original_chars_for_badness += line_chars;
            chars_dropped_by_line_drop += line_chars;
            lines_dropped_count += 1;
            marker_chars_added += LINE_REMOVED_COMMENT.chars().count();
            cleaned_output_string_builder.push_str(LINE_REMOVED_COMMENT);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        processed_line_segment_buf.clear();
        current_line_removed_chars_buffer_buf.clear();
        // line_after_tag_handling_buf is cleared inside strip_tags_custom

        // Step 5.1 & 5.4: Use strip_tags_custom with a reusable buffer
        // on the rule-A/B-stripped decoded line content
        let removed_from_tags_count = strip_tags_custom(
            &post_rule_strip,
            &mut line_after_tag_handling_buf,
        );

        // Iterate the result of tag stripping for character filtering, keeping track of LaTeX math spans.
        let chars: Vec<char> = line_after_tag_handling_buf.chars().collect();
        let mut idx = 0usize;
        let mut in_math = carry_math_state;
        let mut math_chars_this_line: usize = 0;
        let mut outside_math_original_chars: usize = 0;

        while idx < chars.len() {
            let ch = chars[idx];
            if ch == '$' && idx + 1 < chars.len() && chars[idx + 1] == '$' {
                in_math = !in_math;
                processed_line_segment_buf.push(ch);
                processed_line_segment_buf.push(chars[idx + 1]);
                math_chars_this_line += 2;
                idx += 2;
                continue;
            }

            if in_math {
                processed_line_segment_buf.push(ch);
                math_chars_this_line += 1;
                idx += 1;
                continue;
            }

            outside_math_original_chars += 1;

            // Codepoint fold (ligatures, enclosed / dingbat / math-alphanumeric
            // digits, vulgar fractions, Unicode whitespace variants) bypasses
            // the allowed/unusual check — replacements are ASCII or a regular
            // space.
            if let Some(replacement) = normalize::fold_codepoint(ch) {
                processed_line_segment_buf.push_str(replacement);
                idx += 1;
                continue;
            }

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

            let should_remove_char = is_unicode_noise_char(ch)
                || (is_char_in_unusual_set && !is_char_allowed_by_scripts);

            if should_remove_char {
                if !ch.is_whitespace() {
                    current_line_removed_chars_buffer_buf.push(ch);
                }
            } else {
                processed_line_segment_buf.push(ch);
            }
            idx += 1;
        }

        carry_math_state = in_math;

        // If the line only contained math content, preserve it but skip scoring contributions.
        if outside_math_original_chars == 0 && math_chars_this_line > 0 {
            // Math pass-through: count chars as kept content (math is legitimate).
            content_chars_kept += processed_line_segment_buf.chars().count();
            cleaned_output_string_builder.push_str(&processed_line_segment_buf);
            cleaned_output_string_builder.push('\n');
            continue;
        }

        let removed_chars_on_line_for_comment_decision =
            removed_from_tags_count + current_line_removed_chars_buffer_buf.chars().count();

        let trimmed_segment_for_comment_check = processed_line_segment_buf.trim();
        let mut line_content_to_add = String::new();

        // Check if the line (after processing up to character filtering) is exclusively an HTML comment.
        let is_exclusively_comment = if !trimmed_segment_for_comment_check.is_empty()
            && COMMENT_REGEX.is_match(trimmed_segment_for_comment_check)
        {
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
            if !processed_line_segment_buf.trim().is_empty() {
                // P_trimmed is not empty
                if removed_chars_on_line_for_comment_decision >= min_comment_chars {
                    line_content_to_add.push_str(processed_line_segment_buf.trim_end());
                    line_content_to_add.push(' ');
                    line_content_to_add.push_str(TEXT_MISSING_COMMENT);
                    inline_tmc_additions_count += 1;
                } else {
                    line_content_to_add.push_str(&processed_line_segment_buf);
                }
            } else {
                // processed_line_segment_buf is empty or whitespace only
                if removed_chars_on_line_for_comment_decision >= min_comment_chars
                    && line.chars().any(|c| !c.is_whitespace())
                {
                    // original line had content
                    line_content_to_add.push_str(TEXT_MISSING_COMMENT);
                    standalone_tmc_replacements_on_processed_lines_count += 1;
                } else {
                    line_content_to_add.push_str(&processed_line_segment_buf);
                }
            }
        }

        let kept_chars_total = line_content_to_add.chars().count();
        sum_kept_line_content_chars += kept_chars_total.saturating_sub(math_chars_this_line); // exclude math spans
        original_chars_for_badness += line.chars().count().saturating_sub(math_chars_this_line);

        // Per-char filter accounting. Entity-decode + rule A/B span strip +
        // tag strip + per-char unicode filter together shrink input chars to
        // processed_line_segment_buf chars. Marker additions happen AFTER
        // (line_content_to_add), so they don't pollute this delta.
        let input_chars_this_line = line.chars().count();
        let post_per_char_chars = processed_line_segment_buf.chars().count();
        chars_dropped_by_per_char_filter +=
            input_chars_this_line.saturating_sub(post_per_char_chars);

        let line_to_write = if is_exclusively_comment {
            // Input line was ITSELF a comment (pass-through) — don't touch.
            line_content_to_add.clone()
        } else {
            // Chain line-level normalizations AFTER cleaning so that chars
            // removed by per-char filter / rule-A/B-span-strip /
            // entity-decode collapse cleanly — e.g. a word stripped
            // mid-line leaves `foo  bar` (2 spaces) which whitespace-run
            // bucketing collapses to `foo bar`. Normalize passes are
            // marker-safe (they operate on specific patterns that don't
            // overlap with `<!-- text-missing -->`), so inline-TMC lines
            // normalize too. Order matters:
            //   dot-leader (tiered bucket) -> separator line -> ellipsis run
            //   -> malformed-entity fallback -> whitespace run (tiered bucket).
            // Dots and whitespace both use the tiered bucket rule:
            //   {2}→1, {3}→3, {4..10}→3, {>10}→20.
            let mut s = line_content_to_add.clone();
            if let Some(n) = normalize_layout_leader_runs(&s) {
                s = n;
            }
            if let Some(n) = md_module::normalize_separator_line(&s) {
                s = n;
            }
            if let Some(n) = normalize::normalize_ellipsis_runs(&s) {
                s = n;
            }
            if let Some(n) = normalize::normalize_malformed_entities(&s) {
                s = n;
            }
            if let Some(n) = normalize::normalize_whitespace_runs(&s) {
                s = n;
            }
            s
        };
        // Normalize-pass accounting: delta between pre-normalize (with any
        // inline markers already attached) and post-normalize output.
        // Saturating because rare expansions (e.g. `…` folded to ASCII
        // triple-dot ".....") would underflow.
        let pre_normalize_chars = line_content_to_add.chars().count();
        let post_normalize_chars = line_to_write.chars().count();
        chars_dropped_by_normalization +=
            pre_normalize_chars.saturating_sub(post_normalize_chars);

        // Content-chars-kept accounting: output chars on this line EXCLUDING
        // any marker chars that were added inline (space + TMC) or that
        // wholly replaced the line (standalone TMC / exclusive-comment).
        if is_exclusively_comment {
            // Output line IS a comment marker that came from the INPUT
            // (pass-through; we didn't add it). Attribute to passthrough.
            marker_chars_passthrough += post_normalize_chars;
        } else if line_content_to_add.contains(TEXT_MISSING_COMMENT) {
            // Inline TMC addition OR standalone TMC replacement — we ADDED
            // this marker. Standalone case: processed_line_segment_buf.trim()
            // is empty and the whole line becomes the marker. Inline case:
            // output = trimmed_content + " " + TMC, so marker part is
            // (1 space + TMC.len()). Normalize doesn't run on this path
            // (line_to_write == line_content_to_add).
            let inline_marker_chars = TEXT_MISSING_COMMENT.chars().count();
            if processed_line_segment_buf.trim().is_empty() {
                marker_chars_added += post_normalize_chars;
            } else {
                let marker_span = inline_marker_chars + 1; // " " + TMC
                content_chars_kept += post_normalize_chars.saturating_sub(marker_span);
                marker_chars_added += marker_span.min(post_normalize_chars);
            }
        } else {
            // Normal kept line — all output chars are content.
            content_chars_kept += post_normalize_chars;
        }

        cleaned_output_string_builder.push_str(&line_to_write);
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
    total_placeholder_content_penalty +=
        inline_tmc_additions_count * (TEXT_MISSING_COMMENT.len() + 1); // +1 for the space

    // Penalty for TEXT_MISSING_COMMENT that replaced an entire processed line.
    // The content added was TMC. We want to keep 0 from original for this line part.
    // So, penalty is length of TMC.
    total_placeholder_content_penalty +=
        standalone_tmc_replacements_on_processed_lines_count * TEXT_MISSING_COMMENT.len();

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
            num_table_removed_comments_as_full_lines_in_output += 1;
        }
    }
    total_placeholder_content_penalty +=
        num_table_removed_comments_as_full_lines_in_output * TABLE_REMOVED_COMMENT.len();

    let adjusted_kept_chars_for_badness =
        sum_kept_line_content_chars.saturating_sub(total_placeholder_content_penalty);

    // If an input line was pre-existing TABLE_REMOVED_COMMENT, the main
    // path classified it as kept content. Move those chars from content to
    // marker_chars_passthrough so content_chars_kept reflects true content.
    let trc_marker_chars = TABLE_REMOVED_COMMENT.chars().count();
    let trc_reclass_total = num_table_removed_comments_as_full_lines_in_output * trc_marker_chars;
    marker_chars_passthrough += trc_reclass_total;
    content_chars_kept = content_chars_kept.saturating_sub(trc_reclass_total);

    let stats = CleanStats {
        content_chars_kept,
        chars_dropped_by_line_drop,
        chars_dropped_by_normalization,
        chars_dropped_by_per_char_filter,
        lines_dropped_count,
        marker_chars_passthrough,
        marker_chars_added,
        original_chars_for_badness,
        sum_kept_line_content_chars: adjusted_kept_chars_for_badness,
    };

    (final_cleaned_text, stats)
}

/// Build (allowed_chars, unusual_chars) for the requested script set.
/// Ensures `punctuation`, `numbers`, `common_symbols` are always included
/// and that whitespace chars are always allowed.
fn build_script_char_sets(scripts_to_keep: &[String]) -> (HashSet<char>, HashSet<char>) {
    let mut allowed_chars = HashSet::new();
    for key in scripts_to_keep {
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
    allowed_chars.insert(' ');
    allowed_chars.insert('\t');
    allowed_chars.insert('\n');
    let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
    (allowed_chars, unusual_chars)
}

/// Python-exposed function to clean a single string
#[pyfunction]
pub fn clean_text(
    text: &str,
    scripts_to_keep: Vec<String>,
    min_chars_for_comment: Option<usize>,
) -> PyResult<String> {
    let (allowed_chars, unusual_chars) = build_script_char_sets(&scripts_to_keep);
    let (cleaned_string, _, _) =
        core_clean_text(text, &allowed_chars, &unusual_chars, min_chars_for_comment);
    Ok(cleaned_string)
}

/// Python-exposed variant that also returns per-doc char accounting.
///
/// Returns `(cleaned_text, stats_dict)` where `stats_dict` has integer keys:
/// - `content_chars_kept`: output chars excluding all comment markers
/// - `chars_dropped_by_line_drop`: chars in lines replaced by a line-drop marker
/// - `chars_dropped_by_normalization`: chars collapsed by dot/whitespace/separator/table/ellipsis normalizers
/// - `chars_dropped_by_per_char_filter`: chars stripped by entity-decode / rule-A/B / tag-strip / unicode filter
/// - `lines_dropped_count`: number of line-drop marker emissions
/// - `marker_chars_passthrough`: input chars whose line was a marker (pass-through)
/// - `marker_chars_added`: marker chars we emitted during cleaning
/// - `original_chars_for_badness`: back-compat badness-scoring input
/// - `sum_kept_line_content_chars`: back-compat badness-scoring output
#[pyfunction]
#[pyo3(signature = (text, scripts_to_keep, min_chars_for_comment=None, enable_latex_repetition_crop=false, latex_char_threshold=30, latex_line_threshold=3))]
pub fn clean_text_with_stats(
    py: Python<'_>,
    text: &str,
    scripts_to_keep: Vec<String>,
    min_chars_for_comment: Option<usize>,
    enable_latex_repetition_crop: bool,
    latex_char_threshold: usize,
    latex_line_threshold: usize,
) -> PyResult<(String, PyObject)> {
    use pyo3::types::PyDict;
    // Wave-2 (2026-04-23): optional LaTeX repetition cropping runs
    // BEFORE the cleaner's main passes, so OCR-hallucinated repetitions
    // inside `$$…$$` segments are truncated before any other pass sees
    // them. Off by default — caller opts in.
    let preprocessed: String;
    let text_ref: &str = if enable_latex_repetition_crop {
        preprocessed = crate::latex_module::crop_latex_repetitions(
            text,
            true,
            latex_char_threshold,
            latex_line_threshold,
        );
        &preprocessed
    } else {
        text
    };
    let (allowed_chars, unusual_chars) = build_script_char_sets(&scripts_to_keep);
    let (cleaned_string, stats) = core_clean_text_with_stats(
        text_ref,
        &allowed_chars,
        &unusual_chars,
        min_chars_for_comment,
    );
    let dict = PyDict::new(py);
    dict.set_item("content_chars_kept", stats.content_chars_kept)?;
    dict.set_item("chars_dropped_by_line_drop", stats.chars_dropped_by_line_drop)?;
    dict.set_item("chars_dropped_by_normalization", stats.chars_dropped_by_normalization)?;
    dict.set_item("chars_dropped_by_per_char_filter", stats.chars_dropped_by_per_char_filter)?;
    dict.set_item("lines_dropped_count", stats.lines_dropped_count)?;
    dict.set_item("marker_chars_passthrough", stats.marker_chars_passthrough)?;
    dict.set_item("marker_chars_added", stats.marker_chars_added)?;
    dict.set_item("original_chars_for_badness", stats.original_chars_for_badness)?;
    dict.set_item("sum_kept_line_content_chars", stats.sum_kept_line_content_chars)?;
    Ok((cleaned_string, dict.into()))
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
    pub badness_score_all_chars: Option<f64>, // New score based on all chars in processed lines
    pub badness_score_non_ws: Option<f64>,    // Existing badness score, now explicitly named
}

// Internal function to perform text analysis and return the SLIMMED DOWN struct
pub fn perform_text_analysis(
    text: &str,
    allowed_chars_ref: &HashSet<char>,
    unusual_chars_ref: &HashSet<char>,
    scripts_for_percentage_and_specific_counts: &[String],
    calculate_specific_counts: bool,
    min_chars_for_comment: Option<usize>,
) -> SlimTextAnalysisResult {
    let original_total_chars_abs = text.chars().count();
    let original_non_whitespace_chars_abs = text.chars().filter(|c| !c.is_whitespace()).count();

    let (cleaned_text, original_chars_processed_lines, kept_chars_processed_lines) =
        core_clean_text(
            text,
            allowed_chars_ref,
            unusual_chars_ref,
            min_chars_for_comment,
        );

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
        let metrics = scan_script_metrics(&cleaned_text);
        let include_greek = scripts_for_percentage_and_specific_counts
            .iter()
            .any(|script| script == "greek");
        let include_latin = scripts_for_percentage_and_specific_counts
            .iter()
            .any(|script| script == "latin");

        if include_greek {
            greek_char_count_cleaned = Some(metrics.greek_char_count as usize);
        }
        if include_latin {
            latin_char_count_cleaned = Some(metrics.latin_char_count as usize);
        }
        cleaned_non_whitespace_chars_val = Some(metrics.non_whitespace_chars as usize);
    } else {
        cleaned_non_whitespace_chars_val =
            Some(cleaned_text.chars().filter(|c| !c.is_whitespace()).count());
    }

    // Calculate badness_score_non_ws
    let removed_non_whitespace_chars = original_non_whitespace_chars_abs
        .saturating_sub(cleaned_non_whitespace_chars_val.unwrap_or(0));
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
pub fn analyze_text(
    py: Python,
    text: &str,
    scripts_to_keep: Vec<String>,
    calculate_specific_counts: bool,
    min_chars_for_comment: Option<usize>,
) -> PyResult<HashMap<String, PyObject>> {
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
        min_chars_for_comment,
    );

    let mut results = HashMap::new();
    results.insert(
        "original_total_chars".to_string(),
        analysis_result.original_total_chars.to_object(py),
    );
    results.insert(
        "cleaned_total_chars".to_string(),
        analysis_result.cleaned_total_chars.to_object(py),
    );
    results.insert(
        "original_non_whitespace_chars".to_string(),
        analysis_result
            .original_non_whitespace_chars
            .unwrap_or(0)
            .to_object(py),
    );
    results.insert(
        "cleaned_non_whitespace_chars".to_string(),
        analysis_result
            .cleaned_non_whitespace_chars_after_clean
            .unwrap_or(0)
            .to_object(py),
    );

    // The definition of removed_chars_count for the old badness_score was (original_total_chars - cleaned_total_chars).
    // Let's keep that for a general removed count, and use specific badness scores from analysis_result.
    let removed_chars_count_total = analysis_result
        .original_total_chars
        .saturating_sub(analysis_result.cleaned_total_chars);
    results.insert(
        "removed_chars_count_total".to_string(),
        removed_chars_count_total.to_object(py),
    );

    // Add the two badness scores
    results.insert(
        "badness_score_all_chars".to_string(),
        analysis_result
            .badness_score_all_chars
            .unwrap_or(0.0)
            .to_object(py),
    );
    results.insert(
        "badness_score_non_ws".to_string(),
        analysis_result
            .badness_score_non_ws
            .unwrap_or(0.0)
            .to_object(py),
    );

    // The old "badness_score" key used original_total_chars - cleaned_total_chars / original_non_whitespace_chars.
    // This is different from badness_score_non_ws if removed whitespace is significant.
    // For clarity, I am only exposing the two new specific badness scores.
    // If the old one is critical, it can be re-calculated here.

    if calculate_specific_counts {
        if let Some(greek_count) = analysis_result.greek_char_count_after_clean {
            results.insert("greek_chars_cleaned".to_string(), greek_count.to_object(py));
            if analysis_result
                .cleaned_non_whitespace_chars_after_clean
                .unwrap_or(0)
                > 0
            {
                let percentage_greek = greek_count as f64
                    / analysis_result
                        .cleaned_non_whitespace_chars_after_clean
                        .unwrap_or(1) as f64
                    * 100.0;
                results.insert(
                    "percentage_greek_cleaned".to_string(),
                    percentage_greek.to_object(py),
                );
            } else {
                results.insert("percentage_greek_cleaned".to_string(), 0.0.to_object(py));
            }
        }
        if let Some(latin_count) = analysis_result.latin_char_count_after_clean {
            results.insert("latin_chars_cleaned".to_string(), latin_count.to_object(py));
            if analysis_result
                .cleaned_non_whitespace_chars_after_clean
                .unwrap_or(0)
                > 0
            {
                let percentage_latin = latin_count as f64
                    / analysis_result
                        .cleaned_non_whitespace_chars_after_clean
                        .unwrap_or(1) as f64
                    * 100.0;
                results.insert(
                    "percentage_latin_cleaned".to_string(),
                    percentage_latin.to_object(py),
                );
            } else {
                results.insert("percentage_latin_cleaned".to_string(), 0.0.to_object(py));
            }
        }
    }

    results.insert(
        "cleaned_text".to_string(),
        analysis_result.cleaned_text_content.to_object(py),
    );

    Ok(results)
}

/// Python-exposed function to list available script keys
#[pyfunction]
pub fn list_available_scripts() -> PyResult<Vec<String>> {
    Ok(SCRIPT_SETS
        .keys()
        .filter(|&k| **k != *"unusual")
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_allowed_chars() -> HashSet<char> {
        let mut allowed_chars = HashSet::new();
        for key in ["greek", "latin", "punctuation", "numbers", "common_symbols"] {
            if let Some(script_set) = SCRIPT_SETS.get(key) {
                allowed_chars.extend(script_set);
            }
        }
        allowed_chars.insert(' ');
        allowed_chars.insert('\t');
        allowed_chars.insert('\n');
        allowed_chars
    }

    #[test]
    fn core_clean_text_decoded_glyph_tag_stripped_keeps_prose() {
        // Wave-2 (Case 7): entity-decode + GLYPH-strip pre-passes mean
        // GLYPH<...> markers (even when entity-encoded) are removed
        // inline, leaving the surrounding prose. Old behavior: line-drop
        // with marker. New behavior: keep prose.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "prefix GLYPH&lt;c=3,font=/QCMXYA+CenturyGothic&gt; suffix\n";
        let (cleaned, _, _) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // GLYPH<...> deleted → "prefix  suffix\n" (extra space collapsed
        // by whitespace normalize). The /uni-style font-name path is
        // also covered by the same regex.
        assert!(cleaned.contains("prefix"), "got {:?}", cleaned);
        assert!(cleaned.contains("suffix"), "got {:?}", cleaned);
        assert!(!cleaned.contains("GLYPH"), "got {:?}", cleaned);
        assert!(!cleaned.contains("&lt;"), "got {:?}", cleaned);
    }

    #[test]
    fn core_clean_text_normalizes_long_dot_leaders_without_badness_penalty() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Chapter ..........................................  85\n";
        let (cleaned, original_chars, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // Tiered bucket: 42 dots (>10) → 20 dots; 2 spaces → 1 space.
        assert_eq!(cleaned, "Chapter .................... 85\n");
        assert_eq!(original_chars, input.trim_end_matches('\n').chars().count());
        assert_eq!(kept_chars, original_chars);
    }

    #[test]
    fn core_clean_text_bare_glyph_code_stripped_keeps_prose() {
        // Wave-2 (Case 7): GLYPH<\d+> deleted, prose preserved.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "prefix GLYPH<236> suffix\n";
        let (cleaned, _, _) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("prefix"), "got {:?}", cleaned);
        assert!(cleaned.contains("suffix"), "got {:?}", cleaned);
        assert!(!cleaned.contains("GLYPH"), "got {:?}", cleaned);
    }

    #[test]
    fn core_clean_text_rejects_bare_glyph_word() {
        // Bare `GLYPH` — structural trigger, line-drop with marker.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "some text with GLYPH in the middle\n";
        let (cleaned, _, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, format!("{LINE_REMOVED_COMMENT}\n"));
        assert_eq!(kept_chars, 0);
    }

    #[test]
    fn core_clean_text_span_strips_ps_uni_glyph_names_in_prose() {
        // Per 2026-04-22 Gemini wave: /uni<hex> is now SPAN-stripped, not
        // line-rejected, when coverage predicate is not met. Matches
        // should disappear; surrounding prose should remain.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "foo /uni03B1 /uni03B2 /uni03B3 bar\n";
        let (cleaned, _, _) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // Prose "foo" + "bar" survive; /uni spans removed.
        assert!(cleaned.contains("foo"), "got {:?}", cleaned);
        assert!(cleaned.contains("bar"), "got {:?}", cleaned);
        assert!(!cleaned.contains("/uni03B1"), "got {:?}", cleaned);
    }

    #[test]
    fn core_clean_text_dense_rule_b_matches_now_stripped_to_empty() {
        // Wave-2 (Case 7): the wave-2 GLYPH/uni/gN strip pre-pass
        // deletes ALL `/g<N>` markers up front. A line that was
        // entirely `/g<N>` tokens reduces to whitespace + becomes
        // empty post-strip; the cleaner emits the line-removed
        // marker for the now-empty content. Old behavior used
        // rule-B density to decide; new behavior is even stricter
        // (markers gone unconditionally).
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "/g302/g544/g306/g542/g304/g538/g652/g305/g536/g545/g541/g547\n";
        let (cleaned, _, _) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // After strip, line is empty/whitespace → either dropped
        // entirely or replaced by line-removed marker. Either way,
        // no /g<N> tokens survive.
        assert!(!cleaned.contains("/g"), "got {:?}", cleaned);
    }

    #[test]
    fn core_clean_text_rejects_pdf_font_subset_form() {
        // /<SUBSET>+<FontName> — Adobe font-subset convention, still
        // line-drop (user classified ✅ not-flagged).
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Text /XQDMQS+CenturyGothic in it.\n";
        let (cleaned, _, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, format!("{LINE_REMOVED_COMMENT}\n"));
        assert_eq!(kept_chars, 0);
    }

    #[test]
    fn core_clean_text_span_strips_ps_glyph_literals_in_prose() {
        // `/hyphenminus /space /period ...` — rule A literals. Per
        // 2026-04-22 Gemini wave: SPAN-strip unconditionally; surrounding
        // prose (here "hello"/"world") should remain.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        for input in [
            "foo /hyphenminus bar\n",
            "x /space y\n",
            "a /period b\n",
        ] {
            let (cleaned, _, _) =
                core_clean_text(input, &allowed_chars, &unusual_chars, None);
            assert_ne!(cleaned, format!("{LINE_REMOVED_COMMENT}\n"),
                       "rule A literals should not line-drop: {:?} → {:?}",
                       input, cleaned);
            assert!(!cleaned.contains("/hyphenminus") && !cleaned.contains("/space")
                    && !cleaned.contains("/period"),
                    "rule A literal should be stripped from {:?} → {:?}", input, cleaned);
        }
    }

    #[test]
    fn core_clean_text_bare_hyphenminus_still_line_drops() {
        // `hyphenminus` (no slash) is a BAD_LINE_AC structural trigger —
        // still line-drops per user-review classification ✅.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "hello hyphenminus world\n";
        let (cleaned, _, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, format!("{LINE_REMOVED_COMMENT}\n"));
        assert_eq!(kept_chars, 0);
    }

    #[test]
    fn core_clean_text_does_not_reject_legitimate_slash_word() {
        // Guard: /united-nations /university-of-X in URLs must survive.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "See https://example.com/united-nations/report.\n";
        let (cleaned, _, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_ne!(cleaned, format!("{LINE_REMOVED_COMMENT}\n"));
        assert!(kept_chars > 0);
    }

    #[test]
    fn core_clean_text_strips_lrm_rlm_direction_marks() {
        // LRM (U+200E) and RLM (U+200F) are invisible bidi-direction marks
        // inserted by MediaWiki around foreign-language translations. They
        // have no semantic purpose in Greek text and must strip.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Μεσοπόλεμος (λατινικά: Interbellum\u{200E}\u{200E}, γερμανικά: Zwischenkriegszeit\u{200F})\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(!cleaned.contains('\u{200E}'));
        assert!(!cleaned.contains('\u{200F}'));
        assert!(cleaned.contains("Interbellum"));
        assert!(cleaned.contains("Zwischenkriegszeit"));
    }

    #[test]
    fn core_clean_text_strips_unicode_noise_chars() {
        // Wave-2 changed the order: soft hyphen U+00AD is now stripped
        // by the wave-2 `strip_soft_hyphens` pre-pass (silent invisible
        // strip). U+F0B7 is not in our Adobe Symbol map (passes
        // through preprocessing, then stripped by per-char filter).
        // U+FFFD and U+03A2 (non-existent Greek codepoint) are still
        // stripped by per-char filter.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "A\u{00AD}B \u{F0B7} C\u{FFFD}D \u{03A2}\n";
        let (cleaned, _, _) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // Essential invariants: noise chars all gone, prose letters kept.
        assert!(cleaned.contains('A'));
        assert!(cleaned.contains('B'));
        assert!(cleaned.contains('C'));
        assert!(cleaned.contains('D'));
        assert!(!cleaned.contains('\u{00AD}'));
        assert!(!cleaned.contains('\u{F0B7}'));
        assert!(!cleaned.contains('\u{FFFD}'));
        assert!(!cleaned.contains('\u{03A2}'));
    }

    #[test]
    fn core_clean_text_normalizes_separator_line() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Before\n------\nAfter\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, "Before\n---\nAfter\n");
    }

    #[test]
    fn core_clean_text_normalizes_gfm_table_separator() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "| A | B |\n| :------- | -------: |\n| 1 | 2 |\n";
        let (cleaned, original_chars, kept_chars) =
            core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("| :--- | ---: |"));
        // Semantics-preserving normalization => badness neutral on the
        // separator row (the kept count for that row equals the original).
        assert_eq!(kept_chars, original_chars);
    }

    #[test]
    fn core_clean_text_skips_fenced_code_block() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Four spaces of indentation and `....` inside a fenced block must survive.
        let input = "Prose\n```\n    indented...\n----\n```\nMore prose\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("    indented..."));
        // The `----` inside the fence must NOT collapse to `---`.
        let fence_block: Vec<&str> =
            cleaned.lines().skip_while(|l| !l.starts_with("```")).collect();
        assert!(fence_block.iter().any(|l| *l == "----"));
    }

    #[test]
    fn core_clean_text_folds_math_italic_latin() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Let 𝑥 + 𝑦 = 𝑧.\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, "Let x + y = z.\n");
    }

    #[test]
    fn core_clean_text_normalizes_toc_whitespace_leader_via_bucket() {
        // A TOC line where title and page number are separated by a long
        // whitespace run (PDF table-of-contents layout). The tiered bucket
        // whitespace rule bucketizes the run to 20 spaces (>10 → 20),
        // preserving the visual TOC signal without a TOC-specific heuristic.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Κεφάλαιο 1 Εισαγωγή                              5\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // Original had 30 spaces between "Εισαγωγή" and "5"; tiered → 20.
        let expected = format!("Κεφάλαιο 1 Εισαγωγή{}5\n", " ".repeat(20));
        assert_eq!(cleaned, expected);
    }

    #[test]
    fn core_clean_text_collapses_ellipsis_runs() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "wait……… then\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, "wait… then\n");
    }

    #[test]
    fn core_clean_text_preserves_polytonic_greek() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Polytonic chars in U+1F00..U+2000 must survive; previously they
        // passed through by coincidence, now explicit in the `greek` set.
        let input = "Λόγος πολυτονικός: ἀγαθός, εὐδαιμονία.\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains('ἀ')); // U+1F00 GREEK SMALL LETTER ALPHA WITH PSILI
        assert!(cleaned.contains('ὐ')); // U+1F50 GREEK SMALL LETTER UPSILON WITH PSILI
    }

    #[test]
    fn core_clean_text_preserves_non_greek_latin_scripts() {
        // Policy (2026-04-21): Armenian/Hebrew/Arabic/Georgian carry semantic
        // meaning; Apertus's multilingual training covers them. We preserve
        // them rather than strip.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Greek κείμενο \u{10A0} και \u{0531} και \u{0627} συνεχίζει.\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains('\u{10A0}')); // Georgian letter an
        assert!(cleaned.contains('\u{0531}')); // Armenian capital ayb
        assert!(cleaned.contains('\u{0627}')); // Arabic alef
        assert!(cleaned.contains("Greek"));
        assert!(cleaned.contains("κείμενο"));
    }

    #[test]
    fn core_clean_text_folds_math_greek_to_plain_greek() {
        // Math-italic Greek letters in a Greek corpus are almost always OCR
        // residue of italicized Greek in equations. Fold (not strip) to the
        // regular Greek codepoint Apertus tokenizes efficiently.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Let 𝛼 + 𝛽 = 𝛾 in Greek.\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(cleaned, "Let α + β = γ in Greek.\n");
    }

    #[test]
    fn core_clean_text_composite_roundtrip() {
        // End-to-end exercise: polytonic + math italic + ligature + separator
        // + whitespace run + ellipsis + malformed entity + code fence, all in
        // one document. Asserts the composed behavior.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "\
# Document

Let 𝑥    be defined as follows……

Separator:
--------

The eﬃcient λόγος ἀγαθός &amp 𝐴.

```
    4 spaces stay
----
```

| H1 | H2 |
| :------- | ---: |
| a | b |
";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        // 4-space indented line inside code fence is preserved.
        assert!(cleaned.contains("    4 spaces stay"));
        // `----` inside code fence is NOT collapsed.
        assert!(cleaned.lines().any(|l| l == "----"));
        // Outside the fence, separator collapses.
        assert!(cleaned.contains("\n---\n"));
        // Math italic folds to ASCII. Tiered whitespace: 4 spaces → 5.
        assert!(cleaned.contains("Let x     be"));
        // Original 4-space run no longer present.
        assert!(!cleaned.contains("x    be"));
        // Ligature folds.
        assert!(cleaned.contains("efficient"));
        // Polytonic preserved.
        assert!(cleaned.contains("λόγος"));
        assert!(cleaned.contains("ἀγαθός"));
        // Ellipsis collapses to single.
        assert!(cleaned.contains("follows…"));
        assert!(!cleaned.contains("follows……"));
        // Malformed entity fallback.
        assert!(cleaned.contains("& A"));
        assert!(!cleaned.contains("&amp"));
        // GFM table separator normalized.
        assert!(cleaned.contains("| :--- | ---: |"));
    }

    // -----------------------------------------------------------------
    // Char accounting regression suite (added 2026-04-22)
    // -----------------------------------------------------------------

    /// Helper: assert the INPUT-side char accounting invariant:
    ///   input_chars ≈ content_kept + line_drop + normalize + per_char
    ///                 + marker_chars_passthrough
    /// marker_chars_added is NOT part of input — those are chars we emitted
    /// into output that weren't in the input (LINE_REMOVED_COMMENT, inline
    /// TMC additions, etc.).
    fn assert_accounting_invariant(input: &str, stats: &CleanStats) {
        let input_chars = input
            .lines()
            .map(|l| l.chars().count())
            .sum::<usize>();
        let accounted = stats.content_chars_kept
            + stats.chars_dropped_by_line_drop
            + stats.chars_dropped_by_normalization
            + stats.chars_dropped_by_per_char_filter
            + stats.marker_chars_passthrough;
        // Entity decoding can shrink chars (`&amp;` 5→1) so accounting may
        // undercount slightly. We only assert we don't OVER-count.
        assert!(
            accounted <= input_chars + 2, // small slack for edge cases
            "accounting overshoot: input={input_chars} accounted={accounted} stats={stats:?}"
        );
        // And that we don't massively undercount either.
        // Wave-2: preprocessing passes (entity decode, GLYPH strip,
        // soft-hyphen, paragraph reflow) can each subtract chars
        // counted ONCE in `chars_dropped_by_normalization`, but
        // input_chars is the original length. So undercount slack
        // needs to allow for substantial pre-pass deletions. Use a
        // generous fraction-based slack.
        let slack = (input_chars / 10).max(20);
        assert!(
            accounted + slack >= input_chars,
            "accounting undershoot: input={input_chars} accounted={accounted} slack={slack} stats={stats:?}"
        );
    }

    #[test]
    fn accounting_clean_greek_text_goes_to_content_kept() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Καλημέρα κόσμε.\nΚαι πάλι.\n";
        let (_cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(stats.chars_dropped_by_line_drop, 0);
        assert_eq!(stats.lines_dropped_count, 0);
        assert_eq!(stats.marker_chars_passthrough, 0);
        assert_eq!(stats.marker_chars_added, 0);
        assert!(stats.content_chars_kept > 0);
        assert_accounting_invariant(input, &stats);
    }

    #[test]
    fn accounting_line_drop_bumps_counter_and_chars() {
        // Wave-2: changed trigger from `/g<N>` density (now stripped
        // up-front by Case 7 GLYPH-marker pass) to a confirmed
        // line-drop trigger — `hyphenminus` literal is still in the
        // RULE_A literal set and reliably triggers line removal.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "Καλημέρα.\nhyphenminus hyphenminus hyphenminus hyphenminus hyphenminus\nΕπίλογος.\n";
        let (cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert!(stats.lines_dropped_count >= 1,
                "expected at least one line drop, got {stats:?}");
        assert!(stats.chars_dropped_by_line_drop > 0,
                "expected line-drop chars, got {stats:?}");
        assert!(cleaned.contains(LINE_REMOVED_COMMENT));
        let marker_chars = LINE_REMOVED_COMMENT.chars().count();
        assert!(stats.marker_chars_added >= marker_chars,
                "LINE_REMOVED_COMMENT should be in marker_chars_added: {stats:?}");
        assert_eq!(stats.marker_chars_passthrough, 0,
                "no pass-through markers in this input: {stats:?}");
    }

    #[test]
    fn accounting_content_chars_excludes_line_removed_marker() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "hello hyphenminus world\n";
        let (cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        // The whole line dropped — content_chars_kept MUST be 0 even
        // though the output has the marker in it.
        assert_eq!(stats.content_chars_kept, 0);
        assert!(cleaned.contains(LINE_REMOVED_COMMENT));
        assert_eq!(stats.marker_chars_added, LINE_REMOVED_COMMENT.chars().count());
        assert_eq!(stats.marker_chars_passthrough, 0);
    }

    #[test]
    fn accounting_normalization_tracks_separator_collapse() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // 20-dash separator → collapses to `---` (3 chars). normalize delta
        // should reflect the reduction.
        let input = "hello\n--------------------\nworld\n";
        let (cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("\n---\n"), "expected collapsed separator, got {cleaned:?}");
        assert!(stats.chars_dropped_by_normalization >= 17,
                "expected ≥17 normalization chars dropped, got {stats:?}");
        assert_eq!(stats.chars_dropped_by_line_drop, 0);
        assert_accounting_invariant(input, &stats);
    }

    #[test]
    fn accounting_escaped_underscore_separator_normalizes_to_dash_and_counts() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Markdown-escaped divider (common in EU legislative corpus).
        // Raw bytes: \ _ \ _ \ _ \ _ \ _ = 10 chars; should collapse to `---`.
        let input = "ΠΕΡΙ: ΝΟΜΟΘΕΣΙΑΣ\n\\_\\_\\_\\_\\_\nΑιτιολογική έκθεση.\n";
        let (cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("\n---\n"),
                "escaped-underscore divider should collapse to `---`, got {cleaned:?}");
        assert!(stats.chars_dropped_by_normalization > 0);
        assert_accounting_invariant(input, &stats);
    }

    #[test]
    fn accounting_per_char_filter_tracks_unusual_script_strip() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Cyrillic letters aren't in the default greek/latin/french/spanish
        // allowed set; they're in `unusual` (stripped per SCRIPT_SETS policy).
        let input = "Καλημέρα Здравствуйте.\n";
        let (_cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        // All 12 cyrillic chars should be stripped by the per-char filter.
        assert!(stats.chars_dropped_by_per_char_filter >= 12,
                "expected ≥12 per-char-filter chars dropped, got {stats:?}");
        assert_eq!(stats.chars_dropped_by_line_drop, 0);
        assert_accounting_invariant(input, &stats);
    }

    #[test]
    fn accounting_mixed_doc_invariant_holds() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Wave-2: replaced the dense `/g<N>` line (now stripped early
        // by Case 7) with a `hyphenminus`-density line that still
        // triggers line-drop reliably.
        let input = "Καλημέρα, Здравствуйте.\n\
hyphenminus hyphenminus hyphenminus hyphenminus hyphenminus\n\
--------------------\n\
\\_\\_\\_\\_\\_\n\
Επίλογος.\n";
        let (_cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert!(stats.lines_dropped_count >= 1,
                "expected at least one line drop, got {stats:?}");
        assert!(stats.chars_dropped_by_line_drop > 0);
        assert!(stats.chars_dropped_by_normalization > 0);
        assert!(stats.chars_dropped_by_per_char_filter > 0);
        assert!(stats.content_chars_kept > 0);
        assert_accounting_invariant(input, &stats);
    }

    #[test]
    fn normalization_collapses_whitespace_left_after_cleaning() {
        // Per 2026-04-22 user guidance: normalization runs AFTER cleaning
        // so that when a word/span is stripped, the gap it leaves collapses
        // cleanly. Input: "hello /hyphenminus world" →
        //   after rule-A strip: "hello  world" (2 spaces)
        //   after whitespace-run normalize: "hello world" (1 space)
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let input = "hello /hyphenminus world\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains("hello world\n"),
                "expected single space between hello/world, got {cleaned:?}");
        assert!(!cleaned.contains("hello  world"),
                "double space should have been collapsed, got {cleaned:?}");
    }

    #[test]
    fn normalization_fires_on_inline_tmc_lines_too() {
        // When enough chars are stripped to trigger inline TMC
        // (>=5 unicode-filter removals on the line), normalize should STILL
        // run on the surviving prose so long whitespace runs bucket-collapse.
        // Before 2026-04-22 fix: normalize was skipped whenever
        // line_content_to_add contained TEXT_MISSING_COMMENT, leaving raw
        // 6-space gaps in the output.
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Cyrillic word stripped by per-char filter → 6 consecutive spaces
        // (3 + 3 around the removed word). bucket_run_length(6) = 5, so a
        // 6-space run means normalize didn't fire.
        let input = "Καλημέρα   Здравствуйте   world\n";
        let (cleaned, _, _) = core_clean_text(input, &allowed_chars, &unusual_chars, None);
        assert!(cleaned.contains(TEXT_MISSING_COMMENT));
        assert!(!cleaned.contains("      "),
                "6-space run should have been bucket-collapsed by normalize, got {cleaned:?}");
    }

    #[test]
    fn accounting_rule_a_span_strip_goes_to_per_char_filter_not_line_drop() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        // Rule A literals `/hyphenminus /space` inside otherwise-valid prose.
        // Should span-strip (per-char filter) — NOT line-drop.
        let input = "foo /hyphenminus /space bar\n";
        let (_cleaned, stats) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        assert_eq!(stats.lines_dropped_count, 0);
        assert_eq!(stats.chars_dropped_by_line_drop, 0);
        assert!(stats.chars_dropped_by_per_char_filter >= 18,
                "expected rule-A literals (`/hyphenminus`=12 + `/space`=6) stripped, got {stats:?}");
        assert_accounting_invariant(input, &stats);
    }

    // -----------------------------------------------------------------
    // Performance baseline — regression fence (added 2026-04-22)
    // -----------------------------------------------------------------

    /// Build a representative mixed-content doc: Greek prose, rule-B dense
    /// line, separator, escaped-underscore divider, unusual-script strip,
    /// GFM table, code fence, malformed entities — roughly everything the
    /// cleaner handles in one pass, blown up to ~8 KB.
    fn bench_doc() -> String {
        let block = "\
Καλημέρα κόσμε. Η γλώσσα μας είναι πλούσια. λόγος ἀγαθός.
Η πρόταση περιέχει &amp πολλά &lt σύμβολα.
/g302/g544/g306/g542/g304/g538/g652/g305/g536/g545/g541/g547
foo /hyphenminus /space /period bar /uni03B1 /uni03B2
--------------------
\\_\\_\\_\\_\\_\\_\\_\\_
Καλημέρα Здравствуйте ქართული.
| Column | Value |
| :------- | ---: |
| α | 1 |

Some dots in a row......
```
code fence content     stays
----
```
Επίλογος.
";
        // Blow up ~12x to get a representative ~8 KB doc.
        let mut out = String::with_capacity(block.len() * 12);
        for _ in 0..12 {
            out.push_str(block);
        }
        out
    }

    /// Regression fence: assert cleaner throughput stays above a
    /// conservative minimum. The baseline (measured 2026-04-22 on the
    /// author's laptop, release build, single-threaded) is ~40 M chars/sec
    /// on this mixed-content doc. The threshold below is deliberately well
    /// under that (5 M chars/sec) — this test should only trip on major
    /// regressions, not normal CI-machine variability.
    #[test]
    fn perf_mixed_doc_throughput_floor() {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let doc = bench_doc();
        let doc_chars = doc.chars().count();
        let iterations = 50;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let (_cleaned, _stats) =
                core_clean_text_with_stats(&doc, &allowed_chars, &unusual_chars, None);
        }
        let elapsed = start.elapsed();
        let total_chars = doc_chars * iterations;
        let chars_per_sec = total_chars as f64 / elapsed.as_secs_f64();
        let min_chars_per_sec = 5_000_000.0;
        assert!(
            chars_per_sec >= min_chars_per_sec,
            "throughput regression: {chars_per_sec:.0} chars/sec < {min_chars_per_sec:.0} floor \
             ({total_chars} chars in {:.3}s)",
            elapsed.as_secs_f64(),
        );
        // Print so `cargo test -- --nocapture` shows the actual number.
        eprintln!(
            "[perf] core_clean_text_with_stats: {chars_per_sec:.0} chars/sec ({iterations} x {doc_chars} chars in {:.3}s)",
            elapsed.as_secs_f64(),
        );
    }

    // ------------------------------------------------------------------
    // Phase B end-to-end structural-equivalence regression tests.
    //
    // Runs the full `core_clean_text_with_stats` pipeline (Phase A
    // MD-syntax + Phase B content-modifying) on realistic inputs and
    // asserts `md_verify::verify_md_structural` passes. These catch
    // regressions where the cleaner accidentally drops / reorders /
    // fuses content in ways that violate the "output tokens are a
    // monotone subsequence of input tokens" invariant.
    //
    // Phase B safeguards: docs with MD-syntax chars (`|`, `#`, `---`)
    // in syntactic positions must still have those chars in the
    // cleaner output. Regression net against future per-char-filter
    // misconfigurations.
    //
    // See `docs/MD_MODULE_ARCHITECTURE.md`.
    // ------------------------------------------------------------------

    fn run_full_cleaner(input: &str) -> String {
        let allowed_chars = default_allowed_chars();
        let unusual_chars = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let (cleaned, _) =
            core_clean_text_with_stats(input, &allowed_chars, &unusual_chars, None);
        cleaned
    }

    #[test]
    fn phase_b_structural_equiv_on_simple_prose() {
        let input = "# Title\n\nFirst paragraph of Greek prose. Δεύτερη πρόταση.\n\nSecond paragraph.\n";
        let out = run_full_cleaner(input);
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(
            r.is_structural_equivalent(),
            "structural equivalence violated: {:?}",
            r
        );
    }

    #[test]
    fn phase_b_structural_equiv_on_entity_decode_and_glyph_strip() {
        // Phase B deletions (GLYPH strip) + Phase A entity decode.
        // Both change raw chars. Structural subsequence should still hold.
        let input = "# Heading\n\nΗ εργασία &amp; GLYPH<216> αναφέρεται.\n";
        let out = run_full_cleaner(input);
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(
            r.is_structural_equivalent(),
            "{:?}",
            r
        );
        assert!(r.token_retention_pct < 1.0, "expected some tokens dropped");
    }

    #[test]
    fn phase_b_table_cells_preserved_after_cleaning() {
        let input = "| Col A | Col B | Col C |\n| ---- | ---- | ---- |\n| α | β | γ |\n| 1 | 2 | 3 |\n";
        let out = run_full_cleaner(input);
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(r.table_cells_subsequence, "{:?}", r);
    }

    #[test]
    fn phase_b_mixed_content_doc_passes_structural() {
        let input = concat!(
            "# Top heading\n\n",
            "First soft-wrapped\nparagraph of Greek prose.\n\n",
            "| a | b |\n| ---------- | ---------- |\n| α | β |\n\n",
            "----------\n\n",
            "## Section two\n\n",
            "- item alpha\n- item beta\n\n",
            "Final paragraph.\n"
        );
        let out = run_full_cleaner(input);
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(r.is_structural_equivalent(), "{:?}", r);
    }

    // --- Phase B safeguards: MD-syntax chars must survive ---

    #[test]
    fn phase_b_preserves_heading_marker() {
        let input = "# My Heading\n\nbody text.\n";
        let out = run_full_cleaner(input);
        assert!(
            out.contains("# My Heading") || out.contains("#My Heading"),
            "heading `#` stripped by Phase B: output={:?}",
            out
        );
    }

    #[test]
    fn phase_b_preserves_table_pipes() {
        let input = "| col1 | col2 |\n| --- | --- |\n| a | b |\n";
        let out = run_full_cleaner(input);
        // Count pipes — should have the same structural count.
        let in_pipes = input.chars().filter(|&c| c == '|').count();
        let out_pipes = out.chars().filter(|&c| c == '|').count();
        assert_eq!(
            in_pipes, out_pipes,
            "table pipes lost: in={} out={} output={:?}",
            in_pipes, out_pipes, out
        );
    }

    #[test]
    fn phase_b_preserves_hr_thematic_break() {
        let input = "before\n\n---\n\nafter\n";
        let out = run_full_cleaner(input);
        assert!(
            out.contains("---"),
            "HR `---` stripped by Phase B: output={:?}",
            out
        );
    }

    #[test]
    fn phase_b_preserves_fenced_code_backticks() {
        let input = "before\n\n```\ncode body\n```\n\nafter\n";
        let out = run_full_cleaner(input);
        let in_fences = input.matches("```").count();
        let out_fences = out.matches("```").count();
        assert_eq!(
            in_fences, out_fences,
            "fenced code markers lost: in={} out={} output={:?}",
            in_fences, out_fences, out
        );
    }

    #[test]
    fn phase_b_preserves_list_markers() {
        let input = "- alpha item\n- beta item\n- gamma item\n";
        let out = run_full_cleaner(input);
        // Each `- ` at line start must survive.
        let in_markers = input.lines().filter(|l| l.starts_with("- ")).count();
        let out_markers = out.lines().filter(|l| l.starts_with("- ")).count();
        assert_eq!(
            in_markers, out_markers,
            "list markers dropped: in={} out={} output={:?}",
            in_markers, out_markers, out
        );
    }

    #[test]
    fn phase_b_preserves_blockquote_markers() {
        let input = "> quoted text\n> continued\n";
        let out = run_full_cleaner(input);
        assert!(out.contains(">"), "blockquote marker dropped: {:?}", out);
    }

    #[test]
    fn phase_b_v6_11_nbsp_does_not_fuse_words() {
        // v6-11 regression: Docling emits NBSP (U+00A0) as the default
        // word-separator on many PDFs. Prior cleaner stripped it as
        // "unusual Latin-1 Supplement char", fusing Greek words into
        // 70+ char blobs. Fix (2026-04-24): fold_codepoint now folds
        // U+00A0 → U+0020 so downstream sees real whitespace.
        let input = "Η\u{00A0}εργασία\u{00A0}αυτή\u{00A0}έχει\u{00A0}σκοπό.\n";
        let out = run_full_cleaner(input);
        // After the fix, words should still be separated by whitespace.
        assert!(
            out.contains("Η εργασία")
                || out.contains("Η\u{00A0}εργασία"),
            "NBSP fusion regressed — output lost word separator: {:?}",
            out
        );
        // Structural subsequence should hold (no fusion in token space).
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(
            r.is_structural_equivalent(),
            "NBSP doc should pass structural equivalence: {:?}",
            r
        );
        assert_ne!(
            r.subsequence_failure_kind.as_deref(),
            Some("fusion"),
            "should NOT be classified as fusion anymore: {:?}",
            r
        );
    }

    // -----------------------------------------------------------------
    // Commit 11 RED test — optional-pipe GFM tables through full cleaner.
    //
    // Per the reviewer, this Markdown is a valid GFM table:
    //   a | b
    //   --- | ---
    //   1 | 2
    // Today's cleaner ordering (reflow first, table-sep canonicalization
    // much later) means `--- | ---` is NOT detected as a hard break by
    // `line_is_hard_break` (doesn't start/end with pipe, doesn't match
    // SEPARATOR_LINE_REGEX). Reflow joins it with the header → table
    // destroyed before the GFM-sep pass ever sees it.
    //
    // Fix in Commit 13: route cleaner through md_module::normalize_md_syntax
    // as single Phase A entrypoint so GFM-sep canonicalization runs BEFORE
    // reflow.
    // -----------------------------------------------------------------

    #[test]
    fn red_until_c13_optional_pipe_gfm_table_survives_full_cleaner() {
        let input = "a | b\n--- | ---\n1 | 2\n";
        let out = run_full_cleaner(input);
        // Structural equivalence: block count + tokens. Passes only if
        // the cleaner preserves the table as a table (header row,
        // separator row, body row).
        let r = crate::md_verify::verify_md_structural(input, &out);
        assert!(
            r.is_structural_equivalent(),
            "optional-pipe GFM table destroyed by cleaner — reflow fused \
             rows before table-sep canonicalization. Fix in Commit 13: \
             route cleaner through md_module::normalize_md_syntax as \
             single Phase A entrypoint. out={:?} report={:?}",
            out, r
        );
    }

    #[test]
    fn phase_b_other_unicode_spaces_also_preserved() {
        // Narrow NBSP / thin space / em space / etc. all fold to
        // regular space so word boundaries survive.
        let input = "alpha\u{2009}beta\u{202F}gamma\u{2003}delta\n";
        let out = run_full_cleaner(input);
        // All four words should still appear as distinct tokens.
        for word in ["alpha", "beta", "gamma", "delta"] {
            assert!(
                out.contains(word),
                "word `{}` lost in output: {:?}",
                word,
                out
            );
        }
        // Words should be whitespace-separated (not fused).
        assert!(!out.contains("alphabeta"), "fusion across thin space: {:?}", out);
    }

    // -----------------------------------------------------------------
    // Commit 14 — shared non-destructive canonicalization.
    //
    // The verifier's `canonicalize_for_verify` now delegates to
    // `md_module::non_destructive_canonicalize`. These tests assert the
    // invariant that drove the extraction: on inputs where the cleaner
    // would delete nothing, cleaner output must equal canonicalize
    // output. Future drift in either code path trips this gate.
    // -----------------------------------------------------------------

    /// Permissive allowed-set covering everything in the default
    /// script sets plus the 0..=127 ASCII range, so the cleaner has
    /// nothing to drop at the per-char filter on the sample inputs
    /// used by the drift-prevention tests.
    fn permissive_allowed_chars() -> HashSet<char> {
        let mut allowed = default_allowed_chars();
        for ch in 0u32..=127 {
            if let Some(c) = char::from_u32(ch) {
                allowed.insert(c);
            }
        }
        allowed
    }

    fn assert_cleaner_matches_canonicalize(input: &str) {
        let allowed = permissive_allowed_chars();
        let unusual = SCRIPT_SETS.get("unusual").cloned().unwrap_or_default();
        let (cleaned, _stats) =
            core_clean_text_with_stats(input, &allowed, &unusual, None);
        let canonical = md_module::non_destructive_canonicalize(input);
        assert_eq!(
            cleaned.trim_end_matches('\n'),
            canonical.trim_end_matches('\n'),
            "cleaner output diverged from non_destructive_canonicalize:\n\
             input={:?}\ncleaner={:?}\ncanonical={:?}",
            input,
            cleaned,
            canonical,
        );
    }

    #[test]
    fn drift_cleaner_eq_canonicalize_on_plain_prose() {
        assert_cleaner_matches_canonicalize(
            "Η εργασία αυτή έχει σκοπό την περιγραφή.\n",
        );
    }

    #[test]
    fn drift_cleaner_eq_canonicalize_on_optional_pipe_table() {
        assert_cleaner_matches_canonicalize("a | b\n--- | ---\n1 | 2\n");
    }

    #[test]
    fn drift_cleaner_eq_canonicalize_on_hr_collapse_with_adjacent_prose() {
        assert_cleaner_matches_canonicalize(
            "before paragraph.\n\n----------\n\nafter paragraph.\n",
        );
    }

    #[test]
    fn drift_cleaner_eq_canonicalize_on_soft_wrapped_paragraph() {
        assert_cleaner_matches_canonicalize(
            "first soft-wrapped\npiece of content\nhere.\n",
        );
    }

    #[test]
    fn drift_cleaner_eq_canonicalize_on_gfm_table_and_hr() {
        assert_cleaner_matches_canonicalize(concat!(
            "# Heading\n\n",
            "| A | B |\n| ---------- | ---------- |\n| 1 | 2 |\n\n",
            "----------\n\n",
            "Paragraph soft-wrap\nacross two lines.\n",
        ));
    }
}
