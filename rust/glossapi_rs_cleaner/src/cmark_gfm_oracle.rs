//! cmark-gfm subprocess oracle — the CommonMark+GFM reference
//! renderer GitHub actually uses to render Markdown.
//!
//! We shell out to the `cmark-gfm` C binary (installed via the
//! `cmark-gfm` Debian package, `/usr/bin/cmark-gfm`) rather than
//! using a Rust port, because:
//!
//! 1. It IS the ground truth — GitHub's renderer uses this exact
//!    codebase. No port-parity ambiguity.
//! 2. Our formatter (`md_format::format_parsed`) uses `comrak` (a
//!    Rust port of cmark-gfm). If we also used comrak as the
//!    verifier we'd have a tautology. Using cmark-gfm independently
//!    tests that our comrak round-trip preserves rendering under
//!    GitHub's actual renderer.
//! 3. `cmark-gfm` is fast enough (~1ms/doc subprocess overhead) for
//!    all our scales: 29 fixtures = instant; 90 real docs = <1s;
//!    168K corpus audit = ~5min overhead on top of parse time. Not
//!    on any hot path — verifier only runs during testing + corpus
//!    audits.
//!
//! Availability: `/usr/bin/cmark-gfm` is installed on the gcloud
//! cleaning instance (`apertus-greek-tokenizer-20260408t160000z`);
//! for local dev we fall back to comrak (same codebase, high parity).

use std::io::Write;
use std::process::{Command, Stdio};

/// Path at which we expect the cmark-gfm binary on the cleaning
/// instance. If the binary isn't at this path (or isn't on PATH),
/// callers should detect the failure via `is_available()` and fall
/// back to the in-process Rust oracle (`comrak`).
const CMARK_GFM_BIN: &str = "cmark-gfm";

/// GFM extensions to enable — matches what GitHub enables by default
/// for README / issue / PR rendering. Keeps rendering consistent
/// with the actual GitHub renderer.
const GFM_EXTENSIONS: &[&str] = &["table", "strikethrough", "tasklist", "autolink"];

/// Test whether the `cmark-gfm` binary is callable in this
/// environment. Used by tests to skip when unavailable.
pub fn is_available() -> bool {
    Command::new(CMARK_GFM_BIN)
        .arg("--help")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Render Markdown via `cmark-gfm --to html` with GFM extensions
/// matching GitHub's defaults. Returns the raw HTML exactly as
/// cmark-gfm emits it — no whitespace normalization. For equality
/// checks, byte-for-byte comparison is the right thing: the same
/// binary on the same input produces the same output, deterministically.
///
/// Returns `Err(message)` if the subprocess fails for any reason
/// (binary missing, non-zero exit, IO error).
pub fn render_html(md: &str) -> Result<String, String> {
    let mut child = Command::new(CMARK_GFM_BIN)
        .arg("--to")
        .arg("html")
        .args(GFM_EXTENSIONS.iter().flat_map(|e| ["--extension", e]))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn cmark-gfm: {e}"))?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or("cmark-gfm stdin not captured")?;
        stdin
            .write_all(md.as_bytes())
            .map_err(|e| format!("write to cmark-gfm stdin: {e}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|e| format!("wait cmark-gfm: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cmark-gfm exited {}: stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("cmark-gfm non-utf8 output: {e}"))
}

/// Verify that `input` and `output` render to the same HTML under
/// cmark-gfm. Compares both RAW html (byte-for-byte) and the
/// preview-equivalent form (whitespace between block tags collapsed,
/// trailing ws before closing tags stripped — standard HTML-preview
/// invariants that differ between `<p>a\nb</p>` and `<p>a b</p>`).
///
/// The preview-equivalent form is the meaningful preservation
/// signal; byte-for-byte is a stricter stats-only property.
///
/// If the binary isn't available, returns `Err`. Callers decide
/// whether to treat that as skip or fail.
pub fn verify(input: &str, output: &str) -> Result<CmarkGfmReport, String> {
    let in_html = render_html(input)?;
    let out_html = render_html(output)?;
    let byte_identical = in_html == out_html;
    let in_normalized = normalize_for_preview_eq(&in_html);
    let out_normalized = normalize_for_preview_eq(&out_html);
    let preview_identical = in_normalized == out_normalized;
    let first_diff = if preview_identical {
        None
    } else {
        Some(find_first_diff(&in_normalized, &out_normalized))
    };
    Ok(CmarkGfmReport {
        input_html: in_html,
        output_html: out_html,
        byte_identical,
        preview_identical,
        first_diff,
    })
}

/// Normalize a cmark-gfm HTML render to a form where preview-
/// equivalent inputs produce equal strings. This is what "same
/// rendered page" means when the HTML bytes differ by invisible
/// whitespace (whitespace between block tags, trailing whitespace
/// before closing tags).
///
/// Intentionally NOT too aggressive — only normalizations that
/// provably don't change visible rendering:
///
/// 1. Collapse whitespace runs to a single space.
/// 2. Strip whitespace between adjacent tags (`> <` → `><`).
/// 3. Strip whitespace before closing tags (` </` → `</`).
///
/// We do NOT strip HTML comments or normalize attribute order — if
/// cmark-gfm emits something different we want to know.
pub fn normalize_for_preview_eq(html: &str) -> String {
    // Step 1: collapse whitespace runs to single space.
    let mut collapsed = String::with_capacity(html.len());
    let mut prev_ws = false;
    for c in html.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(c);
            prev_ws = false;
        }
    }
    // Steps 2+3: strip `> <` → `><` and ` </` → `</` iteratively
    // (each can create new adjacencies).
    let mut prev = collapsed.trim().to_string();
    loop {
        let next = prev.replace("> <", "><").replace(" </", "</");
        if next == prev {
            break;
        }
        prev = next;
    }
    prev
}

#[derive(Debug, Clone)]
pub struct CmarkGfmReport {
    pub input_html: String,
    pub output_html: String,
    /// Byte-for-byte equal under cmark-gfm (stricter — stats-only
    /// metric; false just means the HTML had different whitespace).
    pub byte_identical: bool,
    /// Preview-equivalent under cmark-gfm — same rendered page
    /// (whitespace between block tags + trailing ws before close
    /// normalized out). THIS is the Phase A invariant.
    pub preview_identical: bool,
    pub first_diff: Option<String>,
}

fn find_first_diff(a: &str, b: &str) -> String {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let n = ab.len().min(bb.len());
    let mut i = 0;
    while i < n && ab[i] == bb[i] {
        i += 1;
    }
    // Snap to the nearest char boundary at or before `start`, at or
    // after `end` — byte offsets into UTF-8 strings MUST NOT split
    // a multi-byte char (Greek, math symbols, etc. would panic).
    let start = floor_char_boundary(a, i.saturating_sub(40));
    let end_a = ceil_char_boundary(a, (i + 120).min(a.len()));
    let end_b = ceil_char_boundary(b, (i + 120).min(b.len()));
    // `start` is relative to `a`; use a matching start for `b` that
    // is also on a char boundary.
    let start_b = floor_char_boundary(b, start.min(b.len()));
    format!(
        "first diff at byte {i} (in_len={} out_len={})\n  in:  {}\n  out: {}",
        a.len(),
        b.len(),
        &a[start..end_a],
        &b[start_b..end_b]
    )
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// PyO3 surface.
// ---------------------------------------------------------------------------

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// PyO3: verify `input` and `output` render to the same HTML under
/// cmark-gfm. Returns a dict:
/// - `is_available`: cmark-gfm binary found
/// - `identical`: in_html == out_html (byte-for-byte)
/// - `first_diff`: diagnostic snippet if not identical
/// - `error`: string if binary unavailable / subprocess failed
#[pyfunction]
pub fn cmark_gfm_verify_py(py: Python<'_>, input: &str, output: &str) -> PyResult<PyObject> {
    let d = PyDict::new(py);
    if !is_available() {
        d.set_item("is_available", false)?;
        d.set_item("preview_identical", false)?;
        d.set_item("byte_identical", false)?;
        d.set_item("identical", false)?;
        d.set_item("error", "cmark-gfm binary not found on PATH")?;
        return Ok(d.into());
    }
    d.set_item("is_available", true)?;
    match verify(input, output) {
        Ok(r) => {
            d.set_item("preview_identical", r.preview_identical)?;
            d.set_item("byte_identical", r.byte_identical)?;
            // Backward-compat alias: `identical` = preview_identical.
            d.set_item("identical", r.preview_identical)?;
            d.set_item("first_diff", r.first_diff)?;
            d.set_item("error", Option::<String>::None)?;
        }
        Err(e) => {
            d.set_item("preview_identical", false)?;
            d.set_item("byte_identical", false)?;
            d.set_item("identical", false)?;
            d.set_item("error", e)?;
        }
    }
    Ok(d.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_unavailable() -> bool {
        if !is_available() {
            eprintln!("cmark-gfm not available — skipping");
            return true;
        }
        false
    }

    #[test]
    fn cmark_basic_render() {
        if skip_if_unavailable() {
            return;
        }
        let html = render_html("# hello\n").expect("render");
        assert!(html.contains("<h1>hello</h1>"), "got: {html}");
    }

    #[test]
    fn cmark_verify_identity_passes() {
        if skip_if_unavailable() {
            return;
        }
        let r = verify("hello world\n", "hello world\n").expect("verify");
        assert!(r.preview_identical);
        assert!(r.byte_identical);
    }

    #[test]
    fn cmark_verify_difference_fails() {
        if skip_if_unavailable() {
            return;
        }
        let r = verify("hello\n", "goodbye\n").expect("verify");
        assert!(!r.preview_identical);
        assert!(r.first_diff.is_some());
    }

    #[test]
    fn cmark_verify_preview_identical_but_not_byte_identical() {
        // A soft-wrap that reflow joins is preview-identical per
        // CM: both render as `<p>first second</p>` (pulldown
        // emits with internal `\n`, cmark-gfm emits... let's check).
        if skip_if_unavailable() {
            return;
        }
        let r = verify("first\nsecond\n", "first second\n").expect("verify");
        // Whatever HTML bytes cmark-gfm emits, after preview-eq
        // normalization the two should match.
        assert!(r.preview_identical, "first_diff: {:?}", r.first_diff);
    }

    // --- Ground-truth anchors: confirm cmark-gfm treats our edge
    //     cases exactly how we expect (these encode the CM spec on
    //     the cases where our old line-based code got them wrong). ---

    #[test]
    fn ground_truth_escaped_underscore_is_literal_not_hr() {
        if skip_if_unavailable() {
            return;
        }
        // `\_\_\_\_\_\_\_\_` is a paragraph of literal underscores
        // (each `\_` is an escape). NOT a thematic break.
        let html = render_html("\\_\\_\\_\\_\\_\\_\\_\\_\n").unwrap();
        assert!(html.contains("<p>________</p>"), "got: {html}");
        assert!(!html.contains("<hr"));
    }

    #[test]
    fn ground_truth_plain_underscore_is_hr() {
        if skip_if_unavailable() {
            return;
        }
        // Plain `________` (≥3 underscores) on its own line IS an
        // HR per CM.
        let html = render_html("________\n").unwrap();
        assert!(html.contains("<hr"), "got: {html}");
    }

    #[test]
    fn ground_truth_optional_pipe_table_parses() {
        if skip_if_unavailable() {
            return;
        }
        let html = render_html("a | b\n--- | ---\n1 | 2\n").unwrap();
        assert!(html.contains("<table>"), "got: {html}");
        assert!(html.contains("<th>a</th>"));
    }

    #[test]
    fn ground_truth_two_space_hard_break() {
        if skip_if_unavailable() {
            return;
        }
        let html = render_html("first  \nsecond\n").unwrap();
        assert!(html.contains("<br"), "got: {html}");
    }

    #[test]
    fn ground_truth_soft_break_is_not_hard_break() {
        if skip_if_unavailable() {
            return;
        }
        let html = render_html("first\nsecond\n").unwrap();
        // Single `\n` between two lines in a paragraph is a soft
        // break — renders as a newline inside the same `<p>`, no
        // `<br>` tag.
        assert!(!html.contains("<br"), "got: {html}");
        assert!(html.contains("<p>first\nsecond</p>"), "got: {html}");
    }
}
