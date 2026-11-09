//! CLI adapter: retain the audited Python per-file deadline and JSONL scheduler.
//! The workers call this crate's PyO3 core, never Python PDF/font repair parsing.
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !matches!(
        args.first().map(String::as_str),
        Some("classify" | "repair")
    ) {
        eprintln!("usage: pdfrepair classify --pdfs PDF_OR_DIR --out REPORT | repair --pdfs PDF_OR_DIR --out-dir DIR --report REPORT");
        std::process::exit(2);
    }
    let python = std::env::var("GLOSSAPI_PDFREPAIR_PYTHON").unwrap_or_else(|_| "python3".into());
    let result = std::process::Command::new(python)
        .args(["-m", "glossapi.pdf_repair.rust_backend", "--cli"])
        .args(args)
        .status();
    match result {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1)
        }
    }
}
