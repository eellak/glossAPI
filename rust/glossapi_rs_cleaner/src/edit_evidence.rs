//! Optional producer-side edit capture. Off unless a caller requests evidence.
//! Coordinates refer to the surface immediately before each ordered transform.
use std::cell::RefCell;
use std::path::Path;

thread_local! {
    #[allow(clippy::type_complexity)] // (rule, start, end, replacement) edit records
    static EDITS: RefCell<Option<Vec<(String, usize, usize, String)>>> = const { RefCell::new(None) };
}

pub fn enabled() -> bool { EDITS.with(|state| state.borrow().is_some()) }

pub fn begin() {
    EDITS.with(|state| *state.borrow_mut() = Some(Vec::new()));
}

pub fn change(rule: &str, before: &str, after: &str, offset: usize) {
    EDITS.with(|state| {
        let mut state = state.borrow_mut();
        let Some(edits) = state.as_mut() else { return };
        if before == after { return; }
        let left: Vec<char> = before.chars().collect();
        let right: Vec<char> = after.chars().collect();
        let prefix = left.iter().zip(&right).take_while(|(a,b)| a == b).count();
        let suffix = left[prefix..].iter().rev().zip(right[prefix..].iter().rev())
            .take_while(|(a,b)| a == b).count();
        edits.push((rule.to_string(), offset + prefix, offset + left.len() - suffix,
            right[prefix..right.len()-suffix].iter().collect()));
    });
}

pub fn finish(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let edits = EDITS.with(|state| state.borrow_mut().take()).unwrap_or_default();
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(["rule_id", "start", "end", "replacement"])?;
    for (rule, start, end, replacement) in edits {
        writer.write_record([rule, start.to_string(), end.to_string(), replacement])?;
    }
    writer.flush()?;
    Ok(())
}
