//! OS-level I/O failures surface as Python `OSError(errno, message)` (native API level 2, K4g fix 4).
//!
//! Before level 2 the cleaner printed a failed stage-one or edits-CSV write and carried on, and returned every other
//! write failure as `ValueError` text, so the errno was lost: a full disk in clean was typed `clean_process_failed`
//! instead of `disk_full`. Now an `io::Error` that carries an errno (ENOSPC, EDQUOT, EIO, EACCES, ENOENT ...) is
//! raised as `OSError(errno, message)`; Python picks the errno's subclass (e.g. `FileNotFoundError`). An error
//! without an errno (invalid UTF-8 input, a CSV or Parquet format problem, a table-analysis failure) is a content
//! problem and keeps its previous behaviour and text.
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use std::error::Error as StdError;
use std::fmt::Display;
use std::sync::Mutex;

/// The errno of the first `io::Error` in `error`'s cause chain that carries one. A `csv::Error` keeps its `io::Error`
/// in its kind (it has no `source`); a Parquet error keeps it as its source; `io::Error::new(kind, inner)` keeps it
/// as its inner error.
pub(crate) fn errno_of(error: &(dyn StdError + 'static)) -> Option<i32> {
    let mut current = Some(error);
    while let Some(e) = current {
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            if let Some(code) = io.raw_os_error() {
                return Some(code);
            }
            if let Some(code) = io.get_ref().and_then(|inner| errno_of(inner)) {
                return Some(code);
            }
        }
        if let Some(csv) = e.downcast_ref::<csv::Error>() {
            if let csv::ErrorKind::Io(io) = csv.kind() {
                if let Some(code) = errno_of(io) {
                    return Some(code);
                }
            }
        }
        current = e.source();
    }
    None
}

/// `OSError(errno, "<context>: <error>")` for an OS-level failure, otherwise `ValueError` with the same text (the
/// text every call site produced before).
pub(crate) fn py_err(context: impl Display, error: &(dyn StdError + 'static)) -> PyErr {
    let message = format!("{context}: {error}");
    match errno_of(error) {
        Some(code) => PyOSError::new_err((code, message)),
        None => PyValueError::new_err(message),
    }
}

/// A stage wrapper's error: an `OSError` passes through unchanged (its errno is the cause the caller types); any
/// other error becomes the stage's `ValueError` text, as before.
pub(crate) fn stage_err(py: Python<'_>, error: PyErr, wrap: impl FnOnce(&PyErr) -> String) -> PyErr {
    if error.is_instance_of::<PyOSError>(py) {
        error
    } else {
        PyValueError::new_err(wrap(&error))
    }
}

/// The first OS-level failure of a parallel loop, raised after the loop. A per-file loop used to print such a
/// failure and continue, leaving a file or report row missing without an error.
#[derive(Default)]
pub(crate) struct FirstOsError(Mutex<Option<PyErr>>);

impl FirstOsError {
    /// Keep `error` when it is OS-level (has an errno) and none is kept yet. Returns whether it is OS-level.
    pub(crate) fn io(&self, context: impl Display, error: &(dyn StdError + 'static)) -> bool {
        if errno_of(error).is_none() {
            return false;
        }
        self.keep(py_err(context, error));
        true
    }

    /// Keep `error` when it is an `OSError` raised by a per-file operation. Returns whether it is one.
    pub(crate) fn python(&self, py: Python<'_>, error: PyErr) -> Result<(), PyErr> {
        if error.is_instance_of::<PyOSError>(py) {
            self.keep(error);
            Ok(())
        } else {
            Err(error)
        }
    }

    fn keep(&self, error: PyErr) {
        let mut first = self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if first.is_none() {
            *first = Some(error);
        }
    }

    /// `Err(first OS error)` when one was kept.
    pub(crate) fn into_result(self) -> PyResult<()> {
        match self.0.into_inner().unwrap_or_else(|poisoned| poisoned.into_inner()) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn errno_is_found_through_io_csv_and_parquet_wrappers() {
        let full = io::Error::from_raw_os_error(28);
        assert_eq!(errno_of(&full), Some(28));
        let wrapped = io::Error::other(io::Error::from_raw_os_error(122));
        assert_eq!(errno_of(&wrapped), Some(122));
        let csv_error = csv::Error::from(io::Error::from_raw_os_error(28));
        assert_eq!(errno_of(&csv_error), Some(28));
        let parquet_error = parquet::errors::ParquetError::from(io::Error::from_raw_os_error(5));
        assert_eq!(errno_of(&parquet_error), Some(5));
    }

    #[test]
    fn content_errors_have_no_errno() {
        let utf8 = io::Error::new(io::ErrorKind::InvalidData, "stream did not contain valid UTF-8");
        assert_eq!(errno_of(&utf8), None);
        let parquet_error = parquet::errors::ParquetError::General("bad schema".into());
        assert_eq!(errno_of(&parquet_error), None);
        let boxed: Box<dyn StdError> = "Missing column".into();
        assert_eq!(errno_of(boxed.as_ref()), None);
    }

    #[test]
    fn real_write_failures_carry_their_errno() {
        let root = std::env::temp_dir().join(format!("glossapi-cleaner-os-error-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("a-file");
        std::fs::write(&file, "x").unwrap();
        // The parent of the target is a file: ENOTDIR (20), a real OS error from the kernel.
        let error = std::fs::write(file.join("child.md"), "text").unwrap_err();
        assert_eq!(errno_of(&error), Some(20));
        let csv_error = csv::Writer::from_path(file.join("child.csv")).unwrap_err();
        assert_eq!(errno_of(&csv_error), Some(20));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
