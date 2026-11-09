fn main() {
    // Cargo (unlike maturin) does not add macOS extension-module linker flags.
    if std::env::var_os("CARGO_FEATURE_PYTHON").is_some() {
        pyo3_build_config::add_extension_module_link_args();
    }
}
