fn main() {
    // PyO3 0.30 no longer emits the macOS extension-module linker flags itself (pyo3 0.19 did), so a plain
    // `cargo build` of this cdylib on macOS needs them here; maturin and setuptools-rust pass them anyway.
    pyo3_build_config::add_extension_module_link_args();
}
