# Dependency setup notes

`setup_glossapi.sh --mode docling` creates the layout/extraction environment from
`requirements-glossapi-docling.txt` and installs the Rust extensions in editable mode.
Use `--venv PATH`, `--python PATH` and `--run-tests` to select the environment and
run its complete test suite. External OCR engines are installed and configured
separately; see [OCR plugin migration](../docs/ocr_plugin_migration.md).

Docling owns its Torch and model dependencies. Math/code enrichment remains part
of this environment. The guided installer can select browser, Docling and docs
extras without provisioning a separate OCR runtime.
