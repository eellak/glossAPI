# Lightweight PDF Corpus

GlossAPI ships 20 single-page PDFs that exercise the extraction pipeline without Docling or GPU dependencies. Use them to validate a local install, reproduce regressions, or give new contributors instant feedback.

- Location: `samples/lightweight_pdf_corpus/`
- Inputs: `pdfs/` (pre-generated assets) and `manifest.json`
- Baseline outputs: `expected_outputs.json` (Markdown produced with the safe PyPDFium backend)

## Quick run

```bash
python - <<'PY'
from pathlib import Path
from glossapi import Corpus

input_dir = Path("samples/lightweight_pdf_corpus/pdfs")
output_dir = Path("artifacts/lightweight_pdf_run")
output_dir.mkdir(parents=True, exist_ok=True)

corpus = Corpus(input_dir, output_dir)
corpus.extract(input_format="pdf")
PY
```

Compare `artifacts/lightweight_pdf_run/markdown/` with `samples/lightweight_pdf_corpus/expected_outputs.json` to confirm the extractor is stable.

## Regenerate (optional)

```bash
python samples/lightweight_pdf_corpus/generate_pdfs.py          # writes into ./pdfs
python samples/lightweight_pdf_corpus/generate_pdfs.py --overwrite
```

Regeneration is stdlib-only; `samples/lightweight_pdf_corpus/requirements.txt` is intentionally empty. Commit updated PDFs, manifest, and expected outputs together.

## Where it is used
- `tests/test_pipeline_smoke.py` wires the corpus into a fast regression check.
- Docs and README reference it as the recommended first task for new contributors.
- Feel free to adapt the manifest with new edge cases—`generate_pdfs.py` keeps the assets reproducible.
