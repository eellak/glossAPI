# Lightweight PDF Corpus

GlossAPI ships two single-page PDFs that exercise the extraction pipeline without Docling or GPU dependencies. Use them
to validate a local install, reproduce regressions, or give new contributors instant feedback.

- Location: `samples/lightweight_pdf_corpus/`
- Inputs: `pdfs/alpha.pdf` and `pdfs/beta.pdf`
- Baseline outputs: `expected_outputs.json`, the Markdown lines the safe PyPDFium backend produces for each stem

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

Compare `artifacts/lightweight_pdf_run/markdown/` with `samples/lightweight_pdf_corpus/expected_outputs.json` to
confirm the extractor is stable.

## Where it is used

- `tests/test_corpus_phases_lightweight.py` copies the two PDFs into a temporary corpus and checks safe extraction
  against `expected_outputs.json`, cleaning (it skips when the native cleaner is not installed) and sectioning. It
  needs no GPU or Docling.
- The README and the onboarding guide use it as the first task for new contributors.

The PDFs are checked in as files; there is no generator script. When you change them, update `expected_outputs.json`
in the same commit.
