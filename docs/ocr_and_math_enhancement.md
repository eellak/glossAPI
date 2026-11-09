# OCR and Math Enrichment

OCR remediation uses the pinned external Greek OCR toolkit, the only supported OCR engine. See
[OCR plugin migration](ocr_plugin_migration.md) for configuration and page routing.
Math/code enrichment stays in the optional Docling layout stack (`python install_glossapi.py layout --device cuda126`,
or `cuda132`). Math enrichment consumes existing Docling JSON from an explicit Docling extraction. The safe PDFium
backend does not create these inputs.
Use `Corpus.ocr(mode="math_only", fix_bad=False)` to enrich existing Docling JSON
without requiring an OCR plugin. `ocr_bad_then_math` OCRs the selected pages (default `page_routing="all"`: every page page quality routes)
and then enriches non-OCR documents with available JSON.

## Running Phase‑1 (Extract)

```python
from glossapi import Corpus
c = Corpus('IN','OUT')

# Emit JSON + formula index for Phase‑2
c.extract(
    input_format='pdf',
    phase1_backend='docling',   # optional layout extra; required for Docling JSON/formula artifacts
    accel_type='CUDA',
    export_doc_json=True,        # json/<stem>.docling.json(.zst) (the default, Docling backend only)
    emit_formula_index=True,     # request json/<stem>.formula_index.jsonl alongside the JSON
)
```

Outputs:
- `markdown/<stem>.md`
- `json/<stem>.docling.json(.zst)` and `json/<stem>.formula_index.jsonl`
- `json/metrics/<stem>.metrics.json` and `json/metrics/<stem>.per_page.metrics.json`

## Running Phase‑2 (Enrich)

```python
from glossapi import Corpus
c = Corpus('OUT','OUT')  # same folder for both

# GPU formula/code decoding from JSON (writes enriched MD to markdown/<stem>.md)
c.formula_enrich_from_json(
    device='cuda',
    batch_size=12,       # tune for your GPU
)
```

Outputs:
- `markdown/<stem>.md` — enriched Markdown overwrites the plain MD
- `json/<stem>.latex_map.jsonl` — LaTeX strings + acceptance/metrics per item

## Early‑stop & Post‑Processing Guards (Formula)

To keep LaTeX well‑formed and fast:
- Generation‑time (applied inside the decoder):
  - `GLOSSAPI_LATEX_EARLYSTOP` = `1|0` (default 1): enable early‑stop criteria for the HF generate path.
  - `GLOSSAPI_LATEX_MAX_CHARS` (default 3000): decoded‑length stop gate.
  - `GLOSSAPI_LATEX_MAX_REPEAT` (default 50): stop on excessive last‑token repetition.
  - `GLOSSAPI_LATEX_LEN_STRIDE` (default 16): decoding stride for the length check.
  - `GLOSSAPI_LATEX_MAX_NEW_TOKENS` (optional): cap new tokens at the decoder level (injected only if caller doesn’t specify one).

- Post‑processing (applied on the generated string):
  - `GLOSSAPI_LATEX_POST_ONLY_FAILED` = `1|0` (default 1): only sanitize when output looks pathological.
  - `GLOSSAPI_LATEX_POST_REPEAT_GATE` (default 50): consider output failed if the last token repeats more than this gate.
  - `GLOSSAPI_LATEX_POST_WINDDOWN` (default 12): clamp the repeated tail token to this run length.
  - `GLOSSAPI_LATEX_POST_MAX_CHARS` (default 3000): cap text length; prefers whitespace/`\` boundary.

The policy is centralized in `glossapi.text_sanitize`. Phase‑2 enrichment and the per‑page metrics both use the same sanitizer so counts and truncation flags are consistent.

## Artifact Placement Summary

```
OUT/
├── markdown/
│   └── <stem>.md                     # enriched Markdown (canonical)
├── json/
│   ├── <stem>.docling.json(.zst)
│   ├── <stem>.formula_index.jsonl
│   ├── <stem>.latex_map.jsonl
│   └── metrics/
│       ├── <stem>.metrics.json
│       └── <stem>.per_page.metrics.json
```
