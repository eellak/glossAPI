# Greek OCR default

`python install_glossapi.py ocr --device cpu` installs the external
`greek-ocr-toolkit[docling]` plugin, Docling and the pinned inference dependencies.
Select `mps` on Apple Silicon for MPS recognition plus CoreML detection, or
`cuda126` / `cuda132` on Linux (including aarch64) for CUDA recognition plus CPU
detection. The installer pins the matching Torch/torchvision pair before resolving
OCR. Standard `onnxruntime` supplies detection; there is no CUDA detector backend.
The device names select wheel sources; available wheels and actual accelerator
execution still require qualification on the target machine.

The sole toolkit source pin is `src/glossapi/ocr_package.txt`. Both the installer
and the dynamic `glossapi[ocr]` wheel extra read that file. Its public GitHub URL
is marked **PENDING-PUBLIC-SHA** until the coordinator publishes the reviewed
release. Replace that one requirement line with the public SHA, then rerun the
CPU smoke. No mirror or glossAPI-side model downloader is used.

For development before publication:

```sh
python install_glossapi.py ocr --device cpu \
  --ocr-toolkit-source 'git+file:///Users/foivoskarounos-zamparloukos/Projects/greek-ocr-toolkit-main-next@138cff524'
# On our Linux machines, use file:///absolute/path/to/commit-bound-archive instead.
# GLOSSAPI_OCR_TOOLKIT_SOURCE supplies the same override without a CLI argument.
```

The override changes the install command, while wheel metadata retains the public
source pin. The Mac qualification used a code-only `git archive` because the full
tracked tree alone exceeds its 2 GB experiment limit; see the integration notes.

After extraction and cleaning:

```python
corpus.ocr(mode='ocr_bad')
# Existing callers can still supply an external plugin's configured ocr_options.
```

Or plan and run the routed OCR DAG:

```sh
glossapi plan --files /absolute/path/book.pdf --stages ocr,export --json
glossapi run --files /absolute/path/book.pdf --stages ocr,export --device ocr=cpu
```

`glossapi run` inserts page-quality inspection and feeds only `page_needs_ocr`
pages to OCR. Good native pages retain their text. It publishes OCR results before
export and fails explicitly when a selected page is unsupported. Omitting OCR
from `--stages` keeps the existing text-layer pipeline. A missing or mismatched
OCR package leaves the DAG waiting with an `install_glossapi.py ocr` hint; another
OCR engine is never silently selected.

Defaults: `paddle_small_heuristics_r8`, detector `small`, preset `clean_scan`,
recognition `x1`, model repository `fffoivos/greek-ocr`, revision
`2a6b019be728e5be70cdbd981414c630b63fca08`, manifest SHA256
`85d455d2cae2b756ab17e7114a3de13d7d41426d1fa5093028a179cf0c7d1345`.
The three preset names currently have identical rules; no per-page preset
selection is inferred. Explicit plugin options remain available, including
segmenter/rules, seeds, preset, local `release_dir` and devices. Options are
recorded in the ledger. Physical page routing and the runtime call-log destination
are supplied by glossAPI.

The scheduler assigns CUDA/MPS recognition, otherwise CPU. It reserves CPU tokens
for the detector/correction work alongside any accelerator and admits one OCR
operation per accelerator at a time. Both recognition and detector devices are
recorded. CLI planning uses a hardware hint without importing Torch; execution
validates the selected backend without changing to another engine.

`GLOSSAPI_CACHE_DIR` (default `$XDG_CACHE_HOME/glossapi`, or
`~/.cache/glossapi`) owns the cache root. For explicit operation settings,
`ocr.cache_root` overrides it. GlossAPI sets `GREEKOCR_CACHE_DIR` for the plugin,
which alone acquires and verifies the model and official upstream detector.
The actual plugin acquisition receipt supplies repository, requested revision,
resolved commit and manifest hash to the ledger. `release_dir` remains an explicit
local override. Prewarm both plugin HF caches and Docling layout/table assets
before offline use; a sealed recognition directory alone is insufficient.

`settings.ocr.options` accepts plugin fields for `glossapi run`; `settings.ocr`
also accepts `device_id`, `cache_root`, `num_threads`, `max_pages` and
`timeout_seconds`. `--device ocr=cuda:N` selects a device pool for a saved plan.
Explicit device options must agree with their pool. Model/seed overrides are
recorded; they do not inherit the default model's qualification.

Run the synthetic CPU fixture with `GREEKOCR_CACHE_DIR` and `OCR_LAYOUT_DIR`
pointing at warm caches:

```sh
python -m pytest tests/fixtures/greekocr/test_smoke.py -q
GLOSSAPI_OCR_INTEGRATION=1 python -m pytest \
  tests/test_greekocr_default.py::test_real_default_stage_cpu_and_provenance -q
```

The latter test blocks Python socket access and checks the actual stage's model
provenance. The optional mixed native/scan public-run test additionally requires
`GLOSSAPI_OCR_RUN_INTEGRATION=1` and current ledger-aware native extensions.
`tests/test_greekocr_cohort.py` compares independent direct segmentation and
recognition against the routed plugin on at most two needs-OCR pages from an
explicit owned cohort PDF. It is a correctness test, with no speed claims.
The 72-page owner-private equality check stays with the Mac coordinator/OCR session.

DeepSeek and the old bundled C1/RapidOCR engines remain removed. Printed upright
Greek/Latin text is the qualified input scope; handwriting, whole-page rotation,
vertical text and general non-text rejection are outside that scope.
