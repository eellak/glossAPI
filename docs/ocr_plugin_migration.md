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
and the dynamic `glossapi[ocr]` wheel extra read that file. It pins public commit
`bb5aa6f` on the toolkit's `dev/glossapi-ocr-default` branch: a code-only snapshot
of the reviewed release `138cff524` (identical `greekocr/`, `pyproject.toml` and
`README.md`). To move to a new toolkit release, replace that requirement line and
the `# greekocr-source-fingerprint` line beside it (commit, SHA256 over the
installed `greekocr/**/*.py` files, file count), then rerun the CPU smoke. No mirror
or glossAPI-side model downloader is used.

Runtime admission verifies the installed toolkit source, not only its version: the
PEP 610 `direct_url.json` VCS commit must equal the pin; an install without VCS
metadata must match the recorded code fingerprint. Any other source is rejected
with a message naming both sources. The installed source (commit, code
fingerprint, override status) is part of the frozen execution identity.

To test an unpublished toolkit checkout:

```sh
python install_glossapi.py ocr --device cpu \
  --ocr-toolkit-source 'git+file:///absolute/path/to/greek-ocr-toolkit@<commit>'
# GLOSSAPI_OCR_TOOLKIT_SOURCE supplies the same override without a CLI argument.
```

The override changes the install command, while wheel metadata retains the public
source pin. At run time the same `GLOSSAPI_OCR_TOOLKIT_SOURCE` value must be set:
it marks the installed source as a development override, is admitted only when it
describes that installed source, and is recorded as such. The Mac qualification used a code-only `git archive` because the full
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
pages to OCR. Default routing profile **v4** trusts an image-dominant existing
text layer (image coverage at least 0.5) only with measured Greek word validity
at least 0.9 and junk-token share strictly below 0.10 on Greek-dominant pages,
with no PUA/control contamination. Unknown validity routes to OCR; known validity
with unknown junk share cannot qualify the Greek layer. The independent settings
`image_dominant_min_valid_word_rate` and `image_dominant_max_junk_share` configure
this bar and are recorded on every page and in the extraction ledger. Every other
effective routing threshold (`image_dominant_ratio`, `image_dominant_min_chars`,
`min_valid_word_rate` and the fixed `min_chars`, `min_image_ratio`,
`min_greek_share`, `max_other_share`) is recorded too, so
`page_quality.replay_page_selection(row)` replays a sidecar row with its own
thresholds. Non-image and Latin routing retain their previous settings and rules.

Junk is the union of unknown words of up to three letters, unknown Latin fragments
of up to four letters, tokens mixing letter scripts, and Greek/Latin tokens with
digit intrusions. The denominator is alphanumeric runs containing letters;
standalone numbers do not count. Dictionary normalization follows validity lookup.
Each page records counts, category counts, nullable share, and whether the bar
applies. The batch CLI accepts `--image-dominant-min-valid-word-rate` and
`--image-dominant-max-junk-share`.
The existing scan gate (fewer than 50 extracted characters) remains in force;
a short valid caption cannot certify the raster body.
Clean Latin-majority layers (Greek share <= 0.5 of Greek + Latin letters) retain
the v3 decision when they have at least 600 characters, no PUA/control characters,
and other-character share <= 0.3. Their page record sets
`validity_not_applicable=true`; Greek vocabulary validity cannot qualify Latin text.
The density bound follows `image_dominant_min_chars` and never bypasses the scan gate.
Below the image threshold the v3 routing logic applies. V2/v3 remain selectable
through `page_quality(..., rule_profile="v3")` or the batch CLI's
`--rule-profile`; their previous dictionary defaults are preserved.

V4 measures Greek vocabulary membership offline on image-dominant pages with the
packaged Hunspell el_GR 0.9 surface entries. Automatic lookup is scoped to these
pages so unconfigured non-image routing stays identical to v3. The source revision,
scope, SHA256 and licence are recorded
in `dictionary_identity`; every page records `rule_profile` and its `decision`.
Dictionary provenance and licence: `src/glossapi/data/wordlists/README.md` in the
source tree (installed as `glossapi/data/wordlists/README.md`).
Explicit word sets/paths or `GLOSSAPI_GREEK_WORDLIST` override the default and
retain their previous all-page scope.
This modern Greek lexicon does not establish OCR accuracy and may over-route
valid polytonic or specialist text. Accent folding remains opt-in.

Good native pages retain their text. The stage publishes OCR results before
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
operation per accelerator at a time. Plugin defaults are resolved before the run
is frozen, so an omitted macOS `detector_device` (CoreML) claims the Apple
accelerator like MPS recognition. Requested and resolved devices plus recognition
and detector devices are recorded; a requested device that disagrees with
`ocr_options.device` is rejected before any pool or converter is chosen. CLI planning uses a hardware hint without importing Torch; execution
validates the selected backend without changing to another engine.

`GLOSSAPI_CACHE_DIR` (default `$XDG_CACHE_HOME/glossapi`, or
`~/.cache/glossapi`) owns the cache root. For explicit operation settings,
`ocr.cache_root` overrides it. GlossAPI sets `GREEKOCR_CACHE_DIR` for the plugin,
which alone acquires and verifies the model and official upstream detector.
A `GREEKOCR_SEGMENTER_MODEL_DIR` detector override is recorded as such (actual
path and SHA256 next to the expected upstream identity) and its file hashes are
bound into the frozen execution identity.
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
