# Synthetic Docling CPU smoke — final design (G3)

Copy this directory only. It contains an unchanged image-only two-page PDF, six public-domain Homer lines, its source transcription and the observed uncorrected OCR baseline. This is an execution/output regression, not a recognition benchmark. No font, model, seed or private scan is included. Do not rebuild the PDF during testing; its metadata may change its hash.

Use the reviewed toolkit commit fb068b45234a12206953a5c0889631773a33fa41 (or the coordinator's final public merge), Docling 2.81.0 and the dependencies in ../HANDOFF_GLOSSAPI.md. The runner selects R8/small/clean_scan/x1, CPU detector and recognizer, no filters. It fetches the pinned official upstream ONNX and x1 via model_repo/model_variant/model_revision/model_manifest_sha256. It supplies no local model source or detector override. No plugin tables=False is passed; Docling table structure is disabled separately.

Set GREEKOCR_CACHE_DIR to glossAPI's cache root. Prewarm fetch_small_detector() and fetch_model() with the pins in question 1 of the hand-off, plus the pinned Heron snapshot. Both HF caches must remain present for offline reuse. The runner enforces HF offline mode; missing assets fail.

```sh
export GREEKOCR_CACHE_DIR=/absolute/path/to/glossapi-cache
export OCR_LAYOUT_DIR=/absolute/path/to/heron-snapshot
unset GREEKOCR_SEGMENTER_MODEL_DIR
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest test_smoke.py -q \
  -p no:cacheprovider --noconftest --basetemp=/private/tmp/greekocr-copy-smoke-pytest
```

Test dependencies: pytest and pypdf. The test checks PDF identity, empty text layers, actual plugin replacement on both pages, upstream/model acquisition identities, CPU options, six kept lines, zero drops and exact normalized OCR text/hash. Timing is recorded, never asserted. Remove the named temporary directory after inspecting the result. Use a fresh directory for each runner invocation.

`expected.json` was refreshed from G3's real socket-blocked cached conversion. It is observed OCR, not ground truth; the source/observed apostrophe difference is preserved. The offline proof is ../offline_probe.py, with its command and receipt in question 5 of the hand-off. This smoke's subprocess test alone does not inherit socket monkeypatches from pytest.

`build_fixture.py` is an optional Mac authoring aid requiring Pillow and local Times New Roman; the font is not redistributed.
