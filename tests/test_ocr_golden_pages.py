from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

import pytest

from glossapi import Corpus
from glossapi.corpus.phase_clean import _render_combined_ocr_debug_page


# The real-page OCR goldens are an external fixture (built by glossapi.scripts.build_ocr_golden_pages), never part of
# the repository: owner-private scans must not enter tests. Point GLOSSAPI_OCR_GOLDEN_DIR at the fixture root (the
# directory holding manifest.jsonl, inputs/ and expected/) to run this test; without it the test skips with a reason.
GOLDEN_DIR_ENV = "GLOSSAPI_OCR_GOLDEN_DIR"


def _golden_dir() -> Path:
    value = os.environ.get(GOLDEN_DIR_ENV, "").strip()
    if not value:
        pytest.skip(f"{GOLDEN_DIR_ENV} is not set: the real-page OCR golden fixture is external and owner-supplied")
    return Path(value).expanduser()


def _fixture_path(root: Path, recorded: str) -> Path:
    """Resolve a manifest path inside the fixture root.

    The builder records absolute paths under its output directory (<out>/inputs/<case>.md, <out>/expected/<case>.md).
    A relocated fixture keeps that two-level layout, so the last two components are resolved against the root;
    relative paths are taken as root-relative. Nothing outside the root is read.
    """
    path = Path(recorded)
    if not path.is_absolute():
        return root / path
    return root / path.parent.name / path.name


def _load_manifest_rows(root: Path) -> list[dict]:
    manifest_path = root / "manifest.jsonl"
    # The variable was set explicitly, so a missing manifest is a misconfiguration, not an absent fixture.
    assert manifest_path.exists(), f"{GOLDEN_DIR_ENV}={root} has no manifest.jsonl"
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _format_diff(case_id: str, expected: str, actual: str) -> str:
    diff = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"{case_id}:expected",
            tofile=f"{case_id}:actual",
            lineterm="",
            n=3,
        )
    )
    return "\n".join(diff[:120])


def test_combined_ocr_real_goldens_match_exact_output(tmp_path: Path) -> None:
    root = _golden_dir()
    rows = _load_manifest_rows(root)
    assert len(rows) >= 300, f"Expected hundreds of real OCR golden cases, got {len(rows)}"

    corpus = Corpus(input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    corpus.input_dir.mkdir(parents=True, exist_ok=True)
    corpus.output_dir.mkdir(parents=True, exist_ok=True)
    noise_mod = corpus._load_rust_extension(
        "glossapi_rs_noise",
        "rust/glossapi_rs_noise/Cargo.toml",
        required_attrs=("find_numeric_debug_page_spans", "evaluate_page_character_noise"),
    )

    mismatches: list[str] = []
    for row in rows:
        case_id = str(row["case_id"])
        input_path = _fixture_path(root, str(row["input_path"]))
        expected_path = _fixture_path(root, str(row["expected_path"]))
        page_text = input_path.read_text(encoding="utf-8")
        expected = expected_path.read_text(encoding="utf-8")
        actual = _render_combined_ocr_debug_page(
            page_text,
            noise_mod=noise_mod,
            min_progress_steps=10,
            min_repeat_steps=8,
            min_same_digit_steps=10,
            word_rep_threshold=4,
            word_min_period=3,
            word_window=96,
        )["annotated_page"]
        if actual != expected:
            mismatches.append(_format_diff(case_id, expected, actual))
            if len(mismatches) >= 5:
                break

    assert not mismatches, "\n\n".join(mismatches)


def test_golden_root_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """T0-G03: no hard-coded fixture path; without the variable the golden test skips with a reason."""
    monkeypatch.delenv(GOLDEN_DIR_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match=GOLDEN_DIR_ENV):
        _golden_dir()
    monkeypatch.setenv(GOLDEN_DIR_ENV, "  ")
    with pytest.raises(pytest.skip.Exception, match=GOLDEN_DIR_ENV):
        _golden_dir()


def test_golden_manifest_paths_resolve_inside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relocated fixture is read from its root, never from the builder's original absolute paths."""
    monkeypatch.setenv(GOLDEN_DIR_ENV, str(tmp_path))
    root = _golden_dir()
    assert _fixture_path(root, "/elsewhere/build/inputs/case_001.md") == tmp_path / "inputs" / "case_001.md"
    assert _fixture_path(root, "expected/case_001.md") == tmp_path / "expected" / "case_001.md"
    with pytest.raises(AssertionError, match="no manifest.jsonl"):
        _load_manifest_rows(root)
