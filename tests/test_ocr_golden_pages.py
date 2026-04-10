from __future__ import annotations

import difflib
import json
from pathlib import Path

from glossapi import Corpus
from glossapi.corpus.phase_clean import _render_combined_ocr_debug_page


GOLDEN_DIR = Path(
    "/home/foivos/data/openarchives_ocr_ingest_20260403/debug/ocr_golden_pages_first300_20260410"
)


def _load_manifest_rows() -> list[dict]:
    manifest_path = GOLDEN_DIR / "manifest.jsonl"
    assert manifest_path.exists(), f"Missing OCR golden manifest: {manifest_path}"
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
    rows = _load_manifest_rows()
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
        input_path = Path(str(row["input_path"]))
        expected_path = Path(str(row["expected_path"]))
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
