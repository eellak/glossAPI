from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples" / "lightweight_pdf_corpus"
PDF_DIR = SAMPLES_DIR / "pdfs"
EXPECTED_OUTPUTS_PATH = SAMPLES_DIR / "expected_outputs.json"


@pytest.fixture(scope="module")
def expected_outputs() -> dict[str, list[str]]:
    with EXPECTED_OUTPUTS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _copy_sample_pdfs(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for pdf_path in PDF_DIR.glob("*.pdf"):
        shutil.copy2(pdf_path, dest / pdf_path.name)


@pytest.fixture
def corpus_paths(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _copy_sample_pdfs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def test_phase_extract_safe_backend_matches_expected(
    corpus_paths: tuple[Path, Path],
    expected_outputs: dict[str, list[str]],
) -> None:
    input_dir, output_dir = corpus_paths
    corpus = Corpus(input_dir, output_dir)

    corpus.extract(input_format="pdf", phase1_backend="safe", use_gpus="none")

    markdown_dir = output_dir / "markdown"
    assert markdown_dir.exists(), "Markdown output directory missing after extract()"

    for stem, expected_lines in expected_outputs.items():
        md_path = markdown_dir / f"{stem}.md"
        assert md_path.exists(), f"Missing markdown for {stem}"
        actual_lines = [line.rstrip("\n") for line in md_path.read_text(encoding="utf-8").splitlines()]
        assert actual_lines == expected_lines, f"Extracted markdown mismatch for {stem}"


def test_phase_clean_populates_clean_markdown(corpus_paths: tuple[Path, Path]) -> None:
    pytest.importorskip("glossapi_rs_cleaner")

    input_dir, output_dir = corpus_paths
    corpus = Corpus(input_dir, output_dir)

    corpus.extract(input_format="pdf", phase1_backend="safe", use_gpus="none")
    corpus.clean()

    clean_dir = output_dir / "clean_markdown"
    assert clean_dir.exists(), "clean_markdown directory not created"

    markdown_files = sorted(clean_dir.glob("*.md"))
    assert markdown_files, "No cleaned markdown files produced"

    for md_file in markdown_files:
        content = md_file.read_text(encoding="utf-8").strip()
        assert content, f"Cleaned markdown {md_file.name} is empty"


def test_phase_section_emits_expected_parquet(corpus_paths: tuple[Path, Path]) -> None:
    input_dir, output_dir = corpus_paths
    corpus = Corpus(input_dir, output_dir)

    corpus.extract(input_format="pdf", phase1_backend="safe", use_gpus="none")
    corpus.section()

    sections_path = output_dir / "sections" / "sections_for_annotation.parquet"
    assert sections_path.exists(), "sections_for_annotation.parquet not written"

    df = pd.read_parquet(sections_path)
    assert not df.empty, "Section parquet unexpectedly empty"
    assert "filename" in df.columns
    filenames = set(df["filename"].tolist())
    expected_stems = {path.stem for path in PDF_DIR.glob("*.pdf")}
    assert expected_stems == filenames, "Section output missing filenames"
    assert df["section_length"].gt(0).all(), "Section length should be positive for all rows"
