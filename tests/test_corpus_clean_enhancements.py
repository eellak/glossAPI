from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus


LATEX_MOJIBAKE_MD = """# Sample Document

Regular intro line.

$$
GLYPH&lt;1&gt;GLYPH&lt;2&gt;GLYPH&lt;3&gt;GLYPH&lt;4&gt;GLYPH&lt;5&gt;
GLYPH&lt;6&gt;GLYPH&lt;7&gt;GLYPH&lt;8&gt;GLYPH&lt;9&gt;GLYPH&lt;10&gt;
$$

Conclusion line with actual content.
"""


UPPER_GLYPH_MD = """## GLYPH ARTEFACT

GLYPH&lt;1&gt; GLYPH&lt;2&gt; GLYPH&lt;3&gt; GLYPH&lt;4&gt; GLYPH&lt;5&gt; GLYPH&lt;6&gt;
GLYPH&lt;7&gt; GLYPH&lt;8&gt; GLYPH&lt;9&gt; GLYPH&lt;10&gt; GLYPH&lt;11&gt; GLYPH&lt;12&gt;
GLYPH&lt;13&gt; GLYPH&lt;14&gt; GLYPH&lt;15&gt; GLYPH&lt;16&gt; GLYPH&lt;17&gt; GLYPH&lt;18&gt;
"""


def _build_corpus(tmp_path: Path) -> Corpus:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    return Corpus(input_dir=input_dir, output_dir=output_dir)


def _run_clean_and_read_row(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    write_cleaned_files: bool = True,
) -> pd.Series:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    corpus.clean(write_cleaned_files=write_cleaned_files)
    parquet = corpus.output_dir / "download_results" / "download_results.parquet"
    df = pd.read_parquet(parquet)
    row = df[df["filename"] == f"{stem}.pdf"]
    assert not row.empty, "Expected metrics entry for generated markdown"
    return row.iloc[0]


def test_clean_skips_latex_blocks_for_mojibake(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_and_read_row(corpus, LATEX_MOJIBAKE_MD, stem="latex-case")
    filter_value = (row.get("filter") or "").lower()
    assert "mojibake>0.1" not in filter_value
    score = row.get("mojibake_badness_score")
    assert pd.notna(score)
    assert float(score) < 0.1


def test_clean_supports_score_only_mode(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_and_read_row(
        corpus,
        "Plain content without artefacts.",
        stem="score-only",
        write_cleaned_files=False,
    )
    cleaned_dir = corpus.cleaned_markdown_dir
    assert not any(cleaned_dir.glob("*.md")), "No cleaned markdown files should be written"
    assert corpus.markdown_dir == corpus.output_dir / "markdown"
    assert not bool(row.get("needs_ocr", False))
    assert (row.get("filter") or "") == "ok"


def test_clean_flags_uppercase_glyph_noise(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_and_read_row(corpus, UPPER_GLYPH_MD, stem="glyph-artefact")
    score = float(row.get("mojibake_badness_score") or 0.0)
    assert score >= 0.5
    filter_value = row.get("filter") or ""
    assert "mojibake>0.1" in filter_value or "non_greek_text" in filter_value
    assert bool(row.get("needs_ocr", False))
