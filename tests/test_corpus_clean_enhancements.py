from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from glossapi import Corpus
from glossapi.corpus.phase_clean import (
    _find_word_repeat_spans,
    _find_word_repeat_spans_python,
    _normalize_alnum_with_map_skip_tags,
)
from glossapi.scripts.table_markdown_audit import audit_table, write_clean_markdown_file
from glossapi.scripts.review_manifest_materialize import materialize_manifest_categories


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


def _run_clean_ocr_and_read_row(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    drop_bad: bool = False,
) -> pd.Series:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    corpus.clean_ocr(drop_bad=drop_bad)
    parquet = corpus.output_dir / "download_results" / "download_results.parquet"
    df = pd.read_parquet(parquet)
    row = df[df["filename"] == f"{stem}.pdf"]
    assert not row.empty, "Expected OCR metrics entry for generated markdown"
    return row.iloc[0]


def _run_clean_ocr_and_read_cleaned_text(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    write_cleaned_files: bool = True,
) -> str:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    corpus.clean_ocr(write_cleaned_files=write_cleaned_files)
    cleaned_path = corpus.cleaned_markdown_dir / f"{stem}.md"
    assert cleaned_path.exists(), f"Expected cleaned markdown output at {cleaned_path}"
    return cleaned_path.read_text(encoding="utf-8")


def _run_clean_ocr_debug_export(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    max_pages: int | None = 1000,
) -> tuple[list[dict], Path]:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    debug_dir = corpus.output_dir / "ocr_debug"
    rows = corpus.clean_ocr_debug(debug_dir, max_pages=max_pages)
    return rows, debug_dir


def _run_clean_ocr_numeric_debug_export(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    max_pages: int | None = 1000,
) -> tuple[list[dict], Path]:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    debug_dir = corpus.output_dir / "ocr_numeric_debug"
    rows = corpus.clean_ocr_numeric_debug(debug_dir, max_pages=max_pages)
    return rows, debug_dir


def _run_clean_ocr_numeric_word_debug_docs(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    max_docs: int | None = 100,
) -> tuple[list[dict], Path]:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    debug_dir = corpus.output_dir / "ocr_numeric_word_debug"
    rows = corpus.clean_ocr_numeric_word_debug_docs(debug_dir, max_docs=max_docs)
    return rows, debug_dir


def _run_clean_ocr_hybrid_debug_export(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    max_docs: int | None = 100,
) -> tuple[list[dict], Path]:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    debug_dir = corpus.output_dir / "ocr_hybrid_debug"
    rows = corpus.clean_ocr_hybrid_debug(debug_dir, max_docs=max_docs)
    return rows, debug_dir


def _run_clean_ocr_latex_slot_progression_debug_export(
    corpus: Corpus,
    markdown_text: str,
    *,
    stem: str = "sample",
    max_docs: int | None = 1000,
) -> tuple[list[dict], Path]:
    md_path = corpus.markdown_dir / f"{stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")
    debug_dir = corpus.output_dir / "ocr_latex_slot_progression_debug"
    rows = corpus.clean_ocr_latex_slot_progression_debug(debug_dir, max_docs=max_docs)
    return rows, debug_dir


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


def test_clean_ocr_populates_script_metrics(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_ocr_and_read_row(
        corpus,
        "Αυτή είναι η πρώτη σελίδα.\n<--- Page Split --->\nΚαὶ αὕτη εἶναι ἡ δευτέρα.",
        stem="ocr-script-metrics",
    )
    assert float(row.get("percentage_greek") or 0.0) > 70.0
    assert float(row.get("latin_percentage") or 0.0) < 5.0
    assert float(row.get("polytonic_ratio") or 0.0) > 0.0
    assert not bool(row.get("ocr_noise_suspect", False))
    assert (row.get("filter") or "") == "ok"


def test_clean_ocr_writes_cleaned_markdown_with_combined_loop(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    content = _run_clean_ocr_and_read_cleaned_text(
        corpus,
        (
            "1111 1 1 1 1 1 1 1 1 1 1\n"
            "<--- Page Split --->\n"
            "1. Από το 2020, η αγορά των εργασιών των εργασιών των εργασιών των εργασιών των εργασιώ\n"
            "<table><tr><th>Name</th><th>Score</th></tr><tr><td>Alice</td><td>10</td></tr></table>\n"
        ),
        stem="ocr-clean-shared-loop",
    )
    assert "<--- Page Split --->" in content
    assert "<match of type" not in content
    assert "1111 1 1 1 1 1 1 1 1 1 1" not in content
    assert "των εργασιών των εργασιών" not in content
    assert "<table>" not in content
    assert "| Name" in content
    assert "| Alice" in content
    assert corpus.markdown_dir == corpus.cleaned_markdown_dir


def test_clean_ocr_drops_sentence_shell_and_repeated_row_tables(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    content = _run_clean_ocr_and_read_cleaned_text(
        corpus,
        (
            "Πρόλογος\n"
            "<table><tr><td rowspan=\"2\">Η οινοφόρος άμπελος αναπτύχθηκε στην Αρμενία, νότια της Κασπίας</td><td></td></tr><tr><td></td></tr></table>\n"
            "<table><tr><th>State</th><th>Value</th></tr><tr><td>Alpha</td><td>10</td></tr><tr><td>Beta</td><td>20</td></tr><tr><td>Alpha</td><td>10</td></tr><tr><td>Beta</td><td>20</td></tr></table>\n"
            "Επίλογος\n"
        ),
        stem="ocr-clean-drop-tables",
    )
    assert "<table>" not in content
    assert "Η οινοφόρος άμπελος" not in content
    assert "| Alpha" not in content
    assert "Πρόλογος" in content
    assert "Επίλογος" in content


def test_clean_ocr_supports_score_only_mode(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    md_path = corpus.markdown_dir / "ocr-clean-score-only.md"
    md_path.write_text("Κανονικό περιεχόμενο.\n", encoding="utf-8")
    corpus.clean_ocr(write_cleaned_files=False)
    assert not any(corpus.cleaned_markdown_dir.glob("*.md"))
    assert corpus.markdown_dir == corpus.output_dir / "markdown"


def test_clean_ocr_ignores_numeric_lists_and_dotted_values(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_ocr_and_read_row(
        corpus,
        "1. 2. 3. 4. 5. 6. 7.\n9.9.9.9.9\n",
        stem="ocr-non-repeat-noise",
        drop_bad=True,
    )
    assert not bool(row.get("ocr_noise_suspect", False))
    assert int(row.get("ocr_repeat_phrase_run_max") or 0) == 0
    assert int(row.get("ocr_repeat_line_run_max") or 0) == 0
    flags = row.get("ocr_noise_flags") or ""
    assert flags == ""
    assert "ocr_noise" not in (row.get("filter") or "")
    assert "ocr-non-repeat-noise" in corpus.good_files


def test_clean_ocr_flags_repeated_phrase_noise(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    row = _run_clean_ocr_and_read_row(
        corpus,
        "0 0 0 0 0 0\n1.1\n1.1\n1.1\n1.1\n1.1\n1.1\n",
        stem="ocr-repeat-noise",
        drop_bad=True,
    )
    assert bool(row.get("ocr_noise_suspect", False))
    assert int(row.get("ocr_repeat_phrase_run_max") or 0) >= 6
    assert int(row.get("ocr_repeat_line_run_max") or 0) >= 6
    flags = row.get("ocr_noise_flags") or ""
    assert "repeat_phrase_run" in flags
    assert "repeat_line_run" in flags
    assert "ocr_noise" in (row.get("filter") or "")
    assert "ocr-repeat-noise" not in corpus.good_files


def test_clean_ocr_debug_exports_annotated_pages(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_debug_export(
        corpus,
        (
            "Κανονική πρώτη σελίδα.\n"
            "<--- Page Split --->\n"
            "1. 2. 3. 4. 5. 6. 7.\n"
            "0 0 0 0 0 0\n"
            "1.1\n1.1\n1.1\n1.1\n1.1\n1.1\n"
        ),
        stem="ocr-debug-source",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["page_number"] == 2
    assert row["page_index_in_file"] == 2
    assert row["match_count"] >= 2
    assert "repeat_phrase_run" in row["match_types"]
    assert "repeat_line_run" in row["match_types"]

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "1. 2. 3. 4. 5. 6. 7." in content
    assert "<match of type repeat_phrase_run>0 0 0 0 0 0</match>" in content
    assert "<match of type repeat_line_run>1.1</match>" in content

    manifest = debug_dir / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_clean_ocr_debug_respects_sample_limit(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    md_path = corpus.markdown_dir / "ocr-debug-many.md"
    md_path.write_text(
        (
            "0 0 0 0 0 0\n"
            "<--- Page Split --->\n"
            "0 0 0 0 0 0\n"
            "<--- Page Split --->\n"
            "0 0 0 0 0 0\n"
        ),
        encoding="utf-8",
    )
    debug_dir = corpus.output_dir / "ocr_debug"
    rows = corpus.clean_ocr_debug(debug_dir, max_pages=2, sample_seed=0)
    assert len(rows) == 2
    manifest = debug_dir / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_clean_ocr_numeric_debug_flags_ascending_sequences(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        (
            "Κανονικό κείμενο.\n"
            "<--- Page Split --->\n"
            "1. 2. 3. 4. 5. 6. 7. 8. 9. 10.\n"
        ),
        stem="ocr-numeric-progress",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["page_number"] == 2
    assert "ascending_numeric_sequence" in row["match_types"]

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert (
        "<match of type ascending_numeric_sequence>1. 2. 3. 4. 5. 6. 7. 8. 9. 10</match>"
        in content
    )


def test_clean_ocr_numeric_debug_flags_compact_repeated_numbers(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        "2.2.2.2.2.2.2.2.\n",
        stem="ocr-numeric-compact-repeat",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "repeat_numeric_run" in row["match_types"]

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type repeat_numeric_run>2.2.2.2.2.2.2.2</match>" in content


def test_clean_ocr_numeric_debug_flags_same_digit_runs(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        "1111 1 1 1 111 11 1 111 1 11\n",
        stem="ocr-numeric-same-digit",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "same_digit_numeric_run" in row["match_types"]

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert (
        "<match of type same_digit_numeric_run>1111 1 1 1 111 11 1 111 1 11</match>"
        in content
    )


def test_clean_ocr_numeric_debug_merges_close_same_category_spans(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        "1111 1 1 1 1 1 1 1 1 1 1 xy 1111 1 1 1 1 1 1 1 1 1 1\n",
        stem="ocr-numeric-gap-merge",
    )
    assert len(rows) == 1
    exported = Path(rows[0]["output_path"])
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert content.count("<match of type same_digit_numeric_run>") == 1
    assert (
        "<match of type same_digit_numeric_run>"
        "1111 1 1 1 1 1 1 1 1 1 1 xy 1111 1 1 1 1 1 1 1 1 1 1"
        "</match>"
        in content
    )


def test_clean_ocr_numeric_debug_flags_numeric_page_collapse(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    tokens = ("22 2 22 6 22 8 22 1 22 7 22 5 " * 12).strip()
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        tokens + "\n",
        stem="ocr-numeric-page-collapse",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert tokens in content


def test_clean_ocr_numeric_debug_page_collapse_ignores_punctuation_only_tokens(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    tokens = ("1 1 . 1 1 . 2 2 . 2 2 . " * 16).strip()
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        tokens + "\n",
        stem="ocr-numeric-page-collapse-punct",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert tokens in content


def test_clean_ocr_numeric_debug_page_collapse_ignores_container_tokens(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    numeric_body = ("11 11 11 22 22 22 33 33 33 44 44 44 " * 8).strip()
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        f"```\n( {numeric_body} )\n```\n",
        stem="ocr-numeric-page-collapse-fenced",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert numeric_body in content


def test_clean_ocr_numeric_debug_page_collapse_accepts_dotted_numeric_tokens(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    dotted_tokens = " ".join(f"{major}.{minor}." for major in range(1, 6) for minor in range(1, 21))
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        dotted_tokens + "\n",
        stem="ocr-numeric-page-collapse-dotted",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert dotted_tokens in content


def test_clean_ocr_numeric_debug_page_collapse_accepts_compact_numeric_atom_pages(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    compact_tokens = " ".join(["1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1."] * 20)
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        compact_tokens + "\n",
        stem="ocr-numeric-page-collapse-compact-atoms",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert compact_tokens in content


def test_clean_ocr_numeric_debug_flags_numeric_block_after_heading(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    numeric_block = "\n\n".join(
        f"{i}.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.{i}.1.1.1.1.1.1.1.1.1.1.1.1.1.1"
        for i in range(1, 27)
    )
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        f"1\n\n## ΑΠΡΙΛΙΟΣ\n\n1\n\n{numeric_block}\n",
        stem="ocr-numeric-block-heading",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_block_collapse" in row["match_types"]
    assert row["match_count"] == 1

    exported = Path(row["output_path"])
    assert exported.exists()
    assert exported.parent == debug_dir
    content = exported.read_text(encoding="utf-8")
    assert "## ΑΠΡΙΛΙΟΣ" in content
    assert "<match of type numeric_block_collapse>" in content
    assert numeric_block in content


def test_clean_ocr_numeric_word_debug_docs_runs_numeric_then_word(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        (
            "1111 1 1 1 1 1 1 1 1 1 1\n"
            "<--- Page Split --->\n"
            "1. Από το 2020, η αγορά των εργασιών των εργασιών των εργασιών των εργασιών των εργασιώ\n"
            "<table><tr><td>Standard name</td><td>Standard name</td><td>Standard name</td></tr></table>\n"
        ),
        stem="ocr-number-word-doc",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["page_count"] == 2
    assert row["matched_page_count"] == 2
    assert row["numeric_match_count"] >= 1
    assert row["word_match_count"] >= 1
    assert "word_repeat" in row["match_types"]

    exported = debug_dir / "ocr-number-word-doc.md"
    content = exported.read_text(encoding="utf-8")
    assert "<--- Page Split --->" in content
    assert content.count("<match of type same_digit_numeric_run>") == 1
    assert "<match of type word_repeat" in content
    assert "των εργασιώ</match>" in content
    assert "<match of type word_repeat period=12 reps=3>Standard name" not in content

    summary = json.loads((debug_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["doc_count"] == 1
    assert summary["numeric_match_count"] >= 1
    assert summary["word_match_count"] >= 1

    page_metrics = (debug_dir / "page_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(page_metrics) == 2


def test_rust_word_repeat_spans_match_python_reference(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    corpus._load_rust_extension(
        "glossapi_rs_noise",
        "rust/glossapi_rs_noise/Cargo.toml",
        required_attrs=("find_word_repeat_spans",),
    )
    cases = [
        "των εργασιών των εργασιών των εργασιών των εργασιών των εργασιώ",
        "1.1 Hypergeometric function 1.1.1 Hypergeometric function 1.1.2 Hypergeometric function 1.1.3 Hypergeometric function",
        r"\Delta \Delta \Delta \Delta \Delta",
        "το σημείο 1, το σημείο 2, το σημείο 3, το σημείο 4, το σημείο 5, το σημείο 6",
    ]
    for text in cases:
        normalized, _ = _normalize_alnum_with_map_skip_tags(text)
        assert _find_word_repeat_spans(
            normalized,
            rep_threshold=4,
            min_period=3,
            window=96,
        ) == _find_word_repeat_spans_python(
            normalized,
            rep_threshold=4,
            min_period=3,
            window=96,
        )


def test_clean_ocr_numeric_word_debug_docs_flags_empty_html_table_collapse(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    empty_table = (
        "<table>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "<tr><td></td><td></td><td></td></tr>"
        "</table>\n"
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        empty_table,
        stem="ocr-empty-table",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["table_match_count"] == 1
    assert "table_repeat" in row["match_types"]

    content = (debug_dir / "ocr-empty-table.md").read_text(encoding="utf-8")
    assert "<match of type table_repeat kind=empty_table_collapse" in content
    assert "<table>" not in content
    assert "|" in content

    summary = json.loads((debug_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["table_match_count"] == 1


def test_clean_ocr_numeric_word_debug_docs_flags_repeated_html_table_rows(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    repeated_table = (
        "<table>"
        "<tr><th>State</th><th>Value</th></tr>"
        "<tr><td>Alpha</td><td>10</td></tr>"
        "<tr><td>Beta</td><td>20</td></tr>"
        "<tr><td>Alpha</td><td>10</td></tr>"
        "<tr><td>Beta</td><td>20</td></tr>"
        "</table>\n"
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated_table,
        stem="ocr-repeated-table-rows",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["table_match_count"] == 1
    assert "table_repeat" in row["match_types"]

    content = (debug_dir / "ocr-repeated-table-rows.md").read_text(encoding="utf-8")
    assert "<match of type table_repeat kind=repeated_rows" in content
    assert "dup_rows=2" in content
    assert "<table>" not in content
    assert "| Alpha" in content or "| Beta" in content


def test_clean_ocr_numeric_word_debug_docs_ignores_small_distinct_html_table(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        (
            "<table>"
            "<tr><th>Name</th><th>Score</th></tr>"
            "<tr><td>Alice</td><td>10</td></tr>"
            "<tr><td>Bob</td><td>11</td></tr>"
            "</table>\n"
        ),
        stem="ocr-distinct-table",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["table_match_count"] == 0
    assert "table_repeat" not in row["match_types"]

    content = (debug_dir / "ocr-distinct-table.md").read_text(encoding="utf-8")
    assert "<match of type table_repeat" not in content
    assert "<table>" not in content
    assert "| Name" in content
    assert "| Alice" in content


def test_clean_ocr_numeric_word_debug_docs_flags_sentence_shell_table(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        (
            "<table><tr><td rowspan=\"2\">"
            "Η οινοφόρος άμπελος αναπτύχθηκε στην Αρμενία, νότια της Κασπίας"
            "</td><td></td></tr><tr><td></td></tr></table>\n"
        ),
        stem="ocr-sentence-shell-table",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["table_match_count"] == 1
    assert "table_repeat" in row["match_types"]

    content = (debug_dir / "ocr-sentence-shell-table.md").read_text(encoding="utf-8")
    assert "<match of type table_repeat kind=sentence_shell_table" in content
    assert "Η οινοφόρος άμπελος αναπτύχθηκε στην Αρμενία, νότια της Κασπίας" in content
    assert "<table>" not in content


def test_clean_ocr_numeric_word_debug_docs_transfers_pure_numeric_repeats_to_numeric(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        "12 12 12 12 12 12 12 12 12 12 12 12\n",
        stem="ocr-number-transfer",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["numeric_match_count"] >= 1
    assert row["word_match_count"] == 0
    assert "numeric_repeat" in row["match_types"]
    assert "word_repeat" not in row["match_types"]

    content = (debug_dir / "ocr-number-transfer.md").read_text(encoding="utf-8")
    assert "<match of type numeric_repeat>12 12 12 12 12 12 12 12 12 12 12 12</match>" in content


def test_clean_ocr_numeric_word_debug_docs_flags_hybrid_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        "1.1 Hypergeometric function 1.1.1 Hypergeometric function 1.1.2 Hypergeometric function 1.1.3 Hypergeometric function 1.1.4 Hypergeometric function\n",
        stem="ocr-combined-hybrid",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["hybrid_match_count"] >= 1
    assert "hybrid_repeat" in row["match_types"]

    content = (debug_dir / "ocr-combined-hybrid.md").read_text(encoding="utf-8")
    assert "<match of type hybrid_repeat kind=same_body_progression" in content

    summary = json.loads((debug_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["hybrid_match_count"] >= 1


def test_clean_ocr_numeric_word_debug_docs_ignores_latex_in_shared_repeat(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\[ S=\frac{1}{16\pi}\int\sqrt{-g}d^{4}x\left[\phi R-\frac{\omega(\phi)}{\phi}\phi_{,a}\phi^{,a}+2\phi\lambda(\phi)\right]+S_{M} \quad (149) \]" + "\n",
        stem="ocr-latex-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["word_match_count"] == 0
    assert row["latex_match_count"] == 0
    assert "word_repeat" not in row["match_types"]
    content = (debug_dir / "ocr-latex-ignore.md").read_text(encoding="utf-8")
    assert "<match of type word_repeat" not in content


def test_clean_ocr_numeric_word_debug_docs_flags_latex_structural_prefix_repeat(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\( \varepsilon_{H} = \frac{1}{2} \left( \frac{1}{2} \left( \frac{1}{2} \left( \frac{1}{2} \left( x \right) \right) \right) \right) \)"
        + "\n",
        stem="ocr-latex-structural-repeat",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    assert "latex_repeat" in row["match_types"]
    content = (debug_dir / "ocr-latex-structural-repeat.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" in content


def test_clean_ocr_numeric_word_debug_docs_flags_latex_markup_repeat(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\[ u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub>u<sub>α</sub> \]"
        + "\n",
        stem="ocr-latex-markup-repeat",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    assert "latex_repeat" in row["match_types"]
    content = (debug_dir / "ocr-latex-markup-repeat.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" in content


def test_clean_ocr_numeric_word_debug_docs_flags_latex_text_wrapper_noise(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\[ K:\mathrm{\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa} \]"
        + "\n",
        stem="ocr-latex-text-wrapper-noise",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    assert "latex_repeat" in row["match_types"]
    content = (debug_dir / "ocr-latex-text-wrapper-noise.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" in content


def test_clean_ocr_numeric_word_debug_docs_flags_unclosed_latex_text_wrapper_noise(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\[ K:\mathrm{\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa\iota\kappa \]"
        + "\n",
        stem="ocr-latex-unclosed-text-wrapper-noise",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    assert "latex_repeat" in row["match_types"]
    content = (debug_dir / "ocr-latex-unclosed-text-wrapper-noise.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" in content


def test_clean_ocr_numeric_word_debug_docs_ignores_latex_delimiter_bookkeeping(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"\[ \delta R^{\mu\nu}=g^{\mu\alpha}g^{\nu\beta}\left(\nabla_{\kappa}\left(\delta g_{\nu\alpha}\right)\right). \]"
        + "\n",
        stem="ocr-latex-bookkeeping-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] == 0
    assert "latex_repeat" not in row["match_types"]
    content = (debug_dir / "ocr-latex-bookkeeping-ignore.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" not in content


def test_clean_ocr_numeric_word_debug_docs_flags_consecutive_repeated_latex_segments(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\[ N_{bd} = \frac{f_{bk} \cdot l_e \cdot \pi \cdot d_b}{\gamma_b} \]",
            r"\[ N_{bd} = \frac{f_{bk} \cdot l_e \cdot \pi \cdot d_b}{\gamma_b} \]",
            r"\[ N_{bd} = \frac{f_{bk} \cdot l_e \cdot \pi \cdot d_b}{\gamma_b} \]",
            r"\[ N_{bd} = \frac{f_{bk} \cdot l_e \cdot \pi \cdot d_b}{\gamma_b} \]",
        ]
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-consecutive-exact",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    content = (debug_dir / "ocr-latex-consecutive-exact.md").read_text(encoding="utf-8")
    assert content.count("<match of type latex_repeat") == 1


def test_clean_ocr_numeric_word_debug_docs_flags_consecutive_latex_template_run(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\( N_{bd} = \frac{0,6Mpa \cdot 0,4m \cdot \pi \cdot 0,02m}{1,5} = 10,05KN \)",
            r"\( N_{bd} = \frac{0,6Mpa \cdot 0,4m \cdot \pi \cdot 0,03m}{1,5} = 15,07KN \)",
            r"\( N_{bd} = \frac{0,6Mpa \cdot 0,4m \cdot \pi \cdot 0,04m}{1,5} = 20,10KN \)",
            r"\( N_{bd} = \frac{0,6Mpa \cdot 0,4m \cdot \pi \cdot 0,05m}{1,5} = 25,12KN \)",
        ]
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-consecutive-template",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    content = (debug_dir / "ocr-latex-consecutive-template.md").read_text(encoding="utf-8")
    assert content.count("<match of type latex_repeat") == 1


def test_clean_ocr_numeric_word_debug_docs_flags_consecutive_short_delta_atoms(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join([r"\( \Delta \)", r"\( \Delta \)", r"\( \Delta \)", r"\( \Delta \)"])
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-delta-run",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    content = (debug_dir / "ocr-latex-delta-run.md").read_text(encoding="utf-8")
    assert content.count("<match of type latex_repeat") == 1


def test_clean_ocr_numeric_word_debug_docs_ignores_diagrammatic_short_latex_symbols(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        (
            r"\( Q^{I} \) : \( \uparrow\uparrow\uparrow \) + \( \uparrow\downarrow\downarrow \) + ..."
            "\n\n"
            r"\( Q^{IV} \) : \( \uparrow\uparrow\uparrow \) + \( \downarrow\downarrow\downarrow \) + ..."
            "\n"
        ),
        stem="ocr-latex-diagram-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] == 0
    assert "latex_repeat" not in row["match_types"]
    content = (debug_dir / "ocr-latex-diagram-ignore.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" not in content


def test_clean_ocr_numeric_word_debug_docs_grows_derivative_ladder_template_run(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\[  \frac{d^2\Psi}{dr_*^2} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^2\Psi}{dr_*^2} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^3\Psi}{dr_*^3} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^3\Psi}{dr^3} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^4\Psi}{dr_*^4} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^4\Psi}{dr^4} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^5\Psi}{dr_*^5} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^5\Psi}{dr^5} + (\omega^2 - V(r))\Psi = 0  \]",
        ]
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-derivative-ladder",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] >= 1
    content = (debug_dir / "ocr-latex-derivative-ladder.md").read_text(encoding="utf-8")
    assert content.count("<match of type latex_repeat") == 1
    assert content.index(r"\frac{d^5\Psi}{dr^5}") < content.index("</match>")


def test_clean_ocr_numeric_word_debug_docs_ignores_small_parameterized_formula_family(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\( f_{11}(k) = (1 - 0.0561)^{k-1}0.0561 \)",
            r"\( f_{12}(k) = (1 - 0.0617)^{k-1}0.0617 \)",
            r"\( f_{21}(k) = (1 - 0.1057)^{k-1}0.1057 \)",
            r"\( f_{22}(k) = (1 - 0.1724)^{k-1}0.1724 \)",
        ]
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-parameter-family-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] == 0
    content = (debug_dir / "ocr-latex-parameter-family-ignore.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" not in content


def test_clean_ocr_numeric_word_debug_docs_ignores_symbol_inventory_run(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "   ".join(
        [
            r"\( \tilde{p}_{(1,1)(1,2)}^{\prime} \)",
            r"\( \tilde{p}_{(1,1)(2,0)}^{\prime} \)",
            r"\( \tilde{p}_{(1,1)(1,0)}^{\prime} \)",
            r"\( \tilde{p}_{(2,0)(1,0)}^{\prime} \)",
            r"\( \tilde{p}_{(2,0)(2,1)}^{\prime} \)",
            r"\( \tilde{p}_{(2,0)(2,0)}^{\prime} \)",
        ]
    )
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        repeated + "\n",
        stem="ocr-latex-symbol-inventory-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] == 0
    content = (debug_dir / "ocr-latex-symbol-inventory-ignore.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" not in content


def test_clean_ocr_numeric_word_debug_docs_ignores_delta_definition_atoms(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        r"where \( \Delta \) CFF = \( \Delta \) CFF(t) - \( \Delta \) CFF(t-1)." + "\n",
        stem="ocr-latex-delta-definition-ignore",
        max_docs=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["latex_match_count"] == 0
    content = (debug_dir / "ocr-latex-delta-definition-ignore.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat" not in content


def test_clean_ocr_numeric_debug_flags_vertical_numeric_pages_with_longer_integers(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    vertical_numbers = "\n\n".join(str(i) for i in range(0, 121))
    rows, debug_dir = _run_clean_ocr_numeric_debug_export(
        corpus,
        vertical_numbers + "\n",
        stem="ocr-vertical-numeric-page",
    )
    assert len(rows) == 1
    row = rows[0]
    assert "numeric_page_collapse" in row["match_types"]

    content = Path(row["output_path"]).read_text(encoding="utf-8")
    assert "<match of type numeric_page_collapse>" in content
    assert "100" in content
    assert "120" in content


def test_clean_ocr_numeric_word_debug_docs_records_bad_char_metrics(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_numeric_word_debug_docs(
        corpus,
        "Κανονική γραμμή\n<--- Page Split --->\n## \x01\x02\x00 漢 \uf0b7\n",
        stem="ocr-bad-char-metrics",
        max_docs=1,
    )
    assert len(rows) == 1

    page_metric_rows = [
        json.loads(line)
        for line in (debug_dir / "page_metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(page_metric_rows) == 2
    second_page = page_metric_rows[1]
    assert second_page["bad_char_count"] >= 4
    assert second_page["bad_char_ratio"] > 0.0
    assert second_page["control_count"] >= 3
    assert second_page["cjk_count"] >= 1
    assert second_page["private_use_count"] >= 1

    summary = json.loads((debug_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["bad_char_ratio"]["max"] > 0.0


def test_clean_ocr_numeric_word_debug_docs_respects_doc_offset(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    (corpus.markdown_dir / "a-first.md").write_text("χωρίς επανάληψη\n", encoding="utf-8")
    (corpus.markdown_dir / "b-second.md").write_text(
        r"\( \Delta \)" + "\n\n" + r"\( \Delta \)" + "\n\n" + r"\( \Delta \)" + "\n\n" + r"\( \Delta \)" + "\n",
        encoding="utf-8",
    )

    debug_dir = corpus.output_dir / "ocr_numeric_word_debug"
    rows = corpus.clean_ocr_numeric_word_debug_docs(debug_dir, max_docs=1, doc_offset=1)

    assert len(rows) == 1
    row = rows[0]
    assert row["source_stem"] == "b-second"
    assert row["latex_match_count"] >= 1
    assert not (debug_dir / "a-first.md").exists()
    assert (debug_dir / "b-second.md").exists()


def test_clean_ocr_hybrid_debug_flags_same_body_numbered_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Απόκτηση της αξίας του αξιώματος. "
            "2. Απόκτηση της αξίας του αξιώματος. "
            "3. Απόκτηση της αξίας του αξιώματος. "
            "4. Απόκτηση της αξίας του αξιώματος. "
            "5. Απόκτηση της αξίας του αξιώματος.\n"
        ),
        stem="ocr-hybrid-same-body",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-hybrid-same-body__debug_page_00001.md").read_text(encoding="utf-8")
    assert "<match of type hybrid_repeat kind=same_body_progression" in content
    assert "5. Απόκτηση της αξίας του αξιώματος." in content


def test_clean_ocr_hybrid_debug_flags_hierarchical_heading_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1.1 Hypergeometric function "
            "1.1.1 Hypergeometric function "
            "1.1.2 Hypergeometric function "
            "1.1.3 Hypergeometric function "
            "1.1.4 Hypergeometric function "
            "1.1.5 Hypergeometric function\n"
        ),
        stem="ocr-hybrid-hierarchical",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-hybrid-hierarchical__debug_page_00001.md").read_text(encoding="utf-8")
    assert "<match of type hybrid_repeat kind=same_body_progression" in content
    assert "1.1.5 Hypergeometric function" in content


def test_clean_ocr_hybrid_debug_extends_partial_tail_of_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Σχεδία 1.1. Σχεδία 1.2. Σχεδία 1.3. Σχεδία 1.4. Σχεδία 1.5. Σχεδ\n"
        ),
        stem="ocr-hybrid-partial-tail",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-hybrid-partial-tail__debug_page_00001.md").read_text(encoding="utf-8")
    assert "1.5. Σχεδ" in content
    assert content.index("1.5. Σχεδ") < content.index("</match>")


def test_clean_ocr_hybrid_debug_flags_body_cycle_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Εισαγωγή 2. Φυσικοχημικές ιδιότητες 3. Φάσεις 4. Επιπλοκές "
            "5. Εισαγωγή 6. Φυσικοχημικές ιδιότητες 7. Φάσεις 8. Επιπλοκές "
            "9. Εισαγωγή 10. Φυσικοχημικές ιδιότητες 11. Φάσεις 12. Επιπλοκές\n"
        ),
        stem="ocr-hybrid-cycle",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-hybrid-cycle__debug_page_00001.md").read_text(encoding="utf-8")
    assert "<match of type hybrid_repeat kind=body_cycle_progression" in content
    assert "cycle=4" in content


def test_clean_ocr_hybrid_debug_flags_inline_numeric_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Από το σημείο 1, το σημείο 2, το σημείο 3, "
            "το σημείο 4, το σημείο 5, το σημείο 6, το σημείο 7.\n"
        ),
        stem="ocr-hybrid-inline-progress",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-hybrid-inline-progress__debug_page_00001.md").read_text(encoding="utf-8")
    assert "<match of type hybrid_repeat kind=inline_numeric_progression" in content
    assert "το σημείο 7" in content


def test_clean_ocr_hybrid_debug_ignores_short_inline_numeric_run(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Από το σημείο 1, το σημείο 2, το σημείο 3, "
            "το σημείο 4, το σημείο 5.\n"
        ),
        stem="ocr-hybrid-inline-short-ignore",
        max_docs=1,
    )
    assert rows == []
    assert not any(debug_dir.glob("*.md"))


def test_clean_ocr_hybrid_debug_ignores_diverse_numbered_list(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            "1. Εισαγωγή 2. Μέθοδοι 3. Αποτελέσματα 4. Συζήτηση 5. Συμπεράσματα\n"
        ),
        stem="ocr-hybrid-diverse-ignore",
        max_docs=1,
    )
    assert rows == []
    assert not any(debug_dir.glob("*.md"))


def test_clean_ocr_hybrid_debug_ignores_markup_number_progression(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path)
    rows, debug_dir = _run_clean_ocr_hybrid_debug_export(
        corpus,
        (
            '<img src="image_1.png" alt="Π" > '
            '<img src="image_2.png" alt="Π" > '
            '<img src="image_3.png" alt="Π" > '
            '<img src="image_4.png" alt="Π" >\n'
        ),
        stem="ocr-hybrid-markup-ignore",
        max_docs=1,
    )
    assert rows == []
    assert not any(debug_dir.glob("*.md"))


def test_clean_ocr_latex_slot_progression_debug_flags_derivative_ladder(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\[  \frac{d^2\Psi}{dr_*^2} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^2\Psi}{dr^2} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^3\Psi}{dr_*^3} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^3\Psi}{dr^3} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^4\Psi}{dr_*^4} + (\omega^2 - V(r))\Psi = 0  \]",
            r"\[  \frac{d^4\Psi}{dr^4} + (\omega^2 - V(r))\Psi = 0  \]",
        ]
    )
    rows, debug_dir = _run_clean_ocr_latex_slot_progression_debug_export(
        corpus,
        repeated + "\n",
        stem="ocr-latex-slot-derivative",
        max_docs=1,
    )
    assert len(rows) == 1
    content = (debug_dir / "ocr-latex-slot-derivative__debug_page_00001.md").read_text(encoding="utf-8")
    assert "<match of type latex_repeat kind=slot_progression" in content
    assert content.count("<match of type latex_repeat") == 1


def test_clean_ocr_latex_slot_progression_debug_ignores_small_parameter_family(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path)
    repeated = "\n\n".join(
        [
            r"\( f_{11}(k) = (1 - 0.0561)^{k-1}0.0561 \)",
            r"\( f_{12}(k) = (1 - 0.0617)^{k-1}0.0617 \)",
            r"\( f_{21}(k) = (1 - 0.1057)^{k-1}0.1057 \)",
            r"\( f_{22}(k) = (1 - 0.1724)^{k-1}0.1724 \)",
        ]
    )
    rows, debug_dir = _run_clean_ocr_latex_slot_progression_debug_export(
        corpus,
        repeated + "\n",
        stem="ocr-latex-slot-parameter-family-ignore",
        max_docs=1,
    )
    assert rows == []
    assert not any(debug_dir.glob("*.md"))


def test_review_manifest_materialize_creates_labeled_copies(tmp_path: Path) -> None:
    source_dir = tmp_path / "contexts"
    source_dir.mkdir()
    first = source_dir / "case_001.txt"
    second = source_dir / "case_002.txt"
    first.write_text("alpha body\n", encoding="utf-8")
    second.write_text("beta body\n", encoding="utf-8")

    manifest = tmp_path / "semantic_review_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "path": str(first),
                        "label": "fits_semantically",
                        "confidence": "high",
                        "notes": "complete",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "path": str(second),
                        "label": "fits_but_truncated_or_incomplete",
                        "confidence": "medium",
                        "notes": "cut off",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "categorized"
    summary = materialize_manifest_categories(
        manifest,
        output_dir,
        category_name="semantic_fit",
    )

    assert summary["row_count"] == 2
    fit_copy = output_dir / "by_label" / "fits_semantically" / "case_001.txt"
    trunc_copy = output_dir / "by_label" / "fits_but_truncated_or_incomplete" / "case_002.txt"
    assert fit_copy.exists()
    assert trunc_copy.exists()

    fit_text = fit_copy.read_text(encoding="utf-8")
    assert "REVIEW_CATEGORY: semantic_fit" in fit_text
    assert "REVIEW_LABEL: fits_semantically" in fit_text
    assert "=== REVIEW_SOURCE_CONTENT ===" in fit_text
    assert "alpha body" in fit_text


def test_table_markdown_audit_preserves_semantic_inline_html() -> None:
    audit = audit_table(
        Path("/tmp/demo.md"),
        1,
        1,
        (
            "<table><tr><td>Line A<br/>Line B</td>"
            "<td>x<sub>i</sub><sup>2</sup></td>"
            "<td><a href=\"https://example.com\">source</a></td>"
            "<td><img alt=\"diagram\" src=\"x.png\"/></td></tr></table>"
        ),
    )
    assert audit.convertible is True
    assert audit.markdown is not None
    assert "Line A<br>Line B" in audit.markdown
    assert "x<sub>i</sub><sup>2</sup>" in audit.markdown
    assert "[source](https://example.com)" in audit.markdown
    assert "diagram" in audit.markdown


def test_table_markdown_audit_writes_clean_markdown_file(tmp_path: Path) -> None:
    audit = audit_table(
        Path("/tmp/demo.md"),
        1,
        7,
        "<table><tr><td>Α</td><td>Β</td></tr><tr><td>1</td><td>2</td></tr></table>",
    )
    output = write_clean_markdown_file(tmp_path, audit)
    assert output is not None
    path = Path(output)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("## ORIGINAL_HTML")
    assert "## GITHUB_MD" in text
    assert "<table>" in text
    assert "Α" in text
    assert "1" in text
