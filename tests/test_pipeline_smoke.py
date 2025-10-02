import os
from pathlib import Path

import onnxruntime as ort
import pandas as pd
import pytest
import torch

from glossapi import Corpus


pytest.importorskip("docling")
pytest.importorskip("glossapi_rs_cleaner")


def _write_pdf(path: Path, text: str | None) -> None:
    """Create a tiny PDF containing the provided text (or a blank page)."""

    def _obj(obj_id: int, body: str) -> bytes:
        return f"{obj_id} 0 obj\n{body}\nendobj\n".encode("utf-8")

    escaped = "" if text is None else text.replace("(", r"\(").replace(")", r"\)")
    if escaped:
        stream_body = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n"
    else:
        stream_body = "BT\nET\n"
    stream_bytes = stream_body.encode("utf-8")

    objects = [
        _obj(1, "<< /Type /Catalog /Pages 2 0 R >>"),
        _obj(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        _obj(
            3,
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        _obj(
            4,
            f"<< /Length {len(stream_bytes)} >>\nstream\n{stream_body}\nendstream",
        ),
        _obj(5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    output = bytearray(header)
    offsets: list[int] = []
    for chunk in objects:
        offsets.append(len(output))
        output.extend(chunk)

    xref_start = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    output.extend(trailer.encode("ascii"))

    path.write_bytes(bytes(output))


def _assert_dir_contents(
    root: Path,
    allowed_suffixes: set[str],
    *,
    allowed_dirs: set[str] = frozenset(),
    allowed_exact: set[str] = frozenset(),
) -> None:
    assert root.exists(), f"Missing directory: {root}"
    for entry in root.iterdir():
        if entry.is_dir():
            if entry.name in allowed_dirs:
                continue
            pytest.fail(f"Unexpected subdirectory {entry} in {root}")
        else:
            name = entry.name
            if name in allowed_exact:
                continue
            if not any(name.endswith(suffix) for suffix in allowed_suffixes):
                pytest.fail(f"Unexpected file {entry} in {root}")


def test_pipeline_smoke_and_artifacts(tmp_path):
    assert torch.cuda.is_available(), "CUDA GPU expected for pipeline smoke test"
    providers = ort.get_available_providers()
    assert "CUDAExecutionProvider" in providers, f"CUDAExecutionProvider missing: {providers}"

    device_idx = 0
    if torch.cuda.device_count() > 1:
        device_idx = torch.cuda.current_device()

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    good_pdf = corpus_dir / "text.pdf"
    blank_pdf = corpus_dir / "blank.pdf"
    _write_pdf(good_pdf, "A simple line of text")
    _write_pdf(blank_pdf, None)

    corpus = Corpus(input_dir=corpus_dir, output_dir=corpus_dir)

    corpus.extract(
        input_format="pdf",
        accel_type="CUDA",
        num_threads=1,
        emit_formula_index=True,
        phase1_backend="docling",
        force_ocr=True,
        use_gpus="single",
        devices=[device_idx],
    )

    corpus.clean()

    parquet_path = corpus_dir / "download_results" / "download_results.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    needs = df.set_index("filename").get("needs_ocr")
    assert bool(needs.get("blank.pdf")), "Blank PDF should be flagged for OCR"
    assert not bool(needs.get("text.pdf"))

    corpus.ocr(
        mode="ocr_bad",
        use_gpus="single",
        devices=[device_idx],
        math_enhance=False,
    )

    corpus.section()

    markdown_dir = corpus_dir / "markdown"
    clean_markdown_dir = corpus_dir / "clean_markdown"
    json_dir = corpus_dir / "json"
    metrics_dir = json_dir / "metrics"
    downloads_dir = corpus_dir / "downloads"
    sections_dir = corpus_dir / "sections"

    _assert_dir_contents(downloads_dir, {".pdf"})
    _assert_dir_contents(
        markdown_dir,
        {".md"},
        allowed_dirs={"problematic_files", "timeout_files"},
        allowed_exact={".processing_state.pkl"},
    )
    _assert_dir_contents(
        clean_markdown_dir,
        {".md"},
        allowed_dirs={"timeout_files", "problematic_files"},
        allowed_exact={".processing_state.pkl"},
    )
    if json_dir.exists():
        _assert_dir_contents(
            json_dir,
            {".docling.json", ".docling.json.zst", ".formula_index.jsonl"},
            allowed_dirs={"metrics"},
        )
        if metrics_dir.exists():
            _assert_dir_contents(metrics_dir, {".json"})
    _assert_dir_contents(corpus_dir / "download_results", {".parquet"})
    _assert_dir_contents(sections_dir, {".parquet"})

    sections_file = sections_dir / "sections_for_annotation.parquet"
    assert sections_file.exists()
