#!/usr/bin/env python3
"""
Minimal DeepSeek OCR integration smoke test.

This script runs the GlossAPI DeepSeek backend on a tiny sample PDF and
verifies that real Markdown output is produced. It requires the DeepSeek-OCR
weights to be available under ``../deepseek-ocr/DeepSeek-OCR`` relative to
the repository root (override via ``DEEPSEEK_MODEL_DIR``).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "samples" / "lightweight_pdf_corpus" / "pdfs"
DEFAULT_MODEL_ROOT = (REPO_ROOT / ".." / "deepseek-ocr").resolve()


def ensure_model_available(model_root: Path) -> None:
    expected = model_root / "DeepSeek-OCR" / "model-00001-of-000001.safetensors"
    if not expected.exists() or expected.stat().st_size < 1_000_000:
        raise FileNotFoundError(
            f"Expected DeepSeek-OCR weights at {expected}. "
            "Download the checkpoint (huggingface.co/deepseek-ai/DeepSeek-OCR) "
            "or set DEEPSEEK_MODEL_DIR to the directory that contains them."
        )


def run_smoke(model_root: Path) -> None:
    from glossapi import Corpus

    ensure_model_available(model_root)
    sample_pdf = SAMPLES_DIR / "sample01_plain.pdf"
    if not sample_pdf.exists():
        raise FileNotFoundError(f"Sample PDF not found: {sample_pdf}")

    with tempfile.TemporaryDirectory(prefix="deepseek_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        input_dir = tmp_dir / "input"
        output_dir = tmp_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        target_pdf = input_dir / sample_pdf.name
        shutil.copy2(sample_pdf, target_pdf)

        dl_dir = output_dir / "download_results"
        dl_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "filename": sample_pdf.name,
                    "url": "",
                    "needs_ocr": True,
                    "ocr_success": False,
                }
            ]
        )
        parquet_path = dl_dir / "download_results.parquet"
        df.to_parquet(parquet_path, index=False)

        os.environ.setdefault("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "0")
        os.environ.setdefault(
            "GLOSSAPI_DEEPSEEK_VLLM_SCRIPT",
            str(model_root / "run_pdf_ocr_vllm.py"),
        )
        os.environ.setdefault(
            "GLOSSAPI_DEEPSEEK_PYTHON",
            sys.executable,
        )
        ld_extra = os.environ.get("GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH") or str(
            model_root / "libjpeg-turbo" / "lib"
        )
        os.environ["GLOSSAPI_DEEPSEEK_LD_LIBRARY_PATH"] = ld_extra
        os.environ["LD_LIBRARY_PATH"] = (
            f"{ld_extra}:{os.environ.get('LD_LIBRARY_PATH','')}".rstrip(":")
        )

        corpus = Corpus(input_dir=input_dir, output_dir=output_dir)
        corpus.ocr(
            backend="deepseek",
            fix_bad=True,
            math_enhance=False,
            reprocess_completed=True,
        )

        md_path = output_dir / "markdown" / (sample_pdf.stem + ".md")
        if not md_path.exists():
            raise RuntimeError(f"DeepSeek OCR did not produce {md_path}")
        if not md_path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"DeepSeek Markdown output is empty: {md_path}")


def main() -> None:
    model_dir_env = os.environ.get("DEEPSEEK_MODEL_DIR")
    if model_dir_env:
        model_root = Path(model_dir_env).expanduser().resolve()
    else:
        model_root = DEFAULT_MODEL_ROOT
    run_smoke(model_root)


if __name__ == "__main__":
    main()
