from __future__ import annotations

from pathlib import Path

import pandas as pd

from glossapi.scripts.openarchives_ocr_run_node import (
    _normalize_download_results,
    _prepare_download_input,
)


def test_prepare_download_input_adds_url_and_filename_base() -> None:
    df = pd.DataFrame(
        [
            {
                "filename": "ABC_001.pdf",
                "pdf_url": "https://example.com/a.pdf",
                "needs_ocr": True,
            }
        ]
    )
    out = _prepare_download_input(df)
    assert out.loc[0, "url"] == "https://example.com/a.pdf"
    assert out.loc[0, "filename_base"] == "ABC_001"


def test_normalize_download_results_preserves_shard_filename_and_metadata() -> None:
    shard = pd.DataFrame(
        [
            {
                "filename": "ABC_001.pdf",
                "pdf_url": "https://example.com/a.pdf",
                "filename_base": "ABC_001",
                "needs_ocr": True,
                "source_doc_id": "doc-1",
            }
        ]
    )
    dl = pd.DataFrame(
        [
            {
                "filename": "ABC_001.pdf",
                "filename_base": "ABC_001",
                "download_success": True,
                "download_error": "",
                "url": "https://example.com/a.pdf",
            }
        ]
    )
    out = _normalize_download_results(shard_df=shard, download_results_df=dl)
    assert out.loc[0, "filename"] == "ABC_001.pdf"
    assert out.loc[0, "source_doc_id"] == "doc-1"
    assert bool(out.loc[0, "download_success"]) is True
    assert bool(out.loc[0, "needs_ocr"]) is True
