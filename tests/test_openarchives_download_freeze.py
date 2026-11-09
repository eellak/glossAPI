from __future__ import annotations

from pathlib import Path

import pandas as pd

import glossapi.scripts.openarchives_download_freeze as freeze_mod
from glossapi.scripts.openarchives_download_freeze import main


def test_download_freeze_dry_run_materializes_manifest(tmp_path: Path) -> None:
    src = tmp_path / "input.parquet"
    pd.DataFrame(
        [
            {
                "filename": "ABC_001.pdf",
                "pdf_url": "https://example.com/a.pdf",
                "needs_ocr": True,
            }
        ]
    ).to_parquet(src, index=False)

    work_root = tmp_path / "work"
    rc = main(["--input-parquet", str(src), "--work-root", str(work_root), "--dry-run"])
    assert rc == 0
    assert (work_root / "manifests" / "download_input.parquet").exists()
    assert (work_root / "download_results" / "download_results.parquet").exists()


def test_download_freeze_uses_pdf_only_auto_mode(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "input.parquet"
    pd.DataFrame(
        [
            {
                "filename": "ABC_001.pdf",
                "pdf_url": "https://example.com/a.pdf",
                "needs_ocr": True,
            }
        ]
    ).to_parquet(src, index=False)

    observed = {}

    class DummyCorpus:
        def __init__(self, *args, **kwargs):
            observed["init"] = kwargs

        def download(self, **kwargs):
            observed["download"] = kwargs
            return pd.DataFrame(
                [
                    {
                        "url": "https://example.com/a.pdf",
                        "filename": "ABC_001.pdf",
                        "download_success": True,
                        "download_error": "",
                        "file_ext": "pdf",
                    }
                ]
            )

    monkeypatch.setattr(freeze_mod, "Corpus", DummyCorpus)

    work_root = tmp_path / "work"
    rc = main(["--input-parquet", str(src), "--work-root", str(work_root)])

    assert rc == 0
    assert observed["download"]["download_mode"] == "auto"
    assert observed["download"]["supported_formats"] == ["pdf"]
