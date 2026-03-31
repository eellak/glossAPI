from __future__ import annotations

from pathlib import Path

import pandas as pd

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
