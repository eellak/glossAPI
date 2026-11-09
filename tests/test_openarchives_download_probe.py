from __future__ import annotations

from pathlib import Path

import pandas as pd

from glossapi.scripts.openarchives_download_probe import _prepare_probe_frame


def test_prepare_probe_frame_limits_per_host_and_adds_runtime_columns() -> None:
    df = pd.DataFrame(
        [
            {"filename": "a.pdf", "pdf_url": "https://ikee.lib.auth.gr/file/a.pdf"},
            {"filename": "b.pdf", "pdf_url": "https://ikee.lib.auth.gr/file/b.pdf"},
            {"filename": "c.pdf", "pdf_url": "https://ikee.lib.auth.gr/file/c.pdf"},
            {"filename": "d.pdf", "pdf_url": "https://dspace.lib.ntua.gr/file/d.pdf"},
            {"filename": "e.pdf", "pdf_url": "https://dspace.lib.ntua.gr/file/e.pdf"},
        ]
    )

    out = _prepare_probe_frame(
        df,
        samples_per_host=2,
        max_hosts=2,
        seed=7,
    )

    counts = out.groupby("host").size().to_dict()
    assert counts["ikee.lib.auth.gr"] == 2
    assert counts["dspace.lib.ntua.gr"] == 2
    assert set(out["url"]) <= set(df["pdf_url"])
    assert set(out["base_domain"]) == {"https://ikee.lib.auth.gr", "https://dspace.lib.ntua.gr"}
