"""T1-G06: frozen behaviour oracles for the broad ftfy and PyYAML ranges.

`gloss_extract` calls `ftfy.fix_text` with its defaults, which repair far more than NFC; the download policy is read
with `yaml.safe_load`, whose YAML 1.1 scalars and merge keys differ from JSON/TOML. Any ftfy or PyYAML release inside
the declared ranges, or a replacement helper/loader, must keep these outputs. The ftfy cases are the
`ftfy_behavior_probe.json` samples recorded under ftfy 6.3.1 (deps_20261028, T1).
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest

pytest.importorskip("ftfy")
yaml = pytest.importorskip("yaml")

from glossapi.download_policy import load_download_policy, read_policy_file  # noqa: E402
from glossapi.gloss_extract import GlossExtract  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

FTFY_CASES = [
    ("Î±Î²", "αβ"),  # Greek UTF-8 read as Latin-1 (mojibake)
    ("FranÃ§ais", "Français"),
    ("â€™", "'"),  # mojibake curly apostrophe, then uncurled
    ("﻿alpha", "alpha"),  # byte-order mark
    ("\u001b[31mred\u001b[0m", "red"),  # terminal escapes
    ("ＡＢＣ", "ABC"),  # full-width letters
    ("ﬁ", "fi"),  # Latin ligature
    ("é", "é"),  # NFC
    ("&#x3b1;", "α"),  # HTML entity (unescape_html="auto")
    ("&amp;", "&"),
    ("“quote”", '"quote"'),  # uncurl quotes
    ("a\r\nb", "a\nb"),  # line breaks
    ("x\u0000y", "xy"),  # control characters
    ("\u0085", "…"),  # C1 control read as Windows-1252
    ("\ud800", "�"),  # lone surrogate
    ("ἄνθρωπος", "ἄνθρωπος"),  # polytonic kept
]


@pytest.mark.parametrize(("text", "expected"), FTFY_CASES, ids=[ascii(text) for text, _ in FTFY_CASES])
def test_gloss_extract_text_fix_is_frozen(text: str, expected: str) -> None:
    assert GlossExtract()._fix_greek_text(text) == expected


def test_sample_download_policy_safe_load_is_frozen() -> None:
    path = REPO / "samples" / "openarchives_download_policy.yml"
    data = read_policy_file(path)
    assert data["default"] == {
        "downloader": "standard", "request_timeout": 60, "ssl_verify": True, "per_domain_concurrency": 8,
        "domain_concurrency_floor": 1, "domain_concurrency_ceiling": 12, "skip_failed_after": 3, "sleep": 0.25,
    }
    assert [rule["match"]["domains"] for rule in data["rules"]][:2] == [["ikee.lib.auth.gr"], ["dspace.lib.ntua.gr"]]
    olympias = data["rules"][2]
    assert olympias["ssl_verify"] is False and olympias["sleep"] == 1.0 and isinstance(olympias["sleep"], float)
    assert data["rules"][6]["browser_timeout_ms"] == 90000 and len(data["rules"]) == 8
    policy = load_download_policy(path)
    assert policy.default_downloader == "standard"
    assert [rule.downloader for rule in policy.rules] == [
        "standard", "standard", "standard", "standard", "auto", "standard", "auto", "standard"]


def test_policy_yaml_1_1_scalars_and_merge_keys_are_frozen(tmp_path: Path) -> None:
    """Counterexamples a JSON/TOML/YAML 1.2 loader would read differently."""
    path = tmp_path / "policy.yml"
    path.write_text(
        "base: &base {ssl_verify: on, sleep: 0.5}\n"
        "default:\n  <<: *base\n  ssl_verify: no\n"
        "values: [yes, off, 010, 1e3, 2026-11-02, ~, '1']\n",
        encoding="utf-8",
    )
    data = read_policy_file(path)
    assert data["base"] == {"ssl_verify": True, "sleep": 0.5}
    assert data["default"] == {"ssl_verify": False, "sleep": 0.5}
    assert data["values"] == [True, False, 8, "1e3", datetime.date(2026, 11, 2), None, "1"]
