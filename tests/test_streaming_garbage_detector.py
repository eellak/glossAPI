from pathlib import Path

import pytest

from glossapi.ocr.utils.cleaning import (
    StreamingGarbageDetector,
    detect_approx_token_triplet_repeat,
)


DOWNLOAD_EXPORT = (
    Path.home()
    / "Downloads"
    / "deepseek_ocr_43pdfs_allpages_20260331"
)


def _stream_detect(text: str, *, chunk_size: int) -> tuple[bool, str | None]:
    detector = StreamingGarbageDetector()
    for idx in range(0, len(text), max(1, int(chunk_size))):
        if detector.feed(text[idx : idx + chunk_size]):
            return True, detector.triggered_reason
    return False, detector.triggered_reason


def _load_real_markdown_garbage() -> str:
    root = DOWNLOAD_EXPORT / "corrections_markdown_garbage"
    if not root.exists():
        pytest.skip(f"missing local export: {root}")
    for path in sorted(root.glob("*__markdown_original.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "\uf0b7" in text or "" in text or "" in text:
            return text
    pytest.skip("no local symbol-garbage sample found")


def _load_real_empty_page_numeric_garbage() -> str:
    if not DOWNLOAD_EXPORT.exists():
        pytest.skip(f"missing local export: {DOWNLOAD_EXPORT}")
    preferred = DOWNLOAD_EXPORT / (
        "000008__04afb897cb954a76fe378b2ca22f2f059097876fa60a57666de75e37319e5968__p0008__markdown_original.md"
    )
    candidates = [preferred] if preferred.exists() else sorted(DOWNLOAD_EXPORT.glob("*__markdown_original.md"))
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "1. 2. 3." in text:
            return text
    pytest.skip("no local numeric-list garbage sample found")


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 17])
def test_streaming_detector_catches_symbol_garbage_across_chunks(chunk_size):
    text = "Κανονικό κείμενο\n" + (" " * 20)
    triggered, reason = _stream_detect(text, chunk_size=chunk_size)
    assert triggered is True
    assert reason == "symbol_garbage"


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 11])
def test_streaming_detector_catches_numeric_list_garbage_across_chunks(chunk_size):
    text = " ".join(f"{idx}." for idx in range(1, 25))
    triggered, reason = _stream_detect(text, chunk_size=chunk_size)
    assert triggered is True
    assert reason == "numeric_list_garbage"


def test_streaming_detector_ignores_non_ascii_digit_glyphs():
    triggered, reason = _stream_detect("x³ y² z¹", chunk_size=1)
    assert triggered is False
    assert reason is None


@pytest.mark.parametrize("chunk_size", [1, 3, 9, 23])
def test_streaming_detector_real_faulty_page_from_downloads(chunk_size):
    text = _load_real_markdown_garbage()
    triggered, reason = _stream_detect(text, chunk_size=chunk_size)
    assert triggered is True
    assert reason == "symbol_garbage"


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 21])
def test_streaming_detector_real_empty_page_generation_from_downloads(chunk_size):
    text = _load_real_empty_page_numeric_garbage()
    triggered, reason = _stream_detect(text, chunk_size=chunk_size)
    assert triggered is True
    assert reason == "numeric_list_garbage"


def test_detect_approx_token_triplet_repeat_catches_exact_triplets():
    detail = detect_approx_token_triplet_repeat(
        [91, 42, 13, 91, 42, 13, 91, 42, 13],
        max_period=3,
        max_frac=0.0,
        min_k=0,
    )

    assert detail is not None
    assert detail["rule"] == "approx_token_triplet_repeat"
    assert detail["period"] == 3
    assert detail["first_second_hamming"] == 0
    assert detail["second_third_hamming"] == 0


def test_detect_approx_token_triplet_repeat_avoids_false_positive_on_small_period_noise():
    detail = detect_approx_token_triplet_repeat(
        [1, 2, 3, 4, 5, 6],
        max_period=2,
        max_frac=0.1,
        min_k=0,
    )

    assert detail is None


def test_detect_approx_token_triplet_repeat_clamps_single_token_period_to_exact_match():
    assert detect_approx_token_triplet_repeat(
        [1, 2, 3],
        max_period=1,
        max_frac=1.0,
        min_k=1,
    ) is None

    detail = detect_approx_token_triplet_repeat(
        [7, 7, 7],
        max_period=1,
        max_frac=1.0,
        min_k=1,
    )

    assert detail is not None
    assert detail["period"] == 1
