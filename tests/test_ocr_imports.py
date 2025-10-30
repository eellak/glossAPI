import importlib.util

import pytest


def test_import_ocr_package_is_lightweight():
    # Import should not require heavy stacks
    import glossapi.ocr as ocr

    assert hasattr(ocr, "deepseek")
    assert hasattr(ocr, "rapidocr")

    # New subpackages remain importable lazily
    import glossapi.ocr.deepseek.runner as deepseek_runner
    import glossapi.ocr.rapidocr.dispatch as rapid_dispatch

    assert ocr.deepseek.runner is deepseek_runner
    assert ocr.rapidocr.dispatch is rapid_dispatch
    assert ocr.deepseek_runner is deepseek_runner
    assert ocr.rapidocr_dispatch is rapid_dispatch
    assert hasattr(deepseek_runner, "run_for_files")
    assert hasattr(rapid_dispatch, "run_via_extract")

    # Utilities module always available (pure Python)
    from glossapi.ocr.utils import json_io as utils_json

    assert hasattr(utils_json, "export_docling_json")

    if importlib.util.find_spec("docling") is not None:
        try:
            from glossapi.ocr.rapidocr import pool as rapid_pool
        except ModuleNotFoundError:
            pytest.skip("Docling optional dependencies not available")
        else:
            assert hasattr(rapid_pool, "GLOBAL_RAPID_OCR_POOL")

    if importlib.util.find_spec("docling_core") is not None:
        try:
            from glossapi.ocr.math import enrich_from_docling_json, RoiEntry
        except ModuleNotFoundError:
            pytest.skip("Docling-core optional dependencies not available")
        else:
            assert callable(enrich_from_docling_json)
            assert RoiEntry.__module__.startswith("glossapi.ocr.math.enrich")
