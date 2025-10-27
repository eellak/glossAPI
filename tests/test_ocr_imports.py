def test_import_ocr_package_is_lightweight():
    # Import should not require heavy stacks
    import glossapi.ocr as ocr

    assert hasattr(ocr, "deepseek_runner")
    assert hasattr(ocr, "rapidocr_dispatch")

