"""Docling entry point, matching docling.models.plugins.defaults.ocr_engines."""


def ocr_engines():
    from .model import GlossapiOcrModel

    return {"ocr_engines": [GlossapiOcrModel]}
