"""Temporary wrappers around Docling's RapidOCR integration.

The upstream Docling release (2.48.x) does not tolerate RapidOCR returning
``None`` for a given crop. That bubbles up as an AttributeError inside the
conversion loop and the entire document fails. Until Docling includes a fix, we
wrap the loader so that ``None`` simply means "no detections" and processing
continues. Once Docling ships a release with the guard we can drop this shim and
revert to the vanilla ``RapidOcrModel``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Type

import numpy

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, RapidOcrOptions
from docling.models.rapid_ocr_model import RapidOcrModel as _RapidOcrModel
from docling.models.rapid_ocr_model import TextCell, _log
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle


class SafeRapidOcrModel(_RapidOcrModel):
    """Drop-in RapidOCR wrapper that copes with ``None`` OCR results.

    Docling 2.48.0 assumes ``self.reader`` always returns an object with
    ``boxes/txts/scores``. RapidOCR occasionally yields ``None`` for problematic
    crops, which crashes the extractor. We normalise the return value before the
    original list(zip(...)) call and treat anything unexpected as "no boxes".
    Remove this once Docling hardens the upstream implementation.
    """

    # NOTE: keep signature identical so StandardPdfPipeline can instantiate it.
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: RapidOcrOptions,
        accelerator_options,
    ):
        if enabled:
            try:
                from ._rapidocr_paths import resolve_packaged_onnx_and_keys

                resolved = resolve_packaged_onnx_and_keys()

                _log.warning(
                    'SafeRapidOcrModel initial options: det=%s rec=%s cls=%s keys=%s',
                    getattr(options, 'det_model_path', None),
                    getattr(options, 'rec_model_path', None),
                    getattr(options, 'cls_model_path', None),
                    getattr(options, 'rec_keys_path', None),
                )

                if resolved.det:
                    options.det_model_path = resolved.det
                if resolved.rec:
                    options.rec_model_path = resolved.rec
                if resolved.cls:
                    options.cls_model_path = resolved.cls
                if resolved.keys:
                    options.rec_keys_path = resolved.keys

                try:
                    from rapidocr.ch_ppocr_rec import main as _rapidocr_rec_main

                    if not getattr(_rapidocr_rec_main.TextRecognizer, '_glossapi_patch', False):
                        original_get_character_dict = _rapidocr_rec_main.TextRecognizer.get_character_dict

                        def _patched_get_character_dict(self, cfg):
                            try:
                                current_keys = cfg.get('keys_path', None)
                                current_rec_keys = cfg.get('rec_keys_path', None)
                                if current_rec_keys is None and current_keys is not None:
                                    cfg['rec_keys_path'] = current_keys
                                    _log.warning('Patched RapidOCR cfg: set rec_keys_path from keys_path=%s', current_keys)
                                else:
                                    _log.warning('Patched RapidOCR cfg: existing rec_keys_path=%s keys_path=%s', current_rec_keys, current_keys)
                            except Exception:
                                _log.warning('RapidOCR cfg inspection failed', exc_info=True)
                            return original_get_character_dict(self, cfg)

                        _rapidocr_rec_main.TextRecognizer.get_character_dict = _patched_get_character_dict
                        _rapidocr_rec_main.TextRecognizer._glossapi_patch = True
                except Exception:
                    _log.warning('Failed to patch RapidOCR TextRecognizer for keys fallback', exc_info=True)

                _log.warning(
                    'SafeRapidOcrModel using packaged assets: det=%s rec=%s cls=%s keys=%s',
                    options.det_model_path,
                    options.rec_model_path,
                    options.cls_model_path,
                    options.rec_keys_path,
                )
            except Exception:
                _log.warning(
                    'SafeRapidOcrModel bootstrap failed to resolve packaged assets',
                    exc_info=True,
                )

        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions

    def _normalise_result(self, result):
        """Return an iterable of (bbox, text, score) triples.

        RapidOCR returns ``None`` or semi-populated structures in some corner
        cases. We swallow those and log a one-line warning so the page still
        progresses through the pipeline.
        """

        if result is None:
            _log.warning("RapidOCR returned None; skipping crop")
            return []
        boxes = getattr(result, "boxes", None)
        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or txts is None or scores is None:
            _log.warning("RapidOCR returned incomplete data; treating crop as empty")
            return []
        try:
            return list(zip(boxes.tolist(), txts, scores))
        except Exception as exc:  # pragma: no cover - defensive only
            _log.warning("RapidOCR result normalisation failed: %s", exc)
            return []

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)

                all_ocr_cells = []
                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue
                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    im = numpy.array(high_res_image)
                    raw_result = self.reader(
                        im,
                        use_det=self.options.use_det,
                        use_cls=self.options.use_cls,
                        use_rec=self.options.use_rec,
                    )
                    result = self._normalise_result(raw_result)
                    del high_res_image
                    del im

                    if not result:
                        continue

                    cells = [
                        TextCell(
                            index=ix,
                            text=line[1],
                            orig=line[1],
                            confidence=line[2],
                            from_ocr=True,
                            rect=BoundingRectangle.from_bounding_box(
                                BoundingBox.from_tuple(
                                    coord=(
                                        (line[0][0][0] / self.scale) + ocr_rect.l,
                                        (line[0][0][1] / self.scale) + ocr_rect.t,
                                        (line[0][2][0] / self.scale) + ocr_rect.l,
                                        (line[0][2][1] / self.scale) + ocr_rect.t,
                                    ),
                                    origin=CoordOrigin.TOPLEFT,
                                )
                            ),
                        )
                        for ix, line in enumerate(result)
                    ]
                    all_ocr_cells.extend(cells)

                self.post_process_cells(all_ocr_cells, page)

            from docling.datamodel.settings import settings

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page


def patch_docling_rapidocr() -> bool:
    """Replace Docling's RapidOcrModel with the safe shim if available."""

    try:
        import docling.models.rapid_ocr_model as rapid_module
    except Exception:  # pragma: no cover - Docling missing
        return False

    current = getattr(rapid_module, "RapidOcrModel", None)
    if current is SafeRapidOcrModel:
        return False

    rapid_module.RapidOcrModel = SafeRapidOcrModel
    try:
        from docling.models.factories import get_ocr_factory  # type: ignore
        import logging
    except Exception:
        return True

    try:
        factory = get_ocr_factory()
        options_type = SafeRapidOcrModel.get_options_type()

        if hasattr(factory, "classes"):
            factory.classes[options_type] = SafeRapidOcrModel
        elif hasattr(factory, "_classes"):
            factory._classes[options_type] = SafeRapidOcrModel
        logging.getLogger(__name__).info(
            "Registered SafeRapidOcrModel for %s", options_type
        )
        try:
            from docling.pipeline import standard_pdf_pipeline as _std_pdf  # type: ignore
            from docling.datamodel.pipeline_options import RapidOcrOptions  # type: ignore
            from functools import lru_cache
        except Exception as _exc:  # pragma: no cover - best effort
            logging.getLogger(__name__).warning(
                "Docling factory patch limited to local mutation: %s", _exc
            )
        else:
            original_get_factory = getattr(
                _std_pdf.get_ocr_factory, "__wrapped__", _std_pdf.get_ocr_factory
            )

            def _ensure_safe(factory_obj):
                try:
                    current = factory_obj.classes.get(RapidOcrOptions)
                    if current is not SafeRapidOcrModel:
                        factory_obj.classes[RapidOcrOptions] = SafeRapidOcrModel
                except AttributeError:
                    current = getattr(factory_obj, "_classes", {}).get(RapidOcrOptions)
                    if current is not SafeRapidOcrModel:
                        getattr(factory_obj, "_classes", {})[RapidOcrOptions] = SafeRapidOcrModel
                return factory_obj

            @lru_cache(maxsize=None)
            def _patched_get_ocr_factory(allow_external_plugins: bool = False):
                return _ensure_safe(original_get_factory(allow_external_plugins))

            _patched_get_ocr_factory.__wrapped__ = original_get_factory  # type: ignore[attr-defined]
            _std_pdf.get_ocr_factory = _patched_get_ocr_factory  # type: ignore[attr-defined]
            try:
                _ensure_safe(_std_pdf.get_ocr_factory(False))
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).warning(
            "Failed to re-register SafeRapidOcrModel: %s", exc
        )
    return True


__all__ = ["SafeRapidOcrModel", "patch_docling_rapidocr"]
