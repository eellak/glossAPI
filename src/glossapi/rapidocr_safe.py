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
            return []
        boxes = getattr(result, "boxes", None)
        txts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or txts is None or scores is None:
            _log.warning("RapidOCR returned incomplete data; treating as empty")
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
        from docling.models import ocr_factory  # type: ignore

        factory = ocr_factory.OCRModelFactory()
        factory._classes["rapidocr"] = SafeRapidOcrModel
    except Exception as exc:  # pragma: no cover - best effort
        import logging

        logging.getLogger(__name__).warning(
            "Failed to re-register SafeRapidOcrModel: %s", exc
        )
    return True


__all__ = ["SafeRapidOcrModel", "patch_docling_rapidocr"]
