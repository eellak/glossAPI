"""Mode A: replace routed native textline cells before Docling layout."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from docling.models.base_ocr_model import BaseOcrModel
from docling_core.types.doc.page import TextCell

from .options import GlossapiOcrOptions
from .recognizer.ckpt_io import sha256_file
from .recognizer.strips import extract_strip

_log = logging.getLogger(__name__)

# Frozen sidecar scans have 0..49 chars (numbers/captions); the shortest
# garbage-text page has 66. This coverage heuristic is not a body detector.
RASTER_BODY_CHAR_LIMIT = 50
RASTER_BODY_IMAGE_RATIO = 0.5


class GlossapiOcrModel(BaseOcrModel):
    def __init__(self, *, enabled, artifacts_path, options, accelerator_options):
        super().__init__(enabled=enabled, artifacts_path=artifacts_path,
                         options=options, accelerator_options=accelerator_options)
        self.routes: dict[str, set[int]] = {}
        self.route_metrics: dict[str, dict[int, dict]] = {}
        self.unsupported_counts: Counter = Counter()
        self.call_log: list[dict] = []
        self.recognizer = None
        if not enabled:
            return
        if options.routing_sidecar is not None:
            for line in options.routing_sidecar.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                filename = row.get("filename") or Path(row["path"]).name
                # Sidecars may reference repaired_pdfs; the stable filename binds
                # them to original inputs. Conflicting duplicate filenames fail closed.
                pages = {int(p["page"]) for p in row["pages"]
                         if p["page_needs_ocr"] is True}
                metrics = {int(p["page"]): p for p in row["pages"]}
                if filename in self.routes and (self.routes[filename] != pages or
                                               self.route_metrics[filename] != metrics):
                    raise ValueError(f"conflicting routing rows for {filename}")
                self.routes[filename] = pages
                self.route_metrics[filename] = metrics

    @classmethod
    def get_options_type(cls):
        return GlossapiOcrOptions

    def _get_recognizer(self):
        if self.recognizer is None:
            from .recognizer.inference import Recognizer

            opts = self.options
            if opts.checkpoint_sha256 is not None:
                if sha256_file(str(opts.checkpoint)) != opts.checkpoint_sha256:
                    raise ValueError("OCR checkpoint sha256 mismatch")
                # A .pt pin must not silently resolve to different sibling weights.
                if opts.checkpoint.suffix == ".pt" and opts.checkpoint.with_suffix(".safetensors").exists():
                    raise ValueError("pinned .pt has a safetensors sibling; pin that artifact explicitly")
            self.recognizer = Recognizer(str(opts.checkpoint), str(opts.codec), opts.device)
        return self.recognizer

    def _record(self, record):
        self.call_log.append(record)
        if record["status"] == "unsupported":
            self.unsupported_counts[record["reason"]] += 1
            _log.warning("OCR unsupported: %s page %s: %s",
                         record["filename"], record["page"], record["reason"])
        if self.options.call_log is not None:
            self.options.call_log.parent.mkdir(parents=True, exist_ok=True)
            with self.options.call_log.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _install(cells, page):
        # Equivalent to BaseOcrModel.post_process_cells under full replacement,
        # restricted to THIS page; never mutate the pipeline's global OCR option.
        parsed = page.parsed_page
        parsed.textline_cells = cells
        parsed.has_lines = bool(cells)
        parsed.word_cells = [c for c in parsed.word_cells if c.from_ocr]
        parsed.char_cells = [c for c in parsed.char_cells if c.from_ocr]
        parsed.has_words = bool(parsed.word_cells)
        parsed.has_chars = bool(parsed.char_cells)

    def __call__(self, conv_res, page_batch):
        if not self.enabled:
            yield from page_batch
            return
        filename = conv_res.input.file.name
        for page in page_batch:
            replace = (self.options.replace_policy == "all" or
                       page.page_no in self.routes.get(filename, set()))
            if not replace:
                yield page
                continue
            record = {"filename": filename, "page": page.page_no,
                      "mode": self.options.mode, "crop_scale": self.options.crop_scale,
                      "status": "unsupported", "timings_s": {}}
            if self.options.mode == "detector":
                record["reason"] = "detector_out_of_scope"
                self._record(record)
                yield page
                continue
            parsed = page.parsed_page
            if parsed is None or not parsed.textline_cells:
                record.update(reason="no_textlayer_lines", source_line_count=0)
                self._record(record)
                yield page
                continue
            assert parsed.has_lines, "textline cells present but has_lines=False"
            source = list(parsed.textline_cells)
            record["source_line_count"] = len(source)
            metrics = self.route_metrics.get(filename, {}).get(page.page_no, {})
            # Prefer the routing sidecar's original text-layer measurement. Older
            # minimal routing files retain native-line behavior without scan evidence.
            chars = metrics.get("chars", sum(len(c.text.strip()) for c in source))
            raster_body = (chars < RASTER_BODY_CHAR_LIMIT and
                           (metrics.get("decision") == "scanned_page" or
                            (metrics.get("image_area_ratio") or 0) >= RASTER_BODY_IMAGE_RATIO))
            record["body_support"] = dict(
                textlayer_chars=chars, chars_source="sidecar" if "chars" in metrics else "native_lines",
                decision=metrics.get("decision"), image_area_ratio=metrics.get("image_area_ratio"),
                char_limit=RASTER_BODY_CHAR_LIMIT, image_ratio_threshold=RASTER_BODY_IMAGE_RATIO)
            start = perf_counter()
            recognizer = self._get_recognizer()
            record["timings_s"]["recognizer_setup"] = perf_counter() - start
            start = perf_counter()
            image = page.get_image(scale=self.options.crop_scale)
            record["timings_s"]["render"] = perf_counter() - start
            if image is None:
                record["reason"] = "no_page_image"
                self._record(record)
                yield page
                continue
            start = perf_counter()
            gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            strips = []
            for cell in source:
                bb = cell.rect.to_bounding_box().to_top_left_origin(page_height=page.size.height)
                l, t, r, b = bb.scaled(self.options.crop_scale).as_tuple()
                strip = extract_strip(gray, (math.floor(l), math.floor(t), math.ceil(r), math.ceil(b)))
                if strip is None:
                    break
                strips.append(strip)
            record["timings_s"]["crop"] = perf_counter() - start
            if len(strips) != len(source):
                # Preserve the whole page if one cell cannot satisfy the crop contract.
                record["reason"] = "invalid_line_crop"
                self._record(record)
                yield page
                continue
            start = perf_counter()
            predictions = recognizer.recognize(strips)
            record["timings_s"]["recognise"] = perf_counter() - start
            assert len(predictions) == len(source)
            ours = [TextCell(index=i, from_ocr=True, rect=cell.rect,
                             text=text, orig=text, confidence=confidence)
                    for i, (cell, (text, confidence)) in enumerate(zip(source, predictions))]
            if raster_body:
                # Recognize and log existing furniture, but do not claim to recover
                # the raster body or replace its native cells with footer OCR.
                record.update(reason="raster_body", installed=False)
            else:
                self._install(ours, page)
                record.update(status="replaced", installed=True)
            record.update(output_line_count=len(ours), recognition_performed=True)
            if self.options.call_log is not None:
                record["lines"] = [{"source_rect": old.rect.model_dump(mode="json"),
                                    "rect": new.rect.model_dump(mode="json"),
                                    "source_text": old.text, "text": new.text,
                                    "confidence": new.confidence, "from_ocr": new.from_ocr}
                                   for old, new in zip(source, ours)]
            self._record(record)
            yield page
