"""Options for the routed, per-page Docling OCR replacement engine."""
from pathlib import Path
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import Field, model_validator


class GlossapiOcrOptions(OcrOptions):
    # Docling's factory reads the discriminator on the CLASS, as for bundled engines.
    kind: ClassVar[Literal["glossapi"]] = "glossapi"
    lang: list[str] = Field(default_factory=lambda: ["el", "en"])
    checkpoint: Path
    codec: Path
    mode: Literal["textlayer", "detector"] = "textlayer"
    crop_scale: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    device: str = "cpu"
    routing_sidecar: Path | None = None
    replace_policy: Literal["routed_pages", "all"] = "routed_pages"
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    call_log: Path | None = None

    @model_validator(mode="after")
    def validate_replacement(self):
        if self.force_full_page_ocr:
            raise ValueError("glossapi uses per-page replacement; force_full_page_ocr must be False")
        if self.replace_policy == "routed_pages" and self.routing_sidecar is None:
            raise ValueError("routed_pages requires a routing_sidecar")
        return self
