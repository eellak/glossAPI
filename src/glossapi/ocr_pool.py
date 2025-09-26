"""Shared RapidOCR engine pooling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, Optional, Union, Type

from docling.datamodel.pipeline_options import RapidOcrOptions


@dataclass(frozen=True)
class _PoolKey:
    device: str
    det_model_path: str
    rec_model_path: str
    cls_model_path: str
    lang: Tuple[str, ...]
    text_score: float
    use_det: bool
    use_cls: bool
    use_rec: bool


class RapidOcrEnginePool:
    """Process-local cache of RapidOCR models keyed by configuration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._cache: Dict[_PoolKey, object] = {}

    def _make_key(self, device: str, opts: RapidOcrOptions) -> _PoolKey:
        lang = tuple(opts.lang or [])
        return _PoolKey(
            device=str(device),
            det_model_path=str(getattr(opts, "det_model_path", "")),
            rec_model_path=str(getattr(opts, "rec_model_path", "")),
            cls_model_path=str(getattr(opts, "cls_model_path", "")),
            lang=lang,
            text_score=float(getattr(opts, "text_score", 0.0)),
            use_det=bool(getattr(opts, "use_det", True)),
            use_cls=bool(getattr(opts, "use_cls", False)),
            use_rec=bool(getattr(opts, "use_rec", True)),
        )

    def get(
        self,
        device: str,
        opts: RapidOcrOptions,
        factory: Callable[[], object],
        *,
        expected_type: Optional[Union[Type[object], tuple[Type[object], ...]]] = None,
    ) -> object:
        key = self._make_key(device, opts)
        with self._lock:
            model = self._cache.get(key)
            if expected_type is not None and model is not None and not isinstance(model, expected_type):
                self._cache.pop(key, None)
                model = None
            if model is None:
                model = factory()
                if expected_type is None or isinstance(model, expected_type):
                    self._cache[key] = model
            return model

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


GLOBAL_RAPID_OCR_POOL = RapidOcrEnginePool()

__all__ = ["RapidOcrEnginePool", "GLOBAL_RAPID_OCR_POOL"]
