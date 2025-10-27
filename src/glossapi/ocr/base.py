from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DeepSeekConfig:
    model_dir: Optional[Path] = None
    dtype: str = "auto"  # e.g. "fp16", "bf16", "auto"
    max_tokens: int = 8192
    render_dpi: int = 220
    gpu_util: float = 0.9
    tensor_parallel: int = 1
    save_images: bool = False

