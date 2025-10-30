from __future__ import annotations

import numpy as np


def is_mostly_blank_pix(pixmap, *, tolerance: int = 8, max_fraction: float = 0.0015) -> bool:
    """Heuristic check whether a PyMuPDF pixmap is mostly blank.

    Accepts a `fitz.Pixmap` instance. Returns True if the page is blank-ish.
    """
    buf = getattr(pixmap, "samples", None)
    if not buf:
        return True
    channels = 4 if getattr(pixmap, "alpha", False) else 3
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return True
    arr = arr.reshape(-1, channels)
    if channels == 4:
        arr = arr[:, :3]
    if arr.size == 0:
        return True
    if arr.shape[0] > 65536:
        samples = arr[::64]
    else:
        samples = arr
    samples16 = samples.astype(np.int16, copy=False)
    base = samples16[0]
    diff = np.abs(samples16 - base)
    if diff.max() <= tolerance:
        return True
    mask = np.any(diff > tolerance, axis=1)
    return float(mask.mean()) <= max_fraction

