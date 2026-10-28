"""Stage 3 — line box -> recognizer strip, per the frozen contract.

REFERENCES
----------
- Strip contract: docs/data_contracts.md §2 (FROZEN): grayscale, height 48 px exactly, width
  clamped to [64, 2560], dark-on-light nominal, T = ceil(W/4).
- LANCZOS resampling for downscale of text: standard practice (e.g. the sibling program's eval
  harness used height-32 LANCZOS); ours: the polarity heuristic below (border-vs-otsu ink
  fraction), documented as v0.
- v1 upgrade (pinned): interface-match crop training with detector-jitter, docs/architecture/02.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

STRIP_H = 48
MIN_W, MAX_W = 64, 2560


def extract_strip(page: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Crop + normalize one line strip. Returns float array 48xW in [0,1] (1=white), or None.

    Vertical pad is PROPORTIONAL (10% of band height per side) so the ink band lands at
    ~40/48 px of the strip — the same crop-normalized geometry the factory renders training
    strips with (recipe ink_target_h; docs/review_feedback.md §3). Keep the two in sync.
    """
    x0, y0, x1, y1 = box
    pad_y = max(1, round(0.1 * (y1 - y0)))
    y0, y1 = y0 - pad_y, y1 + pad_y
    crop = page[max(0, y0) : min(page.shape[0], y1), max(0, x0) : x1]
    if crop.size == 0 or crop.shape[0] < 4:
        return None
    # polarity: strips must be dark-on-light; if ink fraction says inverted, flip.
    ink_frac = float((crop < 0.5).mean())
    if ink_frac > 0.5:
        crop = 1.0 - crop
    h, w = crop.shape
    new_w = int(round(w * STRIP_H / h))
    if new_w < 8:
        return None
    if new_w > MAX_W:
        # LETTERBOX (shared rule with the factory renderer): shrink aspect-preserving to fit
        # MAX_W, pad vertically — never squeeze, never clip.
        scale = MAX_W / new_w
        new_h = max(8, int(round(STRIP_H * scale)))
        img = Image.fromarray((crop * 255).astype(np.uint8), mode="L").resize(
            (MAX_W, new_h), Image.LANCZOS
        )
        small = np.asarray(img).astype(np.float32) / 255.0
        bg = float(np.median(small[:2, :]))
        arr = np.full((STRIP_H, MAX_W), bg, dtype=np.float32)
        top = (STRIP_H - new_h) // 2
        arr[top : top + new_h, :] = small
        return arr
    img = Image.fromarray((crop * 255).astype(np.uint8), mode="L").resize(
        (new_w, STRIP_H), Image.LANCZOS
    )
    arr = np.asarray(img).astype(np.float32) / 255.0
    if new_w < MIN_W:  # pad right with background estimate to the contract minimum
        bg = float(np.median(arr[:, -2:]))
        pad = np.full((STRIP_H, MIN_W - new_w), bg, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr
