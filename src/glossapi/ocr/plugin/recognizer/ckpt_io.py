"""Safe checkpoint I/O (WP-A item ii, REV H appendix): weights-only safetensors bound to
codec+arch+head, and NO unsafe pickle (`torch.load(weights_only=False)`) on the inference path.

Training `.pt` files contain resume state as well as model weights. The preferred deployment
artifact is `.safetensors` (pure tensors plus string metadata). Without a sibling, `.pt` is tried
with `torch.load(weights_only=True)`. Recognizer scopes the NumPy resume constructors needed by
the supplied checkpoint through safe_globals; unrestricted pickle loading is never enabled.
This package has no checkpoint converter. If restricted loading fails, obtain a compatible
safetensors artifact from the trusted training/export environment and pin that artifact explicitly.

Metadata keys bound into every safetensors file: arch, head, codec_version, codec_sha256, n_classes
(+ any extras the caller passes, e.g. seed, best_metric). All values are stored as strings.
"""

from __future__ import annotations

import hashlib
import os

import torch

META_KEYS = ("arch", "head", "codec_version", "codec_sha256", "n_classes")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_meta(arch: str, head: str, codec_version: str, n_classes: int,
               codec_path: str | None = None, **extra) -> dict:
    """Assemble the codec+arch+head binding metadata (all values coerced to str at save time)."""
    meta = {
        "arch": arch,
        "head": head,
        "codec_version": str(codec_version),
        "codec_sha256": (sha256_file(codec_path) if codec_path and os.path.exists(codec_path) else ""),
        "n_classes": int(n_classes),
    }
    meta.update(extra)
    return meta


def save_safetensors(state_dict: dict, meta: dict, path: str) -> bool:
    """Write model weights + string metadata to `path` (.safetensors). Returns False (no-op) if the
    safetensors package is unavailable, so training never crashes on its absence — the `.pt` is still
    written by the caller. Install `safetensors` on training hosts to get the safe artifact."""
    try:
        from safetensors.torch import save_file
    except ImportError:
        return False
    sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    md = {k: str(v) for k, v in meta.items()}
    save_file(sd, path, metadata=md)
    return True


def load_safetensors(path: str):
    """Safe load of a .safetensors -> (state_dict, metadata dict). Never executes pickle."""
    from safetensors.torch import load_file, safe_open

    sd = load_file(path)
    with safe_open(path, framework="pt") as f:
        md = dict(f.metadata() or {})
    return sd, md


def load_model_weights(ckpt_path: str):
    """INFERENCE entry — returns (state_dict, meta). No unrestricted pickle loading.

    Resolution order: a `.safetensors` (given directly, or a sibling of a `.pt`) is loaded safely.
    A legacy `.pt` with no sibling is tried with `weights_only=True`; if it carries resume state
    (RNG/optimizer) outside the caller's scoped safe globals, it raises. Obtain a compatible
    safetensors artifact from the trusted training/export environment; no converter is bundled."""
    if ckpt_path.endswith(".safetensors"):
        return load_safetensors(ckpt_path)
    sibling = os.path.splitext(ckpt_path)[0] + ".safetensors"
    if os.path.exists(sibling):
        # WP-A cert F5: a sibling older than the .pt is a stale-weights hazard (resume-in-place / partial
        # rsync that moved only the .pt) — warn loud, since inference would silently load OLD weights.
        if os.path.exists(ckpt_path) and os.path.getmtime(sibling) < os.path.getmtime(ckpt_path):
            print(f"[warn] {sibling} is OLDER than {ckpt_path} — loading a possibly STALE safetensors "
                  f"sibling (export fresh weights in the trusted training environment). [WP-A cert F5]")
        return load_safetensors(sibling)
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:  # resume-state .pt (rng/opt) is not weights_only-safe
        raise RuntimeError(
            f"restricted weights-only load failed for checkpoint: {ckpt_path} ({e}). "
            "Obtain compatible .safetensors weights from the trusted training/export environment "
            "and pin that artifact explicitly; this package does not bundle a checkpoint converter."
        ) from e
    if not isinstance(ck, dict) or "model" not in ck:  # WP-A cert F8: clearer than a bare KeyError
        raise RuntimeError(
            f"{ckpt_path} is not a train.py checkpoint (no 'model' key) — expected a best.pt/last.pt "
            f"wrapper dict or a .safetensors; got a bare state-dict or unexpected object."
        )
    meta = {k: ck[k] for k in ("arch", "head", "codec_version", "n_classes", "seed") if k in ck}
    return ck["model"], meta
