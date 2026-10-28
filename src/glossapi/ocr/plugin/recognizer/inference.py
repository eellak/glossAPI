"""Stage 4 — recognizer inference over extracted strips (greedy CTC).

REFERENCES
----------
- Model + recipe: docs/architecture/04; the arch is selected by checkpoint (train/ registry —
  the M-vs-R A/B in docs/iteration_2_plan.md decides the model of record).
- Greedy CTC decode: Graves et al. 2006 (best-path); implementation = train/cer.py
  (collapse repeats, drop blanks, ids -> NFD string).
- Confidence: mean over timesteps of max softmax prob — UNCALIBRATED (calibration is a pinned
  later stage, docs/architecture/06 §calibration); consumers must not treat it as a probability.
"""
from __future__ import annotations

import math
import importlib
import unicodedata

import numpy as np
import torch

from . import ckpt_io
from .codec import Codec, load_codec, greedy_decode_line
from .model import DET_PAD_MARGIN, build_model, masked_forward

class Recognizer:
    def __init__(self, ckpt_path: str, codec_path: str, device: str = "cpu") -> None:
        self.codec: Codec = load_codec(codec_path)
        # Load (state_dict, metadata) via ckpt_io: prefer explicit/sibling safetensors;
        # otherwise try the supplied resume .pt using torch's restricted weights-only loader.
        # The plugin checks any checkpoint pin before entering this loader.
        # Only NumPy's resume-state constructors are allowlisted; torch still uses
        # weights_only=True. No unrestricted pickle load or checkpoint conversion.
        try:
            multiarray = importlib.import_module("numpy._core.multiarray")
        except ImportError:  # NumPy 1.x, also supported by glossAPI
            multiarray = importlib.import_module("numpy.core.multiarray")
        with torch.serialization.safe_globals([
            multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype("uint32")),
        ]):
            sd, meta = ckpt_io.load_model_weights(ckpt_path)
        # WP-A cert F4: ENFORCE the codec binding (it was stored but never checked) — a same-size or
        # reordered codec (WP-C v2 is in flight) would otherwise decode SILENTLY WRONG text.
        if meta.get("codec_version") and meta["codec_version"] != self.codec.version:
            raise ValueError(f"checkpoint codec_version {meta['codec_version']!r} != codec "
                             f"{self.codec.version!r} — refusing to decode with a mismatched codec")
        _csha = meta.get("codec_sha256") or ""
        if _csha and _csha != ckpt_io.sha256_file(codec_path):
            raise ValueError(f"checkpoint codec_sha256 != {codec_path} — codec changed under a "
                             f"version-frozen checkpoint")
        n_classes = int(meta["n_classes"])  # metadata values are strings in safetensors
        if n_classes != self.codec.n_classes or meta.get("head", "nfd") != "nfd":
            raise ValueError("checkpoint must use this codec's NFD CTC head and class count")
        arch = meta.get("arch", "compact")
        try:  # arch-registry signature (rec/ A/B lane); returns (model, codec)
            built = build_model(n_classes=n_classes, arch=arch, verbose=False)
        except TypeError:  # pre-registry signature
            built = build_model(n_classes=n_classes, verbose=False)
        self.model = built[0] if isinstance(built, tuple) else built
        self.model.load_state_dict(sd)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def recognize(self, strips: list[np.ndarray], bs: int = 16) -> list[tuple[str, float]]:
        """strips: list of 48xW float arrays (1=white). Returns (text, conf) per strip, in order."""
        order = sorted(range(len(strips)), key=lambda i: strips[i].shape[1])
        results: list[tuple[str, float]] = [("", 0.0)] * len(strips)
        for start in range(0, len(order), bs):
            idxs = order[start : start + bs]
            ws = [strips[i].shape[1] for i in idxs]
            # WP-A F1 (deterministic own-bucket padding): pad EVERY strip — including the batch-widest —
            # with a right halo of its OWN background at least DET_PAD_MARGIN px wide (>= the recognizer
            # conv receptive field). Then each strip's boundary frames see its own background, never the
            # tensor's implicit conv zero-pad or another strip's width, so the reading is deterministic
            # regardless of batch composition / wmax. The mask (input_lengths) handles attention; this
            # halo handles the conv boundary — together they make serving batch-invariant.
            canvas_w = max(ws) + DET_PAD_MARGIN
            batch = torch.zeros(len(idxs), 1, 48, canvas_w)
            for j, i in enumerate(idxs):
                s = strips[i]
                bg = float(np.median(s[:, -2:]))
                batch[j, 0, :, : s.shape[1]] = torch.from_numpy(s)
                batch[j, 0, :, s.shape[1] :] = bg  # ALWAYS fill the remainder with this strip's own bg
            batch = ((batch - 0.5) / 0.5).to(self.device)
            in_lens = torch.tensor([math.ceil(w / 4) for w in ws])
            log_probs = masked_forward(self.model, batch, in_lens)  # [B, T, C] (mask-aware arches)
            probs = log_probs.exp()
            for j, i in enumerate(idxs):
                t_i = math.ceil(ws[j] / 4)
                lp = probs[j, :t_i]
                ids = lp.argmax(dim=-1).tolist()
                # greedy_decode_line yields the internal NFD form (codec atoms are NFD codepoints);
                # WP-A: user-facing serve output is NFC. The NFD/CER/audit path (eval_cer, dump_preds,
                # grader) is unaffected — it decodes ids directly, not via this serve method.
                text = unicodedata.normalize("NFC", greedy_decode_line(ids, self.codec))
                conf = float(lp.max(dim=-1).values.mean())
                results[i] = (text, conf)
        return results
