"""Compact SVTRv2-style CTC line recognizer (PyTorch only) + the arch registry.

Contracts (frozen): docs/data_contracts.md §2 (strip: H=48, T = ceil(W/4)),
docs/architecture/04_recognizer.md (design rationale).

Architectures (docs/iteration_2_plan.md §"The recognizer setup A/B") — same interface,
selected via build_model(..., arch=...) / train.py --arch:
  compact      : this file's SVTRv2CTC (arm M, ours from-memory) — the default.
  svtrv2       : vendored OpenOCR SVTRv2 encoder + RCTC decoder, base dims (arm R-base);
                 rec/svtrv2_model.py over rec/openocr_vendored/ (imported lazily).
  svtrv2_small : same, dims scaled to match compact's capacity (arm R-small).

Input : grayscale [B, 1, 48, W], already normalized as (px/255 - 0.5) / 0.5.
Output: log-probs [B, T, n_classes] (log_softmax over classes), T = ceil(W/4) exactly.
Classes: id 0 = CTC blank, ids 1..len(atoms) = codec v1 atoms (n_classes = len(atoms)+1).

Topology (pinned):
  stem   : Conv(1->48, 3x3, s2)+BN+GELU -> Conv(48->64, 3x3, s2)+BN+GELU   -> [B,  64, 12, ceil(W/4)]
  stage1 : 3 x LocalBlock(64)
  down1  : Conv(64->128, 3x3, stride (2,1))+BN+GELU                        -> [B, 128,  6, ceil(W/4)]
  stage2 : 6 x LocalBlock(128)
  down2  : Conv(128->256, 3x3, stride (2,1))+BN+GELU                       -> [B, 256,  3, ceil(W/4)]
  stage3 : + sinusoidal 2D positional encoding (added once), then
           3 x GlobalBlock(256): pre-LN transformer block, MHSA 8 heads over the
           flattened 3 x ceil(W/4) token grid, MLP ratio 4
  head   : mean-pool over H -> [B, T, 256] -> LayerNorm -> Linear(256 -> n_classes)
           -> log_softmax

LocalBlock is ConvNeXt-style: DWConv 7x7 + LayerNorm (channels-last) + pointwise
MLP (4x, GELU) + residual.
"""

from __future__ import annotations

import contextlib
import inspect
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import load_codec

STRIP_HEIGHT = 48  # data_contracts §2
STRIDE = 4  # T = ceil(W / STRIDE) — the frozen stride contract
DET_PAD_MARGIN = 128  # WP-A F1: right own-background halo (px) added to EVERY strip's canvas so its
# boundary conv frames see its own background, not the tensor edge / other strips. Must exceed the
# recognizer conv receptive field (measured ~110px: DWConv7x7 x9 LocalBlocks ~+-27 T-frames ~+-110px);
# 128 is the next 128-multiple (contract §2 bucket step). Makes serving/eval batch-deterministic.


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class ConvBNAct(nn.Module):
    """3x3 Conv + BatchNorm + GELU (stem and downsample unit)."""

    def __init__(self, c_in: int, c_out: int, stride) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LocalBlock(nn.Module):
    """ConvNeXt-style block: DWConv7x7 + LN (channels-last) + MLP(4x, GELU) + residual."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c)
        self.fc2 = nn.Linear(4 * c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # -> channels-last [B,H,W,C]
        x = self.norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        x = x.permute(0, 3, 1, 2)  # -> [B,C,H,W]
        return residual + x


class GlobalBlock(nn.Module):
    """Pre-LN transformer block: MHSA (8 heads) + MLP (ratio 4) over [B, N, C] tokens."""

    def __init__(self, c: int, n_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, mlp_ratio * c)
        self.fc2 = nn.Linear(mlp_ratio * c, c)

    def forward(self, x: torch.Tensor, key_padding_mask=None, attn_mask=None) -> torch.Tensor:
        y = self.norm1(x)
        # key_padding_mask [B, N] True == right-pad frame -> excluded from attention (WP-A): without
        # it a strip's valid frames attend to batch-companion padding, so the same strip reads
        # differently depending on which strips share its batch (train + serve nondeterminism).
        # attn_mask [N, N] bool (True == drop) = the local WINDOW for arm A32 (banded along T;
        # None -> full attention, the compact/certified path is byte-identical).
        out = self.attn(y, y, y, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                        need_weights=False)[0]
        if attn_mask is not None:
            # Windowed arches only: a PAD-query row (t_idx >= valid_T) can have its whole window fall on
            # pad keys (short strip batched with a long one) -> fully-masked softmax row -> NaN. Those
            # frames are discarded downstream (CTC reads only input_lengths; decode slices to :valid_T),
            # so zeroing them is harmless and stops NaN from poisoning the residual/grad. Valid queries
            # always keep >=1 in-window valid key (their own frame, |dt|=0), so they never NaN.
            out = torch.nan_to_num(out, nan=0.0)
        x = x + out
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


def sincos_pos_embed_2d(h: int, w: int, dim: int) -> torch.Tensor:
    """Sinusoidal 2D positional encoding, [h*w, dim] (row-major: index = y*w + x).

    Half the channels encode the H coordinate, half the W coordinate; each half is
    the standard sin/cos frequency ladder (base 10000).
    """
    assert dim % 4 == 0, "PE dim must be divisible by 4"
    d_axis = dim // 2

    def axis(n: int) -> torch.Tensor:
        pos = torch.arange(n, dtype=torch.float32)
        k = torch.arange(d_axis // 2, dtype=torch.float32)
        omega = torch.exp(-math.log(10000.0) * k / (d_axis // 2))
        ang = pos[:, None] * omega[None, :]  # [n, d_axis/2]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [n, d_axis]

    emb_h = axis(h)  # [h, d_axis]
    emb_w = axis(w)  # [w, d_axis]
    pe = torch.zeros(h, w, dim, dtype=torch.float32)
    pe[..., :d_axis] = emb_h[:, None, :]
    pe[..., d_axis:] = emb_w[None, :, :]
    return pe.reshape(h * w, dim)


ENC_DIM = 256  # compact encoder feature width (the P1 head arms in heads.py build on this)


class SVTRv2CTC(nn.Module):
    def __init__(self, n_classes: int, with_head: bool = True, attn_window: Optional[int] = None,
                 height_collapse: bool = False) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2 (blank + atoms), got {n_classes}")
        if attn_window is not None and attn_window < 1:
            raise ValueError(f"attn_window must be >= 1 or None, got {attn_window}")
        self.n_classes = n_classes
        # Arm A32 (S19-BKO): local windowed attention along T. attn_window=w -> each query attends
        # only to keys within radius w//2 along the T axis (H stays fully connected; H=3). ZERO extra
        # params (same projections; the window is a banded attn_mask). None = full attention (compact).
        self.attn_window = attn_window
        self._winmask_cache: dict = {}  # (h, t) -> cpu fp32 [h*t, h*t] additive band mask

        self.stem = nn.Sequential(ConvBNAct(1, 48, stride=2), ConvBNAct(48, 64, stride=2))
        self.stage1 = nn.Sequential(*[LocalBlock(64) for _ in range(3)])
        self.down1 = ConvBNAct(64, 128, stride=(2, 1))
        self.stage2 = nn.Sequential(*[LocalBlock(128) for _ in range(6)])
        self.down2 = ConvBNAct(128, 256, stride=(2, 1))
        # Arm B+A64 (S19-BKO): LEARNED height collapse H=3 -> 1 (Conv2d kernel (3,1), no T-mixing) BEFORE
        # stage3, so global attention runs over T tokens (not 3*T) — 3x fewer tokens. Combined with
        # attn_window=64 this is the Diaz-2021 "collapse then windowed 1-D attention" topology. Adds
        # params (~197k); compact/A32 keep height_collapse=False and mean-pool H after stage3.
        self.hcollapse = nn.Conv2d(256, 256, kernel_size=(3, 1)) if height_collapse else None
        self.stage3 = nn.ModuleList([GlobalBlock(256, n_heads=8, mlp_ratio=4) for _ in range(3)])
        self.norm = nn.LayerNorm(256)
        # arm B (nfd) owns the head here; the P1 A/C arms build headless (with_head=False)
        # and attach their own head to encode() (heads.py). See docs/architecture/04 §4.4.
        self.head = nn.Linear(256, n_classes) if with_head else None

        self._pe_cache: dict = {}  # (h, w) -> cpu fp32 [1, h*w, C]

    def _pos_embed(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w)
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = sincos_pos_embed_2d(h, w, 256).unsqueeze(0)  # [1, h*w, 256] cpu fp32
            self._pe_cache[key] = pe
        return pe.to(device=device, dtype=dtype)

    def _frame_key_padding_mask(self, input_lengths, h: int, t: int, device) -> torch.Tensor:
        """[B, h*t] bool mask, True where a token is a right-pad frame (t_idx >= valid_T).

        Tokens are row-major (h*T + t); a frame is pad iff its column index t_idx >= the sample's
        valid_T, across all h rows. valid_T is clamped to [1, t] so no attention row is fully masked
        (a fully-masked row yields NaN in softmax)."""
        il = input_lengths.to(device=device).clamp(min=1, max=t)  # [B]
        t_idx = torch.arange(t, device=device)  # [T]
        frame_pad = t_idx[None, :] >= il[:, None]  # [B, T] True where t_idx >= valid_T
        return frame_pad[:, None, :].expand(il.shape[0], h, t).reshape(il.shape[0], h * t)

    def _window_attn_mask(self, h: int, t: int, device) -> torch.Tensor:
        """[h*t, h*t] BOOL mask (True == DISALLOW) = local T-window (arm A32).

        Token n is row-major (h_idx*t + t_idx), so t_idx = n % t. A pair (i, j) is kept iff
        |t_idx(i) - t_idx(j)| <= attn_window//2, for ALL h rows (H fully connected). Shared across
        the batch + heads (a 2D attn_mask), so it costs O((h*t)^2) once, not per-sample. Bool (not a
        float additive mask) to MATCH the bool key_padding_mask — nn.MultiheadAttention deprecates
        mixing a bool key_padding_mask with a float attn_mask."""
        key = (h, t)
        m = self._winmask_cache.get(key)
        if m is None:
            r = self.attn_window // 2
            ti = torch.arange(t)
            allow = (ti[:, None] - ti[None, :]).abs() <= r  # [t, t] bool (T-band)
            m = ~allow.repeat(h, h)  # [h*t, h*t] bool: True == DISALLOW (element (a*t+p,b*t+q)=~allow[p,q])
            self._winmask_cache[key] = m
        return m.to(device=device)

    def encode(self, x: torch.Tensor, input_lengths=None) -> torch.Tensor:
        """Encoder trunk -> normed per-frame features [B, T, 256], T = ceil(W/4).

        Shared by all P1 head arms (heads.py). The arm-specific head consumes this;
        arm B's head lives on this module (self.head).

        input_lengths [B] (valid T per sample = ceil_div(valid_width, STRIDE)) builds a
        key_padding_mask so a strip's valid frames never attend to batch-companion right-pad (WP-A).
        None -> no masking (backward-compatible; exact only for single/full-width strips)."""
        assert x.dim() == 4 and x.shape[1] == 1, f"expected [B,1,48,W], got {tuple(x.shape)}"
        assert x.shape[2] == STRIP_HEIGHT, f"strip height must be {STRIP_HEIGHT}, got {x.shape[2]}"
        w_in = x.shape[3]
        t_expected = ceil_div(w_in, STRIDE)  # frozen contract: T = ceil(W/4)

        x = self.stem(x)  # [B, 64, 12, T]
        x = self.stage1(x)
        x = self.down1(x)  # [B, 128, 6, T]
        x = self.stage2(x)
        x = self.down2(x)  # [B, 256, 3, T]

        b, c, h, t = x.shape
        assert (
            h == 3 and t == t_expected
        ), f"stride contract violated: got H={h}, T={t}, expected H=3, T=ceil({w_in}/4)={t_expected}"

        if self.hcollapse is not None:  # Arm B+A64: learned H=3 -> 1 collapse (no T-mixing) before stage3
            x = self.hcollapse(x)  # [B, 256, 1, T]
            b, c, h, t = x.shape  # h is now 1; the h*T machinery below runs identically with h=1

        x = x.flatten(2).transpose(1, 2)  # [B, h*T, 256], row-major (h*T + t)
        x = x + self._pos_embed(h, t, x.device, x.dtype)  # added once, before the global blocks
        kpm = self._frame_key_padding_mask(input_lengths, h, t, x.device) if input_lengths is not None else None
        wmask = self._window_attn_mask(h, t, x.device) if self.attn_window is not None else None
        for blk in self.stage3:
            x = blk(x, kpm, wmask)

        x = x.reshape(b, h, t, c).mean(dim=1)  # mean-pool over H -> [B, T, 256]
        return self.norm(x)  # [B, T, 256] normed features (pre-head)

    def forward(self, x: torch.Tensor, input_lengths=None) -> torch.Tensor:
        feats = self.encode(x, input_lengths)  # [B, T, 256]
        if self.head is None:
            raise RuntimeError("SVTRv2CTC built with with_head=False; use encode() + an external head")
        return F.log_softmax(self.head(feats), dim=-1)  # [B, T, n_classes]


def masked_forward(model, x, input_lengths=None):
    """Call the model, passing input_lengths ONLY to forwards that accept it (WP-A).

    The compact SVTRv2CTC + the P1 head wrappers (heads.py) take input_lengths and build the
    key_padding_mask; the vendored svtrv2 forward (rec/, SVTRv2RCTC) takes x only. Call sites use
    this so a masked run never TypeErrors on a non-mask-aware arch. Capability is detected once via
    the forward signature and cached on the model's class."""
    cls = type(model)
    ok = getattr(cls, "_accepts_input_lengths", None)
    if ok is None:
        ok = "input_lengths" in inspect.signature(model.forward).parameters
        cls._accepts_input_lengths = ok
    if ok and input_lengths is not None:
        return model(x, input_lengths=input_lengths)
    return model(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


ARCHS = {"compact": SVTRv2CTC}
ARCH_CHOICES = ("compact",)


def build_model(
    codec_path: Optional[str] = None,
    n_classes: Optional[int] = None,
    arch: str = "compact",
    verbose: bool = True,
):
    """Build the recognizer. Exactly one of codec_path / n_classes decides the head size.

    arch selects the architecture from ARCHS (default compact — backward-compatible).
    Returns (model, codec) — codec is None when built from a bare n_classes (selftest).
    """
    codec = None
    if codec_path is not None:
        codec = load_codec(codec_path)  # validates: ids contiguous, max id == len(atoms)
        if n_classes is not None and n_classes != codec.n_classes:
            raise ValueError(f"n_classes {n_classes} != codec n_classes {codec.n_classes}")
        n_classes = codec.n_classes
    if n_classes is None:
        raise ValueError("need --codec (or explicit n_classes) to size the head")
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choices: {', '.join(ARCH_CHOICES)}")
    model = ARCHS[arch](n_classes)
    if verbose:
        n = count_params(model)
        print(f"[model] arch={arch}: n_classes={n_classes}, params={n:,} ({n / 1e6:.2f}M)")
    return model, codec


def amp_dtype_for(device: torch.device):
    """bf16 autocast on cuda when supported; fp32 (no autocast) on cpu/mps."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def autocast_ctx(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
