from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os


@dataclass
class LatexPolicy:
    # Generation-time early-stop policy
    earlystop_enabled: bool = True
    repeat_gate: int = 50           # stop when tail token repeats past this run
    max_chars: int = 3000           # decoded-length stop gate
    len_stride: int = 16            # decode-length check stride (lower = more checks)
    max_new_tokens: int = 0         # 0 disables token cap injection

    # Post-processing policy
    post_only_failed: bool = True   # apply only if clearly pathological
    post_repeat_gate: int = 50      # treat as failed if tail_run exceeds this
    post_winddown: int = 12         # clamp repeated tail token to this count
    post_max_chars: int = 3000      # hard cap on text length (prefer boundary)


def _get_env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    try:
        v = os.getenv(name)
        if v is None:
            return default
        return str(v).strip().lower() not in {"0", "false", "no"}
    except Exception:
        return default


def load_latex_policy() -> LatexPolicy:
    return LatexPolicy(
        earlystop_enabled=_get_env_bool("GLOSSAPI_LATEX_EARLYSTOP", True),
        repeat_gate=_get_env_int("GLOSSAPI_LATEX_MAX_REPEAT", 50),
        max_chars=_get_env_int("GLOSSAPI_LATEX_MAX_CHARS", 3000),
        len_stride=_get_env_int("GLOSSAPI_LATEX_LEN_STRIDE", 16),
        max_new_tokens=_get_env_int("GLOSSAPI_LATEX_MAX_NEW_TOKENS", 0),
        post_only_failed=_get_env_bool("GLOSSAPI_LATEX_POST_ONLY_FAILED", True),
        post_repeat_gate=_get_env_int("GLOSSAPI_LATEX_POST_REPEAT_GATE", 50),
        post_winddown=_get_env_int("GLOSSAPI_LATEX_POST_WINDDOWN", 12),
        post_max_chars=_get_env_int("GLOSSAPI_LATEX_POST_MAX_CHARS", 3000),
    )


def tail_run(s: str) -> int:
    toks = (s or "").split()
    if not toks:
        return 0
    last = toks[-1]
    run = 1
    i = len(toks) - 2
    while i >= 0 and toks[i] == last:
        run += 1
        i -= 1
    return run


def sanitize_latex(text: str, policy: LatexPolicy | None = None) -> Tuple[str, Dict[str, Any]]:
    """Apply tail-repeat clamp and length cap; return sanitized text and info flags.

    - If policy.post_only_failed is True, we only apply changes when tail_run exceeds
      policy.post_repeat_gate or when text length exceeds policy.post_max_chars.
    """
    p = policy or load_latex_policy()
    s = text or ""
    info = {
        "orig_len": len(s),
        "truncated_by_repeat": False,
        "truncated_by_len": False,
        "tail_token": "",
        "tail_run": 0,
        "post_applied": False,
    }

    # Decide whether to post-process
    must = False
    try:
        r = tail_run(s)
    except Exception:
        r = 0
    info["tail_run"] = r
    toks = s.split()
    info["tail_token"] = toks[-1] if toks else ""
    if r > int(p.post_repeat_gate) or len(s) > int(p.post_max_chars):
        must = True
    if p.post_only_failed and not must:
        return s, info
    info["post_applied"] = True

    # Repeat clamp (winddown)
    if r > int(p.post_winddown) and toks:
        keep = int(p.post_winddown)
        # drop tail to exactly `keep` repeats
        i = len(toks) - 1
        last = toks[-1]
        while i >= 1 and toks[i - 1] == last:
            i -= 1
        toks = toks[: i + keep]
        s = " ".join(toks)
        info["truncated_by_repeat"] = True

    # Length cap (prefer whitespace or backslash boundary)
    if len(s) > int(p.post_max_chars):
        cut_ws = max(s.rfind(" ", 0, int(p.post_max_chars)), s.rfind("\n", 0, int(p.post_max_chars)), s.rfind("\t", 0, int(p.post_max_chars)))
        # also attempt LaTeX boundary at backslash
        cut_bs = s.rfind("\\", 0, int(p.post_max_chars))
        cut = max(cut_ws, cut_bs)
        cut = cut if cut != -1 else int(p.post_max_chars)
        s = s[:cut].rstrip()
        info["truncated_by_len"] = True

    return s, info


def accept_latex(text: str) -> float:
    """Lightweight sanity score: 1.0 accepted, 0.0 rejected.

    - Balanced braces
    - No obviously dangerous tokens
    - Modest max length (gate); prefer to handle via policy but keep here as guard
    """
    try:
        s = text or ""
        if len(s) > 6000:  # hard fail beyond extreme size
            return 0.0
        bal = 0
        for ch in s:
            if ch == '{':
                bal += 1
            elif ch == '}':
                bal -= 1
            if bal < 0:
                return 0.0
        if bal != 0:
            return 0.0
        bad_toks = ["\\includegraphics", "\\write18"]
        if any(tok in s for tok in bad_toks):
            return 0.0
        return 1.0
    except Exception:
        return 0.0


__all__ = [
    "LatexPolicy",
    "load_latex_policy",
    "sanitize_latex",
    "tail_run",
    "accept_latex",
]

