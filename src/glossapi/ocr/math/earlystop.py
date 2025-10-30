from __future__ import annotations

"""
Lightweight early-stop injection for Docling CodeFormulaModel generation.

If the underlying recognizer uses HuggingFace `generate`, we attach
StoppingCriteria to stop on excessive tail repetition and cap decoded length.

Safe by default: if we cannot find a `generate` method or `transformers`
is unavailable, this becomes a no-op. Post-generation sanitization remains
the final guard in math enrichment.
"""

from typing import Any, Optional
import logging
import os

from glossapi.text_sanitize import load_latex_policy  # type: ignore

log = logging.getLogger(__name__)

__all__ = ["attach_early_stop"]


def _get_env_int(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v is not None and str(v).strip() != "" else default
    except Exception:
        return default


def attach_early_stop(code_formula_model: Any) -> bool:
    """Attempt to attach early-stop criteria to the underlying HF generate.

    Returns True if a wrapper was installed; False otherwise.
    Controlled via policy defaults with env fallbacks.
    """
    try:
        policy = load_latex_policy()
        enabled = bool(policy.earlystop_enabled)
        repeat_gate = int(policy.repeat_gate)
        max_chars = int(policy.max_chars)
        stride = int(policy.len_stride)
        max_new_tokens = int(policy.max_new_tokens)
    except Exception:
        enabled = os.getenv("GLOSSAPI_LATEX_EARLYSTOP", "1").strip().lower() not in {"0", "false", "no"}
        repeat_gate = _get_env_int("GLOSSAPI_LATEX_MAX_REPEAT", 50)
        max_chars = _get_env_int("GLOSSAPI_LATEX_MAX_CHARS", 3000)
        stride = _get_env_int("GLOSSAPI_LATEX_LEN_STRIDE", 16)
        max_new_tokens = _get_env_int("GLOSSAPI_LATEX_MAX_NEW_TOKENS", 0)
    if not enabled:
        return False

    target = None
    tokenizer = None
    try:
        for name in ("model", "_model", "hf_model", "lm", "recognizer"):
            if hasattr(code_formula_model, name):
                cand = getattr(code_formula_model, name)
                if hasattr(cand, "generate"):
                    target = cand
                    break
        if target is None and hasattr(code_formula_model, "generate"):
            target = code_formula_model
        for tname in ("tokenizer", "_tokenizer", "hf_tokenizer", "processor"):
            if hasattr(code_formula_model, tname):
                tokenizer = getattr(code_formula_model, tname)
                break
        if tokenizer is None and target is not None:
            for tname in ("tokenizer", "_tokenizer", "hf_tokenizer"):
                if hasattr(target, tname):
                    tokenizer = getattr(target, tname)
                    break
    except Exception:
        target = None

    if target is None or not hasattr(target, "generate"):
        return False

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore
    except Exception:
        return False

    class StopOnRepeat(StoppingCriteria):  # type: ignore
        def __init__(self, max_run: int = 50) -> None:
            super().__init__()
            self.max_run = int(max_run)
            self._last: Optional[int] = None
            self._run = 0

        def __call__(self, input_ids, scores, **kwargs):
            try:
                tid = int(input_ids[0, -1])
                if self._last is None or tid != self._last:
                    self._run = 1
                else:
                    self._run += 1
                self._last = tid
                return self._run > self.max_run
            except Exception:
                return False

    class StopOnLength(StoppingCriteria):  # type: ignore
        def __init__(self, max_chars: int = 3000, stride: int = 16, tokenizer: Optional[Any] = None) -> None:
            super().__init__()
            self.max_chars = int(max_chars)
            self.stride = max(1, int(stride))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs):
            try:
                step = int(input_ids.shape[1])
                if (step % self.stride) != 0:
                    return False
                if self.tokenizer is None:
                    return False
                text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                return len(text) >= self.max_chars
            except Exception:
                return False

    criteria = StoppingCriteriaList(
        [StopOnRepeat(repeat_gate), StopOnLength(max_chars=max_chars, stride=stride, tokenizer=tokenizer)]
    )

    try:
        original_generate = target.generate  # type: ignore[attr-defined]

        def wrapped_generate(*args, **kwargs):
            if criteria is not None and "stopping_criteria" not in kwargs:
                kwargs["stopping_criteria"] = criteria
            if max_new_tokens and "max_new_tokens" not in kwargs:
                kwargs["max_new_tokens"] = int(max_new_tokens)
            return original_generate(*args, **kwargs)

        setattr(target, "generate", wrapped_generate)
        log.info("Attached early-stop criteria to CodeFormulaModel generate()")
        return True
    except Exception:
        return False
