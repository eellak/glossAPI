"""Regression tests for math early-stop state isolation (fix/math-earlystop-state-reset)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal Transformers stub so tests run without the real package installed
# ---------------------------------------------------------------------------

def _install_transformers_stub():
    if "transformers" in sys.modules:
        return
    mod = types.ModuleType("transformers")

    class StoppingCriteria:
        def __call__(self, input_ids, scores, **kwargs):
            return False

    class StoppingCriteriaList(list):
        def __call__(self, input_ids, scores, **kwargs):
            return any(c(input_ids, scores, **kwargs) for c in self)

    mod.StoppingCriteria = StoppingCriteria
    mod.StoppingCriteriaList = StoppingCriteriaList
    sys.modules["transformers"] = mod


_install_transformers_stub()

import numpy as np  # noqa: E402
import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ids(*token_ids):
    """Return a 2-D (1, n) int64 array matching the [batch, seq] shape HF expects."""
    return np.array([list(token_ids)], dtype=np.int64)


def _attach(model, *, repeat_gate=2, max_new_tokens=0, monkeypatch,
            max_chars=3000, stride=16):
    """Wrap model.generate() via attach_early_stop using env-var config."""
    monkeypatch.setenv("GLOSSAPI_LATEX_EARLYSTOP", "1")
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_REPEAT", str(repeat_gate))
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_CHARS", str(max_chars))
    monkeypatch.setenv("GLOSSAPI_LATEX_LEN_STRIDE", str(stride))
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_NEW_TOKENS", str(max_new_tokens))

    import glossapi.ocr.math.earlystop as es_mod
    orig = es_mod.load_latex_policy
    es_mod.load_latex_policy = lambda: (_ for _ in ()).throw(RuntimeError("no policy"))
    try:
        from glossapi.ocr.math.earlystop import attach_early_stop
        result = attach_early_stop(model)
    finally:
        es_mod.load_latex_policy = orig
    return result


def _make_recording_model():
    """Return a (model, calls) pair where calls accumulates kwargs per generate()."""
    calls = []

    class FakeModel:
        def generate(self, input_ids, stopping_criteria=None, **kwargs):
            calls.append({"input_ids": input_ids, "stopping_criteria": stopping_criteria, "kwargs": kwargs})
            return input_ids

    return FakeModel(), calls


# ---------------------------------------------------------------------------
# Test A: repetition detection within a single generation
# ---------------------------------------------------------------------------

def test_stop_on_repeat_triggers_within_one_call(monkeypatch):
    """StopOnRepeat fires when a token repeats beyond max_run within one call."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)
    model.generate(_make_ids(1))
    sc = calls[0]["stopping_criteria"][0]  # StopOnRepeat

    assert not sc(_make_ids(7), None)       # _run=1, 1>2 → False
    assert not sc(_make_ids(7, 7), None)    # _run=2, 2>2 → False
    assert sc(_make_ids(7, 7, 7), None)     # _run=3, 3>2 → True


# ---------------------------------------------------------------------------
# Test B: state resets between independent generations
# ---------------------------------------------------------------------------

def test_state_resets_between_generations(monkeypatch):
    """Each generate() call gets a fresh StopOnRepeat; prior run counter does not bleed."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)

    model.generate(_make_ids(1))
    sc1 = calls[0]["stopping_criteria"][0]
    # Exhaust sc1's counter
    sc1(_make_ids(7), None)       # _run=1
    sc1(_make_ids(7, 7), None)    # _run=2
    assert sc1(_make_ids(7, 7, 7), None)  # _run=3 > 2 → True

    model.generate(_make_ids(1))
    sc2 = calls[1]["stopping_criteria"][0]

    assert sc1 is not sc2, "each call must receive a distinct StopOnRepeat instance"
    # sc2 starts fresh — first occurrence of token 7 must not stop
    assert not sc2(_make_ids(7), None)


# ---------------------------------------------------------------------------
# Test C: independent criteria instances per call
# ---------------------------------------------------------------------------

def test_separate_calls_get_separate_criteria_instances(monkeypatch):
    """Stopping-criteria objects differ across generate() calls."""
    model, calls = _make_recording_model()
    _attach(model, monkeypatch=monkeypatch)

    model.generate(_make_ids(1))
    model.generate(_make_ids(2))

    c1 = calls[0]["stopping_criteria"]
    c2 = calls[1]["stopping_criteria"]
    assert c1 is not c2
    assert c1[0] is not c2[0]  # StopOnRepeat
    assert c1[1] is not c2[1]  # StopOnLength


# ---------------------------------------------------------------------------
# Test D: caller-provided stopping_criteria is preserved
# ---------------------------------------------------------------------------

def test_caller_provided_criteria_not_replaced(monkeypatch):
    """When caller supplies stopping_criteria, wrapper must not overwrite it."""
    model, calls = _make_recording_model()
    _attach(model, monkeypatch=monkeypatch)

    custom = object()
    model.generate(_make_ids(1), stopping_criteria=custom)

    assert calls[0]["stopping_criteria"] is custom


# ---------------------------------------------------------------------------
# Test E: StopOnLength still functional after the fix
# ---------------------------------------------------------------------------

def test_stop_on_length_still_works(monkeypatch):
    """StopOnLength fires when decoded text exceeds max_chars."""
    model, calls = _make_recording_model()
    _attach(model, monkeypatch=monkeypatch)
    model.generate(_make_ids(1))
    sc_len = calls[0]["stopping_criteria"][1]  # StopOnLength

    fake_tok = MagicMock()
    fake_tok.decode.return_value = "x" * 3001
    sc_len.tokenizer = fake_tok
    sc_len.stride = 1

    assert sc_len(_make_ids(1), None)


# ---------------------------------------------------------------------------
# Test F: max_new_tokens injection remains unchanged
# ---------------------------------------------------------------------------

def test_max_new_tokens_injected(monkeypatch):
    """max_new_tokens is injected when configured and not supplied by caller."""
    model, calls = _make_recording_model()
    _attach(model, max_new_tokens=128, monkeypatch=monkeypatch)

    model.generate(_make_ids(1))
    assert calls[0]["kwargs"].get("max_new_tokens") == 128

    # Caller-supplied value must not be overwritten
    model.generate(_make_ids(1), max_new_tokens=64)
    assert calls[1]["kwargs"].get("max_new_tokens") == 64


# ---------------------------------------------------------------------------
# Test G: exact repetition threshold boundary
# ---------------------------------------------------------------------------

def test_stop_on_repeat_exact_boundary(monkeypatch):
    """Criterion fires at _run == max_run+1, not before."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=3, monkeypatch=monkeypatch)
    model.generate(_make_ids(1))
    sc = calls[0]["stopping_criteria"][0]

    assert not sc(_make_ids(5), None)       # _run=1, 1>3 → False
    assert not sc(_make_ids(5, 5), None)    # _run=2
    assert not sc(_make_ids(5, 5, 5), None) # _run=3, 3>3 → False (not strictly greater)
    assert sc(_make_ids(5, 5, 5, 5), None)  # _run=4, 4>3 → True


# ---------------------------------------------------------------------------
# Test H: token change resets the run counter
# ---------------------------------------------------------------------------

def test_token_change_resets_run_counter(monkeypatch):
    """When the last token changes, StopOnRepeat resets _run to 1."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)
    model.generate(_make_ids(1))
    sc = calls[0]["stopping_criteria"][0]

    # Build up a run on token 7
    sc(_make_ids(7), None)       # _run=1
    sc(_make_ids(7, 7), None)    # _run=2 (would stop at _run=3)

    # Switch to token 8 — counter must reset
    assert not sc(_make_ids(8), None)   # _run=1, different token
    # Back to token 7 also resets
    assert not sc(_make_ids(7), None)   # _run=1, again reset


# ---------------------------------------------------------------------------
# Test I: StopOnLength respects stride (not fired between stride steps)
# ---------------------------------------------------------------------------

def test_stop_on_length_respects_stride(monkeypatch):
    """StopOnLength skips the check at steps not divisible by stride."""
    model, calls = _make_recording_model()
    _attach(model, stride=4, max_chars=5, monkeypatch=monkeypatch)
    model.generate(_make_ids(1))
    sc_len = calls[0]["stopping_criteria"][1]

    fake_tok = MagicMock()
    fake_tok.decode.return_value = "x" * 100  # > max_chars
    sc_len.tokenizer = fake_tok

    # Step 3 is not divisible by 4 → no check → no stop despite long text
    assert not sc_len(_make_ids(*range(3)), None)
    # Step 4 is divisible by 4 → check → stop
    assert sc_len(_make_ids(*range(4)), None)


# ---------------------------------------------------------------------------
# Test J: StopOnLength instances per call are independent (no shared tokenizer state)
# ---------------------------------------------------------------------------

def test_stop_on_length_instances_independent(monkeypatch):
    """StopOnLength from one call does not share tokenizer mutation with another."""
    model, calls = _make_recording_model()
    _attach(model, monkeypatch=monkeypatch)

    model.generate(_make_ids(1))
    model.generate(_make_ids(2))

    sl1 = calls[0]["stopping_criteria"][1]
    sl2 = calls[1]["stopping_criteria"][1]
    assert sl1 is not sl2

    # Mutating tokenizer on sl1 does not affect sl2
    sl1.tokenizer = MagicMock()
    assert sl2.tokenizer is not sl1.tokenizer


# ---------------------------------------------------------------------------
# Test K: disabled configuration — attach_early_stop returns False
# ---------------------------------------------------------------------------

def test_disabled_config_returns_false_and_no_wrap(monkeypatch):
    """With GLOSSAPI_LATEX_EARLYSTOP=0, attach_early_stop is a no-op."""
    monkeypatch.setenv("GLOSSAPI_LATEX_EARLYSTOP", "0")
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_REPEAT", "2")
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_CHARS", "3000")
    monkeypatch.setenv("GLOSSAPI_LATEX_LEN_STRIDE", "16")
    monkeypatch.setenv("GLOSSAPI_LATEX_MAX_NEW_TOKENS", "0")

    model, calls = _make_recording_model()
    original_generate = model.generate

    import glossapi.ocr.math.earlystop as es_mod
    orig = es_mod.load_latex_policy
    es_mod.load_latex_policy = lambda: (_ for _ in ()).throw(RuntimeError("no policy"))
    try:
        from glossapi.ocr.math.earlystop import attach_early_stop
        result = attach_early_stop(model)
    finally:
        es_mod.load_latex_policy = orig

    assert result is False
    # Bound methods differ on each access; compare the underlying function
    assert model.generate.__func__ is original_generate.__func__, (
        "generate must not be wrapped when disabled"
    )


# ---------------------------------------------------------------------------
# Test L: double attachment — criteria injected exactly once per call
# ---------------------------------------------------------------------------

def test_double_attachment_injects_criteria_once_per_call(monkeypatch):
    """Calling attach_early_stop twice wraps twice, but criteria still injected once."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)  # second wrap

    model.generate(_make_ids(1))

    # The outer wrapper injects criteria; the inner sees stopping_criteria already set
    # and does not replace it. The model's generate receives exactly one criteria object.
    sc = calls[0]["stopping_criteria"]
    assert sc is not None
    # Must be a list-like with exactly the two criteria (not doubled)
    assert len(sc) == 2


# ---------------------------------------------------------------------------
# Test M: batch pattern — many calls on same model, all get fresh criteria
# ---------------------------------------------------------------------------

def test_batch_flush_pattern_fresh_criteria_each_call(monkeypatch):
    """Simulate flush_batch calling generate() per batch item; each gets fresh state."""
    model, calls = _make_recording_model()
    _attach(model, repeat_gate=2, monkeypatch=monkeypatch)

    # Drive 5 independent generate() calls, as flush_batch would
    for _ in range(5):
        model.generate(_make_ids(1))

    criteria_ids = [id(c["stopping_criteria"]) for c in calls]
    # Every call must have received a distinct criteria list
    assert len(set(criteria_ids)) == 5, "each of 5 calls must receive a unique criteria list"

    # Exhaust counter on call 0's StopOnRepeat; call 1's must be unaffected
    sc0 = calls[0]["stopping_criteria"][0]
    sc1_obj = calls[1]["stopping_criteria"][0]

    sc0(_make_ids(7), None)
    sc0(_make_ids(7, 7), None)
    assert sc0(_make_ids(7, 7, 7), None)   # sc0 tripped

    assert not sc1_obj(_make_ids(7), None)  # sc1 fresh: first token, not stopped
