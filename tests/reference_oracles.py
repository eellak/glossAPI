"""Pinned reference packages for the parity/oracle tests (pyproject `reference` dependency group).

The oracles are the libraries K2 replaced or kept only as references: tenacity, aiofiles, dask and
scikit-learn (replaced by own code; scikit-learn 1.6.1 scores the shipped joblib model as the oracle of
the neutral section classifier) and pypdf/fontTools (the Python font-repair reference). Runtime installs no
longer pull them in. Where the group is expected to be installed (the default `test`/`dev` groups,
`setup_glossapi.sh --run-tests`, checkpoint runs) set ``GLOSSAPI_REFERENCE_REQUIRED=1``: a missing
package or a different version then fails the test instead of silently skipping its oracle.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os

import pytest

ENV = "GLOSSAPI_REFERENCE_REQUIRED"
# import name -> (distribution, pinned version); must equal the pyproject `reference` group.
PINS = {
    "tenacity": ("tenacity", "9.2.1"),
    "aiofiles": ("aiofiles", "25.1.0"),
    "dask": ("dask", "2026.8.0"),
    "pypdf": ("pypdf", "6.21.3"),
    "fontTools": ("fonttools", "4.68.0"),
    "sklearn": ("scikit-learn", "1.6.1"),
}


def required() -> bool:
    return os.environ.get(ENV, "").strip().lower() in {"1", "true", "yes"}


def problem(module: str) -> str | None:
    """Why the pinned oracle is unusable here, or None."""
    distribution, version = PINS[module]
    try:
        importlib.import_module(module)
    except ImportError as exc:
        return f"reference oracle {module!r} is not importable ({exc}); install the `reference` dependency group"
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return f"reference oracle {module!r} has no {distribution} distribution metadata"
    if installed != version:
        return f"reference oracle {distribution}=={installed}, but the `reference` group pins {version}"
    return None


def oracle(module: str):
    """Import a pinned oracle: skip when it is optional here, fail when the group is required."""
    if module not in PINS:
        raise KeyError(f"{module!r} is not a pinned reference oracle")
    if not required():
        return pytest.importorskip(module)
    reason = problem(module)
    if reason:
        pytest.fail(f"{ENV}=1: {reason}", pytrace=False)
    return importlib.import_module(module)
