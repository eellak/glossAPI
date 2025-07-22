"""GlossAPI package root.

This lightweight wrapper ensures the compiled Rust extension and the actual
Python sources (located in `pipeline/src/glossapi`) are import-able when the
package is installed via `pip`.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# 1. Ensure the compiled Rust extension is importable
# (installed as `glossapi_rs_cleaner` by maturin) – nothing to do.

# 2. Expose the pure-Python implementation directory.
_src_dir = Path(__file__).resolve().parent.parent / "pipeline" / "src"
if _src_dir.exists() and str(_src_dir) not in sys.path:
    # Insert *ahead* of site-packages to prefer editable sources.
    sys.path.insert(0, str(_src_dir))

# 3. Lazily re-export key sub-modules for convenience.

def __getattr__(name: str):
    """On first attribute access, defer to the real module under _src_dir."""
    try:
        module = importlib.import_module(f"glossapi.{name}")
        setattr(sys.modules[__name__], name, module)
        return module
    except ModuleNotFoundError:
        raise AttributeError(name) from None

# For `from glossapi import Corpus` style imports
try:
    from glossapi.corpus import Corpus  # type: ignore # noqa: F401
except Exception:  # pragma: no cover – might fail in minimal env
    # fall back silently; attribute access will still work via __getattr__
    pass
