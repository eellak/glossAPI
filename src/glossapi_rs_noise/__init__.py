"""Placeholder package for the Rust extension module `glossapi_rs_noise`.

This empty package satisfies `maturin`'s `python-source` validation during
editable installs. The real native extension is built by the Rust crate and
will be provided at install/build time.
"""

# Allow importing the compiled extension if present; keep metadata generation
# and static analysis happy when it's not yet built.
try:
    # the compiled extension may be a top-level module or a submodule; try
    # both import styles without failing when running metadata generation.
    from . import glossapi_rs_noise  # type: ignore
except Exception:
    pass
