"""Actionable errors at optional feature boundaries."""
from contextlib import contextmanager


@contextmanager
def optional_dependencies(package: str):
    """Keep the missing distribution and the installable feature in the error."""
    try:
        yield
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            f"Missing optional dependency {missing!r}; install {package} with "
            f"`pip install 'glossapi[{package}]'`."
        ) from exc
