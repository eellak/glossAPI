"""K4g gap closed in K2 batch 3: every name glossapi exports resolves; the broken `Sampler` export is gone.

`glossapi.sampler` was deleted in 848ab4b (Oct 2025) while `__all__` and the lazy `__getattr__` still named
`Sampler`, so `glossapi.Sampler` raised ModuleNotFoundError. No document, script or test uses it.
"""
import importlib
import importlib.util

import pytest

import glossapi


@pytest.mark.parametrize('name', glossapi.__all__)
def test_every_exported_name_resolves_or_names_its_extra(name):
    try:
        value = getattr(glossapi, name)
    except ImportError as exc:  # optional feature modules: the error names the installable extra
        assert 'glossapi[' in str(exc), str(exc)
    else:
        assert value.__name__ == name


def test_sampler_is_not_exported():
    assert 'Sampler' not in glossapi.__all__
    assert importlib.util.find_spec('glossapi.sampler') is None
    with pytest.raises(AttributeError):
        glossapi.Sampler
