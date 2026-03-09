"""GlossAPI library."""

from __future__ import annotations

__all__ = [
    'GlossSection',
    'GlossSectionClassifier',
    'Corpus',
    'Sampler',
    'Section',
    'GlossDownloader',
]

def __getattr__(name: str):
    # Lazy access for heavy modules to keep top‑level import light.
    if name == 'Corpus':
        from .corpus.corpus_orchestrator import Corpus  # type: ignore
        return Corpus
    if name == 'GlossSectionClassifier':
        from .gloss_section_classifier import GlossSectionClassifier  # type: ignore
        return GlossSectionClassifier
    if name == 'Sampler':
        from .sampler import Sampler  # type: ignore
        return Sampler
    if name == 'GlossSection':
        from .gloss_section import GlossSection  # type: ignore
        return GlossSection
    if name == 'Section':
        from .gloss_section import Section  # type: ignore
        return Section
    if name == 'GlossDownloader':
        from .gloss_downloader import GlossDownloader  # type: ignore
        return GlossDownloader
    raise AttributeError(name)

try:
    from importlib.metadata import version as _pkg_version
    __version__: str = _pkg_version(__name__)
except Exception:
    __version__ = "0.1.1"
