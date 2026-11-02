"""Provenance ledger: the corpus's SQLite fact journal (``corpus.sqlite3``). Importing this package loads only the stdlib.

The default stage profile (``cutover``) writes it and decides from it: plan, run, resume, retention and the read
commands (status, report) use its runs, tasks, attempts, files and events. ``GLOSSAPI_STAGE_PROFILE=legacy`` runs the
pre-ledger stage code as a comparison control; its wrappers only observe into the ledger (``glossapi.stages.profile``).
"""
from .writer import LedgerWriter, LedgerError, open_readonly

__all__ = ["LedgerWriter", "LedgerError", "open_readonly", "import_corpus", "inventory_corpus"]


def __getattr__(name):
    if name in ("import_corpus", "inventory_corpus"):
        from . import importer
        return getattr(importer, name)
    raise AttributeError(name)
