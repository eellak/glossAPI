"""Shadow provenance ledger. Importing this package loads only the stdlib.

No pipeline decision may consult this ledger during the shadow phase.
"""
from .writer import LedgerWriter, LedgerError, open_readonly

__all__ = ["LedgerWriter", "LedgerError", "open_readonly"]
