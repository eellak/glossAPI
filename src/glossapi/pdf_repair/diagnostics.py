"""Scoped third-party logging capture for one isolated PDF worker."""

from contextlib import contextmanager
import logging
from typing import Iterator


class _DiagnosticHandler(logging.Handler):
    """Count library warnings while forwarding their messages at DEBUG."""

    def __init__(self, counts: dict[str, int]) -> None:
        super().__init__(logging.WARNING)
        self.counts = counts

    def emit(self, record: logging.LogRecord) -> None:
        library = record.name.split(".", 1)[0]
        logger = logging.getLogger(__name__)
        if record.levelno < logging.ERROR:
            self.counts[library] += 1
            logger.debug("%s: %s", record.name, record.getMessage())
        else:
            logger.log(record.levelno, "%s: %s", record.name, record.getMessage())


@contextmanager
def capture_library_warnings() -> Iterator[dict[str, int]]:
    """Quiet pypdf/fontTools warnings for this file, restoring logging on exit.

    CLI workers process exactly one PDF in their own process. Warnings across
    analysis, repair and post-repair analysis count as individual occurrences.
    Errors retain their original logging severity and are not warning counts.
    """
    counts = {"fontTools": 0, "pypdf": 0}
    handler = _DiagnosticHandler(counts)
    saved = []
    try:
        for name in counts:
            logger = logging.getLogger(name)
            saved.append((logger, logger.handlers[:], logger.level, logger.propagate))
            logger.handlers = [handler]
            logger.setLevel(logging.WARNING)
            logger.propagate = False
        yield counts
    finally:
        for logger, handlers, level, propagate in saved:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate
        handler.close()
