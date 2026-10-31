"""The two run settings, in the owner's exact words (PLAN rev 5 section 2.2, unchanged from rev 4 section 2.3).

This module is the single source of the settings text shown beside the modes in ``glossapi plan/run --help``, in
``plan --json`` (``settings_help``) and in the retention summary. It imports only the standard library.
"""
from __future__ import annotations

from typing import Final

SETTINGS_CONTRACT: Final[dict[str, dict[str, str]]] = {
    'debug': {
        'off': ('An intermediate is released only after every planned reader has succeeded on its exact hash. '
                'Supplied files, selected final products, evidence and the ledger are kept.'),
        'on': "Keep every stage's intermediates and attempts, including bounded inner HTTP request bodies.",
    },
    'memory_saving': {
        'normal': 'Never delete a supplied original or a downloaded file.',
        'extreme': ('Delete only downloads this run owns, after the last planned reader succeeds and outputs are durable. '
                    "Never a supplied original, another run's file, or a path reached through a link. Requires a disk budget."),
    },
}
SETTINGS_DEFAULTS: Final[dict[str, str]] = {'debug': 'off', 'memory_saving': 'normal'}
PLAN_FIXED_NOTE: Final[str] = ('Stages are fixed at the start of a run, and evaluation adds no readers. A plan amendment that '
                               'adds a reader is a new plan; it cannot justify a past deletion.')


def setting_help(name: str) -> str:
    """One line per value, the value name first: ``off: <contract>. on: <contract>``."""
    return ' '.join(f'{value}: {text}' for value, text in SETTINGS_CONTRACT[name].items())


def option_help(name: str) -> str:
    values = '/'.join(SETTINGS_CONTRACT[name])
    return f'{name} = {values} (default {SETTINGS_DEFAULTS[name]}). {setting_help(name)}'
