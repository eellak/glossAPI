"""Fix round 19 (ASTRA-FIXES2c14 M1-R1): a malformed start never makes the next start reuse a generation, and a pass
is current only if its verdict follows the latest ``validation_started`` event.

Regressions for Astra's minimal probe (a malformed start, then a plain ``begin_validation``: the old generation-1 pass
must not revive over a deleted export) and the six public-resume variants (``true``, ``1.0``, ``"1"``, ``null``,
``[1]``, ``{}``).
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from glossapi import Corpus
from glossapi.ledger import writer as W
from glossapi.ledger.writer import LedgerWriter, open_readonly
from test_fix17_astra import _caught, _first_run, _states, _surfaces


def _append_start(writer, run_id, generation):
    with writer.transaction():
        writer.event(run_id, event_kind='run', reason_code='validation_started', message='validation started',
                     details={'generation': generation})


def _generation_one_pass(tmp_path):
    output, run = _first_run(tmp_path)
    good = Corpus(tmp_path / 'empty', output).run(resume=run['run_id'], retry_failed=True)
    assert good['state'] == 'completed' and good['completion_gate']['generation'] == 1
    return output, run['run_id']


def test_a_new_start_alone_cannot_revalidate_an_old_pass(tmp_path, capfd):
    """Astra's minimal probe: a malformed start, then a normal ``begin_validation`` (no gate): the next generation is
    2, never 1 again, and the generation-1 pass (now before the latest start) is not current; the export is deleted."""
    output, run_id = _generation_one_pass(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        _append_start(writer, run_id, True)
    assert not _surfaces(output, run_id, capfd)['certification']['certified']
    (output / 'export.jsonl').unlink()
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        generation = writer.begin_validation(run_id)
    assert generation == 2
    seen = _surfaces(output, run_id, capfd)
    assert not seen['certification']['certified'] and 'completed' not in _states(seen), seen
    assert seen['certification']['stale_reason'] == 'validation_pending'


@pytest.mark.parametrize('bad_start', [True, 1.0, '1', None, [1], {}])
def test_a_real_resume_after_a_malformed_start_never_revives_the_old_pass(tmp_path, monkeypatch, capfd, bad_start):
    """Astra's public-resume variants: a real generation-1 pass, one malformed start, the export deleted, then a public
    resume whose validation refuses and whose refusal cannot be recorded. Its new generation is 2 and no surface says
    completed (the deleted export then keeps the gate refusing until it is restored: nothing certifies it)."""
    output, run_id = _generation_one_pass(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        _append_start(writer, run_id, bad_start)
    assert 'completed' not in _states(_surfaces(output, run_id, capfd))
    (output / 'export.jsonl').unlink()
    real_event = LedgerWriter.event
    entered = []

    def event(self, *args, **kwargs):
        if kwargs.get('reason_code') == 'completion_gate' and kwargs.get('message') == 'refused':
            raise sqlite3.OperationalError('database or disk is full')
        return real_event(self, *args, **kwargs)

    def validate(writer):
        with open_readonly(output / 'corpus.sqlite3') as connection:
            entered.append(W.latest_validation_generation(connection))
        raise W.UnresolvedRecoveryError([{'path': str(output / 'export.jsonl'), 'reason': 'missing_at_flush'}])
    with monkeypatch.context() as patch:
        patch.setattr(LedgerWriter, 'event', event)
        patch.setattr(LedgerWriter, 'verify_excluded_copies', validate)
        result, error = _caught(lambda: Corpus(tmp_path / 'empty', output).run(resume=run_id, retry_failed=True))
    assert result is None and entered == [2] and not (output / 'export.jsonl').exists()
    seen = _surfaces(output, run_id, capfd)
    assert not seen['certification']['certified'] and 'completed' not in _states(seen), seen
    with open_readonly(output / 'corpus.sqlite3') as connection:
        generations = [json.loads(row[0]).get('generation') for row in
                       connection.execute("SELECT details_json FROM events WHERE reason_code='validation_started' ORDER BY event_id")]
    assert [value for value in generations if W.exact_generation(value)] == [1, 2]  # no well-typed generation reused


def test_a_pass_before_the_latest_start_is_never_current_even_with_matching_fields(tmp_path, capfd):
    """A verdict copied to look current (same generation, same fingerprint) but recorded before the latest start event
    is not current: the order of events decides, not the payload alone."""
    output, run_id = _generation_one_pass(tmp_path)
    with LedgerWriter(output / 'corpus.sqlite3') as writer:
        _append_start(writer, run_id, 1)  # a later start record repeating generation 1 (not something the writer makes)
    seen = _surfaces(output, run_id, capfd)
    assert not seen['certification']['certified'] and seen['certification']['stale_reason'] == 'validation_pending', seen
