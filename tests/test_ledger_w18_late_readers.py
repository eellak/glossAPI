"""W18 M15: public wrappers opened after a debug-off release re-derive or refuse; never a silently shorter product."""
import json

import pytest

from glossapi import Corpus

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_public_wrappers_after_debug_off_release_rederive_or_refuse(tmp_path):
    from test_ledger_fixset_lg import pdf
    sources = [pdf(tmp_path/'in/a.pdf', 'Alpha heading and body text'), pdf(tmp_path/'in/b.pdf', 'Bravo heading and body text')]
    reference = Corpus(tmp_path/'empty', tmp_path/'ref')
    reference.run(sources, stages=('extract', 'clean', 'export'), settings={'debug': 'on', 'extract': {'repair_fonts': False}})
    reference.section()
    reference.jsonl(tmp_path/'ref-again.jsonl')
    released = Corpus(tmp_path/'empty', tmp_path/'off')
    released.run(sources, stages=('extract', 'clean', 'export'), settings={'debug': 'off', 'extract': {'repair_fonts': False}})
    later = Corpus(tmp_path/'empty', tmp_path/'off')
    outcomes = {}
    for name, call in (('section', lambda: later.section()), ('jsonl', lambda: later.jsonl(tmp_path/'off-again.jsonl'))):
        try:
            call(); outcomes[name] = 'ok'
        except Exception as error:  # a typed refusal is acceptable; a shorter product is not
            outcomes[name] = type(error).__name__+': '+str(error)
    if outcomes['jsonl'] == 'ok':
        again, expected = records(tmp_path/'off-again.jsonl'), records(tmp_path/'ref-again.jsonl')
        assert len(again) == len(expected), outcomes
        assert [row['document'] for row in again] == [row['document'] for row in expected]
    if outcomes['section'] == 'ok':
        import pandas as pd
        ours = pd.read_parquet(tmp_path/'off/sections/sections_for_annotation.parquet')
        theirs = pd.read_parquet(tmp_path/'ref/sections/sections_for_annotation.parquet')
        assert len(ours) == len(theirs) and sorted(ours['filename']) == sorted(theirs['filename']), outcomes
