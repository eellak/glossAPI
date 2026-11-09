"""W18 M12: supplied originals are byte- and inode-identical after every run profile (K4(b)).

Part 1 publishes compatibility products by hard link (K4(c) decision 3). No profile may link, rewrite, touch or add a
name beside a file the caller supplied: same sha256, size, inode, device, link count 1, mtime, and no new names in its
directory, after streaming debug off and on, and after the sequential adapter profile.
"""
import hashlib
import os

import pytest

from glossapi import Corpus

pytest.importorskip('glossapi_rs_cleaner')


@pytest.fixture(autouse=True)
def scratch_log(tmp_path, monkeypatch):
    monkeypatch.setenv('GLOSSAPI_CONVERSION_LOG_FILE', str(tmp_path/'conversion.log'))


def identity(path):
    info = os.stat(path)
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=info.st_size, ino=info.st_ino, dev=info.st_dev,
                nlink=info.st_nlink, mtime_ns=info.st_mtime_ns)


def originals(tmp_path):
    from test_ledger_fixset_lg import pdf
    return [pdf(tmp_path/'supplied'/'alpha.pdf', 'ALPHA original'), pdf(tmp_path/'supplied'/'bravo.pdf', 'BRAVO original')]


@pytest.mark.parametrize('profile', ['streaming-debug-off', 'streaming-debug-on', 'sequential-adapters'])
def test_supplied_originals_are_byte_and_inode_identical_after_runs(tmp_path, monkeypatch, profile):
    sources = originals(tmp_path)
    before = {path: identity(path) for path in sources}
    names = sorted(os.listdir(sources[0].parent))
    assert all(value['nlink'] == 1 for value in before.values())
    corpus = Corpus(tmp_path/'empty', tmp_path/'out')
    if profile == 'sequential-adapters':
        monkeypatch.setenv('GLOSSAPI_STAGE_PROFILE', 'sequential-adapters')
        corpus._shadow_ledger_enabled = True
        corpus.extract(input_format='pdf', file_paths=sources, repair_fonts=False, accel_type='CPU', num_threads=1, batched=False, phase1_backend='safe')
        corpus.clean(num_threads=1)
        corpus.jsonl(corpus.output_dir/'export.jsonl')
    else:
        result = corpus.run(sources, stages=('extract', 'clean', 'sections', 'export'),
                            settings={'debug': profile.rsplit('-', 1)[1], 'extract': {'repair_fonts': False}})
        assert result['state'] == 'completed', result
    assert (corpus.output_dir/'export.jsonl').is_file()
    after = {path: identity(path) for path in sources}
    assert after == before, {str(path): (before[path], after[path]) for path in sources if before[path] != after[path]}
    assert sorted(os.listdir(sources[0].parent)) == names
