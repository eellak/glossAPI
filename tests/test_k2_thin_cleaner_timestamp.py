"""K2 chrono call → std SystemTime helper: the cleaner's temporary pipeline directory keeps its
name format (text_cleaner_rs_pipeline_<unix nanoseconds>) and its cleanup/hand-over behaviour.
The epoch/before-epoch/overflow cases are Rust unit tests (pipeline_module::timestamp_tests)."""
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest

pytest.importorskip('glossapi_rs_cleaner')

SCRIPT = r'''
import os, sys
import glossapi_rs_cleaner as cleaner
source, output, report, evidence = sys.argv[1:5]
cleaner.run_complete_pipeline(source, output, report, ['greek', 'latin'], 1, True,
                              **({'evidence_directory': evidence} if evidence != '-' else {}))
'''


def run_cleaner(tmp_path, evidence):
    source = tmp_path/'in'; source.mkdir()
    (source/'doc.md').write_text('# Τίτλος\n\nΕλληνικό κείμενο με λίγες λέξεις.\n', encoding='utf-8')
    temp = tmp_path/'tmpdir'; temp.mkdir()
    before = time.time_ns()
    result = subprocess.run([sys.executable, '-c', SCRIPT, str(source), str(tmp_path/'out'), str(tmp_path/'report.parquet'),
                             str(evidence) if evidence else '-'], text=True, capture_output=True,
                            env={**os.environ, 'TMPDIR': str(temp)})
    after = time.time_ns()
    assert result.returncode == 0, result.stderr[-3000:]
    created = re.findall(r'Creating main temporary directory for pipeline: (\S+)\.\.\.', result.stdout)
    assert len(created) == 1, result.stdout[-3000:]
    name = Path(created[0]).name
    match = re.fullmatch(r'text_cleaner_rs_pipeline_(-?\d+)', name)
    assert match and before <= int(match.group(1)) <= after, name
    return temp, Path(created[0]), result


def test_temporary_directory_name_and_cleanup_without_evidence(tmp_path):
    temp, created, _ = run_cleaner(tmp_path, None)
    assert created.parent == temp
    assert not created.exists() and list(temp.iterdir()) == []
    assert (tmp_path/'out/doc.md').is_file() and (tmp_path/'report.parquet').is_file()


def test_evidence_directory_receives_the_stage_tree(tmp_path):
    evidence = tmp_path/'evidence'; evidence.mkdir()
    temp, created, _ = run_cleaner(tmp_path, evidence)
    assert created.parent == evidence and not created.exists()
    assert (evidence/'native_stages/stage1_cleaned_files/doc.md').is_file()
    assert [path.name for path in evidence.iterdir() if path.name.startswith('text_cleaner_rs_pipeline_')] == []
    assert list(temp.iterdir()) == []
