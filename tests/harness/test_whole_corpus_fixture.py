"""Global selection/IDs/row groups and export shard membership are frozen."""
import subprocess
import sys


def test_whole_corpus_compatibility_fixture(tmp_path):
    # The gate exercises native parquet/section kernels. Keep their worker
    # pools and allocator state out of the parent process-resource tests.
    result = subprocess.run([sys.executable, "-m", "tools.harness.whole_corpus_fixture",
                             "--work", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
