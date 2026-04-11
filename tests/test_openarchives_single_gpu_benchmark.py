from pathlib import Path

from glossapi.scripts.openarchives_single_gpu_benchmark import _resolve_python_bin


def test_resolve_python_bin_keeps_virtualenv_symlink_path(tmp_path):
    repo_root = tmp_path / "repo"
    explicit = repo_root / "dependency_setup" / ".venvs" / "deepseek" / "bin" / "python"
    explicit.parent.mkdir(parents=True, exist_ok=True)

    target = tmp_path / "shared" / "python3.11"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")
    explicit.symlink_to(target)

    resolved = _resolve_python_bin(repo_root, str(explicit))

    assert resolved == explicit
