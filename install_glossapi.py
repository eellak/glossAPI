from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_repo_src() -> None:
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> int:
    _bootstrap_repo_src()
    from glossapi.scripts.install_glossapi import main as _main

    return int(_main())


if __name__ == "__main__":
    raise SystemExit(main())
