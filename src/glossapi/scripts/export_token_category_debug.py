from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from glossapi.scripts.token_category_debug_common import load_rust_extension


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export token-category debug pages and structured review artifacts.",
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--category-specs", required=True, type=Path)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--synthetic-page-target-chars", type=int, default=4000)
    parser.add_argument("--synthetic-page-min-header-chars", type=int, default=1200)
    parser.add_argument("--synthetic-page-hard-max-chars", type=int, default=6000)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    category_specs = args.category_specs.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_mod = load_rust_extension(
        project_root=Path(__file__).resolve().parents[3],
        module_name="glossapi_rs_noise",
        manifest_relative="rust/glossapi_rs_noise/Cargo.toml",
        required_attrs=("export_token_category_debug_pages",),
    )
    rows = list(
        noise_mod.export_token_category_debug_pages(
            str(input_dir),
            str(output_dir),
            str(category_specs),
            args.num_threads,
            args.max_pages,
            int(args.sample_seed),
            int(args.synthetic_page_target_chars),
            int(args.synthetic_page_min_header_chars),
            int(args.synthetic_page_hard_max_chars),
        )
    )
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    else:
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "page_count": len(rows),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
