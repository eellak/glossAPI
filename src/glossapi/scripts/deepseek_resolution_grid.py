"""Compare DeepSeek-OCR-2 prompt, token and runtime levers on the vLLM path.

vLLM 0.18.0 overrides processor resolution kwargs; that grid is out of scope.

Timing uses a config-level first-inference-to-last-write window, excluding
model load; page_metrics[].infer_sec is a generation diagnostic. CER is exact character Levenshtein
on whitespace-collapsed text, with a 20,000-character cap (never truncated).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from glossapi.ocr.deepseek.runner import DEFAULT_VLLM_SCRIPT, _auto_vllm_batch_size, _build_cli_command
from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _join_page_outputs, _split_page_outputs

_BASELINE = {
    "--render-dpi": 144,
    "--base-size": 1024,
    "--image-size": 768,
    "--crop-mode": True,
    "--ocr-profile": "markdown_grounded",
    "--max-new-tokens": 2048,
    "--repair-mode": "auto",
    "--gpu-memory-utilization": 0.9,
}
CONFIGS = {
    "baseline": dict(_BASELINE),
    "plain_ocr": {**_BASELINE, "--ocr-profile": "plain_ocr"},
    "tokens_1024": {**_BASELINE, "--max-new-tokens": 1024},
    "batch_32": {**_BASELINE, "--batch-size": 32},
    "batch_64": {**_BASELINE, "--batch-size": 64},
    "norepair": {**_BASELINE, "--repair-mode": "off"},
    "mem_0.95": {**_BASELINE, "--gpu-memory-utilization": 0.95},
    "fp8kv_off": {**_BASELINE, "--disable-fp8-kv": True},
}
CER_LENGTH_CAP = 20_000
TIMING_BASIS = "runtime_summary.steady_state: first inference start to last output write end"
GEN_TIMING_BASIS = "page_metrics.infer_sec (amortized generation wall seconds, including repair)"
PAGE_MARKER = re.compile(r"(?m)^\s*<!-- page:(\d+) -->[ \t]*\n?")


def resolve_pdfs(specs: list[str]) -> tuple[Path, list[str]]:
    """Accept directories, PDF paths, and newline/JSON lists relative to the list."""
    paths = []
    for spec in specs:
        path = Path(spec).expanduser().resolve()
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() == ".pdf"))
        elif path.suffix.lower() == ".pdf":
            paths.append(path)
        else:
            content = path.read_text(encoding="utf-8")
            entries = json.loads(content) if path.suffix.lower() == ".json" else content.splitlines()
            if not isinstance(entries, list) or not all(isinstance(p, str) for p in entries):
                raise ValueError(f"Expected a list of PDF paths: {path}")
            paths.extend((path.parent / Path(p.strip()).expanduser()).resolve() for p in entries if p.strip())
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No PDFs found")
    stems = set()
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {path}")
        if path.stem in stems:
            raise ValueError(f"Duplicate PDF stem (runner artifact collision): {path.stem}")
        stems.add(path.stem)
    input_dir = Path(os.path.commonpath([str(p.parent) for p in paths]))
    return input_dir, [str(p.relative_to(input_dir)) for p in paths]


def select_configs(names: list[str]) -> list[str]:
    names = [name for item in names for name in item.split(",")]
    if names == ["all"]:
        return list(CONFIGS)
    unknown = set(names) - CONFIGS.keys()
    if unknown:
        raise ValueError(f"Unknown configs: {', '.join(sorted(unknown))}")
    return list(dict.fromkeys(["baseline", *names]))


def build_runner_argv(
    config: str, *, input_dir: Path, output_dir: Path, files: list[str],
    model_dir: Path, python_bin: str = sys.executable,
    max_pages: int | None = None, batch_size: int | None = None,
) -> list[str]:
    flags = CONFIGS[config]
    batch_size = flags.get("--batch-size", batch_size)
    if batch_size is None:
        batch_size = _auto_vllm_batch_size(runtime_backend="vllm", file_list=files,
                                          input_root=input_dir, max_pages=max_pages)
    return _build_cli_command(
        input_dir=input_dir, output_dir=output_dir, files=files, page_ranges=None,
        model_dir=model_dir, python_bin=Path(python_bin), script=DEFAULT_VLLM_SCRIPT,
        max_pages=max_pages, content_debug=False, device="cuda",
        ocr_profile=flags["--ocr-profile"], prompt_override=None, attn_backend="vllm",
        base_size=flags["--base-size"], image_size=flags["--image-size"],
        crop_mode="--crop-mode" in flags, render_dpi=flags["--render-dpi"],
        max_new_tokens=flags["--max-new-tokens"], repetition_penalty=None,
        no_repeat_ngram_size=None, runtime_backend="vllm", vllm_batch_size=batch_size,
        gpu_memory_utilization=flags["--gpu-memory-utilization"],
        disable_fp8_kv=flags.get("--disable-fp8-kv", False), repair_mode=flags["--repair-mode"],
    )


def wrap_env_file(argv: list[str], env_file: Path | None) -> list[str]:
    if env_file is None:
        return argv
    # Paths and argv are positional parameters, never interpolated shell code.
    return ["bash", "-c", 'source "$1" && shift && exec "$@"', "deepseek-grid", str(env_file), *argv]


def run_runner(argv: list[str], log_path: Path) -> None:
    """Monkeypatchable subprocess boundary: one sequential process per config."""
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)


def fake_runner(argv: list[str], log_path: Path) -> None:
    """Synthetic artifact writer; never reads PDF contents or loads a model."""
    def value(flag):
        return argv[argv.index(flag) + 1]

    output_dir = Path(value("--output-dir"))
    start = argv.index("--files") + 1
    end = next((i for i in range(start, len(argv)) if argv[i].startswith("--")), len(argv))
    page_count = min(2, int(value("--max-pages"))) if "--max-pages" in argv else 2
    baseline = output_dir.name == "baseline"
    sec = 1.0 if baseline else 0.6
    metrics_dir = output_dir / "json" / "metrics"
    markdown_dir = output_dir / "markdown"
    metrics_dir.mkdir(parents=True)
    markdown_dir.mkdir(parents=True)
    for filename in argv[start:end]:
        stem = Path(filename).stem
        texts = [(f"Synthetic Greek OCR rehearsal {stem} page {page}. " * 10).strip()
                 for page in range(1, page_count + 1)]
        metrics = {
            "page_count": page_count, "model": "deepseek-ai/DeepSeek-OCR-2",
            "source_file": filename, "source_stem": stem, "source_start_page": 1,
            "render_sec": 0.1, "infer_sec_total": sec * page_count,
            "wall_time_sec": sec * page_count + 0.1,
            "repair_summary": {"repair_mode": value("--repair-mode"), "pages_repaired": 0},
            "page_metrics": [
                {"page_number": page, "infer_sec": sec, "repair_strategy": "none",
                 "prompt_tokens": 1132, "output_tokens": 100,
                 "repair_attempted": False, "repair_applied": False,
                 "page_dropped_after_repair": False, "empty_page_skipped": False}
                for page in range(1, page_count + 1)
            ],
        }
        (metrics_dir / f"{stem}.metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (markdown_dir / f"{stem}.md").write_text(_join_page_outputs(texts) + "\n", encoding="utf-8")
    from datetime import timedelta, timezone

    start_time = datetime(2026, 10, 6, tzinfo=timezone.utc)
    wall = sec * page_count * len(argv[start:end])
    (output_dir / "runtime_summary.json").write_text(json.dumps({"steady_state": {
        "first_batch_started_at": start_time.isoformat(),
        "last_batch_finished_at": (start_time + timedelta(seconds=wall)).isoformat(),
        "first_batch_to_last_batch_window_sec": wall,
    }}), encoding="utf-8")
    log_path.write_text("SYNTHETIC dry-run: no OCR, GPU, or model was used.\n", encoding="utf-8")


def _seconds(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {label}: {value}")
    return result


def pipeline_wall_seconds(output_dir: Path) -> float:
    """One overlapping-work window, never a sum of document wall clocks."""
    steady = json.loads((output_dir / "runtime_summary.json").read_text(encoding="utf-8"))["steady_state"]
    start_value = steady.get("first_batch_started_at")
    end_value = steady.get("last_write_completed_at") or steady.get("last_batch_finished_at")
    if not isinstance(start_value, str) or not isinstance(end_value, str):
        raise ValueError("Missing first-inference or last-write runtime timestamp")
    start = datetime.fromisoformat(start_value.replace("Z", "+00:00"))
    end = datetime.fromisoformat(end_value.replace("Z", "+00:00"))
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Runtime timestamps must include a timezone")
    return _seconds((end - start).total_seconds(), "post-load pipeline wall seconds")


def parse_document(config: str, metrics_path: Path, markdown_path: Path, source_file: str) -> list[dict]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    count = int(metrics["page_count"])
    if count <= 0:
        raise ValueError(f"No rendered pages: {metrics_path}")
    markdown = markdown_path.read_text(encoding="utf-8")
    if "<--- Page Split --->" in markdown:
        texts = _split_page_outputs(markdown)
    elif PAGE_MARKER.search(markdown):
        markers = list(PAGE_MARKER.finditer(markdown))
        if [int(m.group(1)) for m in markers] != list(range(1, count + 1)):
            raise ValueError(f"Noncontiguous page markers: {markdown_path}")
        texts = [markdown[m.end():markers[i + 1].start() if i + 1 < len(markers) else len(markdown)]
                 for i, m in enumerate(markers)]
    else:
        texts = [markdown]
    if len(texts) != count:
        raise ValueError(f"Markdown/page-count mismatch: {markdown_path}")
    texts = [PAGE_MARKER.sub("", text).strip() for text in texts]
    pages = metrics["page_metrics"]
    if len(pages) != count or any(not isinstance(p, dict) for p in pages):
        raise ValueError(f"Incomplete page metrics: {metrics_path}")
    by_page = {int(p["page_number"]): p for p in pages}
    if set(by_page) != set(range(1, count + 1)):
        raise ValueError(f"Noncontiguous page metrics: {metrics_path}")
    rows = []
    runtime_path = metrics_path.parent.parent.parent / "runtime_summary.json"
    wall = pipeline_wall_seconds(runtime_path.parent) if runtime_path.is_file() else None
    for number in range(1, count + 1):
        page = by_page[number]
        sec = _seconds(page["infer_sec"], "infer_sec")
        # Count generated work, including repair even when its output is discarded.
        # Never substitute characters or max_new_tokens for missing counts.
        tokens = page.get("output_tokens")
        if tokens is not None:
            tokens = int(tokens)
            if page.get("repair_attempted"):
                repair_tokens = page.get("repair_output_tokens")
                tokens = None if repair_tokens is None else tokens + int(repair_tokens)
            if tokens is not None and tokens < 0:
                raise ValueError("Negative output token count")
        rows.append({
            "config": config, "source_file": source_file,
            "page_number": int(metrics.get("source_start_page", 1)) + number - 1,
            "generation_seconds_per_page": sec, "generation_timing_basis": GEN_TIMING_BASIS,
            # Amortization across ALL config pages is completed in main after
            # every document is parsed. Retain the whole window on each row.
            "pipeline_wall_seconds": wall, "wall_seconds_per_page": None,
            "timing_basis": TIMING_BASIS,
            "prompt_tokens": page.get("prompt_tokens"),
            "repair_prompt_tokens": page.get("repair_prompt_tokens"),
            "mm_processor_kwargs": metrics.get("mm_processor_kwargs"),
            "mm_processor_kwargs_honoured": metrics.get("mm_processor_kwargs_honoured"),
            "first_pass_output_tokens": page.get("output_tokens"),
            "repair_output_tokens": page.get("repair_output_tokens"),
            "output_tokens": tokens, "text": texts[number - 1],
            "repair_flags": {key: val for key, val in page.items()
                             if key.startswith("repair_") or key in (
                                 "page_dropped_after_repair", "drop_reason", "empty_page_skipped",
                                 "garbage_early_stop_applied", "early_stop_rules")},
            # Document clocks overlap other documents and cannot be summed into throughput.
            "document_wall_time_sec": metrics.get("wall_time_sec"),
            "document_render_sec": metrics.get("render_sec"),
            "metrics_file": str(metrics_path),
        })
    return rows


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def levenshtein(reference: str, variant: str) -> int:
    """Exact bit-vector Levenshtein (unit insert/delete/substitute costs), pure Python."""
    if not reference:
        return len(variant)
    if not variant:
        return len(reference)
    if reference == variant:
        return 0
    if len(reference) > len(variant):
        reference, variant = variant, reference
    masks = {}
    for i, char in enumerate(reference):
        masks[char] = masks.get(char, 0) | (1 << i)
    positive, negative, distance = ~0, 0, len(reference)
    last_bit = 1 << (len(reference) - 1)
    for char in variant:
        equal = masks.get(char, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        distance += bool(plus & last_bit) - bool(minus & last_bit)
        plus = (plus << 1) | 1
        minus <<= 1
        positive = minus | ~(vertical | plus)
        negative = plus & vertical
    return distance


def fidelity(reference: str, variant: str, length_cap: int = CER_LENGTH_CAP) -> dict:
    reference, variant = normalize_text(reference), normalize_text(variant)
    capped = max(len(reference), len(variant)) > length_cap
    return {
        "baseline_chars": len(reference), "variant_chars": len(variant),
        "cer": None if capped or not reference else levenshtein(reference, variant) / len(reference),
        "coverage": len(variant) / len(reference) if reference else None,
        "cer_length_cap_exceeded": capped,
    }


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(rows: list[dict], statuses: dict[str, dict], *, dry_run: bool) -> dict:
    baseline = {(r["source_file"], r["page_number"]): r for r in rows if r["config"] == "baseline"}
    for row in rows:
        ref = baseline.get((row["source_file"], row["page_number"]))
        row.update(fidelity(ref["text"], row["text"]) if ref else {
            "baseline_chars": None, "variant_chars": len(normalize_text(row["text"])),
            "cer": None, "coverage": None, "cer_length_cap_exceeded": False,
        })
    def generation(row):
        return row.get("generation_seconds_per_page", row["wall_seconds_per_page"])

    def wall_seconds(pages):
        if not pages:
            return None
        windows = [r.get("pipeline_wall_seconds") for r in pages]
        if any(v is not None for v in windows):
            if any(v is None for v in windows) or len(set(windows)) != 1:
                return None
            return _seconds(windows[0], "pipeline_wall_seconds")
        # Direct callers may supply already amortized pipeline wall seconds.
        # parse_document never substitutes infer_sec here when a sidecar is absent.
        values = [r["wall_seconds_per_page"] for r in pages]
        return sum(_seconds(v, "wall_seconds_per_page") for v in values) if all(v is not None for v in values) else None

    base_sec = _median(generation(r) for r in baseline.values())
    base_wall = wall_seconds(list(baseline.values()))
    summaries = {}
    for config, status in statuses.items():
        pages = [r for r in rows if r["config"] == config]
        sec = _median(generation(r) for r in pages)
        gen_ratio = sec / base_sec if sec is not None and base_sec else None
        wall = wall_seconds(pages)
        wall_per_page = wall / len(pages) if wall is not None else None
        ratio = wall / base_wall if wall is not None and base_wall else None
        eligible = [r for r in pages if r["baseline_chars"] is not None and r["baseline_chars"] >= 200]
        cer = _median(r["cer"] for r in eligible)
        complete = bool(baseline) and statuses["baseline"]["status"] == "ok" and status["status"] == "ok"
        complete = complete and {(r["source_file"], r["page_number"]) for r in pages} == set(baseline)
        assessed = bool(eligible) and all(r["cer"] is not None for r in eligible)
        passes = bool(complete and assessed and cer <= 0.02 and ratio is not None and ratio <= 0.7)
        tokens = [r["output_tokens"] for r in pages]
        summaries[config] = {
            **status, "flags": CONFIGS[config], "pages": len(pages),
            "median_generation_seconds_per_page": sec, "gen_ratio": gen_ratio,
            "pipeline_wall_seconds": wall, "wall_seconds_per_page_per_gpu": wall_per_page,
            "median_seconds_per_page_per_gpu": wall_per_page, "ratio_to_baseline": ratio,
            "wall_ratio": ratio,
            "speedup_fraction": 1 - ratio if ratio is not None else None,
            "median_tokens_per_page": _median(tokens),
            "median_output_tokens_per_page": _median(tokens),
            "pages_reporting_tokens": sum(t is not None for t in tokens),
            "median_prompt_tokens_per_page": _median(r.get("prompt_tokens") for r in pages),
            "pages_reporting_prompt_tokens": sum(r.get("prompt_tokens") is not None for r in pages),
            "median_repair_prompt_tokens_per_page": _median(r.get("repair_prompt_tokens") for r in pages),
            "fidelity_pages": len(eligible), "median_cer": cer,
            "median_coverage": _median(r["coverage"] for r in pages),
            "complete_baseline_comparison": complete, "passes_gate": passes,
            "accepted": passes and not dry_run,
        }
    return {
        "dry_run": dry_run, "synthetic": dry_run, "gpu_count": 1,
        "timing_basis": TIMING_BASIS, "generation_timing_basis": GEN_TIMING_BASIS, "model_load_excluded": True,
        "cer_method": "exact pure-Python bit-vector Levenshtein / normalized baseline length",
        "cer_length_cap": CER_LENGTH_CAP,
        "gate": {"min_baseline_chars": 200, "max_median_cer": 0.02, "max_time_ratio": 0.7},
        "configs": summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdfs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", default=os.environ.get("GLOSSAPI_DEEPSEEK_MODEL_DIR"))
    parser.add_argument("--configs", nargs="+", default=["all"])
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--batch-size", type=int, help="Default: production page-count selection, capped at 160")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.batch_size is not None and args.batch_size <= 0) or (args.max_pages is not None and args.max_pages <= 0):
        parser.error("--batch-size and --max-pages must be positive")
    if not args.model_dir and not args.dry_run:
        parser.error("Set --model-dir or GLOSSAPI_DEEPSEEK_MODEL_DIR")
    try:
        input_dir, files = resolve_pdfs(args.pdfs)
        configs = select_configs(args.configs)
        out = args.out.expanduser().resolve()
        model_dir = Path(args.model_dir or "synthetic-model-not-loaded").expanduser().resolve()
        python_bin = os.path.expanduser(args.python_bin)
        env_file = args.env_file.expanduser().resolve() if args.env_file else None
        if env_file is not None and not env_file.is_file():
            raise ValueError(f"Missing env file: {env_file}")
        # Fail before launching anything if old artifacts could contaminate this run.
        targets = [out / c for c in configs] + [out / "results.jsonl", out / "summary.json", out / "config.json"]
        if any(p.exists() and (not p.is_dir() or any(p.iterdir())) for p in targets):
            raise ValueError("Output contains previous grid artifacts; use a fresh --out directory")
        batch_size = args.batch_size if args.batch_size is not None else _auto_vllm_batch_size(
            runtime_backend="vllm", file_list=files, input_root=input_dir, max_pages=args.max_pages)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    rows, statuses, commands = [], {}, {}
    for config in configs:
        run_dir = out / config
        run_dir.mkdir(exist_ok=True)
        runner_argv = build_runner_argv(
            config, input_dir=input_dir, output_dir=run_dir, files=files,
            model_dir=model_dir, python_bin=python_bin, max_pages=args.max_pages, batch_size=batch_size,
        )
        command = wrap_env_file(runner_argv, env_file)
        commands[config] = {"runner_argv": runner_argv, "launch_argv": command}
        print(f"{config}: {'synthetic rehearsal' if args.dry_run else 'running'}", flush=True)
        try:
            if args.dry_run:
                fake_runner(runner_argv, run_dir / "runner.log")
            else:
                run_runner(command, run_dir / "runner.log")
            config_rows = []
            for filename in files:
                stem = Path(filename).stem
                doc_rows = parse_document(config, run_dir / "json" / "metrics" / f"{stem}.metrics.json",
                                          run_dir / "markdown" / f"{stem}.md", filename)
                if args.max_pages is not None and len(doc_rows) > args.max_pages:
                    raise ValueError(f"Runner exceeded --max-pages for {filename}")
                config_rows.extend(doc_rows)
            wall = pipeline_wall_seconds(run_dir)
            for row in config_rows:
                row["wall_seconds_per_page"] = wall / len(config_rows)
            rows.extend(config_rows)
            statuses[config] = {"status": "ok"}
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            statuses[config] = {"status": "failed", "error": str(exc)}
            print(f"{config}: failed: {exc}", file=sys.stderr)
    summary = summarize(rows, statuses, dry_run=args.dry_run)
    (out / "config.json").write_text(json.dumps({
        "input_dir": str(input_dir), "files": files, "model_dir": str(model_dir),
        "max_pages_per_pdf": args.max_pages, "batch_size": batch_size,
        "requested_batch_size": args.batch_size,
        "dry_run": args.dry_run, "env_file": str(env_file) if env_file else None, "commands": commands,
    }, indent=2), encoding="utf-8")
    (out / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} page rows to {out}", flush=True)
    return int(any(s["status"] != "ok" for s in statuses.values()))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
