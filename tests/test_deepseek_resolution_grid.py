import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from glossapi.scripts import deepseek_resolution_grid as grid
from glossapi.ocr.deepseek import run_pdf_ocr_vllm as real


@pytest.mark.parametrize("name,profile,tokens,batch,memory,repair,fp8off", [
    ("baseline", "markdown_grounded", 2048, 8, 0.9, "auto", False),
    ("plain_ocr", "plain_ocr", 2048, 8, 0.9, "auto", False),
    ("tokens_1024", "markdown_grounded", 1024, 8, 0.9, "auto", False),
    ("batch_32", "markdown_grounded", 2048, 32, 0.9, "auto", False),
    ("batch_64", "markdown_grounded", 2048, 64, 0.9, "auto", False),
    ("norepair", "markdown_grounded", 2048, 8, 0.9, "off", False),
    ("mem_0.95", "markdown_grounded", 2048, 8, 0.95, "auto", False),
    ("fp8kv_off", "markdown_grounded", 2048, 8, 0.9, "auto", True),
])
def test_exact_runner_argv(name, profile, tokens, batch, memory, repair, fp8off):
    argv = grid.build_runner_argv(
        name, input_dir=Path("/input"), output_dir=Path("/output"),
        files=["one.pdf", "sub/two.pdf"], model_dir=Path("/model"),
        python_bin="/venv/python", max_pages=3, batch_size=8,
    )
    assert argv == [
        "/venv/python", str(grid.DEFAULT_VLLM_SCRIPT),
        "--input-dir", "/input", "--output-dir", "/output", "--model-dir", "/model",
        "--files", "one.pdf", "sub/two.pdf", "--max-pages", "3",
        "--device", "cuda", "--ocr-profile", profile, "--attn-backend", "vllm",
        "--base-size", "1024", "--image-size", "768", "--crop-mode", "--render-dpi", "144",
        "--max-new-tokens", str(tokens), "--batch-size", str(batch),
        "--gpu-memory-utilization", str(memory),
        *(["--disable-fp8-kv"] if fp8off else []), "--repair-mode", repair,
    ]


def test_no_max_pages_and_config_selection():
    argv = grid.build_runner_argv("baseline", input_dir=Path("/input"), output_dir=Path("/output"),
                                  files=["a.pdf"], model_dir=Path("/model"))
    assert "--max-pages" not in argv
    assert argv[0] == sys.executable
    assert grid.select_configs(["tokens_1024,mem_0.95", "tokens_1024"]) == ["baseline", "tokens_1024", "mem_0.95"]
    assert grid.select_configs(["all"]) == list(grid.CONFIGS)
    with pytest.raises(ValueError, match="Unknown configs"):
        grid.select_configs(["other"])
    for old in ('img640', 'small_768_512', 'nocrop_1024', 'dpi110', 'baseline_norepair'):
        with pytest.raises(ValueError, match="Unknown configs"):
            grid.select_configs([old])


def write_real_schema(tmp_path):
    # _write_outputs + _run_jobs_to_outputs in the real runner. infer_sec is
    # already first-pass plus repair (0.3 + 0.2); wall clocks overlap documents.
    metrics = {
        "page_count": 3, "model": "deepseek-ai/DeepSeek-OCR-2",
        "source_file": "doc.pdf", "source_stem": "doc", "source_start_page": 5,
        "source_end_page": 7, "ocr_profile": "markdown_grounded", "runtime_backend": "vllm",
        "base_size": 1024, "image_size": 768, "crop_mode": True, "render_dpi": 144,
        "max_new_tokens": 2048, "batch_size": 8, "repair_mode": "auto",
        "render_sec": 2.0, "infer_sec_total": 0.8, "wall_time_sec": 999.0,
        "repair_summary": {"repair_mode": "auto", "pages_flagged": 2, "pages_repaired": 1},
        "page_metrics": [
            {"page_number": 1, "infer_sec": 0.5, "raw_chars": 20, "final_chars": 10,
             "repair_strategy": "plain", "repair_reason": "early_stop_markdown_garbage",
             "repair_attempted": True, "repair_applied": True, "repair_infer_sec": 0.2,
             "repair_profile": "plain_ocr", "page_dropped_after_repair": False,
             "garbage_early_stop_applied": True, "early_stop_rules": ["numeric_list_garbage"]},
            {"page_number": 2, "infer_sec": 0.0, "repair_strategy": "skip_empty",
             "repair_reason": "empty_page", "empty_page_skipped": True,
             "repair_attempted": False, "repair_applied": False},
            {"page_number": 3, "infer_sec": 0.3, "repair_attempted": True,
             "repair_applied": False, "page_dropped_after_repair": True,
             "drop_reason": "repeat_garbage_cutoff"},
        ],
    }
    mp, md = tmp_path / "doc.metrics.json", tmp_path / "doc.md"
    mp.write_text(json.dumps(metrics), encoding="utf-8")
    # _join_page_outputs gives no page:1 marker unless content-debug is enabled.
    md.write_text("<!-- page:1 -->\nκαλημέρα\n<!-- page:2 -->\n<--- Page Split --->\n"
                  "\n<!-- page:3 -->\n<--- Page Split --->\n", encoding="utf-8")
    return mp, md, metrics


def test_real_metrics_schema_and_blank_pages(tmp_path):
    mp, md, _ = write_real_schema(tmp_path)
    rows = grid.parse_document("baseline", mp, md, "doc.pdf")
    assert [r["page_number"] for r in rows] == [5, 6, 7]
    assert [r["text"] for r in rows] == ["καλημέρα", "", ""]
    assert [r["generation_seconds_per_page"] for r in rows] == [0.5, 0.0, 0.3]
    assert all(r["wall_seconds_per_page"] is None for r in rows)
    assert all(r["output_tokens"] is None for r in rows)
    assert rows[0]["repair_flags"]["repair_infer_sec"] == 0.2  # no double counting
    assert rows[1]["repair_flags"]["empty_page_skipped"] is True
    assert rows[2]["repair_flags"]["page_dropped_after_repair"] is True
    assert rows[0]["document_wall_time_sec"] == 999.0


def test_optional_reported_token_counts(tmp_path):
    mp, md, metrics = write_real_schema(tmp_path)
    metrics["page_metrics"][0].update(output_tokens=40, repair_output_tokens=10,
                                     prompt_tokens=1132, repair_prompt_tokens=1120)
    metrics["page_metrics"][1]["output_tokens"] = 0
    metrics["page_metrics"][2]["output_tokens"] = 40  # repair count unknown
    mp.write_text(json.dumps(metrics), encoding="utf-8")
    rows = grid.parse_document("baseline", mp, md, "doc.pdf")
    assert [r["output_tokens"] for r in rows] == [50, 0, None]
    assert rows[0]['prompt_tokens'] == 1132
    assert rows[0]['repair_prompt_tokens'] == 1120
    assert rows[0]['first_pass_output_tokens'] == 40
    assert rows[0]['repair_output_tokens'] == 10


def test_markers_only_format(tmp_path):
    mp, md, _ = write_real_schema(tmp_path)
    md.write_text("<!-- page:1 -->\none\n<!-- page:2 -->\n\n<!-- page:3 -->\nthree", encoding="utf-8")
    assert [r["text"] for r in grid.parse_document("baseline", mp, md, "doc.pdf")] == ["one", "", "three"]


@pytest.mark.parametrize("damage", ["missing_metric", "missing_text", "nan_timing", "duplicate_page"])
def test_incomplete_or_invalid_artifacts_rejected(tmp_path, damage):
    mp, md, metrics = write_real_schema(tmp_path)
    if damage == "missing_metric":
        metrics["page_metrics"].pop()
    elif damage == "missing_text":
        md.write_text("one page only", encoding="utf-8")
    elif damage == "nan_timing":
        metrics["page_metrics"][0]["infer_sec"] = float("nan")
    else:
        metrics["page_metrics"][1]["page_number"] = 1
    mp.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError):
        grid.parse_document("baseline", mp, md, "doc.pdf")


@pytest.mark.parametrize("reference,variant,distance", [
    ("kitten", "sitting", 3), ("abc", "ac", 1), ("abc", "axbc", 1),
    ("abc", "adc", 1), ("", "abc", 3), ("abc", "", 3), ("", "", 0),
    ("Ελλάδα", "Ελλαδα", 1), ("abc", "abc", 0), ("abc", "xyz", 3),
])
def test_exact_levenshtein(reference, variant, distance):
    assert grid.levenshtein(reference, variant) == distance


def test_levenshtein_against_independent_matrix_oracle():
    strings = ["".join(p) for n in range(5) for p in itertools.product("ab", repeat=n)]
    for a in strings:
        for b in strings:
            matrix = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
            for i in range(len(a) + 1):
                matrix[i][0] = i
            for j in range(len(b) + 1):
                matrix[0][j] = j
            for i in range(1, len(a) + 1):
                for j in range(1, len(b) + 1):
                    matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1,
                                       matrix[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            assert grid.levenshtein(a, b) == matrix[-1][-1]


def test_cer_coverage_whitespace_and_cap():
    result = grid.fidelity("  abc\n  def  ", "abc\tdef")
    assert result["cer"] == 0 and result["coverage"] == 1
    assert result["baseline_chars"] == 7
    assert grid.fidelity("abcd", "abc")["cer"] == 0.25
    assert grid.fidelity("abcd", "abc")["coverage"] == 0.75
    assert grid.fidelity("", "abc")["cer"] is None
    capped = grid.fidelity("abcdef", "abc", length_cap=5)
    assert capped["cer"] is None and capped["cer_length_cap_exceeded"] is True
    assert capped["coverage"] == 0.5


def row(config, text, seconds, page=1, tokens=None):
    return {"config": config, "source_file": "doc.pdf", "page_number": page,
            "text": text, "wall_seconds_per_page": seconds, "output_tokens": tokens}


@pytest.mark.parametrize("seconds,changes,chars,expected", [
    (0.7, 4, 200, True), (0.70001, 4, 200, False),
    (0.7, 5, 200, False), (0.6, 0, 199, False), (0.6, 0, 200, True),
])
def test_combined_gate_boundaries(seconds, changes, chars, expected):
    rows = [row("baseline", "a" * chars, 1.0, tokens=100),
            row("tokens_1024", "b" * changes + "a" * (chars - changes), seconds, tokens=80)]
    summary = grid.summarize(rows, {c: {"status": "ok"} for c in ["baseline", "tokens_1024"]}, dry_run=False)
    assert summary["configs"]["tokens_1024"]["passes_gate"] is expected
    assert summary["configs"]["tokens_1024"]["accepted"] is expected
    assert summary["configs"]["baseline"]["passes_gate"] is False
    assert summary["configs"]["tokens_1024"]["median_tokens_per_page"] == 80


def test_gate_uses_medians_and_excludes_short_baseline_pages():
    rows = []
    for page, sec, text in [(1, 0.5, "a" * 200), (2, 0.7, "a" * 200), (3, 5.0, "b" * 200)]:
        rows.extend([row("baseline", "a" * 200, 1.0, page), row("tokens_1024", text, sec, page)])
    rows.extend([row("baseline", "a" * 10, 1.0, 4), row("tokens_1024", "bad", 0.7, 4)])
    for item in rows:
        item["generation_seconds_per_page"] = item["wall_seconds_per_page"]
        # Config windows are independent of generation medians.
        item["pipeline_wall_seconds"] = 4.0 if item["config"] == "baseline" else 2.8
    summary = grid.summarize(rows, {c: {"status": "ok"} for c in ["baseline", "tokens_1024"]}, dry_run=True)
    variant = summary["configs"]["tokens_1024"]
    assert variant["median_cer"] == 0 and variant["fidelity_pages"] == 3
    assert variant["gen_ratio"] == 0.7 and variant["wall_ratio"] == 0.7
    assert variant["passes_gate"] is True and variant["accepted"] is False


def test_missing_pages_failed_status_and_capped_text_cannot_pass():
    statuses = {c: {"status": "ok"} for c in ["baseline", "tokens_1024"]}
    rows = [row("baseline", "a" * 200, 1, 1), row("baseline", "a" * 200, 1, 2),
            row("tokens_1024", "a" * 200, 0.5, 1)]
    assert grid.summarize(rows, statuses, dry_run=False)["configs"]["tokens_1024"]["passes_gate"] is False
    rows = [row("baseline", "a" * (grid.CER_LENGTH_CAP + 1), 1),
            row("tokens_1024", "a" * (grid.CER_LENGTH_CAP + 1), 0.5)]
    assert grid.summarize(rows, statuses, dry_run=False)["configs"]["tokens_1024"]["passes_gate"] is False
    statuses["baseline"] = {"status": "failed"}
    rows = [row("baseline", "a" * 200, 1), row("tokens_1024", "a" * 200, 0.5)]
    assert grid.summarize(rows, statuses, dry_run=False)["configs"]["tokens_1024"]["passes_gate"] is False


def test_pdf_lists_and_duplicate_stems(tmp_path):
    pdf = tmp_path / "pages.pdf"
    pdf.touch()
    other = tmp_path / "other"
    other.mkdir()
    (other / "more.PDF").touch()
    listing = tmp_path / "pdfs.json"
    listing.write_text(json.dumps(["pages.pdf", "other/more.PDF", "pages.pdf"]), encoding="utf-8")
    root, files = grid.resolve_pdfs([str(listing)])
    assert root == tmp_path and files == ["pages.pdf", "other/more.PDF"]
    txt = tmp_path / "pdfs.txt"
    txt.write_text("pages.pdf\n\nother/more.PDF\n", encoding="utf-8")
    assert grid.resolve_pdfs([str(txt)]) == (root, files)
    (other / "pages.pdf").touch()
    with pytest.raises(ValueError, match="Duplicate PDF stem"):
        grid.resolve_pdfs([str(pdf), str(other / "pages.pdf")])


def test_env_file_wrapper_preserves_literal_arguments(tmp_path):
    env = tmp_path / "env file ' $(literal).sh"
    env.write_text("export E4_GRID_TEST_VALUE=loaded\n", encoding="utf-8")
    code = "import os,sys,json; print(json.dumps([os.environ['E4_GRID_TEST_VALUE'],sys.argv[1:]]))"
    argv = [sys.executable, "-c", code, "a b", "$(not-a-command)", "quote'arg"]
    result = subprocess.run(grid.wrap_env_file(argv, env), check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == ["loaded", argv[3:]]
    assert grid.wrap_env_file(argv, None) == argv


def test_dry_run_all_configs_end_to_end(tmp_path, monkeypatch):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    (pdfs / "sample.pdf").write_bytes(b"not read by fake runner")
    out = tmp_path / "out"
    calls = []
    actual_fake = grid.fake_runner

    def fake(argv, log):
        calls.append(argv)
        actual_fake(argv, log)

    monkeypatch.setattr(grid, "fake_runner", fake)
    monkeypatch.setattr(grid, "run_runner", lambda *a: pytest.fail("Real runner called in dry-run"))
    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_MODEL_DIR", "~/weights")
    assert grid.main(["--pdfs", str(pdfs), "--out", str(out), "--configs", "all", "--dry-run"]) == 0
    assert len(calls) == 8
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 16 and all(r["cer"] == 0 for r in rows)
    assert all(r["output_tokens"] == 100 and r["prompt_tokens"] == 1132 for r in rows)
    assert {r["page_number"] for r in rows} == {1, 2}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["synthetic"] is True and summary["model_load_excluded"] is True
    for name, config in summary["configs"].items():
        assert config["passes_gate"] is (name != "baseline")
        assert config["accepted"] is False
        assert config["median_prompt_tokens_per_page"] == 1132
        assert config["median_output_tokens_per_page"] == 100
        assert config["pages_reporting_prompt_tokens"] == 2
        assert config["median_coverage"] == 1
        assert config["median_seconds_per_page_per_gpu"] == (1 if name == "baseline" else 0.6)
    receipt = json.loads((out / "config.json").read_text())
    assert receipt["model_dir"] == str(Path("~/weights").expanduser())


def test_module_cli_rehearsal_and_max_pages(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.touch()
    out = tmp_path / "out"
    argv = [sys.executable, "-m", "glossapi.scripts.deepseek_resolution_grid",
            "--pdfs", str(pdf), "--out", str(out), "--dry-run",
            "--configs", "norepair", "--max-pages", "1"]
    result = subprocess.run(argv, check=True, capture_output=True, text=True, env=os.environ.copy())
    assert "Wrote 2 page rows" in result.stdout
    rows = (out / "results.jsonl").read_text().splitlines()
    assert len(rows) == 2


def test_real_dispatch_one_process_per_config_failure_reported(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.touch()
    env = tmp_path / "env.sh"
    env.touch()
    calls = []

    def runner(argv, log):
        calls.append(argv)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(7, argv)
        grid.fake_runner(argv[6:], log)  # unwrap bash env-file prefix

    monkeypatch.setattr(grid, "run_runner", runner)
    out = tmp_path / "out"
    rc = grid.main(["--pdfs", str(pdf), "--out", str(out), "--model-dir", "/model",
                    "--configs", "tokens_1024", "mem_0.95", "--env-file", str(env)])
    assert rc == 1 and len(calls) == 3
    assert all(c[0] == "bash" for c in calls)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["configs"]["tokens_1024"]["status"] == "failed"
    assert summary["configs"]["tokens_1024"]["passes_gate"] is False
    assert summary["configs"]["mem_0.95"]["status"] == "ok"


@pytest.mark.parametrize("args", [["--batch-size", "0"], ["--max-pages", "0"]])
def test_positive_limits_required(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        grid.main(["--pdfs", str(tmp_path), "--out", str(tmp_path / "out"), "--dry-run", *args])
    assert exc.value.code == 2


def test_existing_output_rejected_before_any_runner(tmp_path, monkeypatch):
    pdf = tmp_path / "sample.pdf"
    pdf.touch()
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("old result", encoding="utf-8")
    monkeypatch.setattr(grid, "fake_runner", lambda *a: pytest.fail("Runner called with stale artifacts"))
    with pytest.raises(SystemExit) as exc:
        grid.main(["--pdfs", str(pdf), "--out", str(out), "--dry-run"])
    assert exc.value.code == 2


# Adapted from the read-only CP2 reviewer fake-engine probes.
class InlineThread:
    """Run the one-page producer synchronously; create no background thread/job."""
    def __init__(self, target, **kwargs):
        self.target = target
    def start(self):
        self.target()
    def is_alive(self):
        return False
    def join(self, **kwargs):
        pass


def runner_capture(monkeypatch, tmp_path, config, *, request_kwargs=True, use_profile_defaults=False,
                   processor_flags=(), honoured_report=None, output_token_ids=(1,) * 10):
    engines, requests = [], []
    class FakeLLM:
        def __init__(self, **kwargs):
            engines.append(kwargs)
            if honoured_report is not None:
                self.mm_processor_kwargs_honoured = honoured_report
        def generate(self, batch, **kwargs):
            requests.append([
                {'prompt': p['prompt'],
                 'image_size': p['multi_modal_data']['image'].size,
                 'image_mode': p['multi_modal_data']['image'].mode,
                 'request_keys': sorted(p),
                 'multimodal_keys': sorted(p['multi_modal_data']),
                 'mm_processor_kwargs': p.get('mm_processor_kwargs')}  for p in batch
            ])
            return [SimpleNamespace(outputs=[SimpleNamespace(text='αβγ δεζ ' * 40, token_ids=output_token_ids)], prompt_token_ids=[2] * 23)
                    for _ in batch]
    fake = ModuleType('vllm')
    fake.__path__ = []
    fake.LLM = FakeLLM
    fake.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    inputs = ModuleType('vllm.inputs')
    inputs.TextPrompt = type('TextPrompt', (), {'__annotations__': {'mm_processor_kwargs': dict} if request_kwargs else {}})
    monkeypatch.setitem(sys.modules, 'vllm.inputs', inputs)
    monkeypatch.setitem(sys.modules, 'vllm', fake)
    # Prevent any real model library from loading during the fake engine probe.
    monkeypatch.setitem(sys.modules, 'transformers', None)
    monkeypatch.setattr(real.threading, 'Thread', InlineThread)
    monkeypatch.setattr(real, '_iter_pdf_jobs', lambda *a: [
        {'pdf_path': str(tmp_path / 'doc.pdf'), 'stem': 'doc', 'source_name': 'doc.pdf',
         'source_stem': 'doc', 'start_page': 1, 'end_page': None}])
    monkeypatch.setattr(real, '_count_rendered_pages', lambda *a, **kw: 1)
    monkeypatch.setattr(real, '_iter_rendered_pages', lambda *a, **kw: iter([Image.new('RGB', (100, 100), 'black')]))
    out = tmp_path / config
    argv = grid.build_runner_argv(config, input_dir=tmp_path, output_dir=out,
                                files=['doc.pdf'], model_dir=Path('/fake-model'), python_bin='/fake-python')
    if use_profile_defaults:
        for flag in ('--base-size', '--image-size'):
            index = argv.index(flag)
            del argv[index:index + 2]
        argv.remove('--crop-mode')
    monkeypatch.setattr(sys, 'argv', [*argv[1:], *processor_flags])
    assert real.main() == 0
    metadata = json.loads((out / 'json/metrics/doc.metrics.json').read_text())
    return {'engines': engines, 'requests': requests, 'metadata': metadata, 'out': out,
            'declared': {k: metadata[k] for k in ['base_size', 'image_size', 'crop_mode']}}


@pytest.mark.parametrize('flags,expected', [
    (('--image-size', '640'), {'base_size': 1024, 'image_size': 640, 'crop_mode': True}),
    (('--base-size', '768', '--image-size', '512'),
     {'base_size': 768, 'image_size': 512, 'crop_mode': True}),
])
def test_nondefault_processor_warning_and_honoured_flag(monkeypatch, tmp_path, caplog, flags, expected):
    with caplog.at_level('WARNING', logger=real.__name__):
        captured = runner_capture(monkeypatch, tmp_path, 'baseline', processor_flags=flags)
    assert "vLLM 0.18.0's DeepSeek-OCR-2 processor ignores" in caplog.text
    assert captured['engines'][0]['mm_processor_kwargs'] == expected
    assert captured['requests'][0][0]['mm_processor_kwargs'] == expected
    assert captured['metadata']['mm_processor_kwargs'] == expected
    assert captured['metadata']['mm_processor_kwargs_honoured'] is False
    assert captured['metadata']['page_metrics'][0]['prompt_tokens'] == 23
    assert captured['metadata']['page_metrics'][0]['output_tokens'] == 10
    runtime = json.loads((captured['out'] / 'runtime_summary.json').read_text())
    assert runtime['steady_state']['first_batch_to_last_batch_window_sec'] > 0


def test_no_crop_refused_before_engine_build(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(real, '_load_vllm', lambda *a, **kw: pytest.fail('Engine built before refusal'))
    argv = grid.build_runner_argv('baseline', input_dir=tmp_path, output_dir=tmp_path/'out',
                                  files=['doc.pdf'], model_dir=Path('/fake'))
    monkeypatch.setattr(sys, 'argv', [*argv[1:], '--no-crop-mode'])
    with pytest.raises(SystemExit) as exc:
        real.main()
    assert exc.value.code == 2
    assert '--no-crop-mode is unsupported by the vLLM DeepSeek-OCR-2 backend' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


def test_default_processor_does_not_claim_override_support(monkeypatch, tmp_path, caplog):
    captured = runner_capture(monkeypatch, tmp_path, 'baseline')
    assert captured['metadata']['mm_processor_kwargs_honoured'] is None
    assert 'processor ignores' not in caplog.text


@pytest.mark.parametrize('reported', [True, False])
def test_explicit_engine_honoured_report_is_preserved(monkeypatch, tmp_path, reported):
    captured = runner_capture(monkeypatch, tmp_path, 'baseline',
                              processor_flags=('--image-size', '640'), honoured_report=reported)
    assert captured['metadata']['mm_processor_kwargs_honoured'] is reported


@pytest.mark.parametrize('ids,expected', [(None, None), ((), 0), ((1, 2, 3), 3)])
def test_runner_output_token_counts_preserve_missing_and_empty(monkeypatch, tmp_path, ids, expected):
    captured = runner_capture(monkeypatch, tmp_path, 'baseline', output_token_ids=ids)
    assert captured['metadata']['page_metrics'][0]['output_tokens'] == expected


def test_runner_records_repair_tokens_in_results_and_summary(monkeypatch, tmp_path):
    postprocess = real._postprocess_page_text
    def force_repair(text, *, prompt, content_debug):
        text, metrics = postprocess(text, prompt=prompt, content_debug=content_debug)
        if prompt == real._profile_defaults('markdown_grounded')['prompt']:
            metrics['early_stops'] = 1
        return text, metrics
    monkeypatch.setattr(real, '_postprocess_page_text', force_repair)
    captured = runner_capture(monkeypatch, tmp_path, 'baseline')
    assert len(captured['requests']) == 2
    page = captured['metadata']['page_metrics'][0]
    assert page['repair_attempted'] is True
    assert page['output_tokens'] == page['repair_output_tokens'] == 10
    assert page['prompt_tokens'] == page['repair_prompt_tokens'] == 23
    out = captured['out']
    rows = grid.parse_document('baseline', out/'json/metrics/doc.metrics.json', out/'markdown/doc.md', 'doc.pdf')
    assert rows[0]['output_tokens'] == 20
    summary = grid.summarize(rows, {'baseline': {'status': 'ok'}}, dry_run=False)['configs']['baseline']
    assert summary['median_output_tokens_per_page'] == 20
    assert summary['median_prompt_tokens_per_page'] == 23
    assert summary['median_repair_prompt_tokens_per_page'] == 23


def test_tokens_1024_reports_truncation_coverage_and_rejects():
    rows = [row('baseline', 'a' * 400, 1, tokens=200),
            row('tokens_1024', 'a' * 200, 0.6, tokens=100)]
    rows[0]['prompt_tokens'] = rows[1]['prompt_tokens'] = 1132
    candidate = grid.summarize(rows, {c: {'status': 'ok'} for c in grid.CONFIGS}, dry_run=False)['configs']['tokens_1024']
    assert rows[1]['coverage'] == candidate['median_coverage'] == 0.5
    assert candidate['median_cer'] == 0.5
    assert candidate['wall_ratio'] == 0.6
    assert candidate['median_prompt_tokens_per_page'] == 1132
    assert candidate['median_output_tokens_per_page'] == 100
    assert candidate['passes_gate'] is candidate['accepted'] is False


def test_cp2_pipeline_wall_gate_rejects_faster_generation(tmp_path):
    rows = []
    for config, infer, wall in [('baseline', 1.0, 1.1), ('mem_0.95', 0.6, 2.0)]:
        out = tmp_path / config
        real._write_outputs(out, 'doc', 'α' * 200, 1, extra_metrics={
            'wall_time_sec': 999.0, 'render_sec': wall-infer,
            'page_metrics': [{'page_number': 1, 'infer_sec': infer}]})
        write_runtime(out, wall)
        rows.extend(grid.parse_document(config, out/'json/metrics/doc.metrics.json', out/'markdown/doc.md', 'doc.pdf'))
    result = grid.summarize(rows, {c: {'status': 'ok'} for c in ['baseline', 'mem_0.95']}, dry_run=False)
    candidate = result['configs']['mem_0.95']
    assert candidate['accepted'] is False
    assert candidate['gen_ratio'] == pytest.approx(0.6)
    assert candidate['wall_ratio'] == pytest.approx(2.0/1.1)


def write_runtime(out, wall):
    from datetime import datetime, timedelta, timezone
    start = datetime(2026, 10, 6, tzinfo=timezone.utc)
    (out / 'runtime_summary.json').write_text(json.dumps({'steady_state': {
        'first_batch_started_at': start.isoformat(),
        'last_batch_finished_at': (start + timedelta(seconds=wall)).isoformat(),
        'first_batch_to_last_batch_window_sec': wall,
    }}))


@pytest.mark.parametrize('pages,max_pages,expected', [(60, None, 60), (200, None, 160), (60, 7, 7)])
def test_cp2_batch_default_matches_production(tmp_path, pages, max_pages, expected):
    from pypdf import PdfWriter
    pdf = tmp_path / 'sample.pdf'
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    writer.write(pdf)
    argv = grid.build_runner_argv('baseline', input_dir=tmp_path, output_dir=tmp_path/'out',
                                  files=[pdf.name], model_dir=Path('/model'), max_pages=max_pages)
    assert argv[argv.index('--batch-size') + 1] == str(expected)


@pytest.mark.parametrize('profile,expected', [
    ('baseline', {'base_size': 1024, 'image_size': 768, 'crop_mode': True}),
    ('plain_ocr', {'base_size': 768, 'image_size': 512, 'crop_mode': True}),
])
def test_runner_explicitly_passes_profile_defaults(monkeypatch, tmp_path, profile, expected):
    capture = runner_capture(monkeypatch, tmp_path, profile, use_profile_defaults=True)
    assert capture['engines'][0]['mm_processor_kwargs'] == expected
    assert capture['requests'][0][0]['mm_processor_kwargs'] == expected
    assert capture['metadata']['mm_processor_kwargs'] == expected
    assert capture['metadata']['mm_processor_kwargs_honoured'] is (False if profile == 'plain_ocr' else None)


def test_old_request_schema_still_gets_engine_kwargs(monkeypatch, tmp_path):
    capture = runner_capture(monkeypatch, tmp_path, 'baseline', request_kwargs=False,
                             processor_flags=('--image-size', '640'))
    assert capture['engines'][0]['mm_processor_kwargs']['image_size'] == 640
    assert capture['requests'][0][0]['mm_processor_kwargs'] is None
    assert capture['metadata']['request_mm_processor_kwargs_passed'] is False


def test_runtime_window_includes_write_tail(monkeypatch, tmp_path):
    clock = {'now': 1_000.0}
    monkeypatch.setattr(real.time, 'time', lambda: clock['now'])
    generate = real._run_vllm_batch
    write = real._write_outputs

    def timed_generate(*args, **kwargs):
        result = generate(*args, **kwargs)
        clock['now'] += 0.6
        return result

    def timed_write(*args, **kwargs):
        result = write(*args, **kwargs)
        clock['now'] += 1.4
        return result

    monkeypatch.setattr(real, '_run_vllm_batch', timed_generate)
    monkeypatch.setattr(real, '_write_outputs', timed_write)
    capture = runner_capture(monkeypatch, tmp_path, 'baseline')
    assert grid.pipeline_wall_seconds(capture['out']) == 2.0
    runtime = json.loads((capture['out'] / 'runtime_summary.json').read_text())['steady_state']
    assert runtime['first_batch_to_last_batch_window_sec'] == 2.0


def test_missing_runtime_never_substitutes_generation_or_document_wall(tmp_path):
    rows = []
    for config, sec in [('baseline', 1.0), ('mem_0.95', 0.6)]:
        out = tmp_path / config
        real._write_outputs(out, 'doc', 'a' * 200, 1, extra_metrics={
            'wall_time_sec': sec, 'page_metrics': [{'page_number': 1, 'infer_sec': sec}]})
        rows.extend(grid.parse_document(config, out/'json/metrics/doc.metrics.json', out/'markdown/doc.md', 'doc.pdf'))
    candidate = grid.summarize(rows, {c: {'status': 'ok'} for c in ['baseline', 'mem_0.95']}, dry_run=False)['configs']['mem_0.95']
    assert candidate['gen_ratio'] == 0.6
    assert candidate['wall_ratio'] is None
    assert candidate['accepted'] is False


def test_overlapping_documents_share_one_config_window(tmp_path):
    rows = []
    for config, wall in [('baseline', 10.0), ('mem_0.95', 7.0)]:
        out = tmp_path / config
        for stem, page_count in [('one', 1), ('two', 2)]:
            real._write_outputs(out, stem, grid._join_page_outputs(['a'*200]*page_count), page_count,
                                extra_metrics={'wall_time_sec': 999.0,
                                               'page_metrics': [{'page_number': p, 'infer_sec': 1.0}
                                                                for p in range(1, page_count+1)]})
        write_runtime(out, wall)
        for stem in ('one', 'two'):
            rows.extend(grid.parse_document(config, out/'json/metrics'/f'{stem}.metrics.json',
                                            out/'markdown'/f'{stem}.md', f'{stem}.pdf'))
    candidate = grid.summarize(rows, {c: {'status': 'ok'} for c in ['baseline', 'mem_0.95']}, dry_run=False)['configs']['mem_0.95']
    assert candidate['pipeline_wall_seconds'] == 7.0
    assert candidate['wall_seconds_per_page_per_gpu'] == pytest.approx(7.0/3)
    assert candidate['wall_ratio'] == 0.7 and candidate['gen_ratio'] == 1.0
    assert candidate['accepted'] is True


def test_missing_runtime_fails_config_in_cli(tmp_path, monkeypatch):
    pdf = tmp_path / 'sample.pdf'
    pdf.touch()
    def runner(argv, log):
        grid.fake_runner(argv, log)
        Path(argv[argv.index('--output-dir') + 1], 'runtime_summary.json').unlink()
    monkeypatch.setattr(grid, 'run_runner', runner)
    out = tmp_path / 'out'
    assert grid.main(['--pdfs', str(pdf), '--out', str(out), '--configs', 'tokens_1024',
                      '--model-dir', '/fake', '--batch-size', '8']) == 1
    summary = json.loads((out/'summary.json').read_text())
    assert all(c['status'] == 'failed' and c['accepted'] is False for c in summary['configs'].values())


@pytest.mark.parametrize('damage', ['backwards', 'naive_timestamps', 'null_start'])
def test_invalid_runtime_window_rejected(tmp_path, damage):
    write_runtime(tmp_path, 2.0)
    path = tmp_path / 'runtime_summary.json'
    runtime = json.loads(path.read_text())
    steady = runtime['steady_state']
    if damage == 'backwards':
        steady['first_batch_started_at'], steady['last_batch_finished_at'] = steady['last_batch_finished_at'], steady['first_batch_started_at']
    elif damage == 'naive_timestamps':
        steady['first_batch_started_at'] = '2026-10-06T00:00:00'
        steady['last_batch_finished_at'] = '2026-10-06T00:00:02'
    else:
        steady['first_batch_started_at'] = None
    path.write_text(json.dumps(runtime))
    with pytest.raises(ValueError):
        grid.pipeline_wall_seconds(tmp_path)
