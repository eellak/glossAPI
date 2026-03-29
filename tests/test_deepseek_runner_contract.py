import sys
from pathlib import Path

import pandas as pd
import pytest


def _mk_corpus(tmp_path: Path):
    from glossapi import Corpus

    root = tmp_path / "corpus"
    root.mkdir()
    return Corpus(input_dir=root, output_dir=root)


def test_deepseek_backend_rejects_stub_mode(tmp_path, monkeypatch):
    corpus = _mk_corpus(tmp_path)

    dl_dir = corpus.output_dir / "download_results"
    dl_dir.mkdir(parents=True, exist_ok=True)
    fname = "doc.pdf"
    df = pd.DataFrame(
        [{"filename": fname, corpus.url_column: "", "needs_ocr": True, "ocr_success": False}]
    )
    parquet_path = dl_dir / "download_results.parquet"
    df.to_parquet(parquet_path, index=False)
    (corpus.input_dir / fname).write_bytes(b"%PDF-1.4\n%real\n")

    monkeypatch.setenv("GLOSSAPI_DEEPSEEK_ALLOW_STUB", "1")

    with pytest.raises(RuntimeError, match="stub execution has been removed"):
        corpus.ocr(backend="deepseek", fix_bad=True, math_enhance=False)

    updated = pd.read_parquet(parquet_path).set_index("filename")
    assert bool(updated.loc[fname, "ocr_success"]) is False
    assert bool(updated.loc[fname, "needs_ocr"]) is True


def test_progress_artifacts_stay_out_of_canonical_markdown(tmp_path):
    from glossapi.ocr.deepseek.run_pdf_ocr_transformers import _write_outputs, _write_progress

    output_dir = tmp_path / "output"
    _write_progress(
        output_dir=output_dir,
        stem="doc",
        page_outputs=["page one"],
        total_pages=5,
        completed_pages=1,
    )

    canonical_markdown = output_dir / "markdown" / "doc.md"
    progress_markdown = output_dir / "sidecars" / "ocr_progress" / "doc.partial.md"
    progress_json = output_dir / "json" / "metrics" / "doc.progress.json"

    assert not canonical_markdown.exists()
    assert progress_markdown.exists()
    assert progress_json.exists()

    _write_outputs(output_dir=output_dir, stem="doc", markdown="final", page_count=5)

    assert canonical_markdown.exists()
    assert canonical_markdown.read_text(encoding="utf-8") == "final\n"
    assert not progress_markdown.exists()


def test_deepseek_runner_multi_uses_visible_device_isolation(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import runner

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    files = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
    weights = {"a.pdf": 40, "b.pdf": 30, "c.pdf": 20, "d.pdf": 10}
    for name in files:
        (input_dir / name).write_bytes(b"%PDF-1.4\n%stub\n")

    class DummyCorpus:
        def __init__(self, input_dir: Path, output_dir: Path):
            self.input_dir = input_dir
            self.output_dir = output_dir

    class FakePopen:
        calls = []

        def __init__(self, cmd, stdout=None, stderr=None, env=None):
            self.cmd = list(cmd)
            self.env = dict(env or {})
            self.returncode = 0
            FakePopen.calls.append(self)

            args = list(cmd)
            out_root = Path(args[args.index("--output-dir") + 1])
            lane_files = []
            idx = args.index("--files") + 1
            while idx < len(args) and not args[idx].startswith("--"):
                lane_files.append(args[idx])
                idx += 1
            md_dir = out_root / "markdown"
            metrics_dir = out_root / "json" / "metrics"
            md_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            for name in lane_files:
                stem = Path(name).stem
                (md_dir / f"{stem}.md").write_text("ok\n", encoding="utf-8")
                (metrics_dir / f"{stem}.metrics.json").write_text(
                    "{\n  \"page_count\": 1\n}\n",
                    encoding="utf-8",
                )

        def wait(self):
            return self.returncode

    script = tmp_path / "run_pdf_ocr_transformers.py"
    script.write_text("# stub\n", encoding="utf-8")
    model_dir = tmp_path / "DeepSeek-OCR-2"
    model_dir.mkdir()

    monkeypatch.setattr(runner, "_page_count", lambda path: weights[path.name])
    monkeypatch.setattr(runner.subprocess, "Popen", FakePopen)

    results = runner.run_for_files(
        DummyCorpus(input_dir, output_dir),
        files,
        model_dir=model_dir,
        python_bin=Path(sys.executable),
        vllm_script=script,
        use_gpus="multi",
        devices=[2, 5],
        workers_per_gpu=2,
    )

    assert sorted(results) == ["a", "b", "c", "d"]
    assert len(FakePopen.calls) == 4

    seen_files = []
    seen_visible_devices = []
    for call in FakePopen.calls:
        args = call.cmd
        assert "--device" in args
        assert args[args.index("--device") + 1] == "cuda"
        seen_visible_devices.append(call.env.get("CUDA_VISIBLE_DEVICES"))
        idx = args.index("--files") + 1
        while idx < len(args) and not args[idx].startswith("--"):
            seen_files.append(args[idx])
            idx += 1

    assert sorted(seen_files) == sorted(files)
    assert sorted(seen_visible_devices) == ["2", "2", "5", "5"]


def test_deepseek_runner_builds_speed_control_flags(tmp_path):
    from glossapi.ocr.deepseek import runner

    script = tmp_path / "run_pdf_ocr_transformers.py"
    script.write_text("# stub\n", encoding="utf-8")
    model_dir = tmp_path / "DeepSeek-OCR-2"
    model_dir.mkdir()

    cmd = runner._build_cli_command(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        files=["doc.pdf"],
        model_dir=model_dir,
        python_bin=Path(sys.executable),
        script=script,
        max_pages=3,
        content_debug=False,
        device="cuda",
        ocr_profile="plain_ocr",
        attn_backend="sdpa",
        base_size=640,
        image_size=448,
        crop_mode=False,
        render_dpi=120,
    )

    assert "--ocr-profile" in cmd
    assert cmd[cmd.index("--ocr-profile") + 1] == "plain_ocr"
    assert "--attn-backend" in cmd
    assert cmd[cmd.index("--attn-backend") + 1] == "sdpa"
    assert "--base-size" in cmd
    assert cmd[cmd.index("--base-size") + 1] == "640"
    assert "--image-size" in cmd
    assert cmd[cmd.index("--image-size") + 1] == "448"
    assert "--no-crop-mode" in cmd
    assert "--render-dpi" in cmd
    assert cmd[cmd.index("--render-dpi") + 1] == "120"


def test_deepseek_model_load_falls_back_to_eager_when_sdpa_is_unsupported(tmp_path, monkeypatch):
    from glossapi.ocr.deepseek import run_pdf_ocr_transformers as cli

    class DummyModel:
        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

    monkeypatch.setattr(
        cli.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: "tokenizer",
    )

    calls: list[str] = []

    def fake_from_pretrained(*_args, **kwargs):
        attn = kwargs.get("_attn_implementation")
        calls.append(attn)
        if attn == "sdpa":
            raise ValueError(
                "DeepseekOCR2ForCausalLM does not support an attention implementation through "
                "torch.nn.functional.scaled_dot_product_attention yet."
            )
        return DummyModel()

    monkeypatch.setattr(cli.AutoModel, "from_pretrained", fake_from_pretrained)

    _tokenizer, _model, attn_impl = cli._load_model(tmp_path, "cpu", "auto")

    assert calls == ["sdpa", "eager"]
    assert attn_impl == "eager"
