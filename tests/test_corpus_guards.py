import inspect
import queue
import sys
from types import SimpleNamespace

import pytest

from glossapi.corpus import Corpus


class DummyExtractor:
    """Minimal extractor stub to capture configuration during prime_extractor."""

    def __init__(self, *args, **kwargs):
        self.enable_calls = []
        self.configure_calls = []
        self.ensure_calls = []
        self.last_policy = None
        self.last_max_batch_files = None
        self.last_prefer_safe_backend = None
        self.use_pypdfium_backend = None
        self.export_doc_json = None
        self.emit_formula_index = None

    def enable_accel(self, threads=None, type=None):
        self.enable_calls.append({"threads": threads, "type": type})

    def configure_batch_policy(self, policy, *, max_batch_files=None, prefer_safe_backend=None):
        record = {
            "policy": policy,
            "max_batch_files": max_batch_files,
            "prefer_safe_backend": prefer_safe_backend,
        }
        self.configure_calls.append(record)
        self.last_policy = policy
        self.last_max_batch_files = max_batch_files
        self.last_prefer_safe_backend = prefer_safe_backend
        self.use_pypdfium_backend = prefer_safe_backend

    def ensure_extractor(self, **kwargs):
        self.ensure_calls.append(kwargs)
        return SimpleNamespace()


def make_corpus(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    output_dir.mkdir()
    return Corpus(input_dir=input_dir, output_dir=output_dir)


def set_onnx_providers(monkeypatch, providers):
    stub = SimpleNamespace(get_available_providers=lambda: providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", stub)
    return stub


def set_torch_stub(monkeypatch, *, available: bool, device_count: int):
    cuda_ns = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
    )
    torch_ns = SimpleNamespace(cuda=cuda_ns)
    monkeypatch.setattr("glossapi.corpus._maybe_import_torch", lambda force=False: torch_ns)
    return torch_ns


def test_prime_extractor_requires_cuda_for_ocr(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    corpus.extractor = DummyExtractor()

    set_torch_stub(monkeypatch, available=True, device_count=1)
    set_onnx_providers(monkeypatch, ["CPUExecutionProvider"])

    with pytest.raises(RuntimeError) as exc:
        corpus.prime_extractor(
            input_format="pdf",
            accel_type="CUDA",
            force_ocr=True,
            phase1_backend="docling",
        )

    assert "CUDAExecutionProvider" in str(exc.value)


def test_prime_extractor_requires_cuda_for_docling_backend(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    corpus.extractor = DummyExtractor()

    set_torch_stub(monkeypatch, available=False, device_count=0)
    set_onnx_providers(monkeypatch, ["CUDAExecutionProvider"])

    with pytest.raises(RuntimeError) as exc:
        corpus.prime_extractor(
            input_format="pdf",
            accel_type="CUDA",
            phase1_backend="docling",
        )

    assert "Torch CUDA is not available" in str(exc.value)


def test_prime_extractor_configures_safe_backend_for_text_layer(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    corpus.extractor = DummyExtractor()

    set_torch_stub(monkeypatch, available=True, device_count=1)
    set_onnx_providers(monkeypatch, ["CUDAExecutionProvider"])

    corpus.prime_extractor(
        input_format="pdf",
        accel_type="CPU",
        phase1_backend="safe",
    )

    assert corpus.extractor.last_policy == "safe"
    assert corpus.extractor.last_max_batch_files == 1
    assert corpus.extractor.last_prefer_safe_backend is True
    assert corpus.extractor.ensure_calls[0]["enable_ocr"] is False


def test_prime_extractor_configures_docling_backend_for_ocr(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    corpus.extractor = DummyExtractor()

    set_torch_stub(monkeypatch, available=True, device_count=2)
    set_onnx_providers(monkeypatch, ["CUDAExecutionProvider"])

    corpus.prime_extractor(
        input_format="pdf",
        accel_type="CUDA",
        force_ocr=True,
        phase1_backend="auto",
    )

    assert corpus.extractor.last_policy == "docling"
    assert corpus.extractor.last_max_batch_files == 1
    assert corpus.extractor.last_prefer_safe_backend is False
    ensure_kwargs = corpus.extractor.ensure_calls[0]
    assert ensure_kwargs["enable_ocr"] is True
    assert ensure_kwargs["force_full_page_ocr"] is True


def test_prime_extractor_requires_cuda_for_formula_enrichment(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    corpus.extractor = DummyExtractor()

    set_torch_stub(monkeypatch, available=False, device_count=0)
    set_onnx_providers(monkeypatch, ["CUDAExecutionProvider"])

    with pytest.raises(RuntimeError) as exc:
        corpus.prime_extractor(
            input_format="pdf",
            accel_type="CUDA",
            formula_enrichment=True,
            phase1_backend="auto",
        )

    assert "Torch CUDA is not available" in str(exc.value)


def test_gpu_worker_does_not_requeue_failed_paths():
    from glossapi.corpus import gpu_extract_worker_queue

    source = inspect.getsource(gpu_extract_worker_queue)
    assert "work_q.put(_path)" not in source, "GPU worker should not requeue failed paths"
    assert "_clear_current()" in source, "GPU worker should clear status markers"


def test_gpu_worker_requeues_failed_batch(tmp_path, monkeypatch):
    import glossapi.corpus as corpus_mod

    processed_batches = []

    class FakeCorpus:
        def __init__(self, input_dir, output_dir):
            self.input_dir = input_dir
            self.output_dir = output_dir
            self.extract_calls = 0
            self.extractor = SimpleNamespace(max_batch_files=2)

        def prime_extractor(self, *args, **kwargs):
            return None

        def extract(self, *, file_paths=None, **kwargs):
            paths = list(file_paths or [])
            processed_batches.append(paths)
            self.extract_calls += 1
            if self.extract_calls == 1:
                raise RuntimeError("boom")
            return None

    monkeypatch.setattr(corpus_mod, "Corpus", FakeCorpus)
    monkeypatch.setattr("glossapi.Corpus", FakeCorpus)
    monkeypatch.delenv("GLOSSAPI_WORKER_LOG_DIR", raising=False)

    work_q = queue.Queue()
    work_q.put("doc.pdf")
    result_q = queue.Queue()
    status_map: dict = {}

    with pytest.raises(SystemExit) as exit_info:
        corpus_mod.gpu_extract_worker_queue(
            device_id=0,
            in_dir=str(tmp_path),
            out_dir=str(tmp_path),
            work_q=work_q,
            force=False,
            fe=False,
            ce=False,
            use_cls_w=False,
            skip=False,
            input_fmt="pdf",
            threads=1,
            benchmark=False,
            export_json=False,
            emit_index=False,
            backend="safe",
            result_q=result_q,
            status_map=status_map,
            marker_dir=None,
        )
    assert exit_info.value.code == 1

    events = []
    while True:
        try:
            events.append(result_q.get_nowait())
        except queue.Empty:
            break

    batch_events = [ev for ev in events if ev.get("event") == "batch"]
    assert batch_events, "Expected batch failure event from worker"
    assert batch_events[0]["problematic"] == ["doc.pdf"]

    exit_events = [ev for ev in events if ev.get("event") == "exit"]
    assert exit_events, "Expected exit event from worker"
    assert exit_events[0]["exitcode"] == 1

    assert processed_batches == [["doc.pdf"]]
    assert work_q.empty()
    assert status_map == {}
