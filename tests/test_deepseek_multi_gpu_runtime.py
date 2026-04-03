import json
from pathlib import Path
from types import SimpleNamespace


def test_build_env_adds_wheel_managed_cuda_lib_dirs(tmp_path):
    from glossapi.ocr.deepseek import runner

    venv_root = tmp_path / "venv"
    python_bin = venv_root / "bin" / "python"
    python_bin.parent.mkdir(parents=True, exist_ok=True)
    python_bin.write_text("")
    cuda_runtime_lib = venv_root / "lib" / "python3.11" / "site-packages" / "nvidia" / "cuda_runtime" / "lib"
    cublas_lib = venv_root / "lib" / "python3.11" / "site-packages" / "nvidia" / "cublas" / "lib"
    cuda_runtime_lib.mkdir(parents=True, exist_ok=True)
    cublas_lib.mkdir(parents=True, exist_ok=True)

    env = runner._build_env(python_bin=python_bin, visible_device=1, script=None)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    ld_entries = env["LD_LIBRARY_PATH"].split(":")
    assert str(cuda_runtime_lib) in ld_entries
    assert str(cublas_lib) in ld_entries


def test_work_queue_requeues_stale_running_batch(tmp_path):
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(
        db_path,
        batches=[
            {
                "batch_id": 0,
                "pages": 12,
                "files": ["a.pdf"],
                "page_ranges": [],
                "items": [],
            }
        ],
    )

    claimed = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-a",
        stale_after_sec=30.0,
        now_ts=100.0,
    )

    assert claimed["batch_id"] == 0
    assert work_queue.work_queue_counts(db_path)["running"] == 1

    requeued = work_queue.requeue_stale_running_batches(
        db_path,
        stale_after_sec=30.0,
        now_ts=200.0,
    )

    assert requeued == 1
    assert work_queue.work_queue_counts(db_path)["pending"] == 1


def test_work_queue_mark_done_persists_result(tmp_path):
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(
        db_path,
        batches=[
            {
                "batch_id": 1,
                "pages": 8,
                "files": [],
                "page_ranges": ["b.pdf:1:8"],
                "items": [],
            }
        ],
    )

    work_queue.claim_next_batch(
        db_path,
        worker_id="worker-b",
        stale_after_sec=60.0,
        now_ts=50.0,
    )
    work_queue.mark_batch_done(
        db_path,
        batch_id=1,
        worker_id="worker-b",
        result={"pages": 8, "first_infer_started_at": "2026-04-02T10:00:00Z"},
        now_ts=75.0,
    )

    items = list(work_queue.iter_work_items(db_path))

    assert items[0]["status"] == work_queue.STATUS_DONE
    assert items[0]["result"]["pages"] == 8
    assert work_queue.work_queue_counts(db_path)["done"] == 1


def test_work_queue_repair_enqueue_reuses_queue_key(tmp_path):
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(db_path, batches=[])

    inserted = work_queue.enqueue_batches(
        db_path,
        queue_name=work_queue.QUEUE_REPAIR,
        batches=[
            {
                "queue_key": "repair:5:doc",
                "stem": "doc",
                "repair_page_numbers": [2, 5],
                "pages": 2,
            }
        ],
    )
    claimed = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-r",
        stale_after_sec=60.0,
        queue_name=work_queue.QUEUE_REPAIR,
        now_ts=10.0,
    )
    work_queue.mark_batch_done(
        db_path,
        batch_id=claimed["batch_id"],
        worker_id="worker-r",
        result={"pages": 2},
        now_ts=12.0,
    )

    inserted_again = work_queue.enqueue_batches(
        db_path,
        queue_name=work_queue.QUEUE_REPAIR,
        batches=[
            {
                "queue_key": "repair:5:doc",
                "stem": "doc",
                "repair_page_numbers": [2],
                "pages": 1,
            }
        ],
    )
    repair_item = [
        item
        for item in work_queue.iter_work_items(db_path)
        if item["queue_name"] == work_queue.QUEUE_REPAIR
    ][0]

    assert inserted_again == inserted
    assert repair_item["batch_id"] == inserted[0]
    assert repair_item["status"] == work_queue.STATUS_PENDING
    assert repair_item["repair_page_numbers"] == [2]


def test_work_queue_marks_batch_failed_after_one_retry(tmp_path):
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(
        db_path,
        batches=[
            {
                "batch_id": 2,
                "pages": 4,
                "files": ["c.pdf"],
                "page_ranges": [],
                "items": [],
            }
        ],
    )

    first = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-a",
        stale_after_sec=60.0,
        now_ts=10.0,
    )
    work_queue.mark_batch_failed(
        db_path,
        batch_id=first["batch_id"],
        worker_id="worker-a",
        error="first failure",
        max_attempts=2,
        now_ts=20.0,
    )

    second = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-b",
        stale_after_sec=60.0,
        now_ts=30.0,
    )
    work_queue.mark_batch_failed(
        db_path,
        batch_id=second["batch_id"],
        worker_id="worker-b",
        error="second failure",
        max_attempts=2,
        now_ts=40.0,
    )

    item = list(work_queue.iter_work_items(db_path))[0]

    assert item["attempt_count"] == 2
    assert item["status"] == work_queue.STATUS_FAILED
    assert item["worker_id"] == "worker-b"
    assert item["last_error"] == "second failure"


def test_claim_additional_repair_batches_packs_multiple_items(tmp_path):
    from glossapi.ocr.deepseek import run_pdf_ocr_vllm
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(db_path, batches=[])
    inserted = work_queue.enqueue_batches(
        db_path,
        queue_name=work_queue.QUEUE_REPAIR,
        batches=[
            {"queue_key": "repair:1:a", "batch_id": 10, "stem": "a", "repair_page_numbers": [1, 2], "pages": 2},
            {"queue_key": "repair:1:b", "batch_id": 11, "stem": "b", "repair_page_numbers": [3, 4], "pages": 2},
            {"queue_key": "repair:1:c", "batch_id": 12, "stem": "c", "repair_page_numbers": [5], "pages": 1},
        ],
    )
    assert inserted == [10, 11, 12]

    first = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-pack",
        stale_after_sec=60.0,
        queue_name=work_queue.QUEUE_REPAIR,
        now_ts=10.0,
    )
    packed = run_pdf_ocr_vllm._claim_additional_repair_batches(
        db_path,
        worker_id="worker-pack",
        stale_after_sec=60.0,
        first_batch=first,
        target_pages=4,
        target_items=8,
    )

    assert [int(batch["batch_id"]) for batch in packed] == [10, 11]
    counts = work_queue.work_queue_counts(db_path)
    assert counts["by_queue"][work_queue.QUEUE_REPAIR][work_queue.STATUS_RUNNING] == 2
    assert counts["by_queue"][work_queue.QUEUE_REPAIR][work_queue.STATUS_PENDING] == 1


def test_claim_next_phase_batch_switches_to_repair_after_main_drains(tmp_path):
    from glossapi.ocr.deepseek import run_pdf_ocr_vllm
    from glossapi.ocr.deepseek import work_queue

    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(
        db_path,
        batches=[
            {
                "batch_id": 0,
                "pages": 8,
                "files": ["a.pdf"],
                "page_ranges": [],
                "items": [],
            }
        ],
    )
    claimed = work_queue.claim_next_batch(
        db_path,
        worker_id="worker-main",
        stale_after_sec=60.0,
        now_ts=10.0,
    )
    work_queue.mark_batch_done(
        db_path,
        batch_id=claimed["batch_id"],
        worker_id="worker-main",
        result={"pages": 8},
        now_ts=20.0,
    )
    work_queue.enqueue_batches(
        db_path,
        queue_name=work_queue.QUEUE_REPAIR,
        batches=[
            {
                "queue_key": "repair:0:doc",
                "stem": "doc",
                "repair_page_numbers": [2, 5],
                "pages": 2,
            }
        ],
    )

    queue_name, batch, should_wait = run_pdf_ocr_vllm._claim_next_phase_batch(
        db_path,
        worker_id="worker-repair",
        stale_after_sec=60.0,
    )

    assert queue_name == work_queue.QUEUE_REPAIR
    assert batch is not None
    assert batch["queue_key"] == "repair:0:doc"
    assert should_wait is False


def test_runner_runtime_summary_reports_steady_state_windows(tmp_path):
    from glossapi.ocr.deepseek import runner
    from glossapi.ocr.deepseek import work_queue

    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "worker_00.runtime.json").write_text(
        json.dumps(
            {
                "worker_id": "worker_00",
                "engine_ready_at": "2026-04-02T10:00:10Z",
                "first_batch_started_at": "2026-04-02T10:00:20Z",
                "last_batch_finished_at": "2026-04-02T10:05:20Z",
            }
        ),
        encoding="utf-8",
    )
    (runtime_dir / "worker_01.runtime.json").write_text(
        json.dumps(
            {
                "worker_id": "worker_01",
                "engine_ready_at": "2026-04-02T10:00:12Z",
                "first_batch_started_at": "2026-04-02T10:00:24Z",
                "last_batch_finished_at": "2026-04-02T10:04:20Z",
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "work.sqlite"
    work_queue.init_work_db(
        db_path,
        batches=[
            {"batch_id": 0, "pages": 50, "files": ["a.pdf"], "page_ranges": [], "items": []},
            {"batch_id": 1, "pages": 50, "files": ["b.pdf"], "page_ranges": [], "items": []},
        ],
    )
    work_queue.claim_next_batch(db_path, worker_id="worker_00", stale_after_sec=60.0, now_ts=1.0)
    work_queue.mark_batch_done(db_path, batch_id=0, worker_id="worker_00", now_ts=2.0)
    work_queue.claim_next_batch(db_path, worker_id="worker_01", stale_after_sec=60.0, now_ts=3.0)
    work_queue.mark_batch_done(db_path, batch_id=1, worker_id="worker_01", now_ts=4.0)

    summary_path = runner._write_runtime_summary(runtime_dir=runtime_dir, db_path=db_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["queue_counts"]["done"] == 2
    assert summary["steady_state"]["first_batch_started_at"] == "2026-04-02T10:00:20Z"
    assert summary["steady_state"]["all_workers_ready_at"] == "2026-04-02T10:00:12Z"
    assert summary["steady_state"]["last_batch_finished_at"] == "2026-04-02T10:05:20Z"
    assert summary["steady_state"]["first_batch_to_last_batch_window_sec"] == 300.0
    assert summary["steady_state"]["all_workers_ready_to_last_batch_window_sec"] == 308.0
    assert summary["queue_counts"]["by_queue"]["main"]["done"] == 2
    assert summary["queue_counts"]["by_queue"]["repair"]["done"] == 0


def test_runner_preflight_can_ensure_persistence_mode(monkeypatch):
    from glossapi.ocr.deepseek import runner

    responses = [
        [{"index": "0", "persistence_mode": "Disabled"}],
        [{"index": "0", "persistence_mode": "Enabled"}],
    ]

    monkeypatch.setattr(runner, "_query_persistence_mode", lambda *, visible_devices: responses.pop(0))

    calls = {}

    def fake_run(cmd, check, capture_output, text):
        calls["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    status = runner._ensure_gpu_preflight(visible_devices=[0], mode="ensure")

    assert calls["cmd"] == ["sudo", "-n", "nvidia-smi", "-pm", "1"]
    assert status["changed"] is True
    assert status["after"] == [{"index": "0", "persistence_mode": "Enabled"}]


def test_build_cli_command_includes_work_queue_flags(tmp_path):
    from glossapi.ocr.deepseek.runner import _build_cli_command

    cmd = _build_cli_command(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        files=[],
        page_ranges=None,
        model_dir=tmp_path / "model",
        python_bin=Path("/usr/bin/python3"),
        script=tmp_path / "run_vllm.py",
        max_pages=None,
        content_debug=False,
        device="cuda",
        ocr_profile="markdown_grounded",
        prompt_override=None,
        attn_backend="vllm",
        base_size=None,
        image_size=None,
        crop_mode=None,
        render_dpi=144,
        max_new_tokens=2048,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        runtime_backend="vllm",
        vllm_batch_size=96,
        gpu_memory_utilization=0.9,
        disable_fp8_kv=False,
        repair_mode="auto",
        repair_exec_batch_target_pages=48,
        repair_exec_batch_target_items=32,
        work_db=tmp_path / "work.sqlite",
        worker_id="worker_00_gpu0",
        worker_runtime_file=tmp_path / "worker_00.runtime.json",
        work_stale_after_sec=900.0,
        work_heartbeat_sec=10.0,
        work_max_attempts=2,
    )

    assert "--work-db" in cmd
    assert str(tmp_path / "work.sqlite") in cmd
    assert "--worker-id" in cmd and "worker_00_gpu0" in cmd
    assert "--worker-runtime-file" in cmd and str(tmp_path / "worker_00.runtime.json") in cmd
    assert "--work-stale-after-sec" in cmd and "900.0" in cmd
    assert "--work-heartbeat-sec" in cmd and "10.0" in cmd
    assert "--work-max-attempts" in cmd and "2" in cmd
    assert "--repair-exec-batch-target-pages" in cmd and "48" in cmd
    assert "--repair-exec-batch-target-items" in cmd and "32" in cmd


def test_launch_worker_process_uses_start_new_session(monkeypatch):
    from glossapi.ocr.deepseek import runner

    calls = {}

    def fake_popen(cmd, stdout, stderr, env, start_new_session):
        calls["cmd"] = cmd
        calls["start_new_session"] = start_new_session
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    proc = runner._launch_worker_process(["python", "worker.py"], fh=object(), env={"A": "1"})

    assert calls["cmd"] == ["python", "worker.py"]
    assert calls["start_new_session"] is True
    assert proc.pid == 1234


def test_terminate_worker_process_group_signals_group(monkeypatch):
    from glossapi.ocr.deepseek import runner

    signals = []
    monkeypatch.setattr(runner.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    monkeypatch.setattr(runner, "_wait_for_process_group_exit", lambda pgid, *, timeout_sec: True)

    ok = runner._terminate_worker_process_group(
        {
            "worker_id": "worker_00_gpu0",
            "proc": SimpleNamespace(pid=4321),
        }
    )

    assert ok is True
    assert signals == [(4321, runner.signal.SIGTERM)]
