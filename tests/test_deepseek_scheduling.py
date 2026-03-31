from pathlib import Path


def _touch_files(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"%PDF-1.4\n%stub\n")


def test_plan_lanes_balances_weighted_docs_greedily(monkeypatch, tmp_path):
    from glossapi.ocr.deepseek import runner

    weights = {
        "huge.pdf": 500,
        "mid_a.pdf": 300,
        "mid_b.pdf": 300,
        "small_a.pdf": 200,
        "tiny_a.pdf": 100,
        "tiny_b.pdf": 100,
    }
    _touch_files(tmp_path, list(weights))

    monkeypatch.setattr(runner, "_page_count", lambda path: weights[path.name])
    lanes = runner._plan_lanes(
        file_list=["tiny_b.pdf", "mid_a.pdf", "huge.pdf", "small_a.pdf", "tiny_a.pdf", "mid_b.pdf"],
        input_root=tmp_path,
        lane_devices=[0, 1, 2],
        workers_per_gpu=1,
        max_pages=None,
    )

    assert [int(lane["weight"]) for lane in lanes] == [500, 500, 500]
    assigned = [name for lane in lanes for name in lane["files"]]
    assert sorted(assigned) == sorted(weights)
    assert len(assigned) == len(set(assigned))


def test_auto_vllm_batch_size_caps_total_pages(monkeypatch, tmp_path):
    from glossapi.ocr.deepseek import runner

    weights = {
        "a.pdf": 90,
        "b.pdf": 120,
        "c.pdf": 400,
    }
    _touch_files(tmp_path, list(weights))
    monkeypatch.setattr(runner, "_page_count", lambda path: weights[path.name])

    capped = runner._auto_vllm_batch_size(
        runtime_backend="vllm",
        file_list=list(weights),
        input_root=tmp_path,
        max_pages=None,
    )
    reduced = runner._auto_vllm_batch_size(
        runtime_backend="vllm",
        file_list=list(weights),
        input_root=tmp_path,
        max_pages=20,
    )

    assert capped == 160
    assert reduced == 60


def test_auto_scheduler_prefers_exact_fill_for_multi_gpu_vllm():
    from glossapi.ocr.deepseek import runner

    assert runner._resolve_scheduler(
        scheduler="auto",
        runtime_backend="vllm",
        lane_devices=[0, 1],
        workers_per_gpu=1,
    ) == "exact_fill"
    assert runner._resolve_scheduler(
        scheduler="auto",
        runtime_backend="transformers",
        lane_devices=[0, 1],
        workers_per_gpu=1,
    ) == "whole_doc"


def test_fixed_shard_builder_only_splits_large_docs():
    from glossapi.ocr.deepseek.scheduling import SourceDocument, build_fixed_shard_slices

    documents = [
        SourceDocument(name="huge.pdf", pages=310),
        SourceDocument(name="mid.pdf", pages=120),
        SourceDocument(name="small.pdf", pages=40),
    ]

    slices = build_fixed_shard_slices(documents, shard_pages=128, shard_threshold_pages=200)

    assert [item.item_id for item in slices] == [
        "huge.pdf:1:128",
        "huge.pdf:129:256",
        "huge.pdf:257:310",
        "mid.pdf",
        "small.pdf",
    ]


def test_exact_fill_batches_split_documents_to_fill_target():
    from glossapi.ocr.deepseek.scheduling import SourceDocument, build_exact_fill_batches

    documents = [
        SourceDocument(name="a.pdf", pages=200),
        SourceDocument(name="b.pdf", pages=60),
        SourceDocument(name="c.pdf", pages=60),
        SourceDocument(name="d.pdf", pages=20),
    ]

    batches = build_exact_fill_batches(documents, target_batch_pages=160)

    assert [batch.pages for batch in batches] == [160, 160, 20]
    assert [item.item_id for item in batches[0].items] == ["a.pdf:1:160"]
    assert set(item.item_id for item in batches[1].items) == {"a.pdf:161:200", "b.pdf", "c.pdf"}
    assert [item.item_id for item in batches[2].items] == ["d.pdf"]


def test_assign_batches_to_lanes_balances_full_batches():
    from glossapi.ocr.deepseek.scheduling import (
        BatchPlan,
        WorkSlice,
        assign_batches_to_lanes,
    )

    batches = [
        BatchPlan(batch_id=0, items=[WorkSlice("a.pdf", 160, 1, 160)]),
        BatchPlan(batch_id=1, items=[WorkSlice("b.pdf", 160, 1, 160)]),
        BatchPlan(batch_id=2, items=[WorkSlice("c.pdf", 160, 1, 160)]),
        BatchPlan(batch_id=3, items=[WorkSlice("d.pdf", 20, 1, 20)]),
    ]

    lanes = assign_batches_to_lanes(batches, devices=[0, 1], workers_per_gpu=1)

    assert sorted(lane.assigned_pages for lane in lanes) == [180, 320]
    assert [len(lane.batches) for lane in lanes] == [2, 2]


def test_benchmark_planner_exact_fill_mixes_ranges_and_whole_docs():
    from glossapi.ocr.deepseek.scheduling import SourceDocument
    from glossapi.scripts.deepseek_pipeline_benchmark import _plan_lanes

    lanes = _plan_lanes(
        documents=[
            SourceDocument(name="monster.pdf", pages=200),
            SourceDocument(name="tiny.pdf", pages=20),
            SourceDocument(name="mid.pdf", pages=60),
            SourceDocument(name="mid2.pdf", pages=60),
        ],
        devices=[0, 1],
        workers_per_gpu=1,
        scheduler="exact_fill",
        target_batch_pages=160,
        shard_pages=0,
        shard_threshold_pages=0,
    )

    all_ranges = [
        spec
        for lane in lanes
        for batch in lane["batches"]
        for spec in batch.get("page_ranges", [])
    ]
    all_files = [
        name
        for lane in lanes
        for batch in lane["batches"]
        for name in batch.get("files", [])
    ]
    assert "monster.pdf:1:160" in all_ranges
    assert "monster.pdf:161:200" in all_ranges
    assert sorted(all_files) == ["mid.pdf", "mid2.pdf", "tiny.pdf"]


def test_benchmark_planner_whole_doc_preserves_whole_files():
    from glossapi.ocr.deepseek.scheduling import SourceDocument
    from glossapi.scripts.deepseek_pipeline_benchmark import _plan_lanes

    lanes = _plan_lanes(
        documents=[
            SourceDocument(name="monster.pdf", pages=1085),
            SourceDocument(name="a.pdf", pages=200),
            SourceDocument(name="b.pdf", pages=200),
        ],
        devices=[0, 1],
        workers_per_gpu=1,
        scheduler="whole_doc",
        target_batch_pages=160,
        shard_pages=0,
        shard_threshold_pages=0,
    )

    assigned = [name for lane in lanes for batch in lane["batches"] for name in batch["files"]]
    assert sorted(assigned) == ["a.pdf", "b.pdf", "monster.pdf"]


def test_runner_lane_batches_exact_fill_split_large_docs(monkeypatch, tmp_path):
    from glossapi.ocr.deepseek import runner

    weights = {
        "monster.pdf": 200,
        "mid.pdf": 60,
        "mid2.pdf": 60,
        "tiny.pdf": 20,
    }
    _touch_files(tmp_path, list(weights))
    monkeypatch.setattr(runner, "_page_count", lambda path: weights[path.name])

    lanes = runner._plan_lane_batches(
        file_list=list(weights),
        input_root=tmp_path,
        lane_devices=[0, 1],
        workers_per_gpu=1,
        max_pages=None,
        runtime_backend="vllm",
        scheduler="exact_fill",
        target_batch_pages=160,
        shard_pages=0,
        shard_threshold_pages=0,
    )

    all_ranges = [
        spec
        for lane in lanes
        for batch in lane["batches"]
        for spec in batch.get("page_ranges", [])
    ]
    all_files = [
        name
        for lane in lanes
        for batch in lane["batches"]
        for name in batch.get("files", [])
    ]
    assert "monster.pdf:1:160" in all_ranges
    assert "monster.pdf:161:200" in all_ranges
    assert sorted(all_files) == ["mid.pdf", "mid2.pdf", "tiny.pdf"]
