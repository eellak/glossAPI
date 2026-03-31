"""Scheduling helpers for DeepSeek OCR page-range planning.

The core abstraction is a divisible PDF page stream. We can cut a document into
page ranges exactly where a batch boundary needs it, then reconstruct outputs
later by `(doc_id, page_number)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SourceDocument:
    name: str
    pages: int


@dataclass(frozen=True)
class WorkSlice:
    source_name: str
    source_pages: int
    start_page: int
    end_page: int

    @property
    def pages(self) -> int:
        return int(self.end_page) - int(self.start_page) + 1

    @property
    def is_full_document(self) -> bool:
        return int(self.start_page) == 1 and int(self.end_page) == int(self.source_pages)

    @property
    def item_id(self) -> str:
        if self.is_full_document:
            return str(self.source_name)
        return f"{self.source_name}:{int(self.start_page)}:{int(self.end_page)}"

    @property
    def cli_file(self) -> Optional[str]:
        return str(self.source_name) if self.is_full_document else None

    @property
    def cli_page_range(self) -> Optional[str]:
        if self.is_full_document:
            return None
        return self.item_id

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "pages": int(self.pages),
            "file": self.cli_file,
            "page_range": self.cli_page_range,
            "source_name": str(self.source_name),
            "start_page": int(self.start_page),
            "end_page": int(self.end_page),
            "is_full_document": bool(self.is_full_document),
        }


@dataclass
class DocumentCursor:
    name: str
    total_pages: int
    next_page: int = 1

    @property
    def remaining_pages(self) -> int:
        return max(0, int(self.total_pages) - int(self.next_page) + 1)

    def take(self, requested_pages: int) -> WorkSlice:
        take_pages = min(max(1, int(requested_pages)), int(self.remaining_pages))
        start_page = int(self.next_page)
        end_page = start_page + take_pages - 1
        self.next_page = end_page + 1
        return WorkSlice(
            source_name=str(self.name),
            source_pages=int(self.total_pages),
            start_page=int(start_page),
            end_page=int(end_page),
        )


@dataclass
class BatchPlan:
    batch_id: int
    items: List[WorkSlice] = field(default_factory=list)

    @property
    def pages(self) -> int:
        return sum(int(item.pages) for item in self.items)

    def to_dict(self) -> dict:
        return {
            "batch_id": int(self.batch_id),
            "item_ids": [item.item_id for item in self.items],
            "files": [item.cli_file for item in self.items if item.cli_file],
            "page_ranges": [item.cli_page_range for item in self.items if item.cli_page_range],
            "pages": int(self.pages),
            "items": [item.to_dict() for item in self.items],
        }


@dataclass
class LanePlan:
    lane_id: int
    visible_device: int
    batches: List[BatchPlan] = field(default_factory=list)

    @property
    def assigned_pages(self) -> int:
        return sum(int(batch.pages) for batch in self.batches)

    def to_dict(self) -> dict:
        return {
            "lane_id": int(self.lane_id),
            "visible_device": int(self.visible_device),
            "assigned_pages": int(self.assigned_pages),
            "batches": [batch.to_dict() for batch in self.batches],
        }


def build_whole_document_slices(documents: Iterable[SourceDocument]) -> List[WorkSlice]:
    return [
        WorkSlice(
            source_name=str(doc.name),
            source_pages=int(doc.pages),
            start_page=1,
            end_page=int(doc.pages),
        )
        for doc in documents
    ]


def build_fixed_shard_slices(
    documents: Iterable[SourceDocument],
    *,
    shard_pages: int,
    shard_threshold_pages: int,
) -> List[WorkSlice]:
    shard_size = max(0, int(shard_pages))
    threshold = max(0, int(shard_threshold_pages))
    slices: List[WorkSlice] = []
    for doc in documents:
        total_pages = int(doc.pages)
        if shard_size <= 0 or total_pages <= max(threshold, shard_size):
            slices.extend(build_whole_document_slices([doc]))
            continue
        start_page = 1
        while start_page <= total_pages:
            end_page = min(total_pages, start_page + shard_size - 1)
            slices.append(
                WorkSlice(
                    source_name=str(doc.name),
                    source_pages=total_pages,
                    start_page=int(start_page),
                    end_page=int(end_page),
                )
            )
            start_page = end_page + 1
    return slices


def build_exact_fill_batches(
    documents: Iterable[SourceDocument],
    *,
    target_batch_pages: int,
) -> List[BatchPlan]:
    target = max(1, int(target_batch_pages))
    heap: List[tuple[int, int, DocumentCursor]] = []
    for idx, doc in enumerate(documents):
        cursor = DocumentCursor(name=str(doc.name), total_pages=int(doc.pages))
        if cursor.remaining_pages > 0:
            heapq.heappush(heap, (-int(cursor.remaining_pages), idx, cursor))

    batches: List[BatchPlan] = []
    while heap:
        remaining_capacity = int(target)
        items: List[WorkSlice] = []
        while remaining_capacity > 0 and heap:
            _neg_remaining, idx, cursor = heapq.heappop(heap)
            take_pages = min(int(cursor.remaining_pages), int(remaining_capacity))
            items.append(cursor.take(take_pages))
            remaining_capacity -= int(take_pages)
            if cursor.remaining_pages > 0:
                heapq.heappush(heap, (-int(cursor.remaining_pages), idx, cursor))
        batches.append(BatchPlan(batch_id=len(batches), items=items))
    return batches


def pack_slices_into_batches(
    slices: Iterable[WorkSlice],
    *,
    target_batch_pages: int,
) -> List[BatchPlan]:
    target = max(1, int(target_batch_pages))
    ordered = sorted(list(slices), key=lambda item: (-int(item.pages), item.item_id))
    batches: List[BatchPlan] = []
    current: List[WorkSlice] = []
    current_pages = 0

    def flush() -> None:
        nonlocal current, current_pages
        if not current:
            return
        batches.append(BatchPlan(batch_id=len(batches), items=list(current)))
        current = []
        current_pages = 0

    for item in ordered:
        item_pages = int(item.pages)
        if current and current_pages + item_pages > target:
            flush()
        current.append(item)
        current_pages += item_pages
        if current_pages >= target:
            flush()
    flush()
    return batches


def assign_batches_to_lanes(
    batches: Iterable[BatchPlan],
    *,
    devices: List[int],
    workers_per_gpu: int,
) -> List[LanePlan]:
    lanes: List[LanePlan] = []
    lane_id = 0
    for visible_device in devices:
        for _ in range(max(1, int(workers_per_gpu))):
            lanes.append(LanePlan(lane_id=lane_id, visible_device=int(visible_device)))
            lane_id += 1
    for batch in batches:
        lane = min(lanes, key=lambda item: (int(item.assigned_pages), int(item.lane_id)))
        lane.batches.append(batch)
    return lanes

