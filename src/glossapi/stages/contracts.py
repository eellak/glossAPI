"""Version-one, pickleable worker protocol. This module imports only stdlib."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class PackageRequirement:
    package_group: str
    distributions: tuple[str, ...]
    entry_point_group: str | None = None
    entry_point_name: str | None = None


@dataclass(frozen=True)
class ArtifactReference:
    file_id: str
    kind: str
    path: Path
    sha256: str
    bytes: int
    page_range: tuple[int, int] | None = None


@dataclass(frozen=True)
class DocumentSnapshot:
    document_id: str
    version_id: str | None
    source_uri: str
    source_records: tuple[Mapping[str, JsonValue], ...] = ()
    artifacts: tuple[ArtifactReference, ...] = ()


@dataclass(frozen=True)
class StageSpecification:
    operation: str
    input_roles: tuple[str, ...] = ()
    required_output_kinds: tuple[str, ...] = ()
    device_id: str = "cpu"
    settings: Mapping[str, JsonValue] = field(default_factory=dict)
    compatibility_profile: str = "sequential"


@dataclass(frozen=True)
class PlannedOperation:
    document_id: str
    version_id: str | None
    specification: StageSpecification
    dependencies: tuple[str, ...] = ()
    conditional_on: str | None = None
    skip_reason: str | None = None
    estimated_bytes: int | None = None


@dataclass(frozen=True)
class StagePlan:
    operations: tuple[PlannedOperation, ...]
    barriers: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageError:
    error_type: str
    message: str
    traceback: str | None
    retryable: bool


@dataclass(frozen=True)
class StageRequest:
    task_id: str
    attempt_id: str
    lease_token: str
    document_id: str | None
    version_id: str | None
    operation: str
    inputs: tuple[ArtifactReference, ...]
    settings: Mapping[str, JsonValue]
    tools: Mapping[str, JsonValue]
    device_id: str
    work_directory: Path
    deadline_monotonic_ns: int


@dataclass(frozen=True)
class OutputArtifact:
    kind: str
    relative_path: str
    page_range: tuple[int, int] | None
    retention_class: str
    required: bool


@dataclass(frozen=True)
class StageResult:
    task_id: str
    attempt_id: str
    lease_token: str
    status: Literal["ok", "skipped", "failed"]
    reason_code: str
    outputs: tuple[OutputArtifact, ...]
    decisions: tuple[Mapping[str, JsonValue], ...]
    metrics: Mapping[str, JsonValue]
    error: StageError | None = None


class WorkerReporter(Protocol):
    def heartbeat(self, lease_token: str, progress: Mapping[str, JsonValue]) -> None: ...
    def evidence(self, lease_token: str, receipt: Mapping[str, JsonValue]) -> None: ...


class Stage(Protocol):
    name: str
    contract_version: int
    def packages(self, specification: StageSpecification) -> tuple[PackageRequirement, ...]: ...
    def plan(self, documents: Sequence[DocumentSnapshot], *, settings: Mapping[str, JsonValue]) -> StagePlan: ...
    def run_document(self, request: StageRequest, *, reporter: WorkerReporter) -> StageResult: ...
    def finalize(self, request: StageRequest, *, reporter: WorkerReporter) -> StageResult: ...
