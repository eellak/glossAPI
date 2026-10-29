"""Opt-in sequential adapter profile for the existing Corpus serializers.

Until all wrapper products move in step 5, this profile returns the same private
worker outputs through the existing phase controller and its state serializers.
"""
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from .contracts import ArtifactReference, DocumentSnapshot, StageRequest, StageSpecification
from .execution import ReceiptCollector, checked_path, sha256_file
from .extract import ExtractStage


def extract_compatibility_document(extractor, source, output_dir, timeout_dir=None):
    from ..ledger.journal import observe, accept_worker_observations, _active_journal, ResultJournal
    source, output_dir = Path(source).absolute(), Path(output_dir).absolute()
    settings = {"phase1_backend": "safe" if extractor.use_pypdfium_backend else "docling",
        "repair_fonts": extractor.repair_fonts, "export_doc_json": extractor.export_doc_json,
        "emit_formula_index": extractor.emit_formula_index, "benchmark_mode": extractor.benchmark_mode,
        "num_threads": 1, "page_range": getattr(extractor, "page_range", None),
        "long_pdf_page_threshold": extractor.long_pdf_page_threshold, "chunk_size": extractor.chunk_size,
        "max_chunk_timeouts": extractor.max_chunk_timeouts, "per_page_timeout_s": extractor.per_page_timeout_s}
    signature = getattr(extractor, "_last_extractor_cfg", None)
    if signature:
        if signature[1] or signature[-1] is not None:
            raise ValueError("OCR plugin extraction belongs to its separate stage adapter")
        settings.update(images_scale=signature[4], formula_enrichment=signature[5], code_enrichment=signature[6], use_cls=signature[7])
    reference = ArtifactReference(str(uuid.uuid4()), "source_payload", source, sha256_file(source), source.stat().st_size)
    with tempfile.TemporaryDirectory(prefix="glossapi-sequential-") as temporary:
        identifiers = [str(uuid.uuid4()) for _ in range(5)]
        request = StageRequest(*identifiers, "extract.document", (reference,), settings, {}, "cpu", Path(temporary)/"attempt",
            time.monotonic_ns() + int(extractor._calculate_text_timeout(file_path=source)*1e9))
        journal = _active_journal.get()
        if isinstance(journal, ResultJournal):
            from .sequential import SequentialStageRunner
            version = journal.versions[str(source)]
            file_id = journal.writer.connection.execute("SELECT file_id FROM current_file_state WHERE uri=? AND version_id=? AND sha256=? AND presence='present' ORDER BY event_id DESC LIMIT 1",
                (source.as_uri(), version, reference.sha256)).fetchone()[0]
            document_id = journal.writer.connection.execute("SELECT document_id FROM document_versions WHERE version_id=?", (version,)).fetchone()[0]
            snapshot = DocumentSnapshot(document_id, version, source.as_uri(), artifacts=(
                ArtifactReference(file_id, reference.kind, source, reference.sha256, reference.bytes),))
            result, workspace = SequentialStageRunner(journal.writer, journal.corpus.output_dir).execute(
                ExtractStage(), snapshot, StageSpecification("extract.document", ("source_payload",), ("raw_markdown",), settings=settings),
                timeout_seconds=extractor._calculate_text_timeout(file_path=source), run_id=journal.run_id)
        else:
            result = ExtractStage().run_document(request, reporter=ReceiptCollector())
            workspace = request.work_directory
        # Compatibility products use their original path spellings. This only
        # changes location references in the old serializer's chunk receipts.
        for output in result.outputs:
            relative = output.relative_path
            if relative.startswith(("logs/", "inputs/")) or output.kind in ("attempt_diagnostic", "worker_receipt"):
                continue
            path = checked_path(workspace, relative)
            if output.kind == "chunk_manifest":
                import json
                payload = json.loads(path.read_text())
                payload["file"] = str(output_dir.parent/"repaired_pdfs"/source.name) if (workspace/"repaired_pdfs"/source.name).is_file() else str(source)
                serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
            if output.kind == "extract_sidecar":
                import json
                payload = json.loads(path.read_text())
                if payload.get("chunk_manifest_path"):
                    payload["chunk_manifest_path"] = str(output_dir/f"{source.stem}.chunks.json")
                    serialized_payload = json.dumps(payload, ensure_ascii=False)
            destination = output_dir.parent / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if output.kind == "chunk_manifest" or (output.kind == "extract_sidecar" and payload.get("chunk_manifest_path")):
                destination.write_text(serialized_payload)
            else:
                shutil.copy2(path, destination)
        for decision in result.decisions:
            accept_worker_observations(decision.get("backend_observations", ()))
        observe(result.reason_code, path=source, status=result.status, adapter_contract_version=1,
            adapter_metrics=result.metrics, decisions=result.decisions,
            error=None, adapter_error=asdict(result.error) if result.error else None)
        return result.status == "ok"
