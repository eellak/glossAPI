"""Safe/Docling/repair/page-quality adapters; serializers stay in GlossExtract."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import time

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageResult, StageSpecification
from .execution import failed_result, freeze_inputs, outputs_in, run_fenced, sha256_file
from .receipts import capture_backend_observations


def wordlist_identity(path):
    """Identity of the configured Greek word list (GLOSSAPI_GREEK_WORDLIST).

    None when unset; the file's sha256 when it exists; a fixed ``missing:<path as given>`` marker when the setting names
    a file that does not exist. A missing list leaves the dictionary rates null, as before the ledger (K2 native
    coordinator decision), instead of failing the document.
    """
    if not path:
        return None
    return sha256_file(path) if Path(path).is_file() else 'missing:' + path

_worker_extractors = {}


class ExtractStage:
    name = "extract"
    contract_version = 1

    def packages(self, specification):
        requirements = [PackageRequirement("core", ("pypdfium2>=4.30,<5", "ftfy>=6.0.0", "zstandard>=0.22.0"))]
        if specification.settings.get("phase1_backend", "safe") == "docling":
            requirements.append(PackageRequirement("layout", ("docling==2.81.0",)))
        if specification.settings.get("repair_backend") == "python":
            requirements.append(PackageRequirement("font-repair", ("pypdf>=6.0.0", "fonttools>=4.0.0")))
        return tuple(requirements)

    def plan(self, documents, *, settings):
        operation = settings.get("operation", "extract.document")
        required = tuple(settings.get("required_output_kinds", ("page_quality",) if operation == "extract.page_quality" else ("raw_markdown",)))
        return StagePlan(tuple(PlannedOperation(document.document_id, document.version_id,
            StageSpecification(operation, ("source_payload",), required, str(settings.get("device_id", "cpu")), settings),
            skip_reason=None if document.artifacts else "no_matching_inputs") for document in documents))

    def finalize(self, request, *, reporter):
        raise ValueError("Extraction has no aggregate finalize operation")

    def run_document(self, request, *, reporter):
        return run_fenced(self, request, reporter=reporter)

    def _run_document(self, request, *, reporter):
        if len(request.inputs) != 1:
            return failed_result(request, "extract_configuration_invalid", ValueError("Extraction requires one source payload"))
        original = request.inputs[0]
        try:
            request = freeze_inputs(request)
            backend = request.settings.get("phase1_backend", "safe")
            if backend not in ("safe", "docling"):
                raise ValueError("Adapter requires an explicitly resolved backend")
            if backend == "safe" and request.settings.get("page_range"):
                raise ValueError("Safe backend does not implement page-range extraction")
            if request.device_id != "cpu" and backend == "safe":
                raise ValueError("Safe extraction requires CPU")
            if backend == "docling" and request.device_id != "cpu":
                # No silent device substitution; GPU qualification is separate.
                raise ValueError("Sequential adapter currently qualified on CPU only")
            from ..gloss_extract import GlossExtract
            # Process-local converters/models survive across attempts. Every
            # semantic option is part of the key; input/workspace paths are not.
            key = json.dumps({name: value for name, value in request.settings.items() if name != "publication_name"}, sort_keys=True)
            extractor = _worker_extractors.get(key)
            if extractor is None:
                extractor = GlossExtract()
                _worker_extractors.clear()  # bounded model cache per worker
                _worker_extractors[key] = extractor
            extractor.log_file = request.work_directory/'conversion.log'
            extractor.enable_accel(int(request.settings.get("num_threads", 1)), type="CPU")
            extractor.configure_batch_policy("safe" if backend == "safe" else "docling", max_batch_files=1, prefer_safe_backend=backend == "safe")
            extractor.repair_fonts = bool(request.settings.get("repair_fonts", True))
            if request.settings.get("repair_backend") is not None:
                from ..pdf_repair.selector import get_backend
                extractor.repair_backend = request.settings["repair_backend"]
                get_backend(extractor.repair_backend)  # fail explicit unavailable backends before conversion
            extractor.export_doc_json = bool(request.settings.get("export_doc_json", True))
            extractor.emit_formula_index = bool(request.settings.get("emit_formula_index", False))
            extractor.benchmark_mode = bool(request.settings.get("benchmark_mode", False))
            extractor.processing_timeout = max(0.001, (request.deadline_monotonic_ns-time.monotonic_ns()) / 1e9)
            extractor.page_range = tuple(request.settings["page_range"]) if request.settings.get("page_range") else original.page_range
            for name in ("long_pdf_page_threshold", "chunk_size", "max_chunk_timeouts", "per_page_timeout_s"):
                if name in request.settings:
                    setattr(extractor, name, int(request.settings[name]))
            extractor.ensure_extractor(formula_enrichment=bool(request.settings.get("formula_enrichment", False)),
                code_enrichment=bool(request.settings.get("code_enrichment", False)), use_cls=bool(request.settings.get("use_cls", False)),
                images_scale=float(request.settings.get("images_scale", 1.25)),
                profile_timings=not extractor.benchmark_mode)
            return self.run_prepared(request, extractor=extractor, reporter=reporter,
                serialized_source_path=str(original.path))
        except Exception as error:
            return failed_result(request, "input_hash_changed" if str(error) == "input_hash_changed" else "extract_backend_unavailable" if isinstance(error, ImportError) else "extract_configuration_invalid", error)

    def run_prepared(self, request, *, extractor, reporter, serialized_source_path=None):
        """Warm sequential compatibility boundary. Caller owns worker lifetime.

        The ordinary public adapter calls this inside its fenced spawned worker.
        Legacy Corpus still owns its existing worker/converter until step 5/6.
        """
        started = time.monotonic_ns()
        source = request.inputs[0]
        root = request.work_directory
        markdown = root / "markdown"
        markdown.mkdir(parents=True, exist_ok=True)
        (markdown / "timeout_files").mkdir(exist_ok=True)
        required = tuple(request.settings.get("required_output_kinds", ("page_quality",) if request.operation == "extract.page_quality" else ("raw_markdown",)))
        observations = []
        original_provenance = getattr(extractor, "_stage_source_provenance", None)
        try:
            # Hash once per byte stream (K4(c) R4): a snapshot this worker froze moments ago was hashed while its bytes
            # were written (freeze_inputs, same producer scope); only an input this process did not write is re-read.
            from .execution import produced_hash
            written = produced_hash(source.path)
            if (written != (source.sha256, source.bytes) and sha256_file(source.path) != source.sha256) or source.path.stat().st_size != source.bytes:
                raise ValueError("input_hash_changed")
            extractor._stage_source_provenance = {"path": serialized_source_path or str(source.path), "sha256": source.sha256}
            dictionary_path = os.environ.get("GLOSSAPI_GREEK_WORDLIST")
            dictionary_hash = wordlist_identity(dictionary_path)
            with capture_backend_observations() as observations:
                if request.operation == "extract.page_quality":
                    from ..page_quality import page_quality
                    from ..ledger.encoding import pack_evidence
                    options = dict(request.settings.get("page_quality", {}))
                    dictionary = options.get("wordlist", dictionary_path)
                    dictionary_hash = sha256_file(dictionary) if isinstance(dictionary, str) else None
                    report = page_quality(source.path, output_dir=root, **options)
                    dictionary_hash = (report.dictionary_identity or {}).get("sha256")
                    decisions = ({"operation": "extract.page_quality", "input_sha256": source.sha256,
                        "dictionary_sha256": dictionary_hash,
                        "dictionary_identity": report.dictionary_identity,
                        "effective_settings": {**options, "rule_profile": report.rule_profile,
                                               "fold_validity": report.fold_validity,
                                               "image_dominant_min_valid_word_rate": report.image_dominant_min_valid_word_rate,
                                               "image_dominant_max_junk_share": report.image_dominant_max_junk_share},
                        "report": pack_evidence(report.to_dict())},)
                    return StageResult(request.task_id, request.attempt_id, request.lease_token, "ok", "extract_completed",
                        outputs_in(root, required), decisions, {"wall_ns": time.monotonic_ns()-started})
                extractor._font_repair_paths = {}
                extractor._font_repair_originals = {}
                extractor._font_repair_deadlines = {}
                extractor._font_repair_records = {}
                extractor._font_repair_output_dir = markdown
                extractor._preflight_page_counts = {}
                page_count = extractor._get_pdf_page_count(source.path)
                # Open the source for its page count once: the font-repair budget reuses this count.
                extractor._preflight_page_counts[str(source.path.resolve())] = page_count
                chunked = (not extractor.use_pypdfium_backend and page_count is not None
                    and page_count > extractor.long_pdf_page_threshold and not getattr(extractor, "page_range", None))
                if chunked:
                    success = extractor._process_file_chunked(source.path, markdown, markdown / "timeout_files")
                else:
                    success = extractor._process_single_document(source.path, output_dir=markdown, timeout_dir=markdown / "timeout_files")
            if dictionary_path and wordlist_identity(dictionary_path) != dictionary_hash:
                raise ValueError("Input dictionary changed during extraction")
            outputs = outputs_in(root, required)
            missing = set(required) - {output.kind for output in outputs}
            reason = "extract_completed" if success else "extract_exception"
            sidecar = root / "sidecars" / "extract" / f"{source.path.stem}.json"
            metadata = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            backend_status = metadata.get("extract_status")
            if backend_status in ("partial", "failure", "error", "timeout"):
                success = False
                reason = "extract_timeout" if backend_status == "timeout" else "docling_partial" if backend_status == "partial" else "chunk_assembly_failed" if chunked else "docling_failed"
            coverage = next((item["details"]["page_coverage"] for item in observations if "page_coverage" in item.get("details", {})), None)
            if chunked:
                manifest = root / "markdown" / f"{source.path.stem}.chunks.json"
                chunks = json.loads(manifest.read_text()) if manifest.is_file() else {}
                entries = chunks.get("entries", [])
                covered = [page for item in entries if item["status"] == "ok" for page in range(item["page_start"], item["page_end"]+1)]
                complete = covered == list(range(1, (page_count or 0)+1))
                coverage = {"succeeded": covered, "complete": complete, "chunks": entries}
                if not complete:
                    success, reason = False, "chunk_assembly_failed"
                ranges = {f"{source.path.stem}__p{item['page_start']:04d}-{item['page_end']:04d}.md": (item["page_start"], item["page_end"]) for item in entries}
                outputs = tuple(replace(output, page_range=ranges.get(Path(output.relative_path).name))
                    if output.kind == "markdown_chunk" else output for output in outputs)
            selected_range = tuple(request.settings["page_range"]) if request.settings.get("page_range") else source.page_range
            if selected_range:
                outputs = tuple(replace(output, page_range=selected_range) if output.kind in ("raw_markdown", "docling_json") else output for output in outputs)
            if not success:
                reason = next((item["reason_code"] for item in reversed(observations)
                    if item.get("status") == "failed" and item.get("details", {}).get("backend_status") != "pending"), reason)
            if missing and success:
                success, reason = False, "required_extract_artifact_missing"
            # A JSON file must decode AND validate the selected schema before ok.
            if "docling_json" in required and not missing:
                from ..ocr.utils.json_io import load_docling_json
                for output in outputs:
                    if output.kind == "docling_json":
                        payload = (root / output.relative_path).read_bytes()
                        if output.relative_path.endswith(".zst"):
                            import zstandard
                            payload = zstandard.ZstdDecompressor().decompress(payload)
                        if not isinstance(json.loads(payload), dict):
                            raise ValueError("Docling JSON must be an object")
                        load_docling_json(root / output.relative_path)
            complete = (not coverage.get("failed") and coverage.get("complete", True)) if coverage is not None else None
            decisions = ({"input_sha256": source.sha256, "backend_status": backend_status,
                "coverage": coverage, "coverage_complete": complete,
                "coverage_allows_source_release": bool(success and complete), "retention_authority": False,
                "dictionary_sha256": dictionary_hash, "missing_required_kinds": sorted(missing),
                "backend_observations": observations},)
            if not success:
                outputs = tuple(replace(output, retention_class="failed_attempt", required=False) for output in outputs)
            return StageResult(request.task_id, request.attempt_id, request.lease_token,
                "ok" if success else "failed", reason, outputs, decisions,
                {"wall_ns": time.monotonic_ns()-started, "device_id": request.device_id})
        except Exception as error:
            return failed_result(request, "input_hash_changed" if str(error) == "input_hash_changed" else "output_schema_invalid", error, decisions=({"backend_observations": observations},))
        finally:
            extractor._stage_source_provenance = original_provenance
            extractor._preflight_page_counts = {}
