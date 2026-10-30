"""Per-file acquisition using the existing HTTP, policy and browser kernels."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageResult, StageSpecification
from .execution import failed_result, outputs_in, run_fenced
from .receipts import capture_backend_observations, capture_transport


class DownloadStage:
    name = "download"
    contract_version = 1

    async def run_document_async(self,request,*,reporter):
        """Use the controller's exact injected transport, limiter and naming state."""
        import copy
        request.work_directory.mkdir(parents=True,exist_ok=True)
        started=time.monotonic_ns()
        try:
            private=copy.copy(self.downloader)
            private.output_dir=request.work_directory
            private.downloads_dir=request.work_directory/'downloads'
            private.downloads_dir.mkdir()
            method=self.kernel.__func__.__get__(private,type(private)) if hasattr(self.kernel,'__func__') else self.kernel
            from ..ledger.journal import safe_settings
            from ..ledger.urls import url_policy
            with url_policy(request.settings.get('url_policy')),capture_backend_observations() as observations,capture_transport(request.settings['download_mode'],safe_settings(dict(request.settings))) as transport:
                returned=await asyncio.wait_for(method(row_index=request.settings['row_index'],url=request.settings['url'],
                    semaphore=None,rate_limiter=self.rate_limiter,retry_count=request.settings['legacy_outer_retry_count'],
                    filename_base=request.settings['filename_base'],referer=request.settings.get('referer')),
                    timeout=max(0,(request.deadline_monotonic_ns-time.monotonic_ns())/1e9))
            success,filename,extension,message,retry_count=returned
            outputs=outputs_in(request.work_directory,('download_payload',))
            if success and not any(output.kind=='download_payload' for output in outputs):raise ValueError('Transport success has no private payload')
            reason='download_completed' if success else next((item['reason_code'] for item in reversed(observations) if item.get('status')=='failed'),'download_retry_exhausted')
            decisions=({'http':{'requests':transport['requests']},'acquisitions':transport['acquisitions'],
                'assigned_filename':filename,'file_ext':extension,'backend_message':message,'legacy_outer_retry_count':retry_count,
                'backend_observations':observations},)
            return StageResult(request.task_id,request.attempt_id,request.lease_token,'ok' if success else 'failed',reason,outputs,decisions,
                {'wall_ns':time.monotonic_ns()-started,'execution':'controller_async_transport'})
        except Exception as error:
            return failed_result(request,getattr(error,'code','download_exception'),error)

    def packages(self, specification):
        requirements = [PackageRequirement("download", ("aiohttp>=3.14.5,<4", "aiofiles>=23.0.0", "tenacity>=9.2.1,<10", "pyyaml>=6.0"))]
        if specification.settings.get("download_mode", "standard") not in ("standard", "http", "default") or specification.settings.get("download_policy") or specification.settings.get("download_policy_file"):
            requirements.append(PackageRequirement("browser", ("playwright>=1.64.1,<2", "Pillow>=10")))
        return tuple(requirements)

    def plan(self, documents, *, settings):
        # Naming/expansion belongs to the sealed source manifest, before any
        # per-file execution. Never generate a different collision name here.
        return StagePlan(tuple(PlannedOperation(document.document_id, document.version_id,
            StageSpecification("download.acquire", (), ("download_payload",), "cpu",
                {**({"filename_base": document.source_records[0]["filename_base"]} if document.source_records and document.source_records[0].get("filename_base") else {}),
                    **settings, "url": document.source_uri}), skip_reason="duplicate_url" if document.source_records and document.source_records[0].get("is_duplicate") else None if document.source_uri.startswith(("https://", "http://")) else "invalid_source_url")
            for document in documents), ("source_manifest_and_naming_sealed",))

    def finalize(self, request, *, reporter):
        raise ValueError("Download has no aggregate finalize operation")

    def run_document(self, request, *, reporter):
        return run_fenced(self, request, reporter=reporter)

    def _run_document(self, request, *, reporter):
        from ..ledger.urls import url_policy
        # Receipts and logs of this attempt use the run's declared URL policy.
        with url_policy(request.settings.get("url_policy")):
            return self._run_document_with_policy(request, reporter=reporter)

    def _run_document_with_policy(self, request, *, reporter):
        started = time.monotonic_ns()
        settings = dict(request.settings)
        url = settings.pop("url", "")
        filename_base = settings.pop("filename_base", None)
        if filename_base is not None and (Path(filename_base).name != filename_base or filename_base in (".", "..")):
            return failed_result(request, "download_exception", ValueError("Unsafe filename base"))
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            return StageResult(request.task_id, request.attempt_id, request.lease_token, "skipped", "invalid_source_url", (), (), {})
        mode = settings.pop("download_mode", "standard")
        try:
            from ..gloss_downloader import GlossDownloader, RateLimiter
            if mode not in ("standard", "default", "http", "browser", "browser_protected", "auto", "browser_fallback"):
                raise ValueError("Unsupported download mode")
            if mode in ("standard", "default", "http") and not (settings.get("download_policy") or settings.get("download_policy_file")):
                downloader_class = GlossDownloader
            else:
                from ..gloss_browser_downloader import BrowserGlossDownloader
                downloader_class = BrowserGlossDownloader
                settings["default_download_route"] = "browser" if mode in ("browser", "browser_protected") else "auto" if mode in ("auto", "browser_fallback") else "standard"
            row_index = int(settings.pop("row_index", 0))
            outer_retry_count = int(settings.pop("legacy_outer_retry_count", 0))
            referer = settings.pop("referer", None)
            settings.pop("required_output_kinds", None)
            settings.pop("debug", None); settings.pop("memory_saving", None); settings.pop("url_policy", None); settings.pop("transport_url_redacted", None)
            for field in ('disk_budget_bytes','max_source_bytes','max_attempt_bytes'):
                settings.pop(field,None)
            # Configured log destinations are compatibility concerns. A worker
            # writes only inside its attempt, including backend diagnostics.
            settings["progress_log_file"] = str(request.work_directory / "logs" / "download_progress.logs")
            settings["request_log_file"] = str(request.work_directory / "logs" / "download_request.logs")
            with capture_backend_observations() as observations, capture_transport(mode, {**settings, "max_source_bytes":request.settings.get("max_source_bytes")}) as transport:
                downloader = downloader_class(output_dir=str(request.work_directory), **settings)
                rate_limiter = downloader.rate_limiter or RateLimiter(rate_limit=1000000, time_period=1)
                returned = asyncio.run(downloader.download_file(row_index, url, None, rate_limiter,
                    retry_count=outer_retry_count, filename_base=filename_base, referer=referer))
            success, filename, extension, message, retry_count = returned
            required = ("download_payload",)
            outputs = outputs_in(request.work_directory, required)
            if success and (not transport["acquisitions"] or not any(output.kind == "download_payload" for output in outputs)):
                raise ValueError("Transport reported success without a stored acquisition")
            reason = "browser_download_completed" if any(item["reason_code"] == "browser_download_completed" for item in observations) else "download_completed"
            if not success:
                reason = next((item["reason_code"] for item in reversed(observations) if item.get("status") == "failed" or item.get("details", {}).get("backend_status") == "failed"), "download_retry_exhausted")
            decisions = ({"http": {"requests": transport["requests"]}, "acquisitions": transport["acquisitions"],
                "requested_filename_base": filename_base, "assigned_filename": filename, "file_ext": extension,
                "legacy_outer_retry_count": retry_count, "backend_message": message, "backend_observations": observations},)
            return StageResult(request.task_id, request.attempt_id, request.lease_token,
                "ok" if success else "failed", reason, outputs, decisions,
                {"wall_ns": time.monotonic_ns()-started, "device_id": request.device_id})
        except Exception as error:
            return failed_result(request, "download_mode_invalid" if isinstance(error, ValueError) and str(error) == "Unsupported download mode" else "download_exception", error)
