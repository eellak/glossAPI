"""Sequential per-file acquisition behind the unchanged downloader serializer."""
from pathlib import Path
import tempfile
import time
import uuid

from .contracts import DocumentSnapshot, StageRequest, StageSpecification
from .download import DownloadStage
from .execution import ReceiptCollector
from .products import active_journal


def install_download_adapter(corpus,downloader,configuration,download_mode):
    import asyncio
    if not callable(getattr(downloader,'download_file',None)):
        from ..ledger.journal import observe
        observe('legacy_observed',caller_supplied_aggregate_downloader=True,acquisition_provenance='unverified_no_per_file_kernel')
        return
    kernel=downloader.download_file
    lock=asyncio.Lock()
    async def acquire(row_index,url,semaphore,rate_limiter,retry_count=0,filename_base=None,referer=None):
        settings={**configuration,'download_mode':download_mode,'url':url,'row_index':int(row_index),
            'legacy_outer_retry_count':int(retry_count),'filename_base':filename_base,'referer':referer}
        settings.pop('input_parquet',None)
        specification=StageSpecification('download.acquire',(),('download_payload',),settings=settings)
        if semaphore:await semaphore.acquire()
        await lock.acquire()
        try:
            stage=DownloadStage()
            stage.downloader,stage.kernel,stage.rate_limiter=downloader,kernel,rate_limiter
            journal=active_journal()
            if journal:
                from ..ledger.journal import source_key
                from .sequential import SequentialStageRunner
                with journal.writer.transaction():document_id=journal.writer.document(source_key(url))
                result,workspace=await SequentialStageRunner(journal.writer,corpus.output_dir).execute_async(stage,
                    DocumentSnapshot(document_id,None,url),specification,run_id=journal.run_id)
                journal.adapter_stages.add('download');journal.adapter_attempts.add(result.attempt_id)
            else:
                temporary=Path(tempfile.mkdtemp(prefix='glossapi-download-',dir=corpus.output_dir))
                request=StageRequest(*(str(uuid.uuid4()) for _ in range(5)),specification.operation,(),settings,{},'cpu',
                    temporary/'attempt',time.monotonic_ns()+600_000_000_000)
                result=await stage.run_document_async(request,reporter=ReceiptCollector());workspace=request.work_directory
            if not result.decisions:
                return False,'','',result.error.message if result.error else result.reason_code,retry_count
            receipt=result.decisions[0]
            if result.status=='ok':
                from .compatibility import materialize_stage_outputs
                paths={a.relative_path:corpus.output_dir/a.relative_path for a in result.outputs if a.kind=='download_payload'}
                materialize_stage_outputs(corpus,result,workspace,paths)
            elif not journal:
                import shutil
                shutil.rmtree(workspace.parent)
            return result.status=='ok',receipt['assigned_filename'],receipt['file_ext'],receipt['backend_message'],receipt['legacy_outer_retry_count']
        finally:
            lock.release()
            if semaphore:semaphore.release()
    downloader.download_file=acquire
