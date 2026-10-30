"""One fenced async acquisition process with shared rate and domain limits."""
from __future__ import annotations

import asyncio
from dataclasses import asdict,replace
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import time
from urllib.parse import urlsplit

from .contracts import OutputArtifact
from .execution import ReceiptCollector, _stop_process, failed_result
from .workers import read_result


async def _acquire(stage,request,limiters,domains):
    from ..gloss_downloader import GlossDownloader,RateLimiter
    options=dict(request.settings)
    for key in ('url','filename_base','row_index','referer','legacy_outer_retry_count','required_output_kinds','debug','memory_saving','download_mode','disk_budget_bytes','max_source_bytes','max_attempt_bytes','url_policy','transport_url_redacted'):
        options.pop(key,None)
    options['progress_log_file']=str(request.work_directory/'logs/download_progress.logs')
    options['request_log_file']=str(request.work_directory/'logs/download_request.logs')
    mode=request.settings.get('download_mode','standard')
    downloader_class=GlossDownloader
    if mode not in ('standard','default','http') or options.get('download_policy') or options.get('download_policy_file'):
        from ..gloss_browser_downloader import BrowserGlossDownloader
        downloader_class=BrowserGlossDownloader
        options['default_download_route']='browser' if mode in ('browser','browser_protected') else 'auto' if mode in ('auto','browser_fallback') else 'standard'
    downloader=downloader_class(output_dir=request.work_directory/'controller',**options)
    handlers=[]
    # Constructor loggers are module-global. Move only this attempt's handlers
    # to private loggers before the first await; warm concurrent jobs must not
    # append to another attempt, especially after it has been published.
    for field in ('logger','progress_logger'):
        original=getattr(downloader,field)
        private=logging.Logger(original.name+'.'+request.attempt_id,level=original.level)
        from ..ledger.urls import CredentialLogFilter
        private.addFilter(CredentialLogFilter())
        for handler in list(original.handlers):
            if isinstance(handler,logging.FileHandler) and Path(handler.baseFilename).is_relative_to(request.work_directory):
                original.removeHandler(handler);private.addHandler(handler);handlers.append(handler)
        setattr(downloader,field,private)
    key=(options.get('rate_limit',100),options.get('rate_period',60))
    limiter=limiters.setdefault(key,RateLimiter(rate_limit=key[0] or 1000000000,time_period=key[1]))
    domain=urlsplit(request.settings['url']).netloc.lower()
    semaphore=domains.setdefault(domain,asyncio.Semaphore(int(options.get('per_domain_concurrency',5))))
    stage.downloader=downloader
    stage.kernel=downloader.download_file
    stage.rate_limiter=limiter
    from .execution import producer_hash_scope, seal_output_hashes
    with producer_hash_scope():
        try:
            async with semaphore:
                result=await stage.run_document_async(request,reporter=ReceiptCollector())
        finally:
            for handler in handlers:handler.close()
        return seal_output_hashes(result,request.work_directory)


def _serve(connection,parent_pid,capacity):
    os.setsid()
    if Path('/proc/self/stat').exists():
        import ctypes
        def parent_lost(*_):os.killpg(os.getpgrp(),signal.SIGKILL)
        signal.signal(signal.SIGTERM,parent_lost)
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent_pid:parent_lost()
    async def serve():
        active={};limiters={};domains={};closing=False
        while active or not closing:
            while not closing and len(active)<capacity and connection.poll():
                command=connection.recv()
                if command is None:closing=True;break
                stage,request=command
                active[asyncio.create_task(_acquire(stage,request,limiters,domains))]=request
            if not active:
                await asyncio.sleep(.005);continue
            completed,_=await asyncio.wait(active,timeout=.01,return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                request=active.pop(task)
                try:result=task.result()
                except Exception as error:result=failed_result(request,'download_exception',error)
                from ..ledger.journal import loaded_tools
                result=replace(result,metrics={**result.metrics,'tools':loaded_tools(),'worker_pid':os.getpid()})
                request.work_directory.mkdir(parents=True,exist_ok=True)
                try:
                    from .execution import producer_hash_scope, seal_output_hashes, write_text_hashed
                    import hashlib
                    with producer_hash_scope():
                        result=seal_output_hashes(result,request.work_directory)
                    from ..ledger.urls import parse_policy, scrub, scrub_known_value
                    encoded=json.dumps(scrub_known_value(scrub(asdict(result),parse_policy(request.settings.get('url_policy')))),ensure_ascii=False)
                    write_text_hashed(request.work_directory/'worker-result.json',encoded)
                    data=encoded.encode('utf-8')
                    connection.send((request.task_id,request.attempt_id,request.lease_token,
                        'receipt',hashlib.sha256(data).hexdigest(),len(data)))
                except OSError as error:
                    import errno
                    if error.errno!=errno.ENOSPC:raise
                    connection.send((request.task_id,request.attempt_id,request.lease_token,'disk_budget_exceeded'))
    try:asyncio.run(serve())
    except (EOFError,BrokenPipeError):pass
    finally:connection.close()


class AsyncAcquisitionPool:
    def __init__(self,capacity):
        self.capacity=capacity;self.requests={};self.process=None;self.pipe=None

    @property
    def active(self):return len(self.requests)

    def submit(self,stage,request,*,reporter=None):
        if self.active>=self.capacity:raise RuntimeError('Acquisition pool saturated')
        if self.process is None:
            parent,child=multiprocessing.get_context('spawn').Pipe()
            self.process=multiprocessing.get_context('spawn').Process(target=_serve,args=(child,os.getpid(),self.capacity))
            self.process.start();child.close();self.pipe=parent
        self.requests[request.attempt_id]=(request,reporter,time.monotonic()+60,0)
        self.pipe.send((stage,request))

    def poll(self):
        completed=[]
        if self.process is None:return completed
        now=time.monotonic()
        if not self.process.is_alive() or any(now>entry[2] or time.monotonic_ns()>=entry[0].deadline_monotonic_ns for entry in self.requests.values()):
            for request,_,_,_ in self.requests.values():
                request.work_directory.mkdir(parents=True,exist_ok=True)
                completed.append((request,failed_result(request,'worker_lost',TimeoutError('Async acquisition process fenced'))))
            self.close();return completed
        for key,(request,reporter,expiry,heartbeat) in list(self.requests.items()):
            if now>=heartbeat:
                if reporter:reporter.heartbeat(request.lease_token,{'worker_pid':self.process.pid})
                self.requests[key]=(request,reporter,now+60,now+5)
        while self.pipe.poll():
            identity=self.pipe.recv()
            request,_,_,_=self.requests.pop(identity[1])
            if identity[:3]!=(request.task_id,request.attempt_id,request.lease_token):
                self.close();raise ValueError('Stale acquisition lease')
            result=failed_result(request,'disk_budget_exceeded',OSError(28,'Acquisition scratch quota exhausted')) if len(identity)==4 else read_result(request.work_directory/'worker-result.json')
            if (result.task_id,result.attempt_id,result.lease_token)!=identity[:3]:
                self.close();raise ValueError('Stale acquisition receipt')
            if len(identity)!=4:
                digest,size=identity[4:6] if len(identity)==6 else (None,None)
                result=replace(result,outputs=(*result.outputs,OutputArtifact('worker_receipt','worker-result.json',None,'evidence',False,digest,size)))
            completed.append((request,result))
        return completed

    def close(self):
        if self.process is not None:
            _stop_process(self.process);self.pipe.close();self.process.close()
        self.process=None;self.pipe=None;self.requests.clear()
