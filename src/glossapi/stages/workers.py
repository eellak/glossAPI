"""Bounded, warm spawned workers. Only the parent publishes or opens SQLite."""
from __future__ import annotations

from typing import Any, cast

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
import json
import logging
import multiprocessing
import os
from pathlib import Path
import signal
import time

from .contracts import OutputArtifact, StageError, StageResult
from .execution import ReceiptCollector, _stop_process, failed_result

_active_workers = ContextVar('glossapi_stage_workers', default=None)


def read_result(path):
    value = json.loads(Path(path).read_text())
    value['outputs'] = tuple(OutputArtifact(**{**item, 'page_range': tuple(item['page_range']) if item['page_range'] else None}) for item in value['outputs'])
    value['decisions'] = tuple(value['decisions'])
    value['error'] = StageError(**value['error']) if value['error'] else None
    return StageResult(**value)


def _serve(connection, parent_pid):
    os.setsid()
    # A dead orchestrator cannot leave a model worker modifying its workspace.
    # Kill the session group, including backend children, on parent death.
    if Path('/proc/self/stat').exists():
        import ctypes
        def parent_lost(*_):
            os.killpg(os.getpgrp(), signal.SIGKILL)
        signal.signal(signal.SIGTERM, parent_lost)
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != parent_pid:
            parent_lost()
    try:
        while True:
            command = connection.recv()
            if command is None:
                return
            stage, request, aggregate = command
            request.work_directory.mkdir(parents=True, exist_ok=False)
            scratch = request.work_directory/'scratch'
            scratch.mkdir()
            os.environ.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                              GLOSSAPI_CONVERSION_LOG_FILE=str(request.work_directory/'conversion.log'))
            import tempfile
            tempfile.tempdir = str(scratch)
            log_handler=logging.FileHandler(request.work_directory/'conversion.log')
            log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging.getLogger().addHandler(log_handler)
            try:
                from .execution import sha256_file
                if any(not artifact.path.is_file() or artifact.path.stat().st_size!=artifact.bytes or sha256_file(artifact.path)!=artifact.sha256 for artifact in request.inputs):
                    raise ValueError('input_hash_changed')
                method = stage.finalize if aggregate else getattr(stage, '_run_document', stage.run_document)
                from .execution import producer_hash_scope,seal_output_hashes
                with producer_hash_scope():
                    result = method(request, reporter=ReceiptCollector())
                    result = seal_output_hashes(cast(Any,result),request.work_directory)
            except Exception as error:
                result = failed_result(request, 'input_hash_changed' if str(error)=='input_hash_changed' else stage.name+'_exception', error)
            finally:
                loggers=[logging.getLogger(),*(logger for logger in logging.Logger.manager.loggerDict.values() if isinstance(logger,logging.Logger))]
                for logger in loggers:
                    for handler in list(logger.handlers):
                        if isinstance(handler,logging.FileHandler) and Path(handler.baseFilename).is_relative_to(request.work_directory):
                            logger.removeHandler(handler);handler.close()
            if not any(output.relative_path=='conversion.log' for output in result.outputs):
                result=replace(result,outputs=(*result.outputs,OutputArtifact('attempt_diagnostic','conversion.log',None,'intermediate',False)))
            from ..ledger.journal import loaded_tools
            from .execution import seal_output_hashes
            result = seal_output_hashes(replace(result, metrics={**result.metrics, 'tools': loaded_tools(), 'worker_pid': os.getpid()}),request.work_directory)
            target = request.work_directory/'worker-result.json'
            try:
                from .execution import write_text_hashed, sha256_file
                encoded = json.dumps(asdict(result), ensure_ascii=False)
                write_text_hashed(target,encoded)
                import hashlib
                data = encoded.encode('utf-8')
                connection.send((request.task_id, request.attempt_id, request.lease_token, 'receipt', hashlib.sha256(data).hexdigest(), len(data)))
            except OSError as error:
                from .execution import disk_full
                if not disk_full(error):raise  # EDQUOT and a wrapped ENOSPC too (fix 6, F3 audit)
                # The result channel still has a reserved bounded slot when a
                # full scratch filesystem cannot hold its JSON receipt.
                connection.send((request.task_id,request.attempt_id,request.lease_token,'disk_budget_exceeded'))
    except EOFError:
        return
    finally:
        connection.close()


class WarmWorkerPool:
    """One bounded pool for a stage/device; no unbounded executor prefetch."""
    def __init__(self, workers=1):
        if type(workers) is not int or workers < 1:
            raise ValueError('Pool needs a positive worker count')
        self.capacity = workers
        self.slots = []
        self.closed = False

    def _new_slot(self):
        parent, child = multiprocessing.get_context('spawn').Pipe()
        process = multiprocessing.get_context('spawn').Process(target=_serve, args=(child, os.getpid()))
        process.start()
        child.close()
        slot = dict(process=process, pipe=parent, request=None, reporter=None, heartbeat=0)
        self.slots.append(slot)
        return slot

    @property
    def active(self):
        return sum(slot['request'] is not None for slot in self.slots)

    def submit(self, stage, request, *, reporter=None, aggregate=False):
        if self.closed or self.active >= self.capacity:
            raise RuntimeError('Worker pool is closed or saturated')
        slot = next((s for s in self.slots if s['request'] is None), None) or self._new_slot()
        slot.update(request=request, reporter=reporter, heartbeat=0, lease_expiry=time.monotonic()+60)
        slot['pipe'].send((stage, request, aggregate))
        return request.attempt_id

    def poll(self):
        completed = []
        for slot in list(self.slots):
            request = slot['request']
            if request is None:
                continue
            process = slot['process']
            if time.monotonic()>slot['lease_expiry']:
                self._discard(slot)
                request.work_directory.mkdir(parents=True,exist_ok=True)
                completed.append((request,failed_result(request,'worker_lost',TimeoutError('Worker lease expired; late result fenced'))))
                continue
            if time.monotonic() >= slot['heartbeat']:
                if slot['reporter']:
                    slot['reporter'].heartbeat(request.lease_token, {'worker_pid': process.pid})
                slot['heartbeat'] = time.monotonic()+5
                slot['lease_expiry'] = time.monotonic()+60
            if slot['pipe'].poll():
                try:
                    identity = slot['pipe'].recv()
                except (EOFError, OSError):
                    identity = None
                if identity is not None:
                    if identity[:3] != (request.task_id, request.attempt_id, request.lease_token):
                        self._discard(slot)
                        raise ValueError('Stale worker result')
                    result = failed_result(request,'disk_budget_exceeded',OSError(28,'Scratch quota exhausted before result receipt')) if len(identity)==4 else read_result(request.work_directory/'worker-result.json')
                    if (result.task_id, result.attempt_id, result.lease_token) != identity[:3]:
                        self._discard(slot)
                        raise ValueError('Stale worker receipt')
                    if len(identity)!=4:
                        digest,size = identity[4:6] if len(identity)==6 else (None,None)
                        result = replace(result, outputs=(*result.outputs, OutputArtifact('worker_receipt', 'worker-result.json', None, 'evidence', False, digest, size)))
                    slot['request'] = None
                    completed.append((request, result))
                    continue
            if not process.is_alive() or time.monotonic_ns() >= request.deadline_monotonic_ns:
                expired = process.is_alive()
                self._discard(slot)
                result = failed_result(request, 'extract_timeout' if expired and request.operation.startswith('extract.') else 'worker_lost',
                                       TimeoutError('Worker deadline expired; process group fenced') if expired else RuntimeError('Worker process exited'))
                request.work_directory.mkdir(parents=True, exist_ok=True)
                completed.append((request, result))
        return completed

    def execute(self, stage, request, *, reporter):
        self.submit(stage, request, reporter=reporter)
        while True:
            result = self.poll()
            if result:
                return result[0][1]
            time.sleep(.005)

    def _discard(self, slot):
        self.slots.remove(slot)
        _stop_process(slot['process'])
        slot['pipe'].close()
        slot['process'].close()

    def close(self):
        self.closed = True
        for slot in list(self.slots):
            if slot['request'] is None and slot['process'].is_alive():
                slot['pipe'].send(None)
                slot['process'].join(1)
            self._discard(slot)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@contextmanager
def warm_worker_scope():
    """Keep wrapper extraction workers warm for the complete public phase."""
    existing = _active_workers.get()
    if existing is not None:
        yield existing
        return
    pools = {}
    token = _active_workers.set(pools)
    try:
        yield pools
    finally:
        _active_workers.reset(token)
        for pool in pools.values():
            pool.close()


def warm_fenced(stage, request, reporter):
    pools = _active_workers.get()
    if pools is None:
        return None
    key = (stage.name, request.device_id)
    pool = pools.setdefault(key, WarmWorkerPool())
    return pool.execute(stage, request, reporter=reporter)
