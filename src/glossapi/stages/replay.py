"""Project already committed streaming results through legacy serializers."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

from .contracts import ArtifactReference,OutputArtifact
from .execution import sha256_file

_results = ContextVar('glossapi_committed_stage_results', default=None)


@contextmanager
def committed_results(results):
    token=_results.set(results)
    try:
        yield
    finally:
        _results.reset(token)


def replay_result(stage,document,specification):
    results=_results.get()
    if results is None:
        return None
    key=(stage.name,specification.operation,tuple(artifact.sha256 for artifact in document.artifacts))
    candidates=results.get(key,())
    # Equal contents do not merge source identities. The producer binds the
    # document version and exact operation, not a basename checkpoint.
    for version,settings,result,workspace in candidates:
        if version!=document.version_id:
            continue
        if any(settings.get(name)!=value for name,value in specification.settings.items() if name!='compatibility_filename'):
            continue
        if isinstance(result,ArtifactReference):
            if not result.path.is_file() or result.path.stat().st_size!=result.bytes or sha256_file(result.path)!=result.sha256:
                raise ValueError('Committed streaming receipt is missing or corrupt')
            from .workers import read_result
            result=read_result(result.path)
            result=replace(result,outputs=(*result.outputs,OutputArtifact('worker_receipt','worker-result.json',None,'evidence',False)))
        return result,workspace
    return None
