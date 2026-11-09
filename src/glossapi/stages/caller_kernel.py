"""Adapter for explicitly caller-supplied legacy kernel hooks.

These hooks retain their original paths and exception/return identity. Their
effects are sealed as a generation, labelled unverified, and never reusable.
They do not qualify the production backend or its private-worker contract.
"""
from pathlib import Path
import shutil

from .execution import failed_result
from .products import execute_aggregate,generation_outputs,publish_products
from .profile import _caller_kernel_scope
from .receipts import capture_backend_observations
from .contracts import StageSpecification


class CallerKernelStage:
    contract_version=1

    def __init__(self,stage,corpus,callback):
        self.name,self.corpus,self.callback=stage,corpus,callback
        self.returned=None;self.error=None

    def finalize(self,request,*,reporter):
        from ..ledger.journal import inventory,wrapper_return_receipt
        request.work_directory.mkdir(parents=True,exist_ok=True)
        before=inventory(self.corpus.output_dir)
        stale=_not_committed(before)
        token=_caller_kernel_scope.set(True)
        try:
            with capture_backend_observations() as observations:
                self.returned=self.callback()
            outputs=[]
            for name,receipt in inventory(self.corpus.output_dir).items():
                # Unchanged by the callback, and its bytes are the committed generation: nothing to publish. Bytes the
                # ledger does not commit (a failed attempt's, left in place) are published even when the callback wrote
                # the same bytes again, so the path gets its committed generation (fix 13).
                if before.get(name)==receipt and name not in stale:continue
                relative='products/'+str(Path(name).relative_to(self.corpus.output_dir.absolute()))
                destination=request.work_directory/relative;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(name,destination);outputs.append((relative,'caller_kernel_product'))
            return generation_outputs(request,outputs,decisions=({'kernel_profile':'explicit_caller_override',
                'binding':'legacy_unverified','semantic_reuse_eligible':False,'write_scope':'caller_compatibility_paths',
                'wrapper_return':wrapper_return_receipt(self.returned),'backend_observations':observations,
                'removed_paths':sorted(set(before)-set(inventory(self.corpus.output_dir)))},))
        except Exception as error:
            self.error=error
            return failed_result(request,'legacy_observed',error)
        finally:
            _caller_kernel_scope.reset(token)


def _not_committed(before):
    """Files in ``before`` that ``ledger.writer.expected_products`` manages but whose bytes are not the path's committed
    generation (a different digest, a refusal, an expected absence). Files it does not manage are left out."""
    from ..ledger.writer import ExpectedSha, expected_products
    from .products import active_journal
    journal=active_journal()
    if journal is None or not before:
        return set()
    uris={Path(name).absolute().as_uri():name for name in before}
    expected=expected_products(journal.writer.connection,uris=set(uris))
    return {uris[uri] for uri,verdict in expected.items() if uri in uris
            and not (isinstance(verdict,ExpectedSha) and verdict.sha256==before[uris[uri]]['sha256'])}


def execute_caller_kernel(corpus,stage,operation,callback):
    adapter=CallerKernelStage(stage,corpus,callback)
    result,workspace=execute_aggregate(corpus,adapter,[],StageSpecification(operation+'.caller_kernel',(),('generation_manifest',),
        settings={'kernel_profile':'explicit_caller_override','semantic_reuse_eligible':False}))
    if adapter.error:raise adapter.error
    # The products are captures of the files the callback changed. Republishing them is guarded (K2F round 4, CP-K2e
    # F1): metadata destinations go through metadata_target (physical target, lock taken only now, after the callback)
    # and are refused if they changed after capture, instead of reverting a later commit.
    publish_products(corpus,result,workspace,{a.relative_path:corpus.output_dir/Path(a.relative_path).relative_to('products')
        for a in result.outputs if a.kind=='caller_kernel_product'},captured=True)
    return adapter.returned
