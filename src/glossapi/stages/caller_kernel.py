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
        token=_caller_kernel_scope.set(True)
        try:
            with capture_backend_observations() as observations:
                self.returned=self.callback()
            outputs=[]
            for name,receipt in inventory(self.corpus.output_dir).items():
                if before.get(name)==receipt:continue
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


def execute_caller_kernel(corpus,stage,operation,callback):
    adapter=CallerKernelStage(stage,corpus,callback)
    result,workspace=execute_aggregate(corpus,adapter,[],StageSpecification(operation+'.caller_kernel',(),('generation_manifest',),
        settings={'kernel_profile':'explicit_caller_override','semantic_reuse_eligible':False}))
    if adapter.error:raise adapter.error
    publish_products(corpus,result,workspace,{a.relative_path:corpus.output_dir/Path(a.relative_path).relative_to('products')
        for a in result.outputs if a.kind=='caller_kernel_product'})
    return adapter.returned
