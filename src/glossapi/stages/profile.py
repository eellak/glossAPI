"""The public Corpus defaults to sequential adapters; legacy is a control."""
from contextvars import ContextVar
import os

_public_adapter_scope = ContextVar('glossapi_public_adapter_scope', default=False)
_caller_kernel_scope = ContextVar('glossapi_caller_kernel_scope', default=False)


def adapters_enabled(corpus=None):
    if _caller_kernel_scope.get():return False
    configured = os.environ.get('GLOSSAPI_STAGE_PROFILE')
    if configured is not None:
        return configured in ('sequential-adapters','cutover')
    return bool(getattr(corpus, '_stage_adapters_default', False) or _public_adapter_scope.get())


def caller_kernel_override(corpus,stage):
    """Recognize explicit application injections, never missing-package fallback."""
    import sys
    if os.environ.get('GLOSSAPI_STAGE_PROFILE') in ('legacy','sequential-adapters'):return False
    if stage=='clean':return '_load_rust_extension' in vars(corpus)
    if stage=='ocr':
        dispatch=sys.modules.get('glossapi.corpus.ocr.dispatch')
        factory=getattr(dispatch,'_build_converter',None)
        if factory and getattr(factory,'__module__',None)!='glossapi.corpus.ocr.dispatch':return True
        for name in ('glossapi.ocr.math','glossapi.ocr.math.enrich'):
            module=sys.modules.get(name)
            kernel=vars(module).get('enrich_from_docling_json') if module else None
            if kernel and getattr(kernel,'__module__',None)!='glossapi.ocr.math.enrich':return True
    return False
