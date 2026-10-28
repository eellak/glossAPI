import logging
from types import SimpleNamespace
import pytest
from glossapi.corpus.phase_extract import ExtractPhaseMixin


class Harness(ExtractPhaseMixin):
    def __init__(self):
        self.logger=logging.getLogger('test');self._gpu_banner_logged=False
        self.extractor=SimpleNamespace(enable_accel=lambda **kw:None,configure_batch_policy=lambda *a,**kw:None,ensure_extractor=lambda **kw:None)


def test_explicit_cpu_layout_never_probes_cuda(monkeypatch):
    import glossapi.corpus.phase_extract as phase
    monkeypatch.setattr(phase,'_maybe_import_torch',lambda **kw:pytest.fail('CPU layout must not require CUDA'))
    Harness().prime_extractor(phase1_backend='docling',accel_type='CPU')


@pytest.mark.parametrize('kwargs',[{}, {'accel_type':'CPU','formula_enrichment':True}, {'accel_type':'CPU','code_enrichment':True}])
def test_gpu_default_and_enrichment_stay_guarded(monkeypatch,kwargs):
    import glossapi.corpus.phase_extract as phase
    monkeypatch.setattr(phase,'_maybe_import_torch',lambda **kw:None)
    with pytest.raises(RuntimeError):Harness().prime_extractor(phase1_backend='docling',**kwargs)


@pytest.mark.parametrize('page_range',[(0,1),(2,1),(1,1.5)])
def test_page_range_rejects_invalid_before_processing(page_range):
    with pytest.raises(ValueError,match='page_range'):Harness().extract(page_range=page_range,phase1_backend='docling',accel_type='CPU')


def test_docling_conversion_forwards_intact_source_and_page_range():
    from pathlib import Path
    from glossapi.gloss_extract import GlossExtract
    seen=[]
    extractor=GlossExtract.__new__(GlossExtract)
    extractor.page_range=(2,4);extractor._font_repair_output_dir=None;extractor._font_repair_deadlines={}
    extractor.converter=SimpleNamespace(convert=lambda source,**kw:seen.append((source,kw)))
    extractor._convert_with_timeout(Path('original.pdf'),120)
    assert seen==[(Path('original.pdf'),{'raises_on_error':False,'page_range':(2,4)})]
