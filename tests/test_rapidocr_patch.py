import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _clear_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def _install_docling_stub(*, supports_injection: bool) -> None:
    _clear_modules("docling")
    _clear_modules("docling_core")
    _clear_modules("glossapi")

    def register(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    docling = register("docling")
    register("docling.backend")
    register("docling.backend.docling_parse_backend").DoclingParseDocumentBackend = object
    register("docling.backend.docling_parse_v2_backend").DoclingParseV2DocumentBackend = object
    register("docling.backend.pypdfium2_backend").PyPdfiumDocumentBackend = object

    base_models = register("docling.datamodel.base_models")

    class InputFormat:
        PDF = "pdf"
        DOCX = "docx"
        XML_JATS = "xml"
        HTML = "html"
        PPTX = "pptx"
        CSV = "csv"
        MD = "md"

    class ConversionStatus:
        SUCCESS = "success"
        PARTIAL_SUCCESS = "partial"

    class Page:
        def __init__(self):
            self._backend = types.SimpleNamespace(
                is_valid=lambda: True,
                get_page_image=lambda *args, **kwargs: types.SimpleNamespace()
            )

    base_models.InputFormat = InputFormat
    base_models.ConversionStatus = ConversionStatus
    base_models.Page = Page

    pipeline_opts = register("docling.datamodel.pipeline_options")

    class AcceleratorDevice:
        AUTO = "auto"
        CUDA = "cuda"
        MPS = "mps"
        CPU = "cpu"

    class AcceleratorOptions:
        def __init__(self, num_threads=None, device=None):
            self.num_threads = num_threads
            self.device = device

    class PdfPipelineOptions:
        def __init__(self, **_kwargs):
            self.ocr_options = None
            self.do_ocr = False

    class RapidOcrOptions:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.rec_keys_path = None

    class OcrOptions:
        pass

    class LayoutOptions:
        pass

    class TableStructureOptions:
        def __init__(self, mode=None):
            self.mode = mode
            self.do_cell_matching = False

    class TableFormerMode:
        ACCURATE = "accurate"

    class PictureDescriptionApiOptions:
        pass

    pipeline_opts.AcceleratorDevice = AcceleratorDevice
    pipeline_opts.AcceleratorOptions = AcceleratorOptions
    pipeline_opts.PdfPipelineOptions = PdfPipelineOptions
    pipeline_opts.RapidOcrOptions = RapidOcrOptions
    pipeline_opts.OcrOptions = OcrOptions
    pipeline_opts.LayoutOptions = LayoutOptions
    pipeline_opts.TableStructureOptions = TableStructureOptions
    pipeline_opts.TableFormerMode = TableFormerMode
    pipeline_opts.PictureDescriptionApiOptions = PictureDescriptionApiOptions

    register("docling.datamodel.document").ConversionResult = object

    settings_mod = register("docling.datamodel.settings")

    class _Debug:
        def __init__(self):
            self.profile_pipeline_timings = False
            self.visualize_ocr = False

    class _Settings:
        def __init__(self):
            self.debug = _Debug()

    settings_mod.settings = _Settings()

    converter_mod = register("docling.document_converter")

    class DocumentConverter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class PdfFormatOption:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    converter_mod.DocumentConverter = DocumentConverter
    converter_mod.PdfFormatOption = PdfFormatOption
    converter_mod.WordFormatOption = object
    converter_mod.HTMLFormatOption = object
    converter_mod.XMLJatsFormatOption = object
    converter_mod.PowerpointFormatOption = object
    converter_mod.MarkdownFormatOption = object
    converter_mod.CsvFormatOption = object

    register("docling.pipeline.simple_pipeline").SimplePipeline = object

    pipelines_mod = register("docling.pipelines.standard_pdf_pipeline")
    pipeline_mod = register("docling.pipeline.standard_pdf_pipeline")

    if supports_injection:
        class StandardPdfPipeline:
            def __init__(self, opts, ocr_model=None, **_):
                self.opts = opts
                self.ocr_model = ocr_model
    else:
        class StandardPdfPipeline:
            def __init__(self, opts, **_):
                self.opts = opts

    pipelines_mod.StandardPdfPipeline = StandardPdfPipeline
    pipeline_mod.StandardPdfPipeline = StandardPdfPipeline

    rapid_module = register("docling.models.rapid_ocr_model")

    class DummyReader:
        def __call__(self, *_args, **_kwargs):
            return []

    class RapidOcrModel:
        def __init__(self, enabled, artifacts_path, options, accelerator_options):
            self.enabled = enabled
            self.reader = DummyReader()
            self.options = options

        def get_ocr_rects(self, _page):
            return []

        def post_process_cells(self, _cells, _page):
            pass

    class TextCell:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Log:
        @staticmethod
        def warning(_msg):
            pass

    rapid_module.RapidOcrModel = RapidOcrModel
    rapid_module.TextCell = TextCell
    rapid_module._log = _Log()

    utils_mod = register("docling.utils")
    profiling_mod = register("docling.utils.profiling")

    class TimeRecorder:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    profiling_mod.TimeRecorder = TimeRecorder
    utils_mod.profiling = profiling_mod

    register("docling.models")

    core_doc = register("docling_core.types.doc")

    class BoundingBox:
        @staticmethod
        def from_tuple(coord, origin=None):
            return SimpleNamespace(coord=coord, origin=origin)

    class CoordOrigin:
        TOPLEFT = "topleft"

    core_doc.BoundingBox = BoundingBox
    core_doc.CoordOrigin = CoordOrigin

    core_page = register("docling_core.types.doc.page")

    class BoundingRectangle:
        @staticmethod
        def from_bounding_box(box):
            return box

    core_page.BoundingRectangle = BoundingRectangle


def _install_onnxruntime_stub():
    sys.modules['onnxruntime'] = types.SimpleNamespace(
        get_available_providers=lambda: ['CUDAExecutionProvider']
    )


def _make_safe_ocr() -> SimpleNamespace:
    """Return an instantiated SafeRapidOcrModel with stubbed dependencies."""
    rapid_opts = sys.modules['docling.datamodel.pipeline_options'].RapidOcrOptions()
    accel_opts = sys.modules['docling.datamodel.pipeline_options'].AcceleratorOptions(device='cuda:0')
    from glossapi.rapidocr_safe import SafeRapidOcrModel

    return SafeRapidOcrModel(enabled=True, artifacts_path=None, options=rapid_opts, accelerator_options=accel_opts)


@pytest.fixture(autouse=True)
def _cleanup_modules():
    yield
    for name in [n for n in list(sys.modules) if n.startswith('glossapi') and '_rapidocr_paths' not in n]:
        if name.startswith('glossapi_rs_'):
            continue
        sys.modules.pop(name, None)
    _clear_modules('docling')
    _clear_modules('docling_core')
    sys.modules.pop('onnxruntime', None)


def test_patch_runs_on_import():
    _install_docling_stub(supports_injection=True)
    _install_onnxruntime_stub()

    importlib.import_module('glossapi')
    rapid_module = sys.modules['docling.models.rapid_ocr_model']
    from glossapi.rapidocr_safe import SafeRapidOcrModel, patch_docling_rapidocr

    assert rapid_module.RapidOcrModel is SafeRapidOcrModel

    patch_docling_rapidocr()
    assert rapid_module.RapidOcrModel is SafeRapidOcrModel


def test_build_rapidocr_pipeline_injects_when_supported(monkeypatch):
    _install_docling_stub(supports_injection=True)
    _install_onnxruntime_stub()

    glossapi_mod = importlib.import_module('glossapi')
    pipeline = importlib.reload(importlib.import_module('glossapi._pipeline'))

    monkeypatch.setattr(
        pipeline,
        'resolve_packaged_onnx_and_keys',
        lambda: SimpleNamespace(det='det', rec='rec', cls='cls', keys='keys'),
    )

    captured = {}

    def fake_pool_get(device, opts, factory, expected_type):
        model = factory()
        assert isinstance(model, pipeline.SafeRapidOcrModel)
        assert expected_type is pipeline.SafeRapidOcrModel
        captured['device'] = device
        captured['opts'] = opts
        return SimpleNamespace()

    monkeypatch.setattr(pipeline, 'GLOBAL_RAPID_OCR_POOL', SimpleNamespace(get=fake_pool_get))

    engine, opts = pipeline.build_rapidocr_pipeline(device='cuda:0')
    assert hasattr(engine, 'ocr_model')
    assert captured['device'] == 'cuda:0'
    assert opts.do_ocr is True


def test_build_rapidocr_pipeline_falls_back_without_injection(monkeypatch):
    _install_docling_stub(supports_injection=False)
    _install_onnxruntime_stub()

    importlib.import_module('glossapi')
    pipeline = importlib.reload(importlib.import_module('glossapi._pipeline'))

    monkeypatch.setattr(
        pipeline,
        'resolve_packaged_onnx_and_keys',
        lambda: SimpleNamespace(det='det', rec='rec', cls='cls', keys='keys'),
    )

    def fail_pool(*_args, **_kwargs):
        raise AssertionError('Pool should not be used when injection unsupported')

    monkeypatch.setattr(pipeline, 'GLOBAL_RAPID_OCR_POOL', SimpleNamespace(get=fail_pool))

    engine, opts = pipeline.build_rapidocr_pipeline(device='cuda:0')
    converter_mod = importlib.import_module('docling.document_converter')
    assert isinstance(engine, converter_mod.DocumentConverter)
    assert opts.do_ocr is True


def test_safe_rapidocr_normalises_none(monkeypatch):
    _install_docling_stub(supports_injection=True)
    _install_onnxruntime_stub()

    importlib.import_module('glossapi')
    model = _make_safe_ocr()

    assert model._normalise_result(None) == []


def test_safe_rapidocr_normalises_incomplete_and_valid_data(monkeypatch):
    _install_docling_stub(supports_injection=True)
    _install_onnxruntime_stub()

    importlib.import_module('glossapi')
    model = _make_safe_ocr()

    class IncompleteResult:
        boxes = None
        txts = ['foo']
        scores = [0.9]

    assert model._normalise_result(IncompleteResult()) == []

    box = np.array([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    ])

    class FullResult:
        boxes = box
        txts = ['foo']
        scores = [0.9]

    output = model._normalise_result(FullResult())
    assert output == [
        (box[0].tolist(), 'foo', 0.9)
    ]
