"""C1 contract checks and one opt-in, real Corpus.extract CPU cohort run.

Run: GLOSSAPI_RUN_C1=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
     PIP_NO_INDEX=1 .venv-docling/bin/python -m pytest -q -s tests/test_docling_plugin_c1.py
No PDFs/checkpoints are created or copied. Docling page_range limits every conversion
to one frozen page of the ORIGINAL file; baseline converts only the six clean pages.
"""
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter, process_time
import traceback
from types import SimpleNamespace
import unicodedata

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/experiments/glossapi_improvement_20261006"


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def markdown_witness(lines, markdown):
    """Require a changed, distinctive prediction; unchanged common lines cannot pass."""
    candidates = [normalized(c["text"]) for c in lines
                  if normalized(c["text"]) != normalized(c["source_text"])
                  and len(normalized(c["text"])) >= 4]
    counts = Counter(normalized(c["text"]) for c in lines)
    return next((text for text in sorted(candidates, key=len, reverse=True)
                 if counts[text] == 1 and text in normalized(markdown)), None)


def test_markdown_witness_requires_changed_distinctive_text():
    lines = [dict(source_text="common heading", text="common heading"),
             dict(source_text="corrupt body", text="Αναγνωρισμένο σώμα")]
    assert markdown_witness(lines, "common heading") is None
    assert markdown_witness(lines, "Αναγνωρισμένο σώμα") == "Αναγνωρισμένο σώμα"
    assert markdown_witness([lines[1], lines[1]], "Αναγνωρισμένο σώμα") is None


def test_factory_registration():
    pytest.importorskip("docling")
    from docling.models.factories import get_ocr_factory
    from glossapi.ocr.plugin.options import GlossapiOcrOptions
    from glossapi.ocr.plugin.model import GlossapiOcrModel

    factory = get_ocr_factory(allow_external_plugins=True)
    assert "glossapi" in factory.registered_kind
    assert factory.classes[GlossapiOcrOptions] is GlossapiOcrModel
    assert factory.registered_meta[GlossapiOcrOptions].plugin_name == "glossapi"
    assert "glossapi" not in get_ocr_factory(allow_external_plugins=False).registered_kind


def test_page_replacement_boundary(tmp_path, monkeypatch):
    pytest.importorskip("docling")
    import numpy as np
    from PIL import Image
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling_core.types.doc import BoundingBox
    from docling_core.types.doc.page import BoundingRectangle, TextCell
    from glossapi.ocr.plugin.model import GlossapiOcrModel
    from glossapi.ocr.plugin.options import GlossapiOcrOptions

    route = tmp_path / "quality.jsonl"
    route.write_text(json.dumps(dict(filename="doc.pdf", pages=[
        dict(page=1, page_needs_ocr=True), dict(page=2, page_needs_ocr=False),
        dict(page=3, page_needs_ocr=True, chars=0, decision="scanned_page", image_area_ratio=1)])) + "\n")
    def cell(text, from_ocr=False):
        return TextCell(index=0, text=text, orig=text, from_ocr=from_ocr,
                        rect=BoundingRectangle.from_bounding_box(BoundingBox(l=2,t=2,r=40,b=14)))
    def page(number, cells):
        parsed = SimpleNamespace(textline_cells=cells, has_lines=bool(cells),
                                 word_cells=[cell("bad word"), cell("OCR word", True)],
                                 char_cells=[cell("x")], has_words=True, has_chars=True)
        return SimpleNamespace(page_no=number, parsed_page=parsed, size=SimpleNamespace(height=30),
                               get_image=lambda **kwargs: Image.fromarray(np.full((90,150),255,dtype=np.uint8)))
    pages = [page(1,[cell("corrupt")]), page(2,[cell("keep")]), page(3,[])]
    original_rect = pages[0].parsed_page.textline_cells[0].rect.model_dump()
    clean = pages[1].parsed_page
    options = GlossapiOcrOptions(checkpoint=tmp_path/"unused.pt",codec=tmp_path/"unused.json",
                                routing_sidecar=route)
    model = GlossapiOcrModel(enabled=True, artifacts_path=None, options=options,
                            accelerator_options=AcceleratorOptions(device="cpu"))
    monkeypatch.setattr(model,"_get_recognizer",lambda:SimpleNamespace(recognize=lambda strips:[("Αθήνα",.9)]*len(strips)))
    out = list(model(SimpleNamespace(input=SimpleNamespace(file=Path("doc.pdf"))), pages))
    assert out == pages
    replaced = out[0].parsed_page
    assert replaced.textline_cells[0].text == "Αθήνα"
    assert replaced.textline_cells[0].from_ocr
    assert replaced.textline_cells[0].rect.model_dump() == original_rect
    assert [c.text for c in replaced.word_cells] == ["OCR word"]
    assert replaced.has_words and not replaced.char_cells and not replaced.has_chars
    assert out[1].parsed_page is clean and clean.textline_cells[0].text == "keep"
    assert not options.force_full_page_ocr
    assert [(r["page"],r["status"]) for r in model.call_log] == [(1,"replaced"),(3,"unsupported")]
    assert model.unsupported_counts == {"no_textlayer_lines": 1}
    options.mode = "detector"
    assert list(model(SimpleNamespace(input=SimpleNamespace(file=Path("doc.pdf"))), [pages[0]])) == [pages[0]]
    assert model.unsupported_counts["detector_out_of_scope"] == 1


@pytest.mark.parametrize("chars,decision,image_ratio,unsupported", [
    (2, "scanned_page", .85, True),
    (3, "scanned_page", .47, True),
    (5, "scanned_page", .38, True),
    (49, "scanned_page", .36, True),
    (50, "scanned_page", .36, False),
    (3, "garbage_greek_text", 0, False),
    (3, "garbage_greek_text", .5, True),
    (66, "garbage_greek_text", 0, False),
])
def test_raster_body_keeps_native_cells_and_records_recognition(
        tmp_path, monkeypatch, chars, decision, image_ratio, unsupported):
    pytest.importorskip("docling")
    from PIL import Image
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling_core.types.doc import BoundingBox
    from docling_core.types.doc.page import BoundingRectangle, TextCell
    from glossapi.ocr.plugin.model import GlossapiOcrModel
    from glossapi.ocr.plugin.options import GlossapiOcrOptions

    route = tmp_path/"route.jsonl"
    route.write_text(json.dumps(dict(filename="arbitrary.pdf", pages=[dict(
        page=7, page_needs_ocr=True, chars=chars, decision=decision,
        image_area_ratio=image_ratio)]))+"\n")
    cell = TextCell(index=0, text="77", orig="77", from_ocr=False,
                    rect=BoundingRectangle.from_bounding_box(BoundingBox(l=2,t=2,r=40,b=14)))
    parsed = SimpleNamespace(textline_cells=[cell], has_lines=True, word_cells=[cell],
                             char_cells=[cell], has_words=True, has_chars=True)
    page = SimpleNamespace(page_no=7, parsed_page=parsed, size=SimpleNamespace(height=30),
                           get_image=lambda **kw: Image.new("L", (150,90), 255))
    options = GlossapiOcrOptions(checkpoint=tmp_path/"unused.pt", codec=tmp_path/"unused.json",
                                routing_sidecar=route, call_log=tmp_path/"calls.jsonl")
    model = GlossapiOcrModel(enabled=True, artifacts_path=None, options=options,
                            accelerator_options=AcceleratorOptions(device="cpu"))
    calls = []
    def recognize(strips):
        calls.append(len(strips))
        return [("78", .9)]
    monkeypatch.setattr(model, "_get_recognizer", lambda: SimpleNamespace(recognize=recognize))
    assert list(model(SimpleNamespace(input=SimpleNamespace(file=Path("arbitrary.pdf"))), [page])) == [page]
    record = json.loads(options.call_log.read_text())
    assert calls == [1] and record["recognition_performed"]
    assert record["lines"][0]["text"] == "78"
    assert record["body_support"]["textlayer_chars"] == chars
    assert record["body_support"]["chars_source"] == "sidecar"
    assert record["status"] == ("unsupported" if unsupported else "replaced")
    assert record["installed"] is not unsupported
    if unsupported:
        assert record["reason"] == "raster_body" and model.unsupported_counts == {"raster_body": 1}
        assert parsed.textline_cells == [cell] and parsed.word_cells == parsed.char_cells == [cell]
        assert cell.text == "77" and not cell.from_ocr
    else:
        assert parsed.textline_cells[0].text == "78" and parsed.textline_cells[0].from_ocr


def test_checkpoint_failure_instructions_reference_available_export(tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="checkpoint loader requires optional torch")
    from glossapi.ocr.plugin.recognizer import ckpt_io
    def reject(*args, **kwargs):
        assert kwargs["weights_only"] is True
        raise ValueError("unsafe resume object")
    monkeypatch.setattr(ckpt_io.torch, "load", reject)
    with pytest.raises(RuntimeError, match="trusted training/export environment") as exc:
        ckpt_io.load_model_weights(str(tmp_path/"legacy.pt"))
    assert "convert_pt_to_safetensors" not in str(exc.value)


def test_conflicting_sidecar_support_metadata_fails_closed(tmp_path):
    pytest.importorskip("docling", reason="OCR plugin requires optional Docling")
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from glossapi.ocr.plugin.model import GlossapiOcrModel
    from glossapi.ocr.plugin.options import GlossapiOcrOptions
    route = tmp_path/"route.jsonl"
    route.write_text("".join(json.dumps(dict(filename="doc.pdf", pages=[dict(
        page=1, page_needs_ocr=True, chars=chars, decision="scanned_page")]))+"\n"
        for chars in [3, 300]))
    options = GlossapiOcrOptions(checkpoint=Path("unused.pt"), codec=Path("unused.json"),
                                routing_sidecar=route)
    with pytest.raises(ValueError, match="conflicting routing rows"):
        GlossapiOcrModel(enabled=True, artifacts_path=None, options=options,
                        accelerator_options=AcceleratorOptions(device="cpu"))


def test_engine_rejects_batching_before_dispatch(tmp_path):
    from glossapi.corpus import Corpus
    corpus = Corpus(tmp_path, tmp_path / "out")
    with pytest.raises(ValueError, match="batched=True"):
        corpus.extract(ocr_engine="glossapi", batched=True)
    with pytest.raises(ValueError, match="single-device"):
        corpus.extract(ocr_engine="glossapi", batched=False, use_gpus="multi")


def test_engine_options_enable_ocr_and_preserve_defaults():
    pytest.importorskip("docling")
    from glossapi.ocr.docling.pipeline import _apply_common_pdf_options, _resolve_accelerator
    from glossapi.ocr.plugin.options import GlossapiOcrOptions
    acc, _ = _resolve_accelerator("cpu")
    kw = dict(acc=acc,images_scale=1.25,formula_enrichment=False,code_enrichment=False)
    assert not _apply_common_pdf_options(**kw).do_ocr
    options = GlossapiOcrOptions(checkpoint=Path("unused.pt"),codec=Path("unused.json"),replace_policy="all")
    pipeline = _apply_common_pdf_options(**kw,ocr_options=options)
    assert pipeline.do_ocr and pipeline.ocr_options is options
    assert pipeline.allow_external_plugins and not pipeline.ocr_options.force_full_page_ocr


def test_corpus_primes_plugin_on_cpu(tmp_path, monkeypatch):
    pytest.importorskip("docling")
    from glossapi.corpus import Corpus
    from glossapi.ocr.plugin.options import GlossapiOcrOptions
    calls = []
    corpus = Corpus(tmp_path,tmp_path/"out")
    corpus.extractor = SimpleNamespace(
        enable_accel=lambda **kwargs:None,
        configure_batch_policy=lambda *args,**kwargs:calls.append(("policy",args,kwargs)),
        ensure_extractor=lambda **kwargs:calls.append(("ensure",kwargs)),
    )
    route = tmp_path/"routes.jsonl"; route.write_text("")
    corpus.prime_extractor(accel_type="CPU",ocr_engine="glossapi",ocr_checkpoint="checkpoint.pt",
                           ocr_routing_sidecar=route)
    assert corpus._phase1_backend == "docling"
    assert calls[0] == ("policy",("docling",),dict(max_batch_files=1,prefer_safe_backend=False))
    assert calls[1][1]["enable_ocr"] and not calls[1][1]["force_full_page_ocr"]
    assert isinstance(calls[1][1]["ocr_options"],GlossapiOcrOptions)


def test_checkpoint_pin_rejects_changed_artifact(tmp_path):
    pytest.importorskip("docling")
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from glossapi.ocr.plugin.model import GlossapiOcrModel
    from glossapi.ocr.plugin.options import GlossapiOcrOptions
    checkpoint = tmp_path/"changed.pt"; checkpoint.write_bytes(b"changed")
    options = GlossapiOcrOptions(checkpoint=checkpoint,codec=Path("unused.json"),
                                replace_policy="all",checkpoint_sha256="0"*64)
    model = GlossapiOcrModel(enabled=True,artifacts_path=None,options=options,
                            accelerator_options=AcceleratorOptions(device="cpu"))
    with pytest.raises(ValueError,match="sha256 mismatch"):
        model._get_recognizer()


def score_lines(predictions, gt_cells):
    """One-to-one rectangle match (0.01 PDF points); expose coverage separately."""
    from docling_core.types.doc.page import BoundingRectangle
    from glossapi.ocr.plugin.recognizer.codec import levenshtein
    remaining = list(gt_cells)
    edits = chars = matched = 0
    unmatched = []
    for prediction in predictions:
        bb = BoundingRectangle(**prediction["source_rect"]).to_bounding_box().as_tuple()
        matches = [c for c in remaining if max(abs(a-b) for a,b in
                   zip(bb,c.rect.to_bounding_box().as_tuple())) <= .01]
        if len(matches) != 1:
            unmatched.append(prediction["source_rect"])
            continue
        cell = matches[0]
        remaining.remove(cell)
        ref, hyp = normalized(cell.text), normalized(prediction["text"])
        edits += levenshtein(ref,hyp)
        chars += len(ref)
        matched += 1
    return dict(cer=edits/chars if chars else None, edits=edits, reference_chars=chars,
                matched_lines=matched, unmatched_prediction_rects=unmatched,
                unmatched_gt_lines=len(remaining), gt_line_count=len(gt_cells))


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("GLOSSAPI_RUN_C1") != "1", reason="explicit local C1 cohort run only")
def test_real_corpus_extract_cohort(tmp_path, monkeypatch):
    pytest.importorskip("docling")
    import torch
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat, ConversionStatus
    from docling.datamodel.document import InputDocument
    from glossapi.corpus import Corpus
    from glossapi.gloss_extract import GlossExtract
    import glossapi.gloss_extract as extract_module
    from glossapi.ocr.plugin.model import GlossapiOcrModel

    cohort_path = DOCS / "C1_COHORT.json"
    cohort = json.loads(cohort_path.read_text())
    config, pages = cohort["recognizer"], cohort["pages"]
    assert Counter(p["role"] for p in pages) == {"routed":24,"clean":6,"english":2}
    assert cohort["cohort_rule"] == "exclude o5 train+val; reserved eligible"
    assignment = cohort["exclusion_source"]["document_assignment"]
    excluded = set(assignment["train"]) | set(assignment["val"])
    assert not ({p["document_id"] for p in pages} & excluded)
    assert all(p["document_id"] in assignment["reserved"] for p in pages if p["role"] != "english")
    assert max(Counter(p["document_id"] for p in pages if p["role"] == "routed").values()) <= 2
    assert sha(cohort["exclusion_source"]["path"]) == cohort["exclusion_source"]["sha256"]
    assert sha(cohort["routing_source"]["path"]) == cohort["routing_source"]["sha256"]
    assert sha(config["checkpoint"]) == config["sha256"]
    assert sha(ROOT/config["codec"]) == config["codec_sha256"]
    test_factory_registration()  # assertion (a), also part of this real-run preflight
    blockers = []
    if int(importlib.metadata.version("pypdfium2").split(".")[0]) >= 5:
        blockers.append(".venv-docling requires pypdfium2<5; installed " + importlib.metadata.version("pypdfium2"))
    for p in pages:
        assert sha(p["pdf"]) == p["pdf_sha256"]
        if not Path(p["gt_source"]["pdf"]).is_file():
            blockers.append(f"missing disjoint repaired GT: {p['document_id']} page {p['page']}")
        else:
            assert p["gt_source"]["sha256"] and sha(p["gt_source"]["pdf"]) == p["gt_source"]["sha256"]
    if blockers:
        results = dict(status="blocked_before_cohort_run", cohort_sha256=sha(cohort_path),
                       blockers=blockers, cohort_pages=32, routed_pages=24, pages_processed=0,
                       mean_cer=None, clean_identical=None,
                       per_page=[dict(document_id=p["document_id"],page=p["page"],role=p["role"],
                                      cer=None,status="not_run") for p in pages])
        (DOCS/"C1_RESULTS.json").write_text(json.dumps(results,indent=2)+"\n")
        pytest.fail("C1 preflight blocked before rendering/inference:\n" + "\n".join(blockers))

    monkeypatch.setenv("HF_HUB_OFFLINE","1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE","1")
    monkeypatch.setenv("PIP_NO_INDEX","1")
    # Enforce the no-network contract even if a dependency ignores offline flags.
    import socket
    def no_network(*args, **kwargs):
        raise RuntimeError("C1 forbids network connections")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    torch.set_num_threads(2)
    def thread_settings(device="cpu"):
        return dict(device=str(device), torch_num_threads=torch.get_num_threads(),
                    torch_num_interop_threads=torch.get_num_interop_threads(),
                    mkl_available=torch.backends.mkl.is_available(),
                    thread_environment={k:os.getenv(k) for k in ["MKL_NUM_THREADS", "MKL_DYNAMIC",
                        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]})
    initial_thread_settings = thread_settings()
    from glossapi.ocr.plugin.recognizer.inference import Recognizer
    recognize = Recognizer.recognize
    def measured_recognize(self, strips, *args, **kwargs):
        current["capture"]["recognition_execution"] = thread_settings(self.device)
        return recognize(self, strips, *args, **kwargs)
    monkeypatch.setattr(Recognizer, "recognize", measured_recognize)
    from glossapi.ocr.docling.pipeline import build_layout_pipeline
    artifacts = Path.home()/".cache/docling/models"
    assert artifacts.is_dir(), "offline Docling layout/table artifacts missing"
    def offline_pipeline(**kwargs):
        return build_layout_pipeline(**kwargs,options_hook=lambda opts:setattr(opts,"artifacts_path",artifacts))
    monkeypatch.setattr(extract_module,"build_layout_pipeline",offline_pipeline)
    route = tmp_path/"routing.jsonl"
    grouped = {}
    for p in pages:
        grouped.setdefault(Path(p["pdf"]).name,[]).append(p["routing_metrics"])
    route.write_text("".join(json.dumps(dict(filename=name,pages=ps))+"\n" for name,ps in grouped.items()))
    call_log = tmp_path/"calls.jsonl"
    # Scope conversion to frozen pages without manufacturing subset PDFs or replacing
    # the converter. Corpus.extract, its backend, OCR, layout and export are all real.
    current = {}
    convert = GlossExtract._convert_with_timeout
    conversions = Counter()
    def one_page(self,file,timeout_s,**kwargs):
        key = (current["kind"], Path(file).name, current["page"])
        conversions[key] += 1
        if conversions[key] > 1:
            raise RuntimeError(f"C1 forbids a second conversion attempt for {key}")
        start = perf_counter()
        capture = current["capture"] = dict(page=current["page"], kind=current["kind"])
        try:
            result = convert(self,file,timeout_s,page_range=(current["page"],current["page"]),**kwargs)
            capture.update(conversion_status=result.status.value,
                           conversion_errors=[e.model_dump(mode="json") for e in result.errors],
                           converted_pages=len(result.pages),
                           layout_s=float(result.timings["layout"].total()) if "layout" in result.timings else None)
            return result
        except Exception as exc:
            capture["conversion_exception"] = dict(type=type(exc).__name__,message=str(exc))
            raise
        finally:
            capture["conversion_wall_s"] = perf_counter()-start
    monkeypatch.setattr(GlossExtract,"_convert_with_timeout",one_page)
    export = GlossExtract._export_documents
    def time_export(self,results,output_dir):
        start = perf_counter()
        result = export(self,results,output_dir)
        current["capture"]["export_s"] = perf_counter()-start
        return result
    monkeypatch.setattr(GlossExtract,"_export_documents",time_export)
    # Reuse models for this one diagnostic run; production release behavior stays unchanged.
    release = GlossExtract.release_resources
    monkeypatch.setattr(GlossExtract,"release_resources",lambda self:None)
    observed = []
    original_call = GlossapiOcrModel.__call__
    def observe(self,conv_res,batch):
        batch = list(batch)
        before = {p.page_no:p.parsed_page.model_dump(mode="json") for p in batch}
        for page in original_call(self,conv_res,batch):
            observed.append(dict(filename=conv_res.input.file.name,page=page.page_no,
                                 before=before[page.page_no],after=page.parsed_page.model_dump(mode="json")))
            yield page
    monkeypatch.setattr(GlossapiOcrModel,"__call__",observe)
    empty_input = tmp_path/"empty"; empty_input.mkdir()
    engine = Corpus(empty_input,tmp_path/"engine")
    baseline = Corpus(empty_input,tmp_path/"baseline")
    # The legacy CPU Docling prime guard requires CUDA. Prime the baseline explicitly
    # using GlossExtract's real converter, then exercise Corpus.extract(_prepared=True).
    baseline.extractor = GlossExtract()
    baseline.extractor.enable_accel(threads=2,type="CPU")
    baseline.extractor.configure_batch_policy("docling",max_batch_files=1,prefer_safe_backend=False)
    baseline.extractor.repair_fonts = False
    baseline.extractor.export_doc_json = False
    baseline.extractor.ensure_extractor(enable_ocr=False,profile_timings=True)
    per_page = []
    run_start, cpu_start = perf_counter(), process_time()

    def read_calls():
        return [json.loads(line) for line in call_log.read_text().splitlines()] if call_log.exists() else []

    def summary(complete):
        calls = read_calls()
        supported = [p for p in per_page if p["status"] == "replaced" and p["cer"] is not None]
        replaced = [p for p in per_page if p["status"] == "replaced"]
        recognised_lines = sum(len(c.get("lines", [])) for c in calls)
        expected = Counter((Path(p["pdf"]).name,p["page"]) for p in pages if p["role"] == "routed")
        invocation_ok = Counter((c["filename"],c["page"]) for c in calls) == expected
        stage_totals = {k:sum(c["timings_s"].get(k,0) for c in calls)
                        for k in ["recognizer_setup", "render", "crop", "recognise"]}
        stage_totals.update({k:sum(p.get(k) or 0 for p in per_page)
                            for k in ["layout_s", "export_s", "conversion_wall_s", "total_extract_wall_s"]})
        baseline_totals = {k:sum(p.get("baseline",{}).get(k) or 0 for p in per_page)
                           for k in ["layout_s", "export_s", "conversion_wall_s", "total_extract_wall_s"]}
        failures = [dict(document_id=p["document_id"],page=p["page"],errors=p["errors"])
                    for p in per_page if p["errors"]]
        receipt = DOCS/"C1_CALLS.jsonl"
        receipt.write_text(call_log.read_text() if call_log.exists() else "")
        observations_path = tmp_path/"observations.json"
        observations_path.write_text(json.dumps(observed,ensure_ascii=False)+"\n")
        result = dict(
            schema_version=3, status=("passed" if not failures and invocation_ok else "completed_with_failures")
            if complete else "running", cohort_rule=cohort["cohort_rule"],cohort_sha256=sha(cohort_path),
            cohort_pages=len(pages),pages_attempted=len(per_page),
            pages_processed=sum(p.get("conversion_status")==ConversionStatus.SUCCESS.value for p in per_page),
            routed_pages=24,routed_attempts=len(calls),
            replaced_pages=sum(c["status"]=="replaced" for c in calls),scored_pages=len(supported),
            unsupported_pages=sum(c["status"]=="unsupported" for c in calls),
            unsupported_page_list=[dict(filename=c["filename"],page=c["page"],reason=c["reason"],
                                       source_line_count=c.get("source_line_count"),
                                       recognition_performed=c.get("recognition_performed",False),
                                       body_support=c.get("body_support"))
                                   for c in calls if c["status"]=="unsupported"],
            cer_scope="Replaced native-body pages only; unsupported furniture scores are separate diagnostics",
            mean_cer=sum(p["cer"] for p in supported)/len(supported) if supported else None,
            micro_cer=sum(p["edits"] for p in supported)/sum(p["reference_chars"] for p in supported) if supported else None,
            clean_identical=sum(p.get("clean_identical") is True for p in per_page),
            clean_compared=sum("clean_identical" in p for p in per_page),
            unsupported_counts=dict(Counter(c["reason"] for c in calls if c["status"]=="unsupported")),
            runtime={k:importlib.metadata.version(k) for k in ["docling","pypdfium2","torch"]},
            execution=dict(device="cpu",threads=2,crop_scale=config["crop_scale"],network="socket connections disabled",
                           initial_thread_settings=initial_thread_settings,
                           effective_recognition_settings=[dict(json.loads(s)) for s in sorted({
                               json.dumps(p["recognition_execution"],sort_keys=True)
                               for p in per_page if "recognition_execution" in p})],
                           cohort_passes=1,baseline_pages=6,
                           note="Native page_range; real OCR, layout, tables and export; test-only converter reuse"),
            timings=dict(engine_stage_elapsed_s=stage_totals,baseline_stage_elapsed_s=baseline_totals,
                         recognised_lines=recognised_lines,
                         recognition_ms_per_line=1000*stage_totals["recognise"]/recognised_lines if recognised_lines else None,
                         run_wall_s=perf_counter()-run_start,run_process_cpu_s=process_time()-cpu_start,
                         measurement="perf_counter elapsed seconds on CPU; stage sums can overlap"),
            assertions=dict(a_factory_registered=True,b_routed_invocations_exact=invocation_ok,
                            c_geometry_counts=all(p.get("assertions",{}).get("c_geometry_counts",True) for p in per_page),
                            d_clean_identical=sum(p.get("clean_identical") is True for p in per_page)==6,
                            e_replacement_and_complete_cer=all(p.get("assertions",{}).get("e_text_changed",True)
                                and p.get("assertions",{}).get("e_complete_gt_rect_match",True) for p in replaced),
                            f_export=all(p["assertions"].get("f_markdown_contains_plugin_text",False) for p in replaced),
                            all_pages_exported=all(p["assertions"].get("export_succeeded",False) for p in per_page)),
            failures=failures,blockers=[] if invocation_ok else ["Selected-page invocation audit incomplete or duplicated"],
            call_log=dict(path=str(receipt.relative_to(ROOT)),sha256=sha(receipt)),
            observations=dict(path=str(observations_path),sha256=sha(observations_path)),
            artifact_dir=str(tmp_path),per_page=per_page)
        (DOCS/"C1_RESULTS.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
        return result

    try:
        for p in pages:
            current.update(page=p["page"],kind="engine",capture={})
            key = f"{p['document_id']}-{p['page']}"
            engine.markdown_dir = tmp_path/"engine"/key
            row = dict(document_id=p["document_id"],page=p["page"],role=p["role"],decision=p["decision"],
                       cer=None,status="failed",assertions={},errors=[])

            def check(name,condition,message):
                row["assertions"][name] = bool(condition)
                if not condition:
                    row["errors"].append(dict(type="AssertionError",assertion=name,message=message))

            before_observed, before_calls = len(observed), len(read_calls())
            start = perf_counter()
            try:
                engine.extract(input_format="pdf",file_paths=[p["pdf"]],num_threads=2,accel_type="CPU",
                               batched=False,skip_existing=False,repair_fonts=False,export_doc_json=False,
                               ocr_engine="glossapi",ocr_checkpoint=config["checkpoint"],
                               ocr_codec=ROOT/config["codec"],ocr_routing_sidecar=route,
                               ocr_crop_scale=config["crop_scale"],ocr_device="cpu",
                               ocr_checkpoint_sha256=config["sha256"],ocr_call_log=call_log)
                row["total_extract_wall_s"] = perf_counter()-start
                row.update(current["capture"])
                md_path = engine.markdown_dir/f"{p['document_id']}.md"
                check("export_succeeded",row.get("conversion_status")==ConversionStatus.SUCCESS.value
                      and row.get("converted_pages")==1 and md_path.is_file(),"Conversion/export failed")
                md = md_path.read_bytes() if md_path.is_file() else b""
                row.update(markdown_sha256=hashlib.sha256(md).hexdigest(),markdown_bytes=len(md))
                observations = observed[before_observed:]
                check("plugin_page_observed",len(observations)==1,"Expected one actual plugin page observation")
                calls = read_calls()[before_calls:]
                if p["role"] == "routed":
                    check("b_page_invocation",len(calls)==1 and
                          (calls[0]["filename"],calls[0]["page"])==(Path(p["pdf"]).name,p["page"]),
                          "Expected exactly one routed attempt")
                else:
                    check("b_unrouted_no_call",not calls,"Unrouted page logged as selected")
                if len(observations)!=1 or (p["role"]=="routed" and len(calls)!=1):
                    raise RuntimeError("Conversion produced no unique page/call evidence")
                observation = observations[0]
                check("observed_page_identity",(observation["filename"],observation["page"])==
                      (Path(p["pdf"]).name,p["page"]),"Observed page is not the frozen page")
                if p["role"] == "routed":
                    call = calls[0]
                    row.update(status=call["status"],timings_s=call["timings_s"],
                               source_line_count=call.get("source_line_count"),body_support=call.get("body_support"))
                    if call["status"] == "unsupported":
                        row["reason"] = call["reason"]
                        check("unsupported_unchanged",observation["before"]==observation["after"],
                              "Unsupported page changed")
                        if call["reason"] == "raster_body":
                            check("unsupported_furniture_recognised",call.get("recognition_performed") is True
                                  and not call["installed"] and len(call["lines"])==call["source_line_count"],
                                  "Raster-body page lacks existing-line recognition diagnostics")
                    elif call["status"] == "replaced":
                        before = observation["before"]["textline_cells"]
                        after = observation["after"]["textline_cells"]
                        check("c_geometry_counts",len(before)==len(after)==call["source_line_count"]==call["output_line_count"]
                              and [c["rect"] for c in before]==[c["rect"] for c in after],
                              "Source/output counts or rectangles differ")
                        check("c_ocr_cells",all(c["from_ocr"] for c in after)
                              and all(c["from_ocr"] for c in observation["after"]["word_cells"])
                              and all(c["from_ocr"] for c in observation["after"]["char_cells"]),
                              "Native line/word/character cells remain")
                        check("e_text_changed",normalized(" ".join(c["text"] for c in before))!=
                              normalized(" ".join(c["text"] for c in after)),"Replacement equals native text")
                        row["markdown_witness"] = markdown_witness(call["lines"], md.decode())
                        check("f_markdown_contains_plugin_text",row["markdown_witness"] is not None,
                              "Markdown lacks changed, distinctive recognised text")
                    else:
                        raise RuntimeError(f"Unexpected OCR status: {call['status']}")
                    if call.get("recognition_performed"):
                        row["recognition_ms_per_line"] = 1000*call["timings_s"]["recognise"]/len(call["lines"])
                        gt = InputDocument(path_or_stream=Path(p["gt_source"]["pdf"]),format=InputFormat.PDF,
                                           backend=DoclingParseDocumentBackend)
                        backend = gt._backend.load_page(p["page"]-1)
                        try:
                            score = score_lines(call["lines"],list(backend.get_text_cells()))
                        finally:
                            backend.unload(); gt._backend.unload()
                        complete_match = (score["matched_lines"]==call["source_line_count"]
                                          and score["unmatched_gt_lines"]==0 and score["cer"] is not None)
                        if call["status"] == "replaced":
                            row.update(score)
                            check("e_complete_gt_rect_match",complete_match,"GT rect match incomplete; full-page CER unavailable")
                            if not complete_match:
                                row["matched_line_cer"], row["cer"] = row["cer"], None
                        else:
                            row["recognition_diagnostic"] = dict(scope="Existing furniture lines only; no raster-body CER", **score)
                            check("diagnostic_complete_gt_rect_match",complete_match,"Furniture GT rect match incomplete")
                else:
                    row["status"] = "untouched"
                    check("unrouted_cells_unchanged",observation["before"]==observation["after"],"Unrouted parsed cells changed")
                    if p["role"] == "english":
                        row["readable_english_sample"] = normalized(" ".join(
                            c["text"] for c in observation["before"]["textline_cells"]))[:500]
                        check("english_text_present",len(row["readable_english_sample"])>=100,"English control lacks readable text")
            except Exception as exc:
                row["errors"].append(dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            finally:
                row.setdefault("total_extract_wall_s",perf_counter()-start)
                row.update(current["capture"])

            # Run every clean baseline even if its plugin extraction/contract failed.
            if p["role"] == "clean":
                current.update(kind="baseline",capture={})
                baseline.markdown_dir = tmp_path/"baseline"/key
                baseline_start = perf_counter()
                try:
                    baseline.extract(input_format="pdf",file_paths=[p["pdf"]],num_threads=2,accel_type="CPU",
                                     phase1_backend="docling",batched=False,skip_existing=False,
                                     repair_fonts=False,export_doc_json=False,_prepared=True)
                    clean_md = (baseline.markdown_dir/f"{p['document_id']}.md").read_bytes()
                    row["baseline_markdown_sha256"] = hashlib.sha256(clean_md).hexdigest()
                    engine_md = (engine.markdown_dir/f"{p['document_id']}.md").read_bytes()
                    row["clean_identical"] = clean_md==engine_md
                    check("d_clean_byte_identity",row["clean_identical"],"Clean Markdown bytes differ")
                except Exception as exc:
                    row["errors"].append(dict(type=type(exc).__name__,stage="clean_baseline",
                                              message=str(exc),traceback=traceback.format_exc()))
                finally:
                    row["baseline"] = dict(current["capture"],total_extract_wall_s=perf_counter()-baseline_start)
            per_page.append(row)
            summary(complete=False)  # durable partial measurements if a later page fails
            print(f"C1 {key}: {row['status']} CER={row['cer']} errors={len(row['errors'])}",flush=True)
        result = summary(complete=True)
    finally:
        for corpus in [engine,baseline]:
            if corpus.extractor is not None:
                release(corpus.extractor)
    if result["status"] != "passed":
        pytest.fail(f"C1 completed all {len(per_page)} pages; {len(result['failures'])} page failures. "
                    "Measurements and exceptions saved in C1_RESULTS.json")
