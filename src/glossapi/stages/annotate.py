"""Batch prediction barrier, unchanged classifier kernels, private label lineage."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .contracts import PackageRequirement, StagePlan, StageSpecification, StageResult
from .execution import failed_result, freeze_inputs, sha256_file
from .products import execute_aggregate, generation_outputs, publish_products, resolved_input
from .receipts import capture_backend_observations


class AnnotateStage:
    name='annotate'
    contract_version=1

    def __init__(self,corpus=None,section_keys=None):
        self.corpus=corpus
        self.section_keys=section_keys or {}

    def packages(self,specification):
        return (PackageRequirement('annotate',('scikit-learn==1.6.1','joblib>=1.0.0')),)

    def plan(self,documents,*,settings):
        return StagePlan((),('annotation_all_sections_and_model',))

    def run_document(self,request,*,reporter):
        raise ValueError('Compatibility prediction requires the annotation batch barrier')

    def finalize(self,request,*,reporter):
        request.work_directory.mkdir(parents=True,exist_ok=True)
        try:
            request=freeze_inputs(request)
            from ..corpus.phase_annotate import AnnotatePhaseMixin
            import pandas as pd
            view=copy.copy(self.corpus)
            view._inside_product_stage=True
            view.annotation_mapping=dict(request.settings['mapping'])
            view.filename_to_doctype=dict(request.settings['document_types'])
            view.sections_parquet=request.inputs[0].path
            view.section_classifier_model_path=request.inputs[1].path
            view.classified_parquet=request.work_directory/'classified_sections.parquet'
            view.fully_annotated_parquet=request.work_directory/'fully_annotated_sections.parquet'
            # Every generation starts without a numeric-ID checkpoint. A changed
            # model or input always executes and receives a distinct fingerprint.
            with capture_backend_observations() as observations:
                AnnotatePhaseMixin.annotate.__wrapped__(view,annotation_type=request.settings['annotation_type'],
                    fully_annotate=request.settings['fully_annotate'])
            self.corpus.classifier=view.classifier
            classified=pd.read_parquet(view.classified_parquet)
            raw=pd.read_parquet(str(view.classified_parquet)+'.tmp')
            initial={(str(row.filename),int(row.id)):str(row.predicted_section) for row in raw.itertuples()}
            final=pd.read_parquet(view.fully_annotated_parquet) if view.fully_annotated_parquet.exists() else classified
            labels={(str(row.filename),int(row.id)):str(row.predicted_section) for row in final.itertuples()}
            fallbacks={int(value) for item in observations if item.get('reason_code')=='annotation_prediction_fallback'
                for value in item.get('details',{}).get('row_ids',[])}
            source_rows=pd.read_parquet(request.inputs[0].path)
            parser_fingerprint=sha256_file(Path(__import__('glossapi.gloss_section',fromlist=['__file__']).__file__))
            ordinals={};keys={}
            for row in source_rows.itertuples():
                filename=str(row.filename);ordinal=ordinals.get(filename,0);ordinals[filename]=ordinal+1
                supplied=self.section_keys.get(filename,[])
                keys[(filename,int(row.id))]=supplied[ordinal] if ordinal<len(supplied) else hashlib.sha256(
                    f'{request.inputs[0].sha256}:{parser_fingerprint}:{filename}:{ordinal}'.encode()).hexdigest()
            lineage=[]
            from ..ledger.encoding import snapshot_hash
            mapping_hash=snapshot_hash({'mapping':request.settings['mapping'],'document_types':request.settings['document_types']})
            for row in classified.itertuples():
                identity=(str(row.filename),int(row.id))
                requested=request.settings['annotation_type']
                effective=request.settings['mapping'].get(request.settings['document_types'].get(identity[0]),'text') if requested=='auto' else requested
                lineage.append({'section_key':keys[identity],'compatibility_id':int(row.id),'filename':identity[0],
                    'model_label':initial.get(identity),'model_sha256':request.inputs[1].sha256,'confidence':None,
                    'fallback_reason':'prediction_exception' if int(row.id) in fallbacks else None,
                    'index_label':str(row.predicted_section),'final_label':labels.get(identity),
                    'annotation_type':effective,'requested_annotation_type':requested,'mapping_sha256':mapping_hash,
                    'heuristic_rules':(['index_detection'] if str(row.predicted_section)=='π' else ['index_cumulative_limits']) if initial.get(identity)!=str(row.predicted_section) else [],
                    'full_label_rules':['full_annotation_'+effective] if labels.get(identity)!=str(row.predicted_section) else []})
            paths=[('classified_sections.parquet','classified_parquet')]
            if view.fully_annotated_parquet.exists():paths.append(('fully_annotated_sections.parquet','annotated_parquet'))
            paths.extend((str(path.relative_to(request.work_directory)),'annotation_partition')
                for path in sorted(Path(str(view.classified_parquet)+'.tmp').rglob('*')) if path.is_file())
            warnings=[item for item in observations if not item.get('details',{}).get('annotation_model_labels')]
            return generation_outputs(request,paths,reason='annotate_completed',decisions=({'sections_input_sha256':request.inputs[0].sha256,
                'model_sha256':request.inputs[1].sha256,'mapping':request.settings['mapping'],'document_types':request.settings['document_types'],
                'lineage':lineage,'backend_observations':warnings},))
        except Exception as error:
            return failed_result(request,'annotation_schema_invalid',error)


def annotate_products(corpus,annotation_type,fully_annotate):
    from ..ledger.journal import observe
    if corpus.classifier is None:
        from ..gloss_section_classifier import GlossSectionClassifier
        corpus.classifier=GlossSectionClassifier()
    if not corpus.sections_parquet.exists():
        observe('annotation_sections_missing',path=corpus.sections_parquet,status='skipped');return
    if not corpus.section_classifier_model_path.exists():
        observe('annotation_model_missing',path=corpus.section_classifier_model_path,status='skipped');return
    from .products import active_journal
    from ..ledger.encoding import unpack_evidence
    journal=active_journal();keys={};key_evidence={}
    section_input=resolved_input(corpus,corpus.sections_parquet)
    if journal:
        producer=journal.writer.connection.execute('SELECT producer_attempt_id FROM files WHERE uri=? AND sha256=? ORDER BY rowid DESC LIMIT 1',
            (section_input.as_uri(),sha256_file(section_input))).fetchone()
        if producer:
            for encoded, in journal.writer.connection.execute(f"SELECT e.details_json FROM attempt_inputs i JOIN files f ON f.file_id=i.file_id JOIN {journal.writer.connection.event_table} e ON e.attempt_id=f.producer_attempt_id WHERE i.attempt_id=? AND e.event_kind='decision'",(producer[0],)):
                digest=__import__('json').loads(encoded).get('evidence_sha256')
                row=journal.writer.connection.execute("SELECT json FROM snapshots WHERE sha256=? AND kind='stage_evidence'",(digest,)).fetchone()
                if row is None:continue
                decision=unpack_evidence(__import__('json').loads(row[0]))
                if decision.get('stage')=='sections' and 'section_keys' in decision:
                    keys[decision['compatibility_filename']]=decision['section_keys']
                    key_evidence[decision['compatibility_filename']]=digest
    result,workspace=execute_aggregate(corpus,AnnotateStage(corpus,keys),[
        ('sections_parquet',section_input),('classifier_model',corpus.section_classifier_model_path)],
        StageSpecification('annotate.batch',('sections_parquet','classifier_model'),('classified_parquet','generation_manifest'),
            settings={'annotation_type':annotation_type,'fully_annotate':fully_annotate,'mapping':corpus.annotation_mapping,
                'document_types':corpus.filename_to_doctype,'section_key_evidence':key_evidence,'batch_barrier':True}))
    if result.status!='ok':
        raise ValueError(result.error.message if result.error else result.reason_code)
    paths={output.relative_path:corpus.output_dir/output.relative_path for output in result.outputs
        if output.kind in ('classified_parquet','annotated_parquet','annotation_partition')}
    publish_products(corpus,result,workspace,paths)
