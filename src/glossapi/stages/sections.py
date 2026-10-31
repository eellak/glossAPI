"""Existing section parser and frozen-order compatibility aggregate serializer."""
from __future__ import annotations
from .execution import write_text_hashed

import hashlib
import json
import os
from pathlib import Path

from .contracts import PackageRequirement, PlannedOperation, StagePlan, StageSpecification
from .execution import failed_result, freeze_inputs
from .products import execute_aggregate, generation_outputs, publish_products, resolved_input


def restore_clean_selection(corpus):
    """Reconstruct public selection/text views from successful producer facts."""
    from .products import active_journal,local_path
    from ..ledger.encoding import unpack_evidence
    from .execution import sha256_file
    journal=active_journal()
    if not journal or not journal.cutover:return
    selected={};surfaces={}
    for encoded,settings in journal.writer.connection.execute(f"SELECT e.details_json,c.json FROM {journal.writer.connection.event_table} e JOIN tasks t ON t.task_id=e.task_id JOIN attempts a ON a.attempt_id=e.attempt_id JOIN snapshots c ON c.sha256=a.settings_sha256 WHERE t.stage='clean' AND t.operation LIKE 'clean.selection%' AND e.event_kind='decision' AND EXISTS (SELECT 1 FROM {journal.writer.connection.event_table} done WHERE done.attempt_id=a.attempt_id AND done.event_kind='lifecycle' AND done.status='ok') ORDER BY e.event_id"):
        digest=json.loads(encoded).get('evidence_sha256')
        row=journal.writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()
        if row is None:continue
        decision=unpack_evidence(json.loads(row[0]))
        if 'selected_file_id' not in decision:continue
        fact=journal.writer.connection.execute('SELECT uri,sha256 FROM files WHERE file_id=?',(decision['selected_file_id'],)).fetchone()
        path=local_path(fact['uri'])
        if not path.is_file() or sha256_file(path)!=decision['selected_text_sha256']:continue
        filename=Path(json.loads(settings)['compatibility_filename']).stem
        selected[filename]=decision['accepted_for_sections'];surfaces[filename]=path
    if selected:
        corpus.good_files=[name for name,accepted in selected.items() if accepted]
        roots={path.parent for path in surfaces.values()}
        if len(roots)==1:corpus.markdown_dir=next(iter(roots))



class SectionsStage:
    name='sections'
    contract_version=1

    def __init__(self,sectioner=None):
        self.sectioner=sectioner

    def packages(self,specification):
        return (PackageRequirement('core',('pandas>=3.0.0','pyarrow>=26,<27')),)

    def plan(self,documents,*,settings):
        return StagePlan(tuple(PlannedOperation(d.document_id,d.version_id,
            StageSpecification('sections.document',('selected_text',),('section_fragment',),settings=settings)) for d in documents),
            ('frozen_section_enumeration_and_global_ids',))

    def run_document(self,request,*,reporter):
        request.work_directory.mkdir(parents=True,exist_ok=True)
        try:
            request=freeze_inputs(request)
            source=request.inputs[0]
            from ..gloss_section import GlossSection
            sectioner=self.sectioner or GlossSection()
            rows=sectioner.rows_for_document(source.path.read_text(encoding='utf-8'),request.settings['filename'])
            parser_hash=hashlib.sha256(Path(__import__('glossapi.gloss_section',fromlist=['__file__']).__file__).read_bytes()).hexdigest()
            keys=[hashlib.sha256(f'{source.sha256}:{parser_hash}:{ordinal}'.encode()).hexdigest() for ordinal in range(len(rows))]
            write_text_hashed(request.work_directory/'rows.json',json.dumps(rows,ensure_ascii=False))
            return generation_outputs(request,[('rows.json','section_fragment')],reason='sections_completed' if rows else 'sections_empty',
                decisions=({'input_file_id':source.file_id,'input_text_sha256':source.sha256,'parser_sha256':parser_hash,
                    'section_keys':keys,'section_count':len(rows),'compatibility_filename':request.settings['filename']},))
        except Exception as error:
            return failed_result(request,'sections_exception',error)

    def finalize(self,request,*,reporter):
        request.work_directory.mkdir(parents=True,exist_ok=True)
        try:
            request=freeze_inputs(request)
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            from ..gloss_section import section_schema
            row_counter=1; groups=[]
            with pq.ParquetWriter(request.work_directory/'sections_for_annotation.parquet',schema=section_schema()) as writer:
                for source in request.inputs:
                    rows=json.loads(source.path.read_text())
                    for row in rows:
                        row['id']=row_counter;row['row_id']=f'row_{row_counter}';row_counter+=1
                    if rows:
                        writer.write_table(pa.Table.from_pandas(pd.DataFrame(rows),schema=section_schema(),preserve_index=False))
                        groups.append({'filename':rows[0]['filename'],'first_id':rows[0]['id'],'rows':len(rows)})
            return generation_outputs(request,[('sections_for_annotation.parquet','sections_parquet')],reason='sections_completed',
                decisions=({'enumeration':request.settings['enumeration'],'selection':request.settings['selection'],
                    'row_groups':groups,'row_count':row_counter-1,'numeric_ids':'compatibility_only'},))
        except Exception as error:
            return failed_result(request,'sections_exception',error)


def section_products(corpus,good_filenames):
    from .compatibility import execute_compatibility_stage
    # Preserve os.listdir traversal as a frozen input, never sort it here.
    enumeration=[name for name in os.listdir(corpus.markdown_dir) if name.endswith('.md')]
    fragments=[];temporary_workspaces=[]
    for name in enumeration:
        if Path(name).stem not in good_filenames:
            from ..ledger.journal import observe
            observe('section_not_selected',path=corpus.markdown_dir/name,status='skipped')
            continue
        source=resolved_input(corpus,corpus.markdown_dir/name)
        result,workspace=execute_compatibility_stage(corpus,SectionsStage(corpus.sectioner),[('selected_text',source)],
            StageSpecification('sections.document',('selected_text',),('section_fragment',),settings={'filename':Path(name).stem}))
        if result.status!='ok': raise RuntimeError(result.error.message)
        fragments.append(('section_fragment',workspace/'rows.json'));temporary_workspaces.append(workspace)
    result,workspace=execute_aggregate(corpus,SectionsStage(),fragments,
        StageSpecification('sections.publish',tuple(role for role,path in fragments),('sections_parquet','generation_manifest'),
            settings={'enumeration':enumeration,'selection':list(good_filenames)}))
    publish_products(corpus,result,workspace,{'sections_for_annotation.parquet':corpus.sections_parquet})
    from .products import active_journal
    if not active_journal():
        import shutil
        for temporary in temporary_workspaces: shutil.rmtree(temporary.parent)
