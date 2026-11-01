"""Compatibility metadata is a registered projection, never a mutable checkpoint."""
from pathlib import Path
import shutil

from .contracts import StageSpecification
from .execution import freeze_inputs
from .products import execute_aggregate, generation_outputs, publish_products, current_product, local_path


class MetadataProjectionStage:
    contract_version=1

    def __init__(self,stage):self.name=stage

    def finalize(self,request,*,reporter):
        request.work_directory.mkdir(parents=True,exist_ok=True)
        request=freeze_inputs(request)
        source=request.inputs[0]
        shutil.copyfile(source.path,request.work_directory/'metadata.parquet')
        return generation_outputs(request,[('metadata.parquet','metadata_projection')],
            decisions=({'projection_of_stage':self.name,'input_sha256':source.sha256,
                'authority':'immutable registered generation; compatibility columns are serializer products'},))


def seal_metadata_projections(journal):
    paths=list((journal.corpus.output_dir/'download_results').glob('*.parquet'))
    report=journal.corpus.output_dir/'cleaning_report.parquet'
    if report.is_file() and 'clean' in journal.adapter_stages and report not in getattr(journal,'cleanup_paths',()):paths.append(report)
    from ..parquet_schema import metadata_target
    for path in sorted(paths):
        from .execution import sha256_file
        # Through the metadata choke point (CP-K2a F1, K2F round 3): the projection copies the file and publishes
        # those bytes back to its resolved physical target while holding its lock, so no locked writer commits in
        # between and a symlinked metadata file keeps its link; the ledger records the path as found.
        with metadata_target(path) as written:
            if not written.is_file():continue
            digest=sha256_file(written)
            current=current_product(journal.writer.connection,path)
            if current and current['sha256']==digest:continue
            result,workspace=execute_aggregate(journal.corpus,MetadataProjectionStage(journal.stage),[('compatibility_serialization',path)],
                StageSpecification(journal.stage+'.metadata:'+path.name,('compatibility_serialization',),('metadata_projection','generation_manifest')))
            publish_products(journal.corpus,result,workspace,{'metadata.parquet':path},physical={'metadata.parquet':written})


def restore_metadata_projections(journal):
    from .execution import sha256_file
    for encoded, in journal.writer.connection.execute("SELECT json FROM snapshots WHERE kind='product_projection' ORDER BY rowid DESC"):
        import json
        product=json.loads(encoded);path=local_path(product['product_uri'])
        if path.parent!=journal.corpus.output_dir.absolute()/'download_results' or path.suffix!='.parquet':continue
        current=current_product(journal.writer.connection,path)
        if current!=product:continue
        immutable=local_path(product['immutable_uri'])
        if not immutable.is_file() or sha256_file(immutable,own=True)!=product['sha256']:raise ValueError('Metadata projection corrupt')
        # Restoring the committed projection is a deliberate rollback of uncommitted bytes (cutover profile). It is
        # compared and replaced inside the metadata lock, atomically, so a reader never sees a partial copy and a
        # locked writer's read-modify-write is never interleaved with it (CP-K2a F1).
        from ..parquet_schema import _copy_file_locked, metadata_target
        with metadata_target(path) as written:
            if not written.exists() or sha256_file(written)!=product['sha256']:
                _copy_file_locked(immutable,written)
