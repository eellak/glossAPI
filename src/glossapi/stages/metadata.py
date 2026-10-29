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
    for path in sorted(paths):
        from .execution import sha256_file
        digest=sha256_file(path)
        current=current_product(journal.writer.connection,path)
        if current and current['sha256']==digest:continue
        result,workspace=execute_aggregate(journal.corpus,MetadataProjectionStage(journal.stage),[('compatibility_serialization',path)],
            StageSpecification(journal.stage+'.metadata:'+path.name,('compatibility_serialization',),('metadata_projection','generation_manifest')))
        publish_products(journal.corpus,result,workspace,{'metadata.parquet':path})


def restore_metadata_projections(journal):
    from .execution import sha256_file
    for encoded, in journal.writer.connection.execute("SELECT json FROM snapshots WHERE kind='product_projection' ORDER BY rowid DESC"):
        import json
        product=json.loads(encoded);path=local_path(product['product_uri'])
        if path.parent!=journal.corpus.output_dir.absolute()/'download_results' or path.suffix!='.parquet':continue
        current=current_product(journal.writer.connection,path)
        if current!=product:continue
        immutable=local_path(product['immutable_uri'])
        if not immutable.is_file() or sha256_file(immutable)!=product['sha256']:raise ValueError('Metadata projection corrupt')
        if not path.exists() or sha256_file(path)!=product['sha256']:
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(immutable,path)
