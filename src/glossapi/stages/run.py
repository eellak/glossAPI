"""Frozen run plans, streaming document work and compatibility barriers."""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlsplit

from .contracts import ArtifactReference, DocumentSnapshot, StageSpecification
from .execution import sha256_file
from .replay import committed_results
from .scheduler import ScheduledOperation, StreamingScheduler, local_path
from ..ledger.writer import LedgerWriter
from ..ledger.journal import shadow_stage


def extraction_settings(options=None):
    import os
    return {'phase1_backend':'safe','repair_fonts':True,'export_doc_json':True,'emit_formula_index':False,
        'benchmark_mode':False,'num_threads':1,'page_range':None,'long_pdf_page_threshold':600,'chunk_size':200,
        'max_chunk_timeouts':2,'per_page_timeout_s':20,'images_scale':float(os.environ.get('GLOSSAPI_IMAGES_SCALE','1.25')),'formula_enrichment':False,
        'code_enrichment':False,'use_cls':False,**(options or {})}


def _is_url(value):
    return isinstance(value,str) and value.startswith(('http://','https://'))


def _row_url(row):
    """The remote selector of a manifest row, or None (CP-LEDGER cycle 2, F3)."""
    if isinstance(row,str):
        return row if _is_url(row) else None
    if not isinstance(row,dict):
        return None
    url=row.get('url')
    if isinstance(url,(list,tuple)) or (isinstance(url,str) and url.strip()):
        return url
    selector=next((row[k] for k in ('document_id','source_key','source_uri','path') if isinstance(row.get(k),str) and row.get(k)),None)
    return selector if _is_url(selector) else None


def resource_identity(raw_url,*,source_id=None,dataset_id=None,resource_index=0):
    """Resource identity of one raw source URL (cycle-2 F2).

    A declared, non-secret ``source_id`` is the identity (stable across token
    rotation). Otherwise a URL whose persisted form loses nothing keeps
    ``url:v1:``; any other URL is keyed by an HMAC of the raw URL under the
    corpus identity key. Never the redacted spelling.
    """
    from ..ledger.journal import source_key
    if source_id is not None and not (isinstance(source_id,float) and source_id!=source_id):
        from ..ledger.identity import source_key as declared_key
        return declared_key(None,dataset_id=dataset_id if isinstance(dataset_id,str) and dataset_id else 'glossapi:declared_source_id',
            source_id=source_id,resource_index=resource_index)
    return source_key(raw_url)


def _row_identities(row,urls):
    """Identities of the URLs one manifest row names (list cells: one per URL)."""
    declared=row.get('source_id') if isinstance(row,dict) else None
    dataset=row.get('dataset_id') if isinstance(row,dict) else None
    return [resource_identity(url,source_id=declared,dataset_id=dataset,resource_index=index) for index,url in enumerate(urls)]


def _bind(found,identity,raw):
    from ..ledger.urls import register_url
    register_url(raw)  # exact secret components join the known-secret scrubber
    previous=found.get(identity)
    if previous is not None and previous!=raw:
        from ..ledger.urls import persisted_url
        raise ValueError('ambiguous_source_identity: two different source URLs map to one resource identity ('
            +persisted_url(raw)+'); give each resource its own source_id, or supply one URL per source_id')
    found[identity]=raw


def _transport_urls(value, found):
    """In-memory transport bindings ``resource identity -> raw URL`` (cycle-2 F2/F3).

    Accepts raw URL strings, rows ``{'url': ..., 'source_id': ...}``, manifest
    dicts ``{'documents': [...]}`` and lists of these. Nothing is recorded.
    """
    if isinstance(value, dict) and 'documents' in value and 'url' not in value:
        _transport_urls(value['documents'], found)
    elif isinstance(value, dict):
        url=_row_url(value)
        urls=[item for item in url if _is_url(item)] if isinstance(url,(list,tuple)) else [url] if _is_url(url) else []
        for identity,raw in zip(_row_identities(value,urls),urls):
            _bind(found,identity,raw)
    elif isinstance(value, (list, tuple)):
        for item in value:_transport_urls(item, found)
    elif _is_url(value):
        _bind(found,resource_identity(value),value)
    return found


def _corpus_download_names(root):
    """Download names this corpus already assigned: identity -> name base, and every taken base (K4BP-G4).

    Read from the recorded download results (the table the export joins a
    record's URL by) and the published ``downloads/`` files. A later run keeps
    a source's earlier name and never gives another source a name in use, so
    no record is exported under another URL's provenance.
    """
    import pandas as pd
    from .._naming import canonical_stem
    from ..gloss_downloader import frame_identities
    root=Path(root)
    by_identity,taken={},set()
    directory=root/'download_results'
    for path in sorted(directory.glob('*.parquet')) if directory.is_dir() else ():
        try:frame=pd.read_parquet(path)
        except Exception:continue
        if 'url' not in frame.columns or not {'filename_base','filename'}&set(frame.columns):continue
        bases=[str(value) if isinstance(value,str) and value else canonical_stem(str(name)) if isinstance(name,str) and name else ''
            for value,name in zip(frame['filename_base'] if 'filename_base' in frame.columns else ['']*len(frame),
                                  frame['filename'] if 'filename' in frame.columns else ['']*len(frame))]
        for identity,base in zip(frame_identities(frame,'url'),bases):
            if not base:continue
            taken.add(base)
            if identity and not path.name.endswith('.partial.parquet'):by_identity.setdefault(identity,base)
    downloads=root/'downloads'
    if downloads.is_dir():
        taken|={canonical_stem(path) for path in downloads.iterdir() if path.is_file()}
    return by_identity,taken


def _rename_against_corpus(frames,identities,root):
    """Re-derive ``filename_base`` in every frame from the corpus' earlier names (same rows, same identities)."""
    from ..gloss_downloader import GlossDownloader
    by_identity,taken=_corpus_download_names(root)
    if not taken:
        return False
    keep={by_identity[identity] for identity in identities if identity in by_identity}
    for frame in frames:
        naming=GlossDownloader.__new__(GlossDownloader)
        naming.url_column='url'
        naming._used_filename_bases=set(taken)-keep
        duplicate=frame['is_duplicate'].astype(bool).tolist() if 'is_duplicate' in frame.columns else [False]*len(frame)
        frame['filename_base']=['' if dup else by_identity.get(identity,'') for identity,dup in zip(identities,duplicate)]
        naming.ensure_filename_base(frame)
    return True


def _remote_inventory(rows,root):
    """Expand, deduplicate and name remote rows on their raw URLs, in memory.

    Returns the persisted rows (for the ledger), the raw URL and identity of
    every expanded row, and the persisted download-projection frame. Nothing
    raw is returned in a persisted value. The retained kernels perform no
    network or backend work.
    """
    import pandas as pd
    from ..gloss_downloader import GlossDownloader
    from ..ledger.urls import persisted_url, scrub
    def kernel(frame):
        naming=GlossDownloader.__new__(GlossDownloader)
        naming.url_column='url';naming._used_filename_bases=set()
        return naming.ensure_filename_base(naming._expand_and_mark_duplicates(frame))
    rows=[dict(row) if isinstance(row,dict) else {'url':row} for row in rows]
    for row in rows:
        if 'url' not in row or not isinstance(row['url'],(str,list,tuple)) or (isinstance(row['url'],str) and not row['url'].strip()):
            row['url']=_row_url(row)
    persisted=scrub(rows)
    raw_named=kernel(pd.DataFrame(rows))
    persisted_frame=pd.DataFrame(persisted)
    persisted_named=kernel(persisted_frame.copy())
    if len(raw_named)!=len(persisted_named) or list(raw_named['source_row'])!=list(persisted_named['source_row']) \
            or list(raw_named['url_index'])!=list(persisted_named['url_index']):
        raise ValueError('source manifest expansion differs between raw and persisted URL forms; give each URL its own row')
    raw_urls=[value if isinstance(value,str) else '' for value in raw_named['url']]
    expanded=any(isinstance(row.get('url'),(list,tuple,dict)) or (isinstance(row.get('url'),str) and row['url'].strip().startswith(('[','{')))
                 for row in rows)
    identities=[]
    for index,raw in enumerate(raw_urls):
        identity=None
        if _is_url(raw):
            row=rows[int(raw_named['source_row'].iloc[index])] if expanded else rows[index]
            declared=row.get('source_id')
            identity=resource_identity(raw,source_id=declared,dataset_id=row.get('dataset_id'),
                resource_index=max(0,int(raw_named['url_index'].iloc[index])) if expanded else 0)
        identities.append(identity)
    seen={}
    for identity,raw in zip(identities,raw_urls):
        if identity is not None:
            _bind(seen,identity,raw)
    # A corpus that already holds downloads keeps their names; new sources never take a name in use (K4BP-G4).
    renamed=_rename_against_corpus((raw_named,persisted_named),identities,root)
    same=(list(raw_named['is_duplicate'].astype(bool))==list(persisted_named['is_duplicate'].astype(bool))
          and list(raw_named['filename_base'].astype(str))==list(persisted_named['filename_base'].astype(str)))
    named=persisted_named
    projection=persisted_frame
    keyed=any(isinstance(identity,str) and not identity.startswith('url:v1:') for identity in identities)
    if same and not expanded:
        # One identity per source row: the download serializer records it with
        # each persisted result row and resumes by it (LG3). The frozen names
        # travel with the rows, so the publication replays exactly them
        # whatever the corpus holds by then (K4BP-G1).
        projection=persisted_frame.copy()
        projection['resource_identity']=identities
        projection['filename_base']=list(persisted_named['filename_base'].astype(str))
    elif same and (keyed or renamed):
        same=False  # expanded keyed URLs, or names fixed by the corpus: record per expanded row below
    if not same:
        # Distinct raw URLs persist alike (different credentials or query
        # values): seal the raw duplicate decisions and names as non-secret
        # columns so the unchanged download serializer reproduces them.
        groups=raw_named['__url_norm'].map({value:index for index,value in reversed(list(enumerate(raw_named['__url_norm'])))})
        named=persisted_named.copy()
        named['is_duplicate']=raw_named['is_duplicate'].astype(bool).values
        named['filename_base']=raw_named['filename_base'].astype(str).values
        named['duplicate_of']=[persisted_url(value) if isinstance(value,str) else value for value in raw_named['duplicate_of']]
        named['frozen_duplicate_group']=groups.astype('int64').values
        projection=named.drop(columns=['__url_norm'])
        projection['resource_identity']=identities
    return named.to_dict(orient='records'),raw_urls,identities,projection


def _manifest_selector_uri(row, writer):
    """The source URI a local manifest row names, or None (same selectors as the CLI)."""
    from ..ledger.urls import sanitize_url
    if isinstance(row, str):
        selector = row
    elif isinstance(row, dict):
        selector = next((row[k] for k in ('document_id','source_key','source_uri','path','url') if row.get(k)), None)
        if row.get('document_id') == selector and isinstance(selector, str):
            known = writer.connection.execute('SELECT source_key FROM documents WHERE document_id=?', (selector,)).fetchone()
            selector = known[0] if known else None
    else:
        return None
    if not isinstance(selector, str):
        return None
    if ':v1:' in selector and selector.split(':v1:',1)[0] in ('local_path','url'):
        selector = selector.split(':v1:',1)[1]
    if selector.startswith(('http://','https://')):
        return sanitize_url(selector)
    return Path(unquote(urlsplit(selector).path) if selector.startswith('file:') else selector).resolve().as_uri()


def _publication_names(entries, remote_bases):
    """Collision-free compatibility publication names for local inputs (CP-LEDGER F1).

    Legacy serializers key every product by the source basename/stem. Inputs
    whose canonical stems are unique keep exactly their current names. When
    distinct inputs share one, each local input is published under a name
    derived from its own source identity, so no product is overwritten.
    """
    import hashlib
    from .._naming import canonical_stem
    groups = {}
    for entry in entries:
        if entry['path'] is not None:
            groups.setdefault(canonical_stem(entry['path'].name), []).append(entry)
    names = {}
    taken = {canonical_stem(name) for name in remote_bases} | {stem for stem, members in groups.items() if len(members) == 1}
    for stem, members in groups.items():
        if len(members) == 1 and stem not in {canonical_stem(name) for name in remote_bases}:
            continue
        for entry in members:
            digest = hashlib.sha256(entry['key'].encode('utf-8')).hexdigest()
            width = 10
            while f'{stem}__{digest[:width]}' in taken:
                width += 2
            taken.add(f'{stem}__{digest[:width]}')
            names[entry['key']] = f'{stem}__{digest[:width]}{entry["path"].suffix}'
    return names


def _publication_stems(operations):
    """The compatibility stem of every document of the frozen plan (its download name, publication name or source stem).

    The run's products (document export, section export) contain exactly these documents: a corpus may hold
    products of earlier runs, which stay in the ledger history but are not part of this run's export (K4BI-G3).
    """
    from .._naming import canonical_stem
    stems={}
    for operation in operations:
        version=operation.document.version_id
        if version is None:continue
        if operation.stage=='download':
            stems[version]=operation.specification.settings['filename_base']
        elif operation.specification.operation=='extract.document' and version not in stems:
            published=operation.specification.settings.get('publication_name')
            stems[version]=Path(published).stem if published else canonical_stem(local_path(operation.document.source_uri))
    return stems


def _publication_alias(root, run, name, target):
    """A run-owned alias spelling the frozen publication name of a local input."""
    import os
    alias = Path(root)/'.glossapi/publication-inputs'/run/name
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink():
        if Path(os.readlink(alias)) != Path(target):
            raise ValueError('Publication alias already names a different source: '+name)
    elif alias.exists():
        raise ValueError('Publication alias location is occupied: '+name)
    else:
        os.symlink(Path(target), alias)
    return alias


def prepare_streaming(corpus,sources=(),*,workers=None,stages=('extract','clean','sections'),settings=None,
                      resume=None,retry_failed=False,observer=None,writer=None,credentials=None):
    """Return committed results; public serializers consume them without reruns.

    Acquisition naming is sealed in source order before any worker starts.
    Corpus selection/global IDs, annotation and shard publication remain barriers.
    """
    from ..ledger.journal import source_key
    began=time.monotonic()
    from .retention import resolved_settings
    settings=resolved_settings(settings)
    root=Path(corpus.output_dir).resolve()
    sources=list(sources)
    download_manifest=None
    occurrence_rows=None
    remote_rows=None
    record_bindings=[]
    with nullcontext(writer) if writer is not None else LedgerWriter(root/'corpus.sqlite3') as writer:
        # Raw URLs (userinfo, query and fragment values) stay in this process
        # only, keyed by resource identity. Everything recorded uses the
        # persisted URL form (CP-LEDGER cycle 2, F1/F2).
        transport=_transport_urls(list(credentials or ()),{})
        from ..ledger.urls import register_setting_secrets
        register_setting_secrets({key:settings.get(key) for key in ('download',) if settings.get(key)})
        register_setting_secrets(getattr(corpus,'downloader_config',{}) or {})
        raw_manifest=settings.get('source_manifest')
        manifest_rows=(raw_manifest.get('documents',[]) if isinstance(raw_manifest,dict) else raw_manifest) if raw_manifest is not None else None
        if not resume:
            # Selectors are resolved per row: remote rows are expanded/named on
            # their raw URLs; local rows keep their real paths (cycle-2 F3).
            remote_manifest=[row for row in manifest_rows if _row_url(row) is not None] if isinstance(manifest_rows,list) else []
            remote_given=[str(source) for source in sources if _is_url(str(source))]
            if remote_manifest or remote_given:
                remote_rows=remote_manifest or [{'url':source} for source in remote_given]
                source_rows,raw_urls,identities,frame=_remote_inventory(remote_rows,root)
                for identity,raw in zip(identities,raw_urls):
                    if identity is not None:transport[identity]=raw
                # Retain the persisted source frame for the unchanged download
                # serializer. Resume binds these bytes, not a reconstructed frame.
                from ..ledger.writer import new_id
                download_manifest=root/'.glossapi/source-manifests'/new_id()/'sources.parquet'
                download_manifest.parent.mkdir(parents=True,exist_ok=False)
                frame.to_parquet(download_manifest,index=False)
                settings={**settings,'download_projection_manifest':{'path':str(download_manifest),
                    'sha256':sha256_file(download_manifest),'bytes':download_manifest.stat().st_size},
                    'semantic_files':[*settings.get('semantic_files',()),
                        {'path':str(download_manifest),'sha256':sha256_file(download_manifest)}]}
                remote_entries=[dict(source=row.get('url'),remote=True,raw=raw,key=identity)
                    for row,raw,identity in zip(source_rows,raw_urls,identities)]
                planned=settings.get('planned_document_ids')
                if planned is not None and any(entry['key'] is not None and entry['key'] not in planned for entry in remote_entries):
                    raise ValueError('planned_source_identity_mismatch: a supplied source URL does not key to its planned resource '
                        'identity (different credentials or query values); plan again with these URLs')
            else:
                source_rows,remote_entries=[],[]
            local_given=[source for source in sources if not _is_url(str(source))]
            source_rows=[*source_rows,*({'source_uri':str(source)} for source in local_given)]
            if settings.get('memory_saving')=='extreme':
                # Up front (K4BI-G7/G10): the final products must fit the corpus filesystem. Estimated from the supplied
                # sources' sizes (URL sizes are unknown until download; the exact check runs before the publications).
                from .final_products import PLANNING_TEXT_RATIO, product_need, require_room
                paths=[Path(unquote(urlsplit(str(source)).path) if str(source).startswith('file:') else str(source)) for source in local_given]
                sizes=[path.stat().st_size for path in paths if path.is_file()]
                require_room(root,settings,product_need(stages,int(PLANNING_TEXT_RATIO*sum(sizes)),len(sizes)),when='(estimated at run start)')
        engine=StreamingScheduler(writer,root,workers=workers,cpu_tokens=settings.get('cpu_tokens'),
            memory_bytes=settings.get('memory_budget_bytes'),disk_budget_bytes=settings.get('disk_budget_bytes'),observer=observer)
        engine.transport_urls=dict(transport)
        unavailable={}
        if resume:
            run=resume
            operations,_,run_row=engine._load(run)
            frozen=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run_row['settings_sha256'],)).fetchone()[0])
            stages=frozen.get('requested_stages',stages)
            settings=frozen
        else:
            operations=[]
            from ..ledger.journal import redact_credentials, safe_settings, code_revision
            entries=[]
            # 1. The requested inventory is durable before any fallible per-file
            # inspection: one missing or unreadable input cannot roll back the
            # other inputs' census (CP-LEDGER F3).
            with writer.transaction():
                observation=writer.run(redact_credentials(settings),{'mode':'explicit_source_inventory'}, {},code_revision=code_revision())
                if download_manifest:
                    writer.artifact(observation,'download','source_manifest',download_manifest.as_uri(),
                        sha256=sha256_file(download_manifest),bytes=download_manifest.stat().st_size,
                        ownership='managed',retention_class='initial')
                given=[*remote_entries,*(dict(source=str(source),remote=False,raw=None,key=None) for source in local_given)]
                for ordinal,item in enumerate(given):
                    source=item['source']
                    remote=item['remote']
                    if remote:
                        uri=source if isinstance(source,str) else ''
                        key=item['key'] or source_key(uri)
                    else:
                        uri=Path(unquote(urlsplit(source).path) if source.startswith('file:') else source).resolve().as_uri()
                        key=source_key(uri)
                    proposed=settings.get('planned_document_ids',{}).get(key)
                    known=writer.connection.execute('SELECT document_id FROM documents WHERE source_key=?',(key,)).fetchone()
                    if proposed and known is None:
                        from ..ledger.writer import validate_id,utc_now
                        validate_id(proposed)
                        writer.connection.execute('INSERT INTO documents VALUES(?,?,?)',(proposed,key,utc_now()))
                    entries.append(dict(ordinal=ordinal,source=uri if remote else source,remote=remote,uri=uri,key=key,
                        document=writer.document(key),path=None if remote else local_path(uri)))
                # Occurrence inventory, separate from the deduplicated processing
                # documents. Every manifest row keeps its typed original row and
                # its own selector, in local and mixed manifests alike (F3/F4).
                by_key={entry['key']:entry for entry in entries if not entry['remote']}
                local_manifest=[(index,row) for index,row in enumerate(manifest_rows)
                    if _row_url(row) is None] if isinstance(manifest_rows,list) else []
                occurrences=[(entry,source_rows[entry['ordinal']],str(source_rows[entry['ordinal']].get('source_row',entry['ordinal'])),
                    max(0,int(source_rows[entry['ordinal']].get('url_index',0) or 0))) for entry in entries if entry['remote']]
                if local_manifest:
                    covered=set()
                    first={}
                    for index,row in local_manifest:
                        uri=_manifest_selector_uri(row,writer)
                        entry=by_key.get(source_key(uri)) if uri else None
                        if entry is None:
                            continue
                        covered.add(entry['key'])
                        original=row if isinstance(row,dict) else {'source_uri':row}
                        occurrences.append((entry,{'manifest_row':original,'occurrence_index':index,
                            'duplicate_of_occurrence':first.get(entry['key'])},
                            str(original.get('source_row',index)),0))
                        first.setdefault(entry['key'],index)
                    for entry in entries:
                        if not entry['remote'] and entry['key'] not in covered:
                            occurrences.append((entry,source_rows[entry['ordinal']],str(entry['ordinal']),0))
                    occurrence_rows=[metadata for _,metadata,_,_ in occurrences]
                else:
                    occurrences.extend((entry,source_rows[entry['ordinal']],str(source_rows[entry['ordinal']].get('source_row',entry['ordinal'])),
                        max(0,int(source_rows[entry['ordinal']].get('url_index',0) or 0))) for entry in entries if not entry['remote'])
                manifest=writer.snapshot('source_manifest',{'records':safe_settings(occurrence_rows if occurrence_rows is not None else source_rows)})
                for entry,metadata,row_identity,url_index in occurrences:
                    record=writer.source_record(entry['document'],observation,entry['uri'],metadata,source_kind='url' if entry['remote'] else 'local_path',
                        source_row=row_identity,url_index=url_index,manifest_sha256=manifest)
                    record_bindings.append((record,entry['key']))
            # 2. Per-file inspection outside the inventory transaction.
            unique=[]
            seen=set()
            for entry in entries:
                if entry['key'] in seen:
                    continue
                seen.add(entry['key'])
                unique.append(entry)
                if entry['path'] is not None:
                    try:
                        entry['sha256']=sha256_file(entry['path'])
                        entry['bytes']=entry['path'].stat().st_size
                    except OSError as error:
                        entry['failure']=error
            remote_bases=[source_rows[entry['ordinal']].get('filename_base') for entry in unique if entry['remote'] and source_rows[entry['ordinal']].get('filename_base')]
            publication=_publication_names([entry for entry in unique if 'failure' not in entry],remote_bases)
            version_of={}
            with writer.transaction():
                for entry in unique:
                    ordinal,source,remote,uri,document,path=(entry[name] for name in ('ordinal','source','remote','uri','document','path'))
                    failure=entry.get('failure')
                    if failure is not None:
                        # An absent/unreadable source is its own outcome; the
                        # other inputs proceed and report --why explains it.
                        version=writer.version(document)
                        writer.artifact(observation,'extract','source_payload',uri,version_id=version,ownership='initial_external',
                            retention_class='initial',presence='missing')
                        writer.event(observation,event_kind='decision',reason_code='source_unavailable',version_id=version,
                            details={'source_uri':uri,'path':str(path),'cause':'local_source_absent' if isinstance(failure,FileNotFoundError) else 'local_source_unreadable',
                                'error_type':type(failure).__name__,'errno':failure.errno,'message':str(failure)})
                        unavailable[version]=type(failure).__name__
                    else:
                        digest=entry.get('sha256')
                        version=writer.version(document,source_sha256=digest,stored_sha256=digest)
                    version_of[entry['key']]=version
                    skip='source_unavailable' if failure is not None else None
                    published=publication.get(entry['key'])
                    naming={'publication_name':published} if published else {}
                    artifacts=()
                    if path and failure is None:
                        file_id=writer.artifact(observation,'extract','source_payload',uri,sha256=digest,bytes=entry['bytes'],
                            version_id=version,ownership='initial_external',retention_class='initial')
                        artifacts=(ArtifactReference(file_id,'source_payload',path,digest,entry['bytes']),)
                    snapshot=DocumentSnapshot(document,version,uri,artifacts=artifacts)
                    parent=None
                    if remote:
                        # A finite source epoch fixes names and collisions. The
                        # retained downloader validates extensions/redirects.
                        name=source_rows[ordinal]['filename_base']
                        download_settings={'download_mode':'standard','legacy_outer_retry_count':0,**corpus.downloader_config,**settings.get('download',{}),
                            'url':transport.get(entry['key'],source),'filename_base':name,'row_index':ordinal}
                        if download_settings['url']!=source:
                            # Non-secret marker: the frozen URL is its persisted form and
                            # transport needs the raw URL, bound by resource identity.
                            download_settings['transport_url_redacted']=True
                        if settings.get('url_policy'):
                            # Workers record their receipts under the same declared policy.
                            download_settings['url_policy']=settings['url_policy']
                        parent=f'{ordinal}:download'
                        operations.append(ScheduledOperation(parent,'download',snapshot,
                            StageSpecification('download.acquire',required_output_kinds=('download_payload',),settings=download_settings),
                            skip_reason='duplicate_url' if source_rows[ordinal].get('is_duplicate') else None))
                    if 'extract' in stages:
                        key=f'{ordinal}:extract'
                        operations.append(ScheduledOperation(key,'extract',snapshot,
                            StageSpecification('extract.document',('source_payload',) if artifacts else (),('raw_markdown',),
                                settings={**extraction_settings(settings.get('extract')),**naming}),
                            inputs=((parent,'download_payload','source_payload'),) if parent else (),skip_reason=skip))
                        parent=key
                    if 'clean' in stages:
                        key=f'{ordinal}:clean'
                        operations.append(ScheduledOperation(key,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=((parent,'raw_markdown','selected_text'),),skip_reason=skip))
                        parent=key
                    if 'ocr' in stages:
                        from ..ocr.defaults import default_values, select_device, cache_root
                        ocr_settings = dict(settings.get('ocr',{}))
                        if ocr_settings.get('entry_point_name','glossapi') != 'glossapi':
                            raise ValueError('This run adapter requires the Greek OCR entry point; pass external options to Corpus.ocr()')
                        device = ocr_settings.get('device_id') or ocr_settings.get('options',{}).get('device') or select_device()
                        values = {**default_values(device), **ocr_settings.get('options',{})}
                        quality_key=f'{ordinal}:quality'
                        operations.append(ScheduledOperation(quality_key,'extract',snapshot,
                            StageSpecification('extract.page_quality',required_output_kinds=('page_quality',),
                                settings={'page_quality':settings.get('page_quality',{}),**naming}),
                            inputs=((f'{ordinal}:download','download_payload','source_payload'),) if remote else (),skip_reason=skip))
                        source_snapshot=replace(snapshot,artifacts=() if remote else snapshot.artifacts)
                        edges=((f'{ordinal}:download','download_payload','source_payload'),) if remote else ()
                        ocr_key=f'{ordinal}:ocr'
                        operations.append(ScheduledOperation(ocr_key,'ocr',source_snapshot,
                            StageSpecification('ocr.document',('source_payload',) if not remote else (),('ocr_markdown','ocr_metrics'),device,
                                {**ocr_settings,'options':values,'entry_point_name':'glossapi',
                                 'cache_root':str(cache_root(ocr_settings.get('cache_root'))),**naming}),
                            inputs=(*edges,(f'{ordinal}:extract','raw_markdown','native_markdown'),(quality_key,'page_quality','page_quality')),
                            after=(parent,),cpu_tokens=int(ocr_settings.get('num_threads',1)),skip_reason=skip))
                        parent=f'{ordinal}:clean_after_ocr'
                        operations.append(ScheduledOperation(parent,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial.artifact:after_ocr',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=((ocr_key,'ocr_markdown','selected_text'),),skip_reason=skip))
                    if 'sections' in stages:
                        filename=Path(published).stem if published else path.stem if path else name
                        operations.append(ScheduledOperation(f'{ordinal}:sections','sections',replace(snapshot,artifacts=()),
                            StageSpecification('sections.document',required_output_kinds=('section_fragment',),settings={'filename':filename}),
                            inputs=((parent,'clean_markdown','selected_text'),),skip_reason=skip))
            run=engine.freeze(operations,settings={**settings,'requested_stages':list(stages)})
            corpus._active_run_id=run
            # Inventory-to-run and occurrence-to-version bindings: each source
            # occurrence reports its own version and this run's disposition (F4).
            with writer.transaction():
                writer.event(observation,event_kind='decision',reason_code='source_inventory_bound',
                    details={'execution_run_id':run,'occurrences':[{'source_record_id':record,'version_id':version_of.get(key)}
                        for record,key in record_bindings]})
        corpus._retention_settings={**settings,'requested_stages':list(stages)}
        result=engine.execute(run,resume=bool(resume),retry_failed=retry_failed)
        result['wall_seconds']=time.monotonic()-began
        replays={}
        download_replays={}
        selected_sources=[]
        source_versions={}
        per_version={}
        for operation in operations:per_version.setdefault(operation.document.version_id,[]).append(operation.key)
        completed_versions={version for version,keys in per_version.items() if all(key in result['attempts'] for key in keys)}
        for ordinal,operation in enumerate(operations):
            if operation.stage=='download':
                task=next(task['task_id'] for task in writer.task_states(run) if task['ordinal']==ordinal)
                latest=writer.latest_attempt(task)
                if latest:
                    receipts=[artifact for artifact in engine._outputs(latest['attempt_id']) if artifact.kind=='worker_receipt']
                    download_replays[(operation.document.source_uri,operation.specification.settings['filename_base'])]={
                        'attempt_id':latest['attempt_id'],'receipt':receipts[0] if receipts else None,
                        'status':latest['status'],'reason_code':latest['reason_code'],'message':latest['message']}
            attempt=result['attempts'].get(operation.key)
            if not attempt:
                continue
            if operation.stage=='download' and operation.document.version_id in completed_versions and 'extract' not in stages:
                payload=next(a for a in engine._outputs(attempt) if a.kind=='download_payload')
                selected_sources.append(payload.path);source_versions[str(payload.path)]=operation.document.version_id
            if operation.specification.operation=='extract.document' and operation.document.version_id in completed_versions:
                # Only the document extraction names a publication source. An OCR run's extract.page_quality is
                # also an extract-stage operation; counting it listed every source twice, so the extract
                # publication replayed (and registered) each raw markdown twice (K4C2-G2).
                if operation.document.artifacts:source=operation.document.artifacts[0]
                else:
                    producer=next(edge[0] for edge in operation.inputs if edge[2]=='source_payload')
                    source=next(artifact for artifact in engine._outputs(result['attempts'][producer],include_deleted=True) if artifact.kind=='download_payload')
                published=operation.specification.settings.get('publication_name')
                # A colliding basename publishes under its frozen collision-free
                # name through a run-owned alias of the same bytes (CP-LEDGER F1).
                path=_publication_alias(root,run,published,source.path) if published else source.path
                selected_sources.append(path)
                source_versions[str(path)]=operation.document.version_id
            references=writer.connection.execute('SELECT sha256 FROM attempt_inputs WHERE attempt_id=? ORDER BY rowid',(attempt,)).fetchall()
            key=(operation.stage,operation.specification.operation,tuple(row[0] for row in references))
            receipt=next(artifact for artifact in engine._outputs(attempt) if artifact.kind=='worker_receipt')
            # Keep only a hash-bound reference at the corpus barrier. Full edit
            # receipts are read for one projection at a time, never prefetched.
            replays.setdefault(key,[]).append((operation.document.version_id,dict(operation.specification.settings),receipt,receipt.path.parent))
        result['sources']=[str(path) for path in selected_sources]
        failed_versions={task['version_id'] for task in writer.task_states(run) if task['status']=='failed' and task['version_id'] is not None}
        result['document_counts']={'selected':len(per_version),'included':len(completed_versions),
            'omitted':len(per_version)-len(completed_versions),'failed':len(failed_versions)}
        unavailable_versions={task['version_id'] for task in writer.task_states(run)
            if task['reason_code']=='source_unavailable' and task['version_id'] is not None}
        result['source_inventory']={'unavailable':len(unavailable_versions)}
        corpus._streaming_planned_versions=[version for version in per_version if version is not None]
        corpus._streaming_completed_versions=completed_versions
        # The frozen per-version section filename: the annotated-section census reads rows by it (W18 M4).
        corpus._streaming_section_names={operation.document.version_id:operation.specification.settings.get('filename')
            for operation in operations if operation.stage=='sections' and operation.document.version_id}
        corpus._streaming_publication_stems=_publication_stems(operations)
        result['plan_hash']=writer.connection.execute('SELECT plan_sha256 FROM runs WHERE run_id=?',(run,)).fetchone()[0]
        if resume:
            settings=json.loads(writer.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',(run,)).fetchone()[0])
    corpus._streaming_source_versions=source_versions
    corpus._streaming_download_replays=download_replays if settings.get('download_projection_manifest') else None
    corpus._streaming_download_manifest=settings.get('download_projection_manifest')
    return result,replays


def _run_corpus(corpus,files=(),*,stages=('extract','clean','sections','export'),workers=None,settings=None,
               resume=None,retry_failed=False,plan=None,observer=None,credentials=None):
    """Execute document DAG, then declared compatibility publication barriers."""
    settings=settings or {}
    files=list(files)
    if resume and not retry_failed:
        writer=corpus._scheduler_writer
        row=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' ORDER BY event_id DESC LIMIT 1",(resume,)).fetchone()
        digest=json.loads(row[0]).get('result_sha256') if row else None
        if digest:
            from ..ledger.encoding import snapshot_hash
            from .scheduler import execution_environment
            run=writer.connection.execute('SELECT * FROM runs WHERE run_id=?',(resume,)).fetchone()
            if snapshot_hash({'schema_version':1,'kind':'environment',**execution_environment()})!=run['environment_sha256']:
                raise ValueError('Frozen execution environment changed; create an amended run')
            saved=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0])
            for file_id in saved['retained_file_ids']:
                binding=writer.connection.execute('SELECT uri,sha256,bytes FROM files WHERE file_id=?',(file_id,)).fetchone()
                path=local_path(binding['uri'])
                if not path.is_file() or path.stat().st_size!=binding['bytes'] or sha256_file(path)!=binding['sha256']:
                    raise ValueError('Retained recovery frontier missing or corrupt; reacquire bound ancestors in an amended run')
            from .retention import RetentionController
            frozen=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run['settings_sha256'],)).fetchone()[0])
            RetentionController(writer,corpus.output_dir,resume,frozen).recover()
            return saved['result']
    prerequisites={'clean':('extract',),'ocr':('extract','clean'),'sections':('clean','extract'),'annotate':('sections','clean','extract'),'export':('clean','extract')}
    stages=tuple(dict.fromkeys([dependency for stage in stages for dependency in (*prerequisites.get(stage,()),stage)]))
    if plan is not None:
        from ..ledger.encoding import snapshot_hash
        body=plan.get('data',plan)
        if snapshot_hash({key:value for key,value in body.items() if key!='plan_hash'})!=body.get('plan_hash'):
            raise ValueError('Supplied plan hash does not match')
        if any(item.get('code')=='empty_selection' for item in body.get('conflicts') or ()):
            # An explicitly empty selection never runs, and never widens to the corpus (CP-K3 F1).
            from ..plan_preview import refuse_conflicts
            refuse_conflicts({'conflicts':[item for item in body['conflicts'] if item.get('code')=='empty_selection']})
        from ..ledger.urls import is_keyed_key, source_uri_of_key
        from ..ledger.urls import scrub
        explicit=body.get('explicit_manifest')
        raw_manifest=None
        if explicit is not None:
            # The plan records the manifest in its persisted form. The caller's
            # raw manifest (in memory, from --manifest or --credentials) must
            # persist to exactly the planned one; identities are checked below.
            for candidate in [*(credentials or ()),[item for item in (credentials or ()) if isinstance(item,dict) and 'documents' not in item]]:
                rows=candidate.get('documents') if isinstance(candidate,dict) and 'documents' in candidate else candidate
                planned=explicit.get('documents') if isinstance(explicit,dict) else explicit
                if isinstance(rows,list) and scrub(rows)==planned:
                    raw_manifest={'documents':rows} if isinstance(explicit,dict) else rows
                    break
        bindings=_transport_urls(list(credentials or ()),{}) if credentials else {}
        if explicit is not None and raw_manifest is None and any(is_keyed_key(document['source_key']) for document in body['documents']):
            # Persisted manifest rows cannot be expanded or keyed again.
            from .scheduler import CREDENTIALS_NOT_PERSISTED
            raise ValueError(CREDENTIALS_NOT_PERSISTED+' (supply the original manifest rows, for example --credentials MANIFEST.json)')
        files=[]
        for document in body['documents']:
            observation=document.get('source_observation',{})
            uri=observation.get('source_uri') or source_uri_of_key(document['source_key'])
            if is_keyed_key(document['source_key']):
                # A keyed (credential/query-bearing) source binds only to the raw
                # URL whose resource identity is exactly the planned one.
                raw=bindings.get(document['source_key'])
                if raw is None and raw_manifest is None:
                    from .scheduler import CREDENTIALS_NOT_PERSISTED
                    raise ValueError(CREDENTIALS_NOT_PERSISTED+' (planned source: '+str(uri)+')')
                files.append(raw or uri)
                continue
            files.append(uri)
            if uri.startswith('file:'):
                path=local_path(uri)
                if not path.exists() and observation.get('bytes') is None and not observation.get('stored_sha256'):
                    # Planned while absent and still absent: the run records it
                    # as its own unavailable outcome (CP-LEDGER F3).
                    continue
                if not path.is_file() or (observation.get('bytes') is not None and
                    (path.stat().st_size,path.stat().st_mtime_ns)!=(observation['bytes'],observation['mtime_ns'])):
                    raise ValueError('Supplied plan source inventory changed')
                # The bytes observed at planning; the stored hash only when the plan observed none (CP-K3 F3).
                expected=observation.get('sha256') or observation.get('stored_sha256')
                if expected and sha256_file(path)!=expected:
                    raise ValueError('Supplied plan source hash changed')
        stages=tuple(dict.fromkeys(step['stage'] for document in body['documents'] for step in document['steps']))
        workers={pool['stage']:pool['workers'] for pool in body['pools']}
        settings=body['settings']
        if 'ocr' in stages:
            pool=next(pool for pool in body['pools'] if pool['stage']=='ocr')
            settings={**settings,'ocr':{**settings.get('ocr',{}),'device_id':pool['device_id']}}
        settings={**settings,'planned_document_ids':{document['source_key']:document['document_id'] for document in body['documents']},
            # The approved plan --json identity is part of the frozen run settings (K3A-M3), so a later reader ties
            # the run to the plan the user approved (the CLI envelope's bound_plan_hash).
            'approved_plan_hash':body['plan_hash']}
        if explicit is not None:
            settings={**settings,'source_manifest':raw_manifest if raw_manifest is not None else explicit}
    if resume and (files or workers or settings):
        raise ValueError('Resume restores its frozen plan; semantic overrides are forbidden')
    if 'math' in stages:
        raise ValueError('Math requires an explicit operation DAG')
    import inspect
    extraction_options=settings.get('extract',{})
    semantic_options=set(extraction_settings()) & set(inspect.signature(corpus.extract).parameters)
    if any(name not in semantic_options for name in extraction_options):
        raise ValueError('Extraction setting requires an explicit operation DAG')
    if int(extraction_options.get('num_threads') or 1)!=1:
        raise ValueError('Native thread overrides require explicit CPU-token declarations in an operation DAG')
    publication_stages=[stage for stage in ('extract','clean','ocr','sections','annotate','export') if stage in stages]
    if not resume and any(str(source).startswith(('http://','https://')) for source in files):
        publication_stages.insert(0,'download')
    if not resume:
        import os
        from ..section_model import model_files
        # A neutral classifier contributes its manifest and every declared file (K2 W2b).
        semantic_files=[*model_files(corpus.section_classifier_model_path),Path(corpus.extraction_model_path)]
        wordlist=os.environ.get('GLOSSAPI_GREEK_WORDLIST')
        if wordlist:semantic_files.append(Path(wordlist))
        identity=[{'path':str(path.resolve()),'sha256':sha256_file(path)} for path in semantic_files if path.is_file()]
        if wordlist and not Path(wordlist).is_file():
            # A configured but missing word list: extraction proceeds without it (null rates) and the run identity
            # records the fixed marker (K2 native coordinator decision); present files keep their existing entries.
            from .extract import wordlist_identity
            identity.append({'path':wordlist,'sha256':wordlist_identity(wordlist)})
        settings={**settings,'semantic_files':identity}
    result,replays=prepare_streaming(corpus,files,workers=workers,stages=stages,
        settings={**settings,'publication_barriers':publication_stages},
        resume=resume,retry_failed=retry_failed,observer=observer,writer=corpus._scheduler_writer,credentials=credentials)
    _bind_barrier_context(corpus._scheduler_writer,result.get('run_id'))
    if result.get('run_id') and result['state'] in ('completed','completed_with_failures') and getattr(corpus,'_retention_settings',{}).get('memory_saving')=='extreme':
        # Before any publication (K4BI-G7): the exact cleaned text of this run decides whether the final products fit;
        # a typed refusal now, never an exhausted disk mid-publication. Resume after enlarging the filesystem.
        from .final_products import product_need, require_room
        from .disk_budget import DiskBudget
        produced=set(result['attempts'].values())
        texts=[row['bytes'] or 0 for row in corpus._scheduler_writer.file_catalog().values()
            if row['kind']=='clean_markdown' and row['presence']=='present' and row['producer_attempt_id'] in produced]
        require_room(corpus.output_dir,corpus._retention_settings,product_need(corpus._retention_settings.get('requested_stages',stages),sum(texts),len(texts)),
            when='(exact, before the publications)',
            present=DiskBudget(corpus._scheduler_writer,corpus.output_dir,result['run_id'],corpus._retention_settings).present_bytes())
    if result.get('run_id') and getattr(corpus,'_retention_settings',{}).get('memory_saving')=='extreme':
        # Streaming retirement (W18 M9) runs as document readers finish. A resume after a kill inside it (or a run
        # whose last completion raced it) finishes it here, before any publication (K4BP-G2).
        from .retention import retire_downloaded_sources
        retire_downloaded_sources(corpus._scheduler_writer,corpus.output_dir,result['run_id'],corpus._retention_settings)
    if result['state'] not in ('completed','completed_with_failures'):
        return result
    corpus._streaming_ocr_replays=replays
    if resume:
        from ..ledger.writer import open_readonly
        with open_readonly(Path(corpus.output_dir)/'corpus.sqlite3') as connection:
            row=connection.execute('SELECT settings_sha256 FROM runs WHERE run_id=?',(resume,)).fetchone()
            settings=json.loads(connection.execute('SELECT json FROM snapshots WHERE sha256=?',row).fetchone()[0])
            stages=settings['requested_stages']
    calls={
        # A resumed run reads redacted credential markers; the publication replays
        # committed acquisitions and never receives a placeholder as a credential.
        'download':lambda:corpus.download(corpus._streaming_download_manifest['path'],**{key:value for key,value in settings.get('download',{}).items()
            if value!=__import__('glossapi.ledger.journal',fromlist=['REDACTED']).REDACTED}),
        'extract':lambda:corpus.extract(**{'input_format':'pdf','file_paths':result['sources'],'phase1_backend':'safe',
            'accel_type':'CPU','num_threads':1,'skip_existing':False,'batched':False,'repair_fonts':True,**settings.get('extract',{})}),
        'clean':lambda:corpus.clean(num_threads=1),
        'ocr':lambda:publish_ocr_results(corpus),
        'sections':corpus.section,'annotate':corpus.annotate,
        'export':lambda:corpus.jsonl(Path(corpus.output_dir)/'export.jsonl')}
    if not result['sources'] and 'export' in stages:
        metadata=Path(corpus.output_dir)/'download_results/download_results.parquet'
        if not metadata.exists():
            import pandas as pd
            from ..parquet_schema import update_metadata_parquet
            # Create only if still missing inside the metadata lock (K2 native): never replace a concurrent writer's file.
            update_metadata_parquet(metadata,lambda current:None if metadata.exists() else pd.DataFrame({'filename':pd.Series(dtype='str')}))
        writer=corpus._scheduler_writer
        with writer.transaction():
            for version, in writer.connection.execute('SELECT DISTINCT version_id FROM tasks WHERE run_id=? AND version_id IS NOT NULL',(result['run_id'],)).fetchall():
                writer.event(result['run_id'],event_kind='decision',reason_code='export_canonical_markdown_missing',version_id=version,
                    details={'product':'document_jsonl','included':False,'decision':'omit','cause':'upstream_failed','empty_selected_frontier':True})
    from .disk_budget import DiskBudget
    # One whole-run disk observer for the publication barriers; it starts from
    # the scheduler's persisted peak and observes before each retention step.
    publication_budget=DiskBudget(corpus._scheduler_writer,corpus.output_dir,result['run_id'],getattr(corpus,'_retention_settings',settings))
    # Each document version owns its compatibility locations for this run.
    corpus._publication_claims={}
    with committed_results(replays):
        for stage in ('download','extract','clean','ocr','sections','annotate','export'):
            if stage in stages or (stage=='download' and corpus._streaming_download_manifest):
                published=False
                if not result['sources'] and stage not in ('download','export'):
                    skip_barrier(corpus,result['run_id'],stage)
                else:
                    published=publish_barrier(corpus,result['run_id'],stage,calls[stage],retry_failed=retry_failed,
                        empty_export=not result['sources'] and stage=='export')
                # The next compatibility reader has now consumed these private
                # producer records. Retire at that frontier, not at run close.
                released_stage = {'clean':'extract','sections':'clean','annotate':'sections'}.get(stage)
                if released_stage and published:
                    publication_budget.observe()
                    from .retention import retain_completed_run
                    attempts = [attempt for key,attempt in result['attempts'].items()
                        if key.endswith(':'+released_stage)]
                    retain_completed_run(corpus._scheduler_writer,corpus.output_dir,result['run_id'],
                        getattr(corpus,'_retention_settings',settings),attempts=attempts,
                        observer=getattr(corpus,'_retention_observer',None))
    if 'export' in stages:
        reconcile_export_census(corpus,result)
    if 'annotate' in stages:
        reconcile_section_census(corpus,result)
    from .retention import retain_completed_run
    writer=corpus._scheduler_writer
    # Every wrapper generation is a child of this fixed publication plan.
    wrapper_runs=[json.loads(row[0])['wrapper_run_id'] for row in writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message LIKE 'wrapper:%'",(result['run_id'],))]
    all_attempts=list(result['attempts'].values())
    for wrapper in wrapper_runs:
        all_attempts.extend(row[0] for row in writer.connection.execute('SELECT attempt_id FROM attempts a JOIN tasks t USING(task_id) WHERE t.run_id=?',(wrapper,)))
    publication_states=[task['status'] for task in writer.task_states(result['run_id']) if task['operation'].endswith('.compatibility_publish')]
    publication_budget.observe()
    if all(status=='ok' for status in publication_states):
        retain_completed_run(writer,corpus.output_dir,result['run_id'],getattr(corpus,'_retention_settings',settings),attempts=all_attempts,observer=getattr(corpus,"_retention_observer",None))
    from .retention import retention_summary
    # Current present/free bytes after retention; the high-water keeps the
    # scheduler and publication peaks instead of a fresh zero (LS7-G04).
    result['disk']=publication_budget.report()
    result['disk']['deleted_logical_bytes']=retention_summary(writer.connection)['deleted_logical_bytes']
    result['outputs']=[]
    # Both final products (PLAN 2.3): the document export and, when annotation ran, the annotated-section export.
    products=[('jsonl','export.jsonl')] if 'export' in stages or (Path(corpus.output_dir)/'export.jsonl').is_file() else []
    if 'annotate' in stages:
        products+=[('classified_sections_parquet','classified_sections.parquet'),('annotated_sections_parquet','fully_annotated_sections.parquet')]
    for kind,name in products:
        output=Path(corpus.output_dir)/name
        if not output.is_file():continue
        # The barrier just published these bytes and registered their hash (hash once per byte stream).
        published=writer.connection.execute("SELECT sha256,bytes FROM current_file_state WHERE uri=? AND producer_attempt_id IS NOT NULL "
            "AND ownership='managed' AND presence='present' ORDER BY event_id DESC LIMIT 1",(output.absolute().as_uri(),)).fetchone()
        size=output.stat().st_size
        digest=published['sha256'] if published and published['bytes']==size else sha256_file(output)
        result['outputs'].append({'kind':kind,'path':str(output),'sha256':digest,'bytes':size})
    from collections import Counter
    counts=Counter(task['status'] for task in writer.task_states(result['run_id']))
    result['counts']={status:counts[status] for status in ('planned','running','ok','skipped','failed')}
    result['state']='completed_with_failures' if counts['failed'] or result.get('source_inventory',{}).get('unavailable') else 'completed'
    with corpus._scheduler_writer.transaction():
        corpus._scheduler_writer.event(result['run_id'],event_kind='run',reason_code='legacy_observed',details={'state':result['state'],
            'disk_high_water_bytes':result['disk']['high_water_bytes'],'disk_high_water_scope':'run','result_sha256':corpus._scheduler_writer.snapshot('retained_run_result',{
                'result':{key:value for key,value in result.items() if key!='trace'},
                'retained_file_ids':[row['file_id'] for row in writer.file_catalog().values() if row['presence']=='present'
                    and row['ownership'] in ('managed','initial_external') and local_path(row['uri']).is_file()]})})
    return result


def _bind_barrier_context(writer,run_id):
    """The run-close barrier records per-barrier files (debug=on) or one summary (debug=off) for this run."""
    if not run_id:return
    row=writer.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',(run_id,)).fetchone()
    debug=(json.loads(row[0]).get('debug','off') if row else 'off')
    writer.barrier_context={'run_id':run_id,'debug':debug}
    writer.retained_intermediates=debug=='on'


def reconcile_export_census(corpus,result):
    """Final include/omit counts from per-version export membership (CP-LEDGER F1).

    A version counts as included only when the export generation's own
    membership receipt includes it; every other planned version gets one
    explicit omission decision that ledger-only reports can explain.
    """
    from ..ledger.reader import evidence
    writer=corpus._scheduler_writer
    run=result['run_id']
    planned=getattr(corpus,'_streaming_planned_versions',None) or []
    completed=getattr(corpus,'_streaming_completed_versions',set())
    if not planned:
        return
    table=writer.connection.event_table
    tasks=writer.task_states(run)
    barrier=next((task for task in tasks if task['operation']=='export.compatibility_publish'),None)
    exported=set()
    if barrier is not None and barrier['status']=='ok':
        latest=writer.connection.execute(f"SELECT details_json FROM {table} WHERE run_id=? AND event_kind='run' AND message='wrapper:export' ORDER BY event_id DESC LIMIT 1",(run,)).fetchone()
        if latest:
            wrapper_run=json.loads(latest[0])['wrapper_run_id']
            for task in writer.task_states(wrapper_run):
                if task['stage']!='export':
                    continue
                for event in writer.connection.execute(f"SELECT details_json FROM {table} WHERE task_id=? AND event_kind='decision' ORDER BY event_id",(task['task_id'],)).fetchall():
                    details=evidence(writer.connection,{'details_json':event[0]})
                    if 'selected_versions' in details:
                        exported={item['version_id'] for item in details['selected_versions'] if item.get('included') and item.get('version_id')}
    unavailable={task['version_id'] for task in tasks if task['reason_code']=='source_unavailable' and task['version_id'] is not None}
    omitted=[version for version in planned if version not in exported]
    recorded={row[0] for row in writer.connection.execute(
        f"SELECT DISTINCT version_id FROM {table} WHERE run_id=? AND event_kind='decision' AND reason_code='export_canonical_markdown_missing' AND version_id IS NOT NULL",(run,))} if omitted else set()
    with writer.transaction():
        for version in omitted:
            if version in recorded:
                continue
            cause='source_unavailable' if version in unavailable else 'upstream_failed' if version not in completed else 'not_in_export_membership'
            writer.event(run,event_kind='decision',reason_code='export_canonical_markdown_missing',version_id=version,
                details={'product':'document_jsonl','included':False,'decision':'omit','cause':cause,'basis':'final_export_membership'})
    result['document_counts']={**result['document_counts'],'included':len(planned)-len(omitted),'omitted':len(omitted)}


def reconcile_section_census(corpus,result):
    """The annotated-section export's own census (PLAN 2.3, W18 M4).

    Every planned version gets exactly one decision for product
    ``annotated_sections``: include with its section count in the published
    ``fully_annotated_sections.parquet``, or omit with a cause
    (``source_unavailable``, ``upstream_failed`` or ``no_sections``).
    """
    import pandas as pd
    writer=corpus._scheduler_writer
    run=result['run_id']
    planned=getattr(corpus,'_streaming_planned_versions',None) or []
    if not planned:
        return
    names=getattr(corpus,'_streaming_section_names',None) or {}
    completed=getattr(corpus,'_streaming_completed_versions',set())
    tasks=writer.task_states(run)
    barrier=next((task for task in tasks if task['operation']=='annotate.compatibility_publish'),None)
    product=Path(corpus.output_dir)/'fully_annotated_sections.parquet'
    counts={}
    if barrier is not None and barrier['status']=='ok' and product.is_file():
        counts=pd.read_parquet(product,columns=['filename'])['filename'].astype(str).value_counts().to_dict()
    unavailable={task['version_id'] for task in tasks if task['reason_code']=='source_unavailable' and task['version_id'] is not None}
    table=writer.connection.event_table
    recorded={row[0] for row in writer.connection.execute(
        f"SELECT DISTINCT version_id FROM {table} WHERE run_id=? AND event_kind='decision' AND reason_code IN ('export_included','annotation_sections_missing') "
        "AND version_id IS NOT NULL",(run,))}
    included=0
    with writer.transaction():
        for version in planned:
            count=int(counts.get(str(names.get(version)),0)) if names.get(version) else 0
            included+=bool(count)
            if version in recorded:
                continue
            common={'product':'annotated_sections','path':str(product),'basis':'published_annotated_sections_rows'}
            if count:
                writer.event(run,event_kind='decision',reason_code='export_included',version_id=version,
                    details={**common,'included':True,'decision':'include','section_count':count})
            else:
                cause=('source_unavailable' if version in unavailable else 'upstream_failed'
                    if version not in completed or barrier is None or barrier['status']!='ok' else 'no_sections')
                writer.event(run,event_kind='decision',reason_code='annotation_sections_missing',version_id=version,
                    details={**common,'included':False,'decision':'omit','cause':cause})
    result['section_counts']={'selected':len(planned),'included':included,'omitted':len(planned)-included}


@shadow_stage('ocr', 'ocr.wrapper')
def publish_ocr_results(self):
    """Publish committed OCR bytes at the existing single-writer barrier."""
    from .workers import read_result
    from .compatibility import materialize_stage_outputs
    from ..corpus.ocr.artifacts import persist_ocr_success, refresh_cleaner_after_ocr
    corpus=self
    replays=corpus._streaming_ocr_replays
    filenames=[]
    for (stage,operation,hashes), candidates in replays.items():
        if stage != 'ocr':
            continue
        for version,settings,receipt,workspace in candidates:
            result=read_result(receipt.path)
            if result.reason_code != 'ocr_completed':
                continue
            paths={output.relative_path:corpus.output_dir/output.relative_path for output in result.outputs
                if output.kind in ('ocr_markdown','ocr_metrics','ocr_native','ocr_page_state')}
            materialize_stage_outputs(corpus,result,workspace,paths)
            markdown=next(output for output in result.outputs if output.kind=='ocr_markdown')
            filenames.append(Path(markdown.relative_path).with_suffix('.pdf').name)
    if filenames:
        persist_ocr_success(corpus,filenames=filenames,backend_norm='docling')
        refresh_cleaner_after_ocr(corpus)


def skip_barrier(corpus,run,stage):
    writer=corpus._scheduler_writer
    task=next((task for task in writer.task_states(run) if task['operation']==stage+'.compatibility_publish'),None)
    if task['status'] in ('ok','skipped','failed'):return
    with writer.transaction():
        attempt=writer.attempt(task['task_id'],worker_id='single-publisher',device_id='cpu',fingerprint={'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
        writer.finish(run,task['task_id'],attempt,status='skipped',reason_code='upstream_failed')


def publish_barrier(corpus,run,stage,callback,*,retry_failed=False,empty_export=False):
    """Recover a wrapper commit even if death precedes its barrier commit."""
    writer=corpus._scheduler_writer
    task=next((task for task in writer.task_states(run) if task['operation']==stage+'.compatibility_publish'),None)
    if task is None:
        raise ValueError('Publication barrier is absent from the frozen plan')
    if task['status'] in ('failed','skipped') and not retry_failed:return False
    parents=[writer.task_state(row[0]) for row in writer.connection.execute('SELECT producer_task_id FROM task_dependencies WHERE task_id=?',(task['task_id'],))]
    unavailable=any(parent['operation'].endswith('.compatibility_publish') and parent['stage']!='download' and parent['status']!='ok' for parent in parents)
    if unavailable and not empty_export:
        skip_barrier(corpus,run,stage);return False
    previous=writer.latest_attempt(task['task_id'])
    wrapper=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message=? ORDER BY event_id DESC LIMIT 1",(run,'wrapper:'+stage)).fetchone()
    wrapper_run=json.loads(wrapper[0])['wrapper_run_id'] if wrapper else None
    wrapper_ok=False
    if wrapper_run:
        wrapper_operations=(stage+'.wrapper',stage+'.clean' if stage=='clean' else stage+'.jsonl' if stage=='export' else stage+'.extract')
        wrapper_ok=any(task['scope_key'].startswith('corpus:') and task['status']=='ok' and task['operation'] in wrapper_operations for task in writer.task_states(wrapper_run))
    if task['status']=='ok':
        for row in writer.connection.execute('SELECT f.uri,i.sha256 FROM attempt_inputs i JOIN files f USING(file_id) WHERE attempt_id=?',(previous['attempt_id'],)):
            path=local_path(row['uri'])
            state=writer.connection.execute('SELECT file_id,presence FROM current_file_state WHERE uri=? AND sha256=? ORDER BY event_id DESC LIMIT 1',(row['uri'],row['sha256'])).fetchone()
            if state and state['presence']=='deleted':
                from .retention import retention_facts
                if any(value['reason_code'] in ('delete_completed','delete_recovered') for value in retention_facts(writer.connection,state['file_id'])['deletions']):continue
            if not path.is_file() or sha256_file(path)!=row['sha256']:
                raise ValueError('Committed publication is missing or corrupt')
        return True
    if previous and task['status']=='running' and not wrapper_ok:
        marker=writer.connection.execute(f"SELECT event_id FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' "
            "AND reason_code='disk_full' AND event_id>? ORDER BY event_id DESC LIMIT 1",(run,previous.get('event_id') or 0)).fetchone()
        from .interruption import read as read_interruption
        recorded=read_interruption(corpus.output_dir,run)
        cause={'details':{'interruption_cause':'disk_full','errno':28,'disk_full_event_id':marker[0] if marker else None,
            'recorded_at':recorded.get('recorded_at') if recorded else None}} if marker or recorded else {}
        with writer.transaction():
            writer.finish(run,task['task_id'],previous['attempt_id'],status='failed',reason_code='attempt_interrupted',**cause)
            if wrapper_run:
                # The killed wrapper's own tasks end with the barrier attempt: nothing is left running (K4BP-G1).
                for stale in writer.task_states(wrapper_run):
                    latest=writer.latest_attempt(stale['task_id']) if stale['status']=='running' else None
                    if latest:
                        writer.finish(wrapper_run,stale['task_id'],latest['attempt_id'],status='failed',reason_code='attempt_interrupted')
    if not (previous and task['status']=='running' and wrapper_ok):
        with writer.transaction():
            attempt=writer.attempt(task['task_id'],worker_id='single-publisher',device_id='cpu',
                fingerprint={'frozen_run_id':run,'stage':stage,'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
            writer.event(run,event_kind='lifecycle',reason_code='stage_started',status='running',task_id=task['task_id'],attempt_id=attempt)
        corpus._scheduler_publication={'run_id':run,'task_id':task['task_id']}
        try:
            callback()
        except Exception as error:
            if disk_full(error):
                raise  # an exhausted filesystem stops the run with its typed cause; resume repeats this barrier (W18 M10)
            import traceback
            with writer.transaction():
                writer.finish(run,task['task_id'],attempt,status='failed',reason_code=stage+'_exception',
                    error_type=type(error).__name__,message=str(error),traceback=''.join(traceback.format_exception(error)))
            return False
        finally:
            corpus._scheduler_publication=None
        wrapper=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE run_id=? AND event_kind='run' AND message=? ORDER BY event_id DESC LIMIT 1",(run,'wrapper:'+stage)).fetchone()
        wrapper_run=json.loads(wrapper[0])['wrapper_run_id']
    else:
        attempt=previous['attempt_id']
    outputs=writer.connection.execute("SELECT f.* FROM files f JOIN attempts a ON a.attempt_id=f.producer_attempt_id JOIN tasks t ON t.task_id=a.task_id WHERE t.run_id=? AND f.uri LIKE ?",(wrapper_run,(Path(corpus.output_dir).resolve()/'.glossapi/artifacts').as_uri()+'/%')).fetchall()
    wrapper_operation={'download':'download.wrapper','extract':'extract.wrapper','clean':'clean.clean','ocr':'ocr.wrapper','sections':'sections.wrapper','annotate':'annotate.wrapper','export':'export.jsonl'}[stage]
    outcome=next((task for task in writer.task_states(wrapper_run) if task['operation']==wrapper_operation),None)
    with writer.transaction():
        for index,row in enumerate(outputs):
            path=local_path(row['uri'])
            if not path.is_file():
                raise ValueError('Publication artifact is absent')
            writer.input(attempt,row['file_id'],'published:'+str(index),row['sha256'])
        if stage=='extract':
            for source in writer.connection.execute("SELECT DISTINCT i.file_id,i.sha256 FROM attempt_inputs i JOIN attempts a USING(attempt_id) JOIN tasks t USING(task_id) JOIN files f USING(file_id) WHERE t.run_id=? AND f.kind IN ('download_payload','source_payload','initial_input')",(wrapper_run,)):
                writer.input(attempt,source['file_id'],'source_reader:'+source['file_id'],source['sha256'])
        status='ok' if outcome and outcome['status']=='ok' else 'failed'
        details={'wrapper_run_id':wrapper_run,'bound_product_files':len(outputs),'recovered_wrapper_commit':wrapper_ok}
        if status=='failed' and stage=='download' and outcome is not None:
            # The publication replays committed acquisitions only. A source whose acquisition failed is that
            # document's own typed outcome (its download.acquire task); it does not fail the corpus-wide
            # publication of the others (K4BP-L1).
            failed=[row for row in writer.task_states(run) if row['operation']=='download.acquire' and row['status']=='failed']
            if failed and outcome['reason_code']!='download_exception':
                status='ok'
                details['documents_with_failed_acquisition']=[{'version_id':row['version_id'],'reason_code':row['reason_code']} for row in failed]
        failure={} if status=='ok' else {'message':'wrapper '+(outcome['operation'] if outcome else wrapper_operation)+' ended '
            +(str(outcome['status'])+' ('+str(outcome['reason_code'])+')' if outcome else 'without an outcome')}
        writer.finish(run,task['task_id'],attempt,status=status,reason_code=stage+'_completed' if status=='ok' else stage+'_exception',
            details=details,**failure)
    if status=='ok' and stage=='extract':
        from .retention import retire_downloaded_sources
        retire_downloaded_sources(writer,corpus.output_dir,run,getattr(corpus,'_retention_settings',{}))
    return status=='ok'


def _declared_url_policy(writer,options):
    if options.get('resume'):
        row=writer.connection.execute('SELECT settings_sha256 FROM runs WHERE run_id=?',(options['resume'],)).fetchone()
        if row is None:
            return None
        from ..ledger.reader import snapshot
        return snapshot(writer.connection,row[0]).get('url_policy')
    if options.get('plan') is not None:
        return options['plan'].get('data',options['plan']).get('settings',{}).get('url_policy')
    return (options.get('settings') or {}).get('url_policy')


class DiskFullError(OSError):
    """The corpus or staging filesystem ran out of space during a run (errno 28): a typed, resumable interruption."""
    code='disk_full'

    def __init__(self,message,*,run_id=None,location=None):
        import errno
        super().__init__(errno.ENOSPC,message)
        self.details={'errno':errno.ENOSPC,'run_id':run_id,'location':location,
            'action':'free space (or enlarge the filesystem), then resume: glossapi run --resume '+(run_id or 'RUN_ID')}


def disk_full(error):
    """True when an exception (or its cause chain) is an exhausted filesystem: ENOSPC or SQLite's disk-full."""
    import errno
    import sqlite3
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error,OSError) and error.errno in (errno.ENOSPC,errno.EDQUOT):return True
        if isinstance(error,sqlite3.Error) and (getattr(error,'sqlite_errorcode',None)==13 or 'disk is full' in str(error)):return True
        error=error.__cause__ or error.__context__
    return False


def _record_disk_full(writer,corpus,error):
    """Name the exhausted disk, so status, report and the resume can tell it from process death.

    First in the record allocated at run start (overwritten in place: it needs no new block on a full disk), then,
    best effort, as a ledger event.
    """
    run=getattr(corpus,'_active_run_id',None)
    if not run:return
    from .interruption import record
    record(corpus.output_dir,run,error)
    try:
        if writer.connection.in_transaction:writer.connection.rollback()
        with writer.transaction():
            writer.event(run,event_kind='run',reason_code='disk_full',status=None,
                message='filesystem full (errno 28); free space, then resume this run',
                details={'cause':'disk_full','errno':28,'error_type':type(error).__name__,'error':str(error)[:500]})
    except Exception:
        pass  # the disk may be too full even for this record; the typed error below still reaches the caller


def run_corpus(corpus,*args,**options):
    """Hold corpus exclusion across document work and every publication barrier."""
    try:
        return _locked_run_corpus(corpus,*args,**options)
    except DiskFullError:
        raise
    except Exception as error:
        if disk_full(error):
            # A typed explanation instead of an escaped sqlite/OSError (W18 M10).
            raise DiskFullError('disk_full: the corpus filesystem is full (errno 28): '+str(error),
                run_id=getattr(corpus,'_active_run_id',None),location=str(corpus.output_dir)) from error
        raise
    finally:
        corpus._active_run_id=None


def _locked_run_corpus(corpus,*args,**options):
    began=time.monotonic()
    from ..ledger.urls import url_policy
    corpus._active_run_id=options.get('resume')
    with LedgerWriter(Path(corpus.output_dir)/'corpus.sqlite3') as writer:
        corpus._scheduler_writer=writer
        # The disk_full record exists before any write can exhaust the disk (K4BI-G2).
        from .interruption import allocate
        allocate(corpus.output_dir)
        # Copies the previous run close left unflushed are verified (and republished if torn) first (K4C-M2).
        writer.verify_excluded_copies()
        try:
            # The caller's declared URL policy (public query fields, secret
            # headers) is frozen in the run settings and active for every
            # recording boundary of this run, including a resume.
            with url_policy(_declared_url_policy(writer,options)):
                result=_run_corpus(corpus,*args,**options)
            result['total_wall_seconds']=time.monotonic()-began
            if result.get('state') in ('completed','completed_with_failures'):
                from .interruption import clear
                clear(corpus.output_dir)
            return result
        except Exception as error:
            if disk_full(error):
                _record_disk_full(writer,corpus,error)
            raise
        finally:
            corpus._scheduler_writer=None
            corpus._streaming_download_replays=None
            corpus._streaming_ocr_replays=None
            corpus._publication_claims=None
            corpus._streaming_planned_versions=None
            corpus._streaming_completed_versions=None
            corpus._streaming_publication_stems=None
