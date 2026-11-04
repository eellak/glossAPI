"""Frozen run plans, streaming document work and compatibility barriers."""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import json
from pathlib import Path
import time
from typing import Any, cast
from urllib.parse import unquote, urlsplit

from .contracts import ArtifactReference, DocumentSnapshot, StageSpecification
from .execution import disk_full, sha256_file  # disk_full: one definition, shared with the worker typing
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
    """Resource identity of one source URL: a declared ``source_id``, else ``url:v1:``."""
    from ..ledger.journal import source_key
    if source_id is not None and not (isinstance(source_id,float) and source_id!=source_id):
        from ..ledger.identity import source_key as declared_key
        return declared_key(None,dataset_id=dataset_id if isinstance(dataset_id,str) and dataset_id else 'glossapi:declared_source_id',
            source_id=source_id,resource_index=resource_index)
    return source_key(raw_url)


URL_INDEX_MAX=2**63-1  # the ledger stores url_index as a SQLite INTEGER (signed 64-bit)


def _url_index_value(value: Any, index: int | None = None) -> int | None:
    """The url_index a cell declares: an int, None for a missing cell, or the typed refusal (CP-K4g F3, CP-K4gc2 R3).

    Accepted: ``int`` (not bool), numpy/pandas integer scalars, and finite integral ``float`` (Python or numpy) from 0 to
    ``URL_INDEX_MAX``. Integers stay exact. A missing cell is None, NaN or pandas NA. Every other type (bool,
    strings, ``Fraction``, ``Decimal``, other numeric types) is refused with ``invalid_manifest``: nothing is converted
    lossily before it is judged, and an out-of-range value is a typed refusal, never an OverflowError."""
    import sys
    numpy: Any = sys.modules.get('numpy')  # a numpy scalar implies numpy is loaded; never import it here
    pandas: Any = sys.modules.get('pandas')
    if value is None or (pandas is not None and value is pandas.NA):
        return None
    number: int | None = None
    if isinstance(value, bool) or (numpy is not None and isinstance(value, numpy.bool_)):
        number = None
    elif type(value) is int or (numpy is not None and isinstance(value, numpy.integer)):
        number = int(value)
    elif type(value) is float or (numpy is not None and isinstance(value, numpy.floating)):
        floating = float(value)  # exact for Python floats; numpy floats widen exactly to a Python float
        if floating != floating:
            return None  # NaN: a missing cell
        if floating not in (float('inf'), float('-inf')) and floating.is_integer():
            number = int(floating)
    if number is None or number < 0 or number > URL_INDEX_MAX:
        from ..input_manifest import ManifestError
        raise ManifestError('invalid_manifest', f'url_index must be an integer from 0 to {URL_INDEX_MAX}', field='url_index',
                            index=index, value=str(value))
    return number


def row_resource_index(row: Any, *, index: int | None = None) -> int:
    """The resource index of a one-URL source row (CRED-F8): its declared ``url_index`` (``_url_index_value``), else 0.

    Plan (``cli._plan_data``) and run (``_remote_inventory``) both key a declared ``source_id`` by
    (dataset_id, source_id, this index), so two rows of one source_id with url_index 0 and 1 are two resources in
    both. A list-valued URL cell is expanded by run and keyed by each URL's position instead."""
    number = _url_index_value(row.get('url_index') if isinstance(row, dict) else None, index)
    return 0 if number is None else number


def normalize_url_index_cells(manifest: Any) -> Any:
    """A manifest whose url_index cells are plain ints or None (CP-K4gc2 R4), so plan hashing and the identities plan
    and run build see the same value. Invalid cells raise the typed refusal; the caller's objects are not mutated."""
    def documents(value: Any) -> Any:
        rows = value
        if not isinstance(rows, list):
            return value
        changed = False
        out = []
        for index, row in enumerate(rows):
            if isinstance(row, dict) and 'url_index' in row:
                normalized = _url_index_value(row['url_index'], index)
                if type(row['url_index']) is not int or isinstance(row['url_index'], bool) or normalized != row['url_index']:
                    row = {**row, 'url_index': normalized}
                    changed = True
            out.append(row)
        return out if changed else value
    if isinstance(manifest, list):
        return documents(manifest)
    if isinstance(manifest, dict) and isinstance(manifest.get('documents'), list):
        rows = documents(manifest['documents'])
        return manifest if rows is manifest['documents'] else {**manifest, 'documents': rows}
    return manifest


def _bind(found,identity,url,*,index=None):
    """One URL per resource identity (CRED-F8): plan and run refuse a second, different URL with the same typed error."""
    previous=found.get(identity)
    if previous is not None and previous!=url:
        from ..input_manifest import ManifestError
        raise ManifestError('invalid_manifest','ambiguous_source_identity: two different source URLs map to one resource identity ('
            +url+'); give each resource its own source_id or url_index, or supply one URL per source_id',
            field='source_id',index=index,reason='ambiguous_source_identity',identity=identity,urls=[previous,url])
    found[identity]=url


def _corpus_download_names(root: Any) -> tuple[dict[str, str], set[str]]:
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


def _ledger_compatibility_names(writer: Any) -> dict[str, set[str]]:
    """Every compatibility name a frozen plan of this corpus ever gave a source: ``{stem: {source_key, ...}}``.

    Download names (``download.acquire`` ``filename_base``) and local publication names (``extract.document``
    ``publication_name``, else the local source's own stem), joined to the document's source identity. A name held by
    one source identity is never given to another in a later run, so a filename match can never borrow another
    source's metadata row (CP-K4a F1).
    """
    from .._naming import canonical_stem
    claimed: dict[str, set[str]] = {}
    if writer is None:
        return claimed
    rows: list[Any]=writer.connection.execute("""SELECT t.operation,s.json,d.source_key FROM tasks t
        JOIN snapshots s ON s.sha256=t.specification_sha256
        JOIN document_versions v ON v.version_id=t.version_id JOIN documents d ON d.document_id=v.document_id
        WHERE t.operation IN ('download.acquire','extract.document')""").fetchall()
    for operation,specification,key in rows:
        frozen = cast(dict[str, Any], json.loads(specification))
        operation_json = cast(dict[str, Any], frozen.get('scheduler_operation') or {})
        specification_json = cast(dict[str, Any], operation_json.get('specification') or {})
        settings = cast(dict[str, Any], specification_json.get('settings') or frozen.get('settings') or {})
        name: Any
        if operation=='download.acquire':
            name=settings.get('filename_base')
        elif settings.get('publication_name'):
            name=Path(settings['publication_name']).stem
        elif str(key).startswith('local_path:v1:'):
            name=canonical_stem(local_path(str(key).split(':v1:',1)[1]))
        else:
            continue  # a remote document's extraction uses its download name, recorded above
        if name:
            claimed.setdefault(canonical_stem(str(name)),set()).add(key)
    return claimed


def _rename_against_corpus(frames: Any, identities: Any, root: Any, claimed: dict[str, set[str]] | None = None) -> bool:
    """Re-derive ``filename_base`` in every frame from the corpus' earlier names (same rows, same identities).

    ``claimed`` adds the names earlier frozen plans gave any source identity (``_ledger_compatibility_names``): a URL
    keeps a name the ledger records for its own identity and never takes one recorded for another identity.
    """
    from ..gloss_downloader import GlossDownloader
    by_identity,taken=_corpus_download_names(root)
    for name,keys in (claimed or {}).items():
        taken.add(name)
        for key in keys:
            if key in identities and len(keys)==1:by_identity.setdefault(key,name)
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


def _remote_inventory(rows,root,claimed: dict[str, set[str]] | None = None):
    """Expand, deduplicate and name remote rows, in memory.

    Returns the expanded source rows (for the ledger), the URL and identity of
    every expanded row, and the download-projection frame. The retained kernels
    perform no network or backend work.
    """
    import pandas as pd
    from ..gloss_downloader import GlossDownloader
    rows=[dict(row) if isinstance(row,dict) else {'url':row} for row in rows]
    for row in rows:
        if 'url' not in row or not isinstance(row['url'],(str,list,tuple)) or (isinstance(row['url'],str) and not row['url'].strip()):
            row['url']=_row_url(row)
    frame=pd.DataFrame(rows)
    naming=GlossDownloader.__new__(GlossDownloader)
    naming.url_column='url';naming._used_filename_bases=set()
    named=naming.ensure_filename_base(naming._expand_and_mark_duplicates(frame.copy()))
    urls=[value if isinstance(value,str) else '' for value in named['url']]
    expanded=any(isinstance(row.get('url'),(list,tuple,dict)) or (isinstance(row.get('url'),str) and row['url'].strip().startswith(('[','{')))
                 for row in rows)
    identities=[];positions=[]
    for index,url in enumerate(urls):
        identity=None
        position=int(named['source_row'].iloc[index]) if expanded else index
        if _is_url(url):
            row=rows[position]
            # A list cell's URL is keyed by its position; a one-URL row by its declared url_index (CRED-F8), as plan does.
            cell=row.get('url')
            listed=isinstance(cell,(list,tuple,dict)) or (isinstance(cell,str) and cell.strip().startswith(('[','{')))
            identity=resource_identity(url,source_id=row.get('source_id'),dataset_id=row.get('dataset_id'),
                resource_index=max(0,int(named['url_index'].iloc[index])) if listed else row_resource_index(row,index=position))
        identities.append(identity);positions.append(position)
    seen={}
    for identity,url,position in zip(identities,urls,positions):
        if identity is not None:
            _bind(seen,identity,url,index=position)
    # A corpus that already holds downloads keeps their names; new sources never take a name in use (K4BP-G4).
    renamed=_rename_against_corpus((named,),identities,root,claimed)
    declared=any(isinstance(identity,str) and identity.startswith('source:v1:') for identity in identities)
    projection=frame
    if not expanded:
        # One identity per source row: the download serializer records it with
        # each result row and resumes by it. The frozen names travel with the
        # rows, so the publication replays exactly them whatever the corpus
        # holds by then (K4BP-G1).
        projection=frame.copy()
        projection['resource_identity']=identities
        projection['filename_base']=list(named['filename_base'].astype(str))
    elif declared or renamed:
        # Expanded rows with declared identities, or names fixed by the corpus: record per expanded row.
        projection=named.drop(columns=['__url_norm'])
        projection['resource_identity']=identities
    return named.to_dict(orient='records'),urls,identities,projection


def _manifest_selector_uri(row, writer):
    """The source URI a local manifest row names, or None (same selectors as the CLI)."""
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
        return selector
    return Path(unquote(urlsplit(selector).path) if selector.startswith('file:') else selector).resolve().as_uri()


def _publication_names(entries: list[dict[str, Any]], remote_bases: list[Any], claimed: dict[str, set[str]] | None = None) -> dict[str, str]:
    """Collision-free compatibility publication names for local inputs (CP-LEDGER F1, CP-K4a F1).

    Legacy serializers key every product by the source basename/stem. Inputs
    whose canonical stems are unique keep exactly their current names. When
    distinct inputs share one, each local input is published under a name
    derived from its own source identity, so no product is overwritten.
    A stem that an earlier frozen plan gave a *different* source identity
    (``claimed``: ``{stem: {source_key}}`` from the ledger, remote or local) is
    reserved too: the local input then takes its identity-derived name, so its
    products and metadata row never meet the other source's row.
    """
    import hashlib
    from .._naming import canonical_stem
    claimed = claimed or {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if entry['path'] is not None:
            groups.setdefault(canonical_stem(entry['path'].name), []).append(entry)
    names: dict[str, str] = {}
    remote = {canonical_stem(name) for name in remote_bases}
    def held(stem: str, members: list[dict[str, Any]]) -> bool:
        return bool(claimed.get(stem, set()) - {entry['key'] for entry in members})
    taken = remote | set(claimed) | {stem for stem, members in groups.items() if len(members) == 1}
    for stem, members in groups.items():
        if len(members) == 1 and stem not in remote and not held(stem, members):
            continue
        for entry in members:
            digest = hashlib.sha256(entry['key'].encode('utf-8')).hexdigest()
            # A source that already holds an identity-derived name keeps it (reintroduction is stable).
            own = sorted(name for name, keys in claimed.items() if keys == {entry['key']} and name.startswith(f'{stem}__{digest[:10]}'))
            if own:
                names[entry['key']] = f'{own[0]}{entry["path"].suffix}'
                continue
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


def _publication_sources(operations: Any, writer: Any) -> dict[str, dict[str, Any]]:
    """Per planned compatibility stem: its frozen source (``url``: the download URL, None for a local input) and the
    URLs and identities of every *other* source an earlier frozen plan downloaded under that name (``foreign_urls``,
    ``foreign_keys``). The export refuses a planned stem's metadata row that records one of those other sources
    (CP-K4a F1: a filename match never lends another source's provenance). Rows the caller supplied or a legacy
    download wrote (no frozen plan gave the name to another identity) keep their legacy meaning."""
    stems = cast(dict[Any, str], _publication_stems(operations))
    remote: dict[Any, str] = {operation.document.version_id:operation.document.source_uri for operation in operations if operation.stage=='download'}
    documents: dict[Any, Any] = {operation.document.version_id:operation.document.document_id for operation in operations}
    from .compatibility import download_name_claims
    claims = download_name_claims(writer.connection)
    sources: dict[str, dict[str, Any]] = {}
    for version,stem in stems.items():
        known: Any = writer.connection.execute('SELECT source_key FROM documents WHERE document_id=?',(documents.get(version),)).fetchone()
        own = known[0] if known else None
        foreign = sorted((key,uri) for key,uri in claims.get(stem,set()) if key!=own)
        sources[stem] = dict(url=remote.get(version), foreign_urls=sorted({uri for _,uri in foreign}), foreign_keys=sorted({key for key,_ in foreign}))
    return sources


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
                      resume=None,retry_failed=False,observer=None,writer=None):
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
        raw_manifest=settings.get('source_manifest')
        manifest_rows=(raw_manifest.get('documents',[]) if isinstance(raw_manifest,dict) else raw_manifest) if raw_manifest is not None else None
        # Public targets only (PLAN §9): every supplied source and manifest row, and the download settings, before
        # anything is recorded. A resume re-checks its frozen operations in StreamingScheduler._load.
        from ..ledger.urls import refuse_credentials
        refuse_credentials(urls=[*sources,*(manifest_rows if isinstance(manifest_rows,list) else [])],
            settings=[settings.get('download'),getattr(corpus,'downloader_config',None)])
        if not resume:
            # Selectors are resolved per row: remote rows are expanded and named;
            # local rows keep their real paths (cycle-2 F3).
            remote_manifest=[row for row in manifest_rows if _row_url(row) is not None] if isinstance(manifest_rows,list) else []
            remote_given=[str(source) for source in sources if _is_url(str(source))]
            if remote_manifest or remote_given:
                remote_rows=remote_manifest or [{'url':source} for source in remote_given]
                source_rows,urls,identities,frame=_remote_inventory(remote_rows,root,_ledger_compatibility_names(writer))
                # Retain the source frame for the unchanged download serializer.
                # Resume binds these bytes, not a reconstructed frame.
                from ..ledger.writer import new_id
                download_manifest=root/'.glossapi/source-manifests'/new_id()/'sources.parquet'
                download_manifest.parent.mkdir(parents=True,exist_ok=False)
                frame.to_parquet(download_manifest,index=False)
                settings={**settings,'download_projection_manifest':{'path':str(download_manifest),
                    'sha256':sha256_file(download_manifest),'bytes':download_manifest.stat().st_size},
                    'semantic_files':[*settings.get('semantic_files',()),
                        {'path':str(download_manifest),'sha256':sha256_file(download_manifest)}]}
                remote_entries=[dict(source=row.get('url'),remote=True,key=identity)
                    for row,identity in zip(source_rows,identities)]
                planned=settings.get('planned_document_ids')
                if planned is not None and any(entry['key'] is not None and entry['key'] not in planned for entry in remote_entries):
                    raise ValueError('planned_source_identity_mismatch: a supplied source URL does not key to its planned resource '
                        'identity; plan again with these URLs')
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
                require_room(root,cast(dict[str, Any],settings),product_need(cast(Any,stages),int(PLANNING_TEXT_RATIO*sum(sizes)),len(sizes)),when='(estimated at run start)')
        engine=StreamingScheduler(writer,root,workers=workers,cpu_tokens=settings.get('cpu_tokens'),
            memory_bytes=settings.get('memory_budget_bytes'),disk_budget_bytes=settings.get('disk_budget_bytes'),observer=observer)
        unavailable={}
        if resume:
            run=resume
            operations,_,run_row=engine._load(run)
            frozen=json.loads(writer.connection.execute('SELECT json FROM snapshots WHERE sha256=?',(run_row['settings_sha256'],)).fetchone()[0])
            stages=frozen.get('requested_stages',stages)
            settings=frozen
        else:
            operations=[]
            from ..ledger.journal import safe_settings, code_revision
            entries=[]
            # 1. The requested inventory is durable before any fallible per-file
            # inspection: one missing or unreadable input cannot roll back the
            # other inputs' census (CP-LEDGER F3).
            with writer.transaction():
                observation=writer.run(settings,{'mode':'explicit_source_inventory'}, {},code_revision=code_revision())
                if download_manifest:
                    writer.artifact(observation,'download','source_manifest',download_manifest.as_uri(),
                        sha256=sha256_file(download_manifest),bytes=download_manifest.stat().st_size,
                        ownership='managed',retention_class='initial')
                given=[*remote_entries,*(dict(source=str(source),remote=False,key=None) for source in local_given)]
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
            publication=_publication_names(cast(list[dict[str, Any]],[entry for entry in unique if 'failure' not in entry]),cast(list[Any],remote_bases),_ledger_compatibility_names(writer))
            version_of={}
            from ..markdown_layout import mode as layout_mode
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
                    published=cast(Any,publication.get(entry['key']))
                    naming: dict[str, Any]={'publication_name':published} if published else {}
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
                            'url':source,'filename_base':name,'row_index':ordinal}
                        parent=f'{ordinal}:download'
                        operations.append(ScheduledOperation(parent,'download',snapshot,
                            StageSpecification('download.acquire',required_output_kinds=('download_payload',),settings=download_settings),
                            skip_reason='duplicate_url' if source_rows[ordinal].get('is_duplicate') else None))
                    if 'extract' in stages:
                        key=f'{ordinal}:extract'
                        operations.append(ScheduledOperation(key,'extract',snapshot,
                            StageSpecification('extract.document',('source_payload',) if artifacts else (),('raw_markdown',),
                                settings=cast(Any,{**extraction_settings(settings.get('extract')),**naming})),
                            inputs=((parent,'download_payload','source_payload'),) if parent else (),skip_reason=skip))
                        parent=key
                    # K5-A2b: in the default Markdown mode the safe extractor's sidecar (its layout strip record) is
                    # the clean's bound layout input, so text metrics are measured on the markup-free text. Plain mode
                    # and Docling extraction: one input, as before.
                    layout_edges=(((f'{ordinal}:extract','extract_sidecar','layout_metadata'),) if 'extract' in stages
                        and layout_mode()=='layout_v2' and extraction_settings(settings.get('extract')).get('phase1_backend','safe')=='safe' else ())
                    if 'clean' in stages:
                        key=f'{ordinal}:clean'
                        operations.append(ScheduledOperation(key,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=cast(Any,((parent,'raw_markdown','selected_text'),*layout_edges)),skip_reason=skip))
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
                                settings=cast(Any,{'page_quality':settings.get('page_quality',{}),**naming})),
                            inputs=((f'{ordinal}:download','download_payload','source_payload'),) if remote else (),skip_reason=skip))
                        source_snapshot=replace(snapshot,artifacts=() if remote else snapshot.artifacts)
                        edges=((f'{ordinal}:download','download_payload','source_payload'),) if remote else ()
                        ocr_key=f'{ordinal}:ocr'
                        operations.append(ScheduledOperation(ocr_key,'ocr',source_snapshot,
                            StageSpecification('ocr.document',('source_payload',) if not remote else (),('ocr_markdown','ocr_metrics'),device,
                                cast(Any,{**ocr_settings,'options':values,'entry_point_name':'glossapi',
                                 'cache_root':str(cache_root(ocr_settings.get('cache_root'))),**naming})),
                            inputs=(*edges,(f'{ordinal}:extract','raw_markdown','native_markdown'),(quality_key,'page_quality','page_quality'),*layout_edges),
                            after=(parent,),cpu_tokens=int(ocr_settings.get('num_threads',1)),skip_reason=skip))
                        parent=f'{ordinal}:clean_after_ocr'
                        operations.append(ScheduledOperation(parent,'clean',replace(snapshot,artifacts=()),
                            StageSpecification('clean.initial.artifact:after_ocr',required_output_kinds=('clean_markdown','clean_metrics'),settings={'num_threads':1}),
                            inputs=((ocr_key,'ocr_markdown','selected_text'),*layout_edges),skip_reason=skip))
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
        if 'annotate' in stages:
            # Overlap the annotation barrier's legacy model import/unpickle with the DAG (K4(c) part 3, decision 3).
            from ..gloss_section_classifier import preload_legacy_model
            model_path:object=getattr(corpus,'section_classifier_model_path',None)
            preload_legacy_model(model_path)
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
        corpus._streaming_publication_sources=_publication_sources(operations,writer)
        result['plan_hash']=writer.connection.execute('SELECT plan_sha256 FROM runs WHERE run_id=?',(run,)).fetchone()[0]
        if resume:
            settings=json.loads(writer.connection.execute('SELECT s.json FROM runs r JOIN snapshots s ON s.sha256=r.settings_sha256 WHERE r.run_id=?',(run,)).fetchone()[0])
    corpus._streaming_source_versions=source_versions
    corpus._streaming_download_replays=download_replays if settings.get('download_projection_manifest') else None
    corpus._streaming_download_manifest=settings.get('download_projection_manifest')
    return result,replays


def _run_corpus(corpus,files=(),*,stages=('extract','clean','sections','export'),workers=None,settings=None,
               resume=None,retry_failed=False,plan=None,observer=None):
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
        from ..ledger.identity import source_uri_of_key
        # The plan holds the manifest rows as given; the run expands and keys them
        # again, and their identities must equal the planned ones (checked there).
        explicit=body.get('explicit_manifest')
        files=[]
        for document in body['documents']:
            observation=document.get('source_observation',{})
            uri=observation.get('source_uri') or source_uri_of_key(document['source_key'])
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
            settings={**settings,'source_manifest':explicit}
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
        resume=resume,retry_failed=retry_failed,observer=observer,writer=corpus._scheduler_writer)
    _bind_barrier_context(corpus._scheduler_writer,result.get('run_id'))
    if result.get('run_id') and result['state'] in ('completed','completed_with_failures') and getattr(corpus,'_retention_settings',{}).get('memory_saving')=='extreme':
        # Before any publication (K4BI-G7): the exact cleaned text of this run decides whether the final products fit;
        # a typed refusal now, never an exhausted disk mid-publication. Resume after enlarging the filesystem.
        from .final_products import check_exact
        publications=[*stages,*(['download'] if getattr(corpus,'_streaming_download_manifest',None) else [])]
        check_exact(corpus,result,publications,when='(exact, before the publications)')
    if result.get('run_id') and getattr(cast(Any,corpus),'_retention_settings',{}).get('memory_saving')=='extreme':
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
    calls: dict[str, Any]={
        # The publication replays committed acquisitions under the run's download settings.
        'download':lambda:corpus.download(corpus._streaming_download_manifest['path'],**settings.get('download',{})),
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
    # The OCR publication re-cleans every raw markdown in markdown/, so public raw markdown is still read until this
    # invocation's OCR barrier has returned. retention's own check (the OCR publication task is not ok) misses a
    # --retry-failed resume, where that task is ok from the first pass but publishes again because the covered
    # versions changed: the clean frontier released the other documents' raw markdown first, and the repeated OCR
    # publication then dropped their clean text and marked them extract_failed (K4g fix 4).
    ocr_barrier_pending='ocr' in stages
    with committed_results(replays):
        for stage in ('download','extract','clean','ocr','sections','annotate','export'):
            if stage in stages or (stage=='download' and corpus._streaming_download_manifest):
                published=False
                if not result['sources'] and stage not in ('download','export'):
                    skip_barrier(corpus,result['run_id'],stage)
                else:
                    if getattr(corpus,'_retention_settings',{}).get('memory_saving')=='extreme':
                        # Again before EACH publication's first write, from the state on disk now: this publication and
                        # every later one (CP-K4c3 R3; CP-K4ac2 R3 re-checked only the export).
                        from .final_products import check_exact, PUBLICATION_ORDER
                        later=PUBLICATION_ORDER[PUBLICATION_ORDER.index(stage):]
                        pending=[s for s in later if s in stages or (s=='download' and corpus._streaming_download_manifest)]
                        check_exact(corpus,result,pending,when=f'(exact, at the {stage} publication)')
                    published=publish_barrier(corpus,result['run_id'],stage,calls[stage],retry_failed=retry_failed,
                        empty_export=not result['sources'] and stage=='export')
                if stage=='ocr':
                    ocr_barrier_pending=False
                # The next compatibility reader has now consumed these private
                # producer records. Retire at that frontier, not at run close.
                released_stage = {'clean':'extract','sections':'clean','annotate':'sections'}.get(stage)
                if released_stage and published and not _has_failed_documents(corpus._scheduler_writer,result['run_id']):
                    publication_budget.observe()
                    from .retention import retain_completed_run
                    attempts = [attempt for key,attempt in result['attempts'].items()
                        if key.endswith(':'+released_stage)]
                    retain_completed_run(corpus._scheduler_writer,corpus.output_dir,result['run_id'],
                        getattr(corpus,'_retention_settings',settings),attempts=attempts,
                        observer=getattr(corpus,'_retention_observer',None),keep_public_raw=ocr_barrier_pending)
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
        # A run that ends with failed documents keeps the products of its document attempts: a --retry-failed resume
        # that brings the failed ones to ok re-publishes every corpus product from them (K4g B1c), and the completing
        # resume releases them. The publications' own scratch and temporaries are released as before.
        if _has_failed_documents(writer,result['run_id']):
            documents=set(result['attempts'].values())
            all_attempts=[attempt for attempt in all_attempts if attempt not in documents]
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


OCR_PUBLISHED_KINDS=('ocr_markdown','ocr_metrics','ocr_native','ocr_page_state')


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
                if output.kind in OCR_PUBLISHED_KINDS}
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


from .retention import PublicationInputsReleasedError  # noqa: E402  (typed refusal, shared with scheduler recovery)


def _republication_inputs_missing(corpus,stage):
    """Committed document products a repeated per-document publication (extract, ocr) materializes again, that are
    gone. The corpus-level publications (clean, sections, annotate, export) recompute from the public products of the
    publications before them and read no released private file."""
    if stage not in ('extract','ocr'):
        return []
    from .workers import read_result
    writer=corpus._scheduler_writer
    missing=[]
    for (replay_stage,_,_),candidates in (getattr(corpus,'_streaming_ocr_replays',None) or {}).items():
        if replay_stage!=stage:
            continue
        for _,_,receipt,workspace in candidates:
            for output in read_result(receipt.path).outputs:
                relative=output.relative_path
                if relative.startswith(('logs/','inputs/')) or output.kind in ('attempt_diagnostic','worker_receipt'):
                    continue
                if stage=='ocr' and output.kind not in OCR_PUBLISHED_KINDS:
                    continue  # publish_ocr_results materializes only these
                path=Path(workspace)/relative
                if path.is_file():
                    continue
                if output.kind=='repaired_pdf' and writer.connection.execute(
                        "SELECT 1 FROM current_file_state WHERE uri=? AND presence='deleted' LIMIT 1",(path.as_uri(),)).fetchone():
                    continue  # released under extreme after extraction; the publication skips it too
                missing.append(path)
    return missing


def _has_failed_documents(writer,run):
    """Whether document releases are suspended (``retention.document_releases_suspended``, the one shared rule)."""
    from .retention import document_releases_suspended
    return document_releases_suspended(writer,run)


def _publication_coverage(corpus):
    """The identity of the document versions a corpus publication covers now (those whose every document stage is
    ok): a sha256 over the sorted version ids, or None outside a streaming run. A hash, not the list, so a barrier
    record stays the same size for any corpus."""
    completed=getattr(corpus,'_streaming_completed_versions',None)
    if completed is None:
        return None
    from ..ledger.encoding import snapshot_hash
    return snapshot_hash(sorted(str(version) for version in completed if version is not None))


def _published_coverage(writer,run,task_id,attempt):
    """The versions the barrier's committed attempt recorded as covered, or None for a commit made before coverage was
    recorded (that commit keeps its old meaning: current)."""
    if not attempt:
        return None
    row=writer.connection.execute(f"SELECT details_json FROM {writer.connection.event_table} WHERE task_id=? AND attempt_id=? "
        "AND event_kind='lifecycle' AND status='ok' ORDER BY event_id DESC LIMIT 1",(task_id,attempt['attempt_id'])).fetchone()
    if not row:
        return None
    from ..ledger.reader import evidence
    return evidence(writer.connection,{'details_json':row[0]}).get('covered_versions_sha256')


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
    covering=_publication_coverage(corpus)
    if task['status']=='ok' and _published_coverage(writer,run,task['task_id'],previous) not in (None,covering):
        # The document versions this publication covers changed since it committed (a --retry-failed resume brought
        # failed documents to ok, or an earlier commit covered fewer): it is not current, so it publishes again, from
        # the committed document products, which must all still exist (refused, typed, before any write otherwise).
        missing=_republication_inputs_missing(corpus,stage)
        if missing:
            raise PublicationInputsReleasedError(stage,missing)
        task={**task,'status':'stale'}
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
        # Tied to the interrupted barrier attempt's start (CP-K4a F3).
        from .interruption import consume, verified_cause
        ledger=cast(Any,writer.connection)
        found=verified_cause(ledger,cast(Any,corpus.output_dir),cast(str,run),attempt_id=cast(str,previous['attempt_id']),table=ledger.event_table)
        cause={'details':{'interruption_cause':'disk_full','errno':28,'disk_full_event_id':found['event_id'],
            'recorded_at':found['recorded_at']}} if found else {}
        with writer.transaction():
            writer.finish(run,task['task_id'],previous['attempt_id'],status='failed',reason_code='attempt_interrupted',**cause)
            if wrapper_run:
                # The killed wrapper's own tasks end with the barrier attempt: nothing is left running (K4BP-G1).
                for stale in writer.task_states(wrapper_run):
                    latest=writer.latest_attempt(stale['task_id']) if stale['status']=='running' else None
                    if latest:
                        writer.finish(wrapper_run,stale['task_id'],latest['attempt_id'],status='failed',reason_code='attempt_interrupted')
        if found and found['basis']=='record allocated at run start':
            consume(cast(Any,corpus.output_dir),cast(str,run))
    if not (previous and task['status']=='running' and wrapper_ok):
        if previous and task['status'] in ('failed','running'):
            # A failed (or killed) attempt of this publication may have left bytes it never committed: metadata it
            # rewrote in place, mirrors it replaced before their registration. Each was recorded as its own before it
            # was written; the exact recorded bytes are rolled back now, with the space the operator freed (fix 7,
            # R1). A user's edit (other bytes) is never touched. Otherwise the repeated publication would capture its
            # own earlier bytes as unchanged and publish nothing (the OCR publication), or the start would refuse them.
            writer.roll_back_uncommitted_writes(_publication_wrapper_runs(writer,run,stage,task['task_id']),corpus.output_dir,
                observation_run=run)
            if writer.rollback_unresolved:
                # Fix 10 (N1, N3): a target the rollback recognised but could not resolve stops the run here, typed,
                # before the publication repeats over it.
                from ..ledger.writer import UnresolvedRecoveryError
                raise UnresolvedRecoveryError(writer.rollback_unresolved, boundary='publication')
        with writer.transaction():
            attempt=writer.attempt(task['task_id'],worker_id='single-publisher',device_id='cpu',
                fingerprint={'frozen_run_id':run,'stage':stage,'reuse_eligible':False},inputs={'files':[]},settings={},tools={})
            writer.event(run,event_kind='lifecycle',reason_code='stage_started',status='running',task_id=task['task_id'],attempt_id=attempt)
        corpus._scheduler_publication={'run_id':run,'task_id':task['task_id']}
        try:
            callback()
        except Exception as error:
            if disk_full(error):
                # An exhausted filesystem stops the run with its typed cause; resume repeats this barrier (W18 M10). The
                # failed wrapper may already have rewritten compatibility products in place (a state save, a metric
                # merge before the failing write, mirrors replaced before their registration): those bytes were never
                # committed. They are rolled back now if the disk allows (fix 6, F3); on a disk still too full for the
                # copy, the bytes were recorded as this run's own before they were written, and the next start (named
                # copies) and the repeated barrier roll them back once space is freed (fix 7, R1).
                _roll_back_uncommitted_projections(writer,corpus,run,stage,task['task_id'])
                raise
            import traceback
            _roll_back_uncommitted_projections(writer,corpus,run,stage,task['task_id'])  # the same after any failure (fix 6)
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
        if covering is not None:
            details['covered_versions_sha256']=covering
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
        retire_downloaded_sources(writer,corpus.output_dir,run,getattr(cast(Any,corpus),'_retention_settings',{}))
    return status=='ok'


def _publication_wrapper_runs(writer,run,stage,task_id):
    """The wrapper runs of one publication barrier task, oldest first (each attempt of the barrier starts one)."""
    table=writer.connection.event_table
    found=[]
    for encoded, in writer.connection.execute(f"SELECT details_json FROM {table} WHERE run_id=? AND event_kind='run' "
                                              "AND reason_code='legacy_observed' AND message=? ORDER BY event_id",(run,'wrapper:'+stage)):
        details=json.loads(encoded)
        if details.get('publication_task_id')==task_id and details.get('wrapper_run_id'):
            found.append(details['wrapper_run_id'])
    return found


def _roll_back_uncommitted_projections(writer,corpus,run,stage,task_id):
    """Best effort, right after a failed barrier: undo the failed attempt's uncommitted writes
    (``LedgerWriter.roll_back_uncommitted_writes``: exactly the bytes it recorded as its own before writing them), then
    restore every compatibility metadata copy whose current registration is its committed projection
    (``restore_metadata_projections``). Both may need free space (a temporary copy). On a disk still too full, the
    original error still propagates and nothing is lost: the uncommitted bytes stay recorded as this run's own, so the
    next start rolls back a copy the run close named (``verify_excluded_copies``) and the repeated barrier rolls back
    the rest (``publish_barrier``), once space is freed (fix 7, R1). A user's edit never matches such a record."""
    from types import SimpleNamespace
    from .metadata import restore_metadata_projections
    try:
        writer.roll_back_uncommitted_writes(_publication_wrapper_runs(writer,run,stage,task_id),corpus.output_dir,observation_run=run)
    except Exception:
        pass
    try:
        restore_metadata_projections(SimpleNamespace(writer=writer,corpus=corpus))
    except Exception:
        pass


class DiskFullError(OSError):
    """The corpus or staging filesystem ran out of space during a run (errno 28): a typed, resumable interruption."""
    code='disk_full'

    def __init__(self,message,*,run_id=None,location=None):
        import errno
        super().__init__(errno.ENOSPC,message)
        self.details={'errno':errno.ENOSPC,'run_id':run_id,'location':location,
            'action':'free space (or enlarge the filesystem), then resume: glossapi run --resume '+(run_id or 'RUN_ID')}


def _record_disk_full(writer,corpus,error):
    """Name the exhausted disk, so status, report and the resume can tell it from process death.

    First in the record allocated at run start (overwritten in place: it needs no new block on a full disk), then,
    best effort, as a ledger event.
    """
    run=getattr(corpus,'_active_run_id',None)
    if not run:return
    from .interruption import record
    record(corpus.output_dir,run,error)
    if writer.connection is None:
        return  # the failed run-close released the ledger; the record names the cause (CP-K4a F5)
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
    from ..ledger.encoding import run_hash_scope
    try:
        # Own-file hashes are reused only within this run (K4C-G08); the next run start re-hashes.
        with run_hash_scope():
            try:
                return _locked_run_corpus(corpus,*args,**options)
            finally:
                import sys
                classifier=sys.modules.get('glossapi.gloss_section_classifier')
                if classifier is not None:classifier.release_preloaded_models()
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
    corpus._active_run_id=options.get('resume')
    writer=LedgerWriter(Path(corpus.output_dir)/'corpus.sqlite3')
    if True:
        corpus._scheduler_writer=writer
        committed_generation=None  # set only once this start's validation generation is committed (fix 17, L1)
        try:
            # The disk_full record exists before any write can exhaust the disk (K4BI-G2).
            from .interruption import allocate
            allocate(corpus.output_dir)
            # The first act of every start, before start-up validation or any repair: commit a new validation generation
            # to the ledger, durably (fix 16). From then on no earlier pass is current; only a pass verdict for this
            # generation makes the corpus completed again. If even that commit fails, nothing is validated or repaired:
            # the start stops here, typed (disk_full on a full disk or ledger), saying nothing ran.
            try:
                committed_generation=writer.begin_validation(options.get('resume'))
            except Exception as unrecorded:
                from ..ledger.writer import UnresolvedRecoveryError
                raise UnresolvedRecoveryError([{'path': str(Path(corpus.output_dir)/'corpus.sqlite3'), 'reason': 'validation_start_unrecorded',
                    'error': f'{type(unrecorded).__name__}: {unrecorded}'}], boundary='start') from unrecorded
            # Copies the previous run close left unflushed are verified (and republished if torn) first (K4C-M2).
            writer.verify_excluded_copies()
            # Abandoned temporaries of interrupted publications are removed now, verified, even if that publication is
            # never repeated (K5-D, FIX8-G1); an unverifiable one is retained and never blocks.
            writer.sweep_abandoned_temporaries()
            writer.refuse_ambiguous_copies()
            result=_run_corpus(corpus,*args,**options)
            # A run that would report completion passes the completion gate first (fix 13, decision B): inside the
            # close, after its final flush, under the workspace lock; a refusal raises UnresolvedRecoveryError.
            if result.get('run_id') and result.get('state') in ('completed','completed_with_failures'):
                writer.completion_gate_run=result['run_id']
            # The run-close durability barrier and VACUUM are part of the run (CP-K4a F5): an exhausted disk there is
            # recorded like any other, and the record is cleared only after the close succeeded.
            writer.close()
            gate_record=getattr(writer,'completion_gate_record',None)
            if gate_record:
                result['completion_gate']={key:value for key,value in gate_record.items() if key!='failures'}
            result['total_wall_seconds']=time.monotonic()-began
            if result.get('state') in ('completed','completed_with_failures'):
                from .interruption import clear
                clear(corpus.output_dir)
            return result
        except Exception as error:
            if disk_full(error):
                _record_disk_full(writer,corpus,error)
            from ..ledger.writer import PostCloseEditError, UnresolvedRecoveryError
            if isinstance(error,(PostCloseEditError,UnresolvedRecoveryError)) and getattr(error,'boundary','start')!='completion' \
                    and writer.connection is not None:
                # Fix 14 (R3): a refused start (or repeated publication) is a durable verdict, so no earlier pass of
                # this corpus keeps certifying bytes the refusal just found wrong. The refusal stays the outcome.
                try:
                    writer.record_refusal(getattr(corpus,'_active_run_id',None) or options.get('resume'),error,
                        getattr(error,'boundary','start'))
                    error.details['refusal_recorded'] = True
                except Exception as unrecorded:
                    # Fix 15-17 (H3): explicit, never swallowed. The refusal stays the outcome; the ledger could not take its
                    # verdict (e.g. a full ledger). If this start committed its validation generation, every earlier pass
                    # is stale; if that commit itself failed, nothing was validated or repaired.
                    error.details['refusal_recorded'] = False
                    error.details['refusal_record_error'] = f'{type(unrecorded).__name__}: {unrecorded}'
                    if committed_generation is not None:
                        error.add_note('the refusal could not be recorded in the ledger (' + error.details['refusal_record_error']
                                       + f'); this start already committed validation generation {committed_generation}, so no '
                                       'earlier pass is current: the corpus reads not completed until a pass for this generation')
                    else:
                        # Fix 17 (L1): only claim a generation when its commit happened.
                        error.add_note('the refusal could not be recorded in the ledger (' + error.details['refusal_record_error']
                                       + '); validation never started and no new freshness record was committed by this start '
                                       '(nothing was validated or repaired)')
            raise
        finally:
            try:
                writer.close()  # idempotent; after a failed close it has already released the connection and lock
            except Exception as close_error:
                # Reached only while another error propagates (that error stays the run's outcome). A full disk met by
                # this close is still named in the record allocated at run start (fix 6, F3 audit).
                if disk_full(close_error):
                    _record_disk_full(writer,corpus,close_error)
            corpus._scheduler_writer=None
            corpus._streaming_download_replays=None
            corpus._streaming_ocr_replays=None
            corpus._publication_claims=None
            corpus._streaming_planned_versions=None
            corpus._streaming_completed_versions=None
            corpus._streaming_publication_stems=None
            corpus._streaming_publication_sources=None
