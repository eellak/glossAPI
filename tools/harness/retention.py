"""Physical snapshots before allowlisted retention, with causal ledger proof."""

from glossapi.ledger.compact import connect as ledger_connect
from pathlib import Path
import json
import os
import shutil
import sqlite3
from urllib.parse import unquote,urlsplit

from .common import sha256,write_json

ALLOWLIST={'raw_markdown','annotation_partition','repaired_pdf'}


class RetentionSnapshots:
    def __init__(self,corpus,snapshots):
        self.corpus=Path(corpus).resolve()
        self.snapshots=Path(snapshots).resolve()
        self.manifest=self.snapshots/'manifest.json'

    def __call__(self,controller,rows):
        recorded=json.loads(self.manifest.read_text()) if self.manifest.exists() else []
        for row in rows:
            path=Path(unquote(urlsplit(row['uri']).path))
            if not path.is_relative_to(self.corpus) or path.is_relative_to(self.corpus/'.glossapi') or row['kind'] not in ALLOWLIST:
                continue
            if controller.eligibility(row['file_id'])[2]:continue
            relative=path.relative_to(self.corpus)
            saved=self.snapshots/row['file_id']/relative
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,saved)
            with saved.open('rb') as stream:os.fsync(stream.fileno())
            assert saved.stat().st_size==row['bytes'] and sha256(saved)==row['sha256']
            recorded.append(dict(file_id=row['file_id'],run_id=controller.run_id,stage=row['stage'],kind=row['kind'],
                path=str(relative),snapshot=str(saved),sha256=row['sha256'],bytes=row['bytes']))
        if recorded:write_json(self.manifest,recorded)


def verified_snapshots(corpus,snapshots):
    """Never waive a missing product without its physical pre-delete bytes."""
    corpus=Path(corpus).resolve();snapshots=Path(snapshots).resolve()
    manifest=snapshots/'manifest.json'
    if not manifest.exists():return []
    from glossapi.stages.retention import retention_facts
    rows=json.loads(manifest.read_text())
    with ledger_connect('file:'+str(corpus/'corpus.sqlite3')+'?mode=ro&immutable=1',uri=True) as database:
        database.row_factory=sqlite3.Row
        for saved in rows:
            path=Path(saved['snapshot']);assert path.is_relative_to(snapshots)
            assert saved['kind'] in ALLOWLIST and path.stat().st_size==saved['bytes'] and sha256(path)==saved['sha256']
            fact=database.execute('SELECT * FROM current_file_state WHERE file_id=?',(saved['file_id'],)).fetchone()
            assert fact and fact['ownership']=='managed' and fact['kind']==saved['kind'] and fact['sha256']==saved['sha256'] and fact['bytes']==saved['bytes']
            assert fact['uri']==(corpus/saved['path']).as_uri() and fact['presence']=='deleted'
            assert not (corpus/saved['path']).exists()
            proof=retention_facts(database,saved['file_id'])
            assert proof['consumer_ok_event_ids'] and not proof['outstanding_consumers'] and not proof['pins']
            assert any(event['reason_code'] in ('delete_completed','delete_recovered') for event in proof['deletions'])
            saved['retention']=proof
    return rows
