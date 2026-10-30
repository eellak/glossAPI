"""Version-two physical encoding; public rows retain the version-one values.

UUIDs/SHA256s use 16/32-byte SQLite BLOBs. Large canonical JSON uses a
self-identifying zlib frame. Hashes cover the original canonical UTF-8 bytes,
not a compressor-dependent representation. Version one stays read/write plain.
"""
from __future__ import annotations
import re
import json
import sqlite3
import uuid
import zlib

_JSON_FRAME = b'glossapi-json-zlib-v1\x00'
_TREE_FRAME = b'glossapi-json-tree-v1\x00'
_HASH = re.compile('[0-9a-f]{64}\\Z')
_UUID = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\Z')


def encode_json(encoded):
    raw = encoded.encode('utf-8')
    if len(raw) < 256:
        return encoded
    compressed = _JSON_FRAME + zlib.compress(raw, level=1)
    return compressed if len(compressed)+32 < len(raw) else encoded


def decode_value(value):
    if isinstance(value, bytes):
        if value.startswith(_JSON_FRAME):
            return zlib.decompress(value[len(_JSON_FRAME):]).decode('utf-8')
        if len(value) == 32:
            return value.hex()
        if len(value) == 16:
            return str(uuid.UUID(bytes=value))
        raise ValueError('Unknown compact ledger value')
    return value


def _parameter(value):
    if isinstance(value,str):
        if len(value)==64 and _HASH.fullmatch(value):return bytes.fromhex(value)
        if len(value)==36 and _UUID.fullmatch(value):return uuid.UUID(value).bytes
    return value


def _parameters(values, connection):
    if not connection.compact:return values
    def parameter(value):
        if connection.path_root and isinstance(value,str):
            if value.startswith(('@corpus/','@literal/')):value='@literal/'+value
            elif value.startswith(connection.path_root+'/'):value='@corpus/'+value[len(connection.path_root)+1:]
        return _parameter(value)
    if isinstance(values,dict):return {key:parameter(value) for key,value in values.items()}
    return tuple(parameter(value) for value in values)


def _row(cursor,row):
    decoded=tuple(cursor.connection.decode(value) for value in row)
    factory=cursor.connection._public_row_factory
    return factory(cursor,decoded) if factory else decoded


class LedgerCursor(sqlite3.Cursor):
    def execute(self,sql,parameters=()):
        return super().execute(sql,_parameters(parameters,self.connection))
    def executemany(self,sql,parameters):
        return super().executemany(sql,(_parameters(row,self.connection) for row in parameters))


class LedgerConnection(sqlite3.Connection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.compact=False
        self.path_root=None
        self._public_row_factory=None
        sqlite3.Connection.row_factory.__set__(self,_row)
        self.storage_version=super().execute('PRAGMA user_version').fetchone()[0]
        self.compact=self.storage_version>=2
        if self.storage_version>=3:
            self.path_root=super().execute('SELECT uri FROM path_root').fetchone()[0]
    @property
    def event_table(self):
        return 'event_records' if self.compact else 'events'

    def decode(self,value):
        if isinstance(value,bytes) and value.startswith(_TREE_FRAME):
            framed=json.loads(zlib.decompress(value[len(_TREE_FRAME):]))
            value=framed['value']
            for path,digest in framed['references']:
                parent=value
                for key in path[:-1]:parent=parent[key]
                shared=json.loads(self.execute('SELECT json FROM snapshots WHERE sha256=?',(digest,)).fetchone()[0])['value']
                parent[path[-1]]={**shared,**parent[path[-1]]}
            return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        value=decode_value(value)
        if self.path_root and isinstance(value,str) and value.startswith('@literal/'):
            return value[len('@literal/'):]
        if self.path_root and isinstance(value,str) and value.startswith('@corpus/'):
            return self.path_root+'/'+value[len('@corpus/'):]
        return value

    def encode_snapshot(self,value,share):
        """Reference shared configuration; preserve canonical logical hashes."""
        references=[]
        variable={'filename','compatibility_filename','file_id','version_id','document_id',
                  'raw_file_id','source_uri','selected_sources','input_versions'}
        candidates={'settings','specification','tools','effective_settings','environment'}
        def prepare(item,path=()):
            if isinstance(item,list):return [prepare(child,(*path,index)) for index,child in enumerate(item)]
            if not isinstance(item,dict):return item
            result={key:prepare(child,(*path,key)) for key,child in item.items()}
            if path and path[-1] in candidates and value.get('kind')!='shared_config':
                common={key:child for key,child in item.items() if key not in variable}
                if len(json.dumps(common))>=160:
                    digest=share(common)
                    references.append((list(path),digest))
                    return {key:result[key] for key in result if key in variable}
            return result
        prepared=prepare(value)
        # Reference paths nested inside a replaced object are redundant.
        references=[(path,digest) for path,digest in references
                    if not any(len(other)<len(path) and path[:len(other)]==other for other,_ in references)]
        raw=json.dumps({'value':prepared,'references':references},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
        return _TREE_FRAME+zlib.compress(raw,1)
    @property
    def row_factory(self):return self._public_row_factory
    @row_factory.setter
    def row_factory(self,value):
        self._public_row_factory=value
        sqlite3.Connection.row_factory.__set__(self,_row)
    def cursor(self,factory=LedgerCursor):return super().cursor(factory)
    def execute(self,sql,parameters=()):
        return self.cursor().execute(sql,parameters)
    def executemany(self,sql,parameters):
        return self.cursor().executemany(sql,parameters)


def connect(*args,**kwargs):
    return sqlite3.connect(*args,factory=LedgerConnection,**kwargs)
