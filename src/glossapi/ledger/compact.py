"""Version-two physical encoding; public rows retain the version-one values.

UUIDs/SHA256s use 16/32-byte SQLite BLOBs. Large canonical JSON uses a
self-identifying zlib frame. Hashes cover the original canonical UTF-8 bytes,
not a compressor-dependent representation. Version one stays read/write plain.

Frames v2 (K4(c), encoding only): the same zlib streams primed with a fixed
preset dictionary (ledger/json_dictionary.py, pinned by sha256) behind a short
marker. A v2 frame is never 16 or 32 bytes long, so it cannot be mistaken for a
UUID or digest BLOB. Readers decode v1 and v2 frames alike.
"""
from __future__ import annotations
import re
import json
import sqlite3
import uuid
import zlib

_JSON_FRAME = b'glossapi-json-zlib-v1\x00'
_TREE_FRAME = b'glossapi-json-tree-v1\x00'
_JSON_FRAME2 = b'gjz2\x00'
_TREE_FRAME2 = b'gjt2\x00'
_PRIMED = None


def _dictionary():
    from .json_dictionary import DICTIONARY
    return DICTIONARY


def _compress2(marker, raw):
    """zlib (level 6) primed with the fixed dictionary; the primed state is copied, never rebuilt."""
    global _PRIMED
    if _PRIMED is None:
        _PRIMED = zlib.compressobj(6, zlib.DEFLATED, 15, 8, zlib.Z_DEFAULT_STRATEGY, _dictionary())
    stream = _PRIMED.copy()
    return marker + stream.compress(raw) + stream.flush()


def _decompress2(payload):
    stream = zlib.decompressobj(zdict=_dictionary())
    data = stream.decompress(payload) + stream.flush()
    if not stream.eof:
        raise ValueError('Truncated compact ledger frame')
    return data
_HASH = re.compile('[0-9a-f]{64}\\Z')
_UUID = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\Z')


def encode_json(encoded):
    raw = encoded.encode('utf-8')
    if len(raw) < 64:
        return encoded
    compressed = _compress2(_JSON_FRAME2, raw)
    if len(compressed) in (16, 32) or len(compressed)+8 > len(raw):
        return encoded
    return compressed


def decode_value(value):
    if isinstance(value, bytes):
        if len(value) not in (16, 32) and value.startswith(_JSON_FRAME2):
            return _decompress2(value[len(_JSON_FRAME2):]).decode('utf-8')
        if value.startswith(_JSON_FRAME):
            return zlib.decompress(value[len(_JSON_FRAME):]).decode('utf-8')
        if len(value) == 32:
            return value.hex()
        if len(value) == 16:
            # str(uuid.UUID(bytes=value)), without constructing a UUID object (hot path).
            digits = value.hex()
            return f'{digits[:8]}-{digits[8:12]}-{digits[12:16]}-{digits[16:20]}-{digits[20:]}'
        raise ValueError('Unknown compact ledger value')
    return value


def _parameter(value):
    if isinstance(value,str):
        size=len(value)
        if size==64:
            if _HASH.fullmatch(value):return bytes.fromhex(value)
        elif size==36 and _UUID.fullmatch(value):
            # uuid.UUID(value).bytes for the canonical lowercase spelling the pattern admits.
            return bytes.fromhex(value.replace('-',''))
    return value


def _rooted(value, root, prefix):
    """The physical spelling of one parameter (path root, then UUID/digest BLOB); no per-call closure (hot path)."""
    if prefix and isinstance(value,str):
        if value.startswith(('@corpus/','@literal/')):value='@literal/'+value
        elif value.startswith(prefix):value='@corpus/'+value[len(prefix):]
    return _parameter(value)


def _parameters(values, connection):
    if not connection.compact:return values
    root=connection.path_root
    prefix=root+'/' if root else None
    if isinstance(values,dict):return {key:_rooted(value,root,prefix) for key,value in values.items()}
    return tuple([_rooted(value,root,prefix) for value in values])


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
        if self.storage_version>=4:
            # The header names the frame dictionary; refuse one this code does not know (K4C-M3).
            from .json_dictionary import check_known
            check_known(*super().execute('SELECT json_dictionary_id,json_dictionary_sha256 FROM ledger_format').fetchone())
    @property
    def event_table(self):
        return 'event_records' if self.compact else 'events'

    def decode(self,value):
        if not isinstance(value,bytes):
            # Fast path (most values): only a rooted path spelling changes.
            if self.path_root and isinstance(value,str) and value[:1]=='@':
                if value.startswith('@literal/'):return value[len('@literal/'):]
                if value.startswith('@corpus/'):return self.path_root+'/'+value[len('@corpus/'):]
            return value
        tree=None
        if isinstance(value,bytes):
            if len(value) not in (16,32) and value.startswith(_TREE_FRAME2):tree=_decompress2(value[len(_TREE_FRAME2):])
            elif value.startswith(_TREE_FRAME):tree=zlib.decompress(value[len(_TREE_FRAME):])
        if tree is not None:
            framed=json.loads(tree)
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
        framed=_compress2(_TREE_FRAME2,raw)
        return framed if len(framed) not in (16,32) else _TREE_FRAME+zlib.compress(raw,1)
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
