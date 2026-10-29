"""Version-two physical encoding; public rows retain the version-one values.

UUIDs/SHA256s use 16/32-byte SQLite BLOBs. Large canonical JSON uses a
self-identifying zlib frame. Hashes cover the original canonical UTF-8 bytes,
not a compressor-dependent representation. Version one stays read/write plain.
"""
from __future__ import annotations
import re
import sqlite3
import uuid
import zlib

_JSON_FRAME = b'glossapi-json-zlib-v1\x00'
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


def _parameters(values, compact):
    if not compact:return values
    if isinstance(values,dict):return {key:_parameter(value) for key,value in values.items()}
    return tuple(_parameter(value) for value in values)


def _row(cursor,row):
    decoded=tuple(decode_value(value) for value in row)
    factory=cursor.connection._public_row_factory
    return factory(cursor,decoded) if factory else decoded


class LedgerCursor(sqlite3.Cursor):
    def execute(self,sql,parameters=()):
        return super().execute(sql,_parameters(parameters,self.connection.compact))
    def executemany(self,sql,parameters):
        return super().executemany(sql,(_parameters(row,self.connection.compact) for row in parameters))


class LedgerConnection(sqlite3.Connection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.compact=False
        self._public_row_factory=None
        sqlite3.Connection.row_factory.__set__(self,_row)
        self.compact=super().execute('PRAGMA user_version').fetchone()[0]==2
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
