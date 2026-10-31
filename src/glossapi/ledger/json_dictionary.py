"""Fixed zlib preset dictionary of compact ledger JSON frames v2 (K4(c)); never edit: frames depend on it.

Built by experiments/deps_20261028/k4c/build_json_dictionary.py from machine-independent skeletons of
probe-run ledger JSON (keys and vocabulary only; no paths, identifiers, digests or times).
"""
import base64, hashlib, zlib

DICTIONARY_SHA256 = 'ee449a8f1aaa4f8f3a5c35fb118a3ab70793f7ac518cc9b850b3c5bf601190ca'
_ENCODED = (
    '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'
)
DICTIONARY = zlib.decompress(base64.b64decode(_ENCODED))
if hashlib.sha256(DICTIONARY).hexdigest() != DICTIONARY_SHA256:
    raise ImportError("ledger JSON dictionary is corrupt")
