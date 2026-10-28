"""CER scoring + codec decode for the greek-page-ocr training stack.

Contracts (frozen): docs/data_contracts.md §1 (codec v1), docs/architecture/04_recognizer.md §4.3.

- CER is Levenshtein edit distance over **NFD codepoints** (one symbol per codepoint, no
  grapheme clustering). Own two-row O(nm) implementation — no python-Levenshtein (GPL),
  no jiwer. Pure stdlib; numpy not required here.
- Scoring is STRICT per DECISION D-4a: inputs are taken as-is, codepoint-for-codepoint. We
  never re-normalize inside the scorer — NFD's stable sort cannot fix equal-ccc swaps but
  WOULD silently repair cross-ccc ordering errors in a prediction, which strict scoring
  must count.
- Decode: greedy argmax -> collapse repeats -> drop blanks -> ids -> atoms via the codec.
  The joined string IS NFD by construction (labels are stored NFD; atoms are NFD codepoints).
- Ground truth for scoring always comes from `label_ids` decoded through the codec — text is
  never re-encoded (data_contracts §3).

Run `python3 train/cer.py` for the self-check (exit 0 on pass).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence


# ---------------------------------------------------------------------------
# Codec v1 (data_contracts §1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Codec:
    """Loaded codec artifact. ids are contiguous: 0 = CTC blank, atoms 1..len(atoms)."""

    version: str
    blank_id: int
    atoms: tuple  # raw atom dicts, in file order
    atom_by_id: tuple  # index -> single NFD codepoint str; index 0 (blank) is ""
    n_classes: int  # len(atoms) + 1

    def decode(self, ids: Iterable[int]) -> str:
        """Map atom ids (no blanks, already collapsed) to the NFD string of record."""
        table = self.atom_by_id
        out = []
        for i in ids:
            i = int(i)
            if not 1 <= i < self.n_classes:
                raise ValueError(f"id {i} outside codec range 1..{self.n_classes - 1}")
            out.append(table[i])
        return "".join(out)


def load_codec(path: str) -> Codec:
    """Load codec JSON per data_contracts §1 and validate its invariants.

    n_classes = len(atoms) + 1 (blank id 0); asserts max atom id == len(atoms)
    and that ids are contiguous 1..len(atoms).
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    version = obj.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"codec {path}: missing/invalid 'version'")
    blank_id = obj.get("blank_id")
    if blank_id != 0:
        raise ValueError(f"codec {path}: blank_id must be 0, got {blank_id!r}")
    atoms = obj.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError(f"codec {path}: missing/empty 'atoms'")

    n_atoms = len(atoms)
    n_classes = n_atoms + 1
    ids = [a["id"] for a in atoms]
    if len(set(ids)) != n_atoms or min(ids) != 1 or max(ids) != n_atoms:
        raise ValueError(
            f"codec {path}: atom ids must be contiguous 1..{n_atoms} "
            f"(got min={min(ids)}, max={max(ids)}, unique={len(set(ids))})"
        )
    assert max(ids) == n_atoms  # spec-mandated assert: max id == len(atoms)

    table: List[str] = [""] * n_classes  # index 0 = blank, never emitted
    for a in atoms:
        ch = a.get("ch")
        cp = a.get("cp")
        if ch is None and isinstance(cp, str) and cp.startswith("U+"):
            ch = chr(int(cp[2:], 16))
        if not isinstance(ch, str) or len(ch) != 1:
            raise ValueError(f"codec {path}: atom id {a['id']} has no single-codepoint 'ch'/'cp'")
        if isinstance(cp, str) and cp.startswith("U+") and ord(ch) != int(cp[2:], 16):
            raise ValueError(f"codec {path}: atom id {a['id']}: 'ch' {ch!r} disagrees with 'cp' {cp}")
        table[a["id"]] = ch

    return Codec(
        version=version,
        blank_id=0,
        atoms=tuple(atoms),
        atom_by_id=tuple(table),
        n_classes=n_classes,
    )


# ---------------------------------------------------------------------------
# Greedy CTC collapse
# ---------------------------------------------------------------------------


def collapse_ctc(ids: Sequence[int], blank: int = 0) -> List[int]:
    """Collapse repeats then drop blanks (standard CTC greedy path -> label sequence)."""
    out: List[int] = []
    prev = None
    for v in ids:
        v = int(v)
        if v != prev:
            if v != blank:
                out.append(v)
            prev = v
    return out


# ---------------------------------------------------------------------------
# Levenshtein / CER (strict, over NFD codepoints)
# ---------------------------------------------------------------------------


def levenshtein(ref: str, hyp: str) -> int:
    """Edit distance (ins/del/sub, all cost 1) between two codepoint sequences.

    Two-row O(len(ref)*len(hyp)) dynamic program with common prefix/suffix stripping.
    No transpositions: swapping two adjacent codepoints costs 2 edits (strict D-4a).
    """
    if ref == hyp:
        return 0
    # strip common prefix
    i = 0
    nr, nh = len(ref), len(hyp)
    lim = min(nr, nh)
    while i < lim and ref[i] == hyp[i]:
        i += 1
    # strip common suffix (of the remaining parts)
    j = 0
    lim = min(nr, nh) - i
    while j < lim and ref[nr - 1 - j] == hyp[nh - 1 - j]:
        j += 1
    a = ref[i : nr - j]
    b = hyp[i : nh - j]
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(b) < len(a):  # keep the inner loop over the shorter string
        a, b = b, a
    prev = list(range(len(b) + 1))
    cur = [0] * (len(b) + 1)
    for r in range(1, len(a) + 1):
        ac = a[r - 1]
        cur[0] = r
        for c in range(1, len(b) + 1):
            sub = prev[c - 1] + (ac != b[c - 1])
            dele = prev[c] + 1
            ins = cur[c - 1] + 1
            m = sub if sub <= dele else dele
            cur[c] = m if m <= ins else ins
        prev, cur = cur, prev
    return prev[len(b)]


def cer_line(ref: str, hyp: str) -> float:
    """Per-line CER = edits / len(ref). Empty ref: 0.0 if hyp empty, else edits/1 (guarded)."""
    e = levenshtein(ref, hyp)
    return e / max(1, len(ref))


class CorpusCER:
    """Corpus CER accumulator: total edits / total ref length (data_contracts scoring)."""

    def __init__(self) -> None:
        self.edits = 0
        self.ref_len = 0
        self.lines = 0

    def add(self, ref: str, hyp: str) -> int:
        e = levenshtein(ref, hyp)
        self.edits += e
        self.ref_len += len(ref)
        self.lines += 1
        return e

    @property
    def cer(self) -> float:
        return self.edits / max(1, self.ref_len)


def greedy_decode_line(timestep_ids: Sequence[int], codec: Codec) -> str:
    """argmax ids for ONE line (already truncated to its input_length) -> NFD text."""
    return codec.decode(collapse_ctc(timestep_ids, blank=codec.blank_id))
