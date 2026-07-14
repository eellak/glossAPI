from __future__ import annotations

"""Conservative diagnostics for complex VLM-output repetition.

The detectors in this module operate on finalized text.  They complement
generation-time token guards such as ``no_repeat_ngram_size``:

* repeated multi-token blocks catch phrases, HTML rows, and larger structures;
* regular numeric sequences catch model-generated arithmetic progressions.

Single-character and single-token runs are deliberately out of scope.  Those
are handled by cheaper guards in :mod:`glossapi.ocr.utils.cleaning`.
"""

import re
from array import array
from collections import defaultdict, deque
from decimal import Decimal, InvalidOperation
from typing import Deque, Dict, List, Optional, Sequence, Tuple


RULE_REPEATED_TOKEN_BLOCK = "repeated_token_block"
RULE_REGULAR_NUMERIC_SEQUENCE = "regular_numeric_sequence"

TOKEN_PATTERN = re.compile(
    r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*?)?/?>"
    r"|[^\W\d_]+(?:['’][^\W\d_]+)*"
    r"|[-+]?\d+(?:[.,]\d+)*"
    r"|[^\s]",
    re.UNICODE,
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:[.,]\d+)?)(?![\w.])", re.UNICODE)
SEQUENCE_GAP_PATTERN = re.compile(r"[\s,;|/·:–—-]*", re.UNICODE)
HASH_BASE = 1_000_003
HASH_MASK = (1 << 64) - 1


def _normalize_token(value: str) -> str:
    if value.startswith("<") and value.endswith(">"):
        return re.sub(r"\s+", " ", value.strip()).casefold()
    return value.casefold()


def _tokenize(text: str) -> Tuple[List[int], List[int], List[int], List[bool]]:
    token_ids: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    is_content: List[bool] = []
    vocabulary: Dict[str, int] = {}
    for match in TOKEN_PATTERN.finditer(text):
        raw = match.group(0)
        normalized = _normalize_token(raw)
        token_id = vocabulary.setdefault(normalized, len(vocabulary) + 1)
        token_ids.append(token_id)
        starts.append(match.start())
        ends.append(match.end())
        is_content.append(not raw.startswith("<") and any(ch.isalnum() for ch in raw))
    return token_ids, starts, ends, is_content


def _blocks_match(token_ids: List[int], start: int, period: int, repeat_count: int) -> bool:
    pattern = token_ids[start : start + period]
    return all(
        token_ids[start + offset * period : start + (offset + 1) * period] == pattern
        for offset in range(1, repeat_count)
    )


def _rolling_hash_tables(token_ids: List[int]) -> Tuple[array, array]:
    prefixes = array("Q", [0])
    powers = array("Q", [1])
    for token_id in token_ids:
        prefixes.append((prefixes[-1] * HASH_BASE + token_id + 1) & HASH_MASK)
        powers.append((powers[-1] * HASH_BASE) & HASH_MASK)
    return prefixes, powers


def _block_hash(prefixes: Sequence[int], powers: Sequence[int], start: int, end: int) -> int:
    return (prefixes[end] - prefixes[start] * powers[end - start]) & HASH_MASK


def _block_hashes_match(
    prefixes: Sequence[int],
    powers: Sequence[int],
    start: int,
    period: int,
    repeat_count: int,
) -> bool:
    pattern_hash = _block_hash(prefixes, powers, start, start + period)
    return all(
        _block_hash(
            prefixes,
            powers,
            start + offset * period,
            start + (offset + 1) * period,
        )
        == pattern_hash
        for offset in range(1, repeat_count)
    )


def _extend_block_run(token_ids: List[int], start: int, period: int, minimum_repeats: int) -> Tuple[int, int]:
    while start > 0 and all(
        token_ids[start - 1 + offset * period] == token_ids[start - 1]
        for offset in range(1, minimum_repeats)
    ):
        start -= 1
    pattern = token_ids[start : start + period]
    repeat_count = 0
    while start + (repeat_count + 1) * period <= len(token_ids):
        block_start = start + repeat_count * period
        if token_ids[block_start : block_start + period] != pattern:
            break
        repeat_count += 1
    return start, repeat_count


def detect_repeated_token_blocks(
    text: str,
    *,
    min_pattern_tokens: int = 6,
    max_pattern_tokens: int = 256,
    min_repeats: int = 4,
    min_pattern_chars: int = 24,
    min_distinct_tokens: int = 3,
    min_distinct_content_tokens: int = 2,
    anchor_tokens: int = 3,
    max_findings: int = 8,
) -> List[dict]:
    """Find adjacent repetitions of a non-trivial token block.

    Candidate periods are proposed by repeated short anchors, then every block
    is compared exactly.  This follows the same bounded N-gram-loop idea used
    by vLLM while making it applicable to archived text without token IDs.
    """
    if min_pattern_tokens < 2 or min_repeats < 2 or max_pattern_tokens < min_pattern_tokens:
        raise ValueError("invalid repeated-token-block thresholds")
    if anchor_tokens < 1 or anchor_tokens > min_pattern_tokens:
        raise ValueError("anchor_tokens must be between 1 and min_pattern_tokens")
    if max_findings < 1:
        return []

    token_ids, starts, ends, is_content = _tokenize(str(text or ""))
    token_count = len(token_ids)
    if token_count < min_pattern_tokens * min_repeats:
        return []
    prefixes, powers = _rolling_hash_tables(token_ids)

    recent: Dict[Tuple[int, ...], Deque[int]] = defaultdict(deque)
    findings: List[dict] = []
    skip_checks_until = -1
    last_anchor_start = token_count - anchor_tokens

    for current in range(last_anchor_start + 1):
        anchor = tuple(token_ids[current : current + anchor_tokens])
        positions = recent[anchor]
        oldest_allowed = current - max_pattern_tokens
        while positions and positions[0] < oldest_allowed:
            positions.popleft()

        if current >= skip_checks_until:
            for previous in reversed(positions):
                period = current - previous
                if period < min_pattern_tokens or period > max_pattern_tokens:
                    continue
                required_end = previous + period * min_repeats
                if required_end > token_count:
                    continue
                if not _block_hashes_match(prefixes, powers, previous, period, min_repeats):
                    continue
                candidate_pattern_end = previous + period
                distinct_tokens = len(set(token_ids[previous:candidate_pattern_end]))
                distinct_content_tokens = len(
                    {
                        token_ids[index]
                        for index in range(previous, candidate_pattern_end)
                        if is_content[index]
                    }
                )
                pattern_chars = starts[candidate_pattern_end] - starts[previous]
                if (
                    pattern_chars < min_pattern_chars
                    or distinct_tokens < min_distinct_tokens
                    or distinct_content_tokens < min_distinct_content_tokens
                ):
                    continue
                if not _blocks_match(token_ids, previous, period, min_repeats):
                    continue

                run_start, repeat_count = _extend_block_run(token_ids, previous, period, min_repeats)
                pattern_end_token = run_start + period
                evidence_end_token = run_start + repeat_count * period
                pattern_chars = starts[pattern_end_token] - starts[run_start]

                findings.append(
                    {
                        "rule": RULE_REPEATED_TOKEN_BLOCK,
                        "cut_index": starts[run_start],
                        "evidence_end_index": ends[evidence_end_token - 1],
                        "pattern_token_count": period,
                        "pattern_char_count": pattern_chars,
                        "repeat_count": repeat_count,
                        "distinct_token_count": distinct_tokens,
                        "distinct_content_token_count": distinct_content_tokens,
                    }
                )
                skip_checks_until = evidence_end_token
                break

        positions.append(current)
        if len(findings) >= max_findings:
            break

    return findings


def _decimal_value(raw: str) -> Optional[Decimal]:
    value = raw
    if "," in value and "." not in value:
        value = value.replace(",", ".")
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _numeric_detail(
    matches: List[re.Match[str]],
    start: int,
    end: int,
    step: Decimal,
    tolerance: Decimal,
) -> dict:
    return {
        "rule": RULE_REGULAR_NUMERIC_SEQUENCE,
        "cut_index": matches[start].start(),
        "evidence_end_index": matches[end - 1].end(),
        "sequence_term_count": end - start,
        "sequence_step": str(step),
        "sequence_step_tolerance": str(tolerance),
    }


def _display_quantum(value: Decimal) -> Decimal:
    exponent = int(value.as_tuple().exponent)
    return Decimal(1).scaleb(exponent) if exponent < 0 else Decimal(0)


def detect_regular_numeric_sequences(
    text: str,
    *,
    min_terms: int = 32,
    max_gap_chars: int = 12,
    max_findings: int = 8,
) -> List[dict]:
    """Find long adjacent arithmetic progressions in textual number lists."""
    if min_terms < 3:
        raise ValueError("min_terms must be at least 3")
    if max_findings < 1:
        return []

    matches = list(NUMBER_PATTERN.finditer(str(text or "")))
    if len(matches) < min_terms:
        return []
    values = [_decimal_value(match.group(0)) for match in matches]
    findings: List[dict] = []
    run_start = 0
    run_step: Optional[Decimal] = None
    run_step_low: Optional[Decimal] = None
    run_step_high: Optional[Decimal] = None
    run_tolerance = Decimal(0)

    def finish(run_end: int) -> None:
        nonlocal run_start, run_step, run_step_low, run_step_high, run_tolerance
        if run_step is not None and run_step != 0 and run_end - run_start >= min_terms:
            representative = run_step
            if run_step_low is not None and run_step_high is not None:
                representative = min(max(representative, run_step_low), run_step_high)
            findings.append(_numeric_detail(matches, run_start, run_end, representative, run_tolerance))

    for index in range(1, len(matches)):
        previous = values[index - 1]
        current = values[index]
        gap = text[matches[index - 1].end() : matches[index].start()]
        adjacent = len(gap) <= max_gap_chars and SEQUENCE_GAP_PATTERN.fullmatch(gap) is not None
        step = current - previous if previous is not None and current is not None else None
        tolerance = (
            max(_display_quantum(previous), _display_quantum(current))
            if previous is not None and current is not None
            else Decimal(0)
        )
        candidate_low = step - tolerance if step is not None else None
        candidate_high = step + tolerance if step is not None else None
        step_matches = run_step is None or (
            candidate_low is not None
            and candidate_high is not None
            and run_step_low is not None
            and run_step_high is not None
            and max(run_step_low, candidate_low) <= min(run_step_high, candidate_high)
        )
        if adjacent and step is not None and step != 0 and step_matches:
            if run_step is None:
                run_start = index - 1
                run_step = step
                run_step_low = candidate_low
                run_step_high = candidate_high
                run_tolerance = tolerance
            else:
                assert run_step_low is not None and run_step_high is not None
                assert candidate_low is not None and candidate_high is not None
                run_step_low = max(run_step_low, candidate_low)
                run_step_high = min(run_step_high, candidate_high)
                run_tolerance = max(run_tolerance, tolerance)
            continue

        finish(index)
        if len(findings) >= max_findings:
            return findings
        if adjacent and step is not None and step != 0:
            run_start = index - 1
            run_step = step
            run_step_low = candidate_low
            run_step_high = candidate_high
            run_tolerance = tolerance
        else:
            run_start = index
            run_step = None
            run_step_low = None
            run_step_high = None
            run_tolerance = Decimal(0)

    finish(len(matches))
    return findings[:max_findings]


def detect_complex_repetition_details(
    text: str,
    *,
    max_findings: int = 8,
    min_pattern_tokens: int = 6,
    max_pattern_tokens: int = 256,
    min_pattern_repeats: int = 4,
    numeric_sequence_min_terms: int = 32,
) -> List[dict]:
    """Return complex repeated-block and regular-sequence diagnostics.

    The result uses the same ``rule``/``cut_index`` core fields as GlossAPI's
    early-stop diagnostics, with extra evidence metadata for review tooling.
    No text is modified by this function.
    """
    if max_findings < 1:
        return []
    block_findings = detect_repeated_token_blocks(
        text,
        min_pattern_tokens=min_pattern_tokens,
        max_pattern_tokens=max_pattern_tokens,
        min_repeats=min_pattern_repeats,
        max_findings=max_findings,
    )
    sequence_findings = detect_regular_numeric_sequences(
        text,
        min_terms=numeric_sequence_min_terms,
        max_findings=max_findings,
    )
    findings = block_findings + sequence_findings
    findings.sort(key=lambda detail: (int(detail["cut_index"]), str(detail["rule"])))
    return findings[:max_findings]
