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
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple


RULE_REPEATED_TOKEN_BLOCK = "repeated_token_block"
RULE_REGULAR_NUMERIC_SEQUENCE = "regular_numeric_sequence"
TEXT_REMOVED_COMMENT = "<!-- text-removed -->"

TOKEN_PATTERN = re.compile(
    r"</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*?)?/?>"
    r"|[^\W\d_]+(?:['’][^\W\d_]+)*"
    r"|[-+]?\d+(?:[.,]\d+)*"
    r"|[^\s]",
    re.UNICODE,
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d+(?:[.,]\d+)?)(?!\w|[.,]\d)", re.UNICODE)
SEQUENCE_GAP_PATTERN = re.compile(r"[\s,;|/·:–—-]*", re.UNICODE)
HASH_BASE = 1_000_003
HASH_MASK = (1 << 64) - 1
REPEAT_STRUCTURAL_TAGS = {
    "article",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "section",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "ul",
}
OPENING_TAG_NAME_PATTERN = re.compile(r"<([A-Za-z][A-Za-z0-9:_-]*)(?:\s[^<>]*?)?/?>")
CLOSING_TAG_NAME_PATTERN = re.compile(r"</([A-Za-z][A-Za-z0-9:_-]*)\s*>")


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


def _extend_block_run(token_ids: List[int], start: int, period: int) -> Tuple[int, int]:
    """Count complete copies without rotating into preceding context."""
    pattern = token_ids[start : start + period]
    repeat_count = 0
    while start + (repeat_count + 1) * period <= len(token_ids):
        block_start = start + repeat_count * period
        if token_ids[block_start : block_start + period] != pattern:
            break
        repeat_count += 1
    return start, repeat_count


def _tag_name(pattern: re.Pattern[str], raw: str) -> Optional[str]:
    match = pattern.fullmatch(raw)
    return match.group(1).casefold() if match is not None else None


def _boundary_score(
    text: str,
    starts: List[int],
    ends: List[int],
    start: int,
    end: int,
) -> Tuple[int, int, int, int]:
    raw_start = text[starts[start] : ends[start]]
    raw_end = text[starts[end - 1] : ends[end - 1]]
    opening_tag = _tag_name(OPENING_TAG_NAME_PATTERN, raw_start)
    closing_tag = _tag_name(CLOSING_TAG_NAME_PATTERN, raw_end)
    structural_pair = int(opening_tag in REPEAT_STRUCTURAL_TAGS and closing_tag == opening_tag)
    structural_start = int(opening_tag in REPEAT_STRUCTURAL_TAGS)

    start_index = starts[start]
    end_index = ends[end - 1]
    start_boundary = 3 if start_index == 0 else 0
    end_boundary = 3 if end_index == len(text) else 0
    if start_index > 0:
        previous = text[start_index - 1]
        start_boundary = 3 if previous in "\r\n" else 2 if previous.isspace() else 0
    if end_index < len(text):
        following = text[end_index]
        end_boundary = 3 if following in "\r\n" else 2 if following.isspace() else 0
    previous_nonspace = text[:start_index].rstrip()[-1:] if start_index else ""
    textual_pair = int(
        opening_tag is None
        and any(ch.isalnum() for ch in raw_start)
        and (start_index == 0 or previous_nonspace in ".!?…»”\"'")
        and raw_end in {".", "!", "?", "…", "»", "”", "\"", "'"}
    )
    return structural_pair, structural_start, textual_pair, start_boundary + end_boundary


def _canonical_block_run(
    text: str,
    token_ids: List[int],
    starts: List[int],
    ends: List[int],
    is_content: List[bool],
    *,
    initial_start: int,
    period: int,
    min_repeats: int,
) -> Tuple[int, int]:
    """Choose a semantic boundary among equivalent rotations of a token loop."""
    _, initial_repeat_count = _extend_block_run(token_ids, initial_start, period)
    initial_raw = text[starts[initial_start] : ends[initial_start]]
    initial_is_tag = initial_raw.startswith("<") and initial_raw.endswith(">")
    initial_boundary = _boundary_score(
        text,
        starts,
        ends,
        initial_start,
        initial_start + initial_repeat_count * period,
    )
    best_structural: Optional[Tuple[Tuple[int, ...], int, int]] = None
    best_textual: Optional[Tuple[Tuple[int, ...], int, int]] = None
    best_content: Optional[Tuple[Tuple[int, ...], int, int]] = None

    for rotation in range(1, period):
        candidate_start = initial_start + rotation
        if candidate_start + period * min_repeats > len(token_ids):
            break
        if not _blocks_match(token_ids, candidate_start, period, min_repeats):
            continue
        _, repeat_count = _extend_block_run(token_ids, candidate_start, period)
        candidate_end = candidate_start + repeat_count * period
        candidate_boundary = _boundary_score(text, starts, ends, candidate_start, candidate_end)
        if candidate_boundary[0]:
            key = (*candidate_boundary, repeat_count, -rotation)
            if best_structural is None or key > best_structural[0]:
                best_structural = (key, candidate_start, repeat_count)
        if candidate_boundary[2] and is_content[candidate_start]:
            textual_start = candidate_start
            textual_repeat_count = repeat_count
            previous_rotation = initial_start - (period - rotation)
            if previous_rotation >= 0 and _blocks_match(token_ids, previous_rotation, period, min_repeats):
                _, previous_repeat_count = _extend_block_run(token_ids, previous_rotation, period)
                previous_end = previous_rotation + previous_repeat_count * period
                previous_boundary = _boundary_score(
                    text,
                    starts,
                    ends,
                    previous_rotation,
                    previous_end,
                )
                if previous_boundary[2]:
                    textual_start = previous_rotation
                    textual_repeat_count = previous_repeat_count
            key = (textual_repeat_count, -rotation)
            if best_textual is None or key > best_textual[0]:
                best_textual = (key, textual_start, textual_repeat_count)
        if not initial_is_tag and not is_content[initial_start] and is_content[candidate_start]:
            key = (candidate_boundary[3], repeat_count, -rotation)
            if best_content is None or key > best_content[0]:
                best_content = (key, candidate_start, repeat_count)

    if best_structural is not None:
        return best_structural[1], best_structural[2]
    if not initial_boundary[2] and best_textual is not None:
        return best_textual[1], best_textual[2]
    if best_content is not None:
        return best_content[1], best_content[2]
    return initial_start, initial_repeat_count


def _trailing_pattern_prefix(
    text: str,
    token_ids: List[int],
    starts: List[int],
    ends: List[int],
    *,
    run_start: int,
    period: int,
    repeat_count: int,
    min_partial_tokens: int,
) -> Tuple[int, int, bool]:
    """Extend a repeated run over a sufficiently long final partial copy."""
    pattern = token_ids[run_start : run_start + period]
    cursor = run_start + repeat_count * period
    partial_tokens = 0
    while (
        cursor + partial_tokens < len(token_ids)
        and partial_tokens < period
        and token_ids[cursor + partial_tokens] == pattern[partial_tokens]
    ):
        partial_tokens += 1

    trailing_fragment = False
    fragment_is_substantial = False
    next_token = cursor + partial_tokens
    if partial_tokens < period and next_token < len(token_ids):
        expected_token = run_start + partial_tokens
        expected = _normalize_token(text[starts[expected_token] : ends[expected_token]])
        observed = _normalize_token(text[starts[next_token] : ends[next_token]])
        following_gap = (
            text[ends[next_token] : starts[next_token + 1]]
            if next_token + 1 < len(token_ids)
            else text[ends[next_token] :]
        )
        ends_at_strong_boundary = next_token == len(token_ids) - 1 or "\n" in following_gap
        trailing_fragment = (
            (len(observed) >= 3 or (partial_tokens >= min_partial_tokens and bool(observed)))
            and expected.startswith(observed)
            and observed != expected
            and ends_at_strong_boundary
        )
        fragment_is_substantial = trailing_fragment and len(observed) >= 6

    matched_tokens = partial_tokens + int(trailing_fragment)
    full_end_token = cursor
    full_end_index = ends[full_end_token - 1]
    if matched_tokens < min_partial_tokens and not fragment_is_substantial:
        return full_end_index, partial_tokens, trailing_fragment

    removal_end_token = cursor + matched_tokens
    removal_end_index = ends[removal_end_token - 1]
    if not text[removal_end_index:].strip():
        removal_end_index = len(text)
    return removal_end_index, partial_tokens, trailing_fragment


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
    min_trailing_partial_tokens: int = 3,
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
    if min_trailing_partial_tokens < 1:
        raise ValueError("min_trailing_partial_tokens must be positive")
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

                run_start, repeat_count = _canonical_block_run(
                    text,
                    token_ids,
                    starts,
                    ends,
                    is_content,
                    initial_start=previous,
                    period=period,
                    min_repeats=min_repeats,
                )
                pattern_end_token = run_start + period
                evidence_end_token = run_start + repeat_count * period
                pattern_chars = starts[pattern_end_token] - starts[run_start]
                removal_end_index, trailing_partial_tokens, trailing_fragment = _trailing_pattern_prefix(
                    text,
                    token_ids,
                    starts,
                    ends,
                    run_start=run_start,
                    period=period,
                    repeat_count=repeat_count,
                    min_partial_tokens=min_trailing_partial_tokens,
                )

                findings.append(
                    {
                        "rule": RULE_REPEATED_TOKEN_BLOCK,
                        "cut_index": starts[run_start],
                        "evidence_end_index": ends[evidence_end_token - 1],
                        "removal_end_index": removal_end_index,
                        "pattern_token_count": period,
                        "pattern_char_count": pattern_chars,
                        "repeat_count": repeat_count,
                        "distinct_token_count": distinct_tokens,
                        "distinct_content_token_count": distinct_content_tokens,
                        "trailing_partial_token_count": trailing_partial_tokens,
                        "trailing_partial_fragment": trailing_fragment,
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
    text: str,
    matches: List[re.Match[str]],
    values: List[Optional[Decimal]],
    start: int,
    end: int,
    step: Decimal,
    tolerance: Decimal,
    max_gap_chars: int,
) -> dict:
    removal_end_index = matches[end - 1].end()
    trailing_partial = False
    if end < len(matches) and values[end - 1] is not None:
        candidate = matches[end]
        gap = text[matches[end - 1].end() : candidate.start()]
        fragment_end = candidate.end()
        if fragment_end < len(text) and text[fragment_end] in ".," and (
            fragment_end + 1 == len(text) or not text[fragment_end + 1].isdigit()
        ):
            fragment_end += 1
        observed = text[candidate.start() : fragment_end]
        previous_raw = matches[end - 1].group(0)
        expected = values[end - 1] + step
        decimal_separator = "," if "," in previous_raw and "." not in previous_raw else "."
        decimal_places = 0
        if decimal_separator in previous_raw:
            decimal_places = len(previous_raw.rsplit(decimal_separator, 1)[1])
        expected_text = format(expected, f".{decimal_places}f") if decimal_places else format(expected, "f")
        if decimal_separator == ",":
            expected_text = expected_text.replace(".", ",")
        following = text[fragment_end:]
        following_without_horizontal_space = following.lstrip(" \t")
        ends_at_boundary = not following or following_without_horizontal_space.startswith(("\n", "\r"))
        trailing_partial = (
            len(gap) <= max_gap_chars
            and SEQUENCE_GAP_PATTERN.fullmatch(gap) is not None
            and len(observed) >= 2
            and expected_text.startswith(observed)
            and observed != expected_text
            and ends_at_boundary
        )
        if trailing_partial:
            removal_end_index = fragment_end
    return {
        "rule": RULE_REGULAR_NUMERIC_SEQUENCE,
        "cut_index": matches[start].start(),
        "evidence_end_index": matches[end - 1].end(),
        "removal_end_index": removal_end_index,
        "sequence_term_count": end - start,
        "sequence_step": str(step),
        "sequence_step_tolerance": str(tolerance),
        "trailing_partial_number": trailing_partial,
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
            findings.append(
                _numeric_detail(
                    text,
                    matches,
                    values,
                    run_start,
                    run_end,
                    representative,
                    run_tolerance,
                    max_gap_chars,
                )
            )

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
    min_trailing_partial_tokens: int = 3,
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
        min_trailing_partial_tokens=min_trailing_partial_tokens,
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


def _merged_replacement_spans(text: str, details: Sequence[Mapping[str, object]]) -> List[dict]:
    spans: List[dict] = []
    for detail in sorted(details, key=lambda item: (int(item["cut_index"]), str(item["rule"]))):
        start = int(detail["cut_index"])
        end = int(detail.get("removal_end_index", detail["evidence_end_index"]))
        if start < 0 or end <= start or end > len(text):
            raise RuntimeError(f"invalid complex-repetition span: {start}:{end}")
        if spans and (start <= int(spans[-1]["end_index"]) or not text[int(spans[-1]["end_index"]) : start].strip()):
            spans[-1]["end_index"] = max(int(spans[-1]["end_index"]), end)
            spans[-1]["findings"].append(dict(detail))
            continue
        spans.append(
            {
                "start_index": start,
                "end_index": end,
                "findings": [dict(detail)],
            }
        )
    return spans


def _structural_preservation_reason(text: str, detail: Mapping[str, object]) -> Optional[str]:
    if str(detail.get("rule")) != RULE_REPEATED_TOKEN_BLOCK:
        return None
    start = int(detail["cut_index"])
    end = int(detail.get("removal_end_index", detail["evidence_end_index"]))
    raw = text[start:end]
    has_header_cells = re.search(r"</?th(?:\s[^<>]*?)?>", raw, re.IGNORECASE) is not None
    has_data_cells = re.search(r"</?td(?:\s[^<>]*?)?>", raw, re.IGNORECASE) is not None
    if has_header_cells and not has_data_cells:
        return "repeated_html_table_header_cells"
    repeat_count = int(detail.get("repeat_count", 0))
    if "\n" not in raw and repeat_count <= 4 and raw.count("|") >= max(1, repeat_count - 1):
        return "repeated_markdown_table_header_cells"
    return None


def replace_complex_repetitions(
    text: str,
    *,
    replacement: str = TEXT_REMOVED_COMMENT,
    max_passes: int = 16,
    max_findings_per_pass: int = 256,
    min_pattern_tokens: int = 6,
    max_pattern_tokens: int = 256,
    min_pattern_repeats: int = 4,
    numeric_sequence_min_terms: int = 32,
    min_trailing_partial_tokens: int = 3,
    preserve_structural_headers: bool = True,
    metrics: Optional[dict] = None,
) -> str:
    """Replace every detected complex repetition with an explicit HTML comment.

    All non-overlapping findings in a pass are replaced together. Detection is
    then rerun so that edits cannot hide a later finding. The function preserves
    all text outside the reported spans and is idempotent once no findings remain.
    """
    current = str(text or "")
    marker = str(replacement)
    if not marker:
        raise ValueError("replacement must not be empty")
    if max_passes < 1:
        raise ValueError("max_passes must be positive")
    if max_findings_per_pass < 1:
        raise ValueError("max_findings_per_pass must be positive")

    total_replacements = 0
    total_removed_chars = 0
    pass_records: List[dict] = []
    rule_counts: Dict[str, int] = {}

    def record_metrics(preserved: Sequence[Mapping[str, object]]) -> None:
        if metrics is None:
            return
        metrics["complex_repetition_passes"] = len(pass_records)
        metrics["complex_repetition_replacements"] = total_replacements
        metrics["complex_repetition_characters_removed"] = total_removed_chars
        metrics["complex_repetition_rule_counts"] = dict(sorted(rule_counts.items()))
        metrics["complex_repetition_replacement_details"] = pass_records
        metrics["complex_repetition_preserved_findings"] = [dict(item) for item in preserved]

    for pass_index in range(1, max_passes + 1):
        details = detect_complex_repetition_details(
            current,
            max_findings=max_findings_per_pass,
            min_pattern_tokens=min_pattern_tokens,
            max_pattern_tokens=max_pattern_tokens,
            min_pattern_repeats=min_pattern_repeats,
            numeric_sequence_min_terms=numeric_sequence_min_terms,
            min_trailing_partial_tokens=min_trailing_partial_tokens,
        )
        if not details:
            record_metrics([])
            return current

        cleanable_details: List[dict] = []
        preserved_details: List[dict] = []
        for detail in details:
            reason = _structural_preservation_reason(current, detail) if preserve_structural_headers else None
            if reason is None:
                cleanable_details.append(detail)
                continue
            preserved_detail = dict(detail)
            preserved_detail["preservation_reason"] = reason
            preserved_details.append(preserved_detail)
        if not cleanable_details:
            record_metrics(preserved_details)
            return current

        spans = _merged_replacement_spans(current, cleanable_details)
        pieces: List[str] = []
        cursor = 0
        normalized_spans: List[dict] = []
        for span in spans:
            start = int(span["start_index"])
            end = int(span["end_index"])
            pieces.extend((current[cursor:start], marker))
            cursor = end
            rules = sorted({str(item["rule"]) for item in span["findings"]})
            for rule in rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
            normalized_spans.append(
                {
                    "start_index": start,
                    "end_index": end,
                    "removed_char_count": end - start,
                    "rules": rules,
                    "findings": span["findings"],
                }
            )
            total_removed_chars += end - start
        pieces.append(current[cursor:])
        updated = "".join(pieces)
        if updated == current:
            raise RuntimeError("complex-repetition replacement made no progress")
        current = updated
        total_replacements += len(normalized_spans)
        pass_records.append({"pass_index": pass_index, "spans": normalized_spans})

    remaining = detect_complex_repetition_details(
        current,
        max_findings=1,
        min_pattern_tokens=min_pattern_tokens,
        max_pattern_tokens=max_pattern_tokens,
        min_pattern_repeats=min_pattern_repeats,
        numeric_sequence_min_terms=numeric_sequence_min_terms,
        min_trailing_partial_tokens=min_trailing_partial_tokens,
    )
    remaining_cleanable = [
        detail
        for detail in remaining
        if not preserve_structural_headers or _structural_preservation_reason(current, detail) is None
    ]
    if remaining_cleanable:
        raise RuntimeError(f"complex repetition remains after {max_passes} replacement passes")
    preserved_remaining = []
    for detail in remaining:
        reason = _structural_preservation_reason(current, detail) if preserve_structural_headers else None
        if reason is not None:
            preserved_detail = dict(detail)
            preserved_detail["preservation_reason"] = reason
            preserved_remaining.append(preserved_detail)
    record_metrics(preserved_remaining)
    return current
