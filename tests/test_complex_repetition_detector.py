from glossapi.ocr.utils.repetition import (
    TEXT_REMOVED_COMMENT,
    detect_complex_repetition_details,
    replace_complex_repetitions,
)


def test_detects_repeated_html_table_rows_as_a_complex_block():
    first_row = "<tr><td>1</td><td>1</td><td>2</td><td>2</td><td>1</td></tr>"
    repeated_row = "<tr><td>1</td><td>1</td><td>1</td><td>2</td><td>2</td><td>1</td></tr>"
    text = "<table>\n" + first_row + "\n" + "\n".join([repeated_row] * 9) + "\n</table>"

    details = detect_complex_repetition_details(text)

    detail = next(item for item in details if item["rule"] == "repeated_token_block")
    assert detail["cut_index"] <= text.index(repeated_row)
    assert detail["repeat_count"] >= 9
    assert detail["pattern_token_count"] > 1
    assert detail["distinct_content_token_count"] >= 2


def test_detects_regular_numeric_sequence_after_an_irregular_prefix():
    irregular = ["164.799", "164.331", "164.433", "164.383", "164.326"]
    regular = [f"{value // 1000}.{value % 1000:03d}" for value in range(164_071, 163_500, -11)]
    text = "Numerous labels: " + ", ".join([*irregular, *regular])

    details = detect_complex_repetition_details(text)

    detail = next(item for item in details if item["rule"] == "regular_numeric_sequence")
    assert detail["cut_index"] == text.index(regular[0])
    assert detail["sequence_step"] == "-0.011"
    assert detail["sequence_term_count"] == len(regular)


def test_numeric_sequence_allows_one_display_unit_of_rounding_noise():
    values = [f"10.{fraction:03d}" for fraction in range(340, 9, -11)]
    values.extend(["10.000", "9.989", "9.978", "9.967"])
    values.extend(f"9.{fraction:03d}" for fraction in range(956, 700, -11))
    text = ", ".join(values)

    details = detect_complex_repetition_details(text)

    detail = next(item for item in details if item["rule"] == "regular_numeric_sequence")
    assert detail["sequence_step"] == "-0.011"
    assert detail["sequence_step_tolerance"] == "0.001"
    assert detail["sequence_term_count"] == len(values)


def test_excludes_single_character_and_single_token_runs():
    assert detect_complex_repetition_details("x" * 10_000) == []
    assert detect_complex_repetition_details(("word " * 500).strip()) == []


def test_excludes_short_repetitions_and_short_number_lists():
    row = "<tr><td>one</td><td>two</td></tr>"
    text = "<table>" + "\n".join([row] * 3) + "</table>\n" + ", ".join(str(i) for i in range(20))

    assert detect_complex_repetition_details(text) == []


def test_replaces_full_block_run_and_truncated_final_copy():
    row = "<tr><td>alpha</td><td>beta</td><td>gamma</td></tr>\n"
    truncated_row = "<tr><td>alpha</td><td>bet"
    text = "before\n" + (row * 5) + truncated_row
    metrics = {}

    cleaned = replace_complex_repetitions(text, metrics=metrics)

    assert cleaned == "before\n" + TEXT_REMOVED_COMMENT
    assert metrics["complex_repetition_replacements"] == 1
    assert metrics["complex_repetition_passes"] == 1
    detail = metrics["complex_repetition_replacement_details"][0]["spans"][0]["findings"][0]
    assert detail["trailing_partial_token_count"] >= 3
    assert detail["trailing_partial_fragment"] is True
    assert detail["removal_end_index"] == len(text)


def test_replaces_regular_numeric_sequence_without_removing_context():
    sequence = ", ".join(str(value) for value in range(100, 140))
    text = f"prefix [{sequence}] suffix"

    cleaned = replace_complex_repetitions(text)

    assert cleaned == f"prefix [{TEXT_REMOVED_COMMENT}] suffix"


def test_replaces_every_separate_run_and_is_idempotent():
    sections = []
    for index in range(10):
        pattern = f"alpha{index} beta{index} gamma{index} delta{index} epsilon{index} zeta{index}.\n"
        sections.append(pattern * 4)
        sections.append(f"separator {index} remains.\n")
    text = "".join(sections)
    metrics = {}

    cleaned = replace_complex_repetitions(text, metrics=metrics)

    assert cleaned.count(TEXT_REMOVED_COMMENT) == 10
    assert all(f"separator {index} remains." in cleaned for index in range(10))
    assert metrics["complex_repetition_replacements"] == 10
    assert detect_complex_repetition_details(cleaned) == []
    assert replace_complex_repetitions(cleaned) == cleaned


def test_replacement_leaves_simple_runs_out_of_scope():
    text = ("x" * 10_000) + "\n" + (("word " * 500).strip())

    assert replace_complex_repetitions(text) == text


def test_replacement_consumes_substantial_truncated_word_before_new_section():
    pattern = "will confirm that all costs are extra and "
    text = "prefix " + (pattern * 10) + "will confir\n2. Next section"

    cleaned = replace_complex_repetitions(text)

    assert cleaned == "prefix " + TEXT_REMOVED_COMMENT + "\n2. Next section"


def test_numeric_sequence_includes_sentence_final_integer():
    sequence = ", ".join(str(value) for value in range(1, 41))
    text = f"labels: {sequence}.\nNext paragraph"

    cleaned = replace_complex_repetitions(text)

    assert cleaned == f"labels: {TEXT_REMOVED_COMMENT}.\nNext paragraph"


def test_numeric_sequence_consumes_truncated_expected_value_at_line_end():
    sequence = ", ".join(str(value) for value in range(100, 164, 2))
    text = f"labels: {sequence}, 16\nNext paragraph"

    cleaned = replace_complex_repetitions(text)

    assert cleaned == f"labels: {TEXT_REMOVED_COMMENT}\nNext paragraph"


def test_preserves_repeated_html_table_headers_by_default():
    text = "<table><tr>" + ("<th>A/A Repeated score heading</th>" * 4) + "</tr></table>"
    assert detect_complex_repetition_details(text)
    metrics = {}

    cleaned = replace_complex_repetitions(text, metrics=metrics)

    assert cleaned == text
    assert metrics["complex_repetition_replacements"] == 0
    assert metrics["complex_repetition_preserved_findings"][0]["preservation_reason"] == (
        "repeated_html_table_header_cells"
    )
    assert replace_complex_repetitions(text, preserve_structural_headers=False) == (
        "<table><tr>" + TEXT_REMOVED_COMMENT + "</tr></table>"
    )


def test_preserves_short_repeated_markdown_header_row_but_not_long_runaway():
    header = "| REPEATED COLUMN SCORE HEADING VALUE | REPEATED COLUMN SCORE HEADING VALUE | REPEATED COLUMN SCORE HEADING VALUE | REPEATED COLUMN SCORE HEADING VALUE |"
    runaway = "| " + " | ".join(["BROKEN BILLING PERIOD VALUE PLACEHOLDER"] * 8) + " |"

    assert replace_complex_repetitions(header) == header
    assert replace_complex_repetitions(runaway) != runaway


def test_preserves_short_closure_of_open_quote_at_rotated_repeat_boundary():
    cycle = "Ζουράρις;». Όταν πρέπει πάραυτα να λέμε τελικά να πει ο Κ. "
    text = "Τα πρώτα λόγια είναι «Ότι πρέπει πάραυτα να λέμε τελικά να πει ο Κ. " + cycle * 5
    metrics = {}

    cleaned = replace_complex_repetitions(text, metrics=metrics)

    assert cleaned.startswith("Τα πρώτα λόγια είναι «Ότι πρέπει πάραυτα να λέμε τελικά να πει ο Κ. Ζουράρις;». ")
    assert cleaned.rstrip().endswith(TEXT_REMOVED_COMMENT)
    span = metrics["complex_repetition_replacement_details"][0]["spans"][0]
    assert span["findings"][0]["boundary_prefix_preserved_char_count"] > 0
