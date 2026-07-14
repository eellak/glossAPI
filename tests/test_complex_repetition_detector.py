from glossapi.ocr.utils.repetition import detect_complex_repetition_details


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
