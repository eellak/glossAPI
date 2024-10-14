import unittest
import unicodedata

def remove_accents(line):
    line = line.lower()
    accentless_line = ''.join(
        c for c in unicodedata.normalize('NFD', line)
        if unicodedata.category(c) != 'Mn'
    )
    return accentless_line

class TestRemoveAccents(unittest.TestCase):
    def test_basic_lowercasing_and_accent_removal(self):
        input_str = "Κεφάλαιο 55"
        expected = "κεφαλαιο 55"
        self.assertEqual(remove_accents(input_str), expected)

    def test_mixed_capitalization_and_accents(self):
        input_str = "ΑΣΚΗΣΗ 1"
        expected = "ασκηση 1"
        self.assertEqual(remove_accents(input_str), expected)

    def test_multiple_accents_on_single_character(self):
        input_str = "ερωτήσεις"
        expected = "ερωτησεις"
        self.assertEqual(remove_accents(input_str), expected)

    def test_words_without_accents(self):
        input_str = "νεφρώνα"
        expected = "νεφρωνα"
        self.assertEqual(remove_accents(input_str), expected)

    def test_empty_string(self):
        input_str = ""
        expected = ""
        self.assertEqual(remove_accents(input_str), expected)

    def test_string_with_only_accents(self):
        input_str = "άέήίόύώ"
        expected = "αεηιουω"
        self.assertEqual(remove_accents(input_str), expected)

    def test_special_characters_and_punctuation(self):
        input_str = "Κεφάλαιο 6, [ΑΣΚ]ΕΡΩΤΗΣΕΙΣ!"
        expected = "κεφαλαιο 6, [ασκ]ερωτησεις!"
        self.assertEqual(remove_accents(input_str), expected)

    def test_numbers_and_symbols_mixed_with_text(self):
        input_str = "Κεφάλαιο-6_ΕΡΩΤΗΣΕΙΣ"
        expected = "κεφαλαιο-6_ερωτησεις"
        self.assertEqual(remove_accents(input_str), expected)

    def test_words_with_hyphens_and_spaces(self):
        input_str = "Δραστηριότητες-ερωτήσεις"
        expected = "δραστηριοτητες-ερωτησεις"
        self.assertEqual(remove_accents(input_str), expected)

    def test_words_with_multiple_accents(self):
        input_str = "Φύλλο Εργασίας/Αξιολόγησης"
        expected = "φυλλο εργασιας/αξιολογησης"
        self.assertEqual(remove_accents(input_str), expected)

if __name__ == '__main__':
    unittest.main()
