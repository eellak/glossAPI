import re
import unittest
import unicodedata


def remove_accents(line):
    line = line.lower()
    accentless_line = ''.join(
        c for c in unicodedata.normalize('NFD', line)
        if unicodedata.category(c) != 'Mn'
    )
    return accentless_line

def find_excercise_line(line):
    accentless_line = remove_accents(line)
    single_spaced_line = re.sub(r'\s+', ' ', accentless_line).strip()
    
    if len(single_spaced_line) <= 20:
        if askiseis_pattern.search(single_spaced_line):
            return "askiseis"
        if askisi_pattern.search(single_spaced_line):
            return "askisi"
        #if drastiriotites_pattern.search(single_spaced_line):
         #   return "drastiriotites"
        if erotiseis_askiseis_pattern.search(single_spaced_line):
            return "erotiseis_askiseis"
        if erotiseis_pattern.search(single_spaced_line):
            return "erotiseis"
        if fyllo_ergasias_pattern.search(single_spaced_line):
            return "fyllo_ergasias"
        if askiseis_chapter_pattern.search(single_spaced_line):
            return "askiseis_chapter"
        if erotimatologio_pattern.search(single_spaced_line):
            return "erotimatologio"
        if akrostixida_pattern.search(single_spaced_line):
            return "akrostixida"
    return ""

# Patterns to find exercises
askiseis_pattern = re.compile(r"^[αa]σ[κk][ηh]σ[εe][iι]σ", re.UNICODE)
askisi_pattern = re.compile(r"^[αa]σ[κk][ηh]σ[ηh] \d{1,2}", re.UNICODE)
erotiseis_pattern = re.compile(r"^[εe][pρ]ω[τt][ηh]σ[εe][iι]σ", re.UNICODE)
erotiseis_askiseis_pattern = re.compile(r"[εe][pρ]ω[τt][ηh]σ[εe][iι]σ[\s-]*[αa]σ[κk][ηh]σ[εe][iι]σ[\s-]*[πp][pρ][oο][βb]λ[ηh]μ[αa][τt][αa]", re.UNICODE)
fyllo_ergasias_pattern = re.compile(r"φ[υy]λλ[oο] [εe][pρ]γ[αa]σ[iι][αa]σ(\ [αa]ξ[iι][oο]λ[oο]γ[ηh]σ[ηh]σ)?", re.UNICODE)
askiseis_chapter_pattern = re.compile(r"[αa]σ[κk][ηh]σ[εe][iι]σ\s+\w+\s+[κk][εe]φ[αa]λ[αa][iι][oο][υy]", re.UNICODE)
erotimatologio_pattern = re.compile(r"[εe][pρ]ω[τt][ηh]μ[αa][τt][oο]λ[oο]γ[iι][oο]", re.UNICODE)
akrostixida_pattern = re.compile(r"[αa][κk][pρ][oο]σ[τt][iι][xχ][iι]δ[αa]", re.UNICODE)

class TestFindExerciseLine(unittest.TestCase):
    
    def test_find_exercise_line(self):
        line = "AKPOΣTIXIΔA"
        
        # Apply all the transformations step by step and print the output
        print(f"Original line: {line}")
        accentless_line = remove_accents(line)
        print(f"Accentless line: {accentless_line}")
        single_spaced_line = re.sub(r'\s+', ' ', accentless_line).strip()
        print(f"Single spaced line: {single_spaced_line}")
        print(f"Length of single spaced line: {len(single_spaced_line)}")
        
        result = find_excercise_line(line)
        print(f"Result: {result}")
        self.assertEqual(result, "akrostixida")

if __name__ == '__main__':
    unittest.main()
