import re

def contains_weird_characters(line):
    # Updated pattern to include all Greek characters, their variations, and all mentioned punctuation
    allowed_chars = (
        r'a-zA-Z\d\s'  # English letters, digits, and whitespace
        r'\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F'  # Greek and Ancient Greek
        r'\u0387\u00B7'  # Greek punctuation
        r'.,;:!?()\[\]{}'  # Basic punctuation
        r'\-\–\—\―\‒\_'  # Various dashes and underscore
        r'«»'  # Greek quotation marks
        r"''΄'’´"  # Apostrophes and stress mark
        r'…·'  # Ellipsis and ano teleia
        r'+=*/\\|<>~@#$%^&€£¥§©®™°±²³¹½¼¾'  # Special characters and symbols
        r'×÷πΠ∞∑√∫≈≠≤≥←↑→↓↔⇒⇔∀∃∈∩∪⊂⊃⊄⊅⊆⊇⊕⊗⋅∅∇∂∫∮∴∵∼≅≈≡'  # Mathematical symbols
        r'•○●◦◘◙♦♣♠♥♪♫'  # Common bullet points and list markers
    )
    weird_char_pattern = re.compile(f'[^{allowed_chars}]')
    
    return bool(weird_char_pattern.search(line))

# Test cases
test_lines = [
    "This is a normal English sentence.",
    "Αυτή είναι μια κανονική ελληνική πρόταση.",
    "Mixed English and Ελληνικά with numbers 12345 and symbols @#$%^&*().",
    "Mathematical expression: ∫x²dx = x³/3 + C",
    "Ο Μπόξερ είπε: «Είναι νεκρός».",
    "τα ψάρια: μουρούνα, γαύρος, ρέγγες, κολιός …",
    "Κάτι · κάτι με άνω τελεία",
    "«Γιατί, τότε», ρώτησε κάποιος, «τον είχε πολεμήσει με όλα τα μέσα;»",
    "2,95€ (€2.95)",
    "το κόστος του σπιτιού ήταν £260.950,00",
    "Τις Κυριακές δε δούλευε κανένας.",
    "— Αύριο, έλεγε, στις πέντε με πέντε και πέντε, …",
    "Various dashes: - – — ― ‒ _",
    "Greek stress mark: άέίόύώή",
    "Ancient Greek: ἀἁἂἃἄἅἆἇἐἑἒἓἔἕἠἡἢἣἤἥἦἧἰἱἲἳἴἵἶἷὀὁὂὃὄὅὐὑὒὓὔὕὖὗὠὡὢὣὤὥὦὧὰάὲέὴήὶίὸόὺύὼώ",
    "This line has a weird character: ☺",
    "Α´.1. Περίληψη της"
]

for i, line in enumerate(test_lines, 1):
    result = contains_weird_characters(line)
    print(f"Line {i}: {'Contains' if result else 'Does not contain'} weird characters")
    print(f"Text: {line}\n")