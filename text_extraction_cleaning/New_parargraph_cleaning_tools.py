import re
import os

file_stat_list = []

def file_reset_list() :
    return []

def paragraph_maker(text,maxpadding = 2) :
    lines = text.splitlines()
    paragraphs = []
    emptyline = 0
    paragraph = ''
    for i,line in enumerate(lines) :
        if i == len(lines) - 1 :
            paragraph = paragraph + line
            paragraphs.append(paragraph)
            continue
        if line == '' :
            emptyline = emptyline + 1
            if emptyline == maxpadding :
                if paragraph != '' :
                    paragraphs.append(paragraph)
                emptyline = 0
                paragraph = ''
        else :
            paragraph = paragraph + line + '\n'
            emptyline = 0
            if i == len(lines) - 1 :
                paragraphs.append(paragraph)
    return paragraphs

def paragraph_merger(paragraphs,min_par_size,threshold) :
    newparagraphs = []
    for i,paragraph in enumerate(paragraphs) :
        if len(paragraph) < min_par_size :
            if paragraph != paragraphs[-1] :
                if len(paragraphs[i+1]) > threshold :
                    if not (paragraphs[i+1].startswith('##') or paragraphs[i+1].startswith('|') ) :
                        paragraphs[i+1] = paragraph + paragraphs[i+1]
                        continue
        newparagraphs.append(paragraph)
    return newparagraphs

def paragraph_clean_image(paragraphs) :
    newparagraphs = []
    for paragraph in paragraphs :
        if paragraph.startswith('<!-- image -->') :
            continue
        else :
            newparagraphs.append(paragraph)
    return newparagraphs

def paragraph_clean_dotlines(paragraphs) :
    newparagraphs = []
    for paragraph in paragraphs :
        if paragraph.startswith('.....') :
            if paragraph.endswith('.....') :
                continue
        else :
            newparagraphs.append(paragraph)
    return newparagraphs

def paragraph_remove_artifacts(paragraphs,min_length = 5) :
    newparagraphs = []
    for paragraph in paragraphs :
        if len(paragraph) <= min_length :
            continue
        newparagraphs.append(paragraph)
    return newparagraphs

def paragraph_not_char_end(paragraph,chars,print) :
    #if not paragraph.startswith('##') :
    end_search_flag = False
    if not paragraph.endswith(chars) :
        if print :
            paragraph = '[Out:Paragraph does not end with specified char]' + paragraph
        else : paragraph = ''
        end_search_flag = True
    return (paragraph,end_search_flag)

def all_paragraph_not_char_end(paragraphs,chars,print = False) :
    newparagraphs = []
    for i,paragraph in enumerate(paragraphs) :
        newparagraphs.append(paragraph_not_char_end(paragraph,chars,print)[0])
        if paragraph_not_char_end(paragraph,chars,print)[1] == False :
            for t_paragraph in paragraphs[i+1:] :
                newparagraphs.append(t_paragraph)
            return newparagraphs
    return newparagraphs

def remove_content_table_begin(paragraphs,num_of_front = 20,print = False) :
    newparagraphs = []
    table_starting_char = ('|','|')
    for paragraph in paragraphs[:num_of_front] :
        if any(paragraph.startswith(char) for char in table_starting_char) :
            if print :
                paragraph = '[Out:Conent Table]' + paragraph
                newparagraphs.append(paragraph)
        else :
            newparagraphs.append(paragraph)
    for paragraph in paragraphs[num_of_front:] :
        newparagraphs.append(paragraph)
    return newparagraphs

def paragraph_fix_broken_line(paragraphs) :
    ending_lower_pattern = re.compile(r'(.*[α-ωά-ώa-zΑ-ΩΆ-Ώ]\n)|(.*[0-9]\n)|(.*°C\n)|(.*\)\n)|(.*\-\n)|(.*,\n)')
    starting_lower_pattern = re.compile(r'([α-ωά-ώa-z])|([0-9])|(°)|\)|- ')
    newparagraphs = []
    for i,paragraph in enumerate(paragraphs) :
        if re.match(ending_lower_pattern,paragraph) :
            if paragraph != paragraphs[-1] :
                if re.match(starting_lower_pattern,paragraphs[i+1]) :
                    paragraphs[i+1] = paragraph + paragraphs[i+1]
                    continue
        newparagraphs.append(paragraph)
    return newparagraphs

def print_text(paragraphs) :
    for paragraph in paragraphs :
        print(paragraph)
        print('\n\n\n')

def remove_ending_chunk(paragraphs) :
    paragraphs = paragraphs[::-1]
    for i,paragraph in enumerate(paragraphs) :
        if paragraph.startswith('##') :
            paragraphs[i] = '[Out:Paragraph is in ending chunk]' + paragraphs[i]
            return paragraphs[::-1]
        paragraphs[i] = '[Out:Paragraph is in ending chunk]' + paragraphs[i]
        #del paragraphs[-1]
    return paragraphs[::-1]

def remove_numbered_title(paragraphs,pattern) :
    newparagraphs = []
    for paragraph in paragraphs :
        if paragraph.startswith('##') :
            paragraph = re.sub(pattern,'## ',paragraph)
        newparagraphs.append(paragraph)
    return newparagraphs

def util_check_and_remove(paragraph,newparagraphs,tags,ending_tags,print) :
    bibliography_flag = True
    if ending_tags == () : ending_tags = ('##','[Out:')
    else : ending_tags = ('[Out:',) + ending_tags
    if paragraph.startswith(ending_tags) :
        bibliography_flag = False
        newparagraphs.append(paragraph)
    elif print :
        paragraph = '[Out:Paragraph is part of ' + tags[0] + ']' + paragraph
        newparagraphs.append(paragraph)
    return bibliography_flag,newparagraphs

def util_skip_handler(counter,ending_tags,paragraph,flag) :
    if flag == False : counter = counter - 1
    elif paragraph.startswith(('##','[Out:')) : counter = counter - 1
    if counter <= 0 : ending_tags = tuple()
    return counter,ending_tags

def util_stat_creator(tags) :
    file_stat_list.append([tags[0],0])

def util_stat_incrementor() :
    file_stat_list[-1][-1] = file_stat_list[-1][-1] + 1

def stat_assembly(text,paragraphs) :
    file_stat_list.insert(0,['Total Chars',str(len(text))])
    file_stat_list.insert(0,['Total lines',str(len(text.splitlines()))])
    file_stat_list.insert(0,['Total Paragraphs',str(len(paragraphs))])

def remove_taged_paragraphs(paragraphs,tags,ending_tags=tuple(),print = False,skip_paragraphs = 0 ) :
    newparagraphs = []
    bibliography_flag = False
    util_stat_creator(tags)
    for paragraph in paragraphs :
        if paragraph.startswith(tags) :
            bibliography_flag = True
            util_stat_incrementor()
            if print :
                paragraph = '[Out:Paragraph is start of ' + tags[0] + ']' + paragraph
                newparagraphs.append(paragraph)
        elif bibliography_flag == True :
            bibliography_flag,newparagraphs = util_check_and_remove(paragraph,newparagraphs,tags,ending_tags,print)
            if bibliography_flag == True : util_stat_incrementor()
            skip_paragraphs,ending_tags = util_skip_handler(skip_paragraphs,ending_tags,paragraph,bibliography_flag)
        else : newparagraphs.append(paragraph)
    return newparagraphs

def remove_noise(paragraphs,pattern) :
    newparagraphs = []
    for paragraph in paragraphs :
        paragraph = re.sub(pattern,'',paragraph)
        newparagraphs.append(paragraph)
    return newparagraphs

def remove_contained_pattern(paragraphs,pattern,print = False) :
    newparagraphs = []
    for paragraph in paragraphs :
        if re.search(pattern,paragraph):
            if print :
                paragraph = '[Out: Pattern found]' + paragraph
                newparagraphs.append(paragraph)
                continue
            else : continue
        newparagraphs.append(paragraph)
    return newparagraphs

def remove_link(text) :
    text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)','',text)
    return text

def remove_octothrops(text) :
    text = re.sub(r'## ','',text)
    return text

def remove_all_octothrops(paragraphs) :
    newparagraphs = []
    for paragraph in paragraphs :
        newparagraphs.append(remove_octothrops(paragraph))
    return newparagraphs

def test_write_text(paragraphs,file) :
    text = '[Paragraph Begin: Number 0 ]'
    for i,paragraph in enumerate(paragraphs) :
        text = text + paragraph + '[Paragraph End]\n\n\n[Paragraph Begin: Number ' + str(i+1) + ' ]'
    file.write(text)

def write_text(paragraphs,file) :
    text=''
    for i,paragraph in enumerate(paragraphs) :
        text = text + paragraph + '\n'
    file.write(text)

def total_paragraphs(paragraphs) :
    text = ''
    for paragraph in paragraphs :
        text = text + paragraph + '\n'
    return text

#needs testing
def tags_to_pattern(tags) :
    pattern = r''
    for tag in tags :
        pattern = pattern+'|'+tag
    return pattern
        

def remove_paragraph_by_context(paragraphs, begin_tags, print=False, skip_paragraphs=1, end_tags=None):
    newparagraphs = []
    bibliography_flag = False
    c = skip_paragraphs  # Initialize counter if skip_paragraphs > 0
    
    for paragraph in paragraphs:
        if any(paragraph.startswith(tag) for tag in begin_tags):
            bibliography_flag = True
            if print:
                paragraph = '[Out:Paragraph is' + begin_tags[0] + ']' + paragraph
                newparagraphs.append(paragraph)
            continue
        elif bibliography_flag == True:
            if paragraph.startswith(('##','[Out:')) and not any(paragraph.startswith(tag) for tag in begin_tags):
                if skip_paragraphs > 0:
                    # Check if paragraph starts with end tags
                    if end_tags and any(paragraph.startswith(tag) for tag in end_tags):
                        bibliography_flag = False
                    else:
                        c -= 1
                        if c <= 0:
                            bibliography_flag = False
                else:
                    # Όπως πριν
                    bibliography_flag = False
                newparagraphs.append(paragraph)
                continue
            if print:
                paragraph = '[Out:Paragraph is' + begin_tags[0] + ']' + paragraph
                newparagraphs.append(paragraph)
                continue
            else:
                continue
        newparagraphs.append(paragraph)
    return newparagraphs

endings = ('.\n','?\n','!\n',';\n','.\n',';\n','|\n')
# BEGIN TAGS
bibliography_tags = ('## Βιβλιογραφία', '## ΞΕΝΟΓΛΩΣΣΗ','## ΕΛΛΗΝΙΚΗ','## Bl ΒΛΙΟΓΡΑΦΙΑ', '## ΒΛΙΟΓΡΑΦΙΑ','## Π Ι ΝΑ Κ Α Σ Π Ρ Ο Ε Λ Ε Υ Σ Η Σ Τ Ω Ν E Ι ΚΟ Ν Ω Ν',
                         '## ΠΙΝΑΚΑΣ ΠΡΟΕΛΕΥΣΗΣ ΤΩΝ EΙΚΟΝΩΝ','## ΒΙΒΛΙΟΓΡΑΦΙΑ','## ΕΙΚΟΝΟΓΡΑΦΙΚΟ ΥΛΙΚΟ','## Διαδικτυακοί τόποι','## ΚΕΙΜΕΝΑ',
                         '## Ηλεκτρονικές διευθύνσεις στο Internet','## Ξενόγλωσση','## Ελληνική','## ΠΑΡΑΡΤΗΜΑ','## Στ. Φωτογραφικά άλμπουμ',
                         '## Θ. Ιστότοποι','## www.','## Η. Λοιπές μελέτες','## Επιλογή Βιβλιογραφίας','## ΠΑΙΔΑΓΩΓΙΚΕΣ ΜΕΛΕΤΕΣ','## ΠΗΓΕΣ ΦΩΤΟΓ ΡΑΦΙΚΟΥ ΥΛΙΚΟΥ',
                         '## Βιβλιογραφία','## 7. Εικονική βιβλιοθήκη Επιφανειοδραστικών','## ΞΕΝΗ','## Βιβλιογραφικά Βοηθήματα','## ΠΑΡΑΠΟΜΠΕΣ',
                         '## 7. Εικονική βιβλιοθήκη Επιφανειοδραστικών','## ΠΗΓΕΣ ΦΩΤΟΓΡΑΦΙΩΝ','## ΠΗrΕΣ προέλευσης','## Πηγές βοηθητικού υλικού','## Θεωρία',
                         '## Εικονογραφικό υλικό','## Ευχαριστούμε τους παρακάτω εκδοτικούς οίκους','## ΠPOEΛEYΣH TOY EIKONOΓPAΦIKOY YΛIKOY','## Πηγές εικόνων',
                         '## Πρόσθετο εικονιστικό υλικό','ΠΗΓΕΣ ΕΠΟΠΤΙΚΟΥ ΥΛΙΚΟΥ','## ΠΗΓΕΣ ΥΛΙΚΟΥ','Πίνακας συνδέσμων','## Βιβλία-Περιοδικά Ελληνόγλωσσα',
                         '## Ξενόγλωσσα','## ΣΗΜΕΙΩΣΕΙΣ ','## Κλασικά','## Φωτογραφίες και πίνακες που χρησιµοποιήθηκαν','## Ενδεικτική Βιβλιογραφία',
                         '## Ενδεικτική βιβλιογραφία','## βιβλιογραφία','## ΠΗΓΕΣ ΕΠΟΠΤΙΚΟΥ ΥΛΙΚΟΥ','## Εικονιστικό Υλικό','## Πίνακας συνδέσμων διαδικτυακών διευθύνσεων',
                         '## Βιβλία αναφοράς','## Σύνδεσμοι','## ΠΗΓΕΣ','## ΑΠΟ ΣΧΟΛΙΚΑ ΕΓΧΕΙΡΙΔΙΑ','## ΧΕΙΡΟΓΡΑΦΑ','## ΜΟΥΣΙΚΕΣ - ΜΟΥΣΙΚΟΛΟΓΙΚΕΣ ΜΕΛΕΤΕΣ','## Ξενόγλωσσες',
                         'Πηγή: ','## ΞΕΝΟΓΛΩΣΣΑ ΑΡΘΡΑ','## ΕΠΙΛΕΓΜΕΝΕΣ ΠΗΓΕΣ','## 13 Βιβλιογραφία - πηγές','## ΧΡΗΣΙΜΕΣ ΔΙΕΥΘΥΝΣΕΙΣ ΔΙΑΔΙΚΤΥΟΥ','## 8.6 Αναφορές - Βιβλιογραφία',
                         '## 8. Βιβλιογραφία','## ΠΗΓΕΣ ΕΙΚΟΝΩΝ','ΠΗΓΕΣ ΕΙΚΟΝΩΝ','## Ξένη','## Α ΒΙΒΛΙΑ','## ΜΕΤΑΦΡΑΣΜΕΝΗ','## ΞΕΝΗ','## Διεθνής Βιβλιογραφία',
                         '## ΔΙΚΤΥΑΚΟΙ ΤΟΠΟΙ','## ΒΙΒΛΙΟΓΡΑΦΙΚΕΣ ΠΗΓΕΣ','## Πηγές','## Bιβλιογραφία','## Εικόνες εσωφύλλου','## 7. Περιοδικά','## 6. Μη Κυβερνητικές Οργανώσεις',
                         '## 5. Ευρωπαϊκή Ένωση','## 4. Ειδικές Ιστοσελίδες','## 3. Διεθνείς Θεσµοί','## 2. Δηµόσιες και Ανεξάρτητες Υπηρεσίες, Αρχές και πολιτικά κόµµατα',
                         '## 1. Γενικοί Αναζητητές','## Ελληνόγλωσση','## α. Ελληνόγλωσση','## β. Ξενόγλωσση','## Ι. Ελληνική','## II. Ξενόγλωσση','## 1.7. Βιβλιογραφία-Δικτυογραφία',
                         '## 2.3. Βιβλιογραφία - Δικτυογραφία','## 3.6. Βιβλιογραφία - Δικτυογραφία','## 4.7. Βιβλιογραφία - Δικτυογραφία','## 5.10. Βιβλιογραφία-Δικτυογραφία',
                         '## ΧΡΗΣΙΜΕΣ ΔΙΕΥΘΥΝΣΕΙΣ','## ΚΑΤΑΛΟΓΟΣ ΑΡΜΟΔΙΩΝ','## Τεχνικά φυλλάδια','## Περιοδικά και ενημερωτικά φυλλάδια','## Στοιχεία και πληροφορίες','## ΠΑΡΟΡΑΜΑΤΑ',
                         '## 5 - ΠΙΝΑΚΑΣ ΕΤΑΙΡΙΩΝ','## 4 - ΔΙΕΥΘΥΝΣΕΙΣ INTERNET','## 3 - ΤΕΧΝΙΚΑ ΠΕΡΙΟΔΙΚΑ','## 2 - ΠΡΟΤΥΠΑ - ΤΕΧΝΙΚΕΣ ΟΔΗΓΙΕΣ','## 1 - ΤΕΧΝΙΚΑ ΒΙΒΛΙΑ & ΕΓΧΕΙΡΙΔΙΑ',
                         '## Χρήσιμοι Διαδικτυακοί Τόποι','## Χαρακτηριστικές NoSQL Βάσεις Δεδομένων','## https:','## Ηλεκτρονική Βιβλιογραφία','## Ελληνόγλωσση',
                         '## %CE%', '## Αναφορές', '## Ελληνόγλωσση', '## Βιβλιογραφικές Αναφορές')
glossary_tags = ('## ΓΛΩΣΣΑΡΙ, ## Γλωσσάρι','## Λ EΞIKO','## ΛEΞIKO','## ΛΕΞΙΛΟΓΙΟ','## ΓΛΩΣΣΑ ΡΙ','## Ζ. Λε ξ ικά','## Ορολογία','## Γλωσσάρι','## ΓΛΩΣΣΑΡΙΟ',
                     '## Συγκεντρωτικός πίνακας Γραµµατικής','## Β. Γλωσσάρι','## ΓΛΩΣΣΑΡΙ','## ΛΕΞΙΚΟ',']## ορολογίας στην Ελληνική και την Αγγλική','## Λεξικό',
                     '-Λ ΕΞΙΛΟΓΙΟ ΟΡΩΝ','##  Γλωσσάρι')
euritirio_tags = ('## EYPETHPIA','## ΕΥΡΕΤΗΡΙΟ','## Ευρετήριο','## II. Eυρετήριο όρων','## Αλφαβητικό ευρετήριο όρων','## ευρετήριο','## EYPETHPIO')
church_tags = ('## Δ. Λειτουργικά Β ι β λία','## Β . Μουσικά β ι β λία','## Ε. Πατερικά Έργα','## e-kere.gr','## Ι. Π η γές Εικόνων',
                   '## Α\' Γ Υ Μ Ν Α Σ Ι Ο Υ','## Α. Θεωρ η τικά Εκκλ η σιαστικ ή ς Μουσικ ή ς','## Ένα ταξίδι ζωής: Η συνάντηση Θεού και ανθρώπου μέσα από τις βιβλικές διηγήσεις',
                   '## Η Εκκλησία: πορεία ζωής μέσα στην ιστορία','## Η μαρτυρία της Ορθόδοξης Εκκλησίας στον σύγχρονο κόσμο','Εικονογράφηση εξωφύλλου','## Χριστιανισμός και Θρησκεύματα',
                   '## Εισαγωγική σημείωση','## ΘΕΟΛΟΓΙΚΕΣ ΜΕΛΕΤΕΣ','## ΕΝΤΥΠΕΣ ΣΥΛΛΟΓΕΣ ΕΚΚΛΗΣΙΑΣΤΙΚΗΣ ΜΟΥΣΙΚΗΣ','## Κ ΑΤΑ ΛΟΓΟΙ','## Γ. Κατάλογοι μουσικών χειρογράφων',
                   '## χείο Σχολείου Ψαλτικής')
content_tags = ('## ΠΕΡΙΕΧΟΜΕΝΑ','## ΠΕΡΙΕΧΟΜΕΝΟ','## Περιεχόμενο','## Περιεχόμενα','## ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ','## Πίνακας Περιεχομένων','## Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Α','## ΠEPIEXOMENA',
                    '## ΠΕΡΙΕΧΟΜΕΝA','## Περιεχόµενα','## Π Ι ΝΑ Κ Α Σ Π Ε Ρ Ι Ε ΧΟ Μ Ε Ν Ω Ν','## Συνοπτικός πίνακας περιεχοµένου','Πίνακας Περιεχομένων',' Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Α   | Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Α',
                    '## ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ', '## Πίνακας περιεχομένων', '## Π ΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ')
printing_tags = ('## ΣΤΟΙΧΕΙΑ ΕΠΑΝΕΚ∆ΟΣΗΣ','## ΣΤΟΙΧΕΙΑ ΑΡΧΙΚΗΣ ΕΚ∆ΟΣΗΣ','## ΥΠΟΥΡΓΕΙΟ ΠΑΙΔΕΙΑΣ',
                     '## ΣΥΓΓΡΑΦΕΙΣ','## ΚΡΙΤΕΣ','## ΣΥΝΤΟΝΙΣΤΗΣ','## ΙΝΣΤΙΤΟΥΤΟ ΕΚΠΑΙΔΕΥΤΙΚΗΣ ΠΟΛΙΤΙΚΗΣ','## ΓΛΩΣΣΙΚΟΣ ΕΛΕΓΧΟΣ', '## Σχετικές Ηλεκτρονικές διευθύνσεις', 
                     '## Ηλεκτρονική επεξεργασία εικόνας' , '## Ψηφιακή σχεδίαση','## Ηλεκτρονική Τυπογραφία','## Ηλεκτρονική Τυπογραφία','## ΕΠΙΤΡΟΠΗ ΚΡΙΣΗΣ','## ΓΛΩΣΣΙΚΗ ΕΠΙΜΕΛΕΙΑ',
                     '## Επιτροπή αξιολόγησης','## Ομάδα αναθεώρησης','## Εποπτεία και συντονισμός αναθεώρησης','## Καλλιτεχνική επιμέλεια','## ΕΠΟΠΤΕΙΑ ΤΗΣ ΑΝΑΜΟΡΦΩΣΗΣ ΣΤΟ ΠΛΑΙΣΙΟ ΤΟΥ Π.Ι.',
                     '## ΕΠΙΜΕΛΕΙΑ','## Κωδικός Βιβλίου','## Υπεύθυνη Δράσης','## Υπεύθυνος για τo Παιδαγωγικό','## Φιλολογική επιμέλεια','## ΣΗΜΕΙΩΣΕΙΣ',
                     'ΙΝΣΤΙΤΟΥΤΟ ΕΚΠΑΙΔΕΥΤΙΚΗΣ ΠΟΛΙΤΙΚΗΣ','©','## ΣΥΝΤΟΝΙΣΤΡΙΑ','## ΗΛΕΚΤΡΟΝΙΚΗ ΕΠΕΞΕΡΓΑΣΙΑ','## Συντονιστική Επιτροπή του Έργου','## ΥΠΕΥΘΥΝΟΣ ΣΤΟ ΠΛΑΙΣΙΟ',
                     'Email:','## Συγγραφική ομάδα','## ΓΕΝΙΚΗ ΕΠΙΜΕΛΕΙΑ','ΣΤΟΙΧΕΙΑ ΕΠΑΝΕΚ∆ΟΣΗΣ','## Ηλεκτρονικές Διευθύνσεις','## ΠΑΙΔΑΓΩΓΙΚΟ ΙΝΣΤΙΤΟΥΤΟ','## Ομάδα συγγραφής',
                     '## Ομάδα κρίσης','## Γλωσσική επιμέλεια','## Συντονιστής','## ΕΥΧΑΡΙΣΤΙΕΣ','Email:','## Ηλεκτρονική επεξεργασία','## Συντονισμός','## Συγγραφική Ομάδα',
                     '## Στην παρούσα έκδοση','## Καλλιτεχνική Επιμέλεια','## Ανώτατα Εκπαιδευτικά Ιδρύματα','## Τεχνολογικά Εκπαιδευτικά Ιδρύματα','## Ευχαριστίες',
                     '## ΣΤΟΙΧΕΙΑ ΑΡΧΙΚΗΣ ΕΚΔΟΣΗΣ','## ΥΠΕΥΘΥΝΟΙ ΤΟΥ ΜΑΘΗΜΑΤΟΣ','## πραγματοποιήθηκε ΣΤΟΙΧΕΙΑ ΕΠΑΝΕΚ∆ΟΣΗΣ')
exception_tags = ('## ΜΙΚΡΟΒΙΟΛΟΓΙΑ','## <non-compliant-utf8-text>','## Εφαρμογές με Λογισμικό','## Συντομογραφίες','## ● ','## ΝΟΜΟΘΕΣΙΑ','## Ἔα δὴ','## Ενέργεια 2.3.2.',
                      '## Σεκλιζιώτης Σταμάτης','## Συντμήσεις','Ασημάκη Π., Θεοδωροπούλου Μ., Κωνσταντινίδου Ε., Μπόντια Χ.')
summary_tags = ('## Περίληψη','## Abstract','## Summary','## ABSTRACT','## SUMMARY','## ΠΕΡΙΛΗΨΗ')
catalog_tags = ('## Κατάλογος Πινάκων','## Πινάκας Πινάκων','## ΚΑΤΑΛΟΓΟΣ ΣΧΗΜΑΤΩΝ','## ΚΑΤΑΛΟΓΟΣ ΕΙΚΟΝΩΝ','## Πινάκας Εικόνων','## ΠΙΝΑΚΑΣ ΕΙΚΟΝΟΓΡΑΦΗΣΗΣ','## ΚΑΤΑΛΟΓΟΣ ΠΙΝΑΚΩΝ',
'## Κατάλογος Εικόνων','## Κατάλογος Σχημάτων', '## Πίνακας Διαγραμμάτων', '## ΚΑΤΑΛΟΓΟΣ ΣΧΗΜΑΤΩΝ', '## ΚΑΤΑΛΟΓΟΣ ΧΑΡΤΩΝ','## ΚΑΤΑΛΟΓΟΣ ΔΙΑΓΡΑΜΜΑΤΩΝ','## ΚΑΤΑΛΟΓΟΣ ΦΩΤΟΓΡΑΦΙΩΝ')

# END TAGS, ie tags that end a context
CONTENT_and_CATALOG_end_tags = ('## Συντομογραφίες', '## Εισαγωγή', '## Κεφάλαιο 1', '## Περίληψη', '## ΠΕΡΙΛΗΨΗ','## ΚΕΦΑΛΑΙΟ 1: ΕΙΣΑΓΩΓΗ', '##  ΚΕΦΑΛΑΙΟ 1 ο', 'Κεφάλαιο Πρώτο', '## ΚΕΦΑΛΑΙΟ 1ο','## ΓΕΝΙΚΟ ΜΕΡΟΣ', '## Εισαγωγή',
'## ΕΙΣΑΓΩΓΗ', '## ΕΙΣΑΓΩΓΗ:', '## ΠΡΟΛΟΓΟΣ', '## ΠΙΝΑΚΑΣ ΣΥΝΤΜΗΣΕΩΝ', '## Συντμήσεις:')

diofantos_pattern = re.compile(r'IΤΥΕ - ΔΙΟΦΑΝΤΟΣ|«ΔΙΟΦΑΝΤΟΣ|«Διόφαντος»|«ΔΙΟΦΑΝΤΟΣ»')
writer_pattern = re.compile(r'ΣΥΓΓΡΑΦΕΑΣ|Το παρόν εκπονήθηκε αμισθί και εγκρίθηκε|ΚΡΙΤΕΣ-ΑΞΙΟΛΟΓΗΤΕΣ|Σύμβουλος Παιδαγωγικού Ινστιτούτου.|Πρόεδρος:|ΣΥΝΤΟΝΙΣΜΟΣ:|Η συγγραφή και η επιστηµονική επιµέλεια του βιβλίου|Παροράματα βιβλίων Μηχανολογικού Τομέα')
religious_schoolbook_pattern = re.compile(r'Α\) Την εισήγηση για την πλήρη συμμόρφωση των Προγραμμάτων Σπουδών')
legal_note_3966_2011_pattern = re.compile(r'ν. 3966/2011')
noise_pattern = re.compile(r'$(.*?)$|\\_(\\_)+|Ú|È|Ì||Ô|È|Ù|Ë|Á|Ï|Ò|Û|·|Ó|Â|Î|‹|ˆ|∂ ¡ √ Δ ∏ Δ ∞|<non-compliant-utf8-text>')
funding_pattern = re.compile(r'συγχρηματοδοτείται από την Ελλάδα και την Ευρωπαϊκή Ένωση|συγχρηµατοδοτείται από την Ελλάδα και την Ευρωπαϊκή Ένωση|Έργο συγχρηματοδοτούμενο |Έργο συγχρηµατοδοτούµενο')
epal_pattern = re.compile(r'[ΑΓΒAB](\s)*\' ΕΠΑ\.Λ\.|B\'ΕΠΑ\.Λ\.|## Γ\'ΕΠΑ\.Λ\.')
gymnasium_pattern = re.compile(r'[ΑΒΓAB](\s)*\' Γυμνασίου')
exofilo_pattern = re.compile(r'Οι 3 εικόνες με φωτοαπόδοση που χρησιμοποιήθηκαν στο εξώφυλλο|Επεξήγηση του εξωφύλλου:|Στο κέντρο: Μαρσέλ|Αφοί ΤΖΙΦΑ Α.Ε.Β.Ε.')
remove_title_number_pattern = re.compile(r'## \d{1,2}.|## \d{1,2}\.\d{1,2}|## \d{1,2}\.\d{1,2}\.d{1,2}\|## \d{1,2}|## I+\.|## IV\.|## Vi+\.|## [Α-Ω]\.') 