import pandas
import sys
import re
import os
import time

simiosi_pattern = re.compile(r'Σημείωση|Note|Ο τονισμός|The translator|Στην παρούσα ηλεκτρονική μεταγραφή διατηρηθήκαν|Ο πίνακας περιεχομένων|Η έκδοση είχε ατέλειες|Τίτλος:')
end_pattern = re.compile(r'(( )*ΤΕΛΟΣ.?)$|Α Υ Λ Α Ι Α|Τ Ε Λ Ο Σ|Η Σειρά των Αρχαίων Ελλήνων Συγγραφέων|Η Σειρά των Αρχαίων Ελλήνων Συγγραφεων|Ο Εκπαιδευτικός Όμιλος|ΠΑΡΟΡΑΜΑΤΑ|ΥΠΟΣΗΜΕΙΩΣΕΙΣ|ΕΚΛΕΚΤΑ ΕΡΓΑ|ΧΡΗΜΑΤΙΣΤΗΡΙΟΝ ΤΟΥ ΒΙΒΛΙΟΥ|ΔIΟΡΘΩΜΑΤΑ|ΤΥΠΟΓΡΑΦΕΙΟΝ "ΕΣΤΙΑ|Έργα του Ιδίου|(.?ΣΗΜΕΙΩΣΕΙΣ.?)$|ΠΡΟΠΟΜΠΟΙ|(\*\*)$|ΤΟΥ ΑΥΤΟΥ ΣΥΓΓΡΑΦΕΩΣ|( )*ΕΝ ΤΩ ΒΙΒΛΙΟΠΩΛΕΙΩ ΤΗΣ ΕΣΤΙΑΣ|1. &Η ΜΙΝΙΟΝ& του &Γκαίτε&. Το χαριτωμένο επεισόδιο της|   ΕΙΣΑΓΩΓΗ ΡΩΜΑΙΟΚΡΑΤΙΑ|ΣΗΜΕΙΩΣΙΣ ΤΟΥ ΠΟΙΗΤΟΥ|ΒΡΙΣΚΟΝΤΑΙ ΜΕΣΑ|\{Διαφημήσεις\}')
fexi_pattern = re.compile(r'ΦΕΞΗ|ΒΙΒΛΙΟΘΗΚΗ|ΒΙΒΛΙΟΠΩΛΕΙΟΝ|^ΕΚΔΟΤΗΣ|ΕΝ ΑΘΗΝΑΙΣ|ΤΥΠΟΓΡΑΦΕΙΟΥ|ΒΙΒΛΙΟΓΡΑΦΙΑ|ΠΙΝΑΞ ΤΩΝ ΕΙΚΟΝΩΝ|ΓΡΑΦΕΙΑ: Οδός Μιλτιάδου αριθ: 1.|ΣΥΝΔΡΟΜΑΙ: Εσωτερικού ετησία δραχ. 60. — εξάμηνος δραχ. 30. —|           Εξωτερικού ετησία φράγ. 60. — εξάμηνος φράγ. 30. —')
exception_to_fexi_intro_pattern = re.compile(r'ΕΚΔΟΤΙΚΟΣ ΟΙΚΟΣ ΓΕΩΡΓΙΟΥ Δ. ΦΕΞΗ')
unique_line_remove_pattern = re.compile(r'ΒΑΣΙΛΙΚΟΝ ΤΥΠΟΓΡΑΦΕΙΟΝ|Ο κυρ-ΜΑΝΩΛΑΚΗΣ — ΒΑΡΥΧΕΙΜΩΝΙΑ — ΧΡΙΣΤΟΣ ΒΟΣΚΡΕΣ — ΣΕ ΜΙΑ|ΜΕΤΑΦΡΑΣΗ|Π. ΓΡΑΤΣΙΑΤΟΥ|Αθήνα, Τυπογραφείο «Εστία» Κ. Μάισνερ και Ν. Καργαδούρη — 10805.|Εν Αθήναις εκ του Τυπογραφείου της Εστίας 1896-1333.|Τιμάται  δρ.  2.50|Τυπογραφείο της Εστίας, Κ. Μάισνερ και Ν. Καργαδούρη — 1790')
fexi_end_intro = re.compile(r'ΕΝ ΑΘΗΝΑΙΣ|(1[8-9][0-9][0-9].?)$|ΕΙΣΑΓΩΓΗ|ΠΡΟΛΟΓΟΣ|ΠΡΟΣΩΠΑ|ΤΑ ΦΟΡΕΜΑΤΑ|ΕΚΔΟΤΙΚΟΣ ΟΙΚΟΣ ΓΕΩΡΓΙΟΥ ΦΕΞΗ|ΕΙΣΗΓΗΣΙΣ|ΒΙΒΛΙΟΝ|ΤΟΜΟΣ|ΕΚΔΟΤΗΣ|ΕΚΔΟΣΕΙΣ ΦΕΞΗ|ΚΕΦΑΛΑΙΟΝ|Κεφάλαιον|Α\'\.|ΚΕΦΑΛΑΙΟ I.|(\(1[8-9][0-9][0-9]\))$|Η ΥΠΟΘΕΣΙΣ ΤΟΥ ΔΡΑΜΑΤΟΣ|ΠΡΟΛΕΓΟΜΕΝΑ|ΠΡΑΞΙΣ|ΒΙΒΛΙΟ ΠΡΩΤΟ.|Τι είναι βουδδισμός;|PREFACE|ΑΡΙΣΤΟΤΕΛΗΣ ΒΑΛΑΩΡΙΤΗΣ|ΓΙΑ ΤΟΥΤΟ ΤΟ ΒΙΒΛΙΟ|ΠΑΡΑΜΥΘΙ ΧΩΡΙΣ ΟΝΟΜΑ')
content_pattern = re.compile(r'ΠΕΡΙΕΧΟΜΕΝΑ|ΕΜΠΕΡΙΕΧΟΜΕΝΑ|ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ|Π Ι Ν Α Κ Α Σ   Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Ω Ν|Π Ρ Ο Σ Ω Π Α|ΠΡΟΣΩΠΑ|Π Ι Ν Α Κ Α Σ  Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Ω Ν|ΤΑ ΤΗΣ ΤΡΑΓΩΔΙΑΣ ΠΡΟΣΩΠΑ|ΠΕΡΙΕΧΟΜΕΝA|ΠΑΡΑΡΤΗΜΑ|ΠΙΝΑΚΑΣ|( )*ΟΙ ΠΑΡΑΔΑΡΜΕΝΟΙ|Τ Α Π Ρ Ο Σ Ω Π Α Τ Ο Υ Δ Ρ Α Μ Α Τ Ο Σ|ΠΡΟΣΩΠΑ ΤΟΥ ΔΡΑΜΑΤΟΣ|Τα πρόσωπα της τραγωδίας|ΤΑ ΠΡΟΣΩΠΑ ΤΟΥ ΔΡΑΜΑΤΟΣ|ΤΟ \[\?\?\?\?\?\} ΒΙΒΛΙΟΝ ΕΤΥ\[\?\?\?\?\]|ΕΚΔΟΣΕΙΣ|Ιφιγένεια|ΕΡΓΑ ΤΟΥ ΙΔΙΟΥ|ΠΙΝΑΞ ΤΩΝ ΠΕΡΙΕΧΟΜΕΝΩΝ|&Η Ευρώπη κατά τον 19 αιώνα&|&ΠΕΡΙΕΧΟΜΕΝΑ :|(ΦΡΕΣΚΑ ΧΡΟΝΙΑ)$|ΚΕΝΤΡΙΚΗ ΠΩΛΗΣΙΣ')
re_clean_end_pattern = re.compile(r'ΕΚΛΕΚΤΑ ΕΡΓΑ|\*\*\*|\* \* \*|ΠΙΝΑΚΑΣ|ΤΕΥΧΗ ΕΚΔΟΘΕΝΤΑ|Σ Η Μ Ε I Ω Σ Ε Ι Σ|ΠΡΟΠΟΜΠΟΙ|Δ Ι Ο Ρ Θ Ω Σ Ε Ι Σ|_Πίναξ|ΤΕΛΟΣ ΤΟΥ ΠΡΩΤΟΥ ΤΟΜΟΥ|ΠΙΝΑΞ|ΝΤΟΠΙΕΣ ΖΩΓΡΑΦΙΕΣ|ΤΕΛΟΣ|.?1[\)\}\]]|ΠΕΡΙΕΧΟΜΕΝΑ|ΑΝΕΣΤΗ ΚΩΝΣΤΑΝΤΙΝΙΔΟΥ')
out_pattern = re.compile(r'\[Out')

no_greek_pattern = re.compile(r'([Α-Ω]+)|([α-ω]+)', re.UNICODE)
#end_note_pattern = re.compile(r'(1)|1)|1}|1}|[1]|1]')

def remove_simiosi(text) :
    if re.match(simiosi_pattern,text) :
        return True
    return False

def remove_end_re_clean(text) : 
    if re.match(re_clean_end_pattern,text) :
        return True
    return False

def remove_unique_line(text) :
    if re.match(unique_line_remove_pattern,text) :
        return True
    return False

def end_of_text(text) :
    if re.match(end_pattern,text) :
        return True
    return False
    
def fexi_intro(text) :
    if re.search(fexi_pattern,text) :
        return True
    return False

def exception_to_fexi_intro(text) :
    if re.match(exception_to_fexi_intro_pattern,text) :
        return True
    return False

def end_fexi_intro(text) :
    if re.match(fexi_end_intro,text) :
        return True
    return False

def content(text) :
    if re.match(content_pattern,text) :
        return True
    return False

def remove_latin_text(text) :
    lines = text.splitlines()
    newtext = ''
    simiosi_flag = False
    content_flag = False
    content_begin_flag = False
    end_of_text_flag = False
    fexi_intro_flag = True
    extra_white_line = False
    for i,line in enumerate(lines) :
        if line == '' :
            if extra_white_line == True :
                extra_white_line = False
                continue
            content_flag = False
            simiosi_flag = False
            newtext = newtext+line+'\n'
            continue
        else :
            extra_white_line = True
        if content_begin_flag == True :
            content_flag = True
        if fexi_intro(line) :
            fexi_intro_flag = True
            if exception_to_fexi_intro(line) :
                fexi_intro_flag = False
                newtext = newtext+'[Out:Exception to beginning of Introduction]'+line+'\n'
                continue
        if fexi_intro_flag == True :
            if end_fexi_intro(line) :
                fexi_intro_flag = False
                newtext = newtext+'[Out:End of Introduction]'+line+'\n'
                continue
        if i > len(lines)/2 :
            if end_of_text(line) :
                end_of_text_flag = True
        if remove_simiosi(line) :
            simiosi_flag = True
        if content(line) :
            content_flag = True
            content_begin_flag = True
            newtext = newtext+'[Out:Beginning of content(chunk of text)]'+line+'\n'
            continue
        if remove_unique_line(line) :
            newtext = newtext+'[Out:Unique line out]'+line+'\n'
            continue
        if simiosi_flag == True :
            newtext = newtext+'[Out:Simiosis]'+line+'\n'
            continue
        if end_of_text_flag == True :
            newtext = newtext+'[Out:End of text]'+line+'\n'
            continue
        if fexi_intro_flag == True :
            newtext = newtext+'[Out:Part of introduction]'+line+'\n'
            continue
        content_begin_flag = False
        if content_flag == True :
            newtext = newtext+'[Out:Part of content(chunck of text)]'+line+'\n'
            continue
        if i < 105 or i > len(lines) - 105:
            if re.search(no_greek_pattern,line) :
                newtext = newtext + line + '\n' 
            else :
                newtext = newtext+'[Out:Non greek in the first or last 105 lines]'+line+'\n'
        else :
            newtext = newtext + line + '\n'
    return newtext 

def re_remove(text,num_of_file) :
    newtext = ''
    lines = text.splitlines()
    non_capital_pattern = re.compile(r'([α-ω][α-ω])+')
    re_clean_end_found_flag = False
    flag_prob_fail = False
    for i,line in enumerate(lines) :
        if re.match(out_pattern,line) :
            #if i > len(lines)/4 and i < len(lines)/3 and flag_prob_fail == False :
                #print('Probable Failure at ',num_of_file)
                #flag_prob_fail = True
            continue
        if i > 105 and i < len(lines) - 300 : 
            newtext = newtext+line+'\n'
            continue
        if i > len(lines) - 300 :
            if remove_end_re_clean(line) :
                re_clean_end_found_flag = True
        if line == '' :
            newtext = newtext+line+'\n'
            continue
        if re_clean_end_found_flag == True : 
            #newtext = newtext+'[Out:End of text, Second clean]'+line+'\n'
            continue
        if i < 105 : 
            if not re.search(non_capital_pattern,line) :
                #newtext = newtext+'[Out:Pures Capitals/No greek characters in first 105 lines]'+line+'\n'
                continue
        newtext = newtext+line+'\n'
    return newtext

def remove_extras(text) :
    whitespace_pattern = re.compile(r'.*?[Α-ΩΆ-Ώ\[]',re.DOTALL)
    temp = re.match(whitespace_pattern, text)
    if temp :
        temp = temp.group(0)
        text = re.sub(whitespace_pattern, ''+temp[-2:],text,1)
    return text

def remove_publisher_note(text) :
    ekdotis_pattern = re.compile(r'.+(Ο ΕΚΔΟΤΗΣ|Ο εκδότης( )*\n|Αφιερώ( )*\n)',re.DOTALL)
    if re.match(ekdotis_pattern, text) :
        text = re.sub(ekdotis_pattern, '',text,1)
    return text

def remove_special_char(text) :
    special_char_pattern = re.compile(r'$|_|&|\*')
    text = re.sub(special_char_pattern,'',text)
    return text

def precision_cleaning(text) :
    text = remove_publisher_note(text)
    text = remove_extras(text)
    text = remove_special_char(text)
    return text

def clean(pathout,pathin) :
    os.makedirs(pathout, exist_ok=True)
    for i,file in enumerate(os.listdir(pathin)) :
        start = time.time()
        if not file.endswith('.txt'):
            continue
        try:
            with open(pathin+'/'+file, 'r', encoding='utf-8') as infile:
                text = infile.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        text = remove_latin_text(text)
        text = re_remove(text,file)
        text = precision_cleaning(text)
        output_file_path = os.path.join(pathout, file)
        try:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
        except Exception as e:
            print(f"Error writing to {output_file_path}: {e}")
            continue
        end = time.time()
        print('Finished cleaning :',file," taking ",end-start," seconds")

if __name__ == '__main__' :
    if len(sys.argv) < 3 :
        print("Please provide two paths, first for the input and second for the output")
        exit()
    clean(sys.argv[2],sys.argv[1])