import pandas
import sys
import re
import os

def remove_simiosi(text) :
    simiosi_pattern = re.compile(r'Σημείωση|Note|Ο τονισμός|The translator')
    if re.match(simiosi_pattern,text) :
        return True
    return False

def end_of_text(text) :
    end_pattern = re.compile(r'ΤΕΛΟΣ|Α Υ Λ Α Ι Α|Τ Ε Λ Ο Σ')
    if re.match(end_pattern,text) :
        return True
    return False
    
def fexi_intro(text) :
    fexi_pattern = re.compile(r'ΦΕΞΗ|ΒΙΒΛΙΟΘΗΚΗ|ΒΙΒΛΙΟΠΩΛΕΙΟΝ')
    if re.search(fexi_pattern,text) :
        return True
    return False

def exception_to_fexi_intro(text) :
    exception_to_fexi_intro_pattern = re.compile(r'ΕΚΔΟΤΙΚΟΣ ΟΙΚΟΣ ΓΕΩΡΓΙΟΥ Δ. ΦΕΞΗ')
    if re.match(exception_to_fexi_intro_pattern,text) :
        return True
    return False

def end_fexi_intro(text) :
    fexi_end_intro = re.compile(r'ΕΝ ΑΘΗΝΑΙΣ|19[0-9][0-9]|ΕΙΣΑΓΩΓΗ|ΠΡΟΛΟΓΟΣ|ΠΡΟΣΩΠΑ|ΑΡΙΣΤΟΤΕΛΟΥΣ|ΕΚΔΟΤΙΚΟΣ ΟΙΚΟΣ ΓΕΩΡΓΙΟΥ ΦΕΞΗ|ΕΙΣΗΓΗΣΙΣ')
    if re.search(fexi_end_intro,text) :
        return True
    return False

def content(text) :
    content_pattern = re.compile(r'ΠΕΡΙΕΧΟΜΕΝΑ|ΕΜΠΕΡΙΕΧΟΜΕΝΑ|ΠΙΝΑΚΑΣ ΠΕΡΙΕΧΟΜΕΝΩΝ|Π Ι Ν Α Κ Α Σ   Π Ε Ρ Ι Ε Χ Ο Μ Ε Ν Ω Ν')
    if re.match(content_pattern,text) :
        return True
    return False

def remove_latin_text(text) :
    no_greek_pattern = re.compile(r'([Α-Ω]+)|([α-ω]+)', re.UNICODE)
    lines = text.splitlines()
    newtext = ''
    simiosi_flag = False
    content_flag = False
    content_begin_flag = False
    end_of_text_flag = False
    fexi_intro_flag = False
    for line in lines :
        if line == '' :
            content_flag = False
            simiosi_flag = False
            newtext = newtext+line+'\n'
            continue
        if content_begin_flag == True :
            content_flag = True
        if fexi_intro(line) :
            fexi_intro_flag = True
            if exception_to_fexi_intro(line) :
                fexi_intro_flag = False
                newtext = newtext+'[Out]'+line+'\n'
                continue
        if fexi_intro_flag == True :
            if end_fexi_intro(line) :
                fexi_intro_flag = False
                newtext = newtext+'[Out]'+line+'\n'
                continue
        #if end_of_text(line) :
            #end_of_text_flag = True
        if remove_simiosi(line) :
            simiosi_flag = True
        if content(line) :
            content_flag = True
            content_begin_flag = True
            newtext = newtext+'[Out]'+line+'\n'
            continue
        if simiosi_flag == True :
            newtext = newtext+'[Out]'+line+'\n'
            continue
        if end_of_text_flag == True :
            newtext = newtext+'[Out]'+line+'\n'
            continue
        if fexi_intro_flag == True :
            newtext = newtext+'[Out]'+line+'\n'
            continue
        if re.search(no_greek_pattern,line) :
            content_begin_flag = False
            if content_flag == True :
                newtext = newtext+'[Out]'+line+'\n'
                continue
            newtext = newtext + line + '\n'
        else :
            newtext = newtext+'[Out]'+line+'\n'
    return newtext 

def clean(pathout,pathin) :
    os.makedirs(pathout, exist_ok=True)
    for i,file in enumerate(os.listdir(pathin)) :
        if not file.endswith('.txt'):
            continue
        try:
            with open(pathin+'/'+file, 'r', encoding='utf-8') as infile:
                text = infile.read()
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue
        text = remove_latin_text(text)
        output_file_path = os.path.join(pathout, 'text_'+str(i)+'.txt')
        try:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
        except Exception as e:
            print(f"Error writing to {output_file_path}: {e}")
            continue
    

if __name__ == '__main__' :
    if len(sys.argv) < 3 :
        print("Please provide two paths, first for the input and second for the output")
        exit()
    clean(sys.argv[2],sys.argv[1])