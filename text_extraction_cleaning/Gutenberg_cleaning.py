import pandas
import sys
import re

def remove_simiosi(text) :
    simiosi_pattern = re.compile(r'Σημείωση|Note')
    if re.match(simiosi_pattern,text) :
        return True

def remove_latin_text(text) :
    no_greek_pattern = re.compile(r'([Α-Ω]+)|([α-ω]+)', re.UNICODE)
    lines = text.splitlines()
    newtext = ''
    simiosi_flag = False
    for line in lines :
        if line == '' :
            simiosi_flag = False
            continue
        if remove_simiosi(line) :
            simiosi_flag = True
        if simiosi_flag == True :
            continue
        if re.search(no_greek_pattern,line) :
            newtext = newtext + line + '\n'
    return newtext 

def clean(pathout,pathin) :
    df = pandas.read_csv(pathin)
    for i,row in enumerate(df.iterrows()) :
        df['text'][i] = remove_latin_text(df['text'][i])
    df.to_csv(pathout)
    print(df)
    

if __name__ == '__main__' :
    if len(sys.argv) < 3 :
        print("Please provide two paths, first for the input and second for the output")
        exit()
    clean(sys.argv[2],sys.argv[1])