import re  
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2")

class Anonymizer: #handles both email and phone masking in a single loop.
    def __init__(self):
        self.patterns={
            "EMAIL":r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "PHONE":r'\b[0-9]{10}\b'
        }
        self.labels={
            "EMAIL":"[EMAIL]",
            "PHONE":"[PHONE]"
        }
        
    def mask(self,text):
        for label,pattern in self.patterns.items():
            def replace(match,label=label):
                return self.labels[label] 
            text=re.sub(pattern,replace,text) 
        return text 



class NER_masker:
    def __init__(self):
        self.labels={
            "PERSON":"[PER]",
            "ORGANIZATION":"[ORG]",
            "LOCATION":"[LOC]"
        }
    
    def mask(self,text):
        entity=ner_model.predict_entities(text,self.labels)
        spans=[]
        for e in entity:
            spans.append((e['text'],e['label']))

        spans.reverse()
        for tup in spans:
            word,label=tup
            text=text.replace(word,self.labels[label])
        return text 
    

