import pandas as pd
import pickle
import gensim
from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
import pyLDAvis
from pyLDAvis import gensim_models

file1 = open('lda_data_objects.p', 'rb')

lda_objects = pickle.load(file1)

dictionary = lda_objects['dictionary']
dictionary2 = corpora.Dictionary.load('lda_model/lda_model.id2word')
corpus = lda_objects['corpus']

lda_model = gensim.models.ldamodel.LdaModel.load('lda_model/lda_model')

lda_viz = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_viz
