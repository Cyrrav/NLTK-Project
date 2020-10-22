import os
import glob
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm


files_path = 'J:\\AI\\annotated data'

TotalP = 0
TotalGPE = 0
TotalORG = 0

for filename in os.listdir(files_path):
    fname, fextension = os.path.splitext(filename)
    if (fextension == '.txt'):
        file = open(files_path + '\\' + filename, encoding='utf-8')
        rawFile = file.read() #reads everything in the folder that ends in .txt

    def preprocess(sent):  #Pre-processing
        sent = nltk.word_tokenize(sent)
        sent = nltk.pos_tag(sent)
        return sent
    sent = preprocess(rawFile)
    #print(sent) #pre-processing test print

    pattern = 'NP: {<DT>?<JJ>*<NN>}'  #Noun-Phrase chunking to identify named entities
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    #print(cs) #parser test print
    iob_tagged = tree2conlltags(cs)
    #pprint(iob_tagged) #iob_tag test print

    ne_tree = ne_chunk(pos_tag(word_tokenize(rawFile)))
    #print(ne_tree) #ne_Tree test print
    nlp = en_core_web_sm.load()
    doc = nlp(rawFile)
    pprint([(X.text, X.label_) for X in doc.ents])  #print entire thing as tuplets with tags
    #pprint([(X, X.ent_iob_, X.ent_type_) for X in doc]) #print entities

    labels = [x.label_ for x in doc.ents] #all labels
    #print(Counter(labels)) #count all labels per text file (NOT OVERALL)
    PERSON = len([ent for ent in doc.ents if ent.label_ == 'PERSON']) #count how many [PERSON] tags
    print('PERSON: ', PERSON)
    TotalP += PERSON
    ORGANIZATION = len([ent for ent in doc.ents if ent.label_ == 'ORG']) #count how many [ORGANIZATION] tags
    print('ORGANIZATION: ', ORGANIZATION)
    TotalORG += ORGANIZATION
    GPE = len([ent for ent in doc.ents if ent.label_ == 'GPE']) #count how many [GPE] tags
    print('GPE: ', GPE)
    TotalGPE += GPE

print('Total PERSON tags: ', TotalP)
print('Total ORGANIZATION tags: ', TotalORG)
print('Total GPE tags: ', TotalGPE)


