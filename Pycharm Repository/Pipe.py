import urllib.request
from bs4 import BeautifulSoup
import spacy
import neuralcoref
import os
import glob
from pathlib import Path
nlp = spacy.load('en_core_web_sm')

neuralcoref.add_to_pipe(nlp)

'''The following code is for if you want to directly read a url's text:'''
#html = urllib.request.urlopen('example URL here')
#soup = BeautifulSoup(html, 'html.parser')
#text = ''.join([t for t in soup.find_all(text=True) if t.parent.name == 'p' and len(t) >= 25])

''' If you want to read the text from a file however:'''
os.listdir('...\terrorismdataset')
files = Path('...\terrorismdataset').glob('*.txt')
text = list()

for file in files:
    with open(file, 'rb') as f:
        text.append(file.read_text())

doc = nlp(text)
resolved_text = doc._.coref_resolved
print(text)
print('************************************************************************************')
#print(resolved_text)
#print(doc._.coref_clusters)

for ref in doc._.coref_clusters:
    print(ref)
print('************************************************************************************')
sentences = [sent.string.strip() for sent in nlp(resolved_text).sents]
output = [sent for sent in sentences if 'diamond' in
          (' '.join([token.lemma_.lower() for token in nlp(sent)]))]
print('Fact count:', len(output))
for fact in range(len(output)):
    print(str(fact+1)+'.', output[fact])
