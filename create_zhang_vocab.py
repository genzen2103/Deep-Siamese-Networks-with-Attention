import numpy as np 
from nltk.tokenize import word_tokenize
import unidecode
index=0
vocab={}
fct=0

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv

qf = open('../Datasets/Zhang_yahoo.data')

for row in qf:
	row=unidecode.unidecode(str(row))
	q1,q2,l,temp = row.strip().split('\t')
	print(fct)
	fct+=1
	
	tokens = word_tokenize(q1)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if not h in vocab:
					vocab[h]=index
					index+=1
	
	tokens = word_tokenize(q2)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if not h in vocab:
					vocab[h]=index
					index+=1

	
file=open("../Vocabs/Zhang_yahoo_vocab.txt",'w')
file.write("\n".join([ ":".join([str(key),str(vocab[key])]) for key in sorted(vocab.keys()) ]))
print ("Vocab Length:",len(vocab))
