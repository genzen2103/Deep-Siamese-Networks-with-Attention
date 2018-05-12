from nltk.tokenize import word_tokenize
import re
import string 
import numpy as np 
import unidecode
import codecs

vocab = { key:int(value) for key,value in [ row.split(":") for row in open('../../Vocabs/Quora_vocab.txt','r')] }
print(len(vocab))

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv



rct=0

qf = codecs.open("../../Datasets/Quora_Duplicate_Questions.tsv", 'r', 'utf-8')
qf.readline()
for row in qf:
	row=unidecode.unidecode(row)
	qv1 = np.zeros(len(vocab))
	qv2 = np.zeros(len(vocab))
	a,b,c,d,e,f = row.strip().split('\t')

	try:
		f = int(f)
	except:
		print f,row
	tokens = word_tokenize(d)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if h in vocab:
					qv1[vocab[h]]+=1

	tokens = word_tokenize(e)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if h in vocab:
					qv2[vocab[h]]+=1

	data = np.array([qv1,qv2,f])
	np.save('../../Vector_Sets/Quora_Single_files/F'+str(rct)+'.npy',data )		
	rct+=1
	print(rct)
qf.close()
