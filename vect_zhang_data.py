import numpy as np 
from nltk.tokenize import word_tokenize
import unidecode

vocab = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/combined_vocab.txt','r')] }

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv


block=[]
block_num=0
rct=0

qf = open('../Datasets/Zhang_yahoo.data')


for row in qf:
	row=unidecode.unidecode(str(row))
	qv1 = np.zeros(len(vocab))
	qv2 = np.zeros(len(vocab))
	q1,q2,l,temp = row.strip().split('\t')
	
	tokens = word_tokenize(q1)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if h in vocab:
					qv1[vocab[h]]+=1
	
	tokens = word_tokenize(q2)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if h in vocab:
					qv2[vocab[h]]+=1

	block.append(np.array([qv1,qv2,l]))
	rct+=1
	if rct==1000:
		np.save('../Vector_Sets/Zhang_yahoo/B'+str(block_num)+'.npy', np.array(block))
		print(block_num)
		rct=0
		del block
		block=[]
		block_num+=1

	
