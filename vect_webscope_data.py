import numpy as np 
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET

vocab = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/combined_vocab.txt','r')] }

data_file_name = '../Datasets/Yahoo_Webscope_L5/manner-v2.0/manner.xml'

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv



tree = ET.parse(data_file_name)
root = tree.getroot()
print(len(vocab))
block=[]
block_num=0
rct=0

for vess in root:
	q_vect,ans_vect = np.zeros(len(vocab)),np.zeros(len(vocab))
	flag=0
	for entry in vess[0]:
		print entry.tag
		if entry.tag in [ 'subject','content','cat','maincat','subcat'] :
			tokens = word_tokenize(entry.text)
			for t in tokens:
				if t.isalpha():
					str_hash= hash_string(t)
					for h in str_hash:
						if h in vocab:
							q_vect[vocab[h]]+=1
		elif entry.tag in [ 'bestanswer' ] :
			flag+=1
			tokens = word_tokenize(entry.text)
			for t in tokens:
				if t.isalpha():
					str_hash= hash_string(t)
					for h in str_hash:
						if h in vocab:
							ans_vect[vocab[h]]+=1
	if flag:
		block.append(np.array([q_vect,ans_vect,"1"]))
		rct+=1
		if rct==1000:
			np.save('../Vector_Sets/Yahoo_Webscope_L5/B'+str(block_num)+'.npy', np.array(block))
			print(block_num)
			rct=0
			del block
			block=[]
			block_num+=1



