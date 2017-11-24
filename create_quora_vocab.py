from nltk.tokenize import word_tokenize
import re
import string 
import unidecode
vocab={}
wct=0
fct=0



def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv



qf = open('../Datasets/Quora_Duplicate_Questions.tsv')
for row in qf:
	print(fct)
	fct+=1
	row=unidecode.unidecode(str(row))
	a,b,c,d,e,f = row.strip().split('\t')
	tokens = word_tokenize(d)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if not h in vocab:
					vocab.update({h:wct})
					wct+=1
	tokens = word_tokenize(e)
	for t in tokens:
		if t.isalpha():
			str_hash= hash_string(t)
			for h in str_hash:
				if not h in vocab:
					vocab.update({h:wct})
					wct+=1
qf.close()


file=open("../Vocabs/Quora_vocab.txt",'w')
file.write("\n".join([ ":".join([str(key),str(vocab[key])]) for key in sorted(vocab.keys()) ]))
print ("Vocab Length:",len(vocab))