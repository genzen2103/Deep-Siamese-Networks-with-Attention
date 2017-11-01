import numpy as np 
from nltk.tokenize import word_tokenize

vocab={ key.decode('utf-8'):int(value) for key,value in [ row.split(":") for row in open('yahoo_hashed_vocab.dat','rb')] }
max_files=3119

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in xrange(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv


for fct in xrange(max_files):
	print (fct/float(max_files))*100
	qf = open('Yahoo Data/yahoo_answer/C'+str(fct)+'Question.dat')
	qf_vect=np.zeros(len(vocab))
	for row in qf:
		row_list = row.strip().split('\t')
		row_list = row_list[1:]
		for entry in row_list:
			entry=entry.decode("utf-8")
			tokens = word_tokenize(entry)
			for t in tokens:
				if t.isalpha():
					str_hash= hash_string(t)
					for h in str_hash:
						qf_vect[vocab[h]]+=1
	
	af = open('Yahoo Data/yahoo_answer/C'+str(fct)+'Answer.dat')
	ans_vect=np.zeros(len(vocab))
	for row in af:
		ans_list=row.strip().split("|`|")
		for ua_pair in ans_list:
			if len(ua_pair)>0 and "\t" in ua_pair:
				user,ans = ua_pair.split("\t")
				ans=ans.decode("utf-8").replace(".",'')
				tokens = word_tokenize(ans)
				for t in tokens:
					if t.isalpha():
						str_hash= hash_string(t)
						for h in str_hash:
							ans_vect[vocab[h]]+=1
	af.close()
	np.save('full_vect_dataset/C'+str(fct)+'.npy', np.array([qf_vect,ans_vect]))


