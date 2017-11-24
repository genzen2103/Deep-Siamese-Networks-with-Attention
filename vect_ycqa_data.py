import numpy as np 
from nltk.tokenize import word_tokenize
import unidecode

vocab = { key:int(value) for key,value in [ row.split(":") for row in open('../Vocabs/Yahoo_CQA_vocab.txt','r')] }

def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv

max_files=3119
block=[]
block_num=0
rct=0


for fct in range(max_files):
	
	print ((fct/float(max_files))*100)
	
	qf = open('../Datasets/Yahoo_CQA_Data/yahoo_answer/C'+str(fct)+'Question.dat')
	af = open('../Datasets/Yahoo_CQA_Data/yahoo_answer/C'+str(fct)+'Answer.dat')
	
	for qrow in qf:
		
		q_v = np.zeros(len(vocab))
		ans_v = np.zeros(len(vocab))
		
		qrow=unidecode.unidecode(str(qrow))
		row_list = qrow.strip().split('\t')
		row_list = row_list[1:]
		for entry in row_list:
			tokens = word_tokenize(entry)
			for t in tokens:
				if t.isalpha():
					str_hash= hash_string(t)
					for h in str_hash:
						if h in vocab:
							q_v[vocab[h]]+=1

		arow = af.readline()
		arow=unidecode.unidecode(str(arow))
		ans_list=arow.strip().split("|`|")
		for ua_pair in ans_list:
			if len(ua_pair)>0 and "\t" in ua_pair:
				user,ans = ua_pair.split("\t")
				ans=ans.replace(".",'')
				tokens = word_tokenize(ans)
				for t in tokens:
					if t.isalpha():
						str_hash= hash_string(t)
						for h in str_hash:
							if h in vocab:
								ans_v[vocab[h]]+=1	

		block.append(np.array([q_v,ans_v,"1"]))
		rct+=1
		if rct==1000:
			np.save('../Vector_Sets/Yahoo_CQA_Data/B'+str(block_num)+'.npy', np.array(block))
			print(block_num)
			rct=0
			del block
			block=[]
			block_num+=1
	
	qf.close()
	af.close()