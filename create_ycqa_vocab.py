from nltk.tokenize import word_tokenize
import re
import string 
import unidecode
vocab={}
wct=0
max_files=3119



def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv

for fct in range(max_files):
	print ((fct/float(max_files))*100)
	qf = open('../Datasets/Yahoo_CQA_Data/yahoo_answer/C'+str(fct)+'Question.dat')
	for row in qf:
		row=unidecode.unidecode(str(row))
		row_list = row.strip().split('\t')
		row_list = row_list[1:]
		for entry in row_list:
			tokens = word_tokenize(entry)
			for t in tokens:
				if t.isalpha():
					str_hash= hash_string(t)
					for h in str_hash:
						if not h in vocab:
							vocab.update({h:wct})
							wct+=1
	qf.close()

	af = open('../Datasets/Yahoo_CQA_Data/yahoo_answer/C'+str(fct)+'Answer.dat')
	for row in af:
		row=unidecode.unidecode(str(row))
		ans_list=row.strip().split("|`|")
		for ua_pair in ans_list:
			if len(ua_pair)>0 and "\t" in ua_pair:
				user,ans = ua_pair.split("\t")
				ans=ans.replace(".",'')
				tokens = word_tokenize(ans)
				for t in tokens:
					if t.isalpha():
						str_hash= hash_string(t)
						for h in str_hash:
							if not h in vocab:
								vocab.update({h:wct})
								wct+=1
	af.close()

file=open("../Vocabs/Yahoo_CQA_vocab.txt",'w')
file.write("\n".join([ ":".join([str(key),str(vocab[key])]) for key in sorted(vocab.keys()) ]))
print ("Vocab Length:",len(vocab))