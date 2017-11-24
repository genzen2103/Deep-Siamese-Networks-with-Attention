from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
import unidecode
vocab={}
data_file_name = '../Datasets/Yahoo_Webscope_L5/manner-v2.0/manner.xml'
dct=0
index=0


def hash_string(s,n_gram=3):
	s = "#"+s.lower()+"#"
	hv=[]
	for i in range(len(s)-n_gram+1):
		hv.append(s[i:i+n_gram])
	return hv

tree = ET.parse(data_file_name)
root = tree.getroot()

for vess in root:
	dct+=1
	print(dct)
	for doc in vess:
		for entry in doc:
			if entry.tag in [ 'subject','content','cat','maincat','subcat','bestanswer' ] :
				text=unidecode.unidecode(str(entry.text))
				tokens = word_tokenize(text)
				for t in tokens:
					if t.isalpha():
						str_hash= hash_string(t)
						for h in str_hash:
							if not h in vocab:
								vocab.update({h:index})
								index+=1

file=open("../Vocabs/Yahoo_Webscope_L5_vocab.txt",'w')
file.write("\n".join([ ":".join([str(key),str(vocab[key])]) for key in sorted(vocab.keys()) ]))
print ("Vocab Length:",len(vocab))