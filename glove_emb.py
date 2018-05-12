import sys
import os 
import pandas as pd
import numpy as np


def ensureUtf(s, encoding='utf8'):
  if type(s) == bytes:
  	return s.decode(encoding, 'ignore')
  else:
  	return s

		
df = pd.read_csv("../../Datasets/Quora_Duplicate_Questions.tsv",delimiter='\t')
 
# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(ensureUtf(x)))
df['question2'] = df['question2'].apply(lambda x: unicode(ensureUtf(x)))


# exctract word2vec vectors
import spacy
nlp = spacy.load('en_vectors_web_lg')

avg_words_p = 20
avg_words_q = 20

for k in xrange(len(df['question1'])):
	print(k)
	data1 = np.array([d.vector for d in nlp(df['question1'][k]) if d.has_vector ])
	if data1.shape[0]>0 and data1.shape[0]<avg_words_p:
		data1 = np.pad(data1,( (0,avg_words_p-data1.shape[0]),(0,0) ),'constant')
	elif data1.shape[0] == 0:
		data1=np.zeros((avg_words_p,300))
	data1 = data1[0:avg_words_p]
	data2 = np.array([d.vector for d in nlp(df['question2'][k]) if d.has_vector ])
	if data2.shape[0]>0 and data2.shape[0]<avg_words_q:
		data2 = np.pad(data2,( (0,avg_words_q-data2.shape[0]),(0,0) ),'constant')
	elif data2.shape[0] == 0:
		data2=np.zeros((avg_words_q,300))
	data2 = data2[0:avg_words_q]
	np.save('../../Vector_Sets/Quora_Glove_Embs/GLOVE'+str(k)+'.npy', np.array([data1,data2]))

