import sys
import os 
import pandas as pd
import numpy as np


def ensureUtf(s, encoding='utf8'):
  if type(s) == bytes:
  	return s.decode(encoding, 'ignore')
  else:
  	return s

def get_labels(test_ids):
	result={}
	for id in test_ids:
		data = np.load('../../Vector_Sets/Quora_Single_files/F' + id + '.npy')
		result.update({id:data[2]})
	return result

		
df = pd.read_csv("../../Datasets/Quora_Duplicate_Questions.tsv",delimiter='\t')
 
# encode questions to unicode
df['question1'] = df['question1'].apply(lambda x: unicode(ensureUtf(x)))
df['question2'] = df['question2'].apply(lambda x: unicode(ensureUtf(x)))


final={}
file = open('baseline_results.txt','r')
for row in file:
	k,p,v = row.strip().split(',')
	final.update({k:[[p,v]]})


file = open('model1_results.txt','r')
for row in file:
	k,p,v = row.strip().split(',')
	if not k in final:
		final.update({k:[]})
	final[k].append([p,v])

file = open('model2_results.txt','r')
for row in file:
	k,p,v = row.strip().split(',')
	if not k in final:
		final.update({k:[]})
	final[k].append([p,v])

file = open('model3_results.txt','r')
for row in file:
	k,p,v = row.strip().split(',')
	if not k in final:
		final.update({k:[]})
	final[k].append([p,v])


y_test = get_labels(final.keys())


file = open('result.tsv','w')
file.write('P\tQ\tActual Label\tBaseline Prediction\tModel 1 Prediction\tModel 2 Prediction\tModel 3 Prediction')
for k in final:
	if len(final[k])==4:
		i = int(k)
		string = ''
		string+=df['question1'][i].encode('utf8')
		string+='\t'+df['question2'][i].encode('utf8')
		string+='\t'+str(y_test[k])
		string+='\t'+"\t".join([",".join(final[k][i]) for i in xrange(len(final[k]))])
		file.write('\n'+string)
