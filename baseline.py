
from keras.models import Model, Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, Lambda, dot,Dropout,Dot,Concatenate,Reshape,Permute
from keras.optimizers import SGD,Adam
from keras.metrics import cosine_proximity
import numpy as np
from keras import backend as K
from keras.activations import softmax
from keras.models import load_model
from  data_gen import DataGenerator
import math
from keras.models import load_model
import pandas as pd


def ensureUtf(s, encoding='utf8'):
  if type(s) == bytes:
  	return s.decode(encoding, 'ignore')
  else:
  	return s

def show_test_data(test_ids,y_test,pl,preds):
	df = pd.read_csv("../../Datasets/Quora_Duplicate_Questions.tsv",delimiter='\t')
	df['question1'] = df['question1'].apply(lambda x: unicode(ensureUtf(x)))
	df['question2'] = df['question2'].apply(lambda x: unicode(ensureUtf(x)))
	r = []
	for i,id in enumerate(test_ids):
		id = int(id)
		print (df['question1'][id])
		print (df['question2'][id])
		print ("Actual Label: "+str(y_test[i]))
		print ("Predicted Label: "+str(pl[i])+" , "+str(preds[i]))
		r.append(",".join(map(str,[id,pl[i],preds[i]])))
	f = open('baseline_results.txt','w')
	f.write("\n".join(r))
	f.close()

def baseNetwork(inp_dim):
	embeddingSize = 128
	inputShape = (1, inp_dim, 1)
	kernelSizeConv1 = (1,10)
	kernelSizeConv2 = (1,5)
	mPool_PoolSize1 = (1, 100)
	mPool_PoolSize2 = (1, 5)
	mPoolStride1 = (1, 100)
	mPoolStride2 = (1, 4)
	optLayersConv1 = 5 ## No of output layers in conv 1
	optLayersConv2 = 8 ## No of output layers in conv 2
	
	convNet = Sequential()
	
	## Conv layer 1
	convNet.add(Conv2D(optLayersConv1, kernel_size = kernelSizeConv1, input_shape = inputShape))
	convNet.add(MaxPooling2D(pool_size = mPool_PoolSize1, strides = mPoolStride1))
	convNet.add(Activation('relu'))
	
	## Conv layer 2
	convNet.add(Conv2D(optLayersConv2, kernel_size = kernelSizeConv2))
	convNet.add(MaxPooling2D(pool_size = mPool_PoolSize2, strides = mPoolStride2))
	convNet.add(Activation('relu'))
	
	convNet.add(Flatten())
	convNet.add(Dense(embeddingSize)) 
	
	return convNet

def cosine_dist(vects):
	return dot(vects,axes=1,normalize=True)


def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def c_loss(y_true, y_pred):
	margin = 0.40
	return K.mean( y_true * (1-y_pred) + (1 - y_true) * K.maximum(y_pred-margin, 0) )

def get_labels(test_ids):
	result=[]
	for id in test_ids:
		data = np.load('../../Vector_Sets/Quora_Single_files/F' + id + '.npy')
		result.append(int(data[2]))
	return result


if __name__=='__main__':

	batchSize=100
	embeddingSize = 128
	inputDimension = 11522
	total_files = 404290
	glove_emb = 300
	M = 20
	N = 20

	# Model hyper parameters
	noEpochs = 1
	learn_rate = 0.001
	momen = 0.05

	inputP = Input(shape=(1, inputDimension, 1))
	inputQ = Input(shape=(1, inputDimension, 1))
	inputPG =  Input(shape=(M,glove_emb)) 										#1 x m x 300
	inputQG =  Input(shape=(N,glove_emb)) 										#1 x n x 300
	
	baseNw = baseNetwork(inputDimension)
	#baseNw.summary()

	P_rep = baseNw(inputP)  													#1 x 128
	Q_rep = baseNw(inputQ)  													#1 x 128

	distance = Lambda(cosine_dist, output_shape=cos_dist_output_shape)([P_rep, Q_rep])
	model = Model(outputs=distance,inputs=[inputP, inputQ,inputPG,inputQG])
	model.compile(loss = c_loss, optimizer=Adam(lr=learn_rate,decay=0.00005), metrics=['acc'])
	model.summary()

	# Parameters
	params = {'dim': 11522,
			  'batch_size': batchSize,
			  'shuffle': True}

	# Datasets
	full_data_size = 100000

	test_size = int(math.floor(full_data_size*0.001))
	val_size = int(math.floor(full_data_size*0.001))
	train_size = full_data_size - test_size - val_size
	#perm = np.random.permutation(full_data_size)
	perm = range(full_data_size)

	partition = { 	'train':[ str(perm[i]) for i in xrange(train_size) ] , 
					'validation':[ str(perm[i]) for i in xrange(train_size,train_size+val_size) ],
					'test':[ str(perm[i]) for i in xrange(train_size+val_size,train_size+val_size+test_size) ]
					#'test':[ str(perm[i]) for i in xrange(test_size) ]
				}
	print noEpochs,train_size,val_size,test_size
	training_generator = DataGenerator(partition['train'], **params)
	validation_generator = DataGenerator(partition['validation'],  **params)
	test_generator = DataGenerator(partition['test'],  **params)

	print('Traning:')
	model.fit_generator(generator=training_generator,validation_data=validation_generator,epochs = noEpochs ,use_multiprocessing=True,workers=12,verbose=1)

	print('Testing')
	pred_labels = np.array(model.predict_generator(generator=test_generator,use_multiprocessing=True,workers=12,verbose=1))
	y_test = get_labels(partition['test'])
	preds = np.array([ pred_labels[i][0] for i in xrange(pred_labels.shape[0]) ])
	avg = np.mean(preds)
	pl = preds.ravel() >  avg
	acc = np.mean(np.equal(pl, y_test))
	#print "\n".join([ ",".join(map(str,[preds[i],pl[i],y_test[i]])) for i in range(len(y_test))])
	print("Baseline Accuracy : %0.2f%% "% (100 * acc))

	show_test_data(partition['test'],y_test,pl,preds)

	# model.save('my_model.h5')
	# loaded_model = load_model('my_model.h5')
 
	# print('Testing again')
	# pred_labels = np.array(loaded_model.predict_generator(generator=test_generator,use_multiprocessing=True,workers=12,verbose=1))
	# y_test = get_labels(partition['test'])
	# preds = np.array([ pred_labels[i][0] for i in xrange(pred_labels.shape[0]) ])
	# avg = np.mean(preds)
	# pl = preds.ravel() > avg
	# acc = np.mean(np.equal(pl, y_test))
	# #print "\n".join([ ",".join(map(str,[preds[i],pl[i],y_test[i]])) for i in range(len(y_test))])
	# print("Accuracy2 : %0.2f%% "% (100 * acc))