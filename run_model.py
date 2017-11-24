
from keras.models import Model, Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, Lambda,dot,maximum
from keras.optimizers import SGD
from keras.metrics import cosine_proximity
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import load_model
import get_data as datasets

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
	return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1(y_true, y_pred):
	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	return (2.0*((p*r)/(p+r)))


cv_scores = []
cv = 5

batchSize = 100
noEpochs = 50
learn_rate = 0.001
momen = 0.05
inputDimension = 13810
validation_data_split = 0.20
nb_zhang = 5
nb_train = 20
invalid_samples_per_valid = 5

print("Learning rate :", learn_rate)
print("No of epochs :", noEpochs)
print("Batch Size :", batchSize)

while cv:
	
	# Model hyper parameters
	inputQ = Input(shape=(1, inputDimension, 1))
	inputA = Input(shape=(1, inputDimension, 1))
	inputL = Input(shape=(1,))
	baseNw = baseNetwork(inputDimension)
	#baseNw.summary()
	convQues = baseNw(inputQ)
	convAns = baseNw(inputA)
	distance = Lambda(cosine_dist, output_shape=cos_dist_output_shape)([convQues, convAns])
	model = Model(input=[inputQ, inputA], output=distance)
	model.compile(loss = c_loss, optimizer=SGD(lr=learn_rate, momentum=momen), metrics=['acc'])


	print("Training:")
	# Data hyper parameters

	x_val,y_val = datasets.get_zhang_data(2) ## 2 validation blocks from zhang
	validationData = ( [x_val[:,0,:].reshape(x_val.shape[0], 1, inputDimension, 1) , \
													x_val[:,1,:].reshape(x_val.shape[0], 1, inputDimension, 1) ], y_val)

	for e in range(nb_train):
		qq, qql   = datasets.get_quora_data(1)
		qa, qal   = datasets.get_yahoo_webscope_data(1,invalid_samples_per_valid)
		x_train, y_train   =  np.concatenate([qq,qa],0),np.concatenate([qql,qal],0)
		print (x_train.shape, x_train[0][0].shape)
		questions =  x_train[:,0,:].reshape(x_train.shape[0], 1, inputDimension, 1)
		answers = x_train[:,1,:].reshape(x_train.shape[0], 1, inputDimension, 1)
		#model.fit([questions, answers], y_train, batch_size=batchSize, epochs=noEpochs, validation_split=validation_data_split,shuffle=1)
		model.fit([questions, answers], y_train, batch_size=batchSize, epochs=noEpochs, validation_data=validationData ,shuffle=1)

	print("Testing:")
	x_test,y_test = datasets.get_zhang_data(nb_zhang)
	print (x_test.shape,y_test.shape)
	pred_labels = model.predict([ x_test[:,0,:].reshape(x_test.shape[0], 1, inputDimension, 1) , x_test[:,1,:].reshape(x_test.shape[0], 1, inputDimension, 1)  ])
	preds = [ pred_labels[i][0] for i in xrange(pred_labels.shape[0]) ]
	average = sum(preds)/float(len(preds))
	labels = [ 1 if preds[i]>=average else 0 for i in xrange(pred_labels.shape[0])  ]
	acc = sum( [ 1 if labels[i]==y_test[i] else 0 for i in xrange(len(labels)) ] ) /float(len(labels))
	print("Cross_Val : ",cv," Accuracy : ",acc)
	cv_scores.append(acc)
	cv-=1

cv_scores = np.array(cv_scores)
print(cv_scores)
print("Mean : ",cv_scores.mean()," SD : ",cv_scores.std())

print("\nLearning rate :", learn_rate)
print("No of epochs :", noEpochs)
print("Batch Size :", batchSize)

