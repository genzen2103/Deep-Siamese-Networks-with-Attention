
from keras.models import Model, Sequential, Model
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, Lambda,dot,maximum
from keras.optimizers import SGD
from keras.metrics import cosine_proximity
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split

def get_full_data(max_files,invalid_ratio):
	x,y=[],[]
	for i in xrange(max_files):
		q,a = np.load('full_vect_dataset/C'+str(i)+'.npy')
		x.append([q,a])
		y.append(1)
	for i in xrange(invalid_ratio):
		for j in xrange(max_files):
			r = np.random.randint(0,max_files-1)
			while r==j:
				r = np.random.randint(0,max_files-1)
			x.append([ x[j][0] , x[r][1] ])
			y.append(0)
	x = np.array(x)
	y = np.array(y)
	perm = np.random.permutation(x.shape[0])
	x  = x[perm]
	y  = y[perm]
	return [ x, y  ]

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

# def euclidean_distance(vects):
#     x, y = vects
#     return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# def contrastive_loss(y_true, y_pred):
#     margin = 1
#     return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def cosine_dist(vects):
	return dot(vects,axes=1,normalize=True)


def cos_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def c_loss(y_true, y_pred):
	margin = 0.40
	return K.mean( y_true * (1-y_pred) + (1 - y_true) * K.maximum(y_pred-margin, 0) )


def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batch-wise average of recall.

	Computes the recall, a metric for multi-label classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def f1(y_true, y_pred):
	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	return (2.0*((p*r)/(p+r)))


all_files = 3119
invalid_samples_per_valid = 5

X, Y =get_full_data(all_files,invalid_samples_per_valid)

print X.shape, X[0][0].shape

batchSize = 100
inputDimension = X[0][0].shape[0]
test_data_split = 0.20
validation_data_split = 0.25
noEpochs = 80
learn_rate = 0.01
momen = 0.05

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_data_split, random_state=21)

questions =  X_train[:,0,:].reshape(X_train.shape[0], 1, inputDimension, 1)
answers = X_train[:,1,:].reshape(X_train.shape[0], 1, inputDimension, 1)

inputQ = Input(shape=questions.shape[1:])
inputA = Input(shape=answers.shape[1:])
inputL = Input(shape=(1,))
baseNw = baseNetwork(inputDimension)
convQues = baseNw(inputQ)
convAns = baseNw(inputA)

distance = Lambda(cosine_dist, output_shape=cos_dist_output_shape)([convQues, convAns])

model = Model(input=[inputQ, inputA], output=distance)
model.compile(loss = c_loss, optimizer=SGD(lr=learn_rate, momentum=momen), metrics=['acc',precision,recall, f1])
#model.summary()

model.fit([questions, answers], y_train, batch_size=batchSize, epochs=noEpochs, validation_split=validation_data_split)

pred_labels = model.predict([ X_test[:,0,:].reshape(X_test.shape[0], 1, inputDimension, 1) , X_test[:,1,:].reshape(X_test.shape[0], 1, inputDimension, 1)  ])

score = model.evaluate([ X_test[:,0,:].reshape(X_test.shape[0], 1, inputDimension, 1) , X_test[:,1,:].reshape(X_test.shape[0], 1, inputDimension, 1)  ],y_test,verbose=1)

print ("Score:",score)
