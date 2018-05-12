import numpy as np
import keras
import math
total_files = 404290

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim, shuffle=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim)
		# Initialization
		P = []
		Q = []
		PG =[]
		QG =[]
		y = []

		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			data = np.load('../../Vector_Sets/Quora_Single_files/F' + ID + '.npy')
			data[0],data[1] = data[0].reshape(1, self.dim, 1),data[1].reshape(1, self.dim, 1)
			gdata = np.load('../../Vector_Sets/Quora_Glove_Embs/GLOVE' + ID + '.npy')
			#gdata[0],gdata[1] = gdata[0].T , gdata[1].T 
			#print(gdata[0].shape,gdata[1].shape)
			# Store class
			P.append(data[0])
			Q.append(data[1])
			PG.append(gdata[0])
			QG.append(gdata[1])
			y.append(data[2])


		P,Q,y,PG,QG = np.array(P),np.array(Q),np.array(y),np.array(PG),np.array(QG)
		#print(X.shape,y.shape)
		return [P,Q,PG,QG], y



# 	# Parameters
# params = {'dim': 11522,
# 		  'batch_size': 128,
# 		  'shuffle': True}

# # Datasets
# full_data_size = 404291
# test_size = int(math.floor(full_data_size*0.10))
# val_size = int(math.floor(full_data_size*0.09))
# train_size = full_data_size - test_size - val_size
# perm = np.random.permutation(full_data_size)


# partition = { 	'train':[ 'F'+str(perm[i]) for i in xrange(train_size) ] , 
# 				'validation':[ 'F'+str(perm[i]) for i in xrange(train_size,train_size+val_size) ],
# 				'test':[ 'F'+str(perm[i]) for i in xrange(train_size+val_size,train_size+val_size+test_size) ]
# 			}

# training_generator = DataGenerator(partition['train'], **params)
# validation_generator = DataGenerator(partition['validation'],  **params)
# test_generator = DataGenerator(partition['test'],  **params)

# ct=0
# for x,y in training_generator:
# 	try:
# 		pass
# 	except:
# 		print(x.shape,y.shape,ct)
# 	ct+=1
# for x,y in validation_generator:
# 	print(x.shape,y.shape)
# for x,y in test_generator:
# 	print(x.shape,y.shape)