import numpy as np

zhang_total_blocks = 24
quora_total_blocks = 404
yahoo_webscope_total_blocks = 141


def get_zhang_data(n_blocks=1,randomize=True):
	x,y ,seen = [],[],[]
	for i in range(n_blocks):
		p = np.random.randint(0,zhang_total_blocks-1)
		while p in seen:
			p = np.random.randint(0,zhang_total_blocks-1)
		seen.append(p)
		block = np.load('../Vector_Sets/Zhang_yahoo/B'+str(p)+'.npy')
		#print (block.shape)
		for j in range(block.shape[0]):
			x.append( np.array([block[j][0],block[j][1]]))
			y.append(int(block[j][2]))

	x,y = np.array(x),np.array(y)

	if randomize:
		perm = np.random.permutation(x.shape[0])
		x  = x[perm]
		y  = y[perm]
	return x,y


def get_quora_data(n_blocks=1,randomize=True):
	x,y ,seen = [],[],[]
	for i in range(n_blocks):
		p = np.random.randint(0,quora_total_blocks-1)
		while p in seen:
			p = np.random.randint(0,quora_total_blocks-1)
		seen.append(p)
		block = np.load('../Vector_Sets/Quora_Duplicate_Questions/B'+str(p)+'.npy')
		#print (block.shape)
		for j in range(block.shape[0]):
			x.append( np.array([block[j][0],block[j][1]]))
			y.append(int(block[j][2]))

	x,y = np.array(x),np.array(y)

	if randomize:
		perm = np.random.permutation(x.shape[0])
		x  = x[perm]
		y  = y[perm]
	return x,y


def get_yahoo_webscope_data(n_blocks=1,invalid_ratio=3,randomize=True):
	
	x,y,seen=[],[],[]
	
	for i in xrange(n_blocks):
		p = np.random.randint(0,yahoo_webscope_total_blocks-1)
		while p in seen:
			p = np.random.randint(0,yahoo_webscope_total_blocks-1)
		seen.append(p)

		block = np.load('../Vector_Sets/Yahoo_Webscope_L5/B'+str(p)+'.npy')
		#print (block.shape)
		for j in range(block.shape[0]):
			x.append( np.array([block[j][0],block[j][1]]))
			y.append(int(block[j][2]))


		for v in range(invalid_ratio):
			for j in  range(block.shape[0]):
				r = np.random.randint(0,block.shape[0]-1)
				while ( r == j ):
					r = np.random.randint(0,block.shape[0]-1)
				x.append( np.array([block[j][0],block[r][1]]))
				y.append(0)


	x,y = np.array(x),np.array(y)

	if randomize:
		perm = np.random.permutation(x.shape[0])
		x  = x[perm]
		y  = y[perm]

	#print(x.shape,y.shape)
	
	return x,y

