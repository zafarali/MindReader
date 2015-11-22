import numpy as np

def window_generator_1D(X, window_size=300, downsample=1):
	"""
		Yields a windowed version of a single (Iterator)
		@params:
			X: the data in shape: n * 1 (n=number of recordings)
			window_size (=300): the size of the window you want to generate
			downsample(=1): the downsampling of the data (intervals over which you want to pick data) 
	"""
	if len(X.shape)==2:
		a = X[::downsample, :]
	else:
		a = X[::downsample]
	
	for j in range(1,window_size):
		yield np.pad(a[:j], (window_size-j, 0), mode='constant')
	
	for i in range(0,len(a)-window_size):
		yield np.array(a[i: i+window_size])
	
	for j in range(len(a)-window_size, len(a)):
		yield np.pad(a[j:], (0, window_size - (len(a)-j)), mode='constant')


def window_generator_ND(X, window_size=300, downsample=1):
	"""
		Yields a windowed version of a multichannel signal
		@params:
			X: the data in shape n * m (n = number of recordings, m = number of channels)
			window_size (=300): the size of the window we want to generate
			downsample (=1): the downsampling of the data (intervals over which you want to pick data) 
	"""
	assert len(X.shape)>1, 'X is not ND, it is 1D'
	transposed_data = X.T
	num_channels = transposed_data.shape[0]
	generators = [None]*num_channels
	for i in xrange(num_channels):
		generators[i] = window_generator_1D(transposed_data[i,:], window_size=window_size, downsample=downsample)
	
	while True:
		results = []
		for i in xrange(num_channels):
			results.append(generators[i].next())
		yield np.array(results).T

		