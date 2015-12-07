"""
	Contains a whole bunch of tools to feed data into a neuralnetwork
	better. Found from https://www.kaggle.com/bitsofbits/grasp-and-lift-eeg-detection/naive-nnet/code
"""

from os import sys, path

#to allow importing of IOutils
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from nolearn.lasagne import BatchIterator
import IOutils

GLOBALS = {
	'SUBJECTS':range(1,13),
	'SERIES': range(1,9),
	'DOWNSAMPLE': 8,
	'N_OUT':12,
	'SAMPLE_SIZE':512,
	'N_CHANNELS':32
	}

class Source(object):
	def __init__(self):
		self.mean = None
		self.std = None
		self.data = None

	def normalize(self):
		self.data -= self.mean
		sef.data /= self.std


class RawTrainingSource(Source):
	def __init__(self, subjects=SUBJECTS, series=SERIES):
		training_ds = IOutils.data_streamer(patients_list=subjects, series_list=series)

		all_data = list(training_ds)

		X,Y = zip(*all_data)


		self.data = X[0]
		self.events = Y[0]

		self.mean = self.data.mean(axis=0)
		self.std = self.data.std(axis=0)

		self.normalize()


class IndexBatchIterator(BatchIterator):
    def __init__(self, source, SAMPLE_SIZE=GLOBALS['SAMPLE_SIZE'], \
    	N_ELECTRODES=GLOBALS['N_CHANNELS'], DOWNSAMPLE=GLOBALS['DOWNSAMPLE'], \
    	*args, **kwargs):

    	TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
        if source is not None:
            # Tack on (SAMPLE_SIZE-1) copies of the first value so that it is easy to grab
            # SAMPLE_SIZE POINTS even from the first location.
            x = source.data
            self.augmented = np.zeros([len(x)+(SAMPLE_SIZE-1), N_ELECTRODES], dtype=np.float32)
            self.augmented[SAMPLE_SIZE-1:] = x
            self.augmented[:SAMPLE_SIZE-1] = x[0]
        self.Xbuf = np.zeros([self.batch_size, N_ELECTRODES, TIME_POINTS], np.float32) 
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32) 
    
    def transform(self, X_indices, y_indices):
        X_indices, y_indices = super(IndexBatchIterator, self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            sample = self.augmented[ndx:ndx+SAMPLE_SIZE]
            # Reverse so we get most recent point, otherwise downsampling drops the last
            # DOWNSAMPLE-1 points.
            X[i] = sample[::-1][::DOWNSAMPLE].transpose()
            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y


def create_net(train_source, layer_list, layer_factory, batch_size=128, max_epochs=20, update_learning_rate=0.01): 
    """
    	@params:
    		train_source = NO IDEA
    		layer_list = list of (layer, name)
    		layer_factory = a LayerFactory instance
    """
    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
       
    nnet =  NeuralNet(
        layers = layer_list,
        batch_iterator_train = batch_iter_train,
        max_epochs=max_epochs,
        verbose=1000,
        update = nesterov_momentum, 
        update_learning_rate = update_learning_rate,
        update_momentum = 0.9,
        regression = True,
        **layer_factory.kwargs
        )

    return nnet

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1

TRAIN_SIZE = 5*1024


def train_net(layer_list, layer_factory, subjects=SUBJECTS, series=SERIES, **kwargs):
	"""
		Train a neural network
		@params:
			layer_list = list of layers
			layer_factor = the layer factory used
			subjects: the list of subjects to train on
			series: the list of series to use for each subject

			kwargs:
				max_epochs: maximum number of epochs
				update_learning_rate: the learning rate
				batch_size: the size of the batches
	"""

	train_source = RawTrainingSource(subjects=subjects, series=series)

	net = create_net(train_source, layer_list, layer_factory, **kwargs)

	net.fit(train_indices, train_indices)

	





