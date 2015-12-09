import csv
# As is this script scores 0.71+ on the leaderboard. If you download and run
# at home, you can tweak the parameters as described in the Discussion
# to get 0.90+

import sys
import numpy as np
import scipy
import pandas
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from numpy import fft
from numpy.random import randint
# Lasagne (& friends) imports
import theano
from nolearn.lasagne import BatchIterator, NeuralNet
from lasagne.objectives import aggregate, categorical_crossentropy
from lasagne import layers
import lasagne
from lasagne.updates import nesterov_momentum
from theano.tensor.nnet import sigmoid
import gc
from metrics.custom import multiple_auc
from IOutils import VectorTransformer

# Silence some warnings from lasagne
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

SUBJECTS = list(range(1,13))
TRAIN_SERIES = list(range(1,9))
TEST_SERIES = [9,10]

N_ELECTRODES = 32
N_EVENTS = 12

# We train on TRAIN_SIZE randomly selected location each "epoch" (yes, that's
# not really an epoch). One-fifth of these locations are used for validation,
# hence the 5*X format, to make it clear what the number of validation points
# is.
TRAIN_SIZE = 5*1024

SAMPLE_SIZE = 4000 # Larger (2048 perhaps) would be better
# We are downsample without low-pass filtering here. You should probably filter
# to avoid aliading of the data.
DOWNSAMPLE = 8
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE
    

# We encapsulate the event / electrode data in a Source object. 

class Source:

    mean = None
    std = None

    def load_raw_data(self, subjects, series):
        raw_data = []
        if type(subjects) != list:
            subjects = [ subjects ]
        for subject in subjects:
            raw_data.extend([self.read_csv(self.path(subject, i, "data")) for i in series])
            raw_events = [self.read_csv(self.path(subject, i, "events")) for i in series]
        self.data = np.concatenate(raw_data, axis=0).astype(np.float32)
        self.data[np.isnan(self.data)] = 0
        self.events = VectorTransformer().transform(np.concatenate(raw_events, axis=0).astype(np.int32)).reshape(-1, 1)
    
    def normalize(self):
        self.data = self.data - self.mean
        self.data =  self.data / self.std
        
    @staticmethod
    def path(subject, series, kind):
        prefix = "train" if (series in TRAIN_SERIES) else "test"
        return "./data/{0}/subj{1}_series{2}_{3}.csv".format(prefix, subject, series, kind)
    
    csv_cache = {}
    @classmethod
    def read_csv(klass, path):
        if path not in klass.csv_cache:
            if len(klass.csv_cache): # Only cache last value
                klass.csv_cache.popitem() # Need this or we run out of memory in Kaggle scripts
            klass.csv_cache[path] = pandas.read_csv(path, index_col=0).values
        return klass.csv_cache[path]
        
class TrainSource(Source):

    def __init__(self, subject, series_list):
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        # print 'mean:',self.mean
        # print 'sd', self.std
        self.normalize()
        print 'data shapes:', self.data.shape
        print 'event shapes', self.events.shape
        # self.principle_components = scipy.linalg.svd(self.data, full_matrices=False)
        # self.std2 = self.data.std(axis=0)
        # self.data = self.data / self.std2

        
# Note that Test/Submit sources use the mean/std from the training data.
# This is both standard practice and avoids using future data in theano
# test set.
        
class TestSource(Source):

    def __init__(self, subject, series, train_source):
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        # self.principle_components = train_source.principle_components
        self.normalize()

        # self.data = self.data / train_source.std2
        

class SubmitSource(TestSource):

    def __init__(self, subject, a_series, train_source):
        TestSource.__init__(self, subject, [a_series], train_source)

    def load_raw_data(self, subject, series):
        [a_series] = series
        self.data = self.read_csv(self.path(subject, a_series, "data"))
        
        
# Lay out the Neural net.


class LayerFactory:
    """Helper class that makes laying out Lasagne layers more pleasant"""
    def __init__(self):
        self.layer_cnt = 0
        self.kwargs = {}
    def __call__(self, layer, layer_name=None, **kwargs):
        self.layer_cnt += 1
        name = layer_name or "layer{0}".format(self.layer_cnt)
        for k, v in kwargs.items():
            self.kwargs["{0}_{1}".format(name, k)] = v
        return (name, layer) 



class IndexBatchIterator(BatchIterator):
    """Generate BatchData from indices.
    
    Rather than passing the data into the fit function, instead we just pass in indices to
    the data.  The actual data is then grabbed from a Source object that is passed in at
    the creation of the IndexBatchIterator. Passing in a '-1' grabs a random value from
    the Source.
    
    As a result, an "epoch" here isn't a traditional epoch, which looks at all the
    time points. Instead a random subsamle of 0.8*TRAIN_SIZE points from the
    training data are used each "epoch" and 0.2 TRAIN_SIZE points are uses for
    validation.

    """
    def __init__(self, source, *args, **kwargs):
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
        self.Ybuf = np.zeros(self.batch_size, np.int32) 
    
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
            # print '---'
            # print 'i=',i
            # print 'ndx=',ndx
            # print 'X[i].shape=',X[i].shape
            
            # print 'sample.shape=',sample.shape
            # print 'sample[::-1].shape=',sample[::-1].shape

            # print 'sample[::-1][::DOWNSAMPLE].shape=',sample[::-1][::DOWNSAMPLE].shape
            X[i] = sample[::-1][::DOWNSAMPLE].transpose()
            if y_indices is not None:
                Y[i] = self.source.events[ndx]

        Y = None if (y_indices is None) else Y

        return X, Y
    

# Simple / Naive net. Borrows from Daniel Nouri's Facial Keypoint Detection Tutorial 

def create_net(train_source, test_source, batch_size=128, max_epochs=20): 
    
    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
    batch_iter_test  = IndexBatchIterator(test_source, batch_size=batch_size)
    LF = LayerFactory()

    # dense = 196 # larger (1024 perhaps) would be better
    
    layer_list = [
        LF(layers.InputLayer, shape=(None, N_ELECTRODES, TIME_POINTS)), 
        # LF(DropoutLayer, p=0.5),
        # # This first layer condenses N_ELECTRODES down to num_filters.
        # # Since the electrode results are reportedly highly reduntant this
        # # should speed things up without sacrificing accuracy. It may
        # # also increase stability. This was 8 in an earlier version.
        # LF(Conv1DLayer, num_filters=4, filter_size=1, nonlinearity=None),
        # # Try one convolutional layer
        # LF(Conv1DLayer, num_filters=8, filter_size=5, nonlinearity=lasagne.nonlinearities.softmax),
        # # Maxpooling is more typically done with a pool_size of 2
        # LF(MaxPool1DLayer, pool_size=4),
        # # Standard fully connected net from here on out.
        # LF(DropoutLayer, p=0.5),
        # LF(DenseLayer, num_units=dense),
        # LF(DropoutLayer, p=0.5),
        # LF(DenseLayer, num_units=dense),
        # LF(DropoutLayer, p=0.5),


        LF(layers.Conv1DLayer, 'conv1', num_filters=6, filter_size=5, nonlinearity=None, pad='same'),
        LF(layers.Conv1DLayer, 'conv2', num_filters=3, filter_size=2),
        LF(layers.MaxPool1DLayer, 'maxpool1', pool_size=4),
        LF(layers.DenseLayer, 'dense1', num_units=1024),
        LF(layers.DropoutLayer, 'drop2', p=0.5),
        LF(layers.DenseLayer, 'dense2', num_units=512),
        LF(layers.DropoutLayer, 'drop3', p=0.5),
        LF(layers.DenseLayer, layer_name="output", num_units=N_EVENTS, nonlinearity=lasagne.nonlinearities.softmax)
    ]
    
    # def loss(x,t):
    #     return aggregate(categorical_crossentropy(x, t))
    
    
    nnet =  NeuralNet(
        # y_tensor_type = theano.tensor.matrix,
        layers = layer_list,
        batch_iterator_train = batch_iter_train,
        batch_iterator_test = batch_iter_test,
        max_epochs=max_epochs,
        verbose=1000000,
        update = nesterov_momentum,
        update_momentum=0.9, 
        update_learning_rate = 0.01,
        # update_momentum = 0.5,
        **LF.kwargs
        )

    return nnet


# Do the training.

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1


def score(net, samples=256):
    """Compute the area under the curve, ROC score
    
    We take `samples` random samples and compute the ROC AUC
    score on those samples. 
    """
    source = net.batch_iterator_test.source
    test_indices = np.arange(len(source.events))
    np.random.seed(199)
    np.random.shuffle(test_indices)
    predicted = net.predict_proba(test_indices[:samples])
    actual = source.events[test_indices[:samples]]
    return roc_auc_score(actual.reshape(-1), predicted.reshape(-1))
    

def train(factory, subj, series=TRAIN_SERIES, max_epochs=20, valid_series=[1,2], params=None):
    tseries = sorted(set(series) - set(valid_series))
    train_source = TrainSource(subj, tseries)
    test_source = TestSource(subj, valid_series, train_source) # just a place holder
    # i don't think it really does anything when line 265 is commented out
    net = factory(train_source, test_source, max_epochs=max_epochs)
    if params is not None:
        net.load_params_from(params)
    net.fit(train_indices, train_indices)
    # print("Score:", score(net))
    return (net, train_source)
 
# def get_predictions(factory, params, subj, series):
#     train_source = TrainSource(subj, series)
#     test_source = TestSource(subj, valid_series, train_source)

#     net = factory()

def train_all(factory, max_epochs=30, init_epochs=30, valid_series=[1,2]):
    info = {}
    params = None
    for subj in SUBJECTS:
        print("Subject:", subj)
        epochs = max_epochs + init_epochs
        net, train_source = train(factory, subj, epochs, valid_series, params)
        params = net.get_all_params_values()
        info[subj] = (params, train_source)
        init_epochs = 0
    return (factory, info)   

from collections import Counter

def train_cross_subject(factory, train_subject_ids, train_series_ids, test_subject_ids, test_series_ids, max_epochs=30, train_sample_size=0):
    toval = 1
    train_series_ids = sorted(set(train_series_ids) - set([toval]))
    params = None
    for subject in train_subject_ids:
        train_source = TrainSource(subject_id, train_series_ids)
        test_source = TestSource(subject_id, [toval], train_source)
        net = factory(train_source, test_source, max_epochs=max_epochs)
        if params is not None:
            net.load_weights_from(params)

        train_indices = np.zeros(train_sample_size, dtype=np.int32) -1

        net.fit(train_indices, train_indices)

        params = net.get_all_params_values()

    del train_source
    # del test_source


    # net.predict_proba
    val_source = TrainSource(test_subject_ids, test_series_ids)
    # source = net.batch_iterator_test.source

    val_indices = np.arange(len(val_source.events))

    np.random.shuffle(val_indices)

    print len(val_source.events)

    net = factory(val_source, test_source)

    net.load_weights_from(params)

    predicted = net.predict(val_indices[:SAMPLE_SIZE])
    actual = net.events[val_indices[:SAMPLE_SIZE]]

    print 'unique predicted labels:',np.unique(predicted)
    print 'unique actual labels',np.unique(actual)

    return actual, predicted



def train_subject_specific(factory, subject_id, train_series_ids, test_series_ids, max_epochs=30, init_epochs=0, train_sample_size=0):
    info = {}
    params = None
    # toval = int(np.random.choice(train_series_ids))
    # print toval
    toval = 1 # for now fix this
    # print train_series_ids
    # print train_series_ids
    train_source = TrainSource(subject_id, train_series_ids)
    test_source = TestSource(subject_id, [toval] , train_source) # just a place holder
    # i don't think it really does anything when line 265 is commented out
    net = factory(train_source, test_source, max_epochs=max_epochs)
    
    if train_sample_size == 0:
        train_sample_size = len(train_source.events)
    
    train_indices = np.zeros(train_sample_size, dtype=np.int32) -1
    
    # print Counter(train_source.events.T.tolist())

    train_net = factory(train_source, test_source, max_epochs=max_epochs)

    print train_indices.dtype
    train_net.fit(train_indices, train_indices)

    params = train_net.get_all_params_values()
    # get rid of it from memory
    del train_source


    # net.predict_proba
    val_source = TrainSource(subject_id, test_series_ids)
    # source = net.batch_iterator_test.source

    val_indices = np.arange(len(val_source.events))

    np.random.shuffle(val_indices)

    print len(val_source.events)

    test_net = factory(val_source, test_source)

    test_net.load_weights_from(params)

    predicted = test_net.predict(val_indices[:SAMPLE_SIZE])
    actual = val_source.events[val_indices[:SAMPLE_SIZE]]

    print 'unique predicted labels:',np.unique(predicted)
    print 'unique actual labels',np.unique(actual)

    return actual, predicted
    # train(factory, subject_id, )

 
def make_submission(train_info, name):
    factory, info = train_info
    with open(name, 'w') as file:
        file.write("id,HandStart,FirstDigitTouch,BothStartLoadPhase,LiftOff,Replace,BothReleased\n")
        for subj in SUBJECTS:
            weights, train_source = info[subj]
            for series in [9,10]:
                print("Subject:", subj, ", series:", series)
                submit_source = SubmitSource(subj, series, train_source)  
                indices = np.arange(len(submit_source.data))
                net = factory(train_source=None, test_source=submit_source)
                net.load_weights_from(weights)
                probs = net.predict_proba(indices)
                for i, p in enumerate(probs):
                    id = "subj{0}_series{1}_{2},".format(subj, series, i)
                    file.write(id + ",".join(str(x) for x in p) + '\n')
        
        
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print """
            Arguments are as follows:
                MODEL: 'per' or 'cross'
                MAX_EPOCH: the number of epochs to train with
                TRAIN_SIZE: the size of the samples to train each epoch with 
        """
        raise Exception('not enough arguments')

    MODE = sys.argv[1] # cross vs per
    MAX_EPOCH = int(sys.argv[2]) # max epochs
    TRAIN_SIZE2 = int(sys.argv[3]) # the sample size to train with 
    if MODE == 'per':
        print 'Training per patient classifier'
        # train_info = train_all(create_net, max_epochs=25) # Training for longer would likley be better
        Y_true, Y_pred = train_subject_specific(create_net, subject_id=[ 2 ], train_series_ids=range(1,8), \
            test_series_ids=range(8,9), train_sample_size=TRAIN_SIZE2, max_epochs=MAX_EPOCH)
    elif MODE == 'cross':
        print 'Training cross patient classifier'
        Y_true, Y_pred = train_cross_subject(create_net, )


    print 'multiple_auc:',multiple_auc(Y_true, Y_pred)
    print classification_report(Y_true, Y_pred)
    print confusion_matrix(Y_true, Y_pred)
    # make_submission(train_info, "naive_grasp.csv") 
