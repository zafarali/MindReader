
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import IOutils
import random
# from neuralnetworks.templates import BasicCNN
from sklearn.metrics import confusion_matrix, classification_report
import theano
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from neuralnetworks.tools import LayerFactory
from preprocessing.preprocess_utils import window_generator_ND
from metrics.custom import multiple_auc


# In[2]:

WINDOW_SIZE = 300


# In[3]:

LF = LayerFactory()
layers_list = [
    LF(layers.InputLayer, 'input', shape=(None, WINDOW_SIZE, 32 )),
#     LF(layers.DropoutLayer, p=0.5),
    LF(layers.Conv1DLayer, 'conv1', num_filters=6, filter_size=5, nonlinearity=None, pad='same'),
    LF(layers.Conv1DLayer, 'conv2', num_filters=3, filter_size=2),
    LF(layers.MaxPool1DLayer, 'maxpool1', pool_size=4),
#     LF(layers.DropoutLayer, 'drop1', p=0.5),
    LF(layers.DenseLayer, 'dense1', num_units=300),
    LF(layers.DropoutLayer, 'drop2', p=0.5),
    LF(layers.DenseLayer, 'out', num_units=12, nonlinearity=lasagne.nonlinearities.sigmoid)
]


# In[4]:

nn = NeuralNet(layers_list, 
               max_epochs=30, 
               update=nesterov_momentum, 
               update_learning_rate=0.02, 
               verbose=1000, 
               **LF.kwargs)


# In[5]:

training_ds = IOutils.data_streamer(patients_list=range(1,12), series_list=range(1,5))
# nn = BasicCNN(input_shape=(None,42), output_num_units=12, max_epochs=50, hidden=[256, 120], add_drops=[1,1])
vt = IOutils.VectorTransformer()


# In[ ]:

n_repeat_sampling = 1
dataset_count = 0
for X,Y in training_ds:
    X = X.astype(np.float)
    X[np.isnan(X)] = 0
    X = X/X.max()
    wg = window_generator_ND(X, window_size=WINDOW_SIZE)
    dataset_count += 1
    # transform the Ys
    Y_train = vt.transform(Y)
    
    time_point = 0
    X_train = wg
#    print X_train.shape
#    any_nans = np.isnan(X_train).any()
#    any_infs = np.isinf(X_train).any()

#    if any_nans or any_infs:
#        print('NANS WERE FOUND')
#        break

    X_train = np.array(list(wg))
    print X_train.shape
    any_nans = np.isnan(X_train).any()
    any_infs = np.isinf(X_train).any()

#      print X_train.shape
    # fit the classifier
    nn.fit(X_train, Y_train)
    #end windower
#endfor


# In[ ]:

from metrics.custom import multiple_auc

X_test, Y_test = IOutils.data_streamer(patients_list=[12], series_list=[8]).next()
# X_test, Y_test = testing_ds.next()
X_test = X_test.astype(np.float)
X_test[np.isnan(X_test)] = 0
X_test = X_test/X_test.max()


Y_test = vt.transform(Y_test).astype(np.int32).reshape(-1)

wg = window_generator_ND(X_test, window_size=WINDOW_SIZE)

# predicted = nn.predict(X_test)


# print confusion_matrix(Y_test, predicted)


# # In[ ]:

# print classification_report(Y_test, predicted)

# print multiple_auc(Y_test, predicted)


# In[ ]:testing_data = wg

predicted = nn.predict(testing_data)
print confusion_matrix(Y_test, predicted)
print classification_report(Y_test,predicted)

# In[ ]:

testing_data.shape

pickle.dump(nn, open('./CNN_perpatient.pickle', 'w'))
print('Pickle saved')
print('terminate')


# In[ ]:



