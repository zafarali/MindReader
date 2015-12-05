
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import IOutils
import random
from neuralnetworks.templates import BasicNN
from sklearn.metrics import confusion_matrix, classification_report


import sys

if not len(sys.argv)>1:
    print """arguments:
        SUBJECT_ID: Id of the subject you want to train
        N_REPEAT_SAMPLING: number of times to repeat zero sampling
        NUM_ZERO_METRIC: metric to pick zeros (mean, max, min)
        HIDDEN_NODES: number of hidden units
        MAX_EPOCHS: maximum number of epochs
    """
    raise Exception('NEED MORE ARGUMENTS')
# In[2]:

NUM_ZERO_METRIC = sys.argv[3]
# obtain the first 7 datas for the 2nd subject
subject_id = int(sys.argv[1])
training_ds = IOutils.data_streamer2(keeplist=[ (subject_id, i) for i in range(1,7) ]) 
nn = BasicNN(input_shape=(None,42), output_num_units=12, max_epochs=int(sys.argv[5]), hidden_num_units=int(sys.argv[4]))
vt = IOutils.VectorTransformer()


# In[3]:

# number of times to repeat 0 subsampling
n_repeat_sampling = int(sys.argv[2])
dataset_count = 0
for X,Y in training_ds:
    dataset_count += 1
    # transform the Ys
    Y = vt.transform(Y)
#     print('total size before sampling:', len(Y))
    X = X.astype(np.float)
    # normalization for regression
    X[np.isnan(X)] = 0
    X = X/X.max()
    
    # obtains the zeros in the datasets
    X_zero = X[Y==0]
    Y_zero = Y[Y==0]
#     print('number of zeros:',len(Y_zero))
    
    # obtain the nonzeros in the dataset
    X_nonzero = X[Y>0]
    Y_nonzero = Y[Y>0].reshape(-1, 1)
#     print('number of nonzeros', len(Y_nonzero))
    
    # obtain the number of zeros to sample
    vals = Counter(Y.tolist()).values()
    if NUM_ZERO_METRIC == 'mean':
        pick_zeros = np.mean(vals)
    elif NUM_ZERO_METRIC == 'max':
        pick_zeros = np.max(vals)
    elif NUM_ZERO_METRIC == 'min':
        pick_zeros = np.min(vals)
    to_sample = int(pick_zeros)
#     print('number of zeros to sample:',to_sample)
    
    for i in range(n_repeat_sampling):
        # sample the zeros
        X_sampled = np.array(random.sample(X_zero.tolist(), to_sample), dtype=X_zero.dtype)
        Y_sampled = np.zeros((X_sampled.shape[0], 1))
        
        # compile the dataset
        X_train = np.vstack((X_sampled, X_nonzero))
        Y_train = np.vstack((Y_sampled, Y_nonzero))

        # shuffle the data
        zipped = zip(X_train,Y_train)
        random.shuffle(zipped)
        X_train,Y_train = zip(*zipped)
        
        # now all ready!
        X_train = np.array(X_train, dtype=np.float)
        Y_train = np.array(Y_train, dtype=np.int32).reshape(-1)
        
        any_nans = np.isnan(X_train).any()
        any_infs = np.isinf(X_train).any()
        # print('Unique Ys:',np.unique(Y_train).tolist())
        # print('Are any Xs nan?', any_nans)
        # print('Are any Xs inf?', any_infs)
        
        if any_nans or any_infs:
            print('NANS WERE FOUND')
            break
        # fit the classifier
        nn.fit(X_train, Y_train)

        print('dataset: ',dataset_count,', trained: ',(i+1),'/',n_repeat_sampling, )
        #endfor
    #endfor
#     print('total size after sampling:',len(Y_train))
#     print('-----')


# In[ ]:



# In[4]:

# load the last dataset for the subject
testing_ds = IOutils.data_streamer2(keeplist=[(subject_id,8)])
X_test, Y_test = testing_ds.next()

# convert to float
X_test = X_test.astype(np.float)
# remove nans
X_test[np.isnan(X_test)] = 0
X_test = X_test/X_test.max()


Y_test = vt.transform(Y_test).astype(np.int32).reshape(-1)

# predict using your classifier
predicted = nn.predict(X_test)



print confusion_matrix(Y_test, predicted)

print classification_report(Y_test, predicted)




