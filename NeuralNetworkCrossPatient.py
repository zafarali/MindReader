
# coding: utf-8

# In[4]:

import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import IOutils
import random
from neuralnetworks.templates import BasicNN2
from sklearn.metrics import confusion_matrix, classification_report
from itertools import product


# In[5]:

training_ds = IOutils.data_streamer2(keeplist=[ (i, 0) for i in range(1,4) ]) # obtain data for first 11 patients

nn = BasicNN2(input_shape=(None,42), output_num_units=12, max_epochs=100, hidden=[256,128,64])
vt = IOutils.VectorTransformer()


# In[6]:

n_repeat_sampling = 1
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
    to_sample = int(np.mean(Counter(Y.tolist()).values()))
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
        print('Unique Ys:',np.unique(Y_train).tolist())
        print('Are any Xs nan?', any_nans)
        print('Are any Xs inf?', any_infs)
        
        if any_nans or any_infs:
            print('NANS WERE FOUND')
            break
        # fit the classifier
        nn.fit(X_train, Y_train)

        print('dataset: ',dataset_count,', trained: ',i,'/',n_repeat_sampling, )
        #endfor
    #endfor
#     print('total size after sampling:',len(Y_train))
#     print('-----')


# In[ ]:



# In[8]:

testing_ds = IOutils.data_streamer2(keeplist=[(12,0)])

X_test, Y_test = testing_ds.next()
X_test = X_test.astype(np.float)
X_test = X_test.astype(np.float)
X_test[np.isnan(X_test)] = 0
X_test = X_test/X_test.max()
Y_test = vt.transform(Y_test).astype(np.int32).reshape(-1)

predicted = nn.predict(X_test)


# In[ ]:

print confusion_matrix(Y_test, predicted)


# In[ ]:

print classification_report(Y_test, predicted)


# In[ ]:



