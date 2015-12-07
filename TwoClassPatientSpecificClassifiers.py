
# coding: utf-8


import numpy as np
import IOutils
from sklearn.linear_model import LogisticRegression
import random
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import matplotlib.pylab as plt
from collections import Counter
import sys


if not len(sys.argv)>1:
    print """arguments:
        SUBJECT_ID: Id of the subject you want to train
    """
    raise Exception('NEED MORE ARGUMENTS')
    
    
# NUM_ZERO_METRIC = sys.argv[3]
# obtain the first 7 datas for the 2nd subject
subject_id = int(sys.argv[1])
training_ds = IOutils.data_streamer2(keeplist=[ (subject_id, i) for i in range(1,7) ]) 
#nn = BasicNN(input_shape=(None,42), output_num_units=12, max_epochs=int(sys.argv[5]), hidden_num_units=int(sys.argv[4]))
vt = IOutils.VectorTransformer()

linear = LogisticRegression(class_weight = 'auto')
           
# n_repeat_sampling = int(sys.argv[2])
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
    Y_zero = Y[Y==0].reshape(-1,1)
    print('number of zeros:',len(Y_zero))
    print Y_zero
    
    # obtain the nonzeros in the dataset
    X_nonzero = X[Y != 0]
    Y_nonzero = Y[Y != 0].reshape(-1, 1)
    print('number of nonzeros', len(Y_nonzero))
    
    X_train = np.vstack((X_zero, X_nonzero))
    Y_train = np.vstack((Y_zero, Y_nonzero))
    
    zipped = zip(X_train,Y_train)
    random.shuffle(zipped)
    X_train,Y_train = zip(*zipped)
    
    X_train = np.array(X_train, dtype=np.float)
    Y_train = np.array(Y_train, dtype=np.int32).reshape(-1)
           
    linear.fit(X_train, Y_train)

    print('dataset: ',dataset_count)
    

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
predicted = linear.predict(X_test)


print confusion_matrix(Y_test, predicted)

print classification_report(Y_test, predicted)

correct = predicted == Y_test
print ' correct: ', np.sum(correct)/float(len(correct))
print 'Recall:' ,recall_score(Y_test, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_test, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_test, predicted, average='weighted') 
 



