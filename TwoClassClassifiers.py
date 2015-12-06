
# coding: utf-8

import numpy as np
import IOutils
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from sklearn import svm
import random
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



ds = IOutils.data_streamer2() 

vt = IOutils.VectorTransformer()

X_valid, Y_valid = ds.next()

# use as follows: 
Y_valid = vt.transform(Y_valid)


Y_valid [Y_valid != 0] = 1
Y_valid [Y_valid == 0] = 0 

# NaivB = GaussianNB()
# linear = LogisticRegression()
support = svm.SVC(kernel='rbf',C=10)

X_next, Y_next = ds.next()
X_next = X_next/X_next.max()
X_next[-256:, 32:]=0
zipped = zip(X_next,Y_next)
random.shuffle(zipped)
X,Y = zip(*zipped)
X_next = np.array(X)
Y_next = np.array(Y)
X_next = X_next.astype(np.float)
Y_next = vt.transform(Y_next)

Y_next [Y_next != 0] = 1
Y_next [Y_next == 0] = 0 
#for X,Y in ds:        
    #zipped = zip(X,Y)
    #random.shuffle(zipped)
    #X,Y = zip(*zipped)
    #X = np.array(X)
    #Y = np.array(Y)
#Y = vt.transform(Y)

# NaivB.fit(X_next, Y_next)                        
# linear.fit(X_next, Y_next)
support.fit(X_next, Y_next)

X_valid = X_valid.astype(np.float)
X_valid[np.isnan(X_valid)] = 0
predicted = support.predict(X_valid)   
correct = predicted == Y_valid
print ' correct: ', np.sum(correct)/float(len(correct))
print confusion_matrix(Y_valid, predicted)
print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_valid, predicted, average='weighted') 




