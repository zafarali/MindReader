
# coding: utf-8

import numpy as np
import IOutils
# from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import random
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from metrics import custom
import matplotlib.pyplot as plt


ds = IOutils.data_streamer2() 

vt = IOutils.VectorTransformer()

X_valid, Y_valid = ds.next()

# use as follows: 
Y_valid = vt.transform(Y_valid)


# NaivB = GaussianNB()
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
linear = GridSearchCV(LogisticRegression(penalty='l1',class_weight = 'auto'), param_grid)
GridSearchCV(cv=None,estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
            penalty='l2', tol=0.0001), param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})


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


# NaivB.fit(X_next, Y_next)  
linear.fit(X_next, Y_next)                      


X_valid = X_valid.astype(np.float)
X_valid[np.isnan(X_valid)] = 0
predicted = linear.predict(X_valid)   
correct = predicted == Y_valid
print ' correct: ', np.sum(correct)/float(len(correct))

print confusion_matrix(Y_valid, predicted)
cm = confusion_matrix(Y_valid, predicted)
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
pl.xlabel('Predicted')
pl.ylabel('True')
plt.show()

print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
print 'Multiple:', custom.multiple_auc(Y_valid, predicted)
 





