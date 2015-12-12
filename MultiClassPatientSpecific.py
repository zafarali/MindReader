
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
from metrics import custom
import sys

    
subject_id = 2
training_ds = IOutils.data_streamer2(keeplist=[ (subject_id, i) for i in range(1,7) ]) 

vt = IOutils.VectorTransformer()

linear = LogisticRegression(class_weight = 'auto')
           

for X,Y in training_ds:
    
    # transform the Ys
    Y = vt.transform(Y)
    X = X.astype(np.float)
    # normalization for regression
    X[np.isnan(X)] = 0
    X = X/X.max()
    
    zipped = zip(X,Y)
    random.shuffle(zipped)
    X_train,Y_train = zip(*zipped)
    
    X_train = np.array(X_train, dtype=np.float)
    Y_train = np.array(Y_train, dtype=np.int32).reshape(-1)
           
    linear.fit(X_train, Y_train)

    
    

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

cm = confusion_matrix(Y_test, predicted)
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print classification_report(Y_test, predicted)

correct = predicted == Y_test
print ' correct: ', np.sum(correct)/float(len(correct))
print 'Recall:' ,recall_score(Y_test, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_test, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_test, predicted, average='weighted') 
 

print 'Multiple:', custom.multiple_auc(Y_test, predicted)


