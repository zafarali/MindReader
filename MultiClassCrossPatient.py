"""
	Non-zero cross patient!!!!
	this code runs the basic classifiers
"""

import numpy as np
import IOutils
# from sklearn.naive_bayes import GaussianNB
# from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
# from sklearn import svm
import random
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


ds = IOutils.data_streamer2(keeplist=[ (i,0) for i in xrange(1,12) ]) 

vt = IOutils.VectorTransformer()

# NaivB = GaussianNB()
# lr = LogisticRegression(class_weight="auto")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
lr = GridSearchCV(LogisticRegression(penalty='l1',class_weight = 'auto'), param_grid)
GridSearchCV(cv=None,estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
            penalty='l2', tol=0.0001), param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
# svc = svm.SVC(kernel='rbf',C=10,class_weight="auto")
# lrd = LDA() 

for X_next, Y_next in ds:
	X_next = X_next.astype(np.float)/X_next.max()
	X_next[np.isnan(X_next)]=0
	zipped = zip(X_next,Y_next)
	random.shuffle(zipped)
	X,Y = zip(*zipped)
	X_next = np.array(X)
	Y_next = np.array(Y)
	X_next = X_next.astype(np.float)
	Y_next = vt.transform(Y_next)
    
    X_train = X_next
    Y_train = Y_next
    
    zipped = zip(X_train,Y_train)
    random.shuffle(zipped)
	X_train,Y_train = zip(*zipped)
        
    # now all ready!
    X_train = np.array(X_train, dtype=np.float)
    Y_train = np.array(Y_train, dtype=np.int32).reshape(-1)
	

	# NaivB.fit(X_train, Y_train)                        
	# linear.fit(X_train, Y_train)
#     lrd.fit(X_train, Y_train)
	lr.fit(X_train, Y_train)
	# svc.fit(X_train, Y_train)





testing_ds = data_streamer2(keeplist=[(12,0)])

X_test, Y_test = testing_ds.next()
X_test = X_test.astype(np.float)
X_test = X_test.astype(np.float)
X_test[np.isnan(X_test)] = 0
X_test = X_test/X_test.max()
Y_test = vt.transform(Y_test).astype(np.int32)



print 'LOGISTICREGRESSION'
predicted = lr.predict(X_test)  
# predicted = svc.predict(X_test) 
correct = predicted == Y_test
print ' correct: ', np.sum(correct)/float(len(correct))
print confusion_matrix(Y_test, predicted)
print 'Recall:' ,recall_score(Y_test, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_test, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_test, predicted, average='weighted') 
print 'classification_report:', classification_report(Y_test, predicted)
print 'Multiple:', custom.multiple_auc(Y_test, predicted)


print confusion_matrix(Y_test, predicted)

cm = confusion_matrix(Y_test, predicted)
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# print 'SVC'
# predicted = svc.predict(X_valid)   
# correct = predicted == Y_valid
# print ' correct: ', np.sum(correct)/float(len(correct))
# print confusion_matrix(Y_valid, predicted)
# print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
# print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
# print 'F1 Score:',f1_score(Y_valid, predicted, average='weighted') 
# print 'classification_report:', classification_report(Y_valid, predicted)








