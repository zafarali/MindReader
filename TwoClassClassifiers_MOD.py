
# coding: utf-8

import numpy as np
import IOutils
# from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import random
import sys
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from neuralnetworks.templates import BasicNN2

nn = BasicNN2(max_epochs=100, hidden=[200,50,30], input_shape=(None, 42), output_num_units=2)
ds = IOutils.data_streamer2(keeplist=[ (i,0) for i in xrange(1,12) ]) 

vt = IOutils.VectorTransformer()

# NaivB = GaussianNB()
lr = LogisticRegression(class_weight="auto")
#svc = svm.SVC(kernel='rbf',C=10,class_weight="auto")

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

	Y_next[Y_next != 0] = 1
	Y_next[Y_next == 0] = 0 

	# print np.unique(Y_next)
	#for X,Y in ds:        
	    #zipped = zip(X,Y)
	    #random.shuffle(zipped)
	    #X,Y = zip(*zipped)
	    #X = np.array(X)
	    #Y = np.array(Y)
	#Y = vt.transform(Y)

	# NaivB.fit(X_next, Y_next)                        
	# linear.fit(X_next, Y_next)
	nn.fit(X_next, Y_next)
	lr.fit(X_next, Y_next)
#	svc.fit(X_next, Y_next)





ds2 = IOutils.data_streamer2(keeplist=[(12,0)])
X_valid, Y_valid = ds2.next()

# use as follows: 
Y_valid = vt.transform(Y_valid)


Y_valid[Y_valid != 0] = 1
Y_valid[Y_valid == 0] = 0 

X_valid = X_valid.astype(np.float)
X_valid[np.isnan(X_valid)] = 0

print 'NEURAL NETWORK'
predicted = nn.predict(X_valid)   
correct = predicted == Y_valid
print ' correct: ', np.sum(correct)/float(len(correct))
print confusion_matrix(Y_valid, predicted)
print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_valid, predicted, average='weighted') 
print 'classification_report:', classification_report(Y_valid, predicted)

print 'LOGISTICREGRESSION'
predicted = lr.predict(X_valid)   
correct = predicted == Y_valid
print ' correct: ', np.sum(correct)/float(len(correct))
print confusion_matrix(Y_valid, predicted)
print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
print 'F1 Score:',f1_score(Y_valid, predicted, average='weighted') 
print 'classification_report:', classification_report(Y_valid, predicted)


#print 'SVC'
#predicted = svc.predict(X_valid)   
#correct = predicted == Y_valid
#print ' correct: ', np.sum(correct)/float(len(correct))
#print confusion_matrix(Y_valid, predicted)
#print 'Recall:' ,recall_score(Y_valid, predicted, average='weighted')
#print 'Precision:' ,precision_score(Y_valid, predicted, average='weighted')  
#print 'F1 Score:',f1_score(Y_valid, predicted, average='weighted') 
#print 'classification_report:', classification_report(Y_valid, predicted)








