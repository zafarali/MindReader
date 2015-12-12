
"""
	Non-zero patient specific!!!!
	this code runs the basic classifiers
"""
import numpy as np
import matplotlib.pylab as plt
from collections import Counter
import IOutils
import random
# from neuralnetworks.templates import BasicNN2
from sklearn.metrics import confusion_matrix, classification_report
from metrics import custom
# from sklearn.naive_bayes import GaussianNB
# from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
# from sklearn import svm



training_ds = IOutils.data_streamer2(keeplist=[ (1, i) for i in range(1,7) ]) # obtain the first 7 datas for the 1st subject


vt = IOutils.VectorTransformer()


# NaivB = GaussianNB()
# lr = LogisticRegression(class_weight="auto")
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
lr = GridSearchCV(LogisticRegression(penalty='l1',class_weight = 'auto'), param_grid)
GridSearchCV(cv=None,estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
            penalty='l2', tol=0.0001), param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
# svc = svm.SVC(kernel='rbf',C=10,class_weight="auto")
# lrd = LDA() 

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
    Y_nonzero = Y[Y>0].reshape(-1, 1) - 1
#     print('number of nonzeros', len(Y_nonzero))
    
    # obtain the number of zeros to sample
#     to_sample = int(np.max(Counter(Y.tolist()).values()))
#     print('number of zeros to sample:',to_sample)
    
    for i in range(n_repeat_sampling):
        # sample the zeros
#         X_sampled = np.array(random.sample(X_zero.tolist(), to_sample), dtype=X_zero.dtype)
#         Y_sampled = np.zeros((X_sampled.shape[0], 1))
        
        # compile the dataset
#         X_train = np.vstack((X_sampled, X_nonzero))
#         Y_train = np.vstack((Y_sampled, Y_nonzero))

        X_train = X_nonzero
        Y_train = Y_nonzero

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
        lr.fit(X_train, Y_train)

        print('dataset: ',dataset_count,', trained: ',i,'/',n_repeat_sampling, )
    

testing_ds = IOutils.data_streamer2(keeplist=[(2,8)])
X_test, Y_test = testing_ds.next()
X_test = X_test.astype(np.float)
X_test = X_test.astype(np.float)
X_test[np.isnan(X_test)] = 0
X_test = X_test/X_test.max()
Y_test = vt.transform(Y_test).astype(np.int32)


X_test = X_test[Y_test > 0]
Y_test = Y_test[Y_test > 0] - 1

Y_test.reshape(-1)
print np.unique(Y_test)
predicted = lr.predict(X_test)



print classification_report(Y_test, predicted)


print 'Multiple:', custom.multiple_auc(Y_test, predicted)


print confusion_matrix(Y_test, predicted)

cm = confusion_matrix(Y_test, predicted)
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
