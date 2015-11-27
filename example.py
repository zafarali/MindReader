import numpy as np
import IOutils
from sklearn.linear_model import LogisticRegression
from preprocessing.preprocessing import preprocess_sample
import sys
from sklearn.metrics import roc_auc_score as auc

#IOutils.LABEL_NAMES has all the classes we wish to predict
#HI
# initialize logistic regressors
LRs = {}
LRsprocessed = {}
for label_name in IOutils.LABEL_NAMES:
	# each label will have its own logistic regressor
    LRs[label_name] = LogisticRegression()
    LRsprocessed[label_name] = LogisticRegression()

print('Initialized logistic regressors')


# Load training data
# load 1 trial each from 3 patients
train_data = IOutils.data_streamer(mode='train', num_patients=1, num_series=7)


# obtain a validation set
X_valid, Y_valid = train_data.next()
print('Validation set loaded')

# train over remaining data sets one at a time
for X,Y in train_data:
    for i, label_name in enumerate(IOutils.LABEL_NAMES):
        LRs[label_name].fit(X, Y[:,i])
        
        X = preprocess_sample(X, filters=['alpha2'])

        LRsprocessed[label_name].fit(X, Y[:,i])
print('classifiers trained')


# validate the classifiers
for i, label_name in enumerate(IOutils.LABEL_NAMES):

    predicted = LRs[label_name].predict(X_valid)
    # print 'predicted:',predicted.tolist()
    # print 'true:',Y_valid[:,i].tolist()
    # print 'matches:',(predicted==Y_valid[:,i]).tolist()
    # print 'accuracy:',np.sum(predicted==Y_valid[:,i])
    print 'AUC (unprocessed,',label_name,'):',auc(Y_valid[:,i].astype(int), predicted.astype(int))
    # correct = predicted == Y_valid[:,i]

    X_valid = preprocess_sample(X_valid, filters=['alpha2'])

    predicted = LRsprocessed[label_name].predict(X_valid)
    print 'AUC (processed,',label_name,'):',auc(Y_valid[:,i].astype(int), predicted.astype(int))
    # correctprocessed = predicted == Y_valid[:,i]
    # print label_name,' unprocessed correct: ', np.sum(correct)/float(len(correct))
    # print label_name,' processed correct: ',np.sum(correctprocessed)/float(len(correct))


