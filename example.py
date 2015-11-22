import numpy as np
import IOutils
from sklearn.linear_model import LogisticRegression

#IOutils.LABEL_NAMES has all the classes we wish to predict

# initialize logistic regressors
LRs = {}
for label_name in IOutils.LABEL_NAMES:
	# each label will have its own logistic regressor
    LRs[label_name] = LogisticRegression()



# Load training data
# load 1 trial each from 3 patients
train_data = IOutils.data_streamer(mode='train', num_patients=3, num_series=1)


# obtain a validation set
X_valid, Y_valid = train_data.next()


# train over remaining data sets one at a time
for X,Y in train_data:
    for i, label_name in enumerate(IOutils.LABEL_NAMES):
        LRs[label_name].fit(X, Y[:,i])



# validate the classifiers
for i, label_name in enumerate(IOutils.LABEL_NAMES):
    predicted = LRs[label_name].predict(X_valid)
    correct = predicted == Y_valid[:,i]
    print label_name,' correct: ', np.sum(correct)/float(len(correct))