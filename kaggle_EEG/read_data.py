import sys
sys.path.append('..')
import IOutils as io

import numpy as np
import os
import pandas as pd





# Grab train data
data = []
label = []
n_sub = 1
n_series = 8

train_streamer = io.data_streamer2(mode='train')

for k in range(n_sub):
    sub_data = []
    sub_label = []
    for series in range(n_series):
        d, e = train_streamer.next()
        sub_data.append(d)
        sub_label.append(e)

    data.append(sub_data)
    label.append(sub_label)

np.save('eeg_train.npy', [data, label])

del data, label

# Grab test data
data = []
label = []
n_sub = 1
n_series = 2

test_streamer = io.data_streamer2(mode='test')


for k in range(n_sub):
    sub_data = []
    sub_label = []
    for series in range(n_series):
        d, _ = test_streamer.next()
        e = np.zeros([d.shape[0], 6])
        sub_data.append(d)
        sub_label.append(e)

    data.append(sub_data)
    label.append(sub_label)


np.save('eeg_test.npy', [data, label])
