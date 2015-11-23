import numpy as np
import IOutils
import preprocessing.preprocess_utils as utils
from preprocessing.preprocessing import preprocess_sample

ds = IOutils.data_streamer(num_sets=2)
X, Y = ds.next()
# single_channel = X[:,0]
# x = utils.window_generator_1D(single_channel[:50], window_size=10)
# i = 0
# for a in x:
#     print i,a
#     i+=1



X_processed = preprocess_sample(X, filters=['SMR','alpha1', 'theta'])
print X_processed
print X_processed.shape