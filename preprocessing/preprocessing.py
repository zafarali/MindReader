import sys
sys.path.append('..')

import preprocess_utils as utils
import IOutils as io

import os
import re
import numpy as np
import pandas as pd
from scipy import signal
from time import time

#-----------------------
def preprocess_sample(X_raw, normalize=True, filters=utils.FREQUENCY_BANDS.keys(), window_size=300, downsample=1):

    if normalize:
        X_raw = utils.normalize(X_raw)

    wg = utils.window_generator_ND(X_raw, window_size=window_size, downsample=downsample)

    features_extracted = []
    for windowed_data in wg:
        data_point = []
        for filter_name in filters:
            low, high = utils.FREQUENCY_BANDS[filter_name]
            data_point.extend(np.mean(utils.butter_apply(windowed_data, low, high), axis=0).tolist())
        features_extracted.append(data_point)

    return np.array(features_extracted)



#-----------------------
def principal_frequencies(data):
    pass



#-----------------------
def preprocess_all(mode='train'):
    """Computes and saves the spectrographs of all input data"""
    csvlist = io.get_file_list(mode='train', fullpath=True)

    for fname in csvlist:
        data = pd.read_csv(fname).values[:,1:]

        f,t,sxx = utils.spectrogram(data, window='boxcar')
        print(sxx.shape)
        exit()




####################
if __name__ == '__main__':
    t0 = time()

    preprocess_all()
    
    time_msg = "Time elapsed: " + "%.3f"%(time()-t0) + " seconds"
    print(time_msg)


