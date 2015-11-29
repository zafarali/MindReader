import sys
sys.path.append('..')

import preprocess_utils as utils
import IOutils as io

import os
import re
import numpy as np
import pandas as pd
import random
from scipy import signal
from time import time


#---------------------------
def preprocess_sample(X_raw, normalize=True, filters=utils.FREQUENCY_BANDS.keys(), window_size=300, downsample=1, shuffle=True):

    if normalize:
        X_raw = utils.normalize(X_raw)

    if len(X_raw.shape) > 1:
        wg = utils.window_generator_ND(X_raw, window_size=window_size, downsample=downsample)
    else:
        wg = utils.window_generator_1D(X_raw, window_size=window_size, downsample=downsample)

    features_extracted = []
    for windowed_data in wg:
        data_point = []
        for filter_name in filters:
            low, high = utils.FREQUENCY_BANDS[filter_name]

            if len(X_raw.shape) > 1:
                data_point.extend(np.mean(utils.butter_apply(windowed_data, low, high), axis=0).tolist())
            else:
                data_point.append( np.mean( utils.butter_apply(windowed_data, low, high) ) )

        features_extracted.append(data_point)

    return np.array(features_extracted)



#-----------------------
def principal_frequencies(data):
    pass



#-----------------------
def make_all_spectrographs(nperseg=256, mode='train'):
    """Computes and saves the spectrographs of all input data"""
    csvlist = io.get_file_list(mode=mode, fullpath=True)

    for fname in csvlist:
        data = pd.read_csv(fname).values[:,1:]

        # Get the spectrograph
        f,t,sxx = utils.spectrogram(data, window='boxcar', nperseg=nperseg)
        tmp = f
        new_fname = fname[:-4] + '_spectro'
        np.save(new_fname, sxx)

    datadir = io.get_datadir(mode=mode)
    np.save(datadir + 'spectro_freq', f)
    np.save(datadir + 'spectro_time', t)



#-----------------------------
def smoothening(X_raw, normalize=True, window_size=300, downsample=1):

    if len(X_raw.shape) > 1:
        wg = utils.window_generator_ND(X_raw, window_size=window_size, downsample=downsample)
    else:
        wg = utils.window_generator_1D(X_raw, window_size=window_size, downsample=downsample)

    smoothened = []
    for windowed_data in wg:
        if normalize:
            windowed_data = utils.normalize(windowed_data)
        if len(X_raw.shape) > 1:
            smoothened.append( np.mean(windowed_data, axis=0).tolist() )
        else:
            smoothened.appened( np.mean(windowed_data) )

    return np.array(smoothened)





####################
if __name__ == '__main__':
    t0 = time()

    #preprocess_all()
    X = np.array([(x,10*x, 2*x, 3*x) for x in range(2,12)])
    X = utils.running_normalization(X,5)
    print(X)
    
    time_msg = "Time elapsed: " + "%.3f"%(time()-t0) + " seconds"
    print(time_msg)

