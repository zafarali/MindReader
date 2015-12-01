from __future__ import print_function
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
# INCOMPLETE
def principal_frequencies(sxx, N):
    """Outputs a vector of the N frequencies that can be directly concatenate"""

    # Extract max sensor for each frequency
    max_over_sensor = np.max(sxx,axis=1)

    # Extract the N strongest frequenciess
    max_freqs = np.argpartition(max_over_sensor, N, axis=1)[:,sxx.shape[2]-N:]
    max_freqs.sort(axis=1)

    # Normalize
    out = max_freqs / float(sxx.shape[2])
    
    return out

#-----------------------
def printflush(msg):
    print(msg, end="")
    sys.stdout.flush()



#-----------------------
def preprocess_all(norm_wind=None,
                   nperseg=256,
                   mode='train',
                   max_freq_count=10,
                   disp=True):
    """Computes and saves the spectrographs of all input data"""
    csvlist = io.get_file_list(mode=mode, fullpath=True)

    pif = lambda msg: printflush(msg) if disp else None


    for fullpath in csvlist:
        t0 = time()
        fpath, fname = os.path.split(fullpath)
        data = pd.read_csv(fullpath).values[:,1:]
        pif('Processing ' + fname + ' -- ' + str(data.shape[0]) + ' samples...')
        
        # Get the spectrograph
        f,t,sxx = utils.spectrogram(data, window='boxcar', nperseg=nperseg)
        #spectro_fname = fullpath[:-4] + '_spectro'
        #np.save(spectro_fname, sxx)

        # N Principal frequencies (a normalized index)
        max_freqs = principal_frequencies(sxx, max_freq_count)

        # BLow up the max frequencies to match the data array
        repeated_max_freqs = np.empty((data.shape[0], max_freq_count), dtype=max_freqs.dtype)
        tmp = np.zeros((1, max_freqs.shape[1]))
        max_freqs = np.insert(max_freqs, 0, tmp, axis=0)
        for k in range(0,max_freqs.shape[0]-1):
            repeated_max_freqs[k*nperseg:(k+1)*nperseg,:] = np.tile(max_freqs[k,:], (nperseg,1))


        # Execute the running mean
        if norm_wind is not None:
            wind = norm_wind
        else:
            wind = data.shape[0]
        norm_data = utils.running_normalization(data, wind, axis=0)
        pif("\b" + "%.3f"%(time()-t0) + " s\n")

        # Concatenate
        #del data
        final_data = np.append(norm_data, repeated_max_freqs, axis=1)
        #del norm_data

        norm_wind = 'FULL' if norm_wind==None else norm_wind
        final_fname = fullpath[:-4] + '_W' + str(nperseg) + '_norm' + str(norm_wind)
        np.save(final_fname, final_data)
        exit()







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

    preprocess_all(mode='test')
    preprocess_all(mode='train')
    #make_all_spectrographs()


    
    time_msg = "Time elapsed: " + "%.3f"%(time()-t0) + " seconds"
    print(time_msg)

