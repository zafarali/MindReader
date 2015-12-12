import sys
sys.path.append('..')
import IOutils as io

import numpy as np
import os
import pandas as pd
from scipy import signal
from collections import namedtuple

SAMPLING_FREQUENCY = 500
FILTER_ORDER = 4
DTYPE = np.float32

FREQUENCY_BANDS = {
    # 'delta':(1,4),
    # 'theta':(4,8),
    # 'alpha1':(8,10),
    # 'alpha2':(10,12),
    'alpha':(8,13),
    'cbeta':(14,18),
    # 'beta2':(18,26),
    # 'beta':(13,30),
    # 'pgamma':(36,40),
    # 'gamma':(30,45),
    'SMR':(12.5,15)
}


SENSOR_MEAN = np.array([141.43612671,181.4500885,318.70541382,-3.59224248,
                        -6.19295645,293.63870239,5.68691015,296.15979004,
                        74.50315094,-0.9871785,136.28355408,66.4648819,
                        287.72412109,-4.70487118,91.93160248,69.52483368,
                        56.6885643,18.80512428,40.4545784,36.56315613,
                        -56.96159744,147.96061707,211.67466736,90.46417236,
                        123.17406464,153.11029053,235.12339783,-37.77479172,
                        157.26838684,-5.87785339,-49.02427292,-2.42357779],dtype=DTYPE)

SENSOR_STD = np.array([484.43939209,394.33782959,323.73757935,189.50300598,
                       139.62692261,190.29774475,431.20150757,214.77438354,
                       104.06834412,119.83828735,202.79019165,316.48284912,
                       164.62687683,119.49311829,181.1186676,267.18737793,
                       319.74105835,197.75891113,167.47366333,151.26626587,
                       202.29379272,288.68771362,234.14260864,190.28314209,
                       165.09977722,192.34072876,242.53231812,274.65039062,
                       210.33912659,213.33950806,224.74064636,275.12823486],dtype=DTYPE)




#--------------------------
def normalize(x):
    """Returns a zero-mean, std-normalized array of the entries in x"""
    x = x.astype(float)
    ret =  x - np.mean(x, axis=0)
    ret = ret / np.std(x, axis=0)
    return ret


#--------------------------
def butter_filter(low=None, high=None, fs=SAMPLING_FREQUENCY, order=FILTER_ORDER):
    """
    Returns a butter filter
    @params:
        low (=None): the lower cutoff 
        high (=None): the high cutoff
        *if either one of these is None, a highpass or lowpass filter is returned
        *if both are specified a bandpass filter is returned (Note: both cannot be None)
        fs (=500): sampling frequency
        order (=4): the order of the filter
    @returns
        b,a: the coefficients of the numerator and the denominator of the filter
    """
    nyquist = 0.5 * fs
    assert (low is not None) or (high is not None), 'Either high or low must be specified'
    if low is not None and high is None:
        btype = 'lowpass'
        cutoff = low / nyquist
    elif high is not None and low is None:
        btype = 'highpass'
        cutoff = high / nyquist
    else:
        btype = 'bandpass'
        cutoff = [low / nyquist, high / nyquist]
        
    b, a = signal.butter(order, cutoff, btype=btype)
    return b, a


#--------------------------
def butter_apply(X, low=None, high=None, fs=SAMPLING_FREQUENCY, order=FILTER_ORDER):
    """
    Applies a butter filter to a signal
    @params
        X: the raw signal to apply a filter on
        low (=None): the lower cutoff 
        high (=None): the high cutoff
        *if either one of these is None, a highpass or lowpass filter is returned
        *if both are specified a bandpass filter is returned (Note: both cannot be None)
        fs (=500): sampling frequency
        order (=4): the order of the filter
    @returns
        returns a signal where a filter has been applied
    """

    b,a = butter_filter(low=low, high=high, fs=fs, order=order)
    return signal.lfilter(b, a, X, axis=0)


#--------------------------
def window_generator_1D(X, window_size=300, downsample=1, overextend=False):
    """
    Yields a windowed version of a single (Iterator)
    @params:
        X: the data in shape: n * 1 (n=number of recordings)
        window_size (=300): the size of the window you want to generate
        downsample(=1): the downsampling of the data (intervals over which you want to pick data) 
    """
    if len(X.shape)==2:
        a = X[::downsample, :]
    else:
        a = X[::downsample]
    
    for j in range(1,window_size):
        yield np.pad(a[:j], (window_size-j, 0), mode='constant')
    
    for i in range(0,len(a)-window_size+1):
        yield np.array(a[i: i+window_size])
    
    if overextend:
        for j in range(len(a)-window_size, len(a)):
            yield np.pad(a[j:], (0, window_size - (len(a)-j)), mode='constant')


#--------------------------
def window_generator_ND(X, window_size=300, downsample=1):
    """
    Yields a windowed version of a multichannel signal
    @params:
        X: the data in shape n * m (n = number of recordings, m = number of channels)
        window_size (=300): the size of the window we want to generate
        downsample (=1): the downsampling of the data (intervals over which you want to pick data) 
    """
    assert len(X.shape)>1, 'X is not ND, it is 1D'
    transposed_data = X.T
    num_channels = transposed_data.shape[0]
    generators = [None]*num_channels
    for i in xrange(num_channels):
        generators[i] = window_generator_1D(transposed_data[i,:], window_size=window_size, downsample=downsample)
    
    while True:
        results = []
        for i in xrange(num_channels):
            results.append(generators[i].next())
        yield np.array(results).T


#------------------------
def load_csv_data(subid, serid, mode='train', dir_name=None):
    """Loads the data from the appropriate folder, unless dir_name is specified"""
    if dir_name is None:
        file_name = io.get_datadir(mode=mode)
    else:
        file_name = dir_name

    file_name += '/subj'+str(subid)+'_series'+str(serid)
    
    data = pd.read_csv(file_name+'_data.csv')
    events = pd.read_csv(file_name+'_events.csv')
    return data.values[:,1:], events.values[:,1:]


#------------------------
def magsq(x):
    """Calculates the maginitude squared of an input vector x"""
    tmp = x.real**2 + x.imag**2
    return tmp


#-----------------------
def mov_avg(X, N, axis=0):
    """Computes a moving average on the input array. Each point corresponds to
    the mean of the up to N previous points (incl itsef) over the given axis"""
    window = np.ones(N)*(1/float(N))

    # Convolve with MA filter
    ma_arr = np.apply_along_axis(signal.fftconvolve, axis, X, window, mode='full')

    # Remove far edge 
    ma_arr = np.delete(ma_arr, np.s_[-(N-1):], axis=axis)

    # Correct the close edge
    correction = float(N)/np.arange(1,N)
    inplace_mult_start = lambda x,w: np.concatenate((x[:len(w)]*w, x[len(w):]))
    ma_arr = np.apply_along_axis(inplace_mult_start, axis, ma_arr, correction)

    return ma_arr


#------------------------
def running_zeromean(X, N, axis=0):
    """Removes the running average of up to the last N samples from the current sample"""
    if X.shape[0] == N:
        csum = X.cumsum(axis=0)
        csum = csum.astype(DTYPE)/(np.mgrid[1:X.shape[0]+1,0:X.shape[1]][0])
        return X - csum
    else:
        return X - mov_avg(X, N, axis=axis)


#------------------------
def spectrogram(X,
                axis=0,
                noverlap=0,
                fs=SAMPLING_FREQUENCY,
                **kwargs):
    """
    Builds a spectrogram for each column in X (by default)
    All the remaining kwargs are passed to scipy. Refer to scipy.signals.spectrogram()
    
    """
    if len(X.shape) != 2:
        raise Exception('Expected X to have 2 dimensions')


    # Add default values to the kwargs to pass it all to signal.spectrogram
    tmpdict = dict([(x,eval(x)) for x in ['axis', 'noverlap', 'fs']])
    kwargs = dict(kwargs, **tmpdict)

    f, t, sxx = signal.spectrogram(X, **kwargs)
    sxx = np.swapaxes(sxx.astype(np.complex64), 0, 2)
    return f, t, magsq(sxx)



