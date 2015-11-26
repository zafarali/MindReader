import numpy as np
import random
from scipy import signal
import preprocess_utils as utils

def preprocess_sample(X_raw, normalize=True, filters=utils.FREQUENCY_BANDS.keys(), window_size=300, downsample=1):

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


def smoothening(X_raw, normalize=True, window_size=300, downsample=1):

	if normalize:
		X_raw = utils.normalize(X_raw)

	if len(X_raw.shape) > 1:
		wg = utils.window_generator_ND(X_raw, window_size=window_size, downsample=downsample)
	else:
		wg = utils.window_generator_1D(X_raw, window_size=window_size, downsample=downsample)

	smoothened = []
	for windowed_data in wg:
		if len(X_raw.shape) > 1:
			smoothened.append( np.mean(windowed_data, axis=0).tolist() )
		else:
			smoothened.appened( np.mean(windowed_data) )

	return np.array(smoothened)