import numpy as np
from scipy import signal
import preprocess_utils as utils

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